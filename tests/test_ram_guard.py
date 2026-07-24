"""Tests for the RAM safety net.

Two things here would be dangerous if they broke silently: the guard must
never target itself, its ancestors, or whatever holds the DuckDB lock, and it
must never act at all when the machine isn't actually under pressure. Every
test drives the real decision logic through injected fakes -- nothing here
touches a real process.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from market_intelligence.autopilot import ram_guard
from market_intelligence.autopilot.ram_guard import ProcessInfo, _protected_pids


def _proc(pid: int, ppid: int, command: str, *, rss_kb: int = 100_000,
          etime_seconds: int = 3600, stat: str = "S") -> ProcessInfo:  # fmt: skip
    return ProcessInfo(
        pid=pid, ppid=ppid, rss_kb=rss_kb, etime_seconds=etime_seconds, stat=stat, command=command
    )


# --------------------------------------------------------------------------- #
# parsing                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "etime,expected_seconds",
    [
        ("00:05", 5),
        ("02:30", 150),
        ("1:02:30", 3750),
        ("3-01:02:30", 3 * 86400 + 3750),
    ],
)
def test_parse_etime(etime: str, expected_seconds: int) -> None:
    assert ram_guard._parse_etime(etime) == expected_seconds


def test_read_free_pct_parses_memory_pressure_output() -> None:
    output = "System-wide memory free percentage: 42%\nStats:\nPages free: 1234\n"
    assert ram_guard.read_free_pct(lambda: output) == 42.0


def test_read_free_pct_falls_back_to_healthy_on_unparseable_output() -> None:
    assert ram_guard.read_free_pct(lambda: "nonsense") == 100.0


def test_read_free_pct_falls_back_to_healthy_when_command_missing() -> None:
    def _raise() -> str:
        raise OSError("no such command")

    assert ram_guard.read_free_pct(_raise) == 100.0


def test_read_processes_parses_ps_lines() -> None:
    raw = (
        "  123     1  40960 01:02:03 S    /usr/bin/ollama serve\n"
        "  456   123  10240    00:05 Z    <defunct>\n"
    )
    procs = ram_guard.read_processes(lambda: raw)
    assert len(procs) == 2
    assert procs[0].pid == 123
    assert procs[0].rss_kb == 40960
    assert procs[0].command == "/usr/bin/ollama serve"
    assert procs[1].is_zombie


def test_read_processes_skips_unparseable_lines() -> None:
    raw = "not-a-valid-ps-line\n  123     1  40960 01:02:03 S    ollama serve\n"
    procs = ram_guard.read_processes(lambda: raw)
    assert len(procs) == 1


# --------------------------------------------------------------------------- #
# protection: self, ancestors, the DB lock holder                             #
# --------------------------------------------------------------------------- #
def test_protected_pids_walks_the_ancestor_chain() -> None:
    processes = [
        _proc(300, 200, "python worker.py"),
        _proc(200, 100, "zsh"),
        _proc(100, 1, "tmux"),
        _proc(999, 300, "unrelated sibling"),
    ]
    protected = _protected_pids(processes, own_pid=300)
    assert protected == {300, 200, 100}
    assert 999 not in protected


def test_protected_pids_stops_at_missing_parent() -> None:
    processes = [_proc(50, 40, "orphan")]  # ppid 40 not present in the list
    protected = _protected_pids(processes, own_pid=50)
    # 40 is still marked protected -- it's claimed as an ancestor even though
    # its own row wasn't in the snapshot -- but the walk goes no further since
    # nothing is known about *its* parent.
    assert protected == {50, 40}


# --------------------------------------------------------------------------- #
# duplicate detection                                                          #
# --------------------------------------------------------------------------- #
def test_no_candidates_when_only_one_singleton_running() -> None:
    processes = [_proc(1, 0, "ollama serve")]
    protected = _protected_pids(processes, own_pid=99999)
    candidates = ram_guard.find_kill_candidates(processes, protected=protected, lock_holders=set())
    assert candidates == []


def test_duplicate_ollama_keeps_the_older_one() -> None:
    processes = [
        _proc(10, 1, "/opt/homebrew/opt/ollama/bin/ollama serve", etime_seconds=7200),
        _proc(20, 1, "/opt/homebrew/opt/ollama/bin/ollama serve", etime_seconds=30),
    ]
    protected = _protected_pids(processes, own_pid=99999)
    candidates = ram_guard.find_kill_candidates(processes, protected=protected, lock_holders=set())
    assert [c.pid for c in candidates] == [20]
    assert candidates[0].marker == "ollama serve"


def test_self_is_never_a_candidate_and_never_dooms_the_other_copy() -> None:
    # pid 300 is self, the *newer* market_intelligence process. It must never
    # be targeted -- and since it's excluded from the group entirely, the one
    # remaining process (pid 10) is no longer part of a "duplicate pair" and
    # is left alone too. Killing pid 10 just because self happens to share its
    # marker would risk stopping a pre-existing legitimate run (e.g. a manual
    # backfill the daily autopilot invocation raced against on startup) --
    # doing nothing is the safe default when self is one of only two.
    processes = [
        _proc(10, 1, "python -m market_intelligence autopilot run", etime_seconds=7200),
        _proc(300, 200, "python -m market_intelligence autopilot run", etime_seconds=5),
        _proc(200, 1, "zsh"),
    ]
    protected = _protected_pids(processes, own_pid=300)
    lock_holders: set[int] = set()
    candidates = ram_guard.find_kill_candidates(
        processes, protected=protected, lock_holders=lock_holders
    )
    assert candidates == []


def test_genuine_third_party_duplicate_still_caught_alongside_self() -> None:
    # Now there are two *other* market_intelligence processes plus self. Self
    # is excluded, but the remaining two are still a genuine duplicate pair.
    processes = [
        _proc(10, 1, "market-intelligence sec ingest-documents", etime_seconds=7200),
        _proc(11, 1, "market-intelligence sec ingest-documents", etime_seconds=30),
        _proc(300, 200, "market-intelligence autopilot run", etime_seconds=5),
        _proc(200, 1, "zsh"),
    ]
    protected = _protected_pids(processes, own_pid=300)
    candidates = ram_guard.find_kill_candidates(processes, protected=protected, lock_holders=set())
    assert [c.pid for c in candidates] == [11]


def test_db_touching_duplicate_protected_when_it_holds_the_lock() -> None:
    # pid 20 is younger but holds the DuckDB lock -- it must survive even
    # though the age heuristic alone would have picked pid 10.
    processes = [
        _proc(10, 1, "market-intelligence sec ingest-documents", etime_seconds=999),
        _proc(20, 1, "market-intelligence sec ingest-documents", etime_seconds=5),
    ]
    protected = _protected_pids(processes, own_pid=99999)
    candidates = ram_guard.find_kill_candidates(processes, protected=protected, lock_holders={20})
    assert [c.pid for c in candidates] == [10]


def test_db_touching_duplicates_untouched_when_lock_holder_unknown() -> None:
    processes = [
        _proc(10, 1, "market-intelligence sec ingest-documents", etime_seconds=999),
        _proc(20, 1, "market-intelligence sec ingest-documents", etime_seconds=5),
    ]
    protected = _protected_pids(processes, own_pid=99999)
    candidates = ram_guard.find_kill_candidates(processes, protected=protected, lock_holders=None)
    assert candidates == []


def test_non_db_marker_still_evaluated_when_lock_holder_unknown() -> None:
    # ollama doesn't touch the DuckDB file, so an unknown lock holder must not
    # block its (unrelated) duplicate detection.
    processes = [
        _proc(10, 1, "ollama serve", etime_seconds=999),
        _proc(20, 1, "ollama serve", etime_seconds=5),
    ]
    protected = _protected_pids(processes, own_pid=99999)
    candidates = ram_guard.find_kill_candidates(processes, protected=protected, lock_holders=None)
    assert [c.pid for c in candidates] == [20]


def test_wrapper_and_its_child_are_one_launch_not_duplicates() -> None:
    # `uv run market-intelligence ...` shows up as TWO processes -- the uv
    # wrapper and the python worker it spawned -- and both command lines match
    # the same marker. On 2026-07-23 the guard killed the live extractor's
    # worker exactly this way (exit 143). A parent/child chain is one launch,
    # never a duplicate pair.
    processes = [
        _proc(10, 1, "uv run market-intelligence signals extract-relationships", etime_seconds=600),
        _proc(
            11,
            10,
            "python .venv/bin/market-intelligence signals extract-relationships",
            etime_seconds=598,
        ),
    ]
    protected = _protected_pids(processes, own_pid=99999)
    candidates = ram_guard.find_kill_candidates(processes, protected=protected, lock_holders=set())
    assert candidates == []


def test_chain_through_a_non_matching_shell_still_collapses() -> None:
    # wrapper -> zsh (no marker) -> worker: ancestry walks the full process
    # table, so an intermediate non-matching shell doesn't break the chain.
    processes = [
        _proc(10, 1, "uv run market-intelligence sec ingest-documents", etime_seconds=600),
        _proc(11, 10, "zsh -c real-work", etime_seconds=599),
        _proc(
            12, 11, "python .venv/bin/market-intelligence sec ingest-documents", etime_seconds=598
        ),
    ]
    protected = _protected_pids(processes, own_pid=99999)
    candidates = ram_guard.find_kill_candidates(processes, protected=protected, lock_holders=set())
    assert candidates == []


def test_two_unrelated_launch_trees_yield_one_candidate() -> None:
    # Two independent wrapper+child trees ARE duplicates. Exactly one
    # candidate: the younger tree's root -- killing the wrapper takes that
    # launch down as a unit, and the survivor tree is untouched.
    processes = [
        _proc(10, 1, "uv run market-intelligence sec ingest-documents", etime_seconds=7200),
        _proc(
            11, 10, "python .venv/bin/market-intelligence sec ingest-documents", etime_seconds=7199
        ),
        _proc(20, 1, "uv run market-intelligence sec ingest-documents", etime_seconds=60),
        _proc(
            21, 20, "python .venv/bin/market-intelligence sec ingest-documents", etime_seconds=59
        ),
    ]
    protected = _protected_pids(processes, own_pid=99999)
    candidates = ram_guard.find_kill_candidates(processes, protected=protected, lock_holders=set())
    assert [c.pid for c in candidates] == [20]


def test_lock_held_by_a_child_protects_its_whole_tree() -> None:
    # The younger tree's WORKER holds the DuckDB lock. Lock protection must
    # apply to the whole launch tree, so the older tree's root is the one
    # candidate -- the age heuristic alone would have doomed the young worker.
    processes = [
        _proc(10, 1, "uv run market-intelligence sec ingest-documents", etime_seconds=7200),
        _proc(
            11, 10, "python .venv/bin/market-intelligence sec ingest-documents", etime_seconds=7199
        ),
        _proc(20, 1, "uv run market-intelligence sec ingest-documents", etime_seconds=60),
        _proc(
            21, 20, "python .venv/bin/market-intelligence sec ingest-documents", etime_seconds=59
        ),
    ]
    protected = _protected_pids(processes, own_pid=99999)
    candidates = ram_guard.find_kill_candidates(processes, protected=protected, lock_holders={21})
    assert [c.pid for c in candidates] == [10]


def test_unrelated_duplicate_helper_processes_are_ignored() -> None:
    # e.g. VS Code / Electron renderer helpers -- not a recognized marker, so
    # never a candidate no matter how many copies are running.
    processes = [_proc(pid, 1, "Code Helper (Renderer)", etime_seconds=pid) for pid in (10, 20, 30)]
    protected = _protected_pids(processes, own_pid=99999)
    candidates = ram_guard.find_kill_candidates(processes, protected=protected, lock_holders=set())
    assert candidates == []


# --------------------------------------------------------------------------- #
# zombies and high-memory reporting                                           #
# --------------------------------------------------------------------------- #
def test_find_zombies() -> None:
    processes = [_proc(1, 0, "alive", stat="S"), _proc(2, 1, "<defunct>", stat="Z")]
    zombies = ram_guard.find_zombies(processes)
    assert [z.pid for z in zombies] == [2]


def test_find_high_memory_excludes_protected_and_sorts_descending() -> None:
    processes = [
        _proc(1, 0, "self", rss_kb=5_000_000),
        _proc(2, 1, "big_but_protected", rss_kb=9_000_000),
        _proc(3, 1, "medium", rss_kb=2_000_000),
        _proc(4, 1, "small", rss_kb=10_000),
    ]
    protected = {2}
    result = ram_guard.find_high_memory(processes, protected=protected, threshold_mb=1000)
    assert [p.pid for p in result] == [1, 3]


# --------------------------------------------------------------------------- #
# orchestration: run_guard                                                     #
# --------------------------------------------------------------------------- #
def _fake_kill_registry(alive_pids: set[int]) -> tuple[list[tuple[int, int]], object]:
    """A fake ``kill`` that removes a pid from ``alive_pids`` on SIGTERM."""
    calls: list[tuple[int, int]] = []

    def kill(pid: int, sig: int) -> None:
        calls.append((pid, sig))
        if sig == 0:  # liveness probe
            if pid not in alive_pids:
                raise ProcessLookupError(pid)
            return
        if pid not in alive_pids:
            raise ProcessLookupError(pid)
        if sig == 15:  # SIGTERM: pretend it exits cleanly
            alive_pids.discard(pid)

    return calls, kill


def test_run_guard_does_not_act_above_the_pressure_threshold(tmp_path: Path) -> None:
    processes_raw = (
        "  10     1  40960 02:00:00 S    ollama serve\n"
        "  20     1  40960    00:30 S    ollama serve\n"
    )
    result = ram_guard.run_guard(
        database_path=tmp_path / "db.duckdb",
        memory_runner=lambda: "System-wide memory free percentage: 80%\n",
        ps_runner=lambda: processes_raw,
        lsof_runner=lambda _p: "",
        kill=lambda *_: pytest.fail("kill must not be called when RAM is healthy"),
    )
    assert result.acted is False
    assert result.killed == []
    assert len(result.kill_candidates) == 1
    assert result.kill_candidates[0].pid == 20


def test_run_guard_dry_run_never_kills_even_under_pressure(tmp_path: Path) -> None:
    processes_raw = (
        "  10     1  40960 02:00:00 S    ollama serve\n"
        "  20     1  40960    00:30 S    ollama serve\n"
    )
    result = ram_guard.run_guard(
        database_path=tmp_path / "db.duckdb",
        dry_run=True,
        memory_runner=lambda: "System-wide memory free percentage: 5%\n",
        ps_runner=lambda: processes_raw,
        lsof_runner=lambda _p: "",
        kill=lambda *_: pytest.fail("kill must not be called during a dry run"),
    )
    assert result.acted is False
    assert result.killed == []
    assert len(result.kill_candidates) == 1


def test_run_guard_kills_the_duplicate_under_real_pressure(tmp_path: Path) -> None:
    processes_raw = (
        "  10     1  40960 02:00:00 S    ollama serve\n"
        "  20     1  40960    00:30 S    ollama serve\n"
    )
    alive = {10, 20}
    calls, fake_kill = _fake_kill_registry(alive)
    result = ram_guard.run_guard(
        database_path=tmp_path / "db.duckdb",
        memory_runner=lambda: "System-wide memory free percentage: 5%\n",
        ps_runner=lambda: processes_raw,
        lsof_runner=lambda _p: "",
        kill=fake_kill,
        sleep=lambda _seconds: None,
    )
    assert result.acted is True
    assert [k.pid for k in result.killed] == [20]
    assert result.killed[0].signal_used == "SIGTERM"
    assert (20, 15) in calls  # SIGTERM sent
    assert (10, 15) not in calls  # the survivor was never touched


def test_run_guard_escalates_to_sigkill_if_sigterm_does_not_stick(tmp_path: Path) -> None:
    processes_raw = (
        "  20     1  40960    00:30 S    ollama serve\n"
        "  10     1  40960 02:00:00 S    ollama serve\n"
    )
    calls: list[tuple[int, int]] = []

    def stubborn_kill(pid: int, sig: int) -> None:
        calls.append((pid, sig))
        # Never actually removes the pid from "alive" on SIGTERM -- simulates
        # a process that ignores it and must be SIGKILLed.

    result = ram_guard.run_guard(
        database_path=tmp_path / "db.duckdb",
        memory_runner=lambda: "System-wide memory free percentage: 5%\n",
        ps_runner=lambda: processes_raw,
        lsof_runner=lambda _p: "",
        kill=stubborn_kill,
        sleep=lambda _seconds: None,
    )
    assert result.killed[0].signal_used == "SIGKILL"
    assert (20, 9) in calls


def test_run_guard_reports_lsof_failure_as_a_note(tmp_path: Path) -> None:
    def _raise(_path: Path) -> str:
        raise OSError("lsof not found")

    result = ram_guard.run_guard(
        database_path=tmp_path / "db.duckdb",
        memory_runner=lambda: "System-wide memory free percentage: 5%\n",
        ps_runner=lambda: (
            "  10     1  40960 02:00:00 S    market-intelligence sec ingest-documents\n"
            "  20     1  40960    00:30 S    market-intelligence sec ingest-documents\n"
        ),
        lsof_runner=_raise,
        kill=lambda *_: pytest.fail("a DB-touching duplicate must not be killed when lsof failed"),
    )
    assert any("lsof" in note for note in result.notes)
    assert result.killed == []


def test_summary_line_reports_nothing_to_do_on_a_healthy_machine() -> None:
    result = ram_guard.GuardResult(free_pct=90.0, acted=False)
    assert "nothing to do" in result.summary_line


def test_summary_line_distinguishes_would_stop_from_stopped() -> None:
    candidate = ram_guard.KilledProcess(
        pid=20, command="ollama serve", rss_mb=512.0, marker="ollama serve", signal_used=""
    )
    reported = ram_guard.GuardResult(free_pct=10.0, acted=False, kill_candidates=[candidate])
    assert "would stop" in reported.summary_line

    acted = ram_guard.GuardResult(
        free_pct=10.0,
        acted=True,
        killed=[
            ram_guard.KilledProcess(
                pid=20,
                command="ollama serve",
                rss_mb=512.0,
                marker="ollama serve",
                signal_used="SIGTERM",
            )
        ],
    )
    assert "stopped" in acted.summary_line and "would stop" not in acted.summary_line
