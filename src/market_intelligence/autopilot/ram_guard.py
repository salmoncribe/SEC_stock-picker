"""RAM safety net: stop useless duplicate processes before they starve the mini.

The mini has 16GB of RAM and runs a local LLM (ollama) for the connection
engine alongside everything else on the machine. The failure mode worth
guarding against isn't "the model is big" -- ollama and the model size are
known and budgeted for -- it's an *accidental duplicate*: the same collector or
backfill script launched twice (cron fires while a manual run is still going),
or a second ollama server started by habit. A duplicate is pure waste, not
extra throughput: DuckDB is single-writer (see ``orchestrator``), so a second
collector copy only blocks on the first's lock, and a second ollama server just
fights the first for the same port. Both sit in RAM doing nothing.

This module finds exact duplicates of known singleton commands and stops the
redundant copy -- but only once the OS itself reports memory pressure. On a
healthy machine a run only reports what it sees; it never touches anything.
Two things are never touched regardless of pressure: this process and its own
ancestors (so the guard can never kill the session that invoked it), and
whatever process holds the DuckDB file open (so an in-progress backfill is
never interrupted -- checked separately in the logs and files per the
single-writer note, never guessed at here).
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from market_intelligence.logging_config import get_logger

logger = get_logger(__name__)

# Command-line substrings that mark a process as a "singleton": something that
# should only ever have one live copy running at once. A second copy matching
# the same marker is fighting the first for the DuckDB lock or the ollama port,
# never doing additional useful work.
SINGLETON_MARKERS: tuple[str, ...] = (
    "ollama serve",
    "market_intelligence",
    "market-intelligence",
    "backfill_documents.sh",
    "evaluate_filing_events.sh",
)

# Markers whose duplicates might be mid-write to the database, so a duplicate
# among them is only killed once it's confirmed not to hold the DuckDB lock.
_DB_TOUCHING_MARKERS = (
    "market_intelligence",
    "market-intelligence",
    "backfill_documents.sh",
    "evaluate_filing_events.sh",
)

# Below this free-memory percentage the guard is allowed to act; at or above
# it, a run only reports what it *would* do. `memory_pressure`'s free% is the
# same figure macOS itself watches before it starts compressing and swapping,
# so it's a more honest trigger than re-deriving one from raw page counts.
DEFAULT_ACT_BELOW_FREE_PCT = 20.0

# Grace period between SIGTERM and SIGKILL for a duplicate that won't exit.
_KILL_GRACE_SECONDS = 2.0

ProcessRunner = Callable[[], str]
KillFn = Callable[[int, int], None]


@dataclass(frozen=True)
class ProcessInfo:
    """One row of ``ps`` output, parsed just enough to reason about safety."""

    pid: int
    ppid: int
    rss_kb: int
    etime_seconds: int
    stat: str
    command: str

    @property
    def rss_mb(self) -> float:
        return self.rss_kb / 1024

    @property
    def is_zombie(self) -> bool:
        return "Z" in self.stat

    def matches(self, marker: str) -> bool:
        return marker in self.command


@dataclass(frozen=True)
class KilledProcess:
    """A duplicate the guard stopped, with enough context to explain why."""

    pid: int
    command: str
    rss_mb: float
    marker: str
    signal_used: str


@dataclass
class GuardResult:
    """What one guard run found and did.

    ``acted`` is false whenever memory pressure was below the threshold (or
    ``dry_run`` was set) -- in that case ``killed`` is always empty and
    ``kill_candidates`` names what a real run would have stopped instead.
    """

    free_pct: float
    acted: bool
    killed: list[KilledProcess] = field(default_factory=list)
    kill_candidates: list[KilledProcess] = field(default_factory=list)
    zombies: list[ProcessInfo] = field(default_factory=list)
    high_memory: list[ProcessInfo] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def summary_line(self) -> str:
        stopped = self.killed or self.kill_candidates
        verb = "stopped" if self.killed else "would stop"
        if not stopped:
            return f"ram-guard: {self.free_pct:.0f}% free, nothing to do."
        names = ", ".join(f"{k.marker} (pid {k.pid}, {k.rss_mb:.0f}MB)" for k in stopped)
        return f"ram-guard: {self.free_pct:.0f}% free, {verb} {len(stopped)} duplicate(s): {names}"


# --------------------------------------------------------------------------- #
# Reading system state                                                         #
# --------------------------------------------------------------------------- #
def _default_memory_runner() -> str:
    return subprocess.run(
        ["memory_pressure"], capture_output=True, text=True, timeout=10, check=False
    ).stdout


def read_free_pct(runner: ProcessRunner = _default_memory_runner) -> float:
    """Parse macOS's own free-memory percentage from ``memory_pressure``.

    Falls back to 100% (i.e. "assume healthy, take no action") if the command
    is missing or its output can't be parsed -- a guard that can't see clearly
    must never guess itself into killing something.
    """
    try:
        output = runner()
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("ram_guard_memory_pressure_failed", error=str(exc))
        return 100.0
    match = re.search(r"free percentage:\s*(\d+)%", output)
    if not match:
        logger.warning("ram_guard_memory_pressure_unparseable", output=output[:200])
        return 100.0
    return float(match.group(1))


def _default_ps_runner() -> str:
    return subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss=,etime=,stat=,command="],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    ).stdout


def _parse_etime(etime: str) -> int:
    """``ps`` elapsed time (``[[dd-]hh:]mm:ss``) to total seconds."""
    days = 0
    rest = etime
    if "-" in etime:
        day_part, rest = etime.split("-", 1)
        days = int(day_part)
    parts = [int(p) for p in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts[-3], parts[-2], parts[-1]
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def read_processes(runner: ProcessRunner = _default_ps_runner) -> list[ProcessInfo]:
    """Every process on the box, parsed from one ``ps`` call.

    Unparseable lines are skipped rather than raising -- a single odd row (a
    command containing a stray control character, say) must not blind the
    guard to every other process.
    """
    processes: list[ProcessInfo] = []
    for line in runner().splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split(None, 5)
        if len(fields) < 6:
            continue
        pid_s, ppid_s, rss_s, etime_s, stat_s, command = fields
        try:
            processes.append(
                ProcessInfo(
                    pid=int(pid_s),
                    ppid=int(ppid_s),
                    rss_kb=int(rss_s),
                    etime_seconds=_parse_etime(etime_s),
                    stat=stat_s,
                    command=command,
                )
            )
        except ValueError:
            continue
    return processes


def _protected_pids(processes: list[ProcessInfo], own_pid: int | None = None) -> set[int]:
    """This process and every ancestor of it -- never a kill candidate.

    ``own_pid`` defaults to the real running process's pid; tests inject a
    fake one so a synthetic process list can exercise the ancestor walk.
    """
    by_pid = {p.pid: p.ppid for p in processes}
    protected: set[int] = set()
    current: int | None = own_pid if own_pid is not None else os.getpid()
    while current is not None and current not in protected:
        protected.add(current)
        parent = by_pid.get(current)
        if parent is None or parent == current or parent <= 1:
            break
        current = parent
    return protected


def _lock_holder_pids(
    database_path: Path, runner: Callable[[Path], str] | None = None
) -> set[int] | None:
    """PIDs with the DuckDB file open, via ``lsof``. ``None`` means "unknown".

    Returning ``None`` rather than an empty set on failure matters: "lsof
    failed" and "nobody has the file open" must not look the same, because the
    caller treats "unknown" as "protect every DB-touching duplicate" -- the
    safe default when the guard can't actually see who holds the lock.
    """

    def _run(path: Path) -> str:
        return subprocess.run(
            ["lsof", "-t", str(path)], capture_output=True, text=True, timeout=10, check=False
        ).stdout

    run = runner or _run
    try:
        output = run(database_path)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("ram_guard_lsof_failed", error=str(exc))
        return None
    pids: set[int] = set()
    for line in output.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.add(int(line))
    return pids


# --------------------------------------------------------------------------- #
# Deciding what's safe to stop                                                #
# --------------------------------------------------------------------------- #
def _duplicate_groups(
    processes: list[ProcessInfo], protected: set[int]
) -> dict[str, list[ProcessInfo]]:
    """Group non-protected processes by the first singleton marker they match."""
    groups: dict[str, list[ProcessInfo]] = {marker: [] for marker in SINGLETON_MARKERS}
    for proc in processes:
        if proc.pid in protected or proc.is_zombie:
            continue
        for marker in SINGLETON_MARKERS:
            if proc.matches(marker):
                groups[marker].append(proc)
                break
    return {marker: procs for marker, procs in groups.items() if len(procs) > 1}


def _chain_root_pid(proc: ProcessInfo, by_pid: dict[int, int], group_pids: set[int]) -> int:
    """The topmost same-marker ancestor of ``proc``, or ``proc`` itself.

    Walks the full process table, so a chain survives an intermediate
    non-matching process (``uv`` wrapper -> ``zsh -c`` -> python worker). The
    walk is cycle-guarded because ``ps`` snapshots can contain reused pids.
    """
    root = proc.pid
    seen: set[int] = set()
    current = proc.ppid
    while current > 1 and current not in seen:
        if current in group_pids:
            root = current
        seen.add(current)
        parent = by_pid.get(current)
        if parent is None:
            break
        current = parent
    return root


def find_kill_candidates(
    processes: list[ProcessInfo],
    *,
    protected: set[int],
    lock_holders: set[int] | None,
) -> list[KilledProcess]:
    """Duplicate *launches* of a singleton command, the real one kept, rest flagged.

    A single launch is often several matching processes: ``uv run
    market-intelligence ...`` is a wrapper plus the python worker it spawned,
    and both command lines match the same marker. Treating that pair as
    duplicates is how the guard killed the live extractor's worker on
    2026-07-23 -- so matching processes are first collapsed into launch trees
    by ancestry, and only distinct trees count as duplicates. The kill target
    for a losing tree is its root (stopping the wrapper stops the launch); a
    child orphaned by a SIGKILLed root becomes its own root on the next pass.

    Whichever tree holds the DuckDB lock -- via *any* of its processes -- is
    always the one kept; that's certain knowledge, not a guess. Only when no
    tree holds the lock (ollama, or a DB-touching command that isn't mid-write
    right now) does it fall back to keeping the longest-running root, on the
    theory that an accidental re-launch is almost always the newer copy. A
    DB-touching marker is skipped entirely if ``lock_holders`` is unknown
    (lsof failed) -- better to leave a real duplicate alone than guess wrong
    on a live backfill.
    """
    candidates: list[KilledProcess] = []
    by_pid = {p.pid: p.ppid for p in processes}
    for marker, group in _duplicate_groups(processes, protected).items():
        if marker in _DB_TOUCHING_MARKERS and lock_holders is None:
            continue
        group_pids = {p.pid for p in group}
        chains: dict[int, list[ProcessInfo]] = {}
        for proc in group:
            chains.setdefault(_chain_root_pid(proc, by_pid, group_pids), []).append(proc)
        if len(chains) <= 1:
            continue  # one launch tree: a wrapper and its own workers

        roots = {p.pid: p for p in group if p.pid in chains}
        holding_roots = [
            root_pid
            for root_pid, members in chains.items()
            if lock_holders and any(m.pid in lock_holders for m in members)
        ]
        if holding_roots:
            keep_roots = set(holding_roots)
        else:
            survivor = max(roots.values(), key=lambda p: p.etime_seconds)
            keep_roots = {survivor.pid}
        for root_pid, root in sorted(roots.items()):
            if root_pid in keep_roots:
                continue
            candidates.append(
                KilledProcess(
                    pid=root.pid,
                    command=root.command,
                    rss_mb=root.rss_mb,
                    marker=marker,
                    signal_used="",
                )
            )
    return candidates


def find_zombies(processes: list[ProcessInfo]) -> list[ProcessInfo]:
    """Zombies for visibility only -- reaping one means signalling its parent,
    not the zombie itself, so this module reports them rather than acting."""
    return [p for p in processes if p.is_zombie]


def find_high_memory(
    processes: list[ProcessInfo],
    *,
    protected: set[int],
    threshold_mb: float = 1500.0,
    limit: int = 10,
) -> list[ProcessInfo]:
    """Top RSS consumers outside the protected set, for the briefing -- report
    only. Which of a user's own apps are "useless" is a judgment call this
    module doesn't make; it surfaces the candidates and leaves the kill to a
    human."""
    candidates = [
        p
        for p in processes
        if p.pid not in protected and not p.is_zombie and p.rss_mb >= threshold_mb
    ]
    return sorted(candidates, key=lambda p: p.rss_mb, reverse=True)[:limit]


def _terminate(pid: int, *, kill: KillFn, sleep: Callable[[float], None]) -> str:
    """SIGTERM, then SIGKILL after a grace period if it's still alive.

    The liveness probe goes through the same injected ``kill`` (signal 0 is
    the standard "is it alive" no-op) rather than a hardcoded ``os.kill``, so a
    test can simulate "didn't die from SIGTERM" without touching a real pid.
    """
    try:
        kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already-gone"
    except PermissionError:
        return "permission-denied"
    sleep(_KILL_GRACE_SECONDS)
    try:
        kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return "SIGTERM"
    try:
        kill(pid, signal.SIGKILL)
        return "SIGKILL"
    except (ProcessLookupError, PermissionError):
        return "SIGTERM"


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def run_guard(
    *,
    database_path: Path,
    dry_run: bool = False,
    act_below_free_pct: float = DEFAULT_ACT_BELOW_FREE_PCT,
    memory_runner: ProcessRunner = _default_memory_runner,
    ps_runner: ProcessRunner = _default_ps_runner,
    lsof_runner: Callable[[Path], str] | None = None,
    kill: KillFn = os.kill,
    sleep: Callable[[float], None] = time.sleep,
) -> GuardResult:
    """Check memory pressure and stop duplicate processes if it's actually low.

    All I/O is injected (memory/ps/lsof runners, the kill syscall, sleep) so
    tests exercise the real decision logic without touching the real machine.
    """
    free_pct = read_free_pct(memory_runner)
    processes = read_processes(ps_runner)
    protected = _protected_pids(processes)
    lock_holders = _lock_holder_pids(database_path, lsof_runner)

    candidates = find_kill_candidates(processes, protected=protected, lock_holders=lock_holders)
    zombies = find_zombies(processes)
    high_memory = find_high_memory(processes, protected=protected)

    should_act = free_pct < act_below_free_pct and not dry_run
    notes: list[str] = []
    if lock_holders is None:
        notes.append("lsof unavailable: database-touching duplicates were not evaluated")

    if not should_act:
        return GuardResult(
            free_pct=free_pct,
            acted=False,
            kill_candidates=candidates,
            zombies=zombies,
            high_memory=high_memory,
            notes=notes,
        )

    killed: list[KilledProcess] = []
    for candidate in candidates:
        signal_used = _terminate(candidate.pid, kill=kill, sleep=sleep)
        killed.append(
            KilledProcess(
                pid=candidate.pid,
                command=candidate.command,
                rss_mb=candidate.rss_mb,
                marker=candidate.marker,
                signal_used=signal_used,
            )
        )
        logger.warning(
            "ram_guard_killed_duplicate",
            pid=candidate.pid,
            marker=candidate.marker,
            rss_mb=round(candidate.rss_mb, 1),
            signal=signal_used,
        )
    return GuardResult(
        free_pct=free_pct,
        acted=True,
        killed=killed,
        zombies=zombies,
        high_memory=high_memory,
        notes=notes,
    )


__all__ = [
    "DEFAULT_ACT_BELOW_FREE_PCT",
    "SINGLETON_MARKERS",
    "GuardResult",
    "KilledProcess",
    "ProcessInfo",
    "find_high_memory",
    "find_kill_candidates",
    "find_zombies",
    "read_free_pct",
    "read_processes",
    "run_guard",
]
