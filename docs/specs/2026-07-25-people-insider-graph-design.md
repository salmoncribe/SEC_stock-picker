# People, Boards & Insider Graph — Design

**Date:** 2026-07-25
**Status:** Proposed
**Scope:** Add a first-class `Person` entity (officers, directors, 10%+ owners) so the existing
company graph can express who runs and sits on the board of each company, and derive two new
signals — insider cluster-buying and interlocking directorates — from data the platform
**already ingests**. Then extend collection to 8-K executive-change events and institutional/
activist ownership flow.
**Non-goal:** No order execution, no position sizing, no investment advice — same boundary as
[[2026-07-22-signal-graph-design]]. This doc extends that graph; it does not replace it.

---

## 0. What already exists (verified against code, 2026-07-25)

- `collectors/insider.py` / `clients/insider.py` parse SEC's full quarterly Form 3/4/5 bulk
  datasets (SUBMISSION/NONDERIV_TRANS/REPORTINGOWNER) into `insider_transaction` events.
  Real, tested (887 lines), CLI-wired (`sec sync-insider`). Owner name/title/relationship
  are already present in the raw data this collector reads — they are just not promoted past
  an opaque JSON payload field.
- **No `Person` entity exists anywhere.** `entities.py`'s `CompanyIndex` resolves company
  names only.
- `config/forms.yaml` already allow-lists `DEF 14A`, but nothing collects or parses it —
  it's a dead config line.
- `extraction/sections.py` deterministically splits 10-K/10-Q Item sections only; 8-K bodies
  are not sectioned at all. `collectors/filing_events.py` tracks 8-K item **codes**
  (including `5.02`) as direction-less, person-less events.
- `autopilot/graph.py` projects one Obsidian note per company with wikilinks for the 4
  existing company-edge types (customer/supplier/competitor/partner) only.
- Two decisions from [[2026-07-22-signal-graph-design]] make most of this cheap:
  **(E) event taxonomy is an open registry** (`events.event_type` is a string, not an enum —
  new types are config, not migrations) and **(edge_type` on `graph_edges` is likewise open)**.
  New signals below reuse both tables verbatim.

★ The single biggest unlock: because Form 3/4/5 already gives every officer/director a
persistent SEC `reportingOwnerCIK`, promoting that into a `Person` entity makes interlocking
directorates and cluster-buy detection **pure derivations over data already in DuckDB** — no
new network fetch, no new LLM call, no new schema beyond one entity and one join table.

## 1. Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| A | Person identity key | **SEC `reportingOwnerCIK`** | Persistent, already present on every Form 3/4/5 row this platform already stores. Avoids name-matching for the insider population entirely. |
| B | Role storage | **New `role_memberships` table, not a payload field** | A person↔company relationship needs to be queried (joins for interlocks, roster lookups), not just displayed — payload JSON isn't queryable in DuckDB without unpacking. |
| C | Interlocking directorates | **Reuse `graph_edges` with `edge_type = shared_board_member`** | No schema change; inherits the existing candidate→active→dormant lifecycle, dedup, and Obsidian projection for free. |
| D | Insider cluster-buy | **Reuse `events` with `event_type = insider_cluster_buy`** | Same reasoning — flows through the existing `signals/dataset.py` → `signals/impact.py` gate unchanged. |
| E | 8-K executive changes (Phase 4) | **New collector, LLM-extracted** | Unlike A–D this needs a genuine new fetch (8-K body text isn't sectioned yet) and new inference — it does not come for free. Built as its own phase so Phase 3 can ship and get validated first. |
| F | Non-insider people (proxy-only board members, Phase 6+) | **Deferred, DEF 14A parsing not built yet** | Form 3/4/5 already captures anyone who is *currently* an officer/director/10%-owner — the only gap is people named in a proxy who file no Form 3/4/5 themselves (rare — a board seat almost always triggers one). Low value for high (LLM+HTML) cost; revisit only if Phase 3 proves the graph valuable. |
| G | Person name display | **Denormalize `canonical_name` on `people`, keep raw variants** | Form 4 filer names aren't consistently formatted across filings/years; store the most recent as canonical and keep a variants list rather than attempting fuzzy dedup logic now. |

## 2. Architecture

Following the established `clients → collectors → validators → storage → entities` layering
and the existing `analytics/` convention for pure derivations.

**Phase 3 (build first — no new network calls):**
- `entities.py` — add `PersonIndex` alongside the existing `CompanyIndex`, keyed on
  `reporting_owner_cik`.
- `collectors/roles.py` — re-projects `role_memberships` from the REPORTINGOWNER rows
  `collectors/insider.py` already fetched and stored; no new HTTP calls, purely a re-read +
  upsert. Idempotent on `(person_id, company_id)`.
- `analytics/interlocks.py` — pure function: self-join `role_memberships` on `person_id` where
  `company_id` differs → candidate `graph_edges` rows, `edge_type='shared_board_member'`.
- `analytics/cluster_buy.py` — pure function: group existing `insider_transaction` events
  (`transaction_code = 'P'`) by `(company_id, rolling N-day window)`, count distinct
  `person_id`s → candidate `events` rows, `event_type='insider_cluster_buy'`.
- `autopilot/graph.py` — extend the wikilink allow-list to include `shared_board_member`;
  add a new render path: `<vault>/people/<canonical_name>.md` with officer/director links to
  every company in `role_memberships`.
- CLI: `people sync-roles`, `people project-interlocks` — mirrors existing `sec sync-*` /
  `signals project-graph` naming.

**Phase 4 (new fetch + LLM, build second):**
- `clients/edgar_fts.py` — thin client for `efts.sec.gov/LATEST/search-index`, filtered to
  `forms=8-K` and a text match on `"Item 5.02"`, to avoid downloading the full 8-K universe.
- `extraction/eightk.py` — 8-K body extraction. Simpler than `sections.py`: 8-Ks are short,
  no multi-item TOC-artifact problem to solve.
- `collectors/executive_changes.py` — orchestrates fts prefilter → body fetch → Ollama
  extraction (same `LLMProvider` protocol `collectors/relationships.py` already uses, new
  schema-constrained prompt) → `events` row, `event_type='executive_change'`,
  `payload={person_name, person_cik: nullable, role, action: departure|appointment}`.
  `person_cik` is left null until/unless the name resolves against `PersonIndex` — new
  appointees often have no Form 3 on file yet at 8-K filing time, so **do not** invent a CIK;
  follow the same conservative-refusal discipline `CompanyIndex` already uses for ambiguous
  company names.

**Phase 5 (structured, quarterly cadence — build third):**
- `clients/thirteenf.py`, `collectors/institutional.py` — 13F bulk XML → position-delta
  events (`event_type='institutional_position_change'`), no new entity type needed yet
  (filer name/CIK stored on the event payload; a full `institutions` entity is deferred until
  Phase 3's `Person`/`role_memberships` pattern proves worth repeating).
- Same clients extended for 13D/13G (XML-structured since Dec 2024) →
  `event_type='activist_stake'` / `beneficial_ownership_change`.

**Phase 6 (defer, revisit only if 3–5 pay off):** DEF 14A LLM extraction (compensation,
related-party transactions — the one thing Form 3/4/5 genuinely can't give you), Congressional
STOCK Act trades (free third-party JSON, needs ticker-cleanup crosswalk), Wikidata/USPTO
enrichment for exec bios.

## 3. Schema (Phase 3 only — Phases 4/5 add no new tables, only new `event_type`/`edge_type` values)

All new tables carry the standard provenance block (`source`, `source_url`, `content_hash`,
`schema_version`, `validation_status`, `validation_errors`, `collected_time`).

**`people`** — natural key `reporting_owner_cik`
`person_id` (PK), `reporting_owner_cik`, `canonical_name`, `name_variants` (JSON array),
`first_seen_filing_date`, `last_seen_filing_date`.

**`role_memberships`** — natural key `(person_id, company_id)`
`role_id` (PK), `person_id`, `company_id`, `is_officer` (BOOLEAN), `is_director` (BOOLEAN),
`is_ten_pct_owner` (BOOLEAN), `latest_officer_title`, `first_seen`, `last_seen`,
`source_filing_count`. Flags are OR'd across every qualifying filing seen for that person at
that company — a person who files once as director and later as an officer keeps both flags
true, since both were real at different times and the graph is not trying to reconstruct
tenure precisely in Phase 3.

No new tables for `shared_board_member` edges or `insider_cluster_buy` events — see Decisions
C/D. They are new rows with new `edge_type`/`event_type` string values in the tables built for
[[2026-07-22-signal-graph-design]].

## 4. Validation (reuses both existing layers, nothing new to invent)

1. **Record-level** (existing `validation_status` contract): `role_memberships` rows are
   `rejected` if `person_id`/`company_id` don't resolve, same as any other record type.
2. **Signal-level** (existing discovery/holdout gate in `signals/impact.py`): both new derived
   signals are just new `(event_type, edge_type)` cells in the same conditional-impact table.
   `insider_cluster_buy` should be an easy positive control alongside the existing single-Form-4
   control — cluster buying is a *more* concentrated version of the same documented effect
   (Seyhun; Cohen, Malloy & Nguyen), so if single insider purchases already clear the gate,
   clustered ones should clear it with a larger effect size. If they don't, that's a signal the
   clustering window or threshold is mis-tuned, not that the effect is absent.
3. `shared_board_member` edges enter the graph lifecycle at `candidate` like any other edge and
   only reach `active` (and therefore only appear in the vault / fire alerts) once its own
   `(event_type, edge_type)` cell is admitted — an interlock by itself is not a trading signal,
   it's a channel; it only matters paired with an admitted event type propagating across it.

## 5. Data quality notes specific to people

- **Name collisions.** Two different real people can share a name. Because identity is keyed
  on `reporting_owner_cik` (Decision A), this is a non-issue for anyone with a Form 3/4/5 on
  file — it's only a risk in Phase 4 (8-K names, pre-Form-3) and is handled by leaving
  `person_cik` null rather than guessing (Decision E detail above).
- **Title/role staleness.** `latest_officer_title` is exactly that — latest, not historical.
  Phase 3 does not attempt tenure reconstruction (e.g., "was CFO 2019–2022, now CEO"); that's
  a Phase 6+ problem if the graph proves valuable enough to justify it.
- **Interlock noise.** Mutual fund trustees and professional board-sitters can rack up dozens
  of `role_memberships` rows and would otherwise flood the interlock graph with low-information
  edges. `analytics/interlocks.py` should cap or flag persons above a configurable
  company-count threshold rather than silently emitting a fully-connected hub.

## 6. Build order

| Stage | Deliverable | Gate to proceed |
|---|---|---|
| 3a | `people` + `role_memberships` schema, `PersonIndex`, `collectors/roles.py` | Re-projection from existing insider data is idempotent; row counts reconcile against distinct `reportingOwnerCIK`s already in `insider_transaction` |
| 3b | `analytics/interlocks.py` → `shared_board_member` candidate edges | Spot-check 10 known real-world interlocks (e.g., shared directors across watchlist companies) resolve correctly |
| 3c | `analytics/cluster_buy.py` → `insider_cluster_buy` candidate events | Clustered stats (per [[2026-07-22-signal-graph-design]] §6) computed correctly; passes as a second positive control alongside single-Form-4 |
| 3d | Obsidian person notes + interlock wikilinks | Vault regenerates from empty; person notes link to every company in their `role_memberships` |
| 4 | 8-K Item 5.02 executive-change extraction | fts prefilter + LLM extraction spot-checked on 20 real 5.02 filings |
| 5 | 13F / 13D / 13G ownership-flow events | Structured parse reconciles against SEC's own bulk-file row counts |

## 7. Risks

| Risk | Mitigation |
|---|---|
| Interlock graph becomes a low-signal hairball (professional board-sitters) | Configurable company-count cap on person nodes before edge emission (§5) |
| Cluster-buy window/threshold picked arbitrarily | Tune against the discovery split like every other cell, not by inspection |
| 8-K name resolution creates duplicate person records | Never mint a `person_id` from an 8-K name alone; hold as nameless-payload event until a Form 3 CIK match exists |
| DEF 14A temptation creeps back in scope | Explicitly deferred (Decision F) — Form 3/4/5 covers the roster; only revisit for compensation/related-party text |

## 8. Explicit non-goals

No order execution, no broker API, no position sizing, no investment advice — same boundary
as the rest of this platform. This doc only adds *who* to a graph that already models *what*
and *between which companies*.

## 9. Amendment (2026-07-26): the vault is a curated read surface, not a database mirror

The first Phase 3 build (2026-07-26) was structurally correct but rendered every entity
unconditionally: 106,265 person notes, including institutional 10%-owner funds that are not
"connections" in the CEO/board sense, with no notion of whether a fact was still current.
Corrected via direct product feedback the same day. Two rules now govern **everything the
vault renders**, on top of the schema in §3 (no schema change — this is a projection/render
change only):

**9.1 Node/fact inclusion is earned, not assumed.** The vault exists to "put info together and
map out ideas to create edges" (verbatim) — it is not a general-reference graph. A `Person`
gets their own note **only if** they (a) create a `shared_board_member` interlock (sit on 2+
tracked companies), or (b) are tied to a currently-live/unresolved event (see §9.2). This bar
applies to **officers as well as directors** — a title alone (CFO, director) is not sufficient;
a person who holds a role but generates no interlock and no notable event activity does not get
a page. Pure 10%-owner institutional filers are dropped from the people graph entirely — they
are ownership-flow data (Phase 5 territory), not board/executive connections. Everyone who
doesn't clear this bar stays out of Obsidian entirely (`role_memberships` is still fully
queryable in DuckDB) because the vault is an AI reasoning surface, not a reference mirror.
Company notes should spend context only on people/facts that could plausibly help explain why
the stock might move.

**9.2 Relevance is resolution state, not a calendar cutoff.** Rejected an initial proposal of a
fixed day-count TTL (e.g. "drop events after 90 days") — a fact from a year ago can still matter
for a slow-forming thesis that hasn't triggered yet. The correction is not "keep everything
forever"; it is "keep only things with an unanswered or measured question." Every event/fact
rendered in the vault is either **live** (open, unresolved, worth watching) or **graded**
(resolved with measured evidence that can serve as precedent). Resolved-but-ungraded history
stays queryable in DuckDB but is omitted from Obsidian.
- An `insider_cluster_buy` event is **live** until its longest evaluated forward horizon (the
  20-day horizon per the base dataset builder) has elapsed. After that, it is rendered only if
  `event_samples` has a graded outcome (hit/miss, realized abnormal return); otherwise it is
  database history, not AI context.
- A `role_membership` (and the interlock derived from it) is not enough by itself. Interlock
  edges render only once the signal ladder has an **active** propagation cell for
  `shared_board_member`; until then they are extraction/search space, not a reason to spend
  vault context.
- Company and person notes render **Watching** for live facts and **Resolved** only for graded
  facts kept as measured precedent.

**9.3 Consequence: the vault writer must delete, not just add.** Because inclusion criteria can
tighten (as they did here), `autopilot/graph.py` must diff the note set it is about to write
against what's currently on disk and remove files for entities that no longer qualify — the
original writer only ever created/overwrote files. This is required for §9.1 to have any
effect on a previously-over-generated vault, and is required going forward any time an
entity/event drops out of relevance per §9.2.
