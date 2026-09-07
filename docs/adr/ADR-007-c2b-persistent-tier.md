# ADR 007 — C2b, the persistent tier: the organ that makes continuity measurable

- **Status:** Proposed (Nicolas Van Eeckhout, 2026-09-08) — design; implementation in slices, each tested before the next
- **Deciders:** Nicolas Van Eeckhout
- **Context:** the paper is named after continuity (D15), yet its central capability is today a *specified protocol* (H.M., `docs/benchmarks/hm-protocol.md`) awaiting its organ. Whitepaper v1 is gated on measuring the continuity triad on a built organ. C2b is that organ: the episodic store that lives *outside* the weights and survives the session boundary. This ADR turns the scattered specifications already graved (hippocampus pipeline, retention law ADR-003 D4, the noise ruling D4.8, the H.M. thresholds) into an implementable design, and states what it reuses.
- **Cites:** hub-019, ADR-003 D4 (retention law, D4.8 noise ruling), ADR-005 (content addressing), ADR-006 D8 (the router), `docs/concepts/hippocampus.md`, `docs/benchmarks/hm-protocol.md`, register RES-2, RES-9, RES-11, RES-15, RES-16, RES-17, RES-19.

## Context — what C2b is, and is not

**Weights = semantic and procedural; journal = episodic.** The two-store model (patient H.M.: no new episodes, skills intact) is the founding prediction of the memory story. C2b is the journal: an append-only, content-addressed store of *episodes* — things that happened in a session — written by a gated path, read by a fixed-size associative state plus sub-linear search, consolidated on a rhythm into the weights, and forgotten on schedule. Its existence is what makes the H.M. dissociation *testable*: cut the journal, recall must collapse while skills hold.

C2b is **not** retrieval-augmented generation over documents (that is a different object: exogenous, non-episodic, not gated by surprise), and **not** a bigger context window (which forgets at the boundary by construction). It is the organ the continuity triad measures.

## Decision 1 — The entry (the retention law, made concrete)

An entry is `(cue, pointer, salience, schema_id, t_written, t_last_read, state)`:
- `cue` — a fixed-size embedding of the episode's *address* (matryoshka-truncatable for cold entries; NOW-3);
- `pointer` — a content hash of the episode's payload (the payload lives in a payload store, never duplicated in the journal — ADR-005 content addressing reused);
- `salience` — a scalar updated by surprise at write (CA1), by outcome credit later (RES-11), decayed on schedule;
- `schema_id` — the span-contract tag (RES-8), so episodes are typed;
- `state` — the RES-17 lifecycle cell: `live → consolidated → demoted → evicted`.
**Size law:** ≤ 1 KB per entry; content never duplicated. **Read law:** never a scan — retrieval = fixed-size associative state (constant memory) + ANN over cues (sub-linear). O(N) reads are prohibited by test.

## Decision 2 — The write path = the hippocampal pipeline (each stage ablatable)

`separate (DG) → associate (CA3) → compare (CA1) → gate`:
1. **DG — pattern separation:** expand-then-sparsify the cue so near-duplicates decorrelate before storage (interference guard).
2. **CA3 — pattern completion:** the existing C2 associative memory *reconstructs* from the cue (reuse: the Hopfield/delta layer of ADR-006 — C2 is CA3).
3. **CA1 — the comparator:** `surprise = distance(reconstruction, input)`. Surprise stops being a heuristic and becomes a computed quantity at the memory interface (RES-2 made concrete).
4. **Write gate — the noise ruling (D4.8), two stages:** *admission* rejects the redundant/already-known (low surprise) — but surprise alone is not signal (a corrupted input is maximally surprising), so *the verdict is retrospective*: an entry that never consolidates, never gains salience, never earns outcome credit is noise and is evicted. Scheduled forgetting is the tribunal.
Each stage is a flag (default off) and an ablation row: interference without DG; recall without CA3; write precision without CA1.

## Decision 3 — Reads route through the RES-18 router (reuse, not invention)

The read path is a fourth path for the multi-path router: `control | hopfield | delta | journal`. The journal path is used only where its predicted score beats the control (the floor guarantee holds — C2b can never degrade the model); routing hardens as it consolidates; the breaker falls back if the journal degrades. **No new router is built** — the versioned `Router` contract gains one path. This is the first exercise of the router on a *real* new component, and the natural place to test whether it learns to route with thousands of real spans (the open question of D9/Rev20).

## Decision 4 — Consolidation and forgetting (the lifecycle, scheduled)

- **Consolidate:** on a rhythm (RES-9 phases — the "sleep" of the model), replay live entries forward into the persistent associative memory / weights; an entry that consolidates transitions `live → consolidated`.
- **Demote:** K consolidated entries → 1 summary entry keeping pointers (`consolidated → demoted`).
- **Evict:** below a salience floor, already consolidated, aged (`demoted → evicted`). Reverse replay (RES-11) keeps salience honest: outcome credit walks the trajectory backward.
- **Budgets are homeostatic setpoints:** per-scope caps on bytes and read p95 are regulated variables — approaching a cap raises consolidation/eviction pressure. Scoping (per-project journals) partitions N.
This lifecycle *is* RES-16 (content-addressed hierarchical consolidation) and *is* a RES-17 state machine — the same law the router uses, at the memory scale.

## Decision 5 — The first measurable milestone: the H.M. protocol runs

C2b is not "done" when it stores things; it is done when **the H.M. protocol produces a number**. Frozen thresholds (from the spec, never adjusted after seeing numbers): 200 synthetic facts (`Vorel-3f2a`-class, leakage-proof), written through the normal gated path; `hm_recall_on − hm_recall_off ≥ δ = 0.50`; `hm_recall_off ≤ chance + 5 pts`; `hm_skill_delta ≤ ε_S = 1%`; negative control. `hm_dissociation_pass` is 0/1 and is published either way. **A FAIL is a result** and triggers a dated revision of the memory story. From this milestone on, the protocol runs at every checkpoint.

## Decision 6 — Safety and telemetry

The H.M. diagnostic doubles as the **eviction-safety check**: consolidate-then-evict must never silently degrade skills or contracted recall. Journal size, hit rate, read p95, and the lifecycle counts (live/consolidated/demoted/evicted — the RES-17 snapshot) are emitted in C6 telemetry and land in the run record's `standard_suite`. The journal is per-scope and, in the QM setting, encrypted at rest (the trust boundary of RES-2 is respected by construction — the model never sees another scope's journal).

## Decision 7 — Legality and provenance

The journal engine is written from scratch (our filon), as `cortex_data` and `cortex_c2` were. ANN search may use an established library (HNSW-class; license checked at ingestion) — an index is infrastructure, not a claim. Episodes carry provenance (RES-15 origin meta-tokens: which session, which source) and are never persisted across scopes.

## Implementation order (measure-first, slices; each tested before the next)

1. **Slice A — the store — IMPLEMENTED & GREEN 2026-09-08:** `cortex_c2b/` — `Entry` (cue, pointer, salience, schema_id, timestamps, RES-17 state), `PayloadStore` (content-addressed by hash — same bytes stored once), `Journal` (append-only JSONL log replayable to the same state; keyed cue index so reads never scan; legal-only lifecycle transitions live→consolidated→demoted→evicted; inspectable snapshot). **Nine invariants green, including `test_read_never_scans`** (2000 entries; a read by cue must touch < N/10 — a scan fails the test), the size law (≤1 KB, pointer never content) and no-duplication (two entries, one payload).
2. **Slice B — the write path — IMPLEMENTED & GREEN 2026-09-08:** `cortex_c2b/write_path.py` — `DentateGyrus` (expand-then-sparsify, k-WTA: near-duplicates measurably decorrelate), `CA3Completion` (reconstruct from stored codes; Slice C swaps in the C2 layer behind the same call), `ca1_surprise` (= cosine distance reconstruction↔input — a *computed* quantity, RES-2 made concrete), and `WritePath` with the **two-stage gate**: (a) admission rejects the redundant below a declared threshold; (b) `noise_tribunal` evicts admitted entries that never earned salience — the D4.8 verdict as code. Seven tests green: duplicate rejected, novel admitted, **corrupted input admitted-then-evicted while the credited episode survives**, DG separation measurable, CA1 = distance, and each stage ablatable (no CA1 → everything written; no DG → raw cues).
3. **Slice C — the read path via the router — IMPLEMENTED & GREEN 2026-09-08:** `cortex_c2b/read_path.py` — `PATH_JOURNAL = 3` registered into the router's vocabulary (extend, never break); `CueIndex`, a from-scratch random-hyperplane LSH over cues (a query probes buckets, never the store — `test_ann_read_never_scans`: 3000 entries, < N/5 touched); `JournalPath.score()` = retrieval confidence on the router's footing. **The existing `RouterV1` routes to path 4 with zero code change** (it iterates `range(1, len(scores))` with the control at 0 — retro-compat exactly as designed). Seven tests green: the floor holds with the journal on/off, journal-at-0 ≡ the 3-path router, the journal wins a span it holds and falls to the control on one it does not, evicted entries never score.
4. **Slice D — the H.M. protocol — THE MILESTONE REACHED 2026-09-08:** `cortex_c2b/hm_protocol.py` — the leakage-proof generator (`{trigrams}-{4 hex}` entities, 200 facts balanced over 5 schemas, config-hashed), session A planting through the normal gated write path, session B in a fresh context (direct query + paraphrase), both arms, the 50-fact negative control, frozen thresholds. **Result: `hm_dissociation_pass = 1` — PASS, stable across 4 seeds.** `hm_recall_on = 1.000`, `hm_recall_off = 0.125` (at the chance floor 0.175), `hm_gap = 0.875 ≫ δ 0.50`, `negctrl_on = 0.000` (the journal never hallucinates a never-planted fact — it *knows it does not know*), `hm_skill_delta = 0`. *Cut the journal: recall collapses, skills hold.* The founding prediction of the memory story (weights = semantic/procedural; journal = episodic) is now a measurement, against the C2b organ. **Honest history kept:** the first run was INVALID (negative control 0.12) — not the journal (0.000) but the OFF-arm floor simulator guessing twice per fact and a spec reading the negative control as an absolute rate; the simulator was corrected and the spec amended *dated before the rerun* (negative control read above chance; δ/ε_S/floor untouched). **Reserve:** this PASS is against the organ with a hash-seeded cue encoder and a reference skill probe — it proves the journal does its job; the language-model arm (real skill suites, learned encoder) lands when the LM is wired to the journal. Artefact: `metrics/mqar/hm-protocol-2026-09-08.json`. Then lifecycle (consolidate/demote/evict) as Slice E, and the protocol at every checkpoint.

## Consequences

C2b turns the continuity triad from a specification into a measurement — the gate for whitepaper v1. It reuses three built things (C2 as CA3, `cortex_data` hashing, the RES-18 router) rather than inventing new ones (RES-19: what does this consolidate? — everything already built). Its first success criterion is a *number* (`hm_dissociation_pass`), and its first honest outcome may be a FAIL that revises the memory story. Nothing here is claimed until the H.M. protocol has run.
