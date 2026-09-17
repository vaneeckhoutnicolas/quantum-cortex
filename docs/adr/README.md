# Decision records — the local series

One decision per file, in the quantum-meridian discipline: a record is written when the
decision is taken; an amendment gets its own dated paragraph and prior text is kept —
errors are noted, never erased. Ecosystem-level decisions stay in the hub: the founding
decision is **hub decision 019** (`quantum-meridian/docs/decisions/019-quantum-cortex-architecture.md`),
and the full decision log **D1–D23** is mirrored in `quantum-meridian/docs/reference/quantum-cortex.md`.

| # | Decision | Status |
|---|---|---|
| ADR-001 | [Metrics are first-class citizens, by design](ADR-001-metrics-first-class.md) | accepted (2026-08-07) — schema `run-v1` shipped; **ledger live**: N1 control `be1fa8139f59` committed (val_ppl 2.601) |
| ADR-002 | [Training stack, and where portability actually applies](ADR-002-training-stack-portability.md) | proposed (2026-08-07) |
| ADR-003 | [The brain-derived feature map: validated, consistent, performance-guarded](ADR-003-brain-feature-map.md) | accepted (2026-08-08) — 26 rows; features enter by wave order, flags default-off |
| ADR-004 | [The evaluation law: a Pareto frontier, never a scalar (D16)](ADR-004-evaluation-pareto.md) | accepted (2026-09-06) — + linked D17 (open-results, revises D6) |
| ADR-005 | [The data composition layer (NOW-7)](ADR-005-data-composition-layer.md) | accepted (2026-09-06) — **Slice 1 implemented & green**: `cortex_data/` (hashed operator DAG, lazy streaming, five-invariant harness); wired into `train.py` (`mix_hash` in record + schema) |
| ADR-006 | [The C2 ablation protocol (N2)](ADR-006-c2-ablation-protocol.md) | accepted (2026-09-06) — **N2 complete on the C2 axis**: first ablation (hopfield +14% AUC, delta +2%; per-tier 6/2/3), D7 circuit breaker (proven), D8 multi-path router **built A→B→C→D** (learned router 88–93% of oracle on a synthetic task), **D9 Level 1** (multi-seed CIs + paired tests; router on real spans *retrograded* to ~25%; external anchor → recurrent ladder). Confirmation run (5-seed) pending → whitepaper v0 |
| ADR-007 | [C2b, the persistent tier: the organ that makes continuity measurable](ADR-007-c2b-persistent-tier.md) | proposed (2026-09-08) — design in slices A→E, all five implemented and green (E on 2026-09-12); **D8 (2026-09-12): the journal survives a full process restart, persistence declared per component, sealed at rest with the hub's QJE1 framing, sealed by default**; **D9 (2026-09-12): durable first, a storage policy per scope (stop, read_only, memory), free disk space regulated, the protocol reports `persistent` and `claimable`**; **D10 (2026-09-12, decided, not built): sequenced asynchrony, lazy execution of actions decided at a sequence point, with a precondition barrier**; a trade-off declaration next to the cost declaration; reuses C2 (as CA3), `cortex_data` hashing, and the RES-18 router (journal = path 4); **milestone = the Molaison (H.M., after the patient Henry Molaison; defined in docs/benchmarks/hm-protocol.md) protocol produces `hm_dissociation_pass`** (the gate for whitepaper v1) |
| ADR-008 | [The journal in the decode loop: cue encoder, reader, and the cite or abstain contract](ADR-008-journal-in-the-decode-loop.md) | implemented and green on CPU (2026-09-12), amended twice on 2026-09-13 (minimal pair negatives; the pair keeps its shape) and twice on 2026-09-14 (RES-21, the matching head, declared before its run; the decode policies, the closing test of step 4, declared before any measurement), and on 2026-09-16 (the external model arm through a frozen prompt adapter, declared before any measurement, validated 2026-09-17, the models pinned by commit sha in a dated addendum) -- nine invariants validated before the code; the model reads its journal as content through a zero init cross attention in one block, on the router's gate; a learned cue encoder trained by contrast; the curriculum's targets computed from what the journal returned (cite what was shown, else abstain); the protocol's LM arm with strict recall, invalid citation at most 1 %, an attributable failure; **the first measurement is REPRISE step 4 (GPU)** |

Promotions from `../IDEAS-REGISTER-2026-08-07.md` are dated amendments citing the register — never a silent edit.

**Implementation status of ideas** is tracked at the top of the register (`../IDEAS-REGISTER-2026-08-07.md`) — the code that exists is marked there, distinct from designs awaiting their wave.

**The whitepaper** (`../WHITEPAPER.md`) is a specified deliverable with an admission gate: a claim enters a results section only if it is statistical (≥3 seeds, significant), anchored, demonstrated on the real task, reproducible from a committed artefact, and honestly bounded. Findings that fail live as explicit open questions.
