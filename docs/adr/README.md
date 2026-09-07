# Decision records — the local series

One decision per file, in the quantum-meridian discipline: a record is written when the
decision is taken; an amendment gets its own dated paragraph and prior text is kept —
errors are noted, never erased. Ecosystem-level decisions stay in the hub: the founding
decision is **hub decision 019** (`quantum-meridian/docs/decisions/019-quantum-cortex-architecture.md`),
and the full decision log **D1–D18** is mirrored in `quantum-meridian/docs/reference/quantum-cortex.md`.

| # | Decision | Status |
|---|---|---|
| ADR-001 | [Metrics are first-class citizens, by design](ADR-001-metrics-first-class.md) | accepted (2026-08-07) — schema `run-v1` shipped; **ledger live**: N1 control `be1fa8139f59` committed (val_ppl 2.601) |
| ADR-002 | [Training stack, and where portability actually applies](ADR-002-training-stack-portability.md) | proposed (2026-08-07) |
| ADR-003 | [The brain-derived feature map: validated, consistent, performance-guarded](ADR-003-brain-feature-map.md) | accepted (2026-08-08) — 26 rows; features enter by wave order, flags default-off |
| ADR-004 | [The evaluation law: a Pareto frontier, never a scalar (D16)](ADR-004-evaluation-pareto.md) | accepted (2026-09-06) — + linked D17 (open-results, revises D6) |
| ADR-005 | [The data composition layer (NOW-7)](ADR-005-data-composition-layer.md) | accepted (2026-09-06) — **Slice 1 implemented & green**: `cortex_data/` (hashed operator DAG, lazy streaming, five-invariant harness); wired into `train.py` (`mix_hash` in record + schema) |
| ADR-006 | [The C2 ablation protocol (N2)](ADR-006-c2-ablation-protocol.md) | accepted (2026-09-06) — Slices A+B **implemented & green** (C2 layers, circuit breaker D7, MQAR + serial-position); **first result** (hopfield +14% AUC, delta +2%); + D7 circuit breaker, D8 multi-path C2 router (**RES-18 Slice A implemented**) |

Promotions from `../IDEAS-REGISTER-2026-08-07.md` are dated amendments citing the register — never a silent edit.

**Implementation status of ideas** is tracked at the top of the register (`../IDEAS-REGISTER-2026-08-07.md`) — the code that exists is marked there, distinct from designs awaiting their wave.
