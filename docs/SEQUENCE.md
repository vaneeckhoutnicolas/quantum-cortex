# SEQUENCE — what gets integrated, stage by stage

**Purpose.** The build order is normatively scattered across `roadmap.md` (steps and gates), `adr/ADR-003-brain-feature-map.md` (the 26-row map and waves), the `IDEAS-REGISTER` (26 entries), and `FEATURES` (F1–F12). Two months later, reconstructing the micro-sequence from four files is unreasonable. **This document is the synthesis: one page, every stage, everything that ships in it.** Rule: it *synthesizes with pointers, never overrides* — if this page and a normative source disagree, the source wins and this page gets a dated fix. It is regenerated at every gate passage.

**Status stamp:** 2026-09-05 — E1 done (both repos live, hub decision 019 recorded), E2 green — **2 successful checks confirmed** on commit `59976a4`: `ci/smoke-train` in 44 s, `ci/validate-ledger` in 9 s. Next action: **E3**.

## Track 1 — Execution (E-list, from `EXECUTION-2026-08-07.md`)

| Step | What happens | Status |
|---|---|---|
| E1 | Repo born: extract, init, push; hub records decision 019 + reference page | ✅ 2026-09-05 |
| E2 | CI self-test: `validate-ledger` + `smoke-train` green on every push | ✅ 2/2 (smoke 44 s, ledger 9 s) |
| E3 | First real run: Kaggle notebook (T4 + Internet, ~2 h, 25.8M control on 500M FineWeb-Edu byte-tokens) | ⬅ next |
| E4 | Sanity read: `completed`, `anomalies: null`, finite ppl — the run *defines* the baseline | pending |
| E5 | Commit the record + `--regen-latest` → **N1 gate passes**. Proposed here: repo flips **public** at this step (founder's call) | pending |
| E6 | GATE → EuroHPC **Playground** application citing the run_id (portal re-verified at submission) | pending |
| E7 | Stage 1 on EuroHPC: Apptainer, pre-staged data, `provider:"eurohpc"` | pending |

## Track 2 — Build waves (from ADR-003; features from FEATURES/register)

### Pre-N2 — mandatory prerequisite
- **NOW-7 · Data first-class**: the mix becomes declared, config-hashed, ablatable; decontamination policy active. Source leads: OLMo's open mixes; later fed by **RES-12** (legal distillation, teacher mix published).

### N2 = W1 — the C2 core family (memory in the weights)
- Integrate **two associative variants against the control**: Hopfield/energy (RES-1) **and** gated delta-rule (extraction K1–K2 as the numeric starting shape) — both as *stackable attention-layer citizens* (hub-019 constraint, promoted into the hub decision at this step).
- **F1** optimizer ablation: NorMuon-class vs AdamW (re-implemented from papers, no code copied).
- Benchmarks live: **MQAR** via zoology as a dependency (F5–F6), **serial-position curve** (Rev7.b).
- Gate per variant: advancement rule (≥1 capability win, ≤2% ppl, tolerance frozen pre-run).

### N3 = W1b — the C3 family (volatility)
- **NOW-1** oracle-in-pretraining curriculum (token space already reserved in the tokenizer since N1).
- **Thalamic admission gateway** (C3's front door, one admission policy) and **collicular interrupt shortcut** (mid-pass modulation).
- Benchmark: oracle-shift recovery. Law: **NOW-9** (signed oracle provenance + source trust) is binding *before any public C3 endpoint*.

### N4–N5 = W2 — persistent memory, contracts, economics
- **C2b persistent tier**: entry-format law (pointers never blobs), the **DG → CA3 → CA1 write pipeline** (separate → associate → compare; surprise becomes computed), retention law incl. **D4.8** noise ruling (admission by surprise, verdict by consolidation).
- **H.M. protocol goes live** as the standing diagnostic (thresholds frozen in `benchmarks/hm-protocol.md`), including the eviction-safety variant.
- **RES-8**: typed span references + per-span contract stack (C4) + typed diffs; **habit cache** (C6 extension); **NOW-2** speculative graft (the cerebellum drafter) with its quality-per-dollar curve; **NOW-3** graft fidelity across matryoshka resolutions; **F8** DVC for slices/checkpoints.
- Deliverable: **ablation report v0** published with losses → **gates open: EuroHPC Fast Lane + TRC applications** citing run_ids.

### Post-N5 = W3 — regulation and behavior
- **RES-9 endocrine bus** (stress/arousal/reward; homeostat on entropy and budgets; consolidation rhythms; salience single-source), **heterogeneous experts + mixture-of-depths** (Rev7.a), **insula** telemetry-as-input, **RES-4** trust-weighted routing, **NOW-8** organ-use post-training (consult / cite / heed / comply), **NOW-5** the live **continuity demo** (ships once C3 passes).

### W4 — research wave
- **RES-10** connectome (commissure, wiring budget, connectivity profiles), **TEM** structure/content factorization (RES-3), **RES-11** reverse-replay credit assignment, **RES-13** latent recursive refinement, **RES-6** cache-oblivious planning + **RES-5** heterogeneous placement. **RES-7** (ternary, capability-per-bit) is wave-flexible: it can run as a parallel ablation any time after N2.

### Parked (SPEC — unlock conditions on file)
SPEC-1 dual-stream determinism · SPEC-2 oracle landscape editing (after RES-1 data) · SPEC-3 φ/Zeckendorf splitting (after RES-6 exists) · SPEC-4 private-inference flag (after N5 + a named use case).

## Always-on laws (no stage, no exceptions)
A run without its committed record does not exist · every brain-derived feature ships behind a flag, default off · advancement rule per feature · portability (ADR-002 / NOW-4) binding · anti-plagiarism protocol binding (core from scratch) · unknowns are null · **the north metric is the continuity triad (D15): H.M. persistence + oracle-shift revision + reference ratio — never blended**.

## Pointers
Normative: `roadmap.md` · `adr/ADR-003-brain-feature-map.md` · `IDEAS-REGISTER-2026-08-07.md` · `FEATURES-2026-08-07.md` · `EXECUTION-2026-08-07.md` · hub `decisions/019` + `reference/quantum-cortex.md`. Deep-dive schedule: `concepts/README.md`.
