# quantum-cortex — documentation index

**Start with [`RESULTS.md`](RESULTS.md)** — every number, its artefact, its test, its status (measured / gated / held / open). Then [`WHITEPAPER.md`](WHITEPAPER.md) for the argument and [`SEQUENCE.md`](SEQUENCE.md) for what comes next.


Doc organization follows the quantum-meridian discipline (named reading paths, ADRs, glossary, thin roadmap). `docs/concepts/` opened 2026-08-08 with its first document — no empty scaffolding, ever.

## Reading paths

- **New here** → `../README.md` → `../REPRISE-2026-08-07.md` → hub decision 019 (ecosystem founding decision, `quantum-meridian/docs/decisions/019-quantum-cortex-architecture.md`) → `adr/ADR-001-metrics-first-class.md` → `adr/ADR-002-training-stack-portability.md`.
- **Reproduce a run** → `../metrics/README.md` → `../metrics/schema/run-v1.schema.json` → `roadmap.md` (N1) → `EUROHPC-PLAN-2026-08-07.md` (§4 environment checklist).
- **Contribute a component** → `glossary.md` → `IDEAS-REGISTER-2026-08-07.md` → `FEATURES-2026-08-07.md` → `roadmap.md` (gates) → `../CONTRIBUTING.md`.

## Contents

- `adr/` — local decision records, index `adr/README.md` (ecosystem-level decisions stay in the quantum-meridian hub; full decision log mirrored in `quantum-meridian/docs/reference/quantum-cortex.md`).
- `HANDOFF-2026-08-07-ecosystem-context.md` — context capsule for any agent (upload to a dedicated workspace).
- `IDEAS-REGISTER-2026-08-07.md` — innovation candidates, tiers NOW / RES / SPEC.
- `OSS-SURVEY-2026-08-07.md` — surveyed repositories + binding anti-plagiarism protocol + GitOps tooling.
- `FEATURES-2026-08-07.md` — features transposed from surveyed code, with evidence.
- `EUROHPC-PLAN-2026-08-07.md` — eligibility, two-stage access, trainability checklist, sizing.
- `EXECUTION-2026-08-07.md` — the N1 execution list (E0–E7) with the test → EuroHPC gate.
- `EXTRACTION-K3-2026-08-08.md` — knowledge harvested from the two K3 repositories, with per-item legal status (zero code copied).
- `GETTING-STARTED.md` — set up the local Python env and run/verify everything locally (smoke, ledger, the E5 commit loop). Start here to reproduce.
- `SEQUENCE.md` — **the stage-by-stage synthesis**: everything that ships at each step/wave, one page, status-stamped (synthesizes with pointers, never overrides).
- `WHITEPAPER.md` — the paper's living skeleton: sections, the **admission gate** for claims (statistical, anchored, real, reproducible, bounded), what is admissible vs open, and the gates for v0/v1.
- `EXTRACTION-LANDSCAPE-2026-08-08.md` — the August-2026 landscape pass: 15 repos, licenses verified, ideas mapped, legal PASS.
- `concepts/README.md` — concepts index + the deep-dive schedule (a region earns its dive when its wave opens).
- `concepts/brain-atlas.md` — the systematic brain → cortex transposition atlas (mapped / candidate / deliberately not transposed). `concepts/hippocampus.md` — the memory-organ deep dive.
- `adr/ADR-004-evaluation-pareto.md` — the evaluation law (**D16**) and its publication twin (**D17**, open-results, revises D6): a Pareto frontier over declared axes is what we measure *and* what we publish.
- `adr/ADR-006-c2-ablation-protocol.md` — the C2 ablation protocol (N2): Hopfield vs delta-rule vs control, MQAR + serial-position, advancement rule frozen before runs.
- `benchmarks/hm-protocol.md` — the Molaison (H.M., after the patient Henry Molaison; defined in docs/benchmarks/hm-protocol.md) episodic/semantic dissociation diagnostic, spec v1 (thresholds frozen before any run).
- `benchmarks/external-arm.md` — runbook of the external model arm: the Molaison dissociation on an open model (Qwen3 at 1.7, 4 and 8 billion parameters) through the frozen prompt adapter; prerequisites, the six commands in order with their expected outputs, the files produced, how to read a run (declared in ADR-008, amendment 2026-09-16)
- `glossary.md` — coined terms. `roadmap.md` — ordered steps with gates.
- [How to attack this in twenty minutes](FALSIFY.md) — the frozen thresholds, the declared readings, the negative rows, the command that recomputes the aggregates, and what would falsify the central claim.
