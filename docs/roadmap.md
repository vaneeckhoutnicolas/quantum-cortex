# Roadmap — quantum-cortex

Thin and scannable, per the quantum-meridian discipline. Every step is gated; gates are decided by ledger data, not by calendar. Normative references: hub decision 019 (architecture + ablation protocol), ADR-001 (metrics discipline).

| Step | Content | Gate to pass it |
|---|---|---|
| N1 | Training scaffold: one reproducible Kaggle notebook — 30M vanilla transformer control, pinned seeds, config-hash logging, open data slice. Emits its schema-v1 record + regenerates `metrics/LATEST.md`. *Status 2026-08-07: scaffold shipped, in-session CPU smoke passed; gate pending first committed real run (E5). 2026-09-05: repository created and pushed (E1); E2 = CI on every push.* | Record committed; exact rerun instructions verified. |
| N2 | Ablation C2 (associative layer) vs control — MQAR-style benchmark. | Advancement rule (hub decision 019): win on ≥ 1 declared capability, perplexity delta ≤ 2%. |
| N3 | Ablation C3 (oracle channel) — oracle-shift recovery. | Same rule. |
| N4 | Ablation C4 (typed decode contracts) — compliance rate. | Same rule. |
| N5 | Ablation report v0, published with losses. Then, and only then: EuroHPC AI Factories + TPU Research Cloud applications citing ledger `run_id`s. | Report reproducible from the ledger alone. |
| N6 | C5 grafted-zone adapter spec. | A validated core exists (≥ 1 component past its gate). |
| — | Rill + DuckDB dashboard (`dashboard/`, in-repo, open). | ≥ 3 records in the ledger. |
| — | Hardware purchase. | Ablation results + utilization math justify it. Never calendar-driven. |

Brain-derived features enter exclusively through the wave order of **ADR-003** (`adr/ADR-003-brain-feature-map.md`): W1 with N2, W1b with N3, W2 in the N4–N5 window, W3–W4 after N5. Every such feature ships behind a config flag, default off, until its ledger win.
