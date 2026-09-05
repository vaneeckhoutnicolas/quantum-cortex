# metrics/ — the run ledger

Normative decision: **ADR-001** (`docs/adr/ADR-001-metrics-first-class.md`). Summary of the discipline:

- Every training/ablation run appends **exactly one** record to `runs.jsonl`, conforming to `schema/run-v1.schema.json`.
- **A run without its committed record does not exist.** One commit per run: code state + ledger line + regenerated `LATEST.md`.
- Unknown = `null` ("not measured"), never invented, never zero-filled.
- Aborted/invalid runs stay in the ledger with their status — errors are noted, never erased.
- `LATEST.md` always shows: last run, best value per declared benchmark, running win/loss verdicts vs control. It is generated, never hand-edited.
- The ledger is the single source every public claim cites (`run_id`s) — claims and state cannot diverge.

Current state (2026-08-07, honest data): the ledger does not exist yet — it is created by the first N1 run. `LATEST.md` likewise.
