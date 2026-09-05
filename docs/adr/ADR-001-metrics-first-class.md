# ADR 001 — Metrics are first-class citizens, by design

- **Status:** Proposed
- **Date:** 2026-08-07
- **Deciders:** Nicolas Van Eeckhout
- **Series note:** this is the first ADR of the `quantum-cortex` **local** series. The ecosystem-level founding decision remains **hub decision 019** in `quantum-meridian/docs/decisions/`. The local series exists so that external contributors find every cortex decision inside the open repository.

## Context

The Quantum Meridian discipline is "measure first, then decide": the QM agent emits a cleartext metric stream (`turns.json`) and a Rill + DuckDB dashboard reads it. quantum-cortex's entire thesis (hub decision 019) is decided by an ablation protocol — which is meaningless without standardized, comparable, always-current numbers. Community credibility (compute follows a reproducible table, D7) rests on the same numbers. Therefore metrics cannot be an afterthought bolted onto training scripts: they are a first-class value of the model project, by design.

## Decision

1. **Run schema v1** — `metrics/schema/run-v1.schema.json`. Every training or ablation run emits **exactly one** record conforming to it. Unknown values are `null` ("not measured"), never invented and never zero-filled. The schema is versioned: any breaking change is a `run-v2` plus a migration note — never a silent edit.

2. **Append-only ledger** — `metrics/runs.jsonl`, versioned in git. The repository is the source of truth of results (GitOps). **A run without its committed record does not exist.** Invalid or aborted runs stay in the ledger with `status: invalid|aborted` — errors are noted, never erased.

3. **Latest state always visible** — `metrics/LATEST.md` is regenerated after every run (last run, best value per declared benchmark, running win/loss verdicts vs control) and committed **in the same commit** as the ledger line and the code state. One commit per run.

4. **Benchmarkable by design** — the schema's `results.benchmarks` block carries the declared capabilities of hub decision 019 (routing specialization, MQAR-style recall, oracle-shift recovery, invariant compliance) plus perplexity and a standard suite slot. The `comparison` block carries `control_run_id`, `perplexity_delta_pct`, and per-capability verdicts, so the advancement rule of hub decision 019 is computable directly from the ledger.

5. **Serving-time symmetry** — the same standardization philosophy applies at inference through the C6 telemetry (cache affinity / cost / latency / quota). Training ledger and serving telemetry are the two halves of "metrics as first-class values of the LLM".

6. **A dedicated Rill project in this repo** — `dashboard/`, open like everything else: same *approach* as `quantum-metrics-dashboard`, **zero mixing** with it. Hard isolation rules: (a) it reads **only** `metrics/runs.jsonl` and `metrics/schema/`, through **repo-relative paths** — never absolute paths, never usernames (the `turns.sql` lesson, mandatory here because this dashboard ships publicly); (b) it references **no** QM stream (`turns.json` or any proxy/agent/contracts source), and `quantum-metrics-dashboard` references no cortex source — two Rill projects, two data populations, no shared DuckDB source; (c) `quantum-metrics-dashboard` stays local and private, **unchanged** by this decision. Data-population boundary: **training/ablation records are public** (this repo); **serving/usage telemetry** of cortex-as-provider inside a QM installation flows into that user's private QM metrics via C6, like any provider — usage data never enters the public repository. **Activation gate: ≥ 3 records in the ledger** (nothing to compare below that; no untested dashboard is committed before real data exists). Known DuckDB learning to apply: with `union_by_name`, never declare `columns = { x: 'VARCHAR' }`; use `SELECT *, COALESCE(TRY_CAST(x AS VARCHAR), 'default')`.

7. **Doc organization mirrors the quantum-meridian discipline** — `docs/index.md` with named reading paths, `docs/adr/` (this series), `docs/glossary.md`, `docs/roadmap.md`. `docs/concepts/` is reserved and will be created with the first concept document — no empty scaffolding. Git tags follow `checkpoint-NN -m "..."`.

## Consequences

- **N1 is amended:** the training scaffold must append its schema-v1 record and regenerate `metrics/LATEST.md`. A notebook that trains but does not write its record is a failed deliverable.
- Every future ablation claim (blog post, EuroHPC/TRC application, whitepaper) cites `run_id`s from the ledger — claims and state cannot diverge, the exact failure pattern observed in the QM claims-vs-state review.
- Risk accepted: the schema may prove too rigid too early — mitigated by nullable fields and explicit versioning, never by silent edits.
