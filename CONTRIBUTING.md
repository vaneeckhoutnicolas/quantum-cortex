# Contributing

- **Product text in English.** Conversations may happen in any language; everything committed is English.
- **The ledger rules everything:** a training run without its committed `metrics/runs.jsonl` record (schema `run-v1`) does not exist. One commit per run: code state + record + regenerated `metrics/LATEST.md`. Unknown values are `null`, never invented. CI enforces schema validity.
- **Advancement rule (hub decision 019):** a component ships only if it beats the control on ≥ 1 declared capability with ≤ 2% perplexity degradation, tolerance stated before the run. Losses are published.
- **Decisions:** dated ADRs in `docs/adr/` (index: `docs/adr/README.md`); amendments are dated, prior text is kept — errors are noted, never erased. Promotions from the ideas register cite it.
- **Anti-plagiarism protocol is binding:** see `docs/OSS-SURVEY-2026-08-07.md`. Core components (C1-C6 and the model's critical path) are **always written from scratch** - no verbatim reuse in the core, whatever the license. Peripheral needs: pip dependency > re-implementation from the paper > adaptation as a last resort (MIT/BSD/Apache-2.0 only, provenance header + NOTICE entry); unclear licenses are read-only. Understand, rewrite, optimize, criticize - never duplicate.
- **Portability rules (ADR-002):** pure-framework ops; custom kernels Triton-only, with a framework fallback and a measured, ledger-recorded speedup.

## How to contribute

1. **Open an issue first** for anything non-trivial — describe the change before large work.
2. Fork, branch, and keep **commits granular**: one logical change per commit, with a clear message — in this project granular commits are documentation, not just hygiene.
3. Run the CPU smoke before opening a PR: `python train.py --config configs/smoke_cpu.json` (seconds on CPU; correctness only — CPU never pretrains, ADR-002). CI runs the same smoke plus the ledger validation on every push.
4. Open a pull request against `main`, describing what changed and why. Read the [Code of Conduct](CODE_OF_CONDUCT.md); by participating you agree to it.

## Licensing of contributions

This repository is fully open: **Apache-2.0, the whole repository** — no proprietary tier, no licence gate, ever. Contributions are accepted under the same licence (Apache License 2.0, section 5). By submitting a contribution you confirm you have the right to do so. Add yourself to [AUTHORS.md](AUTHORS.md) in your first contribution.
