# quantum-cortex

An open-source **pure LLM** whose architecture implements the Quantum Meridian cognitive concepts — semantic expert zones, associative "magnetism" memory, an oracle volatility channel, typed decode invariants — **in the weights and the decode loop**, not in orchestration code around third-party models. It works standalone, or with external models grafted through the same router interface.

- **Status:** N1 scaffold shipped (2026-08-07): `train.py`, configs, Kaggle notebook, Apptainer recipe, CI smoke — in-session CPU smoke passed. **N1 baseline committed 2026-09-06** — control `be1fa8139f59`, val_perplexity **2.601** on 500M FineWeb-Edu byte-tokens (30,517 steps); the ledger is the truth (`metrics/`). Repository created 2026-09-05 (E1); CI is the self-test on every push (E2). This is the reference every future component must beat — not a capability result; the control is deliberately organ-free.
- **Discipline:** a run without its committed ledger record does not exist (`metrics/`); unknowns are `null`, never invented; errors are noted, never erased; every public claim cites `run_id`s.
- **Start here:** [REPRISE-2026-08-07.md](REPRISE-2026-08-07.md) → [docs/index.md](docs/index.md). AI agents: read [CLAUDE.md](CLAUDE.md) first.
- **Run it yourself (verify everything locally):**
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt
  pytest tests/          # ledger integrity + end-to-end CPU smoke — what CI runs
  ```
  No GPU needed to verify (ADR-002). Full setup and the run-commit loop: `docs/GETTING-STARTED.md`.
- **What trains, when:** the CI smoke trains a *throwaway* proof-of-factory model (weights and record die with the runner); **E3 trains the 25.8M vanilla control** — the baseline, deliberately organ-free, *not yet the cortex*; the waves then train the cortex **one organ per run, against that control**. Full table: `docs/SEQUENCE.md`.
- **Signature capability (D15):** **continuity** — the model that remembers your project and visibly changes its mind when the world changes. Measured as a triad: episodic persistence (the H.M. protocol), oracle-shift revision, and non-regeneration of the known.
- **Ecosystem:** an additional and *independent* member of the Quantum Meridian family — QM-compatible by default, required by nothing, requiring nothing. See `quantum-meridian/docs/reference/quantum-cortex.md`.
- **License:** Apache-2.0 (see LICENSE, NOTICE).
