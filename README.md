# quantum-cortex

An open-source **pure LLM** whose architecture implements the Quantum Meridian cognitive concepts — semantic expert zones, associative "magnetism" memory, an oracle volatility channel, typed decode invariants — **in the weights and the decode loop**, not in orchestration code around third-party models. It works standalone, or with external models grafted through the same router interface.

- **Status:** N1 scaffold shipped (2026-08-07): `train.py`, configs, Kaggle notebook, Apptainer recipe, CI smoke — in-session CPU smoke passed. **Zero completed runs on real data, zero benchmark results** — the ledger is the truth. Repository created 2026-09-05 (E1); CI (`validate-ledger` + `smoke-train`) is the self-test on every push (E2).
- **Discipline:** a run without its committed ledger record does not exist (`metrics/`); unknowns are `null`, never invented; errors are noted, never erased; every public claim cites `run_id`s.
- **Start here:** [REPRISE-2026-08-07.md](REPRISE-2026-08-07.md) → [docs/index.md](docs/index.md). AI agents: read [CLAUDE.md](CLAUDE.md) first.
- **Signature capability (D15):** **continuity** — the model that remembers your project and visibly changes its mind when the world changes. Measured as a triad: episodic persistence (the H.M. protocol), oracle-shift revision, and non-regeneration of the known.
- **Ecosystem:** an additional and *independent* member of the Quantum Meridian family — QM-compatible by default, required by nothing, requiring nothing. See `quantum-meridian/docs/reference/quantum-cortex.md`.
- **License:** Apache-2.0 (see LICENSE, NOTICE).
