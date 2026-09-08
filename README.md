# quantum-cortex

**An open, from-scratch language model built to measure one capability nobody measures: *continuity* — a model that remembers your project and visibly changes its mind when the world changes.**

Independent member of the [Quantum Meridian](https://github.com/vaneeckhoutnicolas/quantum-meridian) ecosystem: the same cognitive concepts that Quantum Meridian runs *as orchestration around third-party models* (expert zones, associative memory, an oracle channel, typed decode contracts) are implemented here *in the weights and the decode loop* of a model trained from scratch. Two planes, one architecture — and both produce measurements.

Built by one researcher with a language model as design, implementation and review partner, under an evidentiary discipline strict enough that **the absence of a result is itself a recorded result**. Every number below links to the artefact and the test that produced it. Apache-2.0.

---

## What exists today (verify each line)

| | Result | Evidence |
|---|---|---|
| **Baseline** | Byte-level GPT-2-style control, 25.9M params, **val_ppl 2.601** on 500M FineWeb-Edu tokens | [`metrics/runs.jsonl`](metrics/runs.jsonl) run `be1fa8139f59`, schema-validated in CI |
| **Continuity, first measurement** | The **H.M. dissociation passes** against the persistent-memory organ: recall **1.000** with the journal, **0.125** without (chance floor), gap **0.875** vs required 0.50; the journal **never hallucinates** a fact it does not hold (negative control 0.000); skills journal-independent. Stable on 4 seeds. | [`metrics/mqar/hm-protocol-2026-09-08.json`](metrics/mqar/hm-protocol-2026-09-08.json) · spec frozen 2026-08-08: [`docs/benchmarks/hm-protocol.md`](docs/benchmarks/hm-protocol.md) · `tests/test_c2b_hm_protocol.py` |
| **Memory ablation (associative recall, MQAR)** | Two associative memories — modern-Hopfield and gated delta-rule — each beat the transformer control on the 12-tier curve (single seed); **no single memory dominates**: Hopfield owns easy/dense tiers, delta the hard/short ones, the control the long ones. Only 6/12 tiers are separable. | [`metrics/mqar/domain-map.md`](metrics/mqar/domain-map.md) · [`mqar-ablation-final-2026-09-07.json`](metrics/mqar/mqar-ablation-final-2026-09-07.json) |
| **The memory router (RES-18)** | A router over paths (control / Hopfield / delta / journal) on a **guaranteed non-regression floor**: a memory is used only where it beats the control. The oracle ceiling is **+21%** over the control, **+6%** over the best single memory. The *learned* router reaches that ceiling on a synthetic task and **not yet on real spans** (12 tiers is too few — recorded, not hidden). | [`cortex_c2/`](cortex_c2/) · [`router-ablation-real-spans-2026-09-07.json`](metrics/mqar/router-ablation-real-spans-2026-09-07.json) · ADR-006 D8–D9 |
| **External anchor** | A from-scratch ladder of pure recurrent models (GLA + selective gate + channel decay + local conv + delta rule, each from its paper): pure recurrents stay ~¼ of the hybrids on MQAR at every rung; the delta rule regresses when pure yet excels hybridised (candidate finding, unconfirmed). | [`cortex_eval/recurrent_ladder.py`](cortex_eval/recurrent_ladder.py) · [`recurrent-ladder-mini-2026-09-07.json`](metrics/mqar/recurrent-ladder-mini-2026-09-07.json) |
| **What the gate refused — twice** | Confirmation at 3 seeds: delta +16%, t = 3.21 < 4.30. At **5 seeds**: delta **+16%, 4/5 seeds, t = 2.65 < 2.78 — 0.13 short**; Hopfield +13%, t = 2.21. **No claim is made.** Rule 6 found the inter-seed variance genuine (not an omitted factor), so no adjusted analysis is admissible; an 8-seed run is the next test — it may pass or hold. | [`confirmation-5seed-2026-09-09-reconstructed.json`](metrics/mqar/confirmation-5seed-2026-09-09-reconstructed.json) · [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md) §4.4 |

**Scope, stated plainly:** everything above is at 25M parameters and MQAR scale. The H.M. pass is a measurement of the *organ* (hash-seeded cue encoder, reference skill probe), not yet of the language model reading its journal in generation. No comparison has yet passed the statistical gate. These are the three open locks, and each has a scheduled key (below).

---

## Why this is worth your time — three contributions

1. **A continuity benchmark that exists before a model scores well on it.** The H.M. protocol (episodic persistence), the oracle-shift protocol (adaptive revision) and the reference-ratio protocol (non-regeneration) are specified, seeded, config-hashed and open. Whoever defines the measurement frames the question. → [`docs/benchmarks/`](docs/benchmarks/)
2. **A memory router that cannot degrade the model.** Control as the floor, routing regime that hardens as it consolidates, a circuit breaker that re-routes instead of killing — versioned behind a retro-compatible contract (v2 was added as one dispatch branch; v1 untouched). → [`cortex_c2/`](cortex_c2/), ADR-006
3. **A method you can audit.** 84 tests run in CI; 7 architecture decision records and a 20-entry decision log, dated and never rewritten; a run ledger where unknowns are `null`; two retrogradations and one invalid run kept in the artefacts. The same discipline, applied to the author's parallel physics program, produced the *completeness test* (rule 6 of the paper's admission gate) that then corrected a result here. → [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md) §2, [`docs/adr/`](docs/adr/)

---

## Reproduce it (no GPU needed to verify)

```bash
git clone https://github.com/vaneeckhoutnicolas/quantum-cortex && cd quantum-cortex
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e . && pip install pytest
pytest tests/ -m "not slow"        # 84 tests: ledger, data-mix invariants, C2 layers, router, journal, H.M.
python -m cortex_c2b.hm_protocol    # the H.M. dissociation, end to end, in seconds
python -m cortex_eval.domain_map    # the per-tier domain map from the committed artefacts
```

GPU runs (the MQAR ablation, the confirmation run) reproduce from the notebooks in [`notebooks/`](notebooks/) on a free Kaggle T4; every run writes its record to the ledger. Full setup: [`docs/GETTING-STARTED.md`](docs/GETTING-STARTED.md).

---

## Where to read next

- **The paper, as it is being written:** [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md) — thesis, method, architecture, results as a domain map, open questions; claims enter only through a six-rule admission gate.
- **Every decision, dated:** [`docs/adr/`](docs/adr/) (ADR-001…007) · [`docs/IDEAS-REGISTER-2026-08-07.md`](docs/IDEAS-REGISTER-2026-08-07.md) (20 ideas, each with an implementation status) · the ecosystem-level log D1–D20 in the [hub](https://github.com/vaneeckhoutnicolas/quantum-meridian/blob/main/docs/reference/quantum-cortex.md).
- **What happens next, and its gates:** [`docs/SEQUENCE.md`](docs/SEQUENCE.md).
- **AI agents:** read [`CLAUDE.md`](CLAUDE.md) first — the working rules are non-negotiable.

## What comes next (gated, in order)

1. **Eight-seed confirmation** — the next test of the memory-vs-control comparison. The 5-seed run held delta at t = 2.65 against a critical value of 2.78; at eight seeds the critical value is 2.36. Whether the effect passes is what the run will decide — its variance may as well grow as shrink, as it did from three to five seeds. A pass would be the first gated comparison and unlock paper v0; a hold would be recorded like the three before it.
2. **The language-model arm of H.M.** — the model reading its own journal in generation (learned cue encoder, real skill suites) → continuity measured on the *model*, not the organ.
3. **C3, the oracle channel** → the second component of the triad (adaptive revision).
4. **A 100M+ run** (EuroHPC Development Access; the author's company is eligible) → whether any of this generalises → paper v1.

Contributions welcome on any of the four — see [`CONTRIBUTING.md`](CONTRIBUTING.md). The most valuable first contribution is a **replication**: run `pytest`, run the H.M. protocol, and open an issue with what you saw.

---

*Author: Nicolas Van Eeckhout (Win2Win SRL, Brussels) · [ORCID 0000-0002-5256-3185](https://orcid.org/0000-0002-5256-3185) · License Apache-2.0 · Companion physics program: [mirror-bh-thermodynamics](https://github.com/vaneeckhoutnicolas/mirror-bh-thermodynamics).*
