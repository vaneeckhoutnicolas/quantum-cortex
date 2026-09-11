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
| **External anchor** | A from-scratch ladder of pure recurrent models (GLA + selective gate + channel decay + local conv + delta rule, each from its paper). At 3 seeds (verified) the hybrids reach **3–4× the best pure recurrent** (+0.048 and +0.062, t = 3.78 / 4.09), held because three seeds set t_crit at 4.30 (df 2). "Delta regresses when pure" was a single-seed artefact: withdrawn. |
| **What the gate refused** | The confirmation run (3 seeds, verified by log and artefact): delta +13% (t 1.76), and the row that matters: the hybrids reach **3–4× the best pure recurrent** (t 3.78 / 4.09) but three seeds set the bar at 4.30. **No claim is made.** A run reported at five seeds during preparation was **withdrawn**: its source log was never retained, so it yields no result. | [`confirmation-20260908T010143.json`](metrics/mqar/confirmation-20260908T010143.json) · [`PROVENANCE-confirmation.json`](metrics/mqar/PROVENANCE-confirmation.json) · [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md) §4.4 |

**Scope, stated plainly:** everything above is at 25M parameters and MQAR scale. The H.M. pass is a measurement of the *organ* (hash-seeded cue encoder, reference skill probe), not yet of the language model reading its journal in generation. No comparison has yet passed the statistical gate. These are the three open locks, and each has a scheduled key (below).

---

## Why this is worth your time — four contributions

1. **A law borrowed from black-hole thermodynamics that decides when a result may be claimed.** The author's parallel physics program ([mirror-bh-thermodynamics](https://github.com/vaneeckhoutnicolas/mirror-bh-thermodynamics)) proves that thermodynamic invariants which look volatile when only the visible horizons are counted become *exact* when the complete root set is counted — the apparent noise is the share of what was left out (R² = 1 under full accounting, verified to forty digits). Transposed here as **rule 6 of the admission gate — the completeness test**: before claiming, regress the result on the declared context; unexplained variance is treated as an *omitted factor*, not as noise, and blocks the claim until it is found or declared a limit. The law has already acted on this repository twice, in both directions: it exposed an ill-posed metric (the router's "93% of the oracle ceiling" was a ratio over a collapsing denominator — replaced by an absolute measure) and it **refused** to rescue a near-miss (the 5-seed delta result at t = 2.65: the per-seed variance regressed on context at R² 0.15–0.20 — genuine, not omitted, so no adjusted analysis was admissible). A method that corrects its author in both directions is the strongest evidence this repository offers that its numbers can be trusted. → [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md) §2.7 and rule 6, [`metrics/mqar/completeness-test-router-2026-09-08.json`](metrics/mqar/completeness-test-router-2026-09-08.json), register RES-20.
2. **A continuity benchmark that exists before a model scores well on it.** The H.M. protocol (episodic persistence), the oracle-shift protocol (adaptive revision) and the reference-ratio protocol (non-regeneration) are specified, seeded, config-hashed and open. Whoever defines the measurement frames the question. → [`docs/benchmarks/`](docs/benchmarks/)
3. **A memory router that cannot degrade the model.** Control as the floor, routing regime that hardens as it consolidates, a circuit breaker that re-routes instead of killing — versioned behind a retro-compatible contract (v2 was added as one dispatch branch; v1 untouched). → [`cortex_c2/`](cortex_c2/), ADR-006
4. **A method you can audit.** 84 tests run in CI; 7 architecture decision records and a 20-entry decision log, dated and never rewritten; a run ledger where unknowns are `null`; two retrogradations and one invalid run kept in the artefacts. The collaboration itself follows a design thinking discipline the author formalised in 2018 (understand before solving; diverge then converge; hold desirability, viability and feasibility together): the researcher supplies the upstream structure, the model supplies rapid divergence and execution, and the ledger arbitrates — a pattern that caught five over claims in one week and is reproducible by others. The same evidentiary discipline was independently converged on by the author's physics program (theorem register, per-claim novelty pre-gates, dated follow-up notes, an archive kept for provenance) — suggesting it is a property of the method, not of either subject. → [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md) §2, [`docs/adr/`](docs/adr/)

---

## Who this is for

The cortex is built for settings where continuity is the decisive property. Where it is not, use a frontier model. Five settings, each tied to what the triad must measure and to the organ that measures it:

- **A long running project companion.** A lawyer on a six month case: a decision taken in January (set an argument aside, on the strength of a ruling) is recalled in May in a fresh session, the ruling is cited rather than re summarised, the discarded argument stays discarded. *Persistence and non regeneration; the H.M. protocol measures exactly this; C2b.*
- **An assistant whose world changes.** A compliance officer: a directive enters into force on Tuesday, the oracle channel receives it, and by Wednesday the model's answers on reporting obligations have changed, visibly and datably, without retraining. *Revision; the oracle organ C3, not yet built.*
- **Private, local, encrypted memory.** A general practitioner keeping follow up notes on her own machine, one encrypted journal per patient, recall across consultations months apart, no record ever sent to an API. *Persistence under a trust boundary; needs scale and ternary weights.*
- **The local model of an agent.** A Quantum Meridian agent asks the cortex for everything that touches the project's history and a frontier model for new code; the cortex signals, through its decode contracts, when a question leaves its domain. *The non regression router, at the scale of models.*
- **The measurement, for other laboratories.** A team with a long memory model runs the open suite (H.M., oracle shift, reference ratio) with its frozen thresholds on their own model and publishes a comparable number. *The product here is the benchmark, not the model.*

What the cortex is not: a general chatbot, a frontier code model, or a competitor on raw capability. The five settings above hold precisely because they rest on a capability those models do not have and do not measure.

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

1. **The recurrent ladder at eight seeds** — the largest effect in the repository (hybrids 3–4× the best pure recurrent, t 3.78 / 4.09 at df 2, t_crit 4.30; at df 7 it is 2.37). Eight seeds rather than five: a bar low enough that the run's outcome, pass or hold, is informative either way. Whether it passes is for the run to decide.
2. **The language-model arm of H.M.** — the model reading its own journal in generation (learned cue encoder, real skill suites) → continuity measured on the *model*, not the organ.
3. **C3, the oracle channel** → the second component of the triad (adaptive revision).
4. **A 100M+ run** (EuroHPC Development Access; the author's company is eligible) → whether any of this generalises → paper v1.

Contributions welcome on any of the four — see [`CONTRIBUTING.md`](CONTRIBUTING.md). The most valuable first contribution is a **replication**: run `pytest`, run the H.M. protocol, and open an issue with what you saw.

---

## License and attribution

Apache License 2.0, unmodified. You may use, modify, fork and redistribute this work freely, on one condition that the License itself imposes (Section 4(d)): **the `NOTICE` file must travel with any copy or derivative work, unmodified**, and with it the statement that quantum-cortex was originally created by Nicolas Van Eeckhout. Removing that attribution is a breach of the License. To cite the work, use [`CITATION.cff`](CITATION.cff) (GitHub renders it as "Cite this repository"). The continuity benchmarks published here carry the same requirement.

---

*Author: Nicolas Van Eeckhout (Win2Win SRL, Brussels) · [ORCID 0000-0002-5256-3185](https://orcid.org/0000-0002-5256-3185) · License Apache-2.0 · Companion physics program: [mirror-bh-thermodynamics](https://github.com/vaneeckhoutnicolas/mirror-bh-thermodynamics).*
