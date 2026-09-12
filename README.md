# quantum-cortex

**An open, from-scratch language model built to measure one capability nobody measures: *continuity* — a model that remembers your project and visibly changes its mind when the world changes.**

Independent member of the [Quantum Meridian](https://github.com/vaneeckhoutnicolas/quantum-meridian) ecosystem: the same cognitive concepts that Quantum Meridian runs *as orchestration around third-party models* (expert zones, associative memory, an oracle channel, typed decode contracts) are implemented here *in the weights and the decode loop* of a model trained from scratch. Two planes, one architecture — and both produce measurements.

Built by one researcher with a language model as design, implementation and review partner, under an evidentiary discipline strict enough that **the absence of a result is itself a recorded result**. Every number below links to the artefact and the test that produced it. Apache-2.0.

---

## What exists today (verify each line)

| | Result | Evidence |
|---|---|---|
| **Baseline** | Byte-level GPT-2-style control, 25.9M params, **val_ppl 2.601** on 500M FineWeb-Edu tokens | [`metrics/runs.jsonl`](metrics/runs.jsonl) run `be1fa8139f59`, schema-validated in CI |
| **Continuity, first measurement** | The **H.M. dissociation passes** against the persistent-memory organ: recall **1.000** with the journal, **0.125** without (chance floor), gap **0.875** vs required 0.50; the journal **never hallucinates** a fact it does not hold (negative control 0.000); skills journal-independent. Stable on 4 seeds. Since 2026-09-12 the same pass holds with the journal **sealed on disk and reopened from the disk alone** for session B (`persistent: true`, `claimable: 1`, 1 seed). | [`metrics/mqar/hm-protocol-2026-09-08.json`](metrics/mqar/hm-protocol-2026-09-08.json) · persistent run: [`hm-protocol-persistent-2026-09-12.json`](metrics/mqar/hm-protocol-persistent-2026-09-12.json) · spec frozen 2026-08-08: [`docs/benchmarks/hm-protocol.md`](docs/benchmarks/hm-protocol.md) · `tests/test_c2b_hm_protocol.py` |
| **The journal in the decode loop is wired (code, tested; no result yet).** The model reads its journal as content through a zero init cross attention in one block, with a learned cue encoder; the curriculum's targets are computed from what the journal actually returned (cite what was shown, else abstain); the protocol's LM arm reports strict recall, invalid citations, and an attributable failure. The first measurement is a GPU fine tune from N1 (REPRISE step 4). | [`docs/adr/ADR-008-journal-in-the-decode-loop.md`](docs/adr/ADR-008-journal-in-the-decode-loop.md) · [`cortex_c2b/lm_bridge.py`](cortex_c2b/lm_bridge.py) · [`tests/test_c2b_lm_bridge.py`](tests/test_c2b_lm_bridge.py) · spec amendment (b) 2026-09-12 |
| **Memory ablation (associative recall, MQAR)** | Two associative memories — modern-Hopfield and gated delta-rule — each beat the transformer control on the 12-tier curve (single seed); **no single memory dominates**: Hopfield owns easy/dense tiers, delta the hard/short ones, the control the long ones. Only 6/12 tiers are separable. | [`metrics/mqar/domain-map.md`](metrics/mqar/domain-map.md) · [`mqar-ablation-final-2026-09-07.json`](metrics/mqar/mqar-ablation-final-2026-09-07.json) |
| **The memory router (RES-18)** | A router over paths (control / Hopfield / delta / journal) on a **guaranteed non-regression floor**: a memory is used only where it beats the control. The oracle ceiling is **+21%** over the control, **+6%** over the best single memory. The *learned* router reaches that ceiling on a synthetic task and **not yet on real spans** (12 tiers is too few — recorded, not hidden). | [`cortex_c2/`](cortex_c2/) · [`router-ablation-real-spans-2026-09-07.json`](metrics/mqar/router-ablation-real-spans-2026-09-07.json) · ADR-006 D8–D9 |
| **External anchor** | A from-scratch ladder of pure recurrent models (GLA + selective gate + channel decay + local conv + delta rule, each from its paper), now at **eight seeds, 1500 steps, paired on seed with the control and the two hybrids** (128 units, the first real cut and resume). At three seeds and 1000 steps the hybrids reached 3-4x the best pure rung; at eight seeds and 1500 steps the comparator changes: **the local convolution rung (L3) takes off on five seeds of eight** (0.47 to 0.67 at 8 pairs) and the statement does not carry over. The hybrids trail L3 and do not beat the control. | [`metrics/mqar/ladder8-ckpt/`](metrics/mqar/ladder8-ckpt/) · [`PROVENANCE-ladder8.json`](metrics/mqar/ladder8-ckpt/PROVENANCE-ladder8.json) · [`docs/RESULTS.md`](docs/RESULTS.md) rows 18-20 |
| **What the gate refused, and the one line it let through** | At eight seeds the six declared tests: five held (hybrids vs the best pure rung: t −2.10 / −1.97; hybrids vs control: t 1.11 / 0.56; control vs the best pure rung: t −2.30) and one gated, structural and negative: **the delta rule pure trails the local convolution pure** (L4 vs L3, t −3.21, df 7), with its reserve written next to it (a bimodal comparator; a sign test does not pass). **No line in favour of a memory thesis.** The three seed run keeps its date (superseded); the five seed run stays withdrawn (its log was never retained). A selection envelope over the separately trained models (0.235) is a ceiling, held twice over: a maximum is significant against its own components by construction, and no router inside one model can compose separately trained models. | [`LATEST-ladder8.json`](metrics/mqar/ladder8-ckpt/LATEST-ladder8.json) · [`metrics/mqar/logs/`](metrics/mqar/logs/) · [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md) §4.4b |

**Scope, stated plainly:** everything above is at 25M parameters and MQAR scale. The H.M. pass is a measurement of the *organ* (hash-seeded cue encoder, reference skill probe), not yet of the language model reading its journal in generation; its first artefact (2026-09-08) was produced within one process (`persistent: false`, not claimable, in the protocol's vocabulary since 2026-09-12); the second (2026-09-12, one seed) reopens a sealed journal from the disk alone (`persistent: true`), and the organ's survival across a real process restart is established by test (ADR-007 D8). No comparison has yet passed the statistical gate. These are the three open locks, and each has a scheduled key (below).

---

## Why this is worth your time — four contributions

1. **A law borrowed from black-hole thermodynamics that decides when a result may be claimed.** The author's parallel physics program ([mirror-bh-thermodynamics](https://github.com/vaneeckhoutnicolas/mirror-bh-thermodynamics)) proves that thermodynamic invariants which look volatile when only the visible horizons are counted become *exact* when the complete root set is counted — the apparent noise is the share of what was left out (R² = 1 under full accounting, verified to forty digits). Transposed here as **rule 6 of the admission gate — the completeness test**: before claiming, regress the result on the declared context; unexplained variance is treated as an *omitted factor*, not as noise, and blocks the claim until it is found or declared a limit. The law has already acted on this repository twice, in both directions: it exposed an ill-posed metric (the router's "93% of the oracle ceiling" was a ratio over a collapsing denominator — replaced by an absolute measure) and it **refused** to rescue a near-miss (the 5-seed delta result at t = 2.65: the per-seed variance regressed on context at R² 0.15–0.20 — genuine, not omitted, so no adjusted analysis was admissible). A method that corrects its author in both directions is the strongest evidence this repository offers that its numbers can be trusted. → [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md) §2.7 and rule 6, [`metrics/mqar/completeness-test-router-2026-09-08.json`](metrics/mqar/completeness-test-router-2026-09-08.json), register RES-20.
2. **A continuity benchmark that exists before a model scores well on it.** The H.M. protocol (episodic persistence), the oracle-shift protocol (adaptive revision) and the reference-ratio protocol (non-regeneration) are specified, seeded, config-hashed and open. Whoever defines the measurement frames the question. → [`docs/benchmarks/`](docs/benchmarks/)
3. **A memory router that cannot degrade the model.** Control as the floor, routing regime that hardens as it consolidates, a circuit breaker that re-routes instead of killing — versioned behind a retro-compatible contract (v2 was added as one dispatch branch; v1 untouched). → [`cortex_c2/`](cortex_c2/), ADR-006
4. **A method you can audit.** 141 tests in CI, 132 in seconds without the training smoke tests; 8 architecture decision records and a 23-entry decision log, dated and never rewritten; a run ledger where unknowns are `null`; two retrogradations and one invalid run kept in the artefacts. The collaboration itself follows a design thinking discipline the author formalised in 2018 (understand before solving; diverge then converge; hold desirability, viability and feasibility together): the researcher supplies the upstream structure, the model supplies rapid divergence and execution, and the ledger arbitrates — a pattern that caught five over claims in one week and is reproducible by others. One consequence is recorded as the project's long horizon (whitepaper §5b): the organ this repository builds, a journal that survives the session boundary and a gate that judges by persistence, is the organ its own AI collaborator lacks; the model that co designed it does not persist between sessions, and the author's ledger reproduced by hand the capability the project measures. That is an ambition, not a result. The same evidentiary discipline was independently converged on by the author's physics program (theorem register, per-claim novelty pre-gates, dated follow-up notes, an archive kept for provenance) — suggesting it is a property of the method, not of either subject. → [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md) §2, [`docs/adr/`](docs/adr/)

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
pytest tests/ -m "not slow"        # 132 tests: ledger, data-mix invariants, C2 layers, router, journal, lifecycle, restart, storage policy, journal in the decode loop, recurrent base hybrid, H.M.
python -m cortex_c2b.hm_protocol    # the H.M. dissociation, end to end, in seconds (persistent: false, in memory)
python -m cortex_c2b.hm_protocol --persistent   # the same, sealed on disk, session B reopened from the disk alone
python train.py --config configs/smoke_journal_cpu.json   # the journal in the decode loop, the whole loop on CPU in a minute (ADR-008); a mechanics smoke, not a result
python -m cortex_eval.resumable_rb --quick --reference-from <a ladder subset dir>   # the recurrent base hybrid family, four arms, in seconds (Rev38); mechanics, not a result
python -m cortex_eval.domain_map    # the per-tier domain map from the committed artefacts
```

GPU runs (the MQAR ablation, the confirmation run) reproduce from the notebooks in [`notebooks/`](notebooks/) on a free Kaggle T4; every run writes its record to the ledger. Full setup: [`docs/GETTING-STARTED.md`](docs/GETTING-STARTED.md).

---

## Where to read next

- **The paper, as it is being written:** [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md) — thesis, method, architecture, results as a domain map, open questions; claims enter only through a seven-rule admission gate.
- **Every decision, dated:** [`docs/adr/`](docs/adr/) (ADR-001…008) · [`docs/IDEAS-REGISTER-2026-08-07.md`](docs/IDEAS-REGISTER-2026-08-07.md) (20 ideas, each with an implementation status) · the ecosystem-level log D1–D23 in the [hub](https://github.com/vaneeckhoutnicolas/quantum-meridian/blob/main/docs/reference/quantum-cortex.md).
- **What happens next, and its gates:** [`docs/SEQUENCE.md`](docs/SEQUENCE.md).
- **AI agents:** read [`CLAUDE.md`](CLAUDE.md) first — the working rules are non-negotiable.

## What comes next (gated, in order)

1. ~~**The recurrent ladder at eight seeds**~~ **Done (2026-09-12): read cold, graved as rows 18 to 20.** It contradicted its hypothesis and produced the run's real finding, L3's bimodal take off. What follows it (Rev38, after item 2): the Σ of the convergence (L3 at 3000 steps on the three seeds at the floor: *not yet, or never*), then the recurrent base hybrid, four declared arms, to learn whether one model can carry both regimes.
2. **The language-model arm of H.M.** — the model reading its own journal in generation (learned cue encoder, real skill suites) → continuity measured on the *model*, not the organ.
3. **C3, the oracle channel** → the second component of the triad (adaptive revision).
4. **A 100M+ run** (EuroHPC Development Access; the author's company is eligible) → whether any of this generalises → paper v1.

Contributions welcome on any of the four — see [`CONTRIBUTING.md`](CONTRIBUTING.md). The most valuable first contribution is a **replication**: run `pytest`, run the H.M. protocol, and open an issue with what you saw.

---

## License and attribution

Apache License 2.0, unmodified. You may use, modify, fork and redistribute this work freely, on one condition that the License itself imposes (Section 4(d)): **the `NOTICE` file must travel with any copy or derivative work, unmodified**, and with it the statement that quantum-cortex was originally created by Nicolas Van Eeckhout. Removing that attribution is a breach of the License. To cite the work, use [`CITATION.cff`](CITATION.cff) (GitHub renders it as "Cite this repository"). The continuity benchmarks published here carry the same requirement.

---

*Author: Nicolas Van Eeckhout (Win2Win SRL, Brussels) · [ORCID 0000-0002-5256-3185](https://orcid.org/0000-0002-5256-3185) · License Apache-2.0 · Companion physics program: [mirror-bh-thermodynamics](https://github.com/vaneeckhoutnicolas/mirror-bh-thermodynamics).*
