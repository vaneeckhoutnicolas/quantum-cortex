# Results — every number, its artefact, its test

The single page a reviewer needs. One row per claim or non-claim; nothing here without a committed artefact and a test that exercises the code path. Status vocabulary: **measured** (a number exists) · **gated** (statistically established per the whitepaper's six-rule gate) · **held** (visible signal, not established) · **open** (no number yet).

| # | Statement | Status | Number | Artefact | Test | Scope / reserve |
|---|---|---|---|---|---|---|
| 1 | Byte-level GPT-2-style control, 25.9M params | measured | val_ppl **2.601**, 500M FineWeb-Edu tokens, 30,517 steps | `metrics/runs.jsonl` (`be1fa8139f59`) | `tests/test_ledger.py` | a baseline, not a comparison |
| 2 | The data mix is a hashed, reproducible object; `mix([S,S]) ≡ S` | measured | 5 structural invariants | `data/mixes/fineweb-edu-n1.json`, `metrics/runs.jsonl` (`mix_hash`) | `tests/test_invariants.py`, `test_wiring.py` | ADR-005 |
| 3 | Hopfield memory > control on MQAR (12-tier AUC) | **held** | +14% (0.1206 vs 0.1062) | `metrics/mqar/mqar-ablation-final-2026-09-07.json` | `tests/test_c2_layers.py`, `test_mqar.py` | single seed; 6/12 tiers separable |
| 4 | Delta-rule memory > control on MQAR (12-tier AUC) | **held** | +2% (0.1083) | same | same | single seed; stabilised (L2-normalised keys) after a v1 divergence |
| 5 | No single memory dominates — each owns a regime | measured | Hopfield 4 / delta 1 / control 1 of 6 separable tiers | `metrics/mqar/domain-map.md` | `tests/test_domain_map.py` | the map is the result; the router is the answer |
| 6 | Hopfield is structure-limited (more slots do not help) | **held** | 0.073 → 0.034 as slots 32 → 256 | `mqar-ablation-final-2026-09-07.json` (capacity sweep) | — | single seed |
| 7 | Router oracle ceiling over the best single path | measured | +21% vs control, +6% vs Hopfield | `metrics/mqar/router-ablation-*.json` | `tests/test_c2_router.py` | an oracle reads the truth — a ceiling, not a router |
| 8 | Learned router reaches the ceiling on a synthetic task | measured | 88–93% of ceiling | `router-ablation-20260907T201204.json` | `tests/test_c2_router_ablation.py` | synthetic routing task |
| 9 | Learned router on **real** MQAR spans | **held / retrograded** | ~25% of ceiling (−38%..+67% across holdouts) | `router-ablation-real-spans-2026-09-07.json` | — | 12 tiers = 8 training points; the ratio metric is ill-posed near a zero ceiling (rule 6) |
| 10 | Delta > control on the discriminating tiers, **5 seeds** | **held** | +0.0134 (+16%), 4/5 seeds, t = 2.65 vs t_crit 2.78 (df 4) — **0.13 short** | `metrics/mqar/confirmation-5seed-2026-09-09-reconstructed.json` | `tests/test_multiseed.py` | 1500 steps, 4 tiers; effect stable across the 3- and 5-seed runs; rule 6: the inter-seed variance is genuine (control level explains R² 0.15–0.20 only) — more seeds, not an adjusted analysis |
| 10b | Hopfield > control on the discriminating tiers, **5 seeds** | **held** | +0.0104 (+13%), 4/5 seeds, t = 2.21 vs 2.78 | same | same | same regime; seed 42 is control-easy (gap −0.004) |
| 11 | Pure recurrents ≪ hybrids at every rung of the ladder | **held** | best pure rung ~¼ of hybrids | `recurrent-ladder-mini-2026-09-07.json` | `tests/test_recurrent_ladder.py` | mini run, single seed |
| 12 | Delta rule regresses when pure, excels hybridised | **held** | L4 pure 0.016 vs hybrid 0.129 | same | same | candidate finding; unconfirmed |
| 13 | **H.M. dissociation** — recall collapses without the journal, skills hold | **measured** (4 seeds) | ON **1.000** / OFF **0.125** (floor 0.175), gap **0.875** ≥ 0.50, negctrl **0.000**, skill Δ 0 → `hm_dissociation_pass = 1` | `metrics/mqar/hm-protocol-2026-09-08.json` | `tests/test_c2b_hm_protocol.py` | against the organ (hash-seeded cue encoder, reference skill probe); first run INVALID on the negative control — simulator corrected, spec amended *dated before rerun*, thresholds untouched |
| 14 | Circuit breaker: aggressive early-abort, bounded retry | measured | 12 clean aborts in production (v1/v2 runs) | `train.py`, ADR-006 D7 | `tests/test_circuit_breaker.py` | — |
| 15 | Oracle-shift revision (triad component 2) | **open** | — | — | — | needs C3 (not built) |
| 16 | Non-regeneration / reference ratio (triad component 3) | **open** | — | — | — | needs RES-8 typed span references (not built) |
| 17 | Anything at 100M+ parameters | **open** | — | — | — | EuroHPC run scheduled |

**Gate status (whitepaper §2.6):** as of 2026-09-09, **zero** comparisons have passed the six-rule gate — the 5-seed confirmation held delta at t = 2.65 (t_crit 2.78) and Hopfield at 2.21; both beat the control on 4/5 seeds. An 8-seed run (t_crit 2.36) is the next test — it may pass or hold; rule 6 found the variance genuine, so no adjusted analysis is admissible.

**How to add a row:** a row needs (a) a committed artefact under `metrics/`, (b) a test in `tests/` exercising the code path, (c) a status from the vocabulary above, (d) the reserve stated. Rows are never deleted; a superseded row keeps its date and gains a pointer to what superseded it.
