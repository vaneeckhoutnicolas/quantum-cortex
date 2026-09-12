# Results — every number, its artefact, its test

The single page a reviewer needs. One row per claim or non-claim; nothing here without a committed artefact and a test that exercises the code path. Status vocabulary: **measured** (a number exists) · **gated** (statistically established per the whitepaper's seven-rule gate) · **held** (visible signal, not established) · **open** (no number yet).

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
| 10 | Delta > control on the discriminating tiers, **3 seeds (run A)** | **held** | +0.0107 (+13%), t = 1.76 vs t_crit 4.30 (df 2) | `metrics/mqar/confirmation-20260908T010143.json` (real Kaggle artefact) | `tests/test_multiseed.py` | 1000 steps, 2 tiers; small effect at 3 seeds |
| 10b | ~~Delta / Hopfield > control, 5 seeds~~ | **withdrawn** | claimed delta t 2.65 / Hopfield t 2.21 (df 4) | `confirmation-5seed-2026-09-09-reconstructed.json` | — | source log never retained as a file; no 5-seed run located in Kaggle (2026-09-09). Kept visible, not cited |
| 11 | Hybrids ≫ best pure recurrent (ladder), **3 seeds (run A)** | **held** | Hopfield-hybrid +0.048 (t 3.78), delta-hybrid +0.062 (t 4.09) vs best pure rung L4; t_crit 4.30 (df 2) | `confirmation-20260908T010143.json` (real Kaggle artefact) | `tests/test_recurrent_ladder.py` | **df-limited by design** (3 seeds), a large effect; next run = ladder at **8 seeds** (t_crit 2.37 at df 7) — no prediction of passage |
| 12 | Delta rule regresses when pure (L4 < L3) | **open — demoted** | L4 − L3 = +0.008, t = 0.90 (3 seeds): not supported | same | same | the earlier mini-run (single seed) was a seed artefact; the candidate finding is withdrawn |
| 13 | **H.M. dissociation** — recall collapses without the journal, skills hold | **measured** (4 seeds) | ON **1.000** / OFF **0.125** (floor 0.175), gap **0.875** ≥ 0.50, negctrl **0.000**, skill Δ 0 → `hm_dissociation_pass = 1` | `metrics/mqar/hm-protocol-2026-09-08.json` | `tests/test_c2b_hm_protocol.py` | against the organ (hash-seeded cue encoder, reference skill probe); first run INVALID on the negative control — simulator corrected, spec amended *dated before rerun*, thresholds untouched; **produced within one process** (`persistent: false`, `claimable: 0` in the protocol's vocabulary since the 2026-09-12 amendment; a claimable pass needs the disk run with session B reopened, `python -m cortex_c2b.hm_protocol --persistent`) |
| 14 | Circuit breaker: aggressive early-abort, bounded retry | measured | 12 clean aborts in production (v1/v2 runs) | `train.py`, ADR-006 D7 | `tests/test_circuit_breaker.py` | — |
| 15 | Oracle-shift revision (triad component 2) | **open** | — | — | — | needs C3 (not built) |
| 16 | Non-regeneration / reference ratio (triad component 3) | **open** | — | — | — | needs RES-8 typed span references (not built) |
| 17 | Anything at 100M+ parameters | **open** | — | — | — | EuroHPC run scheduled |

**Gate status (whitepaper §2.6):** as of 2026-09-09, **zero** comparisons have passed the seven-rule gate. One confirmation run is verified (run A, 3 seeds): memory-vs-control held (small effect), hybrids-vs-pure-recurrent held at df 2 (large effect, t 3.78 / 4.09). A claimed 5-seed run is **withdrawn** (no retained source). Next run: the ladder at 8 seeds — it may pass or hold.

**How to add a row:** a row needs (a) a committed artefact under `metrics/`, (b) a test in `tests/` exercising the code path, (c) a status from the vocabulary above, (d) the reserve stated. Rows are never deleted; a superseded row keeps its date and gains a pointer to what superseded it.
