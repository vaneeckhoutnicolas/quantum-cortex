# SEQUENCE — what gets integrated, stage by stage

**Purpose.** The build order is normatively scattered across `roadmap.md` (steps and gates), `adr/ADR-003-brain-feature-map.md` (the 26-row map and waves), the `IDEAS-REGISTER` (26 entries), and `FEATURES` (F1–F12). Two months later, reconstructing the micro-sequence from four files is unreasonable. **This document is the synthesis: one page, every stage, everything that ships in it.** Rule: it *synthesizes with pointers, never overrides* — if this page and a normative source disagree, the source wins and this page gets a dated fix. It is regenerated at every gate passage.

**Status stamp:** 2026-09-07 — **N1 gate PASSED and N2 complete on the C2 axis.** N1 control `be1fa8139f59` (val_ppl 2.601). NOW-7 data layer built and wired. **First C2 ablation ran** (12 tiers × 3 variants, 9h): both associative memories beat the control (hopfield +14% AUC, delta +2%); per-tier hopfield 6 / delta 2 / control 3; Hopfield is structure-limited. **The RES-18 multi-path C2 router is fully implemented (Slices A→B→C→D) and green**, and its ablation shows the *learned* router captures ~88-93% of the oracle ceiling — RES-18 pays off in practice. **Strategic direction (D19, 2026-09-07): a big fish in a small pond — be the first to prove continuity reproducibly with an individual-trainable model, and publish the method as much as the model.** **Level 1 executed 2026-09-07 (tools built & green, 52 tests):** (a) multi-seed + 95% CIs + paired t-tests — the 5-seed GPU run is the next job; (b) router on **real** MQAR spans → **retrograded**: ~25% of the oracle ceiling (not 93%) — 12 tiers is under-sampled; the router needs thousands of real spans (wire into the LM); (c) external anchor in place — refined into a **recurrent ladder** (GLA + 4 named refinements, each from its paper; L4 = our delta): pure recurrents stay ~1/4 of our hybrids; delta-rule pays only when hybridised (candidate finding). **Deliverable specified 2026-09-07: the whitepaper** — `docs/WHITEPAPER.md`, a living skeleton with an **admission gate** (a claim enters a results section only if statistical ≥3 seeds, anchored, real-not-toy, reproducible, honestly bounded); everything else is an explicit *open question*. Gates for the paper: v0 once the 5-seed C2 ablation passes; v1 once the continuity triad is measured on a built organ (C2b or C3) **and** a 100M+ run exists. **Confirmation run done 2026-09-08 (3 seeds, budget):** delta leads 3/3 seeds (+16%) but t=3.21 < 4.30 — nothing passes the gate yet; Hopfield at the control on hard tiers (its domain is easy tiers). Domain map confirmed. **5-seed run done 2026-09-09: held again** (delta +16% t=2.65 vs 2.78; Hopfield +13% t=2.21; 4/5 seeds each; rule 6: variance genuine). **Next:** an 8-seed run (t_crit 2.36) — the first gate-passing claim and paper v0; then wire the router into the language model; publish the continuity benchmarks (NOW-6); **C2b BUILT (A→D, 2026-09-08) — the H.M. protocol PASSES against the organ (`hm_dissociation_pass = 1`, 4 seeds; recall ON 1.00 / OFF 0.125; journal never hallucinates); Slice E (lifecycle) + the LM arm next** — designed in `adr/ADR-007-c2b-persistent-tier.md` (the persistent tier: entry ≤1 KB, DG→CA3→CA1 write path, two-stage noise tribunal, journal as path 4 of the RES-18 router, scheduled consolidate/demote/evict; milestone = `hm_dissociation_pass` from the H.M. protocol — the gate for paper v1) + C3; 100M+ EuroHPC run → paper v1.

Earlier stamp (2026-09-06, kept): N1 gate passed; NOW-7 Slice 1 green; circuit breaker (D7) added and NaN-abort proven; MQAR harness green.

### The two CI checks (E2), decoded

Both are defined in `.github/workflows/ci.yml` and run on **every push and pull request** — they are the MVP's self-test, catching any future change that breaks training or the ledger discipline before a human even looks.

- **`ci/validate-ledger` (~9 s) — the metrics guardian.** Installs `jsonschema`, parses `metrics/schema/run-v1.schema.json`, then validates **every line** of `metrics/runs.jsonl` against it. Before the first real run (ledger absent) it passes with the explicit notice *"no ledger yet (expected pre-N1)"*. Green = the ledger, or its declared absence, is schema-clean — ADR-001's law ("a run without its valid committed record does not exist") enforced by machine. Red = a malformed or hand-edited record entered the repository.
- **`ci/smoke-train` (~44 s) — the living proof.** On a plain GitHub CPU runner it installs torch (CPU) + numpy + jsonschema, then executes the **real trainer end to end**: `python train.py --config configs/smoke_cpu.json` — a tiny 2-layer model, 195 steps over ~200k synthetic byte-tokens (the seeded pattern with its deliberately unlearnable 5% noise), CPU-only per ADR-002 (CPU proves *correctness*, never pretraining speed). A second step then asserts: a record was appended, it validates against run-v1, `status: completed`, and `anomalies: null` — the last one encodes *"the loss actually decreased"*, since the trainer flags an anomaly when it did not. Green = the whole chain **trains → learns → emits its record → self-validates** on a machine nobody configured. Red = the scaffold itself is broken. Of the 44 seconds, most is dependency install; the actual training takes ~10–15 s.

In one sentence: `validate-ledger` guards **the truth of the numbers**, `smoke-train` guards **the machine that produces them**.

### What is actually trained, stage by stage

| Stage | Model trained | Data | Weights kept? | Purpose |
|---|---|---|---|---|
| CI smoke (every push) | throwaway ~0.13M (2 layers, 64 dim, vocab 256+8 oracle ids) | ~200k **synthetic** byte-tokens (seeded pattern, 5% unlearnable noise) | **No** — weights *and* its run record die with the ephemeral runner; the repo ledger only ever holds committed real runs | prove the factory: data → train → learn → record → validate |
| E3 / N1 (Kaggle) | **the control**: 25.8M vanilla GPT-2-style (deliberately plain: LayerNorm, GELU, tied embeddings, AdamW) | 500M **real** byte-tokens, FineWeb-Edu | checkpoint downloadable from Kaggle (DVC versioning arrives in W2); its **record** is committed at E5 — the ledger's first inhabitant | define the baseline every idea must beat |
| N2 → waves | **the cortex, one organ per run**: same size, same tokens, same seeds as the control, plus the feature under test (Hopfield variant, delta-rule variant, oracle curriculum, …) | same declared mix (NOW-7) | per advancement-rule outcome | each organ earns its place *against the control* |
| post-N5 | the validated **combination**, retrained as one; then the 1–3B scale-up | granted/rented compute | yes — the release candidates | the model people will actually use |

In one sentence: **so far we have only trained the proof that the factory works; E3 trains the reference; the waves train the brain.** "The final model" does not exist yet *by design* — it will be the combination of whatever beats the control on the ledger.

## N2 first result (2026-09-07) — the first Pareto point that decides

The first complete MQAR ablation (12 tiers × control/hopfield/delta, 9h GPU):
**both associative memories beat the control** (AUC: hopfield +14%, delta +2%; verdict "advances"). Per-tier each path has a domain (**hopfield 6, delta 2, control 3**). The **upper-envelope multi-path router (RES-18) reaches +21% vs control, +6% over the best single path** — validating the router design *by data before it is built*. Hopfield is **structure-limited** (capacity sweep: more slots degrade) — the way past its ceiling is combining paths. DeltaMemory was stabilised (L2-normalised keys) and the circuit breaker proved itself. Full artefact: `metrics/mqar/`. **The RES-18 router is now implemented (A→B→C→D) and green** — the learned router (predicts from context, never reads the truth) captures ~88-93% of the oracle's lift over the best single path (`metrics/mqar/router-ablation-*.json`): the multi-path design pays off in practice, confirming the +21% upper-envelope prediction end-to-end. **Next:** wire the router into the real language model; then N3 (C3 oracle) or C2b (persistent memory → continuity D15).

## How we read results (ADR-004 / D16) — read this before interpreting any number

Optimisations are **not linear**, so results are never read as a single score. The law (full text: `adr/ADR-004-evaluation-pareto.md`):
- **A Pareto frontier over declared axes, never a scalar.** Axes: the continuity triad (episodic persistence, oracle-shift revision, non-regeneration) + the two efficiency denominators (per parameter, per bit). The ≤2% perplexity rule is a *guard* to be cleared, not an axis.
- **"Progress" = moving the frontier**, not climbing a number: a candidate advances if it is strictly better on ≥1 axis without regressing another past the guard. The ledger stores the frontier; the N1 control is its lower-left corner (organ-free — a reference, not a capability).
- **Three corollaries:** organs ablated *in isolation to understand*, the validated combination *retrained as one to decide* (non-additivity is expected); the **joint** effect of co-shipped features is recorded next to the sum of isolated effects; each record notes the **addressed composition** that produced it. No composite score is ever computed.

## What we publish (D17) — the twin of D16

*We measure on a frontier, therefore that frontier — and only that — is what we show.* Decision D17 (revises D6, full text in `adr/ADR-004-evaluation-pareto.md`): **open-results, not full open-source.**
- **Public:** the ledger (`metrics/`), the benchmarks, the run-v1 schema, the README (thesis + signature capability) — the Pareto frontier is the public object.
- **Private until an explicit founder decision:** the innovation register, the design ADRs, the concept dives, the extractions, the core code. The recipe stays in the workshop.
- **Reversible**, an assumed IP trade-off for a solo founder; it narrows D6's adoption flywheel, knowingly. **EuroHPC-compatible** (Open Science = open *results*, not open *code*; Win2Win SRL is eligible as EU industry). **Mechanism deferred to N5** — the repo simply stays private until there are real results to show.

## Track 1 — Execution (E-list, from `EXECUTION-2026-08-07.md`)

| Step | What happens | Status |
|---|---|---|
| E1 | Repo born: extract, init, push; hub records decision 019 + reference page | ✅ 2026-09-05 |
| E2 | CI self-test: `validate-ledger` + `smoke-train` green on every push | ✅ 2/2 (smoke 44 s, ledger 9 s) |
| E3 | First real run: Kaggle T4, 25.8M control on 500M FineWeb-Edu byte-tokens. Ran in **3 sessions** (12 h wall at ~10k tok/s ⇒ 13.7 h needed; budgeted `--time-budget-min 645` + auto-resume from Input-mounted checkpoint chained V1→V2→V3). Both August planning errors recorded in EXECUTION, corrected in opposite directions | ✅ 2026-09-06 |
| E4 | Sanity read passed: `completed`, `anomalies: null`, val_ppl 2.601, 30,517 steps | ✅ |
| E5 | Record committed (`be1fa8139f59`) + LATEST regenerated → **N1 gate PASSED**, CI green. Public-visibility switch deferred (founder: open-results stance under discussion) | ✅ 2026-09-06 |
| E6 | GATE → EuroHPC **Development Access** application citing the run_id. **2026-09-06 note:** Win2Win SRL (industry, EU) is eligible; Development Access is continuously open, monthly cut-offs (1st), ~2–3 week access; "open-results" stance is compatible (Open Science = open *results*, not open *code*) — verify the call text at submission | pending |
| E7 | Stage 1 on EuroHPC: Apptainer, pre-staged data, `provider:"eurohpc"` | pending |

## Track 2 — Build waves (from ADR-003; features from FEATURES/register)

### Pre-N2 — mandatory prerequisite
- **NOW-7 · Data first-class** (designed: `adr/ADR-005-data-composition-layer.md`): the mix is a **hashed DAG of typed operators** — lazy/streaming, comparable iff same `mix_hash`, five structural invariants tested from the start (incl. `mix([S,S])≡S`), decontamination as a Bloom-filter algebra, weight a pluggable resolver (the RES-14 data-β seam). Engine from-scratch over licensed sources (FineWeb-Edu ODC-By, OLMo per-shard) on credited recipes. Build wide, execute narrow: **Slice 1** (operator interface + DAG + hashing + source/decontaminate/mix + invariant harness + `mix_hash` schema field + N1 retro-migration) is all N2 needs; memoisation, Bloom-at-scale, and the data-β estimator are later slices. Source leads: OLMo; later fed by **RES-12**.

### N2 = W1 — the C2 core family (memory in the weights)
- Integrate **two associative variants against the control**: Hopfield/energy (RES-1) **and** gated delta-rule (extraction K1–K2 as the numeric starting shape) — both as *stackable attention-layer citizens* (hub-019 constraint, promoted into the hub decision at this step).
- **F1** optimizer ablation: NorMuon-class vs AdamW (re-implemented from papers, no code copied).
- Benchmarks live: **MQAR** via zoology as a dependency (F5–F6), **serial-position curve** (Rev7.b).
- Gate per variant: advancement rule (≥1 capability win, ≤2% ppl, tolerance frozen pre-run).

### N3 = W1b — the C3 family (volatility)
- **NOW-1** oracle-in-pretraining curriculum (token space already reserved in the tokenizer since N1).
- **Thalamic admission gateway** (C3's front door, one admission policy) and **collicular interrupt shortcut** (mid-pass modulation).
- Benchmark: oracle-shift recovery. Law: **NOW-9** (signed oracle provenance + source trust) is binding *before any public C3 endpoint*.

### N4–N5 = W2 — persistent memory, contracts, economics
- **C2b persistent tier**: entry-format law (pointers never blobs), the **DG → CA3 → CA1 write pipeline** (separate → associate → compare; surprise becomes computed), retention law incl. **D4.8** noise ruling (admission by surprise, verdict by consolidation).
- **H.M. protocol goes live** as the standing diagnostic (thresholds frozen in `benchmarks/hm-protocol.md`), including the eviction-safety variant.
- **RES-8**: typed span references + per-span contract stack (C4) + typed diffs; **habit cache** (C6 extension); **NOW-2** speculative graft (the cerebellum drafter) with its quality-per-dollar curve; **NOW-3** graft fidelity across matryoshka resolutions; **F8** DVC for slices/checkpoints.
- Deliverable: **ablation report v0** published with losses → **gates open: EuroHPC Fast Lane + TRC applications** citing run_ids.

### Post-N5 = W3 — regulation and behavior
- **RES-9 endocrine bus** (stress/arousal/reward; homeostat on entropy and budgets; consolidation rhythms; salience single-source), **heterogeneous experts + mixture-of-depths** (Rev7.a), **insula** telemetry-as-input, **RES-4** trust-weighted routing, **NOW-8** organ-use post-training (consult / cite / heed / comply), **NOW-5** the live **continuity demo** (ships once C3 passes).

### W4 — research wave
- **RES-10** connectome (commissure, wiring budget, connectivity profiles), **TEM** structure/content factorization (RES-3), **RES-11** reverse-replay credit assignment, **RES-13** latent recursive refinement, **RES-6** cache-oblivious planning + **RES-5** heterogeneous placement. **RES-7** (ternary, capability-per-bit) is wave-flexible: it can run as a parallel ablation any time after N2.

### Parked (SPEC — unlock conditions on file)
SPEC-1 dual-stream determinism · SPEC-2 oracle landscape editing (after RES-1 data) · SPEC-3 φ/Zeckendorf splitting (after RES-6 exists) · SPEC-4 private-inference flag (after N5 + a named use case).

## Always-on laws (no stage, no exceptions)
A run without its committed record does not exist · every brain-derived feature ships behind a flag, default off · advancement rule per feature · portability (ADR-002 / NOW-4) binding · anti-plagiarism protocol binding (core from scratch) · unknowns are null · **the north metric is the continuity triad (D15): H.M. persistence + oracle-shift revision + reference ratio — never blended**.

## Pointers
Normative: `roadmap.md` · `adr/ADR-003-brain-feature-map.md` · `IDEAS-REGISTER-2026-08-07.md` · `FEATURES-2026-08-07.md` · `EXECUTION-2026-08-07.md` · hub `decisions/019` + `reference/quantum-cortex.md`. Deep-dive schedule: `concepts/README.md`.
