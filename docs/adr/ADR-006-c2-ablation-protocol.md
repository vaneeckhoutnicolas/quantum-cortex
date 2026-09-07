# ADR 006 — The C2 ablation protocol (N2 = Wave 1)

- **Status:** Accepted (Nicolas Van Eeckhout, 2026-09-06)
- **Deciders:** Nicolas Van Eeckhout
- **Context:** N1 is committed (control `be1fa8139f59`, val_ppl 2.601) and the data terrain is a hashed, comparable object (ADR-005). N2 is the project's **first scientific question**: does an associative-memory layer (C2) earn its place against the vanilla control? This ADR fixes the protocol *before* any run, so results are read honestly (ADR-004/D16).
- **Cites:** hub-019 (advancement rule), ADR-003 (waves, flags default-off, guards), ADR-004/D16 (Pareto reading), ADR-005 (mix_hash comparability), register RES-1 (energy/Hopfield), RES-2 (test-time memory), the K3 extraction K1 (KDA = gated delta rule), F5–F6 (MQAR via zoology), Rev7.b (serial-position curve).

## Decision 1 — What is under test

**C2, the associative-memory layer**, in two variants from the *same family* (fast-weight associative memory), each a third residual sub-block inside every transformer `Block`, **behind a config flag, default off** (ADR-003):

- **Variant H — Hopfield/energy (RES-1):** modern-Hopfield-style content-addressable retrieval — a learned key/value memory read by softmax attention over stored patterns (energy-descent retrieval). The "magnetism" formulation.
- **Variant D — gated delta rule (K1, KDA-class):** a per-head recurrent state `S` updated by an **error-correcting rank-one write** `S += β·k·(v−Sᵀk)ᵀ` with a per-channel forget gate — the Widrow-Hoff family, the honest baseline the K3 extraction identified.

**Why two variants of one family:** the honest bar is beating the *incumbent* of the family (delta-rule, industrially validated by KDA/DeltaNet/xLSTM), not only a vanilla transformer. H vs D vs control is the first real Pareto comparison.

## Decision 2 — The control is sacred and equal-at-everything

Every ablation run is identical to the control **except** the feature under test: same `mix_hash` (the FineWeb-Edu manifest, ADR-005), same params budget (the associative layer's parameters are counted; where needed the baseline is width-matched so total params are comparable — the delta is stated in the record), same seed, same tokens, same schedule. **Flags off ⇒ the model is bit-identical to the N1 control** (a test asserts this). No comparison is valid across different `mix_hash`.

## Decision 3 — The benchmarks (frozen here, before any run)

- **MQAR (multi-query associative recall)** via the **zoology** dependency (F5–F6): the canonical probe for associative memory — the capability H and D are supposed to add. Reported as `mqar_accuracy`.
- **Serial-position curve** (Rev7.b): primacy/recency profile measured like an experimental-psychology protocol — cheap, original, and directly probes memory behaviour. The curve is the deliverable, plus a scalar summary.
- **Perplexity** stays the guard (≤2% degradation vs control, hub-019), never the axis.
- **Continuity triad (D15)** is *not yet* fully measurable (H.M. needs C2b, W2) — N2 measures the associative-capability axes; the triad arrives with the persistent tier.

## Decision 4 — The advancement rule, applied

A variant **advances** (its flag may default on in a later build) iff it extends the Pareto frontier (ADR-004): **≥1 declared capability win** (MQAR accuracy, serial-position recall) **with ≤2% perplexity degradation**, tolerance frozen here *before* the run. Losses are published. If neither H nor D clears the bar, that is a recorded result — the vanilla control stands, and the register notes it.

## Decision 5 — How the record carries the comparison

The `comparison` slot (already top-level in run-v1) is populated for ablation runs: `{"baseline_run_id": "be1fa8139f59", "same_mix_hash": true, "capability_deltas": {"mqar_accuracy": …, "serial_position_recall": …}, "ppl_delta_pct": …, "verdict": "advances" | "held"}`. Two ablation runs are comparable to each other and to the control iff they share `mix_hash` (ADR-005). This makes the frontier machine-readable.

## Decision 6 — Sequencing (measure-first, one variable at a time)

1. **Slice A (code, no run) — IMPLEMENTED 2026-09-06:** the two associative layers in the model behind `c2_variant: none | hopfield | delta` (default `none`), parameter-counted, with a **bit-identity test** (flag off ⇒ control unchanged) and a **forward-shape test** (flags on ⇒ model runs, shapes intact). No training yet.
2. **Slice B (benchmarks, no full run) — IMPLEMENTED & GREEN 2026-09-06:** `cortex_eval/` — MQAR by difficulty tiers (`mqar.py`, curriculum kv×seq, the curve as deliverable), the serial-position curve (`serial_position.py`, primacy/recency/U-shape, Rev7.b), and a standalone MQAR runner (`run_mqar.py`, option (a): trains a small model per tier, reusing train.py's VanillaGPT so the C2 layers under test are the exact ones N2 trains). Executable proof: a trained control beats chance on easy MQAR (acc 0.25 vs ~0.03 chance) — the pipeline works; the signal is non-flat. The circuit breaker (D7) applies per tier.
3. **Slice C (the runs):** train control + H + D at equal everything on the FineWeb-Edu manifest; populate `comparison`; publish the frontier. This is the GPU step, at N2 proper.

## Decision 7 — The circuit breaker (amendment, 2026-09-06): aggressive early-abort with bounded refine-and-retry

The advancement rule says when a variable *wins*; this says when a variable *must die*. An ablation that destabilises must not be run to term — it burns GPU on a known result ("it breaks"), and inside a combination an unstable variable masks the others' signal. So a run gains a third verdict beside "advances" and "held": **"aborted — unstable"**, and — the founder's refinement — an abort triggers a **bounded refine-and-retry loop**, exactly as noise is eliminated then re-admitted refined (D4.8).

**Abort criteria (aggressive; declared before the run, never tuned post-hoc — the honesty guard):**
- **Hard:** NaN/Inf in the loss → immediate abort, non-negotiable.
- **Divergence:** train loss > **2× the control's loss at the same step** over a window of 3 consecutive evals → abort (aggressive multiplier; the founder chose aggressive over conservative, paired with retry).
- **Stagnation:** no val improvement over 5 consecutive evals *while the control is still improving* → abort.

**Refine-and-retry (bounded — the guard that keeps aggressive cheaper than conservative):** an abort does not end the variable; it triggers up to **2 retries**, each applying a **pre-declared** refinement in order: (1) tighter gradient clip (grad_clip ×0.5), (2) lower LR (×0.5); a third failure is **permanent death**. Refinements are declared before the run so the loop is deterministic and honest. Without this bound, aggressive-with-retry could cost *more* GPU than conservative-without-retry — the bound is mandatory.

**Ledger (honest data — deaths are published, including resurrected-then-re-killed):** every attempt records `status: "aborted"` with `anomalies` naming cause + step + attempt (e.g. `"diverged at step 4200: loss 2.3× baseline, attempt 2/3"`). A variable that recovers records `"recovered-after-refine"` in its notes with the refinement that saved it. An aborted variable is a **result on the frontier** ("this variable at this setting destabilises"), not a non-observation.

**Conservative on the slow, ruthless on the broken:** NaN is instant death; divergence/stagnation are windowed (3–5 evals) so a slowly-converging recurrent architecture (Mamba2-style "fails then learns") is not killed prematurely. Better a wasted run than a good path buried — except NaN, which is beyond appeal.

**One primitive at three scales (RES-17 instance):** this circuit breaker (protocol level — kill a variable) is the same law as the **hyperdirect veto** (decode level — abort mid-decode on contract violation, ADR-003 basal-ganglia) and **eviction** (memory level — kill noise that never consolidates, D4.8): *cut cleanly what destabilises to preserve the whole*. A variable transitions live → destabilising → killed-or-refined — a textbook RES-17 type×state transition.

## Decision 8 — Multi-path C2: the router design (amendment, 2026-09-07, post-first-result)

The first complete ablation (hopfield +14% AUC, delta +2%, control baseline; per-tier hopfield 6 / delta 2 / **control 3** — each has a domain; upper-envelope router +21% vs control, +6% over Hopfield; Hopfield capacity sweep → **structure-limited**, enlarging it does not help) motivates evolving C2 from *one memory* to a **router of memories on a control floor** (full rationale + unification: register RES-18). Actionable design:

- **Paths:** control (floor, always available), Hopfield, delta. Router selects per span.
- **Floor guarantee:** a memory is used only where it beats the control; else the control. C2 cannot degrade the model. *The data proves the floor is needed — control won 3/12 tiers.* Falling to control is a result, not a failure.
- **Regime = consolidation:** hard routing (pre-compute, one path) where learned/confident/stable; soft (compute + weight) where unsure. Migrate soft→hard as learned (cost falls, precision rises). Hardening threshold **declared, measurable, conservative**.
- **Breaker = fallback:** on degrade/diverge, re-route to the best remaining path above the control (else floor), re-softening. **Declared hysteresis** prevents oscillation. Completes the hyperdirect veto (redirect, not void).
- **Test:** router AUC ≥ max single-path AUC (ceiling shown: +21% envelope); mean cost falls; no flapping.

Implementation order (measure-first, slices, each tested before the next): **(A)** offline upper-envelope oracle — pick per-span the best of the three *known* accuracies, bounding the ceiling (the ablation already gives this: +21%); **(B)** a soft router (learned gate over the three paths) trained on MQAR; **(C)** hardening + hysteresis + breaker-fallback wired; **(D)** the ablation: router vs each single path — must reach the envelope.

## Consequences

`train.py`'s `Config` gains `c2_variant` (default `none`) and the associative layers ship behind it, default off — N1 reproducibility is untouched, and a test proves it. The benchmarks are evaluation-only until Slice C. The advancement rule and tolerances are frozen here, before numbers exist — the honest order. N2 produces the first frontier points that *decide* something; until Slice C runs, nothing is claimed.
