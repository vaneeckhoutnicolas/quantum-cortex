# EXTRACTION — Kimi-K3 & kimi-k3-in-c — 2026-08-08

**Purpose:** harvest everything of value from the two repositories **as knowledge**, with per-item legal status. Method (the copyright lock): copyright protects *expression* (code text), never ideas, algorithms, mathematics, or measured facts. This document re-states mechanisms in our own words and equations, cites file:line as evidence, and ships **zero copied code**. Exposure by construction: none.

---

## Part A — MoonshotAI/Kimi-K3 (loaded 2026-08-08)

**Inventory:** exactly two files — `README.md` and `LICENSE`. **No code exists in this repository**; there is nothing copyable even in principle. Value = architecture facts + the license governing the weights.

**License, read in full ("Kimi K3 License", 52 lines):** an MIT-style broad grant explicitly covering weights, parameters, configuration, inference and training code, and docs — use, copy, modify, sell, fine-tune, derivatives all permitted — under three conditions: (1) keep the copyright/permission notice; (2) a **Model-as-a-Service clause**: an MaaS operator whose aggregate revenue exceeds $20M over any 12 months must sign a separate agreement with Moonshot before commercial use; (3) an **attribution clause**: products above 100M MAU or $20M monthly revenue must display "Kimi K3" prominently. **Reading for Win2Win:** all thresholds are orders of magnitude above us; grafting K3 (C5) via API or self-host is cleared at our scale; conditions documented here for the day they matter. "Kimi" remains Moonshot's trademark — nominative reference only, never in our naming.

**Facts retained (uncopyrightable):** 2.8T total parameters; Stable LatentMoE activating 16 of 896 experts; KDA + AttnRes; 1M context; MXFP4 weights / MXFP8 activations from quantization-aware training; OpenAI-compatible API.

---

## Part B — FareedKhan-dev/kimi-k3-in-c (Apache-2.0 verified; loaded, code read)

Each item: mechanism in our words → evidence → value for cortex → legal status. All items are **ideas/facts (free)**; the expression stays in their repo.

- **K1 · KDA is a gated delta rule — the headline find.** Per head, the recurrent state is a d_k×d_v matrix S. One token: (1) channel-wise forget, row i of S scaled by α_i (a per-key-channel gate, not a scalar); (2) read u = Sᵀk; (3) **rank-one error-correcting write S += β·k·(v−u)ᵀ — (v−u) is a prediction error**, making this a delta rule (Widrow-Hoff family), not accumulation; (4) output o = Sᵀq from the already-updated state. Evidence: `src/core/k3_ops.c:179–221`. **Value:** KDA and our C2 "magnetism" are two members of the same family — fast-weight associative memory (delta-rule vs energy/Hopfield retrieval). The C2 positioning section writes itself, and **N2 gains a recorded design input: the C2 ablation includes a gated-delta-rule baseline variant alongside the Hopfield/energy variant** — beating the family's incumbent, not only a vanilla transformer, is the honest bar.
- **K2 · Gate parametrization.** Decay computed in log space: g = lb·σ(e^{A_log_h}·(z+dt_bias)) with a learned per-head A_log and a lower bound lb, then α = e^g ∈ (e^{lb},1]. Evidence: `k3_ops.c:161–176`. **Value:** stable gating recipe for C2/C3 (the oracle channel modulating gates has a proven numeric shape to start from).
- **K3 · Hybrid layout as explicit config.** Full-attention (MLA) layers are an explicit index list in the config; every other layer is KDA; the first `first_dense` layers carry no MoE. Evidence: `k3_ops.c:81–88`, `K3Cfg`. **Value:** the implementation pattern for our C2 "stackable attention-layer citizen" constraint — layout is data, not code.
- **K4 · AttnRes mechanics.** Periodic residual snapshots are pushed onto a per-depth stack; before attention, earlier snapshots are re-aggregated through a folded (norm gain × scoring projection) vector — learned weighted re-injection of earlier depth states. Evidence: `k3_ops.c:925–1015`. **Value:** study material for the register's AttnRes entry; paper remains the source of truth.
- **K5 · The streaming economics, in measured numbers.** Routed experts are 1.45 TB of a 1.56 TB checkpoint; a decode step touches top-16 in each of 92 MoE layers = 1,472 experts × 17,547,264 bytes ≈ **25.83 GB of weights per token uncached ≈ 21 s/token at ~1.2 GB/s NVMe** — the cache hit rate is the single number that decides viability. Evidence: `src/cache/k3_cache.h` header. **Value:** hard sizing facts for RES-5/RES-6, and a frontier echo of C6's cache-affinity weight.
- **K6 · Cache packed, never dequantized.** One expert is 17.55 MB packed vs 132 MB in floats; mat-vec is memory-bound, so caching the packed form fits 7.5× more experts *and* is faster. Evidence: same header. **Value:** a serving principle for NOW-4 — quantized forms are first-class citizens end-to-end.
- **K7 · Fused dequant-matmul.** The 4-bit form is consumed directly inside the matmul. Evidence: `k3_ops.c:225` region. **Value:** if we ever need such a kernel it is Triton-and-budgeted per ADR-002 — pattern noted, implementation ours.
- **K8 · O_DIRECT streaming with graceful fallback.** The trunk bypasses the page cache (one-pass streams defeat it), on a dedicated async I/O thread, falling back if O_DIRECT is unavailable. Evidence: `src/io/k3_trunk.c:148–170`. **Value:** the disk-as-tier pattern RES-6 will need.
- **K9 · Deterministic scratch budgets.** Every op exposes a `*_scratch(cfg)` sizing function; memory planning is computed, not discovered. Evidence: `k3_ops.c:636, 800`. **Value:** an engineering discipline we adopt (our own implementation) in trainer and serving code.
- **K10 · Double-accumulator RMSNorm.** Sums of thousands of squared fp32 terms accumulate in double so outputs stay byte-identical across budgets. Evidence: `k3_ops.c:90–99`. **Value:** the numeric discipline behind reproducible evals — directly relevant to our ledger's credibility.
- **K11 · "The histogram is not instrumentation."** Hot-expert distribution is measured and feeds the replacement policy — measurement as a policy input, not decoration. Evidence: `k3_cache.h`. **Value:** measure-first embodied in systems code; the exact spirit of RES-5's measured pool.

---

## Part C — Legal verification (requested, performed, **PASS**)

1. **This delivery:** knowledge document only; mechanisms restated, equations, facts, file:line citations; **zero code copied into quantum-cortex** → no copyright exposure by construction.
2. **Kimi-K3 repo:** contains no code to copy; the weights license was read in full — permissive, with MaaS/attribution thresholds far above our scale; graft (C5) cleared, conditions documented.
3. **kimi-k3-in-c:** Apache-2.0 **verified** (includes its §3 patent grant). If a *peripheral* utility were ever adapted: LICENSE + NOTICE attribution + provenance header path exists — but the core rule stands regardless: **C1–C6 from scratch; C2 is implemented from the KDA paper's mathematics and this extraction, never from this code.**
4. **Trademark:** "Kimi" referenced nominatively only.
5. **Patents:** re-implementing published methods carries the usual theoretical surface; this is engineering discipline, not legal advice — a Kern-style legal audit precedes any commercial move (already binding in FEATURES).

## Part D — What changes now

Nothing in N1. Recorded for N2: the C2 ablation matrix includes a gated-delta-rule variant (K1) next to the Hopfield/energy variant, with K2's gate shape as the numeric starting point. Graft clearance for K3 is documented for the day C5 exists.
