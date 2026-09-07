# WHITEPAPER — "Continuity as a first-class capability" — living skeleton

**Status:** deliverable specified 2026-09-07 (D19). This file is the paper's **living skeleton**: it
grows by consolidation (RES-19) — a section is filled only with claims that have passed the
**admission gate** below. Everything else lives in *Open questions*, explicitly labelled. The
skeleton is versioned like everything else: dated additions, prior text kept.

**Why a paper, and why now:** D19 says the method is as much the contribution as the model — the
demonstration that an individual with an AI alter-ego can build something significant *by rigour,
not force*. A paper is how that demonstration reaches the world. Specifying it early keeps every
chantier honest about what it must produce to be *publishable*, not just *interesting*.

## The admission gate (a claim enters a results section only if ALL hold)

1. **Statistical** — measured on ≥ 3 seeds, the gap significant at 95% (paired test, `cortex_eval/multiseed.py`). A single seed is an anecdote, not a claim.
2. **Anchored** — situated against the external references (`cortex_eval/external_baseline.py`, the recurrent ladder) at equal size, tiers, steps.
3. **Real, not toy** — demonstrated on the real task (real MQAR spans / real language-model runs), not only a synthetic surrogate. (Level 1b showed a 93% toy result collapsing to 25% on real spans — this rule exists because of that.)
4. **Reproducible** — a committed artefact (`metrics/`), a config hash, a run_id; the code that produced it is in the repo.
5. **Honestly bounded** — reserves stated (seeds, scale, single capability); losses and retrogradations published alongside.

A finding that fails any rule is an **open question**, not a result — and open questions are a legitimate, valued part of the paper (they are what a solo founder can *pose* to the field).

## Skeleton

### 0. Abstract
*(written last)*

### 1. Thesis — continuity as a capability
- The frontier game is closed (D3); the open question is **continuity** (D15): a model that remembers your project and visibly changes its mind when the world changes.
- Measured as a **triad, never a scalar**: episodic persistence (H.M. protocol), adaptive revision (oracle-shift), non-regeneration of the known (reference ratio). [ADR-004/D16]
- Positioning: a big fish in a small pond — define the question, own the efficiency niche, make the method the contribution. [D19]

### 2. Method — how an individual and an AI alter-ego built this
- Measure-first; a run without its committed record does not exist (ADR-001, the ledger).
- One variable at a time, control equal-at-everything (hub-019), flags default-off (ADR-003).
- Tests from day one — the plumbing invariants caught **five real bugs** before they could poison an ablation (mix duplication semantics ×2, zero-init overwritten by global init, hysteresis ordering, DeltaMemory divergence).
- Errors noted, never erased: dated amendments, retrogradations published (Rev20).
- The circuit breaker (ADR-006 D7): aggressive early-abort with bounded refine-retry — proven in production (12 clean aborts).
- Data as a hashed, comparable object (ADR-005, `mix_hash`).
- *This section is the D19 demonstration; the repo is its evidence.*

### 3. Architecture — the cortex and the C2 memory router
- The six components C1–C6 (hub-019); the brain-derived feature map (ADR-003).
- **C2 as a multi-path router on a control floor** (RES-18): a memory is used only where it beats the control — C2 can never degrade the model; routing regime follows consolidation (hard-on-learned, soft-on-new); the breaker as re-routing fallback with hysteresis. Versioned, retro-compatible `Router` contract (v1 oracle, v2 learned gate).
- Consolidation as one law at three scales (RES-16/RES-19): model, architecture, project.

### 4. Results — what has passed the gate
*(only gate-passing claims; each with artefact + seeds + CI)*
- **[ADMISSIBLE, pending multi-seed]** N1 baseline: control `be1fa8139f59`, val_ppl 2.601 on 500M FineWeb-Edu byte-tokens. *(reproducible, anchored; single run — a baseline, not a comparison, so rule 1 does not apply)*
- **[PENDING gate — needs 5 seeds]** Both associative memories beat the control on MQAR (hopfield +14% AUC, delta +2%, 12 tiers). Per-tier domains: hopfield 6 / delta 2 / control 3. Oracle upper-envelope +21%.
- **[PENDING gate — needs multi-seed]** Hopfield is structure-limited (capacity sweep: more slots degrade).
- **[PENDING gate — needs 1500 steps + multi-seed]** The recurrent ladder: pure recurrents stay ~1/4 of the hybrids at every rung.

### 5. Open questions (valued, explicitly not results)
- **The learned router on real spans** — 93% of the oracle ceiling on a synthetic task, ~25% on real MQAR spans (12 tiers, under-sampled). Needs thousands of real spans (router wired into the LM). [Rev20]
- **Does the delta-rule pay only when hybridised?** — regresses when pure (ladder L4) yet excels inside the hybrid. Candidate finding; unconfirmed. [ladder mini-run]
- **The continuity triad itself** — not yet measurable: C2b (persistent tier) and C3 (oracle) are not built. The paper's central capability is, today, a specified protocol (H.M., `docs/benchmarks/`) awaiting its organs.
- **Scale** — everything is 25M / MQAR-scale; a 100M+ run (EuroHPC) is required before any capability claim generalises.

### 6. Limitations & honest reserves
Single seeds where noted; one capability (associative recall) measured so far; synthetic-vs-real gap demonstrated on our own router; no external pre-trained comparison yet (a non-equalised Mamba/RWKV reference is a separate, license-checked job).

### 7. Reproducibility
`docs/GETTING-STARTED.md`; `pytest tests/` (the CI runs it); the ledger `metrics/runs.jsonl`; artefacts `metrics/mqar/`; every decision dated in `docs/adr/` and the hub decision log D1–D19.

## Gates for the paper itself
- **v0 (internal)** may be drafted once §4 holds ≥ 1 gate-passing comparison (the 5-seed C2 ablation).
- **v1 (submittable)** requires the continuity triad measured on at least one organ (C2b or C3 built) **and** a 100M+ run — i.e. the capability the paper is named after must exist as a measurement, not a specification.
- Target: arXiv (cs.LG), then a workshop; open-results (D17): the paper, ledger and benchmarks are public; the recipe stays the workshop's.
