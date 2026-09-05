# The Cerebellum — the forward model that learns from correction — living groundwork

**Status banner.** Written 2026-09-05 ahead of its wave (founder decision). Groundwork for **NOW-2 (W2)**, zero claims; evidence arrives as dated amendments.

## 1. Biology, the parts that compute

- **More neurons than the rest of the brain combined** — spent almost entirely on one job: **forward models**. The cerebellum predicts the sensory consequences of intended actions so the system can act on predictions instead of waiting for slow feedback.
- **Climbing fibers — the teaching signal.** Each Purkinje cell receives one climbing fiber from the inferior olive that fires on **prediction error**, driving plasticity: the forward model is *trained continuously by its own mistakes*.
- **Granule-layer expansion.** Mossy-fiber inputs are exploded onto ~50 billion tiny granule cells — a massive sparse expansion before the Purkinje stage: the separation trick again (the dentate gyrus's cousin).
- **Timing.** Sub-second temporal precision — the cerebellum is the brain's metronome.
- **Cognition too.** Cerebellar circuits loop with prefrontal and language areas; forward-modeling applies to thought, not only movement.

## 2. Transposition

- **The drafter is the cerebellum [mapped → NOW-2, W2].** A small fast model predicts the big system (the cortex standalone, or the grafted frontier zone), which verifies and overrides only where the prediction fails — act-on-prediction, correct-on-error. The quality-per-dollar curve is the deliverable.
- **Climbing-fiber learning [candidate — new mechanism surfaced by this page].** Every verifier override *is* a labeled error: log the (draft, correction) pairs and **fine-tune the drafter continuously from its own rejections** — online distillation from corrections. Acceptance rate should climb over deployment; the graft literally teaches the brain's fast path. Data loop governed by NOW-7 provenance rules.
- **Granule expansion [noted → C2b kinship].** The expand-then-sparsify motif appears twice in biology (DG, granule layer); our write pipeline already adopts it once — noted as convergent evidence, not a second implementation.
- **Timing [noted → RES-6].** Fine-grained scheduling affinity for the execution plane; recorded for W4, where the planner lives.

## 3. Tests (shapes frozen; thresholds at wave opening)

Quality-per-dollar and tokens/joule vs always-big and always-small baselines · **acceptance-rate trajectory** with climbing-fiber updates on vs frozen drafter (the online-learning ablation) · latency distribution of the draft-verify loop · regression guard: drafter updates must never degrade contract compliance (C4 check in the loop).

## Summary

| Mechanism | Status | Lands on |
|---|---|---|
| Forward-model drafter | mapped | NOW-2 (W2) |
| Climbing-fiber online distillation | candidate — new | NOW-2 + NOW-7 |
| Granule/DG expansion motif | noted | C2b (convergent evidence) |
| Timing | noted | RES-6 (W4) |

Promotion rule unchanged: ledger runs decide.
