# The Thalamus (and colliculi) — admission and interruption — living groundwork

**Status banner.** Written 2026-09-05 **ahead of its wave** (founder decision: every cluster gets its dedicated page upfront). The atlas remains the index; this page carries **groundwork, zero claims** — evidence arrives when **N3 / W1b** opens, as dated amendments. Statuses: [mapped] / [candidate] / [noted].

## 1. Biology, the parts that compute

- **The relay.** Nearly every input to cortex (all senses except olfaction) passes through a thalamic nucleus — the brain runs a single admission architecture, not per-sense side doors.
- **TRN — the thalamic reticular nucleus.** A thin inhibitory shell around the thalamus that gates which relays fire: the classic "attentional searchlight". Admission is *learned and dynamic*, not a fixed filter.
- **Drivers vs modulators (Sherman & Guillery).** Thalamic inputs split into two classes: *drivers* carry content; *modulators* adjust gain without carrying content. The brain types its inputs.
- **First-order vs higher-order nuclei.** First-order relays carry the world inward; higher-order nuclei relay **cortex to cortex** — inter-area traffic re-enters through the same gateway.
- **Colliculi (midbrain, grouped here).** Fast orienting to salient events *before* full cortical processing — a low-latency path that can redirect attention mid-stream.

## 2. Transposition

- **The admission gateway [candidate → C3, W1b].** Every external channel — user input, oracle events, QM tool results — passes **one** gate with one policy: prioritization, filtering, per-channel quotas. No side doors; the thalamic single-architecture principle becomes a hard interface rule.
- **Learned admission (TRN) [candidate].** The gate's mask is trained with the model, not hand-coded: admission scores conditioned on current state — the searchlight as parameters.
- **Typed inputs (drivers/modulators) [mapped — consistency ruling 5 of ADR-003].** Content tokens enter as drivers; modulation travels only on the RES-9 bus. The Sherman distinction is already our law: *no component reads oracle events to self-modulate directly*.
- **Higher-order relay [noted → RES-10].** Inter-zone traffic re-entering through the gateway gives the connectome a natural checkpoint — recorded for W4, where wiring policy lives.
- **Collicular interrupt [candidate → C3, W1b].** A salient signed event may modulate gating **within the current forward pass** under a strict latency budget, instead of waiting for the next context turn. Security note: interrupts obey **NOW-9** (signed provenance, source trust) — an unsigned interrupt is quarantined.

## 3. Tests (frozen shapes; thresholds set at wave opening, before runs)

Admission ablation (gateway vs raw concatenation: oracle-shift recovery, noise-robustness under channel flooding) · interrupt latency and post-interrupt recovery quality · per-channel quota stress (one channel flooding must not starve the others) · TRN-learned vs fixed-priority admission at equal params.

## Summary

| Mechanism | Status | Lands on |
|---|---|---|
| Single admission gateway | candidate | C3 front door (W1b) |
| Learned admission mask (TRN) | candidate | C3 |
| Drivers vs modulators | mapped | ADR-003 ruling 5 / RES-9 |
| Higher-order re-entry | noted | RES-10 (W4) |
| Collicular interrupt | candidate | C3 (W1b), under NOW-9 |

Promotion rule unchanged: ledger runs decide; this page only prepares them.
