# The Endocrine Axis — amygdala, hypothalamus, pituitary, pineal, locus coeruleus — living groundwork

**Status banner.** Written 2026-09-05 ahead of its wave (founder decision). Groundwork for **RES-9 (W3)**, zero claims; evidence arrives as dated amendments. This is the dedicated page the atlas's scattered entries pointed to.

## 1. Biology, the parts that compute

- **Hypothalamus.** The setpoint machine: temperature, energy, arousal held in ranges by closed loops; commands the pituitary.
- **Pituitary.** The actuator — it does not decide, it *executes* hormone release on hypothalamic command (HPA axis: hypothalamus → pituitary → adrenal → cortisol).
- **Cortisol and the inverted U.** Stress performance follows Yerkes–Dodson: moderate arousal improves focused performance, excess degrades it — **the curve has an optimum, it is not monotonic**. Stress also biases memory: consolidation of the salient improves while peripheral detail and retrieval suffer.
- **Amygdala.** Salience and threat tagging; it modulates *what gets consolidated* rather than storing content itself.
- **Locus coeruleus / noradrenaline — adaptive gain.** High *phasic* NA (bursts on task-relevant events) supports **exploitation**; high *tonic* NA supports disengagement and **exploration**. One neuromodulator implements the explore/exploit dial.
- **Pineal.** Melatonin rhythm: the clock that schedules offline states.

## 2. Transposition

- **The bus [mapped → RES-9].** A low-dimensional broadcast (stress / arousal / reward); gates listen to the bus only (ADR-003 ruling 5).
- **The inverted U made law [candidate — new precision].** The stress scalar's effect must be **tested as a curve, never assumed monotonic**: the stress-response benchmark sweeps the scalar and expects an optimum (accuracy on high-stakes items rises then falls; diversity falls throughout). A monotonic implementation would be biologically and empirically wrong.
- **Homeostat + actuator [mapped → RES-9 / C6].** The hypothalamic loop holds entropy bands and token/joule/$ budgets as *regulated states*; the pituitary is the small component that actually writes gate values — decision and actuation kept separate, like the biology.
- **Amygdala = salience, single source [mapped].** One importance score on journal entries (write gain, retrieval weight, consolidation and reverse-replay priority). CA1 computes surprise; amygdala-path computes *importance* — two signals, two owners, no overlap.
- **LC adaptive gain [candidate — new mechanism surfaced by this page].** The arousal channel becomes an **explore/exploit dial for decoding**: phasic mode → low temperature, exploit (high-stakes spans); tonic mode → higher temperature and broader routing, explore (open-ended spans). Wired to C4 criticality and RES-13's depth dial — one coherent "effortful vs exploratory" regime switch.
- **Pineal rhythm [mapped → RES-9].** The consolidation scheduler: periodic replay of the journal into persistent memory (hippocampus page, §4), budget-pressured by the homeostat (retention law, ADR-003 Decision 4).

## 3. Tests (shapes frozen; thresholds at wave opening)

Stress sweep → the inverted-U curve itself is the deliverable (optimum located, both slopes shown, losses published) · setpoint hold under load (entropy band and byte budget maintained while flooding the gateway) · consolidation ablation (± replay phase on long-horizon recall) · salience retrieval lift vs recency-only · explore/exploit switch: quality on high-stakes vs idea-diversity on open-ended tasks, phasic vs tonic settings.

## Summary

| Mechanism | Status | Lands on |
|---|---|---|
| Modulation bus (stress/arousal/reward) | mapped | RES-9 (W3) |
| Inverted-U stress law | candidate — new | RES-9 benchmark design |
| Homeostat + pituitary actuator | mapped | RES-9 / C6 |
| Amygdala salience (single source) | mapped | journal metadata |
| LC adaptive gain → explore/exploit dial | candidate — new | decoding policy · C4 · RES-13 |
| Pineal consolidation rhythm | mapped | RES-9 / Decision 4 |

Promotion rule unchanged: ledger runs decide.
