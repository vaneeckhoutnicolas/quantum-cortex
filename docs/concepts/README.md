# concepts/ — index and deep-dive schedule

**What lives here.** Concept documents: the biology-to-architecture groundwork behind the register and ADR-003. Two exist today, and that is a *policy*, not an oversight:

- **`brain-atlas.md`** — the systematic review of **24 regions**. The amygdala, hypothalamus **and pituitary**, pineal, thalamus, colliculi, basal ganglia, insula, cerebellum, ACC, Broca/Wernicke, corpus callosum and the rest are **all covered there**, each with its status (mapped / candidate / deliberately not transposed) and its landing point in the architecture.
- **`hippocampus.md`** — the first deep dive, because memory is the signature capability (D15) and its wave (W1/W2) comes first.

**The rule (no empty scaffolding, ever):** a region earns its deep dive **when its wave opens** — written at the moment the mechanisms get implemented and ablated, so the document carries evidence, not padding. Until then, the atlas entry is the truth.

## Deep-dive queue (trigger → planned document)

| Region cluster | Trigger | Planned dive |
|---|---|---|
| Thalamus + colliculi (admission gateway, interrupt shortcut) | N3 / W1b opens | `thalamus.md` |
| Basal ganglia (selection, dopamine, habit cache) | habit cache lands (W2) | `basal-ganglia.md` |
| Amygdala + hypothalamus + **pituitary** + pineal + locus coeruleus — the endocrine cluster | RES-9 bus development (W3) | `endocrine-axis.md` |
| Cerebellum (forward model, timing) | NOW-2 speculative-graft work (W2) | `cerebellum.md` |
| Insula + ACC (interoception, conflict) | W3 | `interoception.md` |
| Hemispheres, corpus callosum, connectome profiles | RES-10 (W4) | `connectome.md` |

Anyone may request an earlier dive by opening an issue with the use case; the promotion rule applies as everywhere — a dive without its wave ships no claims, only groundwork.
