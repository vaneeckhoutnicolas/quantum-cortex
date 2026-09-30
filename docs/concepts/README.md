# concepts/ — index and deep-dive schedule

**What lives here.** Concept documents: the biology-to-architecture groundwork behind the register and ADR-003. Two exist today, and that is a *policy*, not an oversight:

- **`brain-atlas.md`** — the systematic review of **24 regions**. The amygdala, hypothalamus **and pituitary**, pineal, thalamus, colliculi, basal ganglia, insula, cerebellum, ACC, Broca/Wernicke, corpus callosum and the rest are **all covered there**, each with its status (mapped / candidate / deliberately not transposed) and its landing point in the architecture.
- **`hippocampus.md`** — the first deep dive, because memory is the signature capability (D15) and its wave (W1/W2) comes first.

**The rule (no empty scaffolding, ever):** a region earns its deep dive **when its wave opens** — written at the moment the mechanisms get implemented and ablated, so the document carries evidence, not padding. Until then, the atlas entry is the truth.

**Amended 2026-09-05 (founder decision):** every queued cluster now gets its dedicated page **upfront, as living groundwork** — substantive biology→architecture prose with a status banner, updated by dated amendments. The wave still gates **claims and evidence**, never prose. Original rule kept above, as always.

## Deep-dive queue (trigger → planned document)

| Region cluster | Trigger | Planned dive |
|---|---|---|
| Thalamus + colliculi (admission gateway, interrupt shortcut) | N3 / W1b opens | `thalamus.md` — **page live** |
| Basal ganglia (selection, dopamine, habit cache) | habit cache lands (W2) | `basal-ganglia.md` — **page live** |
| Amygdala + hypothalamus + **pituitary** + pineal + locus coeruleus — the endocrine cluster | RES-9 bus development (W3) | `endocrine-axis.md` — **page live** |
| Cerebellum (forward model, timing) | NOW-2 speculative-graft work (W2) | `cerebellum.md` — **page live** |
| Insula + ACC (interoception, conflict) | W3 | `interoception.md` — **page live** |
| Hemispheres, corpus callosum, connectome profiles | RES-10 (W4) | `connectome.md` — **page live** |

Anyone may request an earlier dive by opening an issue with the use case; the promotion rule applies as everywhere — a dive without its wave ships no claims, only groundwork.

**Added 2026-09-30.** `door-reconstruction-note-2026-09-30.md` is not a brain concept: it is the door note that reads the founder's thermodynamics program for the organ (the reconstruction theorem and the four part partition), versed here as written, as a declaration (register Rev77; whitepaper §3.5, §4.7c and §5d item 7). It ships no claim.
