# Brain Atlas — the systematic brain → cortex transposition — 2026-08-08

**Validation 2026-08-08:** every candidate in this document was validated by Nicolas and promoted into **ADR-003** (the canonical brain-feature map, wave-ordered, performance-guarded). Statuses below are kept as originally written — states are noted, never erased.

**First document of `docs/concepts/` (opened today, per the no-empty-scaffolding rule).**

**Method.** Region-by-region review of the human brain (sources reviewed 2026-08-08: fr.wikipedia *Aire cérébrale*, fr.wikiversity *Divisions du cerveau*, fr.wikipedia *Liste de régions du cerveau humain*), each region receiving one of three statuses: **[mapped → X]** already carried by an existing component or register entry; **[candidate → target]** recorded design input, promoted only at its own ablation; **[not transposed — reason]** deliberate refusal, stated. Honesty rule: the analogy is a design compass, never an argument — nothing ships without its ledger run. Two anatomical facts anchor the whole exercise: cortical areas differ in **cytoarchitecture** (Brodmann's histological map — areas are not uniform tissue), and area size/position **varies between individuals** — heterogeneity and inter-individual profiles are biological facts, not metaphors.

## 1. Telencephalon — cortex

- **Primary sensory areas (occipital vision, temporal audition, parietal somatosensory)** — modality-specific entry points. [mapped → C5 grafted zones: multimodal models attach as primary areas; the small cortex itself stays text-native.]
- **Associative cortex (parietal-temporal-occipital junctions)** — multimodal integration. [mapped → NOW-3 matryoshka states as the shared association space.]
- **Prefrontal cortex** — planning, judgment, inhibition. [mapped → the deliberation policies: quorum (propose/critique/endorse) and NOW-2 escalation are the slow, expensive path.]
- **Broca (production) vs Wernicke (comprehension)** — asymmetric specialization linked by the **arcuate fasciculus**, a dedicated tract. [mapped → Rev7.a heterogeneous experts (production-heavy vs comprehension-heavy roles); the arcuate becomes a **dedicated high-bandwidth link between specific zone pairs** — RES-10.]
- **Anterior cingulate (ACC)** — conflict monitoring, error detection. [mapped → the anomaly/conflict detector feeding trust scores (RES-4) and the stress scalar (RES-9).]
- **Insula** — interoception, sensing the body's internal state. [candidate → C6/C3: **interoception = the model reads its own telemetry as input** — uncertainty, energy, budget state fed back as tokens; confidence-aware generation. Promoted at its ablation.]
- **Motor cortex** — action execution. [not transposed into the model — actions belong to the body, and **the body is Quantum Meridian** (tools, agents, MCP). The founding division holds.]

## 2. Commissures & white matter

- **Corpus callosum and commissures** — the inter-hemispheric exchange. [mapped → **RES-10**: the connectome as a first-class parameter; a **bottlenecked commissure** between zone groups — limited-capacity broadcast forcing abstraction at the exchange (Global Workspace Theory, credited).]
- **White-matter wiring cost** — axons are expensive; the brain economizes long-range links. [mapped → RES-10's connection budget: sparsity of inter-zone wiring is declared and paid for, never free all-to-all.]

## 3. Limbic system

*The hippocampus is expanded in its own deep dive: `concepts/hippocampus.md` (trisynaptic pipeline, index theory, grid scaffold, replay in three directions, the H.M. ablation).*

- **Hippocampus — CA3 recurrent collaterals** — autoassociative pattern **completion**: the textbook biological Hopfield network. [mapped → C2/RES-1. The anatomy endorses the architecture: our associative memory has a literal biological incumbent.]
- **Hippocampus — dentate gyrus** — pattern **separation**: near-duplicates are made distinct before storage to avoid interference. [candidate → C2b: **decorrelate near-duplicate entries before journal writes** — a concrete anti-interference mechanism for the persistent tier. Promoted at N2+.]
- **Amygdala** — salience, fear, emotional tagging. [mapped → RES-9 salience tags.]
- **Hippocampal replay (during rest/sleep)** — systems consolidation to cortex. [mapped → RES-9 consolidation phases.]

## 4. Basal ganglia

- **Striatum/pallidum loops** — action *selection*, go/no-go gating. [mapped → C1's router as the selector; trust-weighted gating (RES-4).]
- **Dopamine** — reward prediction error driving learning and selection. [mapped → the reward channel of RES-9's bus; delta-rule kinship noted (prediction error is already C2's write signal — K1).]
- **Habit formation** — frequently rewarded sequences compile into cheap automatic paths. [candidate → C6: a **habit cache** — recurrent request patterns bypass full deliberation via cached routing decisions and semantic-cache hits. Promoted at its ablation.]

## 5. Diencephalon

- **Thalamus** — the relay all sensory input passes through; attention-gated admission to cortex. [candidate → formalize **C3's injection point as the thalamic gateway**: every external channel (user input, oracle events, QM tool results) passes one admission gate with prioritization — the ingestion layer gains a name and a policy.]
- **Hypothalamus** — homeostatic setpoints, drives, hormone control via the **pituitary**. [mapped → RES-9 homeostat; the pituitary is the bus's actuator — the component that physically writes gate values.]
- **Pineal gland** — circadian rhythm. [mapped → RES-9 rhythms/consolidation phases.]

## 6. Mesencephalon (midbrain)

- **Superior/inferior colliculi** — fast orienting reflexes to salient stimuli, *before* full cortical processing. [candidate → C3: a **low-latency interrupt shortcut** — a salient oracle event can modulate gating within the current forward pass instead of waiting for the next context turn. Promoted at its ablation.]
- **Reticular formation / locus coeruleus (noradrenaline)** — arousal, vigilance. [mapped → RES-9 arousal channel.]

## 7. Metencephalon

- **Cerebellum** — more neurons than the cortex; fine timing, coordination, and **forward models** predicting outcomes of intended actions. [mapped → **NOW-2's drafter is the cerebellum**: a small fast forward model predicting the cortex, corrected when wrong. Timing/scheduling affinity noted for the execution plane (RES-6).]
- **Pons** — white-matter relay. [not transposed — transport is infrastructure, not cognition.]

## 8. Myelencephalon & spinal cord

- **Medulla (breathing, heart rate, autonomic life support)** — [not transposed into the model — this is the **runtime**: health checks, watchdogs, restart policies live in serving code, deliberately outside the weights.]
- **Spinal reflex arcs** — sensor-to-actuator shortcuts bypassing the brain. [not transposed into the model — reflexes belong to the body: QM-side tool automations.]

## 9. Hemispheric organization & neurodiversity — the founder's insight

Two facts from the sources: areas differ in cytoarchitecture, and their size/position **vary between individuals**. The founder's own framing of his cognition — high inter-hemispheric activity, Asperger — is recorded here as the founding insight of **RES-10**: the **connectome is a first-class, declared, ablatable parameter**, and *connectivity profiles* (local-dense vs long-range-dense, at equal parameters) constitute a family of cognitive styles to be measured, not a single fixed wiring to be assumed. Scientific honesty: the literature on autistic connectivity is heterogeneous across studies; what we transpose is the **principle** — profiles differ between individuals and that diversity is functional — never a clinical claim. Neurodiversity, here, is a design space.

## Summary

| Status | Count | Entries |
|---|---|---|
| mapped | 15 | primary/associative areas, prefrontal, Broca/Wernicke, ACC, corpus callosum, wiring cost, CA3, amygdala, replay, selection loops, dopamine, hypothalamus+pituitary, pineal, reticular/LC, cerebellum |
| candidate | 5 | insula interoception, dentate-gyrus pattern separation, habit cache, thalamic gateway, collicular interrupt shortcut |
| not transposed | 4 | motor cortex (→ QM, the body), pons (transport), medulla (→ runtime), spinal reflexes (→ QM automations) |

Promotion rule unchanged: a candidate enters the register or an ADR only with a defined ablation, and ships only with its ledger run.
