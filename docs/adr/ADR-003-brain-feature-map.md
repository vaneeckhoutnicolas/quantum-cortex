# ADR 003 — The brain-derived feature map: validated, consistent, performance-guarded

- **Status:** Accepted (validated by Nicolas Van Eeckhout, 2026-08-08)
- **Deciders:** Nicolas Van Eeckhout
- **Cites:** IDEAS-REGISTER (rev9), concepts/brain-atlas.md, concepts/hippocampus.md — per the register's promotion rule (dated ADR citing the register, never a silent edit).

## Context

The founder validated the recorded brain-derived candidates and asked for one guarantee: **every brain feature maps to a corresponding, consistent LLM feature — without sacrificing performance**. Nature is the compass; the ledger remains the judge. Caveat stated once: evolution optimizes under metabolic and developmental constraints we do not have — we transpose **computational principles**, never implementation details, and the deliberate non-transpositions stand (motor cortex → QM the body; pons → transport; medulla → runtime; spinal reflexes → QM automations).

## Decision 1 — The canonical map

One table rules them all. Columns: brain feature → cortex feature → owner → test → wave.

| # | Brain | Cortex feature | Owner | Test | Wave |
|---|---|---|---|---|---|
| 1 | Semantic areas | Semantic MoE zones | C1 | routing specialization MI | core (hub decision 019) |
| 2 | Cytoarchitecture varies (Broca) | Heterogeneous expert capacity + mixture-of-depths | C1 | hetero vs uniform, equal params | W3 |
| 3 | Corpus callosum, arcuate, wiring cost | Connectome first-class: bottleneck commissure, dedicated links, profiles | RES-10 | connectome & commissure sweeps | W4 |
| 4 | CA3 pattern completion | Associative retrieval (Hopfield/energy) | C2 / RES-1 | MQAR family | W1 (N2) |
| 5 | Delta-rule kinship (KDA, dopamine RPE) | Gated delta-rule baseline variant | C2 | MQAR family | W1 (N2) |
| 6 | Dentate gyrus separation | Expand-sparsify before journal writes | C2b | interference rate | W2 |
| 7 | CA1 comparator | **Computed** surprise at the memory interface | RES-2 input | write precision | W2 |
| 8 | Index theory | Journal entry = (cue, content-addressed pointer, salience, schema id) — never a blob | C2b / RES-8 | — design law, applies at first C2b code, no ablation needed | W2 |
| 9 | Grid cells / TEM | Structure–content factorization in the memory graph | RES-3 / C2b | zero-shot relational transfer | W4 |
| 10 | Forward replay, pineal rhythm | Consolidation phases (journal → persistent memory) | RES-9 | long-horizon recall ± replay | W3 |
| 11 | Reverse replay | Credit assignment along the journey (journal metadata, no BPTT) | RES-11 | delayed-outcome retrieval precision | W4 |
| 12 | Preplay | Plan sketching in memory space | parked | — | after RES-6/9 |
| 13 | Amygdala | Salience score, single source | RES-9 | retrieval lift | W3 |
| 14 | Cortisol / noradrenaline / dopamine | Modulation bus: stress / arousal / reward | RES-9 | stress-response tradeoff curve | W3 |
| 15 | Hypothalamus + pituitary | Homeostat + gate actuator | RES-9 / C6 | setpoint hold under load | W3 |
| 16 | Thalamus | Admission gateway — the formalization of C3's injection | C3 | shift recovery + admission ablation | W1b (N3) |
| 17 | Colliculi | Low-latency interrupt shortcut (mid-pass modulation) | C3 | interrupt latency & recovery | W1b (N3) |
| 18 | Insula interoception | Own telemetry fed back as input tokens | C6 → C3 | confidence-aware generation quality | W3 |
| 19 | Basal ganglia selection, dopamine gating | Trust-weighted routing | C1 / RES-4 | corrupted-expert robustness | W3 |
| 20 | Habit formation | Habit cache — an extension of the C6 semantic cache | C6 | tokens/$ on recurrent patterns | W2 |
| 21 | ACC conflict monitoring | Anomaly detector feeding trust + stress | RES-4 / RES-9 | — already mapped | — |
| 22 | Primary vs associative areas | Grafted zones + matryoshka association space | C5 / NOW-3 | graft fidelity across resolutions | W2/W4 |
| 23 | Primacy effect, attention sinks | Serial-position curve benchmark | eval suite | the curve itself | W1 (N2) |
| 24 | H.M. dissociation | Standing falsifiable diagnostic: cut journal → skills intact, episodes lost | eval suite | the dissociation holds, or the memory story is wrong | from N2 |
| 25 | Cerebellum forward model | Speculative drafter | NOW-2 | quality per dollar | W2 |
| 26 | Motor / pons / medulla / spinal | — deliberately not transposed | QM / runtime | — | — |

Waves: **W1** = N2 (C2 core family + the two benchmarks). **W1b** = N3 (C3 family). **W2** = N4–N5 window (C2b pipeline & format law, habit cache, drafter, graft fidelity). **W3** = first post-N5 wave (bus, homeostat, heterogeneous experts, insula, trust gating). **W4** = research wave (connectome, TEM, reverse replay). Parked items stay parked.

## Decision 2 — Consistency rulings (one owner per joint)

1. **Surprise has one producer and one consumer:** CA1 *computes* it (distance between reconstruction and input at the memory interface); RES-2 *consumes* it for writes; RES-9 receives it as a salience input. No second surprise heuristic may exist.
2. **The thalamic gateway is C3's front door, not a second mechanism** — it formalizes the existing injection point with an admission policy.
3. **One cache:** the habit cache is a policy *extension* of the C6 semantic cache, never a parallel store.
4. **One salience score**, written on journal entries, consumed by retrieval, consolidation priority, and reverse replay alike.
5. **The bus owns modulation:** C3 emits *events*; RES-9's bus emits *modulation*; gates listen to the bus only — no component reads oracle events to self-modulate directly.
6. **The connectome constrains everyone:** RES-10's wiring graph binds internal zones (C1) and grafted zones (C5) under the same budget.

## Decision 3 — Performance guards (non-negotiable)

1. The **advancement rule** (hub decision 019) applies to every row: ≥ 1 declared-capability win with ≤ 2% perplexity degradation, tolerance stated before the run. Biological beauty wins nothing.
2. **Every brain-derived feature ships behind a config flag, default off** (the SPEC-4 flag principle, generalized): controls stay pure; features earn their default by ledger, not by validation.
3. **Portability holds:** every mechanism respects ADR-002 (pure-framework ops, Triton-and-budgeted kernels) and NOW-4 (runtime matrix) — a feature that cannot ship on the runtime matrix is not done.
4. Design laws without runtime cost (row 8) apply at first implementation without ablation; everything else waits its wave.

## Decision 4 — The journal retention law (amendment, 2026-08-08, same day)

Raised by the founder: journal growth threatens read latency, disk space, and resource pressure. Ruling — the design already bounds all three; this decision makes the bounds law.

1. **Size law (extends row 8):** an entry is *(cue embedding, content-addressed pointer, salience, schema id, timestamps)* — order **≤ 1 KB** (matryoshka-truncated cues can shrink cold entries further). 1M episodes ≈ ~0.3–1 GB class. Content is never duplicated into the journal.
2. **Read law — never a scan:** retrieval = (a) the **fixed-size** associative state (constant memory regardless of history — "a memory that never grows") + (b) **ANN search** (HNSW/IVF-class) over cue embeddings: sub-linear, millisecond-class at 10^6 entries. O(N) reads are prohibited by design.
3. **Write admission:** CA1's computed surprise gates writes (row 7) — unsurprising events are not journaled; DG separation (row 6) decorrelates near-duplicates. The brain does not record everything; neither do we.
4. **Lifecycle = the garbage collector with a purpose:** write-gate → separate → **live (hot)** → **consolidate** (RES-9 replay into persistent associative memory / weights) → **demote** (K entries summarized into 1 summary entry keeping pointers) → **evict** (below a salience floor, already consolidated, aged). Reverse replay (RES-11) keeps salience honest over time — forgetting is a feature, scheduled, never accidental.
5. **Budgets are homeostatic setpoints (RES-9):** per-scope caps on bytes and on read p95 latency are regulated variables — approaching a cap raises consolidation and eviction pressure automatically. Scoping itself (per-project journals, the QM `projectScopeKey` precedent) partitions N.
6. **Tiering (RES-6):** hot index in RAM, warm entries on NVMe, cold summaries compressed — memory is a dial for the journal too; disk is the cheap tier and storage bandwidth is the growing resource.
7. **Safety guard:** the H.M. diagnostic doubles as the **eviction-safety check** — consolidation-then-eviction must never silently degrade skills or contracted recall; journal size, hit rate and read p95 are emitted in C6 telemetry and reported in run records' `standard_suite`.
8. **The noise ruling (amendment, 2026-08-08, same day).** Surprise alone is *not* signal — a corrupted input is maximally surprising, and a pure surprise gate would journal garbage enthusiastically. Definition adopted (MDL view, credited): **noise is the incompressible** — what never admits a shorter description and never improves prediction. Therefore noise identification is a **two-stage process, never a single detector**: (a) *admission* filters the redundant and already-known (CA1 surprise + DG separation — predictive coding, credited); (b) the *verdict* is retrospective — entries that pass the gate but never consolidate, never gain salience, and are never credited by an outcome (RES-11) are thereby identified as noise and evicted. Scheduled forgetting is the tribunal of noise. Scope clause: this law governs **storage** only — the system deliberately *uses* controlled randomness elsewhere (C3's oracle perturbation is exogenous signal, not noise; sampling temperature is a tool), and in training, the loss floor on deliberately unlearnable synthetic noise is an *instrument*: descending below it means the model is memorizing noise — overfitting, detected.

## Consequences

The roadmap gains one sentence pointing here; the two concept documents carry a validation banner (original statuses kept, per the errors-noted culture); the register notes the promotion in rev10; decision D14 is recorded in the hub. N1 remains untouched: the control stays vanilla, flags off — the map only tells the future in what order to arrive.
