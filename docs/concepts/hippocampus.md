# The Hippocampus — deep dive — 2026-08-08

**Validation 2026-08-08:** every candidate in this document was validated by Nicolas and promoted into **ADR-003** (the canonical brain-feature map, wave-ordered, performance-guarded). Statuses below are kept as originally written — states are noted, never erased.

**Why its own document.** The atlas gave the hippocampus three scattered lines (CA3, dentate gyrus, replay — all hippocampal subfields). For a memory-centric architecture, that under-serves the single most relevant organ in the brain. This document transposes the full circuit. Statuses as in the atlas: [mapped] / [candidate] / [not transposed]; promotion only at an ablation.

## 1. The trisynaptic circuit — a memory *pipeline*, not a memory *box*

Entorhinal cortex (EC) → dentate gyrus (DG) → CA3 → CA1 → back to EC/cortex. Each stage has a distinct computational job:

- **EC — the gateway.** The interface through which cortex and hippocampus exchange; carries both raw content and a structural coordinate code (see §3). [mapped → the C2b read/write interface to the journal.]
- **DG — pattern separation.** Sparse expansion coding: project into a much larger, sparser space so that similar inputs become distinct before storage — decorrelation against interference. [candidate → C2b write path stage 1: expand-then-sparsify near-duplicates before journal writes. Already recorded in the atlas; formalized here as a pipeline stage.]
- **CA3 — pattern completion.** Dense recurrent collaterals: an autoassociative attractor network that retrieves the whole from a fragment — the textbook biological Hopfield network. [mapped → C2/RES-1: retrieval by energy descent. The anatomy endorses the architecture.]
- **CA1 — the comparator.** Receives CA3's *reconstruction* and EC's *direct* input, and signals their mismatch: a **novelty/surprise detector by architecture**. [candidate → the surprise signal of RES-2 stops being a heuristic and becomes a computed quantity: surprise = distance(reconstruction, input) at the memory interface. Feeds salience tags (RES-9) and write gating.]

**Pipeline claim (testable):** the C2b write path = separate (DG) → associate (CA3) → compare (CA1), each stage independently ablatable: interference rate without separation; recall without completion; write-precision without the comparator.

## 2. Index theory — the journal stores pointers, not blobs

Teyler & DiScenna's index theory: the hippocampus does not store experiences; it stores **indexes** — pointers binding the distributed cortical patterns that were active — and reactivating an index reactivates the pattern. [mapped → this is the biological blessing of **RES-8's content-addressed span references** and the design law of C2b: a journal entry is *(cue embedding, content-addressed pointer, salience, schema id)* — never a content blob. Storage economics and privacy (I13) both improve: the journal indexes; contexts and outputs hold content.]

## 3. Cognitive maps — grid cells, place cells, and the structure/content split

- **Entorhinal grid cells** (Nobel 2014): a periodic, low-dimensional **coordinate scaffold reused across environments**; **place cells** bind locations in that scaffold to specific content. The system maps *abstract* spaces the same way it maps rooms.
- The rigorous machine-learning incarnation exists and is credited: the **Tolman-Eichenbaum Machine** — factorize a reusable *structural* code from *content*, bind them at the hippocampal stage, and structure transfers zero-shot to new content.
- [candidate → RES-3 gains its scaffold: the memory graph carries a **shared low-dimensional coordinate code** (the "grid") onto which episodic content is bound — structure/content factorization as an explicit C2b design axis, tested on zero-shot transfer of relational structure to unseen entities.]

## 4. The journey, literally — replay in three directions

Place-cell sequences *are* trajectories; sharp-wave ripples re-traverse them:

- **Forward replay** — consolidation of journeys into cortex during rest. [mapped → RES-9 consolidation phases; journal trajectories are the replayed object — the founder's day-one "journey" word lands on its biological referent.]
- **Reverse replay** — after an outcome, the journey is replayed *backwards*, propagating value to the steps that led there: **credit assignment along the trajectory**. [candidate → **RES-11's headline mechanism**: when a ledger-scored or user-scored outcome closes an episode, walk the journal trajectory in reverse, updating the salience and trust contributions of each step — memory-level credit assignment without backprop-through-time.]
- **Preplay** — trajectories sketched *before* acting: planning in map space. [candidate → plan sketching in memory space prior to decoding; parked behind RES-6/RES-9 maturity.]
- **Theta phase compression** — within one oscillation, sequence order is compressed into spike phases. [noted → order-preserving compression of journeys; a note, not an entry.]

## 5. Two memory stores — and a falsifiable prediction

Patient H.M.: hippocampus removed → no *new* episodic memories, **skills intact**. The two-store model (episodic: hippocampus-dependent; semantic/procedural: cortical after consolidation) maps exactly onto our split — **weights = semantic and procedural; journal = episodic; RES-9 rhythms = the transfer between them**. This yields an honest, falsifiable architecture prediction, cheap to run: **the H.M. ablation** — cut the journal at inference and the model must retain skills while losing episode recall; if it doesn't, our memory story is wrong. [candidate → a standing diagnostic in the eval suite from N2 onward.]

## 6. Adult neurogenesis (DG)

New neurons appear precisely where separation happens — capacity grows at the interference frontier. [noted → journal index capacity policy: grow the separation space with corpus growth; an engineering note.]

## Summary

| Mechanism | Status | Lands on |
|---|---|---|
| EC gateway | mapped | C2b interface |
| DG separation | candidate (pipeline stage 1) | C2b writes |
| CA3 completion | mapped | C2 / RES-1 |
| CA1 comparator | candidate | RES-2 surprise, RES-9 salience |
| Index theory | mapped | RES-8 references, C2b entry format |
| Grid scaffold / TEM factorization | candidate | RES-3 / C2b structure-content split |
| Forward replay | mapped | RES-9 consolidation |
| **Reverse replay credit assignment** | **candidate — RES-11 headline** | journal salience/trust updates |
| Preplay planning | candidate (parked) | RES-6/RES-9 |
| Theta compression, neurogenesis | noted | engineering notes |
| H.M. ablation | candidate — standing diagnostic | eval suite (N2+) |

Promotion rule unchanged: ledger runs decide, anatomy only inspires.
