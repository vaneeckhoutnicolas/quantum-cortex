# ADR 004 — The evaluation law: a Pareto frontier, never a scalar (D16)

- **Status:** Accepted (Nicolas Van Eeckhout, 2026-09-06)
- **Deciders:** Nicolas Van Eeckhout
- **Context:** the N1 baseline is committed (`be1fa8139f59`, val_perplexity 2.601). The frontier now has a lower-left corner to reference — so the rule for *reading* results can be fixed before the first capability ablation (N2).
- **Cites / promotes:** hub decision 019 (advancement rule), ADR-003 (waves & guards), D15 (continuity triad). Registered as decision **D16** in the hub decision log.

## Context

Founder's methodological point (2026-09-06): optimisations are not linear, and cannot be. Three independent non-linearities were identified and must be respected by the way we score anything:
1. **Features are not additive.** Two organs together rarely equal the sum of their isolated effects — synergy or interference is the norm.
2. **The addressed composition varies.** The result depends on *which* tokens are mobilised in context; the same journey composed differently yields a different state and a different cost. (This is exactly why D15's continuity is a triad, never a blended score.)
3. **Quality and efficiency are different axes.** A change can cost quality while multiplying efficiency (RES-7 ternary) — "worse" on one axis, "far better" on another.

A single scalar hides all three. This ADR makes the multi-axis reading the law.

## Decision 1 — The declared axes

Every configuration is scored on a fixed, frozen set of axes. No axis is silently added or dropped; changing the set requires a dated amendment here.

- **Capability axes (the continuity triad, D15):** episodic persistence (`hm_recall_on`), oracle-shift revision (`oracle_shift_recovery`), non-regeneration of the known (RES-8 reference ratio). Plus the declared per-wave capability benchmarks (e.g. MQAR at N2).
- **Efficiency axes (thesis D3):** capability **per parameter** and capability **per bit**. Latency / tokens-per-joule / tokens-per-dollar are recorded alongside, from C6 telemetry, as secondary efficiency signals.
- **The perplexity guard is a guard, not an axis.** ≤ 2% ppl degradation (hub-019) is a floor a candidate must clear to be *considered*; it never by itself makes a candidate "better".

## Decision 2 — Pareto, not ranking

- A candidate **advances** if it extends the frontier: strictly better on ≥ 1 declared axis without regressing another beyond the ppl guard. "Progress" = **moving the frontier**, never climbing a single number.
- The ledger is the frontier's storage: each committed record is a point; the N1 control is the current lower-left corner (organ-free — a reference, not a capability result).
- **No composite score is ever computed or published.** Where a summary is unavoidable (a headline), it names the axis it refers to.

## Decision 3 — Three corollaries (all testable, all recorded)

1. **Two measurement regimes.** Organs are ablated **in isolation to understand**; the validated combination is **retrained as one to decide** (the post-N5 re-training already in SEQUENCE). Non-additivity is expected, not a surprise.
2. **Interaction is declared data.** When two features ship together, the record carries the **measured joint effect** next to the sum of the isolated effects — synergy and interference become ledger entries, not anecdotes.
3. **Composition provenance.** Because the result depends on the addressed composition, each record notes the composition that produced it — the *what* of the context, not only the *how much*.

## Decision 4 — Honest credit

Multi-objective optimisation and Pareto frontiers are an established field (credited); what is ours is applying them to the **continuity triad under journaled composition provenance**. This ADR claims a discipline, not an invention.

## Consequences

SEQUENCE gains a standing "How we read results" section pointing here — the first thing a contributor reads before interpreting any number. The record schema already carries per-axis fields (`results.benchmarks.*`) and null-by-default honesty, so no schema change is required today; a `composition` note and a joint-vs-isolated field are added to the record when N2 produces the first comparison (`comparison` is already a top-level schema slot). N1 is untouched. The frontier has its origin; from N2, every run either moves it or is published as a loss that didn't.

## Linked decision — D17 (open-results), the publication counterpart of D16

D16 fixes *what we measure* (a Pareto frontier over declared axes); **D17 fixes what we publish**, and the two are deliberately coupled: **the frontier is the public object; the recipe is the private workshop.** Decision D17 (revises D6), accepted by Nicolas Van Eeckhout 2026-09-06:

- **Public** (the object of D16): the ledger (`metrics/`, dated verifiable numbers), the benchmarks, the `run-v1` schema, and a README stating the thesis and the signature capability. Results are made to be shown.
- **Private until an explicit founder decision**: the innovation register, the *design* ADRs (the architecture), the concept dives, the extractions, and the core code (`train.py`, C1–C6 once written). The recipe is not published.
- **Coherence note (honest):** this narrows D6's open-source adoption flywheel — an assumed IP-protection trade-off for a solo founder. It is reversible: the founder may open more at any time, typically once a module is mature and dated.
- **EuroHPC compatibility (verified 2026-09-06):** eligibility is geographic/institutional (Win2Win SRL, EU industry — eligible), never conditioned on open code. Open Science asks for open *results*, not open *code*. D17 is compatible.
- **Mechanism, deferred by design:** the *principle* is graved now; the concrete split (single private repo with results published out-of-repo, vs. a public "shopfront" repo carrying README + ledger only) is decided at **N5**, when there are real ablation results to show. Until then the repository stays private — no premature switch.

D16 and D17 are twin decisions: *we measure on a frontier, therefore that frontier — and only that — is what we show.*

### Amendment to D17 (2026-09-11) — the public switch is gated on the first model-level continuity measurement

D17 deferred the public/private mechanism to "N5, when there are real results to show". Two candidate triggers were considered since and both are rejected as premature: **a gated architecture result** (e.g. hybrids beating the best pure recurrent on MQAR at 8 seeds — true if it passes, but beside the repository's title, which promises *continuity*), and **a 100M+ run** (scale, not capability). The title and the proof must coincide before the repository is shown. Therefore:

**The public switch happens if and only if the H.M. dissociation passes on the language model** — the LM arm: the journal (C2b) read by the model in generation through router path 4, a learned cue encoder, real skill suites, thresholds frozen (δ 0.50, ε_S 1%, floor chance + 5 pts, negative control read above chance), `hm_dissociation_pass = 1` — with the reserves stated in the same breath (25M parameters; scale declared as a limit; C3 not built; one triad component measured). The organ-level pass of 2026-09-08 does not qualify: it proves the journal, not the model.

This is a harder test than the organ-level one: the model may read the right episode and still answer wrongly; δ ≥ 0.50 is not assumed. A FAIL is a result ("the journal stores, the model cannot yet use it"), points to the next work, and keeps the repository private. The gate applied to the switch itself.

Consequence for the sequence: the 8-seed run still runs and its result enters the documents either way, but it no longer triggers the switch. The path to the switch is: C2b Slice E (lifecycle) → the journal wired into `train.py` (path 4 in generation, learned cue encoder) → the LM arm of H.M. on Kaggle (resume the N1 checkpoint, brief fine-tune with the zero-initialised journal layer, run the protocol) → if PASS: public, `checkpoint-06`, announcement. EuroHPC (100M+) proceeds in parallel and is not a condition of the switch; a v1 "at 25M, continuity measured on the model, scale declared as a limit" is a defensible paper. Estimated Kaggle cost of the LM arm: ~5 h.
