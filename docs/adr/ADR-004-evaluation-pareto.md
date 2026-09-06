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
