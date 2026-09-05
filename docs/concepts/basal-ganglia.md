# The Basal Ganglia — selection, veto, habit — living groundwork

**Status banner.** Written 2026-09-05 ahead of its wave (founder decision). Groundwork, zero claims; evidence arrives with **W2 (habit cache) and W3 (trust gating)** as dated amendments.

## 1. Biology, the parts that compute

- **The loop.** Cortex → striatum → pallidum/SNr → thalamus → cortex: proposals from everywhere in cortex funnel through one selection machine and return gated.
- **Three pathways.** *Direct* = **go** (release the selected action); *indirect* = **no-go** (suppress competitors); *hyperdirect* = **stop** — the fastest route, a global brake that can cancel an action already underway.
- **Dopamine = reward prediction error.** Better-than-expected outcomes reinforce the selection that produced them; the teaching signal is an *error*, the same mathematical species as the delta rule (extraction K1).
- **Habits.** With repetition and reward, control shifts from goal-directed circuitry (dorsomedial striatum) to habitual circuitry (dorsolateral): the same behavior becomes cheap, fast, and model-free — until context change demands re-deliberation.
- **Chunking.** Action sequences compile into single selectable units.

## 2. Transposition

- **Selection = the router, trust-weighted [mapped → C1 / RES-4, W3].** Zone proposals compete; go/no-go gating carries learned trust; deliberately corrupted experts are the robustness test.
- **The hyperdirect veto [candidate → C4, new mechanism surfaced by this page].** A fast global brake wired to the contract layer: a detected invariant breach can **abort a span mid-decode** — not filter it afterwards, *stop it in flight* — the decode-loop twin of a Skald revert. Latency-budgeted like the collicular interrupt.
- **Dopamine [mapped → RES-9 reward channel].** The bus's reward line; its RPE nature links it to C2's delta-rule writes — one error-driven family across memory and selection, worth one line in the whitepaper's unification story (RES-1).
- **Habit cache [candidate → C6 extension, W2].** Recurrent (request-pattern, routing-decision) pairs promote into a cache after repetition-with-reward; a context-shift signal (oracle) **demotes** habits back to deliberation — the biological re-deliberation trigger, transposed. One cache, two policies (ADR-003 ruling 3 holds).
- **Chunking [noted].** Cached macro-decisions (multi-step routing plans) as single entries — follows the habit cache, never precedes it.

## 3. Tests (shapes frozen; thresholds at wave opening)

Trust gating: accuracy under k% corrupted experts vs vanilla gating · veto: abort latency, false-abort rate, contract-compliance lift · habit cache: tokens/$ on recurrent suites, **and** regression check under injected context shift (habits must demote, not persist) · chunk reuse rate.

## Summary

| Mechanism | Status | Lands on |
|---|---|---|
| Trust-weighted go/no-go | mapped | C1 / RES-4 (W3) |
| Hyperdirect veto (mid-decode abort) | candidate — new | C4 (W2/W3) |
| Dopamine RPE | mapped | RES-9 reward · kinship with K1 |
| Habit cache + shift-triggered demotion | candidate | C6 (W2) |
| Action chunking | noted | after habit cache |

Promotion rule unchanged: ledger runs decide.
