# Roadmap — quantum-cortex

**Status stamp: 2026-09-30.** Thin and scannable. Every step is gated by ledger data, not by calendar. This file says where the steps stand; `RESULTS.md` says what was measured and `FALSIFY.md` says how to attack it. Normative references: hub decision 019 (architecture and ablation protocol; mirrored in `docs/hub/`, the hub being private at this edition), ADR-001 (metrics discipline), ADR-003 (the transposition map and its waves), ADR-004 (how a result is read).

## Where the steps stand

| Step | Content | Gate | Status |
|---|---|---|---|
| **N1** | Training scaffold: one reproducible notebook, a plain 26M transformer control, pinned seeds, configuration hash logging, an open data slice, a schema validated ledger record | The scaffold runs end to end and emits a valid record | **passed** 2026-09-07. Control `be1fa8139f59`, validation perplexity 2.601 |
| **N2** | Ablation C2, the associative layer, against the control on an MQAR style benchmark | Win on at least one declared capability, perplexity delta at most 2 % | **passed** 2026-09-07 (both associative memories beat the control), then **extended far beyond it**: the episodic organ C2b, the Molaison protocol, the language model arm and the organ side decode policies, rows 13 to 39 |
| **N3** | Ablation C3, the oracle channel, oracle shift recovery | Same rule | **not built.** The ledger's `oracle_shift_recovery` slot is null in all twelve runs |
| **N4** | Ablation C4, typed decode contracts, compliance rate | Same rule | **partly.** The cite or abstain contract exists and is measured by the protocol's invalid citation rate on every model run; the declared `invariant_compliance_rate` slot is still null |
| **N5** | Ablation report published with its losses; then the applications for granted compute, citing ledger run identifiers | The report reproduces from the ledger alone | **now.** `docs/WHITEPAPER.md` is that report, the landing edition, rows 1 to 40: all eleven measurements of the model alone INVALID and kept, the system holding the four conditions on two live seeds, the organ's security measured in two registers (row 40); the tag follows the founder's reading |
| **N6** | C5, the grafted zone adapter | A validated core exists, at least one component past its gate | **not built.** Map line 22 exists in the source as a comment, not as a component |

Brain derived features enter only through the wave order of **ADR-003**, behind a configuration flag, default off, until their ledger record earns them. The map's own audit is in whitepaper §5c bis: of twenty eight lines, nine are measured, one is declared and unmeasured, one is named and undeclared, one is deliberately excluded, and sixteen have never been built.

## After the v1 tag, in the order the record names them

1. **No training required.** The index's bucketing, the variable row 38 leaves open (more tables, or the sparse tag as the bucket itself), a probe like RES-24's. Widening the negative control past fifty entities, which is what would size the veto margin reserve of rows 39 and 40 (0.005 of mark on one seed). The security doors declared at the tag (register, RES-25 and RES-26): the detectors as code with a two arm suite, membership inference by the mark, the manipulation of retention, the write side separation of near duplicates with its threshold on the marks; the sequence integrity of the journal (a chained counter in every line's associated datum). The reconstruction test on the organ's layer of numbers, whitepaper §5d item 7, once a layer with a first law is identified.
2. **Runs, with GPU.** Stage B of the recurrent base arms, about 4 h 30, the only measurement that can hold the intent of Rev38 by making the trunk take off instead of being pre empted by the attention. Then v8, the reader with an identity initialised local convolution, about 2 h 15, declared and built. Then a fourth seed of v9 if the first three ever divide.
3. **The external model arm.** Its agreement between two executions on one hardware is still pending; then the larger sizes. Door 2 of §5d.
4. **Named in the transposition map, not declared**, each needing a training run and each facing a residual this record measures: line 10, forward replay and consolidation phases, against the trunk that never takes off; line 13, the amygdala's salience together with the times of writing and of last reading, all three stored in every entry since the first line of C2b and never reaching the model; line 28, the chickadee's write side barcode, closed while the probe of row 38 shows no collision to separate.
5. **Steps still open above**: N3, N4's benchmark slot, N6.

## What does not move

The four thresholds of the Molaison protocol, frozen since 2026-09-12. One variable per run. Every reading declared before the run it judges. Every log retained as a file before any number in it is read. A claim whose artefact is missing is treated as absent.
