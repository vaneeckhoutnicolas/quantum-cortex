# The workshop and the public copy

*Added 2026-09-30 (ADR-004, amendment to D17 of that day). Stated so that no reader takes this repository for the whole of the author's files, and so that the commit identifiers cited in the record stay resolvable.*

This public repository is a **filtered copy of the author's private workshop repository**: the same commits, the same authors, the same dates, the same tags, with a declared set of files removed from every commit of the copy. What is public is everything the whitepaper and the record cite as evidence: the code, the tests, the benchmarks with their frozen thresholds, the artefacts and the retained logs, the ledger, the results, the architecture decision records, the ideas register with its dated revisions, the figures, and the mirror of the two hub pages that govern this repository (`docs/hub/`).

What stays in the workshop, removed from the copy's history:

- `docs/working/`: the handover prompts between the author's AI sessions, the resume notes, and the legal and provenance audit made before the tag (its decisions are in `NOTICE`, `LICENSE-DOCS.md` and `DISCLAIMER.md`);
- `CLAUDE.md`: the working rules given to the AI collaborator;
- `docs/HANDOFF-2026-08-07-ecosystem-context.md` and `docs/EXECUTION-2026-08-07.md`: a context capsule and an execution list written for the collaborator.

These files describe *how* the author works with a language model, which is his to keep; the method itself, at the level a reader can reproduce, is stated in the whitepaper (§2, §7b). Nothing in them is evidence: no number of the record depends on them, and the test suite does not read them.

**Commit identifiers.** Filtering rewrites the identifiers of the commits it touches, so a commit identifier cited in a provenance file (`metrics/mqar/PROVENANCE-*.json`) refers to the workshop's history. The mapping from the workshop's identifiers to the copy's, produced by the filtering tool at the moment of the switch, is published here as `docs/COMMIT-MAP.txt` at the first public tag; a reader checks a cited commit by looking up its identifier in that file. The dates of every declaration are those of the workshop's commits, unchanged.

**What the copy is refreshed from.** The workshop remains the place where the work happens; the copy is refreshed from it at each tag, with the same filter, so that the two histories keep the same commits and dates from the first public tag onward.
