# The Molaison Protocol (H.M.) — episodic/semantic dissociation diagnostic — spec v1 — 2026-08-08

**The name (added 2026-09-13; the specification below is unchanged).** H.M. is Henry Molaison (1926 to 2008), the patient whose hippocampus was removed in 1953 to treat his epilepsy. From that day he formed no new episodic memory, yet he kept every skill he had and still learned new motor tasks without remembering the lessons: episodes gone, know how intact. The protocol asks the model the same question: with its journal cut, do the episodes vanish while the skills stay? Arm E is the episodes that must collapse without the journal; arm S is the skills that must survive. The code identifiers (`hm_*` fields, `hm-lm-<run>.json`, `hm_lm.py`) keep the short name; they are in the graved records.

**Status:** specification (written before any implementation, per measure-first). Implementation lands with **W2** (C2b); the spec is an open deliverable (NOW-6). First document of `docs/benchmarks/`.

**Claim under test (refutable):** in quantum-cortex, skills and semantic knowledge live in the **weights**; episodes live in the **journal** (C2b). Therefore cutting the journal at inference must leave skills intact while episodic recall collapses. If the dissociation does not hold, the memory story of hub decision 019 / ADR-003 is wrong — and the loss is published.

## 1. Preconditions

A W1 model (associative layer trained, N2 gate passed) and an implemented C2b (journal write/read path with the ADR-003 row-8 entry format). Hardware: inference-only — minutes on a T4 or CPU; no granted compute required.

## 2. The two arms

### Arm S — skills (must survive)
Run the declared capability benchmarks and the standard suite twice: journal **ON** and journal **OFF** (same model, same seeds, same prompts). Metric: `hm_skill_delta` = max relative degradation across suites. **Declared threshold: ε_S = 1% relative** (revisable only by dated amendment *before* a run).

### Arm E — episodes (must collapse without the journal)
- **Generator (open, seeded, config-hashed):** synthetic episodic facts planted in a session A — entities with generated, training-leakage-proof names (pattern `{consonant-vowel trigrams}-{4 hex}`, e.g. `Vorel-3f2a`), each bound to attributes through 5 schema-tagged templates (person/place/event/object/decision). Default size: **200 facts**, balanced across schemas; every fact written through the normal C2b path (surprise gate active — facts are novel by construction, so they pass).
- **Session B (fresh context, no session-A tokens):** recall queries per fact (direct query + one paraphrase). Scoring: exact-match primary, embedding-similarity ≥ 0.85 secondary.
- Metrics: `hm_recall_on` (journal ON) and `hm_recall_off` (journal OFF). **Declared thresholds:** the dissociation requires `hm_recall_on − hm_recall_off ≥ δ = 0.50` absolute, and `hm_recall_off` at the parametric floor (≤ chance + 5 points). No target is promised for `hm_recall_on` in isolation — its first measurement *defines* the baseline; the *gap* is the claim.

### Verdict
`hm_dissociation_pass = (hm_skill_delta ≤ ε_S) AND (hm_recall_on − hm_recall_off ≥ δ) AND (hm_recall_off ≤ floor)`. PASS or FAIL, both published.

## 3. Eviction-safety variant (ADR-003 Decision 4)

Rerun both arms **after** forced consolidation → demotion → eviction cycles at the byte-budget cap: `hm_skill_delta` must still hold ε_S, and schema-tagged *must-keep* entries (contracted recall) must survive eviction with recall ≥ their pre-eviction value − 2 points. This is the standing proof that scheduled forgetting never silently degrades skills or contracted memory.

## 4. Leakage & validity controls

Entity names are generated (never dictionary words, never real names); a **negative-control set** of 50 never-planted facts must yield near-zero "recall" in both modes (hallucinated recall ≥ 10% invalidates the run); the generator seed, template set and fact list ship with the run's `config_hash`; sessions A and B run in separate processes to preclude context leakage.

## 5. Ledger integration

Results land in the run record's `results.benchmarks.standard_suite` as: `hm_skill_delta`, `hm_recall_on`, `hm_recall_off`, `hm_negctrl_rate`, `hm_dissociation_pass` (0/1), plus `hm_evict_*` for the variant. From W2 onward the protocol runs at **every checkpoint** — it is a standing diagnostic, not a one-shot experiment.

## 6. Failure semantics (honest data)

A FAIL is a result, not an embarrassment: it is recorded, dated, and triggers a written revision of the memory story (dated ADR amendment). Thresholds ε_S, δ, floors are frozen here; changing any of them requires a dated amendment *before* the affected run — never after seeing numbers.

## Amendment 2026-09-08 — the negative control is read above chance (dated before the affected rerun)

**Trigger (recorded honestly):** the first Slice-D run of the protocol against the C2b organ returned `hm_dissociation_pass = 0` with verdict **INVALID** — not because the journal hallucinated (journal-ON negative control = **0.000**: for never-planted entities the cue index returns a neighbour with low similarity and the exact-match check rejects it — the journal *knows it does not know*), but because the journal-OFF arm, which simulates the parametric floor by guessing among the schema's attributes, "recalls" ~chance (12.5%, 8 attributes) by construction, and the spec's absolute 10% invalidation line sits *below* chance. The spec conflated **guessing** (chance-level, legitimate) with **hallucinating** (above chance, illegitimate).

**Amendment:** `hm_negctrl_rate` is defined per arm and read **above chance**: `hm_negctrl_rate = max(negctrl_on, max(0, negctrl_off − chance))`, with `negctrl_on`, `negctrl_off_raw` and `negctrl_off_above_chance` all reported. The 10% invalidation line is unchanged. A journal arm above 10% still invalidates (hallucination); a floor arm at chance does not (guessing).

**Also corrected in the reference simulator (not the spec):** a weights-only floor guesses *once* per fact — the paraphrase probe addresses the same entity and does not reroll the guess.

**Thresholds δ = 0.50, ε_S = 1%, floor = chance + 5 pts are unchanged.** This amendment is dated before the rerun whose verdict it affects; the invalid first run is kept in the artefact history.

## Amendment 2026-09-12 -- the verdict declares its persistence (dated before any run that uses it)

**Trigger (recorded honestly):** the artefact `hm-protocol-2026-09-08.json` was produced with sessions A and B in one process: the journal lived in memory, so the PASS proved survival across a *software* session boundary, not across a restart. ADR-007 Decision 8 defines the boundary as a full process stop and relaunch; Decision 9 adds a storage policy per scope, including a `memory` mode in which nothing is durable. Without this amendment the single process trap would come back through that door.

**Amendment:** the protocol reads the journal's storage state and reports three fields next to the verdict: `persistent` (true only when the journal is on disk, stayed in durable mode through both sessions, and session B **reopened it from the disk alone**, index recomputed), `claimable` (`hm_dissociation_pass` and `persistent`), and `storage` (on disk, mode, policy, sealed, plaintext scope). A pass on a journal in memory, or in `memory` mode, is reported with `persistent: false` and is **not claimable**; the verdict string says so. A measurement run opens its journal with the `stop` policy, so a storage failure aborts the run instead of continuing in memory unnoticed. Reference run: `python -m cortex_c2b.hm_protocol --persistent` (sealed on disk, session B reopened).

**Thresholds d = 0.50, e_S = 1%, floor = chance + 5 pts, the 10% invalidation line, are unchanged.** The 2026-09-08 artefact keeps its status *measured, within one process*; a claimable pass needs the disk run.

## Amendment 2026-09-12 (b) -- the language model arm under the cite or abstain contract (dated before any run on the model)

**Scope.** This amendment defines how the two arms are scored when the protocol runs on the MODEL (ADR-008) rather than on the organ. Thresholds d = 0.50, e_S = 1 %, floor = chance + 5 points, the 10 % invalidation line, are unchanged.

**The contract.** An episodic query is marked `<EPI>`; the answer is `<CITE>` + the local label of a retrieved episode + `<ANS>` + the attribute + newline, or `<UNKNOWN>` + newline. The read window presents up to k retrieved episodes with local labels in a seeded random order; the harness maps a label back to its pointer and verifies that the cited episode holds the claimed attribute.

**Arm E, journal ON.** `hm_recall_on` is STRICT: a valid citation (a label that exists, whose episode holds the attribute) and the exact attribute. Reported next to it: `hm_false_abstention_on` (abstentions on planted facts, the cost of prudence, already paid in the gap, no threshold of its own), `hm_invalid_citation_on` (a claim with a fake provenance: a label of nothing, a malformed claim, or a label whose episode does not hold the claimed attribute; **declared threshold: at most 1 %**), `hm_valid_citation_on`, `hm_attr_exact_given_valid`, and the attribution triple `hm_retrieval_hit` (the encoder), `hm_attention_mass` (the reader), `hm_attr_exact_given_valid` (the answer).

**Arm E, journal OFF.** The reader is absent (no window, gate zero). A citation is impossible, so `hm_recall_off` is zero by construction: the parametric floor in its cleanest form. To keep the floor meaningful, `hm_guess_rate_off` and `hm_attr_hit_off_by_chance` are reported.

**Negative control.** `hm_negctrl_rate` is the rate of CLAIMS (any non abstention) on never planted entities with the journal ON; at or above 10 % the run is INVALID. `hm_negctrl_abstain_on` and `hm_negctrl_abstain_off` are reported.

**Arm S.** `hm_skill_delta` is the relative perplexity degradation with the reader ACTIVE on whatever the journal returns for each ordinary span against the reader absent: retrieval noise must not hurt the model.

**Verdict.** `hm_dissociation_pass = (skill delta <= e_S) AND (recall_on - recall_off >= d) AND (recall_off <= floor) AND (invalid citation <= 1 %) AND (run valid)`. `persistent` and `claimable` as in amendment (a); the strict form runs session B in another process.

**Reference implementation.** `cortex_c2b/hm_lm.py`; the trainer runs it at the end of a journal run and writes `hm-lm.json` next to the checkpoint (its fields in `results.benchmarks.standard_suite`). No run on the model has been made under this amendment yet.
