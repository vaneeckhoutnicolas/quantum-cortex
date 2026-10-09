# ADR-009 — Components and their licences, the source text of every address, English first, the EuroHPC dossier (the founder's decisions of 2026-10-05, graved 2026-10-08)

**Status:** accepted (2026-10-08). Decisions B1 to B4 are the founder's, taken on 2026-10-05 after an outside reading of the public copy at `41e309e` (the review's verdicts are in the register, Rev96, and in the workshop's `docs/working/REVUE-EXTERNE-2026-10-08.md`); the two principles of 2026-10-08 that frame them are his as well: **no licence risk at all** (anything carrying a sub licence, use conditions or a limitation is excluded or rewritten entirely) and **no added complexity** (doors are closed; the direction fixed is finished coherently rather than kept experimental). The facts that motivate each decision were checked against the code on 2026-10-08; what could not be checked from the workshop is marked as the founder's reading, to be reread at the moment it matters. Nothing here is a measurement.

## Decision B1 — No dependency that could become paid, and no licence risk at all

**Decision.** No component (a model, an encoder, an execution engine, a library on the critical path) whose licence could one day impose a payment when the project grows, or which could not be removed once integrated. Hardened on 2026-10-08: a component enters only under Apache 2.0, MIT or BSD **with no condition added**; anything that carries a sub licence, terms of use, a field or revenue limitation, a share alike clause on code, or a licence that must be read twice (Gemma terms on a tokenizer, the LFM Open License of Liquid AI, research licences, Llama) is excluded, and if the idea is useful it is rewritten entirely, as the rule of 2026-10-02 already does for research code (`CONTRIBUTING.md`; `docs/OSS-SURVEY-2026-08-07.md`, rule 7). In doubt, it does not enter.

**Rules that follow.**
1. Apache 2.0, MIT or BSD for every component, including any base that is fine tuned: a fine tuned model inherits its base's licence.
2. Weights pinned by revision and copied locally: a withdrawal from a hub breaks nothing (the external arm already pins its three checkpoints by commit, ADR-008 addendum of 2026-09-17).
3. A standard, open execution engine (`transformers`, `llama.cpp`), without proprietary extension.
4. The organ never depends on an encoder it cannot recompute (Decision B2).
5. If an outside reader is ever used, it is wired through the text contract only (cite a label and give the attribute, or abstain). Before any change of reader, two readers from different providers pass the protocol: the protocol is the exit test.
6. An outside model may always serve as an instrument of measurement without becoming a component; its licence is still read in full before it is declared, and a model whose licence is not clean is not measured either.
7. Every component and every instrument is listed in the table below with its licence as read; the table is part of this record and moves only by a dated line.

**Facts noted by the founder on 2026-10-05, to be reread at every entry (not checked from the workshop).** The LFM Open License v1.0 (Liquid AI) withdraws commercial use of the model and of any derivative by an entity whose yearly revenue reaches ten million dollars, with no grace period and automatic termination; the threshold targets the using entity, hence a customer too: Liquid is excluded. Compatible licences noted that day: BGE M3 (MIT), Qwen3 (Apache 2.0), Ministral 3 in 3, 8 and 14 billion parameters (Apache 2.0), EuroLLM 9B and 22B (Apache 2.0), Kolibri of Aleph Alpha (Apache 2.0, data centre GPUs). The only outside component retained, for later, is BGE M3 (Decision B3); the reader aimed at is the project's own model, trained on EuroHPC.

**The components table (2026-10-08).**

| component | role | licence as read | status |
|---|---|---|---|
| PyTorch | base stack | BSD 3 clause | component |
| numpy | base stack | BSD 3 clause | component |
| jsonschema | the ledger's schema validation | MIT | component |
| cryptography | the journal's cipher (AES-256-GCM) | Apache 2.0 or BSD, at the user's choice | component |
| pytest, build, twine | tests and packaging | MIT, MIT, Apache 2.0 | tooling |
| transformers, accelerate, huggingface_hub | the external arm's loader only | Apache 2.0 | tooling of an instrument |
| FineWeb-Edu (the N1 slice) | pretraining data | ODC-By 1.0, attribution in `NOTICE` | data |
| Qwen3 1.7B, 4B, 8B at the pinned commits | instruments of the external arm | Apache 2.0 | instruments |
| BGE M3 | the organ's semantic addressing, later (B3) | MIT (to be reread at entry) | planned, not entered |
| Vela 2.0 0.3B (vllm-sr) | none | Apache 2.0 weights with a tokenizer under the Gemma terms | excluded (B1) |
| Decision 2.0 models (vllm-sr), Qwen3.5 backbones | none | licence of the backbones not read in full | not entered |
| Liquid AI models (LFM Open License v1.0) | none | use limitation above a revenue threshold | excluded (B1) |
| Kolibri (Aleph Alpha) | none | Apache 2.0 announced; 156 GB in bf16, out of reach of the project's machines | not entered |
| Llama, Gemma, research licences | none | conditions added | excluded (B1) |

## Decision B2 — Keep the source text of every address, from now on

**Decision.** Every journal entry keeps the text its address was computed from, sealed like the content: same cipher, same key, same scope. The day a learned encoder is added (B3), the index is recomputed from the sealed texts without migrating any journal. The hash address stays in place: it depends on no language and remains valid for identifiers (the protocol's invented names; a case number such as `2024/AR/512`).

**The fact that motivates it, checked in the code on 2026-10-08.** `Entry` (`cortex_c2b/__init__.py`) holds `cue` (64 numbers), `pointer` (16 characters, the content hash), `salience`, `schema_id`, `t_written`, `t_last_read`, `state` and `entry_id`, not the text the address was computed from. With the hash encoder that is no problem: the function belongs to the project. With an outside encoder, a change of encoder would orphan every old address.

**Design, declared before the code.** The source text is a sealed payload of its own in the payload store, referenced by a second pointer in the entry (`source_pointer`, 16 characters, `None` when absent); it is never stored inside the entry, so that the entry stays far from the size law (`ENTRY_MAX_BYTES = 1024`; the protocol's entries measure 618 to 650 bytes on 2026-10-08, with one pointer). Existing journals carry no source text: they stay readable, the journal declares `source_text: absent` on opening for every entry without a pointer, and nothing is recoded in silence, as the write path already refuses a journal written under another separation. The rebuild on opening (ADR-007, Decision 8) stays byte for byte identical: the source payload is never replayed into the persistent memory. The lifecycle releases the source payload with the content at eviction (the declared loss covers the content and, since Decision 11, the address; the source text goes with the content), and the phase event says so.

**Tests declared before the code (the code follows the founder's validation of this list, as the recipe requires).**
1. *The round trip.* A write with a source text seals it; a read returns it decoded, equal to the byte; without the key it is unreadable, like the content (`tests/test_c2b_source_text.py`).
2. *The size law.* An entry with a source pointer stays under 1024 bytes at the protocol's sizes, and a source text of any length is accepted, since it lives in the payload store; a direct attempt to put a text inside the entry is refused by the size law as today.
3. *Persistence unchanged.* Every existing persistence test passes without a change; a journal written before this decision opens, declares `source_text: absent`, reads as before, and its rebuild on opening is byte for byte the one of today.
4. *The Molaison rows reproduced number for number.* The hash addresses do not change, so the organ's protocol files (rows 13, 13b) and the session B files of rows 36 to 42 recompute to the digit with the source text present; the record tests run unchanged.
5. *The lifecycle.* An eviction releases the source payload with the content; the phase event carries the count; a reopened scheduler agrees number for number (Decision 8 under Decision 11).

## Decision B3 — English first, multilinguality later

**Decision.** Research continues in English only (the protocol, the model, the data): one variable at a time. Multilinguality comes later, as new rows, with its own fact generator and its own configuration hash.

**BGE M3, later, by design.** BGE M3 (MIT; more than a hundred languages, Dutch and French among them; vectors of 1024 numbers; a lexical mode next to the dense one) is planned as hybrid addressing: the hash for identifiers, BGE M3 for meaning and language. It is not integrated now. Before it enters, four consequences are settled and declared: (1) the exact address kept beside it, since a semantic encoder is weak on arbitrary identifiers and alone would raise the claims on never planted entities; (2) the size law, since 1024 numbers do not fit a 1024 byte entry: a projection or a compression, or a change of law, declared; (3) the write gate, since under semantic addresses a revised decision looks redundant: the revision mechanism (the review's point 3, to be declared as an amendment of ADR-007 after B2 is built) comes first; (4) security, since the system level zero of row 40 holds with a margin of 0.005 of mark because foreign lines mark low: with a semantic encoder that measurement is redone and the veto threshold recalibrated, language by language. Limit kept in mind: the encoder settles the language for finding a note, not for reading it; the reader must understand the line, and the current model learned English only.

## Decision B4 — The EuroHPC dossier states the languages of the training data

**Decision.** The dossier names the languages of the training data explicitly, the one thing costly to change afterwards. The choice between the two options is the founder's: English only (simpler, in the continuity of the current model; other languages by a second pass; bytes need no vocabulary extension), or a share of French and Dutch from the start (useful for Belgium; at small size every added language takes capacity from English). The linked question, his as well, before the dossier: train from zero, or fine tune on EuroHPC an Apache 2.0 base such as EuroLLM (35 languages, itself trained on EuroHPC's MareNostrum 5). The facts stated: Apache 2.0 is irrevocable for the downloaded version and the fine tuned version belongs to the project, origin kept; fine tuning costs a small fraction of a full training; from zero keeps full control of the data and the account of a model built end to end in Europe; if EuroLLM is fine tuned, the language question disappears. **Status: decision pending**; the dossier is prepared with both options (the plan of decision D7, `docs/EUROHPC-PLAN-2026-08-07.md`), and the size study of ADR-008's amendment of 2026-10-08 is what sizes the request.

## Doors closed on 2026-10-08 (the founder's principle of no added complexity)

- Kolibri in the external arm: out of reach of the project's machines (156 GB in bf16, 78 GB in 8 bits); not declared.
- The Merlin Arthur protocol as a training recipe: not declared. Its "Morgana" negative control (a window without the right line, filled with the most similar lines) is not a new door: it is the hardest form of the wider negative control already named in v1.1 and enters that item as a declared variant when the item is declared.
- Long context as a comparator of the organ: not declared.
- Mistral Large 4 and any model reachable only as an online service: not a reference in the sense of rule 2 (same size), not reproducible; not declared.
- A second frame for the external arm (a choice format for decision models): not declared; the size study uses the frozen frame and the text contract.

## Consequences

`CONTRIBUTING.md` and rule 7 of the OSS-SURVEY protocol carry B1 as hardened. The code of B2 follows its five tests. B3 and B4 bind the EuroHPC dossier. The register's Rev96 holds the review's verdicts and the founder's answers; `docs/RESULTS.md` rows 13, 13b, 37 and 39 carry dated amendments; row 13c is declared in ADR-007 (amendment of 2026-10-08) and measured by the founder before it is graved. No prediction.
