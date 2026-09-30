# How to attack this in twenty minutes

This page exists so that a reader can try to break the work quickly, without reading the whitepaper first. Everything below is a place where the record could be caught out, with the file that would catch it.

## 1. The thresholds, and when they were fixed

The Molaison protocol's four conditions are numbers, frozen before any model measurement and unchanged since 2026-09-12: the gap between the recall with the organ and without it at least 0.50; claims on never planted entities strictly under 10 %; invalid citations at most 1 %; the skill delta at most 0.01. They are in `docs/benchmarks/hm-protocol.md` and in `cortex_c2b/hm_protocol.py`, and the git history dates them. If a result here passes only because a threshold moved, the history says so.

## 2. The readings, written before the runs

Every run of §4 was declared in `docs/adr/ADR-008-journal-in-the-decode-loop.md` before it ran, with what each possible outcome would mean, including the outcomes that would sink the idea. Compare a declared reading with the one the row actually gives: `docs/RESULTS.md` rows 21 to 40. A row whose reading does not match its declaration is a finding against us.

## 3. The negative rows

The model alone is INVALID on every one of its eleven measurements, in the same table as the rest, with their attribution: rows 21 to 26, 28, 29, 36, 37 and 39 (row 27 is the Σ of the convergence, not a model measurement); what holds is the system of rows 37 and 39, the model deciding and the organ answering. A record that only shows what worked is not evidence; the fastest check of this one is to read the rows that did not.

## 4. Recompute an aggregate from the artefacts

Nothing here is a number typed into a document. Clone, install, and run the suite:

```
pytest tests/ -m "not slow"
```

Among those tests, `tests/test_recurrent_base_record.py` recomputes every published aggregate of rows 30 to 35 from the committed unit files and fails if a digit differs; `tests/test_c2b_familiarity_mark.py` recomputes the rates of row 36 from the per answer file; `tests/test_c2b_extraction_probe.py` recomputes row 40 from its three files and checks its positive arm against the session B files of rows 36, 37 and 39. `metrics/runs.jsonl` holds one line per training run with its identifier, configuration hash and git commit; `metrics/ARTEFACTS.md` says where each checkpoint and sealed journal lives.

## 5. What we already know is weak

`§6 Limitations & honest reserves` of the whitepaper, and the reserve sentence at the end of every row. The shortest list: most model results rest on one or two seeds; the never planted control rests on 50 entities; the post hoc system reading of row 36 was computed on the seed that suggested the policy and is labelled as not a result; the external model arm has one execution on one hardware and its agreement is pending; the recurrent arms beat their reference with a trunk that never learned the task, which is stated in the same sentence as the result; the organ's veto has a margin of 0.005 of mark on one seed (row 40) and the journal's sequence is not authenticated (whitepaper §4.9, a declared flaw).

## 6. What would falsify the central claim

The claim is that a memory organ outside the weights can hold episodes a model's skills do not depend on, shown by a test that can fail. It fails if: cutting the journal leaves the episodic recall intact (the dissociation is fake); or the skill delta moves when the organ is attached (the organ is not free); or session B, in a new process from the disk alone, does not reproduce the in process arm (the persistence is an artefact of the session). All three are measured in every run's two files, `metrics/mqar/hm-lm-<run_id>.json` and `...-session-b.json`.

If you find one of these, an issue on the repository with the file and the line is the most useful thing anyone can send.

## 7. The organ under attack, in twenty minutes

Whitepaper §4.9 states the organ's security in two registers, by design and by measure, and says which is which. Two measurements can be replayed. **Poisoning** is a rereading of rows 37 and 39: under the organ side policy every claim cites a line the organ holds and answers that line's own value, and nothing is claimed on an entity no line holds; replay it on the committed answers with `python -m cortex_c2b.hm_lm --replay metrics/mqar/hm-lm-2e1dc71913b1-session-b-answers.json --policy mark-veto+value --ckpt none --journal none --config configs/kaggle_t4_journal_v9_seed3.json` (a replay reads only the answers file; the three other arguments are required by the command line and not opened). **Extraction** is row 40: the questions of one domain against the sealed journal of another, zero at the organ level by construction and zero claims at the system level on three seeds; recompute it from the committed files with `pytest tests/test_c2b_extraction_probe.py`, and rerun it on a checkpoint and its sealed journal (where they live: `metrics/ARTEFACTS.md`; the key in `QUANTUM_CORTEX_JOURNAL_KEY`) with `python -m cortex_c2b.hm_lm --extraction-probe --domain-b-seed 1394548301 --ckpt <ckpt.pt> --journal <journal.jsonl> --config <the run's config>`; the seed is the one committed in ADR-008 before the run, and any other seed is refused. The fastest attack on the design itself is the one §4.9 declares for you: the journal authenticates every line and every payload but not their sequence, so delete a line of a sealed journal and reopen it; the cipher does not object, and only an event on an entry the replay has not seen does.
