# How to attack this in twenty minutes

This page exists so that a reader can try to break the work quickly, without reading the whitepaper first. Everything below is a place where the record could be caught out, with the file that would catch it.

## 1. The thresholds, and when they were fixed

The Molaison protocol's four conditions are numbers, frozen before any model measurement and unchanged since 2026-09-12: the gap between the recall with the organ and without it at least 0.50; claims on never planted entities strictly under 10 %; invalid citations at most 1 %; the skill delta at most 0.01. They are in `docs/benchmarks/hm-protocol.md` and in `cortex_c2b/hm_protocol.py`, and the git history dates them. If a result here passes only because a threshold moved, the history says so.

## 2. The readings, written before the runs

Every run of §4 was declared in `docs/adr/ADR-008-journal-in-the-decode-loop.md` before it ran, with what each possible outcome would mean, including the outcomes that would sink the idea. Compare a declared reading with the one the row actually gives: `docs/RESULTS.md` rows 21 to 36. A row whose reading does not match its declaration is a finding against us.

## 3. The negative rows

Six of the nine model measurements are INVALID and are in the same table as the rest, with their attribution: rows 21 to 29 and 36. A record that only shows what worked is not evidence; the fastest check of this one is to read the rows that did not.

## 4. Recompute an aggregate from the artefacts

Nothing here is a number typed into a document. Clone, install, and run the suite:

```
pytest tests/ -m "not slow"
```

Among those tests, `tests/test_recurrent_base_record.py` recomputes every published aggregate of rows 30 to 35 from the committed unit files and fails if a digit differs; `tests/test_c2b_familiarity_mark.py` recomputes the rates of row 36 from the per answer file. `metrics/runs.jsonl` holds one line per training run with its identifier, configuration hash and git commit; `metrics/ARTEFACTS.md` says where each checkpoint and sealed journal lives.

## 5. What we already know is weak

`§6 Limitations & honest reserves` of the whitepaper, and the reserve sentence at the end of every row. The shortest list: most model results rest on one or two seeds; the never planted control rests on 50 entities; the post hoc system reading of row 36 was computed on the seed that suggested the policy and is labelled as not a result; the external model arm has one execution on one hardware and its agreement is pending; the recurrent arms beat their reference with a trunk that never learned the task, which is stated in the same sentence as the result.

## 6. What would falsify the central claim

The claim is that a memory organ outside the weights can hold episodes a model's skills do not depend on, shown by a test that can fail. It fails if: cutting the journal leaves the episodic recall intact (the dissociation is fake); or the skill delta moves when the organ is attached (the organ is not free); or session B, in a new process from the disk alone, does not reproduce the in process arm (the persistence is an artefact of the session). All three are measured in every run's two files, `metrics/mqar/hm-lm-<run_id>.json` and `...-session-b.json`.

If you find one of these, an issue on the repository with the file and the line is the most useful thing anyone can send.
