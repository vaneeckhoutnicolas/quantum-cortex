# The external model arm: the Molaison dissociation on an open model through a frozen prompt

*Runbook. The declaration, the readings and the pinned models are in `docs/adr/ADR-008-journal-in-the-decode-loop.md` (amendment 2026-09-16, addendum 2026-09-17). The frozen protocol is `hm-protocol.md`. Nothing here is a claim; every number comes from a file under `metrics/mqar/`.*

## What it measures, in ten lines

The Molaison dissociation (the H.M. protocol: named after Henry Molaison, the patient whose hippocampus was removed in 1953 and who formed no new episodic memory from that day while keeping every skill he had) asks a model, with its journal cut, whether the episodes vanish while the skills stay. The model arm of the protocol was measured on the 27M cortex trained with the organ (RESULTS rows 21 to 29). This arm asks the same question of a model that was never trained with the organ: an open weight causal language model people use, at three sizes, through a frozen prompt.

The organ is unchanged: session A plants 200 facts through the gated write path into a journal sealed on disk; session B reopens it from the disk alone, in a new process, retrieves a window of k = 4 lines (256 bytes) for each question with the organ's own hash seeded cue (the entity as the address), and hands the window and the question to the model inside a frozen frame. The model answers on one line, `UNKNOWN` or `CITE <label> ANS <attribute>`; the line is mapped onto the protocol's contract tokens and scored with the cortex arm's counting rules. Four conditions, the thresholds of `hm-protocol.md`, the same verdict: strict recall on against off, the negative control (claims on never planted entities), invalid citations, and arm S (the perplexity of ordinary text with the organ's window in the context against the text alone).

What it can say: whether the organ works, unchanged, around a model people use, and how each size fails (door 2 of whitepaper §5d). What it cannot say: anything about size alone, because it changes two things at once against the cortex's arm, the size and the way organ use is taught (a frozen prompt instead of a fine tune). The readings for every outcome were declared before the first run; they are in the amendment.

## Prerequisites

- The repository at the commit that carries the frozen frame `docs/benchmarks/external-arm-prompt-7fda43aced410d65.txt`; a test binds that file to the code.
- Python with `torch` (the project's venv; on the founder's laptop, `C:\git\quantum-cortex\.venv`), plus `pip install transformers accelerate huggingface_hub` (run time only; the test suite never imports them).
- `QUANTUM_CORTEX_JOURNAL_KEY` in the environment of every window that touches the journal (32 bytes hex; the journal is sealed with it; session A and session B need the same key).
- The N1 data slice `data/fineweb_edu_bytes.bin` for arm S (provenance `data/mixes/fineweb-edu-n1.provenance.json`).
- bf16: the published dtype of the pinned models. On a CPU it is used as is; a GPU without bf16 (a T4, the founder's 1660 Ti) is refused by the loader, which never changes the dtype. Memory: about 4 GB for the 1.7B, 8 GB for the 4B, 16.4 GB for the 8B (a 24 GB card on Colab Pro).

## The commands, in order, one at a time (PowerShell, the laptop)

The output of each is pasted and read before the next. `<slug>` is `qwen-qwen3-1.7b`, `qwen-qwen3-4b` or `qwen-qwen3-8b`.

1. `python -m cortex_c2b.external_arm --resolve Qwen/Qwen3-1.7B`
   Expected: `Qwen/Qwen3-1.7B: main = <sha>  pinned = 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e  agrees` (or `DIFFERS` when the hub's `main` moved; the run loads the pin either way, and the difference is recorded).
2. `python -m cortex_c2b.external_arm --frame`
   Expected: `frame: …external-arm-prompt-7fda43aced410d65.txt  hash: 7fda43aced410d65  bytes: 1369`. Any other hash means the frame drifted: stop, nothing runs on a frame that is not the committed one.
3. `python -m cortex_c2b.external_arm --plant --journal runs\external-<slug>`
   Session A, no model. Expected: a JSON with `"admitted": 200`, `"durable": true`, `"policy": "stop"`. The journal directory holds `journal.jsonl`, its payloads and `session-a.json`.
4. `python -u -m cortex_c2b.external_arm --probe --journal runs\external-<slug> --model Qwen/Qwen3-1.7B --device cpu --out metrics\mqar 2>&1 | Tee-Object -FilePath metrics\mqar\logs\external-arm-<slug>.log`
   Session B, a new process. The first run downloads the model (about 4 GB, cached by the hub library). The log carries dated progress lines (`arm E on: 0/200`, …, `arm S: 0/256 spans`), then `[record] retained: metrics\mqar\hm-lm-external-<slug>-7fda43aced410d65.json` and the readout. PowerShell writes the log in UTF-16; read it with that encoding.
5. The same command with `--repeat` and the log name `…-repeat.log`: the second execution, another process, expected to agree number for number.
6. `python -m cortex_c2b.external_arm --agree metrics\mqar\hm-lm-external-<slug>-7fda43aced410d65.json metrics\mqar\hm-lm-external-<slug>-7fda43aced410d65-repeat.json`
   Expected: `agree: number for number`. A `DISAGREE` names the fields; both files are kept and the disagreement is a reserve on the row, never an average.

On Colab (the 4B, the 8B): `notebooks/external_arm_colab.ipynb` runs the same six steps in nine cells, with the secrets `GITHUB_TOKEN` and `QUANTUM_CORTEX_JOURNAL_KEY` and the slice copied once to Google Drive.

## What comes out, and where it goes

- `metrics/mqar/hm-lm-external-<slug>-<prompt hash>.json` and `…-repeat.json`: the record. Verdict, the four conditions, arm S with its frame diagnostic (`hm_skill_delta_frame`), the model (name, revision, dtype, template, tokenizer, hardware, library versions), the prompt hash, the generator hash, the per entity answers of the three arms, and `persistent` / `claimable`.
- `metrics/mqar/logs/external-arm-<slug>.log` and `…-repeat.log`: the console logs, retained as files.
- No line in `metrics/runs.jsonl`: the ledger holds training runs and this arm trains nothing. The artefact and its RESULTS row are the record.

Reading a run: the log as a file first (the progress lines, the `[record]` line), then the JSON (the verdict against the readings declared in the amendment, the per entity answers for a line by line check of any surprising number), then the agreement of the repeat, then one RESULTS row per model with its status and its reserve.

## Durations

Measured, not estimated, once the first run is graved (this section is filled from the retained logs).
