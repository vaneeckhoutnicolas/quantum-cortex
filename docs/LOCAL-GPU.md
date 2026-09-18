# Running the cortex on a local machine (2026-09-14)

Two things run well on a laptop: everything that is protocol and reading (CPU, minutes), and, on a 6 GB NVIDIA card, the step 4 fine tunes at a third to a half of a T4's speed (fp16 with the gradient scaler; the trainer picks it when bf16 is not supported). The configuration hash does not depend on the hardware; the ledger records the provider.

## One time setup (Windows, PowerShell)

```powershell
cd C:\git\quantum-cortex
python --version                               # 3.12
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build; for CPU only: --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
pytest tests/ -m "not slow"                    # about a minute
```

The parent checkpoint and the data slice come from the N1 notebook's Output once (`kaggle kernels output nicolasvaneeckhout/notebook6a34480784 -p C:\kaggle-out\n1`), then:

```powershell
Copy-Item C:\kaggle-out\n1\quantum-cortex\runs\kaggle_t4\ckpt.pt runs\kaggle_t4\ckpt.pt
Copy-Item C:\kaggle-out\n1\quantum-cortex\data\fineweb_edu_bytes.bin data\fineweb_edu_bytes.bin
$env:QUANTUM_CORTEX_JOURNAL_KEY = "<the 32 byte hex key, the same as the Kaggle secret>"
```

## A fine tune (a seed of v7)

```powershell
python train.py --config configs\local_journal_v7_seed2.json 2>&1 | Tee-Object -FilePath metrics\mqar\logs\journal-n1-v7-seed2.log
python -m cortex_c2b.hm_lm --session-b --ckpt runs\journal-n1-v7-seed2\ckpt.pt --journal runs\journal-n1-v7-seed2\journal.jsonl --config configs\local_journal_v7_seed2.json --probe-training-pool
```

The record line printed at the end goes into `metrics/runs.jsonl` (the trainer appends it), `metrics/LATEST.md` is regenerated, and the two `hm-lm-<run_id>*.json` files land under `metrics/mqar/`: the same artefacts as a Kaggle run. If the card runs out of memory, halve `batch_size` in a copy of the config: that changes the hash, so it is another run, declared as such.

## The decode policies and the probes (CPU)

```powershell
python -m cortex_c2b.hm_lm --session-b --policy head --ckpt runs\journal-n1-v7\ckpt.pt --journal runs\journal-n1-v7\journal.jsonl --config configs\kaggle_t4_journal_v7.json
python -m cortex_c2b.hm_lm --session-b --policy head+pointer --ckpt runs\journal-n1-v7\ckpt.pt --journal runs\journal-n1-v7\journal.jsonl --config configs\kaggle_t4_journal_v7.json
```

Each writes `metrics/mqar/hm-lm-<run_id>-session-b-<policy>.json`; retain the console as a log file next to the others.

## v9 on a second and a third seed (2026-09-18, ADR-008 amendment on the organ side policies)

The seeds that did not conceive the policies: `configs/local_journal_v9_seed2.json` (seed 2024) and `configs/local_journal_v9_seed3.json` (seed 7), v9's recipe with the seed and the provider changed. On the GTX 1660 Ti (fp16, about 9 hours; `--resume` after an interruption picks the run up at its last checkpoint):

```powershell
cd C:\git\quantum-cortex; .\.venv\Scripts\Activate.ps1
$env:QUANTUM_CORTEX_JOURNAL_KEY = "<the 64 hex characters>"
python -u train.py --config configs\local_journal_v9_seed2.json 2>&1 | Tee-Object -FilePath metrics\mqar\logs\journal-n1-v9-seed2.log
```

Then session B in a new window, once per policy (the plain one first; each writes its own file, `-mark-veto` and `-mark-veto-value` suffixed):

```powershell
python -m cortex_c2b.hm_lm --session-b --ckpt runs\journal-n1-v9-seed2\ckpt.pt --journal runs\journal-n1-v9-seed2\journal.jsonl --config configs\local_journal_v9_seed2.json
python -m cortex_c2b.hm_lm --session-b --policy mark-veto --ckpt runs\journal-n1-v9-seed2\ckpt.pt --journal runs\journal-n1-v9-seed2\journal.jsonl --config configs\local_journal_v9_seed2.json
python -m cortex_c2b.hm_lm --session-b --policy mark-veto+value --ckpt runs\journal-n1-v9-seed2\ckpt.pt --journal runs\journal-n1-v9-seed2\journal.jsonl --config configs\local_journal_v9_seed2.json
```

On Kaggle the same seeds run through `notebooks/journal_finetune_kaggle.ipynb` with `CONFIG = 'configs/kaggle_t4_journal_v9_seed2.json'` (or `_seed3`), about 2 h 15 of T4, and the three policies through `notebooks/session_b_reread_kaggle.ipynb` with `V = 'v9-seed2'`. The readings are written in ADR-008 before any of this runs.
