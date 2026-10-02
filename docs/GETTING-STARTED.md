# Getting Started — run and verify everything locally

**Why this file.** Nothing in this project is trusted until it has run. Every command below is meant to be **executed and verified on your own machine** — the same commands CI runs — so that no step rests on an assumption. If a claim can be checked locally, check it locally.

## 1. Prerequisites

- **Python 3.10+** (3.12 is used in CI and on Kaggle). Check: `python --version`.
- **git**. That's it for the local loop — no GPU is needed to *verify* anything; a GPU is only needed to *pretrain* (ADR-002: CPU proves correctness, never speed).

## 2. Local environment (once)

`train.py` needs exactly three third-party packages — `numpy`, `torch` (CPU build is fine), and `jsonschema` (used by the ledger validator). A virtual environment keeps them isolated:

**Windows (PowerShell):**
```powershell
cd C:\git\quantum-cortex
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy torch jsonschema
```

**Linux / macOS:**
```bash
cd ~/git/quantum-cortex
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy torch jsonschema
```

To use the organ as a library in a project of your own, without this repository, `pip install quantum-cortex` installs the four packages (`cortex_c2b`, `cortex_c2`, `cortex_data`, `cortex_eval`) and nothing else: no tests, no configurations, no artefacts. One change of behaviour between editions to know about: since v1.1.0 the organ withdraws the address of an evicted episode from its persistent memory in the phase of the eviction, by default (ADR-007, Decision 11, from row 42); a journal written under 1.0.0 and reopened under 1.1.0 rebuilds its memory from the log without the patterns of its evicted entries, and the reopening scheduler's configuration hash names the rule it read the log under; `LifecycleConfig(withdraw_at_eviction=False)` reopens it under the old rule. Everything below assumes the clone.

`.venv/` is git-ignored — never commit it. (No `requirements.txt` is pinned yet by design: the dependency surface is deliberately tiny and stated here; a pinned lockfile arrives with W2/DVC when the data pipeline does.)

## 3. The three things you can verify locally, right now

### a. The CPU smoke — does the whole factory work?
Trains the real trainer end to end on a tiny synthetic slice (seconds, CPU-only), then self-checks that a valid record was produced:
```bash
python train.py --config configs/smoke_cpu.json
```
Expect the loss to fall (≈5.5 → ≈5.0) and a `run-v1` record to append. This is exactly what CI's `smoke-train` job runs on every push. Correctness only — CPU never pretrains.

### b. The ledger validator — is every record schema-clean?
Regenerate `metrics/LATEST.md` from the committed ledger. This also *parses and validates* every line of `metrics/runs.jsonl`, so it fails loudly if a record is malformed:
```bash
python train.py --regen-latest
```
Expect `[ledger] regenerated metrics/LATEST.md` and no traceback. CI's `validate-ledger` job enforces the same schema (`metrics/schema/run-v1.schema.json`) on every push.

### c. Sanity on the ledger itself
```bash
# one record today:
#   count lines
python -c "print(sum(1 for _ in open('metrics/runs.jsonl')))"
#   confirm each line is valid JSON
python -c "import json;[json.loads(l) for l in open('metrics/runs.jsonl')];print('ledger OK')"
```

## 4. Committing a real run (the E5 loop)

After a GPU run finishes (Kaggle, EuroHPC, …) it prints one JSON record. Commit it — code state + record + regenerated LATEST in a single commit (CONTRIBUTING: "a run without its committed record does not exist"):

```bash
# 1) append the printed record as ONE line to metrics/runs.jsonl
# 2) regenerate LATEST from the ledger (never hand-edit LATEST):
python train.py --regen-latest
# 3) one commit for the run:
git add metrics
git commit -m "N<n> <kind> run <run_id>: val_ppl <x> (…)"
git push
```

### Windows gotcha — write JSONL without a BOM
`Add-Content -Encoding utf8` in PowerShell prepends a **BOM** that makes the first line invalid JSON (`Expecting value: line 1 column 1`). Write the record with .NET UTF-8-**no-BOM** instead:
```powershell
$record = '{ …the full one-line JSON record… }'
[System.IO.File]::AppendAllText("$PWD\metrics\runs.jsonl", $record + "`n", (New-Object System.Text.UTF8Encoding($false)))
```
Then verify: `(Get-Content metrics\runs.jsonl).Count` and `python train.py --regen-latest` (must run clean). The `â€"`/`Â·` you may see when printing `LATEST.md` in the console is a *console display* artifact only — the file on disk is correct UTF-8, and CI/GitHub read it fine.

## 5. What "verify locally" buys us

The MVP is self-testing: `smoke-train` guards the machine that produces the numbers, `validate-ledger` guards the truth of the numbers, and both run on every push. Running them locally *before* you push means CI never surprises you — and no result in this project is ever taken on faith. Measure first.

## Pointers
`README.md` (what trains, when) · `CONTRIBUTING.md` (the ledger law, advancement rule, anti-plagiarism protocol) · `docs/SEQUENCE.md` (stage-by-stage synthesis, CI checks decoded).
