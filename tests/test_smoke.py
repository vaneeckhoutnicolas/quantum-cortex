"""End-to-end smoke: the real trainer must train, learn, and self-record.

Mirrors the CI `smoke-train` job as a real pytest. CPU-only (ADR-002:
CPU proves correctness, never a pretraining path). Runs in seconds.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import validate

REPO = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((REPO / "metrics" / "schema" / "run-v1.schema.json").read_text())
LEDGER = REPO / "metrics" / "runs.jsonl"


@pytest.mark.slow
def test_cpu_smoke_trains_and_records():
    before = len(LEDGER.read_text().splitlines()) if LEDGER.exists() else 0
    proc = subprocess.run(
        [sys.executable, "train.py", "--config", "configs/smoke_cpu.json"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"trainer failed:\n{proc.stdout}\n{proc.stderr}"

    lines = LEDGER.read_text().splitlines()
    assert len(lines) == before + 1, "smoke run must append exactly one record"

    rec = json.loads(lines[-1])
    validate(rec, SCHEMA)
    assert rec["run"]["status"] == "completed", rec["run"]
    assert rec["run"]["anomalies"] is None, f"anomaly: {rec['run']['anomalies']}"
    # loss must have decreased below the log(vocab) random-init baseline (~5.55 for byte)
    assert rec["results"]["final_train_loss"] < 5.0, rec["results"]
