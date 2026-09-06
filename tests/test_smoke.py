"""End-to-end smoke: the real trainer must train, learn, and self-record.

Mirrors the CI `smoke-train` job as a real pytest. CPU-only (ADR-002:
CPU proves correctness, never a pretraining path). Runs in seconds.

Isolation (2026-09-06): the smoke run appends a record, so it BACKS UP and
RESTORES the committed ledger — a test must never pollute the real frontier.
(Earlier smoke runs had leaked ppl-133 synthetic records into the ledger.)
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
LATEST = REPO / "metrics" / "LATEST.md"


@pytest.mark.slow
def test_cpu_smoke_trains_and_records():
    ledger_backup = LEDGER.read_text() if LEDGER.exists() else None
    latest_backup = LATEST.read_text() if LATEST.exists() else None
    before = len(ledger_backup.splitlines()) if ledger_backup else 0
    try:
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
        assert rec["results"]["final_train_loss"] < 5.0, rec["results"]
    finally:
        # restore — the committed ledger must be untouched by the test
        if ledger_backup is not None:
            LEDGER.write_text(ledger_backup)
        if latest_backup is not None:
            LATEST.write_text(latest_backup)
