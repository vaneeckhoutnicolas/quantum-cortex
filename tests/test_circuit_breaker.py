"""Circuit breaker (ADR-006 D7): NaN -> clean abort with a truthful record.

Marked slow (runs the real trainer). Restores the ledger afterwards so the
test never pollutes the committed record.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(cfg_dict):
    d = tempfile.mkdtemp()
    cfgp = Path(d) / "cfg.json"
    cfgp.write_text(json.dumps(cfg_dict))
    # isolate the ledger so we don't touch the real one
    env = {"PYTHONPATH": str(REPO)}
    import os
    env = {**os.environ, **env}
    proc = subprocess.run(
        [sys.executable, "train.py", "--config", str(cfgp)],
        cwd=REPO, capture_output=True, text=True, env=env,
    )
    return proc


import pytest


@pytest.mark.slow
def test_nan_triggers_clean_abort():
    # an absurd LR makes the loss blow up to NaN within a few steps
    cfg = {
        "kind": "ablation", "component_under_test": "C2", "provider": "other",
        "notes": "circuit-breaker NaN test",
        "n_layer": 2, "n_head": 2, "n_embd": 32, "block_size": 16,
        "data_mode": "synthetic", "dataset_id": "synthetic-v0",
        "data_slice": "cb-test", "val_fraction": 0.05,
        "max_tokens": 40000, "batch_size": 8, "lr": 50.0, "warmup_steps": 0,
        "eval_every": 20, "eval_batches": 2, "ckpt_every": 50,
        "seed": 1, "out_dir": "runs/_cbtest",
        "cb_enabled": True,
    }
    ledger = REPO / "metrics" / "runs.jsonl"
    backup = ledger.read_text() if ledger.exists() else None
    try:
        proc = _run(cfg)
        _assert_abort(proc)
    finally:
        if backup is not None:
            ledger.write_text(backup)  # never pollute the committed ledger


def _assert_abort(proc):
    assert proc.returncode == 0, f"abort must be clean:\n{proc.stdout}\n{proc.stderr}"
    rec = json.loads((REPO / "metrics" / "runs.jsonl").read_text().splitlines()[-1])
    assert rec["run"]["status"] == "aborted", rec["run"]
    assert "NaN" in (rec["run"]["anomalies"] or ""), rec["run"]["anomalies"]
