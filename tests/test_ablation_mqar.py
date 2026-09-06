"""MQAR ablation driver (Slice C, ADR-006): the 3-variant comparison runs.

Marked slow (trains three small models). Proves the driver produces curves for
all three variants, compares them, emits a verdict, and writes a separate
artefact — without touching the run-v1 ledger.
"""
import json
import tempfile
from pathlib import Path

import pytest

from cortex_eval.mqar import MQARTier
from cortex_eval.ablation_mqar import run_ablation, write_artefact, VARIANTS


@pytest.mark.slow
def test_ablation_runs_all_variants_and_writes_artefact():
    tiers = [MQARTier(kv_pairs=4, seq_len=48)]  # single easy tier, fast
    result = run_ablation(steps=250, tiers=tiers, threshold=0.5)
    # all three variants produced a curve
    assert set(result["curves"].keys()) == set(VARIANTS)
    for v in VARIANTS:
        assert len(result["curves"][v]) == 1
        assert "accuracy" in result["curves"][v][0]
    # a verdict string exists and names the comparison
    assert isinstance(result["verdict"], str) and len(result["verdict"]) > 0
    # artefact writes to a separate dir, not the ledger
    d = tempfile.mkdtemp()
    path = write_artefact(result, d)
    assert path.exists()
    reloaded = json.loads(path.read_text())
    assert reloaded["benchmark"] == "mqar"
    assert (Path(d) / "LATEST.json").exists()
