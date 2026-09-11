"""The resumable ladder: a quota cut loses at most one unit; resume never recomputes.

Simulates Kaggle exactly: run 1 writes units to dir A and is 'killed' by a time
budget; run 2 starts with an EMPTY working dir B, mounts A as an Input
(resume_from), skips every finished unit, and completes. The final aggregate
must equal an uninterrupted run's.
"""
import tempfile
from pathlib import Path
import pytest

from cortex_eval.mqar import MQARTier
from cortex_eval.resumable_ladder import run_resumable_ladder, load_done, unit_id, LADDER

TIER = [MQARTier(kv_pairs=4, seq_len=32)]
KW = dict(steps=30, d_model=32)


@pytest.mark.slow
def test_cut_then_resume_recomputes_nothing_and_matches_uninterrupted():
    seeds = (1, 2)                                   # 5 rungs x 2 seeds x 1 tier = 10 units
    root = Path(tempfile.mkdtemp())
    A = root / "v1_output"
    r1 = run_resumable_ladder(seeds=seeds, tiers=TIER, ckpt_dir=A, time_budget_min=0.02, **KW)
    done_A = load_done([A])
    assert 0 < len(done_A) < 10 and r1["stopped_early"] and not r1["complete"]
    assert (A / "PROGRESS-ladder8.json").exists()
    B = root / "v2_output"
    written_A = {p.name: p.stat().st_mtime for p in A.glob("unit-*.json")}
    r2 = run_resumable_ladder(seeds=seeds, tiers=TIER, ckpt_dir=B, resume_from=[A], **KW)
    assert r2["complete"]
    assert {p.name: p.stat().st_mtime for p in A.glob("unit-*.json")} == written_A   # A untouched
    assert len(list(B.glob("unit-*.json"))) == 10 - len(done_A)                       # B holds only the rest
    C = root / "uninterrupted"
    r3 = run_resumable_ladder(seeds=seeds, tiers=TIER, ckpt_dir=C, **KW)
    assert r3["complete"]
    for rung in LADDER:
        assert r2["per_rung_per_seed"][rung.name] == pytest.approx(r3["per_rung_per_seed"][rung.name], abs=1e-9)


def test_aggregate_is_reproducible_from_units_alone():
    root = Path(tempfile.mkdtemp()); D = root / "d"
    r = run_resumable_ladder(seeds=(1,), tiers=TIER, ckpt_dir=D, steps=10, d_model=32)
    done = load_done([D]); assert len(done) == 5
    assert set(done) == {unit_id(rg, 1, TIER[0]) for rg in LADDER}
