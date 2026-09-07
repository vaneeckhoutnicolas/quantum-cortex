"""Confirmation-run driver (whitepaper v0 gate): the orchestration runs end-to-end."""
import pytest
from cortex_eval.confirmation_run import run_confirmation
from cortex_eval.mqar import MQARTier


@pytest.mark.slow
def test_confirmation_orchestrates_all_three_parts():
    r = run_confirmation(seeds=(1, 2), steps=60, tiers=[MQARTier(kv_pairs=4, seq_len=32)], d_model=32)
    assert set(r["c2_ablation"]["stats"]) == {"none", "hopfield", "delta"}
    assert len(r["recurrent_ladder"]["stats"]) == 5
    assert "gate_readout" in r and "hybrid_vs_pure" in r
