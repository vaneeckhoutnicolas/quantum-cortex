"""MQAR runner (ADR-006 Slice B): the full pipeline actually learns.

Marked slow (trains small models). Proves that generator → model → training →
scoring works end-to-end: on an easy tier, a trained model must beat chance.
This is the executable proof that the C2 judge is real, not just a generator.
"""
import pytest

from cortex_eval.mqar import MQARTier
from cortex_eval.run_mqar import run_curriculum


@pytest.mark.slow
def test_control_learns_easy_mqar_above_chance():
    # easiest tier, enough steps for a tiny model to pick up the association
    tier = MQARTier(kv_pairs=4, seq_len=64, n_symbols=32)
    res = run_curriculum(c2_variant="none", steps=1200,
                         tiers=[tier], d_model=64, n_layer=2, n_head=2)
    acc = res.per_tier[0]["accuracy"]
    # chance is ~1/32 ≈ 0.03; a working pipeline must be well above it
    assert acc > 0.15, f"trained control must beat chance on easy MQAR, got {acc:.3f}"
