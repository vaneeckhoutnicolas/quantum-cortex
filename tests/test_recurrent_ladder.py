"""The recurrent ladder (D19 Level 1c, refined): structure + a rung actually runs."""
import numpy as np
import pytest

from cortex_eval.recurrent_ladder import LADDER, Rung, run_rung, _build
from cortex_eval.mqar import MQARTier


def test_ladder_is_monotone_in_refinements():
    # each rung adds exactly one refinement on top of the previous (one variable at a time)
    flags = [(r.selective_gate, r.channel_decay, r.local_conv, r.delta_rule) for r in LADDER]
    assert flags[0] == (False, False, False, False)
    for a, b in zip(flags, flags[1:]):
        assert sum(b) == sum(a) + 1, "each rung must add exactly one refinement"
    assert LADDER[-1].delta_rule  # L4 = our DeltaMemory family


def test_each_rung_builds_and_forwards():
    import torch
    for rung in LADDER:
        m = _build(rung, d_model=32, n_head=2)
        out = m(torch.randint(0, 264, (2, 16)))
        assert out.shape == (2, 16, 264)
        assert torch.isfinite(out).all(), rung.name


@pytest.mark.slow
def test_a_rung_learns_above_floor():
    # L3 (the strongest pure rung in the mini-run) should beat chance on easy MQAR
    acc = run_rung(LADDER[3], MQARTier(kv_pairs=4, seq_len=48, n_symbols=32),
                   steps=400, seed=1, d_model=48)
    assert acc == acc and acc > 0.05, f"a refined recurrent must beat chance (got {acc})"
