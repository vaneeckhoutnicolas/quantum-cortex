"""The control plus a local convolution (register Rev60, declared 2026-09-17 before
any run): one variable added to the ladder's transformer control, L3's depthwise
causal convolution on the attention's normalised input in every block. The flag
is transparent to the configuration hash at its default, so every recorded hash
stands; the convolution is causal; the extra ladder path is selectable by
`--paths` only and never enters the full ladder; the declared readout computes
the paired tests from unit files alone."""
from __future__ import annotations

import numpy as np
import torch

from cortex_eval.mqar import MQARTier
from cortex_eval.recurrent_base import ARM_BY_NAME
from cortex_eval.resumable_ladder import ARCH, ARCH_EXTRA, LADDER, run_unit, unit_id
from cortex_eval.resumable_rb import REF_CONTROL, REF_CONTROL_CONV, REF_L3, control_conv_readout, rb_unit_id, ref_unit_id
from train import Config, HASH_TRANSPARENT_AT_DEFAULT, VanillaGPT, config_hash


def _tiny(**kw) -> Config:
    return Config(n_layer=2, n_head=2, n_embd=16, block_size=32, vocab_bytes=256, reserved_oracle_tokens=8, **kw)


def test_the_flag_is_transparent_to_the_hash_at_its_default_and_enters_it_when_on():
    assert HASH_TRANSPARENT_AT_DEFAULT["attn_local_conv"] is False
    assert config_hash(_tiny()) == config_hash(_tiny(attn_local_conv=False))
    assert config_hash(_tiny(attn_local_conv=True)) != config_hash(_tiny())


def test_the_convolution_is_absent_by_default_and_depthwise_causal_kernel_three_when_on():
    off = VanillaGPT(_tiny())
    assert all(blk.conv is None for blk in off.blocks)
    on = VanillaGPT(_tiny(attn_local_conv=True))
    for blk in on.blocks:
        assert blk.conv is not None and blk.conv.kernel_size == (3,) and blk.conv.groups == 16
    torch.manual_seed(0)
    on.eval()
    idx = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        base, _ = on(idx)
        idx2 = idx.clone(); idx2[0, 8] = (idx2[0, 8] + 1) % 256   # a later token changed
        moved, _ = on(idx2)
    assert torch.allclose(base[0, :8], moved[0, :8], atol=1e-6)    # nothing before it moves: causal
    assert not torch.allclose(base[0, 8:], moved[0, 8:])


def test_the_extra_path_is_selectable_only_and_trains_through_the_control_s_own_runner():
    assert ARCH_EXTRA == ("none+local-conv",) and "none+local-conv" not in ARCH
    tier = MQARTier(kv_pairs=2, seq_len=16)
    assert unit_id("none+local-conv", 1, tier) == "ARCH-none+local-conv__s1__kv2_seq16"
    acc = run_unit("none+local-conv", tier, steps=2, seed=1, d_model=16)
    assert 0.0 <= acc <= 1.0


def test_the_declared_readout_computes_the_paired_tests_from_units_alone():
    seeds, tiers = [1, 2, 3, 4], [MQARTier(kv_pairs=8, seq_len=128)]
    refs, done = {}, {}
    for i, s in enumerate(seeds):
        refs[ref_unit_id(REF_CONTROL_CONV, s, tiers[0])] = {"accuracy": 0.9 + 0.01 * i}
        refs[ref_unit_id(REF_CONTROL, s, tiers[0])] = {"accuracy": 0.1 + 0.01 * i}
        refs[ref_unit_id(REF_L3, s, tiers[0])] = {"accuracy": 0.8 + 0.02 * i}
        done[rb_unit_id(ARM_BY_NAME["RB-bare"], s, tiers[0])] = {"accuracy": 0.97 - 0.005 * i}
    r = control_conv_readout(refs, done, [ARM_BY_NAME["RB-bare"], ARM_BY_NAME["RB-critical"]], seeds, tiers)
    assert set(r["paired_tests"]) == {"control_conv_vs_control", "control_conv_vs_L3", "RB-bare_full_vs_control_conv"}
    assert r["gate_readout"]["control_conv_vs_control"] == "gated"
    assert r["stats"][REF_CONTROL_CONV]["n"] == 4 and r["per_seed"][REF_CONTROL_CONV] == [0.9, 0.91, 0.92, 0.93]
    assert "RB-critical" not in r["per_seed"]["arms_full"]         # an arm without units is skipped, not invented
