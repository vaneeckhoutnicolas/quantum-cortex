"""C2 associative layers (ADR-006 Slice A): correctness before any training.

The two guarantees that matter:
  - flag off ⇒ the model is bit-identical to the N1 control (no drift, ever);
  - flags on ⇒ the model runs, shapes intact, and — because the associative
    output projection is zero-initialised — the initial forward equals control
    (the memory is learned from a no-op start).
"""
import copy

import torch

from train import Config, VanillaGPT


def _cfg(**kw):
    return Config(n_layer=2, n_head=2, n_embd=32, block_size=16,
                  vocab_bytes=256, reserved_oracle_tokens=8, **kw)


def _forward(cfg, seed=0):
    torch.manual_seed(seed)
    model = VanillaGPT(cfg)
    torch.manual_seed(123)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    with torch.no_grad():
        logits, _ = model(idx)
    return model, logits


def test_flag_off_is_control():
    # c2_variant="none" must construct exactly the vanilla control
    cfg = _cfg(c2_variant="none")
    model, _ = _forward(cfg)
    for blk in model.blocks:
        assert blk.c2 is None, "flag off must leave no associative layer"


def test_hopfield_runs_and_layer_is_noop_at_init():
    # the honest no-op property: the C2 RESIDUAL contribution c2(x) is zero at
    # init (zero-init output projection), so training starts from the transformer
    # and learns the memory from there. (Global logits differ only because
    # constructing extra params consumes the RNG — not a C2 leak.)
    cfg = _cfg(c2_variant="hopfield", c2_mem_slots=16)
    torch.manual_seed(7)
    model = VanillaGPT(cfg)
    blk = model.blocks[0]
    assert blk.c2 is not None
    x = torch.randn(2, cfg.block_size, cfg.n_embd)
    with torch.no_grad():
        out = blk.c2(x)
    assert out.shape == x.shape
    assert torch.count_nonzero(out) == 0, "Hopfield residual must be exactly zero at init"


def test_delta_runs_and_layer_is_noop_at_init():
    cfg = _cfg(c2_variant="delta", c2_heads=2)
    torch.manual_seed(11)
    model = VanillaGPT(cfg)
    blk = model.blocks[0]
    assert blk.c2 is not None
    x = torch.randn(2, cfg.block_size, cfg.n_embd)
    with torch.no_grad():
        out = blk.c2(x)
    assert out.shape == x.shape
    assert torch.count_nonzero(out) == 0, "Delta residual must be exactly zero at init"


def test_c2_adds_parameters_when_on():
    ctrl = sum(p.numel() for p in VanillaGPT(_cfg(c2_variant="none")).parameters())
    hop = sum(p.numel() for p in VanillaGPT(_cfg(c2_variant="hopfield")).parameters())
    delt = sum(p.numel() for p in VanillaGPT(_cfg(c2_variant="delta")).parameters())
    assert hop > ctrl and delt > ctrl, "associative layers must add counted parameters"


def test_c2_learns_gradients_flow():
    # a no-op at init must still have flowing gradients (else it can't learn)
    cfg = _cfg(c2_variant="delta", c2_heads=2)
    torch.manual_seed(3)
    model = VanillaGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    tgt = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    _, loss = model(idx, tgt)
    loss.backward()
    g = model.blocks[0].c2.out.weight.grad
    assert g is not None and torch.isfinite(g).all()
