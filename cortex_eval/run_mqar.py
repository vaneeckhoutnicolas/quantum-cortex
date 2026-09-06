"""cortex_eval.run_mqar — train a small model on MQAR, tier by tier (option a).

The canonical MQAR protocol (Zoology/Kimi): MQAR is an *architecture* probe —
train a small model directly on the synthetic task and read its accuracy. Here
we sweep the difficulty curriculum (kv × seq_len) and, for a given C2 variant
(none | hopfield | delta), produce the accuracy CURVE — the deliverable.

This isolates C2's associative capacity on the memory task itself, cheaply
(small model, synthetic sequences, minutes on CPU). The circuit breaker
(ADR-006 D7) applies naturally: a variant that diverges on a tier is aborted.

Usage (programmatic; a thin CLI is provided):
    from cortex_eval.run_mqar import run_curriculum
    result = run_curriculum(c2_variant="delta", steps=800, tiers=[...])
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np

from cortex_eval.mqar import MQARTier, MQARResult, make_batch, standard_curriculum, SEP


def _lazy_torch():
    import torch
    return torch


def _build_model(vocab: int, d_model: int, n_layer: int, n_head: int,
                 c2_variant: str, block_size: int):
    """Reuse train.py's VanillaGPT with a Config, so the C2 layers under test
    are exactly the ones N2 will train (single source of truth)."""
    from train import Config, VanillaGPT
    cfg = Config(
        n_layer=n_layer, n_head=n_head, n_embd=d_model, block_size=block_size,
        vocab_bytes=vocab - 8, reserved_oracle_tokens=8,   # vocab = symbols + SEP room; keep 8 oracle slot convention
        c2_variant=c2_variant,
    )
    return cfg, VanillaGPT(cfg)


@dataclass
class MQARRunConfig:
    c2_variant: str = "none"
    d_model: int = 64
    n_layer: int = 2
    n_head: int = 2
    steps: int = 800
    batch: int = 32
    lr: float = 3e-3
    seed: int = 1337
    eval_batch: int = 64


def _train_one_tier(rc: MQARRunConfig, tier: MQARTier):
    torch = _lazy_torch()
    torch.manual_seed(rc.seed)
    # vocab must cover symbols (0..n_symbols-1), SEP(255); use full byte vocab
    vocab = 256 + 8
    block = max(tier.seq_len, 2 * tier.kv_pairs + 1 + tier.queries()) + 4
    cfg, model = _build_model(vocab, rc.d_model, rc.n_layer, rc.n_head,
                              rc.c2_variant, block)
    opt = torch.optim.AdamW(model.parameters(), lr=rc.lr)

    diverged = False
    for step in range(rc.steps):
        X, Y = make_batch(tier, rc.batch, seed=rc.seed + step)
        xb = torch.from_numpy(X)
        yb = torch.from_numpy(Y)
        logits, _ = model(xb)                       # (b, T, vocab)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=-100)
        if not torch.isfinite(loss):                # circuit breaker (hard)
            diverged = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    if diverged:
        return float("nan"), True

    # evaluation: exact-match accuracy at query positions on fresh sequences
    model.eval()
    Xe, Ye = make_batch(tier, rc.eval_batch, seed=rc.seed + 10_000)
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(Xe))
    preds = logits.argmax(-1).numpy()
    mask = Ye != -100
    acc = float((preds[mask] == Ye[mask]).mean())
    return acc, False


def run_curriculum(c2_variant: str = "none", steps: int = 800,
                   tiers: list[MQARTier] | None = None, **kw) -> MQARResult:
    rc = MQARRunConfig(c2_variant=c2_variant, steps=steps, **kw)
    tiers = tiers or standard_curriculum()
    result = MQARResult()
    for tier in tiers:
        acc, diverged = _train_one_tier(rc, tier)
        result.add(tier, acc)
        tag = "ABORTED(diverged)" if diverged else f"acc={acc:.3f}"
        print(f"[mqar] variant={c2_variant} kv={tier.kv_pairs} seq={tier.seq_len} -> {tag}")
    return result


def _cli():
    ap = argparse.ArgumentParser(description="MQAR curriculum runner (ADR-006 Slice B)")
    ap.add_argument("--variant", default="none", choices=["none", "hopfield", "delta"])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--quick", action="store_true", help="tiny curriculum for a smoke check")
    args = ap.parse_args()
    tiers = ([MQARTier(kv_pairs=4, seq_len=64), MQARTier(kv_pairs=8, seq_len=64)]
             if args.quick else None)
    res = run_curriculum(c2_variant=args.variant, steps=args.steps, tiers=tiers)
    print(json.dumps(res.as_dict(), indent=2))


if __name__ == "__main__":
    _cli()
