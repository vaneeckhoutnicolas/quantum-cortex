"""cortex_eval.external_baseline — D19 Level 1c: an external anchor.

A number nobody can situate is not a result. This module runs, on OUR MQAR
protocol at EQUAL size, two reference points anyone can reproduce:
  - a standard GPT-2-style transformer (the nanoGPT / VanillaGPT lineage —
    which is exactly our control, so this makes the "control = a known public
    architecture" statement explicit and checkable);
  - an absolute floor: a position-wise MLP with NO attention (cannot do
    associative recall by construction) — the chance-level anchor.
So our variants sit between a public reference (the transformer) and a hard
floor (no-attention MLP). Same tiers, same steps, same seeds — equal at
everything (hub-019). Nothing borrowed: the MLP baseline is written here.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from cortex_eval.mqar import MQARTier, make_batch


def _run_mlp_baseline(tier: MQARTier, steps: int, seed: int, d_model: int = 64,
                      batch: int = 32, lr: float = 3e-3, eval_batch: int = 64) -> float:
    """A position-wise MLP over token embeddings — no attention, no recurrence.
    It sees each position independently, so it CANNOT look back at the key/value
    pairs: an honest floor for associative recall."""
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    vocab = 256 + 8
    emb = nn.Embedding(vocab, d_model)
    mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, vocab))
    params = list(emb.parameters()) + list(mlp.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    for step in range(steps):
        X, Y = make_batch(tier, batch, seed=seed + step)
        logits = mlp(emb(torch.from_numpy(X)))
        loss = nn.functional.cross_entropy(logits.view(-1, vocab),
                                           torch.from_numpy(Y).view(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    Xe, Ye = make_batch(tier, eval_batch, seed=seed + 10_000)
    with torch.no_grad():
        preds = mlp(emb(torch.from_numpy(Xe))).argmax(-1).numpy()
    mask = Ye != -100
    return float((preds[mask] == Ye[mask]).mean())


def run_external_anchor(tiers: list[MQARTier] | None = None, steps: int = 1500,
                        seed: int = 1337, **kw) -> dict:
    from cortex_eval.run_mqar import run_curriculum
    tiers = tiers or [MQARTier(kv_pairs=4, seq_len=128), MQARTier(kv_pairs=8, seq_len=128),
                      MQARTier(kv_pairs=16, seq_len=128)]
    out = {"mlp_no_attention": [], "transformer_control": [], "hopfield": [], "delta": []}
    for tier in tiers:
        out["mlp_no_attention"].append(
            {"kv_pairs": tier.kv_pairs, "seq_len": tier.seq_len,
             "accuracy": _run_mlp_baseline(tier, steps, seed)})
    for name, v in (("transformer_control", "none"), ("hopfield", "hopfield"), ("delta", "delta")):
        res = run_curriculum(c2_variant=v, steps=steps, tiers=tiers, seed=seed, **kw)
        out[name] = res.per_tier
    auc = {k: round(float(np.mean([t["accuracy"] for t in v])), 4) for k, v in out.items()}
    return {
        "benchmark": "mqar_external_anchor",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "regime": {"tiers": [(t.kv_pairs, t.seq_len) for t in tiers], "steps": steps, "seed": seed},
        "curves": out,
        "auc": auc,
        "reading": {
            "floor": "mlp_no_attention — cannot recall by construction (chance-level anchor)",
            "public_reference": "transformer_control — a standard GPT-2-style model (nanoGPT lineage), "
                                "identical to our control: our 'control' IS a known public architecture",
            "our_variants": "hopfield / delta — the C2 memories, measured against the same anchors",
        },
        "note": "D19 Level 1c — equal size, tiers, steps, seed. The MLP floor is written from scratch here.",
    }
