"""cortex_eval.recurrent_ladder — the recurrent-family ladder (D19 Level 1c, refined).

The founder's question: "custom GLA vs official Mamba — isn't there an
intermediate?" — and the answer: the space between them is CONTINUOUS. Mamba is
the canonical gated linear attention PLUS a stack of documented refinements.
So instead of one custom GLA or one official Mamba, we build a LADDER: the
canonical GLA, then each refinement added one at a time behind a flag, each
re-implemented from its paper (zero borrowed code), each measured. Our own
ablation discipline (one variable at a time, equal at everything) applied to
the construction of the reference itself.

The rungs (each credited to the paper that introduced the idea):
  L0  canonical GLA        S_t = g ⊙ S_{t-1} + k vᵀ ; o = q S           (linear attention lineage)
  L1  + input-dependent gate  g_t = σ(W_g x_t)  — Mamba's "selectivity" (Gu & Dao 2023)
  L2  + channel-wise decay    per-channel forget instead of per-head   (Kimi Linear's KDA finding)
  L3  + short local conv      a depthwise conv on the input before mixing (Mamba / RWKV lineage)
  L4  + delta-rule write      S += β k (v − Sᵀk)ᵀ with L2-normalised keys (DeltaNet / KDA)

Reading rule (the founder's curve principle): read the CURVE of MQAR accuracy
vs rung, not one point. Note L4 is exactly our own DeltaMemory — so the top of
the ladder places our component INSIDE the recurrent family, and "L4 pure" vs
"our hybrid (L4 + attention)" isolates the contribution of hybridisation.

Claim discipline: we compare to "GLA + <named rungs>", never to "Mamba" — an
official pre-trained Mamba/RWKV remains a separate, non-equalised external
reference (a later job, license-checked at ingestion).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from cortex_eval.mqar import MQARTier, make_batch

VOCAB = 256 + 8


@dataclass
class Rung:
    name: str
    selective_gate: bool = False    # L1
    channel_decay: bool = False     # L2
    local_conv: bool = False        # L3
    delta_rule: bool = False        # L4


LADDER = [
    Rung("L0-canonical-gla"),
    Rung("L1-+selective-gate", selective_gate=True),
    Rung("L2-+channel-decay", selective_gate=True, channel_decay=True),
    Rung("L3-+local-conv", selective_gate=True, channel_decay=True, local_conv=True),
    Rung("L4-+delta-rule", selective_gate=True, channel_decay=True, local_conv=True, delta_rule=True),
]


def _build(rung: Rung, d_model: int, n_head: int):
    import torch, torch.nn as nn
    dk = d_model // n_head

    class RecurrentRung(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(VOCAB, d_model)
            self.ln = nn.LayerNorm(d_model)
            self.conv = (nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)
                         if rung.local_conv else None)                       # L3: depthwise, causal (crop)
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            # gate: constant-per-head (L0) or input-dependent (L1); per-head or per-channel (L2)
            gate_out = n_head * dk if rung.channel_decay else n_head
            self.gate = (nn.Linear(d_model, gate_out, bias=True) if rung.selective_gate
                         else nn.Parameter(torch.zeros(gate_out)))          # L0: learned constant
            self.beta = nn.Linear(d_model, n_head, bias=True) if rung.delta_rule else None  # L4
            self.out = nn.Linear(d_model, d_model, bias=False)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                     nn.Linear(4 * d_model, d_model))
            self.head = nn.Linear(d_model, VOCAB, bias=False)

        def forward(self, idx):
            x = self.emb(idx)
            b, T, c = x.shape
            z = self.ln(x)
            if self.conv is not None:                                         # L3
                z = self.conv(z.transpose(1, 2))[:, :, :T].transpose(1, 2)    # causal crop
            q, k, v = self.qkv(z).split(c, dim=2)
            q = torch.nn.functional.normalize(q.view(b, T, n_head, dk), dim=-1)
            k = torch.nn.functional.normalize(k.view(b, T, n_head, dk), dim=-1)
            v = v.view(b, T, n_head, dk)
            if rung.selective_gate:                                           # L1
                g_raw = torch.sigmoid(self.gate(z))
            else:                                                             # L0
                g_raw = torch.sigmoid(self.gate).expand(b, T, -1)
            if rung.channel_decay:                                            # L2
                g = g_raw.view(b, T, n_head, dk)
            else:
                g = g_raw.view(b, T, n_head, 1).expand(b, T, n_head, dk)
            beta = torch.sigmoid(self.beta(z)).view(b, T, n_head, 1) if rung.delta_rule else None
            S = torch.zeros(b, n_head, dk, dk, device=x.device)
            outs = []
            for i in range(T):
                S = S * g[:, i].unsqueeze(2)                                   # forget (columns)
                if rung.delta_rule:                                           # L4
                    u = torch.einsum('bhij,bhi->bhj', S, k[:, i])
                    S = S + beta[:, i].unsqueeze(-1) * torch.einsum('bhi,bhj->bhij', k[:, i], v[:, i] - u)
                else:                                                         # L0–L3: plain write
                    S = S + torch.einsum('bhi,bhj->bhij', k[:, i], v[:, i])
                outs.append(torch.einsum('bhi,bhij->bhj', q[:, i], S).reshape(b, c))
            o = torch.stack(outs, 1)
            x = x + self.out(o)
            x = x + self.mlp(self.ln2(x))
            return self.head(x)

    return RecurrentRung()


def run_rung(rung: Rung, tier: MQARTier, steps: int, seed: int, d_model: int = 64,
             n_head: int = 2, batch: int = 32, lr: float = 3e-3, eval_batch: int = 64) -> float:
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    model = _build(rung, d_model, n_head)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        X, Y = make_batch(tier, batch, seed=seed + step)
        logits = model(torch.from_numpy(X))
        loss = nn.functional.cross_entropy(logits.view(-1, VOCAB),
                                           torch.from_numpy(Y).view(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            return float("nan")                     # circuit-breaker spirit
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    Xe, Ye = make_batch(tier, eval_batch, seed=seed + 10_000)
    with torch.no_grad():
        preds = model(torch.from_numpy(Xe)).argmax(-1).numpy()
    mask = Ye != -100
    return float((preds[mask] == Ye[mask]).mean())


def run_ladder(tiers: list[MQARTier] | None = None, steps: int = 1500, seed: int = 1337,
               rungs: list[Rung] | None = None, **kw) -> dict:
    tiers = tiers or [MQARTier(kv_pairs=4, seq_len=128), MQARTier(kv_pairs=8, seq_len=128),
                      MQARTier(kv_pairs=16, seq_len=128)]
    rungs = rungs or LADDER
    curve = []
    for rung in rungs:
        accs = [run_rung(rung, t, steps, seed, **kw) for t in tiers]
        auc = float(np.nanmean(accs)) if any(a == a for a in accs) else float("nan")
        curve.append({"rung": rung.name, "per_tier": accs, "auc": round(auc, 4)})
        print(f"[ladder] {rung.name:22s} -> auc={auc:.4f}")
    return {
        "benchmark": "recurrent_ladder",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "regime": {"tiers": [(t.kv_pairs, t.seq_len) for t in tiers], "steps": steps, "seed": seed},
        "curve": curve,
        "reading": "MQAR AUC vs recurrent refinement rung. L4 (delta-rule, normalised keys) is our own "
                   "DeltaMemory — the top of the ladder places our component inside the recurrent family; "
                   "'L4 pure' vs our hybrid isolates hybridisation. Compare to 'GLA + named rungs', never to 'Mamba'.",
    }
