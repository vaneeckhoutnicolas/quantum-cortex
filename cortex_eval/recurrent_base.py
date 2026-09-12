"""cortex_eval.recurrent_base -- the recurrent base hybrid family (register Rev38; hypotheses, no result yet).

The 8 seed ladder (2026-09-12) showed two regimes: the pure recurrent rung with a
local convolution (L3) takes off on 5 seeds out of 8 at 8 key value pairs
(0.47 to 0.67) and stays at the floor at 16; the transformer control and the
transformer based hybrids are stable around 0.09 and win at 16 pairs. The
founder's frame inverts the hybridisation: L3 as the BASE, the attention as the
ADDED path, zero initialised, taken only where it beats the recurrent. The risk
named before any code: the attention learns the easy solution first and the
recurrent never takes off (pre emption), as in the transformer based hybrids.

Four arms, declared before the run, one variable at a time:

  bare      the added attention open from step zero, no protection -- the base
            line, and the direct measure of pre emption.
  critical  the critical period (founder): the attention's gate stays closed
            until the trunk's training loss plateaus (a measured criterion, not a
            step count), then opens. Until it opens the arm IS L3, to the update.
  cls       complementary learning systems (the assistant's design): the
            attention learns immediately (the fast system); a consolidation
            term trains the trunk only prediction to reproduce the full model's
            prediction (the replay teaches the slow system); where the trunk
            alone reaches the full model, the gate retracts geometrically (the
            fast trace fades). Ratios: retraction 1/2 per phase, attention width
            1/2 of the model.
  cls-phi   the same with phi where a ratio is needed: retraction 1/phi,
            attention width 1/phi^2 (0.382) of the model. A declared arm against
            the 1/2 control -- nowhere else does phi enter.

Every unit records the full model's accuracy (the paired comparisons), the trunk
only accuracy (did the base take off?), the consolidation gap, the gate at the
end, and when the gate opened or retracted. Written from scratch (our filon).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from cortex_data import _hash_obj
from cortex_eval.mqar import MQARTier, make_batch
from cortex_eval.recurrent_ladder import LADDER, Rung, VOCAB, _build

PHI = (1.0 + 5 ** 0.5) / 2.0
L3 = next(r for r in LADDER if r.name == "L3-+local-conv")


@dataclass(frozen=True)
class RBArm:
    name: str
    mode: str                      # bare | critical | cls
    retraction: float = 0.5        # gate multiplier per retraction (cls)
    attn_frac: float = 0.5         # attention inner width as a fraction of d_model
    plateau_window: int = 100      # critical: loss window (steps)
    plateau_tol: float = 0.01      # critical: relative improvement under which the gate opens
    plateau_min_steps: int = 200   # critical: never before
    consolidation_weight: float = 1.0   # cls: weight of the replay term
    retract_every: int = 100       # cls: phase length (steps)
    takeoff_threshold: float = 0.3      # trunk only accuracy that counts as a take off
    gap_epsilon: float = 0.02      # cls: retract when full - trunk only < epsilon (and the trunk took off)

    def config_hash(self) -> str:
        return _hash_obj(asdict(self))


ARMS = [
    RBArm("RB-bare", "bare"),
    RBArm("RB-critical", "critical"),
    RBArm("RB-cls", "cls", retraction=0.5, attn_frac=0.5),
    RBArm("RB-cls-phi", "cls", retraction=1.0 / PHI, attn_frac=1.0 / PHI ** 2),
]
ARM_BY_NAME = {a.name: a for a in ARMS}


def _build_rb(arm: RBArm, d_model: int, n_head: int, seed: int):
    """The trunk is built FIRST, under the seed, so its parameters are exactly a
    standalone L3's (the no op law is testable to the bit); the attention path
    draws its parameters after."""
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    trunk = _build(L3, d_model, n_head)
    attn_dim = max(n_head, int(round(arm.attn_frac * d_model)) // n_head * n_head)

    class RecurrentBaseHybrid(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = trunk
            self.ln_a = nn.LayerNorm(d_model)
            self.qkv = nn.Linear(d_model, 3 * attn_dim, bias=False)
            self.proj = nn.Linear(attn_dim, d_model, bias=False)
            nn.init.zeros_(self.proj.weight)                  # the no op law: at init the model is L3
            self.register_buffer("gate", torch.tensor(1.0 if arm.mode != "critical" else 0.0))
            self.n_head, self.dk = n_head, attn_dim // n_head

        def trunk_hidden(self, idx):
            """The recurrent trunk up to its residual (the expensive scan), shared by
            the full and the trunk only paths."""
            t = self.trunk
            x = t.emb(idx)
            b, T, c = x.shape
            z = t.ln(x)
            z = t.conv(z.transpose(1, 2))[:, :, :T].transpose(1, 2)
            q, k, v = t.qkv(z).split(c, dim=2)
            nh, dk = n_head, c // n_head
            q = torch.nn.functional.normalize(q.view(b, T, nh, dk), dim=-1)
            k = torch.nn.functional.normalize(k.view(b, T, nh, dk), dim=-1)
            v = v.view(b, T, nh, dk)
            g = torch.sigmoid(t.gate(z)).view(b, T, nh, dk)
            S = torch.zeros(b, nh, dk, dk, device=x.device)
            outs = []
            for i in range(T):
                S = S * g[:, i].unsqueeze(2)
                S = S + torch.einsum('bhi,bhj->bhij', k[:, i], v[:, i])
                outs.append(torch.einsum('bhi,bhij->bhj', q[:, i], S).reshape(b, c))
            return x + t.out(torch.stack(outs, 1))

        def attention(self, h):
            b, T, _ = h.shape
            q, k, v = self.qkv(self.ln_a(h)).split(self.n_head * self.dk, dim=2)
            q = q.view(b, T, self.n_head, self.dk).transpose(1, 2)
            k = k.view(b, T, self.n_head, self.dk).transpose(1, 2)
            v = v.view(b, T, self.n_head, self.dk).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * self.dk ** -0.5
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=h.device))
            att = att.masked_fill(~mask, float("-inf")).softmax(-1)
            return self.proj((att @ v).transpose(1, 2).reshape(b, T, self.n_head * self.dk))

        def forward(self, idx, gate=None):
            """Returns (full logits, trunk only logits). The trunk is computed once;
            the two paths differ by the gated attention term only."""
            g = self.gate if gate is None else torch.as_tensor(float(gate))
            t = self.trunk
            h = self.trunk_hidden(idx)
            trunk_only = t.head(h + t.mlp(t.ln2(h)))
            if float(g) == 0.0:
                return trunk_only, trunk_only
            hf = h + g * self.attention(h)
            full = t.head(hf + t.mlp(t.ln2(hf)))
            return full, trunk_only

    return RecurrentBaseHybrid()


def _acc(model, tier: MQARTier, seed: int, batch: int, which: str) -> float:
    import torch
    Xe, Ye = make_batch(tier, batch, seed=seed)
    with torch.no_grad():
        full, trunk = model(torch.from_numpy(Xe))
        preds = (full if which == "full" else trunk).argmax(-1).numpy()
    mask = Ye != -100
    return float((preds[mask] == Ye[mask]).mean())


def run_rb_unit(arm: RBArm, tier: MQARTier, steps: int, seed: int, d_model: int = 64, n_head: int = 2,
                batch: int = 32, lr: float = 3e-3, eval_batch: int = 64) -> dict:
    """One unit: the arm on one tier, one seed. Same data, same optimiser, same
    steps as the pure ladder, so the paired comparison against L3 is exact."""
    import torch, torch.nn as nn
    model = _build_rb(arm, d_model, n_head, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses, opened_at, retractions = [], None, []
    for step in range(steps):
        X, Y = make_batch(tier, batch, seed=seed + step)
        idx, tgt = torch.from_numpy(X), torch.from_numpy(Y)
        full, trunk = model(idx)
        loss = nn.functional.cross_entropy(full.view(-1, VOCAB), tgt.view(-1), ignore_index=-100)
        if arm.mode == "cls" and float(model.gate) > 0.0:
            # the replay: the trunk alone learns the full model's prediction at the queries
            m = (tgt != -100)
            teacher = full.detach()[m].softmax(-1)
            student = trunk[m].log_softmax(-1)
            loss = loss + arm.consolidation_weight * nn.functional.kl_div(student, teacher, reduction="batchmean")
        if not torch.isfinite(loss):
            return {"accuracy": float("nan"), "status": "diverged"}
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(float(loss.detach()))
        # the critical period: open the gate when the trunk's loss has plateaued
        if arm.mode == "critical" and opened_at is None and step + 1 >= max(arm.plateau_min_steps, 2 * arm.plateau_window):
            w = arm.plateau_window
            recent, before = np.mean(losses[-w:]), np.mean(losses[-2 * w:-w])
            if before - recent < arm.plateau_tol * before:
                model.gate.fill_(1.0); opened_at = step + 1
        # the retraction: where the trunk alone has caught up, the fast trace fades
        if arm.mode == "cls" and (step + 1) % arm.retract_every == 0 and float(model.gate) > 1e-3:
            af = _acc(model, tier, seed + 20_000, eval_batch, "full")
            at = _acc(model, tier, seed + 20_000, eval_batch, "trunk")
            if at >= arm.takeoff_threshold and af - at < arm.gap_epsilon:
                model.gate.mul_(arm.retraction); retractions.append(step + 1)
    acc_full = _acc(model, tier, seed + 10_000, eval_batch, "full")
    acc_trunk = _acc(model, tier, seed + 10_000, eval_batch, "trunk")
    return {"accuracy": acc_full, "accuracy_trunk_only": acc_trunk,
            "consolidation_gap": round(acc_full - acc_trunk, 4),
            "took_off": bool(acc_trunk >= arm.takeoff_threshold),
            "gate_final": round(float(model.gate), 4), "opened_at": opened_at, "retractions": retractions,
            "arm_hash": arm.config_hash(), "status": "done"}
