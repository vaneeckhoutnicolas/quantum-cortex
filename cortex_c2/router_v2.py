"""cortex_c2.router_v2 — the learned soft router (RES-18, ADR-006 D8), Slice B.

Slice A's RouterV1 is an ORACLE: it *reads* the true per-path scores. That is the
ceiling, not a usable router — at inference there is no label saying "Hopfield
scores 0.06 here". RouterV2 is the real thing: a small head that **predicts** a
score per path *from the span context alone*, then routes by weighting. It learns
by a supervised signal — its predictions must correlate with the performances
actually observed (the ablation gives those). The central challenge of the whole
router: route without knowing the answer.

RouterV2 honours the *same* `Router` interface as v1 (retro-compat in action:
v2 is added, v1 stays; the version dispatch gains one branch). It emits a **soft**
RouteDecision (a weighted mixture over paths) with a confidence; Slice C turns
high-confidence decisions into hard routing and wires the breaker fallback.

The control floor (RES-18) is preserved: the predicted-best path is used only if
it clears the control's predicted score by a margin; else the mixture is pulled
toward the control. C2 can never be made to degrade the model by the router.

Depends on torch (kept out of cortex_c2/__init__.py so the core stays light).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex_c2 import (Router, RouteDecision, PATH_CONTROL, PATH_HOPFIELD,
                       PATH_DELTA, PATH_NAMES)

N_PATHS = 3  # control, hopfield, delta


class RouterV2(nn.Module, Router):
    """A learned gate that predicts per-path scores from a context vector.

    - `ctx_dim`: dimensionality of the span-context feature the router sees.
    - forward(ctx) -> predicted score per path (b, N_PATHS).
    - route(span_ctx=<tensor>) -> a soft RouteDecision (weights = softmax over the
      predicted scores, with the control floor enforced), plus a confidence.
    - training: `score_loss(ctx, observed)` regresses predictions onto the
      observed per-path performances (the supervised signal from the ablation).

    Version dispatch: `version = 2`; `Router.load_state` gains its v2 branch in
    Slice C when v2 carries persistent state. For now v2's "state" is its weights,
    saved by the normal torch checkpoint — the contract's tag still applies.
    """

    version = 2

    def __init__(self, ctx_dim: int = 32, hidden: int = 64,
                 floor_margin: float = 0.0, temperature: float = 1.0):
        nn.Module.__init__(self)
        self.ctx_dim = ctx_dim
        self.floor_margin = floor_margin
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, N_PATHS),
        )

    # -- prediction ----------------------------------------------------------
    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        """ctx: (b, ctx_dim) -> predicted per-path score (b, N_PATHS)."""
        return self.net(ctx)

    # -- the Router contract -------------------------------------------------
    def route(self, *, span_ctx=None, path_scores: Sequence[float] | None = None,
              state: dict | None = None) -> RouteDecision:
        """Route from the *predicted* scores. span_ctx is the context feature
        (a tensor of shape (ctx_dim,) or (1, ctx_dim)); path_scores is ignored
        (v2 predicts, it does not read the truth — that was v1's oracle)."""
        if span_ctx is None:
            return RouteDecision(path=PATH_CONTROL, confidence=1.0)  # floor
        ctx = span_ctx if span_ctx.dim() == 2 else span_ctx.unsqueeze(0)
        with torch.no_grad():
            pred = self.forward(ctx).squeeze(0)          # (N_PATHS,)
        # control floor: a memory competes only if it clears control + margin
        adj = pred.clone()
        control = pred[PATH_CONTROL]
        for p in (PATH_HOPFIELD, PATH_DELTA):
            if pred[p] <= control + self.floor_margin:
                adj[p] = float("-inf")                    # remove from the mixture
        weights = F.softmax(adj / self.temperature, dim=-1)
        best = int(torch.argmax(weights).item())
        # confidence = how peaked the mixture is (max weight, rescaled)
        conf = float((weights.max() - 1.0 / N_PATHS) / (1.0 - 1.0 / N_PATHS))
        conf = max(0.0, min(1.0, conf))
        return RouteDecision(path=best, confidence=conf,
                             weights=weights.tolist())

    # -- training signal -----------------------------------------------------
    def score_loss(self, ctx: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        """MSE between predicted and observed per-path scores.

        ctx      : (b, ctx_dim)
        observed : (b, N_PATHS) — the true per-path performances for those spans
                   (from the ablation / online measurement). This is how the
                   router learns to predict which path wins, from context alone.
        """
        pred = self.forward(ctx)
        return F.mse_loss(pred, observed)


def make_router_v2(ctx_dim: int = 32, **kw) -> RouterV2:
    return RouterV2(ctx_dim=ctx_dim, **kw)
