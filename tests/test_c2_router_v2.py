"""RouterV2 — the learned soft router (RES-18 Slice B).

The central test: RouterV2 learns to route FROM CONTEXT ALONE (it predicts
per-path scores, it does not read the truth). We build synthetic spans where a
known path wins, train the head on observed performances, and check it routes to
the right path on held-out contexts. Also: it honours the Router contract, emits
soft decisions, and preserves the control floor.
"""
import torch

from cortex_c2 import Router, RouteDecision, make_router, PATH_CONTROL, PATH_HOPFIELD, PATH_DELTA
from cortex_c2.router_v2 import RouterV2, N_PATHS


def test_v2_honours_the_contract():
    r = make_router(2, ctx_dim=8)
    assert isinstance(r, Router) and r.version == 2
    ctx = torch.randn(8)
    d = r.route(span_ctx=ctx)
    assert isinstance(d, RouteDecision)
    assert 0.0 <= d.confidence <= 1.0
    assert not d.is_hard()  # v2 emits soft decisions (a mixture)


def test_v2_no_context_falls_to_floor():
    r = RouterV2(ctx_dim=8)
    d = r.route(span_ctx=None)
    assert d.path == PATH_CONTROL


def test_v2_learns_to_route_from_context():
    # synthetic task: the context's first coordinate encodes which path wins.
    #   ctx[0] < -0.3  -> hopfield best ; ctx[0] > 0.3 -> delta best ; else control
    torch.manual_seed(0)
    ctx_dim = 8
    r = RouterV2(ctx_dim=ctx_dim)
    opt = torch.optim.Adam(r.parameters(), lr=1e-2)

    def make_batch(n):
        ctx = torch.randn(n, ctx_dim)
        obs = torch.zeros(n, N_PATHS)
        for i in range(n):
            c0 = ctx[i, 0].item()
            if c0 < -0.3:      # hopfield wins
                obs[i] = torch.tensor([0.10, 0.30, 0.15])
            elif c0 > 0.3:     # delta wins
                obs[i] = torch.tensor([0.10, 0.15, 0.30])
            else:              # control wins (memory would hurt)
                obs[i] = torch.tensor([0.30, 0.10, 0.10])
        return ctx, obs

    # train the head to predict observed per-path scores from context
    for _ in range(400):
        ctx, obs = make_batch(64)
        loss = r.score_loss(ctx, obs)
        opt.zero_grad(); loss.backward(); opt.step()

    # held-out: does it route to the right path from context alone?
    correct = 0
    total = 300
    tg = torch.Generator().manual_seed(123)
    for _ in range(total):
        ctx = torch.randn(ctx_dim, generator=tg)
        c0 = ctx[0].item()
        expected = (PATH_HOPFIELD if c0 < -0.3 else
                    PATH_DELTA if c0 > 0.3 else PATH_CONTROL)
        d = r.route(span_ctx=ctx)
        correct += (d.path == expected)
    acc = correct / total
    assert acc > 0.80, f"learned router must route from context (got {acc:.2f})"


def test_v2_control_floor_pulls_away_from_losing_memory():
    # a router whose predictions put both memories below control must route to
    # the control (the floor) — verified by forcing predictions via a margin.
    torch.manual_seed(1)
    r = RouterV2(ctx_dim=4, floor_margin=0.05)
    # train it to always predict control highest
    opt = torch.optim.Adam(r.parameters(), lr=1e-2)
    for _ in range(200):
        ctx = torch.randn(32, 4)
        obs = torch.tensor([0.40, 0.10, 0.10]).expand(32, N_PATHS)
        loss = r.score_loss(ctx, obs)
        opt.zero_grad(); loss.backward(); opt.step()
    hits = sum(r.route(span_ctx=torch.randn(4)).path == PATH_CONTROL for _ in range(50))
    assert hits >= 45, "when memories are predicted below control, route to the floor"
