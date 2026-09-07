"""ConsolidatingController (RES-18 Slice C): the founder's three-message design.

Tests the three behaviours as code: hardening (soft→hard on learned decisions),
the control floor, the breaker fallback (re-route + re-soften, never a void),
and the hysteresis (anti-flapping). Pure-Python (drives a stub Router).
"""
from cortex_c2 import Router, RouteDecision, PATH_CONTROL, PATH_HOPFIELD, PATH_DELTA
from cortex_c2.consolidation import ConsolidatingController, ConsolidationConfig


class StubRouter(Router):
    """A router we can steer: always returns a fixed decision (path + confidence)."""
    version = 99
    def __init__(self, path=PATH_HOPFIELD, confidence=0.9):
        self._d = RouteDecision(path=path, confidence=confidence, weights=[0.1, 0.8, 0.1])
    def set(self, path, confidence):
        self._d = RouteDecision(path=path, confidence=confidence, weights=None)
    def route(self, *, span_ctx=None, path_scores=None, state=None):
        return self._d


# --- message 2: hardening (soft → hard on a learned, stable, confident decision)
def test_hardens_after_stable_confident_passes():
    r = StubRouter(path=PATH_HOPFIELD, confidence=0.9)
    ctl = ConsolidatingController(r, ConsolidationConfig(harden_confidence=0.75, harden_stability=5))
    # first passes: soft (not yet hardened)
    for i in range(4):
        d = ctl.route(key="A")
        assert not (ctl.states["A"].hardened_path is not None), f"pass {i}: should still be soft"
    # the 5th confident-agreeing pass hardens it
    ctl.route(key="A")
    assert ctl.states["A"].hardened_path == PATH_HOPFIELD, "should harden after 5 stable passes"
    # once hardened, the decision is HARD (no soft mixture)
    d = ctl.route(key="A")
    assert d.is_hard() and d.path == PATH_HOPFIELD


def test_low_confidence_never_hardens():
    r = StubRouter(path=PATH_HOPFIELD, confidence=0.5)  # below threshold
    ctl = ConsolidatingController(r, ConsolidationConfig(harden_confidence=0.75, harden_stability=5))
    for _ in range(20):
        ctl.route(key="B")
    assert ctl.states["B"].hardened_path is None, "unconfident decisions must never harden"


# --- message 1: the control floor is honoured through the controller
def test_controller_respects_floor_via_router():
    r = StubRouter(path=PATH_CONTROL, confidence=0.9)  # router already chose the floor
    ctl = ConsolidatingController(r)
    d = ctl.route(key="C")
    assert d.path == PATH_CONTROL
    # control is never hardened as a "memory" (it is the floor, always available)
    for _ in range(10):
        ctl.route(key="C")
    assert ctl.states["C"].hardened_path is None


# --- message 3: breaker fallback — re-route, re-soften, never a void
def test_breaker_fallback_reroutes_and_resoftens():
    r = StubRouter(path=PATH_HOPFIELD, confidence=0.9)
    ctl = ConsolidatingController(r)
    # harden to hopfield first
    for _ in range(6):
        ctl.route(key="D")
    assert ctl.states["D"].hardened_path == PATH_HOPFIELD
    # hopfield degrades → breaker fires; delta is above control on this span
    d = ctl.report_breaker(key="D", path=PATH_HOPFIELD,
                           path_scores=[0.10, 0.05, 0.30])  # ctrl, hop(dead), delta
    assert d.path == PATH_DELTA, "fallback must pick the best remaining path above control"
    # the decision is re-softened (hardening dropped) and hopfield is banned
    assert ctl.states["D"].hardened_path is None
    assert ctl.states["D"].banned_path == PATH_HOPFIELD


def test_breaker_fallback_to_floor_when_no_memory_above_control():
    r = StubRouter(path=PATH_HOPFIELD, confidence=0.9)
    ctl = ConsolidatingController(r)
    # both memories below control → fallback is the floor (never a void)
    d = ctl.report_breaker(key="E", path=PATH_HOPFIELD,
                           path_scores=[0.30, 0.05, 0.10])
    assert d.path == PATH_CONTROL


# --- message 3: hysteresis prevents immediate re-selection (anti-flapping)
def test_hysteresis_bans_path_during_cooldown():
    r = StubRouter(path=PATH_HOPFIELD, confidence=0.9)
    ctl = ConsolidatingController(r, ConsolidationConfig(hysteresis_cooldown=3))
    ctl.report_breaker(key="F", path=PATH_HOPFIELD, path_scores=[0.10, 0.05, 0.30])
    # router keeps proposing hopfield, but it is banned → pulled to control
    r.set(PATH_HOPFIELD, 0.9)
    for _ in range(3):
        d = ctl.route(key="F")
        assert d.path == PATH_CONTROL, "banned path must be suppressed during cooldown"
    # after cooldown, hopfield is eligible again
    d = ctl.route(key="F")
    assert d.path == PATH_HOPFIELD, "path is re-eligible once cooldown expires"


def test_snapshot_is_inspectable():
    r = StubRouter(path=PATH_HOPFIELD, confidence=0.9)
    ctl = ConsolidatingController(r)
    for _ in range(6):
        ctl.route(key="G")
    snap = ctl.snapshot()
    assert snap["keys"] == 1 and snap["hardened"] == 1 and snap["soft"] == 0
