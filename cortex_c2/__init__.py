"""cortex_c2 — the multi-path C2 memory router (RES-18, ADR-006 D8), Slice A.

C2 is not one memory nor a choice between Hopfield and delta: it is a ROUTER of
memory paths on a guaranteed control floor. On any span, a memory is used only
where it beats the control; if all are below, the control is used — so C2 can
never degrade the model. (The ablation proved the floor is needed: the control
won 3 of 12 MQAR tiers.)

Slice A ships the FOUNDATION, retro-compatibility included from day one:
  - a VERSIONED, stable Router interface (the contract every future version honours);
  - RouterV1 = the upper-envelope oracle (picks, per span, the best of the known
    path scores — the ceiling the learned router will chase: +21% vs control);
  - version-tagged serialisation + a single-branch version dispatch (adding v2 is
    one more branch, never a change to the existing one);
  - a contract test (the guard) that any RouterVN must pass.

What is deliberately deferred (not by caution but by logic): ACTIVE migration
(transforming a v1 state into a v2 state) — impossible to write correctly before
v2 exists, but MADE POSSIBLE by the contract fixed here. The router-per-block
placement and soft/hard regime + breaker fallback are later slices (B, C).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

# Path identifiers — the control floor is always path 0.
PATH_CONTROL = 0
PATH_HOPFIELD = 1
PATH_DELTA = 2
PATH_NAMES = {PATH_CONTROL: "control", PATH_HOPFIELD: "hopfield", PATH_DELTA: "delta"}

CURRENT_ROUTER_VERSION = 1


@dataclass
class RouteDecision:
    """The stable output contract of any Router version.

    path      : the chosen path id (PATH_CONTROL always available as the floor)
    confidence: [0,1] — how sure the router is (drives soft/hard regime in Slice C)
    weights   : optional soft mixture over paths (None = hard choice of `path`)
    """
    path: int
    confidence: float = 1.0
    weights: Sequence[float] | None = None

    def is_hard(self) -> bool:
        return self.weights is None


class Router(ABC):
    """The versioned, STABLE contract. Every future RouterVN implements exactly
    this signature — versions EXTEND, they never break the interface. This is the
    retro-compatibility promise, fixed at v1 so it precedes (not follows) the code.
    """

    version: int = 0  # each concrete version sets this

    @abstractmethod
    def route(self, *, span_ctx, path_scores: Sequence[float] | None = None,
              state: dict | None = None) -> RouteDecision:
        """Decide which path to use for a span.

        span_ctx    : opaque context handle for the span (features the router may use)
        path_scores : optional known per-path scores (the oracle uses these; a
                      learned router predicts them from span_ctx instead)
        state       : optional persistent router state (version-tagged; see below)
        Returns a RouteDecision honouring the RouteDecision contract.
        """
        raise NotImplementedError

    # -- version-tagged serialisation (the retro-compat machinery, one branch) --
    def dump_state(self, state: dict | None = None) -> dict:
        """Serialise router state WITH its version tag, so any future loader can
        dispatch on it. v1 state is trivial; the envelope carries the tag."""
        return {"router_version": self.version, "state": state or {}}

    @staticmethod
    def load_state(blob: dict) -> dict:
        """Version dispatch — currently one branch (v1). Adding v2 is one more
        `elif`, never a change to this branch. Retro-compat by construction.
        Active migration (v1→v2 transform) is deferred until a v2 exists — the
        contract here makes it POSSIBLE, it does not implement it."""
        v = blob.get("router_version")
        if v == 1:
            return blob.get("state", {})
        # future: elif v == 2: return _migrate_or_read_v2(blob)
        raise ValueError(f"unknown router_version {v!r} — no reader registered")


class RouterV1(Router):
    """v1 = the upper-envelope ORACLE.

    Given the known per-path scores for a span, pick the best path *that beats the
    control floor*; if none beats it, choose the control. This is not a learned
    router — it computes the CEILING (the ablation showed +21% vs control, +6%
    over the best single path). Slice B replaces the oracle with a learned gate
    that PREDICTS path_scores from span_ctx; it will implement this same interface.
    """

    version = 1

    def __init__(self, floor_margin: float = 0.0):
        # a memory must beat the control by at least floor_margin to be chosen
        self.floor_margin = floor_margin

    def route(self, *, span_ctx=None, path_scores: Sequence[float] | None = None,
              state: dict | None = None) -> RouteDecision:
        if not path_scores:
            # no information → fall to the guaranteed floor (never degrade)
            return RouteDecision(path=PATH_CONTROL, confidence=1.0)
        scores = list(path_scores)
        control = scores[PATH_CONTROL]
        # candidate memories that clear the floor by the margin
        best_path, best_score = PATH_CONTROL, control
        for p in range(1, len(scores)):
            if scores[p] > control + self.floor_margin and scores[p] > best_score:
                best_path, best_score = p, scores[p]
        # confidence = normalised gap to the runner-up (how decisive the choice is)
        ordered = sorted(scores, reverse=True)
        gap = (ordered[0] - ordered[1]) if len(ordered) > 1 else 1.0
        conf = float(min(1.0, max(0.0, gap / (abs(ordered[0]) + 1e-6))))
        return RouteDecision(path=best_path, confidence=conf)


def make_router(version: int = CURRENT_ROUTER_VERSION, **kw) -> Router:
    """Factory — version dispatch for construction (one branch today)."""
    if version == 1:
        return RouterV1(**kw)
    raise ValueError(f"no Router implementation for version {version}")


def upper_envelope_auc(per_tier_by_path: dict[str, list[float]]) -> dict:
    """Compute the RouterV1-oracle result over a set of tiers: per tier, the
    router picks max(control, hopfield, delta); the AUC of that choice is the
    ceiling. Returns the envelope AUC and the routed path per tier."""
    control = per_tier_by_path["control"]
    hop = per_tier_by_path["hopfield"]
    delta = per_tier_by_path["delta"]
    n = len(control)
    router = RouterV1()
    routed_paths, routed_scores = [], []
    for i in range(n):
        d = router.route(path_scores=[control[i], hop[i], delta[i]])
        routed_paths.append(PATH_NAMES[d.path])
        routed_scores.append([control[i], hop[i], delta[i]][d.path])
    env_auc = sum(routed_scores) / n
    ctrl_auc = sum(control) / n
    best_single = max(sum(hop) / n, sum(delta) / n, ctrl_auc)
    return {
        "envelope_auc": round(env_auc, 4),
        "vs_control": round(env_auc - ctrl_auc, 4),
        "vs_best_single_path": round(env_auc - best_single, 4),
        "routed_paths": routed_paths,
    }
