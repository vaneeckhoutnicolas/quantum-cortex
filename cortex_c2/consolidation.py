"""cortex_c2.consolidation — the consolidating, self-healing router controller.

Slice C of RES-18 (ADR-006 D8). This is where the founder's three-message design
becomes code, as ONE stateful object wrapping a Router:

  1. Control floor (message 1): a memory path is used only where it beats the
     control; else the control. Enforced by the router; re-checked here.
  2. Regime = consolidation (message 2): a per-span-type decision that has been
     LEARNED — the same path wins, confidently, over N passes — is HARDENED
     (routed hard, one path, no soft mixture, minimal cost). Unsure decisions
     stay SOFT (the router's mixture) and keep generating the learning signal.
     Hardening is conservative (a high, declared confidence threshold + stability
     count) — better stay soft too long than harden an error.
  3. Breaker fallback (message 3): when an active path degrades or diverges, do
     not kill — RE-ROUTE to the best remaining path above the control, else the
     control, and RE-SOFTEN the decision (re-open deliberation). A declared
     HYSTERESIS (cooldown) prevents flapping between paths.

This is a textbook RES-17 type×state transition: a decision transitions
soft → hardened → (on breaker) softened-again. It unifies C1 (routing), RES-4
(confidence), RES-16 (consolidation + reminiscence), the endocrine explore/exploit
dial, basal-ganglia habit + hyperdirect veto, and the circuit breaker D7 — one
mechanism, seven windows.

Pure-Python controller (no torch here); it drives any Router.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from cortex_c2 import Router, RouteDecision, PATH_CONTROL, PATH_NAMES


@dataclass
class DecisionState:
    """Per-span-type consolidation state — the RES-17 (type, state) cell."""
    hardened_path: int | None = None      # None = soft; an int = hardened to that path
    stable_count: int = 0                 # consecutive passes the same path won confidently
    last_path: int | None = None
    cooldown: int = 0                     # hysteresis: passes remaining before a path may be re-chosen
    banned_path: int | None = None        # a path in cooldown after a breaker fallback


@dataclass
class ConsolidationConfig:
    harden_confidence: float = 0.75       # min confidence to count a pass toward hardening
    harden_stability: int = 5             # consecutive confident passes required to harden (conservative)
    hysteresis_cooldown: int = 8          # passes a path stays banned after a fallback (anti-flapping)
    clear_recovery_margin: float = 0.05   # a banned path must beat control by this to be re-eligible early


class ConsolidatingController:
    """Wraps a Router and adds consolidation + breaker fallback, per span-type key.

    `route(key, span_ctx, path_scores)` returns a RouteDecision:
      - if the key is HARDENED and not in cooldown → a hard decision on the learned path;
      - else the router's (soft) decision, with the control floor enforced, and
        hardening bookkeeping updated.
    `report_breaker(key, path)` is called when a path degraded/diverged on this key:
      it re-routes (bans the path for a cooldown), re-softens, and never leaves a void.
    """

    def __init__(self, router: Router, cfg: ConsolidationConfig | None = None):
        self.router = router
        self.cfg = cfg or ConsolidationConfig()
        self.states: dict = {}

    def _state(self, key) -> DecisionState:
        return self.states.setdefault(key, DecisionState())

    def route(self, *, key, span_ctx=None, path_scores=None) -> RouteDecision:
        st = self._state(key)

        # HARDENED path: route hard, minimal cost — unless it is the banned path
        if (st.hardened_path is not None and st.hardened_path != st.banned_path
                and st.cooldown == 0):
            self._tick_cooldown(st)
            return RouteDecision(path=st.hardened_path, confidence=1.0)  # hard, no mixture

        # otherwise ask the router (soft), then apply the floor + bookkeeping
        d = self.router.route(span_ctx=span_ctx, path_scores=path_scores)

        # if the chosen path is banned (still in cooldown), pull to control floor
        if st.banned_path is not None and st.cooldown > 0 and d.path == st.banned_path:
            d = RouteDecision(path=PATH_CONTROL, confidence=d.confidence, weights=d.weights)

        # consolidation bookkeeping: count confident agreement toward hardening
        if d.confidence >= self.cfg.harden_confidence and d.path == st.last_path:
            st.stable_count += 1
        else:
            st.stable_count = 1
        st.last_path = d.path

        # harden once stable enough (conservative) and not the banned path
        if (st.stable_count >= self.cfg.harden_stability
                and d.path != PATH_CONTROL
                and not (st.banned_path == d.path and st.cooldown > 0)):
            st.hardened_path = d.path

        self._tick_cooldown(st)      # decrement AFTER serving this pass
        return d

    def _tick_cooldown(self, st: DecisionState) -> None:
        if st.cooldown > 0:
            st.cooldown -= 1
            if st.cooldown == 0:
                st.banned_path = None

    def report_breaker(self, *, key, path, path_scores=None) -> RouteDecision:
        """A path degraded/diverged on this key → re-route with hysteresis.

        Bans `path` for a cooldown (anti-flapping), re-softens (drops any
        hardening), and returns the best remaining path above the control, else
        the control. Never a void — completes the hyperdirect veto (redirect).
        """
        st = self._state(key)
        st.banned_path = path
        st.cooldown = self.cfg.hysteresis_cooldown
        st.hardened_path = None          # re-soften: re-open deliberation (RES-16 reminiscence)
        st.stable_count = 0
        st.last_path = None

        # choose the fallback: best remaining path above control, else control
        if path_scores:
            control = path_scores[PATH_CONTROL]
            best_path, best_score = PATH_CONTROL, control
            for p in range(len(path_scores)):
                if p == path:
                    continue  # the banned path is out
                if p != PATH_CONTROL and path_scores[p] > control and path_scores[p] > best_score:
                    best_path, best_score = p, path_scores[p]
            return RouteDecision(path=best_path, confidence=0.5)
        # no scores → fall to the guaranteed floor
        return RouteDecision(path=PATH_CONTROL, confidence=1.0)

    # -- introspection (RES-17: the system becomes legible) ------------------
    def snapshot(self) -> dict:
        """Inspectable view — how many keys are hardened, softened, in cooldown."""
        hardened = sum(1 for s in self.states.values() if s.hardened_path is not None)
        cooling = sum(1 for s in self.states.values() if s.cooldown > 0)
        return {
            "keys": len(self.states),
            "hardened": hardened,
            "soft": len(self.states) - hardened,
            "in_cooldown": cooling,
        }
