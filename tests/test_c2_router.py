"""C2 multi-path router (RES-18, ADR-006 D8), Slice A.

Tests the foundation: the versioned contract (the guard any RouterVN must pass),
the control-floor guarantee, and that RouterV1's upper-envelope oracle reaches
the ceiling measured by the ablation (+21% vs control, +6% over the best path).
"""
import numpy as np

from cortex_c2 import (Router, RouterV1, RouteDecision, make_router,
                       upper_envelope_auc, PATH_CONTROL, PATH_HOPFIELD, PATH_DELTA,
                       CURRENT_ROUTER_VERSION)


# --- the contract guard: any RouterVN must honour the stable interface -------
def test_routerv1_honours_the_contract():
    r = make_router(CURRENT_ROUTER_VERSION)
    assert isinstance(r, Router)
    assert r.version == 1
    d = r.route(path_scores=[0.1, 0.3, 0.2])
    assert isinstance(d, RouteDecision)
    assert 0.0 <= d.confidence <= 1.0
    assert d.path in (PATH_CONTROL, PATH_HOPFIELD, PATH_DELTA)


def test_version_tagged_serialisation_roundtrips():
    r = RouterV1()
    blob = r.dump_state({"foo": 1})
    assert blob["router_version"] == 1
    # the version dispatch reads v1; adding v2 is one more branch, never a change
    assert Router.load_state(blob) == {"foo": 1}


def test_unknown_version_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        Router.load_state({"router_version": 999, "state": {}})


# --- the control-floor guarantee (RES-18): never degrade ---------------------
def test_floor_no_memory_beats_control_chooses_control():
    r = RouterV1()
    # both memories below the control → must choose the control (the floor)
    d = r.route(path_scores=[0.30, 0.10, 0.20])
    assert d.path == PATH_CONTROL, "if no memory beats control, route to the floor"


def test_floor_no_information_chooses_control():
    r = RouterV1()
    d = r.route(path_scores=None)  # no scores → never degrade
    assert d.path == PATH_CONTROL


def test_picks_the_winning_memory_when_it_beats_control():
    r = RouterV1()
    assert r.route(path_scores=[0.10, 0.30, 0.20]).path == PATH_HOPFIELD
    assert r.route(path_scores=[0.10, 0.15, 0.40]).path == PATH_DELTA


def test_floor_margin_requires_clear_win():
    r = RouterV1(floor_margin=0.05)
    # hopfield only 0.02 above control (< margin) → stays on the floor
    assert r.route(path_scores=[0.30, 0.32, 0.20]).path == PATH_CONTROL
    # hopfield 0.10 above control (> margin) → chosen
    assert r.route(path_scores=[0.30, 0.40, 0.20]).path == PATH_HOPFIELD


# --- the headline: the oracle reaches the measured ceiling -------------------
def test_upper_envelope_matches_ablation_ceiling():
    # the real per-tier curves from the final ablation (2026-09-07)
    control  = [0.230,0.115,0.024,0.010,0.211,0.121,0.049,0.012,0.238,0.188,0.065,0.009]
    hopfield = [0.234,0.139,0.060,0.024,0.285,0.156,0.062,0.011,0.258,0.150,0.059,0.010]
    delta    = [0.234,0.172,0.057,0.007,0.234,0.072,0.069,0.012,0.242,0.148,0.046,0.005]
    env = upper_envelope_auc({"control": control, "hopfield": hopfield, "delta": delta})
    # the ablation reported: envelope +21% vs control, +6% over the best single path
    assert env["vs_control"] > 0.02, env          # +0.0218 measured
    assert env["vs_best_single_path"] > 0.005, env  # +0.0071 measured
    # the routed paths must include all three (each has a domain: hop/delta/control)
    routed = set(env["routed_paths"])
    assert routed == {"control", "hopfield", "delta"}, \
        "the router must use all three paths — the control floor is used where memory hurts"
