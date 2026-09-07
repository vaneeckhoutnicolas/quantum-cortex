"""Router ablation (RES-18 Slice D): the learned router approaches the oracle.

Marked slow (trains a router). The closing test of N2: the learned router
(predicts from context, never reads truth) must capture most of the oracle's
lift over the best single path — proving the multi-path design pays off in
practice, not just in oracle.
"""
import pytest

from cortex_c2.router_ablation import run_router_ablation


@pytest.mark.slow
def test_learned_router_approaches_oracle_ceiling():
    r = run_router_ablation(train_n=3000, test_n=800, steps=500, seed=1337)
    auc = r["auc"]
    # sanity: the ordering must hold — control < best single < learned <= oracle
    assert auc["control_only"] < auc["best_single_memory"], auc
    assert auc["best_single_memory"] < auc["learned_router"], \
        "the router must beat the best single path (else multi-path is pointless)"
    assert auc["learned_router"] <= auc["oracle_ceiling"] + 1e-6, \
        "the learned router cannot exceed the oracle that reads the truth"
    # the headline: capture a large fraction of the oracle's lift
    frac = r["lift_over_best_single"]["learned_fraction_of_oracle_ceiling"]
    assert frac > 0.6, f"learned router should capture most of the ceiling (got {frac:.2f})"
