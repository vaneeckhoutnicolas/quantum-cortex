"""Ingest of the confirmation artefact: must grave correctly for BOTH outcomes,
never inventing a number, using only the status vocabulary."""
import json, shutil, tempfile
from pathlib import Path

import cortex_eval.ingest_confirmation as ing


def _fake(seeds, d_sig, h_sig):
    n = len(seeds)
    return {
        "regime": {"seeds": seeds, "steps": 1500, "tiers": [[16,128],[8,128],[16,256],[8,256]]},
        "c2_ablation": {
            "stats": {"none": {"mean": 0.082}, "hopfield": {"mean": 0.100}, "delta": {"mean": 0.098}},
            "paired_tests": {
                "hopfield_vs_control": {"mean_diff": 0.018, "t": 3.4 if h_sig else 1.1, "df": n-1, "t_crit_95": 2.776, "significant_95": h_sig},
                "delta_vs_control":    {"mean_diff": 0.016, "t": 3.9 if d_sig else 1.3, "df": n-1, "t_crit_95": 2.776, "significant_95": d_sig},
                "delta_vs_hopfield":   {"mean_diff": -0.002, "t": -0.3, "df": n-1, "t_crit_95": 2.776, "significant_95": False}},
            "per_seed_auc": {"none": [0.08]*n, "hopfield": [0.10]*n, "delta": [0.098]*n},
            "claims": []},
        "gate_readout": {"delta_beats_control_significant": d_sig, "hopfield_beats_control_significant": h_sig},
        "recurrent_ladder": {"L4_vs_L3": {"mean_diff": -0.01, "significant_95": False}},
        "hybrid_vs_pure": {},
    }


def _setup(tmp, art):
    (tmp/"metrics"/"mqar").mkdir(parents=True)
    (tmp/"docs").mkdir()
    (tmp/"metrics"/"mqar"/"LATEST-confirmation.json").write_text(json.dumps(art))
    shutil.copy(ing.RESULTS, tmp/"docs"/"RESULTS.md")
    shutil.copy(ing.WP, tmp/"docs"/"WHITEPAPER.md")
    ing.RESULTS = tmp/"docs"/"RESULTS.md"; ing.WP = tmp/"docs"/"WHITEPAPER.md"


def test_passing_outcome_is_graved_as_gated():
    tmp = Path(tempfile.mkdtemp()); _setup(tmp, _fake([1,2,3,4,5], d_sig=True, h_sig=True))
    a = ing.read_artefact(tmp/"metrics"/"mqar"/"LATEST-confirmation.json")
    assert a["rows"]["delta_vs_control"]["status"] == "gated"
    r = ing.update_results(a); w = ing.update_whitepaper(a)
    assert "**gated**" in r and "| 10b |" in r
    assert "**2** comparison(s) have passed" in r
    assert "**passes**" in w and "now pass" in w


def test_failing_outcome_is_graved_as_held_not_claimed():
    tmp = Path(tempfile.mkdtemp()); _setup(tmp, _fake([1,2,3,4,5], d_sig=False, h_sig=False))
    a = ing.read_artefact(tmp/"metrics"/"mqar"/"LATEST-confirmation.json")
    assert a["rows"]["delta_vs_control"]["status"] == "held"
    r = ing.update_results(a); w = ing.update_whitepaper(a)
    assert "**0** comparison(s) have passed" in r
    assert "does not pass" in w and "not claimed" in w


def test_three_seeds_significant_still_gated_but_two_never():
    # gate rule 1: >= 3 seeds
    tmp = Path(tempfile.mkdtemp()); _setup(tmp, _fake([1,2], d_sig=True, h_sig=True))
    a = ing.read_artefact(tmp/"metrics"/"mqar"/"LATEST-confirmation.json")
    assert a["rows"]["delta_vs_control"]["status"] == "held"   # 2 seeds can never claim
