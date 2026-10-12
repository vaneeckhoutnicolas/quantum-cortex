"""The record tests of the bucket probe's rows (ADR-008, amendment of 2026-10-12, evening).

Row 44: the first run on the seed 2024 journal (2026-10-12, 21:48) stopped on its precondition, as declared.
Its failure file is a file of the record: this test reads it and recomputes what row 44 states, from the file
alone and from row 38's file: six of the seven scalars reproduced, the seventh (the maximum of the never
planted scores) off by more than the declared tolerance; C0 alone retained; the journal unchanged; the stop
written. The attribution (the questions encoded in batches of 200 and 50 where row 38 encoded one at a time)
is pinned on the corrected code by `tests/test_c2b_bucket_probe.py` (the file's cues equal the single question
encoding digit for digit)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from cortex_c2b import bucket_probe as bp

ROOT = Path(__file__).resolve().parent.parent
FAILED = ROOT / "metrics/mqar/bucket-probe-4cfd6f6e1406-FAILED-precondition.json"
ROW38 = ROOT / "metrics/mqar/retrieval-probe-4cfd6f6e1406.json"


@pytest.fixture(scope="module")
def failed():
    return json.loads(FAILED.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def row38():
    return json.loads(ROW38.read_text(encoding="utf-8"))


def test_row_44_the_first_run_stopped_on_its_precondition_and_read_no_configuration(failed):
    assert failed["checkpoint"]["run_id"] == bp.ROW38_RUN_ID == "4cfd6f6e1406"
    assert failed["checkpoint"]["config_hash"] == "d0c0d93ba24555a8"
    assert failed["precondition_row38"]["applies"] is True and failed["precondition_row38"]["ok"] is False
    assert failed["precondition_row38"]["tolerance"] == bp.ROW38_TOLERANCE == 1e-6
    assert len(failed["stopped"]) == 1 and failed["stopped"][0].startswith("the control does not reproduce row 38 on its journal")
    assert set(failed["rows"]) == set(failed["aggregates"]) == set(failed["entry_signatures"]) == set(failed["never_planted"]) == {"C0"}
    assert failed["configurations_not_retained"] == [c["name"] for c in bp.CONFIGS if c["name"] != "C0"] + ["C6"]
    assert failed["control_matches_live_index"] is True and all(failed["subset_relations"].values())


def test_row_44_six_of_the_seven_scalars_reproduce_and_the_seventh_is_off_by_more_than_the_tolerance(failed, row38):
    c0 = failed["aggregates"]["C0"]
    ok, diffs = bp.check_row38(c0, row38, k=failed["k"], n_facts=failed["n_facts"], n_negctrl=failed["n_negctrl"])
    assert not ok and len(diffs) == 1 and diffs[0].startswith("negctrl_best_max:")
    assert diffs == failed["precondition_row38"]["differences"]
    expected = row38["rankings"]["cosine"]
    assert c0["candidate_recall"] == expected["own_in_candidates"] == 0.935
    assert c0["top_k_recall"] == expected["own_in_top_k"] == 0.935
    assert c0["n_in_candidates"] == expected["own_rank"]["n"] == 187
    assert c0["n_margins"] == expected["own_minus_best_other"]["n"] == 183
    assert c0["n_negctrl_scored"] == expected["negctrl_best_score"]["n"] == 49
    assert math.isclose(c0["margin_mean"], expected["own_minus_best_other"]["mean"], rel_tol=0.0, abs_tol=bp.ROW38_TOLERANCE)
    assert c0["negctrl_best_max"] == 0.5618768930435181 and expected["negctrl_best_score"]["max"] == 0.5618757009506226
    gap = c0["negctrl_best_max"] - expected["negctrl_best_score"]["max"]
    assert gap == 5 * 2.0 ** -22 and gap / 2.0 ** -24 == 20.0 and gap > bp.ROW38_TOLERANCE   # twenty float32 ulps at 0.56, both values float32 exact
    assert abs(c0["margin_mean"] - expected["own_minus_best_other"]["mean"]) < 2e-7            # 1.1e-7: the mean of 183 absorbs what one maximum shows
    scored = [r for r in failed["never_planted"]["C0"] if r["best_score"] is not None]
    assert max(scored, key=lambda r: r["best_score"])["j"] == 39 and len(scored) == 49
    assert failed["k"] == row38["k"] == 4 and failed["n_facts"] == 200 and failed["n_negctrl"] == 50


def test_row_44_the_counts_recompute_from_the_rows_and_the_journal_was_not_touched(failed):
    c0 = failed["aggregates"]["C0"]
    rec = bp.aggregate(failed["rows"]["C0"], failed["never_planted"]["C0"], failed["rows"]["C0"], n_entries=failed["journal"]["entries"], k=failed["k"])
    for key, v in rec.items():
        assert c0[key] == pytest.approx(v) if isinstance(v, float) else c0[key] == v, key
    assert failed["journal"]["entries"] == 200 and failed["generator"]["planted_found"] == 200 and failed["journal"]["states"] == {"live": 200}
    assert failed["journal"]["unchanged"] is True and failed["journal"]["opened"].startswith("a copy")
    assert failed["journal"]["sha256_before"] == failed["journal"]["sha256_after"]
    assert len(failed["rows"]["C0"]) == 200 and len(failed["never_planted"]["C0"]) == 50 and len(failed["questions"]) == 200
    assert all(q["planted"] for q in failed["questions"])
    # the one never planted query with an empty candidate set that row 38's count of 49 implies
    assert [r["j"] for r in failed["never_planted"]["C0"] if r["n_candidates"] == 0] == [45]
    # the four hits with a singleton candidate set that row 38's 187 ranks and 183 margins imply
    assert [r["i"] for r in failed["rows"]["C0"] if r["in_candidates"] and r["n_candidates"] == 1] == [4, 73, 129, 140]
    assert c0["misses"] == [7, 9, 41, 46, 52, 54, 72, 118, 124, 134, 157, 164, 182]


def test_row_44_the_file_names_the_code_the_machine_and_the_time_it_ran(failed):
    """What row 44 states about the run comes from the file: the commit of the first build, a clean tree, the
    versions, the thread count, the processor as the encoder's device, the time."""
    assert failed["code_commit"] == "fa8f2bd" and failed["code_tree_clean"] is True
    assert failed["versions"] == {"python": "3.12.10", "numpy": "2.5.3", "torch": "2.6.0+cu124"}
    assert failed["device"]["device"] == "cpu" and failed["device"]["threads"] == 8
    assert failed["timestamp_utc"] == "2026-10-12T19:46:03+00:00"
    assert failed["checkpoint"]["config_sha256"] == "7315713b45ed5881" and failed["journal"]["sha256_before"] == "697be97c15d2edf3"
    assert failed["index_seed"] == 0 and failed["facts_seed"] == 0
