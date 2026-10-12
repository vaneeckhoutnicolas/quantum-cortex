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


# ---- the rerun of 2026-10-12, 21:07 UTC, declared in ADR-008's build precision (13) before it ran ----------------
RUN2 = ROOT / "metrics/mqar/bucket-probe-4cfd6f6e1406-run2.json"


@pytest.fixture(scope="module")
def run2():
    return json.loads(RUN2.read_text(encoding="utf-8"))


def test_the_rerun_reproduces_row_38_bit_for_bit_and_names_the_code_the_device_and_the_time(run2, row38):
    """Precision (13): a seventh scalar equal bit for bit says the encoding path accounts for the whole difference."""
    pc = run2["precondition_row38"]
    assert pc["applies"] is True and pc["ok"] is True and pc["differences"] == [] and pc["tolerance"] == bp.ROW38_TOLERANCE
    assert set(pc["values"]) == {"candidate_recall", "top_k_recall", "n_in_candidates", "n_margins", "margin_mean", "n_negctrl_scored", "negctrl_best_max"}
    assert all(v["difference"] == 0 for v in pc["values"].values())                   # the seven scalars, bit for bit
    c0 = run2["aggregates"]["C0"]
    assert c0["margin_mean"] == row38["rankings"]["cosine"]["own_minus_best_other"]["mean"] == 0.7060376622523575
    assert c0["negctrl_best_max"] == row38["rankings"]["cosine"]["negctrl_best_score"]["max"] == 0.5618757009506226
    ok, diffs = bp.check_row38(c0, row38, k=run2["k"], n_facts=run2["n_facts"], n_negctrl=run2["n_negctrl"])
    assert ok and diffs == []
    assert run2["device"]["device"] == "cuda" and run2["device"]["cuda_available"] is True and run2["device"]["gpu"] == "NVIDIA GeForce GTX 1660 Ti"
    assert run2["device"]["threads"] == 8 and run2["versions"] == {"python": "3.12.10", "numpy": "2.5.3", "torch": "2.6.0+cu124"}
    assert run2["code_commit"] == "dd39b02" and run2["code_tree_clean"] is True and run2["timestamp_utc"] == "2026-10-12T21:07:33+00:00"
    assert run2["checkpoint"]["run_id"] == "4cfd6f6e1406" and run2["journal"]["sha256_before"] == run2["journal"]["sha256_after"] == "697be97c15d2edf3"
    assert run2["journal"]["unchanged"] is True and run2["generator"]["planted_found"] == 200 and "stopped" not in run2
    assert set(run2["rows"]) == {c["name"] for c in bp.CONFIGS} | {"C6"}


def test_the_rerun_against_the_failure_file_the_one_variable_is_the_encoding_path(run2, failed):
    """The same machine, versions and thread count, the code one commit apart (fa8f2bd against dd39b02): the cues
    move on nearly every coordinate (the device and the batch shape), every C0 signature, candidate count and miss
    stays, and the rerun's scalars land on row 38's."""
    import numpy as np
    assert failed["versions"] == run2["versions"] and failed["device"]["threads"] == run2["device"]["threads"] == 8
    assert failed["device"]["device"] == "cpu" and run2["device"]["device"] == "cuda"
    assert failed["code_commit"] == "fa8f2bd" and run2["code_commit"] == "dd39b02"
    qa = np.array(failed["query_cues"]["questions"], dtype=np.float32); qb = np.array(run2["query_cues"]["questions"], dtype=np.float32)
    na = np.array(failed["query_cues"]["never_planted"], dtype=np.float32); nb = np.array(run2["query_cues"]["never_planted"], dtype=np.float32)
    assert qa.shape == qb.shape == (200, 64) and na.shape == nb.shape == (50, 64)
    dq, dn = np.abs(qa - qb), np.abs(na - nb)
    assert 2.3e-6 <= float(dq.max()) < 2.4e-6 and 2.3e-6 <= float(dn.max()) < 2.4e-6         # two orders above the batch shape alone (2.5e-7)
    assert (dq > 0).any(axis=1).all() and (dn > 0).any(axis=1).all()                            # every query moved
    assert [r["signatures_query"] for r in failed["rows"]["C0"]] == [r["signatures_query"] for r in run2["rows"]["C0"]]
    assert [r["n_candidates"] for r in failed["rows"]["C0"]] == [r["n_candidates"] for r in run2["rows"]["C0"]]
    assert [r["in_candidates"] for r in failed["rows"]["C0"]] == [r["in_candidates"] for r in run2["rows"]["C0"]]
    assert failed["aggregates"]["C0"]["misses"] == run2["aggregates"]["C0"]["misses"]
    assert failed["entry_signatures"]["C0"] == run2["entry_signatures"]["C0"]
    assert [r["n_candidates"] for r in failed["never_planted"]["C0"]] == [r["n_candidates"] for r in run2["never_planted"]["C0"]]
    margins = [abs(x["margin"] - y["margin"]) for x, y in zip(failed["rows"]["C0"], run2["rows"]["C0"]) if x["margin"] is not None]
    best = [abs(x["best_score"] - y["best_score"]) for x, y in zip(failed["never_planted"]["C0"], run2["never_planted"]["C0"]) if x["best_score"] is not None]
    assert len(margins) == 183 and max(margins) < 2e-6 and len(best) == 49 and max(best) < 2.2e-6
    assert run2["never_planted"]["C0"][39]["best_score"] == 0.5618757009506226 and failed["never_planted"]["C0"][39]["best_score"] == 0.5618768930435181


# ---- row 45: the bucket probe measured on the three v9 journals (2026-10-12, 21:07 to 21:10 UTC) --------------------
FILES = {"1337": ROOT / "metrics/mqar/bucket-probe-8ceba5db8d0b.json",
         "2024": ROOT / "metrics/mqar/bucket-probe-4cfd6f6e1406-run2.json",
         "7": ROOT / "metrics/mqar/bucket-probe-2e1dc71913b1.json"}
RUN_IDS = {"1337": "8ceba5db8d0b", "2024": "4cfd6f6e1406", "7": "2e1dc71913b1"}
CONFIG_HASHES = {"1337": "36a3dd10bbff84eb", "2024": "d0c0d93ba24555a8", "7": "a727d1cb445afa92"}
JOURNAL_SHA = {"1337": "d0d56e5a72fe4cf3", "2024": "697be97c15d2edf3", "7": "0e0c1d34438c85c3"}      # rows 40 to 42's journals


@pytest.fixture(scope="module")
def three():
    return {s: json.loads(p.read_text(encoding="utf-8")) for s, p in FILES.items()}


def test_row_45_the_three_files_are_the_declared_runs_on_the_record_s_journals_untouched(three):
    for s, d in three.items():
        assert d["checkpoint"]["run_id"] == RUN_IDS[s] and d["checkpoint"]["config_hash"] == CONFIG_HASHES[s]
        assert d["journal"]["sha256_before"] == d["journal"]["sha256_after"] == JOURNAL_SHA[s] and d["journal"]["unchanged"] is True
        assert d["journal"]["entries"] == 200 and d["journal"]["states"] == {"live": 200} and d["journal"]["aliases"] == 0 and d["journal"]["duplicate_pointers"] == 0
        assert d["generator"]["planted_found"] == 200 and d["k"] == 4 and d["n_facts"] == 200 and d["n_negctrl"] == 50
        assert d["index_seed"] == 0 and d["facts_seed"] == 0 and d["budget"] == bp.BUDGET and d["margin"] == bp.MARGIN and d["cue_far"] == bp.CUE_FAR
        assert d["control_matches_live_index"] is True and all(d["subset_relations"].values()) and "stopped" not in d
        assert d["device"]["device"] == "cuda" and d["device"]["gpu"] == "NVIDIA GeForce GTX 1660 Ti" and d["device"]["threads"] == 8
        assert d["versions"] == {"python": "3.12.10", "numpy": "2.5.3", "torch": "2.6.0+cu124"} and d["code_commit"] == "dd39b02" and d["code_tree_clean"] is True
        assert set(d["rows"]) == {c["name"] for c in bp.CONFIGS} | {"C6"}
        assert d["configurations"]["C0"]["planes_sha256"] == "680896ab07072738" and d["configurations"]["C2b"]["planes_sha256"] == "e5703fddba43332f"
        assert d["precondition_row38"]["applies"] is (s == "2024")
    assert three["2024"]["precondition_row38"]["ok"] is True


def test_row_45_every_aggregate_recomputes_from_the_rows_on_the_three_journals(three):
    for s, d in three.items():
        for name, agg in d["aggregates"].items():
            rec = bp.aggregate(d["rows"][name], d["never_planted"][name], d["rows"]["C0"], n_entries=d["journal"]["entries"], k=d["k"])
            for key, v in rec.items():
                assert (agg[key] == pytest.approx(v, abs=1e-12)) if isinstance(v, float) else agg[key] == v, (s, name, key)
            expected_adm = bp.admissible(agg, d["aggregates"]["C0"]) if name not in ("C0", "C6") else (name == "C0")
            assert agg["admissible"] is expected_adm, (s, name)


def test_row_45_the_control_s_misses_are_the_index_s_and_the_exact_signature_loses_nothing(three):
    misses = {"1337": [0, 25, 26, 43, 58, 62, 65, 72, 77, 160, 165, 190, 199], "2024": [7, 9, 41, 46, 52, 54, 72, 118, 124, 134, 157, 164, 182],
              "7": [38, 44, 57, 63, 72, 106, 115, 148, 164, 172, 195]}
    recall = {"1337": 0.935, "2024": 0.935, "7": 0.945}
    for s, d in three.items():
        c0, c1 = d["aggregates"]["C0"], d["aggregates"]["C1"]
        assert c0["candidate_recall"] == c0["top_k_recall"] == recall[s] and c0["misses"] == misses[s]
        assert c0["miss_classes"] == {"index": len(misses[s]), "interference": 0, "cue": 0}
        assert c1["R"] == 0 and c1["L"] == 0 and c1["misses"] == misses[s] and c1["cost"]["mean"] < 0.4 * c0["cost"]["mean"]


def test_row_45_reading_a_c2b_and_c4_propose_every_own_line_and_the_protected_selection_takes_c2b(three):
    js = [three["1337"], three["2024"], three["7"]]
    r = bp.read_three(js)
    assert r["letter"] == "a" and r["candidates_at_1000"] == ["C2b", "C4"] and r["above_control_everywhere"] == ["C2a", "C2b", "C3b", "C4"]
    assert r["selected"] == "C2b" and r["rotations_failed"] == 1 and r["excluded_by_rotation"] == ["C2a"]
    assert r["rotations"] == [{"left_out": i, "chosen": "C2b", "holds_on_left_out": True} for i in range(3)]
    assert r["selected_label"] == ("selected among 10 configurations on the questions of 3 journals, 1 rotation(s) failed; "
                                   "confirmed on the first journal sealed after the adoption (the prospective holdout)")
    assert r["c1_loses_nothing"] is True and r["adopt_c1"] is False
    R = {"1337": 13, "2024": 13, "7": 11}
    for s, d in three.items():
        for n in ("C2b", "C4"):
            a = d["aggregates"][n]
            assert a["candidate_recall"] == 1.0 and a["top_k_recall"] == 1.0 and a["R"] == R[s] and a["L"] == 0 and a["misses"] == [] and a["admissible"] is True
        assert d["aggregates"]["C2b"]["n_margins"] == 200 - sum(1 for row in d["rows"]["C2b"] if row["n_candidates"] == 1)
    # the one miss that excluded C2a, and its cost advantage that the words refused to follow
    assert three["1337"]["aggregates"]["C2a"]["misses"] == [199] and three["1337"]["aggregates"]["C2a"]["candidate_recall"] == 0.995
    assert three["2024"]["aggregates"]["C2a"]["misses"] == [] and three["7"]["aggregates"]["C2a"]["misses"] == []
    p95 = {n: [d["aggregates"][n]["cost"]["p95"] for d in js] for n in ("C0", "C2a", "C2b", "C4")}
    assert p95["C0"] == [11.0, 10.0, 11.0] and p95["C2a"] == [7.0, 7.0, 6.0] and p95["C2b"] == [9.0, 10.0, 10.0] and p95["C4"] == [18.0, 17.0, 20.0]
    means = {n: [round(d["aggregates"][n]["cost"]["mean"], 2) for d in js] for n in ("C0", "C2b", "C4")}
    assert means["C0"] == [5.92, 5.92, 5.75] and means["C2b"] == [5.64, 5.83, 5.9] and means["C4"] == [11.95, 11.79, 12.21]


def test_row_45_the_fly_ranking_on_c0_s_candidates_puts_the_own_line_first_on_every_hit_as_the_cosine_does(three):
    """The four readings of 2026-09-19, decided from the own ranks under the two rankings: the second holds (the same
    top k, the misses are candidate misses). The novelty score is recorded and read nowhere."""
    hits = {"1337": 187, "2024": 187, "7": 189}
    for s, d in three.items():
        rows = [r for r in d["rows"]["C0"] if r["in_candidates"]]
        assert len(rows) == hits[s]
        assert all(r["rank"] == 1 and r["fly_rank_on_c0"] == 1 for r in rows)
        assert all(r["fly_overlap_own"] == r["fly_top_overlap"] for r in rows)
        assert min(r["fly_overlap_own"] for r in rows) > 0.66 and 0.82 < sum(r["fly_overlap_own"] for r in rows) / len(rows) < 0.84
        assert all(r["fly_rank_on_c0"] is None for r in d["rows"]["C0"] if not r["in_candidates"])
        nov = [r["novelty"] for r in d["never_planted"]["C0"]]
        assert all(0.0 <= x <= 1.0 for x in nov) and sum(1 for x in nov if x > 0.0) == (1 if s == "2024" else 0)   # one of 150 queries keeps one index no stored tag activates
        assert max(nov) == (1 / 51 if s == "2024" else 0.0)


def test_row_45_the_reserve_the_never_planted_best_score_rises_under_the_wider_index(three):
    best = {"1337": (0.6859, 0.6900), "2024": (0.5619, 0.7481), "7": (0.6006, 0.6994)}
    for s, d in three.items():
        c0, c2b = d["aggregates"]["C0"], d["aggregates"]["C2b"]
        assert round(c0["negctrl_best_max"], 4) == best[s][0] and round(c2b["negctrl_best_max"], 4) == best[s][1]
        assert c2b["never_planted_cost"]["p95"] <= bp.BUDGET["p95_at_most"] and c2b["never_planted_empty_share"] <= c0["never_planted_empty_share"]


def test_rows_44_and_45_the_provenance_names_the_four_files_by_their_lf_hashes():
    import hashlib
    prov = json.loads((ROOT / "metrics/mqar/PROVENANCE-bucket-probe.json").read_text(encoding="utf-8"))
    assert prov["rows"] == ["44", "45"] and prov["reading"]["letter"] == "a" and prov["reading"]["selected"] == "C2b" and prov["reading"]["rotations_failed"] == 1
    assert set(prov["files"]) == {"bucket-probe-4cfd6f6e1406-FAILED-precondition.json", "bucket-probe-4cfd6f6e1406-run2.json",
                                  "bucket-probe-8ceba5db8d0b.json", "bucket-probe-2e1dc71913b1.json"}
    for name, e in prov["files"].items():
        raw = (ROOT / "metrics/mqar" / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(raw).hexdigest()[:16] == e["sha256_lf"], name
        d = json.loads(raw)
        assert d["checkpoint"]["run_id"] == e["run_id"] and d["checkpoint"]["config_hash"] == e["config_hash"] and d["timestamp_utc"] == e["timestamp_utc"]
        assert d["code_commit"] == e["code_commit"] and d["journal"]["sha256_before"] == e["journal_sha256"]
    assert prov["files"]["bucket-probe-4cfd6f6e1406-FAILED-precondition.json"]["code_commit"] == "fa8f2bd"
    assert all(e["code_commit"] == "dd39b02" for n, e in prov["files"].items() if "FAILED" not in n)
