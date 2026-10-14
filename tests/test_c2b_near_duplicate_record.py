"""Row 48, the near duplicate probe measured (ADR-007, amendment of 2026-10-13, evening; run by the founder 2026-10-14,
read cold the same day): everything recomputed from the sixteen retained files and the provenance, by the declared
functions. The letter is (d): both failures produced. These tests fail if a digit of the row moves."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
M = ROOT / "metrics" / "mqar"

from cortex_c2b import near_duplicate_reading as R           # noqa: E402
from cortex_c2b import near_duplicate_probe as P             # noqa: E402
from cortex_c2b.adoption_reading import binomial_two_sided_p  # noqa: E402
from cortex_c2b.hm_protocol import generate_facts             # noqa: E402

CODE_COMMIT = "7b8c548"
CONFIG_HASH = {"8ceba5db8d0b": "36a3dd10bbff84eb", "4cfd6f6e1406": "d0c0d93ba24555a8", "2e1dc71913b1": "a727d1cb445afa92"}
THRESHOLD = {"8ceba5db8d0b": 0.5360028743743896, "4cfd6f6e1406": 0.6122402548789978, "2e1dc71913b1": 0.5907554626464844}
REFUSED = {"8ceba5db8d0b": {"E1": 200, "E2": 160}, "4cfd6f6e1406": {"E1": 198, "E2": 162}, "2e1dc71913b1": {"E1": 192, "E2": 185}}
# corrected 2026-10-14, 19:30 (Rev120): E2's admitted maxima per checkpoint and the refused pairs under 1e-3 (E1 and E2 together)
E2_MAX = {"8ceba5db8d0b": 0.6312, "4cfd6f6e1406": 0.3616, "2e1dc71913b1": 0.2218}
UNDER_1E3 = {"8ceba5db8d0b": 34, "4cfd6f6e1406": 25, "2e1dc71913b1": 37}
GAP = {}
OWN_CITED = {"8ceba5db8d0b": {"B0": 192, "E0d": 188, "E1": 192, "E2": 172}, "4cfd6f6e1406": {"B0": 190, "E0d": 189, "E1": 186, "E2": 170},
             "2e1dc71913b1": {"B0": 182, "E0d": 182, "E1": 183, "E2": 172}}
CITED_DUPLICATE_E2 = {"8ceba5db8d0b": 15, "4cfd6f6e1406": 19, "2e1dc71913b1": 6}
ORACLE_OWN_E2 = {"8ceba5db8d0b": 200, "4cfd6f6e1406": 200, "2e1dc71913b1": 199}


@pytest.fixture(scope="module")
def loaded():
    return R.load_from_dir(M)


@pytest.fixture(scope="module")
def reading(loaded):
    return R.read_near_duplicates(**loaded)


@pytest.fixture(scope="module")
def prov():
    return json.loads((M / "PROVENANCE-near-duplicates.json").read_text(encoding="utf-8"))


def test_the_sixteen_files_are_there_with_their_hashes_commit_and_checkpoints(prov, loaded):
    assert len(prov["files"]) == 16
    for name, meta in prov["files"].items():
        b = (M / name).read_bytes()
        assert b"\r\n" not in b, name
        assert hashlib.sha256(b).hexdigest()[:16] == meta["sha256_lf"], name
        d = json.loads(b)
        assert d["code_commit"] == CODE_COMMIT == meta["code_commit"], name
    for rid in R.RUN_IDS:
        assert loaded["cues_by_run"][rid]["checkpoint"]["config_hash"] == CONFIG_HASH[rid]
        for arm in R.ARMS:
            f = loaded["files_by_run"][rid][arm]
            assert f["checkpoint"]["run_id"] == rid and f["checkpoint"]["config_hash"] == CONFIG_HASH[rid]
            assert f["hm_mark_oracle"]["threshold"] == THRESHOLD[rid]                    # equal to the bit on the twelve files
            assert f["journal"]["unchanged"] is True and f["index"]["live"]["planes_sha256"] == "e5703fddba43332f"
            R.check_arm_file(f, arm)


def test_stage1_recomputes_its_hash_band_and_refusal_fractions(loaded):
    s1 = loaded["stage1"]
    assert s1["declared_constants"] is True and s1["second_process"]["agrees"] is True
    assert P.numbers_sha256(s1) == s1["numbers_sha256"] == s1["second_process"]["numbers_sha256"]
    assert s1["band"] == {"c_any": 0.95, "c_all": 0.995, "c_half": 0.98}
    fractions = {c: v["refusal_fraction"] for c, v in s1["per_c"].items()}
    assert fractions == {"0.8": 0.0, "0.85": 0.0, "0.9": 0.0, "0.95": 0.155, "0.97": 0.47, "0.98": 0.725, "0.99": 0.955, "0.995": 1.0, "0.999": 1.0}
    # the aggregates from the pairs
    for c, v in s1["per_c"].items():
        rows = [r for r in s1["pairs"] if str(r["c"]) == c]
        assert len(rows) == 200 == v["n"] and sum(1 for r in rows if not r["admitted"]) == v["refused"]
        for r in rows:
            assert abs(r["gate_code_cosine"] - (1.0 - r["surprise"])) < 1e-5 and r["admitted"] == (r["surprise"] >= P.ADMISSION_THRESHOLD)
    assert P.band(fractions_f := {float(c): f for c, f in fractions.items()}) == s1["band"]
    assert s1["geometries"]["legacy"]["planes_sha256"] == "680896ab07072738" and s1["geometries"]["adopted"]["planes_sha256"] == "e5703fddba43332f"


def test_stage2_preconditions_held_and_the_plans_recompute(loaded):
    for rid in R.RUN_IDS:
        cf = loaded["cues_by_run"][rid]
        pre = cf["precondition"]
        assert pre["order_exact"] and pre["within_tolerance"] and not pre["stopped"] and pre["ranks_compared"] == 200
        assert pre["max_abs_diff_cos_query_own"] < 1e-5 and pre["max_abs_diff_best_other_cosine"] < 5e-5 and pre["query_cues_max_abs_diff"] < 3e-6
        assert len(pre["per_question"]) == 200 and all(q["rank"] == 1 for q in pre["per_question"])
        d = cf["distributions"]
        assert d["E1"]["n"] == 1400 and d["E2"]["n"] == 200 and d["E0"]["n"] == 19900 and cf["plans"]["E2_dropped"] == []
        facts, _ = generate_facts(200, 0)
        assert cf["plans"]["E1"] == P.e1_plan(facts, 0)
        assert cf["plans"]["E2"] == {f.entity: P.bump_name(f.entity) for f in facts}
        assert d["E0"]["max"] < 0.79 and d["E1"]["median"] > 0.99 and d["E2"]["median"] > 0.99
        assert cf["row45_file"]["name"].startswith(f"bucket-probe-{rid}")


def test_f1_both_kinds_refused_with_the_partner_nearest_and_no_background_refusal(reading, loaded):
    f1 = reading["F1"]
    assert f1["produced"] and f1["complete"] and f1["defective"] == []
    for rid in R.RUN_IDS:
        for kind in R.KINDS:
            v = f1["per_run"][rid][kind]
            assert v["duplicates"] == 200 and v["duplicates_refused"] == REFUSED[rid][kind] == v["duplicates_refused_partner_nearest"]
            assert v["agrees_with_file"]
        assert f1["per_run"][rid]["E0d"]["e0d_written"] == 200 and f1["per_run"][rid]["E0d"]["e0d_refused"] == 0
        assert f1["per_run"][rid]["B0"]["originals_refused"] == 0
        # every refused near duplicate was refused by the surprise rule, with its partner admitted
        for kind in R.KINDS:
            writes = loaded["files_by_run"][rid][kind]["near_duplicates"]["planting"]["writes"]
            dups = [w for w in writes if w["kind"] == kind]
            assert all(w["partner"]["admitted"] for w in dups)
            assert all(w["reason"] == "redundant (below admission threshold)" for w in dups if not w["admitted"])
            assert all(w["surprise"] < P.ADMISSION_THRESHOLD for w in dups if not w["admitted"])


def test_f2_the_two_primary_tests_and_the_letter(reading):
    f2 = reading["F2"]
    assert f2["complete"] and f2["produced"]
    e1, e2 = f2["tests"]["E1"], f2["tests"]["E2"]
    assert (e1["b_test_only"], e1["c_control_only"], e1["n_pairs"]) == (25, 27, 400) and e1["p"] == pytest.approx(binomial_two_sided_p(25, 27))
    assert (e2["b_test_only"], e2["c_control_only"], e2["n_pairs"]) == (27, 56, 400) and e2["p"] == pytest.approx(binomial_two_sided_p(27, 56))
    assert e2["p"] < 0.002 and e1["p"] > 0.8
    assert f2["holm_rejected"] == {"E2": True, "E1": False} and f2["gains_named"] == []
    assert e2["direction"].startswith("fewer") and e1["direction"].startswith("fewer")
    assert e2["per_seed"]["4cfd6f6e1406"]["b_test_only"] == 11 and e2["per_seed"]["4cfd6f6e1406"]["c_control_only"] == 30
    assert e2["per_seed"]["2e1dc71913b1"]["b_test_only"] == 16 and e2["per_seed"]["2e1dc71913b1"]["c_control_only"] == 26
    assert e1["per_seed"]["4cfd6f6e1406"]["b_test_only"] == 9 and e1["per_seed"]["4cfd6f6e1406"]["c_control_only"] == 12
    assert e1["per_seed"]["2e1dc71913b1"]["b_test_only"] == 16 and e1["per_seed"]["2e1dc71913b1"]["c_control_only"] == 15
    rep = f2["reported_conceiving_seed"]
    assert (rep["8ceba5db8d0b:E2"]["b_test_only"], rep["8ceba5db8d0b:E2"]["c_control_only"]) == (8, 24)
    assert (rep["8ceba5db8d0b:E1"]["b_test_only"], rep["8ceba5db8d0b:E1"]["c_control_only"]) == (12, 8)
    assert reading["letter"] == "d" and reading["exits"] == [] and not reading["precondition_stopped"]


def test_the_attribution_of_the_discordant_pairs_and_the_blindness_of_the_rates(reading, loaded):
    e2 = reading["F2"]["tests"]["E2"]
    # the excess of c under E2 sits on the questions whose duplicate was in the window with both marks at or above the threshold
    assert e2["per_seed"]["4cfd6f6e1406"]["split"]["c"]["duplicate_in_window"] == 19 and e2["per_seed"]["4cfd6f6e1406"]["split"]["c"]["both_marks"] == 19
    assert e2["per_seed"]["2e1dc71913b1"]["split"]["c"]["duplicate_in_window"] == 7
    assert all(ps["split"]["c"]["own_not_admitted"] == 0 and ps["split"]["c"]["lost"] == 0 for k in reading["F2"]["tests"] for ps in reading["F2"]["tests"][k]["per_seed"].values())
    for rid in R.RUN_IDS:
        arms = loaded["files_by_run"][rid]
        for arm in R.ARMS:
            sc = R.side_counts(arms[arm])
            assert sc["agrees_with_file"] and sc["own_cited"] == OWN_CITED[rid][arm]
        sc2 = R.side_counts(arms["E2"])
        assert sc2["cited_is_duplicate"] == CITED_DUPLICATE_E2[rid] and sc2["oracle_cites_own"] == ORACLE_OWN_E2[rid]
        assert sc2["duplicate_in_window"] == sc2["both_marks_at_or_above_threshold"]
        assert sc2["duplicate_cited_given_both_present"] is not None and 0.35 <= sc2["duplicate_cited_given_both_present"] <= 0.5
        # the protocol's strict recall is blind to the wrong line under E2 (the duplicate carries the expected attribute): the own
        # citations fall by at least ten while strict recall moves by at most 0.02 against B0
        assert OWN_CITED[rid]["B0"] - OWN_CITED[rid]["E2"] >= 10
        assert abs(arms["E2"]["hm_recall_on"] - arms["B0"]["hm_recall_on"]) <= 0.025
        assert arms["E2"]["hm_invalid_citation_on"] == 0.0 and arms["E2"]["hm_negctrl_rate"] == 0.0


def test_the_controls_b0_identical_to_row_47_thresholds_reproduced_densities(reading, loaded):
    for rid in R.RUN_IDS:
        b0 = reading["b0_against_record"][rid]
        assert b0["identical_outcomes"] and b0["own_cited"]["b_test_only"] == 0 and b0["own_cited"]["c_control_only"] == 0
        assert b0["strict"]["b_adopted_only"] == 0 and b0["strict"]["c_control_only"] == 0
        for arm in R.ARMS:
            assert reading["thresholds_against_record"][f"{rid}:{arm}"]["label"] == "reproduced"
            assert reading["thresholds_against_record"][f"{rid}:{arm}"]["diff"] == 0.0
        arms = loaded["files_by_run"][rid]
        assert arms["B0"]["journal"]["entries"] == 200 and arms["E0d"]["journal"]["entries"] == 400
        assert arms["E1"]["journal"]["entries"] == 200 + (200 - REFUSED[rid]["E1"]) and arms["E2"]["journal"]["entries"] == 200 + (200 - REFUSED[rid]["E2"])
        assert arms["E0d"]["cost"]["on"]["p95"] >= arms["B0"]["cost"]["on"]["p95"]
    assert reading["drift"] == {rid: False for rid in R.RUN_IDS}


def test_the_gate_s_surprise_on_near_duplicates_against_distinct_facts(loaded):
    """The write side cosines the mechanism's amendment is to be set on: the near duplicates' surprise at most 0.22, the distinct
    facts' never under 0.21 and above 0.45 at the first percentile; the two distributions touch at their extremes on seed 7."""
    import numpy as np
    for rid in R.RUN_IDS:
        arms = loaded["files_by_run"][rid]
        w0 = arms["E0d"]["near_duplicates"]["planting"]["writes"]
        distinct = [x["surprise"] for x in w0 if (x["kind"] == "E0d") or (x["kind"] == "original" and x["nearest_before"] is not None)]
        e1 = [x["surprise"] for x in arms["E1"]["near_duplicates"]["planting"]["writes"] if x["kind"] == "E1"]
        e2 = [x["surprise"] for x in arms["E2"]["near_duplicates"]["planting"]["writes"] if x["kind"] == "E2"]
        assert min(distinct) > 0.21 and np.percentile(distinct, 1) > 0.45
        assert max(e1) < 0.22 and np.median(e1) < 0.07 and np.median(e2) < 0.1 and min(e2) < 1e-3     # an E2 pair with a near identical cue
        # corrected 2026-10-14, 19:30 (Rev120): "at most 0.22" holds for E1 alone; E2's admitted tail reaches 0.36, 0.22 and 0.63
        # (seeds 2024, 7, 1337), a name moved by one character moving the cue far for a few names; every refused pair is under 0.15
        # and the gap between the highest refused near duplicate and the lowest distinct fact holds nothing
        assert max(e2) > 0.22 and abs(max(e2) - E2_MAX[rid]) < 1e-3
        refused = [x["surprise"] for a in ("E1", "E2") for x in arms[a]["near_duplicates"]["planting"]["writes"]
                   if x["kind"] == a and not x["admitted"]]
        assert max(refused) < 0.15 and max(refused) < min(distinct)
        assert sum(1 for s in refused if s < 1e-3) == UNDER_1E3[rid]
        # 23:00: the gap between the highest refused pair and the lowest distinct fact holds admitted near duplicates and no
        # distinct fact (69 of the 103 admitted over the three checkpoints lie in it); the lowest near duplicate surprise is
        # 2.26e-5 (seed 7), so no cut at or above 0.001 admits the 96 and a cut admitting all 1200 lies under the sample's minimum
        admitted = [x["surprise"] for a in ("E1", "E2") for x in arms[a]["near_duplicates"]["planting"]["writes"]
                    if x["kind"] == a and x["admitted"]]
        assert all(s >= 0.15 for s in admitted)
        GAP["admitted"] = GAP.get("admitted", []) + admitted
        GAP["lowest_distinct"] = min(GAP.get("lowest_distinct", 1.0), min(distinct))
        if rid == "2e1dc71913b1":
            assert abs(min(refused) - 2.26e-5) < 1e-6
    # pooled over the three checkpoints: the lowest distinct fact is 0.2129 (seed 7) and 69 admitted near duplicates lie under it
    assert abs(GAP["lowest_distinct"] - 0.2129) < 1e-4
    assert sum(1 for s in GAP["admitted"] if s < GAP["lowest_distinct"]) == 69


def test_provenance_states_the_reading(prov, reading):
    assert prov["letter"] == "d" and prov["F1"]["produced"] and prov["F2"]["produced"]
    assert prov["F1"]["refused_partner_nearest"] == {rid: REFUSED[rid] for rid in R.RUN_IDS}
    assert prov["F2"]["holm_rejected"] == reading["F2"]["holm_rejected"] and prov["F2"]["tests"]["E2"]["c_control_only"] == 56
    assert all(v == "reproduced" for v in prov["thresholds_against_record"].values()) and all(prov["b0_against_record_identical_outcomes"].values())
