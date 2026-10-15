"""Rows 49 and 50: the identity rule at the write gate measured on the five arms (row 49, letter alpha, Decision 12) and the read
side selection replayed on its files and confirmed live once (row 50, Decision 13: R4 adopted; F2 closed under R4), ADR-007,
amendment of 2026-10-14, evening, sections 5 and 6; run by the founder 2026-10-14 23:56 to 2026-10-15 00:50 UTC, read cold at
00:58 UTC. Everything recomputed from the sixteen retained files and the provenance by the declared functions; these tests fail if
a digit of either row moves."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
M = ROOT / "metrics" / "mqar"

from cortex_c2b import near_duplicate_reading as R           # noqa: E402
from cortex_c2b.adoption_reading import binomial_two_sided_p  # noqa: E402
from cortex_c2b.write_path import LEGACY_THRESHOLD            # noqa: E402

CODE_COMMIT = "667fae7"
S2024, S7, S1337 = "4cfd6f6e1406", "2e1dc71913b1", "8ceba5db8d0b"
LIVE = (S2024, S7)
CONFIG_HASH = {S1337: "36a3dd10bbff84eb", S2024: "d0c0d93ba24555a8", S7: "a727d1cb445afa92"}
THRESHOLD = {S1337: 0.5360028743743896, S2024: 0.6122402548789978, S7: 0.5907554626464844}
ROW48_REFUSED = {S1337: {"E1": 200, "E2": 160}, S2024: {"E1": 198, "E2": 162}, S7: {"E1": 192, "E2": 185}}      # row 48's numbers
E4_SHADOW_REFUSED = {S2024: 43, S7: 34, S1337: 38}
E4_SHADOW_BY_SCHEMA = {S2024: {"place": 23, "object": 20}, S7: {"place": 24, "object": 10}, S1337: {"place": 22, "object": 16}}
OWN_CITED = {S2024: {"B0": 190, "E0d": 189, "E1": 101, "E2": 85, "E4": 129}, S7: {"B0": 182, "E0d": 182, "E1": 96, "E2": 101, "E4": 123},
             S1337: {"B0": 192, "E0d": 188, "E1": 85, "E2": 95, "E4": 124}}
CITED_DUPLICATE = {S2024: {"E1": 99, "E2": 115, "E4": 43}, S7: {"E1": 102, "E2": 99, "E4": 49}, S1337: {"E1": 113, "E2": 105, "E4": 49}}
ORACLE_E2 = {S2024: (180, 20), S7: (171, 29), S1337: (177, 23)}
ORACLE_E1 = {S2024: (97, 103), S7: (98, 102), S1337: (101, 99)}
COVERAGE = {S2024: {"B0": 0.95, "E0d": 0.95, "E1": 1.0, "E2": 1.0, "E4": 0.645}, S7: {"B0": 0.91, "E0d": 0.91, "E1": 0.99, "E2": 1.0, "E4": 0.615},
            S1337: {"B0": 0.96, "E0d": 0.94, "E1": 0.99, "E2": 1.0, "E4": 0.62}}
UNDER_FLOOR = {S2024: {"E1": 200, "E2": 182, "E4": 68}, S7: {"E1": 199, "E2": 195, "E4": 62}, S1337: {"E1": 200, "E2": 181, "E4": 64}}
E2_DUP_CITED_BY_SHAPE = {S2024: {"none": 115, "R1": 19, "R2": 0, "R3": 13, "R4": 0}, S7: {"none": 99, "R1": 31, "R2": 0, "R3": 11, "R4": 0},
                         S1337: {"none": 105, "R1": 24, "R2": 0, "R3": 12, "R4": 0}}
E2_R3_ABSTAINED = {S2024: 109, S7: 106, S1337: 104}
E4_COVERAGE_R4 = {S2024: 0.86, S7: 0.86, S1337: 0.87}
E4_FALSE_ABST = {S2024: (71, 28), S7: (77, 28), S1337: (76, 26)}
R4_VS_R1_E2 = {S2024: 19, S7: 31, S1337: 24}
NEIGHBOUR_BASE_OWN = {S2024: 94, S7: 94, S1337: 92}


@pytest.fixture(scope="module")
def prov():
    return json.loads((M / "PROVENANCE-identity-rule.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def loaded():
    return R.load_row49_from_dir(M)


@pytest.fixture(scope="module")
def reading(loaded, prov):
    fifteen = {name: meta["sha256_lf"] for name, meta in prov["files"].items() if "-R4-" not in name}
    return R.read_row49(loaded["files_by_run"], loaded["row48_files_by_run"], loaded["row47_files"], expected_commit=CODE_COMMIT, provenance=fifteen,
                        row48_provenance=loaded["row48_provenance"], row47_provenance=loaded["row47_provenance"])


@pytest.fixture(scope="module")
def replay(loaded, reading):
    return R.replay_row50(loaded["files_by_run"], row49_letter=reading["letter"])


def test_the_sixteen_files_are_there_with_their_hashes_commit_checkpoints_and_thresholds(prov, loaded):
    assert len(prov["files"]) == 16 and sum(1 for n in prov["files"] if "-R4-" in n) == 1
    for name, meta in prov["files"].items():
        b = (M / name).read_bytes()
        assert b"\r\n" not in b, name
        assert hashlib.sha256(b).hexdigest()[:16] == meta["sha256_lf"], name
        d = json.loads(b)
        assert d["code_commit"] == CODE_COMMIT == meta["code_commit"] and d["code_tree_clean"] is True, name
        assert d["checkpoint"]["run_id"] == meta["run_id"] and d["checkpoint"]["config_hash"] == CONFIG_HASH[meta["run_id"]], name
        assert d["hm_mark_oracle"]["threshold"] == THRESHOLD[meta["run_id"]], name                    # equal to the bit on the sixteen
        assert d["journal"]["unchanged"] is True and d["index"]["live"]["planes_sha256"] == "e5703fddba43332f"
        assert d["near_duplicates"]["admission"]["record"] == "identity" == d["near_duplicates"]["admission"]["argument"]
        assert d["versions"]["torch"] == "2.6.0+cu124" and d["versions"]["numpy"] == "2.5.3"
        assert ("-identity-" in name) and (("-R4-" in name) == (d.get("decode_shape") == "R4"))
    for rid in R.RUN_IDS:
        for arm in R.ROW49_ARMS:
            f = loaded["files_by_run"][rid][arm]
            assert f is not None and f["checkpoint"]["run_id"] == rid
            R.check_identity_arm_file(f, arm)
    assert loaded["live_r4"][S2024] is not None and loaded["live_r4"][S7] is None and loaded["live_r4"][S1337] is None


def test_p0_to_p3_hold_and_the_letter_is_alpha(reading):
    assert reading["letter"] == "alpha" and "stopped_at" not in reading
    assert reading["P0"]["ok"] and reading["P0"]["commit"] == CODE_COMMIT and reading["P0"]["provenance_checked"] and reading["P0"]["problems"] == []
    assert reading["references"]["ok"] and reading["references"]["problems"] == []
    p1 = reading["P1"]
    assert p1["ok"] and all(v["diff"] == 0.0 and v["scratch_admitted"] == 200 == v["scratch_facts"] and v["scratch_admission"] == "identity" for v in p1["per_file"].values())
    p2 = reading["P2"]
    assert p2["ok"] and len(p2["per_file"]) == 6
    for v in p2["per_file"].values():                                   # B0 and E0d identical to row 48, entries and questions
        assert v["entries"]["ok"] and v["entries"]["n_differences"] == 0 and v["entries"]["max_salience_drift"] == 0.0
        assert v["questions"]["ok"] and v["questions"]["n_differences"] == 0
    p3 = reading["P3"]
    assert p3["ok"] and p3["legacy_refused_near_duplicates_total"] == 1097 == p3["row48_refusals_recomputed"] == p3["row48_refusals"]
    for rid in R.RUN_IDS:
        for kind in ("E1", "E2"):
            v = p3["per_file"][f"{rid}:{kind}"]
            assert v["legacy_refused"] == ROW48_REFUSED[rid][kind] == v["row48_refused"] and v["n_differences"] == 0 and v["writes"] == 400
        assert p3["per_file"][f"{rid}:B0"]["legacy_refused"] == 0 and p3["per_file"][f"{rid}:E0d"]["legacy_refused"] == 0


def test_f1_is_zero_under_the_rule_on_1200_near_duplicates_and_600_paraphrases(reading, loaded):
    f1 = reading["F1"]
    assert f1["zero"] and f1["totals_as_declared"]
    assert (f1["near_duplicates"], f1["near_duplicates_refused"], f1["paraphrases"], f1["paraphrases_refused"]) == (1200, 0, 600, 0)
    assert f1["originals_refused_beside"] == 0 and f1["e0d_refused_beside"] == 0
    for rid in R.RUN_IDS:
        for arm in R.ROW49_ARMS:
            f = loaded["files_by_run"][rid][arm]
            adm = f["near_duplicates"]["planting"]["admission"]
            assert adm["rule"] == "identity" and adm["source"] == "stamp" and adm["threshold"] == 0.15
            shadow = adm["shadow_legacy"]
            assert shadow["judged"] == f["near_duplicates"]["planting"]["counts"]["entries"] == (200 if arm == "B0" else 400)
            assert adm["near_recorded"] == shadow["legacy_refused"] == shadow["disagreements_with_rule"]
            assert shadow["legacy_refused_originals"] == 0 and shadow["legacy_refused_e0d"] == 0
            writes = f["near_duplicates"]["planting"]["writes"]
            assert all(w["admitted"] for w in writes) and all(w["admission"] == "identity" for w in writes)
            kept = [w for w in writes if w["kind"] in ("E1", "E2", "E4")]
            assert all(w["reason"] == "admitted (identity: different source text)" for w in kept)
            # the near stamp sits on every admitted write whose surprise is under the legacy threshold, and on no other
            assert all((w["near"] is not None) == (w["surprise"] < LEGACY_THRESHOLD and w["nearest_before"] is not None) for w in writes)
            assert sum(1 for w in writes if w["near"] is not None) == adm["near_recorded"]
            # the near stamp coincides write by write with the shadow legacy gate's refusal (its decision is a boolean, True admitted)
            assert all((w["near"] is not None) == (w["legacy_decision"] is False) for w in writes)
        assert loaded["files_by_run"][rid]["E1"]["near_duplicates"]["planting"]["admission"]["shadow_legacy"]["legacy_refused_duplicates"] == ROW48_REFUSED[rid]["E1"]
        assert loaded["files_by_run"][rid]["E2"]["near_duplicates"]["planting"]["admission"]["shadow_legacy"]["legacy_refused_duplicates"] == ROW48_REFUSED[rid]["E2"]
        assert loaded["files_by_run"][rid]["E4"]["near_duplicates"]["planting"]["admission"]["shadow_legacy"]["legacy_refused_paraphrases"] == E4_SHADOW_REFUSED[rid]


def test_e4_planting_the_paraphrases_by_schema(reading):
    e4 = reading["E4"]
    assert e4["label"].startswith("five templates, one per schema, one draw")
    for rid in R.RUN_IDS:
        v = e4["per_run"][rid]
        assert v["paraphrases"] == 200 and v["legacy_refused"] == E4_SHADOW_REFUSED[rid] and v["legacy_refused_share"] == pytest.approx(E4_SHADOW_REFUSED[rid] / 200)
        for schema, x in v["per_schema"].items():
            assert x["n"] == 40
            if schema in ("place", "object"):                        # the near templates: the partner nearest on every one, the shadow refuses a share
                assert x["legacy_refused"] == E4_SHADOW_BY_SCHEMA[rid][schema] and x["nearest_is_partner"] == 40
                assert x["surprise"]["median"] < 0.2 and x["cos_raw"]["median"] > 0.95 and x["under_tribunal_floor"] >= 24
            else:                                                     # the far templates: no refusal, surprise medians above 0.35
                assert x["legacy_refused"] == 0 and x["surprise"]["median"] > 0.35 and x["cos_raw"]["median"] < 0.85 and x["under_tribunal_floor"] <= 1
        assert sum(x["legacy_refused"] for x in v["per_schema"].values()) == E4_SHADOW_REFUSED[rid]


def test_the_three_reported_tests_e2_mcnemar_e1_and_e4_tango(reading):
    t = reading["reported"]["tests"]
    e2 = t["E2"]
    assert (e2["b_test_only"], e2["c_control_only"], e2["n_pairs"]) == (5, 190, 400)
    assert e2["p_two_sided"] == pytest.approx(binomial_two_sided_p(5, 190)) and e2["p_two_sided"] < 1e-40 and e2["significant"] and e2["alpha"] == 0.05
    assert e2["paired_difference_test_minus_control"] == pytest.approx(-0.4625) and e2["direction"].startswith("fewer")
    assert e2["estimand"].startswith("200 near duplicates against 200 distinct facts at equal density")
    assert (e2["per_seed"][S2024]["b_test_only"], e2["per_seed"][S2024]["c_control_only"]) == (0, 104)
    assert (e2["per_seed"][S7]["b_test_only"], e2["per_seed"][S7]["c_control_only"]) == (5, 86)
    assert (e2["per_seed"][S1337]["b_test_only"], e2["per_seed"][S1337]["c_control_only"]) == (3, 96)       # reported, not pooled
    e1 = t["E1"]
    assert (e1["b_test_only"], e1["c_control_only"], e1["n_pairs"]) == (17, 191, 400) and e1["p"] is None
    assert e1["paired_difference_test_minus_control"] == pytest.approx(-0.435)
    assert tuple(e1["tango_95"]) == pytest.approx(R.tango_interval(17, 191, 400)) and e1["tango_95"][1] < -0.37
    assert e1["named_as"].startswith("the contradiction case")
    assert (e1["per_seed"][S2024]["b_test_only"], e1["per_seed"][S2024]["c_control_only"]) == (7, 95)
    assert (e1["per_seed"][S7]["b_test_only"], e1["per_seed"][S7]["c_control_only"]) == (10, 96)
    assert (e1["per_seed"][S1337]["b_test_only"], e1["per_seed"][S1337]["c_control_only"]) == (8, 111)
    e4 = t["E4"]
    assert (e4["b_test_only"], e4["c_control_only"], e4["n_pairs"]) == (19, 138, 400) and e4["p"] is None
    assert e4["paired_difference_test_minus_control"] == pytest.approx(-0.2975) and tuple(e4["tango_95"]) == pytest.approx(R.tango_interval(19, 138, 400))
    assert e4["cited_pointer_is_own_label"] == "original line cited"
    assert e4["citation_support_pooled"] == {"b_test_only": 26, "c_control_only": 53}
    assert (e4["per_seed"][S2024]["citation_support"]["b_test_only"], e4["per_seed"][S2024]["citation_support"]["c_control_only"]) == (11, 28)
    assert (e4["per_seed"][S7]["citation_support"]["b_test_only"], e4["per_seed"][S7]["citation_support"]["c_control_only"]) == (15, 25)
    for arm in ("E1", "E2", "E4"):
        for rid in R.RUN_IDS:
            ps = t[arm]["per_seed"][rid]
            assert ps["own_cited_control"] == OWN_CITED[rid]["E0d"] and ps["own_cited_test"] == OWN_CITED[rid][arm]
            assert ps["split"]["b"]["lost"] == 0 and ps["split"]["c"]["lost"] == 0 and ps["split"]["c"]["own_not_admitted"] == 0
            assert ps["split"]["c"]["same_lines_other_order"] == 0


def test_side_counts_and_the_blindness_of_the_protocol_s_rates(reading, loaded):
    sc = reading["reported"]["side_counts"]
    for rid in R.RUN_IDS:
        for arm in R.ROW49_ARMS:
            v = sc[f"{rid}:{arm}"]
            assert v["agrees_with_file"] and v["own_cited"] == OWN_CITED[rid][arm]
            assert v["cited_is_duplicate"] == CITED_DUPLICATE[rid].get(arm, 0)
            if arm in ("B0", "E0d"):
                assert v["duplicate_in_window"] == 0 and v["oracle_cites_own"] == 200
        e2 = sc[f"{rid}:E2"]
        assert (e2["oracle_cites_own"], e2["oracle_cites_duplicate"]) == ORACLE_E2[rid] and e2["duplicate_in_window"] >= 199
        # the oracle's count is over the 200 reads; on the contested reads alone it is 180, 171, 176 (seed 1337 has one read
        # whose duplicate is absent from the window, cited own by the oracle): 528 of 600 reads, 527 of the 599 contested
        on = loaded["files_by_run"][rid]["E2"]["answers"]["on"]
        contested = [x for x in on if x.get("both_marks_at_or_above_threshold")]
        assert len(contested) == {S2024: 200, S7: 200, S1337: 199}[rid]
        assert sum(1 for x in contested if x["oracle_cites_own"]) == {S2024: 180, S7: 171, S1337: 176}[rid]
        assert sc[f"{rid}:E1"]["duplicate_in_window"] == 200 == sc[f"{rid}:E1"]["both_marks_at_or_above_threshold"]
        assert (sc[f"{rid}:E1"]["oracle_cites_own"], sc[f"{rid}:E1"]["oracle_cites_duplicate"]) == ORACLE_E1[rid]
        assert sc[f"{rid}:E4"]["oracle_cites_own"] == 200
        f = loaded["files_by_run"][rid]
        # E2: strict recall 1.0 while the own line is cited on fewer than 102 of 200 (the duplicate carries the expected value)
        assert f["E2"]["hm_recall_on"] == 1.0 and OWN_CITED[rid]["E2"] <= 101
        # E1: the duplicate carries the second value, so strict recall falls to the own citations' share
        assert f["E1"]["hm_recall_on"] == pytest.approx(OWN_CITED[rid]["E1"] / 200) and f["E1"]["hm_valid_citation_on"] >= 0.99
        assert f["E1"]["hm_invalid_citation_on"] == 0.0 and f["E2"]["hm_invalid_citation_on"] == 0.0


def test_provenance_metrics_coverage_support_contested_reads_and_cost(reading):
    pm = reading["reported"]["provenance_metrics"]
    lc = reading["reported"]["live_cost"]
    for rid in R.RUN_IDS:
        for arm in R.ROW49_ARMS:
            v = pm[f"{rid}:{arm}"]
            assert v["n"] == 200 and v["coverage"] == pytest.approx(COVERAGE[rid][arm]) and v["retrieval_recall"] == 1.0 and v["own_in_candidates"] == 1.0
            assert v["citation_unreadable"] == 0 and v["abstentions"]["verified"] == 0 and v["abstentions"]["false"] == v["abstentions"]["n"]
            assert v["designated_line_cited"] == OWN_CITED[rid][arm]
        assert pm[f"{rid}:E2"]["value_correct"] == 1.0 and pm[f"{rid}:E2"]["citation_supported"] == OWN_CITED[rid]["E2"]
        assert pm[f"{rid}:E2"]["attributed_accuracy"] == pytest.approx(OWN_CITED[rid]["E2"] / 200)
        assert pm[f"{rid}:E2"]["contested"]["forced_choice_error"] == CITED_DUPLICATE[rid]["E2"]
        assert pm[f"{rid}:E1"]["contested"]["n"] == 200 and pm[f"{rid}:E1"]["contested"]["forced_choice_error"] == CITED_DUPLICATE[rid]["E1"]
        assert pm[f"{rid}:E1"]["value_correct"] == pytest.approx(OWN_CITED[rid]["E1"] / 200)
        e4 = pm[f"{rid}:E4"]
        assert e4["citation_supported"] >= 172 and e4["contested"]["forced_choice_error"] == 0            # citing the paraphrase is correct provenance
        assert e4["contested"]["abstention_rate"] > e4["contested"]["abstention_rate_one_clears"] and e4["contested"]["difference"] > 0.15
        assert e4["abstentions"]["false"] == E4_FALSE_ABST[rid][0]
        assert lc[f"{rid}:B0"]["mean"] < 6.0 < 10.0 < lc[f"{rid}:E0d"]["mean"] < min(lc[f"{rid}:{a}"]["mean"] for a in ("E1", "E2", "E4"))
    assert max(v["max"] for v in lc.values()) == 48


def test_tribunal_static_nearest_codes_and_decision_12(reading):
    ts = reading["reported"]["tribunal_static"]
    for rid in R.RUN_IDS:
        for arm in R.ROW49_ARMS:
            v = ts[f"{rid}:{arm}"]
            assert v["under_floor_by_kind"]["original"] == 0 and v["admitted"] == (200 if arm == "B0" else 400)
            if arm in ("E1", "E2", "E4"):
                assert v["under_floor"] == UNDER_FLOOR[rid][arm] == v["under_floor_by_kind"][arm]
            else:
                assert v["under_floor"] == 0
    near_dups = sum(UNDER_FLOOR[rid][a] for rid in R.RUN_IDS for a in ("E1", "E2"))
    paraphrases = sum(UNDER_FLOOR[rid]["E4"] for rid in R.RUN_IDS)
    assert (near_dups, paraphrases) == (1157, 194) and "1157 of the 1200" in reading["decision_12"]
    assert reading["decision_12"] == R.DECISION_12 and reading["labels"] == R.ROW49_LABELS
    nc = reading["reported"]["nearest_codes"]
    for rid in R.RUN_IDS:
        for kind in ("E1", "E2"):
            assert nc[f"{rid}:{kind}"]["partners"] == 200 and nc[f"{rid}:{kind}"]["partners_whose_nearest_is_not_partner"] == 0
            assert 95 <= nc[f"{rid}:{kind}"]["originals_whose_nearest_is_a_partner"] <= 110
        assert nc[f"{rid}:E4"]["partners_whose_nearest_is_not_partner"] == {S2024: 67, S7: 88, S1337: 41}[rid]


def test_row50_validity_and_its_gates(replay, loaded):
    assert replay["valid"] and "unread" not in replay and len(replay["validity"]) == 15 and all(v["ok"] for v in replay["validity"].values())
    assert replay["label"] == R.ROW50_LABEL and set(replay["shapes"]) == set(R.SHAPES)
    gated = R.replay_row50(loaded["files_by_run"], row49_letter="gamma")
    assert gated["unread"].startswith("row 49 is not read alpha") and gated["decision_13"] is None
    short = {rid: dict(arms) for rid, arms in loaded["files_by_run"].items()}
    short[S7]["E4"] = None
    assert "files missing or stopped" in R.replay_row50(short, row49_letter="alpha")["unread"]


def test_row50_the_shapes_on_the_contested_reads(replay):
    pf = replay["per_file"]
    for rid in R.RUN_IDS:
        e2 = pf[f"{rid}:E2"]
        for shape, k in E2_DUP_CITED_BY_SHAPE[rid].items():
            assert e2[shape]["duplicate_cited_on_contested"] == k and e2[shape]["contested"] >= 199
        assert e2["R4"]["own_cited_on_contested"] == e2["R4"]["contested"] and e2["R4"]["coverage"] == 1.0 and e2["R4"]["conditions"]["hold"]
        assert e2["R4"]["risk_among_claims"] == {"other_entity": 0, "wrong_value": 0, "claims": 200} and e2["R4"]["abstentions"] == {"false": 0, "verified": 0}
        assert e2["R3"]["abstained_on_contested"] == E2_R3_ABSTAINED[rid] and not e2["R3"]["conditions"]["hold"]
        assert e2["none"]["duplicate_cited_wilson_95"][0] > 0.42 and e2["R4"]["duplicate_cited_wilson_95"][1] < 0.02
        e1 = pf[f"{rid}:E1"]                                         # the contradiction: R4 abstains on every contested read
        assert e1["R4"]["abstained_on_contested"] == 200 and e1["R4"]["coverage"] == 0.0 and e1["R4"]["own_cited"] == 0
        assert e1["none"]["risk_among_claims"]["wrong_value"] == CITED_DUPLICATE[rid]["E1"] and e1["none"]["risk_among_claims"]["other_entity"] == 0
        assert e1["R2"]["duplicate_cited_on_contested"] == e1["none"]["duplicate_cited_on_contested"]      # the key match is blind to E1
        e4 = pf[f"{rid}:E4"]
        assert e4["none"]["coverage"] == pytest.approx(COVERAGE[rid]["E4"]) and e4["R4"]["coverage"] == pytest.approx(E4_COVERAGE_R4[rid]) == e4["R1"]["coverage"]
        assert (e4["none"]["abstentions"]["false"], e4["R4"]["abstentions"]["false"]) == E4_FALSE_ABST[rid]
        assert e4["R4"]["duplicate_cited_on_contested"] == 0 and e4["R4"]["conditions"]["hold"]
        # the rise is the value reader's abstention lifted where the model had cited the paraphrase (43, 49, 49), plus one on 1337
        assert e4["R4"]["own_cited"] - e4["none"]["own_cited"] == e4["none"]["abstentions"]["false"] - e4["R4"]["abstentions"]["false"] == {S2024: 43, S7: 49, S1337: 50}[rid]
        assert e4["none"]["duplicate_cited_on_contested"] == CITED_DUPLICATE[rid]["E4"]
        for arm in ("B0", "E0d"):                                    # the controls: no shape loses an own citation
            base = pf[f"{rid}:{arm}"]["none"]["own_cited"]
            assert base == OWN_CITED[rid][arm]
            for shape in ("R1", "R2", "R4"):
                assert pf[f"{rid}:{arm}"][shape]["own_cited"] >= base and pf[f"{rid}:{arm}"][shape]["conditions"]["hold"]
        assert pf[f"{rid}:E2"]["R4_against_R1"] == {"disagreements": R4_VS_R1_E2[rid], "attributed": {"ranking_error_of_the_mark": R4_VS_R1_E2[rid], "parse_failure": 0,
                                                                                                        "R4_abstained_on_conflict": 0, "R4_abstained_no_candidate": 0, "R4_lifted_a_value_reader_abstention": 0}}
        a = pf[f"{rid}:E1"]["R4_against_R1"]["attributed"]
        assert a["parse_failure"] == 0 and a["ranking_error_of_the_mark"] == 0 and a["R4_abstained_on_conflict"] >= 198
    assert pf[f"{S2024}:E0d"]["R4"]["own_cited"] == 190 and pf[f"{S2024}:E0d"]["none"]["own_cited"] == 189          # the one citation R4 adds on a control


def test_row50_the_two_reported_blocks_under_the_shapes(replay):
    pf = replay["per_file"]
    for rid in R.RUN_IDS:
        for arm in R.ROW49_ARMS:
            b = pf[f"{rid}:{arm}"]["reported:near_name_never_planted"]
            assert b["none"]["n"] == 50 and b["none"]["claims"] >= 30 and b["R4"]["claims"] == 0 and b["R4"]["cited_neighbour"] == 0
            assert b["R2"]["claims"] == b["none"]["claims"]                              # the key match changes no claim here
            if arm in ("B0", "E0d"):
                assert b["none"]["cited_neighbour"] == b["none"]["claims"] >= 46       # every claim cites the one char neighbour
                assert b["R1"]["claims"] == b["R3"]["claims"] == b["none"]["claims"]
            elif arm in ("E1", "E2"):
                assert b["none"]["claims"] >= 48 and b["R1"]["claims"] == b["none"]["claims"] and 21 <= b["R3"]["claims"] <= 27
            else:                                                                        # E4: the base policy claims less, R1 more
                assert 30 <= b["none"]["claims"] <= 35 and b["R1"]["claims"] == {S2024: 43, S7: 41, S1337: 38}[rid]
                assert b["R3"]["claims"] == {S2024: 35, S7: 30, S1337: 31}[rid]
        q = pf[f"{rid}:E2"]["reported:neighbour_questions"]
        assert q["none"]["n"] == 200 and q["none"]["own_cited"] == NEIGHBOUR_BASE_OWN[rid] and q["none"]["cited_is_original"] + q["none"]["own_cited"] == q["none"]["claims"]
        assert q["R4"]["cited_is_original"] == 0 and q["R4"]["own_cited"] == q["R4"]["claims"] >= 199 and q["R2"]["cited_is_original"] == 0


def test_decision_13_superiority_no_loss_adoption_and_the_closure(replay):
    d = replay["decision_13"]
    s = d["superiority"]
    assert (s["contested_reads"], s["b_R4_only"], s["c_base_only"]) == (400, 214, 0) and s["holds"]
    assert s["p_one_sided"] == pytest.approx(R.mcnemar_one_sided_p(214, 0)) and s["p_one_sided"] < 1e-60 and s["alpha"] == 0.025 and s["min_discordant_one_way"] == 6
    for key, (r4, base) in {f"{S2024}:B0": (190, 190), f"{S2024}:E0d": (190, 189), f"{S7}:B0": (182, 182), f"{S7}:E0d": (182, 182)}.items():
        v = d["no_loss"]["per_control"][key]
        assert v["four_conditions_hold"] and v["not_fewer"] and (v["own_cited_R4"], v["own_cited_base"]) == (r4, base)
    assert d["no_loss"]["holds"] and d["adopted"] is True and d["if_not_adopted"] is None and d["intersection_union"].startswith("both (i) and (ii)")
    c = d["closure"]
    assert (c["loss_pairs_control_only"], c["gain_pairs_test_only"], c["n_paired"], c["n_expected"]) == (0, 28, 400, 400)
    assert c["tango_upper_97_5"] == pytest.approx(R.tango_interval(0, 28, 400)[1]) and c["tango_upper_97_5"] < 0.0 < c["bound"] == 0.02
    assert c["F2_closed_under_R4"] and not c["F2_persists_under_R4"] and c["sentence"] == "F2 closed under R4"
    assert c["split_per_seed"][S2024]["gain_pairs_test_only"] == 10 and c["split_per_seed"][S7]["gain_pairs_test_only"] == 18
    assert all(v["unpaired"] == [] for v in c["split_per_seed"].values())
    assert (d["per_seed"][S1337]["contested"], d["per_seed"][S1337]["b_R4_only"], d["per_seed"][S1337]["c_base_only"]) == (199, 105, 0)   # reported
    assert d["per_seed"][S1337]["closure"]["gain_pairs_test_only"] == 12 and d["per_seed"][S1337]["closure"]["loss_pairs_control_only"] == 0
    assert (d["per_seed"][S2024]["b_R4_only"], d["per_seed"][S7]["b_R4_only"]) == (115, 99) and set(d["reported_seed"]) == {S1337}


def test_the_live_r4_run_reproduces_the_replay_question_by_question(loaded, prov):
    live, e2 = loaded["live_r4"][S2024], loaded["files_by_run"][S2024]["E2"]
    out = R.live_r4_against_replay(live, e2)
    assert out["ok"] and out["differences"] == [] and out["n_differences"] == 0 and out["n_compared"] == 500
    assert out["inputs_sha256"] == {"live": live["_sha256"], "identity_E2": e2["_sha256"]} == {"live": "93baa2bf4a4839c5", "identity_E2": "14b7ad14ccd72086"}
    assert live["decode_shape"] == "R4" and live["near_duplicates"]["decode_shape"] == "R4" and e2["decode_shape"] is None
    assert live["near_duplicates"]["counts"]["own_cited"] == 200 and live["near_duplicates"]["counts"]["cited_is_duplicate"] == 115    # the raw choice, before the shape
    assert live["cost"] == e2["cost"] and live["hm_mark_oracle"]["threshold"] == e2["hm_mark_oracle"]["threshold"]
    assert live["reported_blocks"]["near_name_never_planted"]["claims_n"] == 0 and live["reported_blocks"]["neighbour_questions"]["own_cited"] == 200
    assert prov["row50"]["live_r4"]["n_compared"] == 500 and prov["row50"]["live_r4"]["inputs_sha256"] == out["inputs_sha256"]


def test_the_post_hoc_replay_on_row_48_validates_the_machinery_and_decides_nothing(loaded):
    ph = R.replay_row50(loaded["row48_files_by_run"], post_hoc=True)
    assert ph["post_hoc"] and ph["valid"] and ph["decision_13"] is None and ph["label"].startswith("post hoc")
    assert len(ph["validity"]) == 12 and len(ph["per_file"]) == 12


def test_provenance_states_the_two_rows(prov, reading, replay):
    assert prov["rows"] == "rows 49 and 50" and prov["row49_letter"] == "alpha" == reading["letter"]
    assert prov["row50_decision_13_adopted"] is True and prov["row50_closure"] == "F2 closed under R4"
    r49 = prov["row49"]
    assert r49["exits"]["P3"]["legacy_refused_near_duplicates_total"] == 1097 and r49["F1"]["zero"] and r49["F1"]["near_duplicates_refused"] == 0 and r49["F1"]["paraphrases_refused"] == 0
    assert r49["reported_tests"]["E2"]["b_test_only"] == 5 and r49["reported_tests"]["E2"]["c_control_only"] == 190
    assert r49["reported_tests"]["E1"]["b_test_only"] == 17 and r49["reported_tests"]["E4"]["b_test_only"] == 19
    assert r49["tribunal_static"]["near_duplicates_under_floor"] == 1157 and r49["tribunal_static"]["paraphrases_under_floor"] == 194
    assert r49["decision_12"] == reading["decision_12"]
    for rid in R.RUN_IDS:
        for arm in R.ROW49_ARMS:
            assert r49["side_counts"][f"{rid}:{arm}"]["own_cited"] == OWN_CITED[rid][arm]
    d = prov["row50"]["decision_13"]
    assert d["superiority"]["b_R4_only"] == 214 and d["adopted"] and d["closure"]["sentence"] == "F2 closed under R4"
    assert prov["row50"]["valid"] and all(prov["row50"]["validity"].values())
    for rid in R.RUN_IDS:
        for shape, k in E2_DUP_CITED_BY_SHAPE[rid].items():
            assert prov["row50"]["per_shape"][f"{rid}:E2"][shape]["duplicate_cited_on_contested"] == k == replay["per_file"][f"{rid}:E2"][shape]["duplicate_cited_on_contested"]
    assert prov["checkpoints"][S2024]["threshold"] == THRESHOLD[S2024] and prov["row50"]["row48_post_hoc_replay"]["valid"]
    assert prov["row50"]["row48_post_hoc_replay"]["decision_13"] is None
