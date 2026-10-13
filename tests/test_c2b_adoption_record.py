"""Row 47: the adoption of C2b measured (ADR-008, amendment of 2026-10-13; the measurement (0), (i), (ii)), on the
founder's eighteen files of 2026-10-13: six control replays under the legacy geometry read against the legacy files,
six replays under `--index 16,exact`, six bucket probe draws at index seeds 1 and 2. Everything the row states is
recomputed here from the files: the headers (commit, configuration hashes, the journals' hashes equal to rows 40 to
42's and unchanged, the geometries), the controls' exact reproduction, the cell H1 S1 M+ by the module's own reading,
the paired counts and their attribution, the live signatures equal to the probe's, the vetoes, the never planted
windows, the live cost, the six clean draws, the provenance hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cortex_c2b import adoption_reading as ar
from cortex_c2b import hm_lm
from cortex_c2b.read_path import CueIndex, LEGACY_GEOMETRY, ADOPTED_GEOMETRY, same_geometry

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "metrics/mqar"
SEEDS = {"8ceba5db8d0b": 1337, "4cfd6f6e1406": 2024, "2e1dc71913b1": 7}
CONFIG_HASH = {"8ceba5db8d0b": "36a3dd10bbff84eb", "4cfd6f6e1406": "d0c0d93ba24555a8", "2e1dc71913b1": "a727d1cb445afa92"}
JOURNAL_SHA = {"8ceba5db8d0b": "d0d56e5a72fe4cf3", "4cfd6f6e1406": "697be97c15d2edf3", "2e1dc71913b1": "0e0c1d34438c85c3"}   # rows 40 to 42, 45
LEGACY_FILE = {("8ceba5db8d0b", "plain"): "hm-lm-8ceba5db8d0b-session-b-answers.json", ("8ceba5db8d0b", "mark-veto+value"): "hm-lm-8ceba5db8d0b-session-b-answers.json",
               ("4cfd6f6e1406", "plain"): "hm-lm-4cfd6f6e1406-session-b.json", ("4cfd6f6e1406", "mark-veto+value"): "hm-lm-4cfd6f6e1406-session-b-mark-veto-value.json",
               ("2e1dc71913b1", "plain"): "hm-lm-2e1dc71913b1-session-b-answers.json", ("2e1dc71913b1", "mark-veto+value"): "hm-lm-2e1dc71913b1-session-b-answers-mark-veto-value.json"}
PROBE45 = {"8ceba5db8d0b": "bucket-probe-8ceba5db8d0b.json", "4cfd6f6e1406": "bucket-probe-4cfd6f6e1406-run2.json", "2e1dc71913b1": "bucket-probe-2e1dc71913b1.json"}


def _load(name):
    return json.loads((M / name).read_text(encoding="utf-8"))


def _name(rid, geometry, policy):
    return f"hm-lm-{rid}-session-b-{'control' if geometry == 'legacy' else 'index16exact'}{'' if policy == 'plain' else '-mark-veto-value'}.json"


@pytest.fixture(scope="module")
def files():
    return {(rid, g, pol): _load(_name(rid, g, pol)) for rid in SEEDS for g in ("legacy", "adopted") for pol in ("plain", "mark-veto+value")}


@pytest.fixture(scope="module")
def draws():
    return [_load(f"bucket-probe-{rid}-index{s}.json") for rid in SEEDS for s in (1, 2)]


def test_row_47_the_twelve_session_b_files_carry_the_record_s_headers(files):
    for (rid, g, pol), f in files.items():
        assert f["checkpoint"]["run_id"] == rid and f["checkpoint"]["config_hash"] == CONFIG_HASH[rid]
        assert f["code_commit"] == "58a4eda" and f["code_tree_clean"] is True and f["policy"] == pol
        assert f["journal"]["sha256_before"] == f["journal"]["sha256_after"] == JOURNAL_SHA[rid] and f["journal"]["unchanged"] is True
        assert f["journal"]["entries"] == 200 == f["journal"]["planted_found"] and f["journal"]["opened"].startswith("a copy")
        assert f["device"]["device"] == "cpu" and f["device"]["threads"] == 8 and f["versions"]["torch"] == "2.6.0+cu124" and f["versions"]["numpy"] == "2.5.3"
        assert f["timestamp_utc"].startswith("2026-10-13T1")
        idx = f["index"]
        assert same_geometry(idx["record"], LEGACY_GEOMETRY) and idx["record_source"] == "legacy_by_default" and idx["config"] == {"n_tables": 4, "signature": "collapsed"}
        if g == "legacy":
            assert same_geometry(idx["live"], LEGACY_GEOMETRY) and idx["live"]["planes_sha256"] == "680896ab07072738" and idx["overridden"] is False and idx["argument"] is None
        else:
            assert same_geometry(idx["live"], ADOPTED_GEOMETRY) and idx["live"]["planes_sha256"] == "e5703fddba43332f" and idx["overridden"] is True
            assert idx["argument"] == {"n_tables": 16, "n_bits": 10, "signature": "exact"}
        o = f["hm_mark_oracle"]
        assert same_geometry({**o["threshold_index_geometry"], "seed": 0}, {**LEGACY_GEOMETRY}) or o["threshold_index_geometry"]["signature"] == "collapsed"
        assert o["live_index_geometry"]["signature"] == ("collapsed" if g == "legacy" else "exact")
        assert len(f["answers"]["on"]) == 200 and len(f["answers"]["off"]) == 200 and len(f["answers"]["negctrl"]) == 50
        for a in f["answers"]["on"]:
            assert len(a["signatures_query"]) == (4 if g == "legacy" else 16) and len(a["cue"]) == 64 and isinstance(a["touched"], int)


def test_row_47_the_six_controls_reproduce_the_legacy_files_exactly_and_the_threshold_within_the_reproduction_tolerance(files):
    for rid in SEEDS:
        for pol in ("plain", "mark-veto+value"):
            f = files[(rid, "legacy", pol)]
            c = f["control"]
            assert c["ok"] is True and c["differences"] == [] and c["outcomes_equal"] == {"on": True, "off": True, "negctrl": True}
            assert c["against"].replace("\\", "/").split("/")[-1] == LEGACY_FILE[(rid, pol)]      # a Windows path, read on any platform
            legacy = _load(LEGACY_FILE[(rid, pol)])
            assert c["against_sha256"] == hashlib.sha256((M / LEGACY_FILE[(rid, pol)]).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]
            # recomputed here: the comparison run again on the committed files gives the same verdict
            again = hm_lm._compare_with_control(f, legacy, pol, str(M / LEGACY_FILE[(rid, pol)]))
            assert again["ok"] is True and again["outcomes_equal"] == c["outcomes_equal"] and again["threshold"]["difference"] == c["threshold"]["difference"]
            assert c["threshold"]["within_tolerance"] is True and c["threshold"]["stops"] is False
            assert c["threshold"]["tolerance"] == 1e-6 and c["threshold"]["stop_bound"] == 1e-3
            if rid == "8ceba5db8d0b" and pol == "mark-veto+value":
                assert c["compared_as"].startswith("the post hoc replay") and c["not_comparable_by_replay"] == {"on": [], "negctrl": []}
            else:
                assert c["compared_as"] == "the legacy file's own outcomes"
    # the thresholds: equal to the bit on seeds 1337 and 2024, 1.8e-7 off on seed 7 (a third value for that checkpoint on the record)
    assert files[("8ceba5db8d0b", "legacy", "plain")]["control"]["threshold"]["difference"] == 0.0
    assert files[("4cfd6f6e1406", "legacy", "plain")]["control"]["threshold"]["difference"] == 0.0
    assert files[("2e1dc71913b1", "legacy", "plain")]["control"]["threshold"]["difference"] == pytest.approx(1.7881393432617188e-07)
    assert files[("2e1dc71913b1", "legacy", "plain")]["control"]["threshold"]["min_margin_live"] == pytest.approx(0.009874582290649414)
    # the controls' numbers are rows 36, 37 and 39's
    assert files[("8ceba5db8d0b", "legacy", "plain")]["hm_recall_on"] == 0.75 and files[("8ceba5db8d0b", "legacy", "mark-veto+value")]["hm_recall_on"] == 0.875
    assert files[("4cfd6f6e1406", "legacy", "plain")]["hm_recall_on"] == 0.74 and files[("4cfd6f6e1406", "legacy", "mark-veto+value")]["hm_recall_on"] == 0.845
    assert files[("2e1dc71913b1", "legacy", "plain")]["hm_recall_on"] == 0.795 and files[("2e1dc71913b1", "legacy", "mark-veto+value")]["hm_recall_on"] == 0.89


def test_row_47_the_threshold_is_the_same_under_both_geometries_as_declared(files):
    for rid in SEEDS:
        thr = {(g, pol): files[(rid, g, pol)]["hm_mark_oracle"]["threshold"] for g in ("legacy", "adopted") for pol in ("plain", "mark-veto+value")}
        assert len(set(thr.values())) == 1, (rid, thr)
    assert files[("8ceba5db8d0b", "adopted", "plain")]["hm_mark_oracle"]["threshold"] == pytest.approx(0.5360028743743896)
    assert files[("4cfd6f6e1406", "adopted", "plain")]["hm_mark_oracle"]["threshold"] == pytest.approx(0.6122402548789978)
    assert files[("2e1dc71913b1", "adopted", "plain")]["hm_mark_oracle"]["threshold"] == pytest.approx(0.5907554626464844)


def test_row_47_the_cell_is_h1_s1_m_plus_by_the_declared_reading(files):
    r = ar.read_adoption(files)
    assert r["cell"] == "H1 S1 M+" and r["policy_read"] == "mark-veto+value"
    assert r["H"] == {"letter": "H1", "in_window": 600, "questions": 600, "in_window_under_plain": 600,
                      "per_checkpoint": {"8ceba5db8d0b": 200, "4cfd6f6e1406": 200, "2e1dc71913b1": 200}, "misses": {"8ceba5db8d0b": [], "4cfd6f6e1406": [], "2e1dc71913b1": []}}
    assert r["S"]["letter"] == "S1" and r["S"]["broken"] == {} and all(v["holds"] and v["legacy_holds"] for v in r["S"]["per_seed"].values())
    assert r["S"]["conceiving_seed_reported"]["8ceba5db8d0b"]["holds"] is True
    m = r["M"]
    assert (m["b"], m["c"]) == (48, 23) and m["p"] == pytest.approx(0.004065116124638666) and m["letter"] == "M+"
    assert m["per_seed"] == {"4cfd6f6e1406": {"b": 28, "c": 7}, "2e1dc71913b1": {"b": 20, "c": 16}} and m["conceiving_seed_reported"] == {"8ceba5db8d0b": {"b": 23, "c": 6}}
    assert ar.binomial_two_sided_p(28, 7) == pytest.approx(0.0005, abs=5e-5) and ar.binomial_two_sided_p(20, 16) == pytest.approx(0.6177, abs=5e-4)
    assert (m["plain_reported"]["b"], m["plain_reported"]["c"], m["plain_reported"]["letter"]) == (60, 42, "M0") and m["plain_reported"]["p"] == pytest.approx(0.0918, abs=5e-4)
    assert "the adoption stands" in r["consequence"] and "claimed at the singular until the holdout" in r["consequence"]
    assert r["counted_seeds"] == ["4cfd6f6e1406", "2e1dc71913b1"] and r["reported_not_counted"] == ["8ceba5db8d0b"]


def test_row_47_the_paired_counts_the_gains_are_the_control_s_misses_and_no_line_is_lost(files):
    r = ar.read_adoption(files)
    gains = {"8ceba5db8d0b": 13, "4cfd6f6e1406": 13, "2e1dc71913b1": 11}        # row 45's control misses, all proposed by C2b
    strict_sys = {"8ceba5db8d0b": (175, 192, 23, 6), "4cfd6f6e1406": (169, 190, 28, 7), "2e1dc71913b1": (178, 182, 20, 16)}
    strict_plain = {"8ceba5db8d0b": (150, 164, 30, 16), "4cfd6f6e1406": (148, 163, 37, 22), "2e1dc71913b1": (159, 162, 23, 20)}
    windows = {"8ceba5db8d0b": (2, 4), "4cfd6f6e1406": (0, 1), "2e1dc71913b1": (1, 3)}   # never planted windows at or above the threshold, the probe's bound
    for rid in SEEDS:
        for pol, table in (("mark-veto+value", strict_sys), ("plain", strict_plain)):
            pc = r["paired"][rid][pol]
            assert pc["own_in_window"] == {"control": 200 - gains[rid], "adopted": 200, "gained": gains[rid], "lost": 0}
            assert (pc["strict"]["control"], pc["strict"]["adopted"], pc["strict"]["b_adopted_only"], pc["strict"]["c_control_only"]) == table[rid]
            s = pc["discordant_by_window"]
            assert s["c_newly_present"] == 0 and s["c_lost"] == 0 and s["c_absent_both"] == 0 and s["b_lost"] == 0 and s["b_absent_both"] == 0
            assert s["b_newly_present"] + s["b_already_present"] == pc["strict"]["b_adopted_only"] and s["c_already_present"] == pc["strict"]["c_control_only"]
            assert pc["never_planted"]["windows_at_or_above_threshold"] == {"control": windows[rid][0], "adopted": windows[rid][1]}
            assert sum(sum(row.values()) for row in pc["outcomes"].values()) == 200
        sysc = r["paired"][rid]["mark-veto+value"]
        assert sysc["never_planted"]["claims"] == {"control": 0, "adopted": 0}                      # the veto holds the negative control at zero under both
        assert sysc["outcomes"]["strict"]["invalid"] == 0 and sysc["outcomes"]["abstain"]["invalid"] == 0   # no invalid citation under the system, either geometry
    assert r["paired"]["8ceba5db8d0b"]["mark-veto+value"]["discordant_by_window"]["b_newly_present"] == 11
    assert r["paired"]["4cfd6f6e1406"]["mark-veto+value"]["discordant_by_window"]["b_newly_present"] == 13
    assert r["paired"]["2e1dc71913b1"]["mark-veto+value"]["discordant_by_window"]["b_newly_present"] == 11
    assert r["vetoes"] == {"8ceba5db8d0b": {"legacy": {"on": 4, "negctrl": 5}, "adopted": {"on": 2, "negctrl": 10}, "difference": {"on": -2, "negctrl": 5}},
                           "4cfd6f6e1406": {"legacy": {"on": 6, "negctrl": 7}, "adopted": {"on": 3, "negctrl": 5}, "difference": {"on": -3, "negctrl": -2}},
                           "2e1dc71913b1": {"legacy": {"on": 3, "negctrl": 11}, "adopted": {"on": 6, "negctrl": 10}, "difference": {"on": 3, "negctrl": -1}}}


def test_row_47_the_lost_pairs_are_the_model_s_abstentions_or_the_veto_on_another_line_never_the_own_line_vetoed(files):
    expected = {"8ceba5db8d0b": (4, 2), "4cfd6f6e1406": (4, 3), "2e1dc71913b1": (10, 6)}
    for rid, (abst, vetoed) in expected.items():
        ctl, ad, adp = files[(rid, "legacy", "mark-veto+value")], files[(rid, "adopted", "mark-veto+value")], files[(rid, "adopted", "plain")]
        kinds = {"abstained": 0, "vetoed": 0}
        for x, y, z in zip(ctl["answers"]["on"], ad["answers"]["on"], adp["answers"]["on"]):
            if x["outcome"] == "strict" and y["outcome"] != "strict":
                assert x["own_in_window"] and y["own_in_window"] and y["outcome"] == "abstain"
                if z["kind"] == "unknown":
                    kinds["abstained"] += 1
                else:
                    assert z["kind"] == "cite" and not z["cited_pointer_is_own"]            # another line cited under plain, refused by the veto
                    kinds["vetoed"] += 1
        assert kinds == {"abstained": abst, "vetoed": vetoed}, rid


def test_row_47_the_live_signatures_equal_the_probe_s_on_600_of_600_and_the_own_line_ranks_first(files):
    idx16 = CueIndex(dim=64, seed=0, n_tables=16, signature="exact")
    assert idx16.geometry()["planes_sha256"] == "e5703fddba43332f"
    for rid in SEEDS:
        f, p = files[(rid, "adopted", "mark-veto+value")], _load(PROBE45[rid])
        assert p["configurations"]["C2b"]["planes_sha256"] == "e5703fddba43332f"
        for a, row, q in zip(f["answers"]["on"], p["rows"]["C2b"], p["questions"]):
            assert a["entity"] == q["entity"]
            assert a["signatures_query"] == row["signatures_query"]                              # the replay's cpu cue and the probe's cuda cue bucket alike
            assert [idx16._sig(np.asarray(a["cue"], np.float32), t) for t in range(16)] == a["signatures_query"]
            assert a["own_in_candidates"] is True and a["own_rank"] == 1 and a["own_in_window"] is True and a["touched"] == row["n_candidates"]
        drift = max(float(np.max(np.abs(np.asarray(a["cue"], np.float32) - np.asarray(q, np.float32)))) for a, q in zip(f["answers"]["on"], p["query_cues"]["questions"]))
        assert 1e-6 < drift < 3e-6                                                               # the encoding path's drift, as row 44 measured it


def test_row_47_the_live_cost_is_the_probe_s_and_the_adopted_geometry_costs_what_the_control_costs(files):
    cost = {rid: (files[(rid, "legacy", "plain")]["cost"]["on"], files[(rid, "adopted", "plain")]["cost"]["on"]) for rid in SEEDS}
    assert (cost["8ceba5db8d0b"][0]["p95"], cost["8ceba5db8d0b"][1]["p95"]) == (11.0, 9.0) and (cost["8ceba5db8d0b"][0]["max"], cost["8ceba5db8d0b"][1]["max"]) == (17, 12)
    assert (cost["4cfd6f6e1406"][0]["p95"], cost["4cfd6f6e1406"][1]["p95"]) == (10.0, 10.0) and (cost["4cfd6f6e1406"][0]["max"], cost["4cfd6f6e1406"][1]["max"]) == (18, 12)
    assert (cost["2e1dc71913b1"][0]["p95"], cost["2e1dc71913b1"][1]["p95"]) == (11.0, 10.0) and (cost["2e1dc71913b1"][0]["max"], cost["2e1dc71913b1"][1]["max"]) == (14, 14)
    for rid in SEEDS:
        for g in ("legacy", "adopted"):
            f, p = files[(rid, g, "plain")], _load(PROBE45[rid])
            rows = p["rows"]["C0" if g == "legacy" else "C2b"]
            assert [a["touched"] for a in f["answers"]["on"]] == [row["n_candidates"] for row in rows]   # the live cost is the probe's candidate count, question by question
            assert f["cost"]["on"]["sum"] == sum(row["n_candidates"] for row in rows)
            assert files[(rid, g, "mark-veto+value")]["cost"] == f["cost"]                               # the same reads under both policies


def test_row_47_the_system_s_numbers_under_the_adopted_geometry(files):
    sys_ = {rid: files[(rid, "adopted", "mark-veto+value")] for rid in SEEDS}
    assert sys_["4cfd6f6e1406"]["hm_recall_on"] == 0.95 and sys_["2e1dc71913b1"]["hm_recall_on"] == 0.91 and sys_["8ceba5db8d0b"]["hm_recall_on"] == 0.96
    for rid, f in sys_.items():
        assert f["hm_invalid_citation_on"] == 0.0 and f["hm_negctrl_rate"] == 0.0 and f["hm_retrieval_hit"] == 1.0 and f["hm_recall_off"] == 0.0
        assert f["gate_conditions_without_skill_arm"] == 1 and f["hm_false_abstention_on"] == pytest.approx(1 - f["hm_recall_on"])
        assert f["hm_mark_oracle"]["recall_on_strict"] == 1.0 and f["hm_mark_oracle"]["claims_on"] == 1.0
    assert [sys_[r]["hm_mark_oracle"]["negctrl_claims"] for r in ("8ceba5db8d0b", "4cfd6f6e1406", "2e1dc71913b1")] == [0.08, 0.02, 0.06]   # 4, 1, 3 of 50
    plain = {rid: files[(rid, "adopted", "plain")] for rid in SEEDS}
    assert [plain[r]["hm_recall_on"] for r in ("8ceba5db8d0b", "4cfd6f6e1406", "2e1dc71913b1")] == [0.82, 0.815, 0.81]
    assert [plain[r]["hm_negctrl_rate"] for r in ("8ceba5db8d0b", "4cfd6f6e1406", "2e1dc71913b1")] == [0.2, 0.1, 0.2]         # the model alone, INVALID under both geometries
    assert all(plain[r]["gate_conditions_without_skill_arm"] == 0 for r in SEEDS)


def test_row_47_the_six_draws_are_clean_and_read_as_the_expectation_written_before_the_run(draws):
    d = ar.read_draws(draws)
    assert d["clean_draws"] == 6 == d["draws"] and d["misses"] == 0 and d["inadmissible"] == [] and d["miss_detail"] == []
    assert d["expected_misses_per_draw"] == pytest.approx(0.03253, abs=1e-5) and d["chance_both_clean"] == pytest.approx(0.937, abs=1e-3)
    assert d["per_draw_miss_chance_bound_95"] == pytest.approx(0.776, abs=1e-3) and d["reading"].startswith("two clean draws")
    for f in draws:
        rid, s = f["checkpoint"]["run_id"], f["index_seed"]
        assert f["code_commit"] == "58a4eda" and f["code_tree_clean"] is True and f["checkpoint"]["config_hash"] == CONFIG_HASH[rid]
        assert f["journal"]["sha256_before"] == JOURNAL_SHA[rid] and f["journal"]["unchanged"] is True and f["device"]["device"] == "cuda"
        assert f["precondition_row38"] == {"applies": False} and f["control_matches_live_index"] is True and all(f["subset_relations"].values())
        assert f["index"]["overridden"] is True and f["index"]["record_source"] == "legacy_by_default" and same_geometry(f["index"]["record"], LEGACY_GEOMETRY)
        assert f["index"]["control_live"]["seed"] == s and f["configurations"]["C2b"]["index_seed"] == s
        c2b = f["aggregates"]["C2b"]
        assert c2b["candidate_recall"] == 1.0 and c2b["misses"] == [] and c2b["admissible"] is True and c2b["L"] == 0 and c2b["R"] == len(f["aggregates"]["C0"]["misses"])
        assert c2b["cost"]["p95"] <= 10 and c2b["cost"]["max"] <= 14
    planes = {(f["index_seed"], f["configurations"]["C2b"]["planes_sha256"]) for f in draws}
    assert planes == {(1, "4fea5708987638e1"), (2, "a9276d53f974e855")}                      # one draw per seed, the same on the three journals


def test_row_47_the_provenance_names_the_eighteen_files_by_their_lf_hashes():
    prov = json.loads((M / "PROVENANCE-adoption.json").read_text(encoding="utf-8"))
    assert prov["row"] == "47" and prov["cell"] == "H1 S1 M+"
    assert len(prov["files"]) == 18
    for name, e in prov["files"].items():
        raw = (M / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(raw).hexdigest()[:16] == e["sha256_lf"], name
        d = json.loads(raw)
        assert d["code_commit"] == "58a4eda" == e["code_commit"] and d["timestamp_utc"] == e["timestamp_utc"]
