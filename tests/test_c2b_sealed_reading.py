"""The cold readings of rows 51 and 52 (`cortex_c2b.sealed_reading`, declared before any family runs): the intervals, the
counts and conditions under a rule, the factor product and its differences, M2's paired test under Holm, the primary
tests of B.2 and the letters, on synthetic pass files whose outcomes are chosen by the test; the first family's direct
form on the record's row 47 file."""
import json
import math
from pathlib import Path

import pytest

from cortex_c2b import record_tools
from cortex_c2b import sealed_reading as SR
from cortex_c2b import resumable as RS
from cortex_c2b.hm_protocol import DELTA, NEGCTRL_INVALID
from cortex_c2b.hm_lm import INVALID_CITATION_MAX

MQ = Path(record_tools.REPO_ROOT) / "metrics" / "mqar"


# ---------------------------------------------------------------------------- #
# intervals                                                                      #
# ---------------------------------------------------------------------------- #
def test_clopper_pearson_known_values():
    lo, hi = SR.clopper_pearson(5, 10)
    assert abs(lo - 0.1870860) < 1e-6 and abs(hi - 0.8129140) < 1e-6
    lo, hi = SR.clopper_pearson(1, 100)
    assert abs(lo - 0.000253146) < 1e-8 and abs(hi - 0.0544594) < 1e-6
    assert SR.clopper_pearson(0, 10) == (0.0, pytest.approx(0.3084971, abs=1e-6)) and SR.clopper_pearson(10, 10)[1] == 1.0
    assert SR.clopper_pearson(0, 0) is None and SR.rate(3, 4)["ci95"] == SR.clopper_pearson(3, 4)
    d = SR.difference_interval(50, 100, 40, 100)
    assert abs(d["difference"] - 0.1) < 1e-12 and d["ci95"][0] < 0 < d["ci95"][1] and "Newcombe" in d["label"]
    d2 = SR.difference_interval(95, 100, 5, 100)
    assert d2["ci95"][0] > 0.7
    assert SR.difference_interval(1, 0, 1, 2)["ci95"] is None


# ---------------------------------------------------------------------------- #
# synthetic pass files                                                           #
# ---------------------------------------------------------------------------- #
def rule_rec(outcome: str, own: bool, kind: str = "cite", extra=None):
    return {"kind": kind if outcome != "abstain" else "unknown", "cited_pointer": ("own" if own else "other") if outcome != "abstain" else None, "answered": "v",
            "outcome": outcome, "cited_pointer_is_own": (own if outcome != "abstain" else None), "claim": outcome != "abstain", **(extra or {})}


def record(i, block, outcomes: dict, own_in_window=True, mark_own=0.9, mark_max=0.9, seed=None, raw_kind="cite", raw_label="A"):
    """One pass record whose outcome under each rule is given: outcomes[rule] = (outcome, cites_own)."""
    rules = {}
    for r in SR.RULES:
        o, own = outcomes.get(r, ("abstain", False))
        extra = {"veto_cited_pointer": ("own" if own else "other") if o != "abstain" else None, "shape": {"pointer": ("own" if own else "other") if o != "abstain" else None}} if r == "mark-veto+value" else None
        rules[r] = rule_rec(o, own, extra=extra)
    part = {}
    for r, rr in rules.items():
        if rr["outcome"] == "strict":
            part[r] = "strict_own" if rr["cited_pointer_is_own"] else "strict_other"
        elif rr["outcome"] == "valid":
            part[r] = "valid_wrong_value"
        elif rr["outcome"] == "invalid":
            part[r] = "invalid"
        else:
            part[r] = "model_abstention" if raw_kind == "unknown" else ("veto_abstention" if r == "mark-veto" else "rule_abstention")
    rec = {"index": i, "block": block, "entity": f"Tone-{i:04x}", "query": f"what is Tone-{i:04x}?", "own_pointer": ("own" if block in ("planted", "cross_schema") else None),
           "own_in_window": own_in_window if block in ("planted", "cross_schema") else None, "mark_own": mark_own if own_in_window else None, "mark_max": mark_max,
           "raw_kind": raw_kind, "raw_label": raw_label, "window": [{"label": "A", "pointer": "own" if own_in_window else "other2", "mark": mark_own}, {"label": "B", "pointer": "other", "mark": 0.5}],
           "short_window": False, "own_in_candidates": own_in_window, "own_rank": 1 if own_in_window else None,
           "whole_store": {"own_rank": 1 if own_in_window else 2, "own_cosine": 0.9, "best_other_cosine": 0.5 if own_in_window else 0.95, "compared": 200} if block == "planted" else None,
           "rules": rules, "partition": part}
    if seed is not None:
        rec["seed"] = seed
    if block == "near_name":
        rec.update({"neighbour_in_window": True, "neighbour_mark_at_or_above_threshold": True})
    return rec


def make_pass(index, form, run_id, strict_own_rate: float, invalid_rate: float = 0.0, never_claim_rate: float = 0.0, model_own_rate: float | None = None,
              oracle_own_rate: float = 0.9, n_planted=200, n_never=50, extra_never=0, cross=50, near=50):
    """A pass whose `mark-veto`, V1 and plain strict rates on the own line are chosen; the oracle cites the own line at its rate."""
    answers = []
    i = 0
    for k in range(n_planted):
        strict = k < round(strict_own_rate * n_planted)
        invalid = (not strict) and (k - round(strict_own_rate * n_planted)) < round(invalid_rate * n_planted)
        model_own = (k % 10) < round((model_own_rate if model_own_rate is not None else strict_own_rate) * 10)
        oracle_own = (k % 10) < round(oracle_own_rate * 10)
        o = ("strict", True) if strict else (("invalid", False) if invalid else ("abstain", False))
        outcomes = {"plain": ("strict", True) if model_own else ("valid", False), "mark-veto": o, "mark-veto+value": ("abstain", False), "V1": o, "V1+organ-value": o,
                    "oracle": ("strict", True) if oracle_own else ("valid", False), "oracle+organ-value": ("abstain", False)}
        answers.append(record(i, "planted", outcomes, own_in_window=True)); i += 1
    for k in range(n_never):
        claim = k < round(never_claim_rate * n_never)
        o = ("invalid", False) if claim else ("abstain", False)
        answers.append(record(i, "never_planted", {r: o for r in SR.RULES}, own_in_window=False, seed=30_001, mark_max=0.2)); i += 1
    for k in range(extra_never):
        answers.append(record(i, "never_planted", {r: ("abstain", False) for r in SR.RULES}, own_in_window=False, seed=30_003, mark_max=0.2)); i += 1
    if form == "direct":
        for k in range(cross):
            answers.append(record(i, "cross_schema", {r: ("abstain", False) for r in SR.RULES})); i += 1
        for k in range(near):
            answers.append(record(i, "near_name", {r: ("abstain", False) for r in SR.RULES}, own_in_window=False)); i += 1
    return {"family": "A", "index": index, "form": form, "checkpoint": {"run_id": run_id}, "answers": answers, "n_questions": len(answers), "run_in_sessions": 1,
            "aggregates": {}, "blocks": {}}


def write_family(md: Path, family: str, run_id: str, passes: dict, threshold: float = 0.6):
    meta = {}
    for (index, form), p in passes.items():
        name = f"sealed-{family}-{run_id}-{index}-{form}.pass.json"
        p = dict(p); p["family"] = family
        (md / name).write_text(json.dumps(p) + "\n", encoding="utf-8", newline="\n")
        meta[f"{index}-{form}"] = {"file": name, "sha256_lf": RS.sha256_file(md / name, lf=True)}
    fam = {"family": family, "run_id": run_id, "threshold": {"threshold": threshold}, "passes": meta}
    (md / f"sealed-{family}-{run_id}.json").write_text(json.dumps(fam) + "\n", encoding="utf-8", newline="\n")


def test_counts_conditions_and_factors_on_a_synthetic_pass():
    p = make_pass("I0", "direct", "r1", strict_own_rate=0.6, invalid_rate=0.02, never_claim_rate=0.2, extra_never=150)
    on = SR.records(p, "planted")
    c = SR.counts(on, "mark-veto")
    assert c["n"] == 200 and c["strict_own"] == 120 and c["invalid"] == 4 and c["claims"] == 124 and c["abstain"] == 76
    assert c["coverage"]["rate"] == 124 / 200 and c["selective_risk"]["rate"] == 4 / 124 and c["strict_recall_own"]["rate"] == 0.6
    assert c["partition"] == {"strict_own": 120, "invalid": 4, "veto_abstention": 76}
    cond = SR.conditions(p, "mark-veto")
    assert cond["own_line_gap"]["rate"] == 0.6 >= DELTA and cond["gap"]["rate"] == 0.6 and not cond["holds"] and not cond["holds_with_own_line_gap"]
    assert any("invalid citations" in b for b in cond["broken"]) and any("never planted" in b for b in cond["broken"])
    # the frozen gap (any line) against the own line gap: a pass strict on other lines holds the frozen four and not the own line gap
    other = make_pass("I0", "direct", "r1", strict_own_rate=0.3)
    for a in other["answers"][60:120]:                                           # sixty strict answers citing another line
        if a["block"] == "planted":
            a["rules"]["mark-veto"] = rule_rec("strict", False); a["partition"]["mark-veto"] = "strict_other"
    c2 = SR.conditions(other, "mark-veto")
    assert c2["gap"]["rate"] == 0.6 and c2["own_line_gap"]["rate"] == 0.3 and c2["holds"] and not c2["holds_with_own_line_gap"]
    assert any("own line gap" in b for b in c2["broken_with_own_line_gap"]) and c2["broken"] == []
    assert cond["never_planted"]["first_seed"]["seed"] == 30_001 and cond["never_planted"]["first_seed"]["claims"] == 10 and cond["never_planted"]["all"]["n"] == 200
    assert cond["off_arm"].startswith("the off arm's strict recall is zero by construction")
    good = make_pass("I0", "direct", "r1", strict_own_rate=0.6, invalid_rate=0.0, never_claim_rate=0.0)
    assert SR.conditions(good, "mark-veto")["holds"] and not SR.conditions(good, "mark-veto+value")["holds"]      # +value abstains on A by construction
    f = SR.factors(on, "mark-veto")
    assert f["own_in_window"]["rate"] == 1.0 and f["own_cited_given_in_window"]["rate"] == 0.6 and f["exact_given_own_cited"]["rate"] == 1.0 and f["product"] == pytest.approx(0.6)
    ml = SR.marks_ledger(p, 0.95)
    assert ml["own_line_marks_below_threshold"]["k"] == 200 and ml["never_planted_max_at_or_above"]["all"]["k"] == 0
    assert SR.miss_classes(on)["n_misses"] == 0
    v = SR.veto_ledger(p)
    assert v["citations_refused_own_line"] + v["citations_refused_other_line"] + v["citations_of_a_label_outside_the_window"] == 76 and v["model_abstentions_own_present"] == 0
    assert v["malformed_outputs"] == 0 and "add up" in v["what"]
    assert SR.m3_exactness(on)["rate"] == 1.0
    b = SR.block_report(p, "near_name", 0.6)
    assert b["n"] == 50 and b["enters_a_letter"] is False and b["neighbour_in_window"]["rate"] == 1.0 and b["by_rule"]["V1"]["claims"] == 0


def test_miss_classes_use_the_frozen_classifier():
    recs = [record(0, "planted", {"plain": ("abstain", False)}, own_in_window=False)]
    recs[0]["own_in_candidates"] = False; recs[0]["whole_store"] = {"own_rank": 1, "own_cosine": 0.98, "best_other_cosine": 0.3, "compared": 200}
    m = SR.miss_classes(recs)
    assert m["n_misses"] == 1 and m["not_in_candidates"] == 1 and m["index"] == 1
    recs[0]["whole_store"] = {"own_rank": 3, "own_cosine": 0.5, "best_other_cosine": 0.9, "compared": 200}
    assert SR.miss_classes(recs)["interference"] == 1
    recs[0]["own_in_candidates"] = True
    assert SR.miss_classes(recs)["in_candidates_not_in_window"] == 1 and SR.miss_classes(recs)["unclassed"] == 1


def test_m2_pairs_and_holm_across_families(tmp_path):
    md = tmp_path
    for fam in ("A", "Av", "T0"):
        for run_id in SR.RUN_IDS:
            passes = {}
            for form in SR.FORMS:
                for index in SR.INDEXES:
                    # on A the model cites the own line on 9 of 10, the oracle on 5 of 10: the model better; on Av equal; on T0 the oracle better
                    mo, oo = {"A": (0.9, 0.5), "Av": (0.7, 0.7), "T0": (0.3, 0.9)}[fam]
                    passes[(index, form)] = make_pass(index, form, run_id, strict_own_rate=0.7, model_own_rate=mo, oracle_own_rate=oo, extra_never=(150 if fam == "A" else 0),
                                                      n_never=50)
            write_family(md, fam, run_id, passes)
    reading = SR.read_rows(md)
    m = reading["M"]
    assert set(m["families"]) == {"A", "Av", "T0"} and set(m["holm"]["pvalues"]) == {"A", "Av", "T0"}
    a = m["families"]["A"]["M2"]
    assert a["pooled_live_seeds"]["n_reads_with_own_in_window"] == 800 and a["pooled_live_seeds"]["b_model_only"] > a["pooled_live_seeds"]["c_oracle_only"]
    assert a["separated_under_holm"] and a["direction"] == "model" and a["sentence"] == SR.M2_SENTENCES["model"]
    av = m["families"]["Av"]["M2"]
    assert av["pooled_live_seeds"]["b_model_only"] == av["pooled_live_seeds"]["c_oracle_only"] == 0 and not av["separated_under_holm"] and "resolution" in av
    assert av["sentence"] == SR.M2_SENTENCES["organ"] and "Holm over the 3 families" in av["resolution"] and av["a0_ceiling"] is None
    assert SR.discordant_needed(0.05) == 6 and SR.discordant_needed(0.025) == 7 and SR.discordant_needed(0.05 / 3) == 7
    t0 = m["families"]["T0"]["M2"]
    assert t0["direction"] == "oracle" and t0["sentence"] == SR.M2_SENTENCES["organ"]
    assert m["families"]["A"]["M2"]["third_checkpoint"]["n_reads_with_own_in_window"] == 400
    assert set(m["families"]["A"]["M4_veto_ledger"]) == {f"I0-{f}-{r}" for f in SR.FORMS for r in SR.RUN_IDS}


def test_letters_a_and_r_on_synthetic_families(tmp_path):
    md = tmp_path
    # A: the organ's stack holds the four conditions on every cell; I2 V1 and I1 V1 fail the gap on a live seed; both tests favour the organ: R-a
    for run_id in SR.RUN_IDS:
        passes = {}
        for form in SR.FORMS:
            passes[("I0", form)] = make_pass("I0", form, run_id, strict_own_rate=0.8, extra_never=150)
            for index in ("I1", "I2"):
                passes[(index, form)] = make_pass(index, form, run_id, strict_own_rate=(0.4 if run_id in SR.LIVE_RUN_IDS else 0.45), extra_never=150)
        write_family(md, "A", run_id, passes)
    reading = SR.read_rows(md)
    la = reading["letter_A"]
    assert la["letter"] == "A1" and la["cells"] == 6 and la["cells_holding"] == 6 and la["claimed_forms"] == ["direct", "paraphrase"] and la["not_read"] is None
    assert reading["first_family"]["direct"] == {} and reading["first_family"]["paraphrase"] == {} and "departures_stated" in reading
    assert la["sentence"] == SR.A1_SENTENCE and la["oracle_organ_value_strict_on_A"] and all(v == 0 for v in la["oracle_organ_value_strict_on_A"].values())
    R = reading["R"]
    assert R["letter"] == "R-a" and all(t["outcome"] == "the organ's stack better" for t in R["primary_tests"]["tests"].values())
    t = R["primary_tests"]["tests"]["I0 mark-veto against I2 V1"]
    assert t["live_seeds_pooled"]["pairs"] == 400 and t["live_seeds_pooled"]["b_organ_only"] == 160 and t["live_seeds_pooled"]["c_reference_only"] == 0 and t["third_agrees"]
    assert R["near_name_block_enters"] is False and "index_at_fixed_rule" in R["attribution"]
    fp = reading["factor_product"]
    assert "A against Av" in fp["differences"] and fp["label"].startswith("the hybrid")
    assert reading["near_name_block"]["A"] and reading["cross_schema_block"]["A"]
    # A2 and R-c: one cell fails (the paraphrase form on a live seed), the references equal to the organ
    md2 = tmp_path / "two"; md2.mkdir()
    for run_id in SR.RUN_IDS:
        passes = {}
        for form in SR.FORMS:
            for index in SR.INDEXES:
                rate = 0.3 if (form == "paraphrase" and run_id == SR.LIVE_RUN_IDS[0] and index == "I0") else 0.8
                passes[(index, form)] = make_pass(index, form, run_id, strict_own_rate=rate, extra_never=150)
        write_family(md2, "A", run_id, passes)
    r2 = SR.read_rows(md2)
    assert r2["letter_A"]["letter"] == "A2" and r2["letter_A"]["claimed_forms"] == ["direct"] and len(r2["letter_A"]["failing_cells"]) == 1
    fc = next(iter(r2["letter_A"]["failing_cells"].values()))
    assert set(fc) >= set(SR.COMPONENTS) and fc["broken"]
    assert r2["R"]["letter"] in ("R-b", "R-c")
    for t in r2["R"]["primary_tests"]["tests"].values():
        assert t["outcome"] == "not separated at this size" and t["resolution"] == SR.RESOLUTION_SENTENCE
    # A3: every cell fails
    md3 = tmp_path / "three"; md3.mkdir()
    for run_id in SR.RUN_IDS:
        write_family(md3, "A", run_id, {(i, f): make_pass(i, f, run_id, strict_own_rate=0.2, extra_never=150) for i in SR.INDEXES for f in SR.FORMS})
    r3 = SR.read_rows(md3)
    assert r3["letter_A"]["letter"] == "A3" and r3["R"]["letter"] == "R-b" and r3["R"]["note"] and "vacuity" in r3["R"]["note"]
    # a missing cell: no letter
    md4 = tmp_path / "four"; md4.mkdir()
    for run_id in SR.LIVE_RUN_IDS:
        write_family(md4, "A", run_id, {(i, f): make_pass(i, f, run_id, strict_own_rate=0.8, extra_never=150) for i in SR.INDEXES for f in SR.FORMS})
    r4 = SR.read_rows(md4)
    assert r4["letter_A"]["letter"] is None and "missing" in r4["letter_A"]["not_read"]


def test_load_family_refuses_a_pass_file_whose_hash_moved(tmp_path):
    write_family(tmp_path, "A", "r1", {(i, f): make_pass(i, f, "r1", 0.5) for i in SR.INDEXES for f in SR.FORMS})
    L = SR.load_family(tmp_path, "A", "r1")
    assert set(L["passes"]) == {(i, f) for i in SR.INDEXES for f in SR.FORMS} and L["threshold"] == 0.6
    p = tmp_path / "sealed-A-r1-I0-direct.pass.json"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        SR.load_family(tmp_path, "A", "r1")


@pytest.mark.skipif(not list(MQ.glob("hm-lm-*-session-b-index16exact.json")), reason="the record's row 47 files are needed")
def test_first_family_direct_from_the_record_row47_file():
    p = sorted(MQ.glob("hm-lm-*-session-b-index16exact.json"))[0]
    f = SR.first_family_direct(json.loads(p.read_text(encoding="utf-8")))
    assert f["n"] == 200 and f["label"] == SR.FIRST_FAMILY_LABEL
    fp = f["factors_plain"]
    assert fp["own_in_window"]["n"] == 200 and 0.0 <= fp["product"] <= 1.0 and fp["strict_own"]["k"] == round(fp["product"] * 200)
    assert f["mark_veto_replayed"]["strict_recall_own"]["k"] <= f["plain"]["strict_recall_own"]["k"]       # the veto only removes claims
    assert f["mark_veto_replayed"]["by"].startswith("the frozen replay_outcomes")
