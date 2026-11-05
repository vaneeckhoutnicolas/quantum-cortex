"""The provenance benchmark module (the roadmap's decision of 2026-10-14, point (a); declared 2026-11-05 in
docs/working/PROVENANCE-BENCH-DECLARATION-2026-11-05.md before this code, amended the same day after its committee): a bench
built from the sealed generator, an answers file scored with the record's reading of rows 48 to 50 and the record's frozen
conditions, no model and no organ. Tests 1 to 13 are the declaration's, in its order; 14 to 16 pin the module's own copies
against the record's functions (written with the code they pin, a stated departure from tests first); 17 to 22 are the
committee's additions (the repeated keys, a dropped neighbour, the family blocks scored, the answers file's rules, the bench's
integrity, the n rule)."""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cortex_c2b import provenance_bench as PB       # noqa: E402
from cortex_c2b import content_hash                 # noqa: E402
from cortex_c2b import sealed_families as S         # noqa: E402

SEED, N = 7001, 200                                  # the scale draw for the tests: ten E1, ten E2 at most, one thousand never planted


@pytest.fixture(scope="module")
def bench():
    return PB.build_bench(seed=SEED, n=N, e4=True, families=False)


@pytest.fixture(scope="module")
def bench_families():
    return PB.build_bench(seed=SEED, n=N, e4=False, families=True)


@pytest.fixture(scope="module")
def fams():
    return S.generate_all()


def planted_questions(b):
    return [q for q in b["questions"] if q["block"] in PB.PLANTED_BLOCKS]


def perfect_answers(b, key=None):
    key = key or b["key"]
    out = {}
    for q in b["questions"]:
        k = key[q["id"]]
        if k["designated"] is not None:
            out[q["id"]] = {"value": k["value"], "cited": k["designated"]}
        else:
            out[q["id"]] = {"value": None, "cited": None}
    return {"system": "the key itself", "answers": out}


def first_statements(b):
    """The first statement written about each (entity, schema), over every kind (the declaration's designated line)."""
    first = {}
    for s in b["statements"]:
        first.setdefault((s["entity"], s["schema"]), s["pointer"])
    return first


# 1
def test_build_is_deterministic(tmp_path):
    a = PB.build_bench(seed=SEED, n=N, e4=True); b = PB.build_bench(seed=SEED, n=N, e4=True); c = PB.build_bench(seed=SEED + 1, n=N, e4=True)
    strip = lambda d: {k: v for k, v in d.items() if k != "construction"} | {"construction": {k: v for k, v in d["construction"].items() if k != "time_utc"}}
    assert strip(a) == strip(b) and a["construction"]["content_sha256"] == b["construction"]["content_sha256"]
    assert a["construction"]["content_sha256"] != c["construction"]["content_sha256"]
    pa, pb_ = tmp_path / "a.json", tmp_path / "b.json"
    PB.write_json(pa, a); PB.write_json(pb_, b)
    ta, tb = pa.read_text(encoding="utf-8"), pb_.read_text(encoding="utf-8")
    assert ta.replace(a["construction"]["time_utc"], "T") == tb.replace(b["construction"]["time_utc"], "T")
    assert a["construction"]["content_sha256"] == PB.content_sha256_of(a)                       # the hash is the statements' and the questions'


# 2
def test_pointers_are_the_organs_content_hashes(bench):
    ps = [s["pointer"] for s in bench["statements"]]
    assert len(set(ps)) == len(ps)
    for s in bench["statements"]:
        assert s["pointer"] == content_hash(s["text"].encode("utf-8")) and len(s["pointer"]) == 16


# 3
@pytest.mark.parametrize("which", ["bench", "bench_families"])
def test_the_key_designates_the_first_statement_and_the_supporting_set(request, which):
    b = request.getfixturevalue(which)
    first = first_statements(b)
    by_pointer = {s["pointer"]: s for s in b["statements"]}
    seen = {"never_planted": 0, "near_name": 0, "cross_schema": 0, "planted_like": 0}
    for q in b["questions"]:
        k = b["key"][q["id"]]
        if q["block"] in PB.CLAIM_BLOCKS:
            assert k["designated"] is None and k["supporting"] == [] and k["value"] is None
            seen[q["block"]] += 1
            continue
        seen["planted_like"] += 1
        assert k["designated"] == first[(k["entity"], k["schema"])] == k["supporting"][0]
        kinds = {by_pointer[p]["kind"] for p in k["supporting"]}
        assert kinds <= {"fact", "E4", "family"}
        for p in k["supporting"]:
            assert by_pointer[p]["entity"] == k["entity"] and by_pointer[p]["schema"] == k["schema"] and by_pointer[p]["value"] == k["value"]
        if k["partner"] is not None:
            assert by_pointer[k["partner"]].get("partner_of") == k["designated"] and k["partner"] not in k["supporting"] or by_pointer[k["partner"]]["kind"] == "E4"
    if which == "bench":
        e4 = [q for q in b["questions"] if q["block"] == "contested_E4"]
        assert e4 and all(len(b["key"][q["id"]]["supporting"]) == 2 for q in e4) and seen["planted_like"] == N
    else:
        assert seen["near_name"] == 150 and seen["cross_schema"] == 150 and seen["never_planted"] == 1000 + 300 and seen["planted_like"] == N + 1200


# 4
def test_blocks_and_rates_match_the_declared_construction(bench, bench_families, fams):
    plan = S.scale_plan(SEED, N, SEED + 100)
    c = bench["construction"]
    kinds = {}
    for s in bench["statements"]:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    assert kinds["fact"] == N and kinds["E1"] == plan["partners"]["E1"] and kinds["E2"] == plan["partners"]["E2"] and kinds["E4"] == N
    assert c["e2_dropped"] == plan["partners"]["E2_dropped"] == c["e2_dropped_seen"] == 0 and c["generator_hash"] == plan["generator_hash"] and c["plan_content_sha256"] == plan["content_sha256"]
    assert c["repeated_keys"] == [] and c["versions"]["numpy"] and c["counts"]["questions"] == len(bench["questions"])
    blocks = {}
    for q in bench["questions"]:
        blocks[q["block"]] = blocks.get(q["block"], 0) + 1
    assert blocks["never_planted"] == S.SCALE_NEVER_PLANTED_N and blocks["contested_E1"] == plan["partners"]["E1"] and blocks["contested_E2"] == plan["partners"]["E2"]
    assert sum(blocks.get(b, 0) for b in ("planted", "contested_E1", "contested_E2", "contested_E4")) == N        # one direct question per fact, read in one block; with --e4 none is "planted"
    assert "planted" not in blocks
    fb = {}
    for q in bench_families["questions"]:
        fb[(q.get("family"), q["block"])] = fb.get((q.get("family"), q["block"]), 0) + 1
    seal = json.loads((ROOT / S.SEAL_PATH).read_text(encoding="utf-8"))
    for name in ("A", "T0", "Av"):
        assert fb[(name, "family")] == fams[name]["n_facts"] == 200 and fb[(name, "family_paraphrase")] == 200
        assert fb[(name, "never_planted")] == fams[name]["n_never_planted"] == (200 if name == "A" else 50) and fb[(name, "cross_schema")] == fams[name]["n_cross_schema"] == 50
        assert fb[(name, "near_name")] == 50 == bench_families["construction"]["families"][name]["n_near_name"]                 # the frozen plan's ten per schema, as sealed
        assert bench_families["construction"]["families"][name]["content_sha256"] == seal["families"][name]["content_sha256"]
    assert bench_families["construction"]["counts"]["by_block"]["never_planted"] == 1300
    assert "fifty for T0 and Av, two hundred for A" in PB.BLOCK_WHAT["never_planted"]


# 5
def test_a_perfect_answers_file_scores_one_everywhere(bench):
    t = PB.score(bench, perfect_answers(bench))
    p = t["pooled_planted"]
    assert p["coverage"]["rate"] == 1.0 and p["value_correct"]["rate"] == 1.0 and p["citation_supported"]["rate"] == 1.0 and p["designated_line"]["rate"] == 1.0
    assert p["strict"]["rate"] == 1.0 and p["citation_valid"]["rate"] == 1.0 and p["citation_supported_answered"]["rate"] == 1.0
    assert p["attributed_accuracy"]["rate"] == 1.0 and p["abstentions"]["false"] == 0 and p["citation_invalid"] == {"n": 0, "d": N, "rate": 0.0}
    assert t["blocks"]["never_planted"]["claims"]["n"] == 0 and t["blocks"]["never_planted"]["abstentions"]["verified"] == S.SCALE_NEVER_PLANTED_N
    assert t["conditions"]["holds"] is True and all(t["conditions"][k]["holds"] for k in ("gap", "invalid_citations", "never_planted_claims", "attributed_accuracy_gap"))
    assert t["conditions"]["holds_with_attributed_accuracy_gap"] is True and t["conditions"]["off_arm"]["measured"] is False
    assert t["conditions"]["gap"]["rate"] == p["strict"] and t["conditions"]["invalid_citations"]["rate"]["d"] == N          # the frozen two over the planted questions


# 6
def test_an_all_abstaining_file(bench):
    t = PB.score(bench, {"system": "silence", "answers": {}})
    p = t["pooled_planted"]
    assert p["coverage"]["rate"] == 0.0 and p["abstentions"]["false"] == len(planted_questions(bench)) and p["abstentions"]["verified"] == 0
    assert t["blocks"]["never_planted"]["abstentions"]["verified"] == S.SCALE_NEVER_PLANTED_N
    assert t["conditions"]["gap"]["holds"] is False and t["conditions"]["invalid_citations"]["holds"] is True and t["conditions"]["never_planted_claims"]["holds"] is True
    assert t["conditions"]["holds"] is False and t["conditions"]["holds_with_attributed_accuracy_gap"] is False
    assert p["citation_supported"]["rate"] is None and p["citation_invalid"] == {"n": 0, "d": N, "rate": 0.0}


# 7
def test_contested_reads_are_read_apart(bench):
    answers = {}
    for q in bench["questions"]:
        k = bench["key"][q["id"]]
        if q["block"].startswith("contested_"):
            part = next(s for s in bench["statements"] if s["pointer"] == k["partner"])
            assert part.get("partner_of") == k["designated"] and part["kind"] == q["block"][-2:]
            answers[q["id"]] = {"value": part["value"], "cited": part["pointer"]}
        elif k["designated"] is not None:
            answers[q["id"]] = {"value": k["value"], "cited": k["designated"]}
    t = PB.score(bench, {"system": "cites the partner", "answers": answers})
    e1, e2, e4 = (t["blocks"][f"contested_{k}"] for k in ("E1", "E2", "E4"))
    # the E1 line carries the value it answered, another entity's value than the key's: valid, wrong, unsupported under 4 (b), supported under section 7's wording
    assert e1["cites_partner"]["n"] == e1["n"] and e1["value_correct"]["n"] == 0 and e1["citation_valid"]["n"] == e1["n"] and e1["citation_invalid"]["n"] == 0
    assert e1["citation_supported"]["n"] == 0 and e1["citation_supported_answered"]["n"] == e1["n"] and e1["strict"]["n"] == 0 and e1["attributed_accuracy"]["n"] == 0
    # the E2 line is another name's with the same value: valid and strict under the record's frozen reading, unsupported under both of section 7's
    assert e2["cites_partner"]["n"] == e2["n"] and e2["value_correct"]["n"] == e2["n"] and e2["citation_valid"]["n"] == e2["n"] and e2["strict"]["n"] == e2["n"]
    assert e2["citation_supported"]["n"] == 0 and e2["citation_supported_answered"]["n"] == 0 and e2["attributed_accuracy"]["n"] == 0
    # the E4 paraphrase supports the answer and is not the designated line
    assert e4["cites_partner"]["n"] == e4["n"] and e4["value_correct"]["n"] == e4["n"] and e4["citation_supported"]["n"] == e4["n"] and e4["designated_line"]["n"] == 0
    assert e4["strict"]["n"] == e4["n"] and e4["attributed_accuracy"]["n"] == e4["n"]
    for b in (e1, e2, e4):
        lo, hi = b["cites_partner"]["clopper_pearson_95"]
        assert lo < 1.0 <= hi and b["abstains"]["n"] == 0 and b["cites_designated"]["n"] == 0
    assert t["conditions"]["gap"]["rate"]["n"] == e2["n"] + e4["n"] and t["conditions"]["attributed_accuracy_gap"]["rate"]["n"] == e4["n"]


# 8
def test_claims_on_never_planted_names_are_counted(bench):
    answers = {q["id"]: {"value": "anything", "cited": None} for q in bench["questions"] if q["block"] == "never_planted"}
    answers.update(perfect_answers(bench)["answers"] | answers)
    t = PB.score(bench, {"system": "claims", "answers": answers})
    nb = t["blocks"]["never_planted"]
    assert nb["claims"]["n"] == nb["n"] == S.SCALE_NEVER_PLANTED_N and t["conditions"]["never_planted_claims"]["holds"] is False and t["conditions"]["holds"] is False
    assert t["conditions"]["never_planted_claims"]["rate"] == {"n": 1000, "d": 1000, "rate": 1.0}


# 9
def test_invalid_citations(bench):
    qs = planted_questions(bench)[:4]
    k0, k1, k2, k3 = (bench["key"][q["id"]] for q in qs)
    other = next(s["pointer"] for s in bench["statements"] if s["kind"] == "fact" and s["entity"] != k2["entity"] and s["value"] != k2["value"])
    same_value_other_entity = next((s["pointer"] for s in bench["statements"] if s["entity"] != k3["entity"] and s["value"] == k3["value"]), None)
    answers = {qs[0]["id"]: {"value": k0["value"], "cited": None},               # a value without a pointer: invalid
               qs[1]["id"]: {"value": k1["value"], "cited": "0" * 16},           # a pointer the bench does not hold: invalid
               qs[2]["id"]: {"value": k2["value"], "cited": other}}              # a held line not carrying the answered value: invalid (the record's definition), unsupported
    if same_value_other_entity:
        answers[qs[3]["id"]] = {"value": k3["value"], "cited": same_value_other_entity}   # a held line carrying the value, another entity's: valid and strict, unsupported
    t = PB.score(bench, {"system": "x", "answers": answers})
    p = t["pooled_planted"]
    n_ans = 3 + bool(same_value_other_entity)
    assert p["coverage"]["n"] == n_ans and p["citation_invalid"]["n"] == 3 and p["citation_supported"]["n"] == 0 and p["value_correct"]["n"] == n_ans and p["attributed_accuracy"]["n"] == 0
    assert p["strict"]["n"] == bool(same_value_other_entity) and p["citation_valid"]["n"] == bool(same_value_other_entity) and p["citation_invalid"]["d"] == N
    assert t["conditions"]["invalid_citations"]["holds"] is False                       # 3 of 200 is above 0.01


# 10
def test_answers_file_shapes(bench):
    pub, key = PB.split(bench)
    assert "key" not in pub and set(key["key"]) == {q["id"] for q in bench["questions"]} and key["content_sha256"] == bench["construction"]["content_sha256"]
    assert all(set(s) == {"id", "pointer", "text"} for s in pub["statements"]) and all(set(q) == {"id", "text"} for q in pub["questions"])
    q0 = planted_questions(bench)[0]; k0 = bench["key"][q0["id"]]
    answers = {q0["id"]: {"value": None, "cited": k0["designated"]}, "no-such-question": {"value": "v", "cited": None}}
    t = PB.score(pub, {"system": "x", "answers": answers}, key=key)
    assert t["pooled_planted"]["coverage"]["n"] == 0 and t["pooled_planted"]["abstentions"]["false"] == len(planted_questions(bench))
    assert t["answers_file"]["unknown_question_ids"] == ["no-such-question"] and t["answers_file"]["answered"] == 0 and t["answers_file"]["abstentions_with_a_citation"] == 1
    with pytest.raises(ValueError, match="no key"):
        PB.score(pub, {"system": "x", "answers": {}})                                 # no key: refused
    other = PB.build_bench(seed=SEED + 1, n=N, e4=True)
    with pytest.raises(ValueError, match="not this bench"):
        PB.score(pub, {"system": "x", "answers": {}}, key=PB.split(other)[1])        # another bench's key: refused
    t = PB.score(bench, {"system": "x", "answers": {}, "bench_content_sha256": bench["construction"]["content_sha256"]})
    assert t["answers_file"]["answered"] == 0
    with pytest.raises(ValueError, match="another bench"):
        PB.score(bench, {"system": "x", "answers": {}, "bench_content_sha256": other["construction"]["content_sha256"]})


# 11
def test_cli_writes_once(tmp_path, bench, capsys):
    out = tmp_path / "bench.json"
    PB._cli(["build", "--seed", str(SEED), "--n", str(N), "--e4", "--out", str(out)])
    assert out.exists()
    with pytest.raises(SystemExit, match="never overwritten"):
        PB._cli(["build", "--seed", str(SEED), "--n", str(N), "--e4", "--out", str(out)])
    ans = tmp_path / "answers.json"; ans.write_text(json.dumps(perfect_answers(json.loads(out.read_text(encoding="utf-8")))), encoding="utf-8")
    table = tmp_path / "table.json"
    capsys.readouterr()
    PB._cli(["score", "--bench", str(out), "--answers", str(ans), "--out", str(table)])
    printed = capsys.readouterr().out
    t = json.loads(table.read_text(encoding="utf-8"))
    assert t["bench"]["content_sha256"] == bench["construction"]["content_sha256"] and t["answers_file"]["sha256"] == hashlib.sha256(ans.read_bytes()).hexdigest()
    assert t["module"]["sha256_lf"] == hashlib.sha256((ROOT / "cortex_c2b" / "provenance_bench.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]
    assert t["answers_file"]["sha256"][:16] in printed and t["module"]["sha256_lf"] in printed and "holds True" in printed
    assert t["pooled_planted"]["value_correct"]["rate"] == 1.0
    with pytest.raises(SystemExit, match="never overwritten"):
        PB._cli(["score", "--bench", str(out), "--answers", str(ans), "--out", str(table)])
    with pytest.raises(SystemExit, match="refused: n must be a multiple of five"):
        PB._cli(["build", "--seed", "11", "--n", "23", "--out", str(tmp_path / "x.json")])
    assert not (tmp_path / "x.json").exists()
    bad = tmp_path / "bad.json"; bad.write_text('{"answers": {"q000000": {"value": "a", "cited": null}, "q000000": {"value": "b", "cited": null}}}', encoding="utf-8")
    with pytest.raises(SystemExit, match="appears twice"):
        PB._cli(["score", "--bench", str(out), "--answers", str(bad), "--out", str(tmp_path / "t2.json")])
    notjson = tmp_path / "notjson.txt"; notjson.write_text("hello", encoding="utf-8")
    with pytest.raises(SystemExit, match="cannot be read as JSON"):
        PB._cli(["score", "--bench", str(out), "--answers", str(notjson), "--out", str(tmp_path / "t3.json")])
    assert not (tmp_path / "t2.json").exists() and not (tmp_path / "t3.json").exists()


# 12
def test_no_model_in_the_import_graph():
    import subprocess
    code = ("import sys, importlib; importlib.import_module('cortex_c2b.provenance_bench'); "
            "bad = [m for m in ('torch', 'cortex_c2b.hm_lm', 'cortex_c2b.lm_bridge', 'cortex_c2b.sealed_family') if m in sys.modules]; print(bad)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"


# 13
@pytest.mark.parametrize("arm", ["E1", "E2", "E4"])
def test_the_scorer_equals_the_records_reading_on_a_records_file(arm):
    """On row 49's identity files of one checkpoint (the writes carry `truth`, the answers `query` and `answered`): the bench's
    coverage, strict share (the record's `value_correct`), citation support under 4 (b) on the claims, designated line and
    attributed accuracy equal `provenance_metrics`'; the forced choice error is compared where the record's contested set is
    the whole block (the record reads it by the window, the bench by construction); the abstention split is not compared."""
    from cortex_c2b import near_duplicate_reading as NR
    p = ROOT / "metrics" / "mqar" / f"hm-lm-4cfd6f6e1406-near-dup-{arm}-identity-session-b-index16exact-mark-veto-value.json"
    if not p.exists():
        pytest.skip("the row 49 file is not in this tree")
    f = json.loads(p.read_text(encoding="utf-8"))
    b, answers = PB.bench_from_record_file(f)
    t = PB.score(b, answers)
    pm = NR.provenance_metrics(f)
    pp = t["pooled_planted"]
    rows = f["answers"]["on"]; truth = NR._truth_of(f)
    assert pp["n"] == pm["n"] == 200
    assert pp["coverage"]["rate"] == pm["coverage"] and pp["strict"]["rate"] == pm["value_correct"]
    assert pp["citation_supported"]["n"] == sum(1 for a in rows if a.get("kind") != "unknown" and NR.citation_support_of(a, truth) is True)
    assert pp["citation_supported_answered"]["n"] == sum(1 for a in rows if a.get("kind") != "unknown" and a.get("citation_support_answered") is True)
    assert pp["attributed_accuracy"]["rate"] == pm["attributed_accuracy"] and pp["designated_line"]["n"] == pm["designated_line_cited"]
    assert pp["citation_invalid"]["n"] == sum(1 for a in rows if a.get("outcome") == "invalid")
    blk = t["blocks"][f"contested_{arm}"]
    if pm["contested"]["n"] == blk["n"]:
        assert blk["cites_partner"]["n"] == pm["contested"]["forced_choice_error"]
    assert pm["citation_supported"] >= pp["citation_supported"]["n"]                       # the record counts the support on an abstention's surviving citation too
    assert all(v is None or isinstance(v, str) for v in (a["value"] for a in answers["answers"].values()))


# 14 to 16: the module's own copies against the record's functions (the bench imports nothing of the model, so it carries three
# constructions of its own, pinned here against the originals, which the tests may import)
def test_the_exact_interval_equals_the_records():
    from cortex_c2b.sealed_reading import clopper_pearson as cp_record
    for n in (1, 2, 7, 50, 200, 1000):
        for k in sorted({0, 1, n // 3, n // 2, n - 1, n}):
            assert PB.clopper_pearson(k, n) == list(cp_record(k, n))
    assert PB.clopper_pearson(0, 0) is None


def test_the_family_facts_and_near_name_block_equal_the_runners(fams):
    from cortex_c2b import sealed_family as SF
    protected = PB._near_name_protected(fams)
    for name in ("T0", "A", "Av"):
        a, b = PB._facts_of(fams[name]), SF.facts_of(fams[name])
        assert [(x.entity, x.schema, x.attr, x.statement, x.query, x.paraphrase, x.address_text) for x in a] == [(x.entity, x.schema, x.attr, x.statement, x.query, x.paraphrase, x.address_text) for x in b]
        assert all((x.cue == y.cue).all() for x, y in zip(a, b))
        nb = SF.near_name_block(fams[name], fams)
        assert [(q["entity"], q["schema"], q["query_asked"]) for q in PB._near_name_block(fams[name], fams, protected)] == [(q["entity"], q["schema"], q["query_asked"]) for q in nb["questions"]]


def test_the_invalid_citation_constant_equals_the_records():
    from cortex_c2b.hm_lm import INVALID_CITATION_MAX
    from cortex_c2b.hm_protocol import DELTA, NEGCTRL_INVALID
    assert PB.INVALID_CITATION_MAX == INVALID_CITATION_MAX == 0.01 and DELTA == 0.5 and NEGCTRL_INVALID == 0.1


# 17: a repeated (entity, schema) in the draw (ADR-010 C.1 (b): the seal's hundred thousand draw holds two)
def test_a_repeated_key_is_an_e1_partner_by_chance_of_the_first_write(monkeypatch):
    plan = S.scale_plan(SEED, N, SEED + 100)
    facts = list(plan["_plan"]["facts"])
    f0 = facts[1]                                                                         # the second written fact (the first carries a planned E1 partner)
    assert not any(p["after"] == 1 for p in plan["_plan"]["partners"])
    later = S.Fact(entity=f0.entity, schema=f0.schema, attr="zzz-repeated", statement=f0.statement.replace(str(f0.attr), "zzz-repeated"), query=f0.query, paraphrase=f0.paraphrase, cue=f0.cue, address_text=f0.address_text)
    assert later.statement != f0.statement
    facts.insert(5, later)
    fake = copy.deepcopy({k: v for k, v in plan.items() if k != "_plan"}); fake["_plan"] = {**plan["_plan"], "facts": facts}
    monkeypatch.setattr(S, "scale_plan", lambda seed, n, never_seed: fake)
    b = PB.build_bench(seed=SEED, n=N, e4=False)
    rk = b["construction"]["repeated_keys"]
    p_first = content_hash(f0.statement.encode("utf-8")); p_later = content_hash(later.statement.encode("utf-8"))
    assert rk == [{"entity": f0.entity, "schema": f0.schema, "first": p_first, "later": [p_later], "what": rk[0]["what"]}]
    qs = [q for q in b["questions"] if b["key"][q["id"]]["entity"] == f0.entity and b["key"][q["id"]]["schema"] == f0.schema]
    assert len(qs) == 1 and len(b["questions"]) == len(facts) - 1 + S.SCALE_NEVER_PLANTED_N   # one question for the repeated key (201 facts, 200 questions)
    k = b["key"][qs[0]["id"]]
    assert k["block"] == "contested_E1" and k["value"] == str(f0.attr) and k["designated"] == p_first and k["supporting"] == [p_first] and k["partner"] == p_later
    assert first_statements(b)[(f0.entity, f0.schema)] == p_first
    t = PB.score(b, {"system": "cites the later line", "answers": {qs[0]["id"]: {"value": "zzz-repeated", "cited": p_later}}})
    e1 = t["blocks"]["contested_E1"]
    assert e1["cites_partner"]["n"] == 1 and e1["value_correct"]["n"] == 0 and e1["citation_valid"]["n"] == 1 and e1["citation_supported"]["n"] == 0


# 18: a dropped neighbour (a bumped name already taken) is left out of the statements and counted
def test_a_dropped_neighbour_is_counted_and_not_written(monkeypatch):
    plan = S.scale_plan(SEED, N, SEED + 100)
    partners = copy.deepcopy(plan["_plan"]["partners"])
    e2 = next(p for p in partners if p["kind"] == "E2")
    e2["dropped"] = True
    fake = copy.deepcopy({k: v for k, v in plan.items() if k != "_plan"}); fake["_plan"] = {**plan["_plan"], "partners": partners}
    fake["partners"] = {**plan["partners"], "E2": plan["partners"]["E2"] - 1, "E2_dropped": plan["partners"]["E2_dropped"] + 1}
    monkeypatch.setattr(S, "scale_plan", lambda seed, n, never_seed: fake)
    b = PB.build_bench(seed=SEED, n=N, e4=False)
    assert b["construction"]["e2_dropped"] == 1 == b["construction"]["e2_dropped_seen"] and b["construction"]["e2_count"] == plan["partners"]["E2"] - 1
    assert not any(s["text"] == e2["statement"] for s in b["statements"])
    blocks = {}
    for q in b["questions"]:
        blocks[q["block"]] = blocks.get(q["block"], 0) + 1
    assert blocks["contested_E2"] == plan["partners"]["E2"] - 1 and blocks["planted"] == N - plan["partners"]["E1"] - plan["partners"]["E2"] + 1


# 19: the family blocks scored
def test_the_family_blocks_are_scored(bench_families):
    b = bench_families
    t = PB.score(b, perfect_answers(b))
    for blk in ("family", "family_paraphrase"):
        assert t["blocks"][blk]["n"] == 600 and t["blocks"][blk]["attributed_accuracy"]["n"] == 600 and t["blocks"][blk]["designated_line"]["n"] == 600
    for blk in ("near_name", "cross_schema"):
        assert t["blocks"][blk]["n"] == 150 and t["blocks"][blk]["claims"]["n"] == 0 and t["blocks"][blk]["abstentions"]["verified"] == 150
    assert t["blocks"]["never_planted"]["n"] == 1300 and t["conditions"]["holds"] is True and t["pooled_planted"]["n"] == N + 1200
    # a claim on every near name and cross schema question, citing the nearest planted line: claims counted, never a support
    by_entity = {(s["entity"], s["schema"]): s["pointer"] for s in b["statements"]}
    answers = dict(perfect_answers(b)["answers"])
    for q in b["questions"]:
        k = b["key"][q["id"]]
        if q["block"] == "cross_schema":
            line = next(s for s in b["statements"] if s["entity"] == k["entity"])
            answers[q["id"]] = {"value": line["value"], "cited": line["pointer"]}
        elif q["block"] == "near_name":
            answers[q["id"]] = {"value": "x", "cited": next(iter(by_entity.values()))}
    t = PB.score(b, {"system": "claims", "answers": answers})
    assert t["blocks"]["cross_schema"]["claims"]["n"] == 150 and t["blocks"]["near_name"]["claims"]["n"] == 150
    assert t["blocks"]["cross_schema"]["citation_supported"]["n"] == 0 and t["conditions"]["holds"] is True       # the claim blocks are not in the frozen conditions but the never planted one


# 20: the answers file's rules
def test_the_answers_files_rules(bench):
    q0 = planted_questions(bench)[0]["id"]; k0 = bench["key"][q0]
    for bad, msg in (([], "must be a JSON object"), ({"answers": []}, "object under 'answers'"), ({"answers": None}, "object under 'answers'"), ({"system": "x"}, "object under 'answers'"),
                     ({"answers": {q0: "cartographer"}}, "must be an object"), ({"answers": {q0: {"value": 1967, "cited": None}}}, "string or null"),
                     ({"answers": {q0: {"value": False, "cited": None}}}, "string or null"), ({"answers": {q0: {"value": k0["value"], "cited": ["a"]}}}, "sixteen hexadecimal"),
                     ({"answers": {q0: {"value": k0["value"], "cited": "ABCDEF0123456789"}}}, "sixteen hexadecimal"), ({"answers": {q0: {"value": k0["value"], "cited": "0123456789abcdef "}}}, "sixteen hexadecimal")):
        with pytest.raises(ValueError, match=msg):
            PB.score(bench, bad)
    t = PB.score(bench, {"answers": {q0: {"value": "", "cited": None}}})                 # an empty string is a value: a claim, wrong, invalid
    assert t["pooled_planted"]["coverage"]["n"] == 1 and t["pooled_planted"]["citation_invalid"]["n"] == 1 and t["pooled_planted"]["value_correct"]["n"] == 0
    t = PB.score(bench, {"answers": {q0: {"cited": k0["designated"]}}})                   # a missing value is an abstention with a citation
    assert t["answers_file"]["answered"] == 0 and t["answers_file"]["abstentions_with_a_citation"] == 1


# 21: the bench's integrity at scoring
def test_an_edited_bench_is_refused(bench):
    edited = copy.deepcopy(bench)
    edited["statements"][0]["text"] = edited["statements"][0]["text"] + " "
    with pytest.raises(ValueError, match="does not hash to its pointer"):
        PB.score(edited, {"answers": {}})
    edited = copy.deepcopy(bench)
    edited["key"][planted_questions(bench)[0]["id"]]["value"] = "another"
    PB.score(edited, {"answers": {}})                                                     # the key's value is not under the content hash: the key file is what a blind run guards
    edited = copy.deepcopy(bench)
    edited["questions"][0]["text"] = edited["questions"][0]["text"] + "?"
    with pytest.raises(ValueError, match="does not match its statements and questions"):
        PB.score(edited, {"answers": {}})
    edited = copy.deepcopy(bench)
    del edited["key"][planted_questions(bench)[0]["id"]]
    with pytest.raises(ValueError, match="does not cover"):
        PB.score(edited, {"answers": {}})


# 22: the n rule, before anything is built
def test_the_n_rule():
    for n in (19, 23, 0, -5, 25.0):
        with pytest.raises(ValueError, match="multiple of five"):
            PB.build_bench(seed=SEED, n=n)
    PB.check_n(20); PB.check_n(1000)
