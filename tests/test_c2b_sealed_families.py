"""The sealed families of ADR-010 (part A) and the draws of its scale probe (part C): the rules T1 to T5 pinned on the constants,
the decontamination (A.4), the seal regenerated and compared (A.3), and the independence of the organ's code from the sealed texts
(A.5). No encoder runs here; no number below is a number of the record."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cortex_c2b import sealed_families as S                                          # noqa: E402
from cortex_c2b.hm_protocol import value_of_statement, generate_facts, SCHEMAS, N_FACTS, _TEMPLATES   # noqa: E402
from cortex_c2b.near_duplicate_probe import parse_statement, parse_question, E4_TEMPLATES, bump_name   # noqa: E402
from cortex_c2b.organ_use import training_facts                                       # noqa: E402

SEAL = ROOT / "metrics" / "mqar" / "SEALED-FAMILIES.json"


@pytest.fixture(scope="module")
def families():
    return S.generate_all()


@pytest.fixture(scope="module")
def seal():
    return json.loads(SEAL.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- T1 and T2: the cells and the assignment
def test_t1_four_statement_templates_per_schema_one_per_cell_and_t2_two_question_templates():
    for s in SCHEMAS:
        t = S.A_TEMPLATES[s]
        assert set(t) == {"S1", "S2", "S3", "S4", "Q1", "Q2", "P1", "P2"}
        first = lambda x: x.startswith("{e}")                     # noqa: E731
        last = lambda x: x.rstrip().endswith("{a}")              # noqa: E731
        assert first(t["S1"]) and last(t["S1"])                   # entity first, value last
        assert first(t["S2"]) and not last(t["S2"])               # entity first, value elsewhere
        assert not first(t["S3"]) and last(t["S3"])               # entity elsewhere, value last
        assert not first(t["S4"]) and not last(t["S4"])           # entity elsewhere, value elsewhere
        assert first(t["Q1"]) and not first(t["Q2"]) and first(t["P1"]) and not first(t["P2"])
        for k in ("Q1", "Q2", "P1", "P2"):
            assert "{a}" not in t[k] and t[k].count("{e}") == 1 and t[k].endswith("?")
        for k in ("S1", "S2", "S3", "S4"):
            assert t[k].count("{e}") == 1 and t[k].count("{a}") == 1
    # the assignment: every statement template meets every question template on five facts of each schema; never planted by i mod 2
    seen = {}
    for i in range(40):
        sid, qid, pid = S._assign(S.A_TEMPLATES["person"], i)
        assert qid[1] == pid[1]
        seen[(sid, qid)] = seen.get((sid, qid), 0) + 1
    assert set(seen.values()) == {5} and len(seen) == 8
    assert [S._assign(S.A_TEMPLATES["person"], i, never_planted=True)[1] for i in range(4)] == ["Q1", "Q2", "Q1", "Q2"]
    assert S._assign(S.FIRST_FAMILY_TEMPLATES["person"], 7) == ("S", "Q", "P")


def test_t3_no_shared_bigram_content_word_skeleton_or_value_preposition_with_the_first_family():
    first_bigrams = set().union(*[S.bigrams_of(t) for s in SCHEMAS for t in S.FIRST_FAMILY_TEMPLATES[s].values()])
    first_skeletons = {S.skeleton(t) for s in SCHEMAS for t in S.FIRST_FAMILY_TEMPLATES[s].values()} | {S.skeleton(t) for t in E4_TEMPLATES.values()}
    for s in SCHEMAS:
        same_schema_content = set().union(*[S.content_words(t) for t in S.FIRST_FAMILY_TEMPLATES[s].values()])
        for tid, t in S.A_TEMPLATES[s].items():
            assert not (S.bigrams_of(t) & first_bigrams), (s, tid, S.bigrams_of(t) & first_bigrams)
            assert not (S.content_words(t) & same_schema_content), (s, tid, S.content_words(t) & same_schema_content)
            assert S.skeleton(t) not in first_skeletons, (s, tid, S.skeleton(t))
            if tid.startswith("S"):
                wb = S.word_before_value(t)
                if wb is not None:
                    assert wb != S.word_before_value(S.FIRST_FAMILY_TEMPLATES[s]["S"]), (s, tid, wb)
    assert [S.word_before_value(S.FIRST_FAMILY_TEMPLATES[s]["S"]) for s in SCHEMAS] == ["a", "in", "in", "of", "by"]
    assert S.skeleton("{e} works as a {a}") == "{e} _ as a {a}" and S.bigrams_of("{e} works as a {a}") == {("works", "as"), ("as", "a")}
    assert S.fixed_words("by trade, {a} is what {e} does") == [["by", "trade"], ["is", "what"], ["does"]]


def test_t4_values_and_the_line_budget(families):
    all_values = {v for fam in (S.FIRST_FAMILY_VALUES, S.A_VALUES, S.DEV_VALUES) for vs in fam.values() for v in vs}
    first_values = {v for vs in S.FIRST_FAMILY_VALUES.values() for v in vs}
    fixed_text = " ".join(" ".join(r) for s in SCHEMAS for fam in (S.A_TEMPLATES, S.DEV_TEMPLATES, S.FIRST_FAMILY_TEMPLATES) for t in fam[s].values() for r in S.fixed_words(t))
    for fam_values in (S.A_VALUES, S.DEV_VALUES):                     # eight values per schema, none of the first family's, for both
        for s in SCHEMAS:
            assert len(fam_values[s]) == 8 and len(set(fam_values[s])) == 8
            assert not (set(fam_values[s]) & first_values)
    for s in SCHEMAS:                                                    # T4 is A's rule: no value of A a substring of another value of any family,
        for v in S.A_VALUES[s]:                                          # of a template's fixed text or of a name (the hexadecimal part, below)
            assert not any(v.lower() in o.lower() for o in all_values if o != v), (v, [o for o in all_values if o != v and v.lower() in o.lower()])
            assert v.lower() not in fixed_text, v
    # the development family is kept as drafted (ADR-010 A.1): its values are not held to T4 across families ("tin" sits in "Tindale")
    assert "tin" in S.DEV_VALUES["object"] and "Tindale" in S.A_VALUES["place"]
    for name, fam in families.items():
        for f in fam["facts"]:
            assert len(f["statement"].encode("utf-8")) <= S.LINE_BUDGET_BYTES, (name, f["statement"])
            assert f["attribute"] not in f["entity"] and f["entity"] in f["statement"] and f["attribute"] in f["statement"]
            assert S.NAME_RE.match(f["entity"])
        for v in fam["never_planted"]:
            assert S.NAME_RE.match(v["entity"]) and v["entity"] in v["query"] and v["entity"] in v["paraphrase"]
    # a year value never reads as a name's hexadecimal part (T4: no value is a substring of a name)
    years = S.year_values()
    assert {"1821", "1873", "1803"} <= years
    for fam in families.values():
        for f in fam["facts"] + fam["never_planted"]:
            assert f["entity"][-4:] not in years


def test_t5_no_reader_of_the_tagged_system_or_of_the_probe_reads_a_text_of_a(families):
    A = families["A"]
    for f in A["facts"] + A["never_planted"]:
        assert value_of_statement(f["statement"]) is None and parse_statement(f["statement"]) is None
        assert parse_question(f["query"]) is None and parse_question(f["paraphrase"]) is None
    for c in A["cross_schema"]:
        assert parse_question(c["question"]) is None
    # the development family too; T0 and Av are read by the first family's templates by construction (the value cell)
    for f in families["dev"]["facts"]:
        assert value_of_statement(f["statement"]) is None and parse_statement(f["statement"]) is None and parse_question(f["query"]) is None
    for name in ("T0", "Av"):
        for f in families[name]["facts"]:
            assert value_of_statement(f["statement"]) == (f["schema"], f["entity"], f["attribute"])
    # E4's templates differ from every family's templates (A.4)
    for s in SCHEMAS:
        for fam in (S.A_TEMPLATES, S.DEV_TEMPLATES, S.FIRST_FAMILY_TEMPLATES):
            assert E4_TEMPLATES[s] not in fam[s].values()


# ---------------------------------------------------------------- A.4: the decontamination
def test_decontamination_exact_and_one_character_against_the_declared_sources_and_between_families(families):
    protected = S.protected_names()
    filt = S.NameFilter(protected)
    assert filt.rejects("Tone-4010") == "exact" and filt.rejects("Tone-401f") == "exact"           # a first family name and its moved name
    first = generate_facts(N_FACTS, 0)[0]
    assert all(f.entity in protected and bump_name(f.entity) in protected for f in first)
    tf, tneg, _ = training_facts(200, 100_000)
    assert all(f.entity in protected for f in tf + tneg)
    assert all(f.entity in protected for f in generate_facts(200, 1394548301)[0])
    all_names = {}
    for name, fam in families.items():
        for f in fam["facts"] + fam["never_planted"]:
            assert filt.rejects(f["entity"]) is None, (name, f["entity"])
            all_names.setdefault(f["entity"], []).append(name)
        assert len({f["entity"] for f in fam["facts"] + fam["never_planted"]}) == fam["n_facts"] + fam["n_never_planted"]
    assert all(len(v) == 1 for v in all_names.values())                                             # disjoint across the families
    names = list(all_names)
    masked = {}
    for n in names:
        for m in S._mask_variants(n):
            masked.setdefault(m, []).append(n)
    assert all(len(v) == 1 for v in masked.values())                                                # no two family names one character apart
    assert len(protected) == 39249                                                                  # the union's size at this commit (the sources are frozen seeds)


def test_the_oracle_training_family_and_the_v9_pools_share_no_name_with_any_family(families):
    fam_names = {f["entity"] for fam in families.values() for f in fam["facts"] + fam["never_planted"]}
    for r in (0, 1, 73, 150):
        tf, tneg, rep = training_facts(200, 100_000 + 2 * r)
        assert not ({f.entity for f in tf + tneg} & fam_names)
    assert training_facts(200, 100_000)[2]["generator_hash"] == generate_facts(200, 100_000)[1]     # `training_facts` unchanged and pinned
    assert S.FAMILY_ORDER == ("T0", "A", "Av", "dev")


# ---------------------------------------------------------------- A.3: the seal
def test_the_seal_regenerates_exactly_and_hides_nothing(families, seal):
    assert SEAL.exists() and seal["date"] == S.SEAL_DATE and seal["label"].startswith("a synthetic set locked before execution")
    assert seal["constants_sha256"] == S.content_hash_of(S.constants())
    for name in S.FAMILY_ORDER:
        fresh = S.content_hash_of(S.family_content(families[name]))
        assert seal["families"][name]["content_sha256"] == fresh, name
        assert seal["families"][name]["n_facts"] == 200 and seal["families"][name]["n_cross_schema"] == 50
        assert seal["families"][name]["n_never_planted"] == (200 if name == "A" else 50)
        assert seal["families"][name]["rejections"] == families[name]["rejections"]
        for s in SCHEMAS:
            for tid, cell in seal["families"][name]["templates"][s].items():
                assert set(cell) >= {"template", "nearest_first_family_template", "shared_words", "shared_bigrams", "word_jaccard", "char_4gram_jaccard", "skeleton"}
                if name == "A":
                    assert cell["shared_bigrams"] == [] and cell["word_jaccard"] < 0.5
        ham = seal["families"][name]["never_planted_hamming_distance_counts"]
        assert "1" not in ham and sum(ham.values()) == seal["families"][name]["n_never_planted"]
    assert seal["decontamination"]["protected_names"] == 39249 and seal["scale_draws"] is not None
    # the canonical form is the declared one
    assert S.canonical({"b": 1, "a": [1, "é"]}) == '{"a":[1,"é"],"b":1}'


@pytest.mark.slow
def test_the_scale_draws_regenerate_and_the_name_statistics_hold(seal):
    fresh = S.scale_seal()
    for k, v in seal["scale_draws"].items():
        assert fresh[k]["content_sha256"] == v["content_sha256"], k
        assert fresh[k]["names"] == v["names"] and fresh[k]["partners"] == v["partners"] and fresh[k]["never_planted"] == v["never_planted"]
    big = seal["scale_draws"]["60000"]
    assert big["names"]["repeated_names"] == 10 and big["names"]["hamming_one_pairs"] == 1537 and big["n_facts"] == 100_000   # ADR-010 C.1, from the names alone
    small = [seal["scale_draws"][str(s)]["names"]["hamming_one_pairs"] for s in (50_000, 50_001, 50_002, 50_003, 50_004)]
    assert min(small) == 13 and max(small) == 18
    for k, v in seal["scale_draws"].items():
        assert v["partners"]["E1"] == v["n_facts"] // 20 and v["partners"]["E1"] >= v["partners"]["E2"] and v["never_planted"]["n"] == 1000
        assert v["entries_planned"] == v["n_facts"] + v["partners"]["E1"] + v["partners"]["E2"]
        assert v["sample"]["n"] == 2000 and set(v["never_planted"]["per_schema"].values()) == {200} and v["names"]["what"].startswith("among the facts alone")
    assert seal["versions"]["numpy"] and seal["versions"]["python"]


def test_the_scale_plan_follows_the_declared_rules_on_a_small_draw():
    import numpy as np
    p = S.scale_plan(50_000, 400, 50_100)
    plan = p.pop("_plan")
    # the written order is a seeded permutation of the generator's facts, mixing the schema blocks
    facts0 = generate_facts(400, 50_000)[0]
    assert [f.entity for f in plan["facts"]] == [facts0[i].entity for i in plan["order"]] and sorted(plan["order"]) == list(range(400))
    assert plan["order"] == [int(i) for i in np.random.default_rng(50_007).permutation(400)] and len({f.schema for f in plan["facts"][:20]}) > 1
    kinds = [(x["after"], x["kind"]) for x in plan["partners"]]
    assert all(a % 20 == 0 for a, k in kinds if k == "E1") and all(a % 20 == 10 for a, k in kinds if k == "E2")
    assert len([k for _, k in kinds if k == "E1"]) == 20 and len([k for _, k in kinds if k == "E2"]) == 20
    for x in plan["partners"]:
        f = plan["facts"][x["after"]]; assert f.entity == x["entity"]                   # the partner follows its original in the written order
        if x["kind"] == "E1":
            assert x["attribute"] != f.attr and x["entity"] in x["statement"] and x["attribute"] in _TEMPLATES[f.schema][0].format(e=f.entity, a=x["attribute"])
        elif not x.get("dropped"):
            assert x["neighbour"] == bump_name(x["entity"]) and x["neighbour"] in x["statement"]
    assert p["never_planted"]["n"] == 1000 and p["names"]["n"] == 400 and p["entries_planned"] == 400 + p["partners"]["E1"] + p["partners"]["E2"]
    planted = {f.entity for f in plan["facts"]} | {x["neighbour"] for x in plan["partners"] if x["kind"] == "E2" and not x.get("dropped")}
    for i, v in enumerate(plan["never"]):
        assert v["entity"] not in planted and v["schema"] == SCHEMAS[i % 5] and v["query"] == _TEMPLATES[v["schema"]][1].format(e=v["entity"]) and v["distance"] >= 1
    assert plan["sample"] == sorted(int(i) for i in np.random.default_rng(50_009).choice(400, size=400, replace=False)) and len(plan["sample"]) == 400


# ---------------------------------------------------------------- A.5: no module of the organ or of the policies carries a sealed text
def test_no_module_of_the_organ_or_of_the_policies_contains_a_sealed_template_or_value():
    """A.5: the organ's and the policies' code carries none of the sealed templates (the template text with its placeholders)
    and none of the sealed values as a string literal; the generator module itself and the tests are the only holders."""
    modules = [p for p in (ROOT / "cortex_c2b").glob("*.py") if p.name != "sealed_families.py"] + [p for p in (ROOT / "cortex_c2").glob("*.py")] + [ROOT / "train.py"]
    text = "\n".join(p.read_text(encoding="utf-8") for p in modules)
    templates = {t for s in SCHEMAS for fam in (S.A_TEMPLATES, S.DEV_TEMPLATES) for t in fam[s].values()}
    found = sorted(t for t in templates if t in text)
    assert found == [], found
    values = {v for fam in (S.A_VALUES, S.DEV_VALUES) for vs in fam.values() for v in vs}
    found = sorted(v for v in values if f'"{v}"' in text or f"'{v}'" in text)
    assert found == [], found
    # the first family's templates and values are in the generator and nowhere else among the readers' constants
    assert all(_TEMPLATES[s][0] in (ROOT / "cortex_c2b" / "hm_protocol.py").read_text(encoding="utf-8") for s in SCHEMAS)
