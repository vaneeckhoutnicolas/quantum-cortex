"""Rows 51 and 52: the independent test on the sealed families T0, A and Av, read once each on the three checkpoints of the
frozen v1.1.0 system (row 51, letter A3; M2 separated under Holm on the three families, the direction the oracle's), and the
simple references and the cost (row 52, letter R-b by vacuity; the primary tests not separated at this size), ADR-010 parts
A, B and E with the amendment of 2026-10-15; run by the founder 2026-10-30, 01:28 to 11:23 UTC, read cold at 12:57 UTC.
Everything recomputed from the retained files and the two provenance files by the declared functions; these tests fail if a
digit of either row moves."""
import hashlib
import json
import sys
from pathlib import Path, PureWindowsPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
M = ROOT / "metrics" / "mqar"

from cortex_c2b import sealed_reading as R                      # noqa: E402
from cortex_c2b.adoption_reading import binomial_two_sided_p    # noqa: E402
from cortex_c2b import freeze as F                              # noqa: E402

FAMILY_COMMIT, FIRST_COMMIT = "f42bf9f", "217dd50"
FREEZE_COMMIT = "c5b4818d3db29902f72d1b3078e01c4b3cb1e634"
S2024, S7, S1337 = "4cfd6f6e1406", "2e1dc71913b1", "8ceba5db8d0b"
RUNS = (S1337, S2024, S7)                                        # the rows' order: 1337, 2024, 7
LIVE = (S2024, S7)                                               # the live seeds (adoption_reading.LIVE_SEEDS); the conceiving seed 1337 is the third checkpoint
CONFIG_HASH = {S1337: "36a3dd10bbff84eb", S2024: "d0c0d93ba24555a8", S7: "a727d1cb445afa92"}
THRESHOLD = {S1337: 0.5360028743743896, S2024: 0.6122402548789978, S7: 0.5907554626464844}
FAMILIES, INDEXES, FORMS = ("A", "T0", "Av"), ("I0", "I1", "I2"), ("direct", "paraphrase")
N_QUESTIONS = {"A": {"direct": 500, "paraphrase": 400}, "T0": {"direct": 350, "paraphrase": 250}, "Av": {"direct": 350, "paraphrase": 250}}
ADOPTED = {"n_tables": 16, "n_bits": 10, "signature": "exact", "seed": 0, "planes_sha256": "e5703fddba43332f"}
LEGACY = {"n_tables": 4, "n_bits": 10, "signature": "collapsed", "seed": 0}
# row 51, letter A: the six cells of A under mark-veto on I0 (1337, 2024, 7)
A_MARK_VETO_INVALID = {"direct": (43, 31, 29), "paraphrase": (47, 31, 36)}
A_MARK_VETO_NEVER_PLANTED = {"direct": (11, 4, 3), "paraphrase": (7, 2, 2)}
A_PLAIN_INVALID = {"direct": (102, 88, 103), "paraphrase": (120, 92, 113)}
A_PLAIN_ABSTAIN = {"direct": (98, 112, 97), "paraphrase": (80, 108, 87)}
A_PLAIN_NEVER_PLANTED = {"direct": (92, 71, 88), "paraphrase": (95, 71, 81)}
A_PLAIN_NEVER_PLANTED_FIRST_SEED_DIRECT = (23, 16, 23)
# M2 (plain, the own line cited given in the window, against the mark oracle; the live seeds pooled over both forms)
M2 = {"A": {"n": 466, "both": 122, "neither": 52, "b": 5, "c": 287, "third": (230, 2, 146)},
      "Av": {"n": 799, "b": 0, "c": 106, "s1337": (400, 0, 40)}, "T0": {"n": 800, "b": 0, "c": 55, "s1337": (400, 0, 25)}}
A0_CEILING = {"A": (466, 798), "Av": (799, 800), "T0": (800, 800)}
M3 = {"A": {"direct": ((0, 34), (0, 30), (0, 28)), "paraphrase": ((0, 42), (0, 31), (0, 33))},
      "Av": {"direct": ((0, 179), (0, 177), (0, 169)), "paraphrase": ((0, 181), (0, 173), (0, 174))},
      "T0": {"direct": ((168, 189), (159, 186), (164, 181)), "paraphrase": ((165, 186), (164, 188), (169, 190))}}
M4 = {"citations_refused_own_line": {"direct": (1, 2, 1), "paraphrase": (1, 1, 1)}, "citations_refused_other_line": {"direct": (51, 53, 68), "paraphrase": (69, 54, 64)},
      "citations_of_a_label_outside_the_window": {"direct": (7, 2, 5), "paraphrase": (3, 6, 12)},
      "model_abstentions_own_present": {"direct": (48, 47, 53), "paraphrase": (37, 54, 46)}, "model_abstentions_own_absent": {"direct": (50, 65, 44), "paraphrase": (43, 54, 41)}}
# the attribution by component on A (the direct form, I0)
OWN_IN_WINDOW = {"I0": {"direct": (108, 110, 117), "paraphrase": (122, 115, 124)}, "I1": {"direct": (188, 180, 183)}, "I2": {"direct": (200, 200, 200)}}
MISSES = {"index": (66, 58, 52), "interference": (26, 31, 30), "cue": (0, 0, 0), "in_candidates_not_in_window": (0, 1, 1)}
MARKS_BELOW = (4, 16, 13)
MARKS_BELOW_PARAPHRASE = (3, 16, 11)
NEVER_PLANTED_MARK_AT_OR_ABOVE = {"all": (31, 11, 12), "first_seed": (9, 2, 3), "paraphrase": (38, 11, 9), "Av": (8, 3, 4), "T0": (5, 4, 4)}
# the window sizes under I0 on A (per cell: four, three, two, one, no line), the direct then the paraphrase form; the never planted block's four line and empty windows (direct)
WINDOW_SIZES_A = {"direct": ((161, 19, 13, 7, 0), (165, 17, 15, 2, 1), (168, 15, 12, 5, 0)), "paraphrase": ((166, 18, 11, 5, 0), (157, 24, 10, 8, 1), (168, 19, 10, 3, 0))}
WINDOW_SIZES_A_NEVER_PLANTED_DIRECT = ((143, 4), (149, 4), (159, 3))
M4_AV = {"direct": {"citations_refused_other_line": (5, 5, 11), "citations_of_a_label_outside_the_window": (0, 0, 0), "model_abstentions_own_present": (15, 18, 18), "model_abstentions_own_absent": (0, 0, 1)},
         "paraphrase": {"citations_refused_other_line": (7, 5, 7), "citations_of_a_label_outside_the_window": (1, 0, 0), "model_abstentions_own_present": (11, 20, 19), "model_abstentions_own_absent": (0, 0, 0)}}
M4_T0 = {"direct": {"citations_refused_other_line": (2, 3, 3), "citations_of_a_label_outside_the_window": (0, 0, 1), "model_abstentions_own_present": (9, 11, 15), "model_abstentions_own_absent": (0, 0, 0)},
         "paraphrase": {"citations_refused_other_line": (6, 3, 3), "citations_of_a_label_outside_the_window": (0, 0, 0), "model_abstentions_own_present": (8, 9, 7), "model_abstentions_own_absent": (0, 0, 0)}}
NEIGHBOUR_IN_WINDOW_A = {"I0": ((27, 27, 30), (27, 22, 27)), "I1": ((46, 43, 43), (40, 29, 33)), "I2": ((33, 33, 33), (31, 23, 27))}      # present; mark at or above the threshold
OWN_CITED_GIVEN_WINDOW = {"mark-veto": {"I0": (34, 30, 28), "I1": (41, 35, 32), "I2": (42, 32, 35)}, "plain": {"I0": (35, 32, 29), "I1": (44, 40, 37), "I2": (42, 38, 38)}}
# the factor product under plain on I0, the direct form (1337, 2024, 7): own in window, own cited given in window, exact given own cited
FACTORS = {"A": (((108, 200), (35, 108), (0, 35)), ((110, 200), (32, 110), (0, 32)), ((117, 200), (29, 117), (0, 29))),
           "Av": (((200, 200), (179, 200), (0, 179)), ((200, 200), (177, 200), (0, 177)), ((199, 200), (169, 199), (0, 169))),
           "T0": (((200, 200), (189, 200), (168, 189)), ((200, 200), (186, 200), (159, 186)), ((200, 200), (181, 200), (164, 181))),
           "first": (((200, 200), (192, 200), (164, 192)), ((200, 200), (190, 200), (163, 190)), ((200, 200), (182, 200), (162, 182)))}
PRODUCTS = {"A": (0.0, 0.0, 0.0), "Av": (0.0, 0.0, 0.0), "T0": (0.84, 0.795, 0.82), "first": (0.82, 0.815, 0.81)}
TRANSFER = {"A against Av": {"own_in_window": "not transferring", "own_cited_given_in_window": "not transferring", "exact_given_own_cited": "no fall detectable at this size"},
            "Av against first": {"own_in_window": "no fall detectable at this size", "own_cited_given_in_window": "mixed", "exact_given_own_cited": "not transferring"},
            "T0 against first": {"own_in_window": "no fall detectable at this size", "own_cited_given_in_window": "no fall detectable at this size", "exact_given_own_cited": "no fall detectable at this size"}}
A_AGAINST_AV_WINDOW = (-0.46, -0.45, -0.41)
A_AGAINST_AV_CITED = (-0.571, -0.594, -0.601)
AV_AGAINST_FIRST_CITED = (-0.065, -0.065, -0.061)
AV_AGAINST_FIRST_EXACT = (-0.854, -0.858, -0.890)
# the first family beside A (1337, 2024, 7)
FIRST_DIRECT = {"strict": (164, 163, 162), "invalid_plain": (29, 30, 24), "invalid_mark_veto": (28, 27, 20), "never_planted_plain": (10, 5, 10)}
FIRST_PARAPHRASE = {"strict": (159, 166, 171), "invalid_plain": (35, 22, 21), "invalid_mark_veto": (33, 21, 18), "never_planted_plain": (6, 10, 7)}
T0_PLAIN_STRICT = {"direct": (168, 159, 164), "paraphrase": (165, 164, 169)}
# the two blocks (the direct form, I0; 1337, 2024, 7)
NEAR_NAME = {"plain": {"A": (26, 25, 23), "Av": (40, 41, 45), "T0": (46, 47, 49)}, "mark-veto": {"A": (10, 10, 6), "Av": (39, 38, 41), "T0": (45, 47, 48)},
             "oracle": {"A": (32, 26, 29), "Av": (50, 50, 50), "T0": (50, 50, 50)}, "oracle+organ-value": {"A": (0, 0, 0), "Av": (50, 50, 50), "T0": (50, 50, 50)}}
CROSS_SCHEMA = {"plain": {"A": (25, 19, 31), "Av": (10, 23, 20), "T0": (20, 17, 22)}, "mark-veto": {"A": (8, 7, 10), "Av": (5, 11, 10), "T0": (14, 7, 15)},
                "mark-veto+value": {"A": (0, 0, 0), "Av": (5, 11, 10), "T0": (14, 7, 15)}, "oracle": {"A": (31, 30, 27)}}
# row 52: the rule at a fixed index on A (the direct form; 1337, 2024, 7)
INVALID_BY_RULE_I0 = {"plain": (102, 88, 103), "mark-veto": (43, 31, 29), "mark-veto+value": (0, 0, 0), "V1": (0, 0, 0), "oracle": (16, 5, 7)}
NEVER_PLANTED_BY_RULE = {"I0": {"plain": (92, 71, 88), "mark-veto": (11, 4, 3), "mark-veto+value": (0, 0, 0), "V1": (0, 0, 0), "oracle": (31, 11, 12)},
                         "I1": {"plain": (96, 92, 102), "mark-veto": (17, 7, 7), "oracle": (61, 20, 22)}, "I2": {"plain": (60, 46, 36), "mark-veto": (3, 2, 2), "oracle": (36, 11, 13)}}
MODEL_AGAINST_ORACLE = {"I0": ((108, 35, 103, 1, 69), (110, 32, 94, 2, 64), (117, 29, 103, 1, 75)),            # n, model, oracle, model only, oracle only
                        "I1": ((188, 44, 158, 3, 117), (180, 40, 120, 5, 85), (183, 37, 135, 5, 103)),
                        "I2": ((200, 42, 162, 2, 122), (200, 38, 120, 6, 88), (200, 38, 136, 3, 101))}
ORACLE_HOLDS_ON_A = {"I2-direct-4cfd6f6e1406": (120, 0, 11), "I1-paraphrase-4cfd6f6e1406": (125, 2, 16), "I2-paraphrase-4cfd6f6e1406": (125, 0, 5), "I2-paraphrase-2e1dc71913b1": (144, 1, 3)}
COST_RUNS = [(fam, r) for r in (S2024, S1337, S7) for fam in FAMILIES]


def sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load(name: str) -> dict:
    return json.loads((M / name).read_text(encoding="utf-8"))


def cell_key(index: str, form: str, run: str) -> str:
    return f"{index}-{form}-{run}"


def bounds(values, scale: float = 1.0, ndigits: int | None = 0):
    """The attained minimum and maximum, in the row's unit and precision (a range the row prints is the files' rounded minimum and maximum)."""
    lo, hi = min(values) / scale, max(values) / scale
    return (round(lo, ndigits), round(hi, ndigits)) if ndigits else (round(lo), round(hi))


@pytest.fixture(scope="module")
def prov51():
    return load("PROVENANCE-sealed-set.json")


@pytest.fixture(scope="module")
def prov52():
    return load("PROVENANCE-references.json")


@pytest.fixture(scope="module")
def committed():
    return load("sealed-reading-rows51-52.json")


@pytest.fixture(scope="module")
def reading():
    return R.read_rows(M)


@pytest.fixture(scope="module")
def loaded():
    return {fam: {r: R.load_family(M, fam, r) for r in RUNS} for fam in FAMILIES}


@pytest.fixture(scope="module")
def cost():
    return {(fam, r): load(f"cost-{fam}-{r}.json") for fam, r in COST_RUNS}


def _walk(a, b, path, diffs, tol, counts=None):
    if counts is None:
        counts = {"int": 0, "float": 0, "max_abs_diff": 0.0}
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            diffs.append((path, "keys", sorted(set(a) ^ set(b))))
            return
        for k in a:
            _walk(a[k], b[k], path + (k,), diffs, tol, counts)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append((path, "length", (len(a), len(b))))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, path + (i,), diffs, tol, counts)
    elif isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        if a != b and not (path and path[-1] in ("path", "timestamp_utc")):
            diffs.append((path, "value", (a, b)))
    else:
        counts["int" if isinstance(a, int) else "float"] += 1
        d = abs(float(a) - float(b))
        counts["max_abs_diff"] = max(counts["max_abs_diff"], d)
        if (isinstance(a, int) and a != b) or d > tol:
            diffs.append((path, "number", (a, b)))
    return counts


def test_the_files_are_there_with_their_hashes_commits_checkpoints_and_thresholds(prov51, prov52, committed):
    assert len(prov51["files"]) == 34 and len(prov52["files"]) == 46          # 3 + 9 + 18 + 3 + 1; 36 + 9 + 1
    for prov, commit_of in ((prov51, lambda n: FIRST_COMMIT if n.startswith("sealed-first-") else FAMILY_COMMIT), (prov52, lambda n: FIRST_COMMIT if n.startswith("cost-") else FAMILY_COMMIT)):
        for name, meta in prov["files"].items():
            b = (M / name).read_bytes()
            assert b"\r\n" not in b, name
            assert hashlib.sha256(b).hexdigest()[:16] == meta["sha256_lf"], name
            d = json.loads(b)
            assert d["timestamp_utc"] == meta["timestamp_utc"] and d["timestamp_utc"].startswith("2026-10-30T"), name
            if name == "sealed-reading-rows51-52.json":
                continue
            assert d["code_commit"] == commit_of(name) == meta["code_commit"], name
            if "checkpoint" in d:                                                      # the pass, precondition and cost files; the family file names its run only
                assert d["checkpoint"]["run_id"] == meta["run_id"] and d["checkpoint"]["config_hash"] == CONFIG_HASH[meta["run_id"]], name
                assert d["versions"]["torch"] == "2.6.0+cu124" and d["versions"]["numpy"] == "2.5.3" and d["versions"]["python"] == "3.12.10", name
            else:
                assert d["run_id"] == meta["run_id"], name
            if name.endswith(".pass.json"):
                assert d["family"] == meta["family"] and d["index"] == meta["index"] == d["identity"]["compared"]["index"] and d["form"] == meta["form"], name
                assert d["identity"]["compared"]["oracle_threshold"] == THRESHOLD[meta["run_id"]] and d["identity"]["recorded"]["oracle"]["threshold"] == THRESHOLD[meta["run_id"]], name
                assert d["admission"] == ("surprise" if name.startswith("sealed-first-") else "identity") and d["admission_asked"] == "identity", name
                assert d["decode_shape"] == "R4" and d["live_index_geometry"] == ADOPTED, name
                assert all(s["code_state"]["tree_clean"] is True for s in d["sessions"]), name
                cmp = d["identity"]["compared"]
                assert cmp["freeze_sha256"][:16] == prov["freeze"]["sha256_lf"] and cmp["frozen_tree_sha256"][:16] == prov["freeze"]["tree_sha256"], name
                assert cmp["bm25"] == {"k1": 1.5, "b": 0.75} and cmp["k"] == 4 and cmp["environment"]["PYTHONHASHSEED"] == "0", name
                if name.startswith("sealed-first-"):                                   # the record's journal: no sealed family behind it
                    assert cmp["seal_sha256"] is None and cmp["family_content_sha256"] is None, name
                else:
                    assert cmp["seal_sha256"][:16] == prov["seal"]["sha256_lf"] and len(cmp["family_content_sha256"]) == 64, name
            elif name.startswith("sealed-precondition-"):
                assert d["ok"] is True and d["freeze"]["sha256"][:16] == prov["freeze"]["sha256_lf"], name
            elif name.startswith("cost-"):
                assert d["family"] == meta["family"] and d["started_utc"] == meta["started_utc"] and d["seal"]["seal_path"].endswith("SEALED-FAMILIES.json"), name
            else:                                                                  # the nine family files
                assert d["family"] == meta["family"] and d["admission"] == "identity" and d["decode_shape"] == "R4", name
                for key, pm in d["passes"].items():
                    pname = PureWindowsPath(pm["file"]).name
                    assert pname == f"sealed-{meta['family']}-{meta['run_id']}-{key}.pass.json" and sha_lf(M / pname) == pm["sha256_lf"], pname
                assert sha_lf(M / f"sealed-precondition-{meta['run_id']}.json") == d["precondition"]["sha256_lf"]
    for prov in (prov51, prov52):
        assert sha_lf(M / "FREEZE-v1.1.0.json")[:16] == prov["freeze"]["sha256_lf"] and sha_lf(M / "SEALED-FAMILIES.json")[:16] == prov["seal"]["sha256_lf"]
        for name, h in prov["row47_files"].items():
            assert sha_lf(M / name)[:16] == h
        for name, h in prov["logs"].items():
            assert sha_lf(M / "logs" / name)[:16] == h, name
        for run, meta in prov["checkpoints"].items():
            assert meta["threshold"] == THRESHOLD[run] and meta["config_hash"] == CONFIG_HASH[run] and meta["step"] == 7323
    for name, meta in prov51["a0_files"].items():
        assert sha_lf(M / name)[:16] == meta["sha256_lf"], name
    assert len(prov51["a0_files"]) == 21                                            # 18 on the families, 3 on the first family; the six on row 49's arms are row 54's alone
    for fam in FAMILIES:                                                           # the reading's own file block, the 64 character form
        for r in RUNS:
            assert committed["files"][fam][r]["sha256_lf"] == sha_lf(M / f"sealed-{fam}-{r}.json")


def test_the_freeze_binds_every_run_and_matches_the_frozen_set_when_the_git_objects_are_present(prov51, prov52):
    fz = load("FREEZE-v1.1.0.json")
    assert fz["commit"] == FREEZE_COMMIT and len(fz["files"]) == 104 and fz["tree_sha256"][:16] == "676060c734b6d3d7"
    bound = [load(n) for n in prov51["files"] if n.startswith("sealed-precondition-")]
    assert len(bound) == 3 and all(b["freeze"]["sha256"] == sha_lf(M / "FREEZE-v1.1.0.json") for b in bound)
    passes = [n for prov in (prov51, prov52) for n in prov["files"] if n.endswith(".pass.json")]
    assert len(passes) == 57 and all(load(n)["identity"]["compared"]["freeze_sha256"] == sha_lf(M / "FREEZE-v1.1.0.json") for n in passes)
    tree = F.check_tree(fz, ROOT)                                                     # the frozen files of this checkout against the list: a frozen file that differs fails every record test of the ADR
    assert tree["ok"] and tree["equal"] == 104 and tree["missing"] == [] and tree["differing"] == []
    out = F.check_git_objects(fz, ROOT)
    if out["status"] == F.NOT_RUN:
        pytest.skip(f"the git objects of {FREEZE_COMMIT[:7]} are not readable here")
    assert out["ok"] and out["differing"] == [] and out["file_count_git"] == 104 == out["file_count_list"]


def test_the_preconditions_hold_on_the_three_checkpoints(prov51):
    for r in RUNS:
        p = load(f"sealed-precondition-{r}.json")
        c = p["comparison"]
        assert p["ok"] and c["ok"] and c["n_differences"] == 0 and c["differences"] == []
        assert c["threshold"]["live"] == c["threshold"]["row47"] == THRESHOLD[r] == p["live_replay_threshold"] and c["threshold"]["difference"] == 0.0 and c["threshold"]["within"]
        bm = p["bm25_selection"]["on_journal"]
        assert bm["all_tie"] and bm["best"] == 200 == bm["n_queries"] and set(bm["counts"].values()) == {200} and len(bm["counts"]) == 10
        assert bm["pair"] == {"k1": 1.5, "b": 0.75} and bm["label"] == "the textbook pair, confirmed and not selected by the first family"
        assert prov51["row51"]["preconditions"][r] == {"ok": True, "n_differences": 0, "threshold_difference": 0.0, "bm25_all_tie": True, "bm25_pair": {"k1": 1.5, "b": 0.75}}


def test_the_nine_plantings_admit_200_refuse_0_and_the_shadow_refuses_0(loaded, prov51):
    for fam in FAMILIES:
        for r in RUNS:
            f = loaded[fam][r]["file"]
            pl = f["planting"]
            assert pl["n_facts"] == 200 and pl["counts"] == {"written": 200, "admitted": 200, "refused": 0, "refused_attributed": [], "entries": 200}
            assert pl["index_geometry"] == ADOPTED == pl["journal_record"]
            adm = pl["admission"]
            assert adm["rule"] == "identity" == adm["asked"] and adm["source"] == "stamp" and adm["threshold"] == 0.15 and adm["near_recorded"] == 0
            assert adm["shadow_legacy"] == {"judged": 200, "legacy_refused": 0}
            writes = f["planting_writes"]
            assert len(writes) == 200 and all(w["admitted"] and w["admission"] == "identity" and w["near"] is None for w in writes)
            th = f["threshold"]
            assert th["threshold"] == THRESHOLD[r] == th["row47"] and th["within_tolerance"] and th["scratch_admitted"] == 200 == th["scratch_facts"]
            assert prov51["row51"]["plantings"][f"{fam}-{r}"] == pl["counts"] and prov51["row51"]["shadow_legacy_refused"][f"{fam}-{r}"] == 0
    for r in RUNS:                                                                  # the first family: the record's journals, 200 entries, read through an explicit index
        p = load(f"sealed-first-{r}-I0-paraphrase.pass.json")
        assert p["planting"]["counts"] == {"entries": 200} and p["planting"]["first_family"] is True and p["planting"]["admission"] == "surprise (the record's)"
        assert p["index_record"]["record"] == LEGACY and p["index_record"]["live"] == ADOPTED and p["index_record"]["overridden"] is True
        assert PureWindowsPath(p["planting"]["record_file"]).name == f"hm-lm-{r}-session-b-index16exact.json"
        assert sha_lf(M / f"hm-lm-{r}-session-b-index16exact.json") == p["planting"]["sha256_lf"]


def test_the_fifty_four_passes_ran_in_one_session_each_and_the_first_family_s_in_two(loaded):
    for fam in FAMILIES:
        for r in RUNS:
            for (index, form), p in loaded[fam][r]["passes"].items():
                assert p["n_questions"] == N_QUESTIONS[fam][form] == len(p["answers"]) and p["run_in_sessions"] == 1 and len(p["sessions"]) == 1
                s = p["sessions"][0]
                assert s["reason"] == "complete" and s["cuts"] == [] and s["resumed_from"] is None and s["code_change"] is None and s["code_state"]["commit"] == FAMILY_COMMIT
                assert s["first_unit"] == 0 and s["last_unit"] == N_QUESTIONS[fam][form] - 1 and s["traceback"] is None
                assert p["duplicates_compared"] == [] and len(p["progress_sha256"]) == 64
    for r in RUNS:
        p = load(f"sealed-first-{r}-I0-paraphrase.pass.json")
        assert p["run_in_sessions"] == 2 and len(p["sessions"]) == 2 and p["n_questions"] == 250
        first, second = p["sessions"]
        assert first["reason"] == "death" and first["start"].startswith("2026-10-30T09:49") and first["code_state"]["commit"] == FAMILY_COMMIT
        assert second["reason"] == "complete" and second["start"].startswith("2026-10-30T10:3") and second["resumed_from"] == 0 and second["last_unit"] == 249
        assert second["code_change"]["previous_commit"] == FAMILY_COMMIT and second["code_change"]["commit"] == FIRST_COMMIT == second["code_state"]["commit"]
        assert second["code_change"]["modules_changed"] == ["cortex_c2b/sealed_family.py"] and second["code_change"]["tree_clean"] and second["code_change"]["previous_tree_clean"]
        assert first["traceback"] is None                                              # the error rose at the bridge's construction, before the session's hook; the log carries it
    # the timeline as the files date it (UTC): the preconditions, the families, the first family's passes, the cost runs
    stamp = lambda name: load(name)["timestamp_utc"]  # noqa: E731
    assert [stamp(f"sealed-precondition-{r}.json")[11:19] for r in (S2024, S1337, S7)] == ["01:28:49", "01:32:13", "01:37:32"]
    assert stamp(f"sealed-A-{S2024}.json")[11:19] == "01:53:12" and load(f"sealed-A-{S2024}.json")["planting"]["timestamp_utc"][11:19] == "01:43:50"
    assert load(f"sealed-T0-{S2024}-I2-paraphrase.pass.json")["sessions"][0]["end"][11:19] == "02:09:51" and load(f"sealed-Av-{S2024}.json")["planting"]["timestamp_utc"][11:19] == "05:14:00"
    assert load(f"sealed-Av-{S2024}-I0-direct.pass.json")["sessions"][0]["start"][11:19] == "08:05:22" and stamp(f"sealed-Av-{S2024}.json")[11:19] == "08:12:43"
    assert (load(f"sealed-A-{S1337}.json")["planting"]["timestamp_utc"][11:19], stamp(f"sealed-Av-{S1337}.json")[11:19]) == ("08:16:25", "08:41:35")
    assert (load(f"sealed-A-{S7}.json")["planting"]["timestamp_utc"][11:19], stamp(f"sealed-Av-{S7}.json")[11:19]) == ("08:51:47", "09:15:23")
    assert [stamp(f"sealed-first-{r}-I0-paraphrase.pass.json")[11:19] for r in (S2024, S1337, S7)] == ["10:37:32", "10:38:31", "10:39:31"]
    assert (load(f"cost-A-{S2024}.json")["started_utc"][11:19], stamp(f"cost-Av-{S7}.json")[11:19]) == ("10:47:12", "11:23:05") and stamp("sealed-reading-rows51-52.json")[11:19] == "12:57:07"
    for fam in FAMILIES:                                                               # the first pass starts within thirty seconds of the planting's write on the eight families not held
        for r in RUNS:
            if (fam, r) == ("Av", S2024):
                continue
            from datetime import datetime
            planted = datetime.fromisoformat(loaded[fam][r]["file"]["planting"]["timestamp_utc"])
            started = datetime.fromisoformat(loaded[fam][r]["passes"][("I0", "direct")]["sessions"][0]["start"])
            assert 0 < (started - planted).total_seconds() <= 30, (fam, r)


def test_the_reading_recomputed_from_the_files_equals_the_committed_one(reading, committed, prov51):
    diffs = []
    counts = _walk(committed, reading, (), diffs, 4e-15)
    assert diffs == []
    assert counts["int"] == 30106 and counts["float"] == 26613 and counts["int"] + counts["float"] == 56719 and counts["max_abs_diff"] <= 4e-15
    assert R.summary_lines(reading) == R.summary_lines(committed)
    assert tuple(R.LIVE_RUN_IDS) == LIVE and R.CONCEIVING_RUN_ID == S1337 and tuple(R.RUN_IDS) == RUNS
    # the reading module: the one recorded in every pass's code state at the founder's reading, and the one corrected after it by one line (the loader)
    recorded = {load(f"sealed-{fam}-{r}-I0-direct.pass.json")["sessions"][0]["code_state"]["new_modules_sha256_lf"]["cortex_c2b/sealed_reading.py"] for fam in FAMILIES for r in RUNS}
    assert recorded == {prov51["code"]["sealed_reading_py_sha256_lf_at_the_founders_reading"]} == {"b0ee99b63af67bf4"}
    assert sha_lf(ROOT / "cortex_c2b" / "sealed_reading.py")[:16] == prov51["code"]["sealed_reading_py_sha256_lf_at_the_recomputation"]


def test_letter_a3_the_conditions_hold_on_no_cell_of_a_under_mark_veto(reading):
    la = reading["letter_A"]
    assert la["letter"] == "A3" and la["cells"] == 6 and la["cells_holding"] == 0 and la["claimed_forms"] == [] and la["claim"] is None
    assert la["holds"] == {cell_key("I0", form, r): False for form in FORMS for r in RUNS} and la["by_form"] == {"direct": False, "paraphrase": False}
    assert la["sentence"] == R.A1_SENTENCE and la["label"].startswith("A.2's label") and set(la["construction_table"]) == {"i", "ii", "vii"}
    assert la["reported"].startswith("they hold on none") and la["not_read"] is None
    cells = reading["readings"]["A"]["cells"]
    for form in FORMS:
        for i, r in enumerate(RUNS):
            c = cells[cell_key("I0", form, r)]["rules"]["mark-veto"]["conditions"]
            assert not c["holds"] and not c["holds_with_own_line_gap"] and c["delta"] == 0.5 and c["invalid_max"] == 0.01 and c["never_planted"]["bound"] == 0.1
            assert c["gap"]["k"] == 0 and c["gap"]["n"] == 200 and c["own_line_gap"]["k"] == 0 and c["off_arm"] == R.OFF_ARM_SENTENCE
            assert c["invalid_citations"]["k"] == A_MARK_VETO_INVALID[form][i] and c["never_planted"]["all"]["claims"] == A_MARK_VETO_NEVER_PLANTED[form][i]
            assert c["never_planted"]["all"]["n"] == 200 and c["never_planted"]["first_seed"]["n"] == 50 and c["never_planted"]["first_seed"]["seed"] == 30001
            assert 0.145 <= c["invalid_citations"]["rate"] <= 0.235 and c["never_planted"]["all"]["claim_rate"]["rate"] < 0.1
            assert la["failing_cells"][cell_key("I0", form, r)]["broken"] == c["broken"]
            assert la["oracle_organ_value_strict_on_A"][cell_key("I0", form, r)] == 0


def test_plain_on_a_the_model_alone_on_unseen_forms(reading):
    cells = reading["readings"]["A"]["cells"]
    for index in INDEXES:
        for form in FORMS:
            for i, r in enumerate(RUNS):
                rules = cells[cell_key(index, form, r)]["rules"]
                for rule in ("plain", "mark-veto", "mark-veto+value", "V1", "V1+organ-value", "oracle+organ-value"):
                    assert rules[rule]["counts"]["strict"] == 0 and rules[rule]["counts"]["strict_own"] == 0, (index, form, r, rule)
                for rule in ("mark-veto+value", "V1", "V1+organ-value", "oracle+organ-value"):
                    assert rules[rule]["counts"]["claims"] == 0 and rules[rule]["counts"]["abstain"] == 200, (index, form, r, rule)
                m1 = reading["M"]["families"]["A"]["M1_plain_conditions"].get(cell_key(index, form, r))
                if index == "I0":
                    c = rules["plain"]["counts"]
                    assert c["invalid"] == A_PLAIN_INVALID[form][i] == c["claims"] and c["abstain"] == A_PLAIN_ABSTAIN[form][i] and c["n"] == 200
                    assert m1 is not None and not m1["holds"] and len(m1["broken"]) == 3 and m1["never_planted"]["all"]["claims"] == A_PLAIN_NEVER_PLANTED[form][i]
                    if form == "direct":
                        assert m1["never_planted"]["first_seed"]["claims"] == A_PLAIN_NEVER_PLANTED_FIRST_SEED_DIRECT[i]


def test_m2_the_model_against_the_oracle_separated_under_holm_direction_the_oracle(reading):
    m = reading["M"]
    for fam in FAMILIES:
        m2 = m["families"][fam]["M2"]
        pooled = m2["pooled_live_seeds"]
        assert pooled["n_reads_with_own_in_window"] == M2[fam]["n"] and (pooled["b_model_only"], pooled["c_oracle_only"]) == (M2[fam]["b"], M2[fam]["c"])
        assert pooled["p_two_sided"] == pytest.approx(binomial_two_sided_p(M2[fam]["b"], M2[fam]["c"]))
        assert m2["separated_under_holm"] and m2["direction"] == "oracle" and m2["sentence"] == R.M2_SENTENCES["organ"]
        ceiling = m2["a0_ceiling"]["bag_top_is_own_pooled"]
        assert (ceiling["k"], ceiling["n"]) == A0_CEILING[fam] and len(m2["a0_ceiling"]["files"]) == 4
    a = m["families"]["A"]["M2"]
    assert (a["pooled_live_seeds"]["both"], a["pooled_live_seeds"]["neither"]) == (M2["A"]["both"], M2["A"]["neither"])
    assert a["pooled_live_seeds"]["model_rate"]["k"] == 127 and a["pooled_live_seeds"]["oracle_rate"]["k"] == 409
    third = a["third_checkpoint"]
    assert (third["n_reads_with_own_in_window"], third["b_model_only"], third["c_oracle_only"]) == M2["A"]["third"] and third["p_two_sided"] < 1e-40
    assert all(v.startswith("a partial ceiling") for v in a["a0_ceiling"]["letters"].values())
    assert all(v.startswith("A0 (ii)") for v in m["families"]["Av"]["M2"]["a0_ceiling"]["letters"].values())
    assert all(v.startswith("A0 (iii)") for v in m["families"]["T0"]["M2"]["a0_ceiling"]["letters"].values())
    for fam in ("Av", "T0"):
        third = m["families"][fam]["M2"]["third_checkpoint"]
        assert (third["n_reads_with_own_in_window"], third["b_model_only"], third["c_oracle_only"]) == M2[fam]["s1337"]
    assert m["holm"]["rejected"] == {"A": True, "Av": True, "T0": True} and m["holm"]["alpha"] == 0.05 and m["holm"]["m"] == 3
    assert m["holm"]["pvalues"]["A"] < 1e-77 and m["holm"]["pvalues"]["Av"] < 1e-31 and m["holm"]["pvalues"]["T0"] < 1e-16


def test_m3_exactness_given_an_own_citation_and_m4_the_veto_s_ledger(reading):
    fams = reading["M"]["families"]
    for fam in FAMILIES:
        m3 = fams[fam]["M3_exactness_given_own_citation_mark_veto"]
        for form in FORMS:
            for i, r in enumerate(RUNS):
                assert (m3[cell_key("I0", form, r)]["k"], m3[cell_key("I0", form, r)]["n"]) == M3[fam][form][i], (fam, form, r)
    m4 = fams["A"]["M4_veto_ledger"]
    for form in FORMS:
        for i, r in enumerate(RUNS):
            v = m4[cell_key("I0", form, r)]
            assert v["n"] == 200 and v["malformed_outputs"] == 0
            for key, expected in M4.items():
                assert v[key] == expected[form][i], (form, r, key)
            mv = reading["readings"]["A"]["cells"][cell_key("I0", form, r)]["rules"]["mark-veto"]["counts"]
            assert (v["citations_refused_own_line"] + v["citations_refused_other_line"] + v["citations_of_a_label_outside_the_window"] + v["malformed_outputs"]
                    + v["model_abstentions_own_present"] + v["model_abstentions_own_absent"]) == mv["abstain"] == 200 - mv["claims"]
            for fam, expected in (("Av", M4_AV), ("T0", M4_T0)):
                w = fams[fam]["M4_veto_ledger"][cell_key("I0", form, r)]
                assert w["citations_refused_own_line"] == 0 and w["malformed_outputs"] == 0
                for key, exp in expected[form].items():
                    assert w[key] == exp[i], (fam, form, r, key)
            fs = reading["readings"]["A"]["cells"][cell_key("I0", form, r)]["rules"]["mark-veto"]["conditions"]["never_planted"]["first_seed"]
            assert (fs["n"], fs["claims"]) == (50, {"direct": (3, 1, 1), "paraphrase": (3, 0, 0)}[form][i])


def test_the_attribution_by_component_on_a(reading):
    cells = reading["readings"]["A"]["cells"]
    for i, r in enumerate(RUNS):
        c = cells[cell_key("I0", "direct", r)]
        assert c["rules"]["plain"]["factors"]["own_in_window"]["k"] == OWN_IN_WINDOW["I0"]["direct"][i] and c["n_planted"] == 200
        assert cells[cell_key("I0", "paraphrase", r)]["rules"]["plain"]["factors"]["own_in_window"]["k"] == OWN_IN_WINDOW["I0"]["paraphrase"][i]
        miss = c["misses"]
        assert miss["n_misses"] == 200 - OWN_IN_WINDOW["I0"]["direct"][i] and miss["unclassed"] == MISSES["in_candidates_not_in_window"][i]
        for key in ("index", "interference", "cue", "in_candidates_not_in_window"):
            assert miss[key] == MISSES[key][i], (r, key)
        assert miss["index"] + miss["interference"] + miss["cue"] == miss["not_in_candidates"] and miss["not_in_candidates"] + miss["in_candidates_not_in_window"] == miss["n_misses"]
        marks = c["marks"]
        assert marks["threshold"] == THRESHOLD[r] and (marks["own_line_marks_below_threshold"]["k"], marks["own_line_marks_below_threshold"]["n"]) == (MARKS_BELOW[i], OWN_IN_WINDOW["I0"]["direct"][i])
        assert marks["never_planted_max_at_or_above"]["all"]["k"] == NEVER_PLANTED_MARK_AT_OR_ABOVE["all"][i] and marks["never_planted_max_at_or_above"]["30001"]["k"] == NEVER_PLANTED_MARK_AT_OR_ABOVE["first_seed"][i]
        assert marks["never_planted_max_at_or_above"]["30001"]["n"] == 50 and marks["never_planted_max_at_or_above"]["30003"]["n"] == 150 and marks["never_planted_max_at_or_above"]["all"]["n"] == 200
        pm = cells[cell_key("I0", "paraphrase", r)]["marks"]
        assert (pm["own_line_marks_below_threshold"]["k"], pm["own_line_marks_below_threshold"]["n"]) == (MARKS_BELOW_PARAPHRASE[i], OWN_IN_WINDOW["I0"]["paraphrase"][i])
        assert pm["never_planted_max_at_or_above"]["all"]["k"] == NEVER_PLANTED_MARK_AT_OR_ABOVE["paraphrase"][i]
        for fam, seed in (("Av", "35001"), ("T0", "70001")):
            fm = reading["readings"][fam]["cells"][cell_key("I0", "direct", r)]["marks"]
            assert fm["own_line_marks_below_threshold"]["k"] == 0 and fm["never_planted_max_at_or_above"]["all"]["k"] == NEVER_PLANTED_MARK_AT_OR_ABOVE[fam][i] and fm["never_planted_max_at_or_above"][seed]["n"] == 50
        assert c["m3_exactness_under_mark_veto"]["n"] == OWN_CITED_GIVEN_WINDOW["mark-veto"]["I0"][i] and c["m3_exactness_under_mark_veto"]["k"] == 0
        assert c["rules"]["plain"]["factors"]["own_cited_given_in_window"]["k"] == OWN_CITED_GIVEN_WINDOW["plain"]["I0"][i] and c["rules"]["plain"]["factors"]["exact_given_own_cited"]["k"] == 0
        assert 18 <= c["short_windows"] <= 43
        for index in ("I1", "I2"):
            assert cells[cell_key(index, "direct", r)]["rules"]["plain"]["factors"]["own_in_window"]["k"] == OWN_IN_WINDOW[index]["direct"][i]
    short = [reading["readings"][fam]["cells"][cell_key("I0", form, r)]["short_windows"] for fam in FAMILIES for form in FORMS for r in RUNS]
    assert min(short) == 18 and max(short) == 43 and len(short) == 18                 # the buckets proposed fewer than four lines on 18 to 43 of 200 per I0 cell


def test_the_factor_product_and_its_transfer(reading):
    fp = reading["factor_product"]
    assert fp["label"] == R.DIFFERENCE_INTERVAL_LABEL
    for i, r in enumerate(RUNS):
        for fam in ("A", "Av", "T0", "first"):
            f = fp["per_run"][r][fam]
            (w, wn), (c, cn), (e, en) = FACTORS[fam][i]
            assert (f["own_in_window"]["k"], f["own_in_window"]["n"]) == (w, wn) and (f["own_cited_given_in_window"]["k"], f["own_cited_given_in_window"]["n"]) == (c, cn)
            assert (f["exact_given_own_cited"]["k"], f["exact_given_own_cited"]["n"]) == (e, en) and f["product"] == pytest.approx(PRODUCTS[fam][i], abs=1e-9)
            assert f["product"] == pytest.approx(f["own_in_window"]["rate"] * f["own_cited_given_in_window"]["rate"] * f["exact_given_own_cited"]["rate"]) and f["strict_own"]["k"] == e
            if fam != "first":
                assert f["rule"] == "plain"
        d = fp["differences"]["A against Av"]["by_run"][r]
        assert d["own_in_window"]["difference"] == pytest.approx(A_AGAINST_AV_WINDOW[i], abs=1e-9) and d["own_in_window"]["ci95"][1] < 0
        assert d["own_cited_given_in_window"]["difference"] == pytest.approx(A_AGAINST_AV_CITED[i], abs=5e-4) and d["own_cited_given_in_window"]["ci95"][1] < 0
        assert d["exact_given_own_cited"]["difference"] == 0.0
        d = fp["differences"]["Av against first"]["by_run"][r]
        assert d["own_cited_given_in_window"]["difference"] == pytest.approx(AV_AGAINST_FIRST_CITED[i], abs=5e-4) and d["exact_given_own_cited"]["difference"] == pytest.approx(AV_AGAINST_FIRST_EXACT[i], abs=5e-4)
        assert d["exact_given_own_cited"]["ci95"][1] < -0.79 and abs(d["own_in_window"]["difference"]) <= 0.0051
    for pair, expected in TRANSFER.items():
        assert fp["differences"][pair]["transfer"] == expected, pair


def test_av_and_t0_reported_the_default_policy_holds_on_every_cell(reading):
    for fam in ("Av", "T0"):
        cells = reading["readings"][fam]["cells"]
        for index in INDEXES:
            for form in FORMS:
                for i, r in enumerate(RUNS):
                    rules = cells[cell_key(index, form, r)]["rules"]
                    for rule in ("mark-veto+value", "V1+organ-value"):
                        assert rules[rule]["conditions"]["holds"] and rules[rule]["counts"]["invalid"] == 0, (fam, index, form, r, rule)
                        assert rules[rule]["conditions"]["never_planted"]["all"]["claims"] == 0
                    assert rules["oracle"]["counts"]["strict_own"] >= (199 if fam == "Av" else 200)
                    oc = rules["oracle"]["conditions"]
                    if not oc["holds"]:
                        assert len(oc["broken"]) == 1 and oc["broken"][0].startswith("claims on never planted entities") and 5 <= oc["never_planted"]["all"]["claims"] <= 17
                    else:
                        assert 2 <= oc["never_planted"]["all"]["claims"] <= 4
                    if fam == "Av":
                        assert rules["plain"]["counts"]["strict"] == 0 and 158 <= rules["plain"]["counts"]["invalid"] <= 190
                        assert 157 <= rules["mark-veto+value"]["counts"]["strict"] <= 186 and 156 <= rules["V1+organ-value"]["counts"]["strict"] <= 186
                        assert not rules["V1"]["conditions"]["holds"] and rules["V1"]["counts"]["strict"] == 0
                        if index == "I0":
                            exact = rules["plain"]["factors"]["exact_given_own_cited"]
                            assert exact["k"] == 0 and exact["n"] == M3["Av"][form][i][1]
                    else:
                        assert 154 <= rules["mark-veto+value"]["counts"]["strict"] <= 194 and 154 <= rules["V1+organ-value"]["counts"]["strict"] <= 194
                        assert rules["V1"]["conditions"]["holds"] and 127 <= rules["V1"]["counts"]["strict"] <= 169 and rules["V1"]["counts"]["invalid"] == 0
                        assert 19 <= rules["plain"]["counts"]["invalid"] <= 39 and 1 <= rules["plain"]["conditions"]["never_planted"]["all"]["claims"] <= 14
                        if index == "I0":
                            assert 19 <= rules["plain"]["counts"]["invalid"] <= 28 and 4 <= rules["plain"]["conditions"]["never_planted"]["all"]["claims"] <= 13
                        assert not rules["plain"]["conditions"]["holds"] and not rules["mark-veto"]["conditions"]["holds"] and 17 <= rules["mark-veto"]["counts"]["invalid"] <= 36
                        assert rules["mark-veto"]["conditions"]["broken"] == [f"invalid citations {rules['mark-veto']['counts']['invalid'] / 200} above 0.01"]
                        if index == "I0":
                            assert rules["plain"]["counts"]["strict_own"] == T0_PLAIN_STRICT[form][i]
                    assert cells[cell_key(index, form, r)]["sessions"] == 1
    for fam in ("Av", "T0"):
        assert sum(1 for c in reading["readings"][fam]["cells"].values() if not c["rules"]["oracle"]["conditions"]["holds"]) == 8
        cells = reading["readings"][fam]["cells"].values()
        assert bounds([c["rules"]["mark-veto+value"]["counts"]["strict"] for c in cells]) == ({"Av": (157, 186), "T0": (154, 194)}[fam])
        assert bounds([c["rules"]["V1+organ-value"]["counts"]["strict"] for c in cells]) == ({"Av": (156, 186), "T0": (154, 194)}[fam])
        assert bounds([c["rules"]["oracle"]["conditions"]["never_planted"]["all"]["claims"] for c in cells if not c["rules"]["oracle"]["conditions"]["holds"]]) == ({"Av": (5, 17), "T0": (5, 12)}[fam])
        assert bounds([c["rules"]["oracle"]["conditions"]["never_planted"]["all"]["claims"] for c in cells if c["rules"]["oracle"]["conditions"]["holds"]]) == (2, 4)
    t0c = reading["readings"]["T0"]["cells"]
    assert bounds([c["rules"]["V1"]["counts"]["strict"] for c in t0c.values()]) == (127, 169) and bounds([c["rules"]["mark-veto"]["counts"]["invalid"] for c in t0c.values()]) == (17, 36)
    assert bounds([c["rules"]["plain"]["counts"]["invalid"] for c in t0c.values()]) == (19, 39) and bounds([c["rules"]["plain"]["counts"]["invalid"] for k, c in t0c.items() if k.startswith("I0")]) == (19, 28)
    assert bounds([c["rules"]["plain"]["conditions"]["never_planted"]["all"]["claims"] for c in t0c.values()]) == (1, 14) and bounds([c["rules"]["plain"]["conditions"]["never_planted"]["all"]["claims"] for k, c in t0c.items() if k.startswith("I0")]) == (4, 13)
    assert bounds([c["rules"]["plain"]["counts"]["invalid"] for c in reading["readings"]["Av"]["cells"].values()]) == (158, 190)
    assert sorted(c["rules"]["oracle"]["counts"]["strict_own"] for c in reading["readings"]["Av"]["cells"].values())[:2] == [199, 200]
    # strict and strict_own differ by one on exactly two cells of T0 under plain (a strict answer whose cited line is another line holding the same value)
    differing = [(fam, k) for fam in FAMILIES for k, c in reading["readings"][fam]["cells"].items() for rule in R.RULES if c["rules"][rule]["counts"]["strict"] != c["rules"][rule]["counts"]["strict_own"]]
    assert differing == [("T0", cell_key("I0", "paraphrase", S1337)), ("T0", cell_key("I1", "direct", S7))]
    assert all(t0c[k]["rules"]["plain"]["counts"]["strict"] - t0c[k]["rules"]["plain"]["counts"]["strict_own"] == 1 for _, k in differing)
    t0 = reading["T0_own_in_window"]
    assert len(t0) == 18 and all(v["k"] == 200 and v["n"] == 200 for v in t0.values())
    assert all(c["misses"]["n_misses"] == 0 for k, c in reading["readings"]["T0"]["cells"].items() if k.startswith("I0"))
    assert all(c["misses"] == {"not_classed": "a reference pass: every live entry is a candidate, the adoption's classes are the index's"} for k, c in reading["readings"]["T0"]["cells"].items() if not k.startswith("I0"))
    assert sum(c["misses"]["n_misses"] for k, c in reading["readings"]["Av"]["cells"].items() if k.startswith("I0")) == 1       # Av on I0: one miss in 1 200 reads (the direct form, seed 7), the index's
    # Av, the direct form on I0: the model cites the own line on 179, 177, 169 of 200 and copies its exact value on 0 of 525
    cited = [reading["readings"]["Av"]["cells"][cell_key("I0", "direct", r)]["rules"]["plain"]["factors"]["own_cited_given_in_window"]["k"] for r in RUNS]
    assert cited == [179, 177, 169] and sum(cited) == 525


def test_the_first_family_beside_a(reading):
    ff = reading["first_family"]
    for i, r in enumerate(RUNS):
        d = ff["direct"][r]
        assert d["n"] == 200 and d["threshold"] == THRESHOLD[r] and d["label"] == R.FIRST_FAMILY_LABEL
        assert d["plain"]["strict_recall_own"]["k"] == FIRST_DIRECT["strict"][i] == d["mark_veto_replayed"]["strict_recall_own"]["k"]
        assert d["plain"]["invalid"]["k"] == FIRST_DIRECT["invalid_plain"][i] and d["mark_veto_replayed"]["invalid"]["k"] == FIRST_DIRECT["invalid_mark_veto"][i]
        assert (d["plain"]["never_planted_claims"]["k"], d["plain"]["never_planted_claims"]["n"]) == (FIRST_DIRECT["never_planted_plain"][i], 50) and d["mark_veto_replayed"]["never_planted_claims"]["k"] == 0
        assert d["factors_plain"]["own_in_window"]["k"] == 200
        p = ff["paraphrase"][r]
        assert p["n"] == 200 and p["admission"] == "surprise" and p["label"] == R.FIRST_FAMILY_LABEL
        assert p["plain"]["strict_recall_own"]["k"] == FIRST_PARAPHRASE["strict"][i] == p["mark_veto"]["strict_recall_own"]["k"]
        assert p["plain"]["invalid"]["k"] == FIRST_PARAPHRASE["invalid_plain"][i] and p["mark_veto"]["invalid"]["k"] == FIRST_PARAPHRASE["invalid_mark_veto"][i]
        assert p["plain"]["never_planted_claims"]["k"] == FIRST_PARAPHRASE["never_planted_plain"][i] and p["mark_veto"]["never_planted_claims"]["k"] == 0
        assert p["factors_plain"]["own_in_window"]["k"] == 200 and not p["mark_veto"]["conditions"]["holds"]


def test_the_two_blocks_enter_no_letter(reading):
    for fam in FAMILIES:
        for i, r in enumerate(RUNS):
            nn = reading["near_name_block"][fam][cell_key("I0", "direct", r)]
            assert nn["n"] == 50 and nn["enters_a_letter"] is False
            for rule, expected in NEAR_NAME.items():
                assert nn["by_rule"][rule]["claims"] == expected[fam][i], (fam, r, rule)
            assert nn["by_rule"]["mark-veto+value"]["claims"] == 0 and nn["by_rule"]["V1"]["claims"] == 0 and nn["by_rule"]["V1+organ-value"]["claims"] == 0
            cs = reading["cross_schema_block"][fam][cell_key("I0", "direct", r)]
            assert cs["n"] == 50 and cs["enters_a_letter"] is False
            for rule, expected in CROSS_SCHEMA.items():
                if fam in expected:
                    assert cs["by_rule"][rule]["claims"] == expected[fam][i], (fam, r, rule)
            assert cs["by_rule"]["V1"]["claims"] <= 1
            for index in INDEXES:
                key = cell_key(index, "direct", r)
                assert set(reading["near_name_block"][fam]) == {cell_key(ix, "direct", rr) for ix in INDEXES for rr in RUNS}        # the blocks exist on the direct form
                assert reading["near_name_block"][fam][key]["by_rule"]["mark-veto+value"]["claims"] == 0 and reading["near_name_block"][fam][key]["by_rule"]["V1"]["claims"] == 0
                nb = reading["near_name_block"][fam][key]
                if fam == "A":
                    assert (nb["neighbour_in_window"]["k"], nb["neighbour_mark_at_or_above_threshold"]["k"]) == (NEIGHBOUR_IN_WINDOW_A[index][0][i], NEIGHBOUR_IN_WINDOW_A[index][1][i]), (index, r)
                else:
                    assert nb["neighbour_in_window"]["k"] == 50 == nb["neighbour_mark_at_or_above_threshold"]["k"] and nb["by_rule"]["oracle"]["claims"] == 50 == nb["by_rule"]["oracle+organ-value"]["claims"]
                    if index != "I0":
                        assert reading["cross_schema_block"][fam][key]["by_rule"]["oracle"]["claims"] == 50
    oracle_i0 = {fam: [reading["cross_schema_block"][fam][cell_key("I0", "direct", r)]["by_rule"]["oracle"]["claims"] for r in RUNS] for fam in ("Av", "T0")}
    assert oracle_i0 == {"Av": [49, 45, 47], "T0": [49, 45, 44]} and bounds(oracle_i0["Av"] + oracle_i0["T0"]) == (44, 49)       # 44 to 49 under I0; 50 of 50 under I1 and I2
    for fam in ("Av", "T0"):
        plus_value = [v["by_rule"]["mark-veto+value"]["claims"] for v in reading["cross_schema_block"][fam].values()]
        assert len(plus_value) == 9 and min(plus_value) >= 5 and max(plus_value) <= 16
    assert bounds([v["by_rule"]["mark-veto+value"]["claims"] for fam in ("Av", "T0") for v in reading["cross_schema_block"][fam].values()]) == (5, 16)
    assert reading["R"]["near_name_block_enters"] is False


def test_row52_the_primary_tests_are_not_separated_and_the_letter_is_r_b_by_vacuity(reading, prov52):
    rr = reading["R"]
    assert rr["letter"] == "R-b" and rr["note"].startswith("the organ's stack holds the four conditions on no live seed")
    assert rr["sentence"].startswith("the read side brings no measured advantage over the classical stack on these families") and rr["sentence"].endswith("answered against the organ")
    pt = rr["primary_tests"]
    assert pt["holm_alpha"] == 0.05 and pt["form"] == "direct" and set(pt["tests"]) == set(R.PRIMARY_TESTS)
    for name, t in pt["tests"].items():
        live, third = t["live_seeds_pooled"], t["third_checkpoint"]
        assert (live["pairs"], live["both"], live["neither"], live["b_organ_only"], live["c_reference_only"], live["p_two_sided"]) == (400, 0, 400, 0, 0, 1.0)
        assert (third["pairs"], third["b_organ_only"], third["c_reference_only"], third["p_two_sided"]) == (200, 0, 0, 1.0)
        assert live["organ_strict_own"] == 0 == live["reference_strict_own"] and t["outcome"] == "not separated at this size" and not t["rejected_under_holm"]
        assert t["resolution"] == R.RESOLUTION_SENTENCE and "equal" not in t["outcome"]
    assert rr["conditions"]["organ"] == {r: False for r in RUNS}
    assert rr["conditions"]["buyers"] == {"I2 V1": {r: False for r in RUNS}, "I1 V1": {r: False for r in RUNS}}
    assert prov52["row52"]["letter"] == "R-b" and prov52["row52"]["primary_tests"]["I0 mark-veto against I2 V1"]["outcome"] == "not separated at this size"


def test_row52_the_four_conditions_per_configuration_on_a(reading):
    cells = reading["readings"]["A"]["cells"]
    oracle_holds = {}
    for index in INDEXES:
        for form in FORMS:
            for r in RUNS:
                rules = cells[cell_key(index, form, r)]["rules"]
                for rule in ("plain", "mark-veto", "mark-veto+value", "V1", "V1+organ-value", "oracle+organ-value"):
                    assert not rules[rule]["conditions"]["holds"], (index, form, r, rule)
                c = rules["oracle"]["conditions"]
                if c["holds"]:
                    oracle_holds[cell_key(index, form, r)] = (rules["oracle"]["counts"]["strict_own"], rules["oracle"]["counts"]["invalid"], c["never_planted"]["all"]["claims"])
                else:
                    assert rules["oracle"]["counts"]["invalid"] <= 21 and 3 <= c["never_planted"]["all"]["claims"] <= 62
                    assert all(b.startswith(("invalid citations", "claims on never planted entities", "gap (the frozen strict recall, any line)")) for b in c["broken"])
                    if any(b.startswith("gap") for b in c["broken"]):                # the oracle's own line under the half: seed 2024's two I0 cells
                        assert index == "I0" and r == S2024 and rules["oracle"]["counts"]["strict_own"] == {"direct": 94, "paraphrase": 99}[form]
                assert rules["V1"]["counts"] ["claims"] == 0 and rules["V1"]["counts"]["abstain"] == 200
    assert oracle_holds == ORACLE_HOLDS_ON_A
    failing = [c for c in cells.values() if not c["rules"]["oracle"]["conditions"]["holds"]]
    assert len(failing) == 14 and max(c["rules"]["oracle"]["counts"]["invalid"] for c in failing) == 21 and max(c["rules"]["oracle"]["conditions"]["never_planted"]["all"]["claims"] for c in failing) == 62
    assert bounds([c["rules"]["oracle"]["conditions"]["never_planted"]["all"]["claims"] / 2 for c in cells.values()], ndigits=1) == (1.5, 31.0)   # the mark alone claims on 1.5 to 31 % over the 18 cells
    # V1 abstains on every question of A: no cited line carries the entity together with the answered value; the value is a byte substring of the cited line on 2 of 8 100 questions
    substring = []
    n_questions = 0
    for index in INDEXES:
        for form in FORMS:
            for r in RUNS:
                p = load(f"sealed-A-{r}-{index}-{form}.pass.json")
                for a in p["answers"]:
                    n_questions += 1
                    pl = a["rules"]["plain"]
                    assert a["rules"]["V1"]["claim"] is False
                    if pl["kind"] == "cite" and pl["cited_pointer"] and pl["answered"] and pl["answered"] in p["texts"].get(pl["cited_pointer"], ""):
                        substring.append((r, index, form, a["index"], pl["answered"]))
    assert n_questions == 8100 and substring == [(S2024, "I1", "direct", 33, "on"), (S7, "I1", "direct", 368, "ar")]


def test_row52_the_attribution_by_component(reading):
    at = reading["R"]["attribution"]
    for i, r in enumerate(RUNS):
        for index in INDEXES:
            v = at["index_at_fixed_rule"][f"mark-veto-{index}-{r}"]
            assert (v["own_in_window"]["k"], v["own_in_window"]["n"]) == (OWN_IN_WINDOW[index]["direct"][i], 200)
            assert (v["own_cited_given_in_window"]["k"], v["own_cited_given_in_window"]["n"]) == (OWN_CITED_GIVEN_WINDOW["mark-veto"][index][i], OWN_IN_WINDOW[index]["direct"][i])
            v1 = at["index_at_fixed_rule"][f"V1-{index}-{r}"]
            assert v1["own_cited_given_in_window"]["k"] == 0 and v1["own_in_window"]["k"] == OWN_IN_WINDOW[index]["direct"][i]
            for rule, expected in NEVER_PLANTED_BY_RULE[index].items():
                assert at["rule_at_fixed_index"][f"{index}-{rule}-{r}"]["never_planted_claims"]["k"] == expected[i], (index, rule, r)
            n, model, oracle, b, c = MODEL_AGAINST_ORACLE[index][i]
            mo = at["model_against_oracle_at_fixed_index"][f"{index}-{r}"]
            assert (mo["n_reads_with_own_in_window"], mo["model_rate"]["k"], mo["oracle_rate"]["k"], mo["b_model_only"], mo["c_oracle_only"]) == (n, model, oracle, b, c)
            assert mo["p_two_sided"] == pytest.approx(binomial_two_sided_p(b, c)) and mo["both"] + mo["neither"] + b + c == n
        for rule, expected in INVALID_BY_RULE_I0.items():
            assert at["rule_at_fixed_index"][f"I0-{rule}-{r}"]["invalid_citations"]["k"] == expected[i], (rule, r)
    assert at["model_against_oracle_at_fixed_index"][f"I0-{S1337}"]["p_two_sided"] < 1e-18 and at["model_against_oracle_at_fixed_index"][f"I0-{S7}"]["p_two_sided"] < 1e-20
    # the fuller the window, the smaller the model's share: from I0 to I1 on the three checkpoints, from I1 to I2 on two (seed 7 equal at the third decimal)
    for i, r in enumerate(RUNS):
        shares = [OWN_CITED_GIVEN_WINDOW["mark-veto"][index][i] / OWN_IN_WINDOW[index]["direct"][i] for index in INDEXES]
        assert shares[0] > shares[1] and (shares[1] > shares[2] or (r == S7 and round(shares[1], 3) == round(shares[2], 3) == 0.175))


def test_the_nine_cost_files_pinned_state_protocol_and_no_unstable_repetition(cost, prov52):
    for (fam, r), d in cost.items():
        assert d["family"] == fam and d["run_id"] == r and d["code_commit"] == FIRST_COMMIT and d["launcher"] == "tools/adr010_launcher.py --pin-threads"
        assert d["pinned"] == {"thread_variables": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"},
                               "torch_threads": 1, "torch_interop_threads": 1, "pinned": True, "problems": [], "launcher": True, "interop_set_note": None}
        assert d["environment"]["PYTHONHASHSEED"] == "0" and d["environment"]["torch_threads"] == 1 and d["device"]["device"] == "cpu"
        assert d["label"] == "not a scale result: 200 entries, one processor, the build and the query separated; the scale comparison is part C's"
        assert d["threshold"] == THRESHOLD[r] and d["bm25"]["pair"] == {"k1": 1.5, "b": 0.75} and d["bm25"]["all_tie"]
        pass_file = load(f"sealed-{fam}-{r}-I0-direct.pass.json")
        assert d["planting_sha256_lf"] == pass_file["identity"]["compared"]["planting_record_sha256"] and d["seal"] == pass_file["seal"]
        rs = d["read_side"]
        assert rs["protocol_2"] is None and rs["configurations"] == list(INDEXES) and rs["rules"] == list(R.RULES)
        p1 = rs["protocol_1"]
        assert p1["repetitions"] == 6 and len(p1["orders"]) == 6 and len({tuple(o) for o in p1["orders"]}) == 6 and p1["timer_overhead_ns"] == 100.0
        assert p1["n_questions"] == (400 if fam == "A" else 250) and p1["collections_outside_stages"] in (25, 26)
        for index in INDEXES:
            for stage, v in p1["stages"][index].items():
                if isinstance(v, dict) and "median" in v and stage != "operations":
                    assert v["flag"] is False and v["unstable_repetitions"] == [] and len(v["repetition_medians"]) == 6 and v["unit"] == "ns", (fam, r, index, stage)
                    assert v["n"] == 6 * p1["n_questions"]
            assert p1["stages"][index]["operations"]["what"] == {"I0": "candidates scored (the index's touched increment)", "I1": "cues compared", "I2": "postings read"}[index]
        assert p1["stages"]["I0"]["marks"]["absent"] is True
        ws = d["write_side"]
        assert ws["repetitions"] == 6 and ws["n_statements"] == 200 and ws["cues"] == {"what": ws["cues"]["what"], "largest_abs_difference": 0.0, "compared": 200}
        assert all(v.get("flag") is False for k, v in ws["stages"].items() if k != "append_derived") and ws["stages"]["append_derived"]["label"].startswith("derived")
        assert d["generation"]["n"] == 50
        pc = prov52["row52"]["cost"][f"{fam}-{r}"]
        assert pc["started_utc"] == d["started_utc"] and pc["pinned"] and pc["problems"] == [] and not pc["protocol_2_ran"]
        assert pc["index_step_median_ns"] == {index: p1["stages"][index]["index"]["median"] for index in INDEXES} and pc["read_side_build_bytes"] == d["memory"]["read_side_build_bytes"]


def test_the_cost_by_stage_as_the_row_states_it(cost, prov52):
    """Every range the row prints is the files' attained minimum and maximum at the row's precision (over the nine files; over the
    twenty seven values for a stage timed under each index)."""
    st = lambda d, index, stage, q="median": d["read_side"]["protocol_1"]["stages"][index][stage][q]  # noqa: E731
    us, ms = 1e3, 1e6
    nine = list(cost.values())
    per_index = lambda stage, q="median": [st(d, index, stage, q) for d in nine for index in INDEXES]  # noqa: E731
    assert bounds(per_index("cue"), ms, 1) == (20.0, 23.1) and bounds(per_index("cue", "p95"), ms, 1) == (22.5, 29.0)
    assert bounds([st(d, "I0", "index") for d in nine], us) == (235, 243) and bounds([st(d, "I1", "index") for d in nine], us) == (512, 524)
    assert bounds([st(d, "I2", "index") for d in nine], us) == (65, 89)
    assert bounds([st(d, "I0", "operations") for d in nine]) == (5, 6) and bounds([st(d, "I0", "operations", "p95") for d in nine]) == (9, 14)
    assert all(st(d, "I1", "operations") == 200 for d in nine) and all(st(d, "I2", "operations") == (40 if fam == "A" else 82) for (fam, r), d in cost.items())
    assert bounds([st(d, "I2", "operations", "p95") for d in nine]) == (162, 240)
    assert bounds([st(d, "I1", "marks") for d in nine], us) == (48, 49) and bounds([st(d, "I2", "marks") for d in nine], us) == (89, 92)
    assert bounds(per_index("window"), us) == (30, 34)
    assert bounds([st(d, "I2", "index") + st(d, "I2", "window") for d in nine], us) == (97, 122)                      # I2's stack without the organ's pass (B.1 (b))
    rules = {rule: per_index(rule) for rule in R.RULES}
    assert set(rules["plain"]) == {300.0}
    assert bounds(rules["mark-veto"], us, 1) == (4.2, 5.0) and bounds(rules["mark-veto+value"], us, 1) == (0.9, 1.0) and bounds(rules["V1"], us, 1) == (5.2, 8.1)
    assert bounds(rules["V1+organ-value"], us, 1) == (20.6, 30.9) and bounds(rules["oracle"], us, 1) == (1.0, 1.7) and bounds(rules["oracle+organ-value"], us, 1) == (0.6, 16.9)
    assert bounds(per_index("mark-veto+value", "p95"), us) == (30, 35) and bounds(per_index("oracle+organ-value", "p95"), us) == (18, 20)
    ratios = {index: [st(d, index, "cue") / st(d, index, "index") for d in nine] for index in INDEXES}            # the encoder's pass against each index step
    assert bounds(ratios["I1"], 1, 1) == (38.6, 44.1) and bounds(ratios["I0"]) == (84, 96) and bounds(ratios["I2"]) == (229, 348)
    spreads_index = [st(d, index, "index", "spread_of_medians") for d in nine for index in INDEXES]
    spreads_cue = [st(d, index, "cue", "spread_of_medians") for d in nine for index in INDEXES]
    assert bounds(spreads_index, us, 1) == (0.4, 6.3) and bounds(spreads_cue, ms, 1) == (0.3, 1.3)
    w = lambda d, stage, q="median": d["write_side"]["stages"][stage][q]                                # noqa: E731
    assert bounds([w(d, "cue") for d in nine], ms, 1) == (21.6, 28.6) and bounds([w(d, "gate") for d in nine], us) == (258, 268)
    assert bounds([w(d, "insert_I0") for d in nine], us) == (148, 152) and bounds([w(d, "insert_I1") for d in nine], us, 1) == (3.0, 3.1)
    assert bounds([w(d, "insert_I2") for d in nine], us, 1) == (14.8, 16.8) and bounds([w(d, "store_write") for d in nine], ms, 2) == (1.57, 1.69)
    assert bounds([w(d, "seal_payload") for d in nine], us) == (28, 34) and bounds([w(d, "seal_line") for d in nine], us) == (13, 15)
    assert bounds([w(d, "append_derived") for d in nine], ms, 2) == (1.53, 1.64)
    for d in nine:                                                                   # the derived append: the whole store write minus the sealing, on the medians
        assert w(d, "append_derived") == pytest.approx(w(d, "store_write") - w(d, "seal_payload") - w(d, "seal_line"))
        assert d["write_side"]["durability"].startswith("the write's durability is the organ's as frozen")
    gen = {key: d["generation"]["median"] / ms for key, d in cost.items()}
    assert round(gen[("A", S2024)], 1) == 86.5 and round(gen[("A", S7)], 1) == 86.7 and round(gen[("A", S1337)], 1) == 384.5
    assert bounds([v for key, v in gen.items() if key[0] != "A"]) == (405, 464) and bounds([d["generation"]["p95"] / ms for d in nine]) == (527, 1034)
    assert all(d["generation"]["label"].startswith("the generation, a common stage") for d in nine)
    mem = {key: d["memory"] for key, d in cost.items()}
    assert bounds([m["read_side_build_bytes"]["I0"] for m in mem.values()], 1e3) == (481, 581) and bounds([m["index_sizes"]["I0"]["buckets"] for m in mem.values()]) == (2458, 2786)
    assert bounds([m["read_side_build_bytes"]["I1"] for m in mem.values()], 1e3, 1) == (11.7, 11.8) and bounds([m["read_side_build_bytes"]["I2"] for m in mem.values()], 1e3) == (140, 150)
    assert bounds([m["index_sizes"]["I2"]["tokens"] for m in mem.values()]) == (448, 483) and bounds([m["index_sizes"]["I2"]["postings"] for m in mem.values()]) == (1285, 1520)
    assert bounds([m["journal_open_bytes"] for m in mem.values()], 1e3) == (686, 686) and bounds([m["journal_payloads_read_bytes"] for m in mem.values()], 1e3) == (19, 21)
    assert bounds([m["journal_on_disk_bytes"] for m in mem.values()], 1e3) == (249, 252) and bounds([m["checkpoint_bytes"] for m in mem.values()], 1e6) == (325, 325)
    assert all(m["index_sizes"]["I0"]["tables"] == 16 and m["entries"] == 200 for m in mem.values())
    for (fam, r), d in cost.items():
        assert d["maintenance_proxies"]["tunable_parameters"]["I0"]["count"] == 4 and d["maintenance_proxies"]["tunable_parameters"]["I1"]["count"] == 0
        assert d["maintenance_proxies"]["tunable_parameters"]["I2"] == {"k1": 1.5, "b": 0.75, "count": 2} and d["maintenance_proxies"]["lines_of_code"] == {"I0": 228, "I1": 40, "I2": 95}
        assert d["maintenance_proxies"]["label"] == "proxies, not measurements of maintenance"
        assert prov52["row52"]["cost"][f"{fam}-{r}"]["cue_median_ns"] == st(d, "I0", "cue")


def test_the_provenance_states_the_two_rows(prov51, prov52, reading):
    assert prov51["rows"] == "row 51" and prov51["row51_letter"] == "A3" == reading["letter_A"]["letter"] and prov51["row51_m2_sentence"] == R.M2_SENTENCES["organ"]
    r51 = prov51["row51"]
    assert r51["M2"]["A"] == {"b_model_only": 5, "c_oracle_only": 287, "n_reads_with_own_in_window": 466, "p_two_sided": reading["M"]["families"]["A"]["M2"]["pooled_live_seeds"]["p_two_sided"], "direction": "oracle"}
    assert r51["holm"] == reading["M"]["holm"] and r51["own_in_window_A_direct_I0"] == {S2024: 110, S1337: 108, S7: 117}
    assert r51["factor_product_plain_I0_direct"][S2024] == {"A": 0.0, "Av": 0.0, "T0": 0.795, "first": 0.815} and r51["transfer"] == TRANSFER
    assert r51["first_family_strict_own"] == {S2024: {"direct_plain": 163, "paraphrase_plain": 166}, S1337: {"direct_plain": 164, "paraphrase_plain": 159}, S7: {"direct_plain": 162, "paraphrase_plain": 171}}
    assert len(r51["T0_own_in_window"]) == 18 and set(r51["T0_own_in_window"].values()) == {200}
    assert r51["sessions"]["passes_in_one_session"] == 54 and r51["sessions"]["first_family_passes_in_two_sessions"] == 3 and r51["departures_stated"] == reading["departures_stated"]
    assert prov52["rows"] == "row 52" and prov52["row52_letter"].startswith("R-b (by vacuity") and prov52["row52_primary_tests"].startswith("not separated at this size")
    r52 = prov52["row52"]
    assert r52["own_in_window_A_direct_by_index"] == {"I0": {S2024: 110, S1337: 108, S7: 117}, "I1": {S2024: 180, S1337: 188, S7: 183}, "I2": {S2024: 200, S1337: 200, S7: 200}}
    assert r52["own_cited_given_in_window_mark_veto"] == {"I0": {S2024: 30, S1337: 34, S7: 28}, "I1": {S2024: 35, S1337: 41, S7: 32}, "I2": {S2024: 32, S1337: 42, S7: 35}}
    assert r52["conditions"] == reading["R"]["conditions"] and r52["sentence"] == reading["R"]["sentence"] and len(r52["cost"]) == 9
    for prov in (prov51, prov52):
        assert prov["freeze"]["commit"] == FREEZE_COMMIT and prov["freeze"]["files"] == 104 and prov["device"].startswith("the founder's processor (cpu")
        assert "PYTHONHASHSEED 0" in prov["device"] and prov["hash_note"].startswith("sha256 of the file's bytes with CRLF normalised to LF")


def test_the_windows_the_marks_against_the_index_and_a0_s_denominators(loaded, reading):
    """The window sizes per index (B.1 (a)); I2's tied scores and I0's own ties (B.1 (b), (c)); the marks printed on the reference passes
    against the index's own cosine (one float32 unit in the last place at most); A0's 466 of 798 as the own line in the window."""
    import numpy as np
    from collections import Counter
    sizes = {}
    for fam in FAMILIES:
        for r in RUNS:
            for (index, form), p in loaded[fam][r]["passes"].items():
                planted = Counter(a["window_size"] for a in p["answers"] if a["block"] == "planted")
                never = Counter(a["window_size"] for a in p["answers"] if a["block"] == "never_planted")
                blocks = Counter((a["block"], a.get("seed")) for a in p["answers"])
                expected_never = {"A": {("never_planted", 30001): 50, ("never_planted", 30003): 150}, "T0": {("never_planted", 70001): 50}, "Av": {("never_planted", 35001): 50}}[fam]
                assert blocks == {("planted", None): 200, **expected_never, **({("cross_schema", None): 50, ("near_name", None): 50} if form == "direct" else {})}, (fam, r, index, form)
                if index != "I0":
                    assert set(planted) == {4} and set(never) == {4}, (fam, r, index, form)              # four lines on every question under I1 and I2
                else:
                    sizes[(fam, r, form)] = (planted, never)
    for i, r in enumerate(RUNS):
        for form in FORMS:
            planted, never = sizes[("A", r, form)]
            assert tuple(planted.get(k, 0) for k in (4, 3, 2, 1, 0)) == WINDOW_SIZES_A[form][i], (r, form)
            if form == "direct":
                assert (never[4], never.get(0, 0)) == WINDOW_SIZES_A_NEVER_PLANTED_DIRECT[i] and sum(never.values()) == 200
    full = [sizes[(fam, r, form)][0][4] for fam in ("T0", "Av") for r in RUNS for form in FORMS]
    assert bounds(full) == (163, 182) and all(sizes[(fam, r, form)][0].get(0, 0) == 0 for fam in ("T0", "Av") for r in RUNS for form in FORMS)
    # I2: a tied score among the four lines of every window (the fillers, the earliest written documents at that score); I0: no two bit equal marks in any window
    i2_windows = [a["reference"]["reference_scores"] for fam in FAMILIES for r in RUNS for form in FORMS for a in loaded[fam][r]["passes"][("I2", form)]["answers"]]
    assert len(i2_windows) == 6300 and all(len(set(w)) < len(w) for w in i2_windows)
    i0_windows = [[w["mark"] for w in (a["window"] or [])] for fam in FAMILIES for r in RUNS for form in FORMS for a in loaded[fam][r]["passes"][("I0", form)]["answers"]]
    assert len(i0_windows) == 6300 and all(len(set(w)) == len(w) for w in i0_windows) and sum(len(w) for w in i0_windows) == 23338
    assert all(float(np.float32(m)) == m for w in i0_windows for m in w)
    # the marks printed on the reference passes against the index's own cosine of the own line (float32 arithmetic under both, the norms raised by 1e-8)
    compared = bit_equal = 0
    largest = 0.0
    printed = 0
    for fam in FAMILIES:
        for r in RUNS:
            for index in ("I1", "I2"):
                for form in FORMS:
                    for a in loaded[fam][r]["passes"][(index, form)]["answers"]:
                        for w in (a["window"] or []):
                            printed += 1
                            assert float(np.float32(w["mark"])) == w["mark"]
                        if a["own_in_window"] and a.get("mark_own") is not None and (a.get("whole_store") or {}).get("own_cosine") is not None:
                            compared += 1
                            d = abs(a["mark_own"] - a["whole_store"]["own_cosine"])
                            bit_equal += d == 0
                            largest = max(largest, d)
    assert (compared, bit_equal, printed) == (7998, 6833, 50400) and largest == pytest.approx(1.1920928955078125e-07) and largest < 1e-6
    own_i0 = [(a["mark_own"], a["whole_store"]["own_cosine"]) for fam in FAMILIES for r in RUNS for form in FORMS for a in loaded[fam][r]["passes"][("I0", form)]["answers"]
              if a["own_in_window"] and a.get("mark_own") is not None and (a.get("whole_store") or {}).get("own_cosine") is not None]
    assert len(own_i0) == 3469 and all(m == c for m, c in own_i0)                        # the I0 pass prints the index's own score
    # A0 on the live seeds' I0 passes of A: its top line is the own line exactly where the own line is in the window; one empty planted window of seed 2024 per form not scored
    for r, form, k, scored in ((S2024, "direct", 110, 199), (S2024, "paraphrase", 115, 199), (S7, "direct", 117, 200), (S7, "paraphrase", 124, 200), (S1337, "direct", 108, 200), (S1337, "paraphrase", 122, 200)):
        b = load(f"a0-sealed-A-{r}-I0-{form}.pass.json")["blocks"]["planted"]
        assert (b["n_answers"], b["n_scored"], b["bag"]["top_is_own"]) == (200, scored, k) and k == reading["readings"]["A"]["cells"][cell_key("I0", form, r)]["rules"]["plain"]["factors"]["own_in_window"]["k"]
        assert sizes[("A", r, form)][0].get(0, 0) == 200 - scored
    ceiling = reading["M"]["families"]["A"]["M2"]["a0_ceiling"]["bag_top_is_own_pooled"]
    assert (ceiling["k"], ceiling["n"]) == (110 + 115 + 117 + 124, 199 + 199 + 200 + 200) == (466, 798) == (reading["M"]["families"]["A"]["M2"]["pooled_live_seeds"]["n_reads_with_own_in_window"], 798)
