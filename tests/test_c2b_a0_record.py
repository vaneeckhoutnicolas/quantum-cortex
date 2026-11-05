"""Row 54: A0, the construction's ceiling under byte keys (ADR-008 part A, delivered and run as ADR-010 part E says), read
cold from the twenty seven files `cortex_c2b.a0` wrote on 2026-10-30 on committed files of the record (the E1 arm of row
48 and the E2 arm under the identity rule of row 49, the I0 passes of A, Av and T0 in both forms, the first family's I0
paraphrase passes, on the three checkpoints). Everything recomputed from the retained files, their sources, the log and
`PROVENANCE-a0.json`; these tests fail if a digit of the row moves."""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
M = ROOT / "metrics" / "mqar"

from cortex_c2b import a0 as A0                                   # noqa: E402

S1337, S2024, S7 = "8ceba5db8d0b", "4cfd6f6e1406", "2e1dc71913b1"
RUNS = (S1337, S2024, S7)                                         # the rows' order: 1337, 2024, 7
THRESHOLD = {S1337: 0.5360028743743896, S2024: 0.6122402548789978, S7: 0.5907554626464844}
A0_PY = "ccb01702f23ff325"
NAME_BYTES = 9                                                    # the generator's names: four letters, a dash, four hexadecimal digits

ARM = {"E1": "a0-hm-lm-{r}-near-dup-E1-session-b-index16exact-mark-veto-value.json",
       "E2": "a0-hm-lm-{r}-near-dup-E2-identity-session-b-index16exact-mark-veto-value.json"}
PASS = "a0-sealed-{fam}-{r}-I0-{form}.pass.json"

# ---- the row's numbers, per checkpoint in the order 1337, 2024, 7 --------------------------------------------------
# the planted block of the I0 passes: the reads scored, the own line in the window (A0's top line is the own line alone on
# every one of them), the decoder's raw citation of the own line and of another line on those reads, the mark oracle's
SCORED_A = {"direct": (200, 199, 200), "paraphrase": (200, 199, 200)}
OWN_IN_WINDOW = {"A": {"direct": (108, 110, 117), "paraphrase": (122, 115, 124)},
                 "Av": {"direct": (200, 200, 199), "paraphrase": (200, 200, 200)},
                 "T0": {"direct": (200, 200, 200), "paraphrase": (200, 200, 200)}, "first": {"paraphrase": (200, 200, 200)}}
DECODER_OWN = {"A": {"direct": (35, 32, 29), "paraphrase": (43, 32, 34)}, "Av": {"direct": (179, 177, 169), "paraphrase": (181, 173, 174)},
               "T0": {"direct": (189, 186, 181), "paraphrase": (186, 188, 190)}, "first": {"paraphrase": (192, 187, 189)}}
DECODER_OTHER = {"A": {"direct": (25, 29, 32), "paraphrase": (40, 27, 37)}, "Av": {"direct": (6, 5, 12), "paraphrase": (7, 7, 7)},
                 "T0": {"direct": (2, 3, 3), "paraphrase": (6, 3, 3)}, "first": {"paraphrase": (3, 3, 2)}}
ORACLE_OWN = {"A": {"direct": (103, 94, 103), "paraphrase": (119, 99, 113)}, "Av": {"direct": (200, 200, 199), "paraphrase": (200, 200, 200)},
              "T0": {"direct": (200, 200, 200), "paraphrase": (200, 200, 200)}, "first": {"paraphrase": (200, 200, 200)}}
# A's reads without the own line in the window: A0's top tied, the decoder citing a line, citing A0's top line, the oracle citing a line
A_OWN_ABSENT = {"direct": {"ties": (30, 22, 22), "decoder_cites": (35, 25, 37), "decoder_cites_a0_top": (4, 11, 7), "oracle_cites": (15, 4, 6)},
                "paraphrase": {"ties": (18, 22, 28), "decoder_cites": (34, 27, 30), "decoder_cites_a0_top": (5, 3, 6), "oracle_cites": (11, 2, 6)}}
# the arms (the `on` block): E1 under the legacy gate (the decoder's citation after the veto), E2 under the identity rule (raw)
E1 = {"ties": (0, 2, 8), "decoder_own": (192, 186, 183), "decoder_other": (0, 1, 3), "oracle_own": (200, 198, 198), "nearest_other_at_or_above_0_8": (0, 2, 8)}
E2 = {"ties": (0, 0, 0), "decoder_own": (95, 85, 101), "decoder_other": (105, 115, 99), "oracle_own": (177, 180, 171), "nearest_other_at_or_above_0_8": (199, 200, 200)}
# the letters per file (ADR-008's fixed thresholds on the bag share of the planted block; the decoder's own citation share)
LETTERS = {("A", "direct"): "a partial ceiling", ("A", "paraphrase"): "a partial ceiling", ("T0", "direct"): "A0 (iii)", ("T0", "paraphrase"): "A0 (iii)",
           ("first", "paraphrase"): "A0 (iii)", ("E1", None): "A0 (iii)", ("E2", None): "A0 (ii)"}
LETTERS_AV = {"direct": ("A0 (ii)", "A0 (ii)", "A0 (ii)"), "paraphrase": ("A0 (iii)", "A0 (ii)", "A0 (ii)")}
SHARES_A = {"direct": (0.54, 110 / 199, 0.585), "paraphrase": (0.61, 115 / 199, 0.62)}
# the other blocks of the passes (the direct form; A's never planted block 200 questions, the others' 50)
# the never planted blocks, the twenty one (A 200 questions, the others 50; no own line): the reads scored, A0's top tied, the decoder citing a line, citing A0's
# unique top line, the oracle citing a line, the largest top score (7 of 11 = 0.6364 everywhere but one read at 0.8, A seed 7 paraphrase; 0.5 on three blocks)
NEVER_PLANTED = {("A", "direct"): dict(n=200, scored=(196, 196, 197), ties=(70, 66, 66), decoder_cites=(78, 63, 81), decoder_cites_a0_top=(8, 8, 14), oracle_cites=(31, 11, 12), top_max=(0.6364, 0.6364, 0.6364)),
                 ("A", "paraphrase"): dict(n=200, scored=(200, 198, 197), ties=(66, 62, 65), decoder_cites=(84, 67, 75), decoder_cites_a0_top=(14, 14, 15), oracle_cites=(38, 11, 9), top_max=(0.6364, 0.6364, 0.8)),
                 ("Av", "direct"): dict(n=50, scored=(50, 50, 49), ties=(17, 19, 9), decoder_cites=(16, 8, 9), decoder_cites_a0_top=(3, 1, 3), oracle_cites=(8, 3, 4), top_max=(0.6364, 0.6364, 0.6364)),
                 ("Av", "paraphrase"): dict(n=50, scored=(50, 49, 50), ties=(12, 20, 20), decoder_cites=(10, 12, 10), decoder_cites_a0_top=(1, 3, 1), oracle_cites=(7, 2, 5), top_max=(0.5, 0.6364, 0.6364)),
                 ("T0", "direct"): dict(n=50, scored=(49, 49, 49), ties=(16, 22, 14), decoder_cites=(7, 12, 8), decoder_cites_a0_top=(0, 2, 1), oracle_cites=(5, 4, 4), top_max=(0.6364, 0.6364, 0.6364)),
                 ("T0", "paraphrase"): dict(n=50, scored=(50, 50, 50), ties=(18, 21, 15), decoder_cites=(3, 6, 11), decoder_cites_a0_top=(0, 1, 4), oracle_cites=(4, 4, 7), top_max=(0.5, 0.6364, 0.6364)),
                 ("first", "paraphrase"): dict(n=50, scored=(50, 50, 50), ties=(15, 15, 13), decoder_cites=(6, 10, 7), decoder_cites_a0_top=(1, 2, 2), oracle_cites=(4, 2, 1), top_max=(0.6364, 0.5, 0.6364))}
NEVER_PLANTED_TOP = 7 / 11                                     # nine bytes of the name against a nine byte other name: seven in common at most but one read
NEAR_NAME = {"A": {"top_at_0_8": (27, 27, 30), "ties": (9, 8, 9), "decoder_cites": (25, 25, 23), "decoder_cites_a0_top": (12, 12, 6), "oracle_cites": (32, 26, 29)},
             "Av": {"top_at_0_8": (50, 50, 50), "ties": (0, 0, 0), "decoder_cites": (40, 41, 45), "decoder_cites_a0_top": (39, 38, 41), "oracle_cites": (50, 50, 50)},
             "T0": {"top_at_0_8": (50, 50, 50), "ties": (0, 0, 0), "decoder_cites": (46, 47, 49), "decoder_cites_a0_top": (45, 47, 48), "oracle_cites": (50, 50, 50)}}
# the near name blocks: the decoder's raw citation and the oracle's citation landing on the neighbour's line, of the reads where it is in the window
NEAR_NAME_NEIGHBOUR_CITED = {"A": {"decoder": (7, 10, 5), "oracle": (26, 22, 26)}, "Av": {"decoder": (39, 38, 41), "oracle": (50, 50, 50)}, "T0": {"decoder": (45, 47, 48), "oracle": (50, 50, 50)}}
CROSS_SCHEMA = {"A": {"scored": (50, 49, 50), "own_in_window": (33, 33, 29), "decoder_own": (8, 7, 10), "decoder_other": (8, 9, 9), "oracle_own": (31, 30, 27), "ties_own_absent": (3, 8, 9)},
                "Av": {"scored": (50, 50, 50), "own_in_window": (49, 45, 47), "decoder_own": (5, 9, 10), "decoder_other": (5, 11, 9), "oracle_own": (49, 45, 47), "ties_own_absent": (0, 3, 0)},
                "T0": {"scored": (50, 50, 50), "own_in_window": (49, 45, 44), "decoder_own": (14, 6, 15), "decoder_other": (5, 11, 5), "oracle_own": (49, 45, 44), "ties_own_absent": (1, 1, 2)}}
# the arms' other blocks
NEGCTRL = {"E1": {"scored": (49, 50, 50), "ties": (16, 21, 21), "decoder_cites": (0, 0, 0), "oracle_cites": (4, 1, 3)},
           "E2": {"scored": (50, 50, 50), "ties": (28, 23, 23), "decoder_cites": (13, 9, 12), "decoder_cites_a0_top": (0, 0, 1), "oracle_cites": (4, 3, 3)}}
E2_NEAR_NAME_NEVER_PLANTED = {"ties": (50, 50, 50), "decoder_cites": (50, 49, 48), "oracle_cites": (50, 50, 50)}
E2_NEIGHBOUR = {"own_known": (92, 94, 94), "top_at_1_unique": (200, 200, 200), "oracle_own": (81, 85, 78)}
# the template entity agreement (parsed, disagree) per file kind
PARSED = {"A": 0, "Av": 200, "T0": 200, "first": 200, "E2": 400}
PARSED_E1 = (200, 202, 208)
# pooled over the three checkpoints and the forms, the planted block (PROVENANCE-a0.json, row54)
POOLED = {"A_I0": {"scored": 1198, "own_in_window": 696, "bag_top_is_own": 696, "decoder_cites_own": 205, "decoder_cites_another_line_own_present": 190, "mark_oracle_cites_own": 631},
          "Av_I0": {"scored": 1200, "own_in_window": 1199, "bag_top_is_own": 1199, "decoder_cites_own": 1053, "decoder_cites_another_line_own_present": 44, "mark_oracle_cites_own": 1199},
          "T0_I0": {"scored": 1200, "own_in_window": 1200, "bag_top_is_own": 1200, "decoder_cites_own": 1120, "decoder_cites_another_line_own_present": 20, "mark_oracle_cites_own": 1200},
          "first_I0_paraphrase": {"scored": 600, "own_in_window": 600, "bag_top_is_own": 600, "decoder_cites_own": 568, "decoder_cites_another_line_own_present": 8, "mark_oracle_cites_own": 600},
          "E1_row48": {"scored": 600, "own_in_window": 600, "bag_top_is_own": 590, "ties_with_own_present": 10, "decoder_cites_own": 561, "decoder_cites_another_line_own_present": 4, "mark_oracle_cites_own": 596},
          "E2_identity_row49": {"scored": 600, "own_in_window": 600, "bag_top_is_own": 600, "decoder_cites_own": 281, "decoder_cites_another_line_own_present": 319, "mark_oracle_cites_own": 528}}
# the binned reading, pooled per family over the reads with the own line in the window: (n, the decoder's own citation, the oracle's)
BINS = {"A": {"under_0.5": (612, 172, 553), "0.5_to_under_0.8": (75, 26, 69), "0.8_and_above": (0, 0, 0)},
        "Av": {"under_0.5": (1049, 919, 1049), "0.5_to_under_0.8": (140, 124, 140), "0.8_and_above": (0, 0, 0)},
        "T0": {"under_0.5": (1084, 1011, 1084), "0.5_to_under_0.8": (106, 99, 106), "0.8_and_above": (0, 0, 0)},
        "first": {"under_0.5": (529, 500, 529), "0.5_to_under_0.8": (68, 65, 68), "0.8_and_above": (0, 0, 0)},
        "E1": {"under_0.5": (527, 497, 527), "0.5_to_under_0.8": (61, 56, 61), "0.8_and_above": (10, 6, 6)},
        "E2": {"under_0.5": (1, 1, 1), "0.5_to_under_0.8": (0, 0, 0), "0.8_and_above": (599, 280, 527)}}
# the reads where the own line is alone in its window (no other line, no bin): n, the decoder's own citation, the oracle's
ALONE = {"A": (9, 7, 9), "Av": (10, 10, 10), "T0": (10, 10, 10), "first": (3, 3, 3), "E1": (2, 2, 2), "E2": (0, 0, 0)}


def sha_lf(p: Path, n: int = 16) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:n]


@pytest.fixture(scope="module")
def prov():
    return json.loads((M / "PROVENANCE-a0.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def files(prov):
    out = {}
    for name in prov["files"]:
        out[name] = json.loads((M / name).read_text(encoding="utf-8"))
    assert len(out) == 27
    return out


def arm_file(files, kind, r):
    return files[ARM[kind].format(r=r)]


def pass_file(files, fam, r, form):
    return files[PASS.format(fam=fam, r=r, form=form)]


def planted_reads(d):
    blk = "on" if "on" in d["reads"] else "planted"
    return [x for x in d["reads"][blk] if not x.get("skipped")]


def own_present(reads):
    return [x for x in reads if x.get("own_pointer") and any(l["is_own"] for l in x["lines"])]


def own_absent(reads):
    return [x for x in reads if x.get("own_pointer") and not any(l["is_own"] for l in x["lines"])]


def cites(x):
    return x["decoder_cited_pointer"] is not None


# ---- the files, the hashes, the sources, the log, the code -----------------------------------------------------------
def test_the_twenty_seven_files_their_hashes_the_consoles_printed_values_and_the_log(prov, files):
    assert len(prov["files"]) == 27 and prov["rows"] == "row 54"
    for name, meta in prov["files"].items():
        p = M / name
        assert sha_lf(p) == meta["sha256_lf"] == meta["console_sha256"], name
        assert b"\r\n" not in p.read_bytes()
    log = M / "logs" / "a0-2026-10-30.log"
    assert sha_lf(log) == prov["logs"]["a0-2026-10-30.log"]
    text = log.read_bytes().decode("utf-16")
    printed = dict(re.findall(r"retained: metrics\\mqar\\(a0-\S+)\s+sha256 ([0-9a-f]{16})", text))
    assert len(printed) == 27 and all(printed[n] == prov["files"][n]["console_sha256"] for n in prov["files"])
    # the log holds the retained lines and the module's summaries alone: no command, no time
    assert "timestamp" not in text and "python" not in text


def test_the_sources_are_the_committed_files_the_earlier_provenance_files_pin(prov, files):
    seen = set()
    for name, d in files.items():
        src = M / d["file"]
        full = hashlib.sha256(src.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert full == d["source_sha256_lf"] == prov["sources"][d["file"]]["sha256_lf"], name
        pin = prov["sources"][d["file"]]
        assert pin["pinned_by"] in ("PROVENANCE-sealed-set.json", "PROVENANCE-near-duplicates.json", "PROVENANCE-identity-rule.json") and full.startswith(pin["pinned_value"])
        seen.add(d["file"])
    assert len(seen) == 27
    assert sum(1 for s in prov["sources"].values() if s["pinned_by"] == "PROVENANCE-sealed-set.json") == 21
    assert sum(1 for s in prov["sources"].values() if s["pinned_by"] == "PROVENANCE-near-duplicates.json") == 3
    assert sum(1 for s in prov["sources"].values() if s["pinned_by"] == "PROVENANCE-identity-rule.json") == 3


def test_the_module_is_the_built_one_and_the_files_carry_its_definition(prov, files):
    assert sha_lf(ROOT / "cortex_c2b" / "a0.py") == prov["code"]["a0_py_sha256_lf"] == A0_PY
    for r in RUNS:                                                   # the first family's second sessions recorded the module's hash under 217dd50
        f = json.loads((M / f"sealed-first-{r}-I0-paraphrase.pass.json").read_text(encoding="utf-8"))
        assert [s["code_state"]["new_modules_sha256_lf"]["cortex_c2b/a0.py"] for s in f["sessions"]] == [A0_PY, A0_PY]
    for d in files.values():
        assert d["definition"]["entity_of_line"].startswith("the planting record's text read by the generator's name pattern")
        assert d["definition"]["decisive"] == "a tie at the top is not decisive"
        assert d["threshold"] == THRESHOLD[re.search(r"(8ceba5db8d0b|4cfd6f6e1406|2e1dc71913b1)", d["file"]).group(1)]


def test_every_file_equals_a_recomputation_from_its_source(files):
    for name, d in files.items():
        src = M / d["file"]
        f = json.loads(src.read_text(encoding="utf-8"))
        r = A0.read_file(f, src.name)
        r["source_sha256_lf"] = hashlib.sha256(src.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        r["template_entity_agreement"] = A0.template_entity_agrees(f)
        planted = r["blocks"].get("on") or r["blocks"].get("planted") or {}
        share = (planted.get("bag") or {}).get("share_top_is_own")
        dec = planted.get("decoder", {}).get("cites_own"); nwo = planted.get("n_with_own")
        dshare = (dec / nwo) if (dec is not None and nwo) else None
        r["letter"] = A0.letter_of(share, dshare)
        r["share_top_is_own_planted"] = share; r["decoder_cites_own_share_planted"] = dshare
        assert r == d, name


# ---- what ADR-010 E stated before the reading ------------------------------------------------------------------------
def test_the_own_line_present_scores_one_and_the_full_name_and_is_alone_on_top_except_the_e1_ties(files):
    ties_total = 0
    for name, d in files.items():
        for x in own_present(planted_reads(d)):
            own = [l for l in x["lines"] if l["is_own"]]
            assert len(own) == 1 and own[0]["bag"] == 1.0 and own[0]["aligned"] == NAME_BYTES and own[0]["entity"] == x["query_entity"]
            assert x["bag"]["own_score"] == 1.0 and x["aligned"]["own_score"] == NAME_BYTES and x["aligned"]["own_rank"] in (1, 2) and x["bag"]["own_rank"] in (1, 2)
            if x["bag"]["tie"]:
                ties_total += 1
                assert "-E1-" in name                                # tied only by a line about the same entity: an admitted E1 duplicate
                assert all(l["entity"] == x["query_entity"] for l in x["lines"] if l["bag"] == 1.0) and x["aligned"]["tie"]
            else:
                assert x["bag"]["top_is_own"] and x["aligned"]["top_is_own"]
    assert ties_total == 10


def test_the_name_pattern_equals_the_template_reading_wherever_a_template_parses(files):
    parsed = 0
    for name, d in files.items():
        t = d["template_entity_agreement"]
        assert t["disagree"] == 0 and t["agree"] == t["parsed"]
        kind = "E1" if "-E1-" in name else "E2" if "-E2-identity-" in name else re.search(r"a0-sealed-(A|Av|T0|first)-", name).group(1)
        if kind == "E1":
            r = re.search(r"(8ceba5db8d0b|4cfd6f6e1406|2e1dc71913b1)", name).group(1)
            assert t["parsed"] == PARSED_E1[RUNS.index(r)]
        else:
            assert t["parsed"] == PARSED[kind], name                 # A's statements parse under no template: the pattern is the only source
        parsed += t["parsed"]
    assert parsed == 4810


def test_the_aligned_variant_agrees_with_the_bag_on_the_own_line_on_every_block(files):
    for d in files.values():
        for blk in d["blocks"].values():
            assert blk["aligned"]["top_is_own"] == blk["bag"]["top_is_own"]


def test_the_decoders_field_is_the_raw_citation_except_on_row_48s_files(files):
    for name, d in files.items():
        for blk in d["blocks"].values():
            assert blk["decoder"]["field"] == ("cited_pointer" if "-E1-" in name else "raw_cited_pointer"), name


# ---- the planted block of the passes -------------------------------------------------------------------------------------
@pytest.mark.parametrize("fam,form", [("A", "direct"), ("A", "paraphrase"), ("Av", "direct"), ("Av", "paraphrase"), ("T0", "direct"), ("T0", "paraphrase"), ("first", "paraphrase")])
def test_the_own_line_in_the_window_is_a0s_top_line_and_the_decoder_and_the_oracle_beside(files, fam, form):
    for i, r in enumerate(RUNS):
        d = pass_file(files, fam, r, form)
        reads = planted_reads(d)
        if fam == "A":
            assert len(reads) == SCORED_A[form][i] and d["blocks"]["planted"]["n_answers"] == 200
        else:
            assert len(reads) == 200
        pres, absent = own_present(reads), own_absent(reads)
        assert len(pres) == OWN_IN_WINDOW[fam][form][i] == d["blocks"]["planted"]["bag"]["top_is_own"] == d["blocks"]["planted"]["aligned"]["top_is_own"]
        assert all(x["bag"]["top_is_own"] and not x["bag"]["tie"] for x in pres)
        assert sum(1 for x in pres if x["decoder_cites_own"]) == DECODER_OWN[fam][form][i] == d["blocks"]["planted"]["decoder"]["cites_own"]
        assert sum(1 for x in pres if cites(x) and not x["decoder_cites_own"]) == DECODER_OTHER[fam][form][i]
        assert sum(1 for x in pres if x["mark_oracle_cites_own"]) == ORACLE_OWN[fam][form][i] == d["blocks"]["planted"]["mark_oracle"]["cites_own"]
        assert d["blocks"]["planted"]["n_with_own"] == len(reads)      # every planted read carries its own pointer; the share's denominator is the reads
        if fam == "A":
            a = A_OWN_ABSENT[form]
            assert (sum(1 for x in absent if x["bag"]["tie"]), sum(1 for x in absent if cites(x)), sum(1 for x in absent if x["decoder_agrees_with_bag_top"]), sum(1 for x in absent if x["mark_oracle_pointer"] is not None)) == (a["ties"][i], a["decoder_cites"][i], a["decoder_cites_a0_top"][i], a["oracle_cites"][i])
        else:
            assert len(absent) == (1 if (fam, form, r) == ("Av", "direct", S7) else 0)


def test_the_letters_and_the_shares_as_the_fixed_thresholds_read_them(files):
    for i, r in enumerate(RUNS):
        for (fam, form), letter in LETTERS.items():
            d = arm_file(files, fam, r) if form is None else pass_file(files, fam, r, form)
            assert d["letter"].startswith(letter), (fam, form, r, d["letter"])
            if fam == "A":
                assert abs(d["share_top_is_own_planted"] - SHARES_A[form][i]) < 1e-12 and 0.5 < d["share_top_is_own_planted"] < 0.9
                assert d["share_top_is_own_planted"] == OWN_IN_WINDOW["A"][form][i] / SCORED_A[form][i]
            else:
                assert d["share_top_is_own_planted"] >= 0.96
        for form in ("direct", "paraphrase"):
            d = pass_file(files, "Av", r, form)
            assert d["letter"].startswith(LETTERS_AV[form][i])
            assert d["share_top_is_own_planted"] >= 0.995
            dec = d["decoder_cites_own_share_planted"]
            assert (dec >= 0.9) == (LETTERS_AV[form][i] == "A0 (iii)") and 0.845 <= dec <= 0.905
    # the decoder's own citation shares that split the letters at 0.9: T0 and the first family at or above, Av's five below, E2 under 0.51
    t0 = sorted(pass_file(files, "T0", r, f)["decoder_cites_own_share_planted"] for r in RUNS for f in ("direct", "paraphrase"))
    assert t0[0] == 0.905 and t0[-1] == 0.95
    first = sorted(pass_file(files, "first", r, "paraphrase")["decoder_cites_own_share_planted"] for r in RUNS)
    assert first == [0.935, 0.945, 0.96]
    e2 = sorted(arm_file(files, "E2", r)["decoder_cites_own_share_planted"] for r in RUNS)
    assert e2 == [0.425, 0.475, 0.505]
    e1 = sorted(arm_file(files, "E1", r)["decoder_cites_own_share_planted"] for r in RUNS)
    assert e1 == [0.915, 0.93, 0.96]


# ---- the arms of rows 48 and 49 -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind,exp", [("E1", E1), ("E2", E2)])
def test_the_arms_on_block(files, kind, exp):
    for i, r in enumerate(RUNS):
        d = arm_file(files, kind, r)
        reads = planted_reads(d)
        pres = own_present(reads)
        assert len(reads) == len(pres) == 200 and d["blocks"]["on"]["n_with_own"] == 200
        assert sum(1 for x in pres if x["bag"]["tie"]) == exp["ties"][i]
        assert d["blocks"]["on"]["bag"]["top_is_own"] == 200 - exp["ties"][i]
        assert sum(1 for x in pres if x["decoder_cites_own"]) == exp["decoder_own"][i]
        assert sum(1 for x in pres if cites(x) and not x["decoder_cites_own"]) == exp["decoder_other"][i]
        assert sum(1 for x in pres if x["mark_oracle_cites_own"]) == exp["oracle_own"][i]
        assert sum(1 for x in pres if x["nearest_other_bag"] is not None and x["nearest_other_bag"] >= 0.8) == exp["nearest_other_at_or_above_0_8"][i]
        high = [x for x in pres if x["nearest_other_bag"] is not None and x["nearest_other_bag"] >= 0.8]
        src = json.loads((M / d["file"]).read_text(encoding="utf-8"))
        writes = src["near_duplicates"]["planting"]["writes"]
        if kind == "E1":                                             # the ties are the E1 duplicates the legacy gate admitted (row 48), one tie each
            assert sum(1 for w in writes if w.get("kind") == "E1" and w.get("admitted")) == exp["ties"][i]
        if kind == "E2":                                             # the one byte neighbour in the window: its bag score 0.8 against the own line's 1
            assert all(x["nearest_other_bag"] == 0.8 for x in high)
            assert all(cites(x) for x in pres)                       # the raw decoder cites a line on every read
        else:                                                        # the admitted E1 duplicates: the same entity, the bag score 1, the tie
            assert all(x["nearest_other_bag"] == 1.0 and x["bag"]["tie"] for x in high)


def test_the_arms_other_blocks(files):
    for i, r in enumerate(RUNS):
        for kind in ("E1", "E2"):
            d = arm_file(files, kind, r)
            neg = [x for x in d["reads"]["negctrl"] if not x.get("skipped")]
            e = NEGCTRL[kind]
            assert len(neg) == e["scored"][i] and not any(x.get("own_pointer") for x in neg)
            assert sum(1 for x in neg if x["bag"]["tie"]) == e["ties"][i] and sum(1 for x in neg if cites(x)) == e["decoder_cites"][i]
            assert sum(1 for x in neg if x["mark_oracle_pointer"] is not None) == e["oracle_cites"][i]
            if kind == "E2":
                assert sum(1 for x in neg if x["decoder_agrees_with_bag_top"]) == e["decoder_cites_a0_top"][i]
        d = arm_file(files, "E2", r)
        nn = [x for x in d["reads"]["reported:near_name_never_planted"] if not x.get("skipped")]
        assert len(nn) == 50 and all(x["bag"]["tie"] and x["bag"]["top_score"] == 0.8 for x in nn)   # the original and its admitted variant, both one byte from the name
        assert sum(1 for x in nn if cites(x)) == E2_NEAR_NAME_NEVER_PLANTED["decoder_cites"][i] and sum(1 for x in nn if x["mark_oracle_pointer"] is not None) == 50
        nq = [x for x in d["reads"]["reported:neighbour_questions"] if not x.get("skipped")]
        assert len(nq) == 200 and sum(1 for x in nq if x["bag"]["top_score"] == 1.0 and not x["bag"]["tie"]) == E2_NEIGHBOUR["top_at_1_unique"][i]   # a line carrying the neighbour's name exactly, alone on top
        known = [x for x in nq if x.get("own_pointer")]
        assert len(known) == E2_NEIGHBOUR["own_known"][i] == d["blocks"]["reported:neighbour_questions"]["n_with_own"]
        src = json.loads((M / d["file"]).read_text(encoding="utf-8"))
        assert src["reported_blocks"]["neighbour_questions"]["own_cited"] == len(known)   # the own pointer known through the file's own citation flag alone
        assert all(x["decoder_cites_own"] and x["bag"]["top_is_own"] for x in known)
        assert sum(1 for x in known if x["mark_oracle_cites_own"]) == E2_NEIGHBOUR["oracle_own"][i]


# ---- the other blocks of the passes ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("fam,form", sorted(NEVER_PLANTED))
def test_the_never_planted_blocks(files, fam, form):
    e = NEVER_PLANTED[(fam, form)]
    for i, r in enumerate(RUNS):
        d = pass_file(files, fam, r, form)
        np_ = [x for x in d["reads"]["never_planted"] if not x.get("skipped")]
        assert d["blocks"]["never_planted"]["n_answers"] == e["n"] and len(np_) == e["scored"][i] and not any(x.get("own_pointer") for x in np_)
        assert (sum(1 for x in np_ if x["bag"]["tie"]), sum(1 for x in np_ if cites(x)), sum(1 for x in np_ if x["decoder_agrees_with_bag_top"]), sum(1 for x in np_ if x["mark_oracle_pointer"] is not None)) == (e["ties"][i], e["decoder_cites"][i], e["decoder_cites_a0_top"][i], e["oracle_cites"][i])
        assert round(max(x["bag"]["top_score"] for x in np_), 4) == e["top_max"][i]
        assert all(x["bag"]["top_score"] <= NEVER_PLANTED_TOP + 1e-12 or x["bag"]["top_score"] == 0.8 for x in np_)


def test_one_never_planted_read_in_twenty_one_blocks_meets_a_name_one_byte_away(files):
    at = [(n, x["i"]) for n, d in files.items() if n.startswith("a0-sealed-") for x in d["reads"]["never_planted"] if not x.get("skipped") and x["bag"]["top_score"] > NEVER_PLANTED_TOP + 1e-12]
    assert at == [("a0-sealed-A-2e1dc71913b1-I0-paraphrase.pass.json", 152)]


def test_the_near_name_blocks(files):
    for fam, e in NEAR_NAME.items():
        for i, r in enumerate(RUNS):
            d = pass_file(files, fam, r, "direct")
            nn = [x for x in d["reads"]["near_name"] if not x.get("skipped")]
            assert len(nn) == 50 and not any(x.get("own_pointer") for x in nn)
            assert sum(1 for x in nn if x["bag"]["top_score"] == 0.8) == e["top_at_0_8"][i]     # the planted neighbour, one byte from the name, in the window
            assert all(x["bag"]["top_score"] <= 0.8 for x in nn)
            assert (sum(1 for x in nn if x["bag"]["tie"]), sum(1 for x in nn if cites(x)), sum(1 for x in nn if x["decoder_agrees_with_bag_top"]), sum(1 for x in nn if x["mark_oracle_pointer"] is not None)) == (e["ties"][i], e["decoder_cites"][i], e["decoder_cites_a0_top"][i], e["oracle_cites"][i])
            nb = [x for x in nn if x["bag"]["top_score"] == 0.8]        # the neighbour's line present (its pointer: the line at 0.8)
            on_nb = lambda x, ptr: ptr is not None and ptr in [l["pointer"] for l in x["lines"] if l["bag"] == 0.8]
            dec_nb = sum(1 for x in nb if on_nb(x, x["decoder_cited_pointer"])); orc_nb = sum(1 for x in nb if on_nb(x, x["mark_oracle_pointer"]))
            assert (dec_nb, orc_nb) == (NEAR_NAME_NEIGHBOUR_CITED[fam]["decoder"][i], NEAR_NAME_NEIGHBOUR_CITED[fam]["oracle"][i]), (fam, r)
            if fam != "A":                                           # on Av and T0 the neighbour's line is A0's top line on every read, the oracle's citation on every read
                assert dec_nb == e["decoder_cites_a0_top"][i] and orc_nb == 50


def test_the_cross_schema_blocks(files):
    for fam, e in CROSS_SCHEMA.items():
        for i, r in enumerate(RUNS):
            d = pass_file(files, fam, r, "direct")
            cs = [x for x in d["reads"]["cross_schema"] if not x.get("skipped")]
            pres, absent = own_present(cs), own_absent(cs)
            assert d["blocks"]["cross_schema"]["n_answers"] == 50 and len(cs) == e["scored"][i] and len(pres) == e["own_in_window"][i]
            assert all(x["bag"]["top_is_own"] for x in pres) and d["blocks"]["cross_schema"]["bag"]["top_is_own"] == len(pres)
            assert sum(1 for x in pres if x["decoder_cites_own"]) == e["decoder_own"][i] and sum(1 for x in pres if cites(x) and not x["decoder_cites_own"]) == e["decoder_other"][i]
            assert sum(1 for x in pres if x["mark_oracle_cites_own"]) == e["oracle_own"][i]
            assert sum(1 for x in absent if x["bag"]["tie"]) == e["ties_own_absent"][i]


# ---- the pooled numbers and the binned reading --------------------------------------------------------------------------
def test_the_own_line_present_is_alone_on_top_on_4885_of_4895_reads_and_a_holds_1200_texts(files):
    present = sum(v["own_in_window"] for v in POOLED.values()); alone = sum(v["bag_top_is_own"] for v in POOLED.values())
    assert (present, alone) == (4895, 4885)
    assert sum(d["n_texts"] for n, d in files.items() if n.startswith("a0-sealed-A-")) == 1200       # the family's 200 statements, counted per file


def test_the_pooled_numbers_the_provenance_states(files, prov):
    for key, exp in POOLED.items():
        got = prov["row54"]["pooled_planted_block_per_family"][key]
        for k, v in exp.items():
            assert got[k] == v, (key, k)
        assert got["with_own"] == got["scored"]
    # recomputed from the reads
    sel = {"A_I0": lambda n: n.startswith("a0-sealed-A-"), "Av_I0": lambda n: n.startswith("a0-sealed-Av-"), "T0_I0": lambda n: n.startswith("a0-sealed-T0-"),
           "first_I0_paraphrase": lambda n: n.startswith("a0-sealed-first-"), "E1_row48": lambda n: "-E1-" in n, "E2_identity_row49": lambda n: "-E2-identity-" in n}
    for key, f in sel.items():
        tot = Counter()
        for name in (n for n in files if f(n)):
            reads = planted_reads(files[name])
            pres = own_present(reads)
            tot["scored"] += len(reads); tot["own_in_window"] += len(pres)
            tot["bag_top_is_own"] += sum(1 for x in pres if x["bag"]["top_is_own"])
            tot["decoder_cites_own"] += sum(1 for x in pres if x["decoder_cites_own"])
            tot["decoder_cites_another_line_own_present"] += sum(1 for x in pres if cites(x) and not x["decoder_cites_own"])
            tot["mark_oracle_cites_own"] += sum(1 for x in pres if x["mark_oracle_cites_own"])
        for k, v in POOLED[key].items():
            if k != "ties_with_own_present":
                assert tot[k] == v, (key, k, tot[k])


def test_the_binned_reading_by_the_nearest_other_lines_bag_score(files):
    sel = {"A": lambda n: n.startswith("a0-sealed-A-"), "Av": lambda n: n.startswith("a0-sealed-Av-"), "T0": lambda n: n.startswith("a0-sealed-T0-"),
           "first": lambda n: n.startswith("a0-sealed-first-"), "E1": lambda n: "-E1-" in n, "E2": lambda n: "-E2-identity-" in n}
    for fam, f in sel.items():
        got = {b: [0, 0, 0] for b in BINS[fam]}
        alone = [0, 0, 0]
        for name in (n for n in files if f(n)):
            for x in own_present(planted_reads(files[name])):
                if x["bin"] is None:                                 # the own line alone in its window: no other line to bin by
                    assert len(x["lines"]) == 1 and x["nearest_other_bag"] is None
                    g = alone
                else:
                    g = got[x["bin"]]
                g[0] += 1; g[1] += bool(x["decoder_cites_own"]); g[2] += bool(x["mark_oracle_cites_own"])
        assert {b: tuple(v) for b, v in got.items()} == BINS[fam], fam
        assert tuple(alone) == ALONE[fam], fam
        assert sum(v[0] for v in BINS[fam].values()) + ALONE[fam][0] == POOLED[{"A": "A_I0", "Av": "Av_I0", "T0": "T0_I0", "first": "first_I0_paraphrase", "E1": "E1_row48", "E2": "E2_identity_row49"}[fam]]["own_in_window"]
    # the decoder's own citation share does not move between the two bins under 0.8 on the template families, and halves at 0.8 and above (E2)
    for fam in ("Av", "T0", "first"):
        lo, mid = BINS[fam]["under_0.5"], BINS[fam]["0.5_to_under_0.8"]
        assert abs(lo[1] / lo[0] - mid[1] / mid[0]) < 0.03
    assert round(BINS["E2"]["0.8_and_above"][1] / BINS["E2"]["0.8_and_above"][0], 3) == 0.467                 # 280 of 599: the reads with the neighbour in the window


def test_the_windows_hold_at_most_four_lines_and_a0_scores_every_line_of_them(files):
    for d in files.values():
        for blk, reads in d["reads"].items():
            for x in reads:
                if x.get("skipped"):
                    continue
                assert 1 <= len(x["lines"]) <= 4 and all("bag" in l and "aligned" in l for l in x["lines"])
            assert d["blocks"][blk]["n_not_scored_no_text"] == 0        # every window line has its text in the file


def test_the_provenance_states_the_row_the_command_and_the_recomputation(prov):
    assert prov["recomputation"]["equal_files"] == 27
    assert prov["command"].startswith("python -m cortex_c2b.a0 metrics\\mqar\\hm-lm-*-near-dup-E1-") and "sealed-*-I0-*.pass.json --out metrics\\mqar" in prov["command"]
    assert "no timestamp" in prov["when"]["files"] and "dfe7145" in prov["when"]["commit"]
    kinds = Counter(f["family"] for f in prov["files"].values())
    assert kinds == {"A": 6, "Av": 6, "T0": 6, "first": 3, "E1": 3, "E2-identity": 3}
