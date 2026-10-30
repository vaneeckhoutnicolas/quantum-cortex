"""A0, the construction's ceiling under byte keys (ADR-008 part A; `cortex_c2b/a0.py` delivered with ADR-010's build, part
E): the entity by the name pattern equals the template reading wherever a template parses on rows 48 and 49's files; the
two variants and their bins; a tie at the top is not decisive; the readings' letters on the aggregate share."""
import json
from pathlib import Path

import pytest

from cortex_c2b import record_tools, content_hash
from cortex_c2b import a0 as A

MQ = Path(record_tools.REPO_ROOT) / "metrics" / "mqar"
ROW48 = sorted(MQ.glob("hm-lm-*-near-dup-E1-session-b-index16exact-mark-veto-value.json"))
ROW49 = sorted(MQ.glob("hm-lm-*-near-dup-E2-identity-session-b-index16exact-mark-veto-value.json"))
ROW47 = sorted(MQ.glob("hm-lm-*-session-b-index16exact.json"))


def test_scores_and_entities():
    assert A.entity_in("what is Tone-4010's trade?") == "Tone-4010" and A.entity_in("a summary line") is None
    assert A.bag_score("Tone-4010", "Tone-4010") == 1.0 and A.bag_score("Tone-4010", None) == 0.0
    assert 0.0 < A.bag_score("Tone-4010", "Tone-4011") < 1.0 and A.aligned_score("Tone-4010", "Tone-4011") == 8
    assert A.aligned_score("Tone-4010", "Kilu-4010") == 5 and A.aligned_score("abc", "") == 0
    assert A._bin_of(0.2) == "under_0.5" and A._bin_of(0.5) == "0.5_to_under_0.8" and A._bin_of(0.8) == "0.8_and_above" and A._bin_of(1.0) == "0.8_and_above"
    assert A._bin_of(None) is None


def test_score_read_variants_bins_and_the_tie_rule():
    texts = {"p1": "Tone-4010 works as a cobbler", "p2": "Tone-4011 works as a miller", "p3": "Kilu-77ab works as a sawyer", "p4": "Tone-4010 works as a cobbler too"}
    answer = {"query": "what does Tone-4010 do?", "own_pointer": "p1", "window": [{"label": "A", "pointer": "p2", "mark": 0.9}, {"label": "B", "pointer": "p1", "mark": 0.95}, {"label": "C", "pointer": "p3", "mark": 0.5}],
              "raw_cited_pointer": "p1"}
    s = A.score_read(answer, texts, 0.8)
    assert s["query_entity"] == "Tone-4010" and s["bag"]["top_is_own"] and s["aligned"]["top_is_own"] and not s["bag"]["tie"]
    assert s["bag"]["own_rank"] == 1 and s["bag"]["own_score"] == 1.0 and s["nearest_other_bag"] == A.bag_score("Tone-4011", "Tone-4010")
    assert s["bin"] == "0.8_and_above" and s["decoder_cites_own"] is True and s["decoder_field"] == "raw_cited_pointer" and s["decoder_agrees_with_bag_top"]
    assert s["mark_oracle_pointer"] == "p1" and s["mark_oracle_cites_own"] is True
    tied = dict(answer); tied["window"] = [{"label": "A", "pointer": "p4", "mark": 0.9}, {"label": "B", "pointer": "p1", "mark": 0.95}]
    t = A.score_read(tied, texts, 0.8)
    assert t["bag"]["tie"] and t["bag"]["top_pointer"] is None and not t["bag"]["top_is_own"]       # a tie at the top is not decisive
    assert A.score_read({"query": "x", "window": []}, texts, 0.8) is None
    assert A.score_read({"query": "no name here", "window": [{"pointer": "p1"}]}, texts, 0.8) is None
    older = {"query": "what does Tone-4010 do?", "window": [{"label": "A", "pointer": "p1", "mark": 0.3}], "cited_pointer": "p1", "cited_pointer_is_own": True}
    o = A.score_read(older, texts, 0.8)
    assert o["own_pointer"] == "p1" and o["decoder_field"] == "cited_pointer" and o["mark_oracle_pointer"] is None


def test_letters_as_declared():
    assert A.letter_of(None, None) == "unread"
    assert A.letter_of(0.3, None).startswith("A0 (i)") and A.letter_of(0.5, None).startswith("A0 (i)")
    assert A.letter_of(0.95, 0.4).startswith("A0 (ii)") and A.letter_of(0.95, 0.95).startswith("A0 (iii)") and A.letter_of(0.9, None).startswith("A0 (iii)")
    assert "partial ceiling" in A.letter_of(0.7, 0.9)


@pytest.mark.skipif(not (ROW48 and ROW49 and ROW47), reason="the record's files of rows 47 to 49 are needed")
def test_template_entity_agreement_on_the_record_files_and_a_reading():
    for p in ROW48[:1] + ROW49[:1] + ROW47[:1]:
        f = json.loads(p.read_text(encoding="utf-8"))
        t = A.template_entity_agrees(f)
        assert t["parsed"] > 0 and t["disagree"] == 0, p.name                      # the name pattern equals the template reading wherever a template parses
        r = A.read_file(f, p.name)
        assert r["threshold"] is not None and r["n_texts"] >= 200
        blk = r["blocks"].get("on")
        assert blk and blk["n_scored"] > 0 and blk["n_with_own"] > 0
        assert blk["bag"]["share_top_is_own"] is not None and 0.0 <= blk["bag"]["share_top_is_own"] <= 1.0
        assert set(blk["by_nearest_other_bag"]) == {b for b, _, _ in A.BINS}
        assert blk["decoder"]["field"] in ("raw_cited_pointer", "cited_pointer")
        texts = A.texts_of_file(f)
        own = A.own_pointer_map(f)
        assert all(content_hash(t.encode("utf-8")) == ptr for ptr, t in list(texts.items())[:20])
        assert all(ptr in texts for ptr in list(own.values())[:20])


def test_a_sealed_pass_file_is_read_from_its_own_texts_and_refused_without_them(tmp_path):
    """The committee's finding on the build: a pass file without its texts was read against the first family's statements
    (every entity None, the letter A0 (i) by artefact); the runner now writes `texts` and `own_pointer_of`, and A0 refuses a
    pass file without them."""
    from cortex_c2b import sealed_families as S
    fam = S.generate_all()["A"]
    facts = fam["facts"][:6]
    texts = {content_hash(f["statement"].encode("utf-8")): f["statement"] for f in facts}
    own = {f["entity"]: content_hash(f["statement"].encode("utf-8")) for f in facts}
    ptr = list(texts)
    answers = []
    for i, f in enumerate(facts):
        win = [{"label": "A", "pointer": own[f["entity"]], "mark": 0.9}, {"label": "B", "pointer": ptr[(i + 1) % 6], "mark": 0.4}, {"label": "C", "pointer": ptr[(i + 2) % 6], "mark": 0.3}]
        answers.append({"block": "planted", "entity": f["entity"], "query": f["query"], "own_pointer": own[f["entity"]], "window": win,
                        "retrieval_order": [w["pointer"] for w in win], "raw_cited_pointer": own[f["entity"]], "raw_kind": "cite"})
    f = {"answers": answers, "texts": texts, "own_pointer_of": own, "oracle": {"threshold": 0.5}, "family": "A", "index": "I0", "form": "direct"}
    r = A.read_file(f, "sealed-A-x-I0-direct.pass.json")
    blk = r["blocks"]["planted"]
    assert r["n_texts"] == 6 and blk["n_scored"] == 6 and blk["n_with_own"] == 6 and blk["n_not_scored_no_text"] == 0
    assert blk["bag"]["share_top_is_own"] == 1.0 and blk["decoder"]["cites_own"] == 6 and blk["mark_oracle"]["cites_own"] == 6
    assert all(l["entity"] is not None for rd in r["reads"]["planted"] for l in rd["lines"])
    assert A.template_entity_agrees(f)["parsed"] == 0                             # A's statements parse under no first family template (T5)
    bare = {"answers": answers, "oracle": {"threshold": 0.5}}
    with pytest.raises(ValueError):
        A.read_file(bare, "x")
    partial = dict(f); partial["texts"] = {k: v for k, v in texts.items() if k != ptr[2]}
    r2 = A.read_file(partial, "y")
    assert r2["blocks"]["planted"]["n_not_scored_no_text"] >= 1 and r2["blocks"]["planted"]["n_scored"] < 6


def test_cli_writes_one_file_per_input_and_never_overwrites(tmp_path):
    if not ROW47:
        pytest.skip("the record's row 47 file is needed")
    A._cli([str(ROW47[0]), "--out", str(tmp_path)])
    out = tmp_path / f"a0-{ROW47[0].stem}.json"
    assert out.exists()
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["letter"] and r["source_sha256_lf"] and r["template_entity_agreement"]["disagree"] == 0 and "share_top_is_own_planted" in r
    with pytest.raises(SystemExit):
        A._cli([str(ROW47[0]), "--out", str(tmp_path)])
    # a pattern is expanded by the command itself (PowerShell passes it as written); one that matches nothing is refused
    src = tmp_path / "src"; src.mkdir()
    for i in (1, 2):
        (src / f"copy{i}.json").write_bytes(ROW47[0].read_bytes())
    A._cli([str(src / "copy*.json"), "--out", str(tmp_path / "out")])
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["a0-copy1.json", "a0-copy2.json"]
    with pytest.raises(SystemExit):
        A._cli([str(src / "none*.json"), "--out", str(tmp_path / "out")])
