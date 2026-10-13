"""ADR-008, amendment of 2026-10-13, the measurement (0), (i) and (ii), written before the code: session B
refuses a configuration that does not hash to the checkpoint's; it opens a copy of the journal and hashes the
log and the payload store before and after; it records the geometry from the live object, the device, the
versions, the commit, and per question the live cost (`touched`), the index's query signatures and the cue;
`--index 16,exact` wins over the configuration and the file records the configuration's pair, the journal's
record, the live geometry and `index_overridden`; the control replay compares the per question outcomes and the
counts with a legacy file exactly and stops, retaining a failure file, when one differs; the cold reading
H, S, M and the paired counts recompute from the files alone, the exact binomial on the discordant pairs
decides M, and the two further draws are read by the collision law's expectation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import train
from cortex_c2b import Journal, POLICY_STOP
from cortex_c2b import hm_lm
from cortex_c2b import adoption_reading as ar
from cortex_c2b import record_tools
from cortex_c2b.hm_protocol import generate_facts, DELTA, NEGCTRL_INVALID
from cortex_c2b.hm_lm import INVALID_CITATION_MAX
from cortex_c2b.lm_bridge import JournalBridge
from cortex_c2b.read_path import LEGACY_GEOMETRY, ADOPTED_GEOMETRY, same_geometry

ROOT = Path(__file__).resolve().parent.parent
N, NN = 10, 5                                                                     # the generator balances five schemas: multiples of five


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    """A tiny checkpoint carrying its run id and config hash, its config file, and a sealed journal of the
    protocol's first facts written through the gate under the legacy geometry, unstamped as the record's are
    (written with no path attached)."""
    d = tmp_path_factory.mktemp("adoption")
    cfg_dict = {"n_layer": 2, "n_head": 2, "n_embd": 32, "block_size": 96, "journal": "read", "journal_block": 1,
                "journal_heads": 2, "journal_read_bytes": 96, "journal_cue_dim": 64, "journal_k": 4, "journal_familiarity_mark": True}
    (d / "tiny.json").write_text(json.dumps(cfg_dict), encoding="utf-8")
    cfg = train.load_config(str(d / "tiny.json"))
    torch.manual_seed(0)
    m = train.VanillaGPT(cfg); m.eval()
    key = bytes(range(32))
    j = Journal(d / "journal.jsonl", key=key, policy=POLICY_STOP)
    b = JournalBridge(m, j, k=4, budget_bytes=96, seed=0, shuffle_seed=1, mark=True)
    facts, _ = generate_facts(N, 0)
    for i, f in enumerate(facts):
        b.write(f.statement, f.schema, now=100.0 + i, entity=f.entity)
    torch.save({"model": m.state_dict(), "run_id": "adopt0000000", "config_hash": train.config_hash(cfg), "step": 0}, d / "ckpt.pt")
    return d, key, len(b.pointer_of)


def _run(d, policy="plain", **kw):
    cfg = train.load_config(str(d / "tiny.json"))
    return hm_lm.session_b_from_disk(d / "ckpt.pt", d / "journal.jsonl", cfg, n_facts=N, n_negctrl=NN, seed=0, k=4, budget=96,
                                     window_len=96, policy=policy, **kw)


def test_session_b_refuses_a_configuration_that_does_not_hash_to_the_checkpoint_s(fixture, monkeypatch):
    d, key, _ = fixture
    monkeypatch.setenv("QUANTUM_CORTEX_JOURNAL_KEY", key.hex())
    cfg = train.load_config(str(d / "tiny.json"))
    cfg.journal_k = 3                                                           # another configuration: another hash
    with pytest.raises(ValueError, match="config"):
        hm_lm.session_b_from_disk(d / "ckpt.pt", d / "journal.jsonl", cfg, n_facts=N, n_negctrl=NN, k=3, budget=96, window_len=96)
    torch.save({"model": torch.load(d / "ckpt.pt", weights_only=False)["model"], "run_id": "nohash"}, d / "nohash.pt")
    with pytest.raises(ValueError, match="config hash"):
        hm_lm.session_b_from_disk(d / "nohash.pt", d / "journal.jsonl", train.load_config(str(d / "tiny.json")), n_facts=N, n_negctrl=NN, k=4, budget=96, window_len=96)


def test_session_b_opens_a_copy_hashes_it_and_records_the_geometry_the_device_the_versions_and_the_cost(fixture, monkeypatch):
    d, key, n_planted = fixture
    monkeypatch.setenv("QUANTUM_CORTEX_JOURNAL_KEY", key.hex())
    monkeypatch.setattr(record_tools, "git_state", lambda: ("test0000", True))
    log_before = hashlib.sha256((d / "journal.jsonl").read_bytes()).hexdigest()
    r = _run(d, "plain")
    assert hashlib.sha256((d / "journal.jsonl").read_bytes()).hexdigest() == log_before   # nothing written to the journal of the record
    assert r["journal"]["opened"].startswith("a copy") and r["journal"]["unchanged"] is True
    assert r["journal"]["sha256_before"] == r["journal"]["sha256_after"] == log_before[:16]
    assert r["journal"]["payloads_before"] == r["journal"]["payloads_after"] and r["journal"]["payloads_before"]["n_payload_files"] == n_planted
    assert r["journal"]["planted_found"] == n_planted and r["journal"]["path"] == str(d / "journal.jsonl")
    idx = r["index"]
    assert same_geometry(idx["record"], LEGACY_GEOMETRY) and same_geometry(idx["live"], LEGACY_GEOMETRY)
    assert idx["live"]["planes_sha256"] == "680896ab07072738" and idx["overridden"] is False and idx["argument"] is None
    assert idx["config"] == {"n_tables": 4, "signature": "collapsed"}
    assert r["device"]["device"] == "cpu" and isinstance(r["device"]["threads"], int)
    assert r["versions"]["torch"] == torch.__version__ and r["versions"]["numpy"] == np.__version__
    assert r["code_commit"] == "test0000" and r["code_tree_clean"] is True
    assert r["thresholds"] == {"delta": DELTA, "negctrl_invalid": NEGCTRL_INVALID, "invalid_citation_max": INVALID_CITATION_MAX,
                               "floor": pytest.approx(r["thresholds"]["floor"])}
    assert r["timestamp_utc"].endswith("+00:00")
    for a in r["answers"]["on"] + r["answers"]["negctrl"]:
        assert isinstance(a["touched"], int) and a["touched"] >= 0
        assert len(a["signatures_query"]) == 4 and all(isinstance(s, int) for s in a["signatures_query"])
        assert len(a["cue"]) == 64 and all(isinstance(x, float) for x in a["cue"])
    for a in r["answers"]["on"]:
        assert a["own_in_candidates"] in (True, False) and (a["own_rank"] is None or a["own_rank"] >= 1)
        assert a["own_in_candidates"] or not a["own_in_window"]                # in the window implies among the candidates
        assert isinstance(a["window"], list) and all(set(w) == {"label", "pointer", "mark"} for w in a["window"])
        assert (a["cited_pointer"] is None) == (a["kind"] != "cite") or a["cited_pointer"] is None
        if a["own_in_window"]:
            assert a["own_rank"] is not None and a["own_rank"] <= 4
    assert r["index"]["record_source"] == "stamp"                               # the fixture's journal was written through a bridge: stamped legacy
                                                                                # (the record's journals, written before the stamp, read "legacy_by_default")
    for a in r["answers"]["off"]:
        assert "touched" not in a and "cue" not in a                            # no read on the off arm
    assert r["cost"]["on"]["n"] == len(r["answers"]["on"]) and r["cost"]["on"]["sum"] == sum(a["touched"] for a in r["answers"]["on"])
    assert r["cost"]["negctrl"]["sum"] == sum(a["touched"] for a in r["answers"]["negctrl"])
    json.dumps(r)                                                               # serialisable as the file writes it


def test_the_signatures_and_cues_recorded_make_a_live_miss_attributable(fixture, monkeypatch):
    """The recorded cue, re signed under the live geometry, gives the recorded signatures; the candidate set of
    the question recomputes from them against the journal's entries, as row 45's files do."""
    d, key, _ = fixture
    monkeypatch.setenv("QUANTUM_CORTEX_JOURNAL_KEY", key.hex())
    r = _run(d, "plain")
    from cortex_c2b.read_path import CueIndex
    idx = CueIndex(dim=64, seed=0)
    for a in r["answers"]["on"]:
        cue = np.asarray(a["cue"], dtype=np.float32)
        assert [idx._sig(cue, t) for t in range(4)] == a["signatures_query"]


def test_index_16_exact_wins_over_the_configuration_and_the_file_records_both(fixture, monkeypatch):
    d, key, _ = fixture
    monkeypatch.setenv("QUANTUM_CORTEX_JOURNAL_KEY", key.hex())
    r = _run(d, "plain", index_override=hm_lm.parse_index_argument("16,exact"))
    idx = r["index"]
    assert same_geometry(idx["record"], LEGACY_GEOMETRY) and same_geometry(idx["live"], ADOPTED_GEOMETRY)
    assert idx["live"]["planes_sha256"] == "e5703fddba43332f" and idx["overridden"] is True
    assert idx["config"] == {"n_tables": 4, "signature": "collapsed"} and idx["argument"] == {"n_tables": 16, "n_bits": 10, "signature": "exact"}
    for a in r["answers"]["on"]:
        assert len(a["signatures_query"]) == 16
    v_legacy = _run(d, "mark-veto+value")
    v_adopted = _run(d, "mark-veto+value", index_override=hm_lm.parse_index_argument("16,exact"))
    assert v_adopted["hm_mark_oracle"]["threshold"] == v_legacy["hm_mark_oracle"]["threshold"]        # the same threshold, as declared
    assert same_geometry(v_adopted["hm_mark_oracle"]["threshold_index_geometry"], LEGACY_GEOMETRY)
    assert same_geometry(v_adopted["hm_mark_oracle"]["live_index_geometry"], ADOPTED_GEOMETRY) and same_geometry(v_legacy["hm_mark_oracle"]["live_index_geometry"], LEGACY_GEOMETRY)
    tp = _run(d, "plain", index_override=hm_lm.parse_index_argument("16,exact"), train_pool_seed=100_000, train_pool_n=10)["training_pool_probe"]
    assert same_geometry(tp["index_geometry"], ADOPTED_GEOMETRY)               # the memorisation probe follows the live geometry
    assert hm_lm.parse_index_argument("4,collapsed") == {"n_tables": 4, "n_bits": 10, "signature": "collapsed"}
    for bad in ("16", "sixteen,exact", "16,packed", "0,exact", "16,exact,10"):
        with pytest.raises(ValueError):
            hm_lm.parse_index_argument(bad)
    hashlib.sha256((d / "journal.jsonl").read_bytes())                        # (nothing written: the next test checks the hash again)


def test_the_same_checkpoint_replayed_twice_agrees_and_the_control_comparison_passes(fixture, monkeypatch, tmp_path):
    d, key, _ = fixture
    monkeypatch.setenv("QUANTUM_CORTEX_JOURNAL_KEY", key.hex())
    legacy = _run(d, "mark-veto+value")
    (tmp_path / "legacy.json").write_text(json.dumps(legacy), encoding="utf-8")
    r = _run(d, "mark-veto+value", control_against=tmp_path / "legacy.json")
    c = r["control"]
    assert c["against"] == str(tmp_path / "legacy.json") and c["ok"] is True and c["differences"] == []
    assert c["against_sha256"] == hashlib.sha256((tmp_path / "legacy.json").read_bytes()).hexdigest()[:16]
    assert c["legacy_policy"] == "mark-veto+value" and c["compared_as"] == "the legacy file's own outcomes"
    assert c["threshold"]["legacy"] == c["threshold"]["live"] and c["threshold"]["difference"] == 0.0 and c["threshold"]["tolerance"] == hm_lm.CONTROL_THRESHOLD_TOLERANCE
    assert c["threshold"]["within_tolerance"] is True and c["threshold"]["stops"] is False and c["threshold"]["stop_bound"] == hm_lm.CONTROL_THRESHOLD_STOP
    assert c["threshold"]["min_margin_live"] is not None and c["threshold"]["min_margin_live"] >= 0
    assert c["outcomes_equal"] == {"on": True, "off": True, "negctrl": True} and "stopped" not in r
    assert c["not_comparable_by_replay"] == {"on": [], "negctrl": []} and c["outcomes_differing_by_threshold_only"] == {}
    for k in hm_lm.CONTROL_COUNT_KEYS:
        assert c["counts"][k] == {"legacy": legacy[k], "live": r[k], "equal": True}
    # the oracle is computed once: the threshold the veto used is the one the file records
    assert r["hm_mark_oracle"]["threshold"] == c["threshold"]["live"]
    with pytest.raises(ValueError, match="not a control"):                     # systems reader, finding 3
        _run(d, "mark-veto+value", control_against=tmp_path / "legacy.json", index_override=hm_lm.parse_index_argument("16,exact"))
    no_answers = json.loads(json.dumps(legacy)); no_answers.pop("answers")
    (tmp_path / "noans.json").write_text(json.dumps(no_answers), encoding="utf-8")
    with pytest.raises(ValueError, match="answers"):                           # measurement reader, finding 4b
        _run(d, "mark-veto+value", control_against=tmp_path / "noans.json")


def test_a_plain_legacy_file_is_compared_through_the_post_hoc_replay_when_the_policy_is_the_veto_s(fixture, monkeypatch, tmp_path):
    """Seed 1337's case: the legacy file is the plain answers file; the control under `mark-veto+value` is read
    against the post hoc replay of those answers, by the one function live and replay share."""
    d, key, _ = fixture
    monkeypatch.setenv("QUANTUM_CORTEX_JOURNAL_KEY", key.hex())
    plain = _run(d, "plain")
    (tmp_path / "plain.json").write_text(json.dumps(plain), encoding="utf-8")
    r = _run(d, "mark-veto+value", control_against=tmp_path / "plain.json")
    c = r["control"]
    assert c["legacy_policy"] == "plain" and c["compared_as"].startswith("the post hoc replay")
    assert c["ok"] is True and c["differences"] == []
    replayed = hm_lm.replay_outcomes(plain, "mark-veto+value")
    assert [a["outcome"] for a in r["answers"]["on"]] == replayed["on"]
    assert [a["outcome"] for a in r["answers"]["negctrl"]] == replayed["negctrl"]
    assert [a["outcome"] for a in r["answers"]["off"]] == replayed["off"]
    rates = hm_lm.replay_policy(plain, "mark-veto+value")
    assert rates["hm_recall_on"] == r["hm_recall_on"] and rates["hm_negctrl_rate"] == r["hm_negctrl_rate"]
    assert c["counts"]["hm_retrieval_hit"]["equal"] is True and c["counts"]["hm_retrieval_hit"]["legacy_is"] == "the legacy file's"
    assert hm_lm.not_comparable_by_replay(plain, plain["hm_mark_oracle"]["threshold"]) == {"on": [], "negctrl": []}
    # a question the replay cannot classify (a non own citation at or above the threshold) is named and left out of the differences
    doctored = json.loads(json.dumps(plain)); a0 = doctored["answers"]["on"][0]
    a0.update({"kind": "cite", "outcome": "invalid", "label": "a", "mark_cited": 0.99, "cited_pointer_is_own": False, "answered": "x"})
    assert hm_lm.not_comparable_by_replay(doctored, doctored["hm_mark_oracle"]["threshold"])["on"] == [0]
    (tmp_path / "doc2.json").write_text(json.dumps(doctored), encoding="utf-8")
    r2 = _run(d, "mark-veto+value", control_against=tmp_path / "doc2.json")
    assert r2["control"]["not_comparable_by_replay"]["on"] == [0] and all("question 0 " not in x for x in r2["control"]["differences"])


def test_a_control_that_differs_stops_and_the_differences_are_named(fixture, monkeypatch, tmp_path):
    d, key, _ = fixture
    monkeypatch.setenv("QUANTUM_CORTEX_JOURNAL_KEY", key.hex())
    legacy = _run(d, "plain")
    doctored = json.loads(json.dumps(legacy))
    a0 = doctored["answers"]["on"][0]
    a0["outcome"] = "abstain" if a0["outcome"] != "abstain" else "invalid"
    doctored["hm_mark_oracle"]["threshold"] += 5e-6                            # above the declared tolerance
    (tmp_path / "doctored.json").write_text(json.dumps(doctored), encoding="utf-8")
    r = _run(d, "plain", control_against=tmp_path / "doctored.json")
    c = r["control"]
    assert c["ok"] is False and r["stopped"]
    assert any("on" in x and "question 0" in x for x in c["differences"])
    assert c["threshold"]["within_tolerance"] is False
    assert c["threshold"]["stops"] is (hm_lm.CONTROL_THRESHOLD_STOP is not None and 5e-6 >= hm_lm.CONTROL_THRESHOLD_STOP)
    assert any("threshold" in x for x in c["differences"]) is c["threshold"]["stops"]
    assert c["outcomes_equal"]["on"] is False and c["outcomes_equal"]["off"] is True
    # a legacy file of another checkpoint or another seed is refused outright
    other = json.loads(json.dumps(legacy)); other["checkpoint"]["run_id"] = "someoneelse"
    (tmp_path / "other.json").write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(ValueError, match="run id"):
        _run(d, "plain", control_against=tmp_path / "other.json")


def test_the_command_line_refuses_an_overwrite_and_a_dirty_tree_and_writes_the_failure_file(fixture, monkeypatch, tmp_path, capsys):
    d, key, _ = fixture
    monkeypatch.setenv("QUANTUM_CORTEX_JOURNAL_KEY", key.hex())
    monkeypatch.setattr(record_tools, "git_state", lambda: ("test0000", True))
    base = ["--session-b", "--ckpt", str(d / "ckpt.pt"), "--journal", str(d / "journal.jsonl"), "--config", str(d / "tiny.json"),
            "--n-facts", str(N), "--n-negctrl", str(NN)]
    out = tmp_path / "ctrl.json"
    hm_lm._cli(base + ["--out", str(out)])
    assert out.exists()
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["index"]["overridden"] is False and r["code_commit"] == "test0000"
    with pytest.raises(SystemExit, match="never overwritten"):
        hm_lm._cli(base + ["--out", str(out)])
    out2 = tmp_path / "adopted.json"
    hm_lm._cli(base + ["--index", "16,exact", "--out", str(out2)])
    r2 = json.loads(out2.read_text(encoding="utf-8"))
    assert r2["index"]["overridden"] is True and same_geometry(r2["index"]["live"], ADOPTED_GEOMETRY)
    # a control that fails writes the failure file beside the name asked and exits non zero
    doctored = json.loads(json.dumps(r)); doctored["answers"]["negctrl"][0]["outcome"] = "strict"
    (tmp_path / "doc.json").write_text(json.dumps(doctored), encoding="utf-8")
    out3 = tmp_path / "ctrl2.json"
    with pytest.raises(SystemExit, match="stopped"):
        hm_lm._cli(base + ["--out", str(out3), "--control-against", str(tmp_path / "doc.json")])
    assert not out3.exists() and (tmp_path / "ctrl2-FAILED-control.json").exists()
    failed = json.loads((tmp_path / "ctrl2-FAILED-control.json").read_text(encoding="utf-8"))
    assert failed["stopped"] and failed["control"]["ok"] is False
    monkeypatch.setattr(record_tools, "git_state", lambda: ("dirty000", False))
    with pytest.raises(SystemExit, match="not clean"):
        hm_lm._cli(base + ["--out", str(tmp_path / "x.json")])
    monkeypatch.setattr(record_tools, "git_state", lambda: (None, None))      # systems reader, finding 4: no commit, no file
    with pytest.raises(SystemExit, match="commit"):
        hm_lm._cli(base + ["--out", str(tmp_path / "y.json")])
    monkeypatch.setattr(record_tools, "git_state", lambda: ("test0000", True))
    with pytest.raises(SystemExit, match="--index"):                           # systems reader, finding 15: a bad argument is a refusal, not a traceback
        hm_lm._cli(base + ["--index", "16", "--out", str(tmp_path / "z.json")])
    # the default name is checked before the minutes of replay (systems reader, finding 16)
    outdir = tmp_path / "dir"; outdir.mkdir()
    hm_lm._cli(base + ["--out", str(outdir)])
    assert (outdir / "hm-lm-adopt0000000-session-b.json").exists()
    with pytest.raises(SystemExit, match="never overwritten"):
        hm_lm._cli(base + ["--out", str(outdir)])
    assert (out.read_bytes().count(b"\r\n") == 0) and out.read_bytes().endswith(b"\n")   # LF, as the record's files


def test_the_default_file_names_say_control_or_the_geometry_and_the_policy(fixture, monkeypatch, tmp_path):
    assert hm_lm.session_b_file_name("abc", "plain", None, control=True) == "hm-lm-abc-session-b-control.json"
    assert hm_lm.session_b_file_name("abc", "mark-veto+value", None, control=True) == "hm-lm-abc-session-b-control-mark-veto-value.json"
    assert hm_lm.session_b_file_name("abc", "plain", {"n_tables": 16, "signature": "exact"}, control=False) == "hm-lm-abc-session-b-index16exact.json"
    assert hm_lm.session_b_file_name("abc", "mark-veto+value", {"n_tables": 16, "signature": "exact"}, control=False) == "hm-lm-abc-session-b-index16exact-mark-veto-value.json"
    assert hm_lm.session_b_file_name("abc", "plain", None, control=False) == "hm-lm-abc-session-b.json"      # the name of every file to date


# ---- the cold reading from the files alone ----------------------------------------------------------
def _answers(outcomes, own_in_window, labels=None, marks=None, entities=None):
    out = []
    for i, (o, w) in enumerate(zip(outcomes, own_in_window)):
        a = {"entity": entities[i] if entities else f"E{i}", "outcome": o, "own_in_window": w, "kind": "unknown" if o == "abstain" else "cite",
             "label": (labels[i] if labels else ("a" if o != "abstain" else None)), "cited_pointer_is_own": (o in ("strict", "valid")) if o != "abstain" else None,
             "mark_max": (marks[i] if marks else 0.9), "touched": 5}
        out.append(a)
    return out


def _file(run_id, policy, on_outcomes, on_window, neg_outcomes, neg_marks, thr=0.6, recall_off=0.0, **extra):
    n, nn = len(on_outcomes), len(neg_outcomes)
    strict = on_outcomes.count("strict"); invalid = on_outcomes.count("invalid"); abst = on_outcomes.count("abstain")
    valid = strict + on_outcomes.count("valid")
    claims = sum(1 for o in neg_outcomes if o != "abstain")
    gap = strict / n - recall_off
    four = int(gap >= DELTA and recall_off <= 0.30 and invalid / n <= INVALID_CITATION_MAX and claims / nn < NEGCTRL_INVALID)
    f = {"policy": policy, "checkpoint": {"run_id": run_id}, "hm_recall_on": strict / n, "hm_recall_off": recall_off, "hm_gap": gap,
         "hm_invalid_citation_on": invalid / n, "hm_valid_citation_on": valid / n, "hm_false_abstention_on": abst / n,
         "hm_negctrl_rate": claims / nn, "hm_retrieval_hit": sum(on_window) / n, "gate_conditions_without_skill_arm": four,
         "run_valid": claims / nn < NEGCTRL_INVALID, "hm_mark_oracle": {"threshold": thr},
         "answers": {"on": _answers(on_outcomes, on_window), "off": _answers(["abstain"] * n, [None] * n),
                     "negctrl": _answers(neg_outcomes, [None] * nn, marks=neg_marks)},
         "index": {"live": dict(LEGACY_GEOMETRY if policy.endswith("legacy") else ADOPTED_GEOMETRY)}}
    f.update(extra)
    return f


def test_the_exact_two_sided_binomial_at_one_half_reads_m_as_the_amendment_wrote_it():
    assert ar.binomial_two_sided_p(6, 0) == pytest.approx(0.03125) and ar.binomial_two_sided_p(0, 6) == pytest.approx(0.03125)
    assert ar.binomial_two_sided_p(5, 0) == pytest.approx(0.0625) and ar.binomial_two_sided_p(0, 0) == 1.0
    assert ar.binomial_two_sided_p(3, 3) == 1.0 and ar.binomial_two_sided_p(7, 1) == pytest.approx(2 * (1 + 8) / 256)
    assert ar.read_m(6, 0) == "M+" and ar.read_m(0, 6) == "M-" and ar.read_m(5, 0) == "M0" and ar.read_m(4, 1) == "M0"
    assert ar.read_m(10, 1) == "M+" and ar.read_m(1, 10) == "M-" and ar.read_m(0, 0) == "M0"


def test_the_paired_counts_recompute_from_two_files_and_split_the_discordant_pairs():
    on_c = ["strict", "abstain", "invalid", "strict", "abstain", "valid", "strict", "abstain"]
    w_c = [True, False, True, True, False, True, True, False]
    on_a = ["strict", "strict", "invalid", "abstain", "abstain", "strict", "strict", "strict"]
    w_a = [True, True, True, True, True, True, True, True]
    neg_c = ["abstain", "abstain", "invalid", "abstain"]; neg_a = ["abstain", "abstain", "invalid", "invalid"]
    control = _file("r", "mark-veto+value", on_c, w_c, neg_c, [0.3, 0.5, 0.7, 0.4], thr=0.6)
    adopted = _file("r", "mark-veto+value", on_a, w_a, neg_a, [0.3, 0.65, 0.7, 0.61], thr=0.6)
    pc = ar.paired_counts(control, adopted)
    assert pc["n"] == 8 and pc["own_in_window"] == {"control": 5, "adopted": 8, "gained": 3, "lost": 0}
    assert pc["strict"] == {"control": 3, "adopted": 5, "b_adopted_only": 3, "c_control_only": 1}
    assert pc["discordant_by_window"] == {"b_newly_present": 2, "b_already_present": 1, "b_lost": 0, "b_absent_both": 0,
                                          "c_newly_present": 0, "c_already_present": 1, "c_lost": 0, "c_absent_both": 0}
    lost = _file("r", "mark-veto+value", ["abstain"] + on_a[1:], [False] + w_a[1:], neg_a, [0.3, 0.65, 0.7, 0.61], thr=0.6)
    assert ar.paired_counts(control, lost)["discordant_by_window"]["c_lost"] == 1 and ar.paired_counts(control, lost)["own_in_window"]["lost"] == 1
    assert pc["outcomes"]["strict"]["abstain"] == 1 and pc["outcomes"]["abstain"]["strict"] == 2 and pc["outcomes"]["valid"]["strict"] == 1
    assert sum(sum(row.values()) for row in pc["outcomes"].values()) == 8
    assert pc["never_planted"]["outcomes"]["abstain"]["invalid"] == 1 and pc["never_planted"]["claims"] == {"control": 1, "adopted": 2}
    assert pc["never_planted"]["windows_at_or_above_threshold"] == {"control": 1, "adopted": 3}
    assert pc["citations_changed"] >= 0
    with pytest.raises(ValueError, match="pair"):
        ar.paired_counts(control, _file("other", "mark-veto+value", on_a, w_a, neg_a, [0.3, 0.65, 0.7, 0.61]))


def test_the_vetoes_are_the_citations_plain_makes_that_the_veto_turns_into_abstentions():
    plain = _file("r", "plain", ["strict", "invalid", "invalid", "strict"], [True] * 4, ["invalid", "abstain"], [0.7, 0.2])
    veto = _file("r", "mark-veto+value", ["strict", "abstain", "invalid", "strict"], [True] * 4, ["abstain", "abstain"], [0.7, 0.2])
    assert ar.vetoes(plain, veto) == {"on": 1, "negctrl": 1}


def _reading_files(h_full=True, s_hold=True, b=0, c=0):
    files = {}
    for run_id, n in (("8ceba5db8d0b", 8), ("4cfd6f6e1406", 8), ("2e1dc71913b1", 8)):
        counted = run_id != "8ceba5db8d0b"
        on_c = ["strict"] * 6 + ["abstain"] * 2
        w_c = [True] * 6 + [False] * 2
        on_a = list(on_c); w_a = [True] * 8 if h_full else [True] * 7 + [False]
        if counted:
            for i in range(b):                                                  # adopted strict where the control abstained
                on_a[6 + i % 2] = "strict"
            for i in range(c):                                                  # control strict where the adopted abstains
                on_a[i] = "abstain"
        neg = ["abstain"] * 4
        neg_a = list(neg)
        if not s_hold and counted:
            on_a = ["invalid"] * 3 + on_a[3:]                                   # three invalid citations of eight: above the maximum
        for pol in ("plain", "mark-veto+value"):
            files[(run_id, "legacy", pol)] = _file(run_id, pol + "legacy", on_c, w_c, neg, [0.2] * 4)
            files[(run_id, "legacy", pol)]["policy"] = pol
            files[(run_id, "adopted", pol)] = _file(run_id, pol, on_a, w_a, neg_a, [0.2] * 4)
    return files


def test_the_reading_h_s_m_decides_from_the_files_and_names_its_cell():
    r = ar.read_adoption(_reading_files())
    assert r["H"]["letter"] == "H1" and r["H"]["in_window"] == 24 and r["H"]["questions"] == 24
    assert r["S"]["letter"] == "S1" and set(r["S"]["per_seed"]) == {"4cfd6f6e1406", "2e1dc71913b1"}
    assert r["M"]["letter"] == "M0" and r["M"]["b"] == 0 and r["M"]["c"] == 0 and r["M"]["p"] == 1.0 and r["M"]["policy"] == "mark-veto+value"
    assert r["cell"] == "H1 S1 M0" and "the adoption stands" in r["consequence"]
    assert r["counted_seeds"] == ["4cfd6f6e1406", "2e1dc71913b1"] and r["reported_not_counted"] == ["8ceba5db8d0b"]
    assert set(r["paired"]) == {"8ceba5db8d0b", "4cfd6f6e1406", "2e1dc71913b1"} and set(r["paired"]["4cfd6f6e1406"]) == {"plain", "mark-veto+value"}
    assert r["vetoes"]["4cfd6f6e1406"] == {"legacy": {"on": 0, "negctrl": 0}, "adopted": {"on": 0, "negctrl": 0}, "difference": {"on": 0, "negctrl": 0}}
    r = ar.read_adoption(_reading_files(h_full=False))
    assert r["H"]["letter"] == "H0" and r["cell"].startswith("H0") and "signatures" in r["consequence"]
    r = ar.read_adoption(_reading_files(s_hold=False))
    assert r["S"]["letter"] == "S0" and "invalid citations" in r["S"]["broken"]["4cfd6f6e1406"][0] and "threshold" in r["consequence"]
    r = ar.read_adoption(_reading_files(b=4, c=0))                             # 2 live seeds x up to 2 gains... b counts per seed
    assert r["M"]["b"] == 4 and r["M"]["c"] == 0 and r["M"]["letter"] == "M0"  # four discordant: nothing decidable (p 0.125)
    r = ar.read_adoption(_reading_files(b=0, c=3))                             # three control only strict per seed: six discordant against
    assert r["M"]["b"] == 0 and r["M"]["c"] == 6 and r["M"]["letter"] == "M-" and "M-" in r["cell"] and "window" in r["consequence"]
    files = _reading_files()
    files[("4cfd6f6e1406", "adopted", "plain")]["answers"]["on"][0]["own_in_window"] = False
    with pytest.raises(ValueError, match="different windows"):                 # the reads are the same under both policies by construction
        ar.read_adoption(files)


def test_the_two_further_draws_are_read_by_the_law_s_expectation(tmp_path):
    """On the record's own schema (measurement reader, finding 1: the cosine lives in `questions`, not in the rows): the
    three files of row 45, re labelled as seeds 1 and 2, with a miss doctored in as the probe would record it."""
    import copy
    record = {rid: json.loads((ROOT / "metrics/mqar" / name).read_text(encoding="utf-8")) for rid, name in
              (("8ceba5db8d0b", "bucket-probe-8ceba5db8d0b.json"), ("4cfd6f6e1406", "bucket-probe-4cfd6f6e1406-run2.json"), ("2e1dc71913b1", "bucket-probe-2e1dc71913b1.json"))}

    def probe_file(run_id, seed, recall, admissible, misses):
        f = copy.deepcopy(record[run_id]); f["index_seed"] = seed
        f["aggregates"]["C2b"].update({"candidate_recall": recall, "admissible": admissible, "misses": misses})
        for i in misses:
            row = next(x for x in f["rows"]["C2b"] if x["i"] == i); row["in_candidates"] = False; row["rank"] = None
            next(q for q in f["questions"] if q["i"] == i)["cos_query_own"] = 0.90
        return f
    clean = [probe_file(r, s, 1.0, True, []) for r in ("8ceba5db8d0b", "4cfd6f6e1406", "2e1dc71913b1") for s in (1, 2)]
    d = ar.read_draws(clean)
    assert d["clean_draws"] == 6 and d["misses"] == 0 and d["reading"].startswith("two clean draws")
    assert d["expected_misses_per_draw"] == pytest.approx(0.0325, abs=1e-4) and d["chance_both_clean"] == pytest.approx(0.937, abs=0.002)
    assert d["per_draw_miss_chance_bound_95"] == pytest.approx(0.78, abs=0.01)
    one_miss = clean[:5] + [probe_file("2e1dc71913b1", 2, 199 / 200, True, [17])]
    d = ar.read_draws(one_miss)
    assert d["misses"] == 1 and d["reading"].startswith("one miss") and d["miss_detail"][0]["law_miss_chance_at_cosine"] == pytest.approx(ar.law_miss_chance(0.90))
    assert d["miss_detail"][0] == {"run_id": "2e1dc71913b1", "index_seed": 2, "i": 17, "entity": record["2e1dc71913b1"]["questions"][17]["entity"],
                                   "cos_query_own": 0.90, "law_miss_chance_at_cosine": pytest.approx(ar.law_miss_chance(0.90)),
                                   "miss_class": record["2e1dc71913b1"]["rows"]["C2b"][17]["miss_class"]}
    bad = clean[:5] + [probe_file("2e1dc71913b1", 2, 1.0, False, [])]
    assert ar.read_draws(bad)["reading"].startswith("a draw breaks the budget")
    with pytest.raises(ValueError, match="seeds 1 and 2"):
        ar.read_draws(clean[:4])
    with pytest.raises(ValueError, match="not a declared draw"):               # the stopping rule: no third draw
        ar.read_draws(clean + [probe_file("2e1dc71913b1", 3, 1.0, True, [])])
    with pytest.raises(ValueError, match="one draw per seed"):
        ar.read_draws(clean + [probe_file("2e1dc71913b1", 2, 1.0, True, [])])


def test_the_law_s_expectation_written_before_the_run_recomputes_from_row_45_s_files():
    rec = ar.expected_misses_from_record()
    assert set(rec) == set(ar.EXPECTED_MISSES)
    for rid, e in rec.items():
        assert e == pytest.approx(ar.EXPECTED_MISSES[rid], abs=5e-5), rid
    assert sum(rec.values()) == pytest.approx(0.0325, abs=1e-4)
    assert ar.law_miss_chance(0.944) == pytest.approx(0.002, abs=5e-4) and ar.law_miss_chance(0.90) == pytest.approx(0.022, abs=2e-3)
    assert ar.law_miss_chance(0.85) == pytest.approx(0.085, abs=5e-3)        # the figures of the amendment's Why

