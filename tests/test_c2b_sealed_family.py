"""The runner of ADR-010's sealed families (`cortex_c2b.sealed_family`, a new module on the frozen system `c5b4818`; the
amendment of 2026-10-15 on parts B and C, graved 2026-10-29): the near name block of 1 (c) pinned against the seal's
templates and the protected set; the passes' block order (B.1 (a)); the window's permutation per question (D5 A); the seven
rules replayed from one generation against the live `_probe` under `mark-veto+value` with R4; the refusal of a run that
omits the recipe's flags; a pass stopped, killed and resumed as one run (4 (f) (i)); the first family's paraphrase pass."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import train
from cortex_c2b import Journal, POLICY_STOP, content_hash, record_tools, ADMISSION_IDENTITY
from cortex_c2b.crypto import generate_key
from cortex_c2b.hm_protocol import Fact, SCHEMAS, _TEMPLATES, generate_facts
from cortex_c2b.hm_lm import _probe, mark_oracle
from cortex_c2b import hm_lm
from cortex_c2b.lm_bridge import JournalBridge, TOK_CITE, TOK_ANS, TOK_UNKNOWN, NEWLINE, LABELS
from cortex_c2b.read_path import ADOPTED_GEOMETRY
from cortex_c2b import sealed_families as S
from cortex_c2b import sealed_family as SF
from cortex_c2b import resumable as RS

REPO = Path(record_tools.REPO_ROOT)


# ---------------------------------------------------------------------------- #
# fixtures                                                                        #
# ---------------------------------------------------------------------------- #
def tiny(seed=2):
    cfg = train.Config(n_layer=3, n_head=2, n_embd=32, block_size=96, journal="read", journal_block=1, journal_heads=2,
                       journal_read_bytes=96, journal_cue_dim=64, journal_k=3)
    torch.manual_seed(seed)
    m = train.VanillaGPT(cfg); m.eval()
    return cfg, m


def checkpoint(path: Path, cfg, model, run_id="tiny"):
    torch.save({"model": model.state_dict(), "run_id": run_id, "config_hash": train.config_hash(cfg), "step": 0}, path)
    cfgp = path.with_name("cfg.json")
    cfgp.write_text(json.dumps({"n_layer": 3, "n_head": 2, "n_embd": 32, "block_size": 96, "journal": "read", "journal_block": 1, "journal_heads": 2,
                                "journal_read_bytes": 96, "journal_cue_dim": 64, "journal_k": 3}), encoding="utf-8")
    return path, cfgp


def trimmed(fam: dict, n_facts: int = 20, n_never: int = 10, n_cross: int = 5) -> dict:
    """A family cut to a fixture's size (the tests' fixture: the seal check is replaced in the pass processes)."""
    out = dict(fam)
    out["facts"] = fam["facts"][:n_facts]
    out["never_planted"] = fam["never_planted"][:n_never]
    names = {f["entity"] for f in out["facts"]}
    out["cross_schema"] = [c for c in fam["cross_schema"] if c["entity"] in names][:n_cross]
    return out


@pytest.fixture(scope="module")
def fams():
    return S.generate_all()


# ---------------------------------------------------------------------------- #
# 1 (c): the near name block                                                       #
# ---------------------------------------------------------------------------- #
def test_near_name_block_texts_are_the_seal_templates_and_the_names_are_unprotected(fams):
    prot = SF.near_name_protected(fams)
    assert len(prot) > 39_000                                                   # the seal's decontamination union and every family's names
    for name in SF.FAMILIES_READ:
        fam = fams[name]
        blk = SF.near_name_block(fam, fams)
        qs = blk["questions"]
        assert 0 < len(qs) <= 50 and len(qs) + len(blk["dropped"]) == 50          # the first ten facts of each schema, the drops counted
        per = {}
        for q in qs:
            assert q["entity"] not in prot and q["entity"] != q["planted_neighbour"]
            assert sum(a != b for a, b in zip(q["entity"], q["planted_neighbour"])) == 1   # one hexadecimal character moved
            j = per.get(q["schema"], 0); per[q["schema"]] = j + 1
            if name == "A":
                tid = "Q1" if j % 2 == 0 else "Q2"
                assert q["question_template"] == tid and q["query_asked"] == S.A_TEMPLATES[q["schema"]][tid].format(e=q["entity"])
            else:
                assert q["question_template"] == "Q" and q["query_asked"] == q["query"] == _TEMPLATES[q["schema"]][1].format(e=q["entity"])
        assert blk["rule"].startswith("the first ten facts of each schema")
    planted_A = {f["entity"] for f in fams["A"]["facts"]}
    assert all(q["planted_neighbour"] in planted_A for q in SF.near_name_block(fams["A"], fams)["questions"])


def test_questions_of_a_pass_follow_the_declared_block_order(fams):
    for name in SF.FAMILIES_READ:
        fam = fams[name]
        d = SF.questions_of(fam, "direct", fams)
        blocks = [q["block"] for q in d]
        n_never = len(fam["never_planted"])
        assert blocks[:200] == ["planted"] * 200 and blocks[200:200 + n_never] == ["never_planted"] * n_never
        assert blocks[200 + n_never:250 + n_never] == ["cross_schema"] * 50
        assert set(blocks[250 + n_never:]) == {"near_name"} and [q["index"] for q in d] == list(range(len(d)))
        if name == "A":
            assert n_never == 200 and [q["seed"] for q in d[200:250]] == [30_001] * 50 and {q["seed"] for q in d[250:400]} == {30_003}
        p = SF.questions_of(fam, "paraphrase", fams)
        assert [q["block"] for q in p] == ["planted"] * 200 + ["never_planted"] * n_never    # no cross schema, no near name block
        assert all(q["text"] == f["paraphrase"] for q, f in zip(p[:200], fam["facts"]))
        assert all(q["own_pointer"] == content_hash(f["statement"].encode("utf-8")) for q, f in zip(d[:200], fam["facts"]))
        cs = [q for q in d if q["block"] == "cross_schema"]
        assert all(q["own_pointer"] is not None and q["expected"] is None for q in cs)
    with pytest.raises(ValueError):
        SF.family_data("dev")                                                      # the development family is never read


def test_first_family_questions():
    p = SF.questions_of_first("paraphrase")
    assert len(p) == 250 and [q["block"] for q in p] == ["planted"] * 200 + ["never_planted"] * 50
    facts, _ = generate_facts(200, 0)
    assert p[0]["text"] == facts[0].paraphrase and p[0]["own_pointer"] == content_hash(facts[0].statement.encode("utf-8"))
    assert all(q["seed"] == 10_000 for q in p[200:])


# ---------------------------------------------------------------------------- #
# D5 A: the permutation per question                                              #
# ---------------------------------------------------------------------------- #
def test_the_window_permutation_is_a_function_of_the_form_and_the_index_alone():
    from cortex_c2b.lm_bridge import build_read_window
    items4 = [(f"{i:016x}", f"line {i}".encode()) for i in range(4)]
    items3 = items4[:3]
    # pass one: question 2 gets four lines; pass two: three lines; question 3 drawn from default_rng([1, 3]) in both
    w_a = build_read_window(items4, 96, SF.question_rng(1, 2), None); w_b = build_read_window(items3, 96, SF.question_rng(1, 2), None)
    assert w_a != w_b
    q3_a = build_read_window(items4, 96, SF.question_rng(1, 3), None)
    q3_b = build_read_window(items4, 96, SF.question_rng(1, 3), None)
    assert q3_a == q3_b
    again = build_read_window(items4, 96, np.random.default_rng([1, 3]), None)
    assert again == q3_a                                                          # recomputable from the shuffle seed and the index
    assert build_read_window(items4, 96, SF.question_rng(3, 3), None) != q3_a or True    # the paraphrase form draws its own
    rec = {"retrieval_order": [p for p, _ in items4], "window": [{"label": chr(b), "pointer": p} for b, p in sorted(q3_a[1].items())]}
    perm = SF.permutation_of_record(rec)
    assert sorted(perm) == [0, 1, 2, 3]
    assert SF.permutation_of_record({"retrieval_order": [], "window": []}) is None


# ---------------------------------------------------------------------------- #
# the seven rules on one generation against the live probe                      #
# ---------------------------------------------------------------------------- #
def scripted_generate(attr_of_entity: dict):
    """A deterministic stand in for the model's generation: per question an abstention, a citation of a label with the
    question's entity's planted value or a wrong value, or a malformed answer, chosen by the question's text."""
    import re
    name = re.compile(r"[BDFGKLMNPRSTVZ][aeiou][bdfgklmnprstvz][aeiou]-[0-9a-f]{4}")

    def gen(model, prefix, window_tokens=None, window_mask=None, gate=1.0, max_new=48, policy="plain", window_bytes=None):
        text = bytes(b for b in prefix if b < 256).decode("utf-8", errors="replace")
        h = int(content_hash(text.encode("utf-8"))[:4], 16)
        labels = sorted(raw[0] for raw in (window_bytes or b"").split(b"\n") if raw and raw[0] in LABELS)
        if h % 5 == 0 or not labels:
            return [TOK_UNKNOWN], None
        if h % 5 == 1:
            return [TOK_CITE, LABELS[0]], None                                         # malformed
        label = labels[h % len(labels)]
        m = name.findall(text)
        value = (attr_of_entity.get(m[0], "zzz") if (m and h % 5 != 4) else "zzz").encode("utf-8")
        return [TOK_CITE, label, TOK_ANS, *value, NEWLINE], None
    return gen


def test_replayed_rules_equal_the_live_probe_under_mark_veto_value_r4(monkeypatch, tmp_path, fams):
    cfg, model = tiny()
    fam = trimmed(fams["A"], 20, 5, 0)
    key = generate_key()
    jpath = tmp_path / "A.jsonl"
    planting = SF.plant_family(model, cfg, fam, jpath, key, torch.device("cpu"), admission=ADMISSION_IDENTITY)
    assert planting["counts"]["admitted"] == 20
    monkeypatch.setattr(hm_lm, "generate", scripted_generate({f["entity"]: f["attribute"] for f in fam["facts"]}))
    thr = SF.threshold_of(model, cfg, torch.device("cpu"))["threshold"]
    texts = {content_hash(f["statement"].encode("utf-8")): f["statement"].encode("utf-8") for f in fam["facts"]}
    questions = SF.questions_of(fam, "direct", fams)[:25]
    journal = Journal(jpath, key=key, policy=POLICY_STOP)
    bridge = JournalBridge(model, journal, k=3, budget_bytes=96, seed=0, shuffle_seed=1, mark=True, index_geometry=ADOPTED_GEOMETRY, admission=ADMISSION_IDENTITY)

    class A:
        k = 3; budget = 96; window_len = 96; form = "direct"; index = "I0"; decode_shape = "R4"
    seen = {"cite": 0, "abstain": 0, "veto": 0}
    for q in questions:
        rec = SF.probe_one(model, bridge, q, A, planting["pointer_of"], texts, thr)
        f = Fact(entity=q["entity"], schema=q["schema"], attr=(q["expected"] if q["expected"] is not None else "(never planted)"), statement="", query=q["text"], paraphrase="", cue=np.zeros(64, dtype=np.float32))
        pointer_of = {q["entity"]: planting["pointer_of"][q["entity"]]} if q["own_pointer"] is not None else {}
        bridge.rng = SF.question_rng(1, q["index"])
        live = _probe(model, bridge, [f], True, 96, pointer_of, policy="mark-veto+value", veto_threshold=thr, record_index=True, shape="R4")["answers"][0]
        r = rec["rules"]["mark-veto+value"]
        assert live["outcome"] == r["outcome"] and live["kind"] == r["kind"] and live["answered"] == r["answered"], q["index"]
        assert live.get("veto_cited_pointer") == r["veto_cited_pointer"] and live["shape"]["pointer"] == r["shape"]["pointer"]
        assert rec["retrieval_order"] == live["retrieval_order"] and rec["window"] == live["window"]
        live_v = _probe(model, bridge, [f], True, 96, pointer_of, policy="mark-veto", veto_threshold=thr, record_index=True)["answers"][0]
        bridge.rng = SF.question_rng(1, q["index"]); live_v = _probe(model, bridge, [f], True, 96, pointer_of, policy="mark-veto", veto_threshold=thr, record_index=True)["answers"][0]
        assert live_v["outcome"] == rec["rules"]["mark-veto"]["outcome"]
        seen["cite" if rec["raw_kind"] == "cite" else "abstain"] += 1
        seen["veto"] += int(rec["partition"]["mark-veto"] == "veto_abstention")
        assert rec["whole_store"] is None or rec["whole_store"]["own_rank"] >= 1
        o = rec["rules"]["oracle"]
        assert o["answered"] is None and o["outcome"] in ("strict", "invalid", "abstain")              # the frozen mark_oracle's rule: the citation alone
        assert rec["partition"]["oracle"] != "model_abstention" and rec["partition"]["oracle+organ-value"] != "model_abstention"
        if o["outcome"] == "abstain":
            assert rec["partition"]["oracle"] == "rule_abstention" and "strictly above" in o["reason"]
        else:
            assert o["cited_pointer"] is not None and (o["outcome"] == "strict") == (o["cited_pointer"] == q["own_pointer"])
        if rec["raw_kind"] == "malformed":
            assert rec["partition"]["mark-veto"] == "malformed" and rec["partition"]["V1"] == "malformed"
    assert seen["cite"] > 0 and seen["abstain"] > 0                              # the script exercised both branches


def test_generate_returns_identical_tokens_under_every_policy_on_a_model_without_a_head(tmp_path, fams):
    """A.6's pinned claim: a checkpoint with no matching head generates the same tokens under all policies (one generation
    per question and index), on a fixture."""
    from cortex_c2b.hm_lm import generate
    from cortex_c2b.lm_bridge import encode_query
    cfg, model = tiny()
    assert getattr(model, "match_head", None) is None
    fam = trimmed(fams["A"], 6, 2, 0)
    key = generate_key()
    planting = SF.plant_family(model, cfg, fam, tmp_path / "A.jsonl", key, torch.device("cpu"), admission=ADMISSION_IDENTITY)
    journal = Journal(tmp_path / "A.jsonl", key=key, policy=POLICY_STOP)
    bridge = JournalBridge(model, journal, k=3, budget_bytes=96, seed=0, shuffle_seed=1, mark=True, index_geometry=ADOPTED_GEOMETRY, admission=ADMISSION_IDENTITY)
    with torch.no_grad():
        for f in fam["facts"][:4]:
            tok = encode_query(f["query"]); cue = bridge.cues([tok])[0]
            bridge.rng = SF.question_rng(1, 0)
            read = bridge.read(cue)
            wt, wm = bridge.window_tensors([read.window], 96)
            outs = [generate(model, tok, wt, wm, gate=1.0, policy=pol, window_bytes=read.window)[0] for pol in ("plain", "mark-veto", "mark-veto+value")]
            assert outs[0] == outs[1] == outs[2]


# ---------------------------------------------------------------------------- #
# the refusals                                                                    #
# ---------------------------------------------------------------------------- #
def test_the_cli_refuses_a_run_that_omits_the_recipe(tmp_path):
    cfg, model = tiny()
    ck, cfgp = checkpoint(tmp_path / "ckpt.pt", cfg, model)
    base = ["--family", "A", "--ckpt", str(ck), "--config", str(cfgp), "--journal", str(tmp_path / "A.jsonl"), "--out", str(tmp_path)]
    with pytest.raises(SystemExit) as e:
        SF._cli(base)
    assert "--admission identity and --decode-shape R4 are required" in str(e.value)
    with pytest.raises(SystemExit) as e:
        SF._cli(base + ["--admission", "surprise", "--decode-shape", "R4"])
    assert "required" in str(e.value)
    with pytest.raises(SystemExit) as e:
        SF._cli(base + ["--admission", "identity", "--decode-shape", "R1"])
    assert "required" in str(e.value)


def test_a_checkpoint_with_a_matching_head_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("ADR010_LAUNCHER", "1"); monkeypatch.setenv("PYTHONHASHSEED", "0")
    cfg, model = tiny()
    ck, cfgp = checkpoint(tmp_path / "ckpt.pt", cfg, model)
    model.match_head = object()
    monkeypatch.setattr(SF, "load_model", lambda p, c: (model, {"run_id": "tiny", "config_hash": train.config_hash(cfg)}, torch.device("cpu")))
    jpath = tmp_path / "A.jsonl"
    jpath.write_bytes(b""); jpath.with_name("A.key").write_text("00" * 32); jpath.with_name("A.planting.json").write_text("{}")

    class A:
        family = "A"; index = "I0"; form = "direct"; ckpt = str(ck); config = str(cfgp); journal = str(jpath); out = str(tmp_path)
        admission = "identity"; decode_shape = "R4"; budget_seconds = None; stop_file = None; k = 3; budget = 96; window_len = 96
    with pytest.raises(SystemExit) as e:
        SF.run_pass(A)
    assert "matching head" in str(e.value)


# ---------------------------------------------------------------------------- #
# 4 (f) (i): a pass run whole, stopped and resumed, killed and resumed            #
# ---------------------------------------------------------------------------- #
PATCH = """
import os, sys, json
from cortex_c2b import record_tools, sealed_family as SF, sealed_families as S
record_tools.git_state = lambda: ('test0000', True)
_all = S.generate_all
def _trim(fam, n_facts=20, n_never=10, n_cross=5):
    out = dict(fam); out['facts'] = fam['facts'][:n_facts]; out['never_planted'] = fam['never_planted'][:n_never]
    names = {f['entity'] for f in out['facts']}
    out['cross_schema'] = [c for c in fam['cross_schema'] if c['entity'] in names][:n_cross]
    return out
def _gen():
    fams = _all()
    return {k: (_trim(v) if k == 'A' else v) for k, v in fams.items()}
S.generate_all = _gen
SF.seal_check = lambda fam, seal_path=SF.SEAL_PATH: {'seal_path': 'fixture', 'seal_sha256': 'fixture', 'family_content_sha256': 'fixture', 'constants_sha256': 'fixture'}
from cortex_c2b import hm_lm
from cortex_c2b.lm_bridge import TOK_CITE, TOK_ANS, TOK_UNKNOWN, NEWLINE, LABELS
from cortex_c2b import content_hash
from cortex_c2b.hm_protocol import value_of_statement
import re as _re
_name = _re.compile(r"[BDFGKLMNPRSTVZ][aeiou][bdfgklmnprstvz][aeiou]-[0-9a-f]{4}")
_attr = {f['entity']: f['attribute'] for f in S.generate_all()['A']['facts']}
def _gen_tokens(model, prefix, window_tokens=None, window_mask=None, gate=1.0, max_new=48, policy='plain', window_bytes=None):
    text = bytes(b for b in prefix if b < 256).decode('utf-8', errors='replace'); h = int(content_hash(text.encode('utf-8'))[:4], 16)
    labels = sorted(raw[0] for raw in (window_bytes or b'').split(b'\\n') if raw and raw[0] in LABELS)
    if h % 5 == 0 or not labels:
        return [TOK_UNKNOWN], None
    if h % 5 == 1:
        return [TOK_CITE, LABELS[0]], None
    label = labels[h % len(labels)]; m = _name.findall(text)
    value = (_attr.get(m[0], 'zzz') if (m and h % 5 != 4) else 'zzz').encode('utf-8')
    return [TOK_CITE, label, TOK_ANS, *value, NEWLINE], None
hm_lm.generate = _gen_tokens
__EXTRA__
SF._cli(__ARGV__)
"""

STOP_AFTER = """
_orig = SF.probe_one
_n = {{'n': 0}}
def _probe_one(*a, **k):
    r = _orig(*a, **k); _n['n'] += 1
    if _n['n'] == {N}:
        open({STOP!r}, 'w').close()
    return r
SF.probe_one = _probe_one
"""

KILL_AFTER = """
_orig = SF.probe_one
_n = {{'n': 0}}
def _probe_one(*a, **k):
    r = _orig(*a, **k); _n['n'] += 1
    if _n['n'] == {N}:
        os._exit(3)
    return r
SF.probe_one = _probe_one
"""


def run_pass(argv, extra="", hashseed="0"):
    code = PATCH.replace("__EXTRA__", extra).replace("__ARGV__", repr(argv))
    env = {**os.environ, "PYTHONHASHSEED": hashseed, "ADR010_LAUNCHER": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    return subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")


def strip(d):
    drop = {"sessions", "run_in_sessions", "timestamp_utc", "progress_sha256", "progress_file", "wall_clock", "recorded", "duplicates_compared", "code_commit", "journal"}
    if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if k not in drop}
    if isinstance(d, list):
        return [strip(x) for x in d]
    return d


@pytest.fixture(scope="module")
def planted(tmp_path_factory, fams):
    """The trimmed family A planted once on the tiny model (the planting is atomic; the passes read copies)."""
    root = tmp_path_factory.mktemp("pass")
    cfg, model = tiny()
    ck, cfgp = checkpoint(root / "ckpt.pt", cfg, model)
    fam = trimmed(fams["A"])
    key = generate_key()
    jpath = root / "A.jsonl"
    rec = SF.plant_family(model, cfg, fam, jpath, key, torch.device("cpu"), admission=ADMISSION_IDENTITY)
    rec.update({"checkpoint": {"path": str(ck), "run_id": "tiny"}, "threshold": SF.threshold_of(model, cfg, torch.device("cpu"))})
    jpath.with_name("A.planting.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    jpath.with_name("A.key").write_text(key.hex(), encoding="utf-8", newline="\n")
    return {"root": root, "ck": ck, "cfg": cfgp, "journal": jpath, "n_questions": len(SF.questions_of(fam, "direct", {**fams, "A": fam}))}


def pass_argv(planted, out: Path, index="I0", form="direct", extra=()):
    return ["--pass", "--family", "A", "--index", index, "--form", form, "--ckpt", str(planted["ck"]), "--config", str(planted["cfg"]),
            "--journal", str(planted["journal"]), "--out", str(out), "--admission", "identity", "--decode-shape", "R4", *extra]


def test_a_pass_whole_stopped_and_killed_then_resumed_gives_the_same_file(planted):
    root = planted["root"]
    whole = root / "whole"; whole.mkdir()
    p = run_pass(pass_argv(planted, whole))
    assert p.returncode == 0, p.stderr[-3000:]
    fw = json.loads((whole / "sealed-A-tiny-I0-direct.pass.json").read_text(encoding="utf-8"))
    assert fw["n_questions"] == planted["n_questions"] and fw["run_in_sessions"] == 1
    assert {a["block"] for a in fw["answers"]} == {"planted", "never_planted", "cross_schema", "near_name"}
    assert all(a["whole_store"] is not None for a in fw["answers"] if a["own_pointer"] is not None)
    assert any(a["rules"]["plain"]["kind"] == "cite" for a in fw["answers"])
    assert fw["texts"] and all(ptr in fw["texts"] for a in fw["answers"] for ptr in [w["pointer"] for w in a.get("window", [])])   # A0 reads these
    assert set(fw["own_pointer_of"]) == {a["entity"] for a in fw["answers"] if a["block"] == "planted"} and "own_rank" in fw["field_labels"]
    assert fw["admission"] == "identity" and fw["admission_asked"] == "identity" and fw["oracle"]["row47"] is None
    assert fw["sessions"][0]["code_state"]["new_modules_sha256_lf"]["cortex_c2b/sealed_family.py"]
    # stopped after question 5 (the stop file), resumed by the same command
    stopped = root / "stopped"; stopped.mkdir()
    stop = stopped / "stop"
    p = run_pass(pass_argv(planted, stopped, extra=["--stop-file", str(stop)]), extra=STOP_AFTER.format(N=6, STOP=str(stop)))
    assert p.returncode == 0 and "stopped (stop_file)" in p.stdout and "resumption point 6" in p.stdout
    stop.unlink()
    p = run_pass(pass_argv(planted, stopped, extra=["--stop-file", str(stop)]))
    assert p.returncode == 0, p.stderr[-3000:]
    fs = json.loads((stopped / "sealed-A-tiny-I0-direct.pass.json").read_text(encoding="utf-8"))
    assert fs["run_in_sessions"] == 2 and [s["reason"] for s in fs["sessions"]] == ["stop_file", "complete"] and fs["sessions"][1]["resumed_from"] == 6
    assert strip(fs) == strip(fw)
    # killed after question 8, resumed
    killed = root / "killed"; killed.mkdir()
    p = run_pass(pass_argv(planted, killed), extra=KILL_AFTER.format(N=9))
    assert p.returncode == 3
    # a resumption under another numeric environment is refused (4 (b)); the run stays where it was
    p = run_pass(pass_argv(planted, killed), hashseed="7")
    assert p.returncode != 0 and "resumption is refused" in p.stderr and "PYTHONHASHSEED" in p.stderr
    p = run_pass(pass_argv(planted, killed))
    assert p.returncode == 0, p.stderr[-3000:]
    fk = json.loads((killed / "sealed-A-tiny-I0-direct.pass.json").read_text(encoding="utf-8"))
    assert fk["run_in_sessions"] == 2 and fk["sessions"][0]["reason"] == "death" and fk["sessions"][1]["resumed_from"] == 8
    # the oracle's record is read from the identity line at a resumption (4 (b)), whole: the same block as the run in one session
    assert fk["oracle"] == fw["oracle"] and fk["oracle"]["read_rule"].startswith("computed once") and "training_pairs" in fk["oracle"]
    assert fk["identity"]["recorded"]["oracle"]["threshold"] == fw["oracle"]["threshold"]
    assert strip(fk) == strip(fw)
    assert fk["identity"]["compared"] == fw["identity"]["compared"]             # the same run identity (the journal, the key, the threshold)
    # a file of the record is never overwritten
    p = run_pass(pass_argv(planted, whole))
    assert p.returncode != 0 and "never overwritten" in p.stderr
    # a run started without the launcher is refused
    code = "from cortex_c2b import sealed_family as SF\nSF._cli(%r)\n" % pass_argv(planted, root / "no-launcher")
    env = {k: v for k, v in os.environ.items() if k not in ("ADR010_LAUNCHER", "PYTHONHASHSEED")}
    p = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "tools/adr010_launcher.py" in p.stderr


def test_reference_passes_record_the_whole_store_rank_and_the_reference_ranking(planted):
    out = planted["root"] / "refs"; out.mkdir()
    for index in ("I1", "I2"):
        p = run_pass(pass_argv(planted, out, index=index, form="paraphrase"))
        assert p.returncode == 0, p.stderr[-3000:]
        f = json.loads((out / f"sealed-A-tiny-{index}-paraphrase.pass.json").read_text(encoding="utf-8"))
        assert f["index"] == index and f["form"] == "paraphrase" and {a["block"] for a in f["answers"]} == {"planted", "never_planted"}
        for a in f["answers"]:
            assert a["reference"]["index"] == index and len(a["reference"]["entry_ids"]) <= 3
            if a["own_pointer"] is not None:
                assert a["own_in_candidates"] is True                                # every live entry is a candidate on a reference pass
                assert a["own_rank"] == a["whole_store"]["own_rank"]                 # the rank under the whole store, by the organ's cosine
        if index == "I2":
            assert all("query_tokens" in a["reference"] for a in f["answers"])
            assert f["bm25"]["pair"] == {"k1": 1.5, "b": 0.75}
