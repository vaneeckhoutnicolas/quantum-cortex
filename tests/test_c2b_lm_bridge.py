"""ADR-008 -- the journal in the decode loop: one or more tests per invariant, the
cite-or-abstain contract, and the restart on the MODEL in a new process."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train                                                       # noqa: E402
from cortex_c2 import RouterV1, RouteDecision                      # noqa: E402
from cortex_c2b import Journal, POLICY_STOP                        # noqa: E402
from cortex_c2b.crypto import generate_key                         # noqa: E402
from cortex_c2b.hm_protocol import generate_facts, N_FACTS, N_NEGCTRL   # noqa: E402
from cortex_c2b.lm_bridge import (JournalBridge, CueEncoder, contrastive_loss, build_read_window,  # noqa: E402
                                  encode_query, encode_target, parse_contract, journal_gate,
                                  TOK_EPI, TOK_CITE, TOK_ANS, TOK_UNKNOWN, LABELS, NEWLINE, PATH_JOURNAL)
from cortex_c2b.organ_use import (training_facts, decontaminate, plant_pool, make_example, make_batch,  # noqa: E402
                                  batch_stats, collate, Example, EVAL_SEEDS)
from cortex_c2b.hm_lm import run_hm_lm, INVALID_CITATION_MAX       # noqa: E402


def tiny(journal="read", seed=0, **kw):
    cfg = train.Config(n_layer=3, n_head=2, n_embd=32, block_size=96, journal=journal, journal_block=1,
                       journal_heads=2, journal_read_bytes=96, journal_cue_dim=64, journal_k=3, **kw)
    torch.manual_seed(seed)
    return cfg, train.VanillaGPT(cfg)


# ------------------------------------------------------------------ invariant 1
def test_n1_intact_at_step_zero_and_parent_loads_strictly(tmp_path):
    cfg, m = tiny()
    idx = torch.randint(0, 256, (2, 24))
    w = torch.randint(0, 256, (2, 30)); wm = torch.ones(2, 30, dtype=torch.bool)
    a, _ = m(idx); b, _ = m(idx, window_tokens=w, window_mask=wm); c, _ = m(idx, window_tokens=w, window_mask=wm, gate=0.0)
    assert torch.equal(a, b) and torch.equal(a, c)               # the no-op law, window or not, gate or not
    cfg0, parent = tiny(journal="none", seed=1)
    ck = tmp_path / "parent.pt"
    torch.save({"model": parent.state_dict(), "run_id": "parent01", "config_hash": "x"}, ck)
    run_id, fresh = train.load_parent(m, ck, torch.device("cpu"))
    assert run_id == "parent01" and fresh and all(("reader" in k or "cue_encoder" in k) for k in fresh)
    p_logits, _ = parent(idx)
    m_logits, _ = m(idx, window_tokens=w, window_mask=wm)
    assert torch.allclose(p_logits, m_logits, atol=1e-6)          # the parent, unchanged, plus a silent reader
    bad = {k: v for k, v in parent.state_dict().items() if not k.startswith("blocks.0.attn")}
    torch.save({"model": bad, "run_id": "broken"}, ck)
    with pytest.raises(SystemExit):
        train.load_parent(tiny()[1], ck, torch.device("cpu"))      # a missing non-journal weight is refused


# ------------------------------------------------------------------ invariant 3: the contract and the window
def test_read_window_is_labelled_budgeted_and_shuffled():
    rng = np.random.default_rng(3)
    items = [("p1", b"alpha " * 20), ("p2", b"beta " * 20), ("p3", b"gamma " * 20)]
    window, labels = build_read_window(items, budget=60, rng=rng)
    assert set(labels) <= set(LABELS) and set(labels.values()) == {"p1", "p2", "p3"}
    lines = window.split(b"\n")[:-1]
    assert len(lines) == 3 and all(l[1:2] == b":" for l in lines) and len(window) <= 60 + 3 * 3
    orders = {tuple(build_read_window(items, 60, np.random.default_rng(s))[1].values()) for s in range(20)}
    assert len(orders) > 1                                       # a position is never a shortcut
    assert build_read_window([], 60, rng) == (b"", {})
    assert parse_contract(encode_target(ord("B"), "glazier")) == ("cite", ord("B"), "glazier")
    assert parse_contract(encode_target(None, None)) == ("unknown", None, None)
    assert parse_contract([TOK_CITE, ord("Z"), TOK_ANS] + list(b"x\n"))[0] == "malformed"
    assert parse_contract(list(b"a guess\n"))[0] == "malformed"
    assert encode_query("who")[0] == TOK_EPI and {TOK_EPI, TOK_CITE, TOK_ANS, TOK_UNKNOWN} <= set(range(256, 264))


# ------------------------------------------------------------------ invariant 4
def test_router_gate_on_the_floor():
    assert journal_gate("pinned_on") == 1.0 and journal_gate("pinned_off") == 0.0
    assert journal_gate("router", None) == 0.0
    assert journal_gate("router", RouterV1().route(path_scores=None)) == 0.0          # no score: the control
    dec = RouteDecision(path=PATH_JOURNAL, confidence=0.7)
    assert journal_gate("router", dec) == 0.7
    assert journal_gate("router", RouteDecision(path=0, confidence=0.9)) == 0.0
    assert journal_gate("router", RouteDecision(path=1, confidence=1.0, weights=[0.2, 0.1, 0.1, 0.1, 0.5])) == 0.5


# ------------------------------------------------------------------ invariant 5
def test_cue_encoder_learns_by_contrast_and_retrieval_improves():
    cfg, m = tiny(seed=5)
    facts, _ = generate_facts(20, seed=100_000)
    def hit_rate(model):
        model.eval()
        jb = JournalBridge(model, Journal(), k=3, budget_bytes=96, seed=0)
        plant_pool(jb, facts)
        hits = 0
        for f in facts:
            cue = jb.cues([encode_query(f.query)])[0]
            hits += int(jb.pointer_of.get(f.entity) in jb.read(cue).pointers)   # a rejected write is a miss
        model.train()
        return hits / len(facts)
    before = hit_rate(m)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    jb = JournalBridge(m, Journal(), k=3, budget_bytes=96, seed=0)
    q = [encode_query(f.query) for f in facts]; s = [list(f.statement.encode()) for f in facts]
    l0 = float(contrastive_loss(jb.cue_tensor(q), jb.cue_tensor(s)).detach())
    for _ in range(40):
        loss = contrastive_loss(jb.cue_tensor(q), jb.cue_tensor(s))
        opt.zero_grad(); loss.backward(); opt.step()
    l1 = float(loss.detach())
    after = hit_rate(m)
    assert l1 < l0 and after >= before and after >= 0.8, (l0, l1, before, after)


def test_training_facts_are_disjoint_from_the_frozen_protocol():
    facts, negs, rep = training_facts(40, seed=100_000)
    eval_entities = {f.entity for s in EVAL_SEEDS for f in generate_facts(N_FACTS if s == 0 else N_NEGCTRL, s)[0]}
    assert not ({f.entity for f in facts} & eval_entities) and not ({f.entity for f in negs} & eval_entities)
    assert not ({f.entity for f in facts} & {f.entity for f in negs})
    assert rep["seed"] == 100_000 and rep["n_facts"] + len(rep["removed_collisions"]) >= 40
    with pytest.raises(ValueError):
        training_facts(10, seed=0)
    ev = generate_facts(5, 0)[0]
    kept, removed = decontaminate(ev + facts[:3], ev)              # a planted collision is caught
    assert removed == [f.entity for f in ev] and len(kept) == 3


# ------------------------------------------------------------------ invariant 6: the target rule
def test_targets_follow_what_the_journal_returned():
    cfg, m = tiny(seed=6); m.eval()
    facts, negs, _ = training_facts(30, seed=100_000)
    jb = JournalBridge(m, Journal(), k=3, budget_bytes=96, seed=0)
    plant_pool(jb, facts)
    rng = np.random.default_rng(0)
    f = facts[0]
    pointer = jb.pointer_of[f.entity]
    real_read = jb.read
    # 1. the right episode among the retrieved -> cite it
    jb.read = lambda cue: type(real_read(cue))(window=b"A:" + jb.j.payloads.get(pointer) + b"\n", labels={ord("A"): pointer},
                                                pointers=[pointer], scores=[1.0])
    ex = make_example(jb, f, negative=False, forced_frac=0.0, rng=rng)
    assert ex.kind == "answer" and ex.retrieved_ok and ex.target[:3] == [TOK_CITE, ord("A"), TOK_ANS]
    assert bytes(ex.target[3:-1]).decode() == f.attr and ex.target[-1] == NEWLINE
    # 2. not among the retrieved -> abstain, even though the fact is in the journal
    other = jb.pointer_of[facts[1].entity]
    jb.read = lambda cue: type(real_read(cue))(window=b"A:x\n", labels={ord("A"): other}, pointers=[other], scores=[0.1])
    ex = make_example(jb, f, negative=False, forced_frac=0.0, rng=rng)
    assert ex.kind == "abstain" and not ex.retrieved_ok and ex.target == [TOK_UNKNOWN, NEWLINE]
    # 3. the bootstrap: forced into the window, marked, never counted as retrieved
    ex = make_example(jb, f, negative=False, forced_frac=1.0, rng=rng)
    assert ex.kind == "forced" and not ex.retrieved_ok and ex.target[0] == TOK_CITE and ex.labels[ex.target[1]] == pointer
    # 4. a never-planted entity -> abstain
    ex = make_example(jb, negs[0], negative=True, forced_frac=1.0, rng=rng)
    assert ex.kind == "negative" and ex.target == [TOK_UNKNOWN, NEWLINE]
    # 5. the loss weights: nothing on the query, the asymmetric weight on the decision token of an abstention
    w = ex.weights(asym=2.0)
    assert w[: len(ex.query)] == [0.0] * len(ex.query) and w[len(ex.query)] == 2.0 and w[-1] == 1.0
    jb.read = real_read
    batch = make_batch(jb, facts, negs, 16, rng, negatives_frac=0.25, forced_frac=0.5)
    st = batch_stats(batch)
    assert st["n_negative"] + st["n_answer"] + st["n_abstain"] + st["n_forced"] == 16
    x, y, wts, wt, wm, qs, ss = collate(jb, batch, 2.0, 96)
    assert x.shape == y.shape == wts.shape and wt.shape[1] <= 96 and len(qs) == len(ss) == 16
    for i, e in enumerate(batch):                                      # next token targets, row by row
        n = len(e.tokens) - 1
        assert torch.equal(x[i, 1:n], y[i, : n - 1]) and int(y[i, n - 1]) == e.tokens[-1]
    m.train()


# ------------------------------------------------------------------ invariants 7, 8, 9: the LM arm
def test_lm_arm_reports_attributable_fields_and_never_claims_in_memory(tmp_path):
    cfg, m = tiny(seed=7)
    r = run_hm_lm(m, lambda: Journal(), n_facts=10, n_negctrl=5, k=3, budget=96, window_len=96)
    for k in ("hm_recall_on", "hm_recall_off", "hm_skill_delta", "hm_invalid_citation_on", "hm_false_abstention_on",
              "hm_negctrl_rate", "hm_retrieval_hit", "hm_attention_mass", "hm_dissociation_pass", "persistent", "claimable"):
        assert k in r
    assert r["hm_recall_off"] == 0.0 and r["persistent"] is False and r["claimable"] == 0
    assert "not claimable" in r["verdict"] and r["thresholds"]["invalid_citation_max"] == INVALID_CITATION_MAX
    assert r["attribution"] == {"encoder": r["hm_retrieval_hit"], "reader": r["hm_attention_mass"],
                                "answer": r["hm_attr_exact_given_valid"]}
    key = generate_key(); path = tmp_path / "hm.jsonl"
    r2 = run_hm_lm(m, lambda: Journal(path, key=key, policy=POLICY_STOP), n_facts=10, n_negctrl=5,
                   k=3, budget=96, window_len=96, on_disk=True)
    assert r2["persistent"] is True and r2["storage"]["mode"] == "durable"   # sealed, reopened for session B


SESSION_B = r"""
import json, sys, torch
sys.path.insert(0, sys.argv[1])
import train
from cortex_c2b import Journal, POLICY_STOP
from cortex_c2b.crypto import key_from_env
from cortex_c2b.hm_protocol import generate_facts
from cortex_c2b.lm_bridge import JournalBridge, encode_query
cfg = train.Config(**json.loads(sys.argv[2]))
model = train.VanillaGPT(cfg)
ck = torch.load(sys.argv[3], map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"]); model.eval()
j = Journal(sys.argv[4], key=key_from_env(), policy=POLICY_STOP)     # a new process: the disk alone
jb = JournalBridge(model, j, k=3, budget_bytes=96, seed=0, shuffle_seed=1)
facts, _ = generate_facts(20, seed=0)
pointer_of = json.loads(sys.argv[5])
hits = sum(int(pointer_of[f.entity] in jb.read(jb.cues([encode_query(f.query)])[0]).pointers) for f in facts)
print(json.dumps({"hits": hits, "entries": len(j), "mode": j.mode}))
"""


def test_the_model_and_its_journal_survive_a_full_process_restart(tmp_path):
    """The named boundary (ADR-007 D8): session B in a new process, the model from
    its checkpoint, the journal from the sealed disk, the cues reproduced."""
    cfg, m = tiny(seed=8); m.eval()
    key = generate_key(); path = tmp_path / "journal.jsonl"
    j = Journal(path, key=key, policy=POLICY_STOP)
    jb = JournalBridge(m, j, k=3, budget_bytes=96, seed=0, shuffle_seed=1)
    facts, _ = generate_facts(20, seed=0)
    plant_pool(jb, facts)
    hits_a = sum(int(jb.pointer_of[f.entity] in jb.read(jb.cues([encode_query(f.query)])[0]).pointers) for f in facts)
    ck = tmp_path / "ckpt.pt"
    torch.save({"model": m.state_dict(), "run_id": "tiny"}, ck)
    pointer_of = dict(jb.pointer_of)
    del jb, j, m
    env = {**os.environ, "QUANTUM_CORTEX_JOURNAL_KEY": key.hex(), "PYTHONPATH": str(ROOT)}
    cfg_json = json.dumps({k: getattr(cfg, k) for k in cfg.__dataclass_fields__})
    run = subprocess.run([sys.executable, "-c", SESSION_B, str(ROOT), cfg_json, str(ck), str(path), json.dumps(pointer_of)],
                         capture_output=True, text=True, cwd=ROOT, env=env, timeout=180)
    assert run.returncode == 0, run.stderr
    r = json.loads(run.stdout.strip().splitlines()[-1])
    assert r["entries"] == 20 and r["mode"] == "durable" and r["hits"] == hits_a


# ------------------------------------------------------------------ invariant 9: the trainer, end to end (CI runs the slow ones)
@pytest.mark.slow
@pytest.mark.parametrize("config", ["configs/smoke_journal_cpu.json", "configs/smoke_journal_v2_cpu.json"])
def test_cpu_smoke_trains_the_journal_and_records_the_lm_arm(config):
    ledger, latest = ROOT / "metrics" / "runs.jsonl", ROOT / "metrics" / "LATEST.md"
    lb = ledger.read_text() if ledger.exists() else None
    la = latest.read_text() if latest.exists() else None
    before = len(lb.splitlines()) if lb else 0
    try:
        proc = subprocess.run([sys.executable, "train.py", "--config", config],
                              cwd=ROOT, capture_output=True, text=True, timeout=1500)
        assert proc.returncode == 0, f"trainer failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
        lines = ledger.read_text().splitlines()
        assert len(lines) == before + 1
        rec = json.loads(lines[-1])
        from jsonschema import validate
        validate(rec, json.loads((ROOT / "metrics" / "schema" / "run-v1.schema.json").read_text()))
        assert rec["model"]["architecture_id"].startswith("journal-")
        suite = rec["results"]["benchmarks"]["standard_suite"]
        assert {"hm_recall_on", "hm_recall_off", "hm_skill_delta", "hm_invalid_citation_on", "hm_persistent", "hm_claimable"} <= set(suite)
        out_dir = json.loads((ROOT / config).read_text())["out_dir"]
        assert (ROOT / out_dir / "hm-lm.json").exists()
        for f in (ROOT / "metrics" / "mqar").glob(f"hm-lm-{rec['run_id']}*.json"):
            f.unlink()                                             # a smoke leaves no artefact behind
    finally:
        if lb is not None:
            ledger.write_text(lb)
        if la is not None:
            latest.write_text(la)


def test_session_b_in_a_new_process_reproduces_the_in_process_probe(tmp_path):
    """The named boundary as a run artefact: `python -m cortex_c2b.hm_lm --session-b`
    reopens the checkpoint and the sealed journal in a fresh process and must
    reproduce the ON probe of the in process arm, number for number."""
    from cortex_c2b.crypto import generate_key
    from cortex_c2b import POLICY_STOP
    cfg, m = tiny(seed=9); m.eval()
    key = generate_key(); jpath = tmp_path / "journal.jsonl"
    r_a = run_hm_lm(m, lambda: Journal(jpath, key=key, policy=POLICY_STOP), n_facts=12, n_negctrl=6,
                    k=3, budget=96, window_len=96, on_disk=True)
    ck = tmp_path / "ckpt.pt"
    torch.save({"model": m.state_dict(), "run_id": "tiny", "config_hash": "x", "step": 0}, ck)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({k: getattr(cfg, k) for k in cfg.__dataclass_fields__}), encoding="utf-8")
    env = {**os.environ, "QUANTUM_CORTEX_JOURNAL_KEY": key.hex(), "PYTHONPATH": str(ROOT)}
    out = tmp_path / "session-b.json"
    run = subprocess.run([sys.executable, "-m", "cortex_c2b.hm_lm", "--session-b", "--ckpt", str(ck), "--journal", str(jpath),
                          "--config", str(cfg_path), "--out", str(out), "--n-facts", "12", "--n-negctrl", "6"],
                         capture_output=True, text=True, cwd=ROOT, env=env, timeout=300)
    assert run.returncode == 0, run.stderr
    r_b = json.loads(out.read_text())
    assert r_b["journal"]["mode"] == "durable" and r_b["journal"]["planted_found"] == r_a["generator"]["admitted"]
    for k in ("hm_recall_on", "hm_recall_off", "hm_false_abstention_on", "hm_invalid_citation_on", "hm_retrieval_hit"):
        assert r_b[k] == r_a[k], (k, r_b[k], r_a[k])
    assert r_b["persistent"] is True


def test_fresh_pools_never_repeat_and_stay_decontaminated():
    a, na, _ = training_facts(40, seed=100_000)
    b, nb, _ = training_facts(40, seed=100_002)                     # the second replant's seed (train_seed + 2 x refresh)
    eval_entities = {f.entity for s in EVAL_SEEDS for f in generate_facts(N_FACTS if s == 0 else N_NEGCTRL, s)[0]}
    assert not ({f.entity for f in a} & {f.entity for f in b})          # a new set every replant
    assert not (({f.entity for f in a} | {f.entity for f in b}) & eval_entities)
    assert train.config_hash(train.Config(journal_fresh_pool=True)) != train.config_hash(train.Config())
    assert train.config_hash(train.Config(journal_fresh_pool=False)) == train.config_hash(train.Config())   # off: transparent


def test_session_b_probe_on_the_training_pool_reports_memorisation_evidence(tmp_path):
    from cortex_c2b.crypto import generate_key
    from cortex_c2b import POLICY_STOP
    from cortex_c2b.hm_lm import session_b_from_disk
    cfg, m = tiny(seed=10); m.eval()
    key = generate_key(); jpath = tmp_path / "journal.jsonl"
    run_hm_lm(m, lambda: Journal(jpath, key=key, policy=POLICY_STOP), n_facts=8, n_negctrl=4, k=3, budget=96, window_len=96, on_disk=True)
    ck = tmp_path / "ckpt.pt"; torch.save({"model": m.state_dict(), "run_id": "tiny"}, ck)
    os.environ["QUANTUM_CORTEX_JOURNAL_KEY"] = key.hex()
    try:
        r = session_b_from_disk(ck, jpath, cfg, n_facts=8, n_negctrl=4, k=3, budget=96, window_len=96,
                                train_pool_seed=100_000, train_pool_n=10)
    finally:
        del os.environ["QUANTUM_CORTEX_JOURNAL_KEY"]
    tp = r["training_pool_probe"]
    assert tp["seed"] == 100_000 and tp["n"] == 10 and {"recall_strict", "valid_citation", "invalid_citation", "abstain_rate", "retrieval_hit"} <= set(tp)
