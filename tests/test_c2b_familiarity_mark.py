"""RES-23, the familiarity mark (v9), and v8, the reader with a local convolution:
declared in ADR-008 on 2026-09-17 before either run. The mark is the organ's own
similarity, written into the window line as `A(0.97):`; every route to a window
carries it; the flag is transparent to every recorded hash at its default; the
oracle reads a threshold from the training family only. The v8 convolution starts
as the identity and its keys are the only fresh weights next to the journal's."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train                                                       # noqa: E402
from cortex_c2b import Journal                                     # noqa: E402
from cortex_c2b.hm_protocol import generate_facts                  # noqa: E402
from cortex_c2b.lm_bridge import JournalBridge, build_read_window, encode_query   # noqa: E402
from cortex_c2b.organ_use import plant_pool, make_example, make_paired_negative, training_facts   # noqa: E402
from cortex_c2b.hm_lm import mark_oracle, run_hm_lm               # noqa: E402


def tiny(journal="read", seed=0, **kw):
    cfg = train.Config(n_layer=3, n_head=2, n_embd=32, block_size=96, journal=journal, journal_block=1,
                       journal_heads=2, journal_read_bytes=96, journal_cue_dim=64, journal_k=3, **kw)
    torch.manual_seed(seed)
    return cfg, train.VanillaGPT(cfg)


# ---- the window line ---------------------------------------------------------------
def test_the_marked_window_carries_two_decimals_inside_the_same_budget_and_the_plain_window_is_unchanged():
    items = [("p1", b"alpha " * 20), ("p2", b"beta " * 20), ("p3", b"gamma " * 20)]
    plain, lp = build_read_window(items, 96, np.random.default_rng(3))
    marked, lm = build_read_window(items, 96, np.random.default_rng(3), marks=[0.973, 0.3, -0.2])
    assert lp == lm                                                # the same labels, the same order
    assert all(re.match(rb"^[A-H]\(\d\.\d\d\):", line) for line in marked.split(b"\n") if line)
    assert b"(0.97):" in marked and b"(0.30):" in marked and b"(0.00):" in marked   # clamped at zero
    assert len(marked) <= 96 and len(plain) <= 96
    assert all(re.match(rb"^[A-H]:", line) for line in plain.split(b"\n") if line)
    with pytest.raises(ValueError):
        build_read_window(items, 96, np.random.default_rng(3), marks=[0.5])


# ---- the mark is the organ's similarity, on every route --------------------------------
def _planted_bridge(mark: bool, seed: int = 0):
    cfg, m = tiny(seed=seed)
    b = JournalBridge(m, Journal(), k=3, budget_bytes=96, seed=0, shuffle_seed=1, mark=mark)
    facts, negs, _ = training_facts(12, seed=100_000)
    plant_pool(b, facts)
    return b, facts, negs


def test_the_read_window_s_marks_are_the_index_s_scores_and_the_similarity_by_pointer_agrees():
    b, facts, _ = _planted_bridge(mark=True)
    cue = b.cues([encode_query(facts[0].query)])[0]
    read = b.read(cue)
    assert read.scores and len(read.pointers) == len(read.scores)
    for pointer, score in zip(read.pointers, read.scores):
        assert abs(b.jp.cue_similarity(cue, pointer) - score) < 1e-5     # the same computation
        assert (f"({max(0.0, min(1.0, score)):.2f}):").encode() in read.window
    assert b.jp.cue_similarity(cue, "not-a-pointer") == 0.0


def test_every_curriculum_route_carries_the_mark_when_the_bridge_marks_and_none_when_it_does_not():
    for mark in (True, False):
        b, facts, negs = _planted_bridge(mark=mark)
        rng = np.random.default_rng(0)
        ex = make_example(b, facts[0], False, 0.0, rng)
        forced = make_example(b, facts[1], False, 1.0, rng)
        pair = make_paired_negative(b, facts[2], rng, keep_shape=True)
        for e in (ex, forced, pair):
            lines = [l for l in e.window.split(b"\n") if l]
            if mark:
                assert all(re.match(rb"^[A-H]\(\d\.\d\d\):", l) for l in lines), e.kind
            else:
                assert all(re.match(rb"^[A-H]:", l) for l in lines), e.kind


# ---- the flag and the hash ------------------------------------------------------------
def test_the_two_flags_are_transparent_at_their_defaults_and_enter_the_hash_when_set():
    base = train.config_hash(train.Config())
    assert train.config_hash(train.Config(journal_familiarity_mark=False)) == base
    assert train.config_hash(train.Config(attn_local_conv_init="random")) == base
    assert train.config_hash(train.Config(journal_familiarity_mark=True)) != base
    assert train.config_hash(train.Config(attn_local_conv=True, attn_local_conv_init="identity")) != train.config_hash(train.Config(attn_local_conv=True))
    assert train.HASH_TRANSPARENT_AT_DEFAULT["journal_familiarity_mark"] is False
    assert train.HASH_TRANSPARENT_AT_DEFAULT["attn_local_conv_init"] == "random"


def test_the_v8_and_v9_configurations_are_v6_plus_one_flag_each():
    import json
    fields = train.Config.__dataclass_fields__
    load = lambda name: {k: v for k, v in json.load(open(Path(__file__).resolve().parent.parent / "configs" / name)).items() if k in fields}
    v6, v8, v9 = load("kaggle_t4_journal_v6.json"), load("kaggle_t4_journal_v8.json"), load("kaggle_t4_journal_v9.json")
    assert train.config_hash(train.Config(**v6)) == "ca90d290ad33b472"       # v6 as recorded (ledger 81cb13a684aa)
    diff8 = {k for k in set(v6) | set(v8) if v6.get(k) != v8.get(k)}
    diff9 = {k for k in set(v6) | set(v9) if v6.get(k) != v9.get(k)}
    assert diff8 == {"attn_local_conv", "attn_local_conv_init", "out_dir", "journal_path", "notes"}
    assert diff9 == {"journal_familiarity_mark", "out_dir", "journal_path", "notes"}
    assert len({train.config_hash(train.Config(**c)) for c in (v6, v8, v9)}) == 3


# ---- v8: the identity convolution and the parent loader ----------------------------------
def test_the_identity_convolution_leaves_the_parent_exactly_and_loads_as_the_only_fresh_keys(tmp_path):
    cfg0, parent = tiny(journal="none", seed=1)
    ck = tmp_path / "parent.pt"
    torch.save({"model": parent.state_dict(), "run_id": "parent01", "config_hash": "x"}, ck)
    cfg, m = tiny(attn_local_conv=True, attn_local_conv_init="identity")
    run_id, fresh = train.load_parent(m, ck, torch.device("cpu"))
    assert run_id == "parent01" and all(("reader" in k or "cue_encoder" in k or ".conv." in k) for k in fresh)
    assert any(".conv." in k for k in fresh)
    idx = torch.randint(0, 256, (2, 24))
    a, _ = parent(idx); b, _ = m(idx)
    assert torch.allclose(a, b, atol=1e-6)                        # step zero is the parent exactly
    cfg_r, m_r = tiny(attn_local_conv=True)                        # the ladder's random init, unchanged (row 32)
    train.load_parent(m_r, ck, torch.device("cpu"))
    c, _ = m_r(idx)
    assert not torch.allclose(a, c, atol=1e-4)


# ---- the oracle -----------------------------------------------------------------------
def test_the_mark_oracle_reads_its_threshold_from_the_training_family_and_reports_the_declared_fields():
    b, facts, negs = _planted_bridge(mark=True)
    protocol, _ = generate_facts(10, 0)
    for i, f in enumerate(protocol):
        b.write(f.statement, f.schema, now=100.0 + i, entity=f.entity)
    pointer_of = dict(b.pointer_of)
    neg, _ = generate_facts(5, 10_000)
    r = mark_oracle(b.model, b, protocol, neg, pointer_of, train_seed=100_000, train_n=12)
    assert set(r) >= {"threshold", "training_accuracy", "recall_on_strict", "claims_on", "negctrl_claims", "gap",
                      "passes_without_skill_arm", "training_pairs"}
    assert 0.0 <= r["threshold"] <= 1.0 and 0.0 <= r["recall_on_strict"] <= 1.0 and 0.0 <= r["negctrl_claims"] <= 1.0
    assert r["training_pairs"] > 0


def test_the_arm_records_the_marks_and_the_oracle_with_or_without_the_mark_in_the_window():
    for mark in (True, False):
        cfg, m = tiny()
        r = run_hm_lm(m, lambda: Journal(), n_facts=8, n_negctrl=4, k=3, budget=96, window_len=96, mark=mark)
        assert r["hm_marks"]["written_in_window"] is mark
        assert 0 < r["hm_marks"]["on"]["max"]["n"] <= 8                 # a read may return fewer than k lines, or none
        assert r["hm_marks"]["negctrl"]["max"] is None or r["hm_marks"]["negctrl"]["max"]["n"] <= 4
        assert "threshold" in r["hm_mark_oracle"]


def test_session_b_keeps_every_answer_per_line_with_its_marks(tmp_path, monkeypatch):
    """Rev66: a recording addition, not a variable: the same numbers plus the answers."""
    import os
    from cortex_c2b.hm_lm import session_b_from_disk
    from cortex_c2b.crypto import generate_key
    key = generate_key(); monkeypatch.setenv("QUANTUM_CORTEX_JOURNAL_KEY", key.hex())
    cfg, m = tiny()
    ck = tmp_path / "ckpt.pt"; jp = tmp_path / "journal.jsonl"
    torch.save({"model": m.state_dict(), "run_id": "tiny0001", "config_hash": train.config_hash(cfg)}, ck)
    facts, _ = generate_facts(8, 0)
    from cortex_c2b import Journal, POLICY_STOP
    j = Journal(jp, key=key, policy=POLICY_STOP)
    b = JournalBridge(m, j, k=3, budget_bytes=96, seed=0, shuffle_seed=1, mark=True)
    for i, f in enumerate(facts):
        b.write(f.statement, f.schema, now=100.0 + i, entity=f.entity)
    j.close() if hasattr(j, "close") else None
    r = session_b_from_disk(ck, jp, cfg, n_facts=8, n_negctrl=4, k=3, budget=96, window_len=96)
    ans = r["answers"]
    n = len(ans["on"])                                          # the planted facts session B found (admission by surprise)
    assert 0 < n <= 8 and len(ans["off"]) == n and len(ans["negctrl"]) == len(generate_facts(4, 10_000)[0])
    outcomes = [a["outcome"] for a in ans["on"]]
    assert all(o in ("strict", "valid", "invalid", "abstain") for o in outcomes)
    assert abs(outcomes.count("strict") / n - r["hm_recall_on"]) < 1e-9
    assert abs(outcomes.count("invalid") / n - r["hm_invalid_citation_on"]) < 1e-9
    assert abs(outcomes.count("abstain") / n - r["hm_false_abstention_on"]) < 1e-9
    import json; json.dumps(ans)                                 # serialisable as the file writes it
    for a in ans["on"]:
        assert set(a) >= {"entity", "schema", "attr", "kind", "label", "answered", "outcome", "own_in_window"}
        if a.get("mark_own") is not None:
            assert a["own_in_window"] is True and a["mark_own"] <= a["mark_max"] + 1e-9
    assert all(a["own_in_window"] is None for a in ans["off"])


def test_row_36_answers_file_recomputes_the_session_b_scalars():
    """The per line answers of v9 (session B re read on the Kaggle CPU, 2026-09-18) reproduce the
    committed session B's rates, and carry the three facts row 36 reads from them."""
    import json
    root = Path(__file__).resolve().parent.parent / "metrics" / "mqar"
    r = json.loads((root / "hm-lm-8ceba5db8d0b-session-b-answers.json").read_text())
    b = json.loads((root / "hm-lm-8ceba5db8d0b-session-b.json").read_text())
    on, neg = r["answers"]["on"], r["answers"]["negctrl"]
    n = len(on); assert n == 200 and len(neg) == 50
    outcomes = [a["outcome"] for a in on]
    for key, val in (("hm_recall_on", outcomes.count("strict") / n), ("hm_invalid_citation_on", outcomes.count("invalid") / n),
                     ("hm_false_abstention_on", outcomes.count("abstain") / n), ("hm_valid_citation_on", (outcomes.count("strict") + outcomes.count("valid")) / n),
                     ("hm_negctrl_rate", sum(1 for a in neg if a["outcome"] != "abstain") / 50)):
        assert abs(b[key] - val) < 1e-9 and abs(r[key] - val) < 1e-9, key
    inv = [a for a in on if a["outcome"] == "invalid"]
    assert len(inv) == 28 and sum(1 for a in inv if a.get("cited_pointer_is_own")) == 25
    ab = [a for a in on if a["outcome"] == "abstain"]
    assert len(ab) == 21 and sum(1 for a in ab if a["own_in_window"] is False) == 12
    claims = [a for a in neg if a["outcome"] != "abstain"]
    assert len(claims) == 5 and max(a["mark_max"] for a in claims) < r["hm_mark_oracle"]["threshold"]


def test_the_organ_side_policies_replay_on_the_v9_answers_as_declared_post_hoc():
    """ADR-008 amendment 2026-09-18: `mark-veto` and `mark-veto+value` on the recorded v9 answers
    (a post hoc reading on the seed they were conceived on, never a result): the numbers of row 36."""
    import json
    from cortex_c2b.hm_lm import replay_policy
    r = json.loads((Path(__file__).resolve().parent.parent / "metrics" / "mqar" / "hm-lm-8ceba5db8d0b-session-b-answers.json").read_text())
    v = replay_policy(r, "mark-veto")
    assert v["post_hoc"] and v["hm_negctrl_rate"] == 0.0 and abs(v["hm_invalid_citation_on"] - 0.125) < 1e-9 and v["gate_conditions_without_skill_arm"] == 0
    w = replay_policy(r, "mark-veto+value")
    assert abs(w["hm_recall_on"] - 0.875) < 1e-9 and w["hm_invalid_citation_on"] == 0.0 and w["hm_negctrl_rate"] == 0.0
    assert abs(w["hm_false_abstention_on"] - 0.125) < 1e-9 and w["gate_conditions_without_skill_arm"] == 1


def test_the_organ_side_policies_run_live_in_session_b_and_the_value_policy_never_cites_invalidly(tmp_path, monkeypatch):
    from cortex_c2b.hm_lm import session_b_from_disk
    from cortex_c2b.crypto import generate_key
    from cortex_c2b import Journal, POLICY_STOP
    key = generate_key(); monkeypatch.setenv("QUANTUM_CORTEX_JOURNAL_KEY", key.hex())
    cfg, m = tiny(journal_familiarity_mark=True)
    ck = tmp_path / "ckpt.pt"; jp = tmp_path / "journal.jsonl"
    torch.save({"model": m.state_dict(), "run_id": "tiny0002", "config_hash": train.config_hash(cfg)}, ck)
    facts, _ = generate_facts(8, 0)
    j = Journal(jp, key=key, policy=POLICY_STOP)
    b = JournalBridge(m, j, k=3, budget_bytes=96, seed=0, shuffle_seed=1, mark=True)
    for i, f in enumerate(facts):
        b.write(f.statement, f.schema, now=100.0 + i, entity=f.entity)
    for policy in ("mark-veto", "mark-veto+value"):
        r = session_b_from_disk(ck, jp, cfg, n_facts=8, n_negctrl=4, k=3, budget=96, window_len=96, policy=policy)
        assert r["policy"] == policy
        outs = [a["outcome"] for a in r["answers"]["on"]]
        if policy == "mark-veto+value":
            assert "valid" not in outs                               # the organ's value is exact or the line is not the own one
            assert all(a["kind"] != "cite" or a.get("mark_cited") is None or a["mark_cited"] >= r["hm_mark_oracle"]["threshold"] for a in r["answers"]["on"])


def test_the_post_hoc_system_reading_holds_across_the_veto_threshold_plateau():
    import json
    from cortex_c2b.hm_lm import replay_policy
    r = json.loads((Path(__file__).resolve().parent.parent / "metrics" / "mqar" / "hm-lm-8ceba5db8d0b-session-b-answers.json").read_text())
    for thr in (0.40, 0.536, 0.70, 0.95):
        w = replay_policy(r, "mark-veto+value", threshold=thr)
        assert w["gate_conditions_without_skill_arm"] == 1 and abs(w["hm_recall_on"] - 0.875) < 1e-9, thr
    assert replay_policy(r, "mark-veto+value", threshold=0.97)["hm_recall_on"] < 0.875           # the veto starts refusing own lines
    assert replay_policy(r, "mark-veto+value", threshold=0.20)["hm_negctrl_rate"] > 0.0          # a low threshold lets a claim through


def test_the_second_and_third_seed_configurations_of_v9_change_only_the_seed_and_the_provider():
    import json
    fields = train.Config.__dataclass_fields__
    root = Path(__file__).resolve().parent.parent / "configs"
    load = lambda n: json.load(open(root / n))
    v9 = load("kaggle_t4_journal_v9.json")
    for name, seed in (("local_journal_v9_seed2.json", 2024), ("kaggle_t4_journal_v9_seed2.json", 2024), ("local_journal_v9_seed3.json", 7), ("kaggle_t4_journal_v9_seed3.json", 7)):
        c = load(name)
        diff = {k for k in set(v9) | set(c) if v9.get(k) != c.get(k)}
        assert diff <= {"seed", "provider", "out_dir", "journal_path", "notes"}, (name, diff)
        assert c["seed"] == seed and c["journal_familiarity_mark"] is True
    h = lambda n: train.config_hash(train.Config(**{k: v for k, v in load(n).items() if k in fields}))
    assert h("local_journal_v9_seed2.json") == h("kaggle_t4_journal_v9_seed2.json") != h("kaggle_t4_journal_v9.json")
    assert h("local_journal_v9_seed3.json") == h("kaggle_t4_journal_v9_seed3.json") not in (h("kaggle_t4_journal_v9.json"), h("kaggle_t4_journal_v9_seed2.json"))


def test_the_fly_ranking_is_transparent_and_only_reorders_the_same_candidates():
    """ADR-008 amendment 2026-09-19 (ADR-003 line 6): the expand and sparsify tag is a
    ranking over the buckets every run has used, never a different index."""
    import numpy as np
    from cortex_c2b.read_path import CueIndex, fly_tag, fly_planes, FLY_SPARSITY, FLY_EXPAND
    rng = np.random.default_rng(0)
    vs = {f"e{i}": (lambda v: v / np.linalg.norm(v))(rng.standard_normal(64).astype("float32")) for i in range(120)}
    a, b = CueIndex(seed=1), CueIndex(seed=1, rank="fly")
    for eid, v in vs.items():
        a.add(eid, v); b.add(eid, v)
    assert a.rank == "cosine" and b.rank == "fly" and a.fly_planes is None
    for q in list(vs.values())[:12]:
        ca = {e for e, _ in a.query(q, k=len(a))}
        cb = {e for e, _ in b.query(q, k=len(b))}
        assert ca == cb                                        # the same candidates, a different order
    P = fly_planes(64, seed=3)
    assert P.shape == (FLY_EXPAND, 64) and set(np.unique(P)) <= {0.0, 1.0} and (P.sum(axis=1) == 6).all()
    t = fly_tag(vs["e0"], P)
    assert t.size == round(FLY_SPARSITY * FLY_EXPAND) and len(set(t.tolist())) == t.size
    assert (fly_tag(vs["e0"], P) == t).all()                   # deterministic
    with pytest.raises(ValueError):
        CueIndex(rank="barcode")


def test_the_retrieval_probe_separates_a_bucket_miss_from_a_ranking_miss():
    from cortex_c2b import Journal
    from cortex_c2b.hm_lm import retrieval_probe
    cfg, m = tiny()
    j = Journal()
    b = JournalBridge(m, j, k=3, budget_bytes=96, seed=0, shuffle_seed=1, mark=True)
    facts, _ = generate_facts(24, 0); neg, _ = generate_facts(8, 10_000)
    for i, f in enumerate(facts):
        b.write(f.statement, f.schema, now=100.0 + i, entity=f.entity)
    r = retrieval_probe(m, j, facts, neg, dict(b.pointer_of), k=3)
    assert set(r["rankings"]) == {"cosine", "fly"}
    for rank, v in r["rankings"].items():
        assert 0.0 <= v["own_in_top_k"] <= v["own_in_candidates"] <= 1.0, rank   # top k is a subset of the candidates
        assert v["own_rank"] is None or v["own_rank"]["min"] >= 1.0
    assert r["reading"]["candidate_misses_cosine"] == round(1 - r["rankings"]["cosine"]["own_in_candidates"], 4)


def test_the_oracle_leaves_the_window_shuffle_untouched_so_policies_are_paired():
    """Found on the seed 2024 files of 2026-09-20: the oracle read through the bridge and
    advanced its shuffle, so a run with a veto policy saw different windows from a plain
    run. The oracle now restores the generator state and the three policies are paired."""
    from cortex_c2b import Journal
    from cortex_c2b.hm_lm import mark_oracle
    from cortex_c2b.hm_protocol import generate_facts as gf
    cfg, m = tiny()
    j = Journal()
    b = JournalBridge(m, j, k=3, budget_bytes=96, seed=0, shuffle_seed=1, mark=True)
    facts, _ = gf(16, 0); neg, _ = gf(6, 10_000)
    for i, f in enumerate(facts):
        b.write(f.statement, f.schema, now=100.0 + i, entity=f.entity)
    cue = b.cues([encode_query(facts[0].query)])[0]
    before = [b.read(cue).window for _ in range(3)]
    b.rng = np.random.default_rng(1)                                   # same start as the bridge above
    mark_oracle(m, b, facts, neg, dict(b.pointer_of), train_n=8)
    after = [b.read(cue).window for _ in range(3)]
    assert before == after
