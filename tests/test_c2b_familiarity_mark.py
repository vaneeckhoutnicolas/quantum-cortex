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
