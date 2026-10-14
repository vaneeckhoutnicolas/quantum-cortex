"""Near duplicates at the write gate: the probe (ADR-007, amendment of 2026-10-13, evening; validated 2026-10-14, 11:05),
tests written before the founder's runs. Everything here runs on synthetic cues or a tiny untrained model: no number
below is a number of the record, and none is asserted as one; the tests pin the instrument's mechanics, the fidelity of
its writes to session A's, the fields the readings depend on, the determinism of a second process, and the readings'
rules on constructed files."""
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

import train                                                            # noqa: E402
from cortex_c2b import Journal, POLICY_STOP, content_hash               # noqa: E402
from cortex_c2b.crypto import generate_key                              # noqa: E402
from cortex_c2b.hm_protocol import generate_facts, _ATTRS               # noqa: E402
from cortex_c2b.lm_bridge import JournalBridge                          # noqa: E402
from cortex_c2b.read_path import ADOPTED_GEOMETRY, LEGACY_GEOMETRY      # noqa: E402
from cortex_c2b.write_path import WritePath, DentateGyrus               # noqa: E402
from cortex_c2b import near_duplicate_probe as P                        # noqa: E402
from cortex_c2b import near_duplicate_reading as R                      # noqa: E402
from cortex_c2b import hm_lm, record_tools                              # noqa: E402


def tiny(seed=9):
    cfg = train.Config(n_layer=3, n_head=2, n_embd=32, block_size=96, journal="read", journal_block=1,
                       journal_heads=2, journal_read_bytes=96, journal_cue_dim=64, journal_k=3)
    torch.manual_seed(seed)
    m = train.VanillaGPT(cfg); m.eval()
    return cfg, m


def checkpoint(tmp_path, cfg, model):
    ck = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict(), "run_id": "tiny", "config_hash": train.config_hash(cfg), "step": 0}, ck)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({k: getattr(cfg, k) for k in cfg.__dataclass_fields__}), encoding="utf-8")
    return ck, cfg_path


# ---------------------------------------------------------------- the geometry and the code fact
def test_pair_at_an_exact_cosine_before_rounding():
    rng = np.random.default_rng(3)
    for c in (0.0, 0.5, 0.95, 0.999):
        v = P.unit(rng.standard_normal(64)).astype(np.float32)
        w = P.pair_at(v, c, rng)
        assert w.dtype == np.float32 and abs(float(np.linalg.norm(w)) - 1.0) < 1e-6
        assert abs(P.cosine(v, w) - c) < 1e-6, (c, P.cosine(v, w))


def test_the_identical_cue_with_two_texts_and_two_contents_is_refused_by_the_gate():
    """The fact of the code that frames the declaration (checked on an in memory journal, 2026-10-13): at cosine one the
    failure F1 exists by construction; the same text with the same content is refused as the same address instead."""
    rng = np.random.default_rng(1)
    v = P.unit(rng.standard_normal(64)).astype(np.float32)
    j = Journal(policy=POLICY_STOP); wp = WritePath(j, seed=0)
    assert wp.write(v, b"Bako-3f2a works as a glazier", "person", now=0.0, source="Bako-3f2a works as a glazier").admitted
    r = wp.write(v, b"Bako-3f2a works as a cooper", "person", now=1.0, source="Bako-3f2a works as a cooper")
    # zero to seven decimals, not to the bit: 0.0 on one machine, 2.0e-8 on the founder's processor (the comparator's 1e-8 in the
    # denominator and the four decimal rounding of the stored cue); the fact is the refusal, the number is a float
    assert r.admitted is False and r.surprise < 1e-6 and r.reason == "redundant (below admission threshold)"
    r2 = wp.write(v, b"Bako-3f2a works as a glazier", "person", now=2.0, source="Bako-3f2a works as a glazier")
    assert r2.admitted is False and r2.reason == "redundant (same address, same content)"


def test_the_collision_law_and_the_band():
    assert P.expected_cobucket(1.0)["p_candidate"] == pytest.approx(1.0)
    ps = [P.expected_cobucket(c)["p_candidate"] for c in P.GRID]
    assert ps == sorted(ps) and ps[0] < ps[-1]
    assert P.expected_cobucket(0.0, n_bits=1, n_tables=1)["p_bit"] == pytest.approx(0.5)
    b = P.band({0.8: 0.0, 0.9: 0.1, 0.95: 0.6, 0.99: 1.0})
    assert b == {"c_any": 0.9, "c_all": 0.99, "c_half": 0.95}
    assert P.band({0.8: 0.0, 0.9: 0.0}) == {"c_any": None, "c_all": None, "c_half": None}


# ---------------------------------------------------------------- stage 1
def test_stage1_is_deterministic_and_reconciles_the_gate_with_its_own_surprise():
    d1 = P.stage1(grid=(0.85, 0.99), n_pairs=6)
    d2 = P.stage1(grid=(0.85, 0.99), n_pairs=6)
    assert P.numbers_sha256(d1) == P.numbers_sha256(d2)
    assert set(d1["per_c"]) == {"0.85", "0.99"} and d1["constants"]["dg"] == {"seed": 0, "in_dim": 64, "out_dim": 256, "k": 25}
    for r in d1["pairs"]:
        assert abs(r["cos_before_rounding"] - r["c"]) < 1e-6
        assert abs(r["gate_code_cosine"] - (1.0 - r["surprise"])) < 1e-5          # the gate's surprise is one minus the cosine of the two codes
        assert r["admitted"] == (r["surprise"] >= P.ADMISSION_THRESHOLD)
        assert 0.0 <= r["shared_winner_share"] <= 1.0
        for name in ("legacy", "adopted"):
            assert set(r["candidates"][name]) == {"second_for_first", "second_for_query", "first_for_query"}
        assert abs(r["query_cos_first"] - P.QUERY_COS) < 2e-4                      # the perturbed query's cosine, after the rounding of the stored cue
    pc = d1["per_c"]["0.99"]
    assert set(pc["cobucket"]) == {"legacy", "adopted"} and "expected_exact" in pc and set(d1["band"]) == {"c_any", "c_all", "c_half"}
    assert d1["geometries"]["legacy"]["planes_sha256"] == "680896ab07072738" and d1["geometries"]["adopted"]["planes_sha256"] == "e5703fddba43332f"


def test_stage1_second_process_hash_covers_the_numbers_alone():
    d = P.stage1(grid=(0.9,), n_pairs=3)
    h = P.numbers_sha256(d)
    d["timestamp_utc"] = "later"; d["versions"] = {"x": 1}; d["second_process"] = {"agrees": True}
    assert P.numbers_sha256(d) == h
    d["per_c"]["0.9"]["refused"] += 1
    assert P.numbers_sha256(d) != h


# ---------------------------------------------------------------- the plans
def test_e1_plan_is_seeded_and_never_the_original_attribute():
    facts, _ = generate_facts(20, 0)
    p1, p2 = P.e1_plan(facts, 0), P.e1_plan(facts, 0)
    assert p1 == p2 and all(p1[f.entity] in _ATTRS[f.schema] and p1[f.entity] != f.attr for f in facts)
    assert P.e1_plan(facts, 1) != p1
    f = facts[0]
    assert P.duplicate_statement(f, "E1", p1) == P.statement_of(f, p1[f.entity]) != f.statement


def test_e2_names_move_one_hexadecimal_character_and_are_decontaminated():
    assert P.bump_name("Bako-3f2a") == "Bako-3f2b" and P.bump_name("Bako-3f2f") == "Bako-3f20"
    with pytest.raises(ValueError):
        P.bump_name("Bako-3f2z")
    facts, _ = generate_facts(20, 0)
    plan = P.e2_plan(facts, set())
    assert all(v == P.bump_name(k) for k, v in plan.items())
    protected = {P.bump_name(facts[0].entity)}
    plan2 = P.e2_plan(facts, protected)
    assert plan2[facts[0].entity] is None and sum(v is None for v in plan2.values()) == 1
    assert P.duplicate_statement(facts[0], "E2", plan2) is None
    s = P.duplicate_statement(facts[1], "E2", plan2)
    assert s is not None and facts[1].attr in s and facts[1].entity not in s


def test_e0d_facts_come_from_the_declared_seed_and_avoid_the_protocol_names():
    facts, _ = generate_facts(200, 0); neg, _ = generate_facts(50, 10_000)
    protected = {f.entity for f in facts} | {f.entity for f in neg}
    extra = P.e0d_facts(200, P.E0D_SEED, protected)
    assert all(g.entity not in protected for g in extra) and P.E0D_SEED not in (0, 10_000, 100_000)
    own, _ = generate_facts(200, P.E0D_SEED)
    assert [g.entity for g in extra] == [g.entity for g in own if g.entity not in protected]


# ---------------------------------------------------------------- planting: fidelity to session A and the gate's record
def test_plant_arm_writes_exactly_as_session_a_and_records_the_gate(tmp_path):
    cfg, model = tiny()
    facts, _ = generate_facts(12, 0)
    # the reference: the bridge's own write, as session A plants (run_hm_lm)
    ref = Journal(tmp_path / "ref.jsonl", key=generate_key(), policy=POLICY_STOP, index_geometry=ADOPTED_GEOMETRY)
    bridge = JournalBridge(model, ref, k=3, budget_bytes=96, seed=0, shuffle_seed=1, index_geometry=ADOPTED_GEOMETRY)
    with torch.no_grad():
        for i, f in enumerate(facts):
            bridge.write(f.statement, f.schema, now=float(2 * i), entity=f.entity)
    rec = P.plant_arm(model, cfg, tmp_path / "b0.jsonl", generate_key(), "B0", seed=0, n_facts=12, n_negctrl=6)
    # the same entries in the same order with the same pointers and cues
    assert [w["pointer"] for w in rec["writes"]] == [bridge.pointer_of.get(f.entity) for f in facts]
    assert rec["counts"]["originals"] == len(facts) and rec["counts"]["entries"] == rec["counts"]["originals_admitted"]
    assert rec["index_geometry"]["n_tables"] == 16 and rec["index_geometry"]["signature"] == "exact"
    assert rec["journal_record"]["n_tables"] == 16                            # stamped at construction, before the first write
    for w in rec["writes"]:
        assert w["kind"] == "original" and "nearest_before" in w and "surprise" in w and "reason" in w
    # E1: every duplicate right after its original, with the gate's quantities against the partner, the protocol's pointers untouched
    rec1 = P.plant_arm(model, cfg, tmp_path / "e1.jsonl", generate_key(), "E1", seed=0, n_facts=12, n_negctrl=6)
    kinds = [w["kind"] for w in rec1["writes"]]
    assert kinds == ["original", "E1"] * len(facts)
    for w in rec1["writes"]:
        if w["kind"] == "E1":
            assert w["entity"] is None and w["for_entity"] in {f.entity for f in facts}
            assert set(w["partner"]) >= {"entity", "admitted", "nearest_is_partner"}
            if w["partner"]["admitted"]:
                assert set(w["partner"]) >= {"cos_raw", "cos_separated", "shared_winner_share", "pointer"}
                assert -1.0 <= w["partner"]["cos_separated"] <= 1.0 and 0.0 <= w["partner"]["shared_winner_share"] <= 1.0
            assert w["admitted"] == (w["surprise"] >= P.ADMISSION_THRESHOLD) or w["reason"].startswith("revised") or w["reason"].startswith("redundant (same")
    c = rec1["counts"]
    assert c["duplicates"] == len(facts) and c["duplicates_admitted"] + c["duplicates_refused"] == c["duplicates"]
    assert c["duplicates_refused_partner_nearest"] <= c["duplicates_refused"]
    assert R.f1_counts_from_writes(rec1)["agrees_with_file"] and R.f1_counts_from_writes(rec1)["duplicates"] == c["duplicates"]
    # the nearest stored code is the one CA3 would complete to: recomputed here as `reconstruct` does on the admitted codes
    for w in rec1["writes"]:
        nb = w["nearest_before"]
        if nb is not None:
            assert isinstance(nb["index"], int) and -1.0 <= nb["cosine"] <= 1.0 and nb["entry_id"] and nb["pointer"]
    assert set(rec1["pointer_of_duplicate"]) <= {f.entity for f in facts} and len(rec1["pointer_of_duplicate"]) == c["duplicates_admitted"]
    plan = P.e1_plan(facts, 0)
    for e, ptr in rec1["pointer_of_duplicate"].items():
        f = next(x for x in facts if x.entity == e)
        assert ptr == content_hash(P.statement_of(f, plan[e]).encode("utf-8"))
    # E0d: a distinct fact after each original, the counts of the density control
    rec0 = P.plant_arm(model, cfg, tmp_path / "e0d.jsonl", generate_key(), "E0d", seed=0, n_facts=12, n_negctrl=6)
    assert [w["kind"] for w in rec0["writes"]] == ["original", "E0d"] * len(facts)
    assert rec0["counts"]["e0d_written"] == len(facts) and rec0["counts"]["duplicates"] == 0 and len(rec0["e0d_entities"]) == len(facts)
    with pytest.raises(ValueError):
        P.plant_arm(model, cfg, tmp_path / "x.jsonl", generate_key(), "E9", seed=0, n_facts=12, n_negctrl=6)


def test_session_b_reads_an_arm_and_derives_the_provenance_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(record_tools, "git_state", lambda: ("test0000", True))
    cfg, model = tiny()
    ck, _ = checkpoint(tmp_path, cfg, model)
    key = generate_key(); jp = tmp_path / "e1.jsonl"
    rec = P.plant_arm(model, cfg, jp, key, "E1", seed=0, n_facts=12, n_negctrl=6)
    monkeypatch.setenv("QUANTUM_CORTEX_JOURNAL_KEY", key.hex())
    idx = {"n_tables": 16, "n_bits": 10, "signature": "exact"}
    r = hm_lm.session_b_from_disk(ck, jp, cfg, n_facts=12, n_negctrl=6, seed=0, k=3, budget=96, window_len=96,
                                  policy="mark-veto+value", index_override=idx, near_duplicate_arm="E1", planting_record=rec)
    nd = r["near_duplicates"]
    assert nd["arm"] == "E1" and nd["kind"] == "E1" and nd["planting"]["counts"] == rec["counts"]
    assert nd["pointer_of_duplicate"] == rec["pointer_of_duplicate"] and nd["duplicates_found"] == rec["counts"]["duplicates_admitted"]
    assert r["journal"]["unchanged"] is True and r["index"]["live"]["n_tables"] == 16 and r["journal"]["planted_found"] == rec["counts"]["originals_admitted"]
    thr = r["hm_mark_oracle"]["threshold"]
    for a in r["answers"]["on"]:
        assert set(a) >= {"own_cited", "cited_is_duplicate", "duplicate_in_window", "both_marks_at_or_above_threshold", "oracle_cites_own", "oracle_cites_duplicate"}
        assert a["own_cited"] == (a.get("cited_pointer_is_own") is True)
        dp = nd["pointer_of_duplicate"].get(a["entity"])
        assert a["cited_is_duplicate"] == (dp is not None and a.get("cited_pointer") == dp)
        assert a["duplicate_in_window"] == (dp is not None and dp in [w["pointer"] for w in a["window"]])
        marks = {w["pointer"]: w["mark"] for w in a["window"]}
        own_p = content_hash(next(f.statement for f in generate_facts(12, 0)[0] if f.entity == a["entity"]).encode("utf-8"))
        assert a["both_marks_at_or_above_threshold"] == (dp is not None and own_p in marks and dp in marks and marks[own_p] is not None
                                                          and marks[dp] is not None and marks[own_p] >= thr and marks[dp] >= thr)
    cnt = nd["counts"]
    assert cnt["duplicate_cited_given_both_present_n"] <= cnt["both_marks_at_or_above_threshold"]
    assert cnt["duplicate_cited_given_both_present"] is None or 0.0 <= cnt["duplicate_cited_given_both_present"] <= 1.0
    assert nd["consistent"] is True and nd["problems"] == [] and nd["code_commit_source"].startswith("forwarded")
    sc = R.side_counts(r); assert sc["agrees_with_file"], sc["differences"]
    assert cnt["own_cited"] == sum(a["own_cited"] for a in r["answers"]["on"]) and cnt["cited_is_duplicate"] == sum(a["cited_is_duplicate"] for a in r["answers"]["on"])
    assert cnt["oracle_cites_own"] == sum(a["oracle_cites_own"] for a in r["answers"]["on"])
    # B0: no duplicates, the block still names the arm; an unknown arm is refused
    key0 = generate_key(); jp0 = tmp_path / "b0.jsonl"
    rec0 = P.plant_arm(model, cfg, jp0, key0, "B0", seed=0, n_facts=12, n_negctrl=6)
    monkeypatch.setenv("QUANTUM_CORTEX_JOURNAL_KEY", key0.hex())
    r0 = hm_lm.session_b_from_disk(ck, jp0, cfg, n_facts=12, n_negctrl=6, seed=0, k=3, budget=96, window_len=96,
                                   policy="mark-veto+value", index_override=idx, near_duplicate_arm="B0", planting_record=rec0)
    assert r0["near_duplicates"]["kind"] is None and r0["near_duplicates"]["duplicates_found"] == 0 and r0["near_duplicates"]["consistent"]
    # a file not bound to its journal stops: another arm's planting record, or none
    r_bad = hm_lm.session_b_from_disk(ck, jp0, cfg, n_facts=12, n_negctrl=6, seed=0, k=3, budget=96, window_len=96,
                                      policy="mark-veto+value", index_override=idx, near_duplicate_arm="E1", planting_record=rec0)
    assert r_bad["near_duplicates"]["consistent"] is False and r_bad.get("stopped") and "not bound" in r_bad["stopped"][0]
    r_none = hm_lm.session_b_from_disk(ck, jp0, cfg, n_facts=12, n_negctrl=6, seed=0, k=3, budget=96, window_len=96,
                                       policy="mark-veto+value", index_override=idx, near_duplicate_arm="B0", planting_record=None)
    assert r_none["near_duplicates"]["problems"] == ["no planting record"] and r_none.get("stopped")
    with pytest.raises(ValueError):
        hm_lm.session_b_from_disk(ck, jp0, cfg, n_facts=12, n_negctrl=6, seed=0, k=3, budget=96, window_len=96,
                                  policy="mark-veto+value", index_override=idx, near_duplicate_arm="E9")
    # the pointers of the duplicates recomputed from the plan alone agree with the planting's
    facts, _ = generate_facts(12, 0); neg, _ = generate_facts(6, 10_000)
    protected = {f.entity for f in facts} | {f.entity for f in neg}
    j = Journal(jp, key=key, policy=POLICY_STOP)
    assert P.pointer_of_duplicate(facts, "E1", 0, protected, j.payloads) == rec["pointer_of_duplicate"]
    assert P.pointer_of_duplicate(facts, "B0", 0, protected, j.payloads) == {}


def test_session_b_file_name_carries_the_arm_and_keeps_the_record_names():
    idx = {"n_tables": 16, "n_bits": 10, "signature": "exact"}
    assert hm_lm.session_b_file_name("4cfd6f6e1406", "mark-veto+value", idx, control=False) == "hm-lm-4cfd6f6e1406-session-b-index16exact-mark-veto-value.json"
    assert hm_lm.session_b_file_name("4cfd6f6e1406", "plain", None, control=True) == "hm-lm-4cfd6f6e1406-session-b-control.json"
    assert hm_lm.session_b_file_name("4cfd6f6e1406", "mark-veto+value", idx, control=False, near_dup="E1") == "hm-lm-4cfd6f6e1406-near-dup-E1-session-b-index16exact-mark-veto-value.json"


@pytest.mark.slow
def test_near_duplicate_probe_mode_plants_reads_and_discards_in_new_processes(tmp_path):
    """`python -m cortex_c2b.hm_lm --near-duplicate-probe E2`: the arm planted, the existing session B run in a new process
    with the key by the environment, the journal discarded, the file retained under the declared name and never overwritten;
    the commit replaced in the outer process as the record's new process tests do."""
    cfg, model = tiny()
    ck, cfg_path = checkpoint(tmp_path, cfg, model)
    outdir = tmp_path / "out"; fresh = tmp_path / "fresh" / "j.jsonl"
    argv = ["--near-duplicate-probe", "E2", "--ckpt", str(ck), "--journal", str(fresh), "--config", str(cfg_path), "--out", str(outdir), "--n-facts", "12", "--n-negctrl", "6"]
    code = ("import sys; sys.path.insert(0, %r); from cortex_c2b import record_tools, hm_lm; "
            "record_tools.git_state = lambda: ('test0000', True); hm_lm._cli(%r)" % (str(ROOT), argv))
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=env, timeout=900)
    assert run.returncode == 0, run.stderr[-2000:]
    files = sorted(outdir.glob("*.json"))
    assert [f.name for f in files] == ["hm-lm-tiny-near-dup-E2-session-b-index16exact-mark-veto-value.json"]
    assert not fresh.exists() and not record_tools.payload_dir_of(fresh).exists() and not fresh.with_name("j.planting.json").exists()
    r = json.loads(files[0].read_text(encoding="utf-8"))
    assert r["near_duplicates"]["arm"] == "E2" and r["near_duplicates"]["planting"]["counts"]["duplicates"] == 10
    assert r["code_commit"] == "test0000" and r["near_duplicates"]["planting"]["code_commit"] == "test0000"
    # the fresh journal records the adopted geometry at construction, so the explicit --index agrees with it and overrides nothing
    assert r["policy"] == "mark-veto+value" and r["index"]["overridden"] is False and r["index"]["live"]["n_tables"] == 16
    assert r["index"]["record_source"] == "stamp" and r["index"]["record"]["signature"] == "exact"
    # never overwritten: the same command again is refused before anything is planted
    run2 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=env, timeout=900)
    assert run2.returncode != 0 and "never overwritten" in (run2.stderr + run2.stdout) and not fresh.exists()


# ---------------------------------------------------------------- stage 2: the precondition and the distributions
def _row45_like(model, facts):
    bridge = JournalBridge(model, Journal(policy=POLICY_STOP), k=3, budget_bytes=96, seed=0, index_geometry=ADOPTED_GEOMETRY)
    with torch.no_grad():
        s = np.stack([P.rounded(P.text_cue(bridge, f.statement)) for f in facts])
        q = np.stack([P.question_cue(bridge, f.query) for f in facts])
    qs = []
    for i in range(len(facts)):
        sims = np.array([P.cosine(q[i], s[j]) for j in range(len(facts))]); order = np.argsort(-sims, kind="stable")
        qs.append({"i": i, "entity": facts[i].entity, "cos_query_own": float(sims[i]),
                   "best_other_cosine": float(max(sims[j] for j in range(len(facts)) if j != i)), "rank_under_c6": int(np.where(order == i)[0][0]) + 1})
    return {"checkpoint": {"run_id": "tiny"}, "questions": qs}, q, s


def test_precondition_is_like_for_like_in_two_parts():
    cfg, model = tiny()
    facts, _ = generate_facts(12, 0)
    row45, q, s = _row45_like(model, facts)
    pre = P.precondition(q, s, row45["questions"])
    assert pre["order_exact"] and pre["within_tolerance"] and not pre["stopped"] and pre["max_abs_diff_cos_query_own"] == 0.0
    shifted = json.loads(json.dumps(row45)); shifted["questions"][0]["rank_under_c6"] += 1
    pre2 = P.precondition(q, s, shifted["questions"])
    assert pre2["stopped"] and not pre2["order_exact"] and pre2["order_mismatches"][0]["i"] == 0
    drift = json.loads(json.dumps(row45)); drift["questions"][1]["best_other_cosine"] += 5e-4
    pre3 = P.precondition(q, s, drift["questions"])
    assert not pre3["stopped"] and not pre3["within_tolerance"] and pre3["drift_exceeded_order_intact"]
    small = json.loads(json.dumps(row45)); small["questions"][1]["cos_query_own"] += 5e-5
    assert P.precondition(q, s, small["questions"])["within_tolerance"]
    with pytest.raises(ValueError):
        P.precondition(q, s, row45["questions"][:-1])


def test_cues_stage_records_the_three_distributions_and_the_plans(tmp_path):
    cfg, model = tiny()
    ck, _ = checkpoint(tmp_path, cfg, model)
    facts, _ = generate_facts(12, 0)
    row45, _, _ = _row45_like(model, facts)
    (tmp_path / "row45.json").write_text(json.dumps(row45), encoding="utf-8")
    out = P.cues_stage(ck, cfg, tmp_path / "row45.json", n_facts=12, n_negctrl=6, seed=0)
    n = len(facts)
    assert out["distributions"]["E1"]["n"] == 7 * n and out["distributions"]["E2"]["n"] == n and out["distributions"]["E0"]["n"] == n * (n - 1) // 2
    assert set(out["distributions"]["E1"]["above_grid"]) == {str(c) for c in P.GRID}
    assert out["precondition"]["order_exact"] and len(out["e1_pairs"]) == 7 * n and sum(r["planned"] for r in out["e1_pairs"]) == n
    assert out["plans"]["E1"] == P.e1_plan(facts, 0) and out["constants"]["precondition_tolerance"] == 1e-4
    other = json.loads(json.dumps(row45)); other["checkpoint"]["run_id"] = "another"
    (tmp_path / "other.json").write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(ValueError):
        P.cues_stage(ck, cfg, tmp_path / "other.json", n_facts=12, n_negctrl=6, seed=0)


# ---------------------------------------------------------------- the readings on constructed files
def _planting(arm, refused_partner=0, refused=0, dups=10, e0d_refused=0, n=10):
    """A planting record with its writes, the counts derived from them as the probe derives them."""
    writes = []
    for i in range(n):
        writes.append({"kind": "original", "for_entity": f"e{i}", "entity": f"e{i}", "admitted": True})
        if arm in ("E1", "E2") and i < dups:
            k = i
            is_refused = k < refused
            writes.append({"kind": arm, "for_entity": f"e{i}", "entity": None, "admitted": not is_refused,
                           "partner": {"entity": f"e{i}", "admitted": True, "nearest_is_partner": (is_refused and k < refused_partner)},
                           "nearest_before": {"entity": (f"e{i}" if k < refused_partner else "other")}})
        if arm == "E0d":
            writes.append({"kind": "E0d", "for_entity": f"x{i}", "entity": f"x{i}", "admitted": i >= e0d_refused})
    dl = [w for w in writes if w["kind"] in ("E1", "E2")]
    counts = {"originals": n, "originals_admitted": n, "duplicates": len(dl), "duplicates_admitted": sum(1 for w in dl if w["admitted"]),
              "duplicates_refused": sum(1 for w in dl if not w["admitted"]),
              "duplicates_refused_partner_nearest": sum(1 for w in dl if not w["admitted"] and w["partner"]["nearest_is_partner"]),
              "e0d_written": sum(1 for w in writes if w["kind"] == "E0d"), "e0d_refused": sum(1 for w in writes if w["kind"] == "E0d" and not w["admitted"]),
              "entries": sum(1 for w in writes if w["admitted"])}
    return {"arm": arm, "counts": counts, "writes": writes}


def _file(run_id, arm, own_cited_flags, policy="mark-veto+value", planting="auto", stopped=False, threshold=0.6, n_tables=16):
    if planting == "auto":
        planting = _planting(arm, dups=(len(own_cited_flags) if arm in ("E1", "E2") else 0), n=len(own_cited_flags))
    rows = [{"entity": f"e{i}", "cited_pointer_is_own": (True if x else None), "outcome": ("strict" if x else "abstain"), "kind": ("cite" if x else "unknown"),
             "own_in_window": True, "mark_max": 0.9} for i, x in enumerate(own_cited_flags)]
    f = {"checkpoint": {"run_id": run_id}, "policy": policy, "answers": {"on": rows, "negctrl": []},
         "index": {"live": {"n_tables": n_tables, "n_bits": 10, "signature": "exact"}}, "hm_mark_oracle": {"threshold": threshold},
         "near_duplicates": {"arm": arm, "planting": planting, "consistent": True, "problems": []}}
    if stopped:
        f["stopped"] = ["x"]
    return f


def test_holm_step_down_and_the_resolution_of_the_family():
    assert R.holm({"E1": 0.01, "E2": 0.04}) == {"E1": True, "E2": True}
    assert R.holm({"E1": 0.03, "E2": 0.04}) == {"E1": False, "E2": False}
    assert R.holm({"E1": 0.001, "E2": 0.2}) == {"E1": True, "E2": False}
    assert R.holm({}) == {}
    from cortex_c2b.adoption_reading import binomial_two_sided_p as pb
    # six discordant pairs one way reach 0.03125, under 0.05 for a single test and over 0.025 for the family's first step; seven reach 0.015625
    assert pb(0, 5) == pytest.approx(0.0625) and pb(0, 6) == pytest.approx(0.03125) and pb(0, 7) == pytest.approx(0.015625)
    assert R.holm({"E1": pb(0, 6), "E2": pb(0, 0)}) == {"E1": False, "E2": False}
    assert R.holm({"E1": pb(0, 6), "E2": pb(0, 6)}) == {"E1": False, "E2": False}
    assert R.holm({"E1": pb(0, 7), "E2": pb(0, 6)}) == {"E1": True, "E2": True}
    assert "seven" in R.RESOLUTION_SENTENCE and R.ALPHA == 0.05


def test_own_cited_pairs_and_the_two_primary_tests_read_the_direction():
    live = list(R.LIVE_SEEDS)
    # E1 loses eight own citations against E0d on each live seed, E2 gains eight: only E1 produces F2, E2 is named a gain
    files = {}
    for rid in live:
        ctl = [1] * 20 + [0] * 10
        files[rid] = {"E0d": _file(rid, "E0d", ctl), "B0": _file(rid, "B0", ctl),
                      "E1": _file(rid, "E1", [1] * 12 + [0] * 18), "E2": _file(rid, "E2", [1] * 28 + [0] * 2)}
    pc = R.own_cited_pairs(files[live[0]]["E0d"], files[live[0]]["E1"])
    assert pc["b_test_only"] == 0 and pc["c_control_only"] == 8 and pc["own_cited_control"] == 20 and pc["own_cited_test"] == 12
    f2 = R.read_f2(files)
    assert f2["complete"] and f2["tests"]["E1"]["c_control_only"] == 16 and f2["tests"]["E1"]["b_test_only"] == 0
    assert f2["tests"]["E1"]["direction"].startswith("fewer") and f2["holm_rejected"] == {"E1": True, "E2": True}
    assert f2["produced"] is True and f2["gains_named"] == ["E2"]
    # a gain alone does not produce F2
    for rid in live:
        files[rid]["E1"] = _file(rid, "E1", [1] * 20 + [0] * 10)
    f2b = R.read_f2(files)
    assert f2b["produced"] is False and f2b["gains_named"] == ["E2"] and f2b["tests"]["E1"]["direction"] == "equal"
    # a missing arm leaves its test unread; the conceiving seed is reported and never counted
    del files[live[1]]["E2"]
    f2c = R.read_f2(files)
    assert not f2c["complete"] and any(u[:2] == (live[1], "E2") for u in f2c["unread"]) and "E2" not in f2c["tests"] and "E1" in f2c["tests"]
    files["8ceba5db8d0b"] = {"E0d": _file("8ceba5db8d0b", "E0d", [1] * 30), "E1": _file("8ceba5db8d0b", "E1", [1] * 30)}
    f2d = R.read_f2(files)
    assert "8ceba5db8d0b:E1" in f2d["reported_conceiving_seed"] and f2d["tests"]["E1"]["n_pairs"] == 60 and "p" not in f2d["reported_conceiving_seed"]["8ceba5db8d0b:E1"]
    with pytest.raises(ValueError):
        R.own_cited_pairs(files[live[0]]["E0d"], _file("other", "E1", [1] * 30))
    with pytest.raises(ValueError):
        R.own_cited_pairs(files[live[0]]["E0d"], _file(live[0], "E1", [1] * 30, policy="plain"))
    # a file is refused as an arm it is not, under another policy, under another geometry, or without its planting
    with pytest.raises(ValueError):
        R.check_arm_file(_file(live[0], "E2", [1] * 30), "E1")
    with pytest.raises(ValueError):
        R.check_arm_file(_file(live[0], "E1", [1] * 30, policy="plain"), "E1")
    with pytest.raises(ValueError):
        R.check_arm_file(_file(live[0], "E1", [1] * 30, n_tables=4), "E1")
    with pytest.raises(ValueError):
        R.check_arm_file(_file(live[0], "E1", [1] * 30, planting=None), "E1")
    files[live[0]]["E2"] = _file(live[0], "E1", [1] * 30)                     # an E1 file under the E2 key: the test is unread, not misread
    f2e = R.read_f2(files)
    assert not f2e["complete"] and any(u[0] == live[0] and u[1] == "E2" for u in f2e["unread"])
    # the split names each discordant pair
    pc2 = R.own_cited_pairs(_file(live[0], "E0d", [1] * 20 + [0] * 10), _file(live[0], "E1", [1] * 12 + [0] * 18))
    assert sum(pc2["split"]["c"][k] for k in ("newly_present", "already_present", "lost", "absent_both")) == pc2["c_control_only"] == 8


def test_read_f1_recomputes_from_the_writes_and_needs_the_three_encoders():
    planting = {rid: {arm: _planting(arm, dups=(10 if arm in R.KINDS else 0)) for arm in R.ARMS} for rid in R.RUN_IDS}
    f1 = R.read_f1(planting)
    assert f1["produced"] is False and f1["complete"] and f1["defective"] == []
    planting[R.RUN_IDS[1]]["E2"] = _planting("E2", refused=2, refused_partner=1)
    f1b = R.read_f1(planting)
    assert f1b["produced"] is True and f1b["per_run"][R.RUN_IDS[1]]["E2"]["duplicates_refused_partner_nearest"] == 1
    planting[R.RUN_IDS[1]]["E2"] = _planting("E2", refused=2, refused_partner=0)       # refused, but against another entity: not F1
    assert R.read_f1(planting)["produced"] is False
    bad = _planting("E1", refused=1, refused_partner=1); bad["counts"]["duplicates_refused"] = 0      # a block that disagrees with its writes
    planting[R.RUN_IDS[0]]["E1"] = bad
    f1c = R.read_f1(planting)
    assert f1c["defective"] and f1c["defective"][0][:2] == (R.RUN_IDS[0], "E1")
    del planting[R.RUN_IDS[2]]["E1"]
    f1d = R.read_f1(planting)
    assert not f1d["complete"] and (R.RUN_IDS[2], "E1") in f1d["unread"]
    assert [R.letter(a, b) for a, b in ((True, False), (False, True), (False, False), (True, True))] == ["a", "b", "c", "d"]


def test_the_whole_reading_withholds_the_letter_on_every_exit():
    live = list(R.LIVE_SEEDS)
    files = {rid: {arm: _file(rid, arm, [1] * 20 + [0] * 10) for arm in R.ARMS} for rid in R.RUN_IDS}
    cues = {rid: {"precondition": {"stopped": False, "drift_exceeded_order_intact": False}} for rid in R.RUN_IDS}
    stage1 = {"band": {"c_any": 0.95, "c_all": 0.999, "c_half": 0.999}, "second_process": {"agrees": True}}
    record = {rid: _file(rid, "B0", [1] * 20 + [0] * 10, threshold=0.6) for rid in R.RUN_IDS}
    rd = R.read_near_duplicates(stage1, cues, files, record)
    assert rd["exits"] == [] and rd["letter"] == "c" and rd["F2"]["complete"] and rd["F1"]["complete"]
    assert all(v["within_tolerance"] and v["label"] == "reproduced" for v in rd["thresholds_against_record"].values()) and len(rd["thresholds_against_record"]) == 12
    assert all(rd["b0_against_record"][rid]["identical_outcomes"] for rid in R.RUN_IDS)
    assert any(l.startswith("letter: c") for l in R.summary_lines(rd))
    # the threshold label: a difference beyond 1e-6 is named, not a stop
    record[live[0]]["hm_mark_oracle"]["threshold"] = 0.6 + 5e-6
    rd_t = R.read_near_duplicates(stage1, cues, files, record)
    assert rd_t["thresholds_against_record"][f"{live[0]}:E1"]["label"] == "not reproduced" and rd_t["letter"] == "c"
    # a precondition stop, a missing cues file, a disagreeing second process, a stopped arm, a missing arm: no letter
    c2 = dict(cues); c2[live[0]] = {"precondition": {"stopped": True}}
    rd2 = R.read_near_duplicates(stage1, c2, files, record)
    assert rd2["letter"] is None and rd2["precondition_stopped"]
    c3 = {rid: cues[rid] for rid in live}                                           # 1337's cues file missing
    rd3 = R.read_near_duplicates(stage1, c3, files, record)
    assert rd3["letter"] is None and any("no cues file" in e for e in rd3["exits"])
    rd4 = R.read_near_duplicates({"band": {}, "second_process": {"agrees": False}}, cues, files, record)
    assert rd4["letter"] is None and any("second process" in e for e in rd4["exits"])
    f5 = {rid: dict(arms) for rid, arms in files.items()}; f5[live[1]]["E1"] = _file(live[1], "E1", [1] * 30, stopped=True)
    rd5 = R.read_near_duplicates(stage1, cues, f5, record)
    assert rd5["letter"] is None and any("arm E1 stopped" in e for e in rd5["exits"])
    f6 = {rid: dict(arms) for rid, arms in files.items()}; del f6["8ceba5db8d0b"]["E2"]    # the conceiving seed's arm missing: F1 incomplete
    rd6 = R.read_near_duplicates(stage1, cues, f6, record)
    assert rd6["letter"] is None and any("arm E2 missing" in e for e in rd6["exits"]) and not rd6["F1"]["complete"]
    # the drift is carried into the reading
    c7 = dict(cues); c7[live[0]] = {"precondition": {"stopped": False, "drift_exceeded_order_intact": True}}
    assert R.read_near_duplicates(stage1, c7, files, record)["drift"][live[0]] is True


def test_the_loader_finds_the_retained_files_by_their_names(tmp_path):
    rid = R.RUN_IDS[1]
    (tmp_path / "near-dup-stage1.json").write_text(json.dumps({"band": {}}), encoding="utf-8")
    (tmp_path / f"near-dup-cues-{rid}.json").write_text(json.dumps({"precondition": {"stopped": False}}), encoding="utf-8")
    (tmp_path / f"near-dup-cues-{R.RUN_IDS[0]}-FAILED-precondition.json").write_text(json.dumps({"precondition": {"stopped": True}}), encoding="utf-8")
    (tmp_path / f"hm-lm-{rid}-near-dup-E1-session-b-index16exact-mark-veto-value.json").write_text(json.dumps(_file(rid, "E1", [1])), encoding="utf-8")
    (tmp_path / f"hm-lm-{rid}-session-b-index16exact-mark-veto-value.json").write_text(json.dumps(_file(rid, "B0", [1])), encoding="utf-8")
    d = R.load_from_dir(tmp_path)
    assert d["stage1"] == {"band": {}} and set(d["cues_by_run"]) == {rid, R.RUN_IDS[0]} and d["cues_by_run"][R.RUN_IDS[0]]["precondition"]["stopped"]
    assert set(d["files_by_run"][rid]) == {"E1"} and d["files_by_run"][R.RUN_IDS[2]] == {} and set(d["record_files"]) == {rid}


def test_stage1_command_line_recomputes_in_a_second_process_and_never_overwrites(tmp_path, monkeypatch):
    """The declaration's second process test: the command line computes stage 1, a new process recomputes it from the same
    seeds and the file says they agree (a small grid here, recorded as not the declared constants)."""
    monkeypatch.setattr(record_tools, "git_state", lambda: ("test0000", True))
    P._cli(["--stage1", "--out", str(tmp_path), "--grid", "0.9,0.999", "--n-pairs", "4"])
    out = tmp_path / "near-dup-stage1.json"
    assert out.exists() and b"\r\n" not in out.read_bytes()
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["second_process"]["agrees"] is True and d["numbers_sha256"] == P.numbers_sha256(d) and d["code_commit"] == "test0000"
    assert d["declared_constants"] is False and d["constants"]["grid"] == [0.9, 0.999] and d["constants"]["n_pairs"] == 4
    with pytest.raises(SystemExit):
        P._cli(["--stage1", "--out", str(tmp_path), "--grid", "0.9,0.999", "--n-pairs", "4"])


def test_cues_command_line_names_the_file_and_stops_on_the_precondition(tmp_path, monkeypatch):
    monkeypatch.setattr(record_tools, "git_state", lambda: ("test0000", True))
    cfg, model = tiny()
    ck, cfg_path = checkpoint(tmp_path, cfg, model)
    fake = {"checkpoint": {"run_id": "tiny"}, "precondition": {"stopped": False, "order_exact": True, "within_tolerance": True, "per_question": []},
            "distributions": {"E1": {"n": 0}, "E2": {"n": 0}, "E0": {"n": 0}}, "plans": {"E2_dropped": []}}
    monkeypatch.setattr(P, "cues_stage", lambda *a, **k: json.loads(json.dumps(fake)))
    P._cli(["--cues", "--ckpt", str(ck), "--config", str(cfg_path), "--row45", str(cfg_path), "--out", str(tmp_path)])
    assert (tmp_path / "near-dup-cues-tiny.json").exists()
    with pytest.raises(SystemExit):                                                   # never overwritten, before any compute
        P._cli(["--cues", "--ckpt", str(ck), "--config", str(cfg_path), "--row45", str(cfg_path), "--out", str(tmp_path)])
    fake["precondition"]["stopped"] = True
    out2 = tmp_path / "second"; out2.mkdir()
    with pytest.raises(SystemExit):
        P._cli(["--cues", "--ckpt", str(ck), "--config", str(cfg_path), "--row45", str(cfg_path), "--out", str(out2)])
    assert (out2 / "near-dup-cues-tiny-FAILED-precondition.json").exists()
