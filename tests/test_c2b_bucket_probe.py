"""The bucket probe (ADR-008, amendment of 2026-10-12, evening, validated by the founder at 19:40;
RES-24, roadmap item 1): the probe of the index's buckets on the three v9 journals. Written before
the code, then extended by the committee's round on the build (the same evening). What is pinned: the
hyperplane tables are the live index's (the same draw, the first tables shared); the collapsed signature
is the live `_sig`; the control C0 is the live index's candidate set exactly; the subset relations
declared as properties (C1 within C0, C2, C3 and C4); C4 is the Hamming ball of radius one; C5 is the
published pseudo hash construction, centred per vector, ranked by the tag's overlap; C6 is the whole
store; the per question rows carry what the record test needs and the aggregates recompute from them;
the Hamming distances are the exact bits' for every hyperplane configuration; the budget, the paired
counts and the margin; the three readings, exhaustive, decided from the counts alone, with the protected
selection as a total order and a rotation that can fail; the miss classes per row and per aggregate;
the C0 precondition against row 38's file at the declared tolerance; the command line, its refusals
and one run end to end on a tiny checkpoint and a sealed journal. No model other than a tiny one is
loaded."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

import train
from cortex_c2b import Journal, POLICY_STOP
from cortex_c2b import bucket_probe as bp
from cortex_c2b.hm_protocol import generate_facts, N_FACTS, N_NEGCTRL
from cortex_c2b.lm_bridge import JournalBridge, encode_query
from cortex_c2b.read_path import CueIndex, fly_planes, fly_tag, CUE_DIM


def tiny(journal="read", seed=0, **kw):
    cfg = train.Config(n_layer=3, n_head=2, n_embd=32, block_size=96, journal=journal, journal_block=1,
                       journal_heads=2, journal_read_bytes=96, journal_cue_dim=64, journal_k=3, **kw)
    torch.manual_seed(seed)
    return cfg, train.VanillaGPT(cfg)


def _unit(rng, n):
    v = rng.standard_normal((n, CUE_DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def store():
    rng = np.random.default_rng(5)
    vecs = {f"e{i}": v for i, v in enumerate(_unit(rng, 200))}
    queries = _unit(rng, 60)
    near = np.stack([vecs[f"e{i}"] + 0.15 * rng.standard_normal(CUE_DIM).astype(np.float32) for i in range(60)])
    near = near / np.linalg.norm(near, axis=1, keepdims=True)
    return vecs, queries, near


# ============================================================ 1. the tables, the signatures, the control
def test_the_hyperplane_tables_are_the_live_index_s_and_the_first_tables_are_shared():
    live4, live16 = CueIndex(seed=0, n_tables=4), CueIndex(seed=0, n_tables=16)
    t4, t16 = bp.hyperplane_tables(CUE_DIM, 4, 0), bp.hyperplane_tables(CUE_DIM, 16, 0)
    assert len(t4) == 4 and len(t16) == 16
    assert all(np.array_equal(a, b) for a, b in zip(t4, live4.planes))
    assert all(np.array_equal(a, b) for a, b in zip(t16, live16.planes))
    assert all(np.array_equal(t16[t], t4[t]) for t in range(4))          # C2 nests C0's tables
    assert not np.array_equal(bp.hyperplane_tables(CUE_DIM, 4, 1)[0], t4[0])   # another index seed is another draw


def test_the_collapsed_signature_is_the_live_one_and_collapses_to_278_patterns():
    live = CueIndex(seed=0)
    rng = np.random.default_rng(1)
    for q in _unit(rng, 50):
        bits = (q @ live.planes[0]) > 0
        assert bp.collapsed_signature(bits) == live._sig(q, 0)
    exact, collapsed = set(), set()
    for pat in range(1024):
        bits = np.array([(pat >> i) & 1 for i in range(10)], dtype=bool)
        exact.add(bp.exact_signature(bits)); collapsed.add(bp.collapsed_signature(bits))
    assert len(exact) == 1024 and len(collapsed) == 278
    assert bp.exact_signature(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)) == 1


def test_the_control_c0_is_the_live_index_s_candidate_set_and_its_cost_is_the_touched_increment(store):
    vecs, queries, near = store
    live = CueIndex(seed=0)
    for eid, v in vecs.items():
        live.add(eid, v)
    c0 = bp.BucketIndex(vecs, bp.config("C0"), index_seed=0)
    d = c0.describe()
    assert d["family"] == "hyperplanes" and d["signature"] == "collapsed" and d["rank_by"] == "cosine"
    assert d["n_tables"] == 4 and d["n_bits"] == 10 and d["probe_radius"] == 0
    for q in np.concatenate([queries, near]):
        before = live.touched
        ids = {eid for eid, _ in live.query(q, k=len(live))}
        cands = c0.candidates(q)
        assert set(cands) == ids and len(cands) == len(set(cands)) == live.touched - before


def test_the_hamming_distance_is_the_exact_bits_for_every_hyperplane_configuration_the_collapsed_c0_included(store):
    vecs, queries, _ = store
    c0, c1 = bp.BucketIndex(vecs, bp.config("C0"), index_seed=0), bp.BucketIndex(vecs, bp.config("C1"), index_seed=0)
    q, v = queries[0], vecs["e3"]
    assert c0.bits(q) == c1.bits(q) == c1.signatures(q)                     # the ten bit integers, whatever the recorded key
    assert c0.signatures(q) != c0.bits(q)                                   # the recorded key of C0 stays the collapsed one
    d_exact = [bin(a ^ b).count("1") for a, b in zip(c1.signatures(q), c1.signatures(v))]
    assert [bin(a ^ b).count("1") for a, b in zip(c0.bits(q), c0.bits(v))] == d_exact
    assert all(0 <= d <= 10 for d in d_exact)


# ============================================================ 2. the configurations and their relations
def test_the_declared_configurations_are_the_amendment_s():
    names = [c["name"] for c in bp.CONFIGS]
    assert names == ["C0", "C1", "C2a", "C2b", "C3a", "C3b", "C4", "C5a8", "C5a16", "C5b8", "C5b16"]
    assert bp.config("C6")["family"] == "scan" and "C6" not in names                 # the reference, never a configuration
    assert bp.config("C1")["signature"] == "exact" and bp.config("C1")["n_tables"] == 4 and bp.config("C1")["n_bits"] == 10
    assert (bp.config("C2a")["n_tables"], bp.config("C2b")["n_tables"]) == (8, 16)
    assert (bp.config("C3a")["n_bits"], bp.config("C3b")["n_bits"]) == (8, 6)
    assert bp.config("C4")["probe_radius"] == 1 and bp.config("C4")["signature"] == "exact"
    assert bp.config("C5a8") == {"name": "C5a8", "family": "fly_pseudo", "m_blocks": 8, "probe_radius": 0}
    assert bp.config("C5b16") == {"name": "C5b16", "family": "fly_pseudo", "m_blocks": 16, "probe_radius": 1}
    assert bp.BUDGET == {"p95_at_most": 50, "max_below": 200, "mean_at_most_times_control": 4} and bp.MARGIN == 3
    assert bp.TIE_CANDIDATES == 1.0 and bp.CUE_FAR == 0.10 and bp.ROW38_TOLERANCE == 1e-6


def test_the_subset_relations_declared_as_properties_hold_per_query(store):
    vecs, queries, near = store
    idx = {n: bp.BucketIndex(vecs, bp.config(n), index_seed=0) for n in ("C0", "C1", "C2a", "C2b", "C3a", "C3b", "C4")}
    for q in np.concatenate([queries, near]):
        c = {n: set(i.candidates(q)) for n, i in idx.items()}
        assert c["C1"] <= c["C0"] and c["C1"] <= c["C2a"] <= c["C2b"] and c["C1"] <= c["C3a"] <= c["C3b"] and c["C1"] <= c["C4"]


def test_c3_uses_the_column_prefixes_of_c1_s_planes_not_a_new_draw():
    c1, c3a = bp.BucketIndex({}, bp.config("C1"), index_seed=0), bp.BucketIndex({}, bp.config("C3a"), index_seed=0)
    assert all(np.array_equal(a[:, :8], b) for a, b in zip(c1.tables, c3a.tables))
    assert c3a.describe()["planes_sha256"] != c1.describe()["planes_sha256"]


def test_c4_is_the_hamming_ball_of_radius_one_in_every_table(store):
    vecs, queries, _ = store
    c1, c4 = bp.BucketIndex(vecs, bp.config("C1"), index_seed=0), bp.BucketIndex(vecs, bp.config("C4"), index_seed=0)
    for q in queries[:20]:
        qs = c4.signatures(q)
        expected = set()
        for eid, v in vecs.items():
            es = c4.signatures(v)
            if any(bin(a ^ b).count("1") <= 1 for a, b in zip(qs, es)):
                expected.add(eid)
        assert set(c4.candidates(q)) == expected and set(c1.candidates(q)) <= expected
        assert len(qs) == 4


def test_c5_is_the_published_pseudo_hash_centred_per_vector_and_ranked_by_the_tag_s_overlap(store):
    vecs, queries, _ = store
    c5 = bp.BucketIndex(vecs, bp.config("C5a8"), index_seed=0)
    d = c5.describe()
    assert d["family"] == "fly_pseudo" and d["m_blocks"] == 8 and d["n_tables"] == 1 and d["fly_planes_sha256"] and d["rank_by"] == "overlap"
    P = fly_planes(CUE_DIM, seed=0)
    q = queries[0]
    y = P @ (q - q.mean())
    bits = [float(y[j * 128:(j + 1) * 128].sum()) >= 0.0 for j in range(8)]
    assert c5.signatures(q) == [sum(int(b) << j for j, b in enumerate(bits))]
    assert c5.signatures(q + 0.37) == c5.signatures(q)                      # a constant added to every coordinate: the same key
    ranked = c5.rank(q, c5.candidates(q))
    qt = fly_tag(q, P)
    for eid, score in ranked:
        assert score == pytest.approx(np.intersect1d(qt, fly_tag(vecs[eid], P)).size / qt.size)
    assert [s for _, s in ranked] == sorted([s for _, s in ranked], reverse=True)
    c5b = bp.BucketIndex(vecs, bp.config("C5b8"), index_seed=0)
    assert set(c5.candidates(q)) <= set(c5b.candidates(q))                  # radius one contains radius zero


def test_c6_is_the_whole_store_at_cost_n(store):
    vecs, queries, _ = store
    c6 = bp.BucketIndex(vecs, bp.config("C6"), index_seed=0)
    assert set(c6.candidates(queries[0])) == set(vecs) and len(c6.candidates(queries[0])) == 200
    assert c6.describe()["family"] == "scan"


# ============================================================ 3. the probe on a journal, the rows, the aggregates
@pytest.fixture(scope="module")
def probed():
    cfg, m = tiny()
    j = Journal()
    b = JournalBridge(m, j, k=3, budget_bytes=96, seed=0, shuffle_seed=1)
    facts, _ = generate_facts(24, 0); neg, _ = generate_facts(8, 10_000)
    for i, f in enumerate(facts):
        b.write(f.statement, f.schema, now=100.0 + i, entity=f.entity)
    pointer_of = dict(b.pointer_of)
    out = bp.probe_journal(m, j, facts, neg, pointer_of, k=3, index_seed=0)
    return out, j, facts, neg, pointer_of


def test_the_file_carries_the_live_parameters_the_questions_the_rows_and_the_query_cues(probed):
    out, j, facts, neg, pointer_of = probed
    assert out["probe"].startswith("the bucket probe") and out["k"] == 3 and out["index_seed"] == 0 and out["facts_seed"] == 0
    assert set(out["configurations"]) == {c["name"] for c in bp.CONFIGS} | {"C6"}
    for name, d in out["configurations"].items():
        assert d["name"] == name and "planes_sha256" in d and "family" in d and "rank_by" in d
    assert out["control_matches_live_index"] is True
    assert out["budget"] == bp.BUDGET and out["margin"] == bp.MARGIN and out["cue_far"] == bp.CUE_FAR
    assert len(out["query_cues"]["questions"]) == len(facts) == 20 and len(out["query_cues"]["questions"][0]) == CUE_DIM
    assert len(out["query_cues"]["never_planted"]) == len(neg) == 5          # the generator drops its collisions: 20 of 24, 5 of 8
    assert out["n_facts"] == 20 and out["n_negctrl"] == 5
    qs = out["questions"]                                                    # the per question constants, once
    assert len(qs) == 20 and [q["i"] for q in qs] == list(range(20))
    for key in ("i", "entity", "own_pointer", "own_id", "planted", "cos_query_own", "best_other_cosine", "rank_under_c6"):
        assert key in qs[0], key
    assert all(q["planted"] == (q["own_id"] is not None) for q in qs) and sum(q["planted"] for q in qs) == len(pointer_of)
    rows = out["rows"]["C0"]
    assert len(rows) == 20
    r = rows[0]
    for key in ("i", "planted", "in_candidates", "rank", "n_candidates", "margin", "signatures_query",
                "signatures_own", "hamming", "miss_class"):
        assert key in r, key
    for key in ("entity", "own_pointer", "cos_query_own", "best_other_cosine", "rank_under_c6", "candidate_ids"):
        assert key not in r, key                                             # in `questions`, or not retained
    assert r["in_candidates"] == (r["rank"] is not None) and "rank_cosine" not in r      # the rank is under the configuration's `rank_by`
    assert all(0.0 <= nr["novelty"] <= 1.0 for nr in out["never_planted"]["C0"])
    assert "fly_rank_on_c0" in rows[0] and "fly_overlap_own" in rows[0] and "fly_top_overlap" in rows[0]
    assert "fly_rank_on_c0" not in out["rows"]["C1"][0]
    assert out["versions"]["numpy"] == np.__version__ and "torch" in out["versions"]
    assert out["journal"]["entries"] == len(j) and out["journal"]["duplicate_pointers"] == 0
    sigs = out["entry_signatures"]
    assert set(sigs) == set(out["configurations"]) and sigs["C6"] == {} and len(sigs["C0"]) == out["journal"]["entries"]
    assert all(len(s) == 4 for s in sigs["C1"].values()) and all(len(s) == 16 for s in sigs["C2b"].values())
    assert all(len(s) == 1 for s in sigs["C5a8"].values())


def test_the_query_cues_are_encoded_one_question_at_a_time_as_row_38_did(probed):
    """The first run of 2026-10-12 on the seed 2024 journal stopped on its precondition: the questions were
    encoded in batches of 200 and 50, the float32 kernels differ by batch shape, and the maximum never planted
    score moved by 1.2e-6. The file's cues must equal the single question encoding of `retrieval_probe`, exactly.
    This pins the corrected code's path on the fixture and this machine (under the batched code, 20 questions of 20
    differ here at the ninth digit); it pins nothing about the founder's machine or the trained checkpoint."""
    out, j, facts, neg, pointer_of = probed
    cfg, m = tiny()
    b = JournalBridge(m, j, k=3, budget_bytes=96, seed=0, shuffle_seed=1)
    for i, f in enumerate(facts):
        single = [float("%.9g" % x) for x in b.cues([encode_query(f.query)])[0]]
        assert out["query_cues"]["questions"][i] == single, i
    for j_, f in enumerate(neg):
        single = [float("%.9g" % x) for x in b.cues([encode_query(f.query)])[0]]
        assert out["query_cues"]["never_planted"][j_] == single, j_


def test_every_candidate_set_and_cost_recompute_from_the_file_alone(probed):
    """The entry signatures per configuration and the query signatures per row make the candidate sets a
    function of the file (the ranks need the stored cues, which are the sealed journal's)."""
    out = probed[0]
    for name, cfg in [(c["name"], c) for c in bp.CONFIGS]:
        sigs = out["entry_signatures"][name]
        n_bits = cfg["n_bits"] if cfg["family"] == "hyperplanes" else cfg["m_blocks"]
        for row, q in zip(out["rows"][name], out["questions"]):
            cands = set()
            for t, s in enumerate(row["signatures_query"]):
                keys = {s} | ({s ^ (1 << b) for b in range(n_bits)} if cfg["probe_radius"] == 1 else set())
                cands |= {eid for eid, es in sigs.items() if es[t] in keys}
            assert len(cands) == row["n_candidates"], (name, row["i"])
            assert row["in_candidates"] == (q["own_id"] in cands), (name, row["i"])
    assert all(r["n_candidates"] == out["journal"]["entries"] for r in out["rows"]["C6"])


def test_the_aggregates_recompute_from_the_rows_and_the_subset_relations_hold_on_the_journal(probed):
    out = probed[0]
    planted = [q["planted"] for q in out["questions"]]
    for name, agg in out["aggregates"].items():
        rows, nrows = out["rows"][name], out["never_planted"][name]
        rec = bp.aggregate(rows, nrows, out["rows"]["C0"], n_entries=out["journal"]["entries"], k=out["k"])   # the file's k
        for key, v in rec.items():
            assert agg[key] == pytest.approx(v) if isinstance(v, float) else agg[key] == v, (name, key)
        prow = [r for r, p in zip(rows, planted) if p]
        assert agg["n_planted"] == len(prow) and agg["candidate_recall"] == pytest.approx(sum(r["in_candidates"] for r in prow) / len(prow))
        assert agg["top_k_recall"] <= agg["candidate_recall"]
        assert agg["misses"] == [r["i"] for r in prow if not r["in_candidates"]]
        assert sum(agg["miss_classes"].values()) == len(agg["misses"]) and set(agg["miss_classes"]) == {"index", "interference", "cue"}
        assert all(r["miss_class"] is None for r in rows if r["in_candidates"] or not r["planted"])
    a = out["aggregates"]
    assert a["C0"]["R"] == 0 and a["C0"]["L"] == 0 and a["C6"]["candidate_recall"] == 1.0 and a["C6"]["cost"]["max"] == out["journal"]["entries"]
    assert a["C0"]["admissible"] is True and a["C6"]["admissible"] is False
    assert out["subset_relations"] == {"C1_in_C0": True, "C1_in_C2a": True, "C2a_in_C2b": True, "C1_in_C3a": True,
                                       "C3a_in_C3b": True, "C1_in_C4": True}                    # checked query by query at run time


def test_the_budget_is_read_on_the_95th_percentile_the_maximum_and_the_control_s_mean_over_the_questions():
    def rows(costs):
        return [{"n_candidates": c, "in_candidates": True, "rank": 1} for c in costs]
    ctrl = rows([4] * 20)
    cagg = bp.aggregate(ctrl, rows([3] * 5), ctrl, 200)
    assert bp.admissible(bp.aggregate(rows([10] * 19 + [49]), rows([3] * 5), ctrl, n_entries=200), cagg)
    tail = bp.aggregate(rows([4] * 19 + [200]), rows([3] * 5), ctrl, 200)              # a scan one read in twenty
    assert not bp.admissible(tail, cagg)
    heavy = bp.aggregate(rows([30] * 20), rows([3] * 5), ctrl, 200)                     # p95 under 50 but the mean is 7.5 times the control's
    assert not bp.admissible(heavy, cagg)
    never = bp.aggregate(rows([4] * 20), rows([120] * 5), ctrl, 200)                    # the never planted reads count too: p95 and max
    assert not bp.admissible(never, cagg)
    exact = bp.aggregate(rows([16] * 20), rows([3] * 5), ctrl, 200)                     # the mean at exactly four times passes
    assert bp.admissible(exact, cagg)
    assert bp.aggregate(rows([1, 2, 3, 4]), [], ctrl[:4], 200)["cost"]["p95"] == pytest.approx(np.percentile([1, 2, 3, 4], 95))   # numpy's linear percentile


# ============================================================ 4. the readings, exhaustive, from the counts alone
def _journal(recall_by_cfg: dict, n=200, costs=None, c0_misses=13, p95=None):
    """A synthetic journal summary: per configuration the misses as a set of question ids; every declared
    configuration not named equals the control (R 0, L 0); p95 per configuration, 8.0 by default."""
    costs, p95 = costs or {}, p95 or {}
    c0 = set(range(c0_misses))
    aggs = {}
    for c in bp.CONFIGS:
        name = c["name"]
        misses = set(recall_by_cfg.get(name, c0)) if name != "C0" else c0
        aggs[name] = {"candidate_recall": 1 - len(misses) / n, "misses": sorted(misses),
                      "R": len(c0 - misses) if name != "C0" else 0, "L": len(misses - c0) if name != "C0" else 0,
                      "admissible": costs.get(name, True),
                      "cost": {"p95": p95.get(name, 8.0), "max": 20, "mean": 5.0 if name != "C0" else 4.0}}
    return {"aggregates": aggs, "journal": {"entries": n}}


def test_reading_a_every_own_line_proposed_on_each_journal_with_the_margin_and_a_protected_selection():
    js = [_journal({"C1": range(13), "C2a": [], "C2b": [], "C4": []}, p95={"C2b": 30.0, "C4": 12.0, "C2a": 12.5}) for _ in range(3)]
    r = bp.read_three(js)                                                   # C2a within one candidate of C4: the tie goes to fewer tables
    assert r["letter"] == "a" and r["candidates_at_1000"] == ["C2a", "C2b", "C4"]
    assert r["selected"] == "C4" and r["rotations"] == [{"left_out": i, "chosen": "C4", "holds_on_left_out": True} for i in range(3)]
    assert r["rotations_failed"] == 0 and r["excluded_by_rotation"] == []
    assert r["selected_label"] == ("selected among 10 configurations on the questions of 3 journals, 0 rotation(s) failed; "
                                   "confirmed on the first journal sealed after the adoption (the prospective holdout)")
    assert r["adopt_c1"] is False and r["c1_loses_nothing"] is True


def test_the_rotation_can_fail_and_a_configuration_that_fails_it_is_excluded():
    """The committee's case (2026-10-12): X at 1.000 on two journals at a low cost and 0.995 on the third, Y at
    1.000 everywhere at a higher cost. The pair (0, 1) chooses X by the words, X fails on journal 2, is excluded,
    and the selection reruns: Y on the three rotations, one rotation failed, said in the label."""
    js = [_journal({"C2a": [], "C2b": []}, p95={"C2a": 10.0, "C2b": 30.0}),
          _journal({"C2a": [], "C2b": []}, p95={"C2a": 10.0, "C2b": 30.0}),
          _journal({"C2a": [7], "C2b": []}, p95={"C2a": 10.0, "C2b": 30.0})]
    r = bp.read_three(js)
    assert r["letter"] == "a" and r["candidates_at_1000"] == ["C2b"]
    assert r["rotations_failed"] == 1 and r["excluded_by_rotation"] == ["C2a"]
    assert r["selected"] == "C2b" and all(rot["chosen"] == "C2b" and rot["holds_on_left_out"] for rot in r["rotations"])
    assert r["selected_label"].startswith("selected among 10 configurations on the questions of 3 journals, 1 rotation(s) failed")


def test_a_disagreement_between_rotations_breaks_toward_more_tables_then_the_later_declared_name():
    """Three journals, three configurations at 1.000 everywhere whose pair costs differ by journal, so that each
    rotation chooses another one; the words break toward more tables (C2a over C4 and C1), the later declared
    name among equal table counts."""
    costs = [{"C1": 9.0, "C4": 1.0, "C2a": 1.0}, {"C1": 1.0, "C4": 9.0, "C2a": 1.0}, {"C1": 1.0, "C4": 1.0, "C2a": 9.0}]   # each dear on one journal
    js = [_journal({"C1": [], "C4": [], "C2a": []}, p95=c) for c in costs]
    r = bp.read_three(js)
    assert r["letter"] == "a" and r["rotations_failed"] == 0
    assert [rot["chosen"] for rot in r["rotations"]] == ["C1", "C4", "C2a"] and r["selected"] == "C2a"
    js2 = [_journal({"C1": [], "C4": []}, p95=c) for c in [{"C1": 9.0, "C4": 1.0}, {"C1": 1.0, "C4": 9.0}, {"C1": 1.0, "C4": 1.0}]]
    r2 = bp.read_three(js2)
    assert [rot["chosen"] for rot in r2["rotations"]] == ["C1", "C4", "C1"] and r2["selected"] == "C4"   # equal tables: the later declared name


def test_the_selection_is_a_total_order_independent_of_the_names_order():
    """The tie set is everything within one candidate of the lowest pair mean p95; within it fewer tables, then
    the exact signature, then the lower cost, then the name; the fly counts as one table."""
    js = [_journal({}, p95={"C1": 11.6, "C2a": 10.8, "C2b": 10.0}) for _ in range(3)]
    assert bp._select(["C1", "C2a", "C2b"], js, [0, 1]) == "C2a" == bp._select(["C2b", "C2a", "C1"], js, [0, 1])
    js2 = [_journal({}, p95={"C2b": 10.0, "C2a": 10.9, "C4": 11.8}) for _ in range(3)]
    assert bp._select(["C2b", "C2a", "C4"], js2, [0, 1]) == "C2a" == bp._select(["C4", "C2a", "C2b"], js2, [0, 1])   # C4 is 1.8 above the minimum
    js3 = [_journal({}, p95={"C1": 12.0, "C5a8": 12.9}) for _ in range(3)]
    assert bp._select(["C1", "C5a8"], js3, [0, 1]) == "C5a8"                 # one table beats four within the tie window
    js4 = [_journal({}, p95={"C1": 12.0, "C3a": 12.0}) for _ in range(3)]
    assert bp._select(["C3a", "C1"], js4, [0, 1]) == "C1"                     # equal in every key but the name: the earlier name
    js5 = [_journal({}, p95={"C5a8": 12.0, "C5a16": 12.0}) for _ in range(3)]
    assert bp._select(["C5a16", "C5a8"], js5, [0, 1]) == "C5a8"               # the declared order, not the lexical one
    assert bp._order("C5a8") < bp._order("C5a16") and bp._order("C0") == 0
    assert bp._tables_of("C5b16") == 1 and bp._tables_of("C2b") == 16 and bp._tables_of("C6") == 0


def test_reading_b_above_the_control_everywhere_but_not_every_line_and_the_mixed_case_is_not_a():
    js = [_journal({"C2a": range(3), "C4": range(13)}), _journal({"C2a": range(2), "C4": range(13)}), _journal({"C2a": [5], "C4": range(13)})]
    r = bp.read_three(js)
    assert r["letter"] == "b" and r["above_control_everywhere"] == ["C2a"] and r["selected"] == "C2a"
    assert r["rotations_failed"] == 0 and all(rot["holds_on_left_out"] for rot in r["rotations"])
    mixed = [_journal({"C2a": []}), _journal({"C2a": []}), _journal({"C2a": range(13)})]       # 1.000 on two journals, the control's on the third
    assert bp.read_three(mixed)["letter"] == "c"
    # under (b) the pair's pool is the configurations above the control on the pair with the most hits there
    js2 = [_journal({"C2a": range(3), "C4": range(5)}), _journal({"C2a": range(3), "C4": range(5)}), _journal({"C2a": range(3), "C4": range(5)})]
    r2 = bp.read_three(js2)
    assert r2["letter"] == "b" and r2["above_control_everywhere"] == ["C2a", "C4"] and r2["selected"] == "C2a"


def test_reading_c_nothing_clears_the_margin_and_c1_is_adopted_only_if_it_loses_no_hit():
    noise = [_journal({"C1": range(13), "C2a": range(11), "C4": range(12)}) for _ in range(3)]   # R of 2 and 1: under the margin
    r = bp.read_three(noise)
    assert r["letter"] == "c" and r["above_control_everywhere"] == [] and r["c1_loses_nothing"] is True and r["adopt_c1"] is True
    assert r["selected"] is None and r["rotations"] == []
    loses = [_journal({"C1": list(range(13)) + [40]}) for _ in range(3)]
    r2 = bp.read_three(loses)
    assert r2["letter"] == "c" and r2["c1_loses_nothing"] is False and r2["adopt_c1"] is False
    inadmissible = [_journal({"C1": range(13)}, costs={"C1": False}) for _ in range(3)]      # loses nothing, but inadmissible: not adopted
    r3 = bp.read_three(inadmissible)
    assert r3["letter"] == "c" and r3["c1_loses_nothing"] is True and r3["adopt_c1"] is False


def test_a_journal_where_the_control_proposes_every_line_counts_as_not_below():
    js = [_journal({"C2a": []}, c0_misses=0), _journal({"C2a": []}), _journal({"C2a": []})]     # C0 at 200 of 200 on the first
    r = bp.read_three(js)
    assert r["letter"] == "a" and r["selected"] == "C2a"


def test_inadmissible_configurations_never_enter_a_reading_and_a_file_missing_a_configuration_is_refused():
    js = [_journal({"C2b": []}, costs={"C2b": False}) for _ in range(3)]
    assert bp.read_three(js)["letter"] == "c"
    short = [_journal({"C2b": []}) for _ in range(3)]
    del short[1]["aggregates"]["C5b16"]
    with pytest.raises(ValueError, match="C5b16"):
        bp.read_three(short)


def test_the_miss_classes_follow_the_declared_geometry():
    assert bp.miss_class({"in_candidates": False, "planted": True, "rank_under_c6": 1, "cos_query_own": 0.6, "best_other_cosine": 0.3}) == "index"
    assert bp.miss_class({"in_candidates": False, "planted": True, "rank_under_c6": 3, "cos_query_own": 0.6, "best_other_cosine": 0.7}) == "interference"
    assert bp.miss_class({"in_candidates": False, "planted": True, "rank_under_c6": 1, "cos_query_own": 0.8, "best_other_cosine": 0.8}) == "interference"   # a duplicate at equal cosine, whatever the id order
    assert bp.miss_class({"in_candidates": False, "planted": True, "rank_under_c6": 1, "cos_query_own": 0.1, "best_other_cosine": 0.05}) == "cue"
    assert bp.miss_class({"in_candidates": True, "planted": True, "rank_under_c6": 1, "cos_query_own": 0.9, "best_other_cosine": 0.1}) is None
    assert bp.miss_class({"in_candidates": False, "planted": False, "rank_under_c6": None, "cos_query_own": None, "best_other_cosine": 0.2}) is None
    # the collision law: a per hyperplane agreement of 1 minus theta over pi; C4's chance falls under CUE_FAR near a cosine of 0.18
    assert bp.proposal_probability(1.0, 10, 4, 1) == pytest.approx(1.0) and bp.proposal_probability(0.0, 10, 4, 0) == pytest.approx(1 - (1 - 2 ** -10) ** 4)
    assert bp.proposal_probability(0.19, 10, 4, 1) > bp.CUE_FAR > bp.proposal_probability(0.17, 10, 4, 1)


# ============================================================ 5. the precondition against row 38 and the command line
def test_the_control_s_precondition_names_every_scalar_of_row_38_s_file_at_the_declared_tolerance():
    row38 = json.loads((bp.REPO_ROOT / "metrics/mqar/retrieval-probe-4cfd6f6e1406.json").read_text(encoding="utf-8"))
    c = row38["rankings"]["cosine"]
    good = {"candidate_recall": 0.935, "top_k_recall": 0.935, "n_in_candidates": 187, "n_margins": 183,
            "margin_mean": c["own_minus_best_other"]["mean"], "n_negctrl_scored": 49, "negctrl_best_max": c["negctrl_best_score"]["max"]}
    ok, diffs = bp.check_row38(good, row38, k=4, n_facts=200, n_negctrl=50)
    assert ok and diffs == []
    ok, diffs = bp.check_row38(dict(good, n_margins=184), row38)
    assert not ok and diffs == ["n_margins: 184 != 183"]
    ok, _ = bp.check_row38(dict(good, margin_mean=good["margin_mean"] + 1e-8), row38)        # a float32 ulp of arithmetic order
    assert ok
    ok, diffs = bp.check_row38(dict(good, margin_mean=good["margin_mean"] + 1e-5), row38)
    assert not ok and diffs[0].startswith("margin_mean:")
    ok, diffs = bp.check_row38(dict(good, candidate_recall=0.93), row38)
    assert not ok and diffs == ["candidate_recall: 0.93 != 0.935"]
    ok, diffs = bp.check_row38(good, row38, k=3, n_facts=200, n_negctrl=40)
    assert not ok and diffs == ["k: 3 != 4", "n_negctrl: 40 != 50"]


def _subprocess_cli(args: list[str]):
    import subprocess, sys
    code = ("import sys; sys.path.insert(0, %r); from cortex_c2b import bucket_probe as bp; bp._cli(%r)" % (str(bp.REPO_ROOT), args))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_the_command_line_refuses_another_facts_seed_other_counts_and_an_existing_file_of_the_record(tmp_path):
    kept = tmp_path / "bucket-probe-x.json"; kept.write_text("{}\n", encoding="utf-8")
    base = ["--ckpt", "c", "--journal", "j", "--config", "k", "--seed", "0", "--index-seed", "0", "--out", str(kept)]
    r = _subprocess_cli(base)
    assert r.returncode != 0 and "never overwritten" in (r.stderr + r.stdout) and kept.read_text(encoding="utf-8") == "{}\n"
    kept.unlink()
    failed = tmp_path / "bucket-probe-x-FAILED-precondition.json"; failed.write_text("{}\n", encoding="utf-8")
    r = _subprocess_cli(base)
    assert r.returncode != 0 and "never overwritten" in (r.stderr + r.stdout)        # the failure file is a file of the record too
    failed.unlink()
    r2 = _subprocess_cli([a if a != "0" or i != 7 else "1" for i, a in enumerate(base)])   # --seed 1
    assert r2.returncode != 0 and "seed 0" in (r2.stderr + r2.stdout)
    r3 = _subprocess_cli(base + ["--n-facts", "100"])
    assert r3.returncode != 0 and "not a lever" in (r3.stderr + r3.stdout)
    r4 = _subprocess_cli(base + ["--n-negctrl", "10"])
    assert r4.returncode != 0 and "not a lever" in (r4.stderr + r4.stdout)


def test_the_command_line_refuses_a_journal_path_that_is_not_a_file_and_a_missing_key_before_loading_anything(tmp_path, monkeypatch):
    out = tmp_path / "o.json"
    r = _subprocess_cli(["--ckpt", "c", "--journal", str(tmp_path / "nope" / "journal.jsonl"), "--config", "k", "--out", str(out)])
    assert r.returncode != 0 and "is not a file" in (r.stderr + r.stdout) and not (tmp_path / "nope").exists()   # nothing created
    log = tmp_path / "journal.jsonl"; log.write_text("QJE1:x\n", encoding="utf-8")
    r = _subprocess_cli(["--ckpt", "c", "--journal", str(log), "--config", "k", "--out", str(out)])
    assert r.returncode != 0 and "is not a directory" in (r.stderr + r.stdout)                        # the payload store next to the log
    (tmp_path / "journal.payloads").mkdir()
    import subprocess, sys
    code = ("import os, sys; os.environ.pop('QUANTUM_CORTEX_JOURNAL_KEY', None); sys.path.insert(0, %r); "
            "from cortex_c2b import bucket_probe as bp; bp._cli(%r)" % (str(bp.REPO_ROOT), ["--ckpt", "c", "--journal", str(log), "--config", "k", "--out", str(out)]))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode != 0 and "QUANTUM_CORTEX_JOURNAL_KEY" in (r.stderr + r.stdout) and not out.exists()


def test_the_command_line_pins_the_top_k_to_row_38_s_before_loading_anything(tmp_path):
    cfg = tmp_path / "k3.json"
    cfg.write_text(json.dumps({"n_layer": 3, "n_head": 2, "n_embd": 32, "block_size": 96, "journal": "read", "journal_block": 1,
                               "journal_heads": 2, "journal_read_bytes": 96, "journal_cue_dim": 64, "journal_k": 3}), encoding="utf-8")
    log = tmp_path / "journal.jsonl"; log.write_text("QJE1:x\n", encoding="utf-8"); (tmp_path / "journal.payloads").mkdir()
    import subprocess, sys
    code = ("import os, sys; os.environ['QUANTUM_CORTEX_JOURNAL_KEY'] = '00' * 32; sys.path.insert(0, %r); "
            "from cortex_c2b import bucket_probe as bp; bp._cli(%r)" % (str(bp.REPO_ROOT), ["--ckpt", "c", "--journal", str(log), "--config", str(cfg), "--out", str(tmp_path / "o.json")]))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode != 0 and "journal_k is 3" in (r.stderr + r.stdout)


@pytest.fixture(scope="module")
def sealed_fixture(tmp_path_factory):
    """A tiny checkpoint with a run id and its config hash, its config file, and a journal sealed under a
    throwaway key with the protocol's facts written through the gate: the run the founder makes, in miniature."""
    d = tmp_path_factory.mktemp("e2e")
    cfg_dict = {"n_layer": 3, "n_head": 2, "n_embd": 32, "block_size": 96, "journal": "read", "journal_block": 1,
                "journal_heads": 2, "journal_read_bytes": 96, "journal_cue_dim": 64, "journal_k": 4}
    (d / "tiny.json").write_text(json.dumps(cfg_dict), encoding="utf-8")
    cfg = train.load_config(str(d / "tiny.json"))
    torch.manual_seed(0)
    m = train.VanillaGPT(cfg)
    key = bytes(range(32))
    j = Journal(d / "journal.jsonl", key=key, policy=POLICY_STOP)
    b = JournalBridge(m, j, k=4, budget_bytes=96, seed=0, shuffle_seed=1)
    facts, _ = generate_facts(N_FACTS, 0)
    for i, f in enumerate(facts):
        b.write(f.statement, f.schema, now=100.0 + i, entity=f.entity)
    assert j.payloads.sealed and (d / "journal.jsonl").read_bytes().startswith(b"QJE1:")   # sealed by construction, per line
    torch.save({"model": m.state_dict(), "run_id": "e2e000000000", "config_hash": train.config_hash(cfg)}, d / "ckpt.pt")
    return d, key, len(b.pointer_of)


def test_one_run_end_to_end_writes_the_file_the_record_test_reads_and_refuses_a_second_run(sealed_fixture, monkeypatch, capsys):
    d, key, n_planted = sealed_fixture
    monkeypatch.setenv("QUANTUM_CORTEX_JOURNAL_KEY", key.hex())
    monkeypatch.setattr(bp, "_git_state", lambda: ("test0000", True))      # a fixture run is not a file of the record: the tree's state is not its matter
    sha_before = hashlib.sha256((d / "journal.jsonl").read_bytes()).hexdigest()
    bp._cli(["--ckpt", str(d / "ckpt.pt"), "--journal", str(d / "journal.jsonl"), "--config", str(d / "tiny.json"), "--out", str(d / "out")])
    printed = capsys.readouterr().out
    path = d / "out" / "bucket-probe-e2e000000000.json"
    assert path.exists() and "[record] retained" in printed and "C0 " in printed and "C6 " in printed
    assert hashlib.sha256((d / "journal.jsonl").read_bytes()).hexdigest() == sha_before      # nothing written to the journal
    out = json.loads(path.read_text(encoding="utf-8"))
    assert out["checkpoint"]["run_id"] == "e2e000000000" and out["checkpoint"]["config_hash"] == train.config_hash(train.load_config(str(d / "tiny.json")))
    assert out["checkpoint"]["config_sha256"] == hashlib.sha256((d / "tiny.json").read_bytes()).hexdigest()[:16]
    assert out["journal"]["sha256_before"] == out["journal"]["sha256_after"] == sha_before[:16]
    assert out["journal"]["unchanged"] is True and out["journal"]["payloads_before"] == out["journal"]["payloads_after"]
    assert out["journal"]["payloads_before"]["n_payload_files"] == n_planted and out["journal"]["opened"].startswith("a copy")
    assert sorted(q.name for q in (d / "journal.payloads").iterdir()) == sorted(q.name for q in (d / "journal.payloads").iterdir())
    assert out["code_tree_clean"] is True and out["code_commit"] == "test0000"
    assert out["device"]["cuda_available"] is torch.cuda.is_available()                         # the encoder where row 38's probe ran it
    assert out["device"]["device"] == ("cuda" if torch.cuda.is_available() else "cpu") and isinstance(out["device"]["parallel_info"], str)
    assert (out["device"]["gpu"] is None) == (not torch.cuda.is_available())
    assert out["journal"]["entries"] == n_planted == out["generator"]["planted_found"] and out["journal"]["states"] == {"live": n_planted}
    assert out["precondition_row38"] == {"applies": False} and "stopped" not in out
    assert out["k"] == 4 and out["n_facts"] == N_FACTS and out["n_negctrl"] == N_NEGCTRL and out["facts_seed"] == 0 and out["index_seed"] == 0
    assert out["control_matches_live_index"] is True and all(out["subset_relations"].values())
    assert len(out["rows"]["C0"]) == N_FACTS and len(out["never_planted"]["C0"]) == N_NEGCTRL and len(out["questions"]) == N_FACTS
    assert sum(q["planted"] for q in out["questions"]) == n_planted
    for name, agg in out["aggregates"].items():
        rec = bp.aggregate(out["rows"][name], out["never_planted"][name], out["rows"]["C0"], n_entries=out["journal"]["entries"], k=out["k"])
        assert all((agg[k] == pytest.approx(v)) if isinstance(v, float) else agg[k] == v for k, v in rec.items()), name
    assert path.read_text(encoding="utf-8").count("\n") == 1 and "NaN" not in path.read_text(encoding="utf-8")   # compact, one line, finite
    with pytest.raises(SystemExit, match="never overwritten"):                                     # a second run on the same --out
        bp._cli(["--ckpt", str(d / "ckpt.pt"), "--journal", str(d / "journal.jsonl"), "--config", str(d / "tiny.json"), "--out", str(d / "out")])
    other = d / "k3.json"                                                                           # another config than the run's: refused by its hash
    other.write_text(json.dumps({**json.loads((d / "tiny.json").read_text()), "n_layer": 2}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not the run's config"):
        bp._cli(["--ckpt", str(d / "ckpt.pt"), "--journal", str(d / "journal.jsonl"), "--config", str(other), "--out", str(d / "out2")])
    assert not (d / "out2").exists()
    stripped = d / "ckpt-nohash.pt"                                                                 # a checkpoint without its config hash: refused
    ck = torch.load(d / "ckpt.pt", map_location="cpu", weights_only=False); ck.pop("config_hash"); torch.save(ck, stripped)
    with pytest.raises(SystemExit, match="no config hash"):
        bp._cli(["--ckpt", str(stripped), "--journal", str(d / "journal.jsonl"), "--config", str(d / "tiny.json"), "--out", str(d / "out3")])
    monkeypatch.setenv("QUANTUM_CORTEX_JOURNAL_KEY", "11" * 32)                                      # a wrong key: refused, nothing written
    with pytest.raises(SystemExit, match="does not open under the key"):
        bp._cli(["--ckpt", str(d / "ckpt.pt"), "--journal", str(d / "journal.jsonl"), "--config", str(d / "tiny.json"), "--out", str(d / "out4")])
    assert not (d / "out3").exists() and not (d / "out4").exists()
    assert hashlib.sha256((d / "journal.jsonl").read_bytes()).hexdigest() == sha_before
    monkeypatch.setenv("QUANTUM_CORTEX_JOURNAL_KEY", key.hex())
    monkeypatch.setattr(bp, "_git_state", lambda: ("abc1234", False))      # git answers: a modified tracked file
    with pytest.raises(SystemExit, match="not clean"):
        bp._cli(["--ckpt", str(d / "ckpt.pt"), "--journal", str(d / "journal.jsonl"), "--config", str(d / "tiny.json"), "--out", str(d / "out5")])
    assert not (d / "out5").exists()
    monkeypatch.setattr(bp, "_git_state", lambda: (None, None))            # git absent: recorded as unknown, the record test refuses such a file
    bp._cli(["--ckpt", str(d / "ckpt.pt"), "--journal", str(d / "journal.jsonl"), "--config", str(d / "tiny.json"), "--out", str(d / "out7.json")])
    assert json.loads((d / "out7.json").read_text(encoding="utf-8"))["code_commit"] is None
    monkeypatch.setattr(bp, "_git_state", lambda: ("test0000", True))
    bp._cli(["--ckpt", str(d / "ckpt.pt"), "--journal", str(d / "journal.jsonl"), "--config", str(d / "tiny.json"), "--index-seed", "1", "--out", str(d / "out6")])
    assert (d / "out6" / "bucket-probe-e2e000000000-index1.json").exists()                          # another index seed is in the name
    out1 = json.loads((d / "out6" / "bucket-probe-e2e000000000-index1.json").read_text(encoding="utf-8"))
    assert out1["index_seed"] == 1 and out1["control_matches_live_index"] is True and out1["configurations"]["C0"]["planes_sha256"] != out["configurations"]["C0"]["planes_sha256"]
    assert path.read_bytes().endswith(b"}\n") and not path.read_bytes().endswith(b"\r\n")          # LF, whatever the platform


def test_a_run_that_stops_retains_c0_alone_in_its_failure_file(sealed_fixture, monkeypatch, tmp_path):
    """A journal of the record cannot be faked here, so the stop is forced through the control check: the failure
    file carries the header, the checks and C0; no other configuration's numbers are retained."""
    d, key, n_planted = sealed_fixture
    monkeypatch.setenv("QUANTUM_CORTEX_JOURNAL_KEY", key.hex())
    monkeypatch.setattr(bp, "_git_state", lambda: ("test0000", True))
    real = bp.probe_journal

    def broken(*args, **kw):
        out = real(*args, **kw)
        out["control_matches_live_index"] = False
        return out
    monkeypatch.setattr(bp, "probe_journal", broken)
    target = tmp_path / "probe.json"
    with pytest.raises(SystemExit, match="stopped before any configuration is read"):
        bp._cli(["--ckpt", str(d / "ckpt.pt"), "--journal", str(d / "journal.jsonl"), "--config", str(d / "tiny.json"), "--out", str(target)])
    failed = tmp_path / "probe-FAILED-precondition.json"
    assert failed.exists() and not target.exists()
    out = json.loads(failed.read_text(encoding="utf-8"))
    assert out["stopped"] == ["the probe's control is not the live index's candidate set"]
    assert set(out["rows"]) == set(out["aggregates"]) == set(out["entry_signatures"]) == set(out["never_planted"]) == {"C0"}
    assert out["configurations_not_retained"] == [c["name"] for c in bp.CONFIGS if c["name"] != "C0"] + ["C6"]
    assert len(out["rows"]["C0"]) == N_FACTS and out["precondition_row38"] == {"applies": False}
    with pytest.raises(SystemExit, match="never overwritten"):                                     # the failure file is a file of the record
        bp._cli(["--ckpt", str(d / "ckpt.pt"), "--journal", str(d / "journal.jsonl"), "--config", str(d / "tiny.json"), "--out", str(target)])
