"""ADR-008, amendment of 2026-10-13: the adoption of C2b as the organ's index geometry, items 1 to 6, written
before the code. The geometry is a declared parameter of the index (`signature` collapsed or exact; `geometry()`
with the planes hash); it travels with the journal on every write event beside the separation meta, the first
stamp kept at replay and a later different one refusing the open; an unstamped journal is the legacy geometry;
the recorded geometry is the default at open, a path that builds its own index otherwise is refused, an index
passed explicitly is kept and recorded as overriding; the recipe's two fields are transparent in the hash at the
legacy values and enter it at the adopted ones (the three v9 hashes pinned unchanged); a pair outside the two
declared geometries is refused; the lifecycle's configuration names the geometry; the defaults stay legacy."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

import train
from cortex_c2b import Journal, POLICY_STOP, CUE_DIM, _hash_obj
from cortex_c2b.lm_bridge import JournalBridge
from cortex_c2b.read_path import (CueIndex, JournalPath, LEGACY_GEOMETRY, ADOPTED_GEOMETRY, GEOMETRY_KEYS,
                                  same_geometry)
from cortex_c2b.write_path import WritePath
from cortex_c2b.lifecycle import LifecycleConfig, LifecycleScheduler

V9 = {"configs/kaggle_t4_journal_v9.json": ("36a3dd10bbff84eb", "0662025ee8fe7979"),
      "configs/local_journal_v9_seed2.json": ("d0c0d93ba24555a8", "182bae7a41291fbf"),
      "configs/local_journal_v9_seed3.json": ("a727d1cb445afa92", "2c98acc1ff337564")}


def _cue(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(CUE_DIM).astype(np.float32)


def _old_collapsed_sig(planes: np.ndarray, v: np.ndarray) -> int:
    """The live `_sig` of every run to date, as it was written (the packed byte sum added to the ten bit integer)."""
    bits = (v @ planes) > 0
    return int(np.packbits(bits.astype(np.uint8)).view(np.uint8).astype(np.int64).sum()
               + bits.astype(np.int64) @ (1 << np.arange(len(bits))))


# ---- item 1: the geometry as a declared parameter --------------------------------------------------
def test_the_default_index_is_the_legacy_geometry_and_its_collapsed_signature_is_the_live_one_bit_for_bit():
    idx = CueIndex(dim=CUE_DIM, seed=0)
    assert same_geometry(idx.geometry(), LEGACY_GEOMETRY) and idx.signature == "collapsed"
    assert (idx.n_tables, idx.n_bits, idx.seed) == (4, 10, 0) and len(idx.planes) == 4
    for s in range(50):
        v = _cue(s)
        for t in range(4):
            assert idx._sig(v, t) == _old_collapsed_sig(idx.planes[t], v)


def test_the_exact_signature_is_the_ten_bit_integer_alone_and_the_tables_nest():
    legacy, exact16 = CueIndex(dim=CUE_DIM, seed=0), CueIndex(dim=CUE_DIM, seed=0, n_tables=16, signature="exact")
    assert exact16.signature == "exact" and exact16.n_tables == 16 and len(exact16.planes) == 16
    for t in range(4):
        assert np.array_equal(legacy.planes[t], exact16.planes[t])             # one generator, table by table: C2 nests C0
    for s in range(50):
        v = _cue(s)
        for t in range(16):
            bits = ((v @ exact16.planes[t]) > 0).astype(np.int64)
            assert exact16._sig(v, t) == int(bits @ (1 << np.arange(10))) < 1024
    assert len({exact16._sig(_cue(s), 0) for s in range(3000)}) > 278           # not collapsed to 278 values
    with pytest.raises(ValueError):
        CueIndex(signature="packed")


def test_geometry_returns_the_declared_parameters_and_the_planes_hash_of_row_45_s_files():
    g = CueIndex(dim=CUE_DIM, seed=0).geometry()
    assert set(g) == set(GEOMETRY_KEYS) | {"planes_sha256"} and tuple(GEOMETRY_KEYS) == ("n_tables", "n_bits", "signature", "seed")
    assert g["planes_sha256"] == "680896ab07072738"                           # C0's planes in the three files of row 45
    assert CueIndex(dim=CUE_DIM, seed=0, n_tables=16, signature="exact").geometry()["planes_sha256"] == "e5703fddba43332f"   # C2b's
    assert CueIndex(dim=CUE_DIM, seed=1).geometry()["planes_sha256"] not in ("680896ab07072738", "e5703fddba43332f")
    assert same_geometry(CueIndex(dim=CUE_DIM, seed=0, n_tables=16, signature="exact").geometry(), ADOPTED_GEOMETRY)
    assert not same_geometry(ADOPTED_GEOMETRY, LEGACY_GEOMETRY) and ADOPTED_GEOMETRY["n_bits"] == LEGACY_GEOMETRY["n_bits"] == 10


def test_the_read_law_test_is_the_one_the_amendment_names_and_runs_under_both_geometries():
    """`tests/test_c2b_read_path.py::test_ann_read_never_scans` is parametrised over the two geometries since the
    amendment; this pins that the file carries both (the systems reader of the committee computed 51 touched under
    the legacy geometry and 88 under the adopted one on those cues, both far under the fifth of 3 000)."""
    import importlib.util
    import inspect
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("rp_under_test", Path(__file__).with_name("test_c2b_read_path.py"))
    rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)
    src = inspect.getsource(rp.test_ann_read_never_scans)
    marks = [m for m in getattr(rp.test_ann_read_never_scans, "pytestmark", []) if m.name == "parametrize"]
    assert marks and marks[0].args[1] == [LEGACY_GEOMETRY, ADOPTED_GEOMETRY] and "N // 5" in src


# ---- item 2: the geometry travels with the journal ------------------------------------------------
def _write_through_gate(journal: Journal, jp: JournalPath | None, n: int, seed: int = 0, start: int = 0) -> list:
    wp = WritePath(journal, seed=seed)
    out = []
    for i in range(start, start + n):
        rep = wp.write(_cue(1000 + i), f"line {i}".encode(), "test", now=float(i))
        assert rep.admitted
        if jp is not None:
            jp.on_write(rep.entry)
        out.append(rep.entry)
    return out


def _events(path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_a_fresh_journal_takes_the_first_path_s_geometry_and_stamps_every_write_event_beside_ca3(tmp_path):
    j = Journal(tmp_path / "j.jsonl", plaintext=True)
    assert j.index_record() is None and j.index_geometry is None             # no record yet: nothing written
    jp = JournalPath(j, geometry={"n_tables": 16, "signature": "exact"})
    assert same_geometry(jp.index.geometry(), ADOPTED_GEOMETRY) and jp.index_overridden is False
    assert same_geometry(j.index_geometry, ADOPTED_GEOMETRY) and j.index_geometry["planes_sha256"] == "e5703fddba43332f"
    _write_through_gate(j, jp, 5)
    writes = [ev for ev in _events(tmp_path / "j.jsonl") if ev["ev"] == "write"]
    assert len(writes) == 5 and all("ca3" in ev and "index" in ev for ev in writes)
    assert all(same_geometry(ev["index"], ADOPTED_GEOMETRY) and ev["index"]["planes_sha256"] == "e5703fddba43332f" for ev in writes)
    assert all(ev["ca3"] == {"dg": True, "seed": 0} for ev in writes)         # the separation meta untouched beside it


def test_the_stamp_round_trips_through_the_replay_and_the_recorded_geometry_is_the_default_at_open(tmp_path):
    j = Journal(tmp_path / "j.jsonl", plaintext=True)
    _write_through_gate(j, JournalPath(j, geometry=ADOPTED_GEOMETRY), 6)
    j2 = Journal(tmp_path / "j.jsonl", plaintext=True)
    assert same_geometry(j2.index_record(), ADOPTED_GEOMETRY) and same_geometry(j2.index_geometry, ADOPTED_GEOMETRY)
    jp2 = JournalPath(j2)                                                       # no argument: the record decides
    assert same_geometry(jp2.index.geometry(), ADOPTED_GEOMETRY) and jp2.index.n_tables == 16 and jp2.index_overridden is False
    assert jp2.index.geometry()["planes_sha256"] == j2.index_record()["planes_sha256"]
    assert len(jp2.index) == 6
    # the same seed asked explicitly agrees; another seed or another configuration is refused with both geometries named
    assert same_geometry(JournalPath(Journal(tmp_path / "j.jsonl", plaintext=True), seed=0).index.geometry(), ADOPTED_GEOMETRY)
    with pytest.raises(ValueError, match="records") as e:
        JournalPath(Journal(tmp_path / "j.jsonl", plaintext=True), seed=3)
    assert "exact" in str(e.value) and "3" in str(e.value)
    with pytest.raises(ValueError, match="records"):
        JournalPath(Journal(tmp_path / "j.jsonl", plaintext=True), geometry=LEGACY_GEOMETRY)
    with pytest.raises(ValueError, match="records"):
        Journal(tmp_path / "j.jsonl", plaintext=True, index_geometry=LEGACY_GEOMETRY)     # a configuration given at open


def test_an_unstamped_journal_is_the_legacy_geometry_accepts_an_explicit_index_and_refuses_a_differing_path(tmp_path):
    j = Journal(tmp_path / "old.jsonl", plaintext=True)
    _write_through_gate(j, None, 4)                                             # the code before the amendment: no path attached, no stamp
    assert all("index" not in ev for ev in _events(tmp_path / "old.jsonl") if ev["ev"] == "write")
    j2 = Journal(tmp_path / "old.jsonl", plaintext=True)
    assert same_geometry(j2.index_record(), LEGACY_GEOMETRY) and j2.index_record()["seed"] == 0
    jp = JournalPath(j2)
    assert same_geometry(jp.index.geometry(), LEGACY_GEOMETRY) and jp.index_overridden is False and len(jp.index) == 4
    assert same_geometry(j2.index_geometry, LEGACY_GEOMETRY)                    # what a continued write would stamp
    with pytest.raises(ValueError, match="records"):
        JournalPath(Journal(tmp_path / "old.jsonl", plaintext=True), seed=1)    # a path only reader re indexed silently before; closed
    with pytest.raises(ValueError, match="records"):
        JournalPath(Journal(tmp_path / "old.jsonl", plaintext=True), geometry=ADOPTED_GEOMETRY)
    j3 = Journal(tmp_path / "old.jsonl", plaintext=True)
    live = CueIndex(dim=j3.cue_dim, seed=0, n_tables=16, signature="exact")
    jp3 = JournalPath(j3, index=live)                                           # the declared reader (the probes): kept, recorded as overriding
    assert jp3.index is live and jp3.index_overridden is True and len(live) == 4
    assert same_geometry(j3.index_geometry, LEGACY_GEOMETRY)                    # the journal's own geometry does not move under an overriding reader
    jp4 = JournalPath(Journal(tmp_path / "old.jsonl", plaintext=True), index=CueIndex(dim=CUE_DIM, seed=1))
    assert jp4.index_overridden is True                                         # index seed 1 on a legacy journal: the replication probe's case
    jp5 = JournalPath(Journal(tmp_path / "old.jsonl", plaintext=True), index=CueIndex(dim=CUE_DIM, seed=0))
    assert jp5.index_overridden is False                                        # an explicit index equal to the record overrides nothing


def test_a_legacy_journal_continued_under_the_new_code_stamps_legacy_and_reopens(tmp_path):
    j = Journal(tmp_path / "c.jsonl", plaintext=True)
    _write_through_gate(j, None, 3)
    j2 = Journal(tmp_path / "c.jsonl", plaintext=True)
    jp2 = JournalPath(j2)
    _write_through_gate(j2, jp2, 2, start=3)
    evs = [ev for ev in _events(tmp_path / "c.jsonl") if ev["ev"] == "write"]
    assert [("index" in ev) for ev in evs] == [False, False, False, True, True]
    assert all(same_geometry(ev["index"], LEGACY_GEOMETRY) for ev in evs[3:])
    j3 = Journal(tmp_path / "c.jsonl", plaintext=True)
    assert same_geometry(j3.index_record(), LEGACY_GEOMETRY) and len(JournalPath(j3).index) == 5


def test_a_later_write_event_under_another_geometry_refuses_the_open(tmp_path):
    j = Journal(tmp_path / "two.jsonl", plaintext=True)
    _write_through_gate(j, JournalPath(j), 2)                                  # legacy stamps
    ev = {"ev": "write", "entry": dict(j._entries[next(iter(j._entries))].to_dict()), "ca3": {"dg": True, "seed": 0},
          "index": {**ADOPTED_GEOMETRY, "planes_sha256": "e5703fddba43332f"}}
    ev["entry"]["entry_id"] = "forged0000000000"
    with open(tmp_path / "two.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(ev, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="two index geometries"):
        Journal(tmp_path / "two.jsonl", plaintext=True)
    # the same on an unstamped head: the unstamped writes are the legacy record, a later adopted stamp differs
    j = Journal(tmp_path / "mixed.jsonl", plaintext=True)
    _write_through_gate(j, None, 2)
    with open(tmp_path / "mixed.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(ev, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="two index geometries"):
        Journal(tmp_path / "mixed.jsonl", plaintext=True)


def test_the_stamp_changes_no_recorded_hash_and_no_existing_byte(tmp_path):
    a, b = Journal(tmp_path / "a.jsonl", plaintext=True), Journal(tmp_path / "b.jsonl", plaintext=True)
    _write_through_gate(a, None, 3)                                             # unstamped
    _write_through_gate(b, JournalPath(b, geometry=ADOPTED_GEOMETRY), 3)       # stamped with the adopted geometry
    assert a.scope_config_hash() == b.scope_config_hash()                       # the scope hash excludes the geometry (it is written into every phase event)
    ea, eb = [ev for ev in _events(tmp_path / "a.jsonl") if ev["ev"] == "write"], [ev for ev in _events(tmp_path / "b.jsonl") if ev["ev"] == "write"]
    for x, y in zip(ea, eb):
        assert x["entry"] == y["entry"] and x["ca3"] == y["ca3"]                 # entry ids, pointers, cues, times: identical
        assert {k: v for k, v in y.items() if k != "index"} == x                # the stamp is the only added key
    assert sorted(p.name for p in (tmp_path / "a.payloads").iterdir()) == sorted(p.name for p in (tmp_path / "b.payloads").iterdir())
    assert Journal(tmp_path / "a.jsonl", plaintext=True).scope_config_hash() == Journal(tmp_path / "b.jsonl", plaintext=True).scope_config_hash()


def test_a_memory_journal_records_its_first_path_and_refuses_a_second_path_of_another_geometry():
    j = Journal()
    jp = JournalPath(j, seed=5)                                                 # the curriculum's memory scopes: the run's seed
    assert j.index_record() is None and j.index_geometry["seed"] == 5 and jp.index_overridden is False
    _write_through_gate(j, jp, 2, seed=5)
    assert same_geometry(j.index_record(), {**LEGACY_GEOMETRY, "seed": 5})     # a memory journal keeps its stamps in memory too
    with pytest.raises(ValueError, match="records"):
        JournalPath(j, seed=0)
    assert same_geometry(JournalPath(j).index.geometry(), {**LEGACY_GEOMETRY, "seed": 5})


# ---- items 3 and 5: the recipe names the geometry in its hash; the defaults stay legacy -------------
def test_the_two_configuration_fields_are_transparent_at_legacy_and_enter_the_hash_at_the_adopted_values():
    cfg = train.Config()
    assert (cfg.journal_index_tables, cfg.journal_index_signature) == (4, "collapsed")
    assert train.HASH_TRANSPARENT_AT_DEFAULT["journal_index_tables"] == 4 and train.HASH_TRANSPARENT_AT_DEFAULT["journal_index_signature"] == "collapsed"
    for path, (legacy_hash, adopted_hash) in V9.items():
        c = train.load_config(path)
        assert train.config_hash(c) == legacy_hash, path                       # the three v9 hashes, unchanged (the record's checkpoints)
        assert train.index_geometry_of(c) == {"n_tables": 4, "n_bits": 10, "signature": "collapsed"}
        c.journal_index_tables, c.journal_index_signature = 16, "exact"
        assert train.config_hash(c) == adopted_hash, path                      # a run under the adopted geometry names it in its hash
        assert train.index_geometry_of(c) == {"n_tables": 16, "n_bits": 10, "signature": "exact"}
    d = {k: v for k, v in dataclasses.asdict(train.Config()).items()}
    assert "journal_index_tables" in d and "journal_index_signature" in d


@pytest.mark.parametrize("tables,signature", [(16, "collapsed"), (8, "exact"), (4, "exact"), (16, "fly"), (0, "exact"), (32, "exact")])
def test_a_pair_outside_the_two_declared_geometries_is_refused(tables, signature):
    cfg = train.Config()
    cfg.journal_index_tables, cfg.journal_index_signature = tables, signature
    with pytest.raises(ValueError, match="declared geometries"):
        train.index_geometry_of(cfg)


def test_load_config_accepts_the_two_fields_now_and_refused_them_before(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"journal_index_tables": 16, "journal_index_signature": "exact", "n_layer": 1}), encoding="utf-8")
    c = train.load_config(str(p))
    assert (c.journal_index_tables, c.journal_index_signature) == (16, "exact")
    p.write_text(json.dumps({"journal_index_tablez": 16}), encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown config key"):               # the old code's behaviour on a new key: fail closed
        train.load_config(str(p))


def _tiny_model(tmp_path):
    cfg_dict = {"n_layer": 2, "n_head": 2, "n_embd": 32, "block_size": 96, "journal": "read", "journal_block": 1,
                "journal_heads": 2, "journal_read_bytes": 96, "journal_cue_dim": 64, "journal_k": 4}
    (tmp_path / "tiny.json").write_text(json.dumps(cfg_dict), encoding="utf-8")
    cfg = train.load_config(str(tmp_path / "tiny.json"))
    torch.manual_seed(0)
    return cfg, train.VanillaGPT(cfg)


def test_the_bridge_under_the_adopted_geometry_stamps_a_fresh_journal_and_geometry_reads_it_back(tmp_path):
    cfg, m = _tiny_model(tmp_path)
    cfg.journal_index_tables, cfg.journal_index_signature = 16, "exact"
    key = bytes(range(32))
    j = Journal(tmp_path / "journal.jsonl", key=key, policy=POLICY_STOP)
    b = JournalBridge(m, j, k=4, budget_bytes=96, seed=0, shuffle_seed=1, index_geometry=train.index_geometry_of(cfg))
    assert same_geometry(b.jp.index.geometry(), ADOPTED_GEOMETRY) and b.jp.index_overridden is False
    for i in range(5):
        assert b.write(f"Entity{i} took place in {1990 + i}.", "event", now=float(i), entity=f"Entity{i}").admitted
    j2 = Journal(tmp_path / "journal.jsonl", key=key, policy=POLICY_STOP)
    assert same_geometry(j2.index_record(), ADOPTED_GEOMETRY) and j2.index_record()["planes_sha256"] == "e5703fddba43332f"
    b2 = JournalBridge(m, j2, k=4, budget_bytes=96, seed=0, shuffle_seed=1, index_geometry=train.index_geometry_of(cfg))
    assert same_geometry(b2.jp.index.geometry(), ADOPTED_GEOMETRY) and len(b2.jp.index) == 5
    b3 = JournalBridge(m, Journal(tmp_path / "journal.jsonl", key=key, policy=POLICY_STOP), k=4, budget_bytes=96, seed=0)
    assert same_geometry(b3.jp.index.geometry(), ADOPTED_GEOMETRY)              # no geometry given: the record decides
    with pytest.raises(ValueError, match="records"):                           # the legacy recipe on an adopted journal: refused
        JournalBridge(m, Journal(tmp_path / "journal.jsonl", key=key, policy=POLICY_STOP), k=4, budget_bytes=96, seed=0,
                      index_geometry=train.index_geometry_of(train.Config()))


def test_the_bridge_without_a_geometry_stays_legacy_on_a_fresh_journal(tmp_path):
    cfg, m = _tiny_model(tmp_path)
    b = JournalBridge(m, Journal(), k=4, budget_bytes=96, seed=0)
    assert same_geometry(b.jp.index.geometry(), LEGACY_GEOMETRY) and b.jp.index.signature == "collapsed"   # D1: the defaults do not move
    assert same_geometry(JournalPath(Journal()).index.geometry(), LEGACY_GEOMETRY)
    assert same_geometry(CueIndex().geometry(), {**LEGACY_GEOMETRY, "seed": 0})


# ---- item 4: the lifecycle reads the index and says so ------------------------------------------------
def test_lifecycle_config_names_the_geometry_transparent_at_legacy_and_the_scheduler_refuses_a_mismatch():
    legacy = LifecycleConfig()
    assert (legacy.index_tables, legacy.index_signature) == (4, "collapsed")
    d = dataclasses.asdict(legacy)
    d.pop("withdraw_at_eviction", None) if not d.get("withdraw_at_eviction") else None
    d_without = {k: v for k, v in d.items() if k not in ("index_tables", "index_signature")}
    assert legacy.config_hash() == _hash_obj(d_without)                        # the hash of every phase record of the record, unchanged
    adopted = LifecycleConfig(index_tables=16, index_signature="exact")
    assert adopted.config_hash() != legacy.config_hash()
    j = Journal()
    jp = JournalPath(j)
    LifecycleScheduler(j, WritePath(j), jp, cfg=legacy)                         # legacy on legacy
    with pytest.raises(ValueError, match="geometry"):
        LifecycleScheduler(j, WritePath(j), jp, cfg=adopted)                    # the configuration says 16/exact, the path holds 4/collapsed
    j2 = Journal()
    jp2 = JournalPath(j2, geometry=ADOPTED_GEOMETRY)
    s = LifecycleScheduler(j2, WritePath(j2), jp2, cfg=adopted)
    snap = s.snapshot()
    assert same_geometry(snap["index_geometry"], ADOPTED_GEOMETRY) and snap["config_hash"] == adopted.config_hash()


# ---- the committee's round on the build (2026-10-13, three readers): the cases they ran, pinned ------------------
def test_a_fresh_journal_given_a_geometry_at_construction_is_bound_by_it(tmp_path):
    """Systems reader, finding 1; third reader, finding 1: the path compared against the record alone, None on a fresh
    journal, so a constructor geometry could be contradicted by the path and the stamp would lie."""
    j = Journal(tmp_path / "k.jsonl", plaintext=True, index_geometry=ADOPTED_GEOMETRY)
    with pytest.raises(ValueError, match="records"):
        JournalPath(j, geometry=LEGACY_GEOMETRY)                                # case K
    with pytest.raises(ValueError, match="one geometry"):
        JournalPath(Journal(tmp_path / "j.jsonl", plaintext=True, index_geometry=ADOPTED_GEOMETRY), index=CueIndex(dim=CUE_DIM))   # case J
    jp = JournalPath(Journal(tmp_path / "ok.jsonl", plaintext=True, index_geometry=ADOPTED_GEOMETRY))
    assert same_geometry(jp.index.geometry(), ADOPTED_GEOMETRY) and jp.index_overridden is False
    jp2 = JournalPath(Journal(tmp_path / "ok2.jsonl", plaintext=True, index_geometry=ADOPTED_GEOMETRY), index=CueIndex(dim=CUE_DIM, n_tables=16, signature="exact"))
    assert jp2.index_overridden is False


def test_the_planes_hash_is_compared_at_open_and_a_continued_journal_stamps_the_live_draw(tmp_path):
    """Systems reader, finding 7; third reader, finding 2: a stamp with the four parameters and another planes hash
    (numpy's generator moved) opened silently and the continued writes copied the old hash."""
    j = Journal(tmp_path / "p.jsonl", plaintext=True)
    _write_through_gate(j, JournalPath(j, geometry=ADOPTED_GEOMETRY), 2)
    evs = _events(tmp_path / "p.jsonl")
    forged = dict(evs[0]); forged["entry"] = dict(forged["entry"]); forged["entry"]["entry_id"] = "forged0000000001"
    forged["index"] = {**forged["index"], "planes_sha256": "0000000000000000"}
    (tmp_path / "q.jsonl").write_text(json.dumps(forged, separators=(",", ":")) + "\n", encoding="utf-8")
    import shutil; shutil.copytree(tmp_path / "p.payloads", tmp_path / "q.payloads")
    jq = Journal(tmp_path / "q.jsonl", plaintext=True)                        # opens: the stamp is well formed
    assert jq.index_record()["planes_sha256"] == "0000000000000000"
    with pytest.raises(ValueError, match="generator moved"):
        JournalPath(jq)                                                         # the live draw differs from the recorded one: refused
    live = CueIndex(dim=CUE_DIM, n_tables=16, signature="exact")
    jp = JournalPath(Journal(tmp_path / "q.jsonl", plaintext=True), index=live)   # the declared reader: kept, overriding
    assert jp.index_overridden is True
    # a later stamp with the same parameters and another planes hash refuses the open (two draws in one journal)
    second = dict(forged); second["entry"] = dict(second["entry"]); second["entry"]["entry_id"] = "forged0000000002"
    second["index"] = {**second["index"], "planes_sha256": "e5703fddba43332f"}
    with open(tmp_path / "q.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(second, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="two index geometries"):
        Journal(tmp_path / "q.jsonl", plaintext=True)
    # a continued legacy journal stamps the live draw's hash (the unstamped record carries none)
    c = Journal(tmp_path / "c.jsonl", plaintext=True)
    _write_through_gate(c, None, 2)
    c2 = Journal(tmp_path / "c.jsonl", plaintext=True)
    _write_through_gate(c2, JournalPath(c2), 1, start=2)
    assert _events(tmp_path / "c.jsonl")[2]["index"]["planes_sha256"] == "680896ab07072738"
    assert c2.index_record_source() == "legacy_by_default" and Journal(tmp_path / "p.jsonl", plaintext=True).index_record_source() == "stamp"
    assert Journal().index_record_source() is None


def test_a_geometry_given_at_open_or_stamped_is_validated_before_anything_is_written(tmp_path):
    """Third reader, finding 3: an unvalidated dict was stamped to disk and failed at the next open with a bare error."""
    for bad in ({"n_tables": 16, "signature": "exact"}, {**ADOPTED_GEOMETRY, "n_tables": "16"}, {**ADOPTED_GEOMETRY, "seed": "0"},
                {**ADOPTED_GEOMETRY, "planes_sha256": "zz"}, {**ADOPTED_GEOMETRY, "rank": "fly"}, {**ADOPTED_GEOMETRY, "signature": "packed"},
                {**ADOPTED_GEOMETRY, "n_tables": True}):
        with pytest.raises(ValueError, match="index geometry"):
            Journal(tmp_path / "v.jsonl", plaintext=True, index_geometry=bad)
    assert not (tmp_path / "v.jsonl").exists()
    with pytest.raises(ValueError, match="lacks"):
        JournalPath(Journal(), geometry={"n_tables": 16})
    with pytest.raises(ValueError, match="unknown key"):
        JournalPath(Journal(), geometry={"n_tables": 16, "signature": "exact", "rank": "fly"})
    j = Journal(tmp_path / "w.jsonl", plaintext=True)
    _write_through_gate(j, JournalPath(j), 1)
    ev = _events(tmp_path / "w.jsonl")[0]; ev["entry"] = dict(ev["entry"]); ev["entry"]["entry_id"] = "forged0000000003"
    ev["index"] = {**ev["index"], "n_tables": "4"}
    with open(tmp_path / "w.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(ev, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="index stamp"):
        Journal(tmp_path / "w.jsonl", plaintext=True)                          # a hand edited stamp fails at open, named


def test_an_explicit_index_of_another_dimension_is_refused_and_the_rank_is_not_part_of_the_stamp(tmp_path):
    """Third reader, findings 4 and 5."""
    with pytest.raises(ValueError, match="dimensions"):
        JournalPath(Journal(), index=CueIndex(dim=32, n_tables=16, signature="exact"))
    j = Journal()
    jp = JournalPath(j, index=CueIndex(dim=CUE_DIM, rank="fly"))               # the retrieval probe's case (Rev108): kept
    assert jp.index.rank == "fly" and same_geometry(j.index_geometry, LEGACY_GEOMETRY)     # the buckets' geometry is stamped, not the ranking


def test_sig_casts_to_float32_so_a_recorded_signature_is_the_lookup_s():
    """Third reader, finding 6."""
    idx = CueIndex(dim=CUE_DIM, n_tables=16, signature="exact")
    for s in range(100):
        v32 = _cue(s); v64 = v32.astype(np.float64) + 1e-12
        assert [idx._sig(v64, t) for t in range(16)] == [idx._sig(v32, t) for t in range(16)]


def test_index_geometry_of_refuses_a_coerced_type_that_the_hash_would_name_apart():
    """Systems reader, finding 14: `"16"` hashed to another value and was accepted as 16."""
    cfg = train.Config(); cfg.journal_index_tables, cfg.journal_index_signature = "16", "exact"
    with pytest.raises(ValueError, match="integer"):
        train.index_geometry_of(cfg)
    cfg = train.Config(); cfg.journal_index_tables = True
    with pytest.raises(ValueError, match="integer"):
        train.index_geometry_of(cfg)


def test_a_run_refuses_before_its_first_step_a_journal_at_its_path_written_under_another_geometry(tmp_path, monkeypatch):
    """Systems reader, finding 9: the protocol arm would have refused the journal at the end of the run."""
    monkeypatch.setattr(train, "REPO_ROOT", tmp_path)
    key = bytes(range(32))
    cfg = train.Config(); cfg.journal_path = "runs/x/journal.jsonl"
    assert train.check_existing_journal_geometry(cfg, key) is None            # no journal yet
    (tmp_path / "runs/x").mkdir(parents=True)
    j = Journal(tmp_path / cfg.journal_path, key=key, policy=POLICY_STOP)
    _write_through_gate(j, JournalPath(j, geometry=ADOPTED_GEOMETRY), 2)
    with pytest.raises(SystemExit, match="never re indexed"):
        train.check_existing_journal_geometry(cfg, key)                       # the configuration is legacy, the journal adopted
    cfg.journal_index_tables, cfg.journal_index_signature = 16, "exact"
    assert same_geometry(train.check_existing_journal_geometry(cfg, key), ADOPTED_GEOMETRY)

