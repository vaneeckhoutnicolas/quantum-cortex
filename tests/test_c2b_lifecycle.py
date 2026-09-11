"""Slice E -- the lifecycle (ADR-007 D4/D6): one test per validated invariant, plus
the cascade, the hierarchy and the frozen protocol after a series of phases."""
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cortex_c2b import (Journal, STATE_LIVE, STATE_CONSOLIDATED, STATE_DEMOTED, STATE_EVICTED,
                        SCHEMA_SUMMARY, ENTRY_MAX_BYTES, lifecycle_declaration)
from cortex_c2b.crypto import generate_key, LINE_PREFIX, MAGIC
from cortex_c2b.write_path import WritePath
from cortex_c2b.read_path import JournalPath
from cortex_c2b.lifecycle import LifecycleScheduler, LifecycleConfig, PersistentMemory
from cortex_c2b.hm_protocol import generate_facts, Reader, run_hm_protocol

ORDER = {STATE_LIVE: 0, STATE_CONSOLIDATED: 1, STATE_DEMOTED: 2, STATE_EVICTED: 3}


# --------------------------------------------------------------------------- helpers
def unit(rng, dim=64):
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def near(base, rng, eps=0.02):
    v = base + eps * rng.standard_normal(base.shape).astype(np.float32)
    return v / np.linalg.norm(v)


def make(cfg=None, path=None, memory=None, key=None):
    j = Journal(path, key=key)
    wp = WritePath(j, seed=0)
    jp = JournalPath(j)
    sch = LifecycleScheduler(j, wp, jp, cfg or LifecycleConfig(), memory=memory, now=0.0)
    return j, wp, jp, sch


def drive(sch, n, rng, start=100.0, cue_fn=None):
    """n admitted or not writes of distinct payloads on the logical clock."""
    for i in range(n):
        c = cue_fn(i) if cue_fn else unit(rng)
        sch.write(c, f"payload {i} {rng.integers(1_000_000)}".encode(), "note", now=start + i)


def events(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ------------------------------------------------------------------ invariant 1
def test_replay_rebuilds_lifecycle_state_exactly(tmp_path):
    path = tmp_path / "journal.jsonl"
    cfg = LifecycleConfig(period_writes=6, demote_k=3, demote_min_cosine=0.85, bytes_setpoint=20_000)
    j, wp, jp, sch = make(cfg, path)
    rng = np.random.default_rng(1)
    base = unit(rng)
    drive(sch, 60, rng, cue_fn=lambda i: near(base, rng, eps=0.05) if i % 2 == 0 else unit(rng))
    any_live = next(e for e in j._entries.values() if e.state == STATE_LIVE)
    wp.credit(any_live.entry_id, 0.3)                       # a credit is an event too
    sch.phase(now=500.0)
    ev = events(path)
    kinds = {e["ev"] for e in ev}
    assert {"write", "transition", "salience", "decay", "phase"} <= kinds
    assert any(e["ev"] == "salience" and e["why"] == "credit" for e in ev)
    assert any(e["ev"] == "salience" and e["why"] == "contract" for e in ev)   # the sentinels
    assert any(e["ev"] == "demote" for e in ev), "the near duplicate cluster must have been demoted"

    r = Journal(path)                                        # replay from the log alone
    assert {k: (e.state, e.salience) for k, e in j._entries.items()} == \
           {k: (e.state, e.salience) for k, e in r._entries.items()}
    assert r.summary_sources == j.summary_sources and r.summary_of == j.summary_of
    assert r.summary_level == j.summary_level and r.contracted == j.contracted
    assert {p: set(h) for p, h in j._refs.items()} == {p: set(h) for p, h in r._refs.items()}


# ------------------------------------------------------------------ invariant 2
def test_states_move_forward_only_and_evicted_never_return(tmp_path):
    path = tmp_path / "journal.jsonl"
    cfg = LifecycleConfig(period_writes=8, bytes_setpoint=8_000, decay=0.7)
    j, wp, jp, sch = make(cfg, path)
    rng = np.random.default_rng(2)
    drive(sch, 120, rng)
    seq: dict[str, list[int]] = {}
    for e in events(path):
        if e["ev"] == "write":
            seq[e["entry"]["entry_id"]] = [ORDER[e["entry"]["state"]]]
        elif e["ev"] == "transition":
            seq[e["entry_id"]].append(ORDER[e["to"]])
    for s in seq.values():
        assert all(a < b for a, b in zip(s, s[1:])), "a state moved backward"
    evicted = [e for e in j._entries.values() if e.state == STATE_EVICTED]
    assert evicted, "the budget must have evicted something"
    for e in evicted[:20]:
        hits = jp.retrieve(e.cue, k=5)
        assert all(h.entry_id != e.entry_id and h.state != STATE_EVICTED for h, _, _ in hits)
        assert e.entry_id not in [x.entry_id for x in j.read_by_cue(e.cue)]
    for e in j._entries.values():                            # no dangling pointer
        if e.state != STATE_EVICTED:
            assert e.pointer in j.payloads


def test_payload_released_at_the_last_pointer_only():
    j = Journal()
    rng = np.random.default_rng(3)
    a = j.write(unit(rng), b"same bytes", 0.9, "note", now=1.0)
    b = j.write(unit(rng), b"same bytes", 0.9, "note", now=2.0)
    assert a.pointer == b.pointer and len(j.payloads) == 1
    j.transition(a.entry_id, STATE_EVICTED)
    assert a.pointer in j.payloads and j.references(a.pointer) == {b.entry_id}
    j.transition(b.entry_id, STATE_EVICTED)
    assert a.pointer not in j.payloads and j.references(a.pointer) == set()


# ------------------------------------------------------------------ invariant 3
class BrokenMemory(PersistentMemory):
    def replay(self, cue):
        super().replay(cue)
        return 1.0                                           # nothing ever verifies


def test_consolidation_requires_a_verified_replay():
    cfg = LifecycleConfig(period_writes=8, sentinel_facts=0)
    j, wp, jp, sch = make(cfg, memory=BrokenMemory())
    rng = np.random.default_rng(4)
    drive(sch, 40, rng)
    snap = j.snapshot()
    assert snap[STATE_CONSOLIDATED] == 0 and snap[STATE_LIVE] == 40
    assert sch.replay_failed > 0 and len(sch.memory) == 0      # failed replays are withdrawn
    j2, wp2, jp2, sch2 = make(cfg)
    drive(sch2, 40, rng)
    assert j2.snapshot()[STATE_CONSOLIDATED] > 0
    for e in j2._entries.values():
        if e.state == STATE_CONSOLIDATED:
            r = sch2.memory.reconstruct(e.cue)
            c = np.asarray(e.cue, dtype=np.float32)
            assert 1.0 - float(r @ c / (np.linalg.norm(r) * np.linalg.norm(c))) <= cfg.replay_error_max


# ------------------------------------------------------------------ invariant 4
def test_demotion_keeps_access_through_the_summary_and_builds_a_hierarchy():
    cfg = LifecycleConfig(period_writes=1000, consolidate_batch=16, demote_k=3, demote_min_cosine=0.95,
                          sentinel_facts=0, decay=1.0)
    j, wp, jp, sch = make(cfg)
    rng = np.random.default_rng(5)
    base = unit(rng)
    cues = [near(base, rng, eps=0.01) for _ in range(9)]
    payloads = [f"fact {i}".encode() for i in range(9)]
    for i, (c, p) in enumerate(zip(cues, payloads)):
        rep = wp.write(c, p, "note", now=10.0 + i)
        if not rep.admitted:                                 # near duplicates may be gated: write them in anyway
            e = j.write(c, p, salience=0.9, schema_id="note", now=10.0 + i)
            jp.on_write(e); wp.ca3.store(wp._code(c))
        else:
            jp.on_write(rep.entry)
    sch.phase(now=100.0)                                     # live -> consolidated
    sch.phase(now=200.0)                                     # 9 consolidated -> 3 summaries (level 1)
    assert j.snapshot()["summaries"] == 3
    for c, p in zip(cues, payloads):
        (e, payload, _), = jp.retrieve(c, k=1)
        assert e.schema_id == SCHEMA_SUMMARY and payload == p   # reachable from the source cue, source payload
        assert e.size_bytes() <= ENTRY_MAX_BYTES
        meta = json.loads(j.payloads.get(e.pointer))
        assert len(meta["pointers"]) == 3 and meta["level"] == 1
    assert all(e.state == STATE_DEMOTED for e in j._entries.values() if e.schema_id == "note")
    sch.phase(now=300.0)                                     # summaries live -> consolidated
    sch.phase(now=400.0)                                     # 3 summaries -> 1 summary of summaries
    assert max(j.summary_level.values()) == 2
    (top, payload, _), = jp.retrieve(cues[0], k=1)
    assert j.summary_level[top.entry_id] == 2 and payload == payloads[0]
    for src in j.summary_of:                                 # no cycle
        seen, cur = set(), src
        while cur in j.summary_of:
            assert cur not in seen; seen.add(cur); cur = j.summary_of[cur]


def test_evicting_a_summary_takes_its_sources_and_frees_their_bytes():
    cfg = LifecycleConfig(period_writes=1000, demote_k=3, demote_min_cosine=0.95, sentinel_facts=0)
    j, wp, jp, sch = make(cfg)
    rng = np.random.default_rng(6)
    base = unit(rng)
    ids = []
    for i in range(3):
        e = j.write(near(base, rng, 0.01), f"src {i}".encode(), 0.9, "note", now=float(i)); jp.on_write(e); ids.append(e.entry_id)
    sch.phase(now=50.0); sch.phase(now=60.0)
    (sid, sources), = j.summary_sources.items()
    assert sorted(sources) == sorted(ids)
    ptrs = [j._entries[i].pointer for i in ids]
    j.transition(sid, STATE_EVICTED)
    assert all(j._entries[i].state == STATE_EVICTED for i in ids)
    assert all(p not in j.payloads for p in ptrs)
    assert jp.retrieve(base, k=3) == []


# ------------------------------------------------------------------ invariant 5
def test_eviction_is_a_verdict_never_a_size_cut():
    # extreme byte pressure, but nothing consolidates (batch 0): live salient entries survive
    cfg = LifecycleConfig(period_writes=4, consolidate_batch=0, bytes_setpoint=1_000, sentinel_facts=0)
    j, wp, jp, sch = make(cfg)
    rng = np.random.default_rng(7)
    drive(sch, 40, rng)
    assert sch.pressure()["pressure"] == 1.0
    assert j.snapshot()[STATE_EVICTED] == 0 and j.snapshot()[STATE_LIVE] == 40
    # the tribunal: a live entry that never earned salience is evicted, pressure or not
    noise = next(iter(j._entries.values()))
    j.set_salience(noise.entry_id, 0.05, why="test")
    sch.phase(now=999.0)
    assert j._entries[noise.entry_id].state == STATE_EVICTED
    assert j.snapshot()[STATE_EVICTED] == 1


def test_consolidated_entries_go_first_under_pressure_and_the_floor_never_reaches_one(tmp_path):
    path = tmp_path / "journal.jsonl"
    cfg = LifecycleConfig(period_writes=8, bytes_setpoint=6_000, decay=0.8)
    j, wp, jp, sch = make(cfg, path)
    rng = np.random.default_rng(8)
    drive(sch, 96, rng)
    last = {}
    for e in events(path):
        if e["ev"] == "transition":
            last.setdefault(e["entry_id"], []).append(e["to"])
    for eid, states in last.items():
        if states[-1] == STATE_EVICTED and STATE_CONSOLIDATED not in states and STATE_DEMOTED not in states:
            pytest.fail("a live entry was evicted outside the tribunal rule")
    assert cfg.evict_floor_max < 1.0                       # a contracted entry (salience 1) is never below the floor


# ------------------------------------------------------------------ invariant 6
def test_budget_regulation_converges_into_the_band_and_keeps_the_contracted():
    S, P, band = 30_000, 8, 0.25
    cfg = LifecycleConfig(period_writes=P, bytes_setpoint=S, band=band, decay=0.9)
    j, wp, jp, sch = make(cfg)
    rng = np.random.default_rng(9)
    trace = []
    for i in range(400):
        sch.write(unit(rng), f"payload {i}".encode(), "note", now=100.0 + i)
        if sch.last_phase and sch.last_phase["phase"] == len(trace) + 1:
            trace.append(sch.last_phase["bytes"])
    slack = P * 700                                          # what can arrive between two phases
    warm = trace[15:]
    assert warm, "not enough phases"
    assert all(S * (1 - band) - slack <= b <= S + slack for b in warm), warm
    assert sch.sentinel_probe() == (1.0, 0.0)              # the contracted facts are all still there
    # control: without a setpoint the same stream grows past S
    j2, wp2, jp2, sch2 = make(LifecycleConfig(period_writes=P, bytes_setpoint=None, decay=0.9))
    rng2 = np.random.default_rng(9)
    for i in range(400):
        sch2.write(unit(rng2), f"payload {i}".encode(), "note", now=100.0 + i)
    assert j2.regulated_bytes() > S


# ------------------------------------------------------------------ invariant 7
def test_rhythm_is_deterministic_and_the_config_is_hashed(tmp_path):
    def run(path, seed=11):
        cfg = LifecycleConfig(period_writes=5, bytes_setpoint=15_000)
        j, wp, jp, sch = make(cfg, path)
        rng = np.random.default_rng(seed)
        base = unit(rng)
        drive(sch, 60, rng, cue_fn=lambda i: near(base, rng, eps=0.05) if i % 3 == 0 else unit(rng))
        return sch
    a = run(tmp_path / "a.jsonl"); b = run(tmp_path / "b.jsonl")
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    phases = [e for e in events(tmp_path / "a.jsonl") if e["ev"] == "phase"]
    assert len(phases) == 12 and a.phases == 12
    for k in ("config_hash", "pressure", "bytes", "latency_p95", "live", "consolidated",
              "demoted", "evicted", "summaries", "phases_refused", "refused"):
        assert k in phases[0]
    assert LifecycleConfig().config_hash() == LifecycleConfig().config_hash()
    assert LifecycleConfig(demote_k=5).config_hash() != LifecycleConfig().config_hash()
    assert phases[0]["config_hash"] == LifecycleConfig(period_writes=5, bytes_setpoint=15_000).config_hash()


# ------------------------------------------------------------------ invariant 8
def test_hm_check_refuses_a_phase_that_would_break_contracted_recall():
    cfg = LifecycleConfig(period_writes=1000, sentinel_facts=20)
    j, wp, jp, sch = make(cfg)
    assert len(sch.sentinels) == 20 and sch.sentinel_probe() == (1.0, 0.0)
    sid, _ = sch.sentinels[0]
    j.set_salience(sid, 0.05, why="test")                  # a corrupted salience: the tribunal would take it
    before = {k: e.state for k, e in j._entries.items()}
    rec = sch.phase(now=50.0)
    assert rec["refused"] and sch.refused == 1 and "recall" in rec["reason"]
    assert {k: e.state for k, e in j._entries.items()} == before   # nothing applied
    assert sch.sentinel_probe() == (1.0, 0.0)
    sch.contract(sid, now=51.0)                              # the contract restored: the next phase applies
    rec = sch.phase(now=60.0)
    assert not rec["refused"] and sch.refused == 1 and rec["applied"]["consolidated"] > 0


def test_frozen_protocol_still_passes_after_a_series_of_phases():
    cfg = LifecycleConfig(period_writes=25, consolidate_batch=32, sentinel_facts=0, decay=1.0)
    j, wp, jp, sch = make(cfg)
    facts, _ = generate_facts(200, seed=0)
    neg, _ = generate_facts(50, seed=10_000)
    for i, f in enumerate(facts):
        sch.write(f.cue, f.statement.encode(), f.schema, now=float(i))
    for k in range(10):
        sch.phase(now=1000.0 + k)
    snap = j.snapshot()
    assert snap[STATE_LIVE] == 0 and snap[STATE_EVICTED] == 0
    reader = Reader(jp, chance=0.0)
    assert sum(reader.recall(f, None) for f in facts) / len(facts) == 1.0
    assert sum(reader.recall(f, None) for f in neg) == 0
    assert run_hm_protocol()["hm_dissociation_pass"] == 1  # the protocol itself, untouched


# ------------------------------------------------------ invariant 9 (Decision 8)
ROOT = Path(__file__).resolve().parents[1]

SESSION_B = r"""
import json, os, sys
import numpy as np
from cortex_c2b import Journal, STATE_EVICTED, SCHEMA_SUMMARY, lifecycle_declaration
from cortex_c2b.crypto import key_from_env
from cortex_c2b.write_path import WritePath
from cortex_c2b.read_path import JournalPath
from cortex_c2b.lifecycle import LifecycleScheduler, LifecycleConfig
from cortex_c2b.hm_protocol import generate_facts, Reader
path, cfg = sys.argv[1], LifecycleConfig(**json.loads(sys.argv[2]))
j = Journal(path, key=key_from_env())                      # a NEW process: nothing but the disk
wp, jp = WritePath(j, seed=0), JournalPath(j)
sch = LifecycleScheduler(j, wp, jp, cfg)
facts, _ = generate_facts(40, seed=3); neg, _ = generate_facts(20, seed=10_003)
reader = Reader(jp, chance=0.0)
recall = sum(reader.recall(f, None) for f in facts) / len(facts)
negctrl = sum(reader.recall(f, None) for f in neg) / len(neg)
dup = wp.write(facts[0].cue, facts[0].statement.encode(), facts[0].schema, now=9_000.0).admitted
demoted = [e for e in j._entries.values() if e.schema_id == "note" and e.state == "demoted"]
via_summary = None
if demoted:
    (eff, payload, _), = jp.retrieve(np.asarray(demoted[0].cue, dtype=np.float32), k=1)
    via_summary = [eff.schema_id == SCHEMA_SUMMARY, payload == j.payloads.get(demoted[0].pointer)]
out = {"recall": recall, "negctrl": negctrl, "sentinel": list(sch.sentinel_probe()), "dup_admitted": dup,
       "snapshot": {k: v for k, v in j.snapshot().items() if k in ("live", "consolidated", "demoted", "evicted", "summaries")},
       "memory": len(sch.memory), "phases": sch.phases, "ca3": len(wp.ca3._codes), "via_summary": via_summary,
       "declaration": lifecycle_declaration(j), "writes_since_phase": sch.writes_since_phase}
r = sch.phase(now=10_000.0)                                # continuity: the next sleep applies
out["next_phase_refused"] = r["refused"]
print(json.dumps(out))
"""


@pytest.mark.parametrize("sealed", [True, False])
def test_the_journal_survives_a_full_process_restart(tmp_path, sealed):
    key = generate_key() if sealed else None
    path = tmp_path / "journal.jsonl"
    cfg = LifecycleConfig(period_writes=7, demote_k=3, demote_min_cosine=0.85, bytes_setpoint=200_000,
                          sentinel_facts=20)                 # no pressure: what is recallable before must be after
    j, wp, jp, sch = make(cfg, path, key=key)
    rng = np.random.default_rng(12)
    base = unit(rng)
    drive(sch, 45, rng, cue_fn=lambda i: near(base, rng, eps=0.05) if i % 2 == 0 else unit(rng))
    facts, _ = generate_facts(40, seed=3)                 # session A: episodes planted, not contracted
    for i, f in enumerate(facts):
        sch.write(f.cue, f.statement.encode(), f.schema, now=500.0 + i)
    assert any(e.state == STATE_DEMOTED for e in j._entries.values())
    assert sum(Reader(jp, 0.0).recall(f, None) for f in facts) == len(facts)   # recallable before the restart
    before = {"snapshot": {k: v for k, v in j.snapshot().items() if k in ("live", "consolidated", "demoted", "evicted", "summaries")},
              "memory": len(sch.memory), "phases": sch.phases, "ca3": len(wp.ca3._codes),
              "writes_since_phase": sch.writes_since_phase}
    del sch, jp, wp, j                                    # destroy every Python object
    gc.collect()

    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    files = list((tmp_path / "journal.payloads").iterdir())
    assert files, "the bytes must live on disk"
    if sealed:
        assert all(l.startswith(LINE_PREFIX) for l in lines)
        assert all(f.read_bytes().startswith(MAGIC) for f in files)
        assert b"works as a" not in path.read_bytes() and not any(b"works as a" in f.read_bytes() for f in files)
        with pytest.raises(ValueError):                   # no key: loud, not garbage
            Journal(path)
        with pytest.raises(Exception):                    # wrong key: loud, not garbage
            Journal(path, key=generate_key())
    else:
        assert not any(l.startswith(LINE_PREFIX) for l in lines)
        assert any(b"works as a" in f.read_bytes() for f in files)

    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    if sealed:
        env["QUANTUM_CORTEX_JOURNAL_KEY"] = key.hex()
    cfg_json = json.dumps({k: getattr(cfg, k) for k in cfg.__dataclass_fields__})
    run = subprocess.run([sys.executable, "-c", SESSION_B, str(path), cfg_json],
                         capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
    assert run.returncode == 0, run.stderr
    r = json.loads(run.stdout.strip().splitlines()[-1])
    assert r["recall"] == 1.0 and r["negctrl"] == 0.0           # session B, in another process
    assert r["sentinel"] == [1.0, 0.0]
    assert r["dup_admitted"] is False                          # CA3 rebuilt: the known is still known
    assert r["snapshot"] == before["snapshot"]
    assert r["memory"] == before["memory"] and r["phases"] == before["phases"] and r["ca3"] == before["ca3"]
    assert r["writes_since_phase"] == before["writes_since_phase"]
    assert r["via_summary"] == [True, True]                    # a demoted source still reads through its summary
    d = r["declaration"]
    assert all(d[c]["survives_restart"] for c in ("structure", "bytes", "index", "associative_memory"))
    assert d["bytes"]["encrypted_at_rest"] is sealed and d["structure"]["encrypted_at_rest"] is sealed
    assert r["next_phase_refused"] is False


def test_a_tampered_payload_or_line_fails_loudly(tmp_path):
    key = generate_key()
    path = tmp_path / "journal.jsonl"
    j = Journal(path, key=key)
    rng = np.random.default_rng(13)
    e = j.write(unit(rng), b"a secret episode", 0.9, "note", now=1.0)
    f = tmp_path / "journal.payloads" / e.pointer
    blob = bytearray(f.read_bytes()); blob[-1] ^= 0x01; f.write_bytes(bytes(blob))
    j2 = Journal(path, key=key)
    with pytest.raises(Exception):
        j2.payloads.get(e.pointer)
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0][:-4] + "AAAA"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(Exception):
        Journal(path, key=key)


def test_lifecycle_declaration_names_the_four_components():
    d = lifecycle_declaration(Journal())                   # memory only: nothing survives, and it says so
    assert set(d) == {"structure", "bytes", "index", "associative_memory"}
    assert not any(v["survives_restart"] for v in d.values())
    assert not d["bytes"]["encrypted_at_rest"]
