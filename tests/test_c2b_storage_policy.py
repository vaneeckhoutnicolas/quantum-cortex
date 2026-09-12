"""Decision 8 (sealed by default) and Decision 9 (durable first, storage policy per
scope): one test per invariant, the recovery, the protocol's persistence flag."""
import errno
import json
import os
from collections import namedtuple

import numpy as np
import pytest

import cortex_c2b
from cortex_c2b import (Journal, PayloadStore, StorageExhausted, JournalReadOnly,
                        POLICY_STOP, POLICY_READ_ONLY, POLICY_MEMORY,
                        MODE_DURABLE, MODE_READ_ONLY, MODE_MEMORY, STATE_EVICTED, STATE_DEMOTED,
                        lifecycle_declaration)
from cortex_c2b.crypto import generate_key
from cortex_c2b.write_path import WritePath
from cortex_c2b.read_path import JournalPath
from cortex_c2b.lifecycle import LifecycleScheduler, LifecycleConfig
from cortex_c2b.hm_protocol import run_hm_protocol


def unit(rng, dim=64):
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class DiskFault:
    """A disk that fails on demand: every line and file write raises ENOSPC while
    `on`; `fail_after` lets the n first writes through (to cut a cascade)."""

    def __init__(self, monkeypatch):
        self.on, self.fail_after, self.calls = False, None, 0
        orig_line, orig_file, orig_probe = Journal._write_line, PayloadStore._write_file, Journal._probe_writable

        def failing(self_, *a, orig=None):
            if self.on:
                self.calls += 1
                if self.fail_after is None or self.calls > self.fail_after:
                    raise OSError(errno.ENOSPC, "No space left on device")
            return orig(self_, *a)
        monkeypatch.setattr(Journal, "_write_line", lambda j, l: failing(j, l, orig=orig_line))
        monkeypatch.setattr(PayloadStore, "_write_file", lambda ps, h, p: failing(ps, h, p, orig=orig_file))
        monkeypatch.setattr(Journal, "_probe_writable", lambda j: failing(j, orig=orig_probe))


def fill(j, n, rng, start=0.0):
    return [j.write(unit(rng), f"payload {i} {rng.integers(10**9)}".encode(), 0.9, "note", now=start + i)
            for i in range(n)]


# ------------------------------------------------------------------ Decision 8
def test_sealed_by_default_key_or_declared_test_scope(tmp_path):
    with pytest.raises(ValueError):
        Journal(tmp_path / "j.jsonl")                          # on disk, no key, no declaration: loud
    j = Journal(tmp_path / "j.jsonl", plaintext=True)           # a declared test scope
    assert lifecycle_declaration(j)["storage"]["plaintext_scope"] is True
    assert lifecycle_declaration(Journal())["storage"]["plaintext_scope"] is False   # memory only needs nothing
    k = generate_key()
    sealed = Journal(tmp_path / "s.jsonl", key=k)
    assert lifecycle_declaration(sealed)["structure"]["encrypted_at_rest"] is True
    hashes = {j.scope_config_hash(), sealed.scope_config_hash(),
              Journal(tmp_path / "p.jsonl", plaintext=True, policy=POLICY_STOP).scope_config_hash(),
              Journal().scope_config_hash()}
    assert len(hashes) == 4                                   # sealing, policy and locus are in the scope hash


# ------------------------------------------------------------------ invariants 1, 3, 4: stop
def test_policy_stop_raises_and_leaves_no_half_state(tmp_path, monkeypatch):
    fault = DiskFault(monkeypatch)
    rng = np.random.default_rng(1)
    j = Journal(tmp_path / "j.jsonl", plaintext=True, policy=POLICY_STOP)
    entries = fill(j, 3, rng)
    fault.on = True
    with pytest.raises(StorageExhausted):
        j.write(unit(rng), b"never lands", 0.9, "note", now=10.0)
    with pytest.raises(StorageExhausted):
        j.transition(entries[0].entry_id, cortex_c2b.STATE_CONSOLIDATED)
    assert len(j) == 3 and entries[0].state == cortex_c2b.STATE_LIVE      # memory untouched
    assert cortex_c2b.content_hash(b"never lands") not in j.payloads       # nothing half registered
    assert j.mode == MODE_DURABLE                                         # stop never degrades, it stops
    fault.on = False
    j.write(unit(rng), b"lands", 0.9, "note", now=11.0)
    r = Journal(tmp_path / "j.jsonl", plaintext=True)
    assert {k: e.state for k, e in r._entries.items()} == {k: e.state for k, e in j._entries.items()}


# ------------------------------------------------------------------ invariants 1, 3, 5, 7: read_only
def test_policy_read_only_refuses_keeps_reading_and_resumes(tmp_path, monkeypatch):
    fault = DiskFault(monkeypatch)
    rng = np.random.default_rng(2)
    j = Journal(tmp_path / "j.jsonl", plaintext=True, policy=POLICY_READ_ONLY)
    entries = fill(j, 3, rng)
    fault.on = True
    with pytest.raises(JournalReadOnly):
        j.write(unit(rng), b"refused", 0.9, "note", now=10.0)
    with pytest.raises(JournalReadOnly):
        j.set_salience(entries[0].entry_id, 0.5)
    assert j.mode == MODE_READ_ONLY and j.refused_writes == 2 and len(j) == 3
    d = lifecycle_declaration(j)
    assert d["storage"]["mode"] == MODE_READ_ONLY and d["storage"]["degraded_reason"]
    assert d["structure"]["survives_restart"] is True         # what is on disk stays durable
    assert j.read_by_cue(entries[1].cue)[0].entry_id == entries[1].entry_id   # reads continue
    assert j.payloads.get(entries[1].pointer)
    fault.on = False
    j.write(unit(rng), b"back", 0.9, "note", now=20.0)         # the disk answers: resumed, the gap recorded
    assert j.mode == MODE_DURABLE and j.refused_writes == 0
    assert j.restorations[-1]["from_mode"] == MODE_READ_ONLY and j.restorations[-1]["refused_writes"] == 2
    r = Journal(tmp_path / "j.jsonl", plaintext=True)
    assert len(r) == 4 and len(r.restorations) == 1


# ------------------------------------------------------------------ invariants 1, 3, 7: memory
def test_policy_memory_spills_declares_not_durable_and_flushes_in_order(tmp_path, monkeypatch):
    fault = DiskFault(monkeypatch)
    rng = np.random.default_rng(3)
    j = Journal(tmp_path / "j.jsonl", plaintext=True, policy=POLICY_MEMORY)
    fill(j, 3, rng)
    fault.on = True
    spilled = fill(j, 2, rng, start=10.0)                       # writes continue
    assert j.mode == MODE_MEMORY and len(j.spill) == 2 and j.payloads.pending == 2 and len(j) == 5
    d = lifecycle_declaration(j)
    assert d["structure"]["survives_restart"] is False and d["bytes"]["survives_restart"] is False
    assert d["storage"]["unpersisted_events"] == 2
    assert j.snapshot()["unpersisted_events"] == 2
    fault.on = False
    last = j.write(unit(rng), b"after", 0.9, "note", now=30.0)  # flush, then the restoration record, then this line
    assert j.mode == MODE_DURABLE and not j.spill and j.payloads.pending == 0
    assert j.restorations[-1]["from_mode"] == MODE_MEMORY
    kinds = [json.loads(l)["ev"] for l in (tmp_path / "j.jsonl").read_text().splitlines()]
    ids = [json.loads(l)["entry"]["entry_id"] for l in (tmp_path / "j.jsonl").read_text().splitlines()
           if json.loads(l)["ev"] == "write"]
    assert kinds[-2:] == ["storage_restored", "write"] and ids[-3:] == [e.entry_id for e in spilled] + [last.entry_id]
    r = Journal(tmp_path / "j.jsonl", plaintext=True)
    assert len(r) == 6 and all(r.payloads.get(e.pointer) for e in r._entries.values())   # nothing lost


def test_spill_buffer_is_bounded_and_escalates_to_read_only(tmp_path, monkeypatch):
    fault = DiskFault(monkeypatch)
    rng = np.random.default_rng(4)
    j = Journal(tmp_path / "j.jsonl", plaintext=True, policy=POLICY_MEMORY, spill_max_events=2)
    fault.on = True
    fill(j, 2, rng)
    with pytest.raises(JournalReadOnly):
        j.write(unit(rng), b"one too many", 0.9, "note", now=5.0)
    assert j.mode == MODE_READ_ONLY and len(j.spill) == 2


# ------------------------------------------------------------------ invariant 5: permission at open
def test_permission_denied_at_open_follows_the_policy(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    path = tmp_path / "j.jsonl"
    j = Journal(path, plaintext=True)
    entries = fill(j, 2, rng)
    del j
    fault = DiskFault(monkeypatch); fault.on = True               # probe, lines and files all refused
    with pytest.raises(StorageExhausted):
        Journal(path, plaintext=True, policy=POLICY_STOP)
    ro = Journal(path, plaintext=True, policy=POLICY_READ_ONLY)
    assert ro.mode == MODE_READ_ONLY and len(ro) == 2 and ro.payloads.get(entries[0].pointer)   # readable
    with pytest.raises(JournalReadOnly):
        ro.write(unit(rng), b"no", 0.9, "note", now=9.0)
    mem = Journal(path, plaintext=True, policy=POLICY_MEMORY)
    assert mem.mode == MODE_MEMORY and len(mem) == 2
    mem.write(unit(rng), b"kept in memory", 0.9, "note", now=9.0)
    assert len(mem.spill) == 1 and lifecycle_declaration(mem)["structure"]["survives_restart"] is False
    with pytest.raises(JournalReadOnly):                          # read_only with nothing to read: loud
        Journal(tmp_path / "absent.jsonl", plaintext=True, policy=POLICY_READ_ONLY)


# ------------------------------------------------------------------ invariant 2: the floor, wall and pressure
Usage = namedtuple("usage", "total used free")


def test_disk_floor_is_a_proactive_wall_and_a_regulated_pressure(tmp_path, monkeypatch):
    free = {"v": 5_000}
    monkeypatch.setattr(cortex_c2b.shutil, "disk_usage", lambda p: Usage(10_000, 10_000 - free["v"], free["v"]))
    rng = np.random.default_rng(6)
    j = Journal(tmp_path / "j.jsonl", plaintext=True, policy=POLICY_STOP, disk_free_floor_bytes=1_000,
                disk_check_every=1)
    j.write(unit(rng), b"fine", 0.9, "note", now=1.0)
    free["v"] = 500
    with pytest.raises(StorageExhausted):                         # under the floor: as if the disk were full
        j.write(unit(rng), b"under the floor", 0.9, "note", now=2.0)
    free["v"] = 5_000
    wp, jp = WritePath(j), JournalPath(j)
    sch = LifecycleScheduler(j, wp, jp, LifecycleConfig(sentinel_facts=0, band=0.25))
    for v, expected in ((1_250, 0.0), (1_125, 0.5), (1_000, 1.0), (5_000, 0.0)):
        free["v"] = v
        assert abs(sch.pressure()["pressure_disk"] - expected) < 1e-9


# ------------------------------------------------------------------ the lifecycle under a policy
def test_lifecycle_reports_refusals_and_lets_stop_propagate(tmp_path, monkeypatch):
    fault = DiskFault(monkeypatch)
    rng = np.random.default_rng(7)
    j = Journal(tmp_path / "ro.jsonl", plaintext=True, policy=POLICY_READ_ONLY)
    sch = LifecycleScheduler(j, WritePath(j), JournalPath(j), LifecycleConfig(sentinel_facts=0))
    fault.on = True
    rep = sch.write(unit(rng), b"x", "note", now=1.0)
    assert not rep.admitted and rep.reason.startswith("refused")
    fault.on = False
    j2 = Journal(tmp_path / "stop.jsonl", plaintext=True, policy=POLICY_STOP)
    sch2 = LifecycleScheduler(j2, WritePath(j2), JournalPath(j2), LifecycleConfig(sentinel_facts=0))
    fault.on = True
    with pytest.raises(StorageExhausted):
        sch2.write(unit(rng), b"x", "note", now=1.0)


def test_a_cascade_cut_by_stop_is_consistent_and_repaired_at_the_next_phase(tmp_path, monkeypatch):
    fault = DiskFault(monkeypatch)
    rng = np.random.default_rng(8)
    j = Journal(tmp_path / "j.jsonl", plaintext=True, policy=POLICY_STOP)
    jp = JournalPath(j)
    cfg = LifecycleConfig(period_writes=1000, demote_k=3, demote_min_cosine=0.95, sentinel_facts=0, decay=1.0)
    sch = LifecycleScheduler(j, WritePath(j), jp, cfg)
    base = unit(rng)
    ids = []
    for i in range(3):
        v = base + 0.01 * rng.standard_normal(64).astype(np.float32); v /= np.linalg.norm(v)
        e = j.write(v, f"src {i}".encode(), 0.9, "note", now=float(i)); jp.on_write(e); ids.append(e.entry_id)
    sch.phase(now=50.0); sch.phase(now=60.0)
    (sid, sources), = j.summary_sources.items()
    fault.on, fault.fail_after = True, 1                          # the summary's line lands, the cascade is cut
    with pytest.raises(StorageExhausted):
        j.transition(sid, STATE_EVICTED)
    fault.on = False
    assert j._entries[sid].state == STATE_EVICTED
    assert sum(j._entries[s].state == STATE_DEMOTED for s in sources) >= 1   # orphaned, consistent with the log
    r = Journal(tmp_path / "j.jsonl", plaintext=True)
    assert {k: e.state for k, e in r._entries.items()} == {k: e.state for k, e in j._entries.items()}
    plan = sch.plan(now=70.0)
    assert set(s for s in sources if j._entries[s].state == STATE_DEMOTED) <= set(plan.evict)
    sch.phase(now=70.0)
    assert all(j._entries[s].state == STATE_EVICTED for s in sources)


def test_orphan_payload_file_is_collected_at_open(tmp_path):
    rng = np.random.default_rng(9)
    j = Journal(tmp_path / "j.jsonl", plaintext=True)
    fill(j, 1, rng)
    stray = tmp_path / "j.payloads" / "0123456789abcdef"
    stray.write_bytes(b"a file no entry references")
    r = Journal(tmp_path / "j.jsonl", plaintext=True)
    assert "0123456789abcdef" not in r.payloads and not stray.exists()


# ------------------------------------------------------------------ the addition to invariant 3: the protocol
def test_hm_protocol_reports_persistence_and_claims_only_across_a_reopen(tmp_path, monkeypatch):
    r = run_hm_protocol()                                          # in memory: the organ level pass, not claimable
    assert r["hm_dissociation_pass"] == 1 and r["persistent"] is False and r["claimable"] == 0
    assert "not claimable" in r["verdict"]
    r = run_hm_protocol(journal_path=tmp_path / "hm.jsonl", key=generate_key())   # session B reopened from disk
    assert r["hm_dissociation_pass"] == 1 and r["persistent"] is True and r["claimable"] == 1
    assert r["storage"] == {"on_disk": True, "mode": MODE_DURABLE, "policy": POLICY_STOP,
                            "sealed": True, "plaintext_scope": False}
    fault = DiskFault(monkeypatch); fault.on = True               # a memory-mode journal never yields a claim
    r = run_hm_protocol(journal_path=tmp_path / "hm2.jsonl", plaintext=True, policy=POLICY_MEMORY)
    assert r["persistent"] is False and r["claimable"] == 0
    fault.on = False
    with pytest.raises(StorageExhausted):                         # a measurement run stops, it does not degrade
        fault.on = True
        run_hm_protocol(journal_path=tmp_path / "hm3.jsonl", plaintext=True, policy=POLICY_STOP)
