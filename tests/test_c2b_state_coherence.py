"""The coherence of the organ's three representations under the events that make them diverge
(the founder, 2026-10-12, 01:42, after the correction of the withdrawal under the cascade; the list of
six tests validated by him before any code, as the recipe requires).

The three representations are those Decision 8 of ADR-007 names: the append only log and the state
its events build (entries, states, saliences, links, references), the persistent associative memory
(the patterns), and the state a second process rebuilds at open from the disk alone. The correction of
2026-10-12 found them diverging on one event the restart test never produced, the eviction of a
summary whose demoted sources leave through the store's cascade; these tests put the equality under
the events that can break it, one invariant each:

  1. nested demotion, then the eviction of the top summary: every source down the hierarchy is evicted
     in the store, withdrawn from the memory, released in the bytes (source texts included), and the
     second process is equal;
  2. a restart after every phase of a drive that demotes and evicts summaries, under both flags: the
     reopened scheduler equals the live one phase after phase (memory, snapshot, references, links,
     sentinels, counters);
  3. near duplicates across the admission threshold and the demotion radius: what the gate refuses is
     absent, every group respects the radius, the cascade and the second process agree (the mechanism
     only; the chickadee's measurement keeps its place in v1.1);
  4. a write of B2 failing between its two payloads (the content written, the source text refused),
     under the three storage policies: no entry, no log line, the orphan released at the next open; in
     memory mode both payloads pending and flushed together;
  5. one source text shared by entries of different states, released at its last reference only, the
     second process equal at every step;
  6. the level 1 identity of the reconstruction probe (entries = live + consolidated + K times the
     summaries alive + evicted) under a hierarchy of summaries.

Test code only: no code of the organ changes here. If a test reveals a gap, it becomes a dated finding
before any correction. Nothing here is a number of the record."""
from __future__ import annotations

import errno
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cortex_c2b import (Journal, STATE_LIVE, STATE_CONSOLIDATED, STATE_DEMOTED, STATE_EVICTED, SCHEMA_SUMMARY,   # noqa: E402
                        POLICY_STOP, POLICY_READ_ONLY, POLICY_MEMORY, MODE_MEMORY, MODE_DURABLE, MODE_READ_ONLY,
                        StorageExhausted, JournalReadOnly, content_hash, CUE_DIM)
from cortex_c2b.crypto import generate_key                                                        # noqa: E402
from cortex_c2b.write_path import WritePath                                                        # noqa: E402
from cortex_c2b.read_path import JournalPath                                                       # noqa: E402
from cortex_c2b.lifecycle import LifecycleScheduler, LifecycleConfig                               # noqa: E402
from cortex_c2b import reconstruction_probe as rp                                                  # noqa: E402


# --------------------------------------------------------------------------- helpers
def unit(rng, dim=CUE_DIM):
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def near(base, rng, eps=0.02):
    v = base + eps * rng.standard_normal(base.shape).astype(np.float32)
    return v / np.linalg.norm(v)


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def open_all(path, key, cfg, now=0.0):
    j = Journal(path, key=key)
    wp, jp = WritePath(j, seed=0), JournalPath(j)
    return j, wp, jp, LifecycleScheduler(j, wp, jp, cfg, now=now)


def live_view(j: Journal, sch: LifecycleScheduler, wp: WritePath) -> dict:
    """What the three representations say, in a form two processes can compare."""
    ids = [x for x in sch.memory.ids()]
    return {"memory_ids": sorted(x for x in ids if x is not None), "memory_rows": len(sch.memory),
            "patterns": {x: sch.memory._K[i].tolist() for i, x in enumerate(ids) if x is not None},
            "states": {k: (e.state, e.salience) for k, e in j._entries.items()},
            "refs": {p: sorted(h) for p, h in j._refs.items()},
            "payloads": sorted(j.payloads._sizes), "summary_of": dict(j.summary_of),
            "summary_sources": {k: list(v) for k, v in j.summary_sources.items()}, "contracted": sorted(j.contracted),
            "snapshot": j.snapshot(), "source_text_counts": j.source_text_counts(), "sourced": j.sourced_entries,
            "phases": sch.phases, "refused": sch.refused, "writes_since_phase": sch.writes_since_phase,
            "sentinel": list(sch.sentinel_probe()), "ca3": len(wp.ca3._codes),
            "source_texts": {k: j.source_text(k) for k in j._entries}}


def assert_same_three_representations(a: dict, b: dict) -> None:
    for k in a:
        if k == "patterns":
            assert set(a[k]) == set(b[k])
            for eid in a[k]:
                assert np.allclose(a[k][eid], b[k][eid], atol=1e-6), eid
        else:
            assert a[k] == b[k], k


def hierarchy(path, key, withdraw: bool):
    """Nine near duplicates written through the gate (eps 0.04 around one base, seed 17: every one admitted,
    the pairwise cosines about 0.88, inside a radius of 0.85), each with a source text; five manual phases
    build three level 1 summaries and one level 2 summary over them, the top one consolidated and alone in
    its state, so that the floor can take it at the next phase (a consolidated entry with two consolidated
    neighbours inside the radius would be grouped again before it could be evicted: demotion precedes
    eviction in the plan). The shape is checked from the journal, never assumed."""
    cfg = LifecycleConfig(period_writes=1000, consolidate_batch=16, demote_k=3, demote_min_cosine=0.85,
                          sentinel_facts=0, decay=1.0, withdraw_at_eviction=withdraw)
    j, wp, jp, sch = open_all(path, key, cfg)
    rng = np.random.default_rng(17)
    base = unit(rng)
    ids = []
    for i in range(9):
        rep = sch.write(near(base, rng, eps=0.04), f"fact {i}".encode(), "note", now=10.0 + i, source=f"note|{i}")
        assert rep.admitted, "the family must pass the gate, or the test is vacuous"
        ids.append(rep.entry.entry_id)
    for now in (100.0, 200.0, 300.0, 400.0, 500.0):          # consolidate, demote, consolidate the summaries, demote them, consolidate the top
        sch.phase(now=now)
    tops = [sid for sid, lvl in j.summary_level.items() if lvl == 2]
    assert len(tops) == 1 and j._entries[tops[0]].state == STATE_CONSOLIDATED
    assert sorted(j.summary_level.values()) == [1, 1, 1, 2]                            # three groups of three, one top over them
    assert all(j.summary_level.get(s, 0) == 1 for s in j.summary_sources[tops[0]])    # nested: the top's sources are summaries
    assert [k for k, e in j._entries.items() if e.state == STATE_CONSOLIDATED] == tops  # the top alone is consolidated
    return cfg, j, wp, jp, sch, ids, tops[0]


def descendants(j: Journal, sid: str) -> list[str]:
    """Every source under a summary, down the hierarchy."""
    out = []
    for s in j.summary_sources.get(sid, []):
        out.append(s); out += descendants(j, s)
    return out


# ============================================================================ 1. nested demotion, then the eviction of the top
@pytest.mark.parametrize("withdraw", [True, False])
def test_1_the_eviction_of_a_summary_of_summaries_takes_every_source_down_the_hierarchy_in_the_three_representations(tmp_path, withdraw):
    key = generate_key()
    path = tmp_path / "journal.jsonl"
    cfg, j, wp, jp, sch, ids, top = hierarchy(path, key, withdraw)
    gone = [top] + descendants(j, top)                        # the top and everything under it, summaries and entries
    kept = [k for k in j._entries if k not in gone]
    gone_entries = [k for k in gone if j._entries[k].schema_id != SCHEMA_SUMMARY]
    assert len(gone) >= 5 and gone_entries, "the top must carry a nested hierarchy, or the test is vacuous"
    assert all(j._entries[k].state in (STATE_CONSOLIDATED, STATE_DEMOTED) for k in gone)   # every one has a pattern
    patterns_before = len(sch.memory)
    assert patterns_before == len(j)                          # every replay verified, one pattern per entry
    j.set_salience(top, 0.05, why="test")                     # the floor takes the top summary at the next phase
    rec = sch.phase(now=600.0)
    assert not rec["refused"] and rec["applied"]["evicted"] == 1   # the phase transitions the top itself; the store cascades
    assert all(j._entries[k].state == STATE_EVICTED for k in gone)                                    # the log's state
    assert all(j._entries[k].state != STATE_EVICTED for k in kept)                                    # and nothing else moved
    assert all(j._entries[k].pointer not in j.payloads for k in gone)                                 # the bytes
    assert all(j._entries[k].source_pointer is not None and j.source_text(k) is None for k in gone_entries)   # the source texts
    assert not any((tmp_path / "journal.payloads" / j._entries[k].source_pointer).exists() for k in gone_entries)
    assert all(j.source_text(k) == f"note|{ids.index(k)}".encode() for k in kept if k in ids)       # the kept ones still read
    assert rec["applied"]["source_released"] == len(gone_entries)                                      # the phase says so
    if withdraw:
        assert rec["applied"]["withdrawn"] == len(gone) and len(sch.memory) == patterns_before - len(gone)   # down the hierarchy
        assert not any(x is not None and j._entries[x].state == STATE_EVICTED for x in sch.memory.ids())
    else:
        assert "withdrawn" not in rec["applied"] and len(sch.memory) == patterns_before            # the rule of rows 36 to 41
    a = live_view(j, sch, wp)
    j2, wp2, jp2, sch2 = open_all(path, key, cfg)                                                   # the second process
    assert_same_three_representations(a, live_view(j2, sch2, wp2))


# ============================================================================ 2. a restart after every phase
@pytest.mark.parametrize("withdraw", [True, False])
def test_2_a_scheduler_reopened_after_every_phase_equals_the_live_one_on_a_drive_that_demotes_and_evicts_summaries(tmp_path, withdraw):
    key = generate_key()
    path = tmp_path / "journal.jsonl"
    cfg = LifecycleConfig(period_writes=6, demote_k=3, demote_min_cosine=0.85, bytes_setpoint=20_000, withdraw_at_eviction=withdraw)
    j, wp, jp, sch = open_all(path, key, cfg)
    rng = np.random.default_rng(1)
    base = unit(rng)
    phases_seen, summaries_evicted, compared = 0, 0, 0
    for i in range(72):
        c = near(base, rng, eps=0.05) if i % 2 == 0 else unit(rng)
        sch.write(c, f"payload {i} {rng.integers(1_000_000)}".encode(), "note", now=100.0 + i,
                  source=f"note|{i}" if i % 3 else None)                 # two writes in three carry a text, one declares it absent
        if sch.phases > phases_seen:                                     # a phase fired on this write: restart now
            phases_seen = sch.phases
            j2, wp2, jp2, sch2 = open_all(path, key, cfg)
            assert_same_three_representations(live_view(j, sch, wp), live_view(j2, sch2, wp2))
            compared += 1
    summaries_evicted = sum(1 for e in j._entries.values() if e.schema_id == SCHEMA_SUMMARY and e.state == STATE_EVICTED)
    assert compared == phases_seen >= 10 and summaries_evicted > 0, "the drive must demote and evict summaries, or the test is vacuous"
    assert any(e.state == STATE_EVICTED and e.source_pointer is not None for e in j._entries.values())
    j2, wp2, jp2, sch2 = open_all(path, key, cfg)                        # and once more at the end, then one phase on each side
    ra, rb = sch.phase(now=10_000.0), sch2.phase(now=10_000.0)
    # the same event, written identically, but for one reading that is not an event of the log by design: the read latency
    # proxy (entries touched per read, p95 over the recent reads) is a running measurement of the process, and reads are
    # not logged; a continued scheduler regulates latency from its own reads only (observed 2026-10-12, not corrected)
    assert {k: v for k, v in ra.items() if k != "latency_p95"} == {k: v for k, v in rb.items() if k != "latency_p95"}
    assert_same_three_representations(live_view(j, sch, wp), live_view(j2, sch2, wp2))


# ============================================================================ 3. near duplicates
def test_3_near_duplicates_across_the_admission_threshold_and_the_demotion_radius(tmp_path):
    key = generate_key()
    path = tmp_path / "journal.jsonl"
    cfg = LifecycleConfig(period_writes=1000, consolidate_batch=64, demote_k=3, demote_min_cosine=0.85,
                          sentinel_facts=0, decay=1.0)
    j, wp, jp, sch = open_all(path, key, cfg)
    rng = np.random.default_rng(17)
    bases = [unit(rng) for _ in range(3)]
    admitted, refused, written_cues = [], [], {}
    for b, base in enumerate(bases):
        # three neighbours inside the radius (eps 0.04, one group each), two near duplicates the gate should refuse
        # (eps 0.004), one far cousin (eps 0.4) outside the radius; a fourth neighbour would be grouped with the family's
        # summary at the next demotion and keep it from the floor, since demotion precedes eviction in the plan
        for k, eps in enumerate((0.04, 0.04, 0.04, 0.004, 0.004, 0.4)):
            c = near(base, rng, eps=eps)
            rep = sch.write(c, f"episode {b}.{k}".encode(), "note", now=float(10 * b + k), source=f"ep|{b}.{k}")
            (admitted if rep.admitted else refused).append((b, k, rep))
            if rep.admitted:
                written_cues[rep.entry.entry_id] = c
    assert admitted and refused, "the family must straddle the admission threshold, or the test is vacuous"
    assert all(r.surprise < wp.thr for _, _, r in refused) and all(r.surprise >= wp.thr for _, _, r in admitted)
    assert len(j) == len(admitted)                                               # what the gate refused is nowhere
    for _, _, r in refused:
        assert r.entry is None
    assert j.source_text_counts() == {"present": len(admitted), "absent": 0}
    sch.phase(now=1000.0)                                                        # consolidate
    sch.phase(now=2000.0)                                                        # demote within the radius
    groups = list(j.summary_sources.values())
    assert groups, "no group formed: the near duplicates must demote, or the test is vacuous"
    for members in groups:
        cues = [j._entries[m].cue for m in members]
        assert len(members) == cfg.demote_k
        assert min(cosine(x, y) for i, x in enumerate(cues) for y in cues[i + 1:]) >= cfg.demote_min_cosine
    for sid in list(j.summary_sources):                                          # a summary never pairs two families
        fams = {int(j.payloads.get(j._entries[m].pointer).decode().split()[1].split(".")[0]) for m in j.summary_sources[sid]}
        assert len(fams) == 1
    far = [r.entry.entry_id for b, k, r in admitted if k == 5]                   # the far cousins stay outside every group
    assert far and not any(f in j.summary_of for f in far)
    sch.phase(now=3000.0)                                                        # the summaries consolidate
    for sid in list(j.summary_sources):
        j.set_salience(sid, 0.05, why="test")                                    # every summary below the floor
    rec = sch.phase(now=4000.0)
    assert not rec["refused"] and rec["applied"]["evicted"] == len(groups)       # the phase transitions the summaries; the store cascades
    assert all(j._entries[m].state == STATE_EVICTED for g in groups for m in g)  # the cascade reached every source
    assert rec["applied"]["source_released"] == sum(len(g) for g in groups)      # the texts of the sources left with them
    assert all(j._entries[f].state != STATE_EVICTED for f in far)                # the far cousins stayed
    assert not any(x is not None and j._entries[x].state == STATE_EVICTED for x in sch.memory.ids())
    j2, wp2, jp2, sch2 = open_all(path, key, cfg)
    assert_same_three_representations(live_view(j, sch, wp), live_view(j2, sch2, wp2))
    for eid, c in written_cues.items():                                          # the gate is rebuilt as well: the known stays known
        if j2._entries[eid].state != STATE_EVICTED:
            assert not wp2.write(c, b"again", "note", now=9_000.0).admitted


# ============================================================================ 4. a write failing between its two payloads
class _SecondPayloadFails:
    """Fail the payload file whose pointer is `target` (the source text) for as long as the disk is
    "full" (`lifted` False), as a full disk would; the content file before it is written normally."""

    def __init__(self, store, target: str):
        self.store, self.target, self.calls, self.fired, self.lifted = store, target, 0, False, False
        self.original = store._write_file

    def __call__(self, h: str, payload: bytes) -> None:
        self.calls += 1
        if h == self.target and not self.lifted:
            self.fired = True
            raise OSError(errno.ENOSPC, "no space left on device (simulated between the two payloads)")
        return self.original(h, payload)


def _ptrs(content: bytes, source: str) -> tuple[str, str]:
    return content_hash(content), content_hash(source.encode("utf-8"))


@pytest.mark.parametrize("policy", [POLICY_STOP, POLICY_READ_ONLY])
def test_4_a_source_payload_refused_after_the_content_leaves_no_entry_no_line_and_the_orphan_is_released_at_open(tmp_path, policy):
    key = generate_key()
    path = tmp_path / "journal.jsonl"
    j = Journal(path, key=key, policy=policy)
    rng = np.random.default_rng(31)
    ok = j.write(unit(rng), b"a first episode", 0.9, "note", now=1.0, source="first|address")
    content, source = b"the episode whose text fails", "failing|address"
    cp, sp = _ptrs(content, source)
    fail = _SecondPayloadFails(j.payloads, sp)
    j.payloads._write_file = fail
    with pytest.raises(StorageExhausted if policy == POLICY_STOP else JournalReadOnly):
        j.write(unit(rng), content, 0.9, "note", now=2.0, source=source)
    assert fail.fired and fail.calls == 2                                     # the content went first, the text failed
    assert len(j) == 1 and all(e.pointer != cp for e in j._entries.values())  # no entry
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1                                                    # no log line for the failed write
    assert (tmp_path / "journal.payloads" / cp).exists()                      # the content file, an orphan for now
    assert not (tmp_path / "journal.payloads" / sp).exists()
    if policy == POLICY_READ_ONLY:
        assert j.mode == MODE_READ_ONLY and j.refused_writes == 1
        assert j.payloads.get(ok.pointer) == b"a first episode"              # reads continue
    r = Journal(path, key=key, policy=policy)                                 # the next open: the orphan is garbage
    assert len(r) == 1 and cp not in r.payloads and not (tmp_path / "journal.payloads" / cp).exists()
    assert r.source_text(ok.entry_id) == b"first|address" and r.source_text_counts() == {"present": 1, "absent": 0}
    assert {p: set(h) for p, h in r._refs.items()} == {ok.pointer: {ok.entry_id}, ok.source_pointer: {ok.entry_id}}


def test_4_under_the_memory_policy_both_payloads_wait_together_and_are_flushed_together(tmp_path):
    key = generate_key()
    path = tmp_path / "journal.jsonl"
    j = Journal(path, key=key, policy=POLICY_MEMORY)
    rng = np.random.default_rng(32)
    content, source = b"an episode written while the disk is full", "memory|address"
    cp, sp = _ptrs(content, source)
    fail = _SecondPayloadFails(j.payloads, sp)
    j.payloads._write_file = fail
    e = j.write(unit(rng), content, 0.9, "note", now=1.0, source=source)
    assert fail.fired and j.mode == MODE_MEMORY                               # the policy: keep writing, declared not durable
    assert e.source_pointer == sp and j.payloads.pending == 1                 # the text waits; the content reached the disk
    assert (tmp_path / "journal.payloads" / cp).exists() and not (tmp_path / "journal.payloads" / sp).exists()
    assert len(j.spill) == 1 and j.source_text(e.entry_id) == source.encode()  # the line waits too; the read is served from memory
    fail.lifted = True                                                        # the disk returns
    e2 = j.write(unit(rng), b"after the disk returned", 0.9, "note", now=2.0, source="after|address")
    assert j.mode == MODE_DURABLE and j.payloads.pending == 0 and len(j.spill) == 0
    assert (tmp_path / "journal.payloads" / sp).exists()                      # flushed with the spilled line
    assert len(j.restorations) == 1 and j.restorations[0]["from_mode"] == MODE_MEMORY
    r = Journal(path, key=key, policy=POLICY_MEMORY)                          # the second process reads both episodes with their texts
    assert len(r) == 2 and r.source_text(e.entry_id) == source.encode() and r.source_text(e2.entry_id) == b"after|address"
    assert r.source_text_counts() == {"present": 2, "absent": 0} and len(r.restorations) == 1
    assert {p: set(h) for p, h in r._refs.items()} == {p: set(h) for p, h in j._refs.items()}


# ============================================================================ 5. one source text, several states
def test_5_a_source_text_shared_across_states_is_released_at_its_last_reference_and_the_replay_agrees(tmp_path):
    path = tmp_path / "journal.jsonl"
    j = Journal(path, plaintext=True)
    rng = np.random.default_rng(41)
    text = "shared|address"
    a = j.write(unit(rng), b"content a", 0.9, "note", now=1.0, source=text)   # will stay live
    b = j.write(unit(rng), b"content b", 0.9, "note", now=2.0, source=text)   # will be consolidated
    c = j.write(unit(rng), b"content c", 0.9, "note", now=3.0, source=text)   # will be demoted under a summary
    sp = a.source_pointer
    assert sp == b.source_pointer == c.source_pointer and len(j.payloads) == 4   # three contents, one text
    j.transition(b.entry_id, STATE_CONSOLIDATED)
    j.transition(c.entry_id, STATE_CONSOLIDATED)
    summary = j.write(unit(rng), b'{"kind":"summary"}', 0.9, SCHEMA_SUMMARY, now=4.0)   # a summary carries no source text
    j.demote(summary.entry_id, [c.entry_id]); j.transition(c.entry_id, STATE_DEMOTED)
    assert j.references(sp) == {a.entry_id, b.entry_id, c.entry_id} and summary.source_pointer is None

    def replay_agrees():
        r = Journal(path, plaintext=True)
        assert {p: set(h) for p, h in r._refs.items()} == {p: set(h) for p, h in j._refs.items()}
        assert sorted(r.payloads._sizes) == sorted(j.payloads._sizes)
        assert {k: r.source_text(k) for k in j._entries} == {k: j.source_text(k) for k in j._entries}
        assert r.source_text_counts() == j.source_text_counts() and r.sourced_entries == j.sourced_entries
    replay_agrees()
    j.transition(a.entry_id, STATE_EVICTED)                                  # the live one goes: the text stays, two holders
    assert sp in j.payloads and j.references(sp) == {b.entry_id, c.entry_id} and j.source_text(a.entry_id) is None
    replay_agrees()
    j.transition(summary.entry_id, STATE_EVICTED)                            # the summary goes and takes c with it: one holder
    assert j._entries[c.entry_id].state == STATE_EVICTED and c.pointer not in j.payloads
    assert sp in j.payloads and j.references(sp) == {b.entry_id} and j.source_text(b.entry_id) == text.encode()
    replay_agrees()
    j.transition(b.entry_id, STATE_EVICTED)                                  # the last reference: the text leaves
    assert sp not in j.payloads and j.references(sp) == set() and len(j.payloads) == 0
    assert j.source_text_counts() == {"present": 0, "absent": 0}
    replay_agrees()


# ============================================================================ 6. the level 1 identity under a hierarchy
@pytest.mark.parametrize("withdraw", [True, False])
def test_6_the_level_1_identity_of_the_reconstruction_probe_holds_under_a_hierarchy_of_summaries(tmp_path, withdraw):
    key = generate_key()
    path = tmp_path / "journal.jsonl"
    cfg, j, wp, jp, sch, ids, top = hierarchy(path, key, withdraw)
    alive_before = sum(1 for e in j._entries.values() if e.schema_id == SCHEMA_SUMMARY and e.state != STATE_EVICTED)
    gone = [top] + descendants(j, top)
    j.set_salience(top, 0.05, why="test")
    sch.phase(now=600.0)                                                     # the top summary evicted, the hierarchy with it
    sch.phase(now=700.0)                                                     # one more phase on what remains
    l1 = rp.level_1_counts(path, key, cfg.demote_k)
    assert l1["phases_checked"] == 7 and l1["exact_at_every_phase"] and l1["worst_defect"] == 0
    per = l1["per_phase"]
    assert per[4]["summaries_alive"] == alive_before >= 3                   # two levels alive before the eviction
    assert per[4]["derived_demoted"] == cfg.demote_k * alive_before == per[4]["demoted"]
    assert per[5]["evicted"] == len(gone) == per[5]["derived_evicted"]      # after the top's eviction
    assert per[5]["summaries_alive"] == alive_before - sum(1 for k in gone if j._entries[k].schema_id == SCHEMA_SUMMARY)
    for p in per:
        assert p["entries"] == p["live"] + p["consolidated"] + p["derived_demoted"] + p["derived_evicted"]
    l4 = rp.level_4_content(j, jp)
    assert l4["holds"] and l4["evicted"] == len(gone)                       # no released file present, no evicted entry read
