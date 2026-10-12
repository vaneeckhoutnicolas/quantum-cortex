"""The revision at the same address (ADR-007, amendment of 2026-10-12; the review's point 3, the founder's
decision A3; validated by him on 2026-10-12, 11:25, the six recommendations in A). The six tests declared in
the amendment before any code, in their order, and a seventh on the probe's mechanics:

  1. the rule            same address, different content: one entry, revised, the read returns the new content,
                         the history holds the old one with its time; same content: refused as redundant, no
                         event; a neighbouring address: the surprise rule as today, no revision;
  2. the identity        by the source text when both carry one, resolved before the surprise; without a text,
                         by the rounded cue; a near cue that is not equal never revises;
  3. the three           after revisions a restart gives the same entries, pointers, histories, references,
     representations     memory and index, and the two processes agree on a phase after the restart;
  4. bytes and lifecycle the historical payloads are held until the eviction of the entry and released with
                         it; a revised entry consolidates, demotes and is evicted as before, its state, salience
                         and pattern unchanged by the revision; under a summary it keeps its history and loses
                         it with the summary's eviction;
  5. persistence         a journal written before the amendment opens and reads as before; the record's probes
     unchanged           see no revision event (the suite);
  6. the Molaison rows   the protocol's files recompute to the digit, no revision in them (the suite, and the
                         protocol run again here);
  7. the probe           `cortex_c2b.revision_probe` on a synthetic journal: the seven measures, the thresholds,
                         the letter, the second process.

Nothing here is a number of the record."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cortex_c2b import (Journal, Entry, STATE_LIVE, STATE_CONSOLIDATED, STATE_DEMOTED, STATE_EVICTED, SCHEMA_SUMMARY,   # noqa: E402
                        POLICY_STOP, CUE_DIM, content_hash, lifecycle_declaration)
from cortex_c2b.crypto import generate_key                                                        # noqa: E402
from cortex_c2b.write_path import WritePath                                                        # noqa: E402
from cortex_c2b.read_path import JournalPath                                                       # noqa: E402
from cortex_c2b.lifecycle import LifecycleScheduler, LifecycleConfig                               # noqa: E402
from cortex_c2b.hm_protocol import generate_facts, run_hm_protocol, _cue_for                       # noqa: E402
from cortex_c2b import revision_probe as rvp                                                       # noqa: E402

MQAR = ROOT / "metrics" / "mqar"


def unit(rng, dim=CUE_DIM):
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def near(base, rng, eps=0.02):
    v = base + eps * rng.standard_normal(base.shape).astype(np.float32)
    return v / np.linalg.norm(v)


def events(path: Path, key=None):
    from cortex_c2b.crypto import open_line
    return [json.loads(open_line(key, l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def open_all(path, key, cfg, now=0.0):
    j = Journal(path, key=key)
    wp, jp = WritePath(j, seed=0), JournalPath(j)
    return j, wp, jp, LifecycleScheduler(j, wp, jp, cfg, now=now)


# ============================================================================ 1. the rule
def test_1_same_address_different_content_is_a_dated_revision_of_the_one_entry(tmp_path):
    key = generate_key()
    path = tmp_path / "journal.jsonl"
    j = Journal(path, key=key, policy=POLICY_STOP)
    wp, jp = WritePath(j, seed=0), JournalPath(j)
    address = "decision|argument:originality"
    first = wp.write(_cue_for(address), b"set aside on 12 Jan", "decision", now=1.0, source=address)
    assert first.admitted and not first.revised
    jp.on_write(first.entry)
    eid, p0 = first.entry.entry_id, first.entry.pointer
    rep = wp.write(_cue_for(address), b"reinstated on 3 Mar", "decision", now=2.0, source=address)
    assert rep.revised and not rep.admitted and rep.entry.entry_id == eid and rep.versions == 2
    assert "revised" in rep.reason and rep.surprise == 0.0
    assert len(j) == 1                                                            # one entry, not two
    e = j.get(eid)
    p1 = e.pointer
    assert p1 == content_hash(b"reinstated on 3 Mar") and p1 != p0
    assert e.state == STATE_LIVE and e.t_written == 1.0                           # the identity kept
    (hit, payload, score), = jp.retrieve(_cue_for(address), k=3)                   # the read returns the current version
    assert hit.entry_id == eid and payload == b"reinstated on 3 Mar" and score >= 0.99
    assert j.history(eid) == [(1.0, p0), (2.0, p1)]                               # the history, dated, oldest first
    assert j.payloads.get(p0) == b"set aside on 12 Jan" and p0 in j.payloads      # the past version held
    assert j.references(p0) == {eid} == j.references(p1)
    ev = events(path, key)
    assert [x["ev"] for x in ev] == ["write", "revision"]
    assert ev[1] == {"ev": "revision", "entry_id": eid, "from": p0, "to": p1, "t": 2.0}
    # same address, same content: refused as redundant, as today, and no event
    again = wp.write(_cue_for(address), b"reinstated on 3 Mar", "decision", now=3.0, source=address)
    assert not again.admitted and not again.revised and "redundant" in again.reason
    assert len(events(path, key)) == 2 and j.history(eid) == [(1.0, p0), (2.0, p1)]
    # a neighbouring address: the surprise rule as today, never a revision
    rng = np.random.default_rng(1)
    neighbour = near(_cue_for(address), rng, eps=0.02)
    rep_n = wp.write(neighbour, b"another entity nearby", "decision", now=4.0, source="decision|argument:nearby")
    assert not rep_n.revised and (rep_n.admitted == (rep_n.surprise >= wp.thr))
    assert len(j) == (2 if rep_n.admitted else 1)
    # a revision back to a past content is a revision, and the history keeps every step
    back = wp.write(_cue_for(address), b"set aside on 12 Jan", "decision", now=5.0, source=address)
    assert back.revised and back.versions == 3 and j.get(eid).pointer == p0
    assert j.history(eid) == [(1.0, p0), (2.0, p1), (5.0, p0)]
    assert j.references(p0) == {eid} == j.references(p1) and p1 in j.payloads     # both versions still held


def test_1_an_evicted_entry_is_never_revised_and_a_summary_never_is():
    j = Journal()
    rng = np.random.default_rng(2)
    c = unit(rng)
    e = j.write(c, b"content", 0.9, "note", now=1.0, source="note|x")
    j.transition(e.entry_id, STATE_EVICTED)
    with pytest.raises(ValueError):
        j.revise(e.entry_id, b"other", now=2.0)
    assert j.at_address(c, "note|x") is None                                        # an evicted entry holds no address
    s = j.write(unit(rng), b'{"kind":"summary"}', 0.9, SCHEMA_SUMMARY, now=3.0)
    assert j.at_address(s.cue, None) is None                                         # a summary is never a write's target
    with pytest.raises(ValueError):
        j.revise(e.entry_id, b"content", now=4.0)                                    # same content: nothing to revise


# ============================================================================ 2. the identity
def test_2_the_identity_is_the_source_text_when_both_carry_one_and_the_rounded_cue_otherwise():
    j = Journal()
    wp = WritePath(j, seed=0)
    rng = np.random.default_rng(3)
    # by the text: two different cues (another encoder) with the same text are the same address
    c1, c2 = unit(rng), unit(rng)
    a = wp.write(c1, b"v1", "note", now=1.0, source="entity|A")
    assert a.admitted
    r = wp.write(c2, b"v2", "note", now=2.0, source="entity|A")
    assert r.revised and r.entry.entry_id == a.entry.entry_id and len(j) == 1
    assert j.get(a.entry.entry_id).cue == a.entry.cue                                 # the entry keeps its cue
    # the identity is resolved before the surprise: a far cue with the same text revises instead of being admitted as new
    assert r.surprise == 0.0
    # without a text: the rounded cue, exactly
    c3 = unit(rng)
    b = wp.write(c3, b"w1", "note", now=3.0)
    assert b.admitted
    same = np.asarray(b.entry.cue, dtype=np.float32)                                  # the cue as the journal keeps it
    r2 = wp.write(same, b"w2", "note", now=4.0)
    assert r2.revised and r2.entry.entry_id == b.entry.entry_id and j.history(b.entry.entry_id)[-1][1] == content_hash(b"w2")
    # a near cue that is not equal never revises (identity, never similarity)
    r3 = wp.write(near(c3, rng, eps=0.001), b"w3", "note", now=5.0)
    assert not r3.revised
    # both carry a text and the texts differ: not the same address, whatever the cue (the surprise then decides, as today)
    r4 = wp.write(np.asarray(a.entry.cue, dtype=np.float32), b"w4", "note", now=6.0, source="entity|other")
    assert not r4.revised and not r4.admitted and "redundant" in r4.reason and len(j) == 2
    # one carries a text and the other does not: the rounded cue decides
    r5 = wp.write(same, b"w5", "note", now=7.0, source="entity|B")
    assert r5.revised and r5.entry.entry_id == b.entry.entry_id
    assert j.get(b.entry.entry_id).source_pointer is None                            # the entry's text, absent, does not change


# ============================================================================ 3. the three representations
def _view(j, sch, wp):
    ids = sch.memory.ids()
    return {"states": {k: (e.state, e.salience, e.pointer) for k, e in j._entries.items()},
            "histories": {k: j.history(k) for k in j._entries}, "refs": {p: sorted(h) for p, h in j._refs.items()},
            "payloads": sorted(j.payloads._sizes), "memory_ids": sorted(x for x in ids if x is not None),
            "snapshot": j.snapshot(), "declaration": lifecycle_declaration(j)["structure"], "phases": sch.phases,
            "writes_since_phase": sch.writes_since_phase, "ca3": len(wp.ca3._codes),
            "texts": {k: j.source_text(k) for k in j._entries}}


def test_3_after_revisions_a_restart_gives_the_same_three_representations_and_agrees_on_a_phase(tmp_path):
    key = generate_key()
    path = tmp_path / "journal.jsonl"
    cfg = LifecycleConfig(period_writes=6, demote_k=3, demote_min_cosine=0.85, bytes_setpoint=200_000)   # phases, no pressure:
    j, wp, jp, sch = open_all(path, key, cfg)                                                           # the entries stay alive
    rng = np.random.default_rng(4)
    facts, _ = generate_facts(40, seed=11)
    for i, f in enumerate(facts):
        sch.write(f.cue, f.statement.encode(), f.schema, now=100.0 + i, source=f.address_text)
    revised = 0
    for i, f in enumerate(facts[::3]):                                               # every third fact revised, through the scheduler
        rep = sch.write(f.cue, f"{f.statement} (revised)".encode(), f.schema, now=200.0 + i, source=f.address_text)
        assert rep.revised; revised += 1
    for f in facts[::9]:                                                             # some twice
        rep = sch.write(f.cue, f"{f.statement} (revised twice)".encode(), f.schema, now=300.0, source=f.address_text)
        assert rep.revised and rep.versions == 3
    assert sum(1 for k in j._entries if len(j.history(k)) > 1) == revised
    assert sch.phases == (40 + 14 + 5) // 6 == 9                                     # the revisions counted as writes for the rhythm (40 // 6 would be 6)
    a = _view(j, sch, wp)
    j2, wp2, jp2, sch2 = open_all(path, key, cfg)
    b = _view(j2, sch2, wp2)
    assert a == b
    for k in j._entries:                                                             # the index returns the current version on both sides
        if j._entries[k].state != STATE_EVICTED and j._entries[k].schema_id != SCHEMA_SUMMARY:
            ha, hb = jp.retrieve(j._entries[k].cue, k=1), jp2.retrieve(j._entries[k].cue, k=1)
            assert ha and hb and ha[0][1] == hb[0][1] == j.payloads.get(j._entries[k].pointer)
    ra, rb = sch.phase(now=10_000.0), sch2.phase(now=10_000.0)
    assert {k: v for k, v in ra.items() if k != "latency_p95"} == {k: v for k, v in rb.items() if k != "latency_p95"}
    assert _view(j, sch, wp) == _view(j2, sch2, wp2)


# ============================================================================ 4. the bytes and the lifecycle
def test_4_the_history_is_held_until_the_eviction_and_the_lifecycle_of_a_revised_entry_is_unchanged(tmp_path):
    key = generate_key()
    path = tmp_path / "journal.jsonl"
    cfg = LifecycleConfig(period_writes=1000, consolidate_batch=64, sentinel_facts=0, decay=1.0)
    j, wp, jp, sch = open_all(path, key, cfg)
    rng = np.random.default_rng(5)
    c = unit(rng)
    rep = sch.write(c, b"version one", "note", now=1.0, source="note|one")
    eid = rep.entry.entry_id
    sch.phase(now=10.0)                                                              # consolidated, one pattern
    e = j.get(eid)
    assert e.state == STATE_CONSOLIDATED and len(sch.memory) == 1
    salience_before, pattern_before = e.salience, sch.memory._K.copy()
    r1 = sch.write(c, b"version two", "note", now=20.0, source="note|one")
    r2 = sch.write(c, b"version three", "note", now=30.0, source="note|one")
    assert r1.revised and r2.revised
    assert e.state == STATE_CONSOLIDATED and e.salience == salience_before            # unchanged by the revisions
    assert len(sch.memory) == 1 and np.array_equal(sch.memory._K, pattern_before)   # the pattern untouched
    ptrs = [p for _, p in j.history(eid)]
    assert len(ptrs) == 3 and all(p in j.payloads for p in ptrs)                      # every version held
    assert all((tmp_path / "journal.payloads" / p).exists() for p in ptrs)
    assert j.regulated_bytes() >= sum(len(b) for b in (b"version one", b"version two", b"version three"))
    j.set_salience(eid, 0.05, why="test")                                            # the floor takes it at the next phase
    rec = sch.phase(now=40.0)
    assert rec["applied"]["evicted"] == 1 and j.get(eid).state == STATE_EVICTED
    assert all(p not in j.payloads for p in ptrs) and all(not (tmp_path / "journal.payloads" / p).exists() for p in ptrs)
    assert j.source_text(eid) is None and len(sch.memory) == 0                       # the text and the pattern gone with it
    assert j.history(eid) == [(1.0, ptrs[0]), (20.0, ptrs[1]), (30.0, ptrs[2])]        # the history stays as a record of the tombstone
    assert rec["applied"]["source_released"] == 1
    r = Journal(path, key=key)                                                        # the replay agrees on the tombstone and the bytes
    assert r.history(eid) == j.history(eid) and len(r.payloads) == 0 and r.get(eid).state == STATE_EVICTED


def test_4_a_revised_entry_demoted_under_a_summary_keeps_its_history_and_loses_it_with_the_summary(tmp_path):
    key = generate_key()
    path = tmp_path / "journal.jsonl"
    cfg = LifecycleConfig(period_writes=1000, consolidate_batch=16, demote_k=3, demote_min_cosine=0.85,
                          sentinel_facts=0, decay=1.0)
    j, wp, jp, sch = open_all(path, key, cfg)
    rng = np.random.default_rng(17)
    base = unit(rng)
    ids = []
    for i in range(3):
        rep = sch.write(near(base, rng, eps=0.04), f"fact {i}".encode(), "note", now=10.0 + i, source=f"note|{i}")
        assert rep.admitted; ids.append(rep.entry.entry_id)
    rev = sch.write(np.asarray(j.get(ids[0]).cue, dtype=np.float32), b"fact 0 revised", "note", now=20.0, source="note|0")
    assert rev.revised and len(j.history(ids[0])) == 2
    sch.phase(now=100.0); sch.phase(now=200.0)                                       # consolidated, then demoted into one summary
    assert j.get(ids[0]).state == STATE_DEMOTED and j.summary_of[ids[0]] in j.summary_sources
    assert len(j.history(ids[0])) == 2 and all(p in j.payloads for _, p in j.history(ids[0]))   # the history survives the demotion
    (eff, payload, _), = jp.retrieve(np.asarray(j.get(ids[0]).cue, dtype=np.float32), k=1)
    assert eff.schema_id == SCHEMA_SUMMARY and payload == b"fact 0 revised"           # read through the summary: the current version
    sid = j.summary_of[ids[0]]
    sch.phase(now=300.0)                                                             # the summary consolidates
    j.set_salience(sid, 0.05, why="test")
    sch.phase(now=400.0)                                                             # the summary evicted, its sources with it
    assert j.get(ids[0]).state == STATE_EVICTED
    assert all(p not in j.payloads for _, p in j.history(ids[0]))                     # the history went with the cascade
    r = Journal(path, key=key)
    assert r.history(ids[0]) == j.history(ids[0]) and sorted(r.payloads._sizes) == sorted(j.payloads._sizes)


# ============================================================================ 5. persistence unchanged
def test_5_a_journal_written_before_the_amendment_opens_reads_and_declares_as_before(tmp_path):
    path = tmp_path / "journal.jsonl"
    j = Journal(path, plaintext=True)
    rng = np.random.default_rng(6)
    ids = [j.write(unit(rng), f"episode {i}".encode(), 0.7, "note", now=float(i)).entry_id for i in range(5)]
    assert all(x["ev"] == "write" for x in events(path))                                # no revision event in a journal of before
    r = Journal(path, plaintext=True)
    for i, eid in enumerate(ids):
        assert r.history(eid) == [(float(i), r.get(eid).pointer)]                         # one version, the write
        assert r.payloads.get(r.get(eid).pointer) == f"episode {i}".encode()
    assert lifecycle_declaration(r)["structure"]["revisions"] == {"entries_revised": 0, "past_versions": 0}
    assert {p: set(h) for p, h in j._refs.items()} == {p: set(h) for p, h in r._refs.items()}


# ============================================================================ 6. the Molaison rows
def test_6_the_protocol_recomputes_to_the_digit_and_plants_no_revision():
    r = run_hm_protocol(seed=0)
    file = json.loads((MQAR / "hm-protocol-2026-09-08.json").read_text(encoding="utf-8"))
    for k in ("hm_recall_on", "hm_recall_off", "hm_gap", "hm_negctrl_rate", "hm_negctrl_on", "hm_negctrl_off_raw",
              "hm_skill_delta", "run_valid", "hm_dissociation_pass"):
        assert r[k] == file[k], k
    assert r["generator"]["admitted"] == 200


# ============================================================================ 7. the probe's mechanics
@pytest.fixture(scope="module")
def probe_out(tmp_path_factory):
    return rvp.run(n=60, seed=3)


def test_7_the_probe_reports_the_seven_measures_their_thresholds_and_the_letter(probe_out):
    out = probe_out
    m, t = out["measures"], out["thresholds"]
    assert set(m) == {"current_returned", "past_returned", "unrevised_intact", "history_exact", "entries_exact",
                      "memory_unchanged", "agreement"} == set(t)
    assert out["config"]["n_facts"] == 60 and out["config"]["revised"] == 30 and out["config"]["revised_twice"] == 10
    assert out["config"]["admitted"] == 60 and out["config"]["revisions_made"] == 40
    assert t == {"current_returned": 1.0, "past_returned": 0.0, "unrevised_intact": 1.0, "history_exact": True,
                 "entries_exact": True, "memory_unchanged": True, "agreement": True}
    assert out["process_b"]["process"].startswith("new")
    assert out["verdict"] == out["readings"]["letter"] and out["readings"]["letter"][:1] in "abcd"
    assert m["current_returned"] == 1.0 and m["past_returned"] == 0.0 and m["unrevised_intact"] == 1.0
    assert m["history_exact"] and m["entries_exact"] and m["memory_unchanged"] and m["agreement"]
    assert out["readings"]["letter"].startswith("a:")
    assert out["memory"]["patterns_before"] == out["memory"]["patterns_after"] == 60
    assert len(rvp.summary_lines(out)) == 5


def test_7_the_command_line_never_overwrites_a_retained_file(tmp_path):
    kept = tmp_path / "revision-probe-seed3.json"
    kept.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="never overwritten"):
        rvp._cli(["--n", "60", "--seed", "3", "--out", str(tmp_path)])   # refused before any run
    assert kept.read_text(encoding="utf-8") == "{}\n"
