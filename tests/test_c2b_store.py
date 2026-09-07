"""C2b Slice A — the store's three laws (ADR-007 D1) + append-only provenance.

  size law   : an entry is <= 1 KB; content is never duplicated (payloads by hash)
  read law   : a read never scans — reading 1 cue among N must not touch ~N entries
  lifecycle  : RES-17 transitions are legal-only; state is inspectable
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cortex_c2b import (Journal, Entry, ENTRY_MAX_BYTES, CUE_DIM,
                        STATE_LIVE, STATE_CONSOLIDATED, STATE_DEMOTED, STATE_EVICTED)


def _cue(seed, dim=CUE_DIM):
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


# --- size law -----------------------------------------------------------------
def test_entry_respects_size_law():
    j = Journal()
    e = j.write(_cue(1), b"an episode payload that could be quite long " * 50,
                salience=0.8, schema_id="person", now=1.0)
    assert e.size_bytes() <= ENTRY_MAX_BYTES, e.size_bytes()
    # the entry holds a POINTER, never the content
    assert e.pointer and b"episode" not in e.to_json().encode()


def test_content_never_duplicated():
    j = Journal()
    payload = b"the same episode written twice"
    e1 = j.write(_cue(1), payload, 0.5, "event", now=1.0)
    e2 = j.write(_cue(2), payload, 0.5, "event", now=2.0)
    assert e1.pointer == e2.pointer          # same content → same hash
    assert len(j.payloads) == 1              # stored ONCE
    assert len(j) == 2                       # two entries point to it


# --- read law: never a scan ---------------------------------------------------
def test_read_never_scans():
    j = Journal()
    N = 2000
    for i in range(N):
        j.write(_cue(i), f"payload {i}".encode(), 0.5, "obj", now=float(i))
    target = _cue(777)
    j._reads_touched = 0
    hits = j.read_by_cue(target)
    # a scan would touch ~N entries; a keyed read touches a small bucket
    assert j._reads_touched < N // 10, \
        f"read touched {j._reads_touched} of {N} entries — that is a scan (read law violated)"
    assert any(e.pointer for e in hits)


def test_read_finds_the_written_entry():
    j = Journal()
    c = _cue(42)
    e = j.write(c, b"find me", 0.9, "decision", now=5.0)
    hits = j.read_by_cue(c)
    assert e.entry_id in {h.entry_id for h in hits}
    assert j.payloads.get(e.pointer) == b"find me"


# --- lifecycle: RES-17 transitions --------------------------------------------
def test_lifecycle_legal_transitions():
    j = Journal()
    e = j.write(_cue(1), b"x", 0.5, "place", now=1.0)
    assert e.state == STATE_LIVE
    j.transition(e.entry_id, STATE_CONSOLIDATED)
    j.transition(e.entry_id, STATE_DEMOTED)
    j.transition(e.entry_id, STATE_EVICTED)
    assert j.get(e.entry_id).state == STATE_EVICTED


def test_lifecycle_rejects_illegal_transition():
    j = Journal()
    e = j.write(_cue(1), b"x", 0.5, "place", now=1.0)
    with pytest.raises(ValueError):
        j.transition(e.entry_id, STATE_DEMOTED)   # live → demoted skips consolidation
    j.transition(e.entry_id, STATE_EVICTED)
    with pytest.raises(ValueError):
        j.transition(e.entry_id, STATE_LIVE)      # evicted is terminal


def test_evicted_entries_are_not_returned_by_reads():
    j = Journal()
    c = _cue(9)
    e = j.write(c, b"gone", 0.5, "event", now=1.0)
    j.transition(e.entry_id, STATE_EVICTED)
    assert all(h.entry_id != e.entry_id for h in j.read_by_cue(c))


def test_snapshot_is_inspectable():
    j = Journal()
    for i in range(5):
        j.write(_cue(i), f"p{i}".encode(), 0.5, "obj", now=float(i))
    j.transition(list(j._entries)[0], STATE_CONSOLIDATED)
    s = j.snapshot()
    assert s["entries"] == 5 and s["live"] == 4 and s["consolidated"] == 1


# --- append-only persistence & replay -----------------------------------------
def test_append_only_log_replays_to_same_state():
    d = tempfile.mkdtemp(); p = Path(d) / "journal.jsonl"
    j = Journal(p)
    e = j.write(_cue(3), b"persist me", 0.7, "person", now=10.0)
    j.transition(e.entry_id, STATE_CONSOLIDATED)
    # every event is one appended line; nothing rewritten
    lines = p.read_text().splitlines()
    assert len(lines) == 2 and '"ev":"write"' in lines[0] and '"ev":"transition"' in lines[1]
    # a fresh Journal replays the log to the same state
    j2 = Journal(p)
    assert len(j2) == 1 and j2.get(e.entry_id).state == STATE_CONSOLIDATED
