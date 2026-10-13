"""C2b Slice C — the journal as path 4 of the RES-18 router (ADR-007 D3).

  - the EXISTING RouterV1 routes to path 4 with no code change (retro-compat);
  - the floor guarantee holds with the journal on/off (never degrade);
  - the ANN read never scans (touches buckets, not the store);
  - the journal wins a span it holds and loses one it does not.
"""
import numpy as np
import pytest

from cortex_c2 import RouterV1, PATH_CONTROL, PATH_HOPFIELD, PATH_NAMES
from cortex_c2b import Journal, CUE_DIM, STATE_EVICTED
from cortex_c2b.write_path import WritePath
from cortex_c2b.read_path import (JournalPath, CueIndex, PATH_JOURNAL, path_scores_with_journal, LEGACY_GEOMETRY,
                                  ADOPTED_GEOMETRY)


def _cue(seed):
    return np.random.default_rng(seed).standard_normal(CUE_DIM).astype(np.float32)


# --- retro-compat: the existing router accepts a 4th path untouched --------------
def test_existing_router_routes_to_journal_path_without_change():
    r = RouterV1()
    d = r.route(path_scores=path_scores_with_journal(0.10, 0.12, 0.11, 0.30))
    assert d.path == PATH_JOURNAL                      # index 3 wins
    assert PATH_NAMES[PATH_JOURNAL] == "journal"       # vocabulary extended, not replaced


# --- floor guarantee with the journal on/off -------------------------------------
def test_floor_holds_when_journal_is_below_control():
    r = RouterV1()
    d = r.route(path_scores=path_scores_with_journal(0.30, 0.10, 0.10, 0.05))
    assert d.path == PATH_CONTROL                      # journal below floor → control


def test_journal_off_is_just_the_three_path_router():
    r = RouterV1()
    on = r.route(path_scores=path_scores_with_journal(0.10, 0.25, 0.11, 0.0)).path
    off = r.route(path_scores=[0.10, 0.25, 0.11]).path
    assert on == off == PATH_HOPFIELD                  # journal at 0 changes nothing


# --- the ANN read never scans --------------------------------------------------------
@pytest.mark.parametrize("geometry", [LEGACY_GEOMETRY, ADOPTED_GEOMETRY])
def test_ann_read_never_scans(geometry):
    """The read law, under both declared geometries since the adoption of C2b (ADR-008, 2026-10-13): the test the
    amendment names and moves. A query touches fewer than a fifth of 3 000 Gaussian cues."""
    idx = CueIndex(dim=CUE_DIM, n_tables=geometry["n_tables"], n_bits=geometry["n_bits"], seed=geometry["seed"], signature=geometry["signature"])
    N = 3000
    for i in range(N):
        idx.add(f"e{i}", _cue(i))
    idx.touched = 0
    hits = idx.query(_cue(1234), k=3)
    assert idx.touched < N // 5, f"query touched {idx.touched}/{N} — that is a scan"
    assert len(hits) >= 1


def test_a_passed_index_is_the_one_the_path_holds_even_when_empty():
    """Found 2026-10-12 by the committee on the fly's declaration (register Rev108): an empty
    CueIndex is falsy (it has a length), so `index or CueIndex(...)` silently replaced the index
    the retrieval probe of row 38 passed by a default cosine index; the row's "fly" ranking was
    the cosine ranking twice. The path must keep the object it is given."""
    from cortex_c2b import Journal
    idx = CueIndex(dim=CUE_DIM, seed=0, rank="fly")
    assert len(idx) == 0 and not idx                            # the trap: an empty index is falsy
    jp = JournalPath(Journal(), index=idx, seed=0)
    assert jp.index is idx and jp.index.rank == "fly" and jp.index.fly_planes is not None
    jp2 = JournalPath(Journal(), index=None, seed=0)
    assert jp2.index is not idx and jp2.index.rank == "cosine"


def test_ann_finds_a_stored_cue():
    idx = CueIndex(dim=CUE_DIM, seed=0)
    for i in range(500):
        idx.add(f"e{i}", _cue(i))
    hits = idx.query(_cue(77), k=3)
    assert hits and hits[0][0] == "e77" and hits[0][1] > 0.99


# --- the journal wins what it holds, loses what it does not ------------------------
def test_journal_path_scores_known_high_and_unknown_low():
    j = Journal(); wp = WritePath(j); jp = JournalPath(j)
    known = _cue(5)
    rep = wp.write(known, b"Vorel-3f2a: born in Tartu", "person", now=1.0)
    jp.on_write(rep.entry)
    assert jp.score(known) > 0.9                       # it holds this span
    assert jp.score(_cue(999)) < 0.5                   # it does not hold this one
    # and the router uses it accordingly (control at 0.2 on both spans)
    r = RouterV1()
    d_known = r.route(path_scores=path_scores_with_journal(0.2, 0.1, 0.1, jp.score(known)))
    d_unknown = r.route(path_scores=path_scores_with_journal(0.2, 0.1, 0.1, jp.score(_cue(999))))
    assert d_known.path == PATH_JOURNAL
    assert d_unknown.path == PATH_CONTROL              # floor: journal cannot degrade


def test_evicted_entries_do_not_score():
    j = Journal(); wp = WritePath(j); jp = JournalPath(j)
    c = _cue(8)
    rep = wp.write(c, b"gone soon", "event", now=1.0); jp.on_write(rep.entry)
    j.transition(rep.entry.entry_id, STATE_EVICTED)
    assert jp.score(c) == 0.0
    assert jp.retrieve(c) == []
