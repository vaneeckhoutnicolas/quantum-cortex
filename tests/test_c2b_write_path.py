"""C2b Slice B — the hippocampal write path (ADR-007 D2).

  - a planted duplicate is rejected (redundant, below admission);
  - a novel fact passes;
  - a corrupted input is ADMITTED (maximally surprising) then EVICTED by the
    noise tribunal (surprise alone is not signal — D4.8, two stages, never one);
  - DG separation decorrelates near-duplicates (measurable);
  - each stage is ablatable (the flags change behaviour as the ADR predicts).
"""
import numpy as np

from cortex_c2b import Journal, STATE_LIVE, STATE_EVICTED, CUE_DIM
from cortex_c2b.write_path import WritePath, DentateGyrus, ca1_surprise


def _cue(seed):
    return np.random.default_rng(seed).standard_normal(CUE_DIM).astype(np.float32)


# --- admission: duplicate rejected, novel admitted --------------------------------
def test_novel_fact_is_admitted_and_duplicate_rejected():
    wp = WritePath(Journal(), admission_threshold=0.15)
    c = _cue(1)
    r1 = wp.write(c, b"Vorel-3f2a lives in Tallinn", "person", now=1.0)
    assert r1.admitted and r1.surprise == 1.0          # first ever write: maximal surprise
    r2 = wp.write(c, b"Vorel-3f2a lives in Tallinn", "person", now=2.0)
    assert not r2.admitted and "redundant" in r2.reason  # the same cue again: already known
    assert len(wp.j) == 1


def test_a_different_fact_passes_after_a_first_one():
    wp = WritePath(Journal(), admission_threshold=0.15)
    wp.write(_cue(1), b"fact A", "event", now=1.0)
    r = wp.write(_cue(2), b"fact B", "event", now=2.0)
    assert r.admitted and r.surprise > 0.15
    assert len(wp.j) == 2


# --- the noise tribunal: admitted-then-evicted ------------------------------------
def test_corrupted_input_is_admitted_then_evicted_by_the_tribunal():
    wp = WritePath(Journal(), admission_threshold=0.15)
    good = wp.write(_cue(1), b"a real episode", "event", now=1.0)
    # a corrupted input: pure noise cue, maximally surprising → it IS admitted (stage a)
    noise = np.random.default_rng(999).standard_normal(CUE_DIM).astype(np.float32) * 10
    bad = wp.write(noise, b"garbage", "event", now=2.0)
    assert bad.admitted, "surprise alone admits it — that is the point: admission is not the verdict"
    # the good episode earns outcome credit (RES-11); the noise never does
    wp.credit(good.entry.entry_id, 0.5)
    # stage (b): the tribunal — entries that never earned salience are evicted
    # (set the floor above raw surprise so credit, not surprise, decides)
    for e in wp.j._entries.values():
        e.salience = 0.1 if e.entry_id == bad.entry.entry_id else e.salience
    evicted = wp.noise_tribunal(salience_floor=0.2, now=100.0)
    assert bad.entry.entry_id in evicted
    assert wp.j.get(bad.entry.entry_id).state == STATE_EVICTED
    assert wp.j.get(good.entry.entry_id).state == STATE_LIVE   # the credited one survives


# --- DG: separation is measurable --------------------------------------------------
def test_dg_decorrelates_near_duplicates():
    dg = DentateGyrus(seed=0)
    a = _cue(5)
    b = a + 0.05 * np.random.default_rng(6).standard_normal(CUE_DIM).astype(np.float32)  # near-duplicate
    cos_in = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    sa, sb = dg.separate(a), dg.separate(b)
    cos_out = float(sa @ sb / (np.linalg.norm(sa) * np.linalg.norm(sb) + 1e-8))
    assert cos_in > 0.95                       # very similar in input space
    assert cos_out < cos_in                    # less similar after separation
    assert (sa != 0).sum() == dg.k             # sparse: exactly k winners


# --- CA1: surprise is a computed distance -----------------------------------------
def test_ca1_surprise_is_distance():
    v = _cue(3)
    assert ca1_surprise(None, v) == 1.0                   # no memory: maximal
    assert ca1_surprise(v, v) < 1e-6                      # identical: none
    assert 0.0 <= ca1_surprise(-v, v) <= 1.0              # opposite: clipped to [0,1]


# --- ablations: each stage changes behaviour as predicted --------------------------
def test_without_ca1_everything_is_written():
    wp = WritePath(Journal(), admission_threshold=0.15, use_ca1=False)
    c = _cue(1)
    wp.write(c, b"x", "event", now=1.0)
    r = wp.write(c, b"x", "event", now=2.0)
    assert r.admitted and len(wp.j) == 2       # no comparator → duplicates are not caught


def test_without_dg_codes_are_raw_cues():
    wp = WritePath(Journal(), use_dg=False)
    c = _cue(1)
    assert np.allclose(wp._code(c), c)         # no separation → the raw cue is the code
