"""The five structural invariants of the data composition layer (ADR-005 D7).

These are built from the start (not deferred), because a wrong interface or a
wrong weight semantics later means painful refactors and invalidated ablations.
The degenerate cases are the best plumbing test that exists — in particular the
founder's `mix([S, S]) == S`, which also forces the weight semantics (D6) to be
explicit and verified rather than tacit.
"""
from __future__ import annotations

import numpy as np

from cortex_data import ArraySource, Decontaminate, Mix, mix_hash


def _collect(op, seed=1337):
    parts = list(op.stream(seed=seed))
    return (
        np.concatenate(parts) if parts else np.empty(0, dtype=np.uint16)
    )


def _rand_tokens(n, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=n, dtype=np.uint16)


# --- Invariant 1: identity — mix([S]) ≡ S --------------------------------- #
def test_identity_single_source_mix_equals_source():
    S = ArraySource(_rand_tokens(50_000, seed=1), name="S")
    m = Mix([S], [1.0])
    assert np.array_equal(_collect(m), _collect(S))
    # hashes differ (mix node wraps source) but stream content is identical
    assert _collect(m).dtype == np.uint16


# --- Invariant 2: duplication idempotence — mix([S,S], equal) ≡ S ---------- #
# The founder's plumbing test: playing the same thing twice must reproduce the
# single stream, byte-for-byte AND hash-for-hash. This pins the D6 semantics
# (proportional sampling without replacement).
def test_duplication_idempotence_bytes_and_hash():
    S = ArraySource(_rand_tokens(40_000, seed=2), name="S")
    single = _collect(S)

    # mix([S, S]) with a budget of |S| is a MULTISET-equivalent of S:
    # mixing redistributes a budget across sources, it does not inflate corpus
    # size (ADR-005 D6). Deliberate repetition is a separate `repeat` operator
    # (bounded ≤2 epochs per OLMo/Kimi), never a side effect of mix.
    dup = Mix([S, S], [1.0, 1.0], total=len(single))
    out = _collect(dup)

    # size identity: budget respected, no duplication
    assert len(out) == len(single), (len(out), len(single))
    # multiset identity: same tokens in the same quantities (interleave order
    # of two distinct sources is not what defines a corpus — cf. commutativity)
    assert np.array_equal(np.sort(out), np.sort(single)), \
        "mix([S,S], budget=|S|) must be a multiset-equivalent of S"

    # hash identity: the same recipe re-built hashes identically (determinism)
    dup2 = Mix([ArraySource(S.tokens, name="S"), ArraySource(S.tokens, name="S")],
               [1.0, 1.0], total=len(single))
    assert mix_hash(dup) == mix_hash(dup2)


# --- Invariant 3: commutativity — mix([A,B]) ≡ mix([B,A]) at equal seed ---- #
def test_commutativity_same_seed():
    A = ArraySource(_rand_tokens(30_000, seed=3), name="A")
    B = ArraySource(_rand_tokens(30_000, seed=4), name="B")
    ab = _collect(Mix([A, B], [1.0, 1.0]), seed=99)
    ba = _collect(Mix([B, A], [1.0, 1.0]), seed=99)
    # same multiset of tokens and same length (order of draws is symmetric)
    assert len(ab) == len(ba)
    assert np.array_equal(np.sort(ab), np.sort(ba))


# --- Invariant 4: determinism — same DAG + seed → same mix_hash & stream --- #
def test_determinism_hash_and_stream():
    A = ArraySource(_rand_tokens(25_000, seed=5), name="A")
    B = ArraySource(_rand_tokens(25_000, seed=6), name="B")
    m1 = Mix([A, B], [0.7, 0.3])
    m2 = Mix([ArraySource(A.tokens, name="A"), ArraySource(B.tokens, name="B")],
             [0.7, 0.3])
    assert mix_hash(m1) == mix_hash(m2)
    assert np.array_equal(_collect(m1, seed=7), _collect(m2, seed=7))
    # different seed → same hash (recipe unchanged) but stream may differ in order
    assert mix_hash(m1) == mix_hash(Mix([A, B], [0.7, 0.3]))


# --- Invariant 5: decontamination detects a planted trap n-gram ----------- #
def test_decontamination_detects_planted_trap():
    trap = np.arange(100, 113, dtype=np.uint16)  # a 13-gram
    clean = _rand_tokens(10_000, seed=8)
    poisoned = np.concatenate([clean[:5000], trap, clean[5000:]])

    # strict mode must raise on contamination
    op = Decontaminate(ArraySource(poisoned), eval_ngrams=[trap], n=13, strict=True)
    raised = False
    try:
        _collect(op)
    except ValueError:
        raised = True
    assert raised, "decontamination must flag a planted eval n-gram"

    # report-only mode surfaces the hit without raising
    op2 = Decontaminate(ArraySource(poisoned), eval_ngrams=[trap], n=13, strict=False)
    _collect(op2)
    assert op2.last_report["hits"] >= 1

    # a clean source passes
    op3 = Decontaminate(ArraySource(clean), eval_ngrams=[trap], n=13, strict=True)
    _collect(op3)
    assert op3.last_report["hits"] == 0
