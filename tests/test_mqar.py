"""MQAR harness (ADR-006 Slice B): generator correctness + curve semantics.

A perfect associative lookup must score 100% (the task is well-posed); a random
guesser must score ~1/n_symbols (the task is non-trivial). No model is trained
here — we validate the harness itself.
"""
import numpy as np

from cortex_eval import MQARTier, MQARResult, make_batch, standard_curriculum, accuracy, SEP


def _oracle_predict(X, tier):
    """A perfect memory: for each query position, return the value that followed
    that key earlier in the sequence. Scores the harness's ceiling."""
    b, T = X.shape
    preds, tgts_pos = [], []
    for i in range(b):
        seq = X[i]
        sep = int(np.where(seq == SEP)[0][0])
        # build key->value from the pairs before SEP
        kv = {}
        j = 0
        while j < sep - 1:
            kv[int(seq[j])] = int(seq[j + 1]); j += 2
        # answer each query after SEP
        for p in range(sep + 1, T):
            key = int(seq[p])
            if key in kv:
                preds.append(kv[key]); tgts_pos.append(p)
    return preds


def test_generator_shapes_and_supervision():
    tier = MQARTier(kv_pairs=8, seq_len=128)
    X, Y = make_batch(tier, batch=4, seed=0)
    assert X.shape == Y.shape == (4, 128)
    # exactly `queries` supervised positions per row (targets != -100)
    supervised = (Y != -100).sum(axis=1)
    assert (supervised == tier.queries()).all(), supervised


def test_task_is_wellposed_oracle_scores_perfect():
    tier = MQARTier(kv_pairs=16, seq_len=256)
    X, Y = make_batch(tier, batch=8, seed=1)
    preds = np.array(_oracle_predict(X, tier))
    # gather the true targets at query positions
    tgts = Y[Y != -100]
    assert len(preds) == len(tgts)
    assert accuracy(preds, tgts) == 1.0, "a perfect memory must solve MQAR exactly"


def test_task_is_nontrivial_random_scores_low():
    tier = MQARTier(kv_pairs=16, seq_len=256, n_symbols=128)
    _, Y = make_batch(tier, batch=8, seed=2)
    tgts = Y[Y != -100]
    rng = np.random.default_rng(0)
    rand = rng.integers(0, 128, size=len(tgts))
    assert accuracy(rand, tgts) < 0.1, "random guessing must be far from solving MQAR"


def test_curriculum_covers_the_ladder():
    tiers = standard_curriculum()
    kvs = sorted({t.kv_pairs for t in tiers})
    seqs = sorted({t.seq_len for t in tiers})
    assert kvs == [4, 8, 16, 32] and seqs == [128, 256, 512]


def test_curve_dropoff_summary():
    # a synthetic curve that is perfect until kv=16 then collapses
    r = MQARResult()
    for kv, acc in [(4, 1.0), (8, 0.99), (16, 0.3), (32, 0.05)]:
        r.add(MQARTier(kv_pairs=kv, seq_len=128), acc)
    assert r.dropoff_kv(threshold=0.5) == 16, "dropoff must report where the curve breaks"
