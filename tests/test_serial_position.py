"""Serial-position harness (ADR-006 Slice B, Rev7.b): generator + curve semantics."""
import numpy as np

from cortex_eval import (SerialPositionTask, SerialPositionResult,
                         aggregate_by_position, make_serial_batch)
from cortex_eval.serial_position import SEP, PROBE


def test_generator_shape_and_single_supervision():
    task = SerialPositionTask(list_len=16)
    X, Y, probed = make_serial_batch(task, batch=8, seed=0)
    assert X.shape == Y.shape == (8, 16 + 4)
    # exactly one supervised slot per row (the answer)
    assert ((Y != -100).sum(axis=1) == 1).all()
    # SEP and PROBE present at the expected slots
    assert (X[:, 16] == SEP).all() and (X[:, 17] == PROBE).all()
    # probed position in range
    assert (0 <= probed).all() and (probed < 16).all()


def test_oracle_perfect_recall():
    # a perfect memory returns s_p for probe p — recover it from X and score 100%
    task = SerialPositionTask(list_len=12)
    X, Y, probed = make_serial_batch(task, batch=16, seed=1)
    preds = np.array([int(X[i, probed[i]]) for i in range(len(probed))])  # item at position p
    tgts = Y[Y != -100]
    assert (preds == tgts).all(), "oracle must recall the exact item at the probed position"


def test_random_is_low():
    task = SerialPositionTask(list_len=12, n_symbols=128)
    _, Y, _ = make_serial_batch(task, batch=16, seed=2)
    tgts = Y[Y != -100]
    rng = np.random.default_rng(0)
    rand = rng.integers(0, 128, size=len(tgts))
    assert (rand == tgts).mean() < 0.1


def test_u_shape_signature():
    # a synthetic U-shaped curve: high ends, low middle → positive u_shape
    r = SerialPositionResult(list_len=8,
                             acc_by_pos=[0.9, 0.8, 0.4, 0.3, 0.3, 0.4, 0.8, 0.9])
    d = r.as_dict()
    assert d["primacy"] > d["middle"] and d["recency"] > d["middle"]
    assert d["u_shape"] > 0, "U-shape signature must be positive for a real serial-position curve"


def test_flat_curve_has_zero_u_shape():
    r = SerialPositionResult(list_len=8, acc_by_pos=[0.5] * 8)
    assert abs(r.as_dict()["u_shape"]) < 1e-9
