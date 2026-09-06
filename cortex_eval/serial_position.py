"""cortex_eval.serial_position — the serial-position curve (Rev7.b).

Measures the model's primacy/recency profile like an experimental-psychology
protocol: present a list, then probe recall at each list position, and read the
CURVE of accuracy vs position. Cheap, original, and it probes memory *behaviour*
directly — complementary to MQAR (which probes capacity).

Two named effects (RES-16 gives them roles):
  - primacy  : items early in the list recalled better (consolidation priority);
  - recency  : items late in the list recalled better (still in working state).
An architecture is characterised by the *shape* of this curve. The vanilla
control's attention sink is the transformer's native primacy artifact (credited);
an associative C2 layer should reshape the curve — that reshaping is the signal.

Task (single-answer recall by position): a sequence of distinct symbols
  [s_0, s_1, ..., s_{L-1}, SEP, PROBE_p]
where PROBE_p asks "what symbol was at position p?" The model must emit s_p.
Sweeping p over 0..L-1 and averaging over lists yields accuracy[p] — the curve.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# reuse the same reserved separator as MQAR so vocab accounting is shared
SEP = 255
PROBE = 254  # a distinct "query the position" marker


@dataclass
class SerialPositionTask:
    list_len: int = 16          # L — the number of items to remember
    n_symbols: int = 128        # symbol alphabet (< PROBE, SEP)


def make_batch(task: SerialPositionTask, batch: int, seed: int
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a serial-position batch.

    Returns (X, Y, positions):
      X (b, T)         : the list + SEP + PROBE + a position marker token
      Y (b, T)         : -100 except at the answer slot, where it holds s_p
      positions (b,)   : the probed position p for each row (for per-position aggregation)
    The probed position is encoded as a symbol token right after PROBE (positions
    0..L-1 map to reserved-free symbol ids; L is small so this fits under n_symbols).
    """
    rng = np.random.default_rng(seed)
    L = task.list_len
    # layout: L items + SEP + PROBE + 1 position-id + 1 answer slot
    T = L + 4
    X = np.zeros((batch, T), dtype=np.int64)
    Y = np.full((batch, T), -100, dtype=np.int64)
    probed = np.zeros(batch, dtype=np.int64)

    for b in range(batch):
        items = rng.choice(task.n_symbols, size=L, replace=False)
        X[b, :L] = items
        X[b, L] = SEP
        X[b, L + 1] = PROBE
        p = int(rng.integers(0, L))
        probed[b] = p
        # encode the probed position as a small integer token (0..L-1 < n_symbols)
        X[b, L + 2] = p
        # the answer slot is the last position; supervise it with s_p
        X[b, L + 3] = 0
        Y[b, L + 3] = int(items[p])
    return X, Y, probed


@dataclass
class SerialPositionResult:
    """The deliverable is the curve: accuracy per list position."""
    list_len: int
    acc_by_pos: list[float] = field(default_factory=list)

    def primacy_score(self, frac: float = 0.25) -> float:
        """Mean accuracy over the first `frac` of positions."""
        k = max(1, int(self.list_len * frac))
        return float(np.mean(self.acc_by_pos[:k])) if self.acc_by_pos else float("nan")

    def recency_score(self, frac: float = 0.25) -> float:
        k = max(1, int(self.list_len * frac))
        return float(np.mean(self.acc_by_pos[-k:])) if self.acc_by_pos else float("nan")

    def middle_score(self, frac: float = 0.25) -> float:
        n = len(self.acc_by_pos)
        k = max(1, int(self.list_len * frac))
        lo, hi = k, n - k
        return float(np.mean(self.acc_by_pos[lo:hi])) if hi > lo else float("nan")

    def as_dict(self) -> dict:
        return {
            "list_len": self.list_len,
            "acc_by_pos": self.acc_by_pos,
            "primacy": self.primacy_score(),
            "recency": self.recency_score(),
            "middle": self.middle_score(),
            # the classic U-shape signature: ends higher than the middle
            "u_shape": (self.primacy_score() + self.recency_score()) / 2 - self.middle_score()
                       if self.acc_by_pos else float("nan"),
        }


def aggregate_by_position(preds: np.ndarray, targets: np.ndarray,
                          probed: np.ndarray, list_len: int) -> SerialPositionResult:
    """Given per-example predictions/targets and their probed positions,
    compute accuracy at each list position → the serial-position curve."""
    res = SerialPositionResult(list_len=list_len)
    for p in range(list_len):
        mask = probed == p
        if mask.sum() == 0:
            res.acc_by_pos.append(float("nan"))
        else:
            res.acc_by_pos.append(float((preds[mask] == targets[mask]).mean()))
    return res
