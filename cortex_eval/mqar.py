"""cortex_eval.mqar — Multi-Query Associative Recall, by difficulty tiers.

The canonical probe for associative memory (Arora et al. 2023, the Zoology
lineage; validated as the C2 judge by Kimi Linear's KDA experiments, which
sweep sequence length 256→2048 and report the *curve*, not a point).
Re-implemented from scratch (the paper's format is simple); zero dependency.

Format (Arora): a sequence  [k1,v1, ..., kn,vn, SEP, q1, ..., qm]  where each
query qj repeats an earlier key kj, and the model must emit the associated vj.

The founder's principle (2026-09-06): never measure at a single point that may
be flat — sweep a difficulty scale and read the CURVE. Difficulty rises on two
axes Kimi documents: number of kv pairs and sequence length. An architecture is
characterised by *where its curve drops off*; the gap between two architectures'
drop-off points is the clean signal (never flat, because we always read the
regime where they separate). This also calibrates the (a)->(b) bridge: the tiers
tell us at which difficulty to expect signal before probing a language model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Reserved control tokens live at the top of the byte vocab (0..255); MQAR keys
# and values are drawn from a symbol range that leaves a SEP slot free.
SEP = 255  # separator symbol (distinct from keys/values below)


@dataclass
class MQARTier:
    """One difficulty tier."""
    kv_pairs: int
    seq_len: int
    n_symbols: int = 128        # key/value alphabet size (< SEP)
    n_queries: int | None = None  # default = kv_pairs (query every key once)

    def queries(self) -> int:
        return self.n_queries if self.n_queries is not None else self.kv_pairs


def make_batch(tier: MQARTier, batch: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate an MQAR batch → (inputs, targets), both int64 (b, T).

    targets are -100 (ignore) everywhere except at query positions, where they
    hold the value associated with that query's key. This matches a standard
    cross-entropy-with-ignore training/eval setup.
    """
    rng = np.random.default_rng(seed)
    k = tier.kv_pairs
    q = tier.queries()
    # layout length: 2*k (pairs) + 1 (SEP) + q (queries). Pad/truncate to seq_len.
    core = 2 * k + 1 + q
    T = max(tier.seq_len, core)
    X = np.zeros((batch, T), dtype=np.int64)
    Y = np.full((batch, T), -100, dtype=np.int64)

    for b in range(batch):
        # distinct keys, random values (values may repeat across keys)
        keys = rng.choice(tier.n_symbols, size=k, replace=False)
        vals = rng.integers(0, tier.n_symbols, size=k)
        kv = {int(keys[i]): int(vals[i]) for i in range(k)}

        pos = 0
        for i in range(k):
            X[b, pos] = keys[i]; pos += 1
            X[b, pos] = vals[i]; pos += 1
        X[b, pos] = SEP; pos += 1
        # choose which keys to query (with repetition allowed → "multi-query")
        qkeys = rng.choice(keys, size=q, replace=True)
        for j in range(q):
            X[b, pos] = int(qkeys[j])
            Y[b, pos] = kv[int(qkeys[j])]   # supervise the answer at the query slot
            pos += 1
        # remaining positions (if T > core) stay zero, targets stay ignore
    return X, Y


def standard_curriculum() -> list[MQARTier]:
    """The tier ladder (Kimi/Zoology regime): kv ∈ {4,8,16,32}, seq ∈ {128,256,512}.
    Easy tiers give non-zero signal even for the vanilla control (avoids the
    flat-zero); hard tiers separate the architectures.
    """
    tiers = []
    for seq in (128, 256, 512):
        for kv in (4, 8, 16, 32):
            tiers.append(MQARTier(kv_pairs=kv, seq_len=seq))
    return tiers


def accuracy(logits_at_queries: np.ndarray, targets_at_queries: np.ndarray) -> float:
    """Exact-match accuracy over query positions. Inputs are already gathered
    at query slots: preds (N,) argmax and targets (N,)."""
    if len(targets_at_queries) == 0:
        return float("nan")
    return float((logits_at_queries == targets_at_queries).mean())


@dataclass
class MQARResult:
    """The deliverable is the curve: accuracy per tier."""
    per_tier: list[dict] = field(default_factory=list)

    def add(self, tier: MQARTier, acc: float):
        self.per_tier.append({"kv_pairs": tier.kv_pairs, "seq_len": tier.seq_len, "accuracy": acc})

    def dropoff_kv(self, threshold: float = 0.5) -> int | None:
        """The kv level (at the smallest seq_len) where accuracy first drops
        below threshold — a scalar summary of *where the curve breaks*."""
        by_seq = min((t["seq_len"] for t in self.per_tier), default=None)
        if by_seq is None:
            return None
        row = sorted((t for t in self.per_tier if t["seq_len"] == by_seq),
                     key=lambda t: t["kv_pairs"])
        for t in row:
            if t["accuracy"] < threshold:
                return t["kv_pairs"]
        return None  # never dropped below threshold in this range

    def as_dict(self) -> dict:
        return {"per_tier": self.per_tier, "dropoff_kv_@0.5": self.dropoff_kv()}
