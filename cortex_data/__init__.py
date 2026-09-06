"""cortex_data — the data composition layer (ADR-005, Slice 1).

A data mix is a DAG of typed, composable operators. Each operator is a
lazy, pull-based iterator over byte tokens (uint16 in 0..255 plus reserved
oracle ids); nothing intermediate is materialised. Every node has a
deterministic hash; the sink hash is the mix_hash — two runs are comparable
iff they share it (ADR-005 D2). The engine is written from scratch (D8) and
orchestrates licensed sources via credited recipes.

Slice 1 implements exactly what N2 needs: the operator interface, the hashed
DAG manifest, lazy `source` / `decontaminate` / `mix`, compilation to a
uint16 .bin the trainer already reads, and the five-invariant harness (tests).
Memoisation, Bloom-at-scale and the data-β estimator are later slices.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

TOKEN_DTYPE = np.uint16  # matches train.py's memmap
_CHUNK = 1 << 16         # streaming granularity (tokens)


def _hash_obj(obj) -> str:
    """Deterministic short hash of a JSON-able object."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Operator base                                                               #
# --------------------------------------------------------------------------- #
class Op:
    """A typed, composable, lazy operator over a stream of uint16 tokens.

    Subclasses implement `stream()` (a fresh iterator over token chunks,
    each a 1-D uint16 ndarray) and `_spec()` (the JSON-able description used
    for hashing). `node_hash` is f(spec, input hashes) — content addressing.
    """

    op_type: str = "op"
    inputs: Sequence["Op"] = ()

    def _spec(self) -> dict:
        raise NotImplementedError

    def stream(self, *, seed: int) -> Iterator[np.ndarray]:
        raise NotImplementedError

    # -- hashing -------------------------------------------------------------
    def node_hash(self) -> str:
        return _hash_obj(
            {
                "op": self.op_type,
                "spec": self._spec(),
                "inputs": [c.node_hash() for c in self.inputs],
            }
        )

    def manifest(self) -> dict:
        """The DAG rooted at this node, as a JSON-able recipe (ADR-005 D2)."""
        return {
            "op": self.op_type,
            "spec": self._spec(),
            "hash": self.node_hash(),
            "inputs": [c.manifest() for c in self.inputs],
        }

    # -- materialisation -----------------------------------------------------
    def to_bin(self, path: str | Path, *, seed: int, max_tokens: int | None = None) -> int:
        """Stream this node to a uint16 .bin the trainer reads. Returns token count.

        Only the sink is materialised (ADR-005 D3: no intermediate files).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with open(path, "wb") as fh:
            for chunk in self.stream(seed=seed):
                if max_tokens is not None and written + len(chunk) > max_tokens:
                    chunk = chunk[: max_tokens - written]
                if len(chunk):
                    fh.write(np.ascontiguousarray(chunk, dtype=TOKEN_DTYPE).tobytes())
                    written += len(chunk)
                if max_tokens is not None and written >= max_tokens:
                    break
        return written


# --------------------------------------------------------------------------- #
# Sources                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class BinSource(Op):
    """A licensed corpus already compiled to a uint16 .bin, content-addressed.

    `dataset_id` and `license` are declared for provenance/attribution (D8).
    `content_sha` pins the exact bytes; if absent it is computed on first use.
    """

    path: str
    dataset_id: str
    license: str
    slice: str = ""
    content_sha: str | None = None
    op_type: str = field(default="source", init=False)

    def _resolve(self) -> Path:
        return Path(self.path)

    def _sha(self) -> str:
        if self.content_sha:
            return self.content_sha
        p = self._resolve()
        if not p.exists():
            # hash the spec instead so the DAG is still deterministic offline
            return "unresolved:" + _hash_obj({"path": self.path, "id": self.dataset_id})
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        self.content_sha = h.hexdigest()[:16]
        return self.content_sha

    def _spec(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "license": self.license,
            "slice": self.slice,
            "content_sha": self._sha(),
        }

    def stream(self, *, seed: int) -> Iterator[np.ndarray]:
        p = self._resolve()
        if not p.exists():
            raise FileNotFoundError(f"source .bin missing: {p} ({self.dataset_id})")
        arr = np.memmap(p, dtype=TOKEN_DTYPE, mode="r")
        for i in range(0, len(arr), _CHUNK):
            yield np.asarray(arr[i : i + _CHUNK])


@dataclass
class ArraySource(Op):
    """An in-memory token array — used by tests and small deterministic inputs."""

    tokens: np.ndarray
    name: str = "array"
    op_type: str = field(default="source", init=False)

    def _spec(self) -> dict:
        return {"name": self.name, "content_sha": _hash_obj(self.tokens.tolist())}

    def stream(self, *, seed: int) -> Iterator[np.ndarray]:
        arr = np.asarray(self.tokens, dtype=TOKEN_DTYPE)
        for i in range(0, len(arr), _CHUNK):
            yield arr[i : i + _CHUNK]


# --------------------------------------------------------------------------- #
# Combinators                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class Mix(Op):
    """Proportional sampling WITHOUT replacement (ADR-005 D6).

    Each source contributes a fraction of the final stream proportional to its
    weight, drawn without replacement from that source's tokens. This makes the
    identity and duplication invariants (D7) hold, and is the seam a future
    data-β resolver (RES-14) slots into: weights are *resolved* values here.

    Determinism: block order and per-source sampling are driven by `seed`.
    Weight resolution is a constant today (D5); the resolver seam is the
    `_resolved_weights()` method, overridable later without touching callers.
    """

    sources: Sequence[Op]
    weights: Sequence[float]
    block: int = 4096       # interleave granularity (tokens per draw)
    total: int | None = None  # target output budget; default = weighted mean of
                              # source lengths (so mixing does not inflate size)
    op_type: str = field(default="mix", init=False)

    def __post_init__(self):
        if len(self.sources) != len(self.weights):
            raise ValueError("mix: sources and weights length mismatch")
        if not self.sources:
            raise ValueError("mix: at least one source required")
        self.inputs = tuple(self.sources)

    def _resolved_weights(self) -> list[float]:
        """The RES-14 seam. Today: normalise the declared constants."""
        w = np.asarray(self.weights, dtype=np.float64)
        if (w < 0).any():
            raise ValueError("mix: weights must be non-negative")
        s = w.sum()
        if s <= 0:
            raise ValueError("mix: weights sum to zero")
        return (w / s).tolist()

    def _spec(self) -> dict:
        return {"weights": self._resolved_weights(), "block": self.block}

    def _materialise_inputs(self, seed: int) -> list[np.ndarray]:
        out = []
        for c in self.sources:
            parts = list(c.stream(seed=seed))
            out.append(
                np.concatenate(parts) if parts else np.empty(0, dtype=TOKEN_DTYPE)
            )
        return out

    def stream(self, *, seed: int) -> Iterator[np.ndarray]:
        weights = self._resolved_weights()
        pools = self._materialise_inputs(seed)
        n = len(pools)

        # single-source identity fast-path: mix([S]) is exactly S (D7 identity)
        if n == 1:
            for i in range(0, len(pools[0]), _CHUNK):
                yield pools[0][i : i + _CHUNK]
            return

        # --- signature-aware redundancy folding (the bit-vector insight) ------
        # Two sources with the same content signature are the SAME data. We fold
        # them by content hash and combine their weights, so a budget is spread
        # over the REAL union of tokens, never over overlapping re-draws. This is
        # what makes mix([S,S], budget=|S|) a multiset-equivalent of S: the
        # duplicate is recognised, not re-sampled. (ADR-005 D6; RES-15 phase-1
        # provenance / RES-16 content-addressed consolidation.)
        sigs: dict[str, int] = {}
        folded_pools: list[np.ndarray] = []
        folded_w: list[float] = []
        for pool, w in zip(pools, weights):
            sig = _hash_obj(pool.tolist())
            if sig in sigs:
                folded_w[sigs[sig]] += w
            else:
                sigs[sig] = len(folded_pools)
                folded_pools.append(pool)
                folded_w.append(w)
        pools, weights = folded_pools, folded_w
        m = len(pools)
        s = sum(weights)
        weights = [w / s for w in weights]

        if m == 1:  # everything folded to one source → it IS that source
            budget = int(self.total) if self.total is not None else len(pools[0])
            budget = min(budget, len(pools[0]))
            for i in range(0, budget, _CHUNK):
                yield pools[0][i : min(i + _CHUNK, budget)]
            return

        # Output budget (D6): mixing redistributes a budget, it does not inflate
        # size. Default = weighted mean of (deduplicated) source lengths.
        if self.total is not None:
            budget = int(self.total)
        else:
            budget = int(round(sum(w * len(p) for w, p in zip(weights, pools))))

        # Per-source target, uniform WITHOUT replacement over that source's real
        # tokens (uniform stride, not front-loaded blocks — so a source is
        # covered evenly, and identical sources already folded above).
        targets = [min(int(round(w * budget)), len(p)) for w, p in zip(weights, pools)]
        drift = budget - sum(targets)
        # push leftover onto the largest-capacity source, capped
        order = sorted(range(m), key=lambda i: len(pools[i]) - targets[i], reverse=True)
        k = 0
        while drift > 0 and k < m:
            i = order[k]
            room = len(pools[i]) - targets[i]
            add = min(drift, room)
            targets[i] += add
            drift -= add
            k += 1

        # deterministic interleave of evenly-strided samples from each source
        picks = []
        for i, t in enumerate(targets):
            if t <= 0:
                continue
            idx = np.linspace(0, len(pools[i]) - 1, num=t, dtype=np.int64)
            picks.append((i, idx))
        order_rng = np.random.default_rng(seed ^ 0x9E3779B9)
        cursors = {i: 0 for i, _ in picks}
        remaining = {i: len(idx) for i, idx in picks}
        idx_of = {i: idx for i, idx in picks}
        buf: list[np.ndarray] = []
        buf_len = 0
        total_out = sum(remaining.values())
        produced = 0
        while produced < total_out and any(r > 0 for r in remaining.values()):
            live = [i for i in remaining if remaining[i] > 0]
            probs = np.asarray([remaining[i] for i in live], dtype=np.float64)
            probs /= probs.sum()
            pick = live[int(order_rng.choice(len(live), p=probs))]
            take = min(self.block, remaining[pick])
            sel = idx_of[pick][cursors[pick] : cursors[pick] + take]
            seg = pools[pick][sel]
            cursors[pick] += take
            remaining[pick] -= take
            produced += take
            buf.append(seg)
            buf_len += take
            if buf_len >= _CHUNK:
                yield np.concatenate(buf)
                buf, buf_len = [], 0
        if buf:
            yield np.concatenate(buf)


# --------------------------------------------------------------------------- #
# Transforms                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class Decontaminate(Op):
    """First-class decontamination operator (ADR-005 D4).

    Slice 1: exact n-gram membership against provided eval-set token sequences.
    Emits a contamination report (`last_report`) and, in `strict` mode, raises
    if any eval n-gram is found in the source. The Bloom-filter algebra
    (OR/AND/XOR at scale) is a later slice; the interface here is stable.
    """

    source: Op
    eval_ngrams: Sequence[Sequence[int]] = ()
    n: int = 13
    strict: bool = True
    op_type: str = field(default="decontaminate", init=False)
    last_report: dict = field(default_factory=dict, init=False, compare=False)

    def __post_init__(self):
        self.inputs = (self.source,)

    def _spec(self) -> dict:
        # hash the eval-set signature and policy, not the raw grams
        sig = _hash_obj([list(g) for g in self.eval_ngrams])
        return {"n": self.n, "strict": self.strict, "eval_sig": sig}

    def _eval_set(self) -> set[bytes]:
        out = set()
        for g in self.eval_ngrams:
            arr = np.asarray(g, dtype=TOKEN_DTYPE)
            out.add(arr.tobytes())
        return out

    def stream(self, *, seed: int) -> Iterator[np.ndarray]:
        evals = self._eval_set()
        hits = 0
        checked = 0
        # stream through; check rolling n-grams across chunk boundaries
        carry = np.empty(0, dtype=TOKEN_DTYPE)
        for chunk in self.source.stream(seed=seed):
            buf = np.concatenate([carry, chunk]) if len(carry) else chunk
            if evals and len(buf) >= self.n:
                for i in range(len(buf) - self.n + 1):
                    checked += 1
                    if buf[i : i + self.n].tobytes() in evals:
                        hits += 1
            # keep the last n-1 tokens to catch grams spanning the boundary
            carry = buf[-(self.n - 1):] if self.n > 1 else np.empty(0, dtype=TOKEN_DTYPE)
            yield chunk  # decontaminate is pass-through in Slice 1 (report-only)
        self.last_report = {"n": self.n, "eval_ngrams": len(evals),
                            "checked": checked, "hits": hits}
        if self.strict and hits > 0:
            raise ValueError(
                f"decontamination failed: {hits} eval n-gram(s) present in source "
                f"(n={self.n}). This mix is not comparable until cleaned."
            )


# --------------------------------------------------------------------------- #
# Manifest I/O                                                                 #
# --------------------------------------------------------------------------- #
def write_manifest(root: Op, path: str | Path) -> str:
    """Write the DAG manifest (the reproducible recipe) and return mix_hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mh = root.node_hash()
    doc = {"mix_hash": mh, "dag": root.manifest()}
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return mh


def mix_hash(root: Op) -> str:
    """The sink hash — two runs are comparable iff equal (ADR-005 D2)."""
    return root.node_hash()
