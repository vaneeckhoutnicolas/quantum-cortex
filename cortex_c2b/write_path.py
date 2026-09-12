"""cortex_c2b.write_path — the hippocampal write path (ADR-007 D2), Slice B.

    separate (DG) → associate (CA3) → compare (CA1) → gate

Each stage is a flag (default on here; ablatable to off) so that interference
without DG, recall without CA3, and write precision without CA1 are each a row.

  DG  — pattern separation: expand-then-sparsify the cue so near-duplicates
        decorrelate before storage (the interference guard). A fixed random
        expansion to a larger space followed by k-winners-take-all: two cues that
        are close in input space land on mostly-different sparse codes.
  CA3 — pattern completion: reconstruct the input from what the memory already
        holds. Slice B uses the journal's own stored cues as the associative
        memory (nearest stored cue = the reconstruction); Slice C swaps in the
        C2 Hopfield/delta layer of ADR-006 (C2 *is* CA3) behind the same call.
  CA1 — the comparator: surprise = distance(reconstruction, input). Surprise is
        no longer a heuristic; it is a computed quantity at the memory interface
        (RES-2 made concrete). Feeds salience and the write gate.
  gate — the noise ruling (D4.8), TWO stages, never one detector:
        (a) admission: reject the redundant/already-known (surprise below a
            declared threshold) — but surprise alone is not signal: a corrupted
            input is maximally surprising and would be journaled enthusiastically;
        (b) the verdict is retrospective: an admitted entry that never gains
            salience, never consolidates, never earns outcome credit is NOISE and
            is evicted by the tribunal (`noise_tribunal`). Scheduled forgetting is
            the judge. Slice B implements the tribunal's rule; the schedule (RES-9)
            is Slice E.

Written from scratch (our filon). No torch here — pure numpy over the store.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cortex_c2b import Journal, Entry, STATE_LIVE, STATE_EVICTED, CUE_DIM


# ---------------------------------------------------------------------------- #
# DG — pattern separation                                                      #
# ---------------------------------------------------------------------------- #
class DentateGyrus:
    """Expand-then-sparsify: project to a larger space with a fixed random map,
    keep the k largest activations (k-WTA), renormalise. Near-duplicates
    decorrelate; the separation is measurable as the drop in cosine similarity."""

    def __init__(self, in_dim: int = CUE_DIM, expand: int = 4, sparsity: float = 0.1, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.out_dim = in_dim * expand
        self.W = rng.standard_normal((in_dim, self.out_dim)).astype(np.float32) / np.sqrt(in_dim)
        self.k = max(1, int(self.out_dim * sparsity))

    def separate(self, cue: np.ndarray) -> np.ndarray:
        z = cue.astype(np.float32) @ self.W
        idx = np.argpartition(-np.abs(z), self.k)[: self.k]          # k winners
        out = np.zeros_like(z); out[idx] = z[idx]
        n = np.linalg.norm(out)
        return out / n if n > 0 else out


# ---------------------------------------------------------------------------- #
# CA3 — pattern completion (Slice B: from the journal's stored cues)            #
# ---------------------------------------------------------------------------- #
class CA3Completion:
    """Reconstruct from memory: return the stored code nearest to the input.
    Slice C replaces this with the C2 associative layer behind the same call."""

    def __init__(self):
        self._codes: list[np.ndarray] = []

    def store(self, code: np.ndarray):
        self._codes.append(code.astype(np.float32))

    def reconstruct(self, code: np.ndarray) -> np.ndarray | None:
        if not self._codes:
            return None
        M = np.stack(self._codes)                                       # (n, d)
        sims = M @ code / (np.linalg.norm(M, axis=1) * (np.linalg.norm(code) + 1e-8) + 1e-8)
        return M[int(np.argmax(sims))]


# ---------------------------------------------------------------------------- #
# CA1 — the comparator: surprise as a computed quantity                        #
# ---------------------------------------------------------------------------- #
def ca1_surprise(reconstruction: np.ndarray | None, code: np.ndarray) -> float:
    """surprise = distance(reconstruction, input) in cosine terms ∈ [0, 1].
    No memory yet → maximal surprise (everything is new)."""
    if reconstruction is None:
        return 1.0
    a, b = reconstruction, code
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    return float(np.clip(1.0 - cos, 0.0, 1.0))


# ---------------------------------------------------------------------------- #
# The gated write path                                                          #
# ---------------------------------------------------------------------------- #
@dataclass
class WriteReport:
    admitted: bool
    surprise: float
    reason: str
    entry: Entry | None = None


class WritePath:
    """separate → associate → compare → gate, over a Journal.

    `admission_threshold`: a cue whose CA1 surprise is below it is REDUNDANT
    (already known) and is not admitted (stage a). Admitted entries start LIVE
    with salience = surprise. Stage (b), the retrospective verdict, is
    `noise_tribunal()`.
    """

    def __init__(self, journal: Journal, admission_threshold: float = 0.15,
                 use_dg: bool = True, use_ca3: bool = True, use_ca1: bool = True, seed: int = 0):
        self.j = journal
        self.thr = admission_threshold
        self.use_dg, self.use_ca3, self.use_ca1 = use_dg, use_ca3, use_ca1
        self.dg = DentateGyrus(seed=seed) if use_dg else None
        self.ca3 = CA3Completion()
        # Decision 8 (persistence per component): CA3 is rebuilt from the journal's
        # `ca3` events, in admission order, with the same separation. A journal
        # written under another separation is refused, never silently re-coded.
        self._meta = {"dg": bool(use_dg), "seed": int(seed)}
        for eid, meta in journal.ca3_order:
            if {"dg": meta.get("dg"), "seed": meta.get("seed")} != self._meta:
                raise ValueError(f"CA3 rebuild: journal written with separation {meta}, this write path has {self._meta}")
            self.ca3.store(self._code(np.asarray(journal._entries[eid].cue, dtype=np.float32)))

    def _code(self, cue: np.ndarray) -> np.ndarray:
        return self.dg.separate(cue) if self.dg else cue.astype(np.float32)

    def write(self, cue, payload: bytes, schema_id: str, now: float | None = None) -> WriteReport:
        cue = np.asarray(cue, dtype=np.float32)
        code = self._code(cue)
        recon = self.ca3.reconstruct(code) if self.use_ca3 else None
        surprise = ca1_surprise(recon, code) if self.use_ca1 else 1.0   # no comparator → write everything
        # stage (a): admission — reject the redundant / already-known
        if surprise < self.thr:
            return WriteReport(admitted=False, surprise=surprise, reason="redundant (below admission threshold)")
        entry = self.j.write(cue, payload, salience=surprise, schema_id=schema_id, now=now,
                             ca3=self._meta)                          # one line; a failure cannot split it (D9)
        # store the code of the cue AS JOURNALED (rounded), so a rebuild after a restart
        # produces byte-identical CA3 contents (Decision 8)
        self.ca3.store(self._code(np.asarray(entry.cue, dtype=np.float32)))
        return WriteReport(admitted=True, surprise=surprise, reason="admitted", entry=entry)

    # ---- outcome credit (RES-11 hook): salience earned after the fact ---------
    def credit(self, entry_id: str, amount: float, why: str = "credit"):
        e = self.j.get(entry_id)
        self.j.set_salience(entry_id, e.salience + amount, why=why)   # an event (Slice E, invariant 1)

    # ---- stage (b): the retrospective verdict — the noise tribunal -----------
    def noise_tribunal(self, salience_floor: float = 0.2, min_age: float = 0.0,
                       now: float | None = None) -> list[str]:
        """Evict LIVE entries that never earned salience above the floor and were
        never consolidated — the D4.8 definition: noise is what never improves
        prediction. A corrupted input (maximally surprising at admission) that
        never earns credit is exactly what this catches. Returns evicted ids."""
        import time as _t
        now = _t.time() if now is None else now
        evicted = []
        for eid in noise_candidates(self.j, salience_floor, min_age, now):
            self.j.transition(eid, STATE_EVICTED)
            evicted.append(eid)
        return evicted


def noise_candidates(journal: Journal, salience_floor: float, min_age: float, now: float) -> list[str]:
    """The D4.8 rule as a predicate, shared by the tribunal above and by the
    Slice E scheduler (which plans before it applies): LIVE entries that never
    earned salience above the floor and are old enough. Deterministic order."""
    return [eid for eid, e in sorted(journal._entries.items())
            if e.state == STATE_LIVE and e.salience < salience_floor and (now - e.t_written) >= min_age]
