"""cortex_c2b.read_path — the journal as path 4 of the RES-18 router (ADR-007 D3), Slice C.

No new router is built. The versioned `Router` contract already iterates over
`range(1, len(scores))` with the control at index 0 as the floor — so adding a
path is adding an index, never touching the floor logic. Retro-compatibility in
action: PATH_JOURNAL = 3 joins control / hopfield / delta.

  - The floor guarantee holds for the journal exactly as for the memories: the
    journal path is used on a span only where its (predicted or known) score
    beats the control's; else the control. C2b can never degrade the model.
  - The read is sub-linear (ADR-003 D4 read law): `CueIndex` is an ANN over
    cues — a small from-scratch LSH (random hyperplanes → bucket signatures),
    so a query touches buckets, never the whole journal. HNSW-class libraries
    may replace it later (license-checked); the contract (never scan) holds.
  - `JournalPath.score(span_cue)` is the journal's per-span score: a retrieval
    confidence (best cosine among candidates), so the router can compare it to
    the control on the same footing as the memory paths.

Written from scratch (our filon).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from cortex_c2 import PATH_NAMES, PATH_CONTROL
from cortex_c2b import Journal, Entry, STATE_EVICTED, CUE_DIM

# ---- the fourth path, registered into the router's vocabulary ------------- #
PATH_JOURNAL = 3
if PATH_JOURNAL not in PATH_NAMES:
    PATH_NAMES[PATH_JOURNAL] = "journal"          # extend, never break (retro-compat)


# ---------------------------------------------------------------------------- #
# ANN over cues: random-hyperplane LSH (sub-linear; touches buckets, not all)   #
# ---------------------------------------------------------------------------- #
class CueIndex:
    """Locality-sensitive hashing over cues. `n_bits` hyperplanes → a signature;
    `n_tables` independent tables raise recall. A query probes n_tables buckets
    and ranks the candidates by exact cosine — never the whole store."""

    def __init__(self, dim: int = CUE_DIM, n_bits: int = 10, n_tables: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.planes = [rng.standard_normal((dim, n_bits)).astype(np.float32) for _ in range(n_tables)]
        self.tables: list[dict[int, list[str]]] = [dict() for _ in range(n_tables)]
        self.vecs: dict[str, np.ndarray] = {}
        self.touched = 0                                     # instrumentation for the no-scan test
        self.alias: dict[str, str] = {}                      # Slice E: source id -> summary id
        self.touch_log: deque = deque(maxlen=256)            # Slice E: entries touched per query (latency proxy)

    def _sig(self, v: np.ndarray, t: int) -> int:
        bits = (v @ self.planes[t]) > 0
        return int(np.packbits(bits.astype(np.uint8)).view(np.uint8).astype(np.int64).sum()
                   + bits.astype(np.int64) @ (1 << np.arange(len(bits))))

    def add(self, entry_id: str, cue: np.ndarray):
        v = np.asarray(cue, dtype=np.float32)
        self.vecs[entry_id] = v
        for t in range(len(self.planes)):
            self.tables[t].setdefault(self._sig(v, t), []).append(entry_id)

    def query(self, cue: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        q = np.asarray(cue, dtype=np.float32)
        cand: set[str] = set()
        for t in range(len(self.planes)):
            cand.update(self.tables[t].get(self._sig(q, t), []))
        self.touched += len(cand)
        self.touch_log.append(len(cand))
        if not cand:
            return []
        qn = np.linalg.norm(q) + 1e-8
        scored = []
        for eid in cand:
            v = self.vecs[eid]
            scored.append((eid, float(q @ v / (qn * (np.linalg.norm(v) + 1e-8)))))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def __len__(self):
        return len(self.vecs)

    def candidates(self, cue: np.ndarray) -> list[str]:
        """The keys sharing a bucket with `cue` in any table, sorted -- the
        neighbourhood the Slice E scheduler groups within. Not a read: it is
        not counted in the latency telemetry."""
        q = np.asarray(cue, dtype=np.float32)
        cand: set[str] = set()
        for t in range(len(self.planes)):
            cand.update(self.tables[t].get(self._sig(q, t), []))
        return sorted(cand)

    # ---- Slice E: demotion aliases and the latency proxy ----------------------
    def set_alias(self, source_id: str, summary_id: str) -> None:
        """After a demotion the summary is reachable from every source cue: the
        source stays indexed under its own cue, and a hit on it resolves to the
        summary (ADR-007 D4, keeping pointers). No second vector is stored."""
        self.alias[source_id] = summary_id

    def p95_touched(self) -> float:
        """95th percentile of entries touched per query over the recent window."""
        if not self.touch_log:
            return 0.0
        return float(np.percentile(np.asarray(self.touch_log, dtype=np.float64), 95))


# ---------------------------------------------------------------------------- #
# The journal path: score + retrieve, on the router's footing                  #
# ---------------------------------------------------------------------------- #
class JournalPath:
    """Exposes the journal to the router as path PATH_JOURNAL.

    score(span_cue)   -> the journal's confidence that it holds something for
                         this span: best cosine among ANN candidates (0 if empty).
    retrieve(span_cue)-> the entries behind that score (non-evicted), with payloads.
    """

    def __init__(self, journal: Journal, index: CueIndex | None = None, seed: int = 0):
        """`seed` is the declared seed of the cue index (Decision 8: the index is
        recomputed at every open from the log; the seed is part of the scope's
        configuration, so two opens build the same buckets)."""
        self.j = journal
        self.index = index or CueIndex(dim=journal.cue_dim, seed=seed)
        # index everything already in the journal (demoted sources included: a hit on
        # them resolves to their summary through the alias)
        for eid, e in journal._entries.items():
            if e.state != STATE_EVICTED or eid in journal.summary_of:
                self.index.add(eid, np.asarray(e.cue, dtype=np.float32))
        for src, sid in journal.summary_of.items():
            self.index.set_alias(src, sid)

    def on_write(self, entry: Entry):
        """Keep the index in step with the journal (call after each admitted write)."""
        self.index.add(entry.entry_id, np.asarray(entry.cue, dtype=np.float32))

    def _resolve(self, key: str) -> tuple[Entry, str]:
        """A hit is an entry id or a demoted source id. Returns the effective entry
        (the summary, for a source) and the pointer of the payload to return (the
        source's own payload, alive as long as its summary is)."""
        hit = self.j.get(key)
        effective, target, hops = hit, self.index.alias.get(key), 0
        while target is not None and hops < 64:                  # up the hierarchy (RES-16), no cycle by construction
            effective, target, hops = self.j.get(target), self.index.alias.get(target), hops + 1
        return effective, hit.pointer

    def score(self, span_cue) -> float:
        hits = self.index.query(np.asarray(span_cue, dtype=np.float32), k=1)
        hits = [(eid, s) for eid, s in hits if self._resolve(eid)[0].state != STATE_EVICTED]
        return float(max(0.0, hits[0][1])) if hits else 0.0

    def cue_similarity(self, span_cue, pointer: str) -> float:
        """The organ's own evidence for one line (RES-23, the familiarity mark): the
        cosine between a query's cue and the stored cue of the alive entry whose
        payload is `pointer`, computed exactly as the index scores a retrieval;
        0.0 when no alive entry holds that pointer. Never a value the harness supplies."""
        q = np.asarray(span_cue, dtype=np.float32)
        qn = float(np.linalg.norm(q)) + 1e-8
        for eid, e in self.j._entries.items():
            if e.pointer == pointer and e.state != STATE_EVICTED and eid in self.index.vecs:
                v = self.index.vecs[eid]
                return float(q @ v / (qn * (float(np.linalg.norm(v)) + 1e-8)))
        return 0.0

    def retrieve(self, span_cue, k: int = 3) -> list[tuple[Entry, bytes, float]]:
        out = []
        for eid, s in self.index.query(np.asarray(span_cue, dtype=np.float32), k=k):
            e, pointer = self._resolve(eid)
            if e.state != STATE_EVICTED:
                out.append((e, self.j.payloads.get(pointer), s))
        return out


def path_scores_with_journal(control: float, hopfield: float, delta: float,
                             journal: float) -> list[float]:
    """Assemble the 4-path score vector in router order (control first — the floor)."""
    return [control, hopfield, delta, journal]
