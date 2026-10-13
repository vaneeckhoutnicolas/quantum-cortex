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
# The index geometry (ADR-008, amendment of 2026-10-13): a declared parameter, recorded in the
# journal on every write event beside the separation meta, named in the recipe's hash. Two
# geometries were measured as what they are (row 45): the legacy one, the live index of every
# run to date (four tables of ten bits, the collapsed signature), and the adopted one (sixteen
# tables of ten bits, the exact signature). The code's defaults stay legacy (D1): the adopted
# geometry is declared by a run's configuration and recorded in its journal, never a silent
# change of a library default.
SIGNATURES = ("collapsed", "exact")
GEOMETRY_KEYS = ("n_tables", "n_bits", "signature", "seed")      # what a stamp declares and what a mismatch compares
LEGACY_GEOMETRY = {"n_tables": 4, "n_bits": 10, "signature": "collapsed", "seed": 0}
ADOPTED_GEOMETRY = {"n_tables": 16, "n_bits": 10, "signature": "exact", "seed": 0}


def same_geometry(a: dict | None, b: dict | None) -> bool:
    """Equal on the four declared parameters; the planes hash is information written in files, not a parameter."""
    if a is None or b is None:
        return a is b
    return all(a.get(k) == b.get(k) for k in GEOMETRY_KEYS)


def geometry_stamp(g: dict) -> dict:
    """The four parameters and the planes hash, as a write event carries them."""
    return {k: g[k] for k in GEOMETRY_KEYS} | ({"planes_sha256": g["planes_sha256"]} if "planes_sha256" in g else {})


def validate_geometry(g: dict, where: str = "index geometry") -> dict:
    """A geometry dict as the journal stamps it or a configuration gives it: the four parameters with their types, the
    signature one of the two names, an optional planes hash of sixteen hexadecimal characters, no other key. Returns a
    clean copy; raises ValueError otherwise (a hand edited stamp or a loose dict fails at open, before any write)."""
    import re
    if not isinstance(g, dict):
        raise ValueError(f"{where}: a dict is expected, got {type(g).__name__}")
    extra = set(g) - set(GEOMETRY_KEYS) - {"planes_sha256"}
    if extra:
        raise ValueError(f"{where}: unknown keys {sorted(extra)}")
    missing = [k for k in GEOMETRY_KEYS if k not in g]
    if missing:
        raise ValueError(f"{where}: missing keys {missing}")
    for k in ("n_tables", "n_bits", "seed"):
        if isinstance(g[k], bool) or not isinstance(g[k], int):
            raise ValueError(f"{where}: {k} must be an integer, got {g[k]!r}")
    if g["n_tables"] < 1 or g["n_bits"] < 1:
        raise ValueError(f"{where}: at least one table of at least one bit")
    if g["signature"] not in SIGNATURES:
        raise ValueError(f"{where}: signature is one of {SIGNATURES}, got {g['signature']!r}")
    out = {k: int(g[k]) if k != "signature" else str(g[k]) for k in GEOMETRY_KEYS}
    if "planes_sha256" in g:
        if not isinstance(g["planes_sha256"], str) or not re.fullmatch(r"[0-9a-f]{16}", g["planes_sha256"]):
            raise ValueError(f"{where}: planes_sha256 is sixteen hexadecimal characters, got {g['planes_sha256']!r}")
        out["planes_sha256"] = g["planes_sha256"]
    return out


FLY_EXPAND = 1024                # expanded units against CUE_DIM inputs (the fly: 2000 Kenyon cells for 50 projection neurons)
FLY_SAMPLE = 6                   # inputs each expanded unit samples (the fly: about 6)
FLY_SPARSITY = 0.05              # the winner take all keeps this fraction (the fly's APL silences all but about 5 %)


def fly_planes(dim: int = CUE_DIM, expand: int = FLY_EXPAND, sample: int = FLY_SAMPLE, seed: int = 0) -> np.ndarray:
    """The fixed sparse binary projection of the mushroom body (Dasgupta, Stevens and
    Navlakha, Science 2017): each expanded unit sums a few inputs, drawn once and never
    learned. Returned as an (expand, dim) matrix of zeros and ones."""
    rng = np.random.default_rng(seed)
    P = np.zeros((expand, dim), dtype=np.float32)
    for i in range(expand):
        P[i, rng.choice(dim, size=min(sample, dim), replace=False)] = 1.0
    return P


def fly_tag(cue, planes: np.ndarray, sparsity: float = FLY_SPARSITY) -> np.ndarray:
    """Expand, then sparsify: the indices of the highest firing fraction of the expanded
    units, as a sorted array. Two cues are compared by how many indices they share."""
    y = planes @ np.asarray(cue, dtype=np.float32)
    keep = max(1, int(round(sparsity * y.shape[0])))
    idx = np.argpartition(-y, keep - 1)[:keep]
    return np.sort(idx)


class CueIndex:
    """Locality-sensitive hashing over cues. `n_bits` hyperplanes → a signature;
    `n_tables` independent tables raise recall. A query probes n_tables buckets
    and ranks the candidates by exact cosine — never the whole store."""

    def __init__(self, dim: int = CUE_DIM, n_bits: int = 10, n_tables: int = 4, seed: int = 0,
                 rank: str = "cosine", signature: str = "collapsed"):
        # `rank`: "cosine" (the dense cue, every run to date) or "fly" (the expand and
        # sparsify tag of ADR-003 line 6, declared 2026-09-19, probe only); the candidate
        # buckets are the same in both, only the ranking of the candidates changes.
        # `signature` (ADR-008, 2026-10-13): "collapsed" is the live key of every run to date
        # (the packed byte sum added to the ten bit integer: 278 distinct keys of 1 024, found
        # 2026-10-12), kept under its name; "exact" is the ten bit integer alone (C2b, row 45).
        if rank not in ("cosine", "fly"):
            raise ValueError("rank is cosine or fly")
        if signature not in SIGNATURES:
            raise ValueError(f"signature is one of {SIGNATURES}")
        if n_tables < 1 or n_bits < 1:
            raise ValueError("an index has at least one table of at least one bit")
        self.rank = rank
        self.signature = signature
        self.n_bits, self.n_tables, self.seed = int(n_bits), int(n_tables), int(seed)
        self.fly_planes = fly_planes(dim, seed=seed) if rank == "fly" else None
        self.tags: dict[str, np.ndarray] = {}
        rng = np.random.default_rng(seed)                    # one generator, table by table: the first tables of a larger index are the smaller index's
        self.planes = [rng.standard_normal((dim, n_bits)).astype(np.float32) for _ in range(n_tables)]
        self.tables: list[dict[int, list[str]]] = [dict() for _ in range(n_tables)]
        self.vecs: dict[str, np.ndarray] = {}
        self.touched = 0                                     # instrumentation for the no-scan test
        self.alias: dict[str, str] = {}                      # Slice E: source id -> summary id
        self.touch_log: deque = deque(maxlen=256)            # Slice E: entries touched per query (latency proxy)

    def _sig(self, v: np.ndarray, t: int) -> int:
        v = np.asarray(v, dtype=np.float32)                  # the signature a lookup uses, whatever dtype the caller holds (2026-10-13)
        bits = (v @ self.planes[t]) > 0
        exact = int(bits.astype(np.int64) @ (1 << np.arange(len(bits))))
        if self.signature == "exact":
            return exact
        return int(np.packbits(bits.astype(np.uint8)).view(np.uint8).astype(np.int64).sum()) + exact

    def geometry(self) -> dict:
        """The declared parameters and the hash of the planes actually drawn (the draw depends on numpy's
        generator), read from the live object: what every file that reads a journal records."""
        import hashlib
        h = hashlib.sha256()
        for p in self.planes:
            h.update(np.ascontiguousarray(p, dtype=np.float32).tobytes())
        return {"n_tables": self.n_tables, "n_bits": self.n_bits, "signature": self.signature, "seed": self.seed,
                "planes_sha256": h.hexdigest()[:16]}

    def add(self, entry_id: str, cue: np.ndarray):
        v = np.asarray(cue, dtype=np.float32)
        self.vecs[entry_id] = v
        if self.rank == "fly":
            self.tags[entry_id] = fly_tag(v, self.fly_planes)
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
        if self.rank == "fly":                               # shared indices of the two sparse tags, in [0, 1]
            qt = fly_tag(q, self.fly_planes); n_keep = float(qt.size)
            scored = [(eid, float(np.intersect1d(qt, self.tags[eid], assume_unique=True).size) / n_keep) for eid in cand]
            scored.sort(key=lambda x: -x[1])
            return scored[:k]
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

    def __init__(self, journal: Journal, index: CueIndex | None = None, seed: int | None = None,
                 geometry: dict | None = None):
        """`seed` is the declared seed of the cue index (Decision 8: the index is
        recomputed at every open from the log; the seed is part of the scope's
        configuration, so two opens build the same buckets).

        The geometry (ADR-008, amendment of 2026-10-13). The journal records the geometry
        it was written under (`Journal.index_record()`: the first stamp, legacy for a
        journal written before the stamp existed, None for a journal with no write yet).
        At open the record is the default; `geometry` (a configuration's `n_tables` and
        `signature`, `n_bits` and `seed` optional) or a `seed` that differs from the
        record is refused with both geometries named, never re indexed silently; an
        `index` passed explicitly is kept whatever the record (the rule of Rev108) and
        `index_overridden` says whether it differs from it, so the file can record both.
        A fresh journal takes the first path's geometry as its own and stamps it on
        every write from then on; an overriding reader never moves a journal's geometry."""
        self.j = journal
        record = journal.index_record()
        declared = record if record is not None else journal.index_geometry      # what the journal stamps (committee 2026-10-13: a fresh
        if index is not None:                                                     # journal given a geometry at construction is bound by it)
            # `is None`, not `or`: an empty CueIndex is falsy (it has a length), and `or` replaced the index
            # the retrieval probe of row 38 passed by a default cosine one (found 2026-10-12, register Rev108)
            if index.planes[0].shape[0] != journal.cue_dim:
                raise ValueError(f"index geometry: the index projects {index.planes[0].shape[0]} dimensions, the journal's cues have {journal.cue_dim}")
            self.index = index
            live = index.geometry()
            differs = declared is not None and (not same_geometry(live, declared)
                                                or (declared.get("planes_sha256") not in (None, live["planes_sha256"])))
            if record is None and declared is not None and differs:
                raise ValueError("index geometry: a fresh journal was given "
                                 f"{ {k: declared.get(k) for k in GEOMETRY_KEYS} } at construction and this index is {geometry_stamp(live)}; "
                                 "a journal takes one geometry")
            self.index_overridden = bool(differs)
            if declared is None:
                journal.index_geometry = geometry_stamp(live)                # the buckets' geometry; the ranking (`rank`) is not part of it
        else:
            if geometry is not None:
                g = dict(geometry)
                for k in g:
                    if k not in GEOMETRY_KEYS and k != "planes_sha256":
                        raise ValueError(f"index geometry: unknown key {k!r} in the geometry given")
                for k in ("n_tables", "signature"):
                    if k not in g:
                        raise ValueError(f"index geometry: the geometry given lacks {k!r}")
                g_seed = g.get("seed")
                if seed is not None and g_seed is not None and int(g_seed) != int(seed):
                    raise ValueError(f"index geometry: the geometry names seed {g_seed}, the path seed {seed}")
                want = validate_geometry({"n_tables": g["n_tables"], "n_bits": g.get("n_bits", 10), "signature": g["signature"],
                                          "seed": (seed if seed is not None else (g_seed if g_seed is not None else 0))})
            elif declared is not None:
                want = {k: declared[k] for k in GEOMETRY_KEYS}
                if seed is not None:
                    want["seed"] = int(seed)
            else:
                want = {**LEGACY_GEOMETRY, "seed": int(seed if seed is not None else 0)}
            if declared is not None and not same_geometry(want, declared):
                raise ValueError("index geometry: the journal records "
                                 f"{ {k: declared[k] for k in GEOMETRY_KEYS} }, this path asks {want}; a journal is never re indexed "
                                 "under another geometry silently (ADR-008, 2026-10-13); a declared reader passes its index explicitly")
            self.index = CueIndex(dim=journal.cue_dim, n_bits=want["n_bits"], n_tables=want["n_tables"], seed=want["seed"],
                                  signature=want["signature"])
            live = self.index.geometry()
            if declared is not None and declared.get("planes_sha256") not in (None, live["planes_sha256"]):
                raise ValueError("index geometry: the journal records the planes "
                                 f"{declared['planes_sha256']} and this process draws {live['planes_sha256']} under the same parameters "
                                 "(numpy's generator moved); refused, a declared reader passes its index explicitly (ADR-008, 2026-10-13)")
            self.index_overridden = False
            journal.index_geometry = geometry_stamp(live)                    # the four parameters as recorded, the planes hash of the live draw
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
