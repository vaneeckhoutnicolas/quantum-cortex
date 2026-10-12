"""cortex_c2b.bucket_probe -- the probe of the index's buckets (ADR-008, amendment of
2026-10-12, evening, declared with a committee of four readers and validated by the
founder at 19:40; RES-24, roadmap item 1; register Rev108 and the build's line).

What it measures, on a checkpoint and its sealed journal that already exist (the three
v9 journals), with the checkpoint's own cue encoder as the only model call: for each of
the declared configurations of the buckets, whether the own episode of every protocol
question is among the candidates the buckets return, its rank after the cosine ranking
of those candidates, the number of candidates (the read law's cost: the distinct ids of
the union over the tables, what the live index's `touched` counter increments by), the
cosine between the query cue and the own cue, the best cosine to another stored cue, the
own line's rank over the whole store (C6), the signatures of the query and of the own
entry per table with their Hamming distance; and for the never planted entities the
number of candidates, the best score and the novelty score of Dasgupta, Sheehan, Stevens
and Navlakha (PNAS 2018). The fly's tag as a ranking of C0's candidates is measured at
no extra cost (the reading of 2026-09-19 that row 38 never computed).

The configurations (the amendment's words are the law; this is their code):
  C0  the control: four tables of ten bits, the collapsed signature of the live index
  C1  the exact signature, the same four tables
  C2a, C2b  the exact signature with eight and sixteen tables (the first tables are C1's)
  C3a, C3b  the exact signature with the first eight and six columns of C1's planes
  C4  multi probe on the exact signature: the Hamming ball of radius one, eleven buckets per table
  C5a8, C5a16, C5b8, C5b16  the fly's pseudo hash as the bucket (Sharma and Navlakha 2018):
      the cue centred by its own mean, the 1 024 expanded activations in m blocks (8 or 16),
      one bit per block (the sign of the block's sum), one table, radius 0 (a) or 1 (b),
      the candidates ranked by the overlap of the sparse tags
  C6  the reference: the whole store, never adopted
The subset relations C1 within C0, C2, C3 and C4 are properties of the construction,
pinned by tests and checked query by query at run time, not measurements.

The file (compact JSON, one line, finite numbers only) carries the per question constants
once (`questions`), the rows per configuration, the signatures of every stored entry under
every configuration (`entry_signatures`) and the 250 query cues, so every candidate set and
every cost recompute from the file alone; the ranks need the stored cues, which are the
sealed journal's, which the probe never opens in place (a copy in a temporary directory; the
record's log and payload listing hashed before and after). The committee's two rounds on the
build (2026-10-12, evening: a measurement methodologist and a systems reader on the first
build, a code reader on the corrected one; thirty findings) are applied here and dated in
ADR-008 as the build's precisions (1) to (11).

Nothing here predicts a number. The record test recomputes every aggregate from the rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cortex_c2b import Journal, POLICY_STOP, content_hash
from cortex_c2b.hm_protocol import generate_facts, N_FACTS, N_NEGCTRL
from cortex_c2b.read_path import CueIndex, JournalPath, fly_planes, fly_tag, CUE_DIM, FLY_EXPAND

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---- declared constants (ADR-008, amendment of 2026-10-12, evening) --------------------------------
BUDGET = {"p95_at_most": 50, "max_below": 200, "mean_at_most_times_control": 4}
MARGIN = 3                      # R minus L on every journal
TIE_CANDIDATES = 1.0            # a tie in cost within one candidate per query goes to fewer tables, then the exact signature
CUE_FAR = 0.10                  # a miss is the cue's when the widest hyperplane scheme (C4) would propose it with a chance under this, by the collision law
N_BITS_LIVE, N_TABLES_LIVE = 10, 4

CONFIGS = [
    {"name": "C0", "family": "hyperplanes", "signature": "collapsed", "n_tables": 4, "n_bits": 10, "probe_radius": 0},
    {"name": "C1", "family": "hyperplanes", "signature": "exact", "n_tables": 4, "n_bits": 10, "probe_radius": 0},
    {"name": "C2a", "family": "hyperplanes", "signature": "exact", "n_tables": 8, "n_bits": 10, "probe_radius": 0},
    {"name": "C2b", "family": "hyperplanes", "signature": "exact", "n_tables": 16, "n_bits": 10, "probe_radius": 0},
    {"name": "C3a", "family": "hyperplanes", "signature": "exact", "n_tables": 4, "n_bits": 8, "probe_radius": 0},
    {"name": "C3b", "family": "hyperplanes", "signature": "exact", "n_tables": 4, "n_bits": 6, "probe_radius": 0},
    {"name": "C4", "family": "hyperplanes", "signature": "exact", "n_tables": 4, "n_bits": 10, "probe_radius": 1},
    {"name": "C5a8", "family": "fly_pseudo", "m_blocks": 8, "probe_radius": 0},
    {"name": "C5a16", "family": "fly_pseudo", "m_blocks": 16, "probe_radius": 0},
    {"name": "C5b8", "family": "fly_pseudo", "m_blocks": 8, "probe_radius": 1},
    {"name": "C5b16", "family": "fly_pseudo", "m_blocks": 16, "probe_radius": 1},
]
REFERENCE = {"name": "C6", "family": "scan"}


def config(name: str) -> dict:
    for c in CONFIGS:
        if c["name"] == name:
            return dict(c)
    if name == REFERENCE["name"]:
        return dict(REFERENCE)
    raise KeyError(name)


# ---- the tables and the signatures, exactly the live index's --------------------------------------
def hyperplane_tables(dim: int, n_tables: int, seed: int) -> list[np.ndarray]:
    """The planes `CueIndex(dim, n_bits=10, n_tables, seed)` draws: one generator, table by
    table, so the first tables of a larger index are the smaller index's (C2 nests C0)."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((dim, N_BITS_LIVE)).astype(np.float32) for _ in range(n_tables)]


def exact_signature(bits) -> int:
    b = np.asarray(bits).astype(np.int64)
    return int(b @ (1 << np.arange(len(b))))


def collapsed_signature(bits) -> int:
    """The live `CueIndex._sig`: the packed byte sum added to the ten bit integer (278 distinct
    values over the 1 024 patterns; found 2026-10-12)."""
    b = np.asarray(bits).astype(np.uint8)
    return int(np.packbits(b).view(np.uint8).astype(np.int64).sum()) + exact_signature(b)


def _sha(arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def _cos(a, b) -> float:
    return float(a @ b / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


class BucketIndex:
    """One configuration of the buckets over a fixed set of cues. `candidates(q)` returns the
    distinct ids of the union over the tables (and the probed neighbours), which is what the
    live index's `touched` counter increments by; `rank(q, ids)` ranks them as the configuration's
    family does (the cosine for the hyperplanes and the scan, the tag's overlap for the fly)."""

    def __init__(self, vecs: dict[str, np.ndarray], cfg: dict, index_seed: int = 0, dim: int = CUE_DIM):
        self.cfg, self.index_seed, self.dim = dict(cfg), index_seed, dim
        self.vecs = {k: np.asarray(v, dtype=np.float32) for k, v in vecs.items()}
        self.family = cfg["family"]
        self.tables: list[np.ndarray] = []
        self.buckets: list[dict[int, list[str]]] = []
        self.fly_P = None
        self.tags: dict[str, np.ndarray] = {}
        if self.family == "hyperplanes":
            if cfg["signature"] == "collapsed" and cfg["probe_radius"] != 0:
                raise ValueError("multi probe is declared on the exact signature only")
            self.tables = [T[:, :cfg["n_bits"]] for T in hyperplane_tables(dim, cfg["n_tables"], index_seed)]
            self.buckets = [dict() for _ in self.tables]
            for eid, v in self.vecs.items():
                for t, s in enumerate(self.signatures(v)):
                    self.buckets[t].setdefault(s, []).append(eid)
        elif self.family == "fly_pseudo":
            self.fly_P = fly_planes(dim, seed=index_seed)                # the tag's projection of 2026-09-19, seed 0 as the live one
            self.block = FLY_EXPAND // cfg["m_blocks"]
            self.buckets = [dict()]
            for eid, v in self.vecs.items():
                self.tags[eid] = fly_tag(v, self.fly_P)
                self.buckets[0].setdefault(self.signatures(v)[0], []).append(eid)
        elif self.family == "scan":
            pass
        else:
            raise ValueError(self.family)

    # -- signatures ------------------------------------------------------------------------------
    def bits(self, q) -> list[int]:
        """The exact bits per table (the ten bit integer), for the Hamming distance of every hyperplane
        configuration, the collapsed C0 included; the fly's key for the pseudo hash."""
        q = np.asarray(q, dtype=np.float32)
        if self.family == "hyperplanes":
            return [exact_signature((q @ T) > 0) for T in self.tables]
        return self.signatures(q)

    def signatures(self, q) -> list[int]:
        q = np.asarray(q, dtype=np.float32)
        if self.family == "hyperplanes":
            sig = collapsed_signature if self.cfg["signature"] == "collapsed" else exact_signature
            return [sig((q @ T) > 0) for T in self.tables]
        if self.family == "fly_pseudo":
            y = self.fly_P @ (q - q.mean())                               # the source's first step: centred per vector
            m = self.cfg["m_blocks"]
            bits = [float(y[j * self.block:(j + 1) * self.block].sum()) >= 0.0 for j in range(m)]
            return [sum(int(b) << j for j, b in enumerate(bits))]
        return []

    # -- candidates ------------------------------------------------------------------------------
    def candidates(self, q) -> list[str]:
        if self.family == "scan":
            return sorted(self.vecs)
        out: set[str] = set()
        n_bits = self.cfg["n_bits"] if self.family == "hyperplanes" else self.cfg["m_blocks"]
        for t, s in enumerate(self.signatures(q)):
            keys = [s] + ([s ^ (1 << b) for b in range(n_bits)] if self.cfg["probe_radius"] == 1 else [])
            for key in keys:
                out.update(self.buckets[t].get(key, ()))
        return sorted(out)

    # -- ranking ---------------------------------------------------------------------------------
    def rank(self, q, ids) -> list[tuple[str, float]]:
        q = np.asarray(q, dtype=np.float32)
        if self.family == "fly_pseudo":
            qt = fly_tag(q, self.fly_P); n = float(qt.size)
            scored = [(eid, float(np.intersect1d(qt, self.tags[eid], assume_unique=True).size) / n) for eid in ids]
        else:
            scored = [(eid, _cos(q, self.vecs[eid])) for eid in ids]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored

    def describe(self) -> dict:
        return {"name": self.cfg["name"], "family": self.family, "index_seed": self.index_seed, "entries": len(self.vecs),
                "signature": self.cfg.get("signature"),
                "n_tables": len(self.tables) if self.family == "hyperplanes" else (1 if self.family == "fly_pseudo" else 0),
                "n_bits": self.cfg.get("n_bits"), "m_blocks": self.cfg.get("m_blocks"), "probe_radius": self.cfg.get("probe_radius", 0),
                "rank_by": ("overlap" if self.family == "fly_pseudo" else "cosine"),
                "planes_sha256": _sha(self.tables) if self.tables else None,
                "fly_planes_sha256": _sha([self.fly_P]) if self.fly_P is not None else None}


# ---- the collision law (Charikar 2002), per miss --------------------------------------------------
def proposal_probability(cosine: float, n_bits: int, n_tables: int, radius: int = 0) -> float:
    """The chance that a query at this cosine to a stored cue shares its exact bucket in at least
    one of the tables (radius 0), or lands within Hamming distance one (radius 1): a per
    hyperplane agreement of 1 minus theta over pi (Charikar, STOC 2002)."""
    p = 1.0 - math.acos(max(-1.0, min(1.0, cosine))) / math.pi
    one = p ** n_bits + (n_bits * p ** (n_bits - 1) * (1 - p) if radius == 1 else 0.0)
    return 1.0 - (1.0 - one) ** n_tables


def miss_class(row: dict) -> str | None:
    """The declared classes of a residual miss: the index's (the own line is the nearest stored cue
    and was not proposed), interference (another stored cue is at least as close), the cue's (the own
    line is the nearest, and the widest declared hyperplane scheme, C4, would propose it with a
    chance under CUE_FAR by the collision law: not the bucketing's matter)."""
    if row["in_candidates"] or not row.get("planted", True):
        return None
    if (row["rank_under_c6"] is not None and row["rank_under_c6"] > 1) or \
            (row.get("best_other_cosine") is not None and row.get("cos_query_own") is not None
             and row["best_other_cosine"] >= row["cos_query_own"]):
        return "interference"
    if row["cos_query_own"] is not None and proposal_probability(row["cos_query_own"], 10, 4, radius=1) < CUE_FAR:
        return "cue"
    return "index"


# ---- the probe on one journal ----------------------------------------------------------------------
def probe_journal(model, journal: Journal, facts, neg, pointer_of: dict[str, str], k: int = 4,
                  index_seed: int = 0, configs: list[dict] | None = None) -> dict:
    """The file's content for one journal: the live index's entries, the query cues of the
    checkpoint's encoder, the rows per configuration and question, the never planted rows, the
    aggregates, and the self check that C0 is the live index's candidate set."""
    import torch
    from cortex_c2b.lm_bridge import JournalBridge, encode_query
    device = next(model.parameters()).device
    live = CueIndex(dim=journal.cue_dim, seed=index_seed)
    jp = JournalPath(journal, index=live, seed=index_seed)                  # the path indexes the journal (alive and aliased)
    assert jp.index is live
    bridge = JournalBridge(model, journal, journal_path=jp, k=k, seed=index_seed, device=device)
    vecs = {eid: np.asarray(v, dtype=np.float32) for eid, v in live.vecs.items()}
    pointer_of_id = {eid: jp._resolve(eid)[1] for eid in vecs}
    id_of_pointer: dict[str, str] = {}
    for eid, ptr in pointer_of_id.items():
        id_of_pointer.setdefault(ptr, eid)
    with torch.no_grad():
        q_cues = bridge.cues([encode_query(f.query) for f in facts])
        n_cues = bridge.cues([encode_query(f.query) for f in neg])
    names = [c["name"] for c in (configs or CONFIGS)] + [REFERENCE["name"]]
    indexes = {n: BucketIndex(vecs, config(n), index_seed=index_seed, dim=journal.cue_dim) for n in names}
    all_ids = sorted(vecs)
    fly_P = indexes["C5a8"].fly_P if "C5a8" in indexes else fly_planes(journal.cue_dim, seed=index_seed)
    union_tags = np.unique(np.concatenate([fly_tag(v, fly_P) for v in vecs.values()])) if vecs else np.array([], dtype=int)

    # the self check: C0 is the live index's candidate set, query by query
    control_ok = True
    for q in (np.concatenate([q_cues, n_cues]) if len(facts) + len(neg) else []):
        ids_live = {eid for eid, _ in live.query(q, k=max(1, len(live)))}
        if ids_live != set(indexes["C0"].candidates(q)):
            control_ok = False
    # C6 ranks over the whole store, once per question
    c6_rank, c6_best_other = [], []
    for i, f in enumerate(facts):
        q = q_cues[i]; own = id_of_pointer.get(pointer_of.get(f.entity))
        scored = sorted(((eid, _cos(q, vecs[eid])) for eid in all_ids), key=lambda x: (-x[1], x[0]))
        pos = next((r for r, (eid, _) in enumerate(scored) if eid == own), None)
        c6_rank.append(pos + 1 if pos is not None else None)
        c6_best_other.append(max((s for eid, s in scored if eid != own), default=None))
    questions = []
    for i, f in enumerate(facts):
        own_ptr = pointer_of.get(f.entity); own = id_of_pointer.get(own_ptr)
        questions.append({"i": i, "entity": f.entity, "own_pointer": own_ptr, "own_id": own, "planted": own is not None,
                          "cos_query_own": (_cos(q_cues[i], vecs[own]) if own in vecs else None),
                          "best_other_cosine": c6_best_other[i], "rank_under_c6": c6_rank[i]})
    rows, nrows, entry_signatures = {}, {}, {}
    for name, idx in indexes.items():
        rows[name] = []
        entry_signatures[name] = {eid: idx.signatures(v) for eid, v in vecs.items()} if idx.family != "scan" else {}
        for i, f in enumerate(facts):
            q = q_cues[i]; qa = questions[i]; own = qa["own_id"]
            cands = idx.candidates(q)
            ranked = idx.rank(q, cands)
            pos = next((r for r, (eid, _) in enumerate(ranked) if eid == own), None)
            others = [s for eid, s in ranked if eid != own]
            own_score = next((s for eid, s in ranked if eid == own), None)
            row = {"i": i, "planted": qa["planted"],
                   "in_candidates": pos is not None, "rank": (pos + 1 if pos is not None else None),          # under the configuration's `rank_by`
                   "n_candidates": len(cands), "candidate_ids": cands,
                   "margin": (own_score - max(others) if (own_score is not None and others) else None),
                   "cos_query_own": qa["cos_query_own"], "best_other_cosine": qa["best_other_cosine"], "rank_under_c6": qa["rank_under_c6"],
                   "signatures_query": idx.signatures(q), "signatures_own": (idx.signatures(vecs[own]) if own in vecs else None)}
            row["hamming"] = ([bin(a ^ b).count("1") for a, b in zip(idx.bits(q), idx.bits(vecs[own]))]
                              if own in vecs else None)                    # the exact bits' distance, the collapsed C0 included
            row["miss_class"] = miss_class(row)
            if name == "C0":                                                  # the fly's tag as a ranking of C0's candidates
                fly_ranked = indexes["C5a8"].rank(q, cands) if "C5a8" in indexes else []
                fpos = next((r for r, (eid, _) in enumerate(fly_ranked) if eid == own), None)
                row["fly_rank_on_c0"] = fpos + 1 if fpos is not None else None
                row["fly_overlap_own"] = next((s for eid, s in fly_ranked if eid == own), None)
                row["fly_top_overlap"] = fly_ranked[0][1] if fly_ranked else None
            rows[name].append(row)
        nrows[name] = []
        for j, f in enumerate(neg):
            q = n_cues[j]; cands = idx.candidates(q); ranked = idx.rank(q, cands)
            qt = fly_tag(q, fly_P)
            novelty = float(np.setdiff1d(qt, union_tags, assume_unique=True).size / qt.size) if qt.size else None
            nrows[name].append({"j": j, "entity": f.entity, "n_candidates": len(cands),
                                "best_score": (ranked[0][1] if ranked else None), "novelty": novelty})
    # the subset relations declared as properties, checked query by query, then the ids dropped from the
    # retained rows (the counts, the signatures and the query cues stay; the file must stay small)
    pairs = [("C1", "C0"), ("C1", "C2a"), ("C2a", "C2b"), ("C1", "C3a"), ("C3a", "C3b"), ("C1", "C4")]
    subset = {f"{a}_in_{b}": all(set(ra["candidate_ids"]) <= set(rb["candidate_ids"]) for ra, rb in zip(rows[a], rows[b]))
              for a, b in pairs if a in rows and b in rows}
    for name in names:
        for r in rows[name]:
            r.pop("candidate_ids", None)
    aggregates = {n: aggregate(rows[n], nrows[n], rows["C0"], n_entries=len(vecs), k=k) for n in names}
    for n in names:
        aggregates[n]["admissible"] = admissible(aggregates[n], aggregates["C0"]) if n not in ("C0", "C6") else (n == "C0")
        for r in rows[n]:                                                  # the per question constants live in `questions`
            for key in ("cos_query_own", "best_other_cosine", "rank_under_c6"):
                r.pop(key, None)
    duplicates = len(pointer_of_id) - len(id_of_pointer)
    return {"probe": "the bucket probe (ADR-008, amendment of 2026-10-12, evening; RES-24): the index's buckets under the declared configurations",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "k": k, "index_seed": index_seed, "facts_seed": 0, "n_facts": len(facts), "n_negctrl": len(neg),
            "budget": BUDGET, "margin": MARGIN, "cue_far": CUE_FAR,
            "configurations": {n: idx.describe() for n, idx in indexes.items()},
            "control_matches_live_index": control_ok, "subset_relations": subset,
            "journal": {"entries": len(vecs), "aliases": len(live.alias), "duplicate_pointers": duplicates},
            "questions": questions, "entry_signatures": entry_signatures,
            "query_cues": {"questions": [[float("%.9g" % x) for x in q] for q in q_cues],          # nine significant digits: float32 exactly
                           "never_planted": [[float("%.9g" % x) for x in q] for q in n_cues]},
            "rows": rows, "never_planted": nrows, "aggregates": aggregates,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}


# ---- the aggregates, the budget, the readings -----------------------------------------------------
def _cost(costs: list[int]) -> dict:
    a = np.asarray(costs, dtype=np.float64) if costs else np.asarray([0.0])
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": int(a.max())}


def aggregate(rows: list[dict], nrows: list[dict], control_rows: list[dict], n_entries: int, k: int = 4) -> dict:
    """The aggregates of one configuration from its rows. The recalls and the paired counts are read
    over the planted questions (an entity the gate refused has no own line to propose; the v9
    journals hold all 200), the costs over every question."""
    planted = [r.get("planted", True) for r in rows]
    prow = [r for r, p in zip(rows, planted) if p]
    n = max(1, len(prow))
    hits = [bool(r["in_candidates"]) for r in prow]
    c0 = [bool(r["in_candidates"]) for r, p in zip(control_rows, planted) if p]
    margins = [r["margin"] for r in prow if r.get("margin") is not None]
    nscored = [r["best_score"] for r in nrows if r.get("best_score") is not None]
    out = {"n_planted": len(prow), "candidate_recall": sum(hits) / n,
           "top_k_recall": sum(1 for r in prow if r["rank"] is not None and r["rank"] <= k) / n,
           "n_in_candidates": sum(hits), "misses": [r["i"] for r in prow if not r["in_candidates"]] if prow and "i" in prow[0] else [],
           "R": sum(1 for h, c in zip(hits, c0) if h and not c), "L": sum(1 for h, c in zip(hits, c0) if c and not h),
           "cost": _cost([r["n_candidates"] for r in rows]),
           "share_mean": (float(np.mean([r["n_candidates"] for r in rows])) / n_entries if rows and n_entries else 0.0),
           "empty_share": sum(1 for r in rows if r["n_candidates"] == 0) / max(1, len(rows)),
           "never_planted_cost": _cost([r["n_candidates"] for r in nrows]),
           "never_planted_empty_share": (sum(1 for r in nrows if r["n_candidates"] == 0) / len(nrows) if nrows else 0.0),
           "n_margins": len(margins), "margin_mean": (float(np.mean(margins)) if margins else None),
           "n_negctrl_scored": len(nscored), "negctrl_best_max": (float(max(nscored)) if nscored else None),
           "miss_classes": {c: sum(1 for r in prow if r.get("miss_class") == c) for c in ("index", "interference", "cue")}}
    return out


def admissible(agg: dict, control: dict) -> bool:
    """The budget: on the questions and on the never planted entities alike, the 95th percentile at most
    50 and the maximum below 200; the mean at most four times the control's mean on the same journal."""
    for c in (agg["cost"], agg["never_planted_cost"]):
        if c["p95"] > BUDGET["p95_at_most"] or c["max"] >= BUDGET["max_below"]:
            return False
    return agg["cost"]["mean"] <= BUDGET["mean_at_most_times_control"] * control["cost"]["mean"] + 1e-9


def _tables_of(name: str) -> int:
    c = config(name)
    return c.get("n_tables", 1 if c["family"] == "fly_pseudo" else 0)


def _order(name: str) -> int:
    """The declared order of the configurations (the amendment's list), not the lexical one: C5a8 before C5a16."""
    return [c["name"] for c in CONFIGS].index(name)


def _select(names: list[str], journals: list[dict], idx: list[int]) -> str | None:
    """The declared selection as a total order (the build's precision, 2026-10-12): the cost of a
    configuration on a pair of journals is the mean of its two 95th percentiles; the tie set is every
    candidate within one candidate per query of the lowest cost; within the tie set, fewer tables
    (the fly's pseudo hash counts as one), then the exact signature, then the lower cost, then the name."""
    if not names:
        return None
    cost = {n: float(np.mean([journals[i]["aggregates"][n]["cost"]["p95"] for i in idx])) for n in names}
    low = min(cost.values())
    tie = [n for n in names if cost[n] <= low + TIE_CANDIDATES]
    return min(tie, key=lambda n: (_tables_of(n), config(n).get("signature") != "exact", cost[n], _order(n)))


def read_three(journals: list[dict]) -> dict:
    """The amendment's readings, exhaustive, from the aggregates of the three journals alone. Every
    declared configuration must be in every file (a file of another probe is refused, not read around)."""
    missing = [(i, c["name"]) for i, j in enumerate(journals) for c in CONFIGS if c["name"] not in j["aggregates"]]
    if missing:
        raise ValueError(f"read_three: configurations missing from the files: {missing}")
    names = [c["name"] for c in CONFIGS if c["name"] != "C0"]
    per = {}
    for n in names:
        aggs = [j["aggregates"][n] for j in journals]
        ctrl = [j["aggregates"]["C0"] for j in journals]
        above = []
        for a, c in zip(aggs, ctrl):
            if c["candidate_recall"] == 1.0 and a["candidate_recall"] == 1.0:
                above.append(True)                                        # not below, on a journal the control saturates
            else:
                above.append(a["R"] - a["L"] >= MARGIN)
        per[n] = {"admissible": all(a["admissible"] for a in aggs),
                  "at_1000": [a["candidate_recall"] == 1.0 for a in aggs], "above": above,
                  "recall": [a["candidate_recall"] for a in aggs], "R": [a["R"] for a in aggs], "L": [a["L"] for a in aggs],
                  "p95": [a["cost"]["p95"] for a in aggs]}
    cands_a = sorted(n for n, p in per.items() if p["admissible"] and all(p["at_1000"]) and all(p["above"]))
    above_everywhere = sorted(n for n, p in per.items() if p["admissible"] and all(p["above"]))
    out = {"per_configuration": per, "candidates_at_1000": cands_a, "above_control_everywhere": above_everywhere,
           "selected": None, "rotations": [], "rotations_failed": 0, "excluded_by_rotation": [], "selected_label": None,
           "c1_loses_nothing": None, "adopt_c1": False}
    if "C1" in per:
        out["c1_loses_nothing"] = all(l == 0 for l in per["C1"]["L"])
    pool = cands_a if cands_a else above_everywhere
    if cands_a:
        out["letter"] = "a"
    elif above_everywhere:
        out["letter"] = "b"
        best_hits = max(sum(per[n]["recall"]) for n in above_everywhere)
        pool = [n for n in above_everywhere if sum(per[n]["recall"]) == best_hits]
    else:
        out["letter"] = "c"
        out["adopt_c1"] = bool(out["c1_loses_nothing"]) and per["C1"]["admissible"]      # an inadmissible configuration enters no reading
        return out
    # the rotation, as the words: on each pair of journals the pool is the admissible configurations at 1.000
    # (reading (a)) or above the control with the most hits (reading (b)) on THAT pair; the chosen one must
    # hold on the third (at 1.000 under (a), above the control under (b)); a configuration that fails any
    # rotation is excluded and the selection reruns among the survivors
    def pair_pool(others, excluded):
        if cands_a:
            return [n for n in per if per[n]["admissible"] and n not in excluded
                    and all(per[n]["at_1000"][i] and per[n]["above"][i] for i in others)]
        above_pair = [n for n in per if per[n]["admissible"] and n not in excluded and all(per[n]["above"][i] for i in others)]
        if not above_pair:
            return []
        best = max(sum(per[n]["recall"][i] for i in others) for n in above_pair)
        return [n for n in above_pair if sum(per[n]["recall"][i] for i in others) == best]
    excluded, rotations, failed = set(), [], 0
    while True:
        rotations, chosen_all, any_fail = [], [], False
        for left in range(len(journals)):
            others = [i for i in range(len(journals)) if i != left]
            chosen = _select(pair_pool(others, excluded), journals, others)
            holds = (per[chosen]["at_1000"][left] if cands_a else per[chosen]["above"][left]) if chosen else False
            rotations.append({"left_out": left, "chosen": chosen, "holds_on_left_out": bool(holds)})
            chosen_all.append(chosen)
            if chosen is not None and not holds:
                excluded.add(chosen); any_fail = True; failed += 1
        if not any_fail:
            break
    survivors = [n for n in chosen_all if n is not None]
    if not survivors:
        selected = None
    elif len(set(survivors)) == 1:
        selected = survivors[0]
    else:                                                                   # a disagreement breaks toward more tables, then the later name
        selected = max(set(survivors), key=lambda n: (_tables_of(n), _order(n)))
    out["rotations"], out["selected"], out["rotations_failed"], out["excluded_by_rotation"] = rotations, selected, failed, sorted(excluded)
    out["selected_label"] = (f"selected among {len(CONFIGS) - 1} configurations on the questions of {len(journals)} journals, "
                             f"{failed} rotation(s) failed; confirmed on the first journal sealed after the adoption (the prospective holdout)"
                             if selected is not None else "no configuration survives the rotation; nothing adopted")
    out["adopt_c1"] = selected == "C1"
    return out


# ---- the precondition: C0 reproduces row 38 on the seed 2024 journal ---------------------------------
ROW38_RUN_ID = "4cfd6f6e1406"


ROW38_TOLERANCE = 1e-6          # the two float scalars of row 38 are float32 cosines (a ulp near 0.7 is 6e-8); the integers are exact


def check_row38(summary: dict, row38: dict, k: int | None = None, n_facts: int | None = None, n_negctrl: int | None = None) -> tuple[bool, list[str]]:
    """The precondition: C0 reproduces every scalar of row 38's file; the recalls and the counts exactly, the
    two float means at the declared tolerance; the file's k, n_facts and n_negctrl when given."""
    c = row38["rankings"]["cosine"]
    expected = {"candidate_recall": c["own_in_candidates"], "top_k_recall": c["own_in_top_k"],
                "n_in_candidates": c["own_rank"]["n"], "n_margins": c["own_minus_best_other"]["n"],
                "margin_mean": c["own_minus_best_other"]["mean"], "n_negctrl_scored": c["negctrl_best_score"]["n"],
                "negctrl_best_max": c["negctrl_best_score"]["max"]}
    diffs = []
    for key, e in expected.items():
        v = summary.get(key)
        if key in ("candidate_recall", "top_k_recall"):
            same = v is not None and math.isclose(float(v), float(e), rel_tol=0.0, abs_tol=1e-12)   # a count over 200
        elif isinstance(e, int) and not isinstance(e, bool):
            same = (v == e)
        else:
            same = v is not None and math.isclose(float(v), float(e), rel_tol=0.0, abs_tol=ROW38_TOLERANCE)
        if not same:
            diffs.append(f"{key}: {v} != {e}")
    for key, given in (("k", k), ("n_facts", n_facts), ("n_negctrl", n_negctrl)):
        if given is not None and row38.get(key) != given:
            diffs.append(f"{key}: {given} != {row38.get(key)}")
    return (not diffs), diffs


# ---- the command line -----------------------------------------------------------------------------
def summary_lines(out: dict) -> list[str]:
    lines = [f"bucket probe  entries {out['journal']['entries']}  k {out['k']}  index seed {out['index_seed']}  "
             f"control matches live index {out['control_matches_live_index']}  subset relations {all(out['subset_relations'].values())}"]
    for n, a in out["aggregates"].items():
        lines.append(f"  {n:6s} candidate recall {a['candidate_recall']:.3f}  top k {a['top_k_recall']:.3f}  R {a['R']:3d}  L {a['L']:3d}  "
                     f"cost mean {a['cost']['mean']:.1f} p95 {a['cost']['p95']:.0f} max {a['cost']['max']:3d}  admissible {a['admissible']}")
    return lines


def _cli(argv=None):
    ap = argparse.ArgumentParser(description="the probe of the index's buckets on a checkpoint and its sealed journal (ADR-008, 2026-10-12, evening)")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--journal", required=True)
    ap.add_argument("--config", required=True, help="the run's config json (the model's shape)")
    ap.add_argument("--seed", type=int, default=0, help="the protocol's facts: 0, as every run (an error otherwise)")
    ap.add_argument("--index-seed", type=int, default=0, help="the index seed, 0 as the live index (recorded apart from the facts' seed)")
    ap.add_argument("--n-facts", type=int, default=N_FACTS); ap.add_argument("--n-negctrl", type=int, default=N_NEGCTRL)
    ap.add_argument("--out", default="metrics/mqar", help="a directory (the file is bucket-probe-<run id>.json) or a .json path; never overwritten")
    a = ap.parse_args(argv)
    if a.seed != 0:
        raise SystemExit("bucket probe: the protocol's facts are seed 0 on every journal of the record; give --index-seed for the index")
    if a.n_facts != N_FACTS or a.n_negctrl != N_NEGCTRL:
        raise SystemExit(f"bucket probe: the protocol's counts are {N_FACTS} facts and {N_NEGCTRL} never planted entities; not a lever")
    out_path = Path(a.out)

    def never_overwritten(path: Path) -> Path:
        failed = path.with_name(path.stem + "-FAILED-precondition.json")
        for p in (path, failed):
            if p.exists():
                raise SystemExit(f"bucket probe: {p} exists; a file of the record is never overwritten, give another --out")
        return failed

    if out_path.suffix == ".json":
        failed_path = never_overwritten(out_path)
    jpath = Path(a.journal)
    if not jpath.is_file():                                                 # a wrong path must never create an empty journal
        raise SystemExit(f"bucket probe: {jpath} is not a file; the journal of the record is `<dir>/journal.jsonl`")
    payload_dir = jpath.with_name(jpath.stem + ".payloads")
    if not payload_dir.is_dir():
        raise SystemExit(f"bucket probe: {payload_dir} is not a directory; the sealed journal's payload store sits next to its log")
    from cortex_c2b.crypto import key_from_env
    key = key_from_env()
    if key is None:
        raise SystemExit("bucket probe: QUANTUM_CORTEX_JOURNAL_KEY is not in the environment; the journal of the record is sealed")
    row38_path = REPO_ROOT / "metrics/mqar/retrieval-probe-4cfd6f6e1406.json"
    row38 = json.loads(row38_path.read_text(encoding="utf-8"))                  # read before anything runs
    import shutil
    import subprocess
    import tempfile
    import torch
    import train
    cfg = train.load_config(a.config)
    if cfg.journal_k != row38["k"]:
        raise SystemExit(f"bucket probe: the config's journal_k is {cfg.journal_k}; the declared top k is row 38's, {row38['k']}")
    device = torch.device("cpu")
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    run_id = ck.get("run_id")
    if not run_id:
        raise SystemExit("bucket probe: the checkpoint carries no run id; the file would be nameless")
    if not ck.get("config_hash"):
        raise SystemExit("bucket probe: the checkpoint carries no config hash; the config cannot be checked as the run's")
    if train.config_hash(cfg) != ck["config_hash"]:
        raise SystemExit(f"bucket probe: the config hashes to {train.config_hash(cfg)}, the checkpoint was trained under {ck['config_hash']}; not the run's config")
    model = train.VanillaGPT(cfg).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    if out_path.suffix != ".json":
        out_path = out_path / f"bucket-probe-{run_id}.json"
        failed_path = never_overwritten(out_path)

    def tree_hash(log: Path, payloads: Path) -> dict:
        """The log's sha256 and the sha256 of the payload directory's sorted listing (name, size): what 'nothing is
        written' is checked against, before and after the run."""
        listing = "\n".join(f"{q.name} {q.stat().st_size}" for q in sorted(payloads.iterdir()))
        return {"log": hashlib.sha256(log.read_bytes()).hexdigest()[:16],
                "payloads": hashlib.sha256(listing.encode("utf-8")).hexdigest()[:16], "n_payload_files": len(listing.splitlines())}

    before = tree_hash(jpath, payload_dir)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=10).stdout.strip() or None
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=10).stdout.strip()
        tree_clean = (dirty == "") if commit else None
    except Exception:
        commit, tree_clean = None, None
    facts, gen_hash = generate_facts(a.n_facts, 0)
    neg, _ = generate_facts(a.n_negctrl, 10_000)
    # the journal of the record is never opened in place: the open of a journal probes its log for writability and
    # releases unreferenced payload files; the probe works on a copy in a temporary directory, deleted after
    with tempfile.TemporaryDirectory(prefix="bucket-probe-") as tmp:
        copy = Path(tmp) / jpath.name
        shutil.copy2(jpath, copy)
        shutil.copytree(payload_dir, copy.with_name(copy.stem + ".payloads"))
        try:
            journal = Journal(copy, key=key, policy=POLICY_STOP)
        except Exception as e:                                             # a wrong key: InvalidTag at the first sealed line
            raise SystemExit(f"bucket probe: the journal {jpath} does not open under the key given ({type(e).__name__}: {e})")
        pointer_of = {f.entity: content_hash(f.statement.encode("utf-8")) for f in facts
                      if content_hash(f.statement.encode("utf-8")) in journal.payloads}
        out = probe_journal(model, journal, facts, neg, pointer_of, k=cfg.journal_k, index_seed=a.index_seed)
        states = {s: sum(1 for e in journal._entries.values() if e.state == s) for s in sorted({e.state for e in journal._entries.values()})}
    after = tree_hash(jpath, payload_dir)
    out["checkpoint"] = {"path": str(a.ckpt), "run_id": run_id, "config_hash": ck["config_hash"],
                         "config_path": str(a.config), "config_sha256": hashlib.sha256(Path(a.config).read_bytes()).hexdigest()[:16]}
    out["journal"].update({"path": str(a.journal), "opened": "a copy in a temporary directory, deleted after the run",
                           "sha256_before": before["log"], "sha256_after": after["log"],
                           "payloads_before": before, "payloads_after": after, "unchanged": before == after, "states": states})
    out["generator"] = {"config_hash": gen_hash, "n_facts": a.n_facts, "n_negctrl": a.n_negctrl, "planted_found": len(pointer_of)}
    out["device"] = {"device": str(device), "hardware": platform.processor(), "threads": torch.get_num_threads()}
    out["code_commit"] = commit
    out["code_tree_clean"] = tree_clean
    out["precondition_row38"] = {"applies": run_id == ROW38_RUN_ID}
    stop = []
    if out["journal"]["entries"] == 0 or len(pointer_of) == 0:
        stop.append(f"the journal holds {out['journal']['entries']} indexed entries and {len(pointer_of)} planted facts of the protocol; not a journal of the record")
    if not out["control_matches_live_index"]:
        stop.append("the probe's control is not the live index's candidate set")
    if not all(out["subset_relations"].values()):
        stop.append("a declared subset relation fails: " + ", ".join(k for k, v in out["subset_relations"].items() if not v))
    if run_id == ROW38_RUN_ID:
        ok, diffs = check_row38(out["aggregates"]["C0"], row38, k=cfg.journal_k, n_facts=a.n_facts, n_negctrl=a.n_negctrl)
        out["precondition_row38"].update({"ok": ok, "differences": diffs, "tolerance": ROW38_TOLERANCE})
        if not ok:
            stop.append("the control does not reproduce row 38 on its journal: " + "; ".join(diffs))
    if stop:
        out["stopped"] = stop
        for section in ("rows", "never_planted", "aggregates", "entry_signatures"):   # no configuration is read from a failed run
            out[section] = {n: v for n, v in out[section].items() if n == "C0"}
        out["configurations_not_retained"] = [c["name"] for c in CONFIGS if c["name"] != "C0"] + [REFERENCE["name"]]
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        failed_path.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
        raise SystemExit("bucket probe: stopped before any configuration is read: " + " | ".join(stop) + f"  (retained: {failed_path}, C0 alone)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")   # compact: the rows are many
    for line in summary_lines(out):
        print(line)
    print(f"[record] retained: {out_path}  sha256 {hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    _cli()
