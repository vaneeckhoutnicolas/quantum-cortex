"""cortex_c2b.references -- the simple references of ADR-010 part B, made exact by the amendment of 2026-10-15 on parts B
and C (graved 2026-10-29), as new modules only: the frozen system is `c5b4818` and nothing here changes a frozen module.

Three indexes read one journal, so that they hold the same entries by construction:

  I0  the organ's live index (sixteen tables of ten bits, the exact signature; `CueIndex`), read by the frozen
      `JournalBridge` as every session B of the record reads it;
  I1  an exhaustive cosine over the same cues, no buckets: every live entry ranked by the index's own cosine
      (`CueIndex.query`'s arithmetic: the stored cue as the journal rounds it, four decimals, cast to float32, against
      the query's float32 cue, each norm raised by 1e-8), computed entry by entry as the index computes it and not by a
      batched product, ties on bit equal values by write order (B.1 (c)); labelled "the organ's cues without the
      buckets, a component ablation; no general purpose encoder measured";
  I2  a lexical search, BM25 (B.1 (b)): the document is the stored payload's bytes, the query the question's UTF-8
      bytes, the one token rule (lower cased ASCII, split at every byte outside a to z and 0 to 9, no stemming, no stop
      list), the inverse document frequency ln(1 + (N - df + 0.5) / (df + 0.5)), the term frequency saturation with k1
      and b, |d| the document's token count with repetitions, avgdl their mean over the live documents, the statistics
      maintained at every insertion; every live entry scored (zero when it shares no token), the window the first four
      by score descending then write order ascending, so that the fillers are the earliest written documents.

The reference bridge (B.1 (a)) derives from the frozen `JournalBridge` and keeps its cue encoder, `marks_for`,
`build_read_window`, budget and labels; it replaces retrieval alone: its `read` returns a `ReadResult` whose pointers are
the reference's top four in the reference's rank order and whose scores are the organ's cosines of those lines by the
frozen `marks_for`, so that the window prints the organ's marks and the veto reads them. Its `jp.index` exposes what the
frozen `_probe` reads with `record_index`: `touched` as the reference's operation count per read, `n_tables` and `_sig`
the organ's under the adopted geometry, `candidates` every live entry, `vecs` the stored cues; so on the reference passes
`own_in_candidates` is the own line's liveness and `own_rank` its rank by the organ's cosine under the whole store.

Also here: V1, the external verifier (the citation stands if the cited line contains the question's entity, read from
the question by the generator's name pattern, exactly one match else abstention, and the answered value, byte for byte;
labelled "a key rule, exact on these families by construction"), and the selection of BM25's (k1, b) on the first
family's 200 direct queries by the organ's criterion of row 45 (the own line among the four), computed once before any
family is planted, the ties to (1.5, 0.75). Nothing here is a number of the record.
"""
from __future__ import annotations

import math
import re
from collections import OrderedDict

import numpy as np

from cortex_c2b import Journal, STATE_EVICTED, content_hash
from cortex_c2b.hm_protocol import generate_facts, value_of_statement, N_FACTS
from cortex_c2b.lm_bridge import JournalBridge, ReadResult, build_read_window
from cortex_c2b.read_path import JournalPath

INDEXES = ("I0", "I1", "I2")
I1_LABEL = "the organ's cues without the buckets, a component ablation; no general purpose encoder measured"
I2_LABEL = "a lexical search, BM25 over the stored payloads' bytes; under I2 the window's other lines are the earliest written documents of the tied score"
V1_LABEL = "a key rule, exact on these families by construction"
NAME_PATTERN = re.compile(r"[BDFGKLMNPRSTVZ][aeiou][bdfgklmnprstvz][aeiou]-[0-9a-f]{4}")
K1_GRID = (0.6, 0.9, 1.2, 1.5, 2.0)
B_GRID = (0.4, 0.75)
TEXTBOOK_PAIR = (1.5, 0.75)
TOKEN_RE = re.compile(rb"[a-z0-9]+")


# ---------------------------------------------------------------------------- #
# BM25 (I2)                                                                     #
# ---------------------------------------------------------------------------- #
def tokens(data: bytes | str) -> list[str]:
    """The one token rule: the bytes lower cased in ASCII, split at every byte outside a to z and 0 to 9 (a name gives its
    letters and its hexadecimal part; the apostrophe splits, "'s" gives `s`; a byte of an em dash splits)."""
    b = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    return [t.decode("ascii") for t in TOKEN_RE.findall(b.lower())]


def distinct_in_order(toks: list[str]) -> list[str]:
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out


def idf(n_docs: int, df: int) -> float:
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


class BM25Index:
    """BM25 over documents inserted in write order, the statistics (N, the document frequencies, the lengths, avgdl)
    maintained at every insertion and removal; a read traverses the postings of the query's distinct tokens alone."""

    def __init__(self, k1: float = TEXTBOOK_PAIR[0], b: float = TEXTBOOK_PAIR[1]):
        self.k1, self.b = float(k1), float(b)
        self.postings: dict[str, dict[str, int]] = {}      # token -> {doc -> tf}
        self.doclen: dict[str, int] = {}                   # doc -> token count with repetitions
        self.order: OrderedDict[str, int] = OrderedDict()  # doc -> write order (insertion rank)
        self._next = 0
        self.total_len = 0
        self.touched = 0                                   # postings read, cumulative (the reference's operation count)

    @property
    def n_docs(self) -> int:
        return len(self.doclen)

    @property
    def avgdl(self) -> float:
        return (self.total_len / self.n_docs) if self.n_docs else 0.0

    def insert(self, doc: str, data: bytes | str) -> None:
        if doc in self.doclen:
            raise ValueError(f"document {doc} already indexed")
        toks = tokens(data)
        self.doclen[doc] = len(toks); self.total_len += len(toks)
        self.order[doc] = self._next; self._next += 1
        for t in toks:
            self.postings.setdefault(t, {})
            self.postings[t][doc] = self.postings[t].get(doc, 0) + 1

    def remove(self, doc: str) -> None:
        """An evicted entry leaves the corpus (the statistics follow)."""
        if doc not in self.doclen:
            return
        self.total_len -= self.doclen.pop(doc)
        self.order.pop(doc, None)
        for t in list(self.postings):
            if doc in self.postings[t]:
                del self.postings[t][doc]
                if not self.postings[t]:
                    del self.postings[t]

    def score_of(self, doc: str, qtoks: list[str]) -> float:
        """The score of one document for the query's distinct tokens, in first occurrence order (the summation order)."""
        n, avg = self.n_docs, self.avgdl
        dl = self.doclen[doc]
        s = 0.0
        for t in qtoks:
            post = self.postings.get(t)
            if not post or doc not in post:
                continue
            tf = post[doc]
            s += idf(n, len(post)) * tf * (self.k1 + 1.0) / (tf + self.k1 * (1.0 - self.b + self.b * dl / avg))
        return s

    def rank(self, query: bytes | str, k: int = 4) -> dict:
        """The window: the first `k` documents by score descending then write order ascending; the fillers (score zero)
        the earliest written documents. Returns {"docs": [...], "scores": [...], "postings_read": n, "scored_nonzero": m}."""
        qtoks = distinct_in_order(tokens(query))
        n, avg = self.n_docs, self.avgdl
        scores: dict[str, float] = {}
        read = 0
        for t in qtoks:
            post = self.postings.get(t)
            if not post:
                continue
            read += len(post)
            w = idf(n, len(post))
            for doc, tf in post.items():
                dl = self.doclen[doc]
                scores[doc] = scores.get(doc, 0.0) + w * tf * (self.k1 + 1.0) / (tf + self.k1 * (1.0 - self.b + self.b * dl / avg))
        self.touched += read
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.order[kv[0]]))[:k]
        docs = [d for d, _ in ranked]; vals = [s for _, s in ranked]
        if len(docs) < k:                                  # the fillers: the earliest written documents at score zero
            for doc in self.order:                         # insertion order (write order)
                if len(docs) >= k:
                    break
                if doc not in scores:
                    docs.append(doc); vals.append(0.0)
        return {"docs": docs, "scores": vals, "postings_read": read, "scored_nonzero": len(scores), "query_tokens": qtoks}

    def recompute_all(self, query: bytes | str) -> list[tuple[str, float]]:
        """Every live document scored from the postings' own counts (a test, not a read): (doc, score) by the window's order."""
        qtoks = distinct_in_order(tokens(query))
        return sorted(((d, self.score_of(d, qtoks)) for d in self.doclen), key=lambda kv: (-kv[1], self.order[kv[0]]))

    def parameters(self) -> dict:
        return {"k1": self.k1, "b": self.b}


def bm25_selection(statements: list[str] | None = None, queries: list[str] | None = None,
                   own: list[int] | None = None, k: int = 4) -> dict:
    """The selection of (k1, b) on the first family's 200 direct queries (B.1 (b)): the organ's criterion of row 45, the own
    line among the four; ties to (1.5, 0.75) when among the best, else the lowest k1 then the lowest b. By default the
    documents are the generator's 200 statements of seed 0 in generation order (the first family's journals hold exactly
    those, in that order) and the queries their direct queries; the ten counts are returned with the pair and its label."""
    if statements is None:
        facts, _ = generate_facts(N_FACTS, 0)
        statements = [f.statement for f in facts]; queries = [f.query for f in facts]; own = list(range(len(facts)))
    counts = {}
    for k1 in K1_GRID:
        for b in B_GRID:
            ix = BM25Index(k1, b)
            for i, s in enumerate(statements):
                ix.insert(str(i), s)
            hit = 0
            for q, o in zip(queries, own):
                r = ix.rank(q, k)
                hit += int(str(o) in r["docs"])
            counts[f"{k1},{b}"] = hit
    best = max(counts.values())
    best_pairs = sorted([tuple(float(x) for x in key.split(",")) for key, v in counts.items() if v == best])
    if TEXTBOOK_PAIR in best_pairs:
        pair = TEXTBOOK_PAIR
    else:
        pair = sorted(best_pairs, key=lambda p: (p[0], p[1]))[0]
    tie = len(best_pairs) == len(counts)
    label = ("the textbook pair, confirmed and not selected by the first family" if (tie and pair == TEXTBOOK_PAIR)
             else "selected on the first family's 200 direct queries by the organ's criterion of row 45")
    first_family_label = ("BM25's parameters: the tie rule's, the criterion separating no pair on this protocol" if tie
                          else f"BM25's parameters: ({pair[0]}, {pair[1]}), selected on the first family's 200 direct queries")
    return {"counts": counts, "n_queries": len(queries), "k": k, "best": best, "all_tie": tie, "pair": {"k1": pair[0], "b": pair[1]},
            "label": label, "first_family_label": first_family_label, "grid": {"k1": list(K1_GRID), "b": list(B_GRID)},
            "rule": "the own line among the four on the first family's 200 direct queries; ties to (1.5, 0.75) when among the best, else the lowest k1 then the lowest b"}


# ---------------------------------------------------------------------------- #
# the exhaustive cosine (I1)                                                    #
# ---------------------------------------------------------------------------- #
class ExhaustiveCosine:
    """I1: every live entry ranked by the index's own cosine, computed entry by entry as `CueIndex.query` computes it
    (float32 stored cue, float32 query, each norm raised by 1e-8); ties on bit equal values by write order."""

    def __init__(self, journal: Journal, organ_path: JournalPath):
        self.j = journal
        self.jp = organ_path
        self.touched = 0                                   # cues compared, cumulative
        self._norms: dict[str, np.floating] = {}
        self._live: list[tuple[str, np.ndarray, np.floating]] | None = None   # the live corpus held still (a static journal)

    def _norm(self, eid: str) -> np.floating:
        """The stored cue's norm as the index computes it (numpy's own scalar, its dtype kept: the sum below is the index's
        expression term by term, so the two values are equal bit for bit)."""
        n = self._norms.get(eid)
        if n is None:
            n = np.linalg.norm(self.jp.index.vecs[eid])
            self._norms[eid] = n
        return n

    def live_ids(self) -> list[str]:
        """The live entries in write order (the journal's dict order is the log's)."""
        return [eid for eid, e in self.j._entries.items() if e.state != STATE_EVICTED and eid in self.jp.index.vecs]

    def hold(self, still: bool = True) -> None:
        """Hold the live corpus (ids, stored cues, norms) between reads of a journal nothing writes to (C.2's numbers); off
        again with `hold(False)`. The arithmetic per entry is the same either way."""
        self._live = [(eid, self.jp.index.vecs[eid], self._norm(eid)) for eid in self.live_ids()] if still else None

    def rank(self, cue: np.ndarray, k: int = 4) -> dict:
        q = np.asarray(cue, dtype=np.float32)
        qn = np.linalg.norm(q) + 1e-8
        corpus = self._live if self._live is not None else [(eid, self.jp.index.vecs[eid], self._norm(eid)) for eid in self.live_ids()]
        scored = []
        for pos, (eid, v, n) in enumerate(corpus):
            scored.append((eid, float(q @ v / (qn * (n + 1e-8))), pos))   # the index's arithmetic, the norm precomputed (equal bits)
        self.touched += len(scored)
        scored.sort(key=lambda x: (-x[1], x[2]))
        top = scored[:k]
        return {"docs": [e for e, _, _ in top], "scores": [s for _, s, _ in top], "compared": len(scored)}


# ---------------------------------------------------------------------------- #
# the reference bridge (B.1 (a))                                                 #
# ---------------------------------------------------------------------------- #
class _IndexView:
    """What the frozen `_probe` reads of `bridge.jp.index` on a reference pass: `touched` the reference's operation count,
    `n_tables` and `_sig` the organ's, `candidates` every live entry, `vecs` the stored cues."""

    def __init__(self, organ_path: JournalPath, reference, journal: Journal):
        self._organ = organ_path
        self._ref = reference
        self._j = journal

    @property
    def touched(self) -> int:
        return int(self._ref.touched)

    @property
    def n_tables(self) -> int:
        return self._organ.index.n_tables

    @property
    def vecs(self) -> dict:
        return self._organ.index.vecs

    def _sig(self, v, t: int) -> int:
        return self._organ.index._sig(v, t)

    def geometry(self) -> dict:
        return self._organ.index.geometry()

    def candidates(self, cue) -> list[str]:
        return sorted(eid for eid, e in self._j._entries.items() if e.state != STATE_EVICTED and eid in self._organ.index.vecs)


class ReferenceJournalPath:
    """A journal path whose retrieval is the reference's and whose scores are the organ's cosines of the returned lines
    (the frozen `cue_similarity`); the organ's own path stays underneath for the marks and the index view."""

    def __init__(self, organ_path: JournalPath, reference, name: str):
        self.organ = organ_path
        self.j = organ_path.j
        self.reference = reference
        self.name = name
        self.index = _IndexView(organ_path, reference, organ_path.j)
        self.index_overridden = getattr(organ_path, "index_overridden", False)
        self.last = None                                   # the reference's own ranking of the last read

    def on_write(self, entry) -> None:
        self.organ.on_write(entry)
        if hasattr(self.reference, "insert_entry"):
            self.reference.insert_entry(entry)

    def cue_similarity(self, span_cue, pointer: str) -> float:
        return self.organ.cue_similarity(span_cue, pointer)

    def retrieve(self, span_cue, k: int = 4, query_text: str | None = None):
        q = np.asarray(span_cue, dtype=np.float32)
        if self.name == "I2":
            if query_text is None:
                raise ValueError("I2 ranks the question's bytes: the reference bridge sets the query text before the read")
            r = self.reference.rank(query_text, k=k)
        else:
            r = self.reference.rank(q, k=k)
        out = []
        for eid in r["docs"]:
            e = self.j._entries[eid]
            if e.state == STATE_EVICTED:
                continue
            out.append((e, self.j.payloads.get(e.pointer), self.organ.cue_similarity(q, e.pointer)))
        self.last = {"index": self.name, "entry_ids": list(r["docs"]), "pointers": [self.j._entries[e].pointer for e in r["docs"]],
                     "reference_scores": [float(s) for s in r["scores"]],
                     "operations": int(r.get("postings_read", r.get("compared", 0))),
                     **({"query_tokens": r["query_tokens"], "scored_nonzero": r["scored_nonzero"]} if "query_tokens" in r else {})}
        return out


class BM25Reference:
    """I2 bound to a journal: the live entries' payload bytes as documents in write order."""

    def __init__(self, journal: Journal, k1: float, b: float):
        self.j = journal
        self.bm25 = BM25Index(k1, b)
        for eid, e in journal._entries.items():
            if e.state != STATE_EVICTED:
                self.bm25.insert(eid, journal.payloads.get(e.pointer))

    @property
    def touched(self) -> int:
        return self.bm25.touched

    def insert_entry(self, entry) -> None:
        self.bm25.insert(entry.entry_id, self.j.payloads.get(entry.pointer))

    def rank(self, query_text: str, k: int = 4) -> dict:
        return self.bm25.rank(query_text, k=k)


class ReferenceBridge(JournalBridge):
    """The frozen bridge with retrieval replaced (B.1 (a)): the cue encoder, `marks_for`, `build_read_window`, the budget and
    the labels are the frozen ones; `read` returns the reference's top four in the reference's order with the organ's
    cosines as scores. For I2 the question's text must be set before the read (`set_query`), since `_probe` hands the
    bridge the cue alone."""

    def __init__(self, model, journal: Journal, index_name: str, k: int = 4, budget_bytes: int = 256, seed: int = 0,
                 shuffle_seed: int | None = None, device=None, mark: bool = False, index_geometry: dict | None = None,
                 admission: str | None = None, journal_path: JournalPath | None = None, bm25_parameters: tuple[float, float] | None = None):
        super().__init__(model, journal, journal_path=journal_path, k=k, budget_bytes=budget_bytes, seed=seed, shuffle_seed=shuffle_seed,
                         device=device, mark=mark, index_geometry=index_geometry, admission=admission)
        if index_name not in ("I1", "I2"):
            raise ValueError("the reference bridge serves I1 or I2; I0 is the frozen bridge")
        organ = self.jp
        if index_name == "I1":
            ref = ExhaustiveCosine(journal, organ)
        else:
            k1, b = bm25_parameters or TEXTBOOK_PAIR
            ref = BM25Reference(journal, k1, b)
        self.index_name = index_name
        self.organ_path = organ
        self.jp = ReferenceJournalPath(organ, ref, index_name)
        self._query_text: str | None = None

    def set_query(self, text: str) -> None:
        self._query_text = text

    def read(self, cue: np.ndarray) -> ReadResult:
        hits = self.jp.retrieve(cue, k=self.k, query_text=self._query_text)
        items = [(e.pointer, payload) for e, payload, _ in hits]
        scores = [float(s) for _, _, s in hits]
        window, labels = build_read_window(items, self.budget, self.rng, marks=scores if self.mark else None)
        return ReadResult(window=window, labels=labels, pointers=[p for p, _ in items], scores=scores)

    def last_reference(self) -> dict | None:
        return self.jp.last


class OrganRetrievalReference:
    """A reference whose ranking is the organ's own retrieval (the test of B.1 (a): the reference bridge given it must
    reproduce the frozen bridge's windows byte for byte)."""

    def __init__(self, organ_path: JournalPath):
        self.jp = organ_path
        self.touched = 0

    def rank(self, cue, k: int = 4) -> dict:
        before = self.jp.index.touched
        hits = self.jp.index.query(np.asarray(cue, dtype=np.float32), k=k)
        self.touched += self.jp.index.touched - before
        return {"docs": [e for e, _ in hits], "scores": [s for _, s in hits], "compared": self.jp.index.touched - before}


# ---------------------------------------------------------------------------- #
# V1, the external verifier                                                      #
# ---------------------------------------------------------------------------- #
def entity_of_question(question: str) -> str | None:
    """The question's entity by the generator's name pattern: exactly one match, else None (abstention)."""
    m = NAME_PATTERN.findall(question)
    return m[0] if len(m) == 1 else None


def v1_outcome(question: str, cited_payload: bytes | None, answered: str | None, expected: str,
               organ_value: bool = False) -> dict:
    """The external verifier on the decoder's citation (B.1): the citation stands if the cited line contains the
    question's entity and the answered value, byte for byte; with `organ_value` the value is read from the cited line by
    the organ's reader (`value_of_statement`), so that on a line no template reads the system abstains. Returns
    {kind, answered, outcome, entity, reason} with the record's outcome vocabulary (strict | valid | invalid | abstain)."""
    ent = entity_of_question(question)
    if cited_payload is None:
        return {"kind": "unknown", "answered": None, "outcome": "abstain", "entity": ent, "reason": "no citation"}
    if ent is None:
        return {"kind": "unknown", "answered": None, "outcome": "abstain", "entity": None, "reason": "the question carries no single name"}
    if organ_value:
        v = value_of_statement(cited_payload.decode("utf-8", errors="replace"))
        answered = v[2] if v else None
        if answered is None:
            return {"kind": "unknown", "answered": None, "outcome": "abstain", "entity": ent, "reason": "the organ reads no value on the cited line"}
    if not answered:
        return {"kind": "unknown", "answered": None, "outcome": "abstain", "entity": ent, "reason": "no value answered"}
    ok = ent.encode("utf-8") in cited_payload and answered.encode("utf-8") in cited_payload
    if not ok:
        return {"kind": "unknown", "answered": None, "outcome": "abstain", "entity": ent,
                "reason": "the cited line does not carry the question's entity and the answered value byte for byte"}
    return {"kind": "cite", "answered": answered, "outcome": ("strict" if answered == expected else "valid"), "entity": ent,
            "reason": "the cited line carries the entity and the value"}
