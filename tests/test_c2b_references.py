"""The simple references of ADR-010 part B (the amendment of 2026-10-15 on parts B and C, B.1): BM25 as declared, computed
by hand to twelve decimals on a five statement corpus; the exhaustive cosine bit equal to the index's own arithmetic; V1;
the selection of (k1, b) on the first family; the reference bridge against the frozen bridge. New modules only."""
import math

import numpy as np
import pytest

from cortex_c2b import Journal, POLICY_STOP, content_hash
from cortex_c2b.read_path import CueIndex, JournalPath, ADOPTED_GEOMETRY
from cortex_c2b.write_path import WritePath
from cortex_c2b.hm_protocol import generate_facts, N_FACTS
from cortex_c2b import references as R


# ---------------------------------------------------------------------------- #
# the token rule                                                                 #
# ---------------------------------------------------------------------------- #
def test_the_one_token_rule():
    assert R.tokens("Tone-4010's trade") == ["tone", "4010", "s", "trade"]           # a name gives its letters and its hexadecimal part; "'s" gives s
    assert R.tokens("Where — is it?") == ["where", "is", "it"]                      # the em dash's bytes split, like any byte outside the alphabet
    assert R.tokens(b"ABC def\x00ghi") == ["abc", "def", "ghi"]
    assert R.tokens("") == []
    assert R.distinct_in_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


# ---------------------------------------------------------------------------- #
# BM25 by hand (B.1 (b)): five statements, one repeated query token, one "'s", two lengths, a document sharing no token,
# two documents tied at the window's boundary                                     #
# ---------------------------------------------------------------------------- #
CORPUS = ["Tone-4010's trade is cobbler",                                 # d0: six tokens
          "the long trade of Tone-4010, trade and trade again",            # d1: ten tokens, trade three times
          "Kilu-77ab's trade is miller",                                   # d2: six tokens, tied with d3
          "Rabo-1c9d's trade is sawyer",                                   # d3: six tokens, tied with d2
          "nothing here matches"]                                          # d4: shares no token with the query
QUERY = "what is Tone-4010's trade trade"                                 # "trade" repeated: counted once; "what" meets no document

# the scores written out: N = 5, avgdl = 31 / 5 = 6.2, k1 = 1.5, b = 0.75; idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)) with
# df(is) = 3, df(tone) = 2, df(4010) = 2, df(s) = 3, df(trade) = 4, df(what) = 0; K(d) = k1 (1 - b + b |d| / avgdl)
HAND = {"d0": 3.162520098441034, "d1": 1.7881805259477486, "d2": 1.3857914007015322, "d3": 1.3857914007015322, "d4": 0.0}


def hand_score(doc_tokens: list[str]) -> float:
    """The declared formula term by term, independent of the module (the test's own arithmetic)."""
    N = 5; avgdl = 31 / 5; k1, b = 1.5, 0.75
    corpus = [R.tokens(s) for s in CORPUS]
    s = 0.0
    for t in ["what", "is", "tone", "4010", "s", "trade"]:
        tf = doc_tokens.count(t)
        if tf == 0:
            continue
        df = sum(1 for d in corpus if t in d)
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
        s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc_tokens) / avgdl))
    return s


def test_bm25_scores_by_hand_to_twelve_decimals():
    ix = R.BM25Index(1.5, 0.75)
    for i, s in enumerate(CORPUS):
        ix.insert(f"d{i}", s)
    assert ix.n_docs == 5 and abs(ix.avgdl - 6.2) < 1e-12
    assert abs(R.idf(5, 3) - math.log(1 + 2.5 / 3.5)) < 1e-15 and abs(R.idf(5, 4) - math.log(4 / 3)) < 1e-15
    qtoks = R.distinct_in_order(R.tokens(QUERY))
    assert qtoks == ["what", "is", "tone", "4010", "s", "trade"]
    for d, want in HAND.items():
        assert abs(ix.score_of(d, qtoks) - want) < 1e-12, d
        assert abs(hand_score(R.tokens(CORPUS[int(d[1])])) - want) < 1e-12, d
    r = ix.rank(QUERY, k=3)
    assert r["docs"] == ["d0", "d1", "d2"]                                      # the tie at the boundary: d2 before d3 by write order
    assert all(abs(a - HAND[d]) < 1e-12 for a, d in zip(r["scores"], r["docs"]))
    assert r["postings_read"] == 3 + 2 + 2 + 3 + 4 and r["scored_nonzero"] == 4     # the postings of the five tokens met; "what" reads none
    r4 = ix.rank(QUERY, k=4)
    assert r4["docs"] == ["d0", "d1", "d2", "d3"]
    r5 = ix.rank(QUERY, k=5)
    assert r5["docs"] == ["d0", "d1", "d2", "d3", "d4"] and r5["scores"][-1] == 0.0  # the filler: the earliest written document at score zero
    assert [d for d, _ in ix.recompute_all(QUERY)] == ["d0", "d1", "d2", "d3", "d4"]
    for d, s in ix.recompute_all(QUERY):
        assert abs(s - HAND[d]) < 1e-12


def test_bm25_statistics_follow_every_insertion_and_removal():
    ix = R.BM25Index()
    for i, s in enumerate(CORPUS):
        ix.insert(f"d{i}", s)
    before = ix.rank(QUERY, 5)
    ix.remove("d1")
    assert ix.n_docs == 4 and abs(ix.avgdl - 21 / 4) < 1e-12 and "long" not in ix.postings
    after = ix.rank(QUERY, 5)
    assert after["docs"] == ["d0", "d2", "d3", "d4"] and after["scores"][0] != before["scores"][0]      # N and df moved
    rec = dict(ix.recompute_all(QUERY))
    for d, s in zip(after["docs"], after["scores"]):
        assert abs(rec[d] - s) < 1e-12
    with pytest.raises(ValueError):
        ix.insert("d0", "again")
    ix.remove("zz")                                                             # removing an absent document is nothing
    ix2 = R.BM25Index(); ix2.insert("a", "x y z")
    assert ix2.rank("q", 4)["docs"] == ["a"] and ix2.rank("q", 4)["scores"] == [0.0] and ix2.rank("q", 4)["postings_read"] == 0


def test_the_selection_of_k1_b_on_the_first_family_ties_to_the_textbook_pair():
    sel = R.bm25_selection()
    assert sel["n_queries"] == N_FACTS and set(sel["counts"]) == {f"{k1},{b}" for k1 in R.K1_GRID for b in R.B_GRID}
    assert sel["pair"] == {"k1": 1.5, "b": 0.75}
    if sel["all_tie"]:                                                          # expected from the construction table (vi), not asserted
        assert sel["label"].startswith("the textbook pair, confirmed and not selected")
        assert "the tie rule's" in sel["first_family_label"] and "selected" not in sel["first_family_label"]
    else:
        assert "selected" in sel["label"]
    # the rule on a corpus where the counts differ: the lowest k1 then the lowest b among the best when the textbook pair is not among them
    s2 = R.bm25_selection(["a b", "a c", "b c"], ["a", "c"], [0, 1], k=1)
    assert s2["pair"] in ({"k1": 1.5, "b": 0.75}, {"k1": 0.6, "b": 0.4}) and s2["n_queries"] == 2


# ---------------------------------------------------------------------------- #
# I1: the index's arithmetic, bit for bit                                        #
# ---------------------------------------------------------------------------- #
def planted_journal(n: int = 60, seed: int = 3):
    rng = np.random.default_rng(seed)
    j = Journal(policy=POLICY_STOP, index_geometry=ADOPTED_GEOMETRY)
    wp = WritePath(j, seed=0, admission="identity")
    jp = JournalPath(j, geometry=ADOPTED_GEOMETRY)
    base = rng.standard_normal(64).astype(np.float32)
    for i in range(n):
        c = rng.standard_normal(64).astype(np.float32) if i % 4 else base + 0.05 * rng.standard_normal(64).astype(np.float32)
        rep = wp.write(c / np.linalg.norm(c), f"Tone-{i:04x} is a cobbler".encode(), "person", now=float(i), source=f"text {i}")
        assert rep.admitted
        jp.on_write(rep.entry)
    return j, jp, rng


def test_exhaustive_cosine_is_bit_equal_to_the_index_query_on_every_live_entry():
    j, jp, rng = planted_journal()
    ex = R.ExhaustiveCosine(j, jp)
    for _ in range(50):
        q = rng.standard_normal(64).astype(np.float32)
        full = ex.rank(q, k=len(j._entries))
        want = {eid: float(q @ v / ((np.linalg.norm(q) + 1e-8) * (np.linalg.norm(v) + 1e-8))) for eid, v in jp.index.vecs.items()}
        for eid, s in zip(full["docs"], full["scores"]):
            assert s == want[eid], "not the index's arithmetic bit for bit"                 # CueIndex.query's expression, term by term
        assert full["compared"] == len(j._entries) and ex.touched > 0
        top = jp.index.query(q, k=4)                                                       # on the candidates the buckets propose
        for eid, s in top:
            assert s == want[eid]
    ex.hold(True)
    q = rng.standard_normal(64).astype(np.float32)
    a = ex.rank(q, k=4); ex.hold(False); b = ex.rank(q, k=4)
    assert a == b


def test_exhaustive_cosine_ties_by_write_order_and_drops_the_evicted():
    j, jp, rng = planted_journal(n=8)
    ids = list(j._entries)
    e0 = j._entries[ids[0]]
    rep = WritePath(j, seed=0, admission="identity").write(np.asarray(e0.cue, dtype=np.float32), b"another text", "person", now=99.0, source="the same cue, another text")
    assert rep.admitted
    jp.on_write(rep.entry)
    ex = R.ExhaustiveCosine(j, jp)
    r = ex.rank(np.asarray(e0.cue, dtype=np.float32), k=2)
    assert r["docs"] == [ids[0], rep.entry.entry_id] and r["scores"][0] == r["scores"][1]    # bit equal, the earlier written first
    j.transition(ids[0], "evicted")
    r = ex.rank(np.asarray(e0.cue, dtype=np.float32), k=2)
    assert r["docs"][0] == rep.entry.entry_id and ids[0] not in r["docs"]


# ---------------------------------------------------------------------------- #
# V1                                                                              #
# ---------------------------------------------------------------------------- #
def test_v1_the_external_verifier():
    q = "what is Tone-4010's trade?"
    assert R.entity_of_question(q) == "Tone-4010" and R.entity_of_question("Tone-4010 or Kilu-77ab") is None and R.entity_of_question("nobody") is None
    line = b"Tone-4010's trade is cobbler"
    assert R.v1_outcome(q, line, "cobbler", "cobbler")["outcome"] == "strict"
    assert R.v1_outcome(q, line, "cobbler", "miller")["outcome"] == "valid"
    assert R.v1_outcome(q, line, "miller", "miller")["outcome"] == "abstain"                   # the value is not on the line: refused
    assert R.v1_outcome(q, b"Kilu-77ab's trade is cobbler", "cobbler", "cobbler")["outcome"] == "abstain"   # the entity is not on the line
    assert R.v1_outcome(q, None, "cobbler", "cobbler")["reason"] == "no citation"
    assert R.v1_outcome("Tone-4010 or Kilu-77ab", line, "cobbler", "cobbler")["reason"] == "the question carries no single name"
    assert R.v1_outcome(q, line, None, "cobbler")["reason"] == "no value answered"
    # with the organ's value: read from the line by the first family's templates; a line no template reads is an abstention
    first = generate_facts(5, 0)[0][0]
    qq = first.query; stmt = first.statement.encode("utf-8")
    v = R.v1_outcome(qq, stmt, "whatever", first.attr, organ_value=True)
    assert v["outcome"] == "strict" and v["answered"] == first.attr
    assert R.v1_outcome(q, b"Tone-4010 something no template reads cobbler", "cobbler", "cobbler", organ_value=True)["reason"] == "the organ reads no value on the cited line"


# ---------------------------------------------------------------------------- #
# the reference bridge: the organ's own retrieval reproduces the frozen bridge byte for byte (B.1 (a))
# ---------------------------------------------------------------------------- #
def tiny():
    import torch, train
    cfg = train.Config(n_layer=3, n_head=2, n_embd=32, block_size=96, journal="read", journal_block=1, journal_heads=2,
                       journal_read_bytes=96, journal_cue_dim=64, journal_k=3)
    torch.manual_seed(9)
    m = train.VanillaGPT(cfg); m.eval()
    return cfg, m


def test_reference_bridge_with_the_organ_retrieval_reproduces_the_frozen_windows_byte_for_byte():
    from cortex_c2b.lm_bridge import JournalBridge
    cfg, model = tiny()
    facts, _ = generate_facts(24, 0)
    j = Journal(policy=POLICY_STOP, index_geometry=ADOPTED_GEOMETRY)
    frozen = JournalBridge(model, j, k=4, budget_bytes=96, seed=0, shuffle_seed=1, index_geometry=ADOPTED_GEOMETRY, admission="identity")
    for i, f in enumerate(facts):
        assert frozen.write(f.statement, f.schema, now=float(i), entity=f.entity).admitted
    with pytest.raises(ValueError):
        R.ReferenceBridge(model, j, "I0", k=4, budget_bytes=96, seed=0, shuffle_seed=1, index_geometry=ADOPTED_GEOMETRY, admission="identity")
    ref = R.ReferenceBridge(model, j, "I1", k=4, budget_bytes=96, seed=0, shuffle_seed=1, index_geometry=ADOPTED_GEOMETRY, admission="identity")
    ref.jp.reference = R.OrganRetrievalReference(ref.organ_path)               # the organ's own retrieval behind the reference bridge
    for i, f in enumerate(facts[:12]):
        cue = frozen.cues([list(f.query.encode("utf-8"))])[0]
        frozen.rng = np.random.default_rng([1, i]); ref.rng = np.random.default_rng([1, i])
        a = frozen.read(cue); b = ref.read(cue)
        assert a.pointers == b.pointers and a.window == b.window and a.labels == b.labels      # the windows byte for byte
        # the scores: the frozen bridge's come from the index's `query` (float32 arithmetic), the reference bridge's from the frozen
        # `marks_for` (`cue_similarity`, float64 on the same terms) as the amendment B.1 (a) says: two frozen computations that
        # differ at the seventh decimal, under the record's tolerance (P_TOL 1e-6) and far under the tie band (1e-4); a found fact
        assert len(a.scores) == len(b.scores) and all(abs(x - y) < 1e-6 for x, y in zip(a.scores, b.scores))
        assert b.scores == frozen.marks_for(cue, a.pointers)
        assert ref.last_reference()["index"] == "I1" and ref.jp.index.touched >= 0
        assert ref.jp.index.candidates(cue) == sorted(eid for eid, e in j._entries.items() if e.state != "evicted")
        assert ref.jp.index.n_tables == frozen.jp.index.n_tables and ref.jp.index._sig(cue, 0) == frozen.jp.index._sig(cue, 0)
    r2 = R.ReferenceBridge(model, j, "I2", k=4, budget_bytes=96, seed=0, shuffle_seed=1, index_geometry=ADOPTED_GEOMETRY, admission="identity", bm25_parameters=(1.5, 0.75))
    f = facts[0]
    r2.set_query(f.query); r2.rng = np.random.default_rng([1, 0])
    out = r2.read(frozen.cues([list(f.query.encode("utf-8"))])[0])
    own = content_hash(f.statement.encode("utf-8"))
    assert own in out.pointers and r2.last_reference()["index"] == "I2" and r2.last_reference()["operations"] > 0
    with pytest.raises(ValueError):
        r2.set_query(None); r2.read(frozen.cues([list(f.query.encode("utf-8"))])[0])
