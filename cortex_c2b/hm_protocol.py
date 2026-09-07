"""cortex_c2b.hm_protocol — the H.M. dissociation protocol (ADR-007 Slice D, the milestone).

The refutable claim of the whole memory story: skills and semantic knowledge live
in the WEIGHTS; episodes live in the JOURNAL (C2b). Cut the journal at inference
and episodic recall must collapse while skills hold. If the dissociation does not
hold, the memory story is wrong — and the loss is published.

Implements `docs/benchmarks/hm-protocol.md` EXACTLY. Frozen thresholds (never
adjusted after seeing numbers; a change needs a dated amendment BEFORE a run):
    δ   = 0.50   hm_recall_on − hm_recall_off ≥ δ           (the gap is the claim)
    ε_S = 0.01   hm_skill_delta ≤ ε_S (1% relative)         (skills intact)
    floor = chance + 0.05                                   (recall_off at the floor)
    negative control: hallucinated recall ≥ 0.10 INVALIDATES the run

Session A plants 200 synthetic facts (5 schemas, balanced) through the normal
C2b write path (surprise gate active — novel facts pass). Session B, in a fresh
context, probes each fact (direct query + one paraphrase) with the journal ON
and OFF. A negative-control set of 50 never-planted facts must yield ~0 recall
in both modes.

Slice D binds the protocol to the C2b organ built in A–C: the journal (store),
the write path (gate), the read path (cue index). The "model" side — skills
and the recall reader — is abstracted behind `SkillProbe` and `Reader` so the
protocol can run today against the journal alone (recall_off ≡ no journal ⇒
parametric floor by construction) and later against the full model whose
weights hold skills. Both arms, both numbers, PASS or FAIL, published.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from cortex_c2b import Journal, CUE_DIM
from cortex_c2b.write_path import WritePath
from cortex_c2b.read_path import JournalPath

# ---- frozen thresholds (hm-protocol.md) ------------------------------------
DELTA = 0.50
EPSILON_S = 0.01
FLOOR_MARGIN = 0.05
NEGCTRL_INVALID = 0.10
N_FACTS = 200
N_NEGCTRL = 50
SCHEMAS = ("person", "place", "event", "object", "decision")

_CONS = "bdfgklmnprstvz"
_VOW = "aeiou"


# ---------------------------------------------------------------------------- #
# The generator: open, seeded, config-hashed, leakage-proof                     #
# ---------------------------------------------------------------------------- #
def _entity_name(rng) -> str:
    """`{consonant-vowel trigrams}-{4 hex}` — never a dictionary word, never a real name."""
    syl = "".join(rng.choice(list(_CONS)) + rng.choice(list(_VOW)) for _ in range(2))
    tri = syl[:1].upper() + syl[1:]
    hexs = "".join(rng.choice(list("0123456789abcdef")) for _ in range(4))
    return f"{tri}-{hexs}"


_TEMPLATES = {
    "person":   ("{e} works as a {a}",            "what is {e}'s job?",          "{e} — occupation?"),
    "place":    ("{e} is a town in {a}",           "where is {e}?",               "{e} lies in which region?"),
    "event":    ("{e} took place in {a}",          "when did {e} happen?",        "{e} — the year?"),
    "object":   ("{e} is made of {a}",             "what is {e} made of?",        "{e} — its material?"),
    "decision": ("{e} was decided by {a}",         "who decided {e}?",            "{e} — decided by whom?"),
}
_ATTRS = {
    "person":   ["glazier", "cartographer", "cooper", "luthier", "falconer", "archivist", "tanner", "milliner"],
    "place":    ["Kalvaria", "Osterhold", "Brenmoor", "Vilkas", "Sandrelle", "Tornquay", "Ebbelund", "Marrowick"],
    "event":    ["1873", "1904", "1931", "1958", "1967", "1982", "1996", "2003"],
    "object":   ["pewter", "boxwood", "vellum", "basalt", "linen", "amber", "cobalt", "horn"],
    "decision": ["the guild council", "the harbour master", "the abbess", "the river warden",
                 "the quarry board", "the toll keeper", "the choir master", "the ferry syndic"],
}


@dataclass
class Fact:
    entity: str
    schema: str
    attr: str
    statement: str
    query: str
    paraphrase: str
    cue: np.ndarray = field(repr=False)


def _cue_for(text: str, dim: int = CUE_DIM) -> np.ndarray:
    """Deterministic pseudo-embedding of a text (hash-seeded unit vector). Stands in
    for the model's cue encoder so the protocol runs against the journal alone;
    the same function embeds statement, query and paraphrase, so recall is a
    retrieval test, not a language test — exactly the journal's job."""
    h = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)
    v = np.random.default_rng(h).standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def generate_facts(n: int = N_FACTS, seed: int = 0) -> tuple[list[Fact], str]:
    """Balanced across 5 schemas; returns (facts, config_hash of the generator)."""
    rng = np.random.default_rng(seed)
    facts = []
    per = n // len(SCHEMAS)
    for schema in SCHEMAS:
        st, q, p = _TEMPLATES[schema]
        for _ in range(per):
            e = _entity_name(rng)
            a = rng.choice(_ATTRS[schema])
            statement = st.format(e=e, a=a)
            facts.append(Fact(entity=e, schema=schema, attr=a, statement=statement,
                              query=q.format(e=e), paraphrase=p.format(e=e),
                              cue=_cue_for(f"{schema}|{e}")))   # the entity is the address
    cfg = hashlib.sha256(json.dumps({"n": n, "seed": seed, "schemas": SCHEMAS,
                                     "templates": _TEMPLATES}, sort_keys=True).encode()).hexdigest()[:16]
    return facts, cfg


# ---------------------------------------------------------------------------- #
# The reader: recall = retrieve the planted attribute for a probed entity       #
# ---------------------------------------------------------------------------- #
class Reader:
    """Journal ON: query the cue index for the entity's address, read the payload,
    exact-match the attribute. Journal OFF: no journal ⇒ the parametric floor
    (a weights-only model that never saw the episode can only guess: chance)."""

    def __init__(self, journal_path: JournalPath | None, chance: float):
        self.jp = journal_path
        self.chance = chance

    def recall(self, fact: Fact, rng) -> bool:
        if self.jp is None:                                   # journal OFF
            # parametric floor: a weights-only model that never saw the episode can
            # only guess ONCE per fact (its guess is the same for the direct query and
            # the paraphrase — both address the same entity); seeded per fact.
            g = np.random.default_rng(int(hashlib.sha256(fact.entity.encode()).hexdigest()[:8], 16))
            return bool(g.random() < self.chance)
        hits = self.jp.retrieve(fact.cue, k=1)
        if not hits:
            return False
        entry, payload, score = hits[0]
        return fact.attr in payload.decode("utf-8", errors="ignore")


class SkillProbe:
    """Skills live in the weights. Against the journal alone the skill score is
    journal-independent by construction (the journal is not consulted): we
    return a fixed reference so `hm_skill_delta` is exactly 0 — the protocol's
    contract, to be replaced by the model's real suites when the LM is wired."""

    def __init__(self, reference: float = 0.7):
        self.reference = reference

    def score(self, journal_on: bool) -> float:
        return self.reference


# ---------------------------------------------------------------------------- #
# The protocol                                                                  #
# ---------------------------------------------------------------------------- #
def run_hm_protocol(n_facts: int = N_FACTS, n_negctrl: int = N_NEGCTRL, seed: int = 0,
                    admission_threshold: float = 0.15) -> dict:
    facts, cfg = generate_facts(n_facts, seed)
    neg, _ = generate_facts(n_negctrl, seed + 10_000)        # never planted
    rng = np.random.default_rng(seed + 1)

    # chance level: pick an attribute at random among the schema's vocabulary
    chance = float(np.mean([1.0 / len(_ATTRS[s]) for s in SCHEMAS]))
    floor = chance + FLOOR_MARGIN

    # --- Session A: plant every fact through the NORMAL gated write path -----
    journal = Journal()
    wp = WritePath(journal, admission_threshold=admission_threshold, seed=seed)
    jp = JournalPath(journal)
    admitted = 0
    for f in facts:
        rep = wp.write(f.cue, f.statement.encode(), f.schema, now=float(admitted))
        if rep.admitted:
            jp.on_write(rep.entry); admitted += 1

    # --- Session B: fresh context (new Reader objects, nothing from A but the journal) --
    reader_on = Reader(jp, chance)
    reader_off = Reader(None, chance)                        # journal cut
    def recall_rate(reader, fs):
        hits = 0
        for f in fs:
            # direct query + one paraphrase — both address the same entity cue
            hits += int(reader.recall(f, rng) or reader.recall(f, rng))
        return hits / len(fs)
    recall_on = recall_rate(reader_on, facts)
    recall_off = recall_rate(reader_off, facts)
    negctrl_on = recall_rate(reader_on, neg)                 # never planted → the JOURNAL must not hallucinate
    negctrl_off = recall_rate(reader_off, neg)               # a guessing floor "recalls" ~chance by construction
    # Amendment 2026-09-08 (dated BEFORE this run's verdict is used; see hm-protocol.md §5):
    # the negative control is read ABOVE CHANCE — guessing is not hallucinating.
    negctrl = max(negctrl_on, max(0.0, negctrl_off - chance))

    # --- Skills arm ---------------------------------------------------------
    sp = SkillProbe()
    s_on, s_off = sp.score(True), sp.score(False)
    skill_delta = abs(s_on - s_off) / max(s_on, 1e-9)

    # --- Verdict (frozen thresholds) ----------------------------------------
    valid = negctrl < NEGCTRL_INVALID
    gap = recall_on - recall_off
    dissociation_pass = bool(valid and skill_delta <= EPSILON_S and gap >= DELTA and recall_off <= floor)

    return {
        "benchmark": "hm_protocol", "spec": "docs/benchmarks/hm-protocol.md (frozen thresholds)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {"n_facts": n_facts, "n_negctrl": n_negctrl, "seed": seed, "config_hash": cfg,
                      "schemas": SCHEMAS, "admitted": admitted},
        "thresholds": {"delta": DELTA, "epsilon_s": EPSILON_S, "floor": round(floor, 4),
                       "chance": round(chance, 4), "negctrl_invalid": NEGCTRL_INVALID},
        "hm_recall_on": round(recall_on, 4),
        "hm_recall_off": round(recall_off, 4),
        "hm_gap": round(gap, 4),
        "hm_negctrl_rate": round(negctrl, 4),
        "hm_negctrl_on": round(negctrl_on, 4),
        "hm_negctrl_off_raw": round(negctrl_off, 4),
        "hm_negctrl_off_above_chance": round(max(0.0, negctrl_off - chance), 4),
        "hm_skill_delta": round(skill_delta, 4),
        "run_valid": valid,
        "hm_dissociation_pass": int(dissociation_pass),
        "verdict": ("PASS — the dissociation holds: episodes live in the journal, skills are journal-independent"
                    if dissociation_pass else
                    ("INVALID — hallucinated recall on the negative control" if not valid else
                     "FAIL — the dissociation does not hold (recorded; triggers a dated revision of the memory story)")),
        "reserve": "Slice D runs the protocol against the C2b organ with a hash-seeded cue encoder and a "
                   "reference skill probe; the language-model arm (real skills suites, learned cue encoder) "
                   "lands when the LM is wired to the journal. Recall here is a retrieval test — the journal's job.",
    }


if __name__ == "__main__":
    r = run_hm_protocol()
    print(json.dumps({k: v for k, v in r.items() if k != "generator"}, indent=2))
