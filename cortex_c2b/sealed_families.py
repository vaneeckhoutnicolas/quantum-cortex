"""cortex_c2b.sealed_families -- the sealed families of ADR-010 (part A) and the draws of its scale probe (part C),
generated from constants and sealed by content hash before any encoder runs on them (`metrics/mqar/SEALED-FAMILIES.json`,
committed in the mechanism's build zip of 2026-10-15, before row 49 is read cold; the precision of 2026-10-14, clause (ii)).

Four families of the protocol's generator (`hm_protocol`), the name alphabet and the schemas of the first family:

  T0   the in template control: the first family's templates and vocabularies with new names (facts 70 000; 50 never planted at 70 001)
  A    the unseen formulations: new templates in four structural cells and new vocabularies (facts 30 000; never planted 30 001, fifty,
       and 30 003, one hundred and fifty), built to the rules T1 to T5 of ADR-010 A.1 and pinned by tests on the constants
  Av   the value cell: the first family's templates with A's vocabularies (facts 35 000; 50 never planted at 35 001)
  dev  the development family (the former B): one statement, one query and one paraphrase template per schema, its own vocabularies
       (facts 40 000; 50 never planted at 40 001); public, open to whoever designs a change, never read as a result

Decontamination (A.4): a name is rejected for an exact match and for a one character substitution against the first family and
its negatives (seeds 0 and 10 000), every pool and negative set of the three v9 runs (seeds 100 000 + 2r and 100 001 + 2r for r up
to 150), the density control (seed 20 000, the first 400 facts), the near duplicate arms' neighbours (E2's names), row 40's domain
B (seed 1394548301), the sentinels (seeds 2 and 10 002), and the families generated before it (in the order T0, A, Av, dev); a name
whose hexadecimal part reads as a year value of any family is rejected too (T4: no value is a substring of a name). A rejected
name, and a name already drawn in the same family, is replaced by the generator's next draw; the replacements are counted in the
seal. The texts regenerate from this public code; the seal fixes their content and dates it, it hides nothing.

The scale draws (C.1): the generator's facts at seeds 50 000 to 50 004 (ten thousand each) and 60 000 (one hundred thousand),
written in a seeded order (a permutation of the facts by `default_rng(seed + 7)`, so that the schema blocks of the generator are
mixed and every phase of the cascade sees every schema); near duplicates of kinds E1 and E2 planned at one in twenty facts each
(the fact at position i of the written sequence gets an E1 partner after it when i mod 20 == 0 and an E2 partner when
i mod 20 == 10; eleven thousand entries per ten thousand facts, a hundred and ten thousand per hundred thousand); the never
planted controls (one thousand at 50 100 to 50 104 and 60 100, each with a schema in the generator's balanced order and the
protocol's direct query, exact matches with the journal removed, the distance to the nearest planted name recorded); the read
sample of two thousand questions (indices drawn without replacement by `default_rng(seed + 9)`); from the names alone: the
repeated names and the pairs at Hamming distance one, among the facts (the partners excluded).

This module runs no encoder and writes no journal; the runner of ADR-010 (`cortex_c2b/sealed_family.py`, a later new module) reads
these families on the frozen system. Nothing here is a number of the record.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cortex_c2b.hm_protocol import _TEMPLATES, _ATTRS, _entity_name, SCHEMAS, N_FACTS, N_NEGCTRL, generate_facts, Fact
from cortex_c2b.near_duplicate_probe import E4_TEMPLATES, bump_name, E0D_SEED, E1_PLAN_OFFSET

SEAL_PATH = Path("metrics/mqar/SEALED-FAMILIES.json")
SEAL_DATE = "2026-10-15"
LINE_BUDGET_BYTES = 55           # the window's bytes per line at k = 4 and a budget of 256 with the mark (`build_read_window`)
FUNCTION_WORDS = frozenset("a an the of in on at by from to as is was are do does did what which who whom when where how".split())
FAMILY_ORDER = ("T0", "A", "Av", "dev")
SCHEMA_NEXT = {s: SCHEMAS[(i + 1) % len(SCHEMAS)] for i, s in enumerate(SCHEMAS)}

# ---- the constants (ADR-010 A.1, A.2) ---------------------------------------------------------------------------------------
FIRST_FAMILY_TEMPLATES = {s: {"S": _TEMPLATES[s][0], "Q": _TEMPLATES[s][1], "P": _TEMPLATES[s][2]} for s in SCHEMAS}
FIRST_FAMILY_VALUES = {s: list(_ATTRS[s]) for s in SCHEMAS}

A_TEMPLATES = {
    "person":   {"S1": "{e} spends the week as {a}", "S2": "{e}, {a} by trade, lives nearby", "S3": "the trade of {e}: {a}",
                 "S4": "by trade, {a} is what {e} does", "Q1": "{e} earns a living how?", "Q2": "how does {e} earn a living?",
                 "P1": "{e}: trade?", "P2": "trade of {e}?"},
    "place":    {"S1": "{e} stands within the district of {a}", "S2": "{e}, a settlement of {a}, on the river", "S3": "the district holding {e} is {a}",
                 "S4": "within {a} stands the settlement {e}", "Q1": "{e} belongs to which district?", "Q2": "which district holds {e}?",
                 "P1": "{e}: district?", "P2": "district of {e}?"},
    "event":    {"S1": "{e} dates back to {a}", "S2": "{e}, dated {a}, marked the calendar", "S3": "the calendar entry for {e} reads {a}",
                 "S4": "dated {a}, the calendar entry {e} was kept", "Q1": "{e} dates back to when?", "Q2": "to which date does {e} go back?",
                 "P1": "{e}: dated?", "P2": "dating of {e}?"},
    "object":   {"S1": "{e} was carved from {a}", "S2": "{e}, carved from {a}, sits on the shelf", "S3": "the raw stuff of {e} is {a}",
                 "S4": "from {a} the carvers shaped {e}", "Q1": "{e} was carved from what?", "Q2": "from what was {e} carved?",
                 "P1": "{e}: carved from?", "P2": "stuff of {e}?"},
    "decision": {"S1": "{e} fell to the judgment of {a}", "S2": "{e}, settled through {a}, was closed", "S3": "the matter of {e} rested with {a}",
                 "S4": "settled through {a}, {e} was closed", "Q1": "{e} rested with whom?", "Q2": "with whom did the matter of {e} rest?",
                 "P1": "{e}: settled through?", "P2": "judgment on {e}?"},
}
A_VALUES = {
    "person":   ["weaver", "chandler", "farrier", "scribe", "mason", "fletcher", "dyer", "potter"],
    "place":    ["Ardwick", "Holmby", "Carrow", "Tindale", "Fenwick", "Ravensley", "Oakham", "Bexley"],
    "event":    ["1821", "1847", "1866", "1889", "1912", "1927", "1949", "1975"],
    "object":   ["teak", "bronze", "marble", "ivory", "slate", "walnut", "copper", "jade"],
    "decision": ["the town clerk", "the chapter", "the magistrate", "the mill owners", "the bailiff", "the vestry", "the dean", "the provost"],
}
DEV_TEMPLATES = {
    "person":   {"S": "by trade, {e} is a {a}", "Q": "which trade does {e} practise?", "P": "{e}, what trade?"},
    "place":    {"S": "the district of {a} contains {e}", "Q": "{e} sits in which district?", "P": "district of {e}?"},
    "event":    {"S": "the year {a} saw {e}", "Q": "in which year did {e} occur?", "P": "year of {e}?"},
    "object":   {"S": "the stuff of {e} is {a}", "Q": "of what stuff is {e}?", "P": "{e}, which stuff?"},
    "decision": {"S": "the ruling on {e} came from {a}", "Q": "from whom came the ruling on {e}?", "P": "ruling on {e}, by whom?"},
}
DEV_VALUES = {
    "person":   ["cobbler", "brewer", "sawyer", "glover", "thatcher", "tinker", "ropemaker", "miller"],
    "place":    ["Wetherby", "Lanthorn", "Dunmore", "Kelsey", "Stanwix", "Frome", "Alderley", "Rushmere"],
    "event":    ["1803", "1838", "1859", "1877", "1901", "1936", "1962", "1988"],
    "object":   ["cedar", "granite", "brass", "bone", "flint", "elm", "tin", "onyx"],
    "decision": ["the reeve", "the aldermen", "the prior", "the mint master", "the jurors", "the sheriff", "the almoner", "the burgesses"],
}
FAMILIES = {
    "T0":  {"templates": FIRST_FAMILY_TEMPLATES, "values": FIRST_FAMILY_VALUES, "facts_seed": 70_000, "never_planted": ((70_001, N_NEGCTRL),),
            "what": "the in template control: the first family's templates and vocabularies with new names"},
    "A":   {"templates": A_TEMPLATES, "values": A_VALUES, "facts_seed": 30_000, "never_planted": ((30_001, N_NEGCTRL), (30_003, 150)),
            "what": "the unseen formulations: new templates in four structural cells, new vocabularies"},
    "Av":  {"templates": FIRST_FAMILY_TEMPLATES, "values": A_VALUES, "facts_seed": 35_000, "never_planted": ((35_001, N_NEGCTRL),),
            "what": "the value cell: the first family's templates with A's vocabularies"},
    "dev": {"templates": DEV_TEMPLATES, "values": DEV_VALUES, "facts_seed": 40_000, "never_planted": ((40_001, N_NEGCTRL),),
            "what": "the development family (the former B): public, open to whoever designs a change, never read as a result"},
}
CROSS_SCHEMA_PER_SCHEMA = 10
SCALE_DRAWS = {"ten_thousand": {"seeds": (50_000, 50_001, 50_002, 50_003, 50_004), "n": 10_000, "never_planted_seeds": (50_100, 50_101, 50_102, 50_103, 50_104)},
               "hundred_thousand": {"seeds": (60_000,), "n": 100_000, "never_planted_seeds": (60_100,)}}
SCALE_NEVER_PLANTED_N = 1000
SCALE_SAMPLE_N = 2000            # the read sample of C.2 (the own line in the window at the system's k on two thousand questions)
SCALE_DUP_EVERY = 20             # one E1 partner per twenty facts (i mod 20 == 0) and one E2 partner (i mod 20 == 10), i the written position
SCALE_ORDER_OFFSET = 7           # the written order: a permutation of the facts by default_rng(seed + 7)
SCALE_SAMPLE_OFFSET = 9          # the read sample: indices into the written order by default_rng(seed + 9), without replacement
V9_POOL_REFRESHES = 150          # the three v9 runs replanted at most 147 times: seeds 100 000 + 2r and 100 001 + 2r for r up to 150
DOMAIN_B_SEED = 1394548301       # row 40's domain B
SENTINEL_SEEDS = ((2, N_FACTS), (10_002, N_NEGCTRL))

ALPHABET = {"consonants_initial": "BDFGKLMNPRSTVZ", "consonants": "bdfgklmnprstvz", "vowels": "aeiou", "hex": "0123456789abcdef",
            "pattern": "[BDFGKLMNPRSTVZ][aeiou][bdfgklmnprstvz][aeiou]-[0-9a-f]{4}", "names": 14 * 5 * 14 * 5 * 16 ** 4}
NAME_RE = re.compile("^" + ALPHABET["pattern"] + "$")


# ---- text helpers ---------------------------------------------------------------------------------------------------------------
def fixed_words(template: str) -> list[list[str]]:
    """The words of a template's fixed text, in runs separated by the placeholders (a bigram never spans a placeholder);
    punctuation stripped, lower cased."""
    runs, cur = [], []
    for tok in template.replace("{e}", " {e} ").replace("{a}", " {a} ").split():
        if tok in ("{e}", "{a}"):
            if cur:
                runs.append(cur); cur = []
            continue
        w = re.sub(r"[^a-z0-9']", "", tok.lower()).replace("'s", "").strip("'")
        if w:
            cur.append(w)
    if cur:
        runs.append(cur)
    return runs


def words_of(template: str) -> set[str]:
    return {w for run in fixed_words(template) for w in run}


def content_words(template: str) -> set[str]:
    return {w for w in words_of(template) if w not in FUNCTION_WORDS}


def bigrams_of(template: str) -> set[tuple[str, str]]:
    return {(run[i], run[i + 1]) for run in fixed_words(template) for i in range(len(run) - 1)}


def skeleton(template: str) -> str:
    """The template with its content words masked, the placeholders kept: the structure alone."""
    out = []
    for tok in template.replace("{e}", " {e} ").replace("{a}", " {a} ").split():
        if tok in ("{e}", "{a}"):
            out.append(tok); continue
        core = re.sub(r"[^a-z0-9']", "", tok.lower())
        punct = re.sub(r"[a-z0-9']", "", tok.lower())
        out.append((core if core in FUNCTION_WORDS else "_") + punct)
    return " ".join(out)


def char_ngrams(template: str, n: int = 4) -> set[str]:
    text = " ".join(" ".join(run) for run in fixed_words(template))
    return {text[i:i + n] for i in range(max(0, len(text) - n + 1))}


def word_before_value(template: str) -> str | None:
    """In a statement with the value last, the word written just before the value (None when the value is not last)."""
    t = template.strip()
    if not t.endswith("{a}"):
        return None
    before = t[: -len("{a}")].strip()
    return before.split()[-1] if before else None


def template_overlap(template: str) -> dict:
    """Against the first family's fifteen templates: the nearest by word Jaccard, the shared words and bigrams, the word
    Jaccard and the character four gram overlap (Jaccard) with it."""
    w = words_of(template); b = bigrams_of(template); g = char_ngrams(template)
    best = None
    for s in SCHEMAS:
        for tid, ft in FIRST_FAMILY_TEMPLATES[s].items():
            fw = words_of(ft)
            jac = len(w & fw) / max(1, len(w | fw))
            cand = {"nearest_first_family_template": f"{s}/{tid}", "shared_words": sorted(w & fw), "shared_bigrams": sorted(" ".join(x) for x in (b & bigrams_of(ft))),
                    "word_jaccard": jac, "char_4gram_jaccard": len(g & char_ngrams(ft)) / max(1, len(g | char_ngrams(ft)))}
            if best is None or cand["word_jaccard"] > best["word_jaccard"]:
                best = cand
    return best


# ---- the decontamination set (A.4) --------------------------------------------------------------------------------------------
def _mask_variants(name: str) -> list[str]:
    return [name[:p] + "*" + name[p + 1:] for p in range(len(name))]


class NameFilter:
    """Exact and one character substitution matches against a protected set of names, in constant time per candidate."""

    def __init__(self, names):
        self.exact: set[str] = set()
        self.masked: set[str] = set()
        self.add_all(names)

    def add_all(self, names) -> None:
        for n in names:
            self.exact.add(n); self.masked.update(_mask_variants(n))

    def rejects(self, name: str) -> str | None:
        if name in self.exact:
            return "exact"
        if any(m in self.masked for m in _mask_variants(name)):
            return "one_character"
        return None


def decontamination_sources() -> dict:
    """The declared sources of protected names (A.4), as (label, seed, n) with how they are drawn; the E2 neighbours and the
    year values are derived from them."""
    sources = [("first family facts", 0, N_FACTS), ("first family never planted", 10_000, N_NEGCTRL),
               ("density control E0d (first 400 facts)", E0D_SEED, 2 * N_FACTS), ("row 40 domain B", DOMAIN_B_SEED, N_FACTS)]
    sources += [("sentinels", s, n) for s, n in SENTINEL_SEEDS]
    sources += [("v9 pool", 100_000 + 2 * r, N_FACTS) for r in range(V9_POOL_REFRESHES + 1)]
    sources += [("v9 negatives", 100_001 + 2 * r, N_NEGCTRL) for r in range(V9_POOL_REFRESHES + 1)]
    return {"sources": sources, "derived": ["E2 neighbours of the first family's 200 facts (the last hexadecimal character plus one)",
                                            "the first family's 200 names moved the other way (the last hexadecimal character minus one), which cover the near name block's fifty"],
            "rule": "a name is rejected for an exact match or a one character substitution against the union; a name whose hexadecimal part reads "
                    "as a year value of any family is rejected; a name already drawn in its family is rejected; a rejected name is replaced by the "
                    "generator's next draw and counted"}


def protected_names() -> set[str]:
    names: set[str] = set()
    for _, seed, n in decontamination_sources()["sources"]:
        names.update(f.entity for f in generate_facts(n, seed)[0])
    first = generate_facts(N_FACTS, 0)[0]
    names.update(bump_name(f.entity) for f in first)
    names.update(f.entity[:-1] + ALPHABET["hex"][(ALPHABET["hex"].index(f.entity[-1]) - 1) % 16] for f in first)
    return names


def year_values() -> set[str]:
    out = set(FIRST_FAMILY_VALUES["event"]) | set(A_VALUES["event"]) | set(DEV_VALUES["event"])
    return {v for v in out if re.fullmatch(r"[0-9a-f]{4}", v)}


# ---- the generation -----------------------------------------------------------------------------------------------------------
def _assign(templates: dict, i: int, never_planted: bool = False) -> tuple[str, str, str]:
    """(statement id, query id, paraphrase id) for the i-th fact of a schema block (T1, T2): a family with four statement
    templates takes S(i mod 4 + 1) and the question templates (floor(i / 4) mod 2) + 1, a never planted entity the question
    templates (i mod 2) + 1; a family with one template per form takes S, Q, P."""
    if "S1" in templates:
        q = (i % 2) + 1 if never_planted else ((i // 4) % 2) + 1
        return f"S{(i % 4) + 1}", f"Q{q}", f"P{q}"
    return "S", "Q", "P"


def generate_family(name: str, protected: NameFilter, years: set[str] | None = None) -> dict:
    """One family from its constants: the facts in generation order, the never planted entities, the cross schema questions;
    the names drawn from the generator's alphabet with the declared rejections; the filter is extended with the family's names
    so that the next family is decontaminated against it."""
    spec = FAMILIES[name]
    years = year_values() if years is None else years
    templates, values = spec["templates"], spec["values"]
    rng = np.random.default_rng(spec["facts_seed"])
    per = N_FACTS // len(SCHEMAS)
    in_family: set[str] = set()
    rejections = Counter()

    def draw(r):
        while True:
            n = _entity_name(r)
            why = protected.rejects(n)
            if why is None and n[-4:] in years:
                why = "year_value"
            if why is None and n in in_family:
                why = "in_family"
            if why is None:
                in_family.add(n)
                return n
            rejections[why] += 1

    facts = []
    for s in SCHEMAS:
        for i in range(per):
            e = draw(rng)
            a = str(rng.choice(values[s]))
            sid, qid, pid = _assign(templates[s], i)
            facts.append({"entity": e, "schema": s, "attribute": a, "statement_template": sid, "statement": templates[s][sid].format(e=e, a=a),
                          "query_template": qid, "query": templates[s][qid].format(e=e), "paraphrase_template": pid, "paraphrase": templates[s][pid].format(e=e)})
    never = []
    for seed, n in spec["never_planted"]:
        r2 = np.random.default_rng(seed)
        per_n = n // len(SCHEMAS)
        for s in SCHEMAS:
            for i in range(per_n):
                e = draw(r2)
                a = str(r2.choice(values[s]))
                sid, qid, pid = _assign(templates[s], i, never_planted=True)
                never.append({"entity": e, "schema": s, "attribute": a, "statement_template": sid, "statement": templates[s][sid].format(e=e, a=a),
                              "query_template": qid, "query": templates[s][qid].format(e=e), "paraphrase_template": pid,
                              "paraphrase": templates[s][pid].format(e=e), "seed": seed})
    cross = []
    for s in SCHEMAS:
        block = [f for f in facts if f["schema"] == s][:CROSS_SCHEMA_PER_SCHEMA]
        nxt = SCHEMA_NEXT[s]
        qid = "Q1" if "Q1" in templates[nxt] else "Q"
        for f in block:
            cross.append({"entity": f["entity"], "schema_planted": s, "schema_asked": nxt, "question_template": f"{nxt}/{qid}",
                          "question": templates[nxt][qid].format(e=f["entity"])})
    planted_names = [f["entity"] for f in facts]
    arr = np.array([list(n) for n in planted_names])
    hamming = {}
    for nv in never:
        d = (arr != np.array(list(nv["entity"]))).sum(axis=1)
        k = int(d.argmin())
        hamming[nv["entity"]] = {"nearest_planted": planted_names[k], "distance": int(d[k])}
    protected.add_all(in_family)
    return {"name": name, "what": spec["what"], "facts": facts, "never_planted": never, "cross_schema": cross,
            "rejections": dict(rejections), "n_facts": len(facts), "n_never_planted": len(never), "n_cross_schema": len(cross),
            "never_planted_hamming": hamming, "seeds": {"facts": spec["facts_seed"], "never_planted": [list(x) for x in spec["never_planted"]]}}


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash_of(obj) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def family_content(fam: dict) -> list:
    """The list the seal hashes, in generation order: every planted fact, every never planted entity, every cross schema question
    (A.3's fields)."""
    return ([[f["entity"], f["schema"], f["attribute"], f["statement_template"], f["statement"], f["query_template"], f["query"], f["paraphrase_template"], f["paraphrase"]]
             for f in fam["facts"]]
            + [[v["entity"], v["schema"], v["attribute"], v["statement"], v["query_template"], v["query"], v["paraphrase_template"], v["paraphrase"]] for v in fam["never_planted"]]
            + [[c["entity"], c["schema_asked"], c["question"]] for c in fam["cross_schema"]])


def constants() -> dict:
    """Everything the families are generated from (the constants hash covers it): the templates with their ids, the vocabularies,
    the alphabet and the name rule, the sizes, the seeds, the assignment rules, the decontamination sources and the redraw rule."""
    return {"families": {k: {"templates": v["templates"], "values": v["values"], "facts_seed": v["facts_seed"], "never_planted": [list(x) for x in v["never_planted"]]}
                         for k, v in FAMILIES.items()},
            "order": list(FAMILY_ORDER), "alphabet": ALPHABET, "n_facts": N_FACTS, "schemas": list(SCHEMAS),
            "assignment": {"statements": "the i-th fact of a schema block takes S(i mod 4 + 1) when the family has four statement templates, else S",
                           "questions": "the i-th fact takes Q and P (floor(i/4) mod 2 + 1); a never planted entity takes (i mod 2 + 1); one template per form otherwise",
                           "cross_schema": "the first ten facts of each schema asked the first direct query template of the next schema in the order person, place, event, object, decision",
                           "values": "one draw of the schema's eight values per fact from the family's own stream after the name"},
            "decontamination": decontamination_sources(), "year_values_rejected_as_hex_part": sorted(year_values()),
            "line_budget_bytes": LINE_BUDGET_BYTES, "function_words": sorted(FUNCTION_WORDS),
            "e4_templates": E4_TEMPLATES, "scale_draws": {k: {"seeds": list(v["seeds"]), "n": v["n"], "never_planted_seeds": list(v["never_planted_seeds"])} for k, v in SCALE_DRAWS.items()},
            "scale_rules": {"never_planted_n": SCALE_NEVER_PLANTED_N, "sample_n": SCALE_SAMPLE_N, "order_offset": SCALE_ORDER_OFFSET, "sample_offset": SCALE_SAMPLE_OFFSET,
                            "order": f"the facts written in the order of a permutation drawn by default_rng(seed + {SCALE_ORDER_OFFSET}); i below is the position in that order",
                            "duplicates": f"one E1 partner after the fact at position i when i mod {SCALE_DUP_EVERY} == 0 and one E2 partner when i mod {SCALE_DUP_EVERY} == {SCALE_DUP_EVERY // 2}; "
                                          f"E1: the other attribute of the schema drawn, one draw per partnered fact in the written order, from default_rng(E1_PLAN_OFFSET + seed) ({E1_PLAN_OFFSET} + seed); "
                                          "E2: the last hexadecimal character plus one, dropped when it coincides with a fact's name or with another partner",
                            "never_planted": f"{SCALE_NEVER_PLANTED_N} names drawn by default_rng(never planted seed), a name equal to a fact's or a partner's or already drawn replaced by the next draw; "
                                             "the i-th control takes the schema SCHEMAS[i mod 5] and the protocol's direct query; the Hamming distance to the nearest planted name recorded",
                            "sample": f"{SCALE_SAMPLE_N} positions of the written order drawn without replacement by default_rng(seed + {SCALE_SAMPLE_OFFSET})",
                            "name_statistics": "among the facts alone, the partners excluded; pairs counted with multiplicity"}}


def generate_all() -> dict:
    """The four families in the declared order, each decontaminated against the external sources and the families before it."""
    filt = NameFilter(protected_names())
    years = year_values()
    return {name: generate_family(name, filt, years) for name in FAMILY_ORDER}


def family_seal(fam: dict) -> dict:
    spec = FAMILIES[fam["name"]]
    templates = {s: {tid: {"template": t, **template_overlap(t), "skeleton": skeleton(t), "bytes_max_statement": None} for tid, t in spec["templates"][s].items()} for s in SCHEMAS}
    for f in fam["facts"]:
        cell = templates[f["schema"]][f["statement_template"]]
        cell["bytes_max_statement"] = max(cell["bytes_max_statement"] or 0, len(f["statement"].encode("utf-8")))
    ham = Counter(v["distance"] for v in fam["never_planted_hamming"].values())
    return {"what": fam["what"], "content_sha256": content_hash_of(family_content(fam)), "n_facts": fam["n_facts"], "n_never_planted": fam["n_never_planted"],
            "n_cross_schema": fam["n_cross_schema"], "seeds": fam["seeds"], "rejections": fam["rejections"], "templates": templates,
            "never_planted_hamming_distance_counts": {str(k): v for k, v in sorted(ham.items())},
            "never_planted_hamming": fam["never_planted_hamming"]}


# ---- the scale draws (C.1) --------------------------------------------------------------------------------------------------------
def name_statistics(names: list[str]) -> dict:
    """From the names alone: the repeated names and the pairs at Hamming distance one (exact duplicates not counted as pairs)."""
    c = Counter(names)
    repeated = sum(1 for v in c.values() if v > 1)
    pairs = 0
    items = list(c.items())
    for p in range(len(names[0])):
        buckets = defaultdict(list)
        for n, k in items:
            buckets[n[:p] + "*" + n[p + 1:]].append(k)
        for lst in buckets.values():
            tot = sum(lst); pairs += tot * (tot - 1) // 2 - sum(k * (k - 1) // 2 for k in lst)
    return {"n": len(names), "distinct": len(c), "repeated_names": repeated, "repeated_pairs": sum(v * (v - 1) // 2 for v in c.values()), "hamming_one_pairs": pairs}


def scale_plan(seed: int, n: int, never_seed: int) -> dict:
    """One scale journal's plan: the facts in their written order, the E1 and E2 partners at the declared rate, the never planted
    control with its schema, its question and its distance, the read sample; everything from the names alone, no encoder."""
    facts, gen_hash = generate_facts(n, seed)
    order = [int(i) for i in np.random.default_rng(seed + SCALE_ORDER_OFFSET).permutation(n)]
    written = [facts[i] for i in order]
    names = [f.entity for f in facts]
    name_set = set(names)
    rng = np.random.default_rng(E1_PLAN_OFFSET + seed)
    partners, seen = [], set()
    for i, f in enumerate(written):
        if i % SCALE_DUP_EVERY == 0:
            others = [a for a in _ATTRS[f.schema] if a != f.attr]
            a = str(rng.choice(others))
            partners.append({"after": i, "kind": "E1", "entity": f.entity, "attribute": a, "statement": _TEMPLATES[f.schema][0].format(e=f.entity, a=a)})
        elif i % SCALE_DUP_EVERY == SCALE_DUP_EVERY // 2:
            nb = bump_name(f.entity)
            if nb in name_set or nb in seen:
                partners.append({"after": i, "kind": "E2", "entity": f.entity, "dropped": True, "neighbour": nb})
            else:
                seen.add(nb)
                partners.append({"after": i, "kind": "E2", "entity": f.entity, "neighbour": nb, "attribute": str(f.attr), "statement": _TEMPLATES[f.schema][0].format(e=nb, a=f.attr)})
    kept = [p for p in partners if not p.get("dropped")]
    # the never planted control: a name equal to a fact's, a partner's or an earlier control's is replaced by the next draw; each
    # control takes a schema in the generator's balanced order and the protocol's direct query; the distance to the nearest planted name
    planted = names + [p["neighbour"] for p in kept if p["kind"] == "E2"]
    planted_set = set(planted)
    r2 = np.random.default_rng(never_seed)
    never, replaced = [], 0
    while len(never) < SCALE_NEVER_PLANTED_N:
        e = _entity_name(r2)
        if e in planted_set or e in {x["entity"] for x in never}:
            replaced += 1; continue
        s = SCHEMAS[len(never) % len(SCHEMAS)]
        never.append({"entity": e, "schema": s, "query": _TEMPLATES[s][1].format(e=e)})
    arr = np.array([list(x) for x in planted])
    dist = Counter()
    for v in never:
        d = (arr != np.array(list(v["entity"]))).sum(axis=1)
        k = int(d.argmin()); v["nearest_planted"] = planted[k]; v["distance"] = int(d[k]); dist[int(d[k])] += 1
    sample = sorted(int(i) for i in np.random.default_rng(seed + SCALE_SAMPLE_OFFSET).choice(n, size=min(SCALE_SAMPLE_N, n), replace=False))
    content = ([[f.entity, f.schema, str(f.attr), f.statement] for f in written]
               + [[p["after"], p["kind"], p.get("neighbour") or p["entity"], p.get("statement"), bool(p.get("dropped"))] for p in partners]
               + [[v["entity"], v["schema"], v["query"]] for v in never]
               + [sample])
    return {"seed": seed, "n_facts": n, "generator_hash": gen_hash, "never_planted_seed": never_seed,
            "order": {"what": f"a permutation of the generator's facts by default_rng(seed + {SCALE_ORDER_OFFSET}); the written sequence",
                      "sha256": content_hash_of(order), "first_ten": order[:10]},
            "names": {**name_statistics(names), "what": "among the facts alone, the partners excluded; pairs counted with multiplicity"},
            "entries_planned": n + len(kept),
            "partners": {"E1": sum(1 for p in kept if p["kind"] == "E1"), "E2": sum(1 for p in kept if p["kind"] == "E2"), "E2_dropped": sum(1 for p in partners if p.get("dropped"))},
            "never_planted": {"n": len(never), "replaced": replaced, "distance_to_nearest_planted_counts": {str(k): v for k, v in sorted(dist.items())},
                              "per_schema": dict(Counter(v["schema"] for v in never))},
            "sample": {"n": len(sample), "sha256": content_hash_of(sample), "first_ten": sample[:10]},
            "content_sha256": content_hash_of(content), "_plan": {"facts": written, "partners": partners, "never": never, "order": order, "sample": sample}}


def scale_seal() -> dict:
    out = {}
    for label, spec in SCALE_DRAWS.items():
        for seed, nseed in zip(spec["seeds"], spec["never_planted_seeds"]):
            p = scale_plan(seed, spec["n"], nseed)
            p.pop("_plan")
            out[str(seed)] = {"size": label, **p}
    return out


def build_seal(with_scale: bool = True) -> dict:
    import platform
    fams = generate_all()
    seal = {"what": "the sealed families of ADR-010 (part A) and the draws of its scale probe (part C), generated from the constants of cortex_c2b/sealed_families.py "
                    "and sealed by content hash before any encoder runs on them; a record test regenerates them and compares; no encoder, no journal",
            "declared": "ADR-010, 2026-10-14 (A.1 to A.4, C.1); the precision of 2026-10-14 on the rule on instruments, clause (ii)",
            "date": SEAL_DATE, "label": "a synthetic set locked before execution, not a blind validation: the templates and the generating rules are public",
            "canonical_json": "json.dumps with sorted keys, separators (',', ':'), ensure_ascii false, UTF-8; sha256 in full",
            # the generator's draws run through numpy's Generator; the versions the seal was written under, so that a regeneration that
            # differs is attributed before anything else (not in the hashes)
            "versions": {"python": platform.python_version(), "numpy": np.__version__},
            "constants_sha256": content_hash_of(constants()),
            "families": {name: family_seal(fams[name]) for name in FAMILY_ORDER},
            "decontamination": {**decontamination_sources(), "protected_names": len(protected_names())},
            "scale_draws": scale_seal() if with_scale else None}
    return seal


def _cli(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="the sealed families and scale draws of ADR-010: generate, seal, check")
    ap.add_argument("--write", action="store_true", help="write metrics/mqar/SEALED-FAMILIES.json (refused if it exists: a file of the record is never overwritten)")
    ap.add_argument("--check", action="store_true", help="regenerate and compare with the seal; exit 1 on a difference")
    ap.add_argument("--print", default=None, metavar="FAMILY", help="print a family's facts (T0 | A | Av | dev)")
    ap.add_argument("--no-scale", action="store_true", help="leave the scale draws out (seconds instead of a minute)")
    args = ap.parse_args(argv)
    if args.print:
        fams = generate_all()
        print(json.dumps({k: v for k, v in fams[args.print].items() if k != "never_planted_hamming"}, indent=1, ensure_ascii=False)); return
    if args.write:
        if SEAL_PATH.exists():
            raise SystemExit(f"{SEAL_PATH} exists; a file of the record is never overwritten")
        seal = build_seal(with_scale=not args.no_scale)
        SEAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        SEAL_PATH.write_text(json.dumps(seal, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"retained: {SEAL_PATH}  sha256 {hashlib.sha256(SEAL_PATH.read_bytes()).hexdigest()[:16]}")
        print(json.dumps({k: v["content_sha256"] for k, v in seal["families"].items()}, indent=1)); return
    if args.check:
        seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
        fresh = build_seal(with_scale=seal.get("scale_draws") is not None)
        diffs = [k for k in seal["families"] if seal["families"][k]["content_sha256"] != fresh["families"][k]["content_sha256"]]
        if seal["constants_sha256"] != fresh["constants_sha256"]:
            diffs.append("constants")
        if seal.get("scale_draws") and any(seal["scale_draws"][k]["content_sha256"] != fresh["scale_draws"][k]["content_sha256"] for k in seal["scale_draws"]):
            diffs.append("scale_draws")
        print("seal: " + ("regenerated exactly" if not diffs else "DIFFERS on " + ", ".join(diffs)))
        raise SystemExit(0 if not diffs else 1)
    ap.error("one of --write / --check / --print")


if __name__ == "__main__":
    _cli()
