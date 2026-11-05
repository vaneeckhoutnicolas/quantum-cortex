"""cortex_c2b.provenance_bench -- the provenance benchmark: a bench built from the record's sealed generator, an answers
file scored with the record's reading of rows 48 to 50 and the record's frozen conditions, without the model and without
the organ (the roadmap's decision of 2026-10-14, point (a); declared 2026-11-05 in
docs/working/PROVENANCE-BENCH-DECLARATION-2026-11-05.md before this code, amended the same day after its committee; a new
module on the frozen system).

  build   python -m cortex_c2b.provenance_bench build --seed <s> --n <n> [--e4] [--families] [--split] --out <bench file>
  score   python -m cortex_c2b.provenance_bench score --bench <bench file> [--key <key file>] --answers <answers file> --out <table>

The bench: statements in the order they are to be written (each with the organ's content hash of its text as pointer), the
questions by block, and the key (per question the entity, the schema, the value, the designated line, the supporting lines,
the partner's line on a contested block). The answers file: {"system": <name>, "answers": {<question id>: {"value": <string
or null>, "cited": <pointer or null>}}}; a question absent or with a null value is an abstention. The table: per block and
pooled over the planted blocks, the provenance metrics of ADR-007 section 7 recomputed here (the record's two readings of
the citation support, section 4 (b)'s on the expected value as `near_duplicate_reading.citation_support_of` reads it and
section 7's on the answered value beside it, as the record's files carry both), with every rate as a numerator over a
denominator; the frozen conditions as `hm_lm` scores them (the strict recall: a valid citation, the cited line carrying the
answered value, and the exact value, over the planted questions; invalid citations over the planted questions; claims on the
never planted names), the attributed accuracy gap beside them as `sealed_reading.conditions` keeps the own line gap beside
the frozen one; the bench's and the answers file's hashes. The bench holds a system to the record's contract and claims
nothing of the organ: a user's numbers are the user's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from cortex_c2b import content_hash
from cortex_c2b import sealed_families as S
from cortex_c2b.hm_protocol import DELTA, NEGCTRL_INVALID, SCHEMAS, Fact, address_cue
from cortex_c2b.near_duplicate_probe import E4_TEMPLATES, near_name_plan

INVALID_CITATION_MAX = 0.01             # equal to `cortex_c2b.hm_lm.INVALID_CITATION_MAX` (declared 2026-09-12); copied so that the bench imports nothing of the model; a test pins the equality
PLANTED_BLOCKS = ("planted", "contested_E1", "contested_E2", "contested_E4", "family", "family_paraphrase")
CLAIM_BLOCKS = ("never_planted", "near_name", "cross_schema")
BLOCK_WHAT = {
    "planted": "the direct question of a fact of the scale draw that has no partner",
    "contested_E1": "the direct question of a fact followed by an E1 partner (the same entity, another value; or a later fact of the draw on the same entity and schema, an E1 by chance, ADR-010 C.1 (b))",
    "contested_E2": "the direct question of a fact followed by an E2 partner (a name one byte away, the same value)",
    "contested_E4": "the direct question of a fact followed by its E4 paraphrase (the same fact under another template)",
    "never_planted": "a name the bench never wrote (the scale draw's control of C.2, one thousand; a sealed family's never planted names, fifty for T0 and Av, two hundred for A)",
    "near_name": "a sealed family's never planted name one hexadecimal character from a planted one",
    "cross_schema": "a sealed family's planted entity asked under another schema (no statement under that schema)",
    "family": "a sealed family's direct question",
    "family_paraphrase": "a sealed family's paraphrase question",
}
CONDITIONS_WHAT = ("the frozen conditions (declared 2026-09-12, not moved) as the record's protocol scores them (`cortex_c2b.hm_lm`): the strict recall, a valid citation "
                   "(the cited line held and carrying the answered value) with the exact value, over the planted questions, at or above DELTA; invalid citations (a value given with "
                   "no pointer, a pointer the bench does not hold, or a held line not carrying the answered value), over the planted questions, at or under INVALID_CITATION_MAX; "
                   "claims on the never planted names under NEGCTRL_INVALID; the off arm at the floor by construction is not the bench's to measure (it has no off arm) and is "
                   "written as not measured; `holds` is the conjunction of the three measured. Beside them, as `sealed_reading.conditions` keeps the own line gap beside the frozen "
                   "gap: the attributed accuracy gap (a correct value with a supporting citation, over the planted questions, at or above DELTA) and `holds_with_attributed_accuracy_gap`")
SUPPORT_WHAT = ("citation support in the record's two readings, both over the answered questions: `citation_supported`, section 4 (b)'s as `near_duplicate_reading.citation_support_of` "
                "reads it, the cited line is about the question's entity and carries the key's value; `citation_supported_answered`, section 7's wording, the cited line is about the "
                "entity and carries the value the system answered (a line carrying a wrong value the system answered supports that answer under this reading, and `value_correct` says "
                "the value is wrong); `attributed_accuracy` is a correct value with a citation supported under 4 (b), as the record's `attributed_accuracy`; the record's own files "
                "count the support on every citation, an abstention's surviving citation included, which the bench does not see")
ABSTENTION_WHAT = ("a false abstention: abstained where a line carrying the key's value exists among the bench's statements; verified: where none exists "
                   "(the never planted, near name and cross schema questions); the record's own files split the abstentions by the window the organ "
                   "returned, which the bench does not see")
CONTESTED_WHAT = ("a contested block is by construction (every question whose fact has a partner), where the record's contested reads are by the window "
                  "(both lines in the window at or above the threshold), which the bench does not see; the exact interval is ADR-010's (`sealed_reading.clopper_pearson`), "
                  "row 50's intervals being Wilson's")
MODULE_PATH = Path(__file__)
POINTER_RE = re.compile(r"^[0-9a-f]{16}$")


def module_hash() -> str:
    return hashlib.sha256(MODULE_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float] | None:
    """The exact two sided interval (Clopper and Pearson 1934), the construction of `sealed_reading.clopper_pearson` (sixty
    halvings on the binomial tails, lgamma coefficients, fsum), here so that the bench imports nothing of the model; a test
    pins the equality with the record's function on a grid."""
    if n <= 0:
        return None
    k = int(k); n = int(n)
    logc = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(n + 1)]

    def cdf_at_most(x, p):
        if p <= 0:
            return 1.0
        if p >= 1:
            return 0.0 if x < n else 1.0
        lp, lq = math.log(p), math.log1p(-p)
        return min(1.0, math.fsum(math.exp(logc[i] + i * lp + (n - i) * lq) for i in range(x + 1)))

    def solve(g, target, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if (g(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    lower = 0.0 if k == 0 else solve(lambda p: 1.0 - cdf_at_most(k - 1, p), alpha / 2, True)
    upper = 1.0 if k == n else solve(lambda p: cdf_at_most(k, p), alpha / 2, False)
    return [lower, upper]


def _r(k: int, d: int) -> dict:
    return {"n": int(k), "d": int(d), "rate": (k / d if d else None)}


# ---------------------------------------------------------------------------- #
# the build                                                                       #
# ---------------------------------------------------------------------------- #
def _facts_of(fam: dict) -> list[Fact]:
    """A sealed family's facts as `Fact` objects, as `sealed_family.facts_of` builds them (that module imports the model's;
    this one may not); a test pins the equality."""
    return [Fact(entity=f["entity"], schema=f["schema"], attr=f["attribute"], statement=f["statement"], query=f["query"], paraphrase=f["paraphrase"],
                 cue=address_cue(f"{f['schema']}|{f['entity']}"), address_text=f"{f['schema']}|{f['entity']}") for f in fam["facts"]]


def _near_name_protected(fams: dict) -> set[str]:
    prot = set(S.protected_names())
    for fam in fams.values():
        prot |= {f["entity"] for f in fam["facts"]} | {v["entity"] for v in fam["never_planted"]}
    return prot


def _near_name_block(fam: dict, fams: dict, protected: set[str] | None = None) -> list[dict]:
    """The near name block of a sealed family as `sealed_family.near_name_block` builds it (the frozen `near_name_plan`, the
    question text A's Q1 and Q2 alternately on A, the direct query elsewhere); a test pins the equality. The protected set
    is computed once per build and passed in (the same set for the three families)."""
    plan = near_name_plan(_facts_of(fam), _near_name_protected(fams) if protected is None else protected)
    out, per = [], {s: 0 for s in SCHEMAS}
    for q in plan["questions"]:
        j = per[q["schema"]]; per[q["schema"]] += 1
        if fam["name"] == "A":
            tid = "Q1" if j % 2 == 0 else "Q2"
            text = S.A_TEMPLATES[q["schema"]][tid].format(e=q["entity"])
        else:
            text = q["query"]
        out.append({**q, "query_asked": text, "dropped": plan.get("dropped")})
    return out


class _Bench:
    def __init__(self):
        self.statements, self.questions, self.key = [], [], {}
        self.pointers = set()

    def add_statement(self, text: str, kind: str, schema: str, entity: str, value: str, partner_of: str | None = None, family: str | None = None) -> str:
        p = content_hash(text.encode("utf-8"))
        if p in self.pointers:
            raise ValueError(f"two statements with one text: {text!r}")
        self.pointers.add(p)
        s = {"id": f"s{len(self.statements):06d}", "pointer": p, "text": text, "kind": kind, "schema": schema, "entity": entity, "value": value}
        if partner_of is not None:
            s["partner_of"] = partner_of
        if family is not None:
            s["family"] = family
        self.statements.append(s)
        return p

    def add_question(self, text: str, block: str, entity: str, schema: str, value: str | None, designated: str | None, supporting: list[str],
                     partner: str | None = None, family: str | None = None) -> str:
        qid = f"q{len(self.questions):06d}"
        q = {"id": qid, "text": text, "block": block}
        if family is not None:
            q["family"] = family
        self.questions.append(q)
        self.key[qid] = {"block": block, "entity": entity, "schema": schema, "value": value, "designated": designated, "supporting": list(supporting), "partner": partner}
        if family is not None:
            self.key[qid]["family"] = family
        return qid


def check_n(n: int) -> None:
    """The scale draw's n: at least 20 (one E1 and one E2 partner) and a multiple of five (the generator makes n / 5 facts per
    schema; any other n fails inside the frozen generator)."""
    if not isinstance(n, int) or n < 20 or n % 5 != 0:
        raise ValueError(f"n must be a multiple of five and at least 20 (got {n!r})")


def seal_file() -> Path:
    """The seal file beside the package: the bench must be run from a clone of the repository (the module imported from the
    clone), since the package alone does not carry metrics/."""
    return S.SEAL_PATH if S.SEAL_PATH.is_absolute() else Path(__file__).resolve().parents[1] / S.SEAL_PATH


def build_bench(seed: int, n: int, e4: bool = False, families: bool = False) -> dict:
    """The bench at a seed: the scale draw of C.1 (`sealed_families.scale_plan(seed, n, seed + 100)`: the facts in their written
    order, E1 partners after the facts at written positions i with i mod 20 == 0, E2 partners after those with i mod 20 == 10,
    the never planted control of one thousand names), the E4 paraphrase after each fact when asked, and the three sealed
    families as sealed when asked (their content hashes checked against the seal, which must exist before anything is built).
    A fact whose (entity, schema) the draw already wrote (ADR-010 C.1 (b): the seal's hundred thousand draw holds two) is written
    as a statement and read as an E1 partner by chance of the first: the key's value is the first write's, no second question
    is asked, and the construction lists the repeated keys."""
    check_n(n)
    seal = None
    if families:
        sp = seal_file()
        if not sp.exists():
            raise SystemExit(f"--families needs the seal file {S.SEAL_PATH} (a clone of the repository, the module imported from it); not found at {sp}")
        seal = json.loads(sp.read_text(encoding="utf-8"))
    plan = S.scale_plan(seed, n, seed + 100)
    b = _Bench()
    facts = plan["_plan"]["facts"]; partners = plan["_plan"]["partners"]; never = plan["_plan"]["never"]
    by_after = {}
    dropped = 0
    for p in partners:
        if p.get("dropped"):
            dropped += 1
        else:
            by_after.setdefault(p["after"], []).append(p)
    own, supporting, partner_ptr, kind_of, repeated, value_of = {}, {}, {}, {}, {}, {}
    for i, f in enumerate(facts):
        key = (f.entity, f.schema)
        p0 = b.add_statement(f.statement, "fact", f.schema, f.entity, str(f.attr))
        value_of[p0] = str(f.attr)
        if key in own:                                                        # a repeated (entity, schema): an E1 partner by chance of the first write (C.1 (b); its value
            repeated.setdefault(key, []).append(p0)                           # differs from the first's, else the texts would be one and `add_statement` refused the draw)
            if key not in partner_ptr:
                partner_ptr[key] = p0; kind_of[key] = "E1"
        else:
            own[key] = p0; supporting[key] = [p0]
        anchor = own[key]
        if e4:
            p4 = b.add_statement(E4_TEMPLATES[f.schema].format(e=f.entity, a=f.attr), "E4", f.schema, f.entity, str(f.attr), partner_of=anchor)
            if anchor == p0:
                supporting[key].append(p4)
                if key not in kind_of:
                    partner_ptr[key] = p4; kind_of[key] = "E4"
        for p in by_after.get(i, []):
            if p["kind"] == "E1":
                pp = b.add_statement(p["statement"], "E1", f.schema, f.entity, str(p["attribute"]), partner_of=anchor)
            else:
                pp = b.add_statement(p["statement"], "E2", f.schema, p["neighbour"], str(p["attribute"]), partner_of=anchor)
            partner_ptr[key] = pp; kind_of[key] = p["kind"]                   # an E1 or E2 partner takes precedence over the paraphrase and over a repeated key for the contested block
    asked = set()
    for f in facts:
        key = (f.entity, f.schema)
        if key in asked:
            continue                                                          # one question per (entity, schema): the repeated fact's text is the first's
        asked.add(key)
        value = value_of[own[key]]
        block = f"contested_{kind_of[key]}" if key in kind_of else "planted"
        b.add_question(f.query, block, f.entity, f.schema, value, own[key], supporting[key], partner=partner_ptr.get(key))
    for v in never:
        b.add_question(v["query"], "never_planted", v["entity"], v["schema"], None, None, [])
    construction = {"seed": seed, "n": n, "e4": bool(e4), "families": bool(families), "generator_hash": plan["generator_hash"], "plan_content_sha256": plan["content_sha256"],
                    "never_planted_seed": seed + 100, "rates": {"E1": "one after each fact at a written position i with i mod 20 == 0", "E2": "one after each fact at i mod 20 == 10, a bumped name already taken dropped",
                                                                  "E4": "one after each fact when asked"},
                    "e1_count": plan["partners"]["E1"], "e2_count": plan["partners"]["E2"], "e2_dropped": plan["partners"]["E2_dropped"], "e2_dropped_seen": dropped, "e4_count": n if e4 else 0,
                    "repeated_keys": [{"entity": k[0], "schema": k[1], "first": own[k], "later": v, "what": "a later fact of the draw on an entity and schema already written: its line is an E1 partner by chance of the first write, the key's value the first's, no second question (ADR-010 C.1 (b))"}
                                      for k, v in sorted(repeated.items())],
                    "never_planted_n": len(never), "never_planted_distance_counts": plan["never_planted"]["distance_to_nearest_planted_counts"],
                    "what": "the scale draw of ADR-010 C.1 at this seed from the sealed generator, no encoder, no journal; the sealed families as sealed when asked"}
    if families:
        fams = S.generate_all()
        protected = _near_name_protected(fams)
        construction["families"] = {}
        for name in ("T0", "A", "Av"):
            fam = fams[name]
            fs = S.family_seal(fam)
            if fs["content_sha256"] != seal["families"][name]["content_sha256"]:
                raise SystemExit(f"the family {name} regenerated here differs from the seal; refused")
            construction["families"][name] = {"content_sha256": fs["content_sha256"], "n_facts": fam["n_facts"], "n_never_planted": fam["n_never_planted"], "n_cross_schema": fam["n_cross_schema"]}
            fown = {}
            for f in fam["facts"]:
                p0 = b.add_statement(f["statement"], "family", f["schema"], f["entity"], str(f["attribute"]), family=name)
                fown.setdefault((f["entity"], f["schema"]), p0)
            for f in fam["facts"]:
                k = (f["entity"], f["schema"])
                b.add_question(f["query"], "family", f["entity"], f["schema"], str(f["attribute"]), fown[k], [fown[k]], family=name)
            for f in fam["facts"]:
                k = (f["entity"], f["schema"])
                b.add_question(f["paraphrase"], "family_paraphrase", f["entity"], f["schema"], str(f["attribute"]), fown[k], [fown[k]], family=name)
            for v in fam["never_planted"]:
                b.add_question(v["query"], "never_planted", v["entity"], v["schema"], None, None, [], family=name)
            nn = _near_name_block(fam, fams, protected)
            for q in nn:
                b.add_question(q["query_asked"], "near_name", q["entity"], q["schema"], None, None, [], family=name)
            for c in fam["cross_schema"]:
                b.add_question(c["question"], "cross_schema", c["entity"], c["schema_asked"], None, None, [], family=name)
            construction["families"][name]["n_near_name"] = len(nn)
            construction["families"][name]["near_names_dropped"] = (nn[0]["dropped"] if nn else None)
    content = [[s["pointer"], s["text"], s["kind"]] for s in b.statements] + [[q["id"], q["text"], q["block"]] for q in b.questions]
    construction["content_sha256"] = S.content_hash_of(content)
    construction["module_sha256_lf"] = module_hash()
    import numpy
    construction["versions"] = {"python": sys.version.split()[0], "numpy": numpy.__version__}
    construction["time_utc"] = now_utc()
    construction["counts"] = {"statements": len(b.statements), "questions": len(b.questions), "by_block": dict(Counter(q["block"] for q in b.questions))}
    return {"benchmark": "provenance_bench", "construction": construction, "statements": b.statements, "questions": b.questions, "key": b.key,
            "what": {"blocks": BLOCK_WHAT, "support": SUPPORT_WHAT, "abstentions": ABSTENTION_WHAT, "contested": CONTESTED_WHAT, "conditions": CONDITIONS_WHAT}}


def content_sha256_of(bench: dict, smeta: dict | None = None, qkey: dict | None = None) -> str:
    """The bench's content hash recomputed from its statements and questions (the statements' kinds and the questions' blocks
    from the key file's meta for a split bench's public part)."""
    smeta = smeta if smeta is not None else {s["pointer"]: s for s in bench["statements"]}
    qkey = qkey if qkey is not None else (bench.get("key") or {})
    content = ([[s["pointer"], s["text"], s.get("kind", smeta.get(s["pointer"], {}).get("kind"))] for s in bench["statements"]]
               + [[q["id"], q["text"], q.get("block", qkey.get(q["id"], {}).get("block"))] for q in bench["questions"]])
    return S.content_hash_of(content)


def split(bench: dict) -> tuple[dict, dict]:
    """The public part (the statements as pointer and text, the questions as id and text, the construction) and the key file
    (per question the key with its block; per pointer the statement's kind, schema, entity, value and partner). The public part
    keeps the build's order and the construction's counts per block, so a question's block is inferable from its position: a
    blind run is blind to the key, not to the blocks."""
    pub = {"benchmark": bench["benchmark"], "construction": bench["construction"], "what": bench["what"],
           "statements": [{"id": s["id"], "pointer": s["pointer"], "text": s["text"]} for s in bench["statements"]],
           "questions": [{"id": q["id"], "text": q["text"]} for q in bench["questions"]]}
    key = {"benchmark": "provenance_bench key", "content_sha256": bench["construction"]["content_sha256"], "key": bench["key"],
           "statements": {s["pointer"]: {k: v for k, v in s.items() if k not in ("text",)} for s in bench["statements"]}}
    return pub, key


# ---------------------------------------------------------------------------- #
# the score                                                                       #
# ---------------------------------------------------------------------------- #
def _meta(bench: dict, key: dict | None) -> tuple[dict, dict]:
    """The per question key and the per pointer statement meta, from the bench or from the key file; the bench's content hash
    recomputed from its statements and questions and every pointer checked against its text (a bench edited after its build is
    refused)."""
    if key is not None:
        if key.get("content_sha256") != bench["construction"]["content_sha256"]:
            raise ValueError("the key file is not this bench's (content hashes differ)")
        qkey, smeta = key["key"], key["statements"]
    elif "key" in bench:
        qkey, smeta = bench["key"], {s["pointer"]: s for s in bench["statements"]}
    else:
        raise ValueError("the bench carries no key: pass the key file")
    for s in bench["statements"]:
        if content_hash(s["text"].encode("utf-8")) != s["pointer"]:
            raise ValueError(f"the statement {s.get('id')} does not hash to its pointer: the bench file was edited after its build")
    if content_sha256_of(bench, smeta, qkey) != bench["construction"]["content_sha256"]:
        raise ValueError("the bench's content hash does not match its statements and questions: the bench file was edited after its build")
    if set(qkey) != {q["id"] for q in bench["questions"]}:
        raise ValueError("the key does not cover the bench's questions exactly")
    return qkey, smeta


def check_answers(answers, qkey: dict | None = None) -> dict:
    """The answers file's shape: an object with `answers`, an object mapping question ids to objects {"value": a string or null,
    "cited": a pointer (sixteen hexadecimal characters) or null}; anything else is refused with the first offending id. Returns
    the answers object."""
    if not isinstance(answers, dict):
        raise ValueError("the answers file must be a JSON object")
    given = answers.get("answers")
    if not isinstance(given, dict):
        raise ValueError("the answers file must hold an object under 'answers' (a list or a missing key is refused)")
    for qid, a in given.items():
        if not isinstance(a, dict):
            raise ValueError(f"the answer to {qid!r} must be an object with 'value' and 'cited'")
        v, c = a.get("value"), a.get("cited")
        if v is not None and not isinstance(v, str):
            raise ValueError(f"the value answered to {qid!r} must be a string or null (got {type(v).__name__})")
        if c is not None and not (isinstance(c, str) and POINTER_RE.match(c)):
            raise ValueError(f"the pointer cited for {qid!r} must be sixteen hexadecimal characters or null")
    return given


def _block_table(rows: list[dict]) -> dict:
    """One block's metrics from its scored rows."""
    n = len(rows); answered = [r for r in rows if r["answered"]]; abst = [r for r in rows if not r["answered"]]
    return {"n": n,
            "coverage": _r(len(answered), n),
            "value_correct": _r(sum(1 for r in rows if r["value_correct"]), n),
            "citation_supported": _r(sum(1 for r in answered if r["supported"]), len(answered)),
            "citation_supported_answered": _r(sum(1 for r in answered if r["supported_answered"]), len(answered)),
            "citation_valid": _r(sum(1 for r in rows if r["valid"]), n),
            "citation_invalid": _r(sum(1 for r in rows if r["invalid"]), n),
            "strict": _r(sum(1 for r in rows if r["strict"]), n),
            "designated_line": _r(sum(1 for r in answered if r["designated"]), len(answered)),
            "attributed_accuracy": _r(sum(1 for r in rows if r["value_correct"] and r["supported"]), n),
            "abstentions": {"n": len(abst), "false": sum(1 for r in abst if r["supporting_exists"]), "verified": sum(1 for r in abst if not r["supporting_exists"])}}


def score(bench: dict, answers: dict, key: dict | None = None, answers_sha256: str | None = None) -> dict:
    """The table of one answers file on one bench (the formats in the module's docstring)."""
    qkey, smeta = _meta(bench, key)
    given = check_answers(answers)
    if answers.get("bench_content_sha256") is not None and answers["bench_content_sha256"] != bench["construction"]["content_sha256"]:
        raise ValueError("the answers file names another bench (bench_content_sha256 differs)")
    held = set(smeta)
    unknown = sorted(q for q in given if q not in qkey)
    rows_by_block: dict[str, list[dict]] = {}
    answered_total = abst_with_citation = 0
    for q in bench["questions"]:
        k = qkey[q["id"]]
        a = given.get(q["id"]) or {}
        value = a.get("value"); cited = a.get("cited")
        answered = value is not None
        if not answered and cited is not None:
            abst_with_citation += 1
        if answered:
            answered_total += 1
        line = smeta.get(cited) if cited in held else None
        valid = bool(answered and line is not None and line["value"] == value)        # the record's valid citation: the cited line holds the answered value
        invalid = bool(answered and not valid)                                         # the record's invalid citation: a claim with no valid citation
        value_correct = bool(answered and k["value"] is not None and value == k["value"])
        strict = bool(valid and value_correct)                                         # the record's strict answer
        supported = bool(answered and line is not None and line["entity"] == k["entity"] and k["value"] is not None and line["value"] == k["value"])   # section 4 (b), `citation_support_of`
        supported_answered = bool(answered and line is not None and line["entity"] == k["entity"] and line["value"] == value)                          # section 7's wording
        designated = bool(answered and cited is not None and cited == k["designated"])
        row = {"answered": answered, "value_correct": value_correct, "valid": valid, "invalid": invalid, "strict": strict, "supported": supported,
               "supported_answered": supported_answered, "designated": designated,
               "cites_partner": bool(answered and k.get("partner") is not None and cited == k["partner"]),
               "supporting_exists": bool(k["supporting"]), "cited": cited}
        rows_by_block.setdefault(k["block"], []).append(row)
    blocks = {}
    for block, rows in rows_by_block.items():
        t = _block_table(rows)
        if block in CLAIM_BLOCKS:
            t["claims"] = _r(sum(1 for r in rows if r["answered"]), len(rows))
        if block.startswith("contested_"):
            cp = sum(1 for r in rows if r["cites_partner"])
            t["cites_partner"] = {**_r(cp, len(rows)), "clopper_pearson_95": clopper_pearson(cp, len(rows))}
            t["cites_designated"] = _r(sum(1 for r in rows if r["designated"]), len(rows))
            t["abstains"] = _r(sum(1 for r in rows if not r["answered"]), len(rows))
        t["what"] = BLOCK_WHAT.get(block, "")
        blocks[block] = t
    pooled_rows = [r for blk in PLANTED_BLOCKS for r in rows_by_block.get(blk, [])]
    pooled = _block_table(pooled_rows)
    never_rows = rows_by_block.get("never_planted", [])
    claims = sum(1 for r in never_rows if r["answered"])
    strict = pooled["strict"]; inv = pooled["citation_invalid"]; attributed = pooled["attributed_accuracy"]
    conditions = {
        "gap": {"holds": bool(strict["rate"] is not None and strict["rate"] >= DELTA), "rate": strict, "delta": DELTA,
                "what": "the frozen gate's strict recall: a valid citation (the cited line carrying the answered value) with the exact value, over the planted questions; the off arm is zero by construction, so the gap is this recall"},
        "invalid_citations": {"holds": bool(inv["rate"] is not None and inv["rate"] <= INVALID_CITATION_MAX), "rate": inv, "max": INVALID_CITATION_MAX,
                              "what": "a value given with no pointer, a pointer the bench does not hold, or a held line not carrying the answered value, over the planted questions"},
        "never_planted_claims": {"holds": bool(never_rows) and claims / len(never_rows) < NEGCTRL_INVALID, "rate": _r(claims, len(never_rows)), "bound": NEGCTRL_INVALID,
                                 "what": "a value given on a name the bench never wrote, over the never planted questions"},
        "off_arm": {"measured": False, "what": "the record's off arm (the organ cut) is at the floor by construction and is not the bench's to measure"},
        "attributed_accuracy_gap": {"holds": bool(attributed["rate"] is not None and attributed["rate"] >= DELTA), "rate": attributed, "delta": DELTA,
                                    "what": "beside the frozen gap, as the record keeps the own line gap beside it: a correct value with a citation supported under section 4 (b), over the planted questions"},
        "what": CONDITIONS_WHAT,
    }
    conditions["holds"] = bool(conditions["gap"]["holds"] and conditions["invalid_citations"]["holds"] and conditions["never_planted_claims"]["holds"])
    conditions["holds_with_attributed_accuracy_gap"] = bool(conditions["attributed_accuracy_gap"]["holds"] and conditions["invalid_citations"]["holds"] and conditions["never_planted_claims"]["holds"])
    if answers_sha256 is None:
        answers_sha256 = hashlib.sha256(json.dumps(answers, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"benchmark": "provenance_bench table", "time_utc": now_utc(), "module": {"sha256_lf": module_hash()},
            "bench": {"content_sha256": bench["construction"]["content_sha256"], "seed": bench["construction"].get("seed"), "n": bench["construction"].get("n"),
                      "e4": bench["construction"].get("e4"), "families": bench["construction"].get("families"), "counts": bench["construction"].get("counts")},
            "answers_file": {"system": answers.get("system"), "sha256": answers_sha256, "sha256_what": "sha256 of the file's bytes through the command line, of the canonical JSON (sorted keys, no spaces) through `score`",
                             "answered": answered_total, "abstentions_with_a_citation": abst_with_citation, "unknown_question_ids": unknown},
            "blocks": blocks, "pooled_planted": {**pooled, "blocks": list(PLANTED_BLOCKS)}, "conditions": conditions,
            "what": {"support": SUPPORT_WHAT, "abstentions": ABSTENTION_WHAT, "contested": CONTESTED_WHAT,
                     "rates": "every rate is a numerator n over a denominator d; d is the block's n or its answered count as named"}}


def summary_lines(table: dict) -> list[str]:
    p = table["pooled_planted"]; c = table["conditions"]
    out = [f"bench {table['bench']['content_sha256'][:16]} (seed {table['bench']['seed']}, n {table['bench']['n']}); answers of {table['answers_file']['system']!r}, {table['answers_file']['answered']} answered; "
           f"answers file sha256 {table['answers_file']['sha256'][:16]}; module {table['module']['sha256_lf']}",
           f"planted questions {p['n']}: coverage {p['coverage']['n']} of {p['coverage']['d']}; value correct {p['value_correct']['n']} of {p['value_correct']['d']}; strict {p['strict']['n']}; "
           f"citation supported {p['citation_supported']['n']} of {p['citation_supported']['d']} answered (on the answered value {p['citation_supported_answered']['n']}); invalid {p['citation_invalid']['n']} of {p['citation_invalid']['d']}; "
           f"designated line {p['designated_line']['n']}; attributed accuracy {p['attributed_accuracy']['n']} of {p['attributed_accuracy']['d']}; abstentions {p['abstentions']['n']} (false {p['abstentions']['false']}, verified {p['abstentions']['verified']})"]
    for blk in ("never_planted", "near_name", "cross_schema"):
        if blk in table["blocks"]:
            b = table["blocks"][blk]; out.append(f"{blk}: claims {b['claims']['n']} of {b['claims']['d']}")
    for blk in ("contested_E1", "contested_E2", "contested_E4"):
        if blk in table["blocks"]:
            b = table["blocks"][blk]; lo, hi = b["cites_partner"]["clopper_pearson_95"]
            out.append(f"{blk}: cites the partner {b['cites_partner']['n']} of {b['n']} (exact 95 % {lo:.3f} to {hi:.3f}); the designated line {b['cites_designated']['n']}; abstains {b['abstains']['n']}; value correct {b['value_correct']['n']}")
    never = c["never_planted_claims"]
    out.append(f"conditions: gap {c['gap']['holds']} (strict recall {c['gap']['rate']['n']} of {c['gap']['rate']['d']} against {DELTA}); invalid citations {c['invalid_citations']['holds']} "
               f"({c['invalid_citations']['rate']['n']} of {c['invalid_citations']['rate']['d']} against {INVALID_CITATION_MAX}); never planted claims {c['never_planted_claims']['holds']} "
               f"({never['rate']['n']} of {never['rate']['d']} against {NEGCTRL_INVALID}{'; no never planted question' if never['rate']['d'] == 0 else ''}); off arm not measured; holds {c['holds']}; "
               f"with the attributed accuracy gap ({c['attributed_accuracy_gap']['rate']['n']} of {c['attributed_accuracy_gap']['rate']['d']}) {c['holds_with_attributed_accuracy_gap']}")
    return out


# ---------------------------------------------------------------------------- #
# a bench from a file of the record (the equality test of the declaration, 13)   #
# ---------------------------------------------------------------------------- #
def bench_from_record_file(f: dict) -> tuple[dict, dict]:
    """A bench and an answers file from one near duplicate probe file of row 49 (one whose writes carry `truth` and whose
    answers carry `query`: the identity session b files), so that the bench's scorer can be compared with
    `near_duplicate_reading.provenance_metrics` on the same reads. The system's value is the row's `answered` field (`attr` is
    the planted value); the citation the veto surviving one. The bench's content hash is that of the rebuilt statements and
    questions; a write carrying its text is checked against its pointer."""
    writes = (f.get("near_duplicates") or {}).get("planting", {}).get("writes") or f.get("planting", {}).get("writes") or []
    if not writes:
        from cortex_c2b.near_duplicate_reading import _writes
        writes = _writes(f)
    if not all("query" in a for a in f["answers"]["on"]) or not any(w.get("truth") for w in writes):
        raise ValueError("this file carries no `truth` on its writes or no `query` on its answers: the conversion reads row 49's identity files")
    schema_of = {a["entity"]: a["schema"] for a in f["answers"]["on"]}
    b = _Bench()
    own, lines = {}, {}
    for w in writes:
        if not (w.get("admitted") and w.get("pointer")):
            continue
        t = w.get("truth") or {}
        ent, val = t.get("entity"), t.get("value")
        kind = {"original": "fact"}.get(w.get("kind"), w.get("kind"))
        sch = schema_of.get(w.get("for_entity") or ent, "")
        partner_of = own.get(w.get("for_entity")) if kind != "fact" else None
        if w.get("text") and content_hash(w["text"].encode("utf-8")) != w["pointer"]:
            raise ValueError(f"the write {w['pointer']} does not hash to its text")
        s = {"id": f"s{len(b.statements):06d}", "pointer": w["pointer"], "text": w.get("text") or "", "kind": kind, "schema": sch, "entity": ent, "value": val}
        if partner_of:
            s["partner_of"] = partner_of
        b.statements.append(s); b.pointers.add(w["pointer"]); lines[w["pointer"]] = s
        if kind == "fact":
            own.setdefault(w.get("for_entity") or ent, w["pointer"])
    dup_of = (f.get("near_duplicates") or {}).get("pointer_of_duplicate") or {}
    answers = {}

    def answer_of(a):
        claimed = a.get("kind") != "unknown"
        return {"value": a.get("answered") if claimed else None, "cited": a.get("veto_cited_pointer", a.get("cited_pointer"))}
    for a in f["answers"]["on"]:
        e = a["entity"]; o = own.get(e)
        val = lines[o]["value"] if o else None
        supporting = [p for p, s in lines.items() if s["entity"] == e and s["schema"] == a["schema"] and s["value"] == val and s["kind"] in ("fact", "E4")]
        partner = dup_of.get(e)
        block = "contested_E2" if partner and lines.get(partner, {}).get("kind") == "E2" else ("contested_E1" if partner and lines.get(partner, {}).get("kind") == "E1" else ("contested_E4" if partner else "planted"))
        qid = b.add_question(a["query"], block, e, a["schema"], val, o, supporting, partner=partner)
        answers[qid] = answer_of(a)
    for a in f["answers"].get("negctrl", []):
        qid = b.add_question(a["query"], "never_planted", a["entity"], a["schema"], None, None, [])
        answers[qid] = answer_of(a)
    bench = {"benchmark": "provenance_bench (from a record file)", "statements": b.statements, "questions": b.questions, "key": b.key,
             "what": {"support": SUPPORT_WHAT, "abstentions": ABSTENTION_WHAT, "contested": CONTESTED_WHAT}}
    bench["construction"] = {"content_sha256": content_sha256_of(bench), "from_record_file": True, "counts": {"statements": len(b.statements), "questions": len(b.questions)}}
    return bench, {"system": "the record's own file", "answers": answers}


# ---------------------------------------------------------------------------- #
# command line                                                                    #
# ---------------------------------------------------------------------------- #
def _refuse_existing(p: Path) -> None:
    if p.exists():
        raise SystemExit(f"{p} exists; a file of the bench is never overwritten (choose another name)")


def _no_duplicate_keys(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"the key {k!r} appears twice in one JSON object")
        d[k] = v
    return d


def _read_json(path: Path, what: str):
    try:
        raw = path.read_bytes()
        return raw, json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise SystemExit(f"{what} {path} cannot be read as JSON: {e}")


def _cli(argv=None):
    ap = argparse.ArgumentParser(description="the provenance benchmark: build a bench from the sealed generator; score an answers file with the record's reading")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--seed", type=int, required=True); b.add_argument("--n", type=int, required=True)
    b.add_argument("--e4", action="store_true", help="a paraphrase partner after every fact (row 49's E4)"); b.add_argument("--families", action="store_true", help="the three sealed families as sealed")
    b.add_argument("--split", action="store_true", help="write the public part under --out and the key under <out>.key.json"); b.add_argument("--out", required=True)
    s = sub.add_parser("score"); s.add_argument("--bench", required=True); s.add_argument("--key", default=None); s.add_argument("--answers", required=True); s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        out = Path(args.out); _refuse_existing(out)
        keyp = out.with_name(out.name + ".key.json") if args.split else None
        if keyp is not None:
            _refuse_existing(keyp)
        try:
            bench = build_bench(args.seed, args.n, e4=args.e4, families=args.families)
        except ValueError as e:
            raise SystemExit(f"refused: {e}")
        if args.split:
            pub, key = split(bench); write_json(out, pub); write_json(keyp, key)
            print(f"written {out} and {keyp}  content_sha256 {bench['construction']['content_sha256']}")
        else:
            write_json(out, bench); print(f"written {out}  content_sha256 {bench['construction']['content_sha256']}")
        c = bench["construction"]["counts"]
        print(f"statements {c['statements']}, questions {c['questions']}: {json.dumps(c['by_block'])}")
        return bench
    out = Path(args.out); _refuse_existing(out)
    _, bench = _read_json(Path(args.bench), "the bench file")
    key = _read_json(Path(args.key), "the key file")[1] if args.key else None
    raw, answers = _read_json(Path(args.answers), "the answers file")
    try:
        table = score(bench, answers, key=key, answers_sha256=hashlib.sha256(raw).hexdigest())
    except ValueError as e:
        raise SystemExit(f"refused: {e}")
    table["answers_file"]["path"] = Path(args.answers).name
    write_json(out, table)
    for line in summary_lines(table):
        print(line)
    print(f"written {out}")
    return table


if __name__ == "__main__":
    _cli()
