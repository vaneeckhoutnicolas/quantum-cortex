"""cortex_c2b.near_duplicate_probe -- near duplicates at the write gate: the probe that must produce the
failure before any mechanism (ADR-007, amendment of 2026-10-13, evening; map line 28; RES-26 door 3;
declared with a committee of three readers in two rounds and validated by the founder on 2026-10-14 at
11:05, D1 A, D2 A, D3 A).

Three stages, every number from a file, nothing of the record touched:

  stage 1  controlled pairs of unit cues at an exact cosine c, written in order through the gate
           (`WritePath`, the protocol's own admission threshold) into a fresh memory holding the first
           cue alone: the dentate's transfer (the cosine of the two separated codes), the shared winner
           share, the gate's own surprise and decision, the stored pair's cosine after the journal's
           rounding; per grid value the refusal fraction, from it the band [c_any, c_all] and the
           reference c_half, reported and conditioning no reading; on the read side, by two `CueIndex`
           objects built directly (the legacy and the adopted geometry), whether the second cue is a
           candidate for the first and for a perturbed query, and the two marks. Seconds, no checkpoint,
           no key of the record; a second process must recompute every number bit for bit.
  stage 2  the realistic pairs under the record's encoders: the cues of the protocol's statements and
           questions recomputed on a checkpoint (the processor, one sequence per forward pass); a like
           for like precondition against row 45's file in two parts (the own line's rank exact and
           stopping, the cosines recorded within a tolerance and the drift attributed when exceeded);
           the distributions of E1 (one entity, two contents), E2 (two entities whose names differ by
           one character, one content) and E0 (two distinct entities) cosines; and the gate's decision
           on every write of an arm's journal, with the nearest stored code's identity (`plant_arm`,
           the journal stage 3 reads).
  stage 3  the collision at the read, by the instrument's new mode (`python -m cortex_c2b.hm_lm
           --near-duplicate-probe <arm>`): the arm's journal planted here, then the existing session B
           reading in a new process; F2 on provenance (`cited_pointer_is_own`, `cited_is_duplicate`).

The two failure modes, named before any number: F1, the false redundancy at the write (a near
duplicate whose content differs refused with its partner as the nearest stored code); F2, the
collision at the read (fewer own lines cited under the near duplicates than under the density control,
on the two primary paired tests with Holm at 0.05). The readings are in `near_duplicate_reading`.
Nothing here is a claim. The first number comes from the first file.

Row 49 (ADR-007, amendment of 2026-10-14, evening; declared, built 2026-10-15, not measured): the same
arms planted under the identity rule (`--admission identity`), a fifth arm E4 (the same fact under a
second statement template per schema, `E4_TEMPLATES`, read by this module's own reader and never by
the tagged system's `value_of_statement`), the shadow legacy gate's verdict recorded per write, the
generator's truth per planted line, and two reported blocks after the protocol's (the near name never
planted block, `near_name_plan`; the neighbour questions of the E2 runs). Nothing here is measured.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cortex_c2b import Journal, POLICY_STOP, CUE_DIM, content_hash
from cortex_c2b.hm_protocol import Fact, generate_facts, _ATTRS, _TEMPLATES, N_FACTS, N_NEGCTRL, SCHEMAS
from cortex_c2b.read_path import CueIndex, ADOPTED_GEOMETRY, LEGACY_GEOMETRY
from cortex_c2b.write_path import WritePath, DentateGyrus

# ---- the constants of the declaration (fixed, not decisions) -----------------------------------------
GRID = (0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999)
N_PAIRS = 200
PAIR_SEED = 1                   # the pairs' draw
PERTURB_SEED = 2                # the perturbed query's draw
QUERY_COS = 0.95                # the perturbed query's cosine to the first cue (near the record's smallest own cosine, 0.944)
DG_SEED = 0                     # the write path's separation seed, the protocol's
ADMISSION_THRESHOLD = 0.15      # the protocol's own
INDEX_SEED = 0
E0D_SEED = 20_000               # the density control's distinct facts (not 0, the facts; not 10 000, the negative control; not 100 000, the training family)
E1_PLAN_OFFSET = 200_000        # E1's other attribute, one per fact, a seeded plan
PRECONDITION_TOL = 1e-4         # on a cosine: the journal's four decimal rounding under a device drift can move the fourth decimal
THRESHOLD_TOL = 1e-6            # the reproduction label on the oracle's threshold, row 47's
ARMS = ("B0", "E0d", "E1", "E2", "E4")
KINDS = ("E1", "E2")            # the near duplicate kinds (F1 and F2 are read on them)
PARAPHRASE_ARM = "E4"           # row 49: the same fact under a second statement template (a paraphrase, not a near duplicate)
PARTNER_KINDS = KINDS + (PARAPHRASE_ARM,)   # the writes that carry a partner (the original written just before)
LIVE_N_BITS, LIVE_N_TABLES = int(ADOPTED_GEOMETRY["n_bits"]), int(ADOPTED_GEOMETRY["n_tables"])
HEX = "0123456789abcdef"
NEAR_NAME_PER_SCHEMA = 10       # the near name never planted block: the first ten facts of each schema, their names moved

# E4's templates (ADR-007, amendment of 2026-10-14, evening, section 4 (b)): one second statement template per schema,
# constants of the declaration; read by `parse_statement` below (the protocol's templates tried first) and never by the
# tagged system's `value_of_statement`, so that "outside the templates" keeps its meaning for the sealed families
E4_TEMPLATES = {
    "person":   "the occupation of {e} is {a}",
    "place":    "{e}: a town of {a}",
    "event":    "the year of {e} was {a}",
    "object":   "{e} is {a} by its material",
    "decision": "the decision on {e} came from {a}",
}


# ---- small geometry ---------------------------------------------------------------------------------
def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return (v / n) if n > 0 else v


def pair_at(v: np.ndarray, c: float, rng: np.random.Generator) -> np.ndarray:
    """A unit vector at cosine exactly `c` to the unit vector `v` (before any rounding): c v + sqrt(1 - c^2) u with u a
    unit vector orthogonal to v drawn from `rng`. Computed in float64, returned in float32 (the organ's cue type)."""
    v = unit(v)
    u = rng.standard_normal(v.shape[0]); u = u - float(u @ v) * v; u = unit(u)
    return (c * v + math.sqrt(max(0.0, 1.0 - c * c)) * u).astype(np.float32)


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float(a @ b / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def rounded(cue) -> np.ndarray:
    """The cue as the journal keeps it (`Journal.write`: four decimals)."""
    return np.asarray([round(float(x), 4) for x in np.asarray(cue, dtype=np.float32)], dtype=np.float32)


def shared_winner_share(code_a: np.ndarray, code_b: np.ndarray) -> float:
    """The share of the k winners the two separated codes have in common (the fly's "shared indices" of row 45)."""
    sa, sb = set(np.flatnonzero(code_a).tolist()), set(np.flatnonzero(code_b).tolist())
    return (len(sa & sb) / max(1, len(sa))) if sa else 0.0


def expected_cobucket(c: float, n_bits: int = LIVE_N_BITS, n_tables: int = LIVE_N_TABLES) -> dict:
    """The collision law of random hyperplanes (Charikar 2002), the expectation written before the run for the exact
    signature: a shared bit with probability 1 - theta / pi, a shared bucket in one table with probability that to the
    power n_bits, a co candidate with probability 1 - (1 - p_table) ** n_tables. The collapsed signature folds keys and
    exceeds it; the probe measures both."""
    theta = math.acos(max(-1.0, min(1.0, c)))
    p_bit = 1.0 - theta / math.pi
    p_table = p_bit ** n_bits
    return {"p_bit": p_bit, "p_table": p_table, "p_candidate": 1.0 - (1.0 - p_table) ** n_tables}


def band(refusal_fraction_by_c: dict) -> dict:
    """[c_any, c_all] and c_half from the refusal fractions per grid value: the smallest grid value with a refusal, the
    smallest with every pair refused, the smallest with at least half the pairs refused; None when never reached."""
    cs = sorted(refusal_fraction_by_c)
    c_any = next((c for c in cs if refusal_fraction_by_c[c] > 0.0), None)
    c_all = next((c for c in cs if refusal_fraction_by_c[c] >= 1.0), None)
    c_half = next((c for c in cs if refusal_fraction_by_c[c] >= 0.5), None)
    return {"c_any": c_any, "c_all": c_all, "c_half": c_half}


# ---- stage 1 ----------------------------------------------------------------------------------------
def stage1(grid=GRID, n_pairs: int = N_PAIRS, pair_seed: int = PAIR_SEED, perturb_seed: int = PERTURB_SEED,
           query_cos: float = QUERY_COS, dg_seed: int = DG_SEED, admission: float = ADMISSION_THRESHOLD,
           index_seed: int = INDEX_SEED, dim: int = CUE_DIM, keep_pairs: bool = True) -> dict:
    """The gate's transfer on controlled pairs; deterministic given the seeds (the pairs drawn in grid order then pair
    order from one generator, the perturbations from another)."""
    rng_pair = np.random.default_rng(pair_seed)
    rng_q = np.random.default_rng(perturb_seed)
    dg = DentateGyrus(in_dim=dim, seed=dg_seed)
    geometries = {}
    per_c, pairs_out = {}, []
    for c in grid:
        rows = []
        for i in range(n_pairs):
            v = unit(rng_pair.standard_normal(dim)).astype(np.float32)
            v2 = pair_at(v, float(c), rng_pair)
            q = pair_at(v, float(query_cos), rng_q)
            # the gate: a fresh memory holding v alone, then v2 through the same gate with another text and another content
            j = Journal(policy=POLICY_STOP)
            wp = WritePath(j, admission_threshold=admission, seed=dg_seed)
            r1 = wp.write(v, f"first {c} {i}".encode(), "pair", now=0.0, source=f"first {c} {i}")
            code1_stored = wp._code(np.asarray(r1.entry.cue, dtype=np.float32))     # as CA3 stores it: the rounded cue's code
            code1, code2 = dg.separate(v), dg.separate(v2)                            # the transfer on the unrounded cues
            r2 = wp.write(v2, f"second {c} {i}".encode(), "pair", now=1.0, source=f"second {c} {i}")
            v_r, v2_r = rounded(v), rounded(v2)
            # the read side: two indexes built directly, the cues as journaled (rounded)
            cands = {}
            for name, g in (("legacy", LEGACY_GEOMETRY), ("adopted", ADOPTED_GEOMETRY)):
                idx = CueIndex(dim=dim, n_bits=g["n_bits"], n_tables=g["n_tables"], seed=index_seed, signature=g["signature"])
                if name not in geometries:
                    geometries[name] = idx.geometry()
                idx.add("a", v_r); idx.add("b", v2_r)
                cands[name] = {"second_for_first": "b" in idx.candidates(v_r), "second_for_query": "b" in idx.candidates(q),
                               "first_for_query": "a" in idx.candidates(q)}
            row = {"c": float(c), "i": i, "cos_before_rounding": cosine(v, v2), "cos_stored": cosine(v_r, v2_r),
                   "transfer": cosine(code1, code2), "shared_winner_share": shared_winner_share(code1, code2),
                   "gate_code_cosine": cosine(code1_stored, dg.separate(v2)),
                   "surprise": float(r2.surprise), "admitted": bool(r2.admitted), "reason": r2.reason,
                   "query_cos_first": cosine(q, v_r), "query_cos_second": cosine(q, v2_r), "candidates": cands}
            row["mark_difference"] = row["query_cos_first"] - row["query_cos_second"]
            rows.append(row)
        refused = sum(1 for r in rows if not r["admitted"])
        per_c[float(c)] = {
            "n": len(rows), "refused": refused, "refusal_fraction": refused / max(1, len(rows)),
            "transfer": {"mean": float(np.mean([r["transfer"] for r in rows])), "min": float(min(r["transfer"] for r in rows)), "max": float(max(r["transfer"] for r in rows))},
            "shared_winner_share_mean": float(np.mean([r["shared_winner_share"] for r in rows])),
            "surprise": {"mean": float(np.mean([r["surprise"] for r in rows])), "min": float(min(r["surprise"] for r in rows)), "max": float(max(r["surprise"] for r in rows))},
            "cos_stored_mean": float(np.mean([r["cos_stored"] for r in rows])),
            "mark_difference_mean": float(np.mean([r["mark_difference"] for r in rows])),
            "cobucket": {name: {k: sum(1 for r in rows if r["candidates"][name][k]) / max(1, len(rows)) for k in ("second_for_first", "second_for_query", "first_for_query")}
                         for name in ("legacy", "adopted")},
            "expected_exact": expected_cobucket(float(c)),
            "expected_exact_at_mean_cosine_query_second": expected_cobucket(float(c) * query_cos),   # the law at the mean cosine of (q, v'); the draw decides the pair's
        }
        if keep_pairs:
            pairs_out.extend(rows)
    fractions = {c: per_c[c]["refusal_fraction"] for c in per_c}
    out = {"probe": "near duplicates at the write gate, stage 1: controlled pairs (ADR-007, amendment of 2026-10-13, evening)",
           "constants": {"grid": [float(c) for c in grid], "n_pairs": n_pairs, "pair_seed": pair_seed, "perturb_seed": perturb_seed,
                         "query_cos": query_cos, "dg": {"seed": dg_seed, "in_dim": dim, "out_dim": dg.out_dim, "k": dg.k},
                         "admission_threshold": admission, "index_seed": index_seed},
           "geometries": geometries, "per_c": {str(c): per_c[c] for c in per_c},
           "band": band(fractions), "pairs": pairs_out}
    return out


def numbers_sha256(d: dict, drop: tuple = ("timestamp_utc", "versions", "device", "second_process", "code_commit", "code_tree_clean", "numbers_sha256", "declared_constants")) -> str:
    """The hash of the file's numbers alone, the fields a second process cannot reproduce left out."""
    body = {k: v for k, v in d.items() if k not in drop}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:16]


# ---- stage 2: the plans and the realistic pairs ------------------------------------------------------
def e1_plan(facts: list[Fact], seed: int) -> dict:
    """One other attribute of the same schema per fact, a seeded draw written once (the record's way, `revision_probe._plan`)."""
    rng = np.random.default_rng(E1_PLAN_OFFSET + seed)
    plan = {}
    for f in facts:
        others = [a for a in _ATTRS[f.schema] if a != f.attr]
        plan[f.entity] = str(rng.choice(others))
    return plan


def statement_of(f: Fact, attr: str, entity: str | None = None) -> str:
    st, _, _ = _TEMPLATES[f.schema]
    return st.format(e=(entity if entity is not None else f.entity), a=attr)


def bump_name(name: str) -> str:
    """The generator's name with its last hexadecimal character moved by one, modulo sixteen."""
    last = name[-1]
    if last not in HEX:
        raise ValueError(f"not a generator name: {name!r}")
    return name[:-1] + HEX[(HEX.index(last) + 1) % 16]


def e2_plan(facts: list[Fact], protected_names: set[str]) -> dict:
    """The neighbouring name per fact, the pair dropped (None) when the new name coincides with one of the 250 names of the
    protocol (the 200 planted and the 50 never planted) or with another new name."""
    plan, seen = {}, set()
    for f in facts:
        nb = bump_name(f.entity)
        if nb in protected_names or nb in seen:
            plan[f.entity] = None
        else:
            plan[f.entity] = nb; seen.add(nb)
    return plan


def e0d_facts(n: int, seed: int, protected_names: set[str]) -> list[Fact]:
    """The density control's distinct facts: the generator at `E0D_SEED`, decontaminated against the protocol's names."""
    fs, _ = generate_facts(n, seed)
    return [f for f in fs if f.entity not in protected_names]


def duplicate_statement(f: Fact, kind: str, plan: dict | None) -> str | None:
    if kind == "E1":
        return statement_of(f, plan[f.entity])
    if kind == "E2":
        nb = plan.get(f.entity)
        return None if nb is None else statement_of(f, f.attr, entity=nb)
    if kind == PARAPHRASE_ARM:
        return e4_statement(f)
    raise ValueError(kind)


def partner_truth(f: Fact, kind: str, plan: dict | None) -> tuple[str, str]:
    """The generator's truth of a partner line: (entity, value). E1: the same entity, the plan's other value; E2: the
    neighbour's name, the same value; E4: the same entity and value under another template."""
    if kind == "E1":
        return f.entity, plan[f.entity]
    if kind == "E2":
        return plan[f.entity], f.attr
    if kind == PARAPHRASE_ARM:
        return f.entity, f.attr
    raise ValueError(kind)


def e4_statement(f: Fact) -> str:
    """The same fact under E4's template of its schema (the entity and the value unchanged)."""
    return E4_TEMPLATES[f.schema].format(e=f.entity, a=f.attr)


def _template_regex(template: str) -> str:
    import re
    return "^" + re.escape(template).replace(r"\{e\}", "(?P<e>.+?)").replace(r"\{a\}", "(?P<a>.+)") + "$"


def parse_statement(text: str) -> tuple[str, str, str, str] | None:
    """The probe's and the replay's own reader of a stored line: (schema, entity, value, template_set), the protocol's
    templates tried first (`value_of_statement`, the tagged system's reader), E4's only when none matches; None when
    neither set reads the line (a sealed family's statement, ADR-010 T5). The template sets' precedence is declared:
    a line the protocol reads is read as the protocol reads it."""
    import re
    from cortex_c2b.hm_protocol import value_of_statement
    v = value_of_statement(text)
    if v is not None:
        return v[0], v[1], v[2], "protocol"
    for schema, st in E4_TEMPLATES.items():
        m = re.match(_template_regex(st), text)
        if m:
            return schema, m.group("e"), m.group("a"), "E4"
    return None


def parse_question(text: str) -> tuple[str, str, str] | None:
    """The question's entity read from its text through the protocol's question templates (the direct query, then the
    paraphrase): (schema, entity, form) or None when no template reads it. The decode shapes of row 50 read the entity
    here and never from an answer record's `entity` label (ADR-007, amendment of 2026-10-14, evening, section 6)."""
    import re
    for schema, (_, q, pq) in _TEMPLATES.items():
        for form, tpl in (("query", q), ("paraphrase", pq)):
            pat = "^" + re.escape(tpl).replace(r"\{e\}", "(?P<e>.+?)") + "$"
            m = re.match(pat, text)
            if m:
                return schema, m.group("e"), form
    return None


def move_name(name: str, step: int) -> str:
    """The generator's name with its last hexadecimal character moved by `step`, modulo sixteen (+1: `bump_name`, E2's
    neighbour; -1: the near name never planted block, the other way)."""
    last = name[-1]
    if last not in HEX:
        raise ValueError(f"not a generator name: {name!r}")
    return name[:-1] + HEX[(HEX.index(last) + step) % 16]


def near_name_plan(facts: list[Fact], protected_names: set[str]) -> dict:
    """The near name never planted block (ADR-007, amendment of 2026-10-14, evening, section 5; the committee's finding
    14): the first `NEAR_NAME_PER_SCHEMA` facts of each schema in the generator's order, their names moved by one
    hexadecimal character the other way (minus one modulo sixteen), each a never planted entity one character from a
    planted one; a moved name that coincides with a protected name (the 250 protocol names, E2's neighbours, E0d's
    entities) or with another moved name is dropped and counted. Returns {"questions": [{entity, schema, query,
    planted_neighbour, attr_of_neighbour}], "dropped": [...]}."""
    per: dict[str, int] = {s: 0 for s in SCHEMAS}
    out, dropped, seen = [], [], set()
    for f in facts:
        if per[f.schema] >= NEAR_NAME_PER_SCHEMA:
            continue
        per[f.schema] += 1
        moved = move_name(f.entity, -1)
        if moved in protected_names or moved in seen:
            dropped.append({"planted": f.entity, "moved": moved})
            continue
        seen.add(moved)
        out.append({"entity": moved, "schema": f.schema, "query": _TEMPLATES[f.schema][1].format(e=moved),
                    "planted_neighbour": f.entity, "attr_of_neighbour": str(f.attr)})
    return {"questions": out, "dropped": dropped, "rule": "the first ten facts of each schema, the last hexadecimal character minus one modulo sixteen"}


def neighbour_questions(facts: list[Fact], plan: dict) -> list[Fact]:
    """The E2 runs' neighbour block: the saved neighbour's own question (the protocol's direct query on the neighbour's
    name; its own line is the duplicate's, which carries the original's value), one per pair the plan kept."""
    out = []
    for f in facts:
        nb = plan.get(f.entity)
        if nb is None:
            continue
        out.append(Fact(entity=nb, schema=f.schema, attr=f.attr, statement=statement_of(f, f.attr, entity=nb),
                        query=_TEMPLATES[f.schema][1].format(e=nb), paraphrase=_TEMPLATES[f.schema][2].format(e=nb),
                        cue=f.cue, address_text=f"{f.schema}|{nb}"))
    return out


def quantiles(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "q90": float(np.quantile(a, 0.9)),
            "q95": float(np.quantile(a, 0.95)), "q99": float(np.quantile(a, 0.99)), "min": float(a.min()), "max": float(a.max()),
            "above_grid": {str(c): int((a >= c).sum()) for c in GRID}}


def load_model(ckpt_path, cfg, device=None):
    """The checkpoint on the processor, the configuration hashing to it (fail closed), as session B loads it."""
    import torch
    import train
    device = device or torch.device("cpu")
    model = train.VanillaGPT(cfg).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not ck.get("config_hash"):
        raise ValueError("near duplicate probe: the checkpoint carries no config hash; the config cannot be checked as the run's")
    if train.config_hash(cfg) != ck["config_hash"]:
        raise ValueError(f"near duplicate probe: the config hashes to {train.config_hash(cfg)}, the checkpoint was trained under {ck['config_hash']}")
    model.load_state_dict(ck["model"]); model.eval()
    return model, ck, device


def text_cue(bridge, text: str) -> np.ndarray:
    """The cue of a statement, one sequence per forward pass, exactly as `JournalBridge.write` computes it."""
    return bridge.cues([list(text.encode("utf-8"))])[0]


def question_cue(bridge, query: str) -> np.ndarray:
    from cortex_c2b.lm_bridge import encode_query
    return bridge.cues([encode_query(query)])[0]


def precondition(q_cues: np.ndarray, s_cues_rounded: np.ndarray, row45_questions: list[dict], tol: float = PRECONDITION_TOL,
                 entities: list[str] | None = None, row45_query_cues=None) -> dict:
    """Like for like against row 45's file, in two parts. The order is exact and stops: per question the own line's rank
    among the stored cues by cosine (`rank_under_c6`) equal to the file's. The cosines are recorded: `cos_query_own` and
    `best_other_cosine` per question with their differences to the file's, within `tol` expected, the drift attributed to
    the devices when exceeded with the order intact (the probe continues). Row 45's file holds the best other entry's
    cosine and not its identity, so the identity is recomputed and recorded here and compared to nothing; the file does
    hold the record's question cues (`query_cues.questions`), so their difference to the recomputed ones is recorded too,
    which tells a drift of the question cues (processor against card) from one of the stored statement cues (processor
    against the training device). The questions are in the generator's order in both; `entities` checks it."""
    n = q_cues.shape[0]
    if len(row45_questions) != n:
        raise ValueError(f"precondition: {len(row45_questions)} questions in the file, {n} recomputed")
    rows, diffs_own, diffs_other, order_mismatch, compared = [], [], [], [], 0
    for i in range(n):
        ref = row45_questions[i]
        if entities is not None and ref.get("entity") is not None and ref["entity"] != entities[i]:
            raise ValueError(f"precondition: question {i} is {entities[i]} here and {ref['entity']} in the file; not the same order")
        sims = np.array([cosine(q_cues[i], s_cues_rounded[j]) for j in range(n)])
        order = np.argsort(-sims, kind="stable")
        rank = int(np.where(order == i)[0][0]) + 1
        others = [(float(sims[j]), j) for j in range(n) if j != i]
        other, j_other = max(others)
        own = float(sims[i])
        row = {"i": i, "entity": ref.get("entity"), "rank": rank, "rank_in_file": ref.get("rank_under_c6"),
               "cos_query_own": own, "best_other_cosine": other, "best_other_entity": (entities[j_other] if entities is not None else None),
               "diff_own": (abs(own - float(ref["cos_query_own"])) if ref.get("cos_query_own") is not None else None),
               "diff_other": (abs(other - float(ref["best_other_cosine"])) if ref.get("best_other_cosine") is not None else None)}
        if ref.get("rank_under_c6") is not None:
            compared += 1
            if rank != int(ref["rank_under_c6"]):
                order_mismatch.append({"i": i, "entity": ref.get("entity"), "rank": rank, "rank_in_file": ref["rank_under_c6"]})
        if row["diff_own"] is not None:
            diffs_own.append(row["diff_own"])
        if row["diff_other"] is not None:
            diffs_other.append(row["diff_other"])
        rows.append(row)
    max_own = max(diffs_own) if diffs_own else None; max_other = max(diffs_other) if diffs_other else None
    exceeded = bool((max_own is not None and max_own > tol) or (max_other is not None and max_other > tol))
    q_diff = None
    if row45_query_cues is not None:
        ref_q = np.asarray(row45_query_cues, dtype=np.float32)
        if ref_q.shape == q_cues.shape:
            q_diff = float(np.max(np.abs(ref_q - np.asarray(q_cues, dtype=np.float32))))
    return {"n": n, "ranks_compared": compared, "tolerance": tol, "order_exact": not order_mismatch, "order_mismatches": order_mismatch,
            "max_abs_diff_cos_query_own": max_own, "max_abs_diff_best_other_cosine": max_other,
            "query_cues_max_abs_diff": q_diff,
            "best_other_identity": "recomputed and recorded per question; not compared: row 45's file holds the best other entry's cosine and not its identity",
            "within_tolerance": not exceeded, "drift_exceeded_order_intact": bool(exceeded and not order_mismatch),
            "drift_attribution": ("the cosines differ beyond the tolerance with the order intact: attributed to the devices (the question cues "
                                  "recomputed on the processor against row 45's computed on the card; the stored statement cues recomputed on the "
                                  "processor against the record's produced on the training device); see query_cues_max_abs_diff for which"
                                  if (exceeded and not order_mismatch) else None),
            "stopped": bool(order_mismatch), "per_question": rows}


def cues_stage(ckpt_path, cfg, row45_file: str | Path, n_facts: int = N_FACTS, n_negctrl: int = N_NEGCTRL, seed: int = 0,
               include_e0: bool = True) -> dict:
    """Stage 2 without a journal of the record: the precondition, the three distributions (E1, E2, E0) and the plans."""
    import torch
    from cortex_c2b import record_tools
    from cortex_c2b.lm_bridge import JournalBridge
    model, ck, device = load_model(ckpt_path, cfg)
    facts, gen_hash = generate_facts(n_facts, seed)
    neg, _ = generate_facts(n_negctrl, seed + 10_000)
    protected = {f.entity for f in facts} | {f.entity for f in neg}
    bridge = JournalBridge(model, Journal(policy=POLICY_STOP), k=cfg.journal_k, budget_bytes=cfg.journal_read_bytes,
                           seed=seed, device=device, index_geometry=ADOPTED_GEOMETRY)
    with torch.no_grad():
        s_cues = np.stack([text_cue(bridge, f.statement) for f in facts])
        q_cues = np.stack([question_cue(bridge, f.query) for f in facts])
        s_rounded = np.stack([rounded(x) for x in s_cues])
        row45_bytes = Path(row45_file).read_bytes()
        row45 = json.loads(row45_bytes.decode("utf-8"))
        if row45.get("checkpoint", {}).get("run_id") != ck.get("run_id"):
            raise ValueError("cues stage: the row 45 file is not this checkpoint's")
        if row45.get("stopped"):
            raise ValueError("cues stage: the row 45 file is a stopped run, not the row's file")
        pre = precondition(q_cues, s_rounded, row45["questions"], entities=[f.entity for f in facts],
                           row45_query_cues=(row45.get("query_cues") or {}).get("questions"))
        plan1, plan2 = e1_plan(facts, seed), e2_plan(facts, protected)
        e1, e2, e0 = [], [], []
        e1_rows, e2_rows = [], []
        for i, f in enumerate(facts):
            for a in _ATTRS[f.schema]:
                if a == f.attr:
                    continue
                c = cosine(s_cues[i], text_cue(bridge, statement_of(f, a)))
                e1.append(c); e1_rows.append({"entity": f.entity, "attr": a, "cos": c, "planned": a == plan1[f.entity]})
            nb = plan2[f.entity]
            if nb is not None:
                c = cosine(s_cues[i], text_cue(bridge, statement_of(f, f.attr, entity=nb)))
                e2.append(c); e2_rows.append({"entity": f.entity, "neighbour": nb, "cos": c})
        if include_e0:
            for i in range(len(facts)):
                for j in range(i + 1, len(facts)):
                    e0.append(cosine(s_cues[i], s_cues[j]))
    out = {"probe": "near duplicates at the write gate, stage 2: the realistic pairs under the record's encoder (ADR-007, amendment of 2026-10-13, evening)",
           "checkpoint": {"path": str(ckpt_path), "run_id": ck.get("run_id"), "config_hash": ck.get("config_hash")},
           "generator": {"config_hash": gen_hash, "n_facts": n_facts, "n_negctrl": n_negctrl, "seed": seed},
           "row45_file": {"name": str(Path(row45_file).name), "sha256_lf": hashlib.sha256(row45_bytes.replace(b"\r\n", b"\n")).hexdigest()[:16],
                          "code_commit": row45.get("code_commit")},
           "precondition": pre,
           "distributions": {"E1": quantiles(e1), "E2": quantiles(e2), "E0": quantiles(e0)},
           "e1_pairs": e1_rows, "e2_pairs": e2_rows,
           "plans": {"E1": plan1, "E2": plan2, "E2_dropped": sorted(e for e, nb in plan2.items() if nb is None)},
           "constants": {"precondition_tolerance": PRECONDITION_TOL, "e1_plan_seed": E1_PLAN_OFFSET + seed, "grid": [float(c) for c in GRID]},
           "device": record_tools.device_info(device), "versions": record_tools.versions(),
           "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return out


# ---- stage 2's gate, stage 3's journal: planting an arm -------------------------------------------
def nearest_stored(wp: WritePath, incoming_code: np.ndarray, entity_of_pointer: dict) -> dict | None:
    """The stored code nearest to the incoming one, exactly as `CA3Completion.reconstruct` finds it (the codes CA3 holds,
    in admission order, float32, the argmax of the cosines), mapped to its entry through the journal's `ca3_order`, with the
    entity its payload names; None on an empty memory."""
    codes = wp.ca3._codes
    if not codes:
        return None
    M = np.stack(codes)
    code = np.asarray(incoming_code, dtype=np.float32)
    sims = M @ code / (np.linalg.norm(M, axis=1) * (np.linalg.norm(code) + 1e-8) + 1e-8)
    k = int(np.argmax(sims))
    eid = wp.ca3.entry_ids[k] if k < len(wp.ca3.entry_ids) and wp.ca3.entry_ids[k] is not None else wp.j.ca3_order[k][0]
    e = wp.j._entries[eid]
    return {"entry_id": eid, "pointer": e.pointer, "entity": entity_of_pointer.get(e.pointer), "cosine": float(sims[k]), "index": k}


def write_recorded(bridge, text: str, schema: str, now: float, entity: str | None, partner: dict | None,
                   entity_of_pointer: dict, kind: str, for_entity: str | None = None, stored_as: str | None = None,
                   truth: tuple[str, str] | None = None):
    """One write through the gate exactly as `JournalBridge.write` does it (the cue from the text, the text as source and
    content, the index told of an admission, the entity's pointer kept for the originals only), with the gate's own
    quantities recorded beside it: the incoming code's nearest stored code before the write, and against the partner
    (the original of a near duplicate or of a paraphrase) the raw cosine, the separated codes' cosine, the shared winner
    share. Row 49: the cue as computed and as journaled, the journal's rule, the `near` stamp, and the shadow legacy gate's
    verdict (`legacy_surprise`, `legacy_decision`, `legacy_nearest`) when the write path carries the shadow; `truth`, the
    generator's (entity, value) of the line, kept for the citation support reading."""
    cue = text_cue(bridge, text)
    wp = bridge.wp
    incoming = wp._code(cue)
    near = nearest_stored(wp, incoming, entity_of_pointer)
    rec = {"kind": kind, "text": text, "entity": entity, "for_entity": (for_entity if for_entity is not None else entity),
           "now": now, "nearest_before": near}
    if truth is not None:
        rec["truth"] = {"entity": str(truth[0]), "value": str(truth[1])}
    if partner is not None:
        pcode_stored = wp._code(np.asarray(partner["entry"].cue, dtype=np.float32))
        rec["partner"] = {"entity": partner["entity"], "pointer": partner["entry"].pointer, "admitted": True,
                          "cos_raw": cosine(cue, partner["cue"]), "cos_separated": cosine(incoming, pcode_stored),
                          "shared_winner_share": shared_winner_share(incoming, pcode_stored),
                          "nearest_is_partner": bool(near is not None and near["entry_id"] == partner["entry"].entry_id)}
    elif kind in PARTNER_KINDS:
        rec["partner"] = {"entity": for_entity, "pointer": None, "admitted": False, "nearest_is_partner": False}
    rep = wp.write(cue, text.encode("utf-8"), schema, now=now, source=text)
    if rep.admitted:
        bridge.jp.on_write(rep.entry)
        entity_of_pointer[rep.entry.pointer] = stored_as if stored_as is not None else (entity if entity is not None else for_entity)
    if (rep.admitted or rep.revised) and entity is not None:
        bridge.pointer_of[entity] = rep.entry.pointer
    rec.update({"admitted": bool(rep.admitted), "revised": bool(rep.revised), "surprise": float(rep.surprise), "reason": rep.reason,
                "pointer": (rep.entry.pointer if rep.entry is not None else None),
                # row 49: the rule, the cue as computed and as journaled, the comparator's verdict and the shadow's
                "admission": rep.admission, "near": rep.near,
                "cue_computed": [float(x) for x in cue],
                "cue_journaled": ([float(x) for x in rep.entry.cue] if (rep.admitted and rep.entry is not None) else None),
                "nearest_entry_id": rep.nearest_entry_id, "nearest_cosine": rep.nearest_cosine,
                "legacy_surprise": rep.legacy_surprise, "legacy_decision": rep.legacy_admitted,
                "legacy_nearest": rep.legacy_nearest_entry_id})
    if rep.legacy_nearest_entry_id is not None:
        le = wp.j._entries.get(rep.legacy_nearest_entry_id)
        rec["legacy_nearest_entity"] = entity_of_pointer.get(le.pointer) if le is not None else None
    return rep, rec, cue


def plant_arm(model, cfg, journal_path: str | Path, key: bytes, arm: str, seed: int = 0, n_facts: int = N_FACTS,
              n_negctrl: int = N_NEGCTRL, device=None, admission: str | None = None) -> dict:
    """The arm's fresh journal (the adopted geometry declared at construction, the key generated for the run), 200 writes
    for B0, 400 for the others: the originals in the protocol's order, each followed by its near duplicate (E1, E2), by
    its paraphrase (E4, row 49) or by a distinct fact of the density control (E0d). `admission` (row 49): the journal's
    rule, `surprise` (the code's default, row 48's plantings) or `identity`; the shadow legacy gate judges every write
    beside the rule and its verdict is recorded. Returns the planting record (stage 2's gate numbers for this arm)."""
    import torch
    from cortex_c2b import record_tools, ADMISSION_SURPRISE, ADMISSION_RULES
    from cortex_c2b.lm_bridge import JournalBridge
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    admission = ADMISSION_SURPRISE if admission is None else admission
    if admission not in ADMISSION_RULES:
        raise ValueError(f"admission must be one of {ADMISSION_RULES}, got {admission!r}")
    device = device or torch.device("cpu")
    facts, gen_hash = generate_facts(n_facts, seed)
    neg, _ = generate_facts(n_negctrl, seed + 10_000)
    protected = {f.entity for f in facts} | {f.entity for f in neg}
    plan = None; extra = None
    if arm == "E1":
        plan = e1_plan(facts, seed)
    elif arm == "E2":
        plan = e2_plan(facts, protected)
    elif arm == PARAPHRASE_ARM:
        plan = {f.entity: e4_statement(f) for f in facts}                       # the paraphrase per fact, written for the record
    elif arm == "E0d":
        extra = e0d_facts(n_facts, E0D_SEED, protected)
        if len(extra) < len(facts):                                             # a decontamination dropped some: top up schema by schema
            extra_all, _ = generate_facts(2 * n_facts, E0D_SEED)                # so that the density control keeps the generator's balance
            per = len(facts) // len(SCHEMAS)
            extra = [g for s in SCHEMAS for g in [x for x in extra_all if x.schema == s and x.entity not in protected][:per]]
    journal = Journal(journal_path, key=key, policy=POLICY_STOP, index_geometry=ADOPTED_GEOMETRY)
    mark = bool(getattr(cfg, "journal_familiarity_mark", False))
    bridge = JournalBridge(model, journal, k=cfg.journal_k, budget_bytes=cfg.journal_read_bytes, seed=seed, shuffle_seed=seed + 1,
                           device=device, mark=mark, index_geometry=ADOPTED_GEOMETRY, admission=admission, shadow_legacy=True)
    entity_of_pointer: dict = {}
    writes = []
    with torch.no_grad():
        for i, f in enumerate(facts):
            rep, rec, cue = write_recorded(bridge, f.statement, f.schema, float(2 * i), f.entity, None, entity_of_pointer, "original",
                                           truth=(f.entity, f.attr))
            writes.append(rec)
            partner = {"entity": f.entity, "entry": rep.entry, "cue": cue} if rep.admitted else None
            if arm in PARTNER_KINDS:
                dup = duplicate_statement(f, arm, plan)
                if dup is None:
                    writes.append({"kind": arm, "entity": f.entity, "dropped": True, "now": float(2 * i + 1)})
                    continue
                stored_as = (plan[f.entity] if arm == "E2" else f"{f.entity} ({arm} {'paraphrase' if arm == PARAPHRASE_ARM else 'duplicate'})")
                _, rec2, _ = write_recorded(bridge, dup, f.schema, float(2 * i + 1), None, partner, entity_of_pointer, arm,
                                            for_entity=f.entity, stored_as=stored_as, truth=partner_truth(f, arm, plan))
                writes.append(rec2)
            elif arm == "E0d":
                if i < len(extra):
                    g = extra[i]
                    _, rec2, _ = write_recorded(bridge, g.statement, g.schema, float(2 * i + 1), g.entity, None, entity_of_pointer, "E0d",
                                                truth=(g.entity, g.attr))
                    rec2["distinct_entity"] = g.entity
                    writes.append(rec2)
    dups = [w for w in writes if w.get("kind") in KINDS and not w.get("dropped")]
    refused = [w for w in dups if not w["admitted"]]
    paraphrases = [w for w in writes if w.get("kind") == PARAPHRASE_ARM and not w.get("dropped")]
    originals = [w for w in writes if w.get("kind") == "original"]
    extras = [w for w in writes if w.get("kind") == "E0d"]
    judged = [w for w in writes if not w.get("dropped") and w.get("legacy_decision") is not None]
    record = {"arm": arm, "seed": seed, "n_facts": n_facts, "generator_hash": gen_hash,
              "index_geometry": bridge.jp.index.geometry(), "journal_record": journal.index_record(),
              # row 49: the journal's rule as recorded after the planting, and the shadow legacy gate's own counts
              "admission": {"rule": journal.admission_record(), "source": journal.admission_record_source(),
                            "asked": admission, "threshold": bridge.wp.thr,
                            "near_recorded": len(journal.near_of),
                            "shadow_legacy": {"judged": len(judged),
                                              "legacy_refused": sum(1 for w in judged if w["legacy_decision"] is False),
                                              "legacy_refused_duplicates": sum(1 for w in judged if w["kind"] in KINDS and w["legacy_decision"] is False),
                                              "legacy_refused_paraphrases": sum(1 for w in judged if w["kind"] == PARAPHRASE_ARM and w["legacy_decision"] is False),
                                              "legacy_refused_originals": sum(1 for w in judged if w["kind"] == "original" and w["legacy_decision"] is False),
                                              "legacy_refused_e0d": sum(1 for w in judged if w["kind"] == "E0d" and w["legacy_decision"] is False),
                                              "disagreements_with_rule": sum(1 for w in judged if w["legacy_decision"] != w["admitted"])}},
              "plan": plan, "e0d_entities": ([g.entity for g in extra] if extra is not None else None),
              "counts": {"originals_admitted": sum(1 for w in originals if w["admitted"]), "originals": len(originals),
                         "duplicates": len(dups), "duplicates_admitted": sum(1 for w in dups if w["admitted"]),
                         "duplicates_refused": len(refused),
                         "duplicates_refused_partner_nearest": sum(1 for w in refused if w.get("partner", {}).get("nearest_is_partner")),
                         "duplicates_whose_original_was_refused": sum(1 for w in dups if not w.get("partner", {}).get("admitted", False)),
                         "paraphrases": len(paraphrases), "paraphrases_admitted": sum(1 for w in paraphrases if w["admitted"]),
                         "paraphrases_refused": sum(1 for w in paraphrases if not w["admitted"]),
                         "dropped": sum(1 for w in writes if w.get("dropped")),
                         "e0d_written": len(extras), "e0d_admitted": sum(1 for w in extras if w["admitted"]),
                         "e0d_refused": sum(1 for w in extras if not w["admitted"]),
                         "entries": len(journal)},
              "pointer_of_duplicate": {w["for_entity"]: w["pointer"] for w in (dups + paraphrases) if w["admitted"]},
              "writes": writes, "device": record_tools.device_info(device), "versions": record_tools.versions(),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return record


def pointer_of_duplicate(facts: list[Fact], arm: str, seed: int, protected: set[str], payloads) -> dict:
    """Recomputed in session B from the generator and the plan alone (deterministic), the pointer of each admitted near
    duplicate, or paraphrase (E4): entity of the original -> content hash of the partner's statement, when it is in the
    journal's store."""
    if arm not in PARTNER_KINDS:
        return {}
    plan = e1_plan(facts, seed) if arm == "E1" else e2_plan(facts, protected) if arm == "E2" else None
    out = {}
    for f in facts:
        dup = duplicate_statement(f, arm, plan)
        if dup is None:
            continue
        p = content_hash(dup.encode("utf-8"))
        if p in payloads:
            out[f.entity] = p
    return out


# ---- command line -----------------------------------------------------------------------------------
def _second_process_hash(grid=GRID, n_pairs: int = N_PAIRS) -> str:
    """Stage 1 recomputed in a second process from the same seeds (the repository as the working directory, so that the
    child imports the same code); returns that process's numbers hash."""
    from cortex_c2b import record_tools
    code = ("import json, sys; from cortex_c2b import near_duplicate_probe as p; "
            "d = p.stage1(grid=%r, n_pairs=%r); print(p.numbers_sha256(d))" % (tuple(float(c) for c in grid), int(n_pairs)))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=1800, cwd=str(record_tools.REPO_ROOT),
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError("second process failed: " + proc.stderr[-500:])
    return proc.stdout.strip().splitlines()[-1]


def _cli(argv=None):
    import argparse
    from cortex_c2b import record_tools
    ap = argparse.ArgumentParser(description="near duplicates at the write gate (ADR-007, amendment of 2026-10-13, evening)")
    ap.add_argument("--stage1", action="store_true", help="the controlled pairs through the gate; the file near-dup-stage1.json")
    ap.add_argument("--cues", action="store_true", help="stage 2 without a journal: the precondition against row 45's file and the three distributions on a checkpoint")
    ap.add_argument("--ckpt", default=None); ap.add_argument("--config", default=None)
    ap.add_argument("--row45", default=None, help="with --cues: row 45's bucket probe file of this checkpoint")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-facts", type=int, default=N_FACTS); ap.add_argument("--n-negctrl", type=int, default=N_NEGCTRL)
    ap.add_argument("--out", default=None, help="the directory for the retained file (never overwritten)")
    ap.add_argument("--no-second-process", action="store_true", help="(tests) skip the second process of stage 1")
    ap.add_argument("--grid", default=None, help="(tests) a comma separated grid for stage 1 in place of the declared one; the file records it")
    ap.add_argument("--n-pairs", type=int, default=None, help="(tests) pairs per grid value for stage 1 in place of the declared 200; the file records it")
    args = ap.parse_args(argv)
    commit, tree_clean = record_tools.git_state()
    if commit is None or tree_clean is not True:
        raise SystemExit("near duplicate probe: the code's commit is unknown or the tracked tree is not clean; commit the code that runs, then run")
    outdir = Path(args.out) if args.out else Path("metrics/mqar")
    if args.stage1:
        for q in (outdir / "near-dup-stage1.json", outdir / "near-dup-stage1-FAILED-second-process.json"):
            if q.exists():
                raise SystemExit(f"near duplicate probe: {q} exists; a file of the record is never overwritten")
        grid = tuple(float(x) for x in args.grid.split(",")) if args.grid else GRID
        n_pairs = int(args.n_pairs) if args.n_pairs else N_PAIRS
        d = stage1(grid=grid, n_pairs=n_pairs)
        d["numbers_sha256"] = numbers_sha256(d)
        d["declared_constants"] = (tuple(grid) == tuple(GRID) and n_pairs == N_PAIRS)
        if not args.no_second_process:
            h2 = _second_process_hash(grid, n_pairs)
            d["second_process"] = {"numbers_sha256": h2, "agrees": h2 == d["numbers_sha256"]}
        d.update({"device": record_tools.device_info("cpu"), "versions": record_tools.versions(), "code_commit": commit, "code_tree_clean": tree_clean,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        agrees = d.get("second_process", {}).get("agrees", True)
        out = outdir / ("near-dup-stage1.json" if agrees else "near-dup-stage1-FAILED-second-process.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        print(f"retained: {out}  sha256 {hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")
        print(json.dumps({"band": d["band"], "refusal_fraction": {c: d["per_c"][c]["refusal_fraction"] for c in d["per_c"]},
                          "second_process": d.get("second_process")}, indent=2))
        if not agrees:
            raise SystemExit("near duplicate probe: the second process does not recompute stage 1's numbers; the file is retained as FAILED and the stage is unread")
        return
    if args.cues:
        if not (args.ckpt and args.config and args.row45):
            ap.error("--cues needs --ckpt, --config and --row45")
        import train
        import torch
        cfg = train.load_config(args.config)
        ck_head = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        rid = ck_head.get("run_id"); del ck_head
        if not rid:
            raise SystemExit("near duplicate probe: the checkpoint carries no run id; the file would be nameless")
        for q in (outdir / f"near-dup-cues-{rid}.json", outdir / f"near-dup-cues-{rid}-FAILED-precondition.json"):
            if q.exists():                                                        # before the minutes of compute
                raise SystemExit(f"near duplicate probe: {q} exists; a file of the record is never overwritten")
        try:
            d = cues_stage(args.ckpt, cfg, args.row45, n_facts=args.n_facts, n_negctrl=args.n_negctrl, seed=args.seed)
        except ValueError as e:
            raise SystemExit(f"near duplicate probe: {e}")
        d.update({"code_commit": commit, "code_tree_clean": tree_clean})
        name = f"near-dup-cues-{rid}" + ("-FAILED-precondition" if d["precondition"]["stopped"] else "") + ".json"
        out = outdir / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        print(f"retained: {out}  sha256 {hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")
        pre_short = {k: v for k, v in d["precondition"].items() if k != "per_question"}
        print(json.dumps({"precondition": pre_short, "distributions": {k: {kk: v.get(kk) for kk in ("n", "median", "q95", "max")} for k, v in d["distributions"].items()},
                          "E2_dropped": d["plans"]["E2_dropped"]}, indent=2))
        if d["precondition"]["stopped"]:
            raise SystemExit("near duplicate probe: the precondition stopped (the own line's rank differs from row 45's file); attributed before anything else runs")
        return
    ap.error("one of --stage1 / --cues is required (stage 3 is `python -m cortex_c2b.hm_lm --near-duplicate-probe <arm>`)")


if __name__ == "__main__":
    _cli()
