"""cortex_c2b.sealed_reading -- the cold readings of rows 51 and 52 (ADR-010 A.6, A.7 and B.2 as declared on 2026-10-14; the
amendment of 2026-10-15 on parts B and C, graved 2026-10-29: B.2 (a), the near name block enters no letter; 1 (d), the block
reported per family, index, rule and checkpoint), declared as functions before any family is run. A new module on the
frozen system `c5b4818`; it reads files and writes one reading file; it runs no model and opens no journal.

The inputs are the files the runner retains (`cortex_c2b.sealed_family`): per family (A, Av, T0) and checkpoint the family
file and its six pass files (three indexes, two forms), the record's row 47 `plain` files (the first family's direct form,
replayed under `mark-veto` by the frozen `replay_outcomes`), the first family's paraphrase passes when run, and A0's files
when run (`cortex_c2b.a0`). Every rate carries its exact interval (Clopper and Pearson 1934). The letters:

  A1, A2, A3    under `mark-veto`, the four frozen conditions and the own line gap on the six cells of A (form x checkpoint),
                the construction table restated; A1 on one form only is claimed for that form only;
  M1 to M4      the main risk's four numbers: `plain` against the conditions; the model's own line citation given the own
                line in the window, under `plain`, paired against the mark oracle's choice on the same reads (an exact two
                sided binomial on the discordant pairs, the two live seeds pooled per family over both forms, Holm over the
                families at 0.05, the third checkpoint reported, A0's ceiling beside); exactness given an own line citation
                under `mark-veto`; the veto's ledger;
  R-a, R-b, R-c the primary tests of B.2 on A (strict recall on the own line, the direct form, the two live seeds pooled,
                an exact two sided binomial on the discordant pairs, Holm over the two at 0.05, the third checkpoint agreeing
                in sign for a claim): (I0, `mark-veto`) against (I2, V1) and against (I1, V1); three outcomes per test, "not
                separated at this size" carrying the resolution sentence of the frozen reading module, never "equal".

The factor product (A.6) is read for A, Av, T0 and the first family with the differences A against Av, Av against the first
family, T0 against the first family per factor; the interval on a difference of two independent proportions is the hybrid
of the two exact intervals (Newcombe 1998, method 11), named as such in the file. Nothing here is a number of the record.
"""
from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path, PureWindowsPath

from cortex_c2b.hm_protocol import DELTA, NEGCTRL_INVALID, N_NEGCTRL
from cortex_c2b.hm_lm import INVALID_CITATION_MAX, replay_outcomes
from cortex_c2b.adoption_reading import LIVE_SEEDS, CONCEIVING_SEED, RUN_IDS, binomial_two_sided_p
from cortex_c2b.near_duplicate_reading import holm, RESOLUTION_SENTENCE, ALPHA
from cortex_c2b.bucket_probe import miss_class
from cortex_c2b import resumable as RS

FAMILIES = ("A", "Av", "T0")
INDEXES = ("I0", "I1", "I2")
FORMS = ("direct", "paraphrase")
RULES = ("plain", "mark-veto", "mark-veto+value", "V1", "V1+organ-value", "oracle", "oracle+organ-value")
LIVE_RUN_IDS = tuple(LIVE_SEEDS)                      # the two seeds that did not conceive the policy (rows 37, 39)
CONCEIVING_RUN_ID = next(iter(CONCEIVING_SEED))
ORGAN_STACK = ("I0", "mark-veto")
BUYER_STACKS = (("I2", "V1"), ("I1", "V1"))
PRIMARY_TESTS = {"I0 mark-veto against I2 V1": ("I2", "V1"), "I0 mark-veto against I1 V1": ("I1", "V1")}
OUTCOMES_PRIMARY = ("the organ's stack better", "the reference better", "not separated at this size")
A1_SENTENCE = ("`mark-veto` is not the system's default policy; under the default the system abstains on every question of A by "
               "construction (the construction table, ii)")
M2_SENTENCES = {"organ": "on these formulations the organ's mark reproduces the selection the protocol rewards; the model's measured contribution is its copy and its abstention",
                "model": "on these formulations the model selects better than the organ's mark"}
DIFFERENCE_INTERVAL_LABEL = "the hybrid of the two exact (Clopper and Pearson) intervals, Newcombe 1998 method 11; not an exact unconditional interval"
FIRST_FAMILY_LABEL = "in sample for the organ (geometry, row 45; policy, seed 1337); BM25's parameters as B.1 (b) labels them"
OFF_ARM_SENTENCE = "the off arm's strict recall is zero by construction (a citation is impossible without a window), so the gap condition is a strict recall at or above delta (the construction table, i)"
COMPONENTS = ("the organ's retrieval", "the mark", "the model's reading", "the copy")


# ---------------------------------------------------------------------------- #
# intervals                                                                      #
# ---------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float] | None:
    """The exact two sided interval (Clopper and Pearson 1934) by bisection on the binomial tails (sixty halvings, the
    log binomial coefficients by lgamma, the tail summed by fsum); cached by (k, n, alpha)."""
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
    # the lower bound: P(X >= k | p) = alpha/2, a tail that rises with p; the upper: P(X <= k | p) = alpha/2, a tail that falls
    lower = 0.0 if k == 0 else solve(lambda p: 1.0 - cdf_at_most(k - 1, p), alpha / 2, True)
    upper = 1.0 if k == n else solve(lambda p: cdf_at_most(k, p), alpha / 2, False)
    return lower, upper


def rate(k: int, n: int) -> dict:
    return {"k": int(k), "n": int(n), "rate": (k / n if n else None), "ci95": clopper_pearson(k, n)}


def difference_interval(k1: int, n1: int, k2: int, n2: int) -> dict:
    """p1 - p2 for two independent proportions with the hybrid of the two exact intervals (Newcombe 1998, method 11):
    L = d - sqrt((p1 - l1)^2 + (u2 - p2)^2), U = d + sqrt((u1 - p1)^2 + (p2 - l2)^2)."""
    if not n1 or not n2:
        return {"difference": None, "ci95": None, "label": DIFFERENCE_INTERVAL_LABEL}
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = clopper_pearson(k1, n1); l2, u2 = clopper_pearson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return {"difference": d, "ci95": (max(-1.0, lo), min(1.0, hi)), "label": DIFFERENCE_INTERVAL_LABEL}


# ---------------------------------------------------------------------------- #
# the files                                                                      #
# ---------------------------------------------------------------------------- #
def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_family(metrics_dir: str | Path, family: str, run_id: str) -> dict:
    """The family file and its six pass files, each pass file checked against the sha256_lf the family file names."""
    md = Path(metrics_dir)
    fpath = md / f"sealed-{family}-{run_id}.json"
    fam = load_json(fpath)
    passes = {}
    for key, meta in fam["passes"].items():
        index, form = key.split("-")
        ppath = md / PureWindowsPath(meta["file"]).name            # the family file names the pass file as the run's machine wrote it (either separator); read beside it
        sha = RS.sha256_file(ppath, lf=True)
        if sha != meta["sha256_lf"]:
            raise ValueError(f"{ppath}: sha256_lf {sha} differs from the family file's {meta['sha256_lf']}")
        passes[(index, form)] = load_json(ppath)
    return {"family": family, "run_id": run_id, "file": fam, "path": str(fpath), "passes": passes, "threshold": float(fam["threshold"]["threshold"])}


def records(p: dict, block: str, seed: int | None = None) -> list[dict]:
    out = [a for a in p["answers"] if a["block"] == block]
    if seed is not None:
        out = [a for a in out if a.get("seed") == seed]
    return out


def first_seed(p: dict) -> int | None:
    seeds = [a.get("seed") for a in p["answers"] if a["block"] == "never_planted" and a.get("seed") is not None]
    return min(seeds) if seeds else None


# ---------------------------------------------------------------------------- #
# counts and rates under a rule                                                   #
# ---------------------------------------------------------------------------- #
def counts(recs: list[dict], rule: str) -> dict:
    """The full partition of answerable questions under a rule (A.6), coverage and selective risk side by side, the strict
    recall on the own line; every rate with its exact interval."""
    n = len(recs)
    part: dict[str, int] = {}
    strict_own = strict = valid = invalid = abstain = claims = 0
    for a in recs:
        r = a["rules"][rule]
        part[a["partition"][rule]] = part.get(a["partition"][rule], 0) + 1
        claims += int(r["claim"]); strict += int(r["outcome"] == "strict"); valid += int(r["outcome"] == "valid")
        invalid += int(r["outcome"] == "invalid"); abstain += int(r["outcome"] == "abstain")
        strict_own += int(r["outcome"] == "strict" and r.get("cited_pointer_is_own") is True)
    return {"n": n, "partition": part, "claims": claims, "strict": strict, "strict_own": strict_own, "valid": valid, "invalid": invalid, "abstain": abstain,
            "coverage": rate(claims, n), "selective_risk": rate(claims - strict_own, claims) if claims else {"k": 0, "n": 0, "rate": None, "ci95": None},
            "strict_recall_own": rate(strict_own, n), "invalid_rate": rate(invalid, n),
            "selective_risk_what": "among the claims, those that are not a strict answer citing the own line"}


def claims_on(recs: list[dict], rule: str) -> dict:
    """A block without a planted line (never planted, cross schema, near name): any claim is a claim without an answer."""
    n = len(recs)
    c = sum(1 for a in recs if a["rules"][rule]["claim"])
    return {"n": n, "claims": c, "claim_rate": rate(c, n)}


def conditions(p: dict, rule: str) -> dict:
    """The four frozen conditions (declared 2026-09-12, not moved) under a rule on one pass, and the own line gap beside
    (A.6). The frozen gate (`hm_lm.gate_conditions_without_skill_arm`): the strict recall (a valid citation with the exact
    value, on any line) at or above delta, the construction table (i) making the gap that recall since the off arm is zero
    by construction; invalid citations at or under the maximum; claims on the never planted entities under the bound on
    the first seed's fifty and on all of them (A.6: the condition holds only if it holds on both); the off arm at the floor,
    by construction. The own line gap, beside: the same with the strict recall counted on the own line alone (A.6: "the gap
    computed on strict answers that cite the own line"); `holds` is the frozen four, `holds_with_own_line_gap` the four with
    the own line gap in place of the frozen gap (letter A reads both; the R letters read `holds`)."""
    on = counts(records(p, "planted"), rule)
    fs = first_seed(p)
    never_all = claims_on(records(p, "never_planted"), rule)
    never_first = claims_on(records(p, "never_planted", fs), rule) if fs is not None else never_all
    strict_any = rate(on["strict"], on["n"])
    gap_ok = strict_any["rate"] is not None and strict_any["rate"] >= DELTA
    own_gap_ok = on["strict_recall_own"]["rate"] is not None and on["strict_recall_own"]["rate"] >= DELTA
    invalid_ok = on["invalid_rate"]["rate"] is not None and on["invalid_rate"]["rate"] <= INVALID_CITATION_MAX
    never_ok = all(x["claim_rate"]["rate"] is not None and x["claim_rate"]["rate"] < NEGCTRL_INVALID for x in (never_first, never_all))
    broken = []
    if not gap_ok:
        broken.append(f"gap (the frozen strict recall, any line) {strict_any['rate']} under {DELTA}")
    if not invalid_ok:
        broken.append(f"invalid citations {on['invalid_rate']['rate']} above {INVALID_CITATION_MAX}")
    if not never_ok:
        broken.append(f"claims on never planted entities {never_first['claim_rate']['rate']} (first seed) / {never_all['claim_rate']['rate']} (all) at or above {NEGCTRL_INVALID}")
    broken_own = list(broken)
    if gap_ok and not own_gap_ok:
        broken_own.append(f"own line gap (strict recall on the own line) {on['strict_recall_own']['rate']} under {DELTA}")
    elif not gap_ok:
        broken_own = [b.replace("gap (the frozen strict recall, any line)", "own line gap (strict recall on the own line)") if b.startswith("gap (") and not own_gap_ok else b for b in broken]
    return {"rule": rule, "holds": not broken, "broken": broken, "holds_with_own_line_gap": (invalid_ok and never_ok and own_gap_ok), "broken_with_own_line_gap": broken_own,
            "gap": strict_any, "own_line_gap": on["strict_recall_own"], "delta": DELTA,
            "gap_what": "the frozen gate's strict recall (a valid citation with the exact value on any line) over the planted block; the own line gap beside counts the own line alone",
            "invalid_citations": on["invalid_rate"], "invalid_max": INVALID_CITATION_MAX,
            "never_planted": {"first_seed": {"seed": fs, **never_first}, "all": never_all, "bound": NEGCTRL_INVALID},
            "off_arm": OFF_ARM_SENTENCE, "on": on}


# ---------------------------------------------------------------------------- #
# the own line: the factor product, the misses, the marks                          #
# ---------------------------------------------------------------------------- #
def factors(recs: list[dict], rule: str) -> dict:
    """The factor product (A.6): the own line in the window; the own line cited given in the window; the exact value given
    the own line cited; each an exact rate, and their product."""
    n = len(recs)
    in_w = [a for a in recs if a.get("own_in_window")]
    cited = [a for a in in_w if a["rules"][rule].get("cited_pointer_is_own") is True]
    exact = [a for a in cited if a["rules"][rule]["outcome"] == "strict"]
    f1, f2, f3 = rate(len(in_w), n), rate(len(cited), len(in_w)), rate(len(exact), len(cited))
    prod = (f1["rate"] or 0.0) * (f2["rate"] or 0.0) * (f3["rate"] or 0.0) if (f1["rate"] is not None and f2["rate"] is not None and f3["rate"] is not None) else None
    return {"own_in_window": f1, "own_cited_given_in_window": f2, "exact_given_own_cited": f3, "product": prod, "strict_own": rate(len(exact), n), "rule": rule}


def factor_differences(fa: dict, fb: dict) -> dict:
    """Per factor, a minus b with the hybrid interval and the reading rule of A.6 (applied per live seed by the caller)."""
    out = {}
    for k in ("own_in_window", "own_cited_given_in_window", "exact_given_own_cited"):
        a, b = fa[k], fb[k]
        out[k] = difference_interval(a["k"], a["n"], b["k"], b["n"])
    return out


def transfer_reading(diffs_by_run: dict) -> dict:
    """A factor "not transferring" when its interval lies below zero on both live seeds, "no fall detectable at this size"
    when it contains zero on both, mixed otherwise; no letter depends on it."""
    out = {}
    for k in ("own_in_window", "own_cited_given_in_window", "exact_given_own_cited"):
        cis = [diffs_by_run[r][k]["ci95"] for r in LIVE_RUN_IDS if r in diffs_by_run and diffs_by_run[r][k]["ci95"] is not None]
        if len(cis) < 2:
            out[k] = "not read: a live seed is missing"; continue
        below = all(ci[1] < 0 for ci in cis); contains = all(ci[0] <= 0 <= ci[1] for ci in cis)
        out[k] = "not transferring" if below else ("no fall detectable at this size" if contains else "mixed")
    return out


def miss_classes(recs: list[dict]) -> dict:
    """The retrieval misses (the own line absent from the window) classed by the adoption's classes through the frozen
    `miss_class`: the index's, interference, the cue's; the rank and cosines of the own line under the whole store stand
    for the bucket probe's rank under its widest configuration."""
    out = {"n_misses": 0, "index": 0, "interference": 0, "cue": 0, "not_in_candidates": 0, "in_candidates_not_in_window": 0, "unclassed": 0}
    for a in recs:
        if a.get("own_in_window") or a.get("own_pointer") is None:
            continue
        out["n_misses"] += 1
        ws = a.get("whole_store") or {}
        in_c = bool(a.get("own_in_candidates"))
        if in_c:
            out["in_candidates_not_in_window"] += 1
        else:
            out["not_in_candidates"] += 1
        row = {"in_candidates": in_c, "planted": True, "rank_under_c6": ws.get("own_rank"), "best_other_cosine": ws.get("best_other_cosine"), "cos_query_own": ws.get("own_cosine")}
        c = miss_class(row)
        if c is None:
            out["unclassed"] += 1
        else:
            out[c] += 1
    out["label"] = ("the own line's rank and cosines under the whole store stand for the bucket probe's rank under C6; an in candidates miss is a window miss (the top k), "
                    "classed as none of the three; the cue class uses the frozen collision law at the legacy geometry (ten bits, four tables, radius one), as the adoption read it; "
                    "a miss of a refused write (the gate's) has no class here: under the identity rule on distinct texts there is none by construction")
    return out


def marks_ledger(p: dict, threshold: float) -> dict:
    """The own line marks below the threshold (a planted question whose own line is in the window with a mark under it)
    and the never planted maximal marks at or above it, per seed."""
    on = records(p, "planted")
    below = sum(1 for a in on if a.get("own_in_window") and a.get("mark_own") is not None and a["mark_own"] < threshold)
    out = {"own_line_marks_below_threshold": rate(below, sum(1 for a in on if a.get("own_in_window"))), "threshold": threshold, "never_planted_max_at_or_above": {}}
    nevers = records(p, "never_planted")
    for s in sorted({a.get("seed") for a in nevers}):
        sub = [a for a in nevers if a.get("seed") == s]
        out["never_planted_max_at_or_above"][str(s)] = rate(sum(1 for a in sub if a.get("mark_max") is not None and a["mark_max"] >= threshold), len(sub))
    out["never_planted_max_at_or_above"]["all"] = rate(sum(1 for a in nevers if a.get("mark_max") is not None and a["mark_max"] >= threshold), len(nevers))
    return out


def veto_ledger(p: dict) -> dict:
    """M4: the citations the veto refused under `mark-veto`, the own line and another line apart, and the model's
    abstentions (under `plain`) with the own line present and absent."""
    on = records(p, "planted")
    refused_own = refused_other = refused_outside = malformed = 0
    for a in on:
        if a["raw_kind"] == "cite" and a["rules"]["mark-veto"]["outcome"] == "abstain":
            labels = {w["label"]: w["pointer"] for w in a.get("window", [])}
            ptr = labels.get(a.get("raw_label"))
            if ptr is None:
                refused_outside += 1
            elif ptr == a.get("own_pointer"):
                refused_own += 1
            else:
                refused_other += 1
        elif a["raw_kind"] == "malformed":
            malformed += 1
    abst_present = sum(1 for a in on if a["raw_kind"] == "unknown" and a.get("own_in_window"))
    abst_absent = sum(1 for a in on if a["raw_kind"] == "unknown" and not a.get("own_in_window"))
    return {"n": len(on), "citations_refused_own_line": refused_own, "citations_refused_other_line": refused_other, "citations_of_a_label_outside_the_window": refused_outside,
            "malformed_outputs": malformed, "model_abstentions_own_present": abst_present, "model_abstentions_own_absent": abst_absent,
            "what": "the veto's ledger under mark-veto: the model's citations the veto refused (the own line, another line, a label outside the window apart), its malformed outputs, its abstentions with the own line present and absent; the counts add up to the planted block's non answers under mark-veto"}


def m3_exactness(recs: list[dict]) -> dict:
    """M3: under `mark-veto`, the exact value given an own line citation that survives the veto."""
    cited = [a for a in recs if a["rules"]["mark-veto"]["kind"] == "cite" and a["rules"]["mark-veto"].get("cited_pointer_is_own") is True]
    return rate(sum(1 for a in cited if a["rules"]["mark-veto"]["outcome"] == "strict"), len(cited))


# ---------------------------------------------------------------------------- #
# the near name block and the cross schema block (1 (d), 1 (e))                   #
# ---------------------------------------------------------------------------- #
def block_report(p: dict, block: str, threshold: float) -> dict:
    """Per rule the claims on a block without a planted line; for the near name block the neighbour's presence in the window
    and its mark against the threshold (1 (d)); the block enters no letter (B.2 (a))."""
    recs = records(p, block)
    out = {"n": len(recs), "by_rule": {r: claims_on(recs, r) for r in RULES}, "enters_a_letter": False}
    if block == "near_name" and recs:
        out["neighbour_in_window"] = rate(sum(1 for a in recs if a.get("neighbour_in_window")), len(recs))
        out["neighbour_mark_at_or_above_threshold"] = rate(sum(1 for a in recs if a.get("neighbour_mark_at_or_above_threshold")), len(recs))
        out["construction"] = "every key rule (V1, A0, a lexical search on the name) refuses a near name by construction; the organ's stack claims what its marks admit"
    if block == "cross_schema" and recs:
        out["own_line_in_window"] = rate(sum(1 for a in recs if a.get("own_in_window")), len(recs))
        out["construction"] = "a planted entity asked the direct query of the next schema: any claim is a claim without an answer, reported per rule, not a condition"
    return out


# ---------------------------------------------------------------------------- #
# the cells of a family                                                           #
# ---------------------------------------------------------------------------- #
def read_cell(p: dict, threshold: float) -> dict:
    """One pass (one index, one form, one checkpoint): every rule's counts and conditions, the factor product per rule,
    the misses, the marks, the veto's ledger, M3, the blocks."""
    on = records(p, "planted")
    return {"index": p["index"], "form": p["form"], "run_id": p["checkpoint"]["run_id"], "n_planted": len(on),
            "short_windows": sum(1 for a in on if a.get("short_window")),
            "rules": {r: {"counts": counts(on, r), "conditions": conditions(p, r), "factors": factors(on, r)} for r in RULES},
            "misses": (miss_classes(on) if p["index"] == "I0" else {"not_classed": "a reference pass: every live entry is a candidate, the adoption's classes are the index's"}),
            "marks": marks_ledger(p, threshold), "veto_ledger": veto_ledger(p), "m3_exactness_under_mark_veto": m3_exactness(on),
            "construction": {"mark-veto+value": ("on A every answer is an abstention by construction (the construction table, ii)" if p.get("family") == "A" else None),
                             "V1": "never planted claims are zero by construction (iii); invalid citations are zero when V1 checks the answered value",
                             "short_windows": ("the organ's buckets proposed fewer than k candidates" if p["index"] == "I0" else "zero by construction on a reference pass")},
            "blocks": {b: block_report(p, b, threshold) for b in ("never_planted", "cross_schema", "near_name") if records(p, b)},
            "sessions": p.get("run_in_sessions")}


def read_family(loaded_by_run: dict) -> dict:
    """A family over its checkpoints: the cells per index and form, with the letter's inputs under `mark-veto` on I0."""
    cells = {}
    for run_id, L in loaded_by_run.items():
        for (index, form), p in L["passes"].items():
            cells[f"{index}-{form}-{run_id}"] = read_cell(p, L["threshold"])
    return {"family": next(iter(loaded_by_run.values()))["family"], "cells": cells}


# ---------------------------------------------------------------------------- #
# A1 to A3                                                                        #
# ---------------------------------------------------------------------------- #
def attribute_cell(cell: dict, rule: str = "mark-veto") -> dict:
    """A failing cell attributed by component (A2): the organ's retrieval (the own line absent from the window), the mark
    (the own line present with a mark under the threshold, or a never planted line at or above it), the model's reading
    (the own line present and another decision taken), the copy (the own line cited and the value not exact)."""
    f = cell["rules"][rule]["factors"]; m = cell["marks"]; c = cell["rules"][rule]["conditions"]
    return {"the organ's retrieval": {"own_in_window": f["own_in_window"], "misses": cell["misses"]},
            "the mark": {"own_line_marks_below_threshold": m["own_line_marks_below_threshold"], "never_planted_max_at_or_above": m["never_planted_max_at_or_above"]},
            "the model's reading": {"own_cited_given_in_window": f["own_cited_given_in_window"], "veto_ledger": cell["veto_ledger"]},
            "the copy": {"exact_given_own_cited": f["exact_given_own_cited"]},
            "broken": c["broken"]}


def letter_a(reading_A: dict) -> dict:
    """A1, A2 or A3 from the six cells of A under `mark-veto` on I0 (form x checkpoint); A1 on one form only claimed for that
    form only; `plain` reported under every letter."""
    cells = {}
    for form in FORMS:
        for run_id in RUN_IDS:
            key = f"I0-{form}-{run_id}"
            if key in reading_A["cells"]:
                cells[key] = reading_A["cells"][key]
    holds = {k: bool(c["rules"]["mark-veto"]["conditions"]["holds"] and c["rules"]["mark-veto"]["conditions"]["holds_with_own_line_gap"]) for k, c in cells.items()}
    n_cells = len(holds); n_hold = sum(holds.values())
    by_form = {form: all(v for k, v in holds.items() if k.startswith(f"I0-{form}-")) and sum(1 for k in holds if k.startswith(f"I0-{form}-")) == 3 for form in FORMS}
    if n_cells < 6:
        letter = None
    elif n_hold == 6:
        letter = "A1"
    elif n_hold == 0:
        letter = "A3"
    else:
        letter = "A2"
    out = {"letter": letter, "cells": n_cells, "cells_holding": n_hold, "holds": holds, "by_form": by_form,
           "holds_what": "the four frozen conditions and the own line gap together, per cell (A1 asks for both)",
           "not_read": (None if n_cells == 6 else f"{6 - n_cells} of the six cells are missing: no letter until every cell is read"),
           "claimed_forms": [form for form in FORMS if by_form[form]] if letter not in ("A3", None) else [],
           "sentence": A1_SENTENCE, "label": "A.2's label: a synthetic set locked before execution, not a blind validation",
           "plain": {k: c["rules"]["plain"]["conditions"] for k, c in cells.items()},
           "failing_cells": {k: attribute_cell(c) for k, c in cells.items() if not holds[k]},
           "construction_table": {"i": OFF_ARM_SENTENCE, "ii": "under +value the organ reads no value on A's lines: every answer is an abstention on A by construction",
                                  "vii": "the organ alone (the mark oracle's citation with the organ's value) gives a strict recall of zero on A by construction"},
           "oracle_organ_value_strict_on_A": {k: c["rules"]["oracle+organ-value"]["counts"]["strict_own"] for k, c in cells.items()}}
    if letter == "A1":
        out["claim"] = "the system with the model's own value holds the episodic conditions on formulations it never saw"
    elif letter == "A2":
        out["claim"] = None; out["reported"] = "they hold on some cells and not all: attributed by component for each failing cell; not claimed"
    elif letter == "A3":
        out["claim"] = None; out["reported"] = "they hold on none: the frontier is documented by component before anything else is built"
    else:
        out["claim"] = None; out["reported"] = out["not_read"]
    return out


# ---------------------------------------------------------------------------- #
# M1 to M4                                                                        #
# ---------------------------------------------------------------------------- #
def m2_pairs(passes: list[dict]) -> dict:
    """The model's own line citation under `plain` against the mark oracle's choice, on the reads with the own line in the
    window, pooled over the passes given (the I0 passes of the two live seeds, both forms): the discordant pairs and the
    exact two sided binomial."""
    b = c = both = neither = 0
    for p in passes:
        for a in records(p, "planted"):
            if not a.get("own_in_window"):
                continue
            model = a["rules"]["plain"].get("cited_pointer_is_own") is True
            oracle = a["rules"]["oracle"].get("cited_pointer_is_own") is True
            if model and oracle:
                both += 1
            elif model:
                b += 1
            elif oracle:
                c += 1
            else:
                neither += 1
    n = b + c + both + neither
    return {"n_reads_with_own_in_window": n, "both": both, "neither": neither, "b_model_only": b, "c_oracle_only": c, "p_two_sided": binomial_two_sided_p(b, c),
            "model_rate": rate(both + b, n), "oracle_rate": rate(both + c, n)}


def discordant_needed(alpha_level: float) -> int:
    """The smallest number of discordant pairs all one way whose exact two sided binomial p is at or under the level."""
    n = 1
    while binomial_two_sided_p(n, 0) > alpha_level:
        n += 1
    return n


def m2_resolution_sentence(m: int, pooled_reads: dict) -> str:
    """M2's own resolution (the primary tests' sentence is written for Holm over two on four hundred pairs): under Holm over
    m families the first rejection needs the smallest p at or under alpha / m, so at least n1 discordant pairs all one way,
    the last at least n_last."""
    n_first = discordant_needed(ALPHA / m); n_last = discordant_needed(ALPHA)
    reads = ", ".join(f"{fam}: {v}" for fam, v in pooled_reads.items())
    return (f"under Holm over the {m} families the first rejection needs the smallest p at or under {ALPHA / m:.4g}: at least {n_first} discordant pairs all one way; "
            f"the last rejection {n_last} (a single test at {ALPHA}); so a difference of fewer discordant reads among the pooled reads with the own line in the window ({reads}) is invisible to the family")


def a0_ceiling_of(md: Path, fam: str, run_ids=LIVE_RUN_IDS) -> dict | None:
    """A0's files on the family's I0 passes of the live seeds (`a0-sealed-<family>-<run id>-I0-<form>.pass.json`, written
    by `cortex_c2b.a0`), pooled: the bag share of the own line on top on the planted block and A0's letter per file."""
    files = {}
    for r in run_ids:
        for form in FORMS:
            f = md / f"a0-sealed-{fam}-{r}-I0-{form}.pass.json"
            if f.exists():
                files[f"{r}-{form}"] = load_json(f)
    if not files:
        return None
    k = sum(((v["blocks"].get("planted") or {}).get("bag") or {}).get("top_is_own", 0) for v in files.values())
    n = sum((v["blocks"].get("planted") or {}).get("n_with_own", 0) for v in files.values())
    return {"files": sorted(files), "bag_top_is_own_pooled": rate(k, n), "letters": {name: v.get("letter") for name, v in files.items()},
            "what": "the construction's ceiling under byte keys (ADR-008 part A; ADR-010 E), read from A0's files on the I0 passes; beside M2, enters no letter"}


def read_m(readings: dict, loaded: dict, a0: dict | None = None, metrics_dir: str | Path | None = None) -> dict:
    """The main risk's four numbers per family: M1 `plain` against the conditions per cell; M2 pooled over the live seeds and
    both forms, Holm over the families at 0.05, the third checkpoint reported, A0's ceiling beside (from A0's files when
    present); M3 and M4 per cell."""
    out = {"families": {}, "holm": None}
    pvalues = {}; pooled_reads = {}
    for fam in FAMILIES:
        if fam not in loaded:
            continue
        live = [loaded[fam][r]["passes"][("I0", form)] for r in LIVE_RUN_IDS for form in FORMS if r in loaded[fam] and ("I0", form) in loaded[fam][r]["passes"]]
        third = [loaded[fam][CONCEIVING_RUN_ID]["passes"][("I0", form)] for form in FORMS if CONCEIVING_RUN_ID in loaded[fam] and ("I0", form) in loaded[fam][CONCEIVING_RUN_ID]["passes"]]
        m2 = m2_pairs(live) if live else None
        if m2 is not None:
            pvalues[fam] = m2["p_two_sided"]; pooled_reads[fam] = m2["n_reads_with_own_in_window"]
        cells = readings[fam]["cells"]
        ceiling = (a0 or {}).get(fam)
        if ceiling is None and metrics_dir is not None:
            ceiling = a0_ceiling_of(Path(metrics_dir), fam)
        out["families"][fam] = {"M1_plain_conditions": {k: c["rules"]["plain"]["conditions"] for k, c in cells.items() if k.startswith("I0-")},
                                "M2": {"pooled_live_seeds": m2, "third_checkpoint": (m2_pairs(third) if third else None), "a0_ceiling": ceiling,
                                       "oracle_what": "the frozen mark_oracle's choice: the highest marked line strictly above the threshold (an abstention under it counts against the oracle)"},
                                "M3_exactness_given_own_citation_mark_veto": {k: c["m3_exactness_under_mark_veto"] for k, c in cells.items() if k.startswith("I0-")},
                                "M4_veto_ledger": {k: c["veto_ledger"] for k, c in cells.items() if k.startswith("I0-")}}
    if pvalues:
        rej = holm(pvalues, ALPHA)
        out["holm"] = {"pvalues": pvalues, "rejected": rej, "alpha": ALPHA, "m": len(pvalues)}
        for fam in pvalues:
            m2 = out["families"][fam]["M2"]["pooled_live_seeds"]
            direction = ("model" if m2["b_model_only"] > m2["c_oracle_only"] else ("oracle" if m2["c_oracle_only"] > m2["b_model_only"] else "tie"))
            out["families"][fam]["M2"]["separated_under_holm"] = rej[fam]
            out["families"][fam]["M2"]["direction"] = direction
            out["families"][fam]["M2"]["sentence"] = M2_SENTENCES["model" if (rej[fam] and direction == "model") else "organ"]
            out["families"][fam]["M2"]["sentence_what"] = ("the model selects better, separated under Holm" if (rej[fam] and direction == "model")
                                                            else ("the oracle selects better, separated under Holm: the organ's sentence" if rej[fam]
                                                                  else "not separated at this size: the organ's sentence, with the resolution"))
            if not rej[fam]:
                out["families"][fam]["M2"]["resolution"] = m2_resolution_sentence(len(pvalues), pooled_reads)
    return out


# ---------------------------------------------------------------------------- #
# B.2: the primary tests on A and R-a, R-b, R-c                                   #
# ---------------------------------------------------------------------------- #
def strict_own_by_index(loaded_A_run: dict, index: str, rule: str, form: str = "direct") -> dict[int, bool]:
    p = loaded_A_run["passes"][(index, form)]
    return {a["index"]: (a["rules"][rule]["outcome"] == "strict" and a["rules"][rule].get("cited_pointer_is_own") is True) for a in records(p, "planted")}


def primary_test(loaded_A: dict, buyer: tuple[str, str]) -> dict:
    """(I0, mark-veto) against a buyer stack on strict recall on the own line, the direct form, the two live seeds pooled
    (400 pairs by question index), an exact two sided binomial on the discordant pairs; the third checkpoint's sign."""
    def pairs(run_ids):
        b = c = both = neither = 0
        for r in run_ids:
            if r not in loaded_A:
                continue
            organ = strict_own_by_index(loaded_A[r], *ORGAN_STACK)
            ref = strict_own_by_index(loaded_A[r], *buyer)
            for i in sorted(organ):
                o, x = organ[i], ref.get(i, False)
                if o and x:
                    both += 1
                elif o:
                    b += 1
                elif x:
                    c += 1
                else:
                    neither += 1
        return {"pairs": b + c + both + neither, "both": both, "neither": neither, "b_organ_only": b, "c_reference_only": c, "p_two_sided": binomial_two_sided_p(b, c),
                "organ_strict_own": both + b, "reference_strict_own": both + c}
    live = pairs(LIVE_RUN_IDS); third = pairs((CONCEIVING_RUN_ID,))
    sign = lambda t: (1 if t["b_organ_only"] > t["c_reference_only"] else (-1 if t["c_reference_only"] > t["b_organ_only"] else 0))
    return {"buyer": list(buyer), "organ": list(ORGAN_STACK), "live_seeds_pooled": live, "third_checkpoint": third,
            "sign_live": sign(live), "sign_third": sign(third), "third_agrees": sign(live) != 0 and sign(live) == sign(third)}


def primary_tests(loaded_A: dict) -> dict:
    tests = {name: primary_test(loaded_A, buyer) for name, buyer in PRIMARY_TESTS.items()}
    rej = holm({name: t["live_seeds_pooled"]["p_two_sided"] for name, t in tests.items()}, ALPHA)
    for name, t in tests.items():
        t["rejected_under_holm"] = rej[name]
        if rej[name] and t["third_agrees"] and t["sign_live"] > 0:
            t["outcome"] = OUTCOMES_PRIMARY[0]
        elif rej[name] and t["third_agrees"] and t["sign_live"] < 0:
            t["outcome"] = OUTCOMES_PRIMARY[1]
        else:
            t["outcome"] = OUTCOMES_PRIMARY[2]
            t["resolution"] = RESOLUTION_SENTENCE
            if rej[name] and not t["third_agrees"]:
                t["note"] = "rejected under Holm on the live seeds, the third checkpoint not agreeing in sign: no claim"
    return {"tests": tests, "holm_alpha": ALPHA, "form": "direct", "what": "strict recall on the own line"}


def stack_conditions(loaded_A: dict, index: str, rule: str) -> dict:
    """The four conditions of a configuration on the direct form, per checkpoint."""
    return {r: conditions(loaded_A[r]["passes"][(index, "direct")], rule)["holds"] for r in RUN_IDS if r in loaded_A and (index, "direct") in loaded_A[r]["passes"]}


def letter_r(loaded_A: dict) -> dict:
    """R-a, R-b or R-c from the primary tests and the conditions on the direct form."""
    tests = primary_tests(loaded_A)
    organ_c = stack_conditions(loaded_A, *ORGAN_STACK)
    buyers_c = {f"{i} {r}": stack_conditions(loaded_A, i, r) for i, r in BUYER_STACKS}
    organ_live = all(organ_c.get(r, False) for r in LIVE_RUN_IDS)
    organ_any = any(organ_c.get(r, False) for r in LIVE_RUN_IDS)
    favours_organ = [t["outcome"] == OUTCOMES_PRIMARY[0] for t in tests["tests"].values()]
    each_buyer_fails = all(any(not bc.get(r, False) for r in LIVE_RUN_IDS) for bc in buyers_c.values())
    buyer_holds_where_organ = any(all(bc.get(r, False) for r in LIVE_RUN_IDS if organ_c.get(r, False)) for bc in buyers_c.values())
    note = None
    if organ_live and each_buyer_fails and all(favours_organ):
        letter = "R-a"
        sentence = "the organ's retrieval and veto do what the simpler stack does not, at the cost named"
    elif buyer_holds_where_organ and not any(favours_organ):
        letter = "R-b"
        sentence = ("the read side brings no measured advantage over the classical stack on these families; the write side, the persistence across "
                    "the session boundary, the provenance and the withdrawal were compared with no reference, so no advantage is claimed for them "
                    "either, and the industrial door's first item is answered against the organ")
        if not organ_any:
            note = "the organ's stack holds the four conditions on no live seed: R-b holds by vacuity of its first clause; reported as such"
    else:
        letter = "R-c"
        sentence = "every other outcome, attributed by component"
    return {"letter": letter, "sentence": sentence, "note": note, "primary_tests": tests, "conditions": {"organ": organ_c, "buyers": buyers_c},
            "conditions_what": "the four frozen conditions (the frozen gap, any line) on the direct form per checkpoint; the own line gap is beside in every cell",
            "attribution": attribution_by_component(loaded_A), "near_name_block_enters": False}


def attribution_by_component(loaded_A: dict) -> dict:
    """(R-c)'s attribution: the index at a fixed rule on the own line in the window and the own citation; the rule at a fixed
    index on invalid citations and never planted claims; the model against the oracle at a fixed index."""
    out = {"index_at_fixed_rule": {}, "rule_at_fixed_index": {}, "model_against_oracle_at_fixed_index": {}}
    for r in RUN_IDS:
        if r not in loaded_A:
            continue
        L = loaded_A[r]
        for rule in ("mark-veto", "V1"):
            for index in INDEXES:
                p = L["passes"].get((index, "direct"))
                if p is None:
                    continue
                f = factors(records(p, "planted"), rule)
                out["index_at_fixed_rule"][f"{rule}-{index}-{r}"] = {"own_in_window": f["own_in_window"], "own_cited_given_in_window": f["own_cited_given_in_window"]}
        for index in INDEXES:
            p = L["passes"].get((index, "direct"))
            if p is None:
                continue
            for rule in RULES:
                c = conditions(p, rule)
                out["rule_at_fixed_index"][f"{index}-{rule}-{r}"] = {"invalid_citations": c["invalid_citations"], "never_planted_claims": c["never_planted"]["all"]["claim_rate"]}
            out["model_against_oracle_at_fixed_index"][f"{index}-{r}"] = m2_pairs([p])
    return out


# ---------------------------------------------------------------------------- #
# the first family (row 47's plain files; the paraphrase pass when run)            #
# ---------------------------------------------------------------------------- #
def first_family_direct(row47: dict) -> dict:
    """The first family's direct form from row 47's plain file: `plain` as recorded, `mark-veto` by the frozen replay; the
    factor product under `plain` (the own line cited is the file's own field)."""
    thr = float(row47["hm_mark_oracle"]["threshold"])
    on = row47["answers"]["on"]; neg = row47["answers"]["negctrl"]
    veto = replay_outcomes(row47, "mark-veto", thr)
    n = len(on)
    in_w = [a for a in on if a.get("own_in_window")]
    cited = [a for a in in_w if a.get("cited_pointer_is_own")]
    exact = [a for a in cited if a["outcome"] == "strict"]
    strict_plain = sum(1 for a in on if a["outcome"] == "strict" and a.get("cited_pointer_is_own"))
    strict_veto = sum(1 for a, o in zip(on, veto["on"]) if o == "strict" and a.get("cited_pointer_is_own"))
    return {"run_id": row47["checkpoint"]["run_id"], "threshold": thr, "n": n, "label": FIRST_FAMILY_LABEL,
            "factors_plain": {"own_in_window": rate(len(in_w), n), "own_cited_given_in_window": rate(len(cited), len(in_w)), "exact_given_own_cited": rate(len(exact), len(cited)),
                              "product": (len(exact) / n if n else None), "strict_own": rate(len(exact), n)},
            "plain": {"strict_recall_own": rate(strict_plain, n), "invalid": rate(sum(1 for a in on if a["outcome"] == "invalid"), n),
                      "never_planted_claims": rate(sum(1 for a in neg if a["outcome"] != "abstain"), len(neg))},
            "mark_veto_replayed": {"strict_recall_own": rate(strict_veto, n), "invalid": rate(sum(1 for o in veto["on"] if o == "invalid"), n),
                                   "never_planted_claims": rate(sum(1 for o in veto["negctrl"] if o != "abstain"), len(veto["negctrl"])),
                                   "by": "the frozen replay_outcomes (hm_lm), the policy that failed the invalid citation condition on both live seeds (rows 37 and 39)"}}


def first_family_paraphrase(p: dict) -> dict:
    """The first family's paraphrase pass (A.6: run once on the three checkpoints by `--family first`): `plain` and
    `mark-veto` as the pass records them (the same records as the sealed families', the seven rules replayed), the factor
    product under `plain`, printed beside the direct form's (row 47's file replayed)."""
    on = records(p, "planted")
    return {"run_id": p["checkpoint"]["run_id"], "n": len(on), "label": FIRST_FAMILY_LABEL, "admission": p.get("admission"),
            "plain": {"strict_recall_own": counts(on, "plain")["strict_recall_own"], "invalid": counts(on, "plain")["invalid_rate"],
                      "never_planted_claims": claims_on(records(p, "never_planted"), "plain")["claim_rate"]},
            "mark_veto": {"strict_recall_own": counts(on, "mark-veto")["strict_recall_own"], "invalid": counts(on, "mark-veto")["invalid_rate"],
                          "never_planted_claims": claims_on(records(p, "never_planted"), "mark-veto")["claim_rate"], "conditions": conditions(p, "mark-veto")},
            "factors_plain": factors(on, "plain"),
            "like_with_like": ("the direct form comes from row 47's plain file (cited_pointer_is_own and outcome under plain; mark-veto by the frozen replay_outcomes); "
                               "the paraphrase form from a pass of this build (the seven rules replayed on the recorded output): the same quantities, two writers")}


def factor_product_reading(readings: dict, loaded: dict, first: dict) -> dict:
    """The factor product for A, Av, T0 and the first family per live seed under `plain` on I0 direct, with the differences
    A against Av, Av against the first family, T0 against the first family, per factor, and the transfer reading."""
    out = {"per_run": {}, "differences": {}, "label": DIFFERENCE_INTERVAL_LABEL}
    pairs = (("A", "Av"), ("Av", "first"), ("T0", "first"))
    for r in RUN_IDS:
        fr = {}
        for fam in FAMILIES:
            L = loaded.get(fam, {}).get(r)
            if L is not None and ("I0", "direct") in L["passes"]:
                fr[fam] = factors(records(L["passes"][("I0", "direct")], "planted"), "plain")
        if r in first:
            fr["first"] = first[r]["factors_plain"]
        out["per_run"][r] = fr
    for a, b in pairs:
        diffs = {r: factor_differences(out["per_run"][r][a], out["per_run"][r][b]) for r in RUN_IDS if a in out["per_run"].get(r, {}) and b in out["per_run"].get(r, {})}
        out["differences"][f"{a} against {b}"] = {"by_run": diffs, "transfer": transfer_reading(diffs)}
    return out


# ---------------------------------------------------------------------------- #
# the whole reading                                                                #
# ---------------------------------------------------------------------------- #
def read_rows(metrics_dir: str | Path, a0: dict | None = None) -> dict:
    """Rows 51 and 52 from the files in `metrics_dir`: every family present over its checkpoints, the letters, M1 to M4,
    the primary tests and R, the factor product, the blocks, the first family beside."""
    md = Path(metrics_dir)
    loaded: dict[str, dict] = {}
    for fam in FAMILIES:
        for r in RUN_IDS:
            if (md / f"sealed-{fam}-{r}.json").exists():
                loaded.setdefault(fam, {})[r] = load_family(md, fam, r)
    readings = {fam: read_family(by_run) for fam, by_run in loaded.items()}
    first = {}
    first_para = {}
    for r in RUN_IDS:
        f47 = md / f"hm-lm-{r}-session-b-index16exact.json"
        if f47.exists():
            first[r] = first_family_direct(load_json(f47))
        fp = md / f"sealed-first-{r}-I0-paraphrase.pass.json"
        if fp.exists():
            first_para[r] = first_family_paraphrase(load_json(fp))
    out = {"benchmark": "the cold reading of rows 51 and 52 (ADR-010 A.6, A.7, B.2; the amendment of 2026-10-15 on parts B and C)",
           "files": {fam: {r: {"path": L["path"], "sha256_lf": RS.sha256_file(L["path"], lf=True)} for r, L in by_run.items()} for fam, by_run in loaded.items()},
           "readings": readings,
           "letter_A": (letter_a(readings["A"]) if "A" in readings else {"letter": None, "reported": "the family A is not read: files missing"}),
           "M": read_m(readings, loaded, a0, md),
           "R": (letter_r(loaded["A"]) if "A" in loaded and any(r in loaded["A"] for r in LIVE_RUN_IDS) else {"letter": None, "reported": "A's live seeds are not read"}),
           "factor_product": factor_product_reading(readings, loaded, first),
           "first_family": {"direct": first, "paraphrase": first_para,
                            "what": "the direct form from row 47's plain files (mark-veto by the frozen replay); the paraphrase form from the passes `--family first` when run; both printed beside A's"},
           "near_name_block": {fam: {k: c["blocks"].get("near_name") for k, c in rd["cells"].items() if c["blocks"].get("near_name")} for fam, rd in readings.items()},
           "cross_schema_block": {fam: {k: c["blocks"].get("cross_schema") for k, c in rd["cells"].items() if c["blocks"].get("cross_schema")} for fam, rd in readings.items()},
           "T0_own_in_window": {k: c["rules"]["plain"]["factors"]["own_in_window"] for k, c in readings.get("T0", {"cells": {}})["cells"].items()},
           "declared": "A.7 and B.2 of 2026-10-14; B.2 (a) and 1 (d) of the amendment",
           "departures_stated": {"difference_interval": DIFFERENCE_INTERVAL_LABEL + " (A.6 says 'exact 95 % interval'; stated in ADR-010's built paragraph of 2026-10-30)",
                                 "conditions": "the frozen four conditions decide the R letters; letter A asks for the four and the own line gap together (A.6)",
                                 "factor_product_rule": "the factor product is read under plain on I0, the direct form (A.6 names no rule or form): a choice of the build, stated"},
           "timestamp_utc": RS.utc_now()}
    return out


def summary_lines(reading: dict) -> list[str]:
    lines = []
    la = reading.get("letter_A") or {}
    lines.append(f"A: {la.get('letter')} ({la.get('cells_holding')} of {la.get('cells')} cells under mark-veto on I0); claimed forms {la.get('claimed_forms')}")
    for fam, m in (reading.get("M") or {}).get("families", {}).items():
        m2 = m["M2"]["pooled_live_seeds"] or {}
        lines.append(f"M2 {fam}: b {m2.get('b_model_only')} c {m2.get('c_oracle_only')} p {m2.get('p_two_sided')}: {m['M2'].get('sentence')}")
    R = reading.get("R") or {}
    lines.append(f"R: {R.get('letter')}; " + "; ".join(f"{k}: {t['outcome']}" for k, t in (R.get("primary_tests") or {}).get("tests", {}).items()))
    return lines


def _cli(argv=None):
    ap = argparse.ArgumentParser(description="the cold reading of rows 51 and 52 (a new module; files only)")
    ap.add_argument("--metrics", default="metrics/mqar")
    ap.add_argument("--out", default=None, help="the reading file (default: <metrics>/sealed-reading-rows51-52.json)")
    ap.add_argument("--a0", default=None, help="a JSON file {family: A0's letter and counts} when A0 was run on the I0 passes")
    args = ap.parse_args(argv)
    a0 = load_json(args.a0) if args.a0 else None
    reading = read_rows(args.metrics, a0)
    out = Path(args.out) if args.out else Path(args.metrics) / "sealed-reading-rows51-52.json"
    if out.exists():
        raise SystemExit(f"sealed reading: {out} exists; a file of the record is never overwritten")
    sha = RS.write_json(out, reading)
    print(f"retained: {out}  sha256 {sha}")
    for l in summary_lines(reading):
        print(l)


if __name__ == "__main__":
    _cli()
