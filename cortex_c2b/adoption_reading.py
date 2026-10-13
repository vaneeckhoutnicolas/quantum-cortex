"""cortex_c2b.adoption_reading: the cold reading of the adoption row (ADR-008, amendment of 2026-10-13), from the
session B files alone, in the order the amendment wrote before any number.

The files: for each of the three v9 checkpoints (seeds 1337, 2024, 7), the control replay under the legacy geometry
and the replay under the adopted one (`--index 16,exact`), each under `plain` and `mark-veto+value`: twelve files.
The reading is written as functions of counts and decided here, not in prose:

  H, the hit: H1 when the own line is in the live window on every question of the three checkpoints under the adopted
     geometry (600 of 600 on the record's files); H0 otherwise, attributed to the live signatures against the probe's,
     never to the window (the probe ranked the own line first on 600 of 600 candidate sets).
  S, the conditions: S1 when, on each of the two live seeds (2024 and 7), the four frozen conditions hold under the
     adopted geometry under `mark-veto+value`; S0 otherwise, the condition named.
  M, the strict recall moved: on the questions of the two live seeds under `mark-veto+value`, the discordant pairs
     (strict under one geometry and not under the other) as b (adopted only) and c (control only); M+ when the exact
     two sided binomial at one half rejects equality at alpha 0.05 in favour of the adopted geometry, M- against it,
     M0 otherwise. Seed 1337, the seed that conceived the policy, is reported in every count and counted in none.

The paired counts (`paired_counts`) and the vetoes (`vetoes`) recompute from the per question answers, which the
record test of the row reads again. `read_draws` reads the replication at index seeds 1 and 2 from the bucket
probe's files against the collision law's expectation, written before the run from row 45's recorded cosines.
"""
from __future__ import annotations

import math
from pathlib import Path

from cortex_c2b.hm_protocol import DELTA, NEGCTRL_INVALID, FLOOR_MARGIN, SCHEMAS, _ATTRS
from cortex_c2b.hm_lm import INVALID_CITATION_MAX

LIVE_SEEDS = {"4cfd6f6e1406": 2024, "2e1dc71913b1": 7}        # counted: the seeds that did not conceive the policy (rows 37, 39)
CONCEIVING_SEED = {"8ceba5db8d0b": 1337}                      # reported, counted nowhere (rows 36, 37, 39 say so)
RUN_IDS = tuple(CONCEIVING_SEED) + tuple(LIVE_SEEDS)
SYSTEM_POLICY = "mark-veto+value"
ALPHA = 0.05
OUTCOMES = ("strict", "valid", "invalid", "abstain")
# the collision law's expectation (Charikar 2002; sixteen tables of ten bits) from the own cosines recorded in row 45's
# files, written before the run (ADR-008, 2026-10-13, (ii)); `expected_misses_from_record` recomputes it from those files
EXPECTED_MISSES = {"8ceba5db8d0b": 0.01046, "4cfd6f6e1406": 0.01547, "2e1dc71913b1": 0.00660}
N_BITS, N_TABLES = 10, 16


# ---- the exact two sided binomial at one half -------------------------------------------------------
def binomial_two_sided_p(b: int, c: int) -> float:
    """The chance, under equal recall, of a split at least as unbalanced as (b, c) among b + c discordant pairs:
    twice the lower tail at one half, capped at one; 1.0 when nothing is discordant."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    lower = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2.0 * lower)


def read_m(b: int, c: int) -> str:
    p = binomial_two_sided_p(b, c)
    if p < ALPHA and b > c:
        return "M+"
    if p < ALPHA and c > b:
        return "M-"
    return "M0"


# ---- the paired counts from two files of one checkpoint and one policy -------------------------------
def _check_pair(control: dict, adopted: dict) -> None:
    if control["checkpoint"]["run_id"] != adopted["checkpoint"]["run_id"]:
        raise ValueError("pair: the two files are not of the same checkpoint")
    if control.get("policy", "plain") != adopted.get("policy", "plain"):
        raise ValueError("pair: the two files are not under the same policy")
    for arm in ("on", "negctrl"):
        ec = [a["entity"] for a in control["answers"][arm]]; ea = [a["entity"] for a in adopted["answers"][arm]]
        if ec != ea:
            raise ValueError(f"pair: the {arm} questions are not the same in the same order")


def paired_counts(control: dict, adopted: dict) -> dict:
    """The declared paired counts of (i), the question (entity, attribute) as the unit: the own line in the window gained
    or lost; the four by four table of outcomes (control rows, adopted columns); the strict discordant pairs b and c
    and their split by whether the own line was newly in the window; the citations whose label or own pointer status
    changed; the never planted transitions, claims and windows at or above the oracle's threshold of each file."""
    _check_pair(control, adopted)
    on_c, on_a = control["answers"]["on"], adopted["answers"]["on"]
    table = {o: {q: 0 for q in OUTCOMES} for o in OUTCOMES}
    gained = lost = b = c = 0
    # every discordant pair split by the own line's presence in the window under (control, adopted): newly present (absent,
    # present), already present (present, present), lost (present, absent: the loss that contradicts the probe), absent in both
    split = {f"{side}_{w}": 0 for side in ("b", "c") for w in ("newly_present", "already_present", "lost", "absent_both")}
    cites_changed = 0           # questions cited under at least one geometry whose label or own pointer status differs (a citation that
    for x, y in zip(on_c, on_a):   # becomes an abstention counts: its label goes to None)
        table[x["outcome"]][y["outcome"]] += 1
        wc, wa = bool(x.get("own_in_window")), bool(y.get("own_in_window"))
        gained += int(wa and not wc); lost += int(wc and not wa)
        where = "newly_present" if (wa and not wc) else "already_present" if (wa and wc) else "lost" if (wc and not wa) else "absent_both"
        sx, sy = x["outcome"] == "strict", y["outcome"] == "strict"
        if sy and not sx:
            b += 1; split[f"b_{where}"] += 1
        if sx and not sy:
            c += 1; split[f"c_{where}"] += 1
        if x.get("kind") == "cite" or y.get("kind") == "cite":
            if (x.get("label"), x.get("cited_pointer_is_own")) != (y.get("label"), y.get("cited_pointer_is_own")):
                cites_changed += 1
    neg_c, neg_a = control["answers"]["negctrl"], adopted["answers"]["negctrl"]
    ntable = {o: {q: 0 for q in OUTCOMES} for o in OUTCOMES}
    for x, y in zip(neg_c, neg_a):
        ntable[x["outcome"]][y["outcome"]] += 1
    thr_c, thr_a = control["hm_mark_oracle"]["threshold"], adopted["hm_mark_oracle"]["threshold"]

    def at_or_above(arm, thr):
        return sum(1 for a in arm if a.get("mark_max") is not None and a["mark_max"] >= thr)

    return {"run_id": control["checkpoint"]["run_id"], "policy": control.get("policy", "plain"), "n": len(on_c),
            "own_in_window": {"control": sum(1 for a in on_c if a.get("own_in_window")), "adopted": sum(1 for a in on_a if a.get("own_in_window")),
                              "gained": gained, "lost": lost},
            "strict": {"control": sum(1 for a in on_c if a["outcome"] == "strict"), "adopted": sum(1 for a in on_a if a["outcome"] == "strict"),
                       "b_adopted_only": b, "c_control_only": c},
            "discordant_by_window": split, "outcomes": table, "citations_changed": cites_changed,
            "never_planted": {"n": len(neg_c), "outcomes": ntable,
                              "claims": {"control": sum(1 for a in neg_c if a["outcome"] != "abstain"), "adopted": sum(1 for a in neg_a if a["outcome"] != "abstain")},
                              "windows_at_or_above_threshold": {"control": at_or_above(neg_c, thr_c), "adopted": at_or_above(neg_a, thr_a)},
                              "threshold": {"control": thr_c, "adopted": thr_a}}}


def vetoes(plain: dict, veto: dict) -> dict:
    """The vetoes under one geometry: a citation under `plain` (strict, valid or invalid) that `mark-veto+value` turns into
    an abstention, on the planted questions and on the never planted entities."""
    if plain["checkpoint"]["run_id"] != veto["checkpoint"]["run_id"]:
        raise ValueError("vetoes: the two files are not of the same checkpoint")
    out = {}
    for arm in ("on", "negctrl"):
        out[arm] = sum(1 for x, y in zip(plain["answers"][arm], veto["answers"][arm])
                       if x.get("kind") == "cite" and x["outcome"] in ("strict", "valid", "invalid") and y["outcome"] == "abstain")
    return out


# ---- the reading H, S, M ------------------------------------------------------------------------------------
def _floor(f: dict) -> float:
    if f.get("thresholds", {}).get("floor") is not None:
        return float(f["thresholds"]["floor"])
    chance = sum(1.0 / len(_ATTRS[s]) for s in SCHEMAS) / len(SCHEMAS)
    return chance + FLOOR_MARGIN


def broken_conditions(f: dict) -> list[str]:
    """The frozen conditions a session B file breaks, named, from its rates and the gate's constants (the file's own
    `gate_conditions_without_skill_arm` is the same test as one integer)."""
    out = []
    if f["hm_invalid_citation_on"] > INVALID_CITATION_MAX:
        out.append(f"invalid citations {f['hm_invalid_citation_on']} above {INVALID_CITATION_MAX}")
    if f["hm_negctrl_rate"] >= NEGCTRL_INVALID:
        out.append(f"claims on never planted entities {f['hm_negctrl_rate']} at or above {NEGCTRL_INVALID}")
    if f["hm_gap"] < DELTA:
        out.append(f"gap {f['hm_gap']} under {DELTA}")
    if f["hm_recall_off"] > _floor(f):
        out.append(f"off arm recall {f['hm_recall_off']} above the floor {_floor(f)}")
    return out


CONSEQUENCES = {
    "H0": ("the own line is absent from a live window although the probe ranked it first on every candidate set: attributed by "
           "recomputing the probe's signatures on the recorded cues against the recorded live signatures, before any other reading; "
           "the adoption stands if the live signatures explain the miss and the probe's do not, else the probe is wrong and the amendment stops"),
    "S1 M+": "the adoption stands; the system's numbers under the adopted geometry are the new rows' numbers, rows 36, 37 and 39 keep theirs and say so; M+ is a paired result on the live seeds' questions, claimed at the singular until the holdout",
    "S1 M0": "the adoption stands; the system's numbers under the adopted geometry are the new rows' numbers, rows 36, 37 and 39 keep theirs and say so; the strict recall is not shown to move at this size",
    "S1 M-": ("the own line in every window and the system answering less: attributed per discordant question (the own line present and another "
              "decision taken, or the veto turning a citation into an abstention, split by whether the own line was newly present); the adoption "
              "stands for the organ, the decision is the window's or the policy's and is the next declared variable"),
    "S0": ("a frozen condition breaks under the adopted geometry on a live seed: attributed per line (which line was cited above the threshold; "
           "the margin fallen); the adoption stands for the organ, the threshold is the next declared variable, not tuned here"),
}


def read_adoption(files: dict) -> dict:
    """`files`: {(run_id, "legacy" | "adopted", policy): the session B file as a dict} for the three run ids and the two
    policies. Returns the letters H, S, M with their counts, the cell and its consequence, the paired counts per
    checkpoint and policy, and the vetoes under each geometry with their difference."""
    need = [(r, g, pol) for r in RUN_IDS for g in ("legacy", "adopted") for pol in ("plain", SYSTEM_POLICY)]
    missing = [k for k in need if k not in files]
    if missing:
        raise ValueError(f"read_adoption: files missing for {missing}")
    for r in RUN_IDS:
        for pol in ("plain", SYSTEM_POLICY):
            _check_pair(files[(r, "legacy", pol)], files[(r, "adopted", pol)])
    # H on the three checkpoints under the adopted geometry, read from the system policy's files; the reads are the same under both
    # policies by construction, so the plain files must show the same own line presence on every question, else the files are defective
    for r in RUN_IDS:
        for g in ("legacy", "adopted"):
            wp = [bool(a.get("own_in_window")) for a in files[(r, g, "plain")]["answers"]["on"]]
            ws = [bool(a.get("own_in_window")) for a in files[(r, g, SYSTEM_POLICY)]["answers"]["on"]]
            if wp != ws:
                raise ValueError(f"read_adoption: the plain and {SYSTEM_POLICY} files of {r} under the {g} geometry show different windows on "
                                 f"{sum(1 for p_, s_ in zip(wp, ws) if p_ != s_)} questions; the reads are the same by construction, the files are not")
            jf = files[(r, g, SYSTEM_POLICY)].get("journal", {})
            if jf.get("planted_found") is not None and jf["planted_found"] != len(ws):
                raise ValueError(f"read_adoption: {r} under {g}: {jf['planted_found']} planted facts found against {len(ws)} questions")
    in_window = {pol: sum(sum(1 for a in files[(r, "adopted", pol)]["answers"]["on"] if a.get("own_in_window")) for r in RUN_IDS) for pol in ("plain", SYSTEM_POLICY)}
    questions = sum(len(files[(r, "adopted", SYSTEM_POLICY)]["answers"]["on"]) for r in RUN_IDS)
    H = {"letter": "H1" if in_window[SYSTEM_POLICY] == questions else "H0", "in_window": in_window[SYSTEM_POLICY], "questions": questions,
         "in_window_under_plain": in_window["plain"], "per_checkpoint": {r: sum(1 for a in files[(r, "adopted", SYSTEM_POLICY)]["answers"]["on"] if a.get("own_in_window")) for r in RUN_IDS},
         "misses": {r: [a["entity"] for a in files[(r, "adopted", SYSTEM_POLICY)]["answers"]["on"] if not a.get("own_in_window")] for r in RUN_IDS}}
    # S on the two live seeds under the system policy and the adopted geometry
    per_seed, broken = {}, {}
    for r in LIVE_SEEDS:
        f = files[(r, "adopted", SYSTEM_POLICY)]
        per_seed[r] = {"seed": LIVE_SEEDS[r], "holds": bool(f["gate_conditions_without_skill_arm"]), "broken": broken_conditions(f),
                       "legacy_holds": bool(files[(r, "legacy", SYSTEM_POLICY)]["gate_conditions_without_skill_arm"])}
        if not per_seed[r]["holds"]:
            broken[r] = per_seed[r]["broken"]
    S = {"letter": "S1" if all(v["holds"] for v in per_seed.values()) else "S0", "per_seed": per_seed, "broken": broken,
         "conceiving_seed_reported": {r: {"holds": bool(files[(r, "adopted", SYSTEM_POLICY)]["gate_conditions_without_skill_arm"]),
                                         "broken": broken_conditions(files[(r, "adopted", SYSTEM_POLICY)])} for r in CONCEIVING_SEED}}
    # M on the live seeds' questions under the system policy, pooled, with the per seed counts; plain reported beside
    paired = {r: {pol: paired_counts(files[(r, "legacy", pol)], files[(r, "adopted", pol)]) for pol in ("plain", SYSTEM_POLICY)} for r in RUN_IDS}
    M = {}
    for pol in (SYSTEM_POLICY, "plain"):
        b = sum(paired[r][pol]["strict"]["b_adopted_only"] for r in LIVE_SEEDS)
        c = sum(paired[r][pol]["strict"]["c_control_only"] for r in LIVE_SEEDS)
        M[pol] = {"b": b, "c": c, "p": binomial_two_sided_p(b, c), "letter": read_m(b, c),
                  "per_seed": {r: {"b": paired[r][pol]["strict"]["b_adopted_only"], "c": paired[r][pol]["strict"]["c_control_only"]} for r in LIVE_SEEDS},
                  "conceiving_seed_reported": {r: {"b": paired[r][pol]["strict"]["b_adopted_only"], "c": paired[r][pol]["strict"]["c_control_only"]} for r in CONCEIVING_SEED}}
    Mread = {"policy": SYSTEM_POLICY, **M[SYSTEM_POLICY], "plain_reported": M["plain"], "alpha": ALPHA}
    vet = {}
    for r in RUN_IDS:
        lg, ad = vetoes(files[(r, "legacy", "plain")], files[(r, "legacy", SYSTEM_POLICY)]), vetoes(files[(r, "adopted", "plain")], files[(r, "adopted", SYSTEM_POLICY)])
        vet[r] = {"legacy": lg, "adopted": ad, "difference": {arm: ad[arm] - lg[arm] for arm in lg}}
    cell = f"{H['letter']} {S['letter']} {Mread['letter']}"
    if H["letter"] == "H0":
        consequence = CONSEQUENCES["H0"]
    else:
        parts = []
        if S["letter"] == "S0":
            parts.append(CONSEQUENCES["S0"])
            parts.append(CONSEQUENCES["S1 " + Mread["letter"]].split(";", 1)[-1].strip() if Mread["letter"] == "M-" else "")
        else:
            parts.append(CONSEQUENCES["S1 " + Mread["letter"]])
        consequence = "; ".join(p for p in parts if p)
    return {"H": H, "S": S, "M": Mread, "cell": cell, "consequence": consequence,
            "counted_seeds": list(LIVE_SEEDS), "reported_not_counted": list(CONCEIVING_SEED),
            "paired": paired, "vetoes": vet, "policy_read": SYSTEM_POLICY}


# ---- (ii) the two further draws ---------------------------------------------------------------------------
def law_miss_chance(cosine: float, n_bits: int = N_BITS, n_tables: int = N_TABLES) -> float:
    """One minus the chance that a query at this cosine shares a bucket with its own line in at least one table."""
    from cortex_c2b.bucket_probe import proposal_probability
    return 1.0 - proposal_probability(cosine, n_bits, n_tables)


def expected_misses_from_record(metrics_dir=None) -> dict:
    """The expectation recomputed from row 45's three seed 0 files (the own cosines of the 200 planted questions)."""
    import json
    d = Path(metrics_dir) if metrics_dir else Path(__file__).resolve().parent.parent / "metrics" / "mqar"
    names = {"8ceba5db8d0b": "bucket-probe-8ceba5db8d0b.json", "4cfd6f6e1406": "bucket-probe-4cfd6f6e1406-run2.json", "2e1dc71913b1": "bucket-probe-2e1dc71913b1.json"}
    out = {}
    for rid, name in names.items():
        f = json.loads((d / name).read_text(encoding="utf-8"))
        out[rid] = sum(law_miss_chance(q["cos_query_own"]) for q in f["questions"] if q["planted"])
    return out


def read_draws(probe_files: list[dict]) -> dict:
    """The replication of the geometry at index seeds 1 and 2 on the three journals, from the bucket probe's files: C2b's
    candidate recall and admissibility per draw against the expectation written before the run. Two clean draws show
    that the seed 0 result is not singular among three draws and establish no recall for another draw; one miss is read
    against the law at the question's own cosine (`questions[i].cos_query_own` of the file; the rows carry no cosine); a
    draw that breaks the budget is recorded, not adopted. The stopping rule of the amendment is enforced: exactly seeds 1
    and 2 on each journal, no third draw after a miss, no re reading at another threshold; a file of another index seed
    is refused. The chance of a clean draw is the Poisson approximation exp(-E) of the product over questions (they agree
    to 1e-5 on the record's cosines)."""
    by = {}
    for f in probe_files:
        s = int(f["index_seed"])
        if s not in (1, 2):
            raise ValueError(f"read_draws: index seed {s} is not a declared draw; exactly seeds 1 and 2, no third draw after a miss (ADR-008, 2026-10-13, (ii))")
        if (f["checkpoint"]["run_id"], s) in by:
            raise ValueError(f"read_draws: two files for {f['checkpoint']['run_id']} at index seed {s}; one draw per seed")
        by[(f["checkpoint"]["run_id"], s)] = f
    need = [(r, s) for r in RUN_IDS for s in (1, 2)]
    if any(k not in by for k in need):
        raise ValueError("read_draws: the files of index seeds 1 and 2 on each of the three journals are needed; missing " + str([k for k in need if k not in by]))
    misses, detail, inadmissible, clean = 0, [], [], 0
    for (r, s) in need:
        agg = by[(r, s)]["aggregates"]["C2b"]
        if not agg["admissible"]:
            inadmissible.append((r, s))
        if agg["candidate_recall"] == 1.0 and not agg["misses"]:
            clean += 1
        questions = {q["i"]: q for q in by[(r, s)]["questions"]}
        for i in agg["misses"]:
            row = next(x for x in by[(r, s)]["rows"]["C2b"] if x["i"] == i)
            if row["in_candidates"]:
                raise ValueError(f"read_draws: {r} seed {s} question {i} is listed as a miss and marked in the candidates; the file contradicts itself")
            cos = questions[i]["cos_query_own"]
            misses += 1
            detail.append({"run_id": r, "index_seed": s, "i": i, "entity": questions[i]["entity"], "cos_query_own": cos,
                           "law_miss_chance_at_cosine": law_miss_chance(cos), "miss_class": row.get("miss_class")})
    e = sum(EXPECTED_MISSES.values())
    expectation = {"expected_misses_per_draw": e, "expected_misses_per_journal": dict(EXPECTED_MISSES),
                   "chance_clean_draw": math.exp(-e), "chance_both_clean": math.exp(-2 * e),
                   "per_draw_miss_chance_bound_95": 1.0 - 0.05 ** 0.5}
    if inadmissible:
        reading = f"a draw breaks the budget ({inadmissible}): recorded, not adopted, as (d) of the amendment of 2026-10-12 reads an inadmissible configuration"
    elif misses == 0:
        reading = ("two clean draws per journal: the seed 0 result is not singular among three draws, the expected outcome written before the run "
                   f"(chance {expectation['chance_both_clean']:.3f}); they establish no recall for another draw, the per draw miss chance is bounded "
                   f"only at {expectation['per_draw_miss_chance_bound_95']:.2f} (one sided 95 %); the adoption keeps seed 0 as a declared parameter")
    else:
        reading = (f"one miss" if misses == 1 else f"{misses} misses") + (" under seeds 1 or 2, read against the expectation (about one chance in sixteen over the two draws): "
                   "the adoption keeps seed 0 as a declared parameter, the label says 'one draw', and the draw becomes a variable only if the question's own cosine "
                   "does not account for the miss by the law")
    return {"clean_draws": clean, "draws": len(need), "misses": misses, "miss_detail": detail, "inadmissible": inadmissible, **expectation, "reading": reading}
