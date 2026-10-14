"""cortex_c2b.near_duplicate_reading -- the cold reading of the near duplicate probe's files (ADR-007, amendment
of 2026-10-13, evening; validated 2026-10-14, 11:05), in the declared order: F1 from the gate's decisions in
the planting records, then F2 from the two primary paired tests on provenance, then the letter as the product
of the two facts, with the exits. Every function takes files as dicts and recomputes from their rows; the
counts a file carries in its own blocks are recomputed here from the rows and compared, a difference making the
file defective. Nothing is read from a console.

  F1 produced   in stage 2, under at least one of the three encoders, at least one near duplicate of a declared
                kind refused with its partner as the nearest stored code (E0d's background refusals named beside it)
  F2 produced   one of the two primary tests (E1 against E0d, E2 against E0d, each on `own_cited` per question,
                the two live seeds pooled, the exact two sided binomial at one half on the discordant pairs,
                Holm's step down at a family wise 0.05) significant in the direction of fewer own citations
                under the near duplicates; a significant gain is reported as a gain and does not produce F2
  (a) F1 and not F2   (b) F2 and not F1   (c) neither   (d) both
  exits: a stage 2 precondition stop, a cues file missing, a stage 1 second process that disagrees, a stage 3
  arm missing or stopped: the letter is withheld and the row says what was unread
"""
from __future__ import annotations

import json
from pathlib import Path

from cortex_c2b.adoption_reading import binomial_two_sided_p, LIVE_SEEDS, CONCEIVING_SEED, RUN_IDS, SYSTEM_POLICY
from cortex_c2b.near_duplicate_probe import (ARMS as PROBE_ARMS, KINDS, THRESHOLD_TOL, LIVE_N_TABLES, PARAPHRASE_ARM, PARTNER_KINDS,
                                             parse_statement, parse_question)

ROW48_ARMS = ("B0", "E0d", "E1", "E2")     # row 48's four arms; the reading of row 48 is frozen on them (E4 is row 49's)
ARMS = ROW48_ARMS                          # this module's `ARMS` stays row 48's; the probe's five arms are `PROBE_ARMS`
ROW49_ARMS = PROBE_ARMS                    # row 49's five arms (B0, E0d, E1, E2, E4), under the identity rule

# ---- row 50: the decode shapes (ADR-007, amendment of 2026-10-14, evening, section 6) ------------------------------
SHAPES = ("none", "R1", "R2", "R3", "R4")
TIE_BAND = 1e-4             # the record's tolerance on a cosine (above the measured drift of 4.1e-5): two marks within it are tied
SHAPE_WHAT = {
    "none": "the base policy: the model's citation surviving the veto stands",
    "R1": "the highest mark: among the window lines at or above the threshold, the cited line becomes the one with the highest mark at full precision; "
          "within the tie band the model's line stands if it is among the tied, else the highest in retrieval order",
    "R2": "the key match: among the lines at or above the threshold whose statement parses, the one whose entity equals the question's byte for byte; "
          "none or several, the model's citation stands",
    "R3": "abstention on disagreement: when the model's cited line is not the highest marked among those at or above the threshold, abstain",
    "R4": "the composition (exact key first, similarity second): when the question parses, the candidates are the lines at or above the threshold whose "
          "statement parses and whose entity equals the question's; none: abstain; one value: the candidate with the highest mark (R1's tie rule); "
          "two or more values: abstain; when the question parses under no template: R1",
}


def select_citation(shape: str, cited: str | None, lines: list[dict], threshold: float, question: str) -> dict:
    """The one function the live probe and the replay share (ADR-007, amendment of 2026-10-14, evening, section 6):
    a decode shape acting after `parse_contract` and after the veto, only on a citation that survived it. `cited` is
    the pointer the model cited and the veto kept (None: a model abstention, a malformed output, a label outside the
    window or a vetoed citation, whose outcome every shape keeps); `lines` the window's lines in retrieval order, each
    `{pointer, mark, text}` with the mark at full precision; `question` the recorded question text, read through the
    protocol's question templates and never through an answer record's label. Returns the final citation (`pointer`,
    None for an abstention), whether the shape changed it, and the attribution of the decision; no shape ever
    creates a claim."""
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}, got {shape!r}")
    out = {"shape": shape, "input": cited, "pointer": cited, "changed": False, "abstain": False, "reason": "no citation survived the veto; untouched"}
    if cited is None or shape == "none":
        if cited is not None:
            out["reason"] = "the base policy: the model's citation stands"
        return out
    above = [l for l in lines if l.get("mark") is not None and float(l["mark"]) >= threshold]
    if not above or cited not in {l["pointer"] for l in above}:
        out["reason"] = "the cited line is not among the lines at or above the threshold; untouched (the veto's domain)"
        return out
    top = max(float(l["mark"]) for l in above)
    cited_mark = next(float(l["mark"]) for l in above if l["pointer"] == cited)

    def highest(cands: list[dict]) -> tuple[str, bool]:
        """R1's rule on a candidate list in retrieval order: when the two highest marks are within the tie band, the model's
        line stands if it is one of the two, else the higher in retrieval order (the oracle's tie break); otherwise the
        highest mark at full precision. Returns (pointer, tied)."""
        ordered = sorted(cands, key=lambda l: -float(l["mark"]))             # stable: retrieval order kept among equal marks
        tied = ordered[:2] if (len(ordered) >= 2 and float(ordered[0]["mark"]) - float(ordered[1]["mark"]) <= TIE_BAND) else ordered[:1]
        if any(l["pointer"] == cited for l in tied):
            return cited, len(tied) > 1
        return tied[0]["pointer"], len(tied) > 1

    def finish(pointer, reason, **more):
        out.update({"pointer": pointer, "abstain": pointer is None, "changed": pointer != cited, "reason": reason, **more})
        return out

    if shape == "R1":
        p, tied = highest(above)
        return finish(p, "the highest mark" + (" (within the tie band)" if tied else ""), tied=tied)
    if shape == "R3":
        if cited_mark == top:
            return finish(cited, "the cited line holds the highest mark")
        return finish(None, "the cited line is not the highest marked: abstention on disagreement", top_mark=top, cited_mark=cited_mark)
    q = parse_question(question)
    if shape == "R2":
        if q is None:
            return finish(cited, "the question parses under no template; the model's citation stands", question_parsed=False)
        matches = [l for l in above if (ps := parse_statement(l.get("text") or "")) is not None and ps[1] == q[1]]
        if len(matches) == 1:
            return finish(matches[0]["pointer"], "the one line whose entity equals the question's", question_parsed=True, matches=1)
        return finish(cited, f"{len(matches)} lines match the question's entity; the model's citation stands", question_parsed=True, matches=len(matches))
    # R4
    if q is None:
        p, tied = highest(above)
        return finish(p, "the question parses under no template: R1" + (" (within the tie band)" if tied else ""), question_parsed=False, fallback="R1", tied=tied)
    parsed = [(l, parse_statement(l.get("text") or "")) for l in above]
    unreadable = sum(1 for _, ps in parsed if ps is None)
    cands = [l for l, ps in parsed if ps is not None and ps[1] == q[1]]
    values = sorted({ps[2] for l, ps in parsed if ps is not None and ps[1] == q[1]})
    if not cands:
        return finish(None, "no line at or above the threshold is about the question's entity: abstention", question_parsed=True, candidates=0, values=0, unreadable=unreadable)
    if len(values) >= 2:
        return finish(None, f"{len(cands)} lines about the entity carry {len(values)} values: a conflict is exposed, abstention", question_parsed=True,
                      candidates=len(cands), values=len(values), unreadable=unreadable)
    p, tied = highest(cands)
    return finish(p, "the candidate with the highest mark among the lines about the entity" + (" (within the tie band)" if tied else ""),
                  question_parsed=True, candidates=len(cands), values=1, unreadable=unreadable, tied=tied)

ALPHA = 0.05            # the family wise level, a constant of the declaration
RESOLUTION_SENTENCE = ("under Holm over the two primary tests the smaller p must be at or under 0.025: the first rejection needs at least seven "
                       "discordant pairs all one way (six give 0.03125), the second six once the first is rejected (0.03125 against 0.05); so a loss "
                       "of six questions in four hundred on either kind is invisible to the family, and a single test at 0.05 would need six (five give 0.0625)")


def holm(pvalues: dict, alpha: float = ALPHA) -> dict:
    """Holm's step down over the named tests: the smallest p is compared with alpha / m, the next with alpha / (m - 1), and
    so on; rejection stops at the first failure. Returns {name: rejected}."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, stop = {}, False
    for rank, (name, p) in enumerate(items):
        if stop or p > alpha / (m - rank):
            stop = True; out[name] = False
        else:
            out[name] = True
    return out


# ---- what a file must be ------------------------------------------------------------------------------
def check_arm_file(f: dict, arm: str) -> None:
    """A stage 3 file is the arm the caller says, under the system's policy and the adopted geometry, bound to its planting."""
    nd = f.get("near_duplicates") or {}
    if nd.get("arm") != arm:
        raise ValueError(f"file: labelled arm {nd.get('arm')!r}, read as {arm!r}")
    if f.get("policy") != SYSTEM_POLICY:
        raise ValueError(f"file: policy {f.get('policy')!r}, the declaration reads {SYSTEM_POLICY!r}")
    live = (f.get("index") or {}).get("live") or {}
    if live.get("n_tables") != LIVE_N_TABLES or live.get("signature") != "exact":
        raise ValueError(f"file: live index {live}, the declaration reads sixteen tables with the exact signature")
    if not nd.get("planting"):
        raise ValueError("file: no planting record; the file is not bound to the journal it read")
    if nd.get("consistent") is False:
        raise ValueError("file: not consistent with its planting record: " + "; ".join(nd.get("problems", [])))


def _check_pair(control: dict, test: dict) -> None:
    if control["checkpoint"]["run_id"] != test["checkpoint"]["run_id"]:
        raise ValueError("pair: the two files are not of the same checkpoint")
    if control.get("policy") != test.get("policy"):
        raise ValueError("pair: the two files are not under the same policy")
    ec = [a["entity"] for a in control["answers"]["on"]]; et = [a["entity"] for a in test["answers"]["on"]]
    if ec != et:
        raise ValueError("pair: the planted questions are not the same in the same order")


def own_cited(answer: dict) -> bool:
    """The declared definition, read from the field every session B file carries since row 47, never from a derived one."""
    return answer.get("cited_pointer_is_own") is True


def side_counts(f: dict) -> dict:
    """The three numbers reported beside the tests, recomputed from the rows and compared to the file's own block."""
    rows = f["answers"]["on"]
    both = sum(1 for a in rows if a.get("both_marks_at_or_above_threshold"))
    cited_dup = sum(1 for a in rows if a.get("cited_is_duplicate"))
    given_both = sum(1 for a in rows if a.get("cited_is_duplicate") and a.get("both_marks_at_or_above_threshold"))
    rec = {"own_cited": sum(1 for a in rows if own_cited(a)), "cited_is_duplicate": cited_dup, "duplicate_in_window": sum(1 for a in rows if a.get("duplicate_in_window")),
           "both_marks_at_or_above_threshold": both, "duplicate_cited_given_both_present_n": given_both,
           "duplicate_cited_given_both_present": (given_both / both if both else None),
           "oracle_cites_own": sum(1 for a in rows if a.get("oracle_cites_own")), "oracle_cites_duplicate": sum(1 for a in rows if a.get("oracle_cites_duplicate"))}
    own_block = ((f.get("near_duplicates") or {}).get("counts") or {})
    diffs = {k: (own_block.get(k), v) for k, v in rec.items() if k in own_block and own_block.get(k) != v}
    return {**rec, "agrees_with_file": not diffs, "differences": diffs}


def own_cited_pairs(control: dict, test: dict) -> dict:
    """The paired counts on provenance, the question as the unit: b, questions whose own line is cited under the test arm
    and not under the control; c, the reverse. Fewer own citations under the test arm means c > b. Each discordant pair is
    split so that it can be attributed: by the own line's presence in the window under (control, test), by the duplicate's
    presence and both marks at or above the threshold under the test arm, and by whether the own line was admitted at the
    test arm's planting (from the planting record), so a c pair reads as a collision, a window miss or a gate refusal."""
    _check_pair(control, test)
    admitted = {w["for_entity"]: w["admitted"] for w in (((test.get("near_duplicates") or {}).get("planting") or {}).get("writes") or []) if w.get("kind") == "original"}
    b = c = 0
    split = {s: {"newly_present": 0, "already_present": 0, "lost": 0, "absent_both": 0, "duplicate_in_window": 0, "both_marks": 0, "own_not_admitted": 0} for s in ("b", "c")}
    for x, y in zip(control["answers"]["on"], test["answers"]["on"]):
        oc, ot = own_cited(x), own_cited(y)
        if oc == ot:
            continue
        side = "b" if ot else "c"
        if side == "b":
            b += 1
        else:
            c += 1
        wc, wt = bool(x.get("own_in_window")), bool(y.get("own_in_window"))
        where = "newly_present" if (wt and not wc) else "already_present" if (wt and wc) else "lost" if (wc and not wt) else "absent_both"
        split[side][where] += 1
        split[side]["duplicate_in_window"] += int(bool(y.get("duplicate_in_window")))
        split[side]["both_marks"] += int(bool(y.get("both_marks_at_or_above_threshold")))
        split[side]["own_not_admitted"] += int(admitted.get(y["entity"]) is False)
    return {"run_id": control["checkpoint"]["run_id"], "n": len(control["answers"]["on"]),
            "own_cited_control": sum(1 for a in control["answers"]["on"] if own_cited(a)),
            "own_cited_test": sum(1 for a in test["answers"]["on"] if own_cited(a)), "b_test_only": b, "c_control_only": c, "split": split}


def f1_counts_from_writes(planting: dict) -> dict:
    """F1's numbers recomputed from the planting record's writes and compared to its `counts` block."""
    writes = planting.get("writes") or []
    dups = [w for w in writes if w.get("kind") in KINDS and not w.get("dropped")]
    refused = [w for w in dups if not w.get("admitted")]
    rec = {"duplicates": len(dups), "duplicates_admitted": len(dups) - len(refused), "duplicates_refused": len(refused),
           "duplicates_refused_partner_nearest": sum(1 for w in refused if (w.get("partner") or {}).get("nearest_is_partner")),
           "originals_refused": sum(1 for w in writes if w.get("kind") == "original" and not w.get("admitted")),
           "e0d_written": sum(1 for w in writes if w.get("kind") == "E0d"), "e0d_refused": sum(1 for w in writes if w.get("kind") == "E0d" and not w.get("admitted")),
           "refused_against": sorted({((w.get("nearest_before") or {}).get("entity") or "?") for w in refused if not (w.get("partner") or {}).get("nearest_is_partner")})}
    cnt = planting.get("counts") or {}
    diffs = {k: (cnt.get(k), v) for k, v in rec.items() if k in cnt and cnt.get(k) != v}
    if "originals" in cnt and "originals_admitted" in cnt and cnt["originals"] - cnt["originals_admitted"] != rec["originals_refused"]:
        diffs["originals_refused"] = (cnt["originals"] - cnt["originals_admitted"], rec["originals_refused"])
    return {**rec, "agrees_with_file": not diffs, "differences": diffs}


def read_f1(planting_by_run: dict) -> dict:
    """`planting_by_run`: {run_id: {arm: planting record}}. F1 is produced when at least one near duplicate of E1 or E2 was
    refused with its partner as the nearest stored code under at least one of the three encoders; the two kinds on the three
    checkpoints must all be there for F1 to be read, else it is incomplete and the letter is withheld."""
    per, unread, produced, defective = {}, [], False, []
    for rid in RUN_IDS:
        arms = planting_by_run.get(rid) or {}
        row = {}
        for arm in ROW48_ARMS:
            rec = arms.get(arm)
            if rec is None:
                row[arm] = None
                if arm in KINDS:
                    unread.append((rid, arm))
                continue
            row[arm] = f1_counts_from_writes(rec)
            if not row[arm]["agrees_with_file"]:
                defective.append((rid, arm, row[arm]["differences"]))
            if arm in KINDS and row[arm]["duplicates_refused_partner_nearest"] > 0:
                produced = True
        per[rid] = row
    return {"produced": produced, "complete": not unread, "unread": unread, "defective": defective, "per_run": per,
            "rule": "at least one near duplicate of E1 or E2 refused with its partner as the nearest stored code, under at least one encoder"}


def read_f2(files_by_run: dict) -> dict:
    """`files_by_run`: {run_id: {arm: session B file as a dict}}. The two primary tests, E1 against E0d and E2 against E0d, on
    `own_cited`, the two live seeds pooled, two sided, Holm at 0.05; the direction read from the sign; seed 1337 and the per seed
    counts reported (b and c, no p: the pooled test is the declared one). A missing, stopped or mislabelled arm leaves its
    test unread."""
    tests, reported, unread, side = {}, {}, [], {}
    for kind in KINDS:
        b = c = 0; per_seed = {}; complete = True
        for rid in LIVE_SEEDS:
            ctl = files_by_run.get(rid, {}).get("E0d"); tst = files_by_run.get(rid, {}).get(kind)
            try:
                if ctl is None or ctl.get("stopped"):
                    raise ValueError("E0d missing or stopped")
                check_arm_file(ctl, "E0d")
                if tst is None or tst.get("stopped"):
                    raise ValueError(f"{kind} missing or stopped")
                check_arm_file(tst, kind)
            except ValueError as e:
                complete = False; unread.append((rid, kind, str(e))); continue
            pc = own_cited_pairs(ctl, tst)
            per_seed[rid] = {"seed": LIVE_SEEDS[rid], **pc}
            side[(rid, kind)] = side_counts(tst)
            b += pc["b_test_only"]; c += pc["c_control_only"]
        if complete:
            tests[kind] = {"b_test_only": b, "c_control_only": c, "n_pairs": sum(ps["n"] for ps in per_seed.values()), "p": binomial_two_sided_p(b, c),
                           "direction": ("fewer own citations under the near duplicates" if c > b else "more own citations under the near duplicates" if b > c else "equal"),
                           "per_seed": per_seed}
        for rid in CONCEIVING_SEED:
            ctl = files_by_run.get(rid, {}).get("E0d"); tst = files_by_run.get(rid, {}).get(kind)
            try:
                if ctl is None or tst is None or ctl.get("stopped") or tst.get("stopped"):
                    raise ValueError("missing or stopped")
                check_arm_file(ctl, "E0d"); check_arm_file(tst, kind)
                pc = own_cited_pairs(ctl, tst)
                reported[f"{rid}:{kind}"] = {"seed": CONCEIVING_SEED[rid], **pc}
                side[(rid, kind)] = side_counts(tst)
            except ValueError as e:
                reported[f"{rid}:{kind}"] = {"seed": CONCEIVING_SEED[rid], "unread": str(e)}
    rejected = holm({k: v["p"] for k, v in tests.items()}) if tests else {}
    produced = any(rejected.get(k) and tests[k]["c_control_only"] > tests[k]["b_test_only"] for k in tests)
    gains = [k for k in tests if rejected.get(k) and tests[k]["b_test_only"] > tests[k]["c_control_only"]]
    return {"produced": produced, "tests": tests, "holm_rejected": rejected, "alpha_family_wise": ALPHA, "gains_named": gains,
            "reported_conceiving_seed": reported, "side_counts": {f"{rid}:{kind}": v for (rid, kind), v in side.items()},
            "unread": unread, "complete": not unread, "resolution": RESOLUTION_SENTENCE}


def letter(f1_produced: bool, f2_produced: bool) -> str:
    return {(True, False): "a", (False, True): "b", (False, False): "c", (True, True): "d"}[(bool(f1_produced), bool(f2_produced))]


def thresholds_against_record(files_by_run: dict, record_files: dict) -> dict:
    """Every arm's oracle threshold against row 47's file of the same checkpoint, the reproduction label at 1e-6 (the
    threshold is read on a scratch journal of the training family, so it must reproduce on every arm)."""
    out = {}
    for rid, arms in files_by_run.items():
        ref = record_files.get(rid)
        for arm, f in arms.items():
            if f is None or ref is None:
                continue
            thr, thr_ref = f["hm_mark_oracle"]["threshold"], ref["hm_mark_oracle"]["threshold"]
            d = abs(thr - thr_ref)
            out[f"{rid}:{arm}"] = {"threshold": thr, "record": thr_ref, "diff": d, "within_tolerance": d <= THRESHOLD_TOL,
                                   "label": ("reproduced" if d <= THRESHOLD_TOL else "not reproduced")}
    return out


def b0_against_record(files_by_run: dict, record_files: dict) -> dict:
    """B0 (the 200 facts alone, planted on the processor) against row 47's `index16exact-mark-veto-value` file of the same
    checkpoint, question by question: the strict paired counts and the own citation counts; the differences, if any, are the
    planting machine's and are named; B0 attributes and never stops."""
    from cortex_c2b.adoption_reading import paired_counts
    out = {}
    for rid, arms in files_by_run.items():
        b0, ref = arms.get("B0"), record_files.get(rid)
        if b0 is None or ref is None or b0.get("stopped"):
            out[rid] = None; continue
        check_arm_file(b0, "B0")
        pc = paired_counts(ref, b0)
        oc = own_cited_pairs(ref, b0)
        out[rid] = {"strict": pc["strict"], "outcomes": pc["outcomes"], "own_in_window": pc["own_in_window"], "own_cited": {k: oc[k] for k in ("own_cited_control", "own_cited_test", "b_test_only", "c_control_only")},
                    "identical_outcomes": all(pc["outcomes"][o][o] == sum(pc["outcomes"][o].values()) for o in pc["outcomes"])}
    return out


def read_near_duplicates(stage1: dict | None, cues_by_run: dict, files_by_run: dict, record_files: dict | None = None) -> dict:
    """The whole reading in the declared order. `cues_by_run`: {run_id: the cues file}; `files_by_run`: {run_id: {arm: file}};
    `record_files`: {run_id: row 47's system file} for the B0 comparison and the threshold labels. The exits come first and
    withhold the letter."""
    exits = []
    if stage1 is None:
        exits.append("stage 1 file missing")
    elif stage1.get("second_process") and not stage1["second_process"].get("agrees", True):
        exits.append("stage 1: the second process does not recompute the numbers")
    for rid in RUN_IDS:
        cf = cues_by_run.get(rid)
        if cf is None:
            exits.append(f"{rid}: no cues file (stage 2 not run)")
        elif cf.get("precondition", {}).get("stopped"):
            exits.append(f"{rid}: the stage 2 precondition stopped (the own line's rank differs from row 45's file)")
        for arm in ROW48_ARMS:
            f = files_by_run.get(rid, {}).get(arm)
            if f is None:
                exits.append(f"{rid}: arm {arm} missing")
            elif f.get("stopped"):
                exits.append(f"{rid}: arm {arm} stopped")
            else:
                try:
                    check_arm_file(f, arm)
                except ValueError as e:
                    exits.append(f"{rid}: arm {arm} refused ({e})")
    planting = {rid: {arm: (f.get("near_duplicates") or {}).get("planting") for arm, f in (files_by_run.get(rid) or {}).items() if f is not None}
                for rid in RUN_IDS}
    f1 = read_f1(planting)
    f2 = read_f2(files_by_run)
    out = {"exits": exits, "precondition_stopped": any("precondition stopped" in e for e in exits), "F1": f1, "F2": f2,
           "drift": {rid: ((cues_by_run.get(rid) or {}).get("precondition", {}).get("drift_exceeded_order_intact")) for rid in RUN_IDS},
           "band_stage1": (stage1 or {}).get("band"),
           "thresholds_against_record": (thresholds_against_record(files_by_run, record_files) if record_files else None),
           "b0_against_record": (b0_against_record(files_by_run, record_files) if record_files else None)}
    if exits or not f1["complete"] or not f2["complete"] or f1["defective"]:
        out["letter"] = None
        out["unread"] = "; ".join(exits + [f"F1 unread for {u}" for u in f1["unread"]] + [f"F2 unread for {u[:2]}" for u in f2["unread"]]
                                  + [f"defective planting record {d[:2]}" for d in f1["defective"]])
    else:
        out["letter"] = letter(f1["produced"], f2["produced"])
    return out


def load_from_dir(metrics_dir: str | Path) -> dict:
    """The reading's inputs from a directory of retained files: the stage 1 file, the cues files (a `-FAILED-precondition` one
    counts as that checkpoint's), the twelve arm files by their `checkpoint.run_id` and `near_duplicates.arm`, and row 47's
    `index16exact-mark-veto-value` files as the record files. Returns the four arguments of `read_near_duplicates`."""
    d = Path(metrics_dir)
    stage1 = json.loads((d / "near-dup-stage1.json").read_text(encoding="utf-8")) if (d / "near-dup-stage1.json").exists() else None
    cues, files, record = {}, {}, {}
    for rid in RUN_IDS:
        for name in (f"near-dup-cues-{rid}.json", f"near-dup-cues-{rid}-FAILED-precondition.json"):
            if (d / name).exists():
                cues[rid] = json.loads((d / name).read_text(encoding="utf-8"))
        ref = d / f"hm-lm-{rid}-session-b-index16exact-mark-veto-value.json"
        if ref.exists():
            record[rid] = json.loads(ref.read_text(encoding="utf-8"))
        files[rid] = {}
        for arm in ROW48_ARMS:
            for name in (f"hm-lm-{rid}-near-dup-{arm}-session-b-index16exact-mark-veto-value.json",
                         f"hm-lm-{rid}-near-dup-{arm}-session-b-index16exact-mark-veto-value-FAILED-arm.json"):
                if (d / name).exists():
                    files[rid][arm] = json.loads((d / name).read_text(encoding="utf-8"))
    return {"stage1": stage1, "cues_by_run": cues, "files_by_run": files, "record_files": record}


def summary_lines(reading: dict) -> list[str]:
    lines = []
    if reading["exits"]:
        lines.append("exits: " + "; ".join(reading["exits"]))
    f1 = reading["F1"]; lines.append(f"F1 produced: {f1['produced']} (complete: {f1['complete']})")
    for rid, row in f1["per_run"].items():
        parts = []
        for arm, v in row.items():
            if v is None:
                parts.append(f"{arm} unread")
            elif arm in KINDS:
                parts.append(f"{arm} refused {v['duplicates_refused']} (partner nearest {v['duplicates_refused_partner_nearest']}) of {v['duplicates']}")
            elif arm == "E0d":
                parts.append(f"E0d refused {v['e0d_refused']} of {v['e0d_written']}")
            else:
                parts.append(f"B0 originals refused {v['originals_refused']}")
        lines.append(f"  {rid}: " + ", ".join(parts))
    f2 = reading["F2"]; lines.append(f"F2 produced: {f2['produced']} (complete: {f2['complete']})")
    for k, tst in f2["tests"].items():
        lines.append(f"  {k} against E0d: b {tst['b_test_only']} c {tst['c_control_only']} p {tst['p']:.4g} Holm {f2['holm_rejected'].get(k)} ({tst['direction']})")
    if f2["gains_named"]:
        lines.append("  gains named: " + ", ".join(f2["gains_named"]))
    lines.append(f"letter: {reading.get('letter')}" + (f" (unread: {reading.get('unread')})" if reading.get("unread") else ""))
    return lines


# =====================================================================================================================
# Rows 49 and 50 (ADR-007, amendment of 2026-10-14, evening): the write side under the identity rule, read cold in the
# declared order (`read_row49`), and the read side selection replayed on row 49's files (`replay_row50`). Built
# 2026-10-15, before any file exists; nothing below is a number of the record.
# =====================================================================================================================
import math as _math

IDENTITY = "identity"
P_TOL = 1e-6                    # saliences, surprises and marks compared within it (the record's reproduction label, row 47's)
from cortex_c2b.lifecycle import LifecycleConfig as _LifecycleConfig
TRIBUNAL_FLOOR = float(_LifecycleConfig().tribunal_floor)   # the lifecycle's floor as coded (0.20): an admitted entry under it is evicted at the first phase
ROW49_FILE = "hm-lm-{rid}-near-dup-{arm}-identity-session-b-index16exact-mark-veto-value.json"
ROW50_LIVE_FILE = "hm-lm-{rid}-near-dup-E2-identity-R4-session-b-index16exact-mark-veto-value.json"
ROW48_FILE = "hm-lm-{rid}-near-dup-{arm}-session-b-index16exact-mark-veto-value.json"
ROW47_FILE = "hm-lm-{rid}-session-b-index16exact-mark-veto-value.json"
DECISION_12 = ("the identity rule removes the false redundancy at the write and becomes the default of every new journal in v1.1.0; the "
               "surprise rule stays behind the flag for the legacy journals; as the lifecycle is coded, an admitted entry under the tribunal's "
               "floor is evicted at the next phase, read or not, which on the saliences of row 48 is 1157 of the 1200 near duplicates; that loss "
               "is measured by ADR-010 C and decided after it")
ROW49_LABELS = ("F1 at admission zero by construction, up to a collision of the 64 bit content pointer; through the lifecycle measured by "
                "ADR-010 C; the costs measured on one draw (the generator's seed 0, three checkpoints of one recipe, five paraphrase templates)")
ROW50_LABEL = ("in sample by design (the arms, questions and checkpoints that produced the collision); one draw; by replay, confirmed live once")
CLOSURE_BOUND = 0.02            # the largest movement row 48 attributes to doubling the store with distinct facts
SUPERIORITY_ALPHA = 0.025       # one sided, on E2's contested reads
MIN_DISCORDANT = 6              # at least six discordant pairs all one way (five give 0.03125)


# ---- the statistics, exact ------------------------------------------------------------------------------------------
def mcnemar_one_sided_p(b: int, c: int) -> float:
    """The exact one sided McNemar p: under equal rates, the chance of at least b successes among b + c discordant pairs
    (the direction: the shape cites the own line on more reads than the base); 1.0 when nothing is discordant."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(_math.comb(n, i) for i in range(b, n + 1)) / 2 ** n


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    """Wilson's score interval for k of n (95 % at the default z)."""
    if n <= 0:
        return None
    p = k / n; d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * _math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def _tango_z(b: int, c: int, n: int, delta0: float) -> float:
    """The score statistic of Tango (1998) for the paired difference p12 - p21 at the value delta0, with p12 = b / n (the
    pairs where the first arm cites and the second does not) and p21 = c / n; the restricted maximum likelihood estimate
    of p21 under delta = delta0 is the positive root of 2 n p^2 - A p - c delta0 (1 - delta0) = 0 with
    A = (b + c) - delta0 (2 n - b + c), derived from the trinomial likelihood (b, c, n - b - c)."""
    A = (b + c) - delta0 * (2 * n - b + c)
    p21 = (A + _math.sqrt(A * A + 8.0 * n * c * delta0 * (1.0 - delta0))) / (4.0 * n)
    var = n * (2.0 * p21 + delta0 * (1.0 - delta0))
    if var <= 0:
        return float("inf") if (b - c - n * delta0) > 0 else float("-inf") if (b - c - n * delta0) < 0 else 0.0
    return (b - c - n * delta0) / _math.sqrt(var)


def tango_interval(b: int, c: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Tango's score confidence interval for p12 - p21 at the two sided level 2 Phi(z) - 1 (95 % at the default z), found
    by bisection on each side of the point estimate (b - c) / n; the one sided 97.5 % upper bound is its upper end."""
    if n <= 0:
        raise ValueError("n must be positive")
    est = (b - c) / n

    def root(lo: float, hi: float, target: float) -> float:
        # Z(delta) decreases in delta; find delta with Z(delta) = target
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if _tango_z(b, c, n, mid) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    lower = root(-1.0, est, z) if est > -1.0 else -1.0
    upper = root(est, 1.0, -z) if est < 1.0 else 1.0
    return lower, upper


# ---- what a row 49 file must be ---------------------------------------------------------------------------------------
def check_identity_arm_file(f: dict, arm: str) -> None:
    """A row 49 file is the arm the caller says, under the system's policy, the adopted geometry and the identity rule,
    bound to its planting, with no decode shape applied."""
    check_arm_file(f, arm)
    adm = (f.get("admission") or {})
    if adm.get("record") != IDENTITY:
        raise ValueError(f"file: the journal records the admission rule {adm.get('record')!r}, row 49 reads {IDENTITY!r}")
    if f.get("decode_shape") not in (None, "none"):
        raise ValueError(f"file: a decode shape {f.get('decode_shape')!r} was applied live; row 49's files carry none")
    planting = (f.get("near_duplicates") or {}).get("planting") or {}
    if (planting.get("admission") or {}).get("rule") != IDENTITY:
        raise ValueError("file: the planting record does not say the journal was planted under the identity rule")


def _planting(f: dict) -> dict:
    return ((f.get("near_duplicates") or {}).get("planting")) or {}


def _writes(f: dict) -> list[dict]:
    return [w for w in (_planting(f).get("writes") or []) if not w.get("dropped")]


def _close(a, b, tol: float = P_TOL) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


# ---- P0 to P3 ----------------------------------------------------------------------------------------------------------
def p0_integrity(files_by_run: dict, expected_commit: str | None = None, provenance: dict | None = None) -> dict:
    """P0: the fifteen files present, not stopped, each what its name says and in the slot of its checkpoint; the files' sha256
    against their provenance when one is given (`provenance`: file name to the sixteen character sha256 of the LF normalised
    bytes, the record's form) and always returned for the record test to pin; one code commit on a clean tree across the
    fifteen, equal to the build's when it is named; the journal unchanged by the read; the file bound to its planting; every
    admitted pointer equal to sha256(text)[:16]; the planting's rule and the journal's record both the identity rule."""
    from cortex_c2b import content_hash
    problems, commits, hashes = [], set(), {}
    for rid in RUN_IDS:
        for arm in ROW49_ARMS:
            f = (files_by_run.get(rid) or {}).get(arm)
            key = f"{rid}:{arm}"
            if f is None:
                problems.append(f"{key}: missing"); continue
            if f.get("stopped"):
                problems.append(f"{key}: stopped"); continue
            try:
                check_identity_arm_file(f, arm)
            except ValueError as e:
                problems.append(f"{key}: {e}"); continue
            if (f.get("checkpoint") or {}).get("run_id") != rid:
                problems.append(f"{key}: the file is of run id {(f.get('checkpoint') or {}).get('run_id')!r}, not of its slot")
            if f.get("code_tree_clean") is not True or not f.get("code_commit"):
                problems.append(f"{key}: the code's commit is unknown or the tree was not clean")
            commits.add(f.get("code_commit"))
            if not (f.get("journal") or {}).get("unchanged", False):
                problems.append(f"{key}: the journal was changed by the read")
            bad = [w for w in _writes(f) if w.get("admitted") and content_hash(w["text"].encode("utf-8")) != w.get("pointer")]
            if bad:
                problems.append(f"{key}: {len(bad)} admitted pointers are not sha256(text)[:16]")
            name = ROW49_FILE.format(rid=rid, arm=arm)
            hashes[name] = f.get("_sha256")
            if provenance is not None:
                if name not in provenance:
                    problems.append(f"{key}: no provenance entry for {name}")
                elif f.get("_sha256") is None or provenance[name] != f.get("_sha256"):
                    problems.append(f"{key}: sha256 {f.get('_sha256')} against the provenance's {provenance[name]}")
    if len(commits) > 1:
        problems.append(f"several code commits across the files: {sorted(c for c in commits if c)}")
    if expected_commit is not None and commits and commits != {expected_commit}:
        problems.append(f"the files' commit {sorted(commits)} is not the build's {expected_commit}")
    return {"ok": not problems, "problems": problems, "commit": (next(iter(commits)) if len(commits) == 1 else None), "file_sha256": hashes,
            "provenance_checked": provenance is not None}


def check_reference_files(row48_files_by_run: dict, row47_files: dict, row48_provenance: dict | None, row47_provenance: dict | None) -> dict:
    """The reference files the preconditions read against (row 48's twelve, row 47's three), bound to their provenance when it is
    given (`metrics/mqar/PROVENANCE-near-duplicates.json` and `PROVENANCE-adoption.json`, `files.<name>.sha256_lf`); a reference
    file whose hash differs from the record's is a defect named before anything is read against it."""
    problems = []
    for rid in RUN_IDS:
        for arm in ROW48_ARMS:
            f = (row48_files_by_run.get(rid) or {}).get(arm)
            name = ROW48_FILE.format(rid=rid, arm=arm)
            if f is None:
                problems.append(f"row 48 reference missing: {name}"); continue
            if row48_provenance is not None:
                rec = (row48_provenance.get("files") or {}).get(name, {}).get("sha256_lf")
                if rec is None or f.get("_sha256") != rec:
                    problems.append(f"row 48 reference {name}: sha256 {f.get('_sha256')} against the record's {rec}")
        f = row47_files.get(rid); name = ROW47_FILE.format(rid=rid)
        if f is None:
            problems.append(f"row 47 reference missing: {name}"); continue
        if row47_provenance is not None:
            rec = (row47_provenance.get("files") or {}).get(name, {}).get("sha256_lf")
            if rec is None or f.get("_sha256") != rec:
                problems.append(f"row 47 reference {name}: sha256 {f.get('_sha256')} against the record's {rec}")
    return {"ok": not problems, "problems": problems, "checked_against_record": row48_provenance is not None and row47_provenance is not None}


def p1_thresholds(files_by_run: dict, row47_files: dict) -> dict:
    """P1: the oracle's threshold within 1e-6 of row 47's on every file; the scratch journal under the arm's rule admitted
    its 200 facts."""
    out, problems = {}, []
    for rid in RUN_IDS:
        ref = row47_files.get(rid)
        for arm in ROW49_ARMS:
            f = (files_by_run.get(rid) or {}).get(arm)
            if f is None or ref is None:
                problems.append(f"{rid}:{arm}: no file or no row 47 file"); continue
            o = f["hm_mark_oracle"]
            d = abs(float(o["threshold"]) - float(ref["hm_mark_oracle"]["threshold"]))
            ok = d <= THRESHOLD_TOL and o.get("scratch_admitted") == o.get("scratch_facts") and o.get("scratch_admission") == IDENTITY
            out[f"{rid}:{arm}"] = {"threshold": o["threshold"], "record": ref["hm_mark_oracle"]["threshold"], "diff": d,
                                   "scratch_admitted": o.get("scratch_admitted"), "scratch_facts": o.get("scratch_facts"),
                                   "scratch_admission": o.get("scratch_admission"), "ok": ok}
            if not ok:
                problems.append(f"{rid}:{arm}: threshold difference {d} or scratch journal {o.get('scratch_admitted')} of {o.get('scratch_facts')} under {o.get('scratch_admission')}")
    return {"ok": not problems, "problems": problems, "per_file": out}


ROW48_QUESTION_FIELDS_EXACT = ("entity", "schema", "attr", "kind", "label", "answered", "outcome", "own_in_window", "cited_pointer",
                               "cited_pointer_is_own", "cited_is_highest_mark", "touched", "signatures_query", "own_in_candidates", "own_rank",
                               "own_cited", "cited_is_duplicate", "duplicate_in_window", "both_marks_at_or_above_threshold",
                               "oracle_cites_own", "oracle_cites_duplicate")
ROW48_QUESTION_FIELDS_CLOSE = ("mark_own", "mark_max", "mark_cited")


def compare_entries(f49: dict, f48: dict) -> dict:
    """The entries first: the planting records' writes compared in order (pointer and decision exact, surprise within 1e-6,
    the nearest stored code's entry id exact); a salience difference is attributed to the device, the library versions
    compared beside it."""
    w49, w48 = _writes(f49), _writes(f48)
    diffs = []
    if len(w49) != len(w48):
        diffs.append(f"{len(w49)} writes against {len(w48)}")
    salience_drift = 0.0
    for i, (a, b) in enumerate(zip(w49, w48)):
        if (a.get("pointer"), a.get("admitted"), a.get("text")) != (b.get("pointer"), b.get("admitted"), b.get("text")):
            diffs.append(f"write {i}: pointer, decision or text differ")
        if not _close(a.get("surprise"), b.get("surprise")):
            diffs.append(f"write {i}: surprise {a.get('surprise')} against {b.get('surprise')}")
        salience_drift = max(salience_drift, abs(float(a.get("surprise") or 0) - float(b.get("surprise") or 0)))
        na, nb = (a.get("nearest_before") or {}).get("entry_id"), (b.get("nearest_before") or {}).get("entry_id")
        if na != nb:
            diffs.append(f"write {i}: nearest stored code {na} against {nb}")
    v49, v48 = f49.get("versions") or {}, f48.get("versions") or {}
    return {"ok": not diffs, "differences": diffs[:50], "n_differences": len(diffs), "max_salience_drift": salience_drift,
            "device": {"row49": _planting(f49).get("device"), "row48": _planting(f48).get("device")},
            "versions": {"row49": v49, "row48": v48, "equal": v49 == v48},
            "attribution": (("a salience difference is attributed to the device; the versions " + ("are equal" if v49 == v48 else "differ"))
                            if any("surprise" in d for d in diffs) else None)}


def compare_questions(f49: dict, f48: dict) -> dict:
    """Then the questions: every per question field of row 48's three blocks exact (marks within 1e-6), the window's
    pointers and labels exact and its marks within 1e-6, the cue within 1e-6; the file's counts equal."""
    diffs = []
    for arm in ("on", "off", "negctrl"):
        a49, a48 = f49["answers"].get(arm) or [], f48["answers"].get(arm) or []
        if len(a49) != len(a48):
            diffs.append(f"{arm}: {len(a49)} answers against {len(a48)}"); continue
        for i, (x, y) in enumerate(zip(a49, a48)):
            for k in ROW48_QUESTION_FIELDS_EXACT:
                if k in y and x.get(k) != y.get(k):
                    diffs.append(f"{arm} {i} ({y.get('entity')}): {k} {x.get(k)!r} against {y.get(k)!r}")
            for k in ROW48_QUESTION_FIELDS_CLOSE:
                if k in y and not _close(x.get(k), y.get(k)):
                    diffs.append(f"{arm} {i} ({y.get('entity')}): {k} {x.get(k)} against {y.get(k)}")
            if "window" in y:
                wx, wy = x.get("window") or [], y.get("window") or []
                if [(w["label"], w["pointer"]) for w in wx] != [(w["label"], w["pointer"]) for w in wy] or \
                        any(not _close(p.get("mark"), q.get("mark")) for p, q in zip(wx, wy)):
                    diffs.append(f"{arm} {i} ({y.get('entity')}): the window differs")
            if "cue" in y and (len(x.get("cue") or []) != len(y["cue"]) or any(abs(float(p) - float(q)) > P_TOL for p, q in zip(x["cue"], y["cue"]))):
                diffs.append(f"{arm} {i} ({y.get('entity')}): the cue differs beyond 1e-6")
    counts = {}
    for k in ("hm_recall_on", "hm_recall_off", "hm_false_abstention_on", "hm_invalid_citation_on", "hm_valid_citation_on", "hm_retrieval_hit",
              "hm_negctrl_rate", "hm_negctrl_abstain_on"):
        if f49.get(k) != f48.get(k):
            diffs.append(f"count {k}: {f49.get(k)} against {f48.get(k)}")
        counts[k] = {"row49": f49.get(k), "row48": f48.get(k)}
    return {"ok": not diffs, "differences": diffs[:50], "n_differences": len(diffs), "counts": counts}


def p2_controls(files_by_run: dict, row48_files_by_run: dict) -> dict:
    """P2: on B0 and E0d, the identity files reproduce row 48's question by question, the entries first."""
    out, problems = {}, []
    for rid in RUN_IDS:
        for arm in ("B0", "E0d"):
            f49 = (files_by_run.get(rid) or {}).get(arm); f48 = (row48_files_by_run.get(rid) or {}).get(arm)
            if f49 is None or f48 is None:
                problems.append(f"{rid}:{arm}: no identity file or no row 48 file"); continue
            e = compare_entries(f49, f48)
            q = compare_questions(f49, f48) if e["ok"] else None
            out[f"{rid}:{arm}"] = {"entries": e, "questions": q}
            if not e["ok"]:
                problems.append(f"{rid}:{arm}: the entries differ ({e['n_differences']}): {e['differences'][:3]}")
            elif not q["ok"]:
                problems.append(f"{rid}:{arm}: the questions differ ({q['n_differences']}): {q['differences'][:3]}")
    return {"ok": not problems, "problems": problems, "per_file": out}


ROW48_REFUSALS = 1097           # row 48's refused near duplicates over the six E1 and E2 arms (RESULTS, row 48)
RECORD_RUN_IDS = tuple(RUN_IDS)  # the record's three checkpoints; the 1097 is compared on them alone (a test replaces RUN_IDS)


def p3_shadow(files_by_run: dict, row48_files_by_run: dict) -> dict:
    """P3: on the six E1 and E2 arms the shadow legacy gate reproduces row 48's planting write by write (the same decisions,
    the same 1097 refusals in total, surprises within 1e-6, the nearest entry equal where row 48 recorded it); on B0 and E0d
    it admits every write: the positive control that makes a zero F1 informative. E4 is only checked to carry the shadow's
    verdict on every write: its planting is read after F1, not here."""
    out, problems, refused_total, row48_total = {}, [], 0, 0
    for rid in RUN_IDS:
        for arm in ROW49_ARMS:
            f49 = (files_by_run.get(rid) or {}).get(arm)
            if f49 is None:
                problems.append(f"{rid}:{arm}: no identity file"); continue
            w49 = _writes(f49)
            judged = [w for w in w49 if w.get("legacy_decision") is not None]
            if len(judged) != len(w49):
                problems.append(f"{rid}:{arm}: {len(w49) - len(judged)} writes without the shadow's verdict")
            row = {"writes": len(w49)}
            if arm in KINDS:
                f48 = (row48_files_by_run.get(rid) or {}).get(arm)
                if f48 is None:
                    problems.append(f"{rid}:{arm}: no row 48 file"); out[f"{rid}:{arm}"] = row; continue
                w48 = _writes(f48)
                diffs = []
                if len(w48) != len(w49):
                    diffs.append(f"{len(w49)} writes against {len(w48)}")
                for i, (a, b) in enumerate(zip(w49, w48)):
                    if a.get("text") != b.get("text"):
                        diffs.append(f"write {i}: the texts differ"); continue
                    if a.get("legacy_decision") != b.get("admitted"):
                        diffs.append(f"write {i} ({a.get('kind')}): shadow {a.get('legacy_decision')} against row 48 {b.get('admitted')}")
                    if not _close(a.get("legacy_surprise"), b.get("surprise")):
                        diffs.append(f"write {i}: shadow surprise {a.get('legacy_surprise')} against row 48 {b.get('surprise')}")
                    nb = (b.get("nearest_before") or {}).get("entry_id")
                    if nb is not None and a.get("legacy_nearest") != nb:
                        diffs.append(f"write {i}: shadow nearest {a.get('legacy_nearest')} against row 48 {nb}")
                mine = sum(1 for w in judged if w["kind"] in KINDS and w["legacy_decision"] is False)
                theirs = sum(1 for w in w48 if w.get("kind") in KINDS and not w.get("admitted"))
                refused_total += mine; row48_total += theirs
                row.update({"legacy_refused": mine, "row48_refused": theirs, "differences": diffs[:50], "n_differences": len(diffs), "ok": not diffs})
                if diffs:
                    problems.append(f"{rid}:{arm}: the shadow does not reproduce row 48's planting ({len(diffs)}): {diffs[:3]}")
            elif arm in ("B0", "E0d"):
                refused = sum(1 for w in judged if w["legacy_decision"] is False)
                row.update({"legacy_refused": refused, "ok": refused == 0})
                if refused:
                    problems.append(f"{rid}:{arm}: the shadow refused {refused} writes where row 48's gate admitted all")
            else:
                row["ok"] = len(judged) == len(w49)                                   # E4: the verdict present on every write; read after F1
            out[f"{rid}:{arm}"] = row
    complete = all(f"{rid}:{arm}" in out for rid in RUN_IDS for arm in KINDS)
    if complete and refused_total != row48_total:
        problems.append(f"the shadow refused {refused_total} near duplicates over the six arms and row 48's files hold {row48_total}")
    if complete and tuple(RUN_IDS) == RECORD_RUN_IDS and row48_total != ROW48_REFUSALS:
        problems.append(f"row 48's files hold {row48_total} refused near duplicates; the record says {ROW48_REFUSALS}")
    return {"ok": not problems, "problems": problems, "per_file": out, "legacy_refused_near_duplicates_total": refused_total,
            "row48_refusals_recomputed": row48_total, "row48_refusals": ROW48_REFUSALS}


# ---- F1 under the rule, E4's planting, the reported quantities ---------------------------------------------------------------
def f1_under_rule(files_by_run: dict) -> dict:
    """The refused near duplicates of E1 and E2 (of 1200) and the refused paraphrases of E4 (of 600); zero by construction:
    a refusal can only be a same source text, which the plans exclude, so a refusal is a fault of the code. Refused originals
    and refused distinct facts are reported beside, outside F1 (they would have failed P2 or P3 first)."""
    refused_dups = refused_para = dups = paras = originals_refused = e0d_refused = 0
    per = {}
    for rid in RUN_IDS:
        for arm in ROW49_ARMS:
            f = (files_by_run.get(rid) or {}).get(arm)
            if f is None:
                continue
            ws = _writes(f)
            d = [w for w in ws if w.get("kind") in KINDS]; pz = [w for w in ws if w.get("kind") == PARAPHRASE_ARM]
            rd = sum(1 for w in d if not w.get("admitted")); rp = sum(1 for w in pz if not w.get("admitted"))
            refused_dups += rd; refused_para += rp; dups += len(d); paras += len(pz)
            originals_refused += sum(1 for w in ws if w.get("kind") == "original" and not w.get("admitted"))
            e0d_refused += sum(1 for w in ws if w.get("kind") == "E0d" and not w.get("admitted"))
            per[f"{rid}:{arm}"] = {"duplicates": len(d), "duplicates_refused": rd, "paraphrases": len(pz), "paraphrases_refused": rp,
                                   "reasons": sorted({w.get("reason") for w in d + pz if not w.get("admitted")})}
    complete = all((files_by_run.get(rid) or {}).get(arm) is not None for rid in RUN_IDS for arm in ROW49_ARMS)
    return {"zero": refused_dups == 0 and refused_para == 0,
            "near_duplicates": dups, "near_duplicates_refused": refused_dups, "paraphrases": paras, "paraphrases_refused": refused_para,
            "totals_as_declared": (dups == 1200 and paras == 600) if complete else None,
            "originals_refused_beside": originals_refused, "e0d_refused_beside": e0d_refused, "per_file": per,
            "label": "zero by construction, up to a collision of the 64 bit content pointer; a refusal is a fault of the code"}


def _quantiles(xs: list[float]) -> dict | None:
    if not xs:
        return None
    import numpy as np
    a = np.asarray(xs, dtype=np.float64)
    return {"n": int(a.size), "min": float(a.min()), "q10": float(np.quantile(a, 0.1)), "median": float(np.median(a)), "q90": float(np.quantile(a, 0.9)),
            "max": float(a.max()), "mean": float(a.mean())}


def e4_planting(files_by_run: dict) -> dict:
    """E4's planting: the paraphrases' surprise quantiles, the share the legacy shadow would have refused, the nearest stored
    code being the partner, per schema and per checkpoint; reported under the declared label."""
    from cortex_c2b.hm_protocol import SCHEMAS
    out = {}
    for rid in RUN_IDS:
        f = (files_by_run.get(rid) or {}).get(PARAPHRASE_ARM)
        if f is None:
            out[rid] = None; continue
        pz = [w for w in _writes(f) if w.get("kind") == PARAPHRASE_ARM]
        by_schema = {}
        for s in SCHEMAS:
            ws = [w for w in pz if (ps := parse_statement(w["text"])) is not None and ps[0] == s]
            by_schema[s] = {"n": len(ws), "surprise": _quantiles([float(w["surprise"]) for w in ws]),
                            "legacy_refused": sum(1 for w in ws if w.get("legacy_decision") is False),
                            "nearest_is_partner": sum(1 for w in ws if (w.get("partner") or {}).get("nearest_is_partner")),
                            "cos_raw": _quantiles([float(w["partner"]["cos_raw"]) for w in ws if (w.get("partner") or {}).get("cos_raw") is not None]),
                            "under_tribunal_floor": sum(1 for w in ws if w.get("admitted") and float(w["surprise"]) < TRIBUNAL_FLOOR)}
        out[rid] = {"paraphrases": len(pz), "legacy_refused": sum(1 for w in pz if w.get("legacy_decision") is False),
                    "legacy_refused_share": (sum(1 for w in pz if w.get("legacy_decision") is False) / len(pz)) if pz else None,
                    "surprise": _quantiles([float(w["surprise"]) for w in pz]), "per_schema": by_schema}
    return {"per_run": out, "label": "five templates, one per schema, one draw; no claim on paraphrases in general (B3)"}


def _truth_of(f: dict) -> dict:
    """pointer -> {entity, value, kind} of every admitted line, from the planting record's truth; when the planting carries no
    truth (row 48's files) the truth is read from the planting text through the probe's templates and flagged."""
    out = {}
    for w in _writes(f):
        if not (w.get("admitted") and w.get("pointer")):
            continue
        if w.get("truth"):
            out[w["pointer"]] = {**w["truth"], "kind": w.get("kind"), "rebuilt": False}
        else:
            ps = parse_statement(w.get("text") or "")
            if ps is not None:
                out[w["pointer"]] = {"entity": ps[1], "value": ps[2], "kind": w.get("kind"), "rebuilt": True}
    return out


def citation_support_of(answer: dict, truth: dict, pointer_key: str = "veto_cited_pointer") -> bool | None:
    """The citation support recomputed from the rows: the line cited (the one surviving the veto by default) is about the
    question's entity and carries the expected value, by the planting's truth; None when nothing was cited or the line is
    unreadable."""
    p = answer.get(pointer_key, answer.get("cited_pointer")) if pointer_key == "veto_cited_pointer" else answer.get(pointer_key)
    if p is None:
        return None
    t = truth.get(p)
    if t is None:
        return None
    return bool(t["entity"] == answer["entity"] and t["value"] == answer["attr"])


def provenance_metrics(f: dict, dup_of: dict | None = None) -> dict:
    """Section 7's provenance metrics for one file, recomputed from the rows: coverage, value correctness, citation support
    (on the veto surviving citation), the designated line, attributed accuracy, the abstentions split into false and
    verified, retrieval recall, the completeness of the candidates, and on the contested reads the forced choice error and
    the abstention rate on them minus that on reads where one line clears."""
    rows = f["answers"]["on"]; n = max(1, len(rows))
    truth = _truth_of(f); thr = float(f["hm_mark_oracle"]["threshold"])
    dup_of = dup_of if dup_of is not None else ((f.get("near_duplicates") or {}).get("pointer_of_duplicate") or {})
    own_of = {w["for_entity"]: w["pointer"] for w in _writes(f) if w.get("kind") == "original" and w.get("admitted")}
    claims = [a for a in rows if a.get("kind") != "unknown"]
    support = [citation_support_of(a, truth) for a in rows]
    supported = sum(1 for x in support if x is True); unreadable = sum(1 for a, x in zip(rows, support) if x is None and a.get("veto_cited_pointer", a.get("cited_pointer")))
    attributed = sum(1 for a, x in zip(rows, support) if x is True and a.get("outcome") == "strict")
    abst = [a for a in rows if a.get("kind") == "unknown"]

    def supporting_line(a):
        return any(float(w["mark"]) >= thr and truth.get(w["pointer"], {}).get("entity") == a["entity"] and truth.get(w["pointer"], {}).get("value") == a["attr"]
                   for w in a.get("window") or [] if w.get("mark") is not None)

    def clears(a):
        marks = {w["pointer"]: w.get("mark") for w in a.get("window") or []}
        own, dp = own_of.get(a["entity"]), dup_of.get(a["entity"])
        o = own in marks and marks[own] is not None and marks[own] >= thr
        d = dp is not None and dp in marks and marks[dp] is not None and marks[dp] >= thr
        return o, d
    contested = [a for a in rows if all(clears(a))]
    one_clears = [a for a in rows if sum(clears(a)) == 1]
    return {"n": len(rows), "coverage": len(claims) / n, "value_correct": sum(1 for a in rows if a.get("outcome") == "strict") / n,
            "citation_supported": supported, "citation_unreadable": unreadable, "citation_support_rate_among_citations": (supported / max(1, sum(1 for a in rows if a.get("veto_cited_pointer", a.get("cited_pointer"))))),
            "designated_line_cited": sum(1 for a in rows if a.get("cited_pointer_is_own") is True),
            "attributed_accuracy": attributed / n,
            "abstentions": {"n": len(abst), "false": sum(1 for a in abst if supporting_line(a)), "verified": sum(1 for a in abst if not supporting_line(a))},
            "retrieval_recall": sum(1 for a in rows if a.get("own_in_window")) / n,
            "own_in_candidates": (sum(1 for a in rows if a.get("own_in_candidates")) / n) if any("own_in_candidates" in a for a in rows) else None,
            "contested": {"n": len(contested), "forced_choice_error": sum(1 for a in contested if dup_of.get(a["entity"]) is not None and a.get("cited_pointer") == dup_of.get(a["entity"])),
                          "abstention_rate": (sum(1 for a in contested if a.get("kind") == "unknown") / len(contested)) if contested else None,
                          "abstention_rate_one_clears": (sum(1 for a in one_clears if a.get("kind") == "unknown") / len(one_clears)) if one_clears else None,
                          "difference": ((sum(1 for a in contested if a.get("kind") == "unknown") / len(contested)) - (sum(1 for a in one_clears if a.get("kind") == "unknown") / len(one_clears)))
                                        if (contested and one_clears) else None},
            "live_cost": _cost_of(rows)}


def _cost_of(rows: list[dict]) -> dict | None:
    xs = [int(a["touched"]) for a in rows if "touched" in a]
    if not xs:
        return None
    import numpy as np
    return {"n": len(xs), "mean": float(np.mean(xs)), "p95": float(np.percentile(xs, 95)), "max": int(max(xs))}


def own_cited_pairs_with_order(control: dict, test: dict) -> dict:
    """`own_cited_pairs` with the declared third split: the discordant pairs whose two windows hold the same lines in another
    order (the same pointers, the labels differing)."""
    pc = own_cited_pairs(control, test)
    for s in ("b", "c"):
        pc["split"][s]["same_lines_other_order"] = 0
    for x, y in zip(control["answers"]["on"], test["answers"]["on"]):
        oc, ot = own_cited(x), own_cited(y)
        if oc == ot:
            continue
        side = "b" if ot else "c"
        wx = [(w["label"], w["pointer"]) for w in x.get("window") or []]; wy = [(w["label"], w["pointer"]) for w in y.get("window") or []]
        if wx and {p for _, p in wx} == {p for _, p in wy} and wx != wy:
            pc["split"][side]["same_lines_other_order"] += 1
    return pc


def citation_support_pairs(control: dict, test: dict) -> dict:
    """E4 against E0d on citation support (the citation surviving the veto about the question's entity with the expected value,
    by the planting's truth): b, questions supported under the test arm and not under the control; c, the reverse."""
    _check_pair(control, test)
    tc, tt = _truth_of(control), _truth_of(test)
    b = c = 0
    for x, y in zip(control["answers"]["on"], test["answers"]["on"]):
        sc, st = citation_support_of(x, tc) is True, citation_support_of(y, tt) is True
        b += int(st and not sc); c += int(sc and not st)
    return {"n": len(control["answers"]["on"]), "supported_control": sum(1 for a in control["answers"]["on"] if citation_support_of(a, tc) is True),
            "supported_test": sum(1 for a in test["answers"]["on"] if citation_support_of(a, tt) is True), "b_test_only": b, "c_control_only": c}


def reported_quantities(files_by_run: dict) -> dict:
    """The reported quantities of row 49, written before any number: one paired test (E2 against E0d, exact two sided
    McNemar at 0.05, the live seeds pooled, per seed counts, seed 1337 reported); E1 and E4 against E0d with counts, paired
    differences and Tango intervals, no p; E4 beside on citation support; the attribution split of every discordant pair
    (window, both marks, same lines in another order); the duplicate cited given both present; the oracle's own line count;
    the protocol's rates and section 7's provenance metrics; the live cost; the tribunal's static count; the order dependence
    of the cut table (nearest codes)."""
    tests, per_seed = {}, {}
    for kind in ("E1", "E2", PARAPHRASE_ARM):
        b = c = n = 0; rows = {}
        for rid in RUN_IDS:
            ctl = (files_by_run.get(rid) or {}).get("E0d"); tst = (files_by_run.get(rid) or {}).get(kind)
            if ctl is None or tst is None:
                rows[rid] = None; continue
            pc = own_cited_pairs_with_order(ctl, tst)
            if kind == PARAPHRASE_ARM:
                pc["citation_support"] = citation_support_pairs(ctl, tst)
            rows[rid] = pc
            if rid in LIVE_SEEDS:
                b += pc["b_test_only"]; c += pc["c_control_only"]; n += pc["n"]
        entry = {"b_test_only": b, "c_control_only": c, "n_pairs": n, "per_seed": rows, "paired_difference_test_minus_control": ((b - c) / n) if n else None}
        if kind == "E2":
            entry["p_two_sided"] = binomial_two_sided_p(b, c) if n else None
            entry["alpha"] = 0.05
            entry["significant"] = bool(n and entry["p_two_sided"] < 0.05)
            entry["direction"] = ("fewer own citations under the near duplicates" if c > b else "more own citations under the near duplicates" if b > c else "equal")
            entry["estimand"] = "200 near duplicates against 200 distinct facts at equal density (400 entries each side): not comparable with row 48's 56 against 27"
        else:
            entry["tango_95"] = tango_interval(b, c, n) if n else None
            entry["p"] = None; entry["label"] = "reported with counts, paired difference and Tango interval; no p value"
            if kind == "E1":
                entry["named_as"] = ("the contradiction case: two values of one attribute for one entity, the protocol's truth the first write; the current "
                                     "version question (roadmap 4b), never a failure of the write side")
            else:
                entry["cited_pointer_is_own_label"] = "original line cited"
                entry["citation_support_pooled"] = {"b_test_only": sum(rows[r]["citation_support"]["b_test_only"] for r in LIVE_SEEDS if rows.get(r)),
                                                    "c_control_only": sum(rows[r]["citation_support"]["c_control_only"] for r in LIVE_SEEDS if rows.get(r))}
        tests[kind] = entry
    side, metrics, tribunal, order = {}, {}, {}, {}
    for rid in RUN_IDS:
        for arm in ROW49_ARMS:
            f = (files_by_run.get(rid) or {}).get(arm)
            if f is None:
                continue
            key = f"{rid}:{arm}"
            side[key] = side_counts(f)
            side[key]["rates"] = {k: f.get(k) for k in ("hm_recall_on", "hm_recall_off", "hm_gap", "hm_invalid_citation_on", "hm_valid_citation_on",
                                                          "hm_false_abstention_on", "hm_negctrl_rate", "hm_retrieval_hit", "gate_conditions_without_skill_arm")}
            metrics[key] = provenance_metrics(f)
            if arm == PARAPHRASE_ARM:
                from cortex_c2b.hm_protocol import SCHEMAS
                metrics[key]["per_schema"] = {}
                for s in SCHEMAS:
                    sub = {**f, "answers": {**f["answers"], "on": [a for a in f["answers"]["on"] if a.get("schema") == s]}}
                    metrics[key]["per_schema"][s] = {k2: v for k2, v in provenance_metrics(sub).items() if k2 in ("n", "coverage", "value_correct", "citation_supported", "designated_line_cited", "attributed_accuracy", "abstentions")}
            ws = _writes(f)
            adm = [w for w in ws if w.get("admitted")]
            tribunal[key] = {"admitted": len(adm), "under_floor": sum(1 for w in adm if float(w["surprise"]) < TRIBUNAL_FLOOR),
                             "under_floor_by_kind": {k: sum(1 for w in adm if w.get("kind") == k and float(w["surprise"]) < TRIBUNAL_FLOOR)
                                                     for k in sorted({w.get("kind") for w in adm})},
                             "rule": f"an admitted entry whose salience (its surprise) is under {TRIBUNAL_FLOOR} is evicted at the first phase with no credit, as the lifecycle is coded"}
            kind_of = {w["pointer"]: w.get("kind") for w in ws if w.get("pointer")}
            partners = [w for w in ws if w.get("kind") in PARTNER_KINDS and w.get("partner")]
            order[key] = {"partners_whose_nearest_is_not_partner": sum(1 for w in partners if not w["partner"].get("nearest_is_partner")),
                          "partners": len(partners),
                          "originals_whose_nearest_is_a_partner": sum(1 for w in ws if w.get("kind") == "original"
                                                                      and kind_of.get((w.get("nearest_before") or {}).get("pointer")) in PARTNER_KINDS)}
    return {"tests": tests, "side_counts": side, "provenance_metrics": metrics, "tribunal_static": tribunal, "nearest_codes": order,
            "live_cost": {k: v["live_cost"] for k, v in metrics.items()}}


def read_row49(files_by_run: dict, row48_files_by_run: dict, row47_files: dict, expected_commit: str | None = None,
               provenance: dict | None = None, row48_provenance: dict | None = None, row47_provenance: dict | None = None) -> dict:
    """The cold reading of row 49 in the declared order (ADR-007, amendment of 2026-10-14, evening, section 5): P0 to P3, the
    exits first; then F1 under the rule; then E4's planting; then the reported quantities; then the letter. `files_by_run`:
    {run_id: {arm: file}} for the fifteen identity files; `row48_files_by_run`: row 48's twelve files; `row47_files`: row 47's
    system file per checkpoint; the provenances bind the files to the record when given. (gamma) a precondition fails: stop,
    attribute first, read nothing further; (beta) F1 is not zero: stop, attributed, nothing adopted; (alpha) Decision 12,
    worded as declared."""
    p0 = p0_integrity(files_by_run, expected_commit, provenance)
    refs = check_reference_files(row48_files_by_run, row47_files, row48_provenance, row47_provenance)
    out = {"P0": p0, "references": refs, "P1": None, "P2": None, "P3": None, "F1": None, "E4": None, "reported": None, "letter": None, "labels": ROW49_LABELS}
    if not p0["ok"] or not refs["ok"]:
        out.update({"letter": "gamma", "stopped_at": "P0", "attribution_first": (p0["problems"] + refs["problems"])[:10]}); return out
    p1 = p1_thresholds(files_by_run, row47_files); out["P1"] = p1
    if not p1["ok"]:
        out.update({"letter": "gamma", "stopped_at": "P1", "attribution_first": p1["problems"][:10]}); return out
    p2 = p2_controls(files_by_run, row48_files_by_run); out["P2"] = p2
    if not p2["ok"]:
        out.update({"letter": "gamma", "stopped_at": "P2", "attribution_first": p2["problems"][:10]}); return out
    p3 = p3_shadow(files_by_run, row48_files_by_run); out["P3"] = p3
    if not p3["ok"]:
        out.update({"letter": "gamma", "stopped_at": "P3", "attribution_first": p3["problems"][:10]}); return out
    f1 = f1_under_rule(files_by_run); out["F1"] = f1
    if not f1["zero"]:
        out.update({"letter": "beta", "stopped_at": "F1",
                    "attribution_first": [f"{k}: {v['duplicates_refused']} duplicates, {v['paraphrases_refused']} paraphrases refused ({v['reasons']})"
                                          for k, v in f1["per_file"].items() if v["duplicates_refused"] or v["paraphrases_refused"]]})
        return out
    out["E4"] = e4_planting(files_by_run)
    out["reported"] = reported_quantities(files_by_run)
    out["letter"] = "alpha"
    out["decision_12"] = DECISION_12
    return out


# ---- row 50: the replay -----------------------------------------------------------------------------------------------------
def _texts_of(f: dict) -> dict:
    """pointer -> text of every admitted line of the arm's journal, from the planting record, each bound by its hash."""
    from cortex_c2b import content_hash
    out = {}
    for w in _writes(f):
        if w.get("admitted") and w.get("pointer"):
            if content_hash(w["text"].encode("utf-8")) != w["pointer"]:
                raise ValueError(f"planting text not bound by its hash: {w['pointer']}")
            out[w["pointer"]] = w["text"]
    return out


def _lines_of(answer: dict, texts: dict) -> tuple[list[dict], bool]:
    """The window's lines in retrieval order (the recorded order when the file carries it and covers the window, else the
    window sorted by mark, flagged), each with its text; a pointer absent from the planting is a defect of the file."""
    from cortex_c2b.hm_protocol import _TEMPLATES
    window = answer.get("window") or []
    by_pointer = {w["pointer"]: w for w in window}
    order = answer.get("retrieval_order")
    flagged = order is None or set(order) != set(by_pointer) or len(order) != len(window)
    if flagged:
        order = [w["pointer"] for w in sorted(window, key=lambda w: -(w.get("mark") if w.get("mark") is not None else -1.0))]
    lines = []
    for p in order:
        if p not in texts:
            raise ValueError(f"window pointer {p} resolves to no planting text")
        lines.append({"pointer": p, "mark": by_pointer[p].get("mark"), "text": texts[p], "label": by_pointer[p]["label"]})
    return lines, flagged


def question_text_of(answer: dict) -> tuple[str, bool]:
    """The recorded question text; on a file written before row 49 (no `query`) the protocol's direct query rebuilt from the
    generator's template for the schema and entity, flagged: the probe asked exactly that text."""
    from cortex_c2b.hm_protocol import _TEMPLATES
    if answer.get("query"):
        return answer["query"], False
    return _TEMPLATES[answer["schema"]][1].format(e=answer["entity"]), True


def replay_answer(answer: dict, shape: str, threshold: float, texts: dict, own_pointer: str | None, expected_attr: str) -> dict:
    """One answer under one shape, the live pipeline of `_probe` mirrored: the veto surviving citation from the file
    (`veto_cited_pointer`, or the final cited pointer on a file of row 48, where the two coincide since every line there is a
    protocol statement), the shape, then the system's policy `mark-veto+value` through `policy_outcome` on the final label
    (the one veto function live and in replay). The label is the record's when the shape does not act (the live probe keeps
    the model's label on an abstention by the veto or by the value reader), None when the shape abstains, the new line's when
    it changes the citation."""
    from cortex_c2b.hm_lm import policy_outcome
    lines, flagged = _lines_of(answer, texts)
    labels = {ord(l["label"]): l["pointer"] for l in lines}
    mbp = {l["pointer"]: float(l["mark"]) for l in lines if l.get("mark") is not None}
    payloads = {l["pointer"]: l["text"].encode("utf-8") for l in lines}
    veto_cited = answer.get("veto_cited_pointer", answer.get("cited_pointer"))
    question, rebuilt = question_text_of(answer)
    sel = select_citation(shape, veto_cited, lines, threshold, question)
    # the label as the live probe records it: the model's raw label on an abstention by the veto (recomputed from `raw_label` where the
    # file carries it, the record's label on a file of row 48), the window's label of the surviving citation when the shape lets it stand
    # (recomputed from the window), the new line's when the shape changes it, None when the shape abstains
    if veto_cited is None:
        kind, label, out_label = "unknown", None, answer.get("raw_label", answer.get("label"))
    elif sel["pointer"] is None:
        kind, label, out_label = "unknown", None, None
    else:
        label = next(b for b, p in labels.items() if p == sel["pointer"])
        kind, out_label = "cite", chr(label)
    k2, attr, outcome = policy_outcome(kind, label, labels, mbp, payloads, own_pointer, answer.get("raw_answered"), expected_attr, "mark-veto+value", threshold)
    final = labels.get(label) if k2 == "cite" else None
    return {"shape": sel, "kind": k2, "label": out_label, "answered": attr, "outcome": outcome,
            "cited_pointer": final, "cited_pointer_is_own": ((final == own_pointer) if (final is not None and own_pointer) else None),
            "veto_cited_pointer": veto_cited, "retrieval_order_flagged": flagged, "question_rebuilt": rebuilt}


def _both_clear(answer: dict, own: str | None, dp: str | None, thr: float) -> bool:
    marks = {w["pointer"]: w.get("mark") for w in answer.get("window") or []}
    return bool(own in marks and dp is not None and dp in marks and marks[own] is not None and marks[dp] is not None and marks[own] >= thr and marks[dp] >= thr)


def replay_file(f: dict, shapes: tuple = SHAPES) -> dict:
    """Every answer of the on and negctrl blocks and of the reported blocks under every shape, with the own pointer of each
    question (the planted fact's line from the planting record), the partner's pointer (recomputed from the planting), the
    contested flag recomputed from the window's marks, the pair's raw cosine from the planting, and the truth per cited line."""
    texts = _texts_of(f); truth = _truth_of(f); thr = float(f["hm_mark_oracle"]["threshold"])
    ws = _writes(f)
    own_of = {w["for_entity"]: w["pointer"] for w in ws if w.get("kind") == "original" and w.get("admitted")}
    dup_of = {w["for_entity"]: w["pointer"] for w in ws if w.get("kind") in PARTNER_KINDS and w.get("admitted") and w.get("pointer")}
    cos_raw = {w["for_entity"]: (w.get("partner") or {}).get("cos_raw") for w in ws if w.get("kind") in PARTNER_KINDS}
    out = {"on": [], "negctrl": [], "reported": {}, "without_window": {"on": [], "negctrl": []}}
    for arm in ("on", "negctrl"):
        for a in f["answers"].get(arm) or []:
            if not a.get("window"):
                out["without_window"][arm].append(a["entity"]); continue
            own = own_of.get(a["entity"]) if arm == "on" else None
            dp = dup_of.get(a["entity"]) if arm == "on" else None
            row = {"entity": a["entity"], "attr": a["attr"], "own_pointer": own, "duplicate_pointer": dp, "raw_cosine": cos_raw.get(a["entity"]),
                   "both_clear": _both_clear(a, own, dp, thr),
                   "base": {k: a.get(k) for k in ("kind", "label", "answered", "outcome", "cited_pointer", "cited_pointer_is_own", "cited_is_duplicate",
                                                 "both_marks_at_or_above_threshold", "duplicate_in_window", "own_in_window", "own_cited", "citation_support")},
                   "shapes": {}}
            marks = {w["pointer"]: w.get("mark") for w in a["window"]}
            labels_of = {w["pointer"]: w["label"] for w in a["window"]}
            if own in marks and dp in marks and marks[own] is not None and marks[dp] is not None:
                row["margin_own_minus_duplicate"] = float(marks[own]) - float(marks[dp])
                row["own_first_in_window"] = labels_of[own] < labels_of[dp]
            for sh in shapes:
                r = replay_answer(a, sh, thr, texts, own, a["attr"])
                sel_p = r["shape"].get("pointer") if r["veto_cited_pointer"] is not None else None       # the citation surviving the veto and the shape
                r["cited_is_duplicate"] = (dp is not None and sel_p == dp)                                # the file's definition: the surviving citation's pointer
                r["final_is_duplicate"] = (dp is not None and r["cited_pointer"] == dp)
                r["selected_is_duplicate"] = r["cited_is_duplicate"]
                r["selected_is_own"] = (own is not None and sel_p == own)
                t = truth.get(r["cited_pointer"]) if r["cited_pointer"] else None
                r["cited_other_entity"] = (bool(t["entity"] != a["entity"]) if t else None)
                r["cited_wrong_value"] = (bool(t["entity"] == a["entity"] and t["value"] != a["attr"]) if t else None)
                st = truth.get(r["shape"].get("pointer")) if r["shape"].get("pointer") else None
                r["selected_support"] = (bool(st["entity"] == a["entity"] and st["value"] == a["attr"]) if st else None)
                row["shapes"][sh] = r
            row["supporting_line_above_threshold"] = any(float(w["mark"]) >= thr and truth.get(w["pointer"], {}).get("entity") == a["entity"]
                                                         and truth.get(w["pointer"], {}).get("value") == a["attr"] for w in a["window"] if w.get("mark") is not None)
            out[arm].append(row)
    for name, block in ((f.get("reported_blocks") or {}).items()):
        rows = []
        for a in block.get("answers") or []:
            if not a.get("window"):
                continue
            own = dup_of.get(a.get("original_entity")) if name == "neighbour_questions" else None
            row = {"entity": a["entity"], "base": {k: a.get(k) for k in ("kind", "label", "answered", "outcome", "cited_pointer", "cited_pointer_is_own")}, "own_pointer": own,
                   "original_pointer": a.get("original_pointer"), "neighbour_pointer": a.get("neighbour_pointer"), "shapes": {}}
            for sh in shapes:
                r = replay_answer(a, sh, thr, texts, own, a["attr"])
                r["cited_is_original"] = (a.get("original_pointer") is not None and r["cited_pointer"] == a.get("original_pointer"))
                r["cited_neighbour"] = (a.get("neighbour_pointer") is not None and r["cited_pointer"] == a.get("neighbour_pointer"))
                row["shapes"][sh] = r
            rows.append(row)
        out["reported"][name] = rows
    out["truth_rebuilt"] = any(t.get("rebuilt") for t in truth.values())
    return out


VALIDITY_FIELDS = ("kind", "label", "answered", "outcome", "cited_pointer", "cited_pointer_is_own", "cited_is_duplicate", "own_cited", "both_marks_at_or_above_threshold")


def validity_none_reproduces(f: dict) -> dict:
    """The fourth validity condition, on the rows the replay itself produces: under the shape "none" every per question field of
    the on and negctrl blocks and of the reported blocks that the replay recomputes equals the file's (kind, label, answered,
    outcome, cited pointer, the own and duplicate flags, the contested flag), and the "none" counts equal the file's rates."""
    rp = replay_file(f, shapes=("none",))
    diffs, n = [], 0
    for arm in ("on", "negctrl"):
        for i, row in enumerate(rp[arm]):
            n += 1
            r = row["shapes"]["none"]; base = row["base"]
            for k in ("kind", "label", "answered", "outcome", "cited_pointer"):
                if r[k] != base.get(k):
                    diffs.append(f"{arm} {i} ({row['entity']}): {k} replay {r[k]!r} file {base.get(k)!r}")
            if arm == "on":
                if "cited_pointer_is_own" in base and r["cited_pointer_is_own"] != base.get("cited_pointer_is_own"):
                    diffs.append(f"{arm} {i} ({row['entity']}): cited_pointer_is_own replay {r['cited_pointer_is_own']!r} file {base.get('cited_pointer_is_own')!r}")
                if "cited_is_duplicate" in base and base.get("cited_is_duplicate") is not None and r["cited_is_duplicate"] != base.get("cited_is_duplicate"):
                    diffs.append(f"{arm} {i} ({row['entity']}): cited_is_duplicate replay {r['cited_is_duplicate']!r} file {base.get('cited_is_duplicate')!r}")
                if "own_cited" in base and base.get("own_cited") is not None and (r["cited_pointer_is_own"] is True) != bool(base.get("own_cited")):
                    diffs.append(f"{arm} {i} ({row['entity']}): own_cited replay {r['cited_pointer_is_own'] is True} file {base.get('own_cited')!r}")
                if "both_marks_at_or_above_threshold" in base and base.get("both_marks_at_or_above_threshold") is not None and row["both_clear"] != bool(base.get("both_marks_at_or_above_threshold")):
                    diffs.append(f"{arm} {i} ({row['entity']}): both_marks_at_or_above_threshold recomputed {row['both_clear']} file {base.get('both_marks_at_or_above_threshold')!r}")
    for name, rows in rp["reported"].items():
        for i, row in enumerate(rows):
            n += 1
            r = row["shapes"]["none"]; base = row["base"]
            for k in ("kind", "label", "answered", "outcome", "cited_pointer"):
                if r[k] != base.get(k):
                    diffs.append(f"{name} {i} ({row['entity']}): {k} replay {r[k]!r} file {base.get(k)!r}")
    # the "none" counts against the file's rates (the three blocks' rates are the file's; the off arm has no window)
    on = rp["on"]; n_on = max(1, len(f["answers"]["on"]))
    rates = {"hm_recall_on": sum(1 for r in on if r["shapes"]["none"]["outcome"] == "strict") / n_on,
             "hm_invalid_citation_on": sum(1 for r in on if r["shapes"]["none"]["outcome"] == "invalid") / n_on,
             "hm_false_abstention_on": (sum(1 for r in on if r["shapes"]["none"]["kind"] == "unknown") + len(rp["without_window"]["on"])) / n_on,
             "hm_negctrl_rate": sum(1 for r in rp["negctrl"] if r["shapes"]["none"]["kind"] != "unknown") / max(1, len(f["answers"]["negctrl"]))}
    for k, v in rates.items():
        if abs(float(f.get(k) or 0.0) - v) > 1e-12:
            diffs.append(f"count {k}: replay {v} file {f.get(k)}")
    return {"ok": not diffs, "n": n, "differences": diffs[:50], "n_differences": len(diffs),
            "without_window": rp["without_window"], "truth_rebuilt": rp["truth_rebuilt"]}


def _conditions_from_replay(rows_on: list[dict], rows_neg: list[dict], shape: str, f: dict) -> dict:
    """The four frozen conditions under a shape, from the replayed outcomes: the gap (strict recall on; off is zero by
    construction, no window), off at the floor, invalid citations at or under 0.01, never planted claims under 0.10."""
    from cortex_c2b.hm_protocol import DELTA, NEGCTRL_INVALID
    from cortex_c2b.hm_lm import INVALID_CITATION_MAX
    n_on = max(1, len(f["answers"]["on"]))
    strict = sum(1 for r in rows_on if r["shapes"][shape]["outcome"] == "strict")
    invalid = sum(1 for r in rows_on if r["shapes"][shape]["outcome"] == "invalid")
    n_neg = max(1, len(f["answers"]["negctrl"]))
    claims = sum(1 for r in rows_neg if r["shapes"][shape]["kind"] != "unknown")
    off = float(f.get("hm_recall_off") or 0.0)
    gap = strict / n_on - off
    floor = float((f.get("thresholds") or {}).get("floor") or 0.0)
    hold = (gap >= DELTA) and (off <= floor) and (invalid / n_on <= INVALID_CITATION_MAX) and (claims / n_neg < NEGCTRL_INVALID)
    return {"strict": strict, "invalid": invalid, "never_planted_claims": claims, "gap": gap, "off": off, "hold": bool(hold),
            "own_cited": sum(1 for r in rows_on if r["shapes"][shape]["cited_pointer_is_own"] is True)}


COSINE_BINS = ((0.0, 0.99), (0.99, 0.995), (0.995, 0.999), (0.999, 1.0001))


def _margins_by_cosine(rows: list[dict]) -> dict:
    out = {}
    for lo, hi in COSINE_BINS:
        xs = [r["margin_own_minus_duplicate"] for r in rows if r.get("margin_own_minus_duplicate") is not None and r.get("raw_cosine") is not None and lo <= float(r["raw_cosine"]) < hi]
        out[f"[{lo}, {hi})"] = _quantiles(xs)
    return out


def _file_shape_quantities(rp: dict, f: dict, shapes: tuple, texts: dict) -> dict:
    per = {}
    contested = [r for r in rp["on"] if r["both_clear"]]
    for sh in shapes:
        dup_cited = sum(1 for r in contested if r["shapes"][sh]["cited_is_duplicate"])
        claims = [r for r in rp["on"] if r["shapes"][sh]["kind"] != "unknown"]
        abst = [r for r in rp["on"] if r["shapes"][sh]["kind"] == "unknown"]
        earlier = [r for r in contested if r.get("own_first_in_window") is not None and r["shapes"][sh]["cited_pointer"] in (r["own_pointer"], r["duplicate_pointer"])]
        earlier_cited = sum(1 for r in earlier if (r["own_first_in_window"] and r["shapes"][sh]["cited_pointer"] == r["own_pointer"])
                            or (not r["own_first_in_window"] and r["shapes"][sh]["cited_pointer"] == r["duplicate_pointer"]))
        per[sh] = {"contested": len(contested), "duplicate_cited_on_contested": dup_cited, "duplicate_cited_wilson_95": wilson(dup_cited, len(contested)),
                   "own_cited_on_contested": sum(1 for r in contested if r["shapes"][sh]["cited_pointer_is_own"] is True),
                   "abstained_on_contested": sum(1 for r in contested if r["shapes"][sh]["kind"] == "unknown"),
                   "earlier_line_cited_on_contested": {"cited_one_of_the_two": len(earlier), "earlier": earlier_cited},
                   "own_cited": sum(1 for r in rp["on"] if r["shapes"][sh]["cited_pointer_is_own"] is True),
                   "selected_own": sum(1 for r in rp["on"] if r["shapes"][sh]["selected_is_own"]),
                   "selected_duplicate": sum(1 for r in rp["on"] if r["shapes"][sh]["selected_is_duplicate"]),
                   "selected_supported": sum(1 for r in rp["on"] if r["shapes"][sh]["selected_support"] is True),
                   "coverage": len(claims) / max(1, len(f["answers"]["on"])),
                   "risk_among_claims": {"other_entity": sum(1 for r in claims if r["shapes"][sh]["cited_other_entity"]), "wrong_value": sum(1 for r in claims if r["shapes"][sh]["cited_wrong_value"]), "claims": len(claims)},
                   "abstentions": {"false": sum(1 for r in abst if r["supporting_line_above_threshold"]), "verified": sum(1 for r in abst if not r["supporting_line_above_threshold"])},
                   "conditions": _conditions_from_replay(rp["on"], rp["negctrl"], sh, f)}
    per["margins_by_raw_cosine"] = _margins_by_cosine(rp["on"])
    per["contested_margins_by_raw_cosine"] = _margins_by_cosine(contested)
    for name, rows in rp["reported"].items():
        per[f"reported:{name}"] = {sh: {"n": len(rows), "claims": sum(1 for r in rows if r["shapes"][sh]["kind"] != "unknown"),
                                        "own_cited": sum(1 for r in rows if r["shapes"][sh]["cited_pointer_is_own"] is True),
                                        "cited_is_original": sum(1 for r in rows if r["shapes"][sh]["cited_is_original"]),
                                        "cited_neighbour": sum(1 for r in rows if r["shapes"][sh]["cited_neighbour"])} for sh in shapes}
    if "R4" in shapes and "R1" in shapes:
        dis = [r for r in rp["on"] if r["shapes"]["R4"]["cited_pointer"] != r["shapes"]["R1"]["cited_pointer"]]
        unreadable = sum(1 for r in dis if r["shapes"]["R1"]["cited_pointer"] is not None and parse_statement(texts.get(r["shapes"]["R1"]["cited_pointer"], "")) is None)
        per["R4_against_R1"] = {"disagreements": len(dis),
                                "attributed": {"ranking_error_of_the_mark": sum(1 for r in dis if r["shapes"]["R1"]["cited_other_entity"]),
                                               "parse_failure": unreadable,
                                               "R4_abstained_on_conflict": sum(1 for r in dis if r["shapes"]["R4"]["kind"] == "unknown" and (r["shapes"]["R4"]["shape"].get("values") or 0) >= 2),
                                               "R4_abstained_no_candidate": sum(1 for r in dis if r["shapes"]["R4"]["kind"] == "unknown" and (r["shapes"]["R4"]["shape"].get("candidates") == 0)),
                                               "R4_lifted_a_value_reader_abstention": sum(1 for r in dis if r["shapes"]["R1"]["kind"] == "unknown" and r["shapes"]["R4"]["kind"] != "unknown")}}
    return per


def _decision_13(per_file: dict, replays: dict, files_by_run: dict) -> dict:
    """Decision 13 as declared (section 6): (i) superiority on E2's contested reads, the live seeds pooled, one sided exact McNemar
    at 0.025 with at least six discordant pairs all one way; (ii) no loss on B0 and E0d on each live seed (the four conditions
    under R4, own citations not fewer); the two as one intersection union test; (iii) the closure, E2 against E0d both under R4,
    paired by entity over the live questions, the one sided 97.5 % upper bound of the paired loss by Tango under 0.02; "F2
    persists" by the exact two sided test in the loss direction, computed whatever the adoption, with its split; seed 1337's
    counts reported beside."""
    per_seed = {}
    for rid in RUN_IDS:
        e2 = replays.get(f"{rid}:E2"); e0 = replays.get(f"{rid}:E0d")
        if e2 is None or e0 is None:
            per_seed[rid] = None; continue
        b = c = 0
        for r in e2["on"]:
            if not r["both_clear"]:
                continue
            base_own = r["shapes"]["none"]["cited_pointer_is_own"] is True; r4_own = r["shapes"]["R4"]["cited_pointer_is_own"] is True
            b += int(r4_own and not base_own); c += int(base_own and not r4_own)
        by_entity = {r["entity"]: r for r in e0["on"]}
        bb = cc = nn = 0; unpaired = []
        for y in e2["on"]:
            x = by_entity.get(y["entity"])
            if x is None:
                unpaired.append(y["entity"]); continue
            ox = x["shapes"]["R4"]["cited_pointer_is_own"] is True; oy = y["shapes"]["R4"]["cited_pointer_is_own"] is True
            bb += int(ox and not oy); cc += int(oy and not ox); nn += 1
        per_seed[rid] = {"contested": sum(1 for r in e2["on"] if r["both_clear"]), "b_R4_only": b, "c_base_only": c,
                         "closure": {"loss_pairs_control_only": bb, "gain_pairs_test_only": cc, "n_paired": nn, "unpaired": unpaired,
                                     "without_window": {"E2": e2["without_window"]["on"], "E0d": e0["without_window"]["on"]}}}
    live = [rid for rid in LIVE_SEEDS if per_seed.get(rid) is not None]
    if len(live) != len(LIVE_SEEDS):
        return {"unread": "E2 or E0d missing on a live seed", "per_seed": per_seed}
    b = sum(per_seed[r]["b_R4_only"] for r in live); c = sum(per_seed[r]["c_base_only"] for r in live)
    contested_total = sum(per_seed[r]["contested"] for r in live)
    p1 = mcnemar_one_sided_p(b, c)
    superiority = bool(p1 <= SUPERIORITY_ALPHA and b >= MIN_DISCORDANT and b > c)
    controls, no_loss = {}, True
    for rid in live:
        for arm in ("B0", "E0d"):
            pf = per_file.get(f"{rid}:{arm}")
            if pf is None:
                return {"unread": f"{rid}:{arm} missing", "per_seed": per_seed}
            hold = pf["R4"]["conditions"]["hold"]; not_fewer = pf["R4"]["own_cited"] >= pf["none"]["own_cited"]
            controls[f"{rid}:{arm}"] = {"four_conditions_hold": hold, "own_cited_R4": pf["R4"]["own_cited"], "own_cited_base": pf["none"]["own_cited"], "not_fewer": not_fewer}
            no_loss = no_loss and hold and not_fewer
    adopted = superiority and no_loss
    bb = sum(per_seed[r]["closure"]["loss_pairs_control_only"] for r in live); cc = sum(per_seed[r]["closure"]["gain_pairs_test_only"] for r in live)
    nn = sum(per_seed[r]["closure"]["n_paired"] for r in live)
    expected_n = sum(len(files_by_run[r]["E2"]["answers"]["on"]) for r in live)
    loss_upper = tango_interval(bb, cc, nn)[1] if nn else None
    p_two = binomial_two_sided_p(bb, cc) if nn else None
    persists = bool(p_two is not None and p_two < 0.05 and bb > cc)
    closed = bool(adopted and loss_upper is not None and loss_upper < CLOSURE_BOUND and nn == expected_n)
    return {"superiority": {"contested_reads": contested_total, "b_R4_only": b, "c_base_only": c, "p_one_sided": p1, "alpha": SUPERIORITY_ALPHA,
                            "min_discordant_one_way": MIN_DISCORDANT, "holds": superiority},
            "no_loss": {"per_control": controls, "holds": no_loss},
            "adopted": adopted, "intersection_union": "both (i) and (ii), no multiplicity adjustment (Berger 1982)",
            "closure": {"loss_pairs_control_only": bb, "gain_pairs_test_only": cc, "n_paired": nn, "n_expected": expected_n,
                        "tango_upper_97_5": loss_upper, "bound": CLOSURE_BOUND, "p_two_sided": p_two,
                        "F2_closed_under_R4": closed, "F2_persists_under_R4": persists,
                        "split_per_seed": {r: per_seed[r]["closure"] for r in live},
                        "sentence": ("F2 closed under R4" if closed else "F2 persists under R4" if persists else
                                     ("neither closed nor shown to persist: a non rejection closes nothing (Altman and Bland 1995)" if adopted else "nothing adopted"))},
            "per_seed": per_seed, "reported_seed": {r: per_seed.get(r) for r in CONCEIVING_SEED},
            "if_not_adopted": (None if adopted else "nothing is adopted; the collision stays attributed to the decision; the v1.2 options are named: R0, the key through B2 and the extractor, a calibrated abstention margin")}


def replay_row50(files_by_run: dict, shapes: tuple = SHAPES, post_hoc: bool = False, row49_letter: str | None = None) -> dict:
    """The read side selection replayed on row 49's identity files (or, labelled post hoc, on row 48's legacy files): every file
    of the arms present and not stopped must pass the validity conditions, and on row 49 all fifteen files must be present and
    row 49 read (alpha) before a shape quantity is computed; every answer replayed under every shape; the quantities per shape
    and seed; then Decision 13 and the closure sentence on the identity files alone (never on the post hoc replay)."""
    arms = ROW48_ARMS if post_hoc else ROW49_ARMS
    missing = [f"{rid}:{arm}" for rid in RUN_IDS for arm in arms if (files_by_run.get(rid) or {}).get(arm) is None or (files_by_run[rid][arm] or {}).get("stopped")]
    out = {"post_hoc": post_hoc, "label": ("post hoc: the shapes were conceived on these files; a consistency check of the machinery; the question texts and the "
                                            "truth are rebuilt from the generator (the files of row 48 carry neither); no decision is computed" if post_hoc else ROW50_LABEL),
           "shapes": {s: SHAPE_WHAT[s] for s in shapes}, "validity": {}, "valid": False, "per_file": {}, "decision_13": None}
    if missing:
        out["unread"] = "files missing or stopped: " + ", ".join(missing); return out
    if not post_hoc and row49_letter != "alpha":
        out["unread"] = f"row 49 is not read alpha (letter {row49_letter!r}): the replay reads nothing further"; return out
    replays = {}
    for rid in RUN_IDS:
        for arm in arms:
            f = files_by_run[rid][arm]; key = f"{rid}:{arm}"
            try:
                out["validity"][key] = validity_none_reproduces(f)
                replays[key] = replay_file(f, shapes)
            except ValueError as e:
                out["validity"][key] = {"ok": False, "error": str(e)}
    out["valid"] = all(v.get("ok") for v in out["validity"].values())
    if not out["valid"]:
        out["unread"] = "a validity condition fails on a file: " + "; ".join(k for k, v in out["validity"].items() if not v.get("ok")); return out
    for key, rp in replays.items():
        rid, arm = key.split(":")
        out["per_file"][key] = _file_shape_quantities(rp, files_by_run[rid][arm], shapes, _texts_of(files_by_run[rid][arm]))
    if not post_hoc and "R4" in shapes and "none" in shapes:
        out["decision_13"] = _decision_13(out["per_file"], replays, files_by_run)
    return out


def live_r4_against_replay(live: dict, identity_e2: dict) -> dict:
    """Row 50's precedent of row 47: the live R4 run (E2, identity, seed 2024, a fresh planting) must reproduce the replay's R4
    question by question. The inputs are compared first (the planting's writes: texts, pointers, decisions, surprises within
    1e-6; the windows' labels, pointers and marks; the raw parse and the veto surviving citation), then the outcomes (kind,
    label, answered, outcome, cited pointer) on the on and negctrl blocks and on both reported blocks; the two files' sha256 are
    recorded."""
    diffs = []
    wl, wi = _writes(live), _writes(identity_e2)
    if len(wl) != len(wi):
        diffs.append(f"planting: {len(wl)} writes against {len(wi)}")
    for i, (a, b) in enumerate(zip(wl, wi)):
        if (a.get("text"), a.get("pointer"), a.get("admitted"), a.get("legacy_decision")) != (b.get("text"), b.get("pointer"), b.get("admitted"), b.get("legacy_decision")):
            diffs.append(f"planting write {i}: text, pointer or decision differ")
        if not _close(a.get("surprise"), b.get("surprise")):
            diffs.append(f"planting write {i}: surprise {a.get('surprise')} against {b.get('surprise')}")
    if not _close(live["hm_mark_oracle"]["threshold"], identity_e2["hm_mark_oracle"]["threshold"]):
        diffs.append("the oracle's threshold differs")
    rp = replay_file(identity_e2, shapes=("R4",))
    blocks = [("on", live["answers"]["on"], identity_e2["answers"]["on"], rp["on"]), ("negctrl", live["answers"]["negctrl"], identity_e2["answers"]["negctrl"], rp["negctrl"])]
    for name in ("near_name_never_planted", "neighbour_questions"):
        la = ((live.get("reported_blocks") or {}).get(name) or {}).get("answers"); ia = ((identity_e2.get("reported_blocks") or {}).get(name) or {}).get("answers")
        if la is not None and ia is not None:
            blocks.append((name, la, ia, rp["reported"].get(name) or []))
    n = 0
    for name, la, ia, rows in blocks:
        if len(la) != len(ia):
            diffs.append(f"{name}: {len(la)} live answers against {len(ia)}"); continue
        rows_by_entity = {r["entity"]: r for r in rows}
        for i, (a, b) in enumerate(zip(la, ia)):
            if [(w["label"], w["pointer"]) for w in a.get("window") or []] != [(w["label"], w["pointer"]) for w in b.get("window") or []] or \
                    any(not _close(p.get("mark"), q.get("mark")) for p, q in zip(a.get("window") or [], b.get("window") or [])):
                diffs.append(f"{name} {i} ({a.get('entity')}): the window differs"); continue
            if (a.get("raw_kind"), a.get("raw_label"), a.get("raw_answered"), a.get("veto_cited_pointer")) != (b.get("raw_kind"), b.get("raw_label"), b.get("raw_answered"), b.get("veto_cited_pointer")):
                diffs.append(f"{name} {i} ({a.get('entity')}): the raw parse or the veto surviving citation differ"); continue
            r = rows_by_entity.get(a["entity"])
            if r is None:
                continue                                                                 # no window: nothing to replay, the live file abstained too
            n += 1
            x = r["shapes"]["R4"]
            for k in ("kind", "label", "answered", "outcome", "cited_pointer"):
                if x[k] != a.get(k):
                    diffs.append(f"{name} {i} ({a.get('entity')}): {k} live {a.get(k)!r} against replay {x[k]!r}")
    return {"ok": not diffs, "differences": diffs[:50], "n_differences": len(diffs), "n_compared": n,
            "inputs_sha256": {"live": live.get("_sha256"), "identity_E2": identity_e2.get("_sha256")}}


def load_row49_from_dir(metrics_dir) -> dict:
    """The reading's inputs from a directory of retained files: the fifteen identity files, row 48's twelve files, row 47's three
    system files, the live R4 file, and the two provenance files of the record (rows 48 and 47); each file carries its sha256
    under `_sha256` (the LF normalised bytes, sixteen hexadecimal characters, the record's form)."""
    import hashlib
    d = Path(metrics_dir)

    def load(name):
        q = d / name
        if not q.exists():
            return None
        raw = q.read_bytes()
        f = json.loads(raw.decode("utf-8"))
        f["_sha256"] = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()[:16]
        return f
    files = {rid: {arm: load(ROW49_FILE.format(rid=rid, arm=arm)) for arm in ROW49_ARMS} for rid in RUN_IDS}
    row48 = {rid: {arm: load(ROW48_FILE.format(rid=rid, arm=arm)) for arm in ROW48_ARMS} for rid in RUN_IDS}
    row47 = {rid: load(ROW47_FILE.format(rid=rid)) for rid in RUN_IDS}
    live = {rid: load(ROW50_LIVE_FILE.format(rid=rid)) for rid in RUN_IDS}
    return {"files_by_run": files, "row48_files_by_run": row48, "row47_files": row47, "live_r4": live,
            "row48_provenance": load("PROVENANCE-near-duplicates.json"), "row47_provenance": load("PROVENANCE-adoption.json")}


def read_row49_from_dir(metrics_dir, expected_commit: str, provenance: dict | None = None) -> dict:
    """The cold reading of row 49 from a directory of retained files; the build's commit is required (P0), the fifteen files'
    provenance (name to sha256) checked when given, the reference files bound to the record's provenance files."""
    if not expected_commit:
        raise ValueError("the build's commit is required: P0 checks the files' commit against it")
    d = load_row49_from_dir(metrics_dir)
    return read_row49(d["files_by_run"], d["row48_files_by_run"], d["row47_files"], expected_commit=expected_commit, provenance=provenance,
                      row48_provenance=d["row48_provenance"], row47_provenance=d["row47_provenance"])


def replay_row50_from_dir(metrics_dir, expected_commit: str, provenance: dict | None = None) -> dict:
    """Row 50 from the same directory, after row 49's reading (the replay reads nothing unless row 49 is alpha): the replay on
    row 49's identity files, the post hoc replay on row 48's legacy files, and the live R4 run against the replay's R4."""
    d = load_row49_from_dir(metrics_dir)
    r49 = read_row49(d["files_by_run"], d["row48_files_by_run"], d["row47_files"], expected_commit=expected_commit, provenance=provenance,
                     row48_provenance=d["row48_provenance"], row47_provenance=d["row47_provenance"])
    out = {"row49_letter": r49.get("letter"), "row49": replay_row50(d["files_by_run"], row49_letter=r49.get("letter")),
           "row48_post_hoc": replay_row50(d["row48_files_by_run"], post_hoc=True), "live_r4": {}}
    for rid, live in d["live_r4"].items():
        e2 = (d["files_by_run"].get(rid) or {}).get("E2")
        out["live_r4"][rid] = (live_r4_against_replay(live, e2) if (live is not None and e2 is not None) else None)
    return out
