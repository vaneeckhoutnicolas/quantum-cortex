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
from cortex_c2b.near_duplicate_probe import ARMS, KINDS, THRESHOLD_TOL, LIVE_N_TABLES

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
        for arm in ARMS:
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
        for arm in ARMS:
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
        for arm in ARMS:
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
