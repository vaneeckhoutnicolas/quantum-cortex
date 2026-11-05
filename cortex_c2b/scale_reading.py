"""cortex_c2b.scale_reading -- row 53's cold reading of part C's files (ADR-010 C.4 and the amendment's section 3): the
files `cortex_c2b.scale_probe` retained under `metrics/mqar/` (`scale-encoding-<draw>-<run id>.json`,
`scale-c2-<draw>-<run id>-<gate>.json`, `scale-cascade-<draw>-<run id>-<gate>.json`), read from the files alone, no
model, no key, seconds; the two conditions of C.4 by `scale_probe.read_c4` (the withdrawal under cascade on every journal
and phase, read first; the never planted entities above the threshold at each size, with their intervals and
attribution, under their label); the Decision 13 successor sentence filled in from the pairs (C.2 (b)); the cost curves
reported with their construction, no threshold to pass; every number per draw, encoder and gate, then pooled per size,
encoder and gate with the attained bounds. A cascade's progress file, when the founder's copy is at hand (never
committed; its sha256 in the final file), gives what the final file pools: the sample reads (entries not yet evicted)
apart from the pair reads, the states at every phase, the checks; it is read from a copy, since `resumable.Progress`
repairs a torn last line in place, and a reader owns none of the founder's files. The expected set of the record is the
three encoders' five ten thousand draws under both gates and the hundred thousand draw on the encoder of seed 2024 (C.2
under both gates, the cascade under the identity rule, D3 A); a missing file is listed, never guessed; a final file of
the probe under another name (a refused run retained, a second run; C.3 (b)) is listed apart and read when named
(`--also`), never pooled with the record's; a progress file with no final file (unfinished, refused) is read for C.4's
fault condition alone (`--journals`, `--progress-dir`, `--progress-file`). Nothing here is a number of the record before
the founder runs it; the reading file is never overwritten.

    python -m cortex_c2b.scale_reading --metrics metrics\\mqar [--journals runs\\scale | --progress-dir <dir>]
                                      [--progress-file <path> ...] [--also <final file> ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import statistics
import tempfile
from collections import Counter
from pathlib import Path

from cortex_c2b import scale_probe as SP
from cortex_c2b import resumable as RS
from cortex_c2b.sealed_reading import clopper_pearson

RUN_IDS = ("8ceba5db8d0b", "4cfd6f6e1406", "2e1dc71913b1")          # the record's order: seeds 1337, 2024, 7
SEED_OF = {"8ceba5db8d0b": 1337, "4cfd6f6e1406": 2024, "2e1dc71913b1": 7}
TEN_THOUSAND_DRAWS = (50000, 50001, 50002, 50003, 50004)
HUNDRED_THOUSAND_DRAW = 60000
HUNDRED_THOUSAND_RUN = "4cfd6f6e1406"                                  # C.1: one journal draw, one encoder (seed 2024)
GATES = ("identity", "surprise")
SIZES = {"ten_thousand": TEN_THOUSAND_DRAWS, "hundred_thousand": (HUNDRED_THOUSAND_DRAW,)}
READING_NAME = "scale-reading-row53.json"
THRESHOLD_TOLERANCE = 1e-6
EXPECTED_PHASES = {"ten_thousand": 687, "hundred_thousand": 6875}      # C.1 (a): under the identity rule
FILE_RE = re.compile(r"^scale-(encoding|c2|cascade)-(\d+)-([0-9a-f]{12})(?:-(identity|surprise))?\.json$")


# ---------------------------------------------------------------------------- #
# the expected files and the present ones                                        #
# ---------------------------------------------------------------------------- #
def expected_files() -> list[dict]:
    """The record's set, in the order of G: per encoder the five ten thousand draws (the encoding, C.2 under both gates,
    the cascade under both gates), then the hundred thousand draw on seed 2024's encoder (the encoding, C.2 under both
    gates, the cascade under the identity rule)."""
    out = []
    for r in RUN_IDS:
        for d in TEN_THOUSAND_DRAWS:
            out.append({"name": f"scale-encoding-{d}-{r}.json", "kind": "encoding", "draw": d, "run_id": r, "gate": None, "size": "ten_thousand"})
            for g in GATES:
                out.append({"name": f"scale-c2-{d}-{r}-{g}.json", "kind": "c2", "draw": d, "run_id": r, "gate": g, "size": "ten_thousand"})
            for g in GATES:
                out.append({"name": f"scale-cascade-{d}-{r}-{g}.json", "kind": "cascade", "draw": d, "run_id": r, "gate": g, "size": "ten_thousand"})
    d, r = HUNDRED_THOUSAND_DRAW, HUNDRED_THOUSAND_RUN
    out.append({"name": f"scale-encoding-{d}-{r}.json", "kind": "encoding", "draw": d, "run_id": r, "gate": None, "size": "hundred_thousand"})
    for g in GATES:
        out.append({"name": f"scale-c2-{d}-{r}-{g}.json", "kind": "c2", "draw": d, "run_id": r, "gate": g, "size": "hundred_thousand"})
    out.append({"name": f"scale-cascade-{d}-{r}-identity.json", "kind": "cascade", "draw": d, "run_id": r, "gate": "identity", "size": "hundred_thousand"})
    return out


def sha256_lf(path: Path, n: int = 16) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:n]


def load_files(md: Path) -> dict:
    """Every expected file present under `md` loaded, the missing ones listed, every other `scale-*.json` under `md` but
    the reading files listed apart (a rerun under another name, a refused run retained with a suffix: never read as the
    record's; read apart when named by `--also`)."""
    expected = expected_files()
    names = {e["name"] for e in expected}
    present, missing = {}, []
    for e in expected:
        p = md / e["name"]
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            present[e["name"]] = {**e, "sha256_lf": sha256_lf(p), "data": d}
        else:
            missing.append(e["name"])
    outside = sorted(p.name for p in md.glob("scale-*.json") if p.name not in names and not p.name.startswith("scale-reading-"))
    return {"present": present, "missing": missing, "outside_the_expected_set": outside, "expected": [e["name"] for e in expected]}


def kind_of_final(d: dict) -> str | None:
    """The kind of a final file of the probe from its content: the encoding (a cue manifest), C.2 (a planting and the
    reads' aggregates), the cascade (phases and faults)."""
    if "batched_against_single" in d and "files" in d:
        return "encoding"
    if "planting" in d and "aggregates" in d:
        return "c2"
    if "phases" in d and "faults" in d and "reads_by_phase" in d:
        return "cascade"
    return None


# ---------------------------------------------------------------------------- #
# the digests per file                                                           #
# ---------------------------------------------------------------------------- #
def _dist(values: list[float]) -> dict | None:
    if not values:
        return None
    xs = sorted(values)
    return {"n": len(xs), "min": xs[0], "max": xs[-1], "median": statistics.median(xs)}


def encoding_digest(d: dict) -> dict:
    b = d["batched_against_single"]
    return {"draw": d["draw"], "run_id": d["run_id"], "size": d["size"], "n_statements": d["n_statements"], "n_questions": d["n_questions"], "batch_size": d["batch_size"],
            "largest_abs_cue_difference": b["largest_abs_cue_difference"], "components_rounding_changed": b["components_rounding_changed"], "components": b["components"],
            "signatures_flipped": b["signatures_flipped"], "signatures": b["signatures"], "repeated_names": sorted(d.get("repeated_names") or {}),
            "plan_content_sha256": d["plan_content_sha256"], "sessions": len(d["sessions"]), "session_reasons": [s["reason"] for s in d["sessions"]],
            "start": d["sessions"][0]["start"], "end": d["sessions"][-1]["end"], "threads": d["device"].get("threads"), "timestamp_utc": d["timestamp_utc"],
            "code_commit": d["sessions"][0].get("code_state", {}).get("commit"),
            "freeze_bound": bool(d["identity"]["compared"].get("freeze_sha256")), "pythonhashseed": d["identity"]["compared"].get("environment", {}).get("PYTHONHASHSEED")}


def c2_digest(d: dict) -> dict:
    p, a, t, s = d["planting"], d["aggregates"], d["time_per_query"], d["aggregates"]["sample"]
    nn, e1, e2, kc = a["never_planted"], a["pairs_E1"], a["pairs_E2"], a["key_contract"]

    def pairs(e):
        return {"n": e["n"], "duplicate_admitted": e["duplicate_admitted"], "own_in_window": e["own_in_window"], "duplicate_in_window": e["duplicate_in_window"],
                "contested": e["contested"], "contested_clopper_pearson_95": e["contested_clopper_pearson_95"],
                "mark_ranks_duplicate_first": e["mark_ranks_duplicate_first"], "r1_error_clopper_pearson_95": e["r1_error_clopper_pearson_95"],
                "r4_outcome_by_construction": e["r4_outcome_by_construction"], "r4_not_following": e["r4_not_following"],
                "selection_counts": e["selection_counts"], "margins_by_raw_cosine": e["margins_by_raw_cosine"],
                "repeated_name_pairs_apart": e.get("repeated_name_pairs_apart"), "repeated_name_pairs": e.get("repeated_name_pairs")}

    def timing(block):
        return {ix: {"median_ns": block[ix]["median"], "p95_ns": block[ix]["p95"], "min_ns": block[ix]["min"], "spread_of_medians_ns": block[ix]["spread_of_medians"],
                     "flag": block[ix]["flag"], "n": block[ix]["n"]} for ix in ("I0", "I1", "I2")}

    return {"draw": d["draw"], "run_id": d["run_id"], "gate": d["gate"], "size": d["size"], "entries": d["entries"], "k": d["k"], "bm25": d["bm25"],
            "planting": {"admitted": p["admitted"], "refused": p["refused"], "entries": p["entries"], "shadow": p["shadow"],
                         "by_kind": {k_: {"n": v["n"], "admitted": v["admitted"], "under_floor_0.2": v["salience"]["under_floor_0.2"], "salience_median": v["salience"]["median"]} for k_, v in p["by_kind"].items()},
                         "phases": p["phases"], "sentinels": p["sentinels"], "run_in_sessions": p["run_in_sessions"], "resumption_fault": p["resumption_fault"]},
            "threshold": {"value": d["threshold"]["value"], "row47": d["threshold"]["row47"], "within_1e-6": d["threshold"]["within_1e-6"]},
            "sample": {"n": s["n"], "own_in_window": {"I0": s["i0"]["own_in_window"], "I1": s["i1"]["own_in_window"], "I2": s["i2"]["own_in_window"], **{k_: v["own_in_window"] for k_, v in s["levers"].items()}},
                       "i0_candidates": s["i0"]["candidates"], "i0_own_in_candidates": s["i0"]["own_in_candidates"], "i0_ties_at_boundary": s["i0"]["ties_at_boundary"],
                       "i1_compared": s["i1"]["compared"], "i2_postings_read": s["i2"]["postings_read"], "agree": s["agree"],
                       "lever_candidates": {k_: v["candidates"] for k_, v in s["levers"].items()}, "repeated_names_in_sample": s.get("repeated_names_in_sample")},
            "never_planted": {"n": nn["n"], "at_or_above": nn["max_mark_at_or_above_threshold"], "clopper_pearson_95": nn["clopper_pearson_95"], "by_distance": nn["by_distance"],
                              "ties_at_boundary": nn["ties_at_boundary"], "candidates": nn["candidates"]},
            "pairs_E1": pairs(e1), "pairs_E2": pairs(e2),
            "key_contract": {k_: v for k_, v in kc.items() if k_ != "label"},
            "time_per_query": {"n_queries": t["n_queries"], "repetitions": t["repetitions"], "interleaved": timing(t["interleaved"]), "blocked": timing(t["blocked"]),
                               "flagged_interleaved": t["flagged_interleaved"], "flagged_blocked": t["flagged_blocked"], "timer_overhead_ns": t["timer_overhead_ns"],
                               "torch_threads": (t.get("threads") or {}).get("torch_threads")},
            "index_sizes": d["index_sizes"], "chance_near_twins": d["chance_near_twins"].get("n_repeated_names"), "n_reads": d["n_reads"],
            "sessions": len(d["sessions"]), "session_reasons": [x["reason"] for x in d["sessions"]], "start": d["sessions"][0]["start"], "end": d["sessions"][-1]["end"],
            "code_commit": d.get("code_commit"), "environment_torch_threads": d["environment"]["torch_threads"], "pythonhashseed": d["environment"].get("PYTHONHASHSEED"),
            "freeze_bound": bool(d["identity"]["compared"].get("freeze_sha256")), "timestamp_utc": d["timestamp_utc"]}


def _linear_fit(ys: list[float]) -> dict | None:
    """The least squares line of the phase times against the phase number (the lifecycle's law: a phase's cost growing
    with the journal, so a linear slope in the phase and a quadratic total), reported with its construction."""
    n = len(ys)
    if n < 2:
        return None
    xs = list(range(1, n + 1))
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    icpt = my - slope * mx
    ss_res = sum((y - (icpt + slope * x)) ** 2 for x, y in zip(xs, ys)); ss_tot = sum((y - my) ** 2 for y in ys)
    return {"slope_seconds_per_phase": slope, "intercept_seconds": icpt, "r2": (1 - ss_res / ss_tot) if ss_tot else None, "n": n}


def cascade_digest(d: dict) -> dict:
    pt, pr = d["phase_times_seconds"], d["probe_times_seconds"]
    nd = d["near_duplicates"]["by_kind"]
    recon = {ph: {"reconstructions": v["reconstructions"], "patterns": v["patterns"], "evicted_addresses": v["evicted_address_reading"]["entries"],
                  "still_reconstructed_under_bound": v["evicted_address_reading"]["still_reconstructed_under_bound"],
                  "min_defect_evicted": v["evicted_address_reading"].get("min_defect_evicted"), "max_defect_evicted": v["evicted_address_reading"].get("max_defect_evicted")}
             for ph, v in d["reconstruction_by_phase"].items() if v}
    return {"draw": d["draw"], "run_id": d["run_id"], "gate": d["gate"], "size": d["size"], "lifecycle_config_hash": d["lifecycle_config"]["hash"], "k": d["k"],
            "phases": d["phases"], "expected_phases_identity": d.get("expected_phases_identity"), "entries": d["entries"], "sentinels": d["sentinels"], "admitted": d["admitted"],
            "states": d["states"], "faults": d["faults"], "phases_refused": d["phases_refused"], "replay_failed_total": d["replay_failed_total"],
            "applied_totals": d["applied_totals"], "summaries": d["summaries"],
            "near_duplicates": {k_: {x: v[x] for x in ("admitted", "evicted", "live", "consolidated", "demoted", "read_before_eviction", "grouped_with_partner")} for k_, v in nd.items()},
            "near_duplicates_label": ("read_before_eviction as the final file counts it: over the sample reads alone (the pair lines carry the pair's kind over the line kind "
                                      "`read` and the probe's reader does not see them); the count over every read line comes from the progress file"),
            "reads_by_phase": d["reads_by_phase"], "reconstruction_by_phase": recon, "read_phases": d["read_phases"],
            "n_read_lines": d["n_read_lines"], "n_read_lines_label": "the sample reads' lines alone (the same cause); reads_by_phase is complete",
            "phase_times": {"n": len(pt), "sum_seconds": sum(pt), "first": pt[0] if pt else None, "last": pt[-1] if pt else None, "max": max(pt) if pt else None,
                            "median": statistics.median(pt) if pt else None, "fit": _linear_fit(pt), "label": d["phase_times_label"]},
            "probe_times": {"sum_seconds": sum(pr), "max": max(pr) if pr else None},
            "sessions": len(d["sessions"]), "session_reasons": [s["reason"] for s in d["sessions"]], "start": d["sessions"][0]["start"], "end": d["sessions"][-1]["end"],
            "run_in_sessions": d["run_in_sessions"], "resumption_fault": d["resumption_fault"], "read_lines_recorded_twice_compared": d["read_lines_recorded_twice_compared"],
            "progress_sha256": d["progress_sha256"], "code_commit": d.get("code_commit"), "freeze_bound": bool(d["identity"]["compared"].get("freeze_sha256")),
            "pythonhashseed": d["identity"]["compared"].get("environment", {}).get("PYTHONHASHSEED"), "budget_seconds": [s.get("budget_seconds") for s in d["sessions"]],
            "timestamp_utc": d["timestamp_utc"]}


# ---------------------------------------------------------------------------- #
# the progress files (the founder's copies, verified by the final file's hash)   #
# ---------------------------------------------------------------------------- #
def progress_path_for(c: dict, journals: Path | None, progress_dir: Path | None) -> Path | None:
    """The cascade's progress file: under the run folders (`<journals>/scale-<draw>-<run id>/c3-<gate>/cascade.progress.jsonl`)
    or as a flat copy (`<progress_dir>/cascade-<draw>-<run id>-<gate>.progress.jsonl`)."""
    cands = []
    if journals is not None:
        cands.append(journals / f"scale-{c['draw']}-{c['run_id']}" / f"c3-{c['gate']}" / "cascade.progress.jsonl")
    if progress_dir is not None:
        cands.append(progress_dir / f"cascade-{c['draw']}-{c['run_id']}-{c['gate']}.progress.jsonl")
    for p in cands:
        if p.exists():
            return p
    return None


def progress_copy(path: Path, tmp: Path) -> Path:
    """A reader owns none of the founder's files: `resumable.Progress` repairs a torn last line in place (a partial line
    cut off, kept beside the file), so the progress file is read through a copy under a folder of the reader's."""
    dst = Path(tmp) / f"{abs(hash(str(path))):x}-{path.name}"
    shutil.copyfile(path, dst)
    return dst


def read_progress(path: Path, final: dict, tmp: Path | None = None) -> dict:
    """The cascade's progress file read cold from a copy: its sha256 against the final file's; the checks at every phase;
    the states, the memory's patterns and the phase event's applied counts at every phase; the read lines by phase, the
    sample reads (entries not yet evicted) apart from the pair reads (by kind, the original's and the duplicate's
    states); a torn last line and an open last session reported (an unfinished run's copy)."""
    sha = RS.sha256_file(path)
    equal = (sha == final["progress_sha256"])
    own_tmp = tempfile.TemporaryDirectory() if tmp is None else None
    pr = RS.Progress(progress_copy(path, Path(own_tmp.name) if own_tmp else tmp))
    units = pr.units()
    per_phase = {}
    for i in sorted(units):
        u = units[i]
        per_phase[i] = {"fault": bool((u.get("check") or {}).get("fault")), "states": u.get("states"), "memory_patterns": u.get("memory_patterns"),
                        "applied": (u.get("phase_event") or {}).get("applied"), "entries": u.get("entries"), "lifecycle_seconds": u.get("lifecycle_seconds")}
    reads, read_faults = read_lines(pr)
    by_phase: dict = {}
    for r in reads:
        ph = str(r.get("phase"))
        b = by_phase.setdefault(ph, {"sample": {"n": 0, "own_in_window": 0, "holes": 0, "own_state": Counter()},
                                      "pairs": {k_: {"n": 0, "own_in_window": 0, "duplicate_in_window": 0, "both_in_window": 0, "holes": 0, "own_state": Counter(), "duplicate_state": Counter(),
                                                     "own_in_window_given_own_not_evicted": 0, "own_not_evicted": 0} for k_ in ("E1", "E2")}})
        if r["what"] == "sample":
            s = b["sample"]; s["n"] += 1; s["own_in_window"] += bool(r.get("own_in_window")); s["holes"] += r.get("holes", 0); s["own_state"][str(r.get("own_state"))] += 1
        else:                                                        # the pair's kind: `pair_kind` once the probe writes it apart, `kind` on the files of 2026-10-30
            pk = b["pairs"][r.get("pair_kind") or r.get("kind")]; pk["n"] += 1; pk["own_in_window"] += bool(r.get("own_in_window")); pk["duplicate_in_window"] += bool(r.get("duplicate_in_window"))
            pk["both_in_window"] += bool(r.get("own_in_window") and r.get("duplicate_in_window")); pk["holes"] += r.get("holes", 0)
            pk["own_state"][str(r.get("own_state"))] += 1; pk["duplicate_state"][str(r.get("duplicate_state"))] += 1
            if r.get("own_state") not in ("evicted", "refused", None):
                pk["own_not_evicted"] += 1; pk["own_in_window_given_own_not_evicted"] += bool(r.get("own_in_window"))
    for b in by_phase.values():
        b["sample"]["own_state"] = dict(b["sample"]["own_state"])
        for pk in b["pairs"].values():
            pk["own_state"] = dict(pk["own_state"]); pk["duplicate_state"] = dict(pk["duplicate_state"])
    faults = [i for i, v in per_phase.items() if v["fault"]]
    sessions = pr.sessions()
    tail = next((l for l in reversed(pr.lines()) if l.get("kind") == "tail"), None)
    tail_under = (sum(1 for d_ in (tail.get("decisions") or []) if d_.get("admitted") and d_.get("surprise") is not None and d_["surprise"] < 0.2) if tail else None)
    out = {"path": str(path), "sha256": sha, "equal_to_the_final_files": equal, "phases_recorded": len(units), "faults": faults, "read_faults": read_faults,
           "n_read_lines": len(reads), "n_sample_lines": sum(1 for r in reads if r["what"] == "sample"), "n_pair_lines": sum(1 for r in reads if r["what"] == "pair"),
           "reads_by_phase": by_phase, "per_phase": per_phase, "near_duplicates_read_before_eviction": read_before_eviction(units, reads, final),
           "tail_writes": (len(tail.get("decisions") or []) if tail else None), "tail_under_floor_writes": tail_under,
           "sessions": sessions, "torn_tail": ({k_: v for k_, v in pr.torn_tail.items() if k_ != "kept_as"} if pr.torn_tail else None),   # the copy's side file is nobody's path
           "last_session_closed": bool(sessions) and sessions[-1].get("end") is not None,
           "read_from": "a copy of the file; the founder's file untouched"}
    if own_tmp is not None:
        own_tmp.cleanup()
    return out


def read_lines(pr: RS.Progress) -> tuple[list[dict], list[list]]:
    """The read lines of a progress file, the sample reads and the pair reads alike, kept once by (phase, what, j, partner).
    The probe writes a pair read's line with the pair's kind (E1, E2) in the `kind` field, over the line kind `read` (the
    record's files of 2026-10-30 carry it so), which `scale_probe.read_lines_of` does not see; the lines are taken here by
    their `what` field. A line recorded twice (a session dead inside the reads) is compared, a difference being a fault."""
    seen: dict[tuple, dict] = {}; faults = []
    for l in pr.lines():
        if l.get("what") not in ("sample", "pair") or "phase" not in l or "own_in_window" not in l:
            continue
        k = (l.get("phase"), l.get("what"), l.get("j"), l.get("partner_j"))
        if k in seen:
            a = {x: y for x, y in seen[k].items() if x != "wall_clock"}; b = {x: y for x, y in l.items() if x != "wall_clock"}
            if a != b:
                faults.append(list(k))
            continue
        seen[k] = l
    return list(seen.values()), faults


def read_before_eviction(units: dict, reads: list[dict], final: dict) -> dict:
    """C.3's 'read before it' per near duplicate, recomputed from the progress file over every read line (the sample reads
    and the pair reads), beside the final file's count, which `near_duplicate_fate` took over the sample reads alone (the
    pair lines unseen, `read_lines`); the duplicate's pointer from the recorded decisions, its eviction phase from the
    phase lines. Two counts: over the numbered read phases alone, and with the reads of the last phase (after the last
    write, `phase` "last") included; each split by the duplicate's end state, since a duplicate still live at the end is
    'read before its eviction' at any read (the committee of 2026-11-05, findings 3 and 4 of readers 2 and 1)."""
    pointer_of_j = {int(d["j"]): d["pointer"] for u in units.values() for d in (u.get("decisions") or []) if d.get("admitted") and d.get("pointer")}
    evicted_at = {int(d["j"]): u["phase"] for u in units.values() for d in (u.get("evicted_near_duplicates_detail") or [])}
    rows = (final.get("near_duplicates") or {}).get("rows") or []
    by_pointer = {ptr: j for j, ptr in pointer_of_j.items()}
    last = 10 ** 9                                                                      # the reads of the last phase sort after every numbered phase
    first_seen: dict[int, int] = {}
    for r in sorted(reads, key=lambda r: (r["phase"] if isinstance(r.get("phase"), int) else last)):
        for ptr in r.get("pointers") or []:
            j = by_pointer.get(ptr)
            if j is not None and j not in first_seen:
                first_seen[j] = r["phase"] if isinstance(r.get("phase"), int) else last
    out = {}
    for kind in ("E1", "E2"):
        sub = [r for r in rows if r["kind"] == kind]
        rec = {"numbered": 0, "with_last": 0}; live = {"numbered": 0, "with_last": 0}; evicted = {"numbered": 0, "with_last": 0}; final_count = 0
        for r in sub:
            j = int(r["j"]); ep = evicted_at.get(j, r.get("evicted_at_phase"))
            fs = first_seen.get(j)
            if fs is not None and (ep is None or fs < ep):
                for mode in ("numbered", "with_last") if fs != last else ("with_last",):
                    rec[mode] += 1
                    (evicted if ep is not None else live)[mode] += 1
            final_count += bool(r.get("read_before_eviction"))
        out[kind] = {"rows": len(sub), "read_before_eviction_sample_and_pair_reads": rec["numbered"], "read_before_eviction_in_the_final_file_sample_reads_only": final_count,
                     "evicted": sum(1 for r in sub if r.get("evicted_at_phase") is not None),
                     "read_live_at_the_end": live["numbered"], "read_then_evicted": evicted["numbered"],
                     "read_before_eviction_with_the_last_phase": rec["with_last"], "read_live_at_the_end_with_the_last_phase": live["with_last"], "read_then_evicted_with_the_last_phase": evicted["with_last"]}
    out["what"] = ("the duplicate's line in the window of any read line at a numbered read phase before the phase of its eviction, or at any numbered read phase when it is still live at the end "
                   "(`read_live_at_the_end`, `read_then_evicted`); `..._with_the_last_phase` counts the reads of the last phase, after the last write, as well; the final file's count took the sample reads alone")
    return out


# ---------------------------------------------------------------------------- #
# the pooled numbers                                                             #
# ---------------------------------------------------------------------------- #
def _bounds(values):
    vals = [v for v in values if v is not None]
    return {"min": min(vals), "max": max(vals), "n": len(vals)} if vals else None


def pooled(c2s: list[dict], cascades: list[dict]) -> dict:
    out = {}
    for size in SIZES:
        for gate in GATES:
            sub = [c for c in c2s if c["size"] == size and c["gate"] == gate]
            cas = [c for c in cascades if c["size"] == size and c["gate"] == gate]
            if not sub and not cas:
                continue
            runs_here = {c["run_id"] for c in sub + cas}
            entry: dict = {"c2_files": len(sub), "cascade_files": len(cas), "encoders": [SEED_OF[r] for r in RUN_IDS if r in runs_here]}
            if sub:
                def never(s_):
                    at = sum(c["never_planted"]["at_or_above"] for c in s_); n = sum(c["never_planted"]["n"] for c in s_)
                    return {"at_or_above": at, "n": n, "share": at / n if n else None, "clopper_pearson_95_pooled": list(clopper_pearson(at, n)) if n else None,
                            "per_file_bounds": _bounds([c["never_planted"]["at_or_above"] for c in s_]),
                            "by_distance": {dist: {"n": sum(c["never_planted"]["by_distance"].get(dist, {}).get("n", 0) for c in s_),
                                                   "at_or_above": sum(c["never_planted"]["by_distance"].get(dist, {}).get("at_or_above", 0) for c in s_)}
                                            for dist in sorted({k_ for c in s_ for k_ in c["never_planted"]["by_distance"]}, key=int)},
                            "ties_at_boundary": sum(c["never_planted"]["ties_at_boundary"] for c in s_)}
                entry["never_planted"] = {**never(sub), "per_encoder": {str(SEED_OF[r]): never([c for c in sub if c["run_id"] == r]) for r in RUN_IDS if any(c["run_id"] == r for c in sub)},
                                          "label": SP.NEVER_PLANTED_LABEL, "what": "C.2 asks the control per journal size and encoder: the per encoder counts are the row's; the pool over the encoders is given beside"}
                for kind in ("pairs_E1", "pairs_E2"):
                    per_encoder = {}
                    for r in RUN_IDS:
                        s_ = [c for c in sub if c["run_id"] == r]
                        if s_:
                            n = sum(c[kind]["contested"] for c in s_); m = sum(c[kind]["mark_ranks_duplicate_first"] for c in s_)
                            pairs = sum(c[kind]["n"] for c in s_)
                            per_encoder[str(SEED_OF[r])] = {"contested": n, "mark_ranks_duplicate_first": m, "clopper_pearson_95": list(clopper_pearson(m, n)) if n else None,
                                                            "duplicates_admitted": sum(c[kind]["duplicate_admitted"] for c in s_), "pairs": pairs,
                                                            "contested_clopper_pearson_95": list(clopper_pearson(n, pairs)) if pairs else None,
                                                            "r4_not_following": sum(c[kind]["r4_not_following"] for c in s_), "r4_outcome_by_construction": sum(c[kind]["r4_outcome_by_construction"] for c in s_)}
                    n = sum(c[kind]["contested"] for c in sub); m = sum(c[kind]["mark_ranks_duplicate_first"] for c in sub)
                    pairs = sum(c[kind]["n"] for c in sub)
                    entry[kind] = {"per_encoder": per_encoder, "contested": n, "pairs": pairs, "contested_clopper_pearson_95": list(clopper_pearson(n, pairs)) if pairs else None,
                                   "mark_ranks_duplicate_first": m, "clopper_pearson_95": list(clopper_pearson(m, n)) if n else None,
                                   "per_file_share_bounds": _bounds([c[kind]["mark_ranks_duplicate_first"] / c[kind]["contested"] for c in sub if c[kind]["contested"]]),
                                   "r4_not_following": sum(c[kind]["r4_not_following"] for c in sub)}
                entry["sample_own_in_window"] = {ix: _bounds([c["sample"]["own_in_window"][ix] for c in sub]) for ix in ("I0", "I1", "I2", "16x12", "32x10", "32x12")}
                entry["sample_n"] = _bounds([c["sample"]["n"] for c in sub])
                entry["agree"] = {k_: _bounds([c["sample"]["agree"][k_] for c in sub]) for k_ in ("own_in_both", "same_set", "same_first")}
                entry["i0_candidates_median"] = _bounds([c["sample"]["i0_candidates"]["median"] for c in sub])
                entry["i0_candidates_p95"] = _bounds([c["sample"]["i0_candidates"]["p95"] for c in sub])
                entry["i0_ties_at_boundary"] = sum(c["sample"]["i0_ties_at_boundary"] for c in sub)
                entry["admitted"] = _bounds([c["planting"]["admitted"] for c in sub]); entry["refused"] = _bounds([c["planting"]["refused"] for c in sub])
                entry["admitted_by_kind"] = {k_: _bounds([c["planting"]["by_kind"][k_]["admitted"] for c in sub]) for k_ in ("fact", "E1", "E2")}
                entry["under_floor_by_kind"] = {k_: _bounds([c["planting"]["by_kind"][k_]["under_floor_0.2"] for c in sub]) for k_ in ("fact", "E1", "E2")}
                entry["time_per_query_median_ms"] = {mode: {ix: _bounds([c["time_per_query"][mode][ix]["median_ns"] / 1e6 for c in sub]) for ix in ("I0", "I1", "I2")} for mode in ("interleaved", "blocked")}
                entry["time_per_query_flags"] = sum(len(c["time_per_query"]["flagged_interleaved"]) + len(c["time_per_query"]["flagged_blocked"]) for c in sub)
                entry["index_bytes"] = {ix: _bounds([c["index_sizes"]["traced_bytes"][ix] for c in sub]) for ix in ("I0", "I1", "I2")}
                entry["i0_buckets"] = _bounds([c["index_sizes"]["I0"]["buckets"] for c in sub])
                entry["threshold_within_1e-6_everywhere"] = all(c["threshold"]["within_1e-6"] for c in sub)
                entry["key_contract"] = {k_: sum(c["key_contract"].get(k_, 0) for c in sub) for k_ in ("n_sample_reads_of_cross_schema_repeated_names", "both_lines_in_window", "r4_abstains_with_both_lines", "r4_keeps_own_with_both_lines")}
            if cas:
                entry["cascade"] = {"faults_total": sum(c["faults"]["n"] for c in cas), "phases": _bounds([c["phases"] for c in cas]), "expected_phases_identity": EXPECTED_PHASES[size],
                                    "entries": _bounds([c["entries"] for c in cas]), "admitted": _bounds([c["admitted"] for c in cas]),
                                    "states_at_the_end": {s_: _bounds([c["states"][s_] for c in cas]) for s_ in ("live", "consolidated", "demoted", "evicted")},
                                    "applied_totals": {k_: _bounds([c["applied_totals"][k_] for c in cas]) for k_ in ("consolidated", "replay_failed", "demoted_groups", "evicted", "withdrawn")},
                                    "summaries_written": sum(c["summaries"]["written"] for c in cas), "phases_refused": sum(c["phases_refused"] for c in cas),
                                    "near_duplicates": {k_: {x: _bounds([c["near_duplicates"][k_][x] for c in cas]) for x in ("admitted", "evicted", "live", "read_before_eviction", "grouped_with_partner")} for k_ in ("E1", "E2")},
                                    "reads_last": {x: _bounds([c["reads_by_phase"]["last"][x] for c in cas if c["reads_by_phase"].get("last")]) for x in ("n", "sample_read", "pairs_read", "own_in_window", "holes_total", "duplicates_in_window")},
                                    "still_reconstructed_under_bound_total": sum(v["still_reconstructed_under_bound"] for c in cas for v in c["reconstruction_by_phase"].values()),
                                    "evicted_addresses_read_at_the_last_phase": _bounds([c["reconstruction_by_phase"]["last"]["evicted_addresses"] for c in cas if "last" in c["reconstruction_by_phase"]]),
                                    "min_defect_evicted": _bounds([c["reconstruction_by_phase"]["last"]["min_defect_evicted"] for c in cas if "last" in c["reconstruction_by_phase"]]),
                                    "phase_time_sum_seconds": _bounds([c["phase_times"]["sum_seconds"] for c in cas]), "phase_time_first": _bounds([c["phase_times"]["first"] for c in cas]),
                                    "phase_time_last": _bounds([c["phase_times"]["last"] for c in cas]), "phase_time_max": _bounds([c["phase_times"]["max"] for c in cas]),
                                    "phase_time_slope_seconds_per_phase": _bounds([c["phase_times"]["fit"]["slope_seconds_per_phase"] for c in cas if c["phase_times"]["fit"]]),
                                    "phase_time_fit_r2": _bounds([c["phase_times"]["fit"]["r2"] for c in cas if c["phase_times"]["fit"]]),
                                    "sessions": _bounds([c["sessions"] for c in cas]), "resumption_faults": sum(len(c["resumption_fault"]) for c in cas),
                                    "read_lines_recorded_twice_compared": sum(c["read_lines_recorded_twice_compared"] for c in cas),
                                    "freeze_bound_everywhere": all(c["freeze_bound"] for c in cas), "label": SP.TIME_LABEL}
            out[f"{size}/{gate}"] = entry
    return out


def decision_13_successor(c2s: list[dict]) -> dict:
    """C.2 (b): after row 53 the label of Decision 13 and of §3.5 reads, whatever the numbers, 'read out of sample on fresh
    pairs without the model (row 53): the mark ranks the duplicate first on n of N contested reads per size and encoder',
    the numbers filled in and no adjective added. Filled in here per size, encoder, gate and kind."""
    filled = {}
    for size in SIZES:
        for r in RUN_IDS:
            for gate in GATES:
                sub = [c for c in c2s if c["size"] == size and c["run_id"] == r and c["gate"] == gate]
                if not sub:
                    continue
                for kind in ("E1", "E2"):
                    n = sum(c[f"pairs_{kind}"]["contested"] for c in sub); m = sum(c[f"pairs_{kind}"]["mark_ranks_duplicate_first"] for c in sub)
                    filled[f"{size}/{SEED_OF[r]}/{gate}/{kind}"] = {"n": m, "N": n, "files": len(sub), "clopper_pearson_95": list(clopper_pearson(m, n)) if n else None}
    return {"label": SP.DECISION_13_SUCCESSOR, "in_sample_label_kept_beside": "in sample by design (the model's choice, which row 53 does not read)", "filled": filled}


# ---------------------------------------------------------------------------- #
# the reading                                                                    #
# ---------------------------------------------------------------------------- #
def read_rows(md: Path, journals: Path | None = None, progress_dir: Path | None = None, progress_files: list[Path] = (), also: list[Path] = ()) -> dict:
    """The reading. `progress_files`: progress files of cascades with no final file (unfinished, refused), read for C.4's
    fault condition; `also`: final files of the probe outside the expected set (a refused run retained under its own
    name, a second run), digested apart under their names and read for the fault condition, never pooled."""
    md = Path(md)
    files = load_files(md)
    present = files["present"]
    encodings = {n: encoding_digest(f["data"]) for n, f in present.items() if f["kind"] == "encoding"}
    c2s = {n: c2_digest(f["data"]) for n, f in present.items() if f["kind"] == "c2"}
    cascades = {n: cascade_digest(f["data"]) for n, f in present.items() if f["kind"] == "cascade"}
    extra = {}
    for p in also:
        p = Path(p); d = json.loads(p.read_text(encoding="utf-8")); kind = kind_of_final(d)
        if kind is None:
            raise ValueError(f"{p}: not a final file of the probe")
        extra[p.name] = {"kind": kind, "sha256_lf": sha256_lf(p), "digest": {"encoding": encoding_digest, "c2": c2_digest, "cascade": cascade_digest}[kind](d), "data": d,
                         "what": "a final file outside the expected set, read apart (C.3 (b)); the row carries it beside the record's, never pooled"}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # C.4 by the probe's own function, on the final files present and the extra ones; an unfinished or refused cascade's progress file
        # (no final file) is read for the fault condition through a copy (`checks_from_progress` constructs `Progress`, which repairs in place)
        unfinished = []
        for e in expected_files():
            if e["kind"] == "cascade" and e["name"] not in present:
                p = progress_path_for(e, journals, progress_dir)
                if p is not None:
                    unfinished.append(p)
        unfinished += [Path(p) for p in progress_files]
        unfinished_copies = [progress_copy(p, tmp) for p in unfinished]
        c4 = SP.read_c4([f["data"] for f in present.values() if f["kind"] == "cascade"] + [x["data"] for x in extra.values() if x["kind"] == "cascade"],
                        [f["data"] for f in present.values() if f["kind"] == "c2"], unfinished_copies)
        c4["withdrawal_under_cascade"]["unfinished_or_refused_runs"] = [str(p) for p in unfinished]        # the founder's paths, not the copies'
        for f_ in c4["withdrawal_under_cascade"]["faults"]:
            if "from" in f_:
                f_["from"] = next((str(p) for p, c in zip(unfinished, unfinished_copies) if str(c) == f_["from"]), f_["from"])
        # the progress files of the finished cascades, when at hand
        progress = {}
        for n, c in cascades.items():
            p = progress_path_for(c, journals, progress_dir)
            if p is not None:
                progress[n] = read_progress(p, present[n]["data"], tmp)
    pool = pooled(list(c2s.values()), list(cascades.values()))
    d13 = decision_13_successor(list(c2s.values()))
    everything = list(encodings.values()) + list(c2s.values()) + list(cascades.values())
    # the consistency checks a reader makes first (recorded, not asserted: the record test asserts)
    checks = {"threshold_within_1e-6_on_every_c2_file": all(c["threshold"]["within_1e-6"] for c in c2s.values()),
              "freeze_bound_on_every_file": all(x["freeze_bound"] for x in everything),
              "pythonhashseed_0_on_every_file": all(str(x["pythonhashseed"]) == "0" for x in everything),
              "identity_cascades_at_the_expected_phase_count": all(c["phases"] == EXPECTED_PHASES[c["size"]] for c in cascades.values() if c["gate"] == "identity"),
              "ten_thousand_runs_in_one_session": all(x["sessions"] == 1 for x in everything if x["size"] == "ten_thousand"),
              "sessions_per_file": {n: x["sessions"] for n, x in {**encodings, **c2s, **cascades}.items() if x["sessions"] != 1},
              "progress_files_equal_to_the_final_files_hashes": all(p["equal_to_the_final_files"] for p in progress.values()) if progress else None,
              "progress_files_read": len(progress), "unfinished_or_refused_progress_files_read": len(unfinished), "extra_final_files_read": sorted(extra)}
    summary = summary_lines(files, pool, c4, d13, checks)
    return {"rows": "row 53", "what": "part C's files read cold (ADR-010 C.4 and the amendment's section 3); every number from the files alone",
            "files": {"expected": len(files["expected"]), "present": len(present), "missing": files["missing"], "outside_the_expected_set": files["outside_the_expected_set"],
                      "hashes": {n: f["sha256_lf"] for n, f in present.items()}},
            "encodings": encodings, "c2": c2s, "cascades": cascades, "extra": {n: {k_: v for k_, v in x.items() if k_ != "data"} for n, x in extra.items()},
            "progress": progress, "c4": c4, "pooled": pool, "decision_13_successor": d13, "checks": checks,
            "labels": {"never_planted": SP.NEVER_PLANTED_LABEL, "read_side": SP.READ_SIDE_LABEL, "key_contract": SP.KEY_CONTRACT_LABEL, "time": SP.TIME_LABEL,
                       "cost_curves": "reported with their construction; no threshold to pass"},
            "summary_lines": summary}


def summary_lines(files: dict, pool: dict, c4: dict, d13: dict, checks: dict) -> list[str]:
    lines = [f"files: {len(files['present'])} of {len(files['expected'])} present; missing {len(files['missing'])}; outside the expected set {len(files['outside_the_expected_set'])}"]
    w = c4["withdrawal_under_cascade"]
    unf = len(w.get("unfinished_or_refused_runs") or [])
    lines.append(f"C.4 (1) withdrawal under cascade: {'holds on every journal and phase' if w['holds_on_every_journal_and_phase'] else 'FAULT ' + json.dumps(w['faults'])} "
                 f"({w['runs_read']} runs read, {unf} of them unfinished or refused, from their progress files)")
    for key, e in pool.items():
        if "never_planted" in e:
            for enc, nn in e["never_planted"]["per_encoder"].items():
                lines.append(f"C.4 (2) never planted at or above the threshold, {key}, the encoder of seed {enc}: {nn['at_or_above']} of {nn['n']} (per file {nn['per_file_bounds']['min']} to {nn['per_file_bounds']['max']}; "
                             f"exact 95 % interval {nn['clopper_pearson_95_pooled'][0]:.3f} to {nn['clopper_pearson_95_pooled'][1]:.3f}); by distance "
                             + ", ".join(f"{d_}: {v['at_or_above']} of {v['n']}" for d_, v in nn["by_distance"].items()))
        if "cascade" in e:
            c = e["cascade"]
            lines.append(f"cascade {key}: faults {c['faults_total']}; phases {c['phases']['min']} to {c['phases']['max']}; evicted {c['states_at_the_end']['evicted']['min']} to {c['states_at_the_end']['evicted']['max']}; "
                         f"demoted groups {c['applied_totals']['demoted_groups']['max']}; still reconstructed under the bound {c['still_reconstructed_under_bound_total']}; "
                         f"phase time sum {c['phase_time_sum_seconds']['min']:.0f} to {c['phase_time_sum_seconds']['max']:.0f} s")
    for k_, v in d13["filled"].items():
        lines.append(f"Decision 13 successor, {k_}: the mark ranks the duplicate first on {v['n']} of {v['N']} contested reads")
    lines.append("checks: " + ", ".join(f"{k_} {v}" for k_, v in checks.items()))
    return lines


def _cli(argv=None):
    ap = argparse.ArgumentParser(description="row 53's cold reading of part C's files (ADR-010 C.4)")
    ap.add_argument("--metrics", default="metrics/mqar")
    ap.add_argument("--journals", default=None, help="the run folders (runs\\scale) holding the cascades' progress files, read when present")
    ap.add_argument("--progress-dir", default=None, help="flat copies of the cascades' progress files, cascade-<draw>-<run id>-<gate>.progress.jsonl")
    ap.add_argument("--progress-file", action="append", default=[], help="a progress file of a cascade with no final file (unfinished, refused): read for C.4's fault condition; repeatable")
    ap.add_argument("--also", action="append", default=[], help="a final file of the probe outside the expected set (a refused run retained, a second run): read apart, never pooled; repeatable")
    ap.add_argument("--out", default=None, help=f"the reading file (default <metrics>/{READING_NAME}); never overwritten")
    args = ap.parse_args(argv)
    md = Path(args.metrics)
    out = Path(args.out) if args.out else md / READING_NAME
    if out.exists():
        raise SystemExit(f"scale reading: {out} exists; a file of the record is never overwritten (a second reading goes under another name)")
    r = read_rows(md, Path(args.journals) if args.journals else None, Path(args.progress_dir) if args.progress_dir else None,
                  [Path(p) for p in args.progress_file], [Path(p) for p in args.also])
    r["timestamp_utc"] = RS.utc_now()
    sha = RS.write_json(out, r)
    for line in r["summary_lines"]:
        print(line)
    print(f"retained: {out}  sha256 {sha}")


if __name__ == "__main__":
    _cli()
