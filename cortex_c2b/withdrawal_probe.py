"""cortex_c2b.withdrawal_probe -- the withdrawal at eviction, measured (ADR-007, amendment of
2026-10-01, declared before any measurement; the first item of v1.1; RESULTS row 41 is its
origin).

Row 41 found that the address of every evicted episode survived its eviction in the
persistent memory: an eviction released the content and never withdrew a pattern. The
amendment gives the lifecycle a flag, `withdraw_at_eviction`, under which the pattern of an
entry is withdrawn from the memory in the phase that evicts it, and the rebuild of a
scheduler opened on the journal skips the evicted entries (Decision 8 kept: the memory of a
second process equals the first's). The memory keeps its patterns explicitly, so the
withdrawal is exact by construction; what this probe measures is whether a withdrawn
address is still reconstructed under the bound from the patterns that stay, and from which
neighbour.

The run, on one sealed journal of the record (copied, never modified): the reconstruction
probe twice, by the same code, from the same original.

  control      the flag off: must reproduce the file of row 41 for this journal, number for
               number, on levels 1 to 3b and on the per phase records (reading a); nothing
               below is read if it does not;
  measurement  the flag on: level 1 exact; level 2 identical to the control (the salience
               line does not depend on the memory); level 3 kept addresses all under the
               bound, their maximal defect next to the control's; level 3b, the share of
               withdrawn addresses still reconstructed under the bound, each attributed to
               its nearest kept pattern and the code's radius for a near duplicate
               (readings b1, b2, b3); collateral, a kept address above the bound that was
               under it in the control (reading c); the two processes of each run agreeing
               (reading d).

Thresholds: the lifecycle's own bound (0.05) and its demotion radius (0.90); no new constant.
Nothing here is a claim. The first number comes from the first file.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from cortex_c2b import reconstruction_probe as rp
from cortex_c2b.lifecycle import LifecycleConfig

# the scalars of a row 41 file the control must reproduce (paths into the probe's output)
RECORD_PATHS = [
    ("journal", "sha256"), ("journal", "entries_before"), ("config", "config_hash"), ("config", "now_plant"),
    ("phases_refused",), ("replays", "count"), ("replays", "withdrawn"),
    ("process_a", "level_1_counts", "exact_at_every_phase"), ("process_a", "level_1_counts", "worst_defect"),
    ("process_a", "level_1_counts", "per_phase"),
    ("process_a", "level_2_salience_line", "entries"), ("process_a", "level_2_salience_line", "kept"),
    ("process_a", "level_2_salience_line", "evicted"), ("process_a", "level_2_salience_line", "contracted"),
    ("process_a", "level_2_salience_line", "max_relative_defect"), ("process_a", "level_2_salience_line", "mean_relative_defect"),
    ("process_a", "level_3_memory", "patterns"), ("process_a", "level_3_memory", "replayed"),
    ("process_a", "level_3_memory", "kept"), ("process_a", "level_3_memory", "distribution"),
    ("process_a", "level_3_memory", "evicted_address_reading", "entries"),
    ("process_a", "level_3_memory", "evicted_address_reading", "still_reconstructed_under_bound"),
    ("process_a", "level_3_memory", "evicted_address_reading", "share"),
    ("process_a", "level_3_memory", "positive_control"), ("process_a", "level_4_content"),
    ("process_a", "snapshot"), ("agreement", "number_for_number"),
]
PHASE_KEYS = ("phase", "now", "pressure", "floor_eff", "batch_eff", "planned", "sentinel_recall", "sentinel_negctrl",
              "refused", "applied", "live", "consolidated", "demoted", "evicted", "summaries", "memory_patterns")


def _get(d: dict, path: tuple):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return "<missing>"
        d = d[k]
    return d


def compare_with_record(control: dict, record: dict) -> dict:
    """Reading (a): the control run against the committed file of row 41, scalar by scalar."""
    diffs = []
    for path in RECORD_PATHS:
        a, b = _get(control, path), _get(record, path)
        d = rp._compare(a, b, "/".join(map(str, path))) if isinstance(a, (dict, list)) and isinstance(b, (dict, list)) else \
            ([] if a == b else [f"{'/'.join(map(str, path))}: {a!r} != {b!r}"])
        diffs += d
    pa, pb = control.get("phases", []), record.get("phases", [])
    if len(pa) != len(pb):
        diffs.append(f"phases: {len(pa)} != {len(pb)}")
    else:
        for i, (x, y) in enumerate(zip(pa, pb)):
            for k in PHASE_KEYS:
                if x.get(k) != y.get(k):
                    diffs.append(f"phases[{i}]/{k}: {x.get(k)!r} != {y.get(k)!r}")
    # the per entry rows of levels 2 and 3, number for number
    for lvl in ("level_2_salience_line", "level_3_memory"):
        ra, rb = control["process_a"][lvl]["rows"], record["process_a"][lvl]["rows"]
        if len(ra) != len(rb):
            diffs.append(f"{lvl}/rows: {len(ra)} != {len(rb)}")
        else:
            for i, (x, y) in enumerate(zip(ra, rb)):
                for k in y:
                    if x.get(k) != y.get(k):
                        diffs.append(f"{lvl}/rows[{i}]/{k}: {x.get(k)!r} != {y.get(k)!r}")
    return {"record_file_present": True, "reproduced_number_for_number": not diffs, "differences": diffs[:100],
            "scalars_compared": len(RECORD_PATHS), "phases_compared": len(pb)}


def readings(control: dict, measurement: dict, regression: dict | None) -> dict:
    """The readings of the amendment, in its letters."""
    c3, m3 = control["process_a"]["level_3_memory"], measurement["process_a"]["level_3_memory"]
    c2, m2 = control["process_a"]["level_2_salience_line"], measurement["process_a"]["level_2_salience_line"]
    ev = m3["evicted_address_reading"]
    # reading c: a kept address under the bound in the control and above it in the measurement
    c_rows = {r["entry_id"]: r for r in c3["rows"]}
    collateral = [r["entry_id"] for r in m3["rows"]
                  if r["state"] in ("consolidated", "demoted") and not r["under_bound"]
                  and r["entry_id"] in c_rows and c_rows[r["entry_id"]]["under_bound"]]
    r = {
        "a_control_reproduces_row_41": (regression["reproduced_number_for_number"] if regression else None),
        "a_record_file_present": bool(regression),
        "b_level_1_exact": measurement["readings"]["level_1_exact"],
        "b_level_2_under_threshold": measurement["readings"]["level_2_under_threshold"],
        "b_level_2_identical_to_control": (m2["max_relative_defect"] == c2["max_relative_defect"]
                                           and m2["mean_relative_defect"] == c2["mean_relative_defect"]
                                           and m2["entries"] == c2["entries"]),
        "b_level_3_kept_all_under_bound": measurement["readings"]["level_3_kept_under_bound"],
        "b_level_3_kept_max_defect": m3["kept"]["max_defect"],
        "a_level_3_kept_max_defect_control": c3["kept"]["max_defect"],
        "b_patterns_in_memory": m3["patterns"],
        "a_patterns_in_memory_control": c3["patterns"],
        "b_withdrawals_at_eviction": measurement.get("withdrawals_at_eviction"),
        "b_evicted": ev["entries"],
        "b_still_under_bound_after_withdrawal": ev["still_reconstructed_under_bound"],
        "b_share_still_under_bound": ev["share"],
        "b_with_kept_neighbour_within_radius": ev.get("still_under_bound_with_kept_neighbour_within_radius"),
        "b_without_such_neighbour": ev.get("still_under_bound_without_such_neighbour"),
        "b_radius": ev.get("radius"),
        "b_min_defect_of_withdrawn": ev.get("min_defect_evicted"),
        "b_max_defect_of_withdrawn": ev.get("max_defect_evicted"),
        "c_collateral_kept_addresses": collateral,
        "d_control_two_processes_agree": control["readings"]["two_processes_agree"],
        "d_measurement_two_processes_agree": measurement["readings"]["two_processes_agree"],
        "level_4_holds": measurement["readings"]["level_4_holds"],
    }
    # the letter
    if not (r["d_control_two_processes_agree"] and r["d_measurement_two_processes_agree"]
            and r["b_level_1_exact"] and r["level_4_holds"]):
        letter = "d: construction fault, no row"
    elif regression and not r["a_control_reproduces_row_41"]:
        letter = "a: the control does not reproduce row 41; attributed before anything else is read"
    elif r["c_collateral_kept_addresses"] or not r["b_level_3_kept_all_under_bound"]:
        letter = "c: collateral on kept addresses; the flag stays off; attributed"
    elif not r["b_level_2_identical_to_control"] or not r["b_level_2_under_threshold"]:
        letter = "b: level 2 moved under the flag, which it must not; attributed before any word"
    elif ev["entries"] == 0:
        letter = "b: no eviction in this run; nothing to withdraw"
    elif ev["still_reconstructed_under_bound"] == 0:
        letter = "b1: the withdrawal is exact at the memory's level; no withdrawn address is reconstructed from the kept patterns"
    elif r["b_without_such_neighbour"] == 0:
        letter = "b2: every withdrawn address still reconstructed has a kept neighbour within the radius; the write side separation is the next measurement"
    else:
        letter = "b3: some withdrawn addresses are reconstructed without a kept neighbour within the radius; attributed on the files before any word"
    r["letter"] = letter
    return r


def run(src: Path, key: bytes, phases: int = rp.PHASES, sentinel_seed: int = rp.SENTINEL_SEED,
        record_file: Path | None = None) -> dict:
    control = rp.run(src, key, phases=phases, sentinel_seed=sentinel_seed, withdraw=False)
    measurement = rp.run(src, key, phases=phases, sentinel_seed=sentinel_seed, withdraw=True)
    regression = None
    if record_file is not None and Path(record_file).exists():
        record = json.loads(Path(record_file).read_text(encoding="utf-8"))
        regression = compare_with_record(control, record)
        regression["record_file"] = str(record_file)
    r = readings(control, measurement, regression)
    return {"probe": "withdrawal at eviction (ADR-007 amendment 2026-10-01; v1.1, first item; origin RESULTS row 41)",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "journal": control["journal"],
            "config": {"phases": int(phases), "sentinel_seed": int(sentinel_seed),
                       "config_hash_control": control["config"]["config_hash"],
                       "config_hash_measurement": measurement["config"]["config_hash"],
                       "bound": LifecycleConfig().replay_error_max, "radius": LifecycleConfig().demote_min_cosine},
            "control": control, "measurement": measurement,
            "regression_against_row_41": regression if regression else {"record_file_present": False},
            "readings": r, "verdict": r["letter"]}


def summary_lines(out: dict) -> list[str]:
    r = out["readings"]
    lines = [f"journal {out['journal']['sha256'][:16]}  phases {out['config']['phases']}  hashes control {out['config']['config_hash_control']} / measurement {out['config']['config_hash_measurement']}"]
    reg = out["regression_against_row_41"]
    lines.append("reading a (control vs row 41): " + ("no record file given" if not reg.get("record_file_present") else
                 f"reproduced number for number = {reg['reproduced_number_for_number']} ({reg['scalars_compared']} scalars, {reg['phases_compared']} phases)"))
    lines.append(f"memory patterns: control {r['a_patterns_in_memory_control']}, measurement {r['b_patterns_in_memory']}; withdrawals at eviction {r['b_withdrawals_at_eviction']}")
    lines.append(f"level 2 identical to control: {r['b_level_2_identical_to_control']}; level 1 exact: {r['b_level_1_exact']}; level 4 holds: {r['level_4_holds']}")
    lines.append(f"level 3 kept: all under bound = {r['b_level_3_kept_all_under_bound']}; max defect {r['b_level_3_kept_max_defect']} (control {r['a_level_3_kept_max_defect_control']})")
    lines.append(f"level 3b: {r['b_still_under_bound_after_withdrawal']}/{r['b_evicted']} withdrawn addresses still under the bound; "
                 f"{r['b_with_kept_neighbour_within_radius']} with a kept neighbour within {r['b_radius']}, {r['b_without_such_neighbour']} without; "
                 f"defects of the withdrawn: min {r['b_min_defect_of_withdrawn']}, max {r['b_max_defect_of_withdrawn']}")
    lines.append(f"collateral on kept addresses: {len(r['c_collateral_kept_addresses'])}")
    lines.append(f"two processes agree: control {r['d_control_two_processes_agree']}, measurement {r['d_measurement_two_processes_agree']}")
    lines.append(f"reading: {out['verdict']}")
    return lines


def _cli(argv=None):
    ap = argparse.ArgumentParser(description="the withdrawal at eviction, measured (ADR-007 amendment 2026-10-01)")
    ap.add_argument("--journal", required=True, help="a sealed journal of the record (the journal.jsonl or its directory); copied, never modified")
    ap.add_argument("--out", default=None, help="the directory or file for the retained record")
    ap.add_argument("--run-id", default=None, help="the run whose journal this is (names the file and finds the row 41 file)")
    ap.add_argument("--record", default=None, help="the row 41 file for this journal (default: metrics/mqar/reconstruction-probe-<run-id>.json)")
    ap.add_argument("--phases", type=int, default=rp.PHASES)
    ap.add_argument("--sentinel-seed", type=int, default=rp.SENTINEL_SEED)
    args = ap.parse_args(argv)
    from cortex_c2b.crypto import key_from_env
    key = key_from_env()
    if key is None:
        raise SystemExit("QUANTUM_CORTEX_JOURNAL_KEY missing")
    record = Path(args.record) if args.record else (
        Path(__file__).resolve().parents[1] / "metrics" / "mqar" / f"reconstruction-probe-{args.run_id}.json" if args.run_id else None)
    out = run(Path(args.journal), key, phases=args.phases, sentinel_seed=args.sentinel_seed, record_file=record)
    for line in summary_lines(out):
        print(line)
    if args.out:
        o = Path(args.out)
        if o.is_dir() or not o.suffix:
            o.mkdir(parents=True, exist_ok=True)
            o = o / f"withdrawal-probe-{args.run_id or out['journal']['sha256'][:12]}.json"
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"[record] retained: {o}")


if __name__ == "__main__":
    _cli()
