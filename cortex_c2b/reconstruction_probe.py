"""cortex_c2b.reconstruction_probe -- the reconstruction probe on the organ's layer of
numbers (ADR-007, amendment of 2026-09-30, declared before any measurement; whitepaper
section 3.5, the four words, and section 5d, door 7).

The door asks whether a level of the organ earns the word *reconstructible at measured
defect*: derive what is hidden from what is kept plus the declared constants, compare with
what the log recorded, and call the gap the defect, under a threshold declared before any
number. On the text the organ has two words by construction, backup and declared loss;
this probe measures its layer of numbers, in three levels and a reading:

  level 1  the counts per state along the consolidation index: from the visible counts
           (live, consolidated, the summaries alive) and the constants (K, the number of
           entries) derive the hidden pair (demoted, evicted); exact at every phase, or a
           construction fault (a check, not a claim);
  level 2  the salience line: from an entry's final salience, the decay constant and the
           number of decays it received (a count of applied phases after its
           consolidation, read from the log), derive its salience at consolidation and
           compare with the value the log recorded; threshold, a relative defect of 1e-3
           (the rounding to six decimals per phase is the known residual);
  level 3  the persistent associative memory: reconstruct every replayed cue from the
           memory alone and measure the cosine distance to the cue the log holds;
           threshold, the lifecycle's own replay bound, 0.05, at the end of the run;
  level 3b the reading declared with it: the share of evicted entries whose cue the
           memory still reconstructs under the bound (at v1.0 an eviction never withdraws
           a pattern; the code says so and the probe reports what that means). Under the
           flag `withdraw_at_eviction` (ADR-007 amendment 2026-10-01) the pattern is
           withdrawn at the eviction, and the probe adds the attribution: for every evicted
           address still under the bound, its nearest kept pattern and whether that
           neighbour is within the code's radius for a near duplicate (demote_min_cosine);
  level 4  the content: no released payload has a file, no evicted entry reads
           (declared loss, a check).

The run: a sealed journal of the record is COPIED to a temporary directory and never
modified in place; the lifecycle plants its sentinels from the frozen generator at a seed
that must not collide with the journal's entities (the abort of row 40), then runs the
declared number of phases on the logical clock with no reads between them; then a NEW
PROCESS reopens the copy from the disk alone, rebuilds the memory from the consolidation
order (invariant 9) and recomputes every level: the two must agree number for number.
Nothing here is a claim. The first number comes from the first file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cortex_c2b import (Journal, POLICY_STOP, MODE_DURABLE, STATE_LIVE, STATE_CONSOLIDATED, STATE_DEMOTED,
                        STATE_EVICTED, SCHEMA_SUMMARY, lifecycle_declaration)
from cortex_c2b.crypto import open_line
from cortex_c2b.hm_protocol import generate_facts, value_of_statement
from cortex_c2b.lifecycle import LifecycleConfig, LifecycleScheduler
from cortex_c2b.read_path import JournalPath
from cortex_c2b.write_path import WritePath

PHASES = 40                       # declared: forty phases on the logical clock, no reads between them
SENTINEL_SEED = 2                 # declared: the frozen generator's seed for the sentinels (not 0, 10 000 or 100 000)
INDEX_SEED = 0                    # the cue index's declared seed, as session B of the record opened it
NOW_PLANT = None                  # the sentinels are planted at the journal's last write time plus one (200.0 for the record's journals)
THRESH_SALIENCE_REL = 1e-3        # level 2, declared
THRESH_MEMORY = None              # level 3: the lifecycle's own replay bound (cfg.replay_error_max), never another
RESERVED_SEEDS = (0, 10_000, 100_000)


# ---------------------------------------------------------------------------- #
# The journal: copy, hash, meta                                                  #
# ---------------------------------------------------------------------------- #
def journal_file(path: str | Path) -> Path:
    p = Path(path)
    return p / "journal.jsonl" if p.is_dir() else p


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_journal(src: Path, dst_dir: Path) -> Path:
    """The journal log and its payload directory, copied; the original is never touched."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copyfile(src, dst)
    pay = src.with_name(src.stem + ".payloads")
    if pay.is_dir():
        shutil.copytree(pay, dst.with_name(dst.stem + ".payloads"))
    return dst


def separation_of(journal: Journal) -> dict:
    """The write path's separation (dg, seed) the journal was written with: a journal
    opened under another separation is refused by the write path, so the probe reads it."""
    if not journal.ca3_order:
        return {"dg": True, "seed": 0}
    meta = journal.ca3_order[0][1]
    return {"dg": bool(meta.get("dg", True)), "seed": int(meta.get("seed", 0))}


def entities_of(journal: Journal) -> set[str]:
    """The entities of the journal's planted statements, read by the organ's own parser."""
    ents = set()
    for e in journal._entries.values():
        if e.schema_id == SCHEMA_SUMMARY:
            continue
        try:
            text = journal.payloads.get(e.pointer).decode("utf-8", errors="replace")
        except (KeyError, ValueError):
            continue
        v = value_of_statement(text)
        if v:
            ents.add(v[1])
    return ents


def _open(copy: Path, key: bytes) -> tuple[Journal, WritePath, JournalPath]:
    journal = Journal(copy, key=key, policy=POLICY_STOP)
    sep = separation_of(journal)
    wp = WritePath(journal, use_dg=sep["dg"], seed=sep["seed"])
    jp = JournalPath(journal, seed=INDEX_SEED)
    return journal, wp, jp


# ---------------------------------------------------------------------------- #
# The log, read as events (the same reader as the journal's replay)              #
# ---------------------------------------------------------------------------- #
def iter_events(path: Path, key: bytes | None):
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            yield json.loads(open_line(key, raw))


# ---------------------------------------------------------------------------- #
# The levels                                                                     #
# ---------------------------------------------------------------------------- #
def level_1_counts(path: Path, key: bytes | None, k: int) -> dict:
    """At every phase event of the log: from (live, consolidated, summaries alive, entries)
    and K, derive (demoted, evicted) and compare with the state the events built."""
    state: dict[str, str] = {}
    sources: dict[str, list[str]] = {}
    phases, worst = [], 0
    for ev in iter_events(path, key):
        kind = ev["ev"]
        if kind == "write":
            state[ev["entry"]["entry_id"]] = STATE_LIVE
        elif kind == "transition":
            state[ev["entry_id"]] = ev["to"]
        elif kind == "demote":
            sources[ev["summary"]] = list(ev["sources"])
        elif kind == "phase":
            counts = {s: 0 for s in (STATE_LIVE, STATE_CONSOLIDATED, STATE_DEMOTED, STATE_EVICTED)}
            for s in state.values():
                counts[s] += 1
            n = len(state)
            alive = sum(1 for sid in sources if state.get(sid) != STATE_EVICTED)
            derived_demoted = k * alive
            derived_evicted = n - counts[STATE_LIVE] - counts[STATE_CONSOLIDATED] - derived_demoted
            dd, de = derived_demoted - counts[STATE_DEMOTED], derived_evicted - counts[STATE_EVICTED]
            worst = max(worst, abs(dd), abs(de))
            phases.append({"phase": ev.get("phase"), "refused": bool(ev.get("refused")), "entries": n,
                           "live": counts[STATE_LIVE], "consolidated": counts[STATE_CONSOLIDATED],
                           "summaries_alive": alive, "demoted": counts[STATE_DEMOTED],
                           "evicted": counts[STATE_EVICTED], "derived_demoted": derived_demoted,
                           "derived_evicted": derived_evicted, "defect_demoted": dd, "defect_evicted": de})
    return {"word_by_construction": "theorem (the closure of 3.5, rule 2)", "phases_checked": len(phases),
            "exact_at_every_phase": worst == 0, "worst_defect": worst, "per_phase": phases}


def level_2_salience_line(path: Path, key: bytes | None, journal: Journal, decay: float, thresh: float) -> dict:
    """For every entry ever consolidated (kept, demoted, or since evicted; an evicted entry's
    line ends at its eviction, where the journal freezes its salience): its salience at consolidation
    as the log recorded it, the number of applied decays it received since (the decay
    events that touched it: its state in the event's states and not contracted, as the
    journal applies them), and the derivation from its final salience alone:
    s_c_derived = s_final / decay ** n."""
    sal: dict[str, float] = {}
    state: dict[str, str] = {}
    contracted: set[str] = set()
    at_cons: dict[str, float] = {}
    decays_after: dict[str, int] = {}
    for ev in iter_events(path, key):
        kind = ev["ev"]
        if kind == "write":
            d = ev["entry"]; sal[d["entry_id"]] = float(d["salience"]); state[d["entry_id"]] = STATE_LIVE
        elif kind == "salience":
            sal[ev["entry_id"]] = float(ev["to"])
            if ev.get("why") == "contract":
                contracted.add(ev["entry_id"])
        elif kind == "transition":
            state[ev["entry_id"]] = ev["to"]
            if ev["to"] == STATE_CONSOLIDATED:
                at_cons[ev["entry_id"]] = sal[ev["entry_id"]]
                decays_after[ev["entry_id"]] = 0
        elif kind == "decay":
            states = tuple(ev["states"]); f = float(ev["factor"])
            for eid, st in state.items():
                if st in states and eid not in contracted:
                    sal[eid] = round(sal[eid] * f, 6)
                    if eid in decays_after:
                        decays_after[eid] += 1
    rows, worst, total = [], 0.0, 0.0
    for eid, e in journal._entries.items():
        if eid not in at_cons:                       # every entry ever consolidated: kept, demoted, or since
            continue                                 # evicted (its line ends at its eviction, its salience frozen there)
        n = decays_after[eid]
        s_final = float(e.salience)
        s_c = at_cons[eid]
        derived = s_final / (decay ** n) if n else s_final
        rel = abs(derived - s_c) / s_c if s_c > 0 else (0.0 if derived == s_c else float("inf"))
        worst = max(worst, rel); total += rel
        rows.append({"entry_id": eid, "state": e.state, "contracted": eid in contracted, "decays": n,
                     "salience_final": s_final, "salience_at_consolidation": s_c,
                     "derived": derived, "relative_defect": rel})
    n_rows = len(rows)
    return {"threshold_relative": thresh, "entries": n_rows, "kept": sum(1 for r in rows if r["state"] != STATE_EVICTED),
            "evicted": sum(1 for r in rows if r["state"] == STATE_EVICTED), "contracted": sum(1 for r in rows if r["contracted"]),
            "under_threshold": sum(1 for r in rows if r["relative_defect"] <= thresh),
            "all_under_threshold": n_rows > 0 and all(r["relative_defect"] <= thresh for r in rows),
            "max_relative_defect": worst, "mean_relative_defect": (total / n_rows) if n_rows else None,
            "anchor": "without the log's count of decays, one number stays free: the origin (3.5, rule 1)",
            "rows": rows}


def level_3_memory(journal: Journal, memory, bound: float, replay_errors: list[float] | None,
                   radius: float | None = None) -> dict:
    """Every replayed cue (the consolidation order), reconstructed from the memory alone.
    With `radius` (the code's near duplicate radius), every evicted address is attributed to
    its nearest kept pattern: the maximal cosine with the cue of a consolidated or demoted
    entry, and whether that neighbour lies within the radius (ADR-007 amendment 2026-10-01)."""
    rows, by_state = [], {STATE_CONSOLIDATED: [], STATE_DEMOTED: [], STATE_EVICTED: []}
    kept_ids = [eid for eid in journal.consolidated_order
                if journal._entries[eid].state in (STATE_CONSOLIDATED, STATE_DEMOTED)]
    kept_cues = None
    if radius is not None and kept_ids:
        kc = np.stack([np.asarray(journal._entries[i].cue, dtype=np.float32) for i in kept_ids])
        kept_cues = kc / (np.linalg.norm(kc, axis=1, keepdims=True) + 1e-8)
    for eid in journal.consolidated_order:
        e = journal._entries[eid]
        cue = np.asarray(e.cue, dtype=np.float32)
        q = cue / (np.linalg.norm(cue) + 1e-8)
        r = memory.reconstruct(q)
        d = float(np.clip(1.0 - float(r @ q / (np.linalg.norm(r) + 1e-8)), 0.0, 1.0)) if r is not None else 1.0
        row = {"entry_id": eid, "state": e.state, "defect": d, "under_bound": d <= bound}
        if radius is not None and e.state == STATE_EVICTED:
            if kept_cues is not None:
                cos = kept_cues @ q
                j = int(np.argmax(cos))
                row["nearest_kept"] = {"entry_id": kept_ids[j], "cosine": float(cos[j]),
                                       "within_radius": bool(cos[j] >= radius)}
            else:
                row["nearest_kept"] = None
        rows.append(row)
        by_state.setdefault(e.state, []).append(d)
    kept = [r for r in rows if r["state"] in (STATE_CONSOLIDATED, STATE_DEMOTED)]
    evicted = [r for r in rows if r["state"] == STATE_EVICTED]
    dist = [r["defect"] for r in rows]
    reading = {"entries": len(evicted),
               "still_reconstructed_under_bound": sum(r["under_bound"] for r in evicted),
               "share": (sum(r["under_bound"] for r in evicted) / len(evicted)) if evicted else None}
    if radius is not None:
        still = [r for r in evicted if r["under_bound"]]
        reading["radius"] = radius
        reading["still_under_bound_with_kept_neighbour_within_radius"] = sum(
            1 for r in still if r.get("nearest_kept") and r["nearest_kept"]["within_radius"])
        reading["still_under_bound_without_such_neighbour"] = sum(
            1 for r in still if not (r.get("nearest_kept") and r["nearest_kept"]["within_radius"]))
        reading["max_defect_evicted"] = max((r["defect"] for r in evicted), default=None)
        reading["min_defect_evicted"] = min((r["defect"] for r in evicted), default=None)
    return {"bound": bound, "patterns": len(memory), "replayed": len(rows),
            "kept": {"entries": len(kept), "under_bound": sum(r["under_bound"] for r in kept),
                     "all_under_bound": len(kept) > 0 and all(r["under_bound"] for r in kept),
                     "max_defect": max((r["defect"] for r in kept), default=None),
                     "mean_defect": (sum(r["defect"] for r in kept) / len(kept)) if kept else None},
            "evicted_address_reading": reading,
            "distribution": {"min": min(dist, default=None), "median": float(np.median(dist)) if dist else None,
                             "p95": float(np.percentile(dist, 95)) if dist else None, "max": max(dist, default=None)},
            "positive_control": ({"replays": len(replay_errors), "max_error_at_replay": max(replay_errors, default=None),
                                  "all_under_bound_at_replay": all(x <= bound for x in replay_errors)}
                                 if replay_errors is not None else "not observable in this process"),
            "rows": rows}


def level_4_content(journal: Journal, jp: JournalPath) -> dict:
    """Declared loss, checked: an evicted entry's payload has no file when its pointer is
    unreferenced, and an evicted entry never comes back from a read."""
    payload_dir = journal.payloads._dir
    evicted = [e for e in journal._entries.values() if e.state == STATE_EVICTED]
    files_present, reads_returning = 0, 0
    for e in evicted:
        referenced = e.pointer in journal.payloads
        if not referenced and payload_dir is not None and (payload_dir / e.pointer).exists():
            files_present += 1
        cue = np.asarray(e.cue, dtype=np.float32)
        if any(h[0].entry_id == e.entry_id for h in jp.retrieve(cue, k=4)):
            reads_returning += 1
    return {"word_by_construction": "declared loss (3.5, rule 5; theorem L2)", "evicted": len(evicted),
            "released_payload_files_present": files_present, "evicted_entries_returned_by_a_read": reads_returning,
            "holds": files_present == 0 and reads_returning == 0}


def levels(copy: Path, key: bytes, journal: Journal, jp: JournalPath, memory, cfg: LifecycleConfig,
           replay_errors: list[float] | None) -> dict:
    return {"level_1_counts": level_1_counts(copy, key, cfg.demote_k),
            "level_2_salience_line": level_2_salience_line(copy, key, journal, cfg.decay, THRESH_SALIENCE_REL),
            "level_3_memory": level_3_memory(journal, memory, cfg.replay_error_max, replay_errors,
                                             radius=cfg.demote_min_cosine if cfg.withdraw_at_eviction else None),
            "level_4_content": level_4_content(journal, jp),
            "snapshot": journal.snapshot()}


def _compare(a: dict, b: dict, path: str = "") -> list[str]:
    """Number for number: the two processes must agree on every scalar of the levels."""
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in ("positive_control", "rows", "per_phase"):
                continue
            diffs += _compare(a.get(k), b.get(k), f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs += _compare(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        if not (a == b or (isinstance(a, float) and isinstance(b, float) and a != a and b != b)):
            diffs.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


# ---------------------------------------------------------------------------- #
# Session A: copy, plant the sentinels, run the phases, measure                  #
# ---------------------------------------------------------------------------- #
def session_a(src: Path, copy_dir: Path, key: bytes, phases: int = PHASES, sentinel_seed: int = SENTINEL_SEED,
              cfg: LifecycleConfig | None = None, withdraw: bool = False) -> tuple[dict, Path]:
    if int(sentinel_seed) in RESERVED_SEEDS:
        raise SystemExit("reconstruction probe: the sentinel seed must differ from the protocol's seeds")
    cfg = cfg or LifecycleConfig(sentinel_seed=int(sentinel_seed), withdraw_at_eviction=bool(withdraw))
    if cfg.sentinel_seed != int(sentinel_seed):
        raise SystemExit("reconstruction probe: the configuration's sentinel seed must be the declared one")
    src = journal_file(src)
    src_hash = sha256_file(src)
    copy = copy_journal(src, copy_dir)
    journal, wp, jp = _open(copy, key)
    n_before = len(journal)
    if journal.phase_count:
        raise SystemExit(f"reconstruction probe: the journal already holds {journal.phase_count} phases; the probe runs on a journal without lifecycle history")
    # the collision abort of row 40: no sentinel entity may be one of the journal's
    sentinels, _ = generate_facts(cfg.sentinel_facts, seed=cfg.sentinel_seed)
    negctrl, _ = generate_facts(cfg.sentinel_negctrl, seed=cfg.sentinel_seed + 10_000)
    domain = entities_of(journal)
    collisions = sorted({f.entity for f in sentinels + negctrl} & domain)
    if collisions:
        raise SystemExit(f"reconstruction probe: sentinel entities collide with the journal's: {collisions}")
    replay_errors: list[float] = []
    now_plant = float(max((e.t_written for e in journal._entries.values()), default=-1.0) + 1.0)
    sched = LifecycleScheduler(journal, wp, jp, cfg=cfg, now=now_plant)      # plants the sentinels (an event each)
    original_replay = sched.memory.replay

    def recorded_replay(cue):
        err = original_replay(cue)
        replay_errors.append(float(err))
        return err
    sched.memory.replay = recorded_replay
    records = []
    for i in range(int(phases)):
        rec = sched.phase(now_plant + 1.0 + i)
        records.append({"phase": rec["phase"], "now": rec["now"], "pressure": rec["pressure"],
                        "floor_eff": rec["floor_eff"], "batch_eff": rec["batch_eff"], "planned": rec["planned"],
                        "sentinel_recall": rec["sentinel_recall"], "sentinel_negctrl": rec["sentinel_negctrl"],
                        "refused": rec["refused"], "reason": rec.get("reason"), "applied": rec.get("applied"),
                        "live": rec.get("live"), "consolidated": rec.get("consolidated"),
                        "demoted": rec.get("demoted"), "evicted": rec.get("evicted"),
                        "summaries": rec.get("summaries"), "memory_patterns": rec.get("memory_patterns")})
    # the replays that failed were withdrawn: they are in replay_errors above the bound, by construction
    withdrawn = sum(1 for x in replay_errors if x > cfg.replay_error_max)
    out = {"probe": "reconstruction on the organ's layer of numbers (ADR-007 amendment 2026-09-30; whitepaper 3.5, 5d door 7)",
           "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "journal": {"source": str(src), "sha256": src_hash, "entries_before": n_before,
                       "separation": separation_of(journal), "sealed": True, "copied_to_temp": True},
           "config": {"lifecycle": cfg.__dict__, "config_hash": cfg.config_hash(), "phases": int(phases),
                      "sentinel_seed": int(sentinel_seed), "index_seed": INDEX_SEED, "now_plant": now_plant,
                      "reads_between_phases": 0, "withdraw_at_eviction": bool(cfg.withdraw_at_eviction)},
           "thresholds": {"level_1": "exact", "level_2_relative": THRESH_SALIENCE_REL,
                          "level_3_bound": cfg.replay_error_max},
           "sentinels": {"planted": len(sched.sentinels), "negctrl": len(sched.negctrl),
                         "collision_check": "passed (no sentinel entity among the journal's)"},
           "phases": records,
           "phases_refused": sum(1 for r in records if r["refused"]),
           "replays": {"count": len(replay_errors), "withdrawn": withdrawn},
           "withdrawals_at_eviction": (sum((r.get("applied") or {}).get("withdrawn", 0) for r in records)
                                       if cfg.withdraw_at_eviction else None),
           "storage": lifecycle_declaration(journal)["storage"],
           "process_a": levels(copy, key, journal, jp, sched.memory, cfg, replay_errors)}
    return out, copy


# ---------------------------------------------------------------------------- #
# Session B: a new process, the disk alone                                       #
# ---------------------------------------------------------------------------- #
def session_b(copy: Path, key: bytes, sentinel_seed: int = SENTINEL_SEED, cfg: LifecycleConfig | None = None,
              withdraw: bool = False) -> dict:
    cfg = cfg or LifecycleConfig(sentinel_seed=int(sentinel_seed), withdraw_at_eviction=bool(withdraw))
    journal, wp, jp = _open(journal_file(copy), key)
    sched = LifecycleScheduler(journal, wp, jp, cfg=cfg)          # rebuilds the memory from the consolidation order
    return {"process": "new (nothing of session A but the disk)", "phases_found": journal.phase_count,
            "sentinels_found": len(sched.sentinels), "durable": journal.mode == MODE_DURABLE,
            **levels(journal_file(copy), key, journal, jp, sched.memory, cfg, None)}


def run(src: Path, key: bytes, phases: int = PHASES, sentinel_seed: int = SENTINEL_SEED, keep_copy: Path | None = None,
        withdraw: bool = False) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="reconstruction-probe-")) if keep_copy is None else keep_copy
    try:
        out, copy = session_a(src, tmp, key, phases=phases, sentinel_seed=sentinel_seed, withdraw=withdraw)
        env = dict(os.environ, QUANTUM_CORTEX_JOURNAL_KEY=key.hex())
        proc = subprocess.run([sys.executable, "-m", "cortex_c2b.reconstruction_probe", "--session-b", str(copy),
                               "--sentinel-seed", str(sentinel_seed)] + (["--withdraw"] if withdraw else []),
                              capture_output=True, text=True, env=env, cwd=str(Path(__file__).resolve().parents[1]))
        if proc.returncode != 0:
            raise RuntimeError(f"session B failed:\n{proc.stderr[-3000:]}")
        b = json.loads(proc.stdout.strip().splitlines()[-1])
        diffs = _compare({k: v for k, v in out["process_a"].items()}, {k: b[k] for k in out["process_a"] if k in b})
        out["process_b"] = b
        out["agreement"] = {"number_for_number": not diffs, "differences": diffs[:50]}
        a = out["process_a"]
        out["readings"] = {
            "level_1_exact": a["level_1_counts"]["exact_at_every_phase"],
            "level_2_under_threshold": a["level_2_salience_line"]["all_under_threshold"],
            "level_3_kept_under_bound": a["level_3_memory"]["kept"]["all_under_bound"],
            "level_3b_evicted_address_share": a["level_3_memory"]["evicted_address_reading"]["share"],
            "level_4_holds": a["level_4_content"]["holds"],
            "two_processes_agree": not diffs,
            "withdraw_at_eviction": bool(out["config"]["withdraw_at_eviction"]),
            "level_3b_evicted": a["level_3_memory"]["evicted_address_reading"]["entries"],
            "level_3b_still_under_bound": a["level_3_memory"]["evicted_address_reading"]["still_reconstructed_under_bound"],
            "level_3b_with_neighbour": a["level_3_memory"]["evicted_address_reading"].get("still_under_bound_with_kept_neighbour_within_radius"),
        }
        out["verdict"] = _verdict(out["readings"])
        return out
    finally:
        if keep_copy is None:
            shutil.rmtree(tmp, ignore_errors=True)


def _verdict(r: dict) -> str:
    if not r["level_1_exact"] or not r["two_processes_agree"] or not r["level_4_holds"]:
        return "CONSTRUCTION FAULT -- no row (reading d)"
    words = ["level 1: theorem by construction", "level 4: declared loss by construction"]
    words.append("level 2: reconstructible at measured defect" if r["level_2_under_threshold"] else "level 2: residual above the bound, to be attributed (reading c)")
    words.append("level 3: reconstructible at measured defect" if r["level_3_kept_under_bound"] else "level 3: backup, the memory's capacity is the measured limit (reading b)")
    share = r["level_3b_evicted_address_share"]
    if r.get("withdraw_at_eviction"):
        words.append("level 3b (withdrawal at eviction): " + ("no evicted entry" if share is None else
                     ("the withdrawal is exact at the memory's level and no withdrawn address is reconstructed from the kept patterns (reading b1)" if share == 0 else
                      f"{r['level_3b_still_under_bound']} of {r['level_3b_evicted']} withdrawn addresses are still reconstructed under the bound from the kept patterns, "
                      f"{r['level_3b_with_neighbour']} of them with a kept neighbour within the radius (reading b2 if all, b3 otherwise)")))
    else:
        words.append("level 3b: " + ("no evicted entry" if share is None else
                                     ("the address of an evicted episode survives its eviction in the memory" if share > 0 else "the declared loss covers the address too")))
    return "; ".join(words)


def summary_lines(out: dict) -> list[str]:
    a, r = out["process_a"], out["readings"]
    l2, l3 = a["level_2_salience_line"], a["level_3_memory"]
    return [
        f"journal {out['journal']['sha256'][:16]}  entries before {out['journal']['entries_before']}  phases {out['config']['phases']}  refused {out['phases_refused']}",
        f"end state: live {a['snapshot']['live']}  consolidated {a['snapshot']['consolidated']}  demoted {a['snapshot']['demoted']}  evicted {a['snapshot']['evicted']}  summaries {a['snapshot']['summaries']}",
        f"level 1 (counts): exact at every phase = {r['level_1_exact']} (worst defect {a['level_1_counts']['worst_defect']})",
        f"level 2 (salience line): {l2['under_threshold']}/{l2['entries']} under 1e-3; max relative defect {l2['max_relative_defect']:.3e}",
        f"level 3 (memory): kept {l3['kept']['under_bound']}/{l3['kept']['entries']} under {l3['bound']}; max defect {l3['kept']['max_defect']}; replays {out['replays']['count']}, withdrawn {out['replays']['withdrawn']}",
        f"level 3b (evicted address{', withdrawn at eviction' if out['config'].get('withdraw_at_eviction') else ''}): "
        f"{l3['evicted_address_reading']['still_reconstructed_under_bound']}/{l3['evicted_address_reading']['entries']} still reconstructed under the bound"
        + (f"; {l3['evicted_address_reading'].get('still_under_bound_with_kept_neighbour_within_radius')} with a kept neighbour within {l3['evicted_address_reading'].get('radius')}"
           if out['config'].get('withdraw_at_eviction') else ""),
        f"level 4 (content): holds = {r['level_4_holds']}",
        f"two processes agree number for number: {r['two_processes_agree']}",
        f"verdict: {out['verdict']}",
    ]


def _cli(argv=None):
    ap = argparse.ArgumentParser(description="the reconstruction probe on the organ's layer of numbers")
    ap.add_argument("--journal", default=None, help="a sealed journal of the record (the journal.jsonl or its directory); copied, never modified")
    ap.add_argument("--out", default=None, help="the directory or file for the retained record")
    ap.add_argument("--run-id", default=None, help="the run whose journal this is (names the file)")
    ap.add_argument("--phases", type=int, default=PHASES)
    ap.add_argument("--sentinel-seed", type=int, default=SENTINEL_SEED)
    ap.add_argument("--session-b", default=None, help="(internal) reopen this copy in a new process and measure")
    ap.add_argument("--withdraw", action="store_true", help="withdraw_at_eviction on (ADR-007 amendment 2026-10-01); off by default")
    args = ap.parse_args(argv)
    from cortex_c2b.crypto import key_from_env
    key = key_from_env()
    if key is None:
        raise SystemExit("QUANTUM_CORTEX_JOURNAL_KEY missing")
    if args.session_b:
        print(json.dumps(session_b(Path(args.session_b), key, sentinel_seed=args.sentinel_seed, withdraw=args.withdraw)))
        return
    if not args.journal:
        raise SystemExit("--journal is required")
    out = run(Path(args.journal), key, phases=args.phases, sentinel_seed=args.sentinel_seed, withdraw=args.withdraw)
    for line in summary_lines(out):
        print(line)
    if args.out:
        o = Path(args.out)
        if o.is_dir() or not o.suffix:
            o.mkdir(parents=True, exist_ok=True)
            o = o / f"reconstruction-probe-{args.run_id or out['journal']['sha256'][:12]}.json"
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"[record] retained: {o}")


if __name__ == "__main__":
    _cli()
