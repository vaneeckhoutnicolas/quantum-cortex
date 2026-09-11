"""cortex_eval.resumable_ladder — the 8-seed ladder that survives a quota cut.

The lesson of run B (2026-09-09): a run that writes only at the end loses
everything when Kaggle kills the session. This runner makes the ladder
**resumable at the unit level**:

  unit = (rung, seed, tier) → one small JSON file, written the moment the unit
         finishes, in a checkpoint directory.

On start, the runner scans the checkpoint directory (and any *previous*
checkpoint directories mounted as Kaggle Inputs), loads every finished unit,
and skips them. A session cut at unit 37 of 80 restarts at unit 38. Nothing is
ever recomputed; nothing is ever lost.

Kaggle specifics: a killed session loses /kaggle/working, but the Output of the
previous *version* is retained and can be mounted as an Input of the next
version. So we READ from every directory in `resume_from` (the mounted Inputs)
and WRITE to `ckpt_dir` (the current Output). Same pattern as the N1 trainer's
--resume from an Input-mounted checkpoint.

Time budget: `--time-budget-min` stops cleanly before the quota wall (units
are small, ~6-10 min each, so the loss at the wall is at most one unit).

When every unit exists, the aggregate (per-rung stats, CIs, paired tests) is
computed from the unit files — deterministic, reproducible from the units alone.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cortex_eval.mqar import MQARTier
from cortex_eval.recurrent_ladder import LADDER, Rung, run_rung
from cortex_eval.multiseed import _mean_ci, _paired_test

DEFAULT_SEEDS = (1337, 2024, 7, 42, 99, 3, 11, 2026)      # 8 seeds → df 7, t_crit 2.365
DEFAULT_TIERS = [MQARTier(kv_pairs=16, seq_len=128), MQARTier(kv_pairs=8, seq_len=128)]


def unit_id(rung: Rung, seed: int, tier: MQARTier) -> str:
    return f"{rung.name}__s{seed}__kv{tier.kv_pairs}_seq{tier.seq_len}"


def load_done(dirs: list[Path]) -> dict[str, dict]:
    """Every finished unit across all given directories (later dirs win on duplicates)."""
    done = {}
    for d in dirs:
        if not d or not d.exists():
            continue
        for p in sorted(d.glob("unit-*.json")):
            try:
                u = json.loads(p.read_text())
                if u.get("status") == "done" and u.get("accuracy") == u.get("accuracy"):
                    done[u["unit_id"]] = u
            except Exception:
                continue
    return done


def run_resumable_ladder(seeds=DEFAULT_SEEDS, tiers=None, steps: int = 1500,
                         ckpt_dir: str | Path = "metrics/mqar/ladder8-ckpt",
                         resume_from: list[str | Path] | None = None,
                         time_budget_min: float | None = None, **kw) -> dict:
    tiers = tiers or DEFAULT_TIERS
    ckpt = Path(ckpt_dir); ckpt.mkdir(parents=True, exist_ok=True)
    dirs = [Path(p) for p in (resume_from or [])] + [ckpt]
    done = load_done(dirs)
    units = [(r, s, t) for r in LADDER for s in seeds for t in tiers]
    todo = [u for u in units if unit_id(*u) not in done]
    print(f"[resumable] {len(units)} units total, {len(done)} already done, {len(todo)} to run")
    t0 = time.time()
    stopped_early = False
    for rung, seed, tier in todo:
        if time_budget_min is not None and (time.time() - t0) / 60.0 > time_budget_min:
            print(f"[resumable] time budget reached — stopping cleanly; rerun to resume")
            stopped_early = True
            break
        uid = unit_id(rung, seed, tier)
        t1 = time.time()
        acc = run_rung(rung, tier, steps, seed, **kw)
        rec = {"unit_id": uid, "rung": rung.name, "seed": seed, "kv_pairs": tier.kv_pairs,
               "seq_len": tier.seq_len, "steps": steps, "accuracy": acc,
               "status": "done" if acc == acc else "diverged",
               "wall_s": round(time.time() - t1, 1), "written_utc": datetime.now(timezone.utc).isoformat()}
        (ckpt / f"unit-{uid}.json").write_text(json.dumps(rec) + "\n")   # written THE MOMENT it finishes
        done[uid] = rec
        print(f"[unit] {uid} -> acc={acc:.4f} ({rec['wall_s']}s)  [{len(done)}/{len(units)}]")
    complete = all(unit_id(*u) in done for u in units)
    result = {"benchmark": "recurrent_ladder_8seed_resumable",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "regime": {"seeds": list(seeds), "n_seeds": len(seeds), "steps": steps,
                         "tiers": [(t.kv_pairs, t.seq_len) for t in tiers]},
              "units_done": len(done), "units_total": len(units), "complete": complete,
              "stopped_early": stopped_early}
    if complete:
        result.update(aggregate(done, seeds, tiers))
        (ckpt / "LATEST-ladder8.json").write_text(json.dumps(result, indent=2) + "\n")
        print("[resumable] COMPLETE — aggregate written")
    else:
        (ckpt / "PROGRESS-ladder8.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"[resumable] incomplete ({len(done)}/{len(units)}) — progress written; rerun with this dir as Input to resume")
    return result


def aggregate(done: dict, seeds, tiers) -> dict:
    """Per-rung AUC per seed (mean over tiers), then stats + paired tests. Deterministic
    from the unit files alone — re-runnable offline on the committed units."""
    per_rung: dict[str, list[float]] = {}
    for r in LADDER:
        vals = []
        for s in seeds:
            accs = [done[unit_id(r, s, t)]["accuracy"] for t in tiers]
            vals.append(float(np.mean(accs)))
        per_rung[r.name] = vals
    stats = {r: _mean_ci(v) for r, v in per_rung.items()}
    best_pure = max(per_rung, key=lambda r: np.mean(per_rung[r]))
    return {"per_rung_per_seed": per_rung, "stats": stats, "best_pure_rung": best_pure,
            "L4_vs_L3": _paired_test(per_rung["L4-+delta-rule"], per_rung["L3-+local-conv"]),
            "note": "hybrid-vs-best-pure needs the hybrids' per-seed AUC at the same regime "
                    "(from the C2 ablation units, or a matching hybrid run) — join on seed."}


def _cli():
    ap = argparse.ArgumentParser(description="Resumable 8-seed recurrent ladder")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--ckpt-dir", default="metrics/mqar/ladder8-ckpt")
    ap.add_argument("--resume-from", nargs="*", default=[],
                    help="previous checkpoint dirs (e.g. Kaggle Inputs) whose finished units are reused")
    ap.add_argument("--time-budget-min", type=float, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        run_resumable_ladder(seeds=(1, 2), tiers=[MQARTier(kv_pairs=4, seq_len=32)], steps=40,
                             ckpt_dir=args.ckpt_dir, resume_from=args.resume_from, d_model=32)
    else:
        run_resumable_ladder(seeds=tuple(args.seeds), steps=args.steps, ckpt_dir=args.ckpt_dir,
                             resume_from=args.resume_from, time_budget_min=args.time_budget_min)


if __name__ == "__main__":
    _cli()
