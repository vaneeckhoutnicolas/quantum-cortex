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
from cortex_eval.run_mqar import run_curriculum

# The three ARCHITECTURE paths measured alongside the pure ladder, same seeds, same
# tiers, same steps — so every comparison below is a PAIRED test on identical data.
ARCH = ("none", "hopfield", "delta")     # control / Hopfield hybrid / delta hybrid

DEFAULT_SEEDS = (1337, 2024, 7, 42, 99, 3, 11, 2026)      # 8 seeds → df 7, t_crit 2.365
DEFAULT_TIERS = [MQARTier(kv_pairs=16, seq_len=128), MQARTier(kv_pairs=8, seq_len=128)]


def unit_id(rung, seed: int, tier: MQARTier) -> str:
    """rung is a Rung (pure ladder) or a str in ARCH (control / hybrids)."""
    name = rung.name if isinstance(rung, Rung) else f"ARCH-{rung}"
    return f"{name}__s{seed}__kv{tier.kv_pairs}_seq{tier.seq_len}"


def run_unit(rung, tier: MQARTier, steps: int, seed: int, **kw) -> float:
    if isinstance(rung, Rung):
        return run_rung(rung, tier, steps, seed, **kw)
    res = run_curriculum(c2_variant=rung, steps=steps, tiers=[tier], seed=seed, **kw)
    return float(res.per_tier[0]["accuracy"])


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
                         time_budget_min: float | None = None, paths: list[str] | None = None,
                         relay: str | None = None, **kw) -> dict:
    """`paths` (optional): restrict the run to these path names (e.g. the sigma run of
    the convergence, L3 alone at 3000 steps on the seeds that stayed at the floor).
    A subset writes its units and a SUBSET aggregate (per path stats, no paired
    tests against absent paths); the full readout needs all eight paths."""
    tiers = tiers or DEFAULT_TIERS
    ckpt = Path(ckpt_dir); ckpt.mkdir(parents=True, exist_ok=True)
    if relay:                                            # the checkpoint relay: the last push, before anything else
        import tempfile
        from cortex_data import relay as _relay
        pulled = Path(tempfile.mkdtemp(prefix="relay-")) / ckpt.name
        if _relay.pull(relay, pulled):
            resume_from = list(resume_from or []) + [pulled]
    dirs = [Path(p) for p in (resume_from or [])] + [ckpt]
    done = load_done(dirs)
    all_paths = list(LADDER) + list(ARCH)
    if paths:
        names = {(r.name if isinstance(r, Rung) else f"ARCH-{r}"): r for r in all_paths}
        unknown = [n for n in paths if n not in names]
        if unknown:
            raise SystemExit(f"unknown paths {unknown}; known: {sorted(names)}")
        all_paths = [names[n] for n in paths]
    units = [(r, s, t) for r in all_paths for s in seeds for t in tiers]
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
        acc = run_unit(rung, tier, steps, seed, **kw)
        rec = {"unit_id": uid, "rung": rung.name if isinstance(rung, Rung) else f"ARCH-{rung}", "seed": seed, "kv_pairs": tier.kv_pairs,
               "seq_len": tier.seq_len, "steps": steps, "accuracy": acc,
               "status": "done" if acc == acc else "diverged",
               "wall_s": round(time.time() - t1, 1), "written_utc": datetime.now(timezone.utc).isoformat()}
        (ckpt / f"unit-{uid}.json").write_text(json.dumps(rec) + "\n")   # written THE MOMENT it finishes
        done[uid] = rec
        if relay:
            from cortex_data import relay as _relay
            _relay.push(ckpt, relay, uid)                                  # small files: a blocking push, seconds
        print(f"[unit] {uid} -> acc={acc:.4f} ({rec['wall_s']}s)  [{len(done)}/{len(units)}]")
    complete = all(unit_id(*u) in done for u in units)
    result = {"benchmark": "recurrent_ladder_8seed_resumable",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "regime": {"seeds": list(seeds), "n_seeds": len(seeds), "steps": steps,
                         "tiers": [(t.kv_pairs, t.seq_len) for t in tiers]},
              "units_done": len(done), "units_total": len(units), "complete": complete,
              "stopped_early": stopped_early}
    if complete and paths:
        result["subset"] = list(paths)
        result["stats"] = {}
        for r in all_paths:
            name = r.name if isinstance(r, Rung) else f"ARCH-{r}"
            vals = [float(np.mean([done[unit_id(r, s, t)]["accuracy"] for t in tiers])) for s in seeds]
            per_tier = {f"kv{t.kv_pairs}": [done[unit_id(r, s, t)]["accuracy"] for s in seeds] for t in tiers}
            result["stats"][name] = {**_mean_ci(vals), "per_seed": vals, "per_tier": per_tier}
        tag = "-".join(paths).replace("+", "").replace("/", "")[:40]
        (ckpt / f"SUBSET-{tag}-{steps}steps.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"[resumable] COMPLETE (subset {paths}) — per path stats written; no paired tests on a subset")
    elif complete:
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
    per_arch: dict[str, list[float]] = {}
    for a in ARCH:
        vals = []
        for s in seeds:
            accs = [done[unit_id(a, s, t)]["accuracy"] for t in tiers]
            vals.append(float(np.mean(accs)))
        per_arch[a] = vals
    stats = {r: _mean_ci(v) for r, v in {**per_rung, **{f"ARCH-{a}": v for a, v in per_arch.items()}}.items()}
    best_pure = max(per_rung, key=lambda r: np.mean(per_rung[r]))
    n = len(seeds)
    tests = {
        "hopfield_hybrid_vs_best_pure": _paired_test(per_arch["hopfield"], per_rung[best_pure]),
        "delta_hybrid_vs_best_pure":    _paired_test(per_arch["delta"],    per_rung[best_pure]),
        "control_vs_best_pure":         _paired_test(per_arch["none"],     per_rung[best_pure]),
        "hopfield_vs_control":          _paired_test(per_arch["hopfield"], per_arch["none"]),
        "delta_vs_control":             _paired_test(per_arch["delta"],    per_arch["none"]),
        "L4_vs_L3":                     _paired_test(per_rung["L4-+delta-rule"], per_rung["L3-+local-conv"]),
    }
    gate = {k: bool(v.get("significant_95")) and n >= 3 for k, v in tests.items()}
    return {"per_rung_per_seed": per_rung, "per_arch_per_seed": per_arch, "stats": stats,
            "best_pure_rung": best_pure, "paired_tests": tests, "gate_readout": gate,
            "note": "every test is paired on seed at identical steps/tiers; gated = significant at 95% with >= 3 seeds."}


def _cli():
    ap = argparse.ArgumentParser(description="Resumable 8-seed recurrent ladder")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--ckpt-dir", default="metrics/mqar/ladder8-ckpt")
    ap.add_argument("--resume-from", nargs="*", default=[],
                    help="previous checkpoint dirs (e.g. Kaggle Inputs) whose finished units are reused")
    ap.add_argument("--time-budget-min", type=float, default=None)
    ap.add_argument("--paths", nargs="*", default=None,
                    help="restrict to these path names (e.g. L3-+local-conv for the sigma run)")
    ap.add_argument("--tiers", nargs="*", default=None, help="restrict to these tiers as KVxSEQ (e.g. 8x128 for the tier where L3 takes off); a subset aggregate, never a paired claim across tiers")
    ap.add_argument("--relay", default=None, help="checkpoint relay: a private Kaggle dataset slug pulled before the resume lookup and pushed after every unit")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    tiers = parse_tiers(args.tiers)
    if args.quick:
        run_resumable_ladder(seeds=(1, 2), tiers=tiers or [MQARTier(kv_pairs=4, seq_len=32)], steps=40,
                             ckpt_dir=args.ckpt_dir, resume_from=args.resume_from, d_model=32, paths=args.paths, relay=args.relay)
    else:
        run_resumable_ladder(seeds=tuple(args.seeds), tiers=tiers, steps=args.steps, ckpt_dir=args.ckpt_dir,
                             resume_from=args.resume_from, time_budget_min=args.time_budget_min, paths=args.paths, relay=args.relay)


def parse_tiers(spec):
    """'8x128 16x128' -> [MQARTier(8, 128), MQARTier(16, 128)]; None -> None (the default tiers)."""
    if not spec:
        return None
    out = []
    for s in spec:
        kv, seq = s.lower().split("x")
        out.append(MQARTier(kv_pairs=int(kv), seq_len=int(seq)))
    return out


if __name__ == "__main__":
    _cli()
