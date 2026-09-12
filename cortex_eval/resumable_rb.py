"""cortex_eval.resumable_rb -- the recurrent base hybrid run, resumable (Rev38; declared before the run).

64 units = 4 arms (bare, critical, cls, cls-phi) x 8 seeds x 2 tiers, at the
ladder's 1500 steps, each written the moment it finishes. The references are
NOT retrained: the 8 seed ladder's units (`metrics/mqar/ladder8-ckpt/`) give L3
pure and the transformer control at the same seeds, tiers and steps, so every
comparison is paired on seed on identical data.

The readout, declared 2026-09-12 before any unit ran:
  per arm: mean full accuracy (mean over tiers per seed), trunk only accuracy,
           take off rate (trunk only at 8 pairs >= 0.3), the 16 pair mean, the
           consolidation gap, the gate at the end;
  paired tests (95 %, n >= 3): each arm's full model vs L3 pure; each arm's full
           model vs the control; each arm's trunk only vs L3 pure (did the base
           keep or gain its take off?); cls-phi vs cls (phi against 1/2).
  gated = significant at 95 %; a bimodal comparator gets its reserve written
           next to the number, as for the ladder.
No number here is a result until the units exist and the log is read cold.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cortex_eval.mqar import MQARTier
from cortex_eval.multiseed import _mean_ci, _paired_test
from cortex_eval.recurrent_base import ARMS, ARM_BY_NAME, RBArm, run_rb_unit
from cortex_eval.resumable_ladder import DEFAULT_SEEDS, DEFAULT_TIERS, load_done

REF_L3, REF_CONTROL = "L3-+local-conv", "ARCH-none"


def rb_unit_id(arm: RBArm, seed: int, tier: MQARTier) -> str:
    return f"{arm.name}__s{seed}__kv{tier.kv_pairs}_seq{tier.seq_len}"


def ref_unit_id(name: str, seed: int, tier: MQARTier) -> str:
    return f"{name}__s{seed}__kv{tier.kv_pairs}_seq{tier.seq_len}"


def run_resumable_rb(seeds=DEFAULT_SEEDS, tiers=None, steps: int = 1500,
                     ckpt_dir: str | Path = "metrics/mqar/rb-ckpt",
                     resume_from: list[str | Path] | None = None,
                     reference_from: list[str | Path] | None = None,
                     time_budget_min: float | None = None, arms: list[str] | None = None, **kw) -> dict:
    tiers = tiers or DEFAULT_TIERS
    ckpt = Path(ckpt_dir); ckpt.mkdir(parents=True, exist_ok=True)
    done = load_done([Path(p) for p in (resume_from or [])] + [ckpt])
    refs = load_done([Path(p) for p in (reference_from or [])])
    chosen = [ARM_BY_NAME[a] for a in arms] if arms else list(ARMS)
    units = [(a, s, t) for a in chosen for s in seeds for t in tiers]
    todo = [u for u in units if rb_unit_id(*u) not in done]
    print(f"[resumable-rb] {len(units)} units total, {len(done)} already done, {len(todo)} to run; "
          f"{len(refs)} reference units loaded")
    t0 = time.time(); stopped_early = False
    for arm, seed, tier in todo:
        if time_budget_min is not None and (time.time() - t0) / 60.0 > time_budget_min:
            print("[resumable-rb] time budget reached \u2014 stopping cleanly; rerun to resume")
            stopped_early = True
            break
        uid = rb_unit_id(arm, seed, tier)
        t1 = time.time()
        r = run_rb_unit(arm, tier, steps, seed, **kw)
        rec = {"unit_id": uid, "rung": arm.name, "arm": arm.name, "seed": seed, "kv_pairs": tier.kv_pairs,
               "seq_len": tier.seq_len, "steps": steps, **r,
               "wall_s": round(time.time() - t1, 1), "written_utc": datetime.now(timezone.utc).isoformat()}
        (ckpt / f"unit-{uid}.json").write_text(json.dumps(rec) + "\n")   # written the moment it finishes
        done[uid] = rec
        print(f"[unit] {uid} -> acc={r['accuracy']:.4f} trunk={r.get('accuracy_trunk_only', float('nan')):.4f} "
              f"gate={r.get('gate_final')} ({rec['wall_s']}s)  [{len(done)}/{len(units)}]")
    complete = all(rb_unit_id(*u) in done for u in units)
    result = {"benchmark": "recurrent_base_hybrid_resumable", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "regime": {"seeds": list(seeds), "n_seeds": len(seeds), "steps": steps,
                         "tiers": [(t.kv_pairs, t.seq_len) for t in tiers], "arms": [a.name for a in chosen],
                         "arm_hashes": {a.name: a.config_hash() for a in chosen}},
              "units_done": len(done), "units_total": len(units), "complete": complete, "stopped_early": stopped_early}
    if complete:
        result.update(aggregate(done, refs, chosen, seeds, tiers))
        (ckpt / "LATEST-rb.json").write_text(json.dumps(result, indent=2) + "\n")
        print("[resumable-rb] COMPLETE \u2014 aggregate written")
    else:
        (ckpt / "PROGRESS-rb.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"[resumable-rb] incomplete ({len(done)}/{len(units)}) \u2014 progress written; rerun with this dir as Input")
    return result


def aggregate(done: dict, refs: dict, arms, seeds, tiers) -> dict:
    """Deterministic from the unit files alone; references from the ladder8 units."""
    kv8 = [t for t in tiers if t.kv_pairs == 8]
    kv16 = [t for t in tiers if t.kv_pairs == 16]
    per_full, per_trunk, stats = {}, {}, {}
    for a in arms:
        full = [float(np.mean([done[rb_unit_id(a, s, t)]["accuracy"] for t in tiers])) for s in seeds]
        trunk = [float(np.mean([done[rb_unit_id(a, s, t)]["accuracy_trunk_only"] for t in tiers])) for s in seeds]
        per_full[a.name], per_trunk[a.name] = full, trunk
        st = {"full": _mean_ci(full), "trunk_only": _mean_ci(trunk),
              "take_off_rate": (float(np.mean([done[rb_unit_id(a, s, t)]["took_off"] for s in seeds for t in kv8])) if kv8 else None),
              "kv16_mean": (float(np.mean([done[rb_unit_id(a, s, t)]["accuracy"] for s in seeds for t in kv16])) if kv16 else None),
              "gap_mean": float(np.mean([done[rb_unit_id(a, s, t)]["consolidation_gap"] for s in seeds for t in tiers])),
              "gate_final_mean": float(np.mean([done[rb_unit_id(a, s, t)]["gate_final"] for s in seeds for t in tiers]))}
        stats[a.name] = st
    tests, missing = {}, []
    def ref_values(name):
        try:
            return [float(np.mean([refs[ref_unit_id(name, s, t)]["accuracy"] for t in tiers])) for s in seeds]
        except KeyError:
            missing.append(name); return None
    l3, ctrl = ref_values(REF_L3), ref_values(REF_CONTROL)
    for a in arms:
        if l3 is not None:
            tests[f"{a.name}_full_vs_L3"] = _paired_test(per_full[a.name], l3)
            tests[f"{a.name}_trunk_vs_L3"] = _paired_test(per_trunk[a.name], l3)
        if ctrl is not None:
            tests[f"{a.name}_full_vs_control"] = _paired_test(per_full[a.name], ctrl)
    if "RB-cls-phi" in per_full and "RB-cls" in per_full:
        tests["RB-cls-phi_vs_RB-cls"] = _paired_test(per_full["RB-cls-phi"], per_full["RB-cls"])
    gate = {k: ("gated" if (v.get("significant_95") and len(seeds) >= 3) else "held") for k, v in tests.items()}
    return {"stats": stats, "per_seed": {"full": per_full, "trunk_only": per_trunk},
            "references": {"L3": l3, "control": ctrl, "missing": sorted(set(missing))},
            "paired_tests": tests, "gate_readout": gate,
            "note": "paired on seed at identical steps/tiers against the ladder8 units; gated = significant at 95% with >= 3 seeds; "
                    "a bimodal comparator (L3) gets its reserve written next to the number."}


def _cli():
    ap = argparse.ArgumentParser(description="Resumable recurrent base hybrid run (4 arms)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--arms", nargs="*", default=None, help=f"subset of {[a.name for a in ARMS]}")
    ap.add_argument("--ckpt-dir", default="metrics/mqar/rb-ckpt")
    ap.add_argument("--resume-from", nargs="*", default=[])
    ap.add_argument("--reference-from", nargs="*", default=["metrics/mqar/ladder8-ckpt"],
                    help="dirs holding the ladder8 units (L3 pure and the control at the same seeds/tiers/steps)")
    ap.add_argument("--time-budget-min", type=float, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        run_resumable_rb(seeds=(1, 2), tiers=[MQARTier(kv_pairs=4, seq_len=32)], steps=40, ckpt_dir=args.ckpt_dir,
                         resume_from=args.resume_from, reference_from=args.reference_from, arms=args.arms, d_model=32)
    else:
        run_resumable_rb(seeds=tuple(args.seeds), steps=args.steps, ckpt_dir=args.ckpt_dir, resume_from=args.resume_from,
                         reference_from=args.reference_from, time_budget_min=args.time_budget_min, arms=args.arms)


if __name__ == "__main__":
    _cli()
