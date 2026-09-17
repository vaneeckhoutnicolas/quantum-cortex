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
REF_CONTROL_CONV = "ARCH-none+local-conv"      # the control plus a local convolution (Rev60), a declared reference


def rb_unit_id(arm: RBArm, seed: int, tier: MQARTier) -> str:
    return f"{arm.name}__s{seed}__kv{tier.kv_pairs}_seq{tier.seq_len}"


def ref_unit_id(name: str, seed: int, tier: MQARTier) -> str:
    return f"{name}__s{seed}__kv{tier.kv_pairs}_seq{tier.seq_len}"


def run_resumable_rb(seeds=DEFAULT_SEEDS, tiers=None, steps: int = 1500, relay: str | None = None,
                     ckpt_dir: str | Path = "metrics/mqar/rb-ckpt",
                     resume_from: list[str | Path] | None = None,
                     reference_from: list[str | Path] | None = None,
                     time_budget_min: float | None = None, arms: list[str] | None = None, **kw) -> dict:
    tiers = tiers or DEFAULT_TIERS
    ckpt = Path(ckpt_dir); ckpt.mkdir(parents=True, exist_ok=True)
    if relay:                                            # the checkpoint relay: the last push, before anything else
        import tempfile
        from cortex_data import relay as _relay
        pulled = Path(tempfile.mkdtemp(prefix="relay-")) / ckpt.name
        if _relay.pull(relay, pulled):
            resume_from = list(resume_from or []) + [pulled]
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
        if relay:
            from cortex_data import relay as _relay
            _relay.push(ckpt, relay, uid)
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


def control_conv_readout(refs: dict, done: dict, arms, seeds, tiers) -> dict:
    """Rev60, declared before the run: the control plus a local convolution against
    the control (does the convolution lift the two layer transformer), against L3
    pure (does it reach the recurrent trunk's level) and against each measured arm's
    full accuracy (does the arm add anything beyond a convolved transformer). Paired
    on seed at identical steps and tier; gated = significant at 95 % with >= 3 seeds.
    Deterministic from the unit files alone."""
    def ref_values(name):
        return [float(np.mean([refs[ref_unit_id(name, s, t)]["accuracy"] for t in tiers])) for s in seeds]
    cc, ctrl, l3 = ref_values(REF_CONTROL_CONV), ref_values(REF_CONTROL), ref_values(REF_L3)
    tests = {"control_conv_vs_control": _paired_test(cc, ctrl), "control_conv_vs_L3": _paired_test(cc, l3)}
    per_full = {}
    for a in arms:
        try:
            per_full[a.name] = [float(np.mean([done[rb_unit_id(a, s, t)]["accuracy"] for t in tiers])) for s in seeds]
        except KeyError:
            continue
        tests[f"{a.name}_full_vs_control_conv"] = _paired_test(per_full[a.name], cc)
    gate = {k: ("gated" if (v.get("significant_95") and len(seeds) >= 3) else "held") for k, v in tests.items()}
    return {"benchmark": "control_plus_local_conv_readout", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "regime": {"seeds": list(seeds), "n_seeds": len(seeds), "tiers": [(t.kv_pairs, t.seq_len) for t in tiers]},
            "stats": {REF_CONTROL_CONV: _mean_ci(cc), REF_CONTROL: _mean_ci(ctrl), REF_L3: _mean_ci(l3)},
            "per_seed": {REF_CONTROL_CONV: cc, REF_CONTROL: ctrl, REF_L3: l3, "arms_full": per_full},
            "paired_tests": tests, "gate_readout": gate,
            "note": "Rev60: the readings were declared before the run; a subset readout, paired on seed at identical steps and tier"}


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
    ap.add_argument("--tiers", nargs="*", default=None, help="restrict to these tiers as KVxSEQ (e.g. 8x128 for the tier where L3 takes off); a subset aggregate, never a paired claim across tiers")
    ap.add_argument("--relay", default=None, help="checkpoint relay: a private Kaggle dataset slug pulled before the resume lookup and pushed after every unit")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--control-conv-readout", default=None, metavar="OUT",
                    help="Rev60: no training; read the control plus convolution units from --reference-from and the arms from "
                         "--ckpt-dir, compute the declared paired tests, write them to OUT")
    args = ap.parse_args()
    from cortex_eval.resumable_ladder import parse_tiers
    tiers = parse_tiers(args.tiers)
    if args.control_conv_readout:
        refs = load_done([Path(p) for p in args.reference_from])
        done = load_done([Path(args.ckpt_dir)])
        arms = [ARM_BY_NAME[n] for n in (args.arms or [a.name for a in ARMS])]
        r = control_conv_readout(refs, done, arms, tuple(args.seeds), tiers or list(DEFAULT_TIERS))
        out = Path(args.control_conv_readout); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=2) + "\n")
        print(out); print(json.dumps({k: r[k] for k in ("stats", "paired_tests", "gate_readout")}, indent=1))
        return
    if args.quick:
        run_resumable_rb(seeds=(1, 2), tiers=tiers or [MQARTier(kv_pairs=4, seq_len=32)], steps=40, ckpt_dir=args.ckpt_dir,
                         resume_from=args.resume_from, reference_from=args.reference_from, arms=args.arms, d_model=32, relay=args.relay)
    else:
        run_resumable_rb(seeds=tuple(args.seeds), tiers=tiers, steps=args.steps, ckpt_dir=args.ckpt_dir, resume_from=args.resume_from,
                         reference_from=args.reference_from, time_budget_min=args.time_budget_min, arms=args.arms, relay=args.relay)


if __name__ == "__main__":
    _cli()
