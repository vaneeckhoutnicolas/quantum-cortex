"""cortex_eval.confirmation_run — the run that unlocks whitepaper v0.

One GPU job that turns today's single-seed signals into gate-passing claims
(WHITEPAPER.md admission gate: ≥3 seeds, significant, anchored, real, reproducible).

Three parts, all at the same regime (steps, seeds), on the DISCRIMINATING tiers —
those where the variants actually separated in the 12-tier run (kv 8/16 at
seq 128/256) — so the job fits one Kaggle session:
  1. C2 ablation × 5 seeds  → paired t-tests, CIs  (hopfield/delta vs control)
  2. Recurrent ladder × 5 seeds → is "delta-rule pays only when hybridised" real?
  3. External anchor × 5 seeds  → the three families, statistically

Everything is written to metrics/mqar/confirmation-*.json. Nothing is claimed
here; the artefact feeds the whitepaper's §4 only where the gate passes.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cortex_eval.mqar import MQARTier
from cortex_eval.multiseed import run_multiseed, _mean_ci, _paired_test
from cortex_eval.recurrent_ladder import run_ladder, LADDER
from cortex_eval.external_baseline import _run_mlp_baseline

# the discriminating tiers (from the 12-tier ablation: where the paths separated)
DISCRIMINATING = [MQARTier(kv_pairs=8, seq_len=128), MQARTier(kv_pairs=16, seq_len=128),
                  MQARTier(kv_pairs=8, seq_len=256), MQARTier(kv_pairs=16, seq_len=256)]


def _ladder_multiseed(seeds, tiers, steps, **kw) -> dict:
    per_rung: dict[str, list[float]] = {r.name: [] for r in LADDER}
    for s in seeds:
        res = run_ladder(tiers=tiers, steps=steps, seed=s, **kw)
        for pt in res["curve"]:
            per_rung[pt["rung"]].append(pt["auc"])
    stats = {r: _mean_ci([v for v in vals if v == v]) for r, vals in per_rung.items()}
    # the candidate finding: L4 (pure delta) vs L3 (best pure) — does delta regress when pure?
    l3, l4 = per_rung["L3-+local-conv"], per_rung["L4-+delta-rule"]
    return {"per_seed": per_rung, "stats": stats,
            "L4_vs_L3": _paired_test(l4, l3)}


def _floor_multiseed(seeds, tiers, steps, **kw) -> dict:
    vals = []
    for s in seeds:
        accs = [_run_mlp_baseline(t, steps, s, d_model=kw.get("d_model", 64)) for t in tiers]
        vals.append(float(np.mean(accs)))
    return {"per_seed": vals, "stats": _mean_ci(vals)}


def run_confirmation(seeds=(1337, 2024, 7, 42, 99), steps: int = 1500,
                     tiers=None, **kw) -> dict:
    tiers = tiers or DISCRIMINATING
    print("=== 1/3 C2 ablation × seeds ===")
    abl = run_multiseed(list(seeds), steps=steps, tiers=tiers, **kw)
    print("=== 2/3 recurrent ladder × seeds ===")
    lad = _ladder_multiseed(seeds, tiers, steps, **kw)
    print("=== 3/3 no-attention floor × seeds ===")
    flo = _floor_multiseed(seeds, tiers, steps, **kw)

    # hybrid vs best pure recurrent (the hybrid thesis, statistically)
    best_pure = max(lad["per_seed"], key=lambda r: np.nanmean(lad["per_seed"][r]))
    hyb_vs_pure = {
        "best_pure_rung": best_pure,
        "hopfield_vs_best_pure": _paired_test(abl["per_seed_auc"]["hopfield"], lad["per_seed"][best_pure]),
        "delta_hybrid_vs_best_pure": _paired_test(abl["per_seed_auc"]["delta"], lad["per_seed"][best_pure]),
    }
    return {
        "benchmark": "confirmation_run",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "regime": {"seeds": list(seeds), "steps": steps,
                   "tiers": [(t.kv_pairs, t.seq_len) for t in tiers], "note": "discriminating tiers"},
        "c2_ablation": {"stats": abl["stats"], "paired_tests": abl["paired_tests"], "claims": abl["claims"],
                        "per_seed_auc": abl["per_seed_auc"]},
        "recurrent_ladder": lad,
        "floor_mlp": flo,
        "hybrid_vs_pure": hyb_vs_pure,
        "gate_readout": {
            "hopfield_beats_control_significant": abl["paired_tests"]["hopfield_vs_control"].get("significant_95"),
            "delta_beats_control_significant": abl["paired_tests"]["delta_vs_control"].get("significant_95"),
            "delta_regresses_when_pure_significant": lad["L4_vs_L3"].get("significant_95") and lad["L4_vs_L3"]["mean_diff"] < 0,
            "hybrid_beats_best_pure_significant": hyb_vs_pure["hopfield_vs_best_pure"].get("significant_95"),
        },
        "note": "Feeds WHITEPAPER.md §4 only where the gate passes; otherwise §5 open questions.",
    }


def write_artefact(result: dict, out_dir="metrics/mqar") -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    stamp = result["timestamp_utc"][:19].replace(":", "").replace("-", "")
    p = out / f"confirmation-{stamp}.json"
    p.write_text(json.dumps(result, indent=2) + "\n")
    (out / "LATEST-confirmation.json").write_text(json.dumps(result, indent=2) + "\n")
    return p


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1337, 2024, 7, 42, 99])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        r = run_confirmation(seeds=(1, 2, 3), steps=150,
                             tiers=[MQARTier(kv_pairs=4, seq_len=48)], d_model=48)
    else:
        r = run_confirmation(seeds=tuple(args.seeds), steps=args.steps)
    p = write_artefact(r)
    print("\n=== GATE READOUT ===")
    print(json.dumps(r["gate_readout"], indent=2))
    print("=== C2 claims ===")
    for c in r["c2_ablation"]["claims"]:
        print("  ", c)
    print("=== hybrid vs best pure recurrent ===")
    print(json.dumps(r["hybrid_vs_pure"], indent=2))
    print("artefact:", p)


if __name__ == "__main__":
    _cli()
