"""cortex_eval.multiseed — make N2 unattackable (D19 Level 1a).

A single-seed result is an anecdote. This module runs the C2 ablation across N
seeds and reports, per variant: mean AUC, standard deviation, and a 95%
confidence interval — then tests whether the gaps (hopfield vs control, delta vs
control) are STATISTICALLY SIGNIFICANT. That is the difference between
"Hopfield +14%" and "Hopfield +14% ± 3%, p < 0.05".

Statistics kept simple and honest (from scratch, no scipy dependency):
  - CI 95% via the t-distribution for small N (t critical values tabulated).
  - Significance via a paired comparison across seeds (same seed → same data
    order for all variants, so pairing is legitimate): a paired t-test on the
    per-seed AUC differences. We report the t statistic and whether |t| exceeds
    the two-sided 95% critical value — a claim only when it does.
Honest-data rule: every seed's curve is stored; nothing is dropped.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cortex_eval.mqar import MQARTier, standard_curriculum
from cortex_eval.run_mqar import run_curriculum

VARIANTS = ("none", "hopfield", "delta")

# two-sided 95% t critical values by degrees of freedom (df = n-1)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def _t95(df: int) -> float:
    if df in _T95:
        return _T95[df]
    keys = sorted(_T95)
    for k in keys:
        if k >= df:
            return _T95[k]
    return 1.96


def _auc(curve: list[dict]) -> float:
    accs = [t["accuracy"] for t in curve if t["accuracy"] == t["accuracy"]]
    return float(np.mean(accs)) if accs else float("nan")


def _mean_ci(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    half = _t95(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"mean": round(mean, 4), "sd": round(sd, 4),
            "ci95": [round(mean - half, 4), round(mean + half, 4)] if n > 1 else None,
            "n": n}


def _paired_test(a: list[float], b: list[float]) -> dict:
    """Paired t-test on per-seed differences a - b. Returns t, df, significant@95%."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    if n < 2:
        return {"mean_diff": round(float(d.mean()), 4), "t": None, "df": n - 1,
                "significant_95": None, "note": "need ≥2 seeds"}
    md = float(d.mean()); sd = float(d.std(ddof=1))
    t = md / (sd / math.sqrt(n)) if sd > 1e-12 else float("inf")
    crit = _t95(n - 1)
    return {"mean_diff": round(md, 4), "t": round(t, 3), "df": n - 1,
            "t_crit_95": crit, "significant_95": bool(abs(t) > crit)}


def run_multiseed(seeds: list[int], steps: int = 1500,
                  tiers: list[MQARTier] | None = None, **kw) -> dict:
    tiers = tiers or standard_curriculum()
    per_seed: dict[str, list[float]] = {v: [] for v in VARIANTS}
    curves: dict[str, list[list[dict]]] = {v: [] for v in VARIANTS}
    for s in seeds:
        for v in VARIANTS:
            res = run_curriculum(c2_variant=v, steps=steps, tiers=tiers, seed=s, **kw)
            per_seed[v].append(_auc(res.per_tier))
            curves[v].append(res.per_tier)          # every curve stored, nothing dropped
        print(f"[multiseed] seed={s}  " + "  ".join(f"{v}={per_seed[v][-1]:.4f}" for v in VARIANTS))

    stats = {v: _mean_ci(per_seed[v]) for v in VARIANTS}
    tests = {
        "hopfield_vs_control": _paired_test(per_seed["hopfield"], per_seed["none"]),
        "delta_vs_control": _paired_test(per_seed["delta"], per_seed["none"]),
        "hopfield_vs_delta": _paired_test(per_seed["hopfield"], per_seed["delta"]),
    }
    # honest verdict: a claim only where the paired test is significant
    claims = []
    for name, tst in tests.items():
        if tst.get("significant_95"):
            sign = "+" if tst["mean_diff"] > 0 else ""
            claims.append(f"{name}: {sign}{tst['mean_diff']} AUC (t={tst['t']}, df={tst['df']}) — significant at 95%")
        else:
            claims.append(f"{name}: {tst['mean_diff']} AUC — NOT significant at 95% (t={tst.get('t')})")
    return {
        "benchmark": "mqar_multiseed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "regime": {"seeds": seeds, "n_seeds": len(seeds), "steps": steps, "tiers": len(tiers)},
        "per_seed_auc": per_seed,
        "stats": stats,
        "paired_tests": tests,
        "claims": claims,
        "curves_per_seed": curves,
        "note": "D19 Level 1a — the single-seed N2 result made statistical. Paired tests are legitimate "
                "because the same seed fixes the data order across variants. A gap is claimed only when "
                "significant at 95%; otherwise reported as not significant (honest data).",
    }


def write_artefact(result: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    stamp = result["timestamp_utc"][:19].replace(":", "").replace("-", "")
    path = out / f"mqar-multiseed-{stamp}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    (out / "LATEST-multiseed.json").write_text(json.dumps(result, indent=2) + "\n")
    return path


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Multi-seed C2 ablation with CIs (D19 Level 1a)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1337, 2024, 7, 42, 99])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", default="metrics/mqar")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    tiers = [MQARTier(kv_pairs=4, seq_len=64), MQARTier(kv_pairs=16, seq_len=64)] if args.quick else None
    steps = 300 if args.quick else args.steps
    r = run_multiseed(args.seeds, steps=steps, tiers=tiers)
    p = write_artefact(r, args.out)
    print("\n=== multi-seed stats ===")
    for v, st in r["stats"].items():
        print(f"  {v:9s}: mean {st['mean']}  sd {st['sd']}  CI95 {st['ci95']}  (n={st['n']})")
    print("=== claims ===")
    for c in r["claims"]:
        print("  ", c)
    print(f"artefact: {p}")


if __name__ == "__main__":
    _cli()
