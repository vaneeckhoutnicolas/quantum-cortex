"""cortex_eval.ablation_mqar — Slice C: the C2 ablation on MQAR.

Runs the MQAR difficulty curriculum on the three variants (control / hopfield /
delta) at the *same* regime, produces the three accuracy curves, compares them,
and emits the verdict — the second point on the Pareto frontier, the first that
*decides* between architectures.

Per the founder (2026-09-06): full curriculum first for a solid base, then a
fast pass driven by insights. Results are a SEPARATE artefact (metrics/mqar/),
not a run-v1 ledger record — MQAR is an architecture bench, distinct from the
language-training runs whose `comparison` slot fills at the real FineWeb-Edu
ablations (W1 proper). NOW-6: benchmarks as open, versioned deliverables.

The reading rule (D16 + the founder's curve principle): an architecture is
characterised by *where its curve drops off*; the gap between drop-off points is
the clean signal. The circuit breaker (ADR-006 D7) applies per tier per variant.

Usage:
    python -m cortex_eval.ablation_mqar --steps 1500 --out metrics/mqar
    python -m cortex_eval.ablation_mqar --quick        # tiny, for a smoke check
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from cortex_eval.mqar import MQARTier, standard_curriculum
from cortex_eval.run_mqar import run_curriculum

VARIANTS = ("none", "hopfield", "delta")
REPO = Path(__file__).resolve().parents[1]


def _dropoff(per_tier: list[dict], threshold: float, axis: str = "kv_pairs") -> dict:
    """Where each curve first drops below threshold, per the smallest other-axis."""
    other = "seq_len" if axis == "kv_pairs" else "kv_pairs"
    base = min((t[other] for t in per_tier), default=None)
    if base is None:
        return {}
    row = sorted((t for t in per_tier if t[other] == base), key=lambda t: t[axis])
    for t in row:
        if t["accuracy"] < threshold:
            return {"axis": axis, other: base, "dropoff": t[axis], "below": threshold}
    return {"axis": axis, other: base, "dropoff": None, "below": threshold}  # never dropped


def run_ablation(steps: int = 1500, tiers: list[MQARTier] | None = None,
                 threshold: float = 0.5, seed: int = 1337, **kw) -> dict:
    tiers = tiers or standard_curriculum()
    curves: dict[str, list[dict]] = {}
    t0 = time.time()
    for v in VARIANTS:
        res = run_curriculum(c2_variant=v, steps=steps, tiers=tiers, seed=seed, **kw)
        curves[v] = res.per_tier

    # comparison: drop-off per variant, and the gap vs control
    dropoffs = {v: _dropoff(curves[v], threshold) for v in VARIANTS}
    ctrl_do = dropoffs["none"].get("dropoff")
    deltas = {}
    for v in ("hopfield", "delta"):
        vdo = dropoffs[v].get("dropoff")
        if ctrl_do is None and vdo is None:
            deltas[v] = "both never dropped (raise difficulty)"
        elif vdo is None:
            deltas[v] = "later than control (never dropped)"
        elif ctrl_do is None:
            deltas[v] = "earlier than control (control never dropped)"
        else:
            g = vdo - ctrl_do
            deltas[v] = f"{'+' if g >= 0 else ''}{g} kv vs control ({vdo} vs {ctrl_do})"

    # honest verdict: which variant extends the frontier on this axis
    winners = [v for v in ("hopfield", "delta")
               if isinstance(dropoffs[v].get("dropoff"), int)
               and (ctrl_do is None or dropoffs[v]["dropoff"] > ctrl_do)]
    if dropoffs["hopfield"].get("dropoff") is None and dropoffs["delta"].get("dropoff") is None \
       and ctrl_do is None:
        verdict = "inconclusive — no variant dropped; raise difficulty (more kv / longer seq)"
    elif winners:
        verdict = f"advances: {', '.join(winners)} recall further than control on MQAR"
    else:
        verdict = "held — no associative variant beat the control's MQAR drop-off"

    return {
        "benchmark": "mqar",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "regime": {"steps": steps, "tiers": len(tiers), "threshold": threshold, "seed": seed},
        "curves": curves,
        "dropoffs": dropoffs,
        "control_delta": deltas,
        "verdict": verdict,
        "wall_s": round(time.time() - t0, 1),
        "note": "architecture bench (separate artefact, not a run-v1 record); "
                "circuit breaker per tier; the gap between drop-offs is the signal.",
    }


def write_artefact(result: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = result["timestamp_utc"][:19].replace(":", "").replace("-", "")
    path = out / f"mqar-ablation-{stamp}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    # also refresh a LATEST pointer
    (out / "LATEST.json").write_text(json.dumps(result, indent=2) + "\n")
    return path


def _cli():
    ap = argparse.ArgumentParser(description="MQAR C2 ablation (Slice C, ADR-006)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="metrics/mqar")
    ap.add_argument("--quick", action="store_true", help="tiny curriculum + few steps")
    args = ap.parse_args()
    tiers = None
    steps = args.steps
    if args.quick:
        tiers = [MQARTier(kv_pairs=4, seq_len=64), MQARTier(kv_pairs=16, seq_len=64)]
        steps = 300
    result = run_ablation(steps=steps, tiers=tiers, threshold=args.threshold, seed=args.seed)
    path = write_artefact(result, args.out)
    print("\n=== MQAR ablation verdict ===")
    print(result["verdict"])
    print(f"drop-offs: {json.dumps(result['dropoffs'])}")
    print(f"vs control: {json.dumps(result['control_delta'])}")
    print(f"artefact: {path}")


if __name__ == "__main__":
    _cli()
