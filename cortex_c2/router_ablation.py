"""cortex_c2.router_ablation — Slice D: does the learned router reach the ceiling?

The closing question of N2 (RES-18, ADR-006 D8). Slice A gave the ORACLE ceiling
(+21% vs control, +6% over the best single path) — but the oracle *reads* the
truth. Slice B/C gave a router that *learns* to route from context alone and
consolidates. Slice D measures whether that learned router, in practice,
approaches the oracle ceiling — the test that validates (or not) that the whole
multi-path system actually works, not just in oracle.

Method (measure-first): build a synthetic routing task where each span-type has a
context feature and a known best path (mirroring the ablation's per-tier domains:
Hopfield-favoured, delta-favoured, control-favoured). Train RouterV2 on observed
per-path scores; drive it through the ConsolidatingController; and compare four
readings on held-out spans:
  - control-only  (always the floor)
  - best single memory (whichever single path scores best on average)
  - the ORACLE (RouterV1: reads truth — the ceiling)
  - the LEARNED router (RouterV2 + controller: predicts, never reads truth)
The learned router's gap to the oracle is the Slice-D signal. Reaching most of
the oracle's lift over the best single path means RES-18 pays off in practice.

Pure enough to run on CPU in minutes; a fuller version trains on real MQAR spans.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cortex_c2 import PATH_CONTROL, PATH_HOPFIELD, PATH_DELTA, PATH_NAMES, RouterV1
from cortex_c2.consolidation import ConsolidatingController, ConsolidationConfig

REPO = Path(__file__).resolve().parents[1]
N_PATHS = 3


# --- the synthetic routing task (each span-type has a domain) ----------------
def _span_scores(ctx0: float) -> list[float]:
    """Known per-path scores as a function of the context feature — mirrors the
    ablation's finding that each path has a domain."""
    if ctx0 < -0.33:                       # Hopfield's domain (dense recall)
        return [0.10, 0.30, 0.16]
    elif ctx0 > 0.33:                      # delta's domain (intermediate)
        return [0.10, 0.16, 0.30]
    else:                                  # control's domain (memory hurts here)
        return [0.28, 0.12, 0.12]


def _make_spans(n: int, seed: int):
    rng = np.random.default_rng(seed)
    ctx = rng.standard_normal((n, 8)).astype(np.float32)
    scores = np.array([_span_scores(float(ctx[i, 0])) for i in range(n)], dtype=np.float32)
    return ctx, scores


def _auc_of_choice(scores: np.ndarray, chosen_path: np.ndarray) -> float:
    """Mean achieved score when each span uses its chosen path."""
    return float(np.mean([scores[i, chosen_path[i]] for i in range(len(scores))]))


def run_router_ablation(train_n: int = 4000, test_n: int = 1000,
                        steps: int = 600, seed: int = 1337) -> dict:
    import torch
    from cortex_c2.router_v2 import RouterV2

    # data
    ctx_tr, sc_tr = _make_spans(train_n, seed)
    ctx_te, sc_te = _make_spans(test_n, seed + 1)

    # train the learned router (predict per-path scores from context)
    torch.manual_seed(seed)
    router = RouterV2(ctx_dim=8, floor_margin=0.0)
    opt = torch.optim.Adam(router.parameters(), lr=1e-2)
    Xtr = torch.from_numpy(ctx_tr); Ytr = torch.from_numpy(sc_tr)
    bs = 128
    for step in range(steps):
        idx = torch.randint(0, train_n, (bs,))
        loss = router.score_loss(Xtr[idx], Ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    # --- four readings on held-out spans ---
    n = test_n
    # 1) control-only
    ctrl_choice = np.full(n, PATH_CONTROL)
    auc_control = _auc_of_choice(sc_te, ctrl_choice)
    # 2) best single memory (pick the path with best average score across spans)
    avg = sc_te.mean(axis=0)
    best_single_path = int(np.argmax(avg))
    single_choice = np.full(n, best_single_path)
    auc_best_single = _auc_of_choice(sc_te, single_choice)
    # 3) oracle (RouterV1 reads truth)
    oracle = RouterV1()
    orc_choice = np.array([oracle.route(path_scores=sc_te[i].tolist()).path for i in range(n)])
    auc_oracle = _auc_of_choice(sc_te, orc_choice)
    # 4) learned router (RouterV2 + controller: predicts, never reads truth)
    ctl = ConsolidatingController(router, ConsolidationConfig(harden_confidence=0.6,
                                                             harden_stability=3))
    learned_choice = np.zeros(n, dtype=int)
    Xte = torch.from_numpy(ctx_te)
    for i in range(n):
        # key = the span-type bucket (so consolidation has something to harden)
        c0 = float(ctx_te[i, 0])
        key = "hop" if c0 < -0.33 else "delta" if c0 > 0.33 else "ctrl"
        d = ctl.route(key=key, span_ctx=Xte[i])
        learned_choice[i] = d.path
    auc_learned = _auc_of_choice(sc_te, learned_choice)

    # gaps
    lift_oracle = auc_oracle - auc_best_single
    lift_learned = auc_learned - auc_best_single
    frac_of_ceiling = (lift_learned / lift_oracle) if lift_oracle > 1e-9 else float("nan")

    return {
        "benchmark": "router_ablation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "regime": {"train_n": train_n, "test_n": test_n, "steps": steps, "seed": seed},
        "auc": {
            "control_only": round(auc_control, 4),
            "best_single_memory": round(auc_best_single, 4),
            "oracle_ceiling": round(auc_oracle, 4),
            "learned_router": round(auc_learned, 4),
        },
        "best_single_path": PATH_NAMES[best_single_path],
        "lift_over_best_single": {
            "oracle": round(lift_oracle, 4),
            "learned": round(lift_learned, 4),
            "learned_fraction_of_oracle_ceiling": round(frac_of_ceiling, 3),
        },
        "controller": ctl.snapshot(),
        "verdict": (
            f"learned router captures {frac_of_ceiling*100:.0f}% of the oracle's lift "
            f"over the best single path"
            if lift_oracle > 1e-9 else
            "no oracle lift on this task (paths do not separate)"
        ),
        "note": "RES-18 Slice D — the learned router predicts (never reads truth); "
                "reaching most of the oracle ceiling means the multi-path design pays off. "
                "Synthetic routing task; a fuller run trains on real MQAR spans.",
    }


def write_artefact(result: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    stamp = result["timestamp_utc"][:19].replace(":", "").replace("-", "")
    path = out / f"router-ablation-{stamp}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    (out / "LATEST-router.json").write_text(json.dumps(result, indent=2) + "\n")
    return path


def _cli():
    ap = argparse.ArgumentParser(description="RES-18 router ablation (Slice D)")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out", default="metrics/mqar")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    r = run_router_ablation(steps=200 if args.quick else args.steps,
                            train_n=1000 if args.quick else 4000)
    path = write_artefact(r, args.out)
    print("\n=== RES-18 router ablation (Slice D) ===")
    print(json.dumps(r["auc"], indent=2))
    print(r["verdict"])
    print(f"controller: {json.dumps(r['controller'])}")
    print(f"artefact: {path}")


if __name__ == "__main__":
    _cli()


# ============================================================================ #
# (b) D19 Level 1b — the router on REAL MQAR spans, not the synthetic task     #
# ============================================================================ #
def _real_span_ctx(kv: int, seq: int) -> np.ndarray:
    """Context feature for a real MQAR span-type: normalised (kv, seq) + simple
    interactions. The router must learn which path wins from THIS, never from
    the truth. (8 dims to stay compatible with RouterV2(ctx_dim=8).)"""
    k = math.log2(kv) / 5.0          # kv ∈ {4..32} → ~0.4..1.0
    s = math.log2(seq) / 9.0         # seq ∈ {128..512} → ~0.78..1.0
    return np.array([k, s, k * s, k * k, s * s, k - s, 1.0 - k, 1.0 - s], dtype=np.float32)


def run_router_ablation_on_real_mqar(curves_by_path: dict, steps: int = 800,
                                     seed: int = 1337, holdout_frac: float = 0.34) -> dict:
    """Train RouterV2 on real per-tier MQAR accuracies (from an ablation artefact)
    and test on held-out tiers. curves_by_path = {"none": [...], "hopfield": [...],
    "delta": [...]} where each item is {"kv_pairs","seq_len","accuracy"} (same tier
    order). Held-out tiers are chosen by seed so the router never sees their scores.
    """
    import torch
    from cortex_c2.router_v2 import RouterV2

    tiers = [(t["kv_pairs"], t["seq_len"]) for t in curves_by_path["none"]]
    n = len(tiers)
    scores = np.array([[curves_by_path["none"][i]["accuracy"],
                        curves_by_path["hopfield"][i]["accuracy"],
                        curves_by_path["delta"][i]["accuracy"]] for i in range(n)],
                      dtype=np.float32)
    ctx = np.array([_real_span_ctx(kv, sq) for kv, sq in tiers], dtype=np.float32)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(2, int(round(n * holdout_frac)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    torch.manual_seed(seed)
    router = RouterV2(ctx_dim=8, floor_margin=0.0)
    opt = torch.optim.Adam(router.parameters(), lr=1e-2)
    Xtr = torch.from_numpy(ctx[train_idx]); Ytr = torch.from_numpy(scores[train_idx])
    for _ in range(steps):
        loss = router.score_loss(Xtr, Ytr)
        opt.zero_grad(); loss.backward(); opt.step()

    # readings on held-out tiers
    te_scores = scores[test_idx]
    auc_control = float(te_scores[:, PATH_CONTROL].mean())
    avg = te_scores.mean(axis=0); best_single_path = int(np.argmax(avg))
    auc_best_single = float(te_scores[:, best_single_path].mean())
    oracle = RouterV1()
    orc = [oracle.route(path_scores=te_scores[i].tolist()).path for i in range(len(test_idx))]
    auc_oracle = float(np.mean([te_scores[i, orc[i]] for i in range(len(test_idx))]))
    Xte = torch.from_numpy(ctx[test_idx])
    learned = [router.route(span_ctx=Xte[i]).path for i in range(len(test_idx))]
    auc_learned = float(np.mean([te_scores[i, learned[i]] for i in range(len(test_idx))]))

    lift_o = auc_oracle - auc_best_single
    lift_l = auc_learned - auc_best_single
    frac = (lift_l / lift_o) if lift_o > 1e-9 else float("nan")
    return {
        "benchmark": "router_ablation_real_mqar",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "regime": {"n_tiers": n, "held_out": [tiers[i] for i in test_idx],
                   "train": [tiers[i] for i in train_idx], "steps": steps, "seed": seed},
        "auc_heldout": {
            "control_only": round(auc_control, 4),
            "best_single_memory": round(auc_best_single, 4),
            "oracle_ceiling": round(auc_oracle, 4),
            "learned_router": round(auc_learned, 4),
        },
        "best_single_path": PATH_NAMES[best_single_path],
        "learned_routes_heldout": [PATH_NAMES[p] for p in learned],
        "oracle_routes_heldout": [PATH_NAMES[p] for p in orc],
        "lift_over_best_single": {"oracle": round(lift_o, 4), "learned": round(lift_l, 4),
                                  "learned_fraction_of_oracle_ceiling": round(frac, 3) if frac == frac else None},
        "verdict": (f"on REAL MQAR held-out tiers the learned router captures "
                    f"{frac*100:.0f}% of the oracle lift" if frac == frac else
                    "no oracle lift on held-out tiers (paths do not separate there)"),
        "note": "D19 Level 1b — router trained/tested on real per-tier MQAR accuracies "
                "(held-out tiers never seen). Small-N caveat: 12 tiers only; repeat across seeds/holdouts.",
    }
