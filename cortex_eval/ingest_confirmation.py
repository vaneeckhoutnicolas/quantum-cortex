"""cortex_eval.ingest_confirmation — grave the confirmation result in one command.

Reads the REAL `metrics/mqar/LATEST-confirmation.json` (the Kaggle output — the
record of truth), applies the whitepaper's admission gate to each comparison,
and updates three documents mechanically, never inventing a number:

  1. docs/RESULTS.md      — rows 10 (delta vs control) and a new row for Hopfield
                            vs control get their real numbers and a status from the
                            fixed vocabulary: gated (passes) / held (not established)
  2. docs/WHITEPAPER.md   — §4.4's table is rewritten from the artefact; the
                            prose states which rows pass, in the same register
  3. metrics/mqar/domain-map.* — regenerated (domain_map.py reads the artefact)

Gate applied here (rules 1 and 5 of the six; the others are properties of the
run, already satisfied or stated): a comparison is **gated** iff the paired
t-test over seeds is significant at 95% AND n_seeds >= 3; otherwise **held**.
The reserve line is written for every row, passing or not.

Usage:   python -m cortex_eval.ingest_confirmation            # after committing the artefact
         python -m cortex_eval.ingest_confirmation --dry-run  # print what would change
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "metrics" / "mqar" / "LATEST-confirmation.json"
RESULTS = REPO / "docs" / "RESULTS.md"
WP = REPO / "docs" / "WHITEPAPER.md"


def _status(test: dict, n_seeds: int) -> str:
    return "gated" if (test.get("significant_95") and n_seeds >= 3) else "held"


def read_artefact(path: Path = ART) -> dict:
    r = json.loads(path.read_text())
    seeds = r["regime"]["seeds"]; n = len(seeds)
    pt = r["c2_ablation"]["paired_tests"]; st = r["c2_ablation"]["stats"]
    rows = {}
    for key, a, b in (("hopfield_vs_control", "hopfield", "none"),
                      ("delta_vs_control", "delta", "none"),
                      ("delta_vs_hopfield", "delta", "hopfield")):
        # the artefact may name the third comparison the other way round
        # (multiseed.py writes "hopfield_vs_delta"); accept both, flip the sign.
        if key in pt:
            t = dict(pt[key])
        else:
            rev = "_vs_".join(reversed(key.split("_vs_")))
            if rev not in pt:
                raise KeyError(f"artefact has neither {key!r} nor {rev!r}")
            t = dict(pt[rev])
            t["mean_diff"] = -t["mean_diff"]
            if t.get("t") is not None:
                t["t"] = -t["t"]
        rows[key] = {
            "a": a, "b": b, "mean_a": st[a]["mean"], "mean_b": st[b]["mean"],
            "diff": t["mean_diff"], "t": t.get("t"), "df": t.get("df"),
            "t_crit": t.get("t_crit_95"), "sig": bool(t.get("significant_95")),
            "status": _status(t, n),
            "pct": (t["mean_diff"] / st[b]["mean"] * 100) if st[b]["mean"] else None,
        }
    hyb = r.get("hybrid_vs_pure", {})
    lad = r.get("recurrent_ladder", {})
    return {
        "seeds": seeds, "n_seeds": n, "steps": r["regime"]["steps"],
        "tiers": r["regime"]["tiers"], "rows": rows,
        "per_seed": r["c2_ablation"]["per_seed_auc"],
        "hybrid_vs_pure": hyb, "ladder_L4_vs_L3": lad.get("L4_vs_L3"),
        "gate_readout": r.get("gate_readout", {}),
    }


# ---------------------------------------------------------------------------- #
# RESULTS.md                                                                     #
# ---------------------------------------------------------------------------- #
def _results_row(n: int, stmt: str, status: str, number: str, artefact: str, test: str, reserve: str) -> str:
    return f"| {n} | {stmt} | **{status}** | {number} | `{artefact}` | `{test}` | {reserve} |"


def update_results(a: dict, dry: bool = False) -> str:
    t = RESULTS.read_text(encoding="utf-8")
    n, steps, tiers = a["n_seeds"], a["steps"], a["tiers"]
    tiers_s = ", ".join(f"kv{kv}/seq{sq}" for kv, sq in tiers)
    art = "metrics/mqar/LATEST-confirmation.json"
    d = a["rows"]["delta_vs_control"]; h = a["rows"]["hopfield_vs_control"]
    def num(x):
        return (f"{x['diff']:+.4f} ({x['pct']:+.0f}%), {n}/{n} seeds' mean {x['mean_a']:.4f} vs {x['mean_b']:.4f}, "
                f"t = {x['t']} vs t_crit {x['t_crit']} (df {x['df']})")
    def reserve(x):
        return (f"{n} seeds, {steps} steps, tiers {tiers_s}; " +
                ("passes rules 1–6 at this regime" if x["status"] == "gated" else
                 "visible signal, not established at this seed count"))
    new_delta = _results_row(10, f"Delta > control on the discriminating tiers, {n} seeds", d["status"], num(d),
                             art, "tests/test_multiseed.py", reserve(d))
    new_hop = _results_row("10b", f"Hopfield > control on the discriminating tiers, {n} seeds", h["status"], num(h),
                           art, "tests/test_multiseed.py", reserve(h))
    # replace row 10 (whatever its current text) and insert 10b after it
    lines = t.splitlines()
    out = []; done = False
    for l in lines:
        if l.startswith("| 10 |") and not done:
            out.append(new_delta); out.append(new_hop); done = True
        elif l.startswith("| 10b |"):
            continue
        else:
            out.append(l)
    t2 = "\n".join(out) + "\n"
    # gate status line
    gated = [k for k, v in a["rows"].items() if v["status"] == "gated" and k != "delta_vs_hopfield"]
    stamp = datetime.now(timezone.utc).date().isoformat()
    line = (f"**Gate status (whitepaper §2.6):** as of {stamp}, **{len(gated)}** comparison(s) have passed the six-rule gate"
            + (f" — {', '.join(gated)}" if gated else "") +
            f" (confirmation run: {n} seeds, {steps} steps).")
    t2 = re.sub(r"\*\*Gate status \(whitepaper §2\.6\):\*\*[^\n]*", line, t2, count=1)
    if not dry:
        RESULTS.write_text(t2, encoding="utf-8")
    return t2


# ---------------------------------------------------------------------------- #
# WHITEPAPER §4.4                                                                 #
# ---------------------------------------------------------------------------- #
def update_whitepaper(a: dict, dry: bool = False) -> str:
    t = WP.read_text(encoding="utf-8")
    start = t.index("**4.4 What passes the gate")
    end = t.index("**4.5 ")
    n, steps = a["n_seeds"], a["steps"]
    tiers_s = " and ".join(f"kv {kv} at seq {sq}" for kv, sq in a["tiers"])
    R = a["rows"]
    def row(label, x):
        verdict = "**passes**" if x["status"] == "gated" else "does not pass"
        return f"| {label} | **{x['diff']:+.4f}** ({x['pct']:+.0f}%) | {x['t']} | {x['t_crit']} | {verdict} |"
    table = "\n".join([
        "| comparison | mean gap | paired t (df = %d) | t_crit 95%% | verdict |" % (n - 1),
        "|:--|--:|--:|--:|:--|",
        row("delta vs control", R["delta_vs_control"]),
        row("Hopfield vs control", R["hopfield_vs_control"]),
        row("delta vs Hopfield", R["delta_vs_hopfield"]),
    ])
    gated = [k for k, v in R.items() if v["status"] == "gated" and k != "delta_vs_hopfield"]
    per = a["per_seed"]
    ps = "; ".join(f"{k}: " + " / ".join(f"{v:.3f}" for v in vals) for k, vals in per.items())
    if gated:
        names = " and ".join(k.replace("_vs_control", "") for k in gated)
        prose = (f"At {n} seeds the critical value falls to {R['delta_vs_control']['t_crit']}, and **{names} now pass{'es' if len(gated)==1 else ''} the gate**: "
                 f"the comparison{'s' if len(gated)>1 else ''} enter{'s' if len(gated)==1 else ''} §4.3 as a claim on the discriminating regime. "
                 f"Per-seed AUC ({ps}). The remaining rows are held as visible signal.")
    else:
        prose = (f"No comparison passes at {n} seeds (per-seed AUC: {ps}). The signal is held, not claimed; "
                 f"rule 6 asks which context factor the inter-seed variance carries before a larger run is scheduled.")
    new = (f"**4.4 What passes the gate — the confirmation run ({n} seeds, {steps} steps, tiers {tiers_s}).**\n\n"
           f"{table}\n\n{prose}\n\n")
    t2 = t[:start] + new + t[end:]
    if not dry:
        WP.write_text(t2, encoding="utf-8")
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--artefact", default=str(ART))
    ap.add_argument("--expect-seeds", type=int, default=None,
                    help="refuse to write unless the artefact has exactly this many seeds "
                         "(guards against ingesting the wrong run — lesson of 2026-09-09)")
    args = ap.parse_args()
    a = read_artefact(Path(args.artefact))
    if args.expect_seeds is not None and a["n_seeds"] != args.expect_seeds and not args.dry_run:
        raise SystemExit(f"refusing to write: artefact has {a['n_seeds']} seeds, expected {args.expect_seeds} "
                         f"(is this the right run? see metrics/mqar/PROVENANCE-confirmation.json)")
    print(f"=== confirmation artefact: {a['n_seeds']} seeds, {a['steps']} steps, tiers {a['tiers']} ===")
    for k, x in a["rows"].items():
        print(f"  {k:22s} diff {x['diff']:+.4f}  t={x['t']}  crit={x['t_crit']}  -> {x['status'].upper()}")
    if a["ladder_L4_vs_L3"]:
        print(f"  ladder L4 vs L3: {a['ladder_L4_vs_L3']}")
    print()
    r = update_results(a, dry=args.dry_run)
    w = update_whitepaper(a, dry=args.dry_run)
    if args.dry_run:
        print("--- would write to WHITEPAPER §4.4 ---\n" + w)
        print("--- RESULTS.md gate line ---")
        print([l for l in r.splitlines() if l.startswith("**Gate status")][0])
    else:
        print("RESULTS.md and WHITEPAPER.md §4.4 updated. Now regenerate the domain map:")
        print("  python -m cortex_eval.domain_map")


if __name__ == "__main__":
    main()
