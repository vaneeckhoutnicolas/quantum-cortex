"""cortex_eval.domain_map — the per-domain reading of the C2 ablation.

The confirmation run showed that the "winner" flips between regimes: on the
12-tier average Hopfield leads; on the hard/short discriminating tiers delta
leads (and Hopfield sinks to the control). That is not a contradiction — it is
RES-20 in action: the ranking depends on Σ = which tiers are held. The honest
object is therefore not a ranking but a DOMAIN MAP: for each tier, which path
wins and by what margin, with the evidence basis (single-seed / multi-seed).

This map is (a) what §4 of the whitepaper displays, and (b) exactly what the
RES-18 router must learn — the map is the router's training target (ADR-007
Slice C). "No single memory dominates" is the result; the router is the answer.

Inputs: the 12-tier ablation artefact (single seed) and, when present, the
confirmation artefact (multi-seed on the discriminating tiers). Output: a JSON
map + a markdown table for the paper.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PATHS = ("none", "hopfield", "delta")
LABEL = {"none": "control", "hopfield": "hopfield", "delta": "delta"}


def _load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def build_domain_map(ablation12_path="metrics/mqar/mqar-ablation-final-2026-09-07.json",
                     confirmation_path="metrics/mqar/LATEST-confirmation.json") -> dict:
    a12 = _load(ablation12_path)
    conf = _load(confirmation_path)
    rows = []

    # --- basis 1: the 12-tier single-seed curves ---
    if a12:
        curves = a12["curves"]
        n = len(curves["none"])
        for i in range(n):
            kv, sq = curves["none"][i]["kv_pairs"], curves["none"][i]["seq_len"]
            sc = {p: curves[p][i]["accuracy"] for p in PATHS}
            best = max(sc, key=sc.get)
            ranked = sorted(sc.values(), reverse=True)
            margin = ranked[0] - ranked[1]
            rows.append({"kv_pairs": kv, "seq_len": sq, "scores": sc, "winner": LABEL[best],
                         "margin": round(margin, 4), "basis": "12-tier, 1 seed",
                         "separable": margin >= 0.01})

    # --- basis 2: confirmation (multi-seed) on discriminating tiers — overrides where present ---
    domain_multiseed = {}
    if conf and "c2_ablation" in conf:
        st = conf["c2_ablation"]["stats"]          # per-variant mean over the held tiers
        tests = conf["c2_ablation"]["paired_tests"]
        tiers = conf["regime"]["tiers"]
        means = {p: st[p]["mean"] for p in PATHS}
        best = max(means, key=means.get)
        sig = {
            "hopfield_vs_control": tests["hopfield_vs_control"].get("significant_95"),
            "delta_vs_control": tests["delta_vs_control"].get("significant_95"),
            "hopfield_vs_delta": tests["hopfield_vs_delta"].get("significant_95"),
        }
        domain_multiseed = {
            "tiers": tiers, "means": means, "winner": LABEL[best],
            "n_seeds": conf["regime"]["n_seeds"] if "n_seeds" in conf["regime"] else len(conf["regime"].get("seeds", [])),
            "significant": sig,
            "basis": f"confirmation, {len(conf['regime'].get('seeds', []))} seeds, {conf['regime'].get('steps')} steps",
        }

    # --- the reading ---
    winners = [r["winner"] for r in rows if r["separable"]]
    counts = {LABEL[p]: winners.count(LABEL[p]) for p in PATHS}
    dominated = max(counts, key=counts.get) if winners else None
    no_single = len([c for c in counts.values() if c > 0]) >= 2
    reading = ("no single memory dominates — each path has a domain; the per-tier map is the result "
               "and the multi-path router (RES-18) is the answer" if no_single else
               f"{dominated} dominates the separable tiers")

    return {
        "benchmark": "domain_map",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "law": "RES-20: a ranking that flips between regimes is not a contradiction — it is Σ (which tiers are held). Read the map, not a ranking.",
        "per_tier": rows,
        "separable_winner_counts": counts,
        "discriminating_multiseed": domain_multiseed,
        "reading": reading,
    }


def to_markdown(dm: dict) -> str:
    lines = ["| kv | seq | control | hopfield | delta | winner | margin | basis |",
             "|---:|---:|---:|---:|---:|:---|---:|:---|"]
    for r in dm["per_tier"]:
        s = r["scores"]
        mark = "" if r["separable"] else " (noise-level)"
        lines.append(f"| {r['kv_pairs']} | {r['seq_len']} | {s['none']:.3f} | {s['hopfield']:.3f} | {s['delta']:.3f} | "
                     f"**{r['winner']}**{mark} | {r['margin']:.3f} | {r['basis']} |")
    ms = dm.get("discriminating_multiseed")
    if ms:
        m = ms["means"]
        lines.append("")
        lines.append(f"**Discriminating tiers {ms['tiers']} — {ms['basis']}:** control {m['none']:.4f} · "
                     f"hopfield {m['hopfield']:.4f} · delta {m['delta']:.4f} → winner **{ms['winner']}**; "
                     f"significance@95%: {ms['significant']}")
    lines.append("")
    lines.append(f"**Reading:** {dm['reading']}")
    return "\n".join(lines)


def write_artefact(dm: dict, out_dir="metrics/mqar") -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    p = out / "domain-map.json"
    p.write_text(json.dumps(dm, indent=2) + "\n")
    (out / "domain-map.md").write_text(to_markdown(dm) + "\n")
    return p


if __name__ == "__main__":
    dm = build_domain_map()
    p = write_artefact(dm)
    print(to_markdown(dm))
    print(f"\nartefact: {p}")
