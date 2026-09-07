"""Domain map (the per-domain reading of C2): structure + the reading rule."""
import json, tempfile
from pathlib import Path
from cortex_eval.domain_map import build_domain_map, to_markdown


def _fake_ablation(path):
    tiers=[(4,128),(8,128),(16,128)]
    def c(v): return [{"kv_pairs":t[0],"seq_len":t[1],"accuracy":a} for t,a in zip(tiers,v)]
    json.dump({"curves":{"none":c([0.20,0.10,0.02]),"hopfield":c([0.30,0.11,0.02]),"delta":c([0.20,0.20,0.03])}},
              open(path,"w"))


def test_map_reports_per_tier_winner_and_separability():
    d=tempfile.mkdtemp(); a=Path(d)/"a.json"; _fake_ablation(a)
    dm=build_domain_map(ablation12_path=a, confirmation_path=Path(d)/"none.json")
    rows=dm["per_tier"]
    assert rows[0]["winner"]=="hopfield" and rows[0]["separable"]      # 0.30 vs 0.20
    assert rows[1]["winner"]=="delta" and rows[1]["separable"]         # 0.20 vs 0.11
    assert not rows[2]["separable"]                                    # 0.03 vs 0.02: noise-level


def test_reading_says_no_single_dominates_when_domains_split():
    d=tempfile.mkdtemp(); a=Path(d)/"a.json"; _fake_ablation(a)
    dm=build_domain_map(ablation12_path=a, confirmation_path=Path(d)/"none.json")
    assert "no single memory dominates" in dm["reading"]
    md=to_markdown(dm); assert "hopfield" in md and "delta" in md
