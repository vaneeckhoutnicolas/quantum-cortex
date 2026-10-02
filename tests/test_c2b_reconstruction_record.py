"""RESULTS row 41: the reconstruction probe on the organ's layer of numbers, three sealed
journals, 2026-09-30. Every scalar the row states is recomputed here from the committed
files (the per entry rows of levels 2 and 3, the per phase counts of level 1, the two
processes), and the file's own aggregates are checked against that recomputation. The
thresholds are the declared ones (ADR-007, amendment 2026-09-30). A change to a file
breaks this test; a change to the row without the file cannot pass it."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cortex_c2b.lifecycle import LifecycleConfig                                 # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FILES = {"8ceba5db8d0b": 1337, "4cfd6f6e1406": 2024, "2e1dc71913b1": 7}
ROW41 = {   # the numbers the row states, per journal: (level 2 max relative defect, level 3 max defect, max error at replay)
    "8ceba5db8d0b": (9.61042961316627e-06, 4.2498111724853516e-05, 4.220008850097656e-05),
    "4cfd6f6e1406": (9.67036111916484e-06, 0.00019890069961547852, 0.00019890069961547852),
    "2e1dc71913b1": (8.086696715290153e-06, 4.589557647705078e-05, 4.392862319946289e-05),
}
JOURNAL_SHA = {"8ceba5db8d0b": "d0d56e5a72fe4cf3", "4cfd6f6e1406": "697be97c15d2edf3", "2e1dc71913b1": "0e0c1d34438c85c3"}


def _load(rid: str) -> dict:
    return json.loads((ROOT / "metrics/mqar" / f"reconstruction-probe-{rid}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("rid", list(FILES))
def test_the_declared_run_and_thresholds_are_in_the_file(rid):
    d = _load(rid)
    cfg = LifecycleConfig(sentinel_seed=2, withdraw_at_eviction=False)       # the configuration that produced the files (the flag off; row 42 turned it on)
    assert d["config"]["config_hash"] == cfg.config_hash() == "3212d4c151c5c862"
    assert d["config"]["phases"] == 40 and len(d["phases"]) == 40 and d["config"]["reads_between_phases"] == 0
    assert d["config"]["sentinel_seed"] == 2 and d["config"]["now_plant"] == 200.0
    assert d["thresholds"] == {"level_1": "exact", "level_2_relative": 1e-3, "level_3_bound": cfg.replay_error_max}
    assert d["journal"]["sha256"].startswith(JOURNAL_SHA[rid]) and d["journal"]["entries_before"] == 200
    assert d["journal"]["copied_to_temp"] and d["journal"]["separation"] == {"dg": True, "seed": 0}
    assert d["sentinels"] == {"planted": 20, "negctrl": 10, "collision_check": "passed (no sentinel entity among the journal's)"}
    assert d["phases_refused"] == 0 and d["replays"] == {"count": 220, "withdrawn": 0}


@pytest.mark.parametrize("rid", list(FILES))
def test_level_1_recomputed_from_the_per_phase_counts(rid):
    a = _load(rid)["process_a"]["level_1_counts"]
    assert a["phases_checked"] == 40 and a["exact_at_every_phase"] and a["worst_defect"] == 0
    for p in a["per_phase"]:
        derived_demoted = LifecycleConfig().demote_k * p["summaries_alive"]
        derived_evicted = p["entries"] - p["live"] - p["consolidated"] - derived_demoted
        assert derived_demoted == p["demoted"] == p["derived_demoted"]
        assert derived_evicted == p["evicted"] == p["derived_evicted"]
    last = a["per_phase"][-1]
    assert (last["live"], last["consolidated"], last["demoted"], last["evicted"], last["entries"]) == (0, 20, 0, 200, 220)


@pytest.mark.parametrize("rid", list(FILES))
def test_level_2_recomputed_from_the_rows(rid):
    l2 = _load(rid)["process_a"]["level_2_salience_line"]
    decay = LifecycleConfig().decay
    rows = l2["rows"]
    assert len(rows) == 220 == l2["entries"] and l2["kept"] == 20 and l2["evicted"] == 200 and l2["contracted"] == 20
    rel = []
    for r in rows:
        derived = r["salience_final"] / (decay ** r["decays"]) if r["decays"] else r["salience_final"]
        assert derived == r["derived"]
        rel.append(abs(derived - r["salience_at_consolidation"]) / r["salience_at_consolidation"])
        if r["contracted"]:
            assert r["decays"] == 0 and r["salience_at_consolidation"] == 1.0
        else:
            assert 8 <= r["decays"] <= 16
    assert max(rel) == l2["max_relative_defect"] == ROW41[rid][0] <= 1e-3
    assert abs(sum(rel) / len(rel) - l2["mean_relative_defect"]) < 1e-15
    assert l2["under_threshold"] == 220 and l2["all_under_threshold"]


@pytest.mark.parametrize("rid", list(FILES))
def test_level_3_recomputed_from_the_rows(rid):
    d = _load(rid)
    l3 = d["process_a"]["level_3_memory"]
    rows = l3["rows"]
    assert len(rows) == 220 == l3["replayed"] == l3["patterns"]
    kept = [r["defect"] for r in rows if r["state"] in ("consolidated", "demoted")]
    evicted = [r["defect"] for r in rows if r["state"] == "evicted"]
    assert len(kept) == 20 and len(evicted) == 200
    assert max(kept) == l3["kept"]["max_defect"] and l3["kept"]["under_bound"] == 20 and l3["kept"]["all_under_bound"]
    assert l3["evicted_address_reading"] == {"entries": 200, "still_reconstructed_under_bound": 200, "share": 1.0}
    assert max(kept + evicted) == l3["distribution"]["max"] == ROW41[rid][1] <= 0.05
    assert l3["distribution"]["p95"] == float(np.percentile(kept + evicted, 95)) < 1e-5
    assert l3["positive_control"] == {"replays": 220, "max_error_at_replay": ROW41[rid][2], "all_under_bound_at_replay": True}
    assert d["process_a"]["level_4_content"] == {"word_by_construction": "declared loss (3.5, rule 5; theorem L2)", "evicted": 200,
                                                  "released_payload_files_present": 0, "evicted_entries_returned_by_a_read": 0, "holds": True}


@pytest.mark.parametrize("rid", list(FILES))
def test_the_two_processes_agree_and_the_phases_read_as_the_row_says(rid):
    d = _load(rid)
    assert d["agreement"] == {"number_for_number": True, "differences": []}
    b = d["process_b"]
    assert b["phases_found"] == 40 and b["sentinels_found"] == 20 and b["durable"]
    for level in ("level_2_salience_line", "level_3_memory"):
        for key in ("max_relative_defect", "mean_relative_defect") if level == "level_2_salience_line" else ("distribution", "kept", "evicted_address_reading"):
            assert d["process_a"][level][key] == b[level][key]
    ph = d["phases"]
    assert min(p["sentinel_recall"] for p in ph) == 1.0
    assert next(p["phase"] for p in ph if p["live"] == 0) == 28
    evict = [p["phase"] for p in ph if (p["applied"] or {}).get("evicted", 0) > 0]
    assert (min(evict), max(evict)) == (17, 38)
    assert sum(p["planned"]["demote_groups"] for p in ph) == 0 and max(p["pressure"] for p in ph) == 0.0
    negctrl = [p["sentinel_negctrl"] for p in ph]
    if rid == "4cfd6f6e1406":
        assert negctrl == [0.1] * 32 + [0.0] * 8
    else:
        assert set(negctrl) == {0.0}
    assert d["verdict"].startswith("level 1: theorem by construction; level 4: declared loss by construction; level 2: reconstructible at measured defect; level 3: reconstructible at measured defect; level 3b: the address of an evicted episode survives")
