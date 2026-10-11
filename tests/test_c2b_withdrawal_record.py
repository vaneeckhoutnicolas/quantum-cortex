"""RESULTS row 42: the withdrawal at eviction, measured on the three sealed journals of rows 40
and 41, 2026-10-02 (ADR-007 amendment 2026-10-01, declared before any measurement). Every scalar
the row states is recomputed here from the committed files: the control's agreement with the row 41
file (the declared comparison, and the whole of both sessions and the forty phase records beyond
it), the counts of withdrawals and of patterns from the phase records and the per entry rows, the
defects of the kept and of the withdrawn addresses, the nearest kept pattern of every withdrawn
address, the agreement of the two processes, and the amendment's letter. A change to a file breaks
this test; a change to the row without the file cannot pass it."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cortex_c2b import STATE_EVICTED, STATE_CONSOLIDATED, STATE_DEMOTED       # noqa: E402
from cortex_c2b.lifecycle import LifecycleConfig                                 # noqa: E402
from cortex_c2b import withdrawal_probe as wp                                    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FILES = {"8ceba5db8d0b": 1337, "4cfd6f6e1406": 2024, "2e1dc71913b1": 7}
JOURNAL_SHA = {"8ceba5db8d0b": "d0d56e5a72fe4cf3", "4cfd6f6e1406": "697be97c15d2edf3", "2e1dc71913b1": "0e0c1d34438c85c3"}
HASH_OFF, HASH_ON = "3212d4c151c5c862", "4c11b44d47901881"
ROW42 = {   # per journal: (kept max defect, the control's kept max defect, min and max defect of the withdrawn, the nearest kept cosine of any withdrawn)
    "8ceba5db8d0b": (5.960464477539063e-08, 1.1920928955078125e-07, 0.4221614599227905, 0.9092052727937698, 0.39885279536247253),
    "4cfd6f6e1406": (5.960464477539063e-08, 5.960464477539063e-08, 0.45219236612319946, 0.9282801747322083, 0.4187443256378174),
    "2e1dc71913b1": (5.960464477539063e-08, 5.960464477539063e-08, 0.42718178033828735, 0.9987040276173502, 0.46892398595809937),
}


def _load(rid: str) -> dict:
    return json.loads((ROOT / "metrics/mqar" / f"withdrawal-probe-{rid}.json").read_text(encoding="utf-8"))


def _row41(rid: str) -> dict:
    return json.loads((ROOT / "metrics/mqar" / f"reconstruction-probe-{rid}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("rid", list(FILES))
def test_the_declared_run_is_in_the_file_and_the_hashes_are_the_declared_ones(rid):
    d = _load(rid)
    assert d["config"] == {"phases": 40, "sentinel_seed": 2, "config_hash_control": HASH_OFF, "config_hash_measurement": HASH_ON,
                           "bound": LifecycleConfig().replay_error_max, "radius": LifecycleConfig().demote_min_cosine}
    assert LifecycleConfig(sentinel_seed=2, withdraw_at_eviction=False).config_hash() == HASH_OFF
    assert LifecycleConfig(sentinel_seed=2, withdraw_at_eviction=True).config_hash() == HASH_ON
    assert d["journal"]["sha256"].startswith(JOURNAL_SHA[rid]) and d["journal"]["sha256"] == _row41(rid)["journal"]["sha256"]
    assert d["journal"]["entries_before"] == 200 and d["journal"]["sealed"] and d["journal"]["copied_to_temp"]
    c, m = d["control"], d["measurement"]
    assert c["config"]["withdraw_at_eviction"] is False and c["config"]["config_hash"] == HASH_OFF
    assert m["config"]["withdraw_at_eviction"] is True and m["config"]["config_hash"] == HASH_ON
    for r in (c, m):
        assert r["journal"]["sha256"] == d["journal"]["sha256"] and r["config"]["phases"] == 40 and len(r["phases"]) == 40
        assert r["phases_refused"] == 0 and r["sentinels"]["planted"] == 20 and r["sentinels"]["negctrl"] == 10


@pytest.mark.parametrize("rid", list(FILES))
def test_reading_a_the_control_reproduces_row_41_number_for_number_and_beyond(rid):
    d = _load(rid)
    c, rec = d["control"], _row41(rid)
    reg = wp.compare_with_record(c, rec)                       # the declared comparison, recomputed from the two files
    assert reg["reproduced_number_for_number"] and reg["differences"] == []
    assert (reg["scalars_compared"], reg["phases_compared"]) == (27, 40)
    stored = d["regression_against_row_41"]
    assert stored["record_file_present"] and stored["reproduced_number_for_number"]
    assert (stored["scalars_compared"], stored["phases_compared"], stored["differences"]) == (27, 40, [])
    # beyond the declared comparison: both sessions and the forty phase records are the row 41 file's
    assert c["process_a"] == rec["process_a"] and c["process_b"] == rec["process_b"] and c["phases"] == rec["phases"]
    assert c["withdrawals_at_eviction"] is None and all("withdrawn" not in (p["applied"] or {}) for p in c["phases"])
    ev = c["process_a"]["level_3_memory"]["evicted_address_reading"]
    assert ev == {"entries": 200, "still_reconstructed_under_bound": 200, "share": 1.0}       # row 41, reading (e)


@pytest.mark.parametrize("rid", list(FILES))
def test_reading_b_the_withdrawal_recomputed_from_the_phases_and_the_rows(rid):
    d = _load(rid)
    c, m = d["control"], d["measurement"]
    bound, radius = d["config"]["bound"], d["config"]["radius"]
    l3 = m["process_a"]["level_3_memory"]
    rows = l3["rows"]
    kept = [r for r in rows if r["state"] in (STATE_CONSOLIDATED, STATE_DEMOTED)]
    withdrawn = [r for r in rows if r["state"] == STATE_EVICTED]
    assert len(rows) == 220 and len(kept) == 20 and len(withdrawn) == 200
    evicted_sum = sum((p["applied"] or {}).get("evicted", 0) for p in m["phases"])
    withdrawn_sum = sum((p["applied"] or {}).get("withdrawn", 0) for p in m["phases"])
    assert evicted_sum == withdrawn_sum == m["withdrawals_at_eviction"] == 200     # every evicted entry had a pattern, withdrawn in its phase
    for p in m["phases"]:
        a = p["applied"] or {}
        assert a.get("withdrawn", 0) == a.get("evicted", 0)
    assert l3["patterns"] == m["phases"][-1]["memory_patterns"] == m["process_b"]["level_3_memory"]["patterns"] == 20
    # levels 1 and 2: exact, and identical to the control line for line
    assert m["process_a"]["level_1_counts"] == c["process_a"]["level_1_counts"]
    assert m["process_a"]["level_1_counts"]["exact_at_every_phase"] and m["process_a"]["level_1_counts"]["worst_defect"] == 0
    assert m["process_a"]["level_2_salience_line"] == c["process_a"]["level_2_salience_line"]
    assert m["process_a"]["level_2_salience_line"]["all_under_threshold"]
    # level 3, the kept addresses
    kept_defects = [r["defect"] for r in kept]
    assert max(kept_defects) == l3["kept"]["max_defect"] == ROW42[rid][0] <= bound and l3["kept"]["all_under_bound"]
    assert c["process_a"]["level_3_memory"]["kept"]["max_defect"] == ROW42[rid][1]
    # level 3b, the withdrawn addresses
    ev = l3["evicted_address_reading"]
    defects = [r["defect"] for r in withdrawn]
    assert all(x > bound for x in defects) and not any(r["under_bound"] for r in withdrawn)
    assert ev["entries"] == 200 and ev["still_reconstructed_under_bound"] == 0 and ev["share"] == 0.0 and ev["radius"] == radius
    assert ev["still_under_bound_with_kept_neighbour_within_radius"] == 0 and ev["still_under_bound_without_such_neighbour"] == 0
    assert (min(defects), max(defects)) == (ev["min_defect_evicted"], ev["max_defect_evicted"]) == (ROW42[rid][2], ROW42[rid][3])
    cosines = [r["nearest_kept"]["cosine"] for r in withdrawn]
    assert max(cosines) == ROW42[rid][4] < radius
    for r in withdrawn:
        nk = r["nearest_kept"]
        assert nk["within_radius"] is False and nk["within_radius"] == (nk["cosine"] >= radius)
        assert any(k["entry_id"] == nk["entry_id"] for k in kept)        # the neighbour named is a kept entry
    # level 4, unchanged by construction
    assert m["process_a"]["level_4_content"]["holds"] and m["process_a"]["level_4_content"]["evicted"] == 200


@pytest.mark.parametrize("rid", list(FILES))
def test_readings_c_and_d_and_the_letter(rid):
    d = _load(rid)
    c, m = d["control"], d["measurement"]
    r = d["readings"]
    assert r["c_collateral_kept_addresses"] == []                                   # (c): nothing damaged
    assert c["agreement"] == {"number_for_number": True, "differences": []}       # (d): both configurations
    assert m["agreement"] == {"number_for_number": True, "differences": []}
    assert r["d_control_two_processes_agree"] and r["d_measurement_two_processes_agree"]
    assert wp.readings(c, m, d["regression_against_row_41"]) == r                 # the readings recomputed by the module
    assert r["letter"].startswith("b1:") and d["verdict"] == r["letter"]
    assert (r["a_patterns_in_memory_control"], r["b_patterns_in_memory"], r["b_withdrawals_at_eviction"], r["b_evicted"]) == (220, 20, 200, 200)
    assert r["b_still_under_bound_after_withdrawal"] == 0 and r["b_share_still_under_bound"] == 0.0


@pytest.mark.parametrize("rid", list(FILES))
def test_the_retained_log_is_the_summary_of_the_file(rid):
    d = _load(rid)
    log = (ROOT / "metrics/mqar/logs" / f"withdrawal-probe-{rid}.log").read_bytes().decode("utf-16").splitlines()
    assert len(log) == 10 and log[:-1] == wp.summary_lines(d) and log[-1].startswith("[record] retained:")


@pytest.mark.parametrize("rid", list(FILES))
def test_no_summary_existed_on_the_three_journals_so_the_cascade_correction_of_2026_10_12_leaves_the_row(rid):
    """ADR-007, correction of 2026-10-12: the withdrawal did not reach the sources a summary takes with
    it through the store's cascade. Whether the row 42 measurement could have met that case is read
    here from the files, not reasoned: the forty phase records of both runs carry the demotion and
    eviction counts and the number of summaries alive, and the snapshot the state counts. On the three
    journals no group was ever demoted, no summary ever existed and no entry is demoted at the end, so
    no eviction cascaded and the correction changes nothing of what these files record."""
    d = _load(rid)
    for name in ("control", "measurement"):
        run = d[name]
        assert len(run["phases"]) == 40 and all(not p["refused"] for p in run["phases"])
        assert sum(p["applied"]["demoted_groups"] for p in run["phases"]) == 0
        assert all(p["summaries"] == 0 for p in run["phases"])
        assert sum(p["applied"]["evicted"] for p in run["phases"]) == 200
        snap = run["process_a"]["snapshot"]
        assert snap["demoted"] == 0 and snap["summaries"] == 0 and snap["evicted"] == 200 and snap["consolidated"] == 20
        assert run["process_b"]["snapshot"]["demoted"] == 0 and run["process_b"]["snapshot"]["summaries"] == 0
    assert sum(p["applied"]["withdrawn"] for p in d["measurement"]["phases"]) == 200      # one withdrawal per eviction, none by cascade


# ---------------------------------------------------------------------------- the rerun of 2026-10-12
RERUN = ROOT / "metrics/mqar/withdrawal-probe-8ceba5db8d0b-rerun-2026-10-12.json"
RERUN_LOG = ROOT / "metrics/mqar/logs/withdrawal-probe-8ceba5db8d0b-rerun-2026-10-12.log"
RERUN_SHA_LF = "baa662a719cd057b4211858f98165e00ae08de94c4c3313046e8f562fe04f7f4"
RERUN_LOG_SHA_LF = "2b17eb23511b6bab872be9a473cb7c928a568fb3fd8704429cfd2774848f9123"
RERUN_TIMESTAMP = "2026-10-11T22:40:11+00:00"


def _rerun() -> dict:
    return json.loads(RERUN.read_text(encoding="utf-8"))


def test_the_rerun_of_2026_10_12_on_seed_1337_reproduces_the_row_42_file_number_for_number_recomputed_here():
    """ADR-007, correction of 2026-10-12, its declared rerun: the probe run again by the founder on the journal
    of seed 1337 with the corrected code, both configurations. The comparison with the row 42 file is
    recomputed here from the two files, not read from the rerun file's own field: the journal, the
    configuration, the control and the measurement on the declared scalars, the forty phase records and the
    per entry rows, the readings and the letter. The correction changed nothing of the row on this journal."""
    import hashlib
    rerun, record, row41 = _rerun(), _load("8ceba5db8d0b"), _row41("8ceba5db8d0b")
    assert hashlib.sha256(RERUN.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == RERUN_SHA_LF
    assert rerun["timestamp_utc"] == RERUN_TIMESTAMP and record["timestamp_utc"] == "2026-10-02T21:08:07+00:00"
    assert rerun["journal"]["sha256"] == record["journal"]["sha256"] == row41["journal"]["sha256"]
    assert rerun["config"] == record["config"]
    cmp = wp.compare_with_previous_run(rerun, record)
    assert cmp["equal_number_for_number"] and cmp["differences"] == []
    assert cmp["control"] == {"reproduced_number_for_number": True, "scalars_compared": 27, "phases_compared": 40}
    assert cmp["measurement"] == {"reproduced_number_for_number": True, "scalars_compared": 27, "phases_compared": 40}
    own = rerun["rerun_against_previous"]                                # the file's own field says the same
    assert own["equal_number_for_number"] and own["differences"] == [] and own["previous_timestamp_utc"] == record["timestamp_utc"]
    assert wp.compare_with_record(rerun["control"], row41)["reproduced_number_for_number"]   # reading (a), still true
    assert rerun["verdict"] == record["verdict"] and rerun["verdict"].startswith("b1:")


def test_the_rerun_holds_no_summary_and_no_source_text_so_the_two_changes_of_2026_10_12_leave_its_records():
    rerun = _rerun()
    for name in ("control", "measurement"):
        ph = rerun[name]["phases"]
        assert len(ph) == 40 and all(not p["refused"] for p in ph)
        assert sum(p["applied"]["demoted_groups"] for p in ph) == 0 and all(p["summaries"] == 0 for p in ph)
        assert all("source_released" not in p["applied"] for p in ph)   # ADR-009 B2: the journal of the record holds no source text
        assert sum(p["applied"]["evicted"] for p in ph) == 200
    assert sum(p["applied"]["withdrawn"] for p in rerun["measurement"]["phases"]) == 200
    assert all("withdrawn" not in p["applied"] for p in rerun["control"]["phases"])


def test_the_rerun_log_is_the_summary_of_its_file_and_the_provenance_names_both():
    import hashlib
    rerun = _rerun()
    raw = RERUN_LOG.read_bytes()
    assert hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == RERUN_LOG_SHA_LF
    log = raw.decode("utf-8").splitlines()                                # UTF-8 this time, the provenance says so
    assert len(log) == 11 and log[:-1] == wp.summary_lines(rerun) and log[-1].startswith("[record] retained:")
    assert log[9].startswith("rerun against withdrawal-probe-8ceba5db8d0b.json (2026-10-02T21:08:07+00:00): equal number for number = True")
    prov = json.loads((ROOT / "metrics/mqar/PROVENANCE-withdrawal-probe.json").read_text(encoding="utf-8"))["rerun_2026_10_12"]
    assert prov["file"]["file_sha256_lf"] == RERUN_SHA_LF and prov["file"]["timestamp_utc"] == RERUN_TIMESTAMP
    assert prov["file"]["journal_sha256"] == rerun["journal"]["sha256"] and prov["code_commit"].startswith("ba6fc60")
    assert RERUN_LOG_SHA_LF in prov["file"]["log"]
