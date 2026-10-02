"""The withdrawal at eviction (ADR-007 amendment 2026-10-01, declared before any measurement,
built after the founder's validation). What these tests pin on a small sealed journal: the
configuration hash of the record's files is unchanged while the flag is off; the memory
withdraws a pattern by identity, exactly; under the flag the phase that evicts an entry
withdraws its pattern and says so in its event, and a scheduler reopened on the journal
rebuilds the same memory from the log alone; with the flag off nothing changes (the phase
events carry no withdrawal key, the evicted addresses survive as in row 41); the probe runs
both configurations from the same original, the control's comparison with a record file
reports equality on itself and a difference on a tampered copy, the attribution of every
withdrawn address names a kept neighbour, and the reading is one of the amendment's letters.
Nothing here is a number of the record."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cortex_c2b import Journal, POLICY_STOP, STATE_EVICTED, STATE_CONSOLIDATED, STATE_DEMOTED   # noqa: E402
from cortex_c2b.crypto import generate_key                                       # noqa: E402
from cortex_c2b.hm_protocol import generate_facts                                # noqa: E402
from cortex_c2b.lifecycle import LifecycleConfig, LifecycleScheduler, PersistentMemory   # noqa: E402
from cortex_c2b.read_path import JournalPath                                     # noqa: E402
from cortex_c2b.write_path import WritePath                                      # noqa: E402
from cortex_c2b import reconstruction_probe as rp                                # noqa: E402
from cortex_c2b import withdrawal_probe as wp                                    # noqa: E402

N = 40
JOURNAL_SEED = 5
PHASES_TEST = 40          # enough phases for the tribunal and the floor to evict on a journal without reads


def _plant(tmp_path: Path, key: bytes, n: int = N, seed: int = JOURNAL_SEED) -> Path:
    d = tmp_path / "source"; d.mkdir()
    j = Journal(d / "journal.jsonl", key=key, policy=POLICY_STOP)
    w, jp = WritePath(j, seed=0), JournalPath(j)
    facts, _ = generate_facts(n, seed)
    for i, f in enumerate(facts):
        rep = w.write(f.cue, f.statement.encode(), f.schema, now=float(i))
        if rep.admitted:
            jp.on_write(rep.entry)
    return d / "journal.jsonl"


def test_the_hash_of_the_record_is_unchanged_while_the_flag_is_off():
    assert LifecycleConfig(sentinel_seed=2).config_hash() == "3212d4c151c5c862"          # rows 36 to 41
    assert LifecycleConfig(sentinel_seed=2, withdraw_at_eviction=False).config_hash() == "3212d4c151c5c862"
    on = LifecycleConfig(sentinel_seed=2, withdraw_at_eviction=True).config_hash()
    assert on != "3212d4c151c5c862" and len(on) == 16


def test_the_memory_withdraws_a_pattern_by_identity_exactly():
    rng = np.random.default_rng(0)
    m = PersistentMemory(dim=16)
    cues = {f"e{i}": rng.standard_normal(16).astype(np.float32) for i in range(3)}
    for eid, c in cues.items():
        assert m.replay(c) <= 0.5
        m.label_last(eid)
    assert len(m) == 3 and m.ids() == ["e0", "e1", "e2"]
    before = m._K.copy()
    assert m.withdraw("e1") and len(m) == 2 and m.ids() == ["e0", "e2"]
    assert np.array_equal(m._K, before[[0, 2]])            # exact: the other rows untouched
    assert not m.withdraw("e1")                             # nothing left to withdraw
    m.withdraw_last()
    assert m.ids() == ["e0"]


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    key = generate_key()
    tmp = tmp_path_factory.mktemp("withdrawal")
    src = _plant(tmp, key)
    control = rp.run(src, key, phases=PHASES_TEST, sentinel_seed=rp.SENTINEL_SEED, withdraw=False)
    measurement = rp.run(src, key, phases=PHASES_TEST, sentinel_seed=rp.SENTINEL_SEED, withdraw=True)
    return {"key": key, "src": src, "control": control, "measurement": measurement}


def test_with_the_flag_off_nothing_changes(two_runs):
    c = two_runs["control"]
    assert c["config"]["withdraw_at_eviction"] is False and c["config"]["config_hash"] == "3212d4c151c5c862"
    assert c["withdrawals_at_eviction"] is None
    assert all("withdrawn" not in (p["applied"] or {}) for p in c["phases"])
    ev = c["process_a"]["level_3_memory"]["evicted_address_reading"]
    assert "radius" not in ev
    if ev["entries"]:                                        # as in row 41: an eviction never withdrew a pattern
        assert ev["share"] == 1.0
    assert c["process_a"]["level_3_memory"]["patterns"] == len(c["process_a"]["level_3_memory"]["rows"])


def test_under_the_flag_the_eviction_withdraws_and_the_event_says_so(two_runs):
    m = two_runs["measurement"]
    assert m["config"]["withdraw_at_eviction"] is True and m["config"]["config_hash"] != "3212d4c151c5c862"
    evicted_total = sum((p["applied"] or {}).get("evicted", 0) for p in m["phases"])
    withdrawn_total = sum((p["applied"] or {}).get("withdrawn", 0) for p in m["phases"])
    assert m["withdrawals_at_eviction"] == withdrawn_total
    l3 = m["process_a"]["level_3_memory"]
    kept = sum(1 for r in l3["rows"] if r["state"] in (STATE_CONSOLIDATED, STATE_DEMOTED))
    evicted = sum(1 for r in l3["rows"] if r["state"] == STATE_EVICTED)
    assert evicted > 0, "the rehearsal must evict, or the test is vacuous"
    assert evicted_total >= evicted                          # live entries evicted by the tribunal never had a pattern
    assert withdrawn_total == evicted                        # every consolidated entry evicted had its pattern withdrawn
    assert l3["patterns"] == kept                            # the memory holds the kept patterns and nothing else
    assert m["phases"][-1]["memory_patterns"] == kept


def test_a_reopened_scheduler_rebuilds_the_same_memory_under_the_flag(two_runs):
    m = two_runs["measurement"]
    assert m["agreement"]["number_for_number"], m["agreement"]["differences"]
    assert m["process_b"]["level_3_memory"]["patterns"] == m["process_a"]["level_3_memory"]["patterns"]
    assert m["readings"]["two_processes_agree"] and m["readings"]["withdraw_at_eviction"]


def test_every_withdrawn_address_is_attributed_to_a_kept_neighbour(two_runs):
    l3 = two_runs["measurement"]["process_a"]["level_3_memory"]
    ev = l3["evicted_address_reading"]
    assert ev["radius"] == LifecycleConfig().demote_min_cosine
    rows = [r for r in l3["rows"] if r["state"] == STATE_EVICTED]
    assert len(rows) == ev["entries"]
    for r in rows:
        nk = r["nearest_kept"]
        if l3["patterns"]:
            assert nk is not None and -1.0 <= nk["cosine"] <= 1.0 and nk["within_radius"] == (nk["cosine"] >= ev["radius"])
    still = [r for r in rows if r["under_bound"]]
    assert ev["still_reconstructed_under_bound"] == len(still)
    assert ev["still_under_bound_with_kept_neighbour_within_radius"] + ev["still_under_bound_without_such_neighbour"] == len(still)


def test_the_probe_reads_in_the_amendments_letters_and_the_control_compares(two_runs, tmp_path):
    c, m = two_runs["control"], two_runs["measurement"]
    # the comparison of the control with a record: itself (equal), then a tampered copy (a difference named)
    same = wp.compare_with_record(c, json.loads(json.dumps(c)))
    assert same["reproduced_number_for_number"] and same["differences"] == []
    tampered = json.loads(json.dumps(c))
    tampered["process_a"]["level_2_salience_line"]["max_relative_defect"] += 1e-3
    diff = wp.compare_with_record(c, tampered)
    assert not diff["reproduced_number_for_number"] and any("max_relative_defect" in d for d in diff["differences"])
    # the readings on the two runs
    r = wp.readings(c, m, same)
    assert r["a_control_reproduces_row_41"] is True and r["b_level_1_exact"] and r["level_4_holds"]
    assert r["b_level_2_identical_to_control"]
    assert r["letter"].split(":")[0] in ("b", "b1", "b2", "b3", "c", "d", "a")
    # the whole probe, end to end, with the control as its own record file
    rec = tmp_path / "record.json"; rec.write_text(json.dumps(c), encoding="utf-8")
    out = wp.run(two_runs["src"], two_runs["key"], phases=PHASES_TEST, sentinel_seed=rp.SENTINEL_SEED, record_file=rec)
    assert out["regression_against_row_41"]["reproduced_number_for_number"]
    assert out["config"]["config_hash_control"] == "3212d4c151c5c862" != out["config"]["config_hash_measurement"]
    assert out["verdict"] == out["readings"]["letter"]
    assert len(wp.summary_lines(out)) == 9
