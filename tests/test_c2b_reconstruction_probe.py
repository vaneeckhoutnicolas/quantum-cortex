"""The reconstruction probe (ADR-007 amendment 2026-09-30, declared before any measurement):
the organ's layer of numbers, in three levels and a reading. What these tests pin on a small
sealed journal: the original is copied and never modified; a sentinel entity that collides
with the journal's aborts; a journal that already holds phases is refused; the thresholds in
the file are the declared ones; level 1 is exact at every phase; level 4 holds; every replay
recorded is under the bound (a failed one is withdrawn by the code); the second process, from
the disk alone, agrees with the first number for number; and the level 2 derivation is checked
against a hand computation on one entry. Nothing here is a number of the record."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cortex_c2b import Journal, POLICY_STOP, STATE_EVICTED                       # noqa: E402
from cortex_c2b.crypto import generate_key                                       # noqa: E402
from cortex_c2b.hm_protocol import generate_facts, Fact                          # noqa: E402
from cortex_c2b.lifecycle import LifecycleConfig                                 # noqa: E402
from cortex_c2b.read_path import JournalPath                                     # noqa: E402
from cortex_c2b.write_path import WritePath                                      # noqa: E402
from cortex_c2b import reconstruction_probe as rp                                # noqa: E402

N = 40                    # a small journal: forty facts from the frozen generator at a test seed
JOURNAL_SEED = 5          # not the protocol's 0, not its never planted 10 000, not the mark oracle's 100 000


def _plant(tmp_path: Path, key: bytes, n: int = N, seed: int = JOURNAL_SEED) -> Path:
    d = tmp_path / "source"; d.mkdir()
    j = Journal(d / "journal.jsonl", key=key, policy=POLICY_STOP)
    wp, jp = WritePath(j, seed=0), JournalPath(j)
    facts, _ = generate_facts(n, seed)
    for i, f in enumerate(facts):
        rep = wp.write(f.cue, f.statement.encode(), f.schema, now=float(i))
        if rep.admitted:
            jp.on_write(rep.entry)
    return d / "journal.jsonl"


@pytest.fixture(scope="module")
def probe_run(tmp_path_factory):
    key = generate_key()
    tmp = tmp_path_factory.mktemp("recon")
    src = _plant(tmp, key)
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    out = rp.run(src, key, phases=12, sentinel_seed=rp.SENTINEL_SEED)
    after = hashlib.sha256(src.read_bytes()).hexdigest()
    return {"key": key, "src": src, "out": out, "before": before, "after": after}


def test_the_original_journal_is_never_modified(probe_run):
    assert probe_run["before"] == probe_run["after"]
    assert probe_run["out"]["journal"]["sha256"] == probe_run["before"]
    assert probe_run["out"]["journal"]["copied_to_temp"] is True


def test_the_file_carries_the_declared_thresholds_and_run(probe_run):
    out = probe_run["out"]
    assert out["thresholds"] == {"level_1": "exact", "level_2_relative": 1e-3,
                                 "level_3_bound": LifecycleConfig().replay_error_max}
    assert out["config"]["sentinel_seed"] == rp.SENTINEL_SEED == 2
    assert out["config"]["reads_between_phases"] == 0
    assert out["config"]["phases"] == 12 and len(out["phases"]) == 12
    assert out["config"]["now_plant"] == float(N)                     # the last write time plus one
    assert out["sentinels"]["planted"] == LifecycleConfig().sentinel_facts


def test_level_1_is_exact_at_every_phase_and_level_4_holds(probe_run):
    a = probe_run["out"]["process_a"]
    assert a["level_1_counts"]["exact_at_every_phase"] and a["level_1_counts"]["worst_defect"] == 0
    assert a["level_1_counts"]["phases_checked"] == 12
    assert a["level_4_content"]["holds"]
    # the identity, recomputed here from the last phase: entries = live + consolidated + K * summaries alive + evicted
    last = a["level_1_counts"]["per_phase"][-1]
    assert last["entries"] == last["live"] + last["consolidated"] + last["derived_demoted"] + last["derived_evicted"]


def test_every_recorded_replay_is_under_the_bound(probe_run):
    out = probe_run["out"]
    pc = out["process_a"]["level_3_memory"]["positive_control"]
    bound = out["thresholds"]["level_3_bound"]
    assert pc["replays"] == out["replays"]["count"] > 0
    assert pc["max_error_at_replay"] <= bound or out["replays"]["withdrawn"] > 0
    # the memory holds exactly the verified replays
    assert out["process_a"]["level_3_memory"]["patterns"] == out["replays"]["count"] - out["replays"]["withdrawn"]


def test_the_second_process_agrees_number_for_number(probe_run):
    out = probe_run["out"]
    assert out["agreement"]["number_for_number"], out["agreement"]["differences"]
    assert out["process_b"]["process"].startswith("new")
    assert out["process_b"]["phases_found"] == 12 and out["process_b"]["durable"]
    assert out["readings"]["two_processes_agree"]


def test_level_2_derivation_matches_a_hand_computation(probe_run):
    out = probe_run["out"]
    rows = out["process_a"]["level_2_salience_line"]["rows"]
    assert rows, "no entry was consolidated in twelve phases"
    decay = LifecycleConfig().decay
    for r in rows:
        derived = r["salience_final"] / (decay ** r["decays"]) if r["decays"] else r["salience_final"]
        assert derived == r["derived"]
        if r["contracted"]:
            assert r["decays"] == 0 and r["salience_at_consolidation"] == 1.0
    # the population is every entry ever consolidated, evicted ones included, their line ending at eviction
    l2 = out["process_a"]["level_2_salience_line"]
    assert l2["entries"] == l2["kept"] + l2["evicted"]


def test_a_colliding_sentinel_entity_aborts(tmp_path, monkeypatch):
    key = generate_key()
    src = _plant(tmp_path, key, n=8, seed=JOURNAL_SEED)
    facts, _ = generate_facts(8, JOURNAL_SEED)
    victim = facts[0]
    real = rp.generate_facts

    def colliding(n, seed=0):
        fs, h = real(n, seed)
        return [Fact(**{**fs[0].__dict__, "entity": victim.entity, "statement": victim.statement})] + fs[1:], h
    monkeypatch.setattr(rp, "generate_facts", colliding)
    with pytest.raises(SystemExit, match="collide"):
        rp.session_a(src, tmp_path / "copy", key, phases=1)


def test_a_reserved_sentinel_seed_and_a_journal_with_phases_are_refused(tmp_path, probe_run):
    key = generate_key()
    src = _plant(tmp_path, key, n=8, seed=JOURNAL_SEED)
    with pytest.raises(SystemExit, match="sentinel seed"):
        rp.session_a(src, tmp_path / "copy0", key, phases=1, sentinel_seed=0)
    # a journal that already holds lifecycle phases is refused: the probe runs on a journal without history
    out, copy = rp.session_a(src, tmp_path / "copy1", key, phases=1)
    with pytest.raises(SystemExit, match="phases"):
        rp.session_a(copy, tmp_path / "copy2", key, phases=1)
