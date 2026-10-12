"""RESULTS row 43: the revision probe on the organ alone, run by the founder on 2026-10-12 at the four
seeds of rows 13b and 13c (ADR-007 amendment 2026-10-12, declared before any number, validated at 11:25,
built tests first), read cold from the files the same day. Every scalar the row states is recomputed here
from the committed files: the configuration and the counts of each seed, the seven measures, the histogram
of versions put equal to the plan of revisions, the agreement of the two processes recomputed from their
fields, the amendment's letter recomputed from the parts alone in the declared order, the provenance
hashes; and the probe is run again at each seed and compared to its file number for number, the timestamp
apart, since it needs no key of the record, no checkpoint and no training. A change to a file breaks this
test; a change to the row without the file cannot pass it."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cortex_c2b import revision_probe as rvp                                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (0, 1, 2, 3)
N, THRESHOLD = 200, 0.15
GENERATOR = {0: "a5e22c70a3275ed4", 1: "82bd2ff094d9b030", 2: "80bb909b049846de", 3: "0c37aac8c10a178f"}  # row 13c's
LIFECYCLE_HASH = "2c3b9885efd9213d"      # period 10000 writes, batch 200, no sentinel, decay 1.0, the withdrawal on by default
LF_SHA = {0: "4da2c436466803a1", 1: "48cec6c397850c16", 2: "9e0908be56771e38", 3: "0bfa2f380bcb20c2"}
TIMESTAMPS = {0: "2026-10-12T10:57:25+00:00", 1: "2026-10-12T10:57:33+00:00",
              2: "2026-10-12T10:57:41+00:00", 3: "2026-10-12T10:57:48+00:00"}
ROW43_COUNTS = {"n_facts": N, "admitted": N, "revised": 100, "revised_twice": 33, "revisions_made": 133,
                "revisions_refused": 0}
ROW43_MEASURES = {"current_returned": 1.0, "past_returned": 0.0, "unrevised_intact": 1.0, "history_exact": True,
                  "entries_exact": True, "memory_unchanged": True, "agreement": True}
HISTOGRAM = {1: 100, 2: 67, 3: 33}      # 100 unrevised, 67 revised once, 33 revised twice; 67 + 2 * 33 = 133 revisions
COMPARED = ("current_returned", "past_returned", "unrevised_intact", "history_exact", "entries_exact",
            "versions", "memory_patterns", "revisions_declared")


def _path(seed: int) -> Path:
    return ROOT / "metrics/mqar" / f"revision-probe-seed{seed}.json"


def _load(seed: int) -> dict:
    return json.loads(_path(seed).read_text(encoding="utf-8"))


def _lf_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def _letter_from_parts(out: dict) -> str:
    """The amendment's readings in their declared order, from the file's parts and not from its letter."""
    a, b = out["process_a"], out["process_b"]
    if any(a[k] != b[k] for k in COMPARED):
        return "d"
    if not a["entries_exact"] or not a["history_exact"]:
        return "c"
    if a["past_returned"] > 0.0 or a["current_returned"] < 1.0 or a["unrevised_intact"] < 1.0:
        return "b"
    if out["memory"]["patterns_before"] != out["memory"]["patterns_after"]:
        return "b"
    return "a"


@pytest.mark.parametrize("seed", SEEDS)
def test_each_file_carries_the_graved_configuration_counts_and_measures(seed):
    out = _load(seed)
    c = out["config"]
    assert c["seed"] == c["index_seed"] == seed and c["admission_threshold"] == THRESHOLD
    assert c["generator_hash"] == GENERATOR[seed] and c["lifecycle_hash"] == LIFECYCLE_HASH
    assert {k: c[k] for k in ROW43_COUNTS} == ROW43_COUNTS
    assert out["timestamp_utc"] == TIMESTAMPS[seed]
    assert out["thresholds"] == rvp.THRESHOLDS == {"current_returned": 1.0, "past_returned": 0.0, "unrevised_intact": 1.0,
                                                   "history_exact": True, "entries_exact": True, "memory_unchanged": True,
                                                   "agreement": True}
    assert out["measures"] == ROW43_MEASURES
    assert out["memory"] == {"patterns_before": N, "patterns_after": N}
    assert out["storage"] == {"mode": "durable", "policy": "stop", "degraded_reason": None, "unpersisted_events": 0,
                              "plaintext_scope": False}
    assert out["verdict"] == out["readings"]["letter"] and out["verdict"].startswith("a:")


@pytest.mark.parametrize("seed", SEEDS)
def test_the_histogram_of_versions_equals_the_plan_of_revisions(seed):
    out = _load(seed)
    for proc in ("process_a", "process_b"):
        versions, plan = out[proc]["versions"], out["plan"]
        assert len(versions) == N and len(plan) == 100 and sum(1 for s in plan.values() if len(s) == 2) == 33
        assert {k: sum(1 for v in versions.values() if v == k) for k in HISTOGRAM} == HISTOGRAM
        assert all(versions[e] == 1 + len(seq) for e, seq in plan.items())          # one version per revision made
        assert all(versions[e] == 1 for e in versions if e not in plan)             # the unrevised hold one
        assert sum(v - 1 for v in versions.values()) == 133 == out[proc]["revisions_declared"]["past_versions"]
        assert out[proc]["revisions_declared"]["entries_revised"] == 100 and out[proc]["memory_patterns"] == N


@pytest.mark.parametrize("seed", SEEDS)
def test_the_two_processes_agree_and_the_letter_recomputed_from_the_parts_is_a(seed):
    out = _load(seed)
    a, b = out["process_a"], out["process_b"]
    assert [k for k in COMPARED if a[k] != b[k]] == [] == out["agreement"]["differences"]
    assert out["agreement"]["number_for_number"] is True and b["process"].startswith("new")
    assert _letter_from_parts(out) == "a" == out["readings"]["letter"][:1]
    assert rvp._letter(out["measures"]) == out["verdict"]


def test_the_provenance_file_names_the_four_files_and_their_hashes():
    prov = json.loads((ROOT / "metrics/mqar/PROVENANCE-revision-probe.json").read_text(encoding="utf-8"))
    assert prov["row"] == "43" and prov["code_commit"] == "59d96f2"
    for seed in SEEDS:
        entry = prov["files"][f"revision-probe-seed{seed}.json"]
        assert entry["sha256_lf"] == LF_SHA[seed] == _lf_sha(_path(seed))
        assert entry["timestamp_utc"] == TIMESTAMPS[seed] and entry["generator_hash"] == GENERATOR[seed]
        assert entry["lifecycle_hash"] == LIFECYCLE_HASH and entry["letter"] == "a"
    k = prov["per_seed_constants"]
    assert {x: k[x] for x in ROW43_COUNTS} == ROW43_COUNTS and k["measures"] == ROW43_MEASURES
    assert {int(x): v for x, v in k["versions_histogram"].items()} == HISTOGRAM


@pytest.mark.parametrize("seed", SEEDS)
def test_the_probe_run_again_at_the_seed_reproduces_its_file_number_for_number(seed):
    file, out = _load(seed), rvp.run(n=N, seed=seed)
    for k in ("config", "memory", "plan", "storage", "thresholds", "measures", "agreement", "readings", "verdict"):
        assert out[k] == file[k], k
    for proc in ("process_a", "process_b"):
        assert {k: out[proc][k] for k in file[proc]} == file[proc], proc
