"""`cortex_c2b.scale_reading`, row 53's cold reading of part C's files: the expected set, the digests, the pooling per size,
encoder and gate, C.4's two conditions through `scale_probe.read_c4`, the Decision 13 successor sentence filled in, the
progress files verified by hash and read apart, the reading file never overwritten. Tested on copies of committed files
renamed to other encoders and on constructed progress files; the record's numbers are pinned by the row's record test."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
M = ROOT / "metrics" / "mqar"

from cortex_c2b import scale_reading as SR                       # noqa: E402
from cortex_c2b import scale_probe as SP                         # noqa: E402

S2024, S1337, S7 = "4cfd6f6e1406", "8ceba5db8d0b", "2e1dc71913b1"
SOURCE = {kind: M / name for kind, name in (("encoding", "scale-encoding-50000-4cfd6f6e1406.json"), ("c2-identity", "scale-c2-50000-4cfd6f6e1406-identity.json"),
                                            ("c2-surprise", "scale-c2-50000-4cfd6f6e1406-surprise.json"), ("cascade-identity", "scale-cascade-50000-4cfd6f6e1406-identity.json"),
                                            ("cascade-surprise", "scale-cascade-50000-4cfd6f6e1406-surprise.json"))}
needs_files = pytest.mark.skipif(not all(p.exists() for p in SOURCE.values()), reason="the committed part C files of seed 2024, draw 50000, are needed")


def test_the_expected_set_is_the_records_in_the_order_of_g():
    exp = SR.expected_files()
    assert len(exp) == 79 and len({e["name"] for e in exp}) == 79
    assert [e["run_id"] for e in exp[:25]] == [S1337] * 25 and [e["run_id"] for e in exp[25:50]] == [S2024] * 25 and [e["run_id"] for e in exp[50:75]] == [S7] * 25
    assert exp[0]["name"] == "scale-encoding-50000-8ceba5db8d0b.json" and exp[1]["name"] == "scale-c2-50000-8ceba5db8d0b-identity.json"
    assert exp[3]["name"] == "scale-cascade-50000-8ceba5db8d0b-identity.json" and exp[4]["name"] == "scale-cascade-50000-8ceba5db8d0b-surprise.json"
    tail = exp[75:]
    assert [e["name"] for e in tail] == ["scale-encoding-60000-4cfd6f6e1406.json", "scale-c2-60000-4cfd6f6e1406-identity.json", "scale-c2-60000-4cfd6f6e1406-surprise.json",
                                         "scale-cascade-60000-4cfd6f6e1406-identity.json"]
    assert all(e["size"] == "hundred_thousand" for e in tail) and all(e["size"] == "ten_thousand" for e in exp[:75])
    assert SR.FILE_RE.match("scale-cascade-60000-4cfd6f6e1406-identity.json") and not SR.FILE_RE.match("scale-reading-row53.json")


def _renamed(src: Path, dst_dir: Path, run_id: str, draw: int, gate: str | None, mutate=None) -> Path:
    d = json.loads(src.read_text(encoding="utf-8"))
    d["run_id"] = run_id; d["draw"] = draw
    kind = "encoding" if "encoding" in src.name else "c2" if "-c2-" in src.name else "cascade"
    if mutate:
        mutate(d)
    name = f"scale-{kind}-{draw}-{run_id}" + (f"-{gate}" if gate else "") + ".json"
    (dst_dir / name).write_text(json.dumps(d), encoding="utf-8")
    return dst_dir / name


@needs_files
def test_load_files_lists_the_present_the_missing_and_the_outside(tmp_path):
    md = tmp_path / "mqar"; md.mkdir()
    for kind, src in SOURCE.items():
        gate = kind.split("-")[1] if "-" in kind else None
        _renamed(src, md, S1337, 50000, gate)
    _renamed(SOURCE["cascade-identity"], md, S1337, 50000, "identity", mutate=lambda d: None).rename(md / "scale-cascade-50000-8ceba5db8d0b-identity.json")
    (md / "scale-cascade-50009-8ceba5db8d0b-identity.json").write_text("{}", encoding="utf-8")   # a draw outside the seal's: listed apart, never read
    (md / "scale-cascade-60000-4cfd6f6e1406-identity-refused.json").write_text("{}", encoding="utf-8")   # a refused run retained under its own name (C.3 (b)): listed apart
    (md / "scale-reading-row53.json").write_text("{}", encoding="utf-8")                           # a reading file: not a file of the probe
    f = SR.load_files(md)
    assert len(f["present"]) == 5 and len(f["missing"]) == 74 and f["outside_the_expected_set"] == ["scale-cascade-50009-8ceba5db8d0b-identity.json", "scale-cascade-60000-4cfd6f6e1406-identity-refused.json"]
    assert all(len(v["sha256_lf"]) == 16 for v in f["present"].values())


@needs_files
def test_the_digests_carry_the_files_numbers_and_the_pooling_keeps_the_encoders_apart(tmp_path):
    md = tmp_path / "mqar"; md.mkdir()
    for run_id in (S1337, S2024, S7):
        for draw in (50000, 50001):
            _renamed(SOURCE["encoding"], md, run_id, draw, None)
            for gate in ("identity", "surprise"):
                _renamed(SOURCE[f"c2-{gate}"], md, run_id, draw, gate)
                _renamed(SOURCE[f"cascade-{gate}"], md, run_id, draw, gate)
    r = SR.read_rows(md)
    assert r["files"]["present"] == 30 and len(r["files"]["missing"]) == 49
    src = json.loads(SOURCE["c2-identity"].read_text(encoding="utf-8"))
    c = r["c2"]["scale-c2-50001-2e1dc71913b1-identity.json"]
    assert c["run_id"] == S7 and c["draw"] == 50001 and c["never_planted"]["at_or_above"] == src["aggregates"]["never_planted"]["max_mark_at_or_above_threshold"]
    assert c["pairs_E1"]["contested"] == src["aggregates"]["pairs_E1"]["contested"] and c["sample"]["own_in_window"]["I0"] == src["aggregates"]["sample"]["i0"]["own_in_window"]
    assert c["time_per_query"]["blocked"]["I1"]["median_ns"] == src["time_per_query"]["blocked"]["I1"]["median"]
    cas = r["cascades"]["scale-cascade-50000-8ceba5db8d0b-surprise.json"]
    srcc = json.loads(SOURCE["cascade-surprise"].read_text(encoding="utf-8"))
    assert cas["phases"] == srcc["phases"] and cas["states"] == srcc["states"] and cas["phase_times"]["n"] == len(srcc["phase_times_seconds"])
    assert abs(cas["phase_times"]["sum_seconds"] - sum(srcc["phase_times_seconds"])) < 1e-9 and cas["phase_times"]["fit"]["n"] == srcc["phases"]
    p = r["pooled"]["ten_thousand/identity"]
    assert p["c2_files"] == 6 and p["cascade_files"] == 6 and p["encoders"] == [1337, 2024, 7]            # the record's order
    nn = src["aggregates"]["never_planted"]["max_mark_at_or_above_threshold"]
    assert p["never_planted"]["at_or_above"] == 6 * nn and p["never_planted"]["n"] == 6000 and p["never_planted"]["per_file_bounds"] == {"min": nn, "max": nn, "n": 6}
    assert list(p["never_planted"]["per_encoder"]) == ["1337", "2024", "7"] and p["never_planted"]["per_encoder"]["7"] == {**p["never_planted"]["per_encoder"]["7"], "at_or_above": 2 * nn, "n": 2000}
    assert set(p["pairs_E1"]["per_encoder"]) == {"1337", "2024", "7"} and p["pairs_E1"]["per_encoder"]["1337"]["contested"] == 2 * src["aggregates"]["pairs_E1"]["contested"]
    assert p["cascade"]["faults_total"] == 0 and p["cascade"]["phases"] == {"min": 687, "max": 687, "n": 6}
    d13 = r["decision_13_successor"]["filled"]
    assert d13["ten_thousand/1337/identity/E1"]["N"] == 2 * src["aggregates"]["pairs_E1"]["contested"] and d13["ten_thousand/1337/identity/E1"]["n"] == 2 * src["aggregates"]["pairs_E1"]["mark_ranks_duplicate_first"]
    assert r["c4"]["withdrawal_under_cascade"]["holds_on_every_journal_and_phase"] and r["c4"]["withdrawal_under_cascade"]["runs_read"] == 12
    assert "ten_thousand" in r["c4"]["never_planted_above_threshold"] and len(r["c4"]["never_planted_above_threshold"]["ten_thousand"]) == 12
    assert r["checks"]["identity_cascades_at_the_expected_phase_count"] and r["checks"]["threshold_within_1e-6_on_every_c2_file"]
    assert any(l.startswith("C.4 (1) withdrawal under cascade: holds") for l in r["summary_lines"])


@needs_files
def test_a_fault_in_one_cascade_fails_the_first_condition_and_is_attributed(tmp_path):
    md = tmp_path / "mqar"; md.mkdir()
    _renamed(SOURCE["cascade-identity"], md, S2024, 50000, "identity")

    def fault(d):
        d["faults"] = {"phases": [41], "n": 1, "condition": "FAULT: attributed before anything else (C.4)"}
    _renamed(SOURCE["cascade-identity"], md, S2024, 50001, "identity", mutate=fault)
    r = SR.read_rows(md)
    w = r["c4"]["withdrawal_under_cascade"]
    assert not w["holds_on_every_journal_and_phase"] and w["faults"] == [{"draw": 50001, "gate": "identity", "run_id": S2024, "phases": [41]}] and w["runs_read"] == 2
    assert r["pooled"]["ten_thousand/identity"]["cascade"]["faults_total"] == 1
    assert any("FAULT" in l for l in r["summary_lines"])


def _progress_file(path: Path, phases: int, fault_at: int | None = None) -> None:
    lines = [{"kind": "identity", "compared": {"kind": "scale probe cascade", "draw": 50000, "gate": "identity", "run_id": S2024}},
             {"kind": "session_start", "wall_clock": "2026-10-30T13:25:14+00:00", "command": "x", "resumed_from": None, "code_state": {"commit": "dfe7145"}}]
    for ph in range(1, phases + 1):
        decisions = [{"j": 5, "kind": "fact", "admitted": True, "pointer": "p", "entry_id": "e5"}, {"j": 6, "kind": "E2", "admitted": True, "pointer": "d", "entry_id": "e6"}] if ph == 1 else []
        lines.append({"kind": "unit", "index": ph, "phase": ph, "next_j": 16 * ph, "decisions": decisions, "phase_event": {"applied": {"consolidated": 8, "evicted": ph, "withdrawn": ph, "demoted_groups": 0, "replay_failed": 0}},
                      "check": {"fault": (ph == fault_at), "withdrawn_present": [], "kept_absent": []}, "lifecycle_seconds": 0.05 * ph, "wall_clock": "t",
                      "entries": 20 + 16 * ph, "states": {"live": 16 * ph, "consolidated": 8, "demoted": 0, "evicted": ph}, "summaries": 0, "memory_patterns": 8,
                      "evicted_near_duplicates": ["e6"] if ph == 4 else [], "evicted_near_duplicates_detail": [{"entry_id": "e6", "j": 6, "kind": "E2"}] if ph == 4 else []})
        if ph in (1, 2, 4):
            lines.append({"kind": "read", "phase": ph, "what": "sample", "j": 3, "own_in_window": True, "holes": 1, "candidates": 5, "pointers": ["p"], "marks": [0.9], "max_mark_at_or_above_threshold": True, "own_state": "live"})
            # a pair read's line as the probe writes it: the pair's kind over the line kind `read`
            rec = {"what": "pair", "j": 5, "own_in_window": ph != 4, "holes": 0, "candidates": 5, "pointers": ["p", "d"] if ph != 4 else ["p"], "marks": [0.9], "max_mark_at_or_above_threshold": True,
                   "own_state": "evicted" if ph == 4 else "live", "kind": "E2", "partner_j": 6, "duplicate_in_window": ph != 4, "duplicate_state": "evicted" if ph == 4 else "live"}
            lines.append({"kind": "read", "phase": ph, **rec})           # as `cascade_reads` writes it: the pair's kind lands over the line kind
    lines.append({"kind": "session_end", "wall_clock": "t", "reason": "complete", "first_unit": 1, "last_unit": phases, "resumption_point": None})
    path.write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")


def test_a_progress_file_is_verified_by_the_final_files_hash_and_read_apart(tmp_path):
    p = tmp_path / "cascade-50000-4cfd6f6e1406-identity.progress.jsonl"
    _progress_file(p, phases=5)
    final = {"progress_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
             "near_duplicates": {"rows": [{"j": 6, "kind": "E2", "original_j": 5, "state": "evicted", "evicted_at_phase": 4, "read_before_eviction": False, "grouped_with_partner": None}]}}
    r = SR.read_progress(p, final)
    assert r["equal_to_the_final_files"] and r["phases_recorded"] == 5 and r["faults"] == [] and r["read_faults"] == [] and r["n_read_lines"] == 6
    assert r["n_sample_lines"] == 3 and r["n_pair_lines"] == 3                      # the pair lines read by their `what`, whatever their `kind` field carries
    assert SP.read_lines_of(__import__("cortex_c2b.resumable", fromlist=["Progress"]).Progress(p))[0].__len__() == 3   # the probe's own reader sees the sample lines alone
    rb = r["near_duplicates_read_before_eviction"]["E2"]
    assert rb == {"rows": 1, "read_before_eviction_sample_and_pair_reads": 1, "read_before_eviction_in_the_final_file_sample_reads_only": 0, "evicted": 1,
                  "read_live_at_the_end": 0, "read_then_evicted": 1, "read_before_eviction_with_the_last_phase": 1, "read_live_at_the_end_with_the_last_phase": 0, "read_then_evicted_with_the_last_phase": 1}
    assert r["reads_by_phase"]["4"]["sample"] == {"n": 1, "own_in_window": 1, "holes": 1, "own_state": {"live": 1}}
    e2 = r["reads_by_phase"]["4"]["pairs"]["E2"]
    assert e2["n"] == 1 and e2["own_in_window"] == 0 and e2["duplicate_in_window"] == 0 and e2["own_state"] == {"evicted": 1} and e2["own_not_evicted"] == 0
    assert r["reads_by_phase"]["2"]["pairs"]["E2"]["duplicate_in_window"] == 1 and r["reads_by_phase"]["2"]["pairs"]["E2"]["both_in_window"] == 1
    assert r["reads_by_phase"]["2"]["pairs"]["E2"]["own_in_window_given_own_not_evicted"] == 1 and r["reads_by_phase"]["2"]["pairs"]["E1"]["n"] == 0
    assert r["per_phase"][3]["states"]["evicted"] == 3 and r["per_phase"][3]["applied"]["withdrawn"] == 3 and not r["per_phase"][3]["fault"]
    assert r["sessions"][0]["reason"] == "complete"
    # another file under the same name: the hash differs, the reading says so
    q = tmp_path / "other" / "cascade-50000-4cfd6f6e1406-identity.progress.jsonl"; q.parent.mkdir()
    _progress_file(q, phases=5, fault_at=2)
    r2 = SR.read_progress(q, final)
    assert not r2["equal_to_the_final_files"] and r2["faults"] == [2]


@needs_files
def test_read_rows_finds_the_progress_files_under_the_run_folders_or_as_flat_copies(tmp_path):
    md = tmp_path / "mqar"; md.mkdir()
    final_path = _renamed(SOURCE["cascade-identity"], md, S2024, 50000, "identity")
    flat = tmp_path / "flat"; flat.mkdir()
    p = flat / "cascade-50000-4cfd6f6e1406-identity.progress.jsonl"
    _progress_file(p, phases=3)
    r = SR.read_rows(md, progress_dir=flat)
    pr = r["progress"]["scale-cascade-50000-4cfd6f6e1406-identity.json"]
    assert pr["phases_recorded"] == 3 and pr["equal_to_the_final_files"] is False     # the committed file's hash is its own progress file's, not this constructed one's
    assert r["checks"]["progress_files_read"] == 1 and r["checks"]["progress_files_equal_to_the_final_files_hashes"] is False
    assert pr["read_from"] == "a copy of the file; the founder's file untouched" and pr["torn_tail"] is None and pr["last_session_closed"]
    runs = tmp_path / "runs"; (runs / "scale-50000-4cfd6f6e1406" / "c3-identity").mkdir(parents=True)
    shutil.copy(p, runs / "scale-50000-4cfd6f6e1406" / "c3-identity" / "cascade.progress.jsonl")
    r2 = SR.read_rows(md, journals=runs)
    assert r2["checks"]["progress_files_read"] == 1
    # an unfinished cascade (no final file) with a progress file: read for the fault condition only, listed as unfinished
    final_path.unlink()
    r3 = SR.read_rows(md, journals=runs)
    w = r3["c4"]["withdrawal_under_cascade"]
    assert w["runs_read"] == 1 and w["unfinished_or_refused_runs"] == [str(runs / "scale-50000-4cfd6f6e1406" / "c3-identity" / "cascade.progress.jsonl")] and w["holds_on_every_journal_and_phase"]
    assert r3["checks"]["unfinished_or_refused_progress_files_read"] == 1 and "1 of them unfinished or refused" in r3["summary_lines"][1]
    # the same file named explicitly, and a fault in it attributed with the founder's path
    q = tmp_path / "dead.progress.jsonl"; _progress_file(q, phases=4, fault_at=3)
    r4 = SR.read_rows(md, progress_files=[q])
    w4 = r4["c4"]["withdrawal_under_cascade"]
    assert not w4["holds_on_every_journal_and_phase"] and w4["faults"][0]["phases"] == [3] and w4["faults"][0]["from"] == str(q) and w4["unfinished_or_refused_runs"] == [str(q)]


@needs_files
def test_the_cli_writes_the_reading_once(tmp_path):
    md = tmp_path / "mqar"; md.mkdir()
    for kind, src in SOURCE.items():
        _renamed(src, md, S2024, 50000, kind.split("-")[1] if "-" in kind else None)
    SR._cli(["--metrics", str(md)])
    out = md / SR.READING_NAME
    assert out.exists()
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["rows"] == "row 53" and r["files"]["present"] == 5 and r["timestamp_utc"]
    with pytest.raises(SystemExit):
        SR._cli(["--metrics", str(md)])


@needs_files
def test_the_committed_files_read_as_they_are():
    r = SR.read_rows(M)
    assert r["files"]["present"] >= 25 and r["files"]["outside_the_expected_set"] == []
    for n, c in r["c2"].items():
        assert c["threshold"]["within_1e-6"] and c["freeze_bound"] and str(c["pythonhashseed"]) == "0" and n.startswith(f"scale-c2-{c['draw']}-{c['run_id']}-{c['gate']}")
    for n, c in r["cascades"].items():
        assert c["faults"]["n"] == 0 and c["freeze_bound"] and c["resumption_fault"] == [] and n.startswith(f"scale-cascade-{c['draw']}-{c['run_id']}-{c['gate']}")
        if c["gate"] == "identity":
            assert c["phases"] == SR.EXPECTED_PHASES[c["size"]]
    assert r["c4"]["withdrawal_under_cascade"]["holds_on_every_journal_and_phase"]


def test_a_torn_last_line_is_reported_and_the_founders_file_is_left_untouched(tmp_path):
    p = tmp_path / "cascade-50000-4cfd6f6e1406-identity.progress.jsonl"
    _progress_file(p, phases=3)
    p.write_bytes(p.read_bytes() + b'{"kind": "unit", "index": 4, "phase": 4, "decisions": [')     # a death between the write of a line and its sync
    before = p.read_bytes()
    r = SR.read_progress(p, {"progress_sha256": "x"})
    assert p.read_bytes() == before and not list(tmp_path.glob("*.torn-*"))                        # read through a copy: nothing repaired in place
    assert r["torn_tail"] == {"bytes": 55, "sha256": hashlib.sha256(b'{"kind": "unit", "index": 4, "phase": 4, "decisions": [').hexdigest()} and r["phases_recorded"] == 3 and r["last_session_closed"]
    assert not r["equal_to_the_final_files"]


@needs_files
def test_an_extra_final_file_is_read_apart_and_never_pooled(tmp_path):
    md = tmp_path / "mqar"; md.mkdir()
    _renamed(SOURCE["cascade-identity"], md, S2024, 50000, "identity")
    _renamed(SOURCE["c2-identity"], md, S2024, 50000, "identity")
    extra = tmp_path / "scale-cascade-50000-4cfd6f6e1406-identity-second-run.json"
    d = json.loads(SOURCE["cascade-identity"].read_text(encoding="utf-8")); d["faults"] = {"phases": [9], "n": 1, "condition": "FAULT"}
    extra.write_text(json.dumps(d), encoding="utf-8")
    r = SR.read_rows(md, also=[extra])
    assert list(r["extra"]) == [extra.name] and r["extra"][extra.name]["kind"] == "cascade" and r["extra"][extra.name]["digest"]["faults"]["n"] == 1
    assert r["pooled"]["ten_thousand/identity"]["cascade_files"] == 1 and r["pooled"]["ten_thousand/identity"]["cascade"]["faults_total"] == 0   # the record's file alone is pooled
    w = r["c4"]["withdrawal_under_cascade"]
    assert not w["holds_on_every_journal_and_phase"] and w["runs_read"] == 2 and w["faults"][0]["phases"] == [9]   # the extra run read for the fault condition
    assert r["checks"]["extra_final_files_read"] == [extra.name] and r["checks"]["sessions_per_file"] == {} and r["checks"]["ten_thousand_runs_in_one_session"]
    (tmp_path / "x.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        SR.read_rows(md, also=[tmp_path / "x.json"])                                                # not a final file of the probe: refused
