"""The scale probe of ADR-010 part C (the amendment of 2026-10-15 on parts B and C, graved 2026-10-29) and the resumption
tests of its section 4 (f): a new module on the frozen system, tested on a test plan (never a draw of the seal) and a tiny
model whose encoder is not used (the cues are engineered files, as 4 (c) lets a run load them). Every resumption runs in
a new process from the disk alone, PYTHONHASHSEED fixed in the command; the cascade fixture sealed (here) and plaintext
(the slow battery)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import train
from cortex_c2b import Journal, STATE_EVICTED, STATE_CONSOLIDATED, STATE_DEMOTED, record_tools, POLICY_STOP, ADMISSION_IDENTITY
from cortex_c2b.crypto import open_line
from cortex_c2b.read_path import JournalPath, ADOPTED_GEOMETRY
from cortex_c2b.write_path import WritePath, DentateGyrus
from cortex_c2b.lifecycle import LifecycleScheduler, LifecycleConfig, PersistentMemory
from cortex_c2b.hm_protocol import generate_facts
from cortex_c2b import sealed_families as S
from cortex_c2b import scale_probe as SP
from cortex_c2b import resumable as RS

REPO = Path(record_tools.REPO_ROOT)
SEED = 777
N_FACTS_FIXTURE = 910                          # 910 facts, 46 E1 and 45 E2 partners: 1 001 entries, 62 phases of 16 writes
CLUSTER = (80, 81, 82, 83)                     # one period's four members: pairwise raw cosine >= 0.9, dentate codes apart
RUN_ID = "tinyscale"


# ---------------------------------------------------------------------------- #
# the fixture: a tiny checkpoint, a test plan, engineered cues                    #
# ---------------------------------------------------------------------------- #
def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def tiny_checkpoint(out: Path):
    cfg = train.Config(n_layer=3, n_head=2, n_embd=32, block_size=96, journal="read", journal_block=1, journal_heads=2,
                       journal_read_bytes=96, journal_cue_dim=64, journal_k=3)
    torch.manual_seed(2)
    m = train.VanillaGPT(cfg); m.eval()
    torch.save({"model": m.state_dict(), "run_id": RUN_ID, "config_hash": train.config_hash(cfg), "step": 0}, out / "ckpt.pt")
    (out / "cfg.json").write_text(json.dumps({"n_layer": 3, "n_head": 2, "n_embd": 32, "block_size": 96, "journal": "read", "journal_block": 1,
                                             "journal_heads": 2, "journal_read_bytes": 96, "journal_cue_dim": 64, "journal_k": 3}), encoding="utf-8")
    torch.manual_seed(3)
    m2 = train.VanillaGPT(cfg)
    torch.save({"model": m2.state_dict(), "run_id": RUN_ID, "config_hash": train.config_hash(cfg), "step": 0}, out / "ckpt-other.pt")
    return cfg


def cluster_members(rng, dg):
    """Four cues with pairwise raw cosine at or above 0.91 whose dentate codes are apart (the second to fourth members'
    surprise above the tribunal's floor), so that the frozen scheduler consolidates and then demotes them."""
    best = None
    for _ in range(3000):
        base = unit(rng.standard_normal(64))
        ms = [base]
        for _ in range(3):
            d = rng.standard_normal(64).astype(np.float32); d -= (d @ base) * base; d = unit(d)
            t = rng.uniform(0.25, 0.40)
            ms.append(unit(np.cos(t) * base + np.sin(t) * d))
        raw = min(float(a @ b) for i, a in enumerate(ms) for b in ms[i + 1:])
        codes = [dg.separate(m) for m in ms]
        cc = max(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)) for i, a in enumerate(codes) for b in codes[i + 1:])
        if raw >= 0.91 and (best is None or cc < best[0]):
            best = (cc, raw, ms)
    return best


def engineered_cues(plan: dict, seed: int = 11):
    """Per period of sixteen writes three random cues (consolidated, decaying, evicted), the rest near copies of earlier
    random entries (surprise about zero: refused under the surprise rule, evicted by the tribunal under identity), and the
    cluster in period six; the question cues equal the statement cues of the facts."""
    planned = plan["planned"]; n = len(planned)
    rng = np.random.default_rng(seed)
    dg = DentateGyrus(seed=0)
    cues = np.zeros((n, 64), dtype=np.float32)
    randoms = []
    _, _, members = cluster_members(rng, dg)
    for j in range(n):
        if j in CLUSTER:
            cues[j] = members[CLUSTER.index(j)]
        elif j % 16 in (0, 5, 10) or not randoms:
            cues[j] = unit(rng.standard_normal(64)); randoms.append(j)
        else:
            src = randoms[(j * 7) % len(randoms)]
            cues[j] = unit(cues[src] + 0.004 * rng.standard_normal(64).astype(np.float32))
    qs = SP.questions_of_plan(plan)
    qcues = np.zeros((len(qs), 64), dtype=np.float32)
    r2 = np.random.default_rng(99)
    for q in qs:
        qcues[q["q"]] = cues[q["j"]] if q["kind"] == "fact" else unit(r2.standard_normal(64))
    return cues, qcues


def write_cue_files(rdir: Path, cues, qcues, batch: int = 64):
    cdir = rdir / "cues"; cdir.mkdir(parents=True, exist_ok=True)
    for name, arr in (("statements", cues), ("questions", qcues)):
        for bi in range((arr.shape[0] + batch - 1) // batch):
            with open(cdir / f"{name}-b{bi}.npy", "wb") as fh:
                np.save(fh, arr[bi * batch:(bi + 1) * batch])


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    root = tmp_path_factory.mktemp("scale")
    tiny_checkpoint(root)
    plan = SP.plan_of(SEED, test_n=N_FACTS_FIXTURE)
    cues, qcues = engineered_cues(plan)
    return {"root": root, "plan": plan, "cues": cues, "qcues": qcues}


def run_dir_of(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(exist_ok=True)
    return d


def prepare(fx, name: str) -> Path:
    """A run folder of its own with the cue files (the encodings step done, 4 (c))."""
    base = run_dir_of(fx["root"], name)
    rdir = base / "runs" / f"scale-{SEED}-{RUN_ID}"
    if not (rdir / "cues").exists():
        write_cue_files(rdir, fx["cues"], fx["qcues"])
    return base


def argv_for(fx, base: Path, step: str, gate: str = "identity", extra=()):
    return ["--draw", str(SEED), "--test-plan-n", str(N_FACTS_FIXTURE), "--ckpt", str(fx["root"] / "ckpt.pt"), "--config", str(fx["root"] / "cfg.json"),
            "--journals", str(base / "runs"), "--out", str(base / "out"), "--step", step, "--gate", gate, *extra]


def run_probe(argv, patch: str = "", env_extra: dict | None = None, hashseed: str = "0") -> subprocess.CompletedProcess:
    """The probe in a new process from the disk alone, PYTHONHASHSEED fixed in the command, `record_tools.git_state` pinned;
    `patch` is Python executed before the command (the tests' kills and stops, on the frozen objects' public methods)."""
    code = ("import os, sys\n"
            "from cortex_c2b import record_tools\n"
            "record_tools.git_state = lambda: ('test0000', True)\n"
            "from cortex_c2b import scale_probe as SP\n"
            f"{patch}\n"
            f"SP._cli({argv!r})\n")
    env = {**os.environ, "PYTHONHASHSEED": hashseed, "ADR010_LAUNCHER": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", **(env_extra or {})}
    return subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")


def ok(proc):
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc


def opened_lines(jdir: Path) -> list[str]:
    key = SP.key_of(jdir)
    raw = (jdir / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    return [open_line(key, l) if key is not None else l for l in raw if l.strip()]


def normalised(d):
    """A final file without its session table and what varies with it (4 (a): equal byte for byte but for the session table)."""
    drop = {"sessions", "run_in_sessions", "timestamp_utc", "progress_sha256", "progress_file", "restored_index_at_last_reopening", "phase_times_seconds",
            "probe_times_seconds", "wall_clock", "phase_seconds", "lifecycle_seconds", "probe_seconds", "reopening_seconds", "recorded", "sha256", "code_commit",
            "read_lines_recorded_twice_compared", "journal", "identity", "threads", "timer_overhead_ns"}
    if isinstance(d, dict):
        return {k: normalised(v) for k, v in d.items() if k not in drop}
    if isinstance(d, list):
        return [normalised(x) for x in d]
    return d


def units_of(base: Path, gate: str = "identity") -> dict:
    return RS.Progress(base / "runs" / f"scale-{SEED}-{RUN_ID}" / f"c3-{gate}" / "cascade.progress.jsonl").units()


def jdir_of(base: Path, gate: str = "identity", step: str = "c3") -> Path:
    return base / "runs" / f"scale-{SEED}-{RUN_ID}" / f"{step}-{gate}"


def final_of(base: Path, gate: str = "identity") -> dict:
    return json.loads((base / "out" / f"scale-cascade-{SEED}-{RUN_ID}-{gate}.json").read_text(encoding="utf-8"))


def assert_same_run(whole: Path, other: Path, gate: str = "identity"):
    """The opened logs equal line for line, the lengths equal at every recorded point, the final files equal but for the
    session table (4 (a), 4 (f))."""
    assert opened_lines(jdir_of(whole, gate)) == opened_lines(jdir_of(other, gate))
    uw, uo = units_of(whole, gate), units_of(other, gate)
    assert sorted(uw) == sorted(uo)
    for i in uw:
        assert uw[i]["log"]["length"] == uo[i]["log"]["length"], f"phase {i}"
        assert uw[i]["archive_length"] == uo[i]["archive_length"], f"phase {i}"
        assert uw[i]["index_state"] == uo[i]["index_state"], f"phase {i}"
    fw, fo = final_of(whole, gate), final_of(other, gate)
    assert normalised(fw) == normalised(fo)
    cw, co = fw["identity"]["compared"], fo["identity"]["compared"]          # the identity: the same run but for its own journal's key and stop file
    for k in ("draw", "size", "gate", "admission", "lifecycle_config_hash", "checkpoint_sha256", "plan_content_sha256", "environment", "frozen_tree_sha256", "seal_sha256"):
        assert cw.get(k) == co.get(k), k
    assert cw["log_after_open"]["length"] == co["log_after_open"]["length"] and cw["index_state_after_open"] == co["index_state_after_open"]
    assert fo["resumption_fault"] == []


# ---------------------------------------------------------------------------- #
# the plan and the probe's small pieces                                          #
# ---------------------------------------------------------------------------- #
def test_plan_of_a_test_draw_is_the_seal_generator_in_written_order(fx):
    plan = fx["plan"]
    assert plan["size"] == "test" and plan["n_facts"] == N_FACTS_FIXTURE and plan["n_planned"] == 1001
    assert plan["partners"] == {"E1": 46, "E2": 45} and plan["expected_phases_identity"] == 62
    kinds = [x["kind"] for x in plan["planned"]]
    assert kinds[0] == "fact" and kinds[1] == "E1"                       # the partner right after its fact (i = 0: E1)
    j_e2 = next(i for i, x in enumerate(plan["planned"]) if x["kind"] == "E2")
    assert plan["planned"][j_e2 - 1]["kind"] == "fact" and plan["planned"][j_e2 - 1]["position"] == 10
    texts = [x["statement"] for x in plan["planned"]]
    assert len(set(texts)) == len(texts)                                 # every text distinct (C.1 (a))
    qs = SP.questions_of_plan(plan)
    assert len(qs) == N_FACTS_FIXTURE + S.SCALE_NEVER_PLANTED_N
    L = SP.indexes_of_plan(plan)
    assert len(L["pairs"]) == 91 and len(L["sample_js"]) == N_FACTS_FIXTURE   # the sample is every position at this size
    with pytest.raises(ValueError):
        SP.plan_of(12345)                                                # not a draw of the seal


def test_the_seal_draw_regenerates_to_its_content_hash():
    plan = SP.plan_of(50000)                                             # the first ten thousand draw, checked against the seal
    seal = json.loads((REPO / "metrics/mqar/SEALED-FAMILIES.json").read_text(encoding="utf-8"))
    assert plan["content_sha256"] == seal["scale_draws"]["50000"]["content_sha256"] and plan["seal_checked"]
    assert plan["n_planned"] == 11_000 and plan["expected_phases_identity"] == 687


def test_sentinel_names_share_nothing_with_the_first_scale_draw():
    """C.1 (a): the sentinels of seeds 0 and 2 and their negatives (seed + 10 000) share no name with the draw's journal or its
    never planted control (the first ten thousand draw here; the slow test covers every draw)."""
    plan = SP.plan_of(50000)
    names = {x["entity"] for x in plan["planned"]} | {v["entity"] for v in plan["never"]}
    for seed in (0, 2):
        for s in (seed, seed + 10_000):
            facts, _ = generate_facts(20 if s == seed else 10, s)
            assert not ({f.entity for f in facts} & names), (seed, s)


@pytest.mark.slow
def test_sentinel_names_share_nothing_with_any_scale_draw():
    sentinel = set()
    for seed in (0, 2):
        for s, n in ((seed, 20), (seed + 10_000, 10)):
            sentinel |= {f.entity for f in generate_facts(n, s)[0]}
    for spec in S.SCALE_DRAWS.values():
        for seed in spec["seeds"]:
            plan = SP.plan_of(seed)
            names = {x["entity"] for x in plan["planned"]} | {v["entity"] for v in plan["never"]}
            assert not (sentinel & names), seed


def test_withdrawal_check_sees_a_withdrawn_identity_present_and_a_kept_one_absent():
    j = Journal(policy=POLICY_STOP)
    wp = WritePath(j, seed=0)
    rng = np.random.default_rng(3)
    ids = []
    for i in range(3):
        rep = wp.write(unit(rng.standard_normal(64)), f"p{i}".encode(), "s", now=float(i), source=f"t{i}")
        ids.append(rep.entry.entry_id)
    for eid in ids[:2]:
        j.transition(eid, STATE_CONSOLIDATED)
    mem = PersistentMemory(dim=64)
    for eid in ids[:2]:
        mem.replay(j._entries[eid].cue); mem.label_last(eid)
    assert not SP.withdrawal_check(j, mem)["fault"]
    j.transition(ids[0], STATE_EVICTED)                                   # withdrawn by the frozen code's rule, the memory not told
    c = SP.withdrawal_check(j, mem)
    assert c["fault"] and c["n_withdrawn_present"] == 1
    mem.withdraw(ids[0]); mem.withdraw(ids[1])
    c = SP.withdrawal_check(j, mem)
    assert c["fault"] and c["n_kept_absent"] == 1


def test_cascade_config_is_the_frozen_one_under_the_adopted_geometry():
    cfg = SP.cascade_config()
    assert (cfg.index_tables, cfg.index_signature, cfg.sentinel_seed, cfg.withdraw_at_eviction) == (16, "exact", 2, True)
    assert cfg.period_writes == 16 and cfg.config_hash() != LifecycleConfig().config_hash()


def test_d6_b_restores_the_live_index_exactly(tmp_path):
    """4 (d), D6 B: candidates(cue) equal on every entry's cue, the touch window and the counter equal, before a stop and
    after a reopening of the same journal from the disk."""
    rng = np.random.default_rng(5)
    jpath = tmp_path / "j.jsonl"
    j = Journal(jpath, plaintext=True, policy=POLICY_STOP, index_geometry=ADOPTED_GEOMETRY)
    wp = WritePath(j, seed=0, admission=ADMISSION_IDENTITY); jp = JournalPath(j, geometry=ADOPTED_GEOMETRY)
    cfg = LifecycleConfig(period_writes=8, demote_k=3, demote_min_cosine=0.85, index_tables=16, index_signature="exact", sentinel_seed=2)
    sch = LifecycleScheduler(j, wp, jp, cfg, now=0.0)
    base = unit(rng.standard_normal(64))
    for i in range(80):
        c = unit(base + 0.004 * rng.standard_normal(64).astype(np.float32)) if i % 10 < 3 else unit(rng.standard_normal(64))
        sch.write(c, f"p{i}".encode(), "s", now=float(i), source=f"text {i}")
    for i in range(30):
        jp.retrieve(unit(rng.standard_normal(64)), k=4)
    live = jp.index
    assert any(e.state == STATE_EVICTED for e in j._entries.values())
    state = RS.index_state(jp)
    j2 = Journal(jpath, plaintext=True, policy=POLICY_STOP, index_geometry=ADOPTED_GEOMETRY)
    jp2 = JournalPath(j2, geometry=ADOPTED_GEOMETRY)
    before = {eid: jp2.index.candidates(np.asarray(e.cue, dtype=np.float32)) for eid, e in j2._entries.items()}
    rec = RS.restore_live_index(jp2, j2, state["touch_window"], state["touched"])
    assert rec["evicted_entries_in_index"] > 0 and rec["absent_from_frozen_open"] > 0 and jp2.index is not jp.index
    for eid, e in j2._entries.items():
        cue = np.asarray(e.cue, dtype=np.float32)
        assert jp2.index.candidates(cue) == live.candidates(cue), eid
    assert any(before[eid] != jp2.index.candidates(np.asarray(e.cue, dtype=np.float32)) for eid, e in j2._entries.items())   # the frozen open drops the evicted
    assert list(jp2.index.touch_log) == list(live.touch_log) and jp2.index.touched == live.touched
    assert jp2.index.alias == live.alias and jp2.index.geometry() == live.geometry()


def test_progress_file_cuts_a_torn_last_line(tmp_path):
    p = RS.Progress(tmp_path / "p.jsonl")
    p.write_identity({"compared": {"a": 1}, "recorded": {}})
    p.append({"kind": "unit", "index": 0, "x": 1}); p.close()
    with open(p.path, "ab") as fh:
        fh.write(b'{"kind": "unit", "index": 1, "x"')
    q = RS.Progress(tmp_path / "p.jsonl")
    assert q.torn_tail and q.torn_tail["bytes"] > 0 and (tmp_path / "p.jsonl.torn-1").exists()
    assert [l["kind"] for l in q.lines()] == ["identity", "unit"] and q.units()[0]["x"] == 1


def test_payload_archive_tolerates_a_torn_tail_until_cut(tmp_path):
    a = RS.PayloadArchive(tmp_path / "x.archive")
    a.append("a" * 16, b"hello"); a.append("b" * 16, b"world"); a.close()
    good = a.length
    with open(a.path, "ab") as fh:
        fh.write(RS.ARCHIVE_MAGIC + b"c" * 16 + b"\x00" * 7 + b"\x09" + b"part")
    b = RS.PayloadArchive(tmp_path / "x.archive")
    assert b.torn_tail_bytes > 0 and b.valid_length == good and "a" * 16 in b and "c" * 16 not in b
    with pytest.raises(RS.ResumptionRefused):
        b.append("d" * 16, b"x")
    with pytest.raises(RS.ResumptionRefused):
        b.cut(good + 5, tmp_path / "side")
    rec = b.cut(good, tmp_path / "side")
    assert rec["archive_tail_bytes"] > 0 and b.torn_tail_bytes == 0 and b.get("b" * 16) == b"world"


# ---------------------------------------------------------------------------- #
# 4 (f) (ii): the cascade battery, sealed                                        #
# ---------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def whole(fx):
    base = prepare(fx, "whole")
    ok(run_probe(argv_for(fx, base, "cascade")))
    return base


def test_cascade_whole_run_holds_a_demotion_an_evicted_summary_and_read_phases(whole):
    f = final_of(whole)
    assert f["phases"] == 62 and f["run_in_sessions"] == 1 and f["faults"]["n"] == 0
    assert f["applied_totals"]["demoted_groups"] >= 1                     # a demotion group happened
    assert f["summaries"]["written"] >= 1 and f["summaries"]["evicted"] >= 1
    assert f["summaries"]["sources_of_evicted_summaries_evicted"] == f["summaries"]["sources_of_evicted_summaries"] >= 4   # with its sources
    assert set(f["reads_by_phase"]) >= {"1", "2", "4", "8", "16", "32", "last"} and f["n_read_lines"] > 0
    assert f["reads_by_phase"]["last"]["pairs_read"] == 91 and f["k"] == 3                 # every planted pair's original after the last write; the system's k
    assert f["applied_totals"]["withdrawn"] > 0 and f["states"]["evicted"] > 0
    units = units_of(whole)
    assert len(units) == 62 and all(u["check"]["fault"] is False for u in units.values())
    assert sum(1 for u in units.values() if (u["phase_event"].get("applied") or {}).get("demoted_groups")) >= 1


def demotion_phase(base: Path) -> int:
    return next(i for i, u in sorted(units_of(base).items()) if (u["phase_event"].get("applied") or {}).get("demoted_groups"))


STOP_AT = """
from cortex_c2b import lifecycle as _L
_orig = _L.LifecycleScheduler.phase
def _phase(self, now):
    rec = _orig(self, now)
    if self.phases == {K}:
        open({STOP!r}, 'w').close()
    return rec
_L.LifecycleScheduler.phase = _phase
"""

KILL_BEFORE_PHASE_EVENT = """
from cortex_c2b import lifecycle as _L
_orig = _L.LifecycleScheduler.phase
def _phase(self, now):
    if self.phases + 1 == {K}:
        os._exit(3)                       # the sixteenth write of the period is on the disk; its phase event is not
    return _orig(self, now)
_L.LifecycleScheduler.phase = _phase
"""

KILL_INSIDE_PHASE = """
from cortex_c2b import Journal as _J, STATE_EVICTED as _E, record_tools as _rt
_orig = _J.transition
_count = {{"n": 0}}
def _transition(self, eid, state):
    r = _orig(self, eid, state)
    if state == _E and self.phase_count + 1 == {K} and self.summary_sources:
        _count["n"] += 1
        if _count["n"] == 2:              # after the demote line and after the transitions of an eviction, a .tmp left
            pdir = _rt.payload_dir_of(self._path)
            (pdir / "ffffffffffffffff.tmp").write_bytes(b"a write cut by the death")
            os._exit(3)
    return r
_J.transition = _transition
"""

KILL_AT_SUMMARY_EVICTION = """
from cortex_c2b import Journal as _J, STATE_EVICTED as _E
_orig = _J.transition
def _transition(self, eid, state):
    r = _orig(self, eid, state)
    if state == _E and eid in self.summary_sources:
        os._exit(3)                       # right after a summary's eviction: the store unlinked payload files the retained prefix holds
    return r
_J.transition = _transition
"""

KILL_INSIDE_READS = """
_orig = SP.window_of
_count = {{"n": 0}}
def _window_of(jp, q, k=4):
    r = _orig(jp, q, k)
    _count["n"] += 1
    if _count["n"] == {N}:
        os._exit(3)
    return r
SP.window_of = _window_of
"""


def test_cascade_stopped_at_a_phase_boundary_and_resumed_is_the_same_run(fx, whole):
    base = prepare(fx, "stop-boundary")
    stop = base / "stop"
    p = ok(run_probe(argv_for(fx, base, "cascade", extra=["--stop-file", str(stop)]), patch=STOP_AT.format(K=10, STOP=str(stop))))
    assert "stopped (stop_file) at phase 10" in p.stdout
    assert not list((base / "out").glob("scale-cascade-*")) if (base / "out").exists() else True
    assert max(units_of(base)) == 10
    stop.unlink()
    ok(run_probe(argv_for(fx, base, "cascade", extra=["--stop-file", str(stop)])))
    f = final_of(base)
    assert f["run_in_sessions"] == 2 and [s["reason"] for s in f["sessions"]] == ["stop_file", "complete"]
    assert f["sessions"][0]["resumption_point"] == 10 and f["sessions"][1]["resumed_from"] == 160
    assert f["sessions"][1]["reopening"]["action"] == "plain reopening" and f["sessions"][1]["cuts"] == []
    assert f["sessions"][1]["code_state"]["new_modules_sha256_lf"]["cortex_c2b/scale_probe.py"] and f["sessions"][1]["code_change"] is None
    assert f["read_phases"][-1] == "last" and f["reads_by_phase"]["last"]["n"] > 0 and f["reconstruction_by_phase"]["last"]["reconstructions"] is not None
    assert len(f["phase_times_seconds"]) == 62 == len(f["probe_times_seconds"]) and f["phase_times_label"].startswith("the time is the lifecycle's")
    assert_same_run(whole, base)


def test_cascade_stopped_at_a_read_phase_and_resumed_is_the_same_run(fx, whole):
    base = prepare(fx, "stop-read")
    stop = base / "stop"
    ok(run_probe(argv_for(fx, base, "cascade", extra=["--stop-file", str(stop)]), patch=STOP_AT.format(K=16, STOP=str(stop))))
    assert max(units_of(base)) == 16 and "reads" in units_of(base)[16]
    stop.unlink()
    ok(run_probe(argv_for(fx, base, "cascade", extra=["--stop-file", str(stop)])))
    assert final_of(base)["run_in_sessions"] == 2
    assert_same_run(whole, base)


def test_cascade_killed_after_the_sixteenth_write_before_the_phase_event_and_resumed(fx, whole):
    base = prepare(fx, "kill-before-event")
    p = run_probe(argv_for(fx, base, "cascade"), patch=KILL_BEFORE_PHASE_EVENT.format(K=20))
    assert p.returncode == 3 and max(units_of(base)) == 19
    jdir = jdir_of(base)
    before = (jdir / "journal.jsonl").stat().st_size
    assert before > units_of(base)[19]["log"]["length"]                 # the period's writes beyond the recorded point
    ok(run_probe(argv_for(fx, base, "cascade")))
    f = final_of(base)
    assert f["run_in_sessions"] == 2 and f["sessions"][0]["reason"] == "death" and f["sessions"][1]["cuts"][0]["action"] == "cut"
    assert f["sessions"][1]["cuts"][0]["lines_moved"] == 16 and Path(f["sessions"][1]["cuts"][0]["side_dir"]).exists()
    assert (Path(f["sessions"][1]["cuts"][0]["side_dir"]) / "log-tail.jsonl").exists()
    assert_same_run(whole, base)


def test_cascade_killed_inside_a_phase_after_a_demotion_and_an_eviction_and_resumed(fx, whole):
    K = demotion_phase(whole)
    base = prepare(fx, "kill-inside-phase")
    p = run_probe(argv_for(fx, base, "cascade"), patch=KILL_INSIDE_PHASE.format(K=K))
    assert p.returncode == 3, p.stderr[-2000:]
    assert max(units_of(base)) == K - 1
    jdir = jdir_of(base)
    pdir = record_tools.payload_dir_of(jdir / "journal.jsonl")
    assert (pdir / "ffffffffffffffff.tmp").exists()
    held = RS.held_pointers(jdir / "journal.jsonl", SP.key_of(jdir), False)
    present = {f.name for f in pdir.iterdir()}
    lines_dead = opened_lines(jdir)
    assert any('"ev":"demote"' in l for l in lines_dead)                  # the dead phase wrote a summary and its demote line
    assert lines_dead == opened_lines(jdir_of(whole))[: len(lines_dead)]  # a prefix of the uninterrupted run, cut inside the phase
    ok(run_probe(argv_for(fx, base, "cascade")))
    f = final_of(base)
    cut = f["sessions"][1]["cuts"][0]
    assert cut["action"] == "cut" and cut["lines_moved"] > 16 and "ffffffffffffffff.tmp" in cut["moved"]
    assert f["sessions"][1]["moved"] and f["sessions"][1]["restored"] is not None
    pdir_after = {x.name for x in pdir.iterdir()}
    assert pdir_after == RS.held_pointers(jdir / "journal.jsonl", SP.key_of(jdir), False) | set()  # exactly the held set at the end
    assert not any(x.name.endswith(".tmp") for x in pdir.iterdir())
    side = Path(cut["side_dir"])
    assert side.exists() and (side / "ffffffffffffffff.tmp").exists()      # moved, never deleted
    assert_same_run(whole, base)


def test_cascade_killed_at_a_summary_eviction_restores_the_deleted_payload_files(fx, whole):
    """The case the second committee found (section 8 (7)): a phase's evictions unlink payload files the retained prefix holds;
    the archive restores them before the frozen open."""
    base = prepare(fx, "kill-summary-eviction")
    p = run_probe(argv_for(fx, base, "cascade"), patch=KILL_AT_SUMMARY_EVICTION)
    assert p.returncode == 3, p.stderr[-2000:]
    ok(run_probe(argv_for(fx, base, "cascade")))
    f = final_of(base)
    cut = f["sessions"][1]["cuts"][0]
    assert cut["action"] == "cut" and len(cut["restored"]) > 0 and all(r["why"] == "missing" for r in cut["restored"])
    assert f["sessions"][1]["restored"] == cut["restored"]
    assert_same_run(whole, base)


def test_cascade_killed_inside_a_read_phase_and_resumed_re_reads_and_keeps_the_lines_once(fx, whole):
    base = prepare(fx, "kill-inside-reads")
    n_before_16 = sum(u["reads"]["n"] for i, u in units_of(whole).items() if i < 16 and "reads" in u)
    p = run_probe(argv_for(fx, base, "cascade"), patch=KILL_INSIDE_READS.format(N=n_before_16 + 3))
    assert p.returncode == 3 and max(units_of(base)) == 15
    prog = RS.Progress(jdir_of(base) / "cascade.progress.jsonl")
    partial = [l for l in prog.lines() if l.get("kind") == "read" and l.get("phase") == 16]
    assert len(partial) == 2                                                 # the third read died inside its lookup, its line never written
    ok(run_probe(argv_for(fx, base, "cascade")))
    f = final_of(base)
    assert f["run_in_sessions"] == 2 and f["read_lines_recorded_twice_compared"] == 2 and f["resumption_fault"] == []
    assert_same_run(whole, base)


# ---------------------------------------------------------------------------- #
# 4 (f) (iii): the planting without a scheduler under the surprise rule          #
# ---------------------------------------------------------------------------- #
KILL_AFTER_REFUSAL = """
from cortex_c2b import write_path as _W
_orig = _W.WritePath.write
_count = {{"n": 0}}
def _write(self, cue, payload, schema_id, now=None, source=None):
    rep = _orig(self, cue, payload, schema_id, now=now, source=source)
    if not rep.admitted:
        _count["n"] += 1
        if _count["n"] == {N}:
            os._exit(3)
    return rep
_W.WritePath.write = _write
"""


def test_planting_under_surprise_killed_after_a_refused_write_and_resumed(fx):
    whole = prepare(fx, "c2-whole")
    ok(run_probe(argv_for(fx, whole, "c2", gate="surprise")))
    pw = json.loads((jdir_of(whole, "surprise", "c2") / "planting.json").read_text(encoding="utf-8"))
    assert pw["refused"] > 0 and pw["admitted"] > 0 and pw["admission_record"] == "surprise"
    base = prepare(fx, "c2-kill")
    p = run_probe(argv_for(fx, base, "c2", gate="surprise"), patch=KILL_AFTER_REFUSAL.format(N=200))
    assert p.returncode == 3
    ok(run_probe(argv_for(fx, base, "c2", gate="surprise")))
    pb = json.loads((jdir_of(base, "surprise", "c2") / "planting.json").read_text(encoding="utf-8"))
    assert opened_lines(jdir_of(whole, "surprise", "c2")) == opened_lines(jdir_of(base, "surprise", "c2"))
    assert pb["run_in_sessions"] == 2 and pb["resumption_fault"] == [] and pb["decisions_sha256"] == pw["decisions_sha256"]
    assert normalised({k: v for k, v in pb.items() if k != "identity"}) == normalised({k: v for k, v in pw.items() if k != "identity"})
    c2w = json.loads((whole / "out" / f"scale-c2-{SEED}-{RUN_ID}-surprise.json").read_text(encoding="utf-8"))
    c2b = json.loads((base / "out" / f"scale-c2-{SEED}-{RUN_ID}-surprise.json").read_text(encoding="utf-8"))
    assert c2w["aggregates"] == c2b["aggregates"] and c2w["index_sizes"]["I0"] == c2b["index_sizes"]["I0"]


# ---------------------------------------------------------------------------- #
# 4 (f) (iv): the refusals                                                       #
# ---------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def stopped(fx):
    base = prepare(fx, "refusals")
    stop = base / "stop"
    ok(run_probe(resume_argv(fx, base), patch=STOP_AT.format(K=3, STOP=str(stop))))
    stop.unlink()
    return base


def resume_argv(fx, base: Path) -> list[str]:
    """The command that started the stopped run, stop file included: 4 (b) compares the probe's configuration, whose stop
    file's name is part, so a resumption is "the same command" and a refusal test changes one thing and nothing else."""
    return argv_for(fx, base, "cascade", extra=["--stop-file", str(base / "stop")])


def refused(proc) -> str:
    assert proc.returncode != 0
    return proc.stderr


def copy_of_stopped(stopped: Path, tmp_path: Path, name: str) -> Path:
    """A copy of the stopped run under another base; its command names the copy's own stop file and folders."""
    import shutil
    copy = tmp_path / name
    shutil.copytree(stopped, copy)
    return copy


def test_resumption_by_the_same_command_proceeds_and_another_stop_file_is_refused(fx, stopped, tmp_path):
    """The control of the refusal tests: on an untouched copy the same command resumes and completes; the stop file's
    name is part of the probe's configuration (4 (b)), so a resumption without it is refused on that hash alone."""
    same = copy_of_stopped(stopped, tmp_path, "same")
    err = refused(run_probe(argv_for(fx, same, "cascade")))
    assert "ResumptionRefused" in err and "probe_config_sha256" in err and "checkpoint_sha256" not in err
    ok(run_probe(resume_argv(fx, same)))
    f = final_of(same)
    assert f["run_in_sessions"] == 2 and [s["reason"] for s in f["sessions"]] == ["stop_file", "complete"]
    assert f["sessions"][0]["resumption_point"] == 3 and f["sessions"][1]["resumed_from"] == 48 and f["phases"] == 62 and f["faults"]["n"] == 0


def test_resumption_refuses_a_changed_lifecycle_configuration(fx, stopped):
    patch = "from cortex_c2b import lifecycle as _L\nSP.cascade_config = lambda: _L.LifecycleConfig(index_tables=16, index_signature='exact', sentinel_seed=2, withdraw_at_eviction=True, decay=0.8)\n"
    err = refused(run_probe(resume_argv(fx, stopped), patch=patch))
    assert "ResumptionRefused" in err and "lifecycle_config_hash" in err


def test_resumption_refuses_a_changed_checkpoint(fx, stopped):
    argv = resume_argv(fx, stopped)
    argv[argv.index("--ckpt") + 1] = str(fx["root"] / "ckpt-other.pt")
    err = refused(run_probe(argv))
    assert "ResumptionRefused" in err and "checkpoint_sha256" in err and "probe_config_sha256" not in err


def test_resumption_refuses_a_changed_seal(fx, stopped, tmp_path):
    other = tmp_path / "SEALED-FAMILIES.json"
    seal = json.loads((REPO / "metrics/mqar/SEALED-FAMILIES.json").read_text(encoding="utf-8"))
    seal["date"] = "another day"
    other.write_text(json.dumps(seal), encoding="utf-8")
    patch = f"from cortex_c2b import sealed_families as _S\nfrom pathlib import Path as _P\n_S.SEAL_PATH = _P({str(other)!r})\n"
    err = refused(run_probe(resume_argv(fx, stopped), patch=patch))
    assert "ResumptionRefused" in err and "seal_sha256" in err and "probe_config_sha256" not in err


def test_resumption_refuses_a_changed_numeric_environment(fx, stopped):
    err = refused(run_probe(resume_argv(fx, stopped), hashseed="1"))
    assert "ResumptionRefused" in err and "environment" in err and "probe_config_sha256" not in err
    err = refused(run_probe(resume_argv(fx, stopped), env_extra={"OMP_NUM_THREADS": "2"}))
    assert "ResumptionRefused" in err and "environment" in err


def test_resumption_refuses_a_shorter_log_and_a_log_of_another_hash(fx, stopped, tmp_path):
    """The cut rule of 4 (d) on the log itself, the identity being equal: a log shorter than the recorded point, and a log
    whose bytes before the point differ, are each refused by name (the committee's 4 (f) (iii)); the assertion reads the
    refusal's own sentence, not the traceback's paths."""
    for what in ("shorter", "other_hash"):
        copy = copy_of_stopped(stopped, tmp_path, what)
        jp = jdir_of(copy) / "journal.jsonl"
        data = jp.read_bytes()
        point = units_of(copy)[3]["log"]["length"]
        if what == "shorter":
            jp.write_bytes(data[: point - 10])
        else:
            i = point // 2
            data = bytearray(data); data[i] = (data[i] + 1) % 256
            jp.write_bytes(bytes(data))
        err = refused(run_probe(resume_argv(fx, copy)))
        assert "ResumptionRefused" in err and "probe_config_sha256" not in err, what
        if what == "shorter":
            assert "the log is shorter" in err and "than the recorded point" in err, err[-800:]
        else:
            assert "not the journal the progress file describes" in err or "do not hash to the recorded" in err, err[-800:]


# ---------------------------------------------------------------------------- #
# the plaintext fixture (the amendment, 4 (f): sealed and plaintext)              #
# ---------------------------------------------------------------------------- #
@pytest.mark.slow
def test_cascade_plaintext_whole_killed_inside_a_phase_and_inside_reads(fx):
    whole = prepare(fx, "pt-whole")
    ok(run_probe(argv_for(fx, whole, "cascade", extra=["--plaintext"])))
    assert (jdir_of(whole) / "journal.plaintext").exists() and not (jdir_of(whole) / "journal.key").exists()
    K = demotion_phase(whole)
    base = prepare(fx, "pt-kill-phase")
    assert run_probe(argv_for(fx, base, "cascade", extra=["--plaintext"]), patch=KILL_INSIDE_PHASE.format(K=K)).returncode == 3
    ok(run_probe(argv_for(fx, base, "cascade", extra=["--plaintext"])))
    assert_same_run(whole, base)
    base2 = prepare(fx, "pt-kill-reads")
    n_before_16 = sum(u["reads"]["n"] for i, u in units_of(whole).items() if i < 16 and "reads" in u)
    assert run_probe(argv_for(fx, base2, "cascade", extra=["--plaintext"]), patch=KILL_INSIDE_READS.format(N=n_before_16 + 2)).returncode == 3
    ok(run_probe(argv_for(fx, base2, "cascade", extra=["--plaintext"])))
    assert_same_run(whole, base2)
    assert final_of(base2)["read_lines_recorded_twice_compared"] == 1


def test_a_run_started_without_the_launcher_is_refused(fx, tmp_path):
    base = prepare(fx, "no-launcher")
    code = "from cortex_c2b import scale_probe as SP\nSP._cli(%r)\n" % argv_for(fx, base, "cascade")
    env = {k: v for k, v in os.environ.items() if k not in ("ADR010_LAUNCHER", "PYTHONHASHSEED")}
    p = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "tools/adr010_launcher.py" in p.stderr and not (base / "runs" / f"scale-{SEED}-{RUN_ID}" / "c3-identity").exists()


def test_c2_read_once_and_aggregates_on_a_constructed_journal(tmp_path):
    """C.2 (b) pinned on a constructed journal: an E1 pair whose duplicate's mark is the higher one, a never planted query
    at and under the threshold, I0's tie at the boundary, the key's contract on a cross schema repeated name."""
    from cortex_c2b.hm_protocol import _TEMPLATES
    from cortex_c2b.references import ExhaustiveCosine, BM25Index
    rng = np.random.default_rng(7)
    j = Journal(policy=POLICY_STOP, index_geometry=ADOPTED_GEOMETRY)
    wp = WritePath(j, seed=0, admission=ADMISSION_IDENTITY); jp = JournalPath(j, geometry=ADOPTED_GEOMETRY)
    base = unit(rng.standard_normal(64))
    texts = [("Tone-4010", "person", "cobbler"), ("Tone-4010", "person", "miller"), ("Kilu-77ab", "place", "Dunmore"), ("Rabo-1c9d", "person", "sawyer"),
             ("Vemo-33aa", "person", "glover"), ("Tone-4010", "place", "Frome")]
    cues = [base, unit(base + 0.03 * rng.standard_normal(64).astype(np.float32)), unit(rng.standard_normal(64)), unit(base + 0.4 * rng.standard_normal(64).astype(np.float32)),
            unit(rng.standard_normal(64)), unit(base + 0.05 * rng.standard_normal(64).astype(np.float32))]
    ids, ptrs = [], []
    for i, ((e, sch, a), c) in enumerate(zip(texts, cues)):
        st = _TEMPLATES[sch][0].format(e=e, a=a)
        rep = wp.write(c, st.encode("utf-8"), sch, now=float(i), source=st)
        assert rep.admitted
        jp.on_write(rep.entry); ids.append(rep.entry.entry_id); ptrs.append(rep.entry.pointer)
    ex = ExhaustiveCosine(j, jp); bm = BM25Index(1.5, 0.75)
    for eid in ids:
        bm.insert(eid, j.payloads.get(j._entries[eid].pointer))
    from cortex_c2b.read_path import CueIndex
    levers = {}
    for name, nt, nb in SP.LEVER_VARIANTS:
        ix = CueIndex(dim=64, n_bits=nb, n_tables=nt, seed=0, signature="exact")
        for eid in ids:
            ix.add(eid, np.asarray(j._entries[eid].cue, dtype=np.float32))
        levers[name] = ix
    entry_of_j = {i: ids[i] for i in range(6)}; pointer_of_j = {i: ptrs[i] for i in range(6)}
    q = unit(base + 0.03 * rng.standard_normal(64).astype(np.float32))              # nearer the duplicate than the own line? read the marks
    r = {"kind": "pair-E1", "j": 0, "partner_j": 1, "q": 0, "entity": "Tone-4010", "schema": "person", "text": _TEMPLATES["person"][1].format(e="Tone-4010"), "raw_cosine": 0.99}
    thr = 0.5
    line = SP.read_once(r, cues[1], j, jp, ex, bm, levers, entry_of_j, pointer_of_j, thr, 4)
    pair = line["pair"]
    assert pair["own_in_window"] and pair["duplicate_in_window"] and pair["both_at_or_above_threshold"]
    assert pair["mark_ranks_duplicate_first"] is True and pair["margin"] < 0                 # the query is the duplicate's own cue
    assert pair["selection"]["duplicate"]["R1"]["pointer"] == ptrs[1]                        # R1 keeps the model's citation
    assert pair["selection"]["duplicate"]["R4"]["abstain"] and pair["selection"]["own"]["R4"]["abstain"]   # two lines of one entity, two values: the conflict rule
    assert all(pair["lines_parse"]) and line["i0"]["touched"] > 0 and line["agree"]["same_set"] in (True, False)
    assert "tie_at_boundary" in line["i0"]
    never = {"kind": "never", "q": 0, "entity": "Zuzu-ffff", "schema": "person", "distance": 4, "nearest_planted": "Tone-4010", "text": "what is Zuzu-ffff's trade?"}
    far = SP.read_once(never, unit(rng.standard_normal(64)), j, jp, ex, bm, levers, entry_of_j, pointer_of_j, 0.99, 4)
    assert far["i0"]["max_mark_at_or_above_threshold"] is False
    near_ = SP.read_once(never, cues[0], j, jp, ex, bm, levers, entry_of_j, pointer_of_j, 0.5, 4)
    assert near_["i0"]["max_mark_at_or_above_threshold"] is True
    # the key's contract: a sample read of the cross schema repeated name, the other line of the name in the window
    sample = {"kind": "sample", "j": 0, "q": 0, "entity": "Tone-4010", "schema": "person", "text": _TEMPLATES["person"][1].format(e="Tone-4010"), "repeated_name": True, "cross_schema_repeated": True}
    sample_line = SP.read_once(sample, cues[0], j, jp, ex, bm, levers, entry_of_j, pointer_of_j, 0.5, 4, {0: [ids[5]]})
    kc = sample_line["key_contract"]
    assert kc["other_line_in_window"] is True and kc["both_lines_in_window"] is True and kc["selection_own_as_citation"]["R4"]["abstain"]
    units = [{"read": r, **line}, {"read": never, **far}, {"read": sample, **sample_line}]
    agg = SP.aggregate_c2(units, {"repeated_names": ["Tone-4010"], "repeated_detail": {}})
    assert agg["pairs_E1"]["contested"] == 1 and agg["pairs_E1"]["mark_ranks_duplicate_first"] == 1 and agg["pairs_E1"]["contested_clopper_pearson_95"] == (pytest.approx(0.025, abs=0.01), 1.0)
    assert agg["key_contract"]["both_lines_in_window"] == 1 and agg["key_contract"]["r4_abstains_with_both_lines"] == 1
    assert agg["never_planted"]["n"] == 1 and agg["never_planted"]["clopper_pearson_95"] == (0.0, pytest.approx(0.975, abs=0.001))


def test_a_test_plan_or_a_plaintext_journal_never_writes_under_metrics(fx, tmp_path):
    p = run_probe(["--draw", "777", "--test-plan-n", "100", "--ckpt", str(fx["root"] / "ckpt.pt"), "--config", str(fx["root"] / "cfg.json"),
                   "--journals", str(tmp_path / "runs"), "--out", "metrics/mqar", "--step", "encode"])
    assert p.returncode != 0 and "never writes under metrics/" in p.stderr
