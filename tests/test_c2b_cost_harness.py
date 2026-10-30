"""The cost harness of ADR-010 B.3 (the amendment of 2026-10-15 on parts B and C): the pinned threads read back, refused
otherwise; the stages' structure (twenty one rule rows per index and checkpoint, the read side's four stages, the write
side's, the memory run, the proxies); the instability rule on synthetic repetitions; one end to end run on a fixture
family and a tiny model in a process started as the launcher starts it."""
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import train
from cortex_c2b import record_tools, ADMISSION_IDENTITY
from cortex_c2b.crypto import generate_key
from cortex_c2b import cost_harness as CH
from cortex_c2b import sealed_families as S
from cortex_c2b import sealed_family as SF

REPO = Path(record_tools.REPO_ROOT)


def test_check_pinned_reads_the_four_variables_and_torch_back(monkeypatch):
    for k in CH.THREAD_VARIABLES:
        monkeypatch.delenv(k, raising=False)
    r = CH.check_pinned()
    assert not r["pinned"] and set(CH.THREAD_VARIABLES) <= set(r["problems"]) and r["torch_threads"] == 1
    for k in CH.THREAD_VARIABLES:
        monkeypatch.setenv(k, "1")
    r = CH.check_pinned()
    assert r["thread_variables"] == {k: "1" for k in CH.THREAD_VARIABLES}
    assert r["pinned"] == (r["torch_interop_threads"] == 1)            # in this process the inter op pool may already be started: read back, never assumed


def test_the_cli_refuses_an_unpinned_run(tmp_path):
    code = ("from cortex_c2b import cost_harness as CH\n"
            "CH._cli(['--family', 'A', '--ckpt', 'x', '--config', 'y', '--journal', 'z'])\n")
    env = {k: v for k, v in os.environ.items() if k not in CH.THREAD_VARIABLES}
    p = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "not pinned to one" in p.stderr and "--pin-threads" in p.stderr
    env2 = {**{k: v for k, v in os.environ.items() if k not in ("ADR010_LAUNCHER", "PYTHONHASHSEED")}, **{k: "1" for k in CH.THREAD_VARIABLES}}
    p = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env2, capture_output=True, text=True)
    assert p.returncode != 0 and "tools/adr010_launcher.py" in p.stderr                   # pinned by hand, not started through the launcher


def test_the_launcher_sets_the_environment_before_the_interpreter(tmp_path):
    """`tools/adr010_launcher.py`: PYTHONHASHSEED 0, the mark, unbuffered output, and the four thread variables under
    --pin-threads only; the module runs as `python -m <module>` with the launcher's exit code."""
    (tmp_path / "envprobe.py").write_text("import json, os, sys\nprint(json.dumps({k: os.environ.get(k) for k in sys.argv[1:]}))\nraise SystemExit(7)\n", encoding="utf-8")
    keys = ["PYTHONHASHSEED", "ADR010_LAUNCHER", "PYTHONUNBUFFERED", *CH.THREAD_VARIABLES]
    env = {k: v for k, v in os.environ.items() if k not in keys}
    launcher = str(REPO / "tools" / "adr010_launcher.py")
    p = subprocess.run([sys.executable, launcher, "envprobe", *keys], cwd=str(tmp_path), env=env, capture_output=True, text=True)
    seen = json.loads(p.stdout)
    assert p.returncode == 7 and seen["PYTHONHASHSEED"] == "0" and seen["ADR010_LAUNCHER"] == "1" and seen["PYTHONUNBUFFERED"] == "1"
    assert all(seen[k] is None for k in CH.THREAD_VARIABLES)
    p = subprocess.run([sys.executable, launcher, "--pin-threads", "envprobe", *keys], cwd=str(tmp_path), env=env, capture_output=True, text=True)
    seen = json.loads(p.stdout)
    assert p.returncode == 7 and all(seen[k] == "1" for k in CH.THREAD_VARIABLES) and seen["PYTHONHASHSEED"] == "0"
    p = subprocess.run([sys.executable, launcher], cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert p.returncode == 2 and "usage" in p.stderr


def test_timed_stage_disables_the_collector_and_counts_collections_outside():
    assert gc.isenabled()
    with CH.Timed() as t:
        assert not gc.isenabled()
        x = sum(range(1000))
    assert gc.isenabled() and t.ns is not None and t.ns >= 0 and x == 499500
    gc.callbacks.append(CH._gc_callback)
    try:
        before = CH.Timed.collections_outside
        gc.collect()
        assert CH.Timed.collections_outside > before
        with CH.Timed():
            n = CH.Timed.collections_outside
            gc.collect()                                            # a collection inside a stage is not counted as outside
            assert CH.Timed.collections_outside == n
    finally:
        gc.callbacks.remove(CH._gc_callback)


def test_the_instability_rule_on_synthetic_repetitions():
    base = [[100.0] * 20 for _ in range(6)]
    times = {"I0": {"index": [list(r) for r in base]}}
    s = CH.summarise(times, overhead=0.5)
    v = s["I0"]["index"]
    assert v["median"] == 100.0 and v["spread_of_medians"] == 0.0 and v["unstable_repetitions"] == [] and not v["flag"]
    assert not v["under_100_timer_overheads"]
    times["I0"]["index"][3] = [250.0] * 20                          # a repetition's median above twice the others' median
    v = CH.summarise(times, overhead=0.5)["I0"]["index"]
    assert v["unstable_repetitions"] == [3] and v["flag"] and v["repetition_medians"][3] == 250.0 and v["spread_of_medians"] == 150.0
    times["I0"]["index"][3] = [40.0] * 20                           # under half
    v = CH.summarise(times, overhead=0.5)["I0"]["index"]
    assert v["unstable_repetitions"] == [3] and v["flag"]
    v = CH.summarise(times, overhead=10.0)["I0"]["index"]           # a stage under a hundred timer overheads: reported beside the overhead, not flagged
    assert v["under_100_timer_overheads"] and not v["flag"] and v["unstable_repetitions"] == [3]
    absent = CH.summarise({"I1": {"marks": [[None] * 3] * 6}}, 0.5)["I1"]["marks"]
    assert absent == {"absent": True}
    assert CH.quantiles([]) == {"n": 0} and CH.quantiles([1.0, 2.0, 3.0])["median"] == 2.0


def test_maintenance_proxies_are_labelled_proxies():
    m = CH.maintenance_proxies((1.5, 0.75))
    assert m["label"].startswith("proxies") and m["tunable_parameters"]["I0"]["count"] == 4 and m["tunable_parameters"]["I1"]["count"] == 0
    assert m["tunable_parameters"]["I2"] == {"k1": 1.5, "b": 0.75, "count": 2} and all(v > 0 for v in m["lines_of_code"].values())


# ---------------------------------------------------------------------------- #
# end to end on a fixture family (a process started as the launcher starts it)   #
# ---------------------------------------------------------------------------- #
def tiny(seed=2):
    cfg = train.Config(n_layer=3, n_head=2, n_embd=32, block_size=96, journal="read", journal_block=1, journal_heads=2,
                       journal_read_bytes=96, journal_cue_dim=64, journal_k=3)
    torch.manual_seed(seed)
    m = train.VanillaGPT(cfg); m.eval()
    return cfg, m


PATCH = """
import os, sys, json
from cortex_c2b import record_tools, sealed_family as SF, sealed_families as S, cost_harness as CH
record_tools.git_state = lambda: ('test0000', True)
_all = S.generate_all
def _gen():
    fams = _all(); fam = dict(fams['A']); fam['facts'] = fam['facts'][:12]; fam['never_planted'] = fam['never_planted'][:4]; fam['cross_schema'] = []
    return {**fams, 'A': fam}
S.generate_all = _gen
SF.seal_check = lambda fam, seal_path=SF.SEAL_PATH: {'seal_path': 'fixture', 'seal_sha256': 'fixture', 'family_content_sha256': 'fixture', 'constants_sha256': 'fixture'}
CH._cli(__ARGV__)
"""


@pytest.mark.slow
def test_end_to_end_on_a_fixture_family(tmp_path):
    cfg, model = tiny()
    ck = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict(), "run_id": "tiny", "config_hash": train.config_hash(cfg), "step": 0}, ck)
    cfgp = tmp_path / "cfg.json"
    cfgp.write_text(json.dumps({"n_layer": 3, "n_head": 2, "n_embd": 32, "block_size": 96, "journal": "read", "journal_block": 1, "journal_heads": 2,
                                "journal_read_bytes": 96, "journal_cue_dim": 64, "journal_k": 3}), encoding="utf-8")
    fams = S.generate_all(); fam = dict(fams["A"]); fam["facts"] = fam["facts"][:12]; fam["never_planted"] = fam["never_planted"][:4]; fam["cross_schema"] = []
    key = generate_key(); jpath = tmp_path / "A.jsonl"
    rec = SF.plant_family(model, cfg, fam, jpath, key, torch.device("cpu"), admission=ADMISSION_IDENTITY)
    rec["threshold"] = SF.threshold_of(model, cfg, torch.device("cpu"))
    jpath.with_name("A.planting.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    jpath.with_name("A.key").write_text(key.hex(), encoding="utf-8", newline="\n")
    argv = ["--family", "A", "--ckpt", str(ck), "--config", str(cfgp), "--journal", str(jpath), "--out", str(tmp_path), "--repetitions", "2", "--generation-questions", "3"]
    env = {**os.environ, **{k: "1" for k in CH.THREAD_VARIABLES}, "PYTHONHASHSEED": "0", "ADR010_LAUNCHER": "1"}
    p = subprocess.run([sys.executable, "-c", PATCH.replace("__ARGV__", repr(argv))], cwd=str(REPO), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert p.returncode == 0, p.stderr[-3000:]
    f = json.loads((tmp_path / "cost-A-tiny.json").read_text(encoding="utf-8"))
    assert f["pinned"]["pinned"] and f["pinned"]["torch_interop_threads"] == 1 and f["label"].startswith("not a scale result")
    p1 = f["read_side"]["protocol_1"]
    assert p1["repetitions"] == 2 and p1["n_questions"] == 16 and len(p1["orders"]) == 2
    for ix in ("I0", "I1", "I2"):
        st = p1["stages"][ix]
        assert all(r in st for r in CH.RULE_STAGES) and {"cue", "index", "marks", "window", "operations"} <= set(st)   # 7 rule rows per index: 21 per checkpoint and family
        assert st["index"]["unit"] == "ns" and st["index"]["n"] == 32 and len(st["index"]["repetition_medians"]) == 2
        assert st["operations"]["n"] == 32
        if ix == "I0":
            assert st["marks"].get("absent") is True and "included" in st["marks"]["note"]
        else:
            assert "median" in st["marks"]
    ws = f["write_side"]["stages"]
    assert {"cue", "gate", "insert_I0", "insert_I1", "insert_I2", "store_write", "seal_payload", "seal_line", "append_derived"} <= set(ws)
    assert ws["append_derived"]["label"].startswith("derived") and "absent" in ws["gate"]["note"]
    assert f["write_side"]["durability"].startswith("the write's durability is the organ's as frozen")
    assert set(f["memory"]["read_side_build_bytes"]) == {"I0", "I1", "I2"} and f["memory"]["entries"] == 12
    assert f["memory"]["journal_open_bytes"] > 0 and f["memory"]["journal_in_memory_bytes"] == f["memory"]["journal_open_bytes"] + f["memory"]["journal_payloads_read_bytes"]
    assert f["write_side"]["cues"]["compared"] == 12 and f["write_side"]["cues"]["largest_abs_difference"] == 0.0 and "near" in f["write_side"]["write_event"]
    assert "surviving citation" in f["read_side"]["protocol_1"]["rule_stage_note"]
    assert f["generation"]["n"] == 3 and f["maintenance_proxies"]["label"].startswith("proxies")
    assert f["command"].startswith("python -m cortex_c2b.cost_harness") and "[tools/adr010_launcher.py]" in f["command"]
    assert f["environment"]["thread_variables"] == {k: "1" for k in CH.THREAD_VARIABLES}
