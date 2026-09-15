"""The checkpoint relay (cortex_data/relay.py) against a fake `kaggle` command: a push then a
pull restores a directory tree; the ladder runner pushes after every unit and a fresh run
pulls the finished units back; without the two secrets the relay is silently off."""
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAKE = r'''#!/usr/bin/env python3
"""A fake kaggle CLI: a store directory per dataset id under $FAKE_KAGGLE_STORE."""
import os, shutil, sys
from pathlib import Path
store = Path(os.environ["FAKE_KAGGLE_STORE"])
a = sys.argv[1:]
def dataset_dir(i): return store / i.replace("/", "__")
if a[:2] == ["datasets", "status"]:
    d = dataset_dir(a[2]); print("ready" if d.exists() else "not found"); sys.exit(0 if d.exists() else 1)
if a[:2] in (["datasets", "create"], ["datasets", "version"]):
    src = Path(a[a.index("-p") + 1]); import json
    meta = json.loads((src / "dataset-metadata.json").read_text()); d = dataset_dir(meta["id"])
    if d.exists(): shutil.rmtree(d)
    shutil.copytree(src, d); print("ok"); sys.exit(0)
if a[:2] == ["datasets", "download"]:
    d = dataset_dir(a[a.index("-d") + 1]); dest = Path(a[a.index("-p") + 1])
    if not d.exists(): print("404"); sys.exit(1)
    dest.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir(): shutil.copy(f, dest / f.name)
    sys.exit(0)
print("unknown", a); sys.exit(2)
'''


@pytest.fixture
def fake_kaggle(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"; bindir.mkdir()
    exe = bindir / "kaggle"
    exe.write_text(FAKE); exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    if os.name == "nt":                                   # a .cmd shim on Windows
        (bindir / "kaggle.cmd").write_text(f'@python "{exe}" %*\n')
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("FAKE_KAGGLE_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("KAGGLE_USERNAME", "tester")
    monkeypatch.setenv("KAGGLE_KEY", "secret")
    import importlib
    from cortex_data import relay
    importlib.reload(relay)
    return relay


def test_relay_off_without_secrets(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False); monkeypatch.delenv("KAGGLE_KEY", raising=False)
    from cortex_data import relay
    assert relay.available() is False
    assert relay.push(tmp_path, "x") is False and relay.push_async(tmp_path, "x") is False and relay.pull("x", tmp_path / "y") is False


def test_push_then_pull_restores_the_tree(fake_kaggle, tmp_path):
    relay = fake_kaggle
    src = tmp_path / "runs" / "journal-n1-vx"; (src / "journal.payloads").mkdir(parents=True)
    (src / "ckpt.pt").write_bytes(b"\x00" * 1000); (src / "journal.jsonl").write_text("line\n")
    (src / "journal.payloads" / "abc").write_bytes(b"payload")
    assert relay.available()
    assert relay.exists("quantum-cortex-relay-t") is False
    assert relay.push(src, "quantum-cortex-relay-t", "step 1000") is True
    assert relay.exists("quantum-cortex-relay-t") is True
    dest = tmp_path / "elsewhere" / "journal-n1-vx"
    assert relay.pull("quantum-cortex-relay-t", dest) is True
    assert (dest / "ckpt.pt").read_bytes() == b"\x00" * 1000
    assert (dest / "journal.payloads" / "abc").read_bytes() == b"payload"
    (src / "ckpt.pt").write_bytes(b"\x01" * 10)                                  # a later push replaces the earlier one
    assert relay.push_async(src, "quantum-cortex-relay-t", "step 2000") is True
    relay.wait()
    assert relay.pull("quantum-cortex-relay-t", dest) is True and (dest / "ckpt.pt").read_bytes() == b"\x01" * 10
    assert relay.slug_for("journal-n1-v7_seed2") == "quantum-cortex-relay-journal-n1-v7-seed2"


def test_ladder_runner_pushes_units_and_a_fresh_run_pulls_them(fake_kaggle, tmp_path):
    relay = fake_kaggle
    from cortex_eval.resumable_ladder import run_resumable_ladder
    from cortex_eval.recurrent_ladder import MQARTier
    A = tmp_path / "ckpt-a"
    r1 = run_resumable_ladder(seeds=(1, 2), tiers=[MQARTier(kv_pairs=4, seq_len=32)], steps=20, ckpt_dir=A,
                              d_model=32, paths=["L3-+local-conv"], relay="quantum-cortex-relay-ladder-t")
    assert r1["units_done"] == 2 and relay.exists("quantum-cortex-relay-ladder-t")
    B = tmp_path / "ckpt-b"                                                       # a fresh container: empty ckpt dir
    r2 = run_resumable_ladder(seeds=(1, 2), tiers=[MQARTier(kv_pairs=4, seq_len=32)], steps=20, ckpt_dir=B,
                              d_model=32, paths=["L3-+local-conv"], relay="quantum-cortex-relay-ladder-t")
    assert r2["units_done"] == 2
    assert len(list(B.glob("unit-*.json"))) == 0                                  # nothing recomputed: the two units came back from the relay
    ids = {json.loads(p.read_text())["unit_id"] for p in A.glob("unit-*.json")}
    assert len(ids) == 2
