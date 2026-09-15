"""The checkpoint relay (2026-09-16): a store that survives the container.

Kaggle relaunches an interrupted session in a FRESH container: `/kaggle/working` is empty,
the runs directory kept outside the clone (the resume of 2026-09-13) is gone, and a run of
several hours restarts from step 0 (v7, run `7e343011489f`, relaunched at step 1740). The only
stores that outlive a container are a version's Output, written at the END of a session, and
Kaggle datasets, which the API can update from inside a running session.

This module pushes a directory (a run's checkpoint and sealed journal, or a ladder's unit
files) to a PRIVATE Kaggle dataset after every checkpoint or unit, as ONE tar file inside a
staging directory (no directory mode subtleties), and pulls the latest version back before a
resume. Everything is optional: without the two secrets (`KAGGLE_USERNAME`, `KAGGLE_KEY`) or
without the `kaggle` command, `available()` is False and the callers run as before.

    relay.pull("quantum-cortex-relay-journal-n1-v7", "runs/journal-n1-v7")   # before the resume lookup
    relay.push_async("runs/journal-n1-v7", "quantum-cortex-relay-journal-n1-v7", "step 3000")  # after a checkpoint

Pushes run in a background process and never block training; a push that is still running
when the next checkpoint lands is left to finish and the new one is skipped (the next
checkpoint will push). A dataset that does not exist yet is created on the first push
(private, CC0 metadata on a private store is irrelevant). One dataset per run or per ladder
directory, named after it, so two runs never overwrite each other.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ARCHIVE = "relay.tar"
_inflight: dict[str, subprocess.Popen] = {}


def kaggle_cmd() -> list[str] | None:
    """The kaggle command, or None when it is not installed."""
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    try:                                       # the module without the console script on PATH
        import kaggle  # noqa: F401
        return [sys.executable, "-m", "kaggle.cli"]
    except Exception:
        return None


def available() -> bool:
    return bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY")) and kaggle_cmd() is not None


def dataset_id(slug: str) -> str:
    return f"{os.environ['KAGGLE_USERNAME']}/{slug}"


def _stage(src: Path, slug: str, message: str) -> Path:
    """A staging directory with the metadata and ONE tar of `src`."""
    stage = Path(tempfile.mkdtemp(prefix="relay-"))
    with tarfile.open(stage / ARCHIVE, "w") as tar:
        tar.add(src, arcname=src.name)
    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": f"quantum-cortex relay: {slug}"[:50], "id": dataset_id(slug),
        "licenses": [{"name": "CC0-1.0"}],
        "description": f"checkpoint relay for {src.name}; last push: {message}",
    }, indent=2), encoding="utf-8")
    return stage


def exists(slug: str) -> bool:
    cmd = kaggle_cmd()
    if cmd is None:
        return False
    r = subprocess.run(cmd + ["datasets", "status", dataset_id(slug)], capture_output=True, text=True)
    return r.returncode == 0 and "ready" in (r.stdout + r.stderr).lower()


def push_command(stage: Path, slug: str, message: str, create: bool) -> list[str]:
    cmd = kaggle_cmd() or ["kaggle"]
    if create:
        return cmd + ["datasets", "create", "-p", str(stage), "-q"]
    return cmd + ["datasets", "version", "-p", str(stage), "-m", message[:80], "-q"]


def push(src, slug: str, message: str = "") -> bool:
    """Push `src` now (blocking). Returns True when the command returned 0."""
    if not available():
        return False
    src = Path(src)
    if not src.is_dir():
        return False
    stage = _stage(src, slug, message)
    r = subprocess.run(push_command(stage, slug, message, create=not exists(slug)), capture_output=True, text=True)
    shutil.rmtree(stage, ignore_errors=True)
    if r.returncode != 0:
        print(f"[relay] push failed ({slug}): {(r.stdout + r.stderr).strip()[-300:]}")
    return r.returncode == 0


def push_async(src, slug: str, message: str = "") -> bool:
    """Push in a background process; skipped (False) when the previous push of this slug is
    still running, when the relay is not available, or when `src` is not a directory."""
    if not available():
        return False
    src = Path(src)
    if not src.is_dir():
        return False
    prev = _inflight.get(slug)
    if prev is not None and prev.poll() is None:
        return False
    stage = _stage(src, slug, message)
    cmd = push_command(stage, slug, message, create=not exists(slug))
    _inflight[slug] = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def wait(timeout: float = 600.0) -> None:
    """Wait for the in flight pushes (called at the end of a run so the last checkpoint lands)."""
    t0 = time.time()
    for p in list(_inflight.values()):
        remaining = max(0.0, timeout - (time.time() - t0))
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass


def pull(slug: str, dest) -> bool:
    """Download the latest version of the relay dataset and unpack its tar so that `dest`
    holds what was pushed. Returns True when something landed; False (and nothing touched)
    when the relay is unavailable, the dataset does not exist, or the download failed."""
    if not available() or not exists(slug):
        return False
    dest = Path(dest)
    tmp = Path(tempfile.mkdtemp(prefix="relay-pull-"))
    cmd = kaggle_cmd() or ["kaggle"]
    r = subprocess.run(cmd + ["datasets", "download", "-d", dataset_id(slug), "-p", str(tmp), "--unzip", "--force", "-q"],
                       capture_output=True, text=True)
    tar_path = next(iter(tmp.rglob(ARCHIVE)), None)
    if r.returncode != 0 or tar_path is None:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[relay] pull found nothing for {slug}")
        return False
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        root = members[0].name.split("/")[0] if members else None
        tar.extractall(tmp / "x", filter="data")
    unpacked = tmp / "x" / root if root else None
    if unpacked is None or not unpacked.exists():
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(unpacked), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[relay] pulled {slug} -> {dest}")
    return True


def slug_for(name: str) -> str:
    """A dataset slug from a run or directory name: lowercase, hyphens, under Kaggle's limits."""
    s = "".join(ch if ch.isalnum() else "-" for ch in name.lower()).strip("-")
    while "--" in s:
        s = s.replace("--", "-")
    return ("quantum-cortex-relay-" + s)[:50].rstrip("-")
