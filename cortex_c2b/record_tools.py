"""cortex_c2b.record_tools: what a file of the record carries about the code and the machine that produced it, and
how an instrument reads a sealed journal of the record without touching it (ADR-008, amendment of 2026-10-13; the
practice of the bucket probe of 2026-10-12, shared here so that session B's replays follow it).

  - `git_state()`: the short commit of HEAD and whether the tracked tree is clean (the tests of the instruments replace
    this function: a fixture run is not a file of the record).
  - `versions()`, `device_info(device)`: the software and the hardware, written in every file.
  - `tree_hash(log, payloads)`: the log's sha256 and one sha256 over the payload files' names and bytes, in name order,
    what "nothing is written" is checked against before and after a run.
  - `opened_copy(journal_path, key)`: the journal copied into a temporary directory and opened there with policy stop,
    because the organ's open of a journal probes its log for writability and releases unreferenced payload files; the
    journal of the record is never opened in place by an instrument.
"""
from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def git_state() -> tuple[str | None, bool | None]:
    """The short commit of the repository's HEAD and whether its tracked tree is clean; (None, None) when git does not
    answer. Untracked files do not count: the founder's measurement files are untracked until his commit."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=str(REPO_ROOT),
                                timeout=10).stdout.strip() or None
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                               cwd=str(REPO_ROOT), timeout=10).stdout.strip()
        return commit, ((dirty == "") if commit else None)
    except Exception:
        return None, None


def versions() -> dict:
    import numpy as np
    import torch
    return {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform()}


def device_info(device) -> dict:
    import torch
    cuda = bool(torch.cuda.is_available())
    return {"device": str(device), "cuda_available": cuda, "gpu": (torch.cuda.get_device_name(0) if cuda else None),
            "hardware": platform.processor(), "threads": torch.get_num_threads(), "parallel_info": torch.__config__.parallel_info()}


def payload_dir_of(journal_path: Path) -> Path:
    return journal_path.with_name(journal_path.stem + ".payloads")


def tree_hash(log: Path, payloads: Path) -> dict:
    """The log's sha256 and one sha256 over the payload files' names and bytes, in name order (sixteen hexadecimal
    characters each), with the count of payload files; the same computation as the bucket probe's."""
    if not payloads.is_dir():
        raise ValueError(f"{payloads} is not a directory; the sealed journal's payload store sits next to its log")
    h = hashlib.sha256(); n = 0
    for q in sorted(payloads.iterdir()):
        h.update(q.name.encode("utf-8")); h.update(q.read_bytes()); n += 1
    return {"log": hashlib.sha256(log.read_bytes()).hexdigest()[:16], "payloads": h.hexdigest()[:16], "n_payload_files": n}


@contextmanager
def opened_copy(journal_path, key: bytes | None, prefix: str = "record-copy-"):
    """Yields (journal, copy_path) for a copy of the sealed journal in a temporary directory, deleted after; the
    journal is opened with policy stop. A path that is not a file, a payload store that is not a directory, or a journal
    that does not open under the key, raises ValueError before anything else happens (a wrong path must never create an
    empty journal, and a log without its store must never open as an empty one)."""
    from cortex_c2b import Journal, POLICY_STOP
    jpath = Path(journal_path)
    if not jpath.is_file():
        raise ValueError(f"{jpath} is not a file; the journal of the record is `<dir>/journal.jsonl`")
    payloads = payload_dir_of(jpath)
    if not payloads.is_dir():
        raise ValueError(f"{payloads} is not a directory; the sealed journal's payload store sits next to its log")
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
        copy = Path(tmp) / jpath.name
        shutil.copy2(jpath, copy)
        shutil.copytree(payloads, payload_dir_of(copy))
        try:
            journal = Journal(copy, key=key, policy=POLICY_STOP)
        except Exception as e:                                               # a wrong key: InvalidTag at the first sealed line
            raise ValueError(f"the journal {jpath} does not open under the key given ({type(e).__name__}: {e})") from e
        yield journal, copy
