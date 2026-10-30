"""cortex_c2b.resumable -- a long measurement resumes where it stopped, as one run (ADR-010, the amendment of
2026-10-15 on parts B and C, section 4; declared before any number, graved 2026-10-29; D6 B taken by the founder
on 2026-10-29 at 21:11).

A new module of the ADR-010 build, outside the frozen set of A.5 (the frozen system is `c5b4818`; every file here acts
on public methods and attributes of the frozen objects and changes no frozen module). Three things live here, shared by
the runner of the sealed families (`cortex_c2b.sealed_family`) and by the scale probe (`cortex_c2b.scale_probe`):

  the progress file (4 (b))   one JSON line per recorded point beside the output, never rewritten; the first line is
                              the run identity in two parts (compared at every resumption, recorded only); before each
                              line the journal's log and the payload archive are synchronised to the disk on
                              descriptors of this module's own, then the line is written and the progress file
                              synchronised, so that a recorded point is never ahead of the disk;
  the payload archive (4 (d)) one append only file beside a journal holding every payload file's bytes as sealed on the
                              disk, keyed by the pointer, so that the files a dead phase unlinked are restored before
                              the frozen open; cut to its recorded length at a resumption, the tail moved aside;
  the reopening (4 (d), D6 B) the cut rule on the journal's log (equal: a plain reopening; longer: the tail beyond the
                              recorded point moved into a side folder and the log cut to the recorded length; shorter
                              or another hash: refused), the payload directory made exactly the set the retained log
                              holds (restored from the archive byte for byte, the orphans and the `.tmp` files moved
                              aside, nothing deleted), then the frozen open, then the live index's state restored
                              (every entry's cue in the log's order through the index's own `add`, the 256 touch
                              window and the cumulative counter refilled from the recorded point), so that the opened
                              logs of a resumed and of an uninterrupted run are equal line for line.

Also here: the stop request and the session budget (4 (e)), the session table, the assembly of a final file from the
progress lines (4 (b)), and the running length and sha256 of a log (the recorded point of a cascade or a planting).
Nothing here is a number of the record.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import struct
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cortex_c2b import Journal, POLICY_STOP, STATE_EVICTED
from cortex_c2b.read_path import CueIndex, JournalPath

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
TOUCH_WINDOW = 256                    # the index's touch window (`CueIndex.touch_log`, a deque of maxlen 256; read_path.py)
ARCHIVE_MAGIC = b"QCA1"               # the payload archive's record framing: magic, pointer (16 ascii bytes), length (8 bytes), bytes
STOP_REASONS = ("budget", "stop_file", "complete", "death")


class ResumptionRefused(ValueError):
    """The run cannot continue as one run: the identity differs, the journal is not the one the progress file describes,
    or a held payload is absent from the archive. The caller retains the run under its name and reports it (C.3 (b))."""


# ---------------------------------------------------------------------------- #
# hashing and the numeric environment                                           #
# ---------------------------------------------------------------------------- #
def sha256_file(path: str | Path, lf: bool = False) -> str:
    """The sha256 of a file's bytes; with `lf`, `\\r\\n` read as `\\n` (the record's `sha256_lf` convention)."""
    b = Path(path).read_bytes()
    if lf:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def sha256_text(obj) -> str:
    """The sha256 of a JSON-able object in canonical form (sorted keys, compact separators, UTF-8, ensure_ascii off)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def numeric_environment() -> dict:
    """What the identity compares of the machine and the libraries (4 (b)): the processor, torch's thread counts, the four
    thread variables, PYTHONHASHSEED, the versions of Python, numpy and torch."""
    import torch
    return {"processor": platform.processor() or platform.machine(), "machine": platform.machine(),
            "torch_threads": int(torch.get_num_threads()), "torch_interop_threads": int(torch.get_num_interop_threads()),
            "thread_variables": {k: os.environ.get(k) for k in THREAD_VARIABLES},
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__}}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


NEW_MODULES = ("cortex_c2b/resumable.py", "cortex_c2b/references.py", "cortex_c2b/sealed_family.py", "cortex_c2b/sealed_reading.py",
               "cortex_c2b/cost_harness.py", "cortex_c2b/scale_probe.py", "cortex_c2b/a0.py", "cortex_c2b/freeze.py", "tools/adr010_launcher.py")


def require_launcher() -> None:
    """Every run of this ADR starts through `tools/adr010_launcher.py` (PYTHONHASHSEED set to 0 before the interpreter: the
    ranking of exact ties in `CueIndex.query` depends on the string hash seed); a run started otherwise is refused here."""
    if os.environ.get("ADR010_LAUNCHER") != "1" or os.environ.get("PYTHONHASHSEED") is None:
        raise SystemExit("this run starts through tools/adr010_launcher.py (PYTHONHASHSEED and the launcher's mark set before the interpreter; "
                         "the amendment of 2026-10-15, 4 (b)); refused")


def code_state(root: str | Path | None = None) -> dict:
    """The code that runs: the repository's commit and whether its tracked tree is clean (`record_tools.git_state`), and the
    sha256_lf of the new modules of this ADR, which the freeze file's list does not cover (4 (b): a change of the probe's
    own code during a run is recorded in the session table)."""
    from cortex_c2b import record_tools
    root = Path(root) if root else Path(record_tools.REPO_ROOT)
    commit, clean = record_tools.git_state()
    hashes = {}
    for rel in NEW_MODULES:
        f = root / rel
        hashes[rel] = sha256_file(f, lf=True)[:16] if f.exists() else None
    return {"commit": commit, "tree_clean": clean, "new_modules_sha256_lf": hashes}


def traceback_file(progress_path: str | Path) -> Path:
    """Where a dying session writes its traceback (4 (b)): beside the progress file, numbered."""
    p = Path(progress_path)
    n = 1
    while p.with_name(p.name + f".traceback-{n}.txt").exists():
        n += 1
    return p.with_name(p.name + f".traceback-{n}.txt")


def write_traceback(progress_path: str | Path, unit: str) -> Path:
    import traceback
    f = traceback_file(progress_path)
    f.write_text(f"died at {unit}\n{utc_now()}\n" + traceback.format_exc(), encoding="utf-8", newline="\n")
    return f


def command_line(module: str | None = None, argv: list[str] | None = None) -> str:
    """The command as the record keeps it: `python -m <module> <arguments>`, the module named by the caller (never a path of
    the machine), with the launcher's mark when `tools/adr010_launcher.py` started the process (4 (b))."""
    a0 = Path(sys.argv[0])
    mod = module or (f"cortex_c2b.{a0.stem}" if a0.suffix == ".py" else a0.name)
    args = sys.argv[1:] if argv is None else list(argv)
    launched = "  [tools/adr010_launcher.py]" if os.environ.get("ADR010_LAUNCHER") == "1" else ""
    return f"python -m {mod} " + " ".join(str(a) for a in args) + launched


# ---------------------------------------------------------------------------- #
# the running state of a log: its length and sha256, extended as it grows         #
# ---------------------------------------------------------------------------- #
class LogTracker:
    """The byte length and the running sha256 of an append only file, extended by reading the bytes appended since the
    last update (a cascade's log is read once, not at every point from its beginning)."""

    def __init__(self, path: str | Path, length: int = 0, state: hashlib._hashlib.HASH | None = None):
        self.path = Path(path)
        self.length = int(length)
        self._h = state if state is not None else hashlib.sha256()
        if length and state is None:                        # a resumption: the prefix read once to rebuild the running state
            with open(self.path, "rb") as fh:
                remaining = self.length
                while remaining > 0:
                    chunk = fh.read(min(1 << 20, remaining))
                    if not chunk:
                        raise ResumptionRefused(f"{self.path}: shorter than the recorded length {self.length}")
                    self._h.update(chunk); remaining -= len(chunk)

    def update(self) -> dict:
        """Read what was appended since the last update; return {length, sha256}."""
        if self.path.exists():
            with open(self.path, "rb") as fh:
                fh.seek(self.length)
                while True:
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        break
                    self._h.update(chunk); self.length += len(chunk)
        return self.point()

    def point(self) -> dict:
        return {"length": self.length, "sha256": self._h.hexdigest()}


def prefix_sha256(path: str | Path, length: int) -> str | None:
    """The sha256 of the first `length` bytes of a file, or None when the file is shorter."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        remaining = int(length)
        while remaining > 0:
            chunk = fh.read(min(1 << 20, remaining))
            if not chunk:
                return None
            h.update(chunk); remaining -= len(chunk)
    return h.hexdigest()


def fsync_path(path: str | Path) -> None:
    """Synchronise a file to the disk on a descriptor of our own (read and write access, nothing written)."""
    p = Path(path)
    if not p.exists():
        return
    fd = os.open(str(p), os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------- #
# the payload archive (4 (d))                                                     #
# ---------------------------------------------------------------------------- #
class PayloadArchive:
    """One append only file beside a journal: every payload file's bytes as sealed on the disk, keyed by the pointer.
    Records: magic (4), pointer (16 ascii bytes), length (8 bytes, big endian), the bytes. A pointer appended twice
    keeps its first record (the store writes one file per pointer; the bytes are the same)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._index: dict[str, tuple[int, int]] = {}         # pointer -> (offset of the bytes, length)
        self._fh = None
        self.valid_length = 0                                # the bytes up to the first torn record (a death inside an append)
        self.torn_tail_bytes = 0
        if self.path.exists():
            self._scan()

    @staticmethod
    def path_for(journal_path: str | Path) -> Path:
        jp = Path(journal_path)
        return jp.with_name(jp.stem + ".archive")

    def _scan(self) -> None:
        """Index the whole records; a torn record (a death inside an append) ends the scan, `valid_length` says where: the
        reopening cuts the archive to its recorded length, which is never beyond it (the line was written after the sync)."""
        self._index = {}
        self.valid_length = 0
        size = self.path.stat().st_size
        with open(self.path, "rb") as fh:
            while True:
                start = fh.tell()
                head = fh.read(4 + 16 + 8)
                if not head:
                    break
                if len(head) < 28 or head[:4] != ARCHIVE_MAGIC:
                    break
                length = struct.unpack(">Q", head[20:28])[0]
                off = fh.tell()
                if off + length > size:
                    break
                pointer = head[4:20].decode("ascii")
                fh.seek(length, 1)
                self._index.setdefault(pointer, (off, length))
                self.valid_length = fh.tell()
            if self.valid_length != size:
                self.torn_tail_bytes = size - self.valid_length
            else:
                self.torn_tail_bytes = 0

    def __contains__(self, pointer: str) -> bool:
        return pointer in self._index

    def __len__(self) -> int:
        return len(self._index)

    @property
    def length(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0

    def append(self, pointer: str, blob: bytes) -> bool:
        """Append a payload's bytes (as on the disk) under its pointer; False when the pointer is already archived."""
        if pointer in self._index:
            return False
        if len(pointer) != 16:
            raise ValueError(f"a pointer is sixteen characters, got {pointer!r}")
        if self.torn_tail_bytes:
            raise ResumptionRefused(f"{self.path}: a torn record of {self.torn_tail_bytes} bytes at its end; the archive is cut to its recorded length before anything is appended")
        if self._fh is None:
            self._fh = open(self.path, "ab")
        off = self._fh.tell() + 28
        self._fh.write(ARCHIVE_MAGIC + pointer.encode("ascii") + struct.pack(">Q", len(blob)) + blob)
        self._fh.flush()
        self._index[pointer] = (off, len(blob))
        return True

    def append_from_store(self, payload_dir: str | Path, pointers) -> list[str]:
        """Archive the files of these pointers from the payload directory as they are on the disk (sealed)."""
        done = []
        for p in pointers:
            f = Path(payload_dir) / p
            if p not in self._index and f.exists():
                self.append(p, f.read_bytes()); done.append(p)
        return done

    def get(self, pointer: str) -> bytes:
        off, length = self._index[pointer]
        with open(self.path, "rb") as fh:
            fh.seek(off)
            return fh.read(length)

    def fsync(self) -> None:
        if self._fh is not None:
            self._fh.flush(); os.fsync(self._fh.fileno())
        elif self.path.exists():
            fsync_path(self.path)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush(); os.fsync(self._fh.fileno()); self._fh.close(); self._fh = None

    def cut(self, length: int, side_dir: Path) -> dict:
        """Cut the archive to a recorded length; the tail moved into the side folder. Returns what was cut."""
        self.close()
        cur = self.length
        if cur < length:
            raise ResumptionRefused(f"{self.path}: shorter ({cur}) than its recorded length ({length})")
        if self.path.exists() and length > self.valid_length:
            raise ResumptionRefused(f"{self.path}: the recorded length {length} lies beyond its whole records ({self.valid_length}); not the archive the progress file describes")
        rec = {"archive_length_before": cur, "archive_length_recorded": int(length), "archive_tail_bytes": cur - int(length)}
        if cur > length:
            side_dir.mkdir(parents=True, exist_ok=True)
            with open(self.path, "rb") as fh:
                fh.seek(length); tail = fh.read()
            (side_dir / "archive-tail.bin").write_bytes(tail)
            rec["archive_tail_sha256"] = hashlib.sha256(tail).hexdigest()
            with open(self.path, "r+b") as fh:
                fh.truncate(length)
        if self.path.exists():
            self._scan()
        return rec


# ---------------------------------------------------------------------------- #
# the progress file (4 (b)) and the sessions (4 (e))                             #
# ---------------------------------------------------------------------------- #
class Progress:
    """The JSON lines progress file of a resumable measurement, beside its output. Line kinds: `identity` (first, once),
    `session_start`, `session_end`, `unit` (a recorded point with its payload), `cut` (a reopening's cut and restorations),
    `code_change`. Never rewritten; `append` synchronises the files given before writing, then the line, then itself."""

    def __init__(self, path: str | Path, sync_paths: list[Path] | None = None, archive: PayloadArchive | None = None):
        self.path = Path(path)
        self.sync_paths = [Path(p) for p in (sync_paths or [])]
        self.archive = archive
        self._fh = None
        self.torn_tail: dict | None = None
        if self.path.exists():
            self._repair()

    def _repair(self) -> None:
        """A death between the write of a line and its sync can leave a partial last line: it is cut off (kept beside the
        file as `<name>.torn-<n>`) so that the next line starts at a line start; every whole line stays. A partial line
        is never a recorded point (4 (b)): the point it would have recorded is recomputed."""
        data = self.path.read_bytes()
        if not data or data.endswith(b"\n"):
            return
        cut = data.rfind(b"\n") + 1
        tail = data[cut:]
        n = 1
        while self.path.with_name(self.path.name + f".torn-{n}").exists():
            n += 1
        side = self.path.with_name(self.path.name + f".torn-{n}")
        side.write_bytes(tail)
        with open(self.path, "r+b") as fh:
            fh.truncate(cut)
        fsync_path(self.path)
        self.torn_tail = {"bytes": len(tail), "sha256": hashlib.sha256(tail).hexdigest(), "kept_as": str(side)}

    def exists(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 0

    def lines(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                out.append(json.loads(raw))
        return out

    def identity(self) -> dict | None:
        ls = self.lines()
        return ls[0] if ls and ls[0].get("kind") == "identity" else None

    def units(self) -> dict[int, dict]:
        """The recorded units by index (the first line of an index wins; a later line of the same index is compared by
        the caller at assembly, 4 (c))."""
        out: dict[int, dict] = {}
        for l in self.lines():
            if l.get("kind") == "unit":
                out.setdefault(int(l["index"]), l)
        return out

    def duplicates(self) -> list[dict]:
        """Units recorded twice (a resumption recomputed an index a later line also records): compared at assembly."""
        seen: dict[int, dict] = {}; dups = []
        for l in self.lines():
            if l.get("kind") == "unit":
                i = int(l["index"])
                if i in seen:
                    dups.append({"index": i, "equal": self._unit_equal(seen[i], l), "first": seen[i], "second": l})
                else:
                    seen[i] = l
        return dups

    @staticmethod
    def _unit_equal(a: dict, b: dict) -> bool:
        ka = {k: v for k, v in a.items() if k not in ("wall_clock", "session")}
        kb = {k: v for k, v in b.items() if k not in ("wall_clock", "session")}
        return ka == kb

    def write_identity(self, identity: dict) -> None:
        if self.exists():
            raise ValueError(f"{self.path} exists; the identity is written once, as the first line")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.append({"kind": "identity", **identity})

    def append(self, record: dict) -> None:
        """Synchronise the journal's log and the archive first, then write the line, then synchronise the progress file:
        a recorded point is never ahead of the disk (4 (b))."""
        for p in self.sync_paths:
            fsync_path(p)
        if self.archive is not None:
            self.archive.fsync()
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8", newline="\n")
        self._fh.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")
        self._fh.flush(); os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush(); os.fsync(self._fh.fileno()); self._fh.close(); self._fh = None

    def start_session(self, record: dict) -> dict:
        """The `session_start` line with the code state (4 (b)); when the code state differs from the previous session's, a
        `code_change` line follows, naming the two commits and the modules whose hash moved (recorded, not a refusal: the
        amendment lets the run continue when the fixture test shows equal progress lines)."""
        state = code_state()
        prev = None
        for l in self.lines():
            if l.get("kind") == "session_start" and l.get("code_state"):
                prev = l["code_state"]
        self.append({"kind": "session_start", "code_state": state, **record})
        if prev is not None and prev != state:
            changed = sorted(k for k in set(prev["new_modules_sha256_lf"]) | set(state["new_modules_sha256_lf"])
                             if prev["new_modules_sha256_lf"].get(k) != state["new_modules_sha256_lf"].get(k))
            self.append({"kind": "code_change", "wall_clock": utc_now(), "previous_commit": prev.get("commit"), "commit": state.get("commit"),
                         "modules_changed": changed, "previous_tree_clean": prev.get("tree_clean"), "tree_clean": state.get("tree_clean")})
        return state

    def sessions(self) -> list[dict]:
        """The session table (4 (e)) from the `session_start` and `session_end` lines, in order; a start without an end is
        a death (the reason `death`, the end unknown); the reopening's cut (when something was cut, restored or moved) or
        its plain reopening; a code change between sessions."""
        table: list[dict] = []
        for l in self.lines():
            if l.get("kind") == "session_start":
                table.append({"session": len(table) + 1, "start": l.get("wall_clock"), "end": None, "reason": "death",
                              "resumed_from": l.get("resumed_from"), "reopening_seconds": l.get("reopening_seconds"),
                              "first_unit": l.get("first_unit"), "last_unit": None, "cuts": [], "reopening": None, "restored": l.get("restored"), "moved": l.get("moved"),
                              "code_change": None, "code_state": l.get("code_state"), "command": l.get("command"), "budget_seconds": l.get("budget_seconds"),
                              "torn_progress_tail": l.get("torn_progress_tail"), "phases_at_start": l.get("phases_at_start"), "traceback": None})
            elif l.get("kind") == "session_end" and table:
                table[-1].update({"end": l.get("wall_clock"), "reason": l.get("reason"), "last_unit": l.get("last_unit"), "resumption_point": l.get("resumption_point")})
                if l.get("first_unit") is not None:
                    table[-1]["first_unit"] = l.get("first_unit")
            elif l.get("kind") == "cut" and table:
                table[-1]["cuts"].append({k: v for k, v in l.items() if k not in ("kind",)})
            elif l.get("kind") == "reopening" and table:
                table[-1]["reopening"] = {k: v for k, v in l.items() if k != "kind"}
            elif l.get("kind") == "code_change" and table:
                table[-1]["code_change"] = {k: v for k, v in l.items() if k != "kind"}
            elif l.get("kind") == "death" and table:
                table[-1]["traceback"] = {k: v for k, v in l.items() if k != "kind"}
        return table


class StopFile:
    """The stop request (4 (e)): a file named in the configuration, beside the progress file; its appearance stops the
    session at the next phase boundary or recorded point."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None

    def requested(self) -> bool:
        return bool(self.path is not None and self.path.exists())


class Budget:
    """A session's wall clock budget in seconds, counted from the session's first unit (a cascade: its first write)."""

    def __init__(self, seconds: float | None):
        self.seconds = None if seconds is None else float(seconds)
        self.started_at: float | None = None

    def start(self) -> None:
        if self.started_at is None:
            self.started_at = time.monotonic()

    def elapsed(self) -> float:
        return 0.0 if self.started_at is None else time.monotonic() - self.started_at

    def exceeded(self) -> bool:
        return bool(self.seconds is not None and self.started_at is not None and self.elapsed() >= self.seconds)


# ---------------------------------------------------------------------------- #
# the run identity (4 (b))                                                        #
# ---------------------------------------------------------------------------- #
def make_identity(compared: dict, recorded: dict) -> dict:
    """The first line of a progress file: what is compared at every resumption and what is only recorded."""
    return {"compared": compared, "recorded": {**recorded, "created": utc_now()}}


def compare_identity(recorded_identity: dict, current_compared: dict, ignore: tuple[str, ...] = ()) -> list[str]:
    """The differences between the recorded identity's compared part and the current one, key by key (nested dicts
    compared as a whole; the keys in `ignore` compared by their own rule, a prefix hash, by the caller); empty when the run
    may continue."""
    rec = (recorded_identity or {}).get("compared") or {}
    diffs = []
    for k in sorted(set(rec) | set(current_compared)):
        if k in ignore:
            continue
        a, b = rec.get(k), current_compared.get(k)
        if a != b:
            diffs.append(f"{k}: recorded {json.dumps(a, sort_keys=True)} against now {json.dumps(b, sort_keys=True)}")
    return diffs


# ---------------------------------------------------------------------------- #
# the reopening of a journal at a recorded point (4 (d), D6 B)                    #
# ---------------------------------------------------------------------------- #
def payload_dir_of(journal_path: str | Path) -> Path:
    jp = Path(journal_path)
    return jp.with_name(jp.stem + ".payloads")


def next_side_dir(journal_path: str | Path) -> Path:
    jp = Path(journal_path)
    n = 1
    while True:
        d = jp.with_name(jp.name + f".cut-{n}")
        if not d.exists():
            return d
        n += 1


def held_pointers(journal_path: str | Path, key: bytes | None, plaintext: bool) -> set[str]:
    """The pointers the retained log holds: a replay of the log by the frozen `Journal` on an empty payload directory
    (whose `_refs` is that set). The journal file is read, never written (the open probes writability by appending nothing)."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="held-") as tmp:
        j = Journal(journal_path, key=key, plaintext=plaintext, payload_dir=Path(tmp) / "empty.payloads", policy=POLICY_STOP)
        return set(j._refs.keys())


def reconcile_journal(journal_path: str | Path, key: bytes | None, point: dict, archive: PayloadArchive,
                      plaintext: bool = False) -> dict:
    """The cut rule and the payload directory, before the frozen open (4 (d)). `point`: the last recorded point's
    {"length", "sha256"} of the log and "archive_length". Returns the record of what was done (for the progress file's
    `cut` line and the session table); raises ResumptionRefused when the journal is not the one described."""
    jpath = Path(journal_path)
    rec = {"kind": "cut", "wall_clock": utc_now(), "log_length_recorded": int(point["length"]), "log_sha256_recorded": point["sha256"],
           "log_length_before": jpath.stat().st_size if jpath.exists() else 0, "action": None, "lines_moved": 0, "tail_sha256": None,
           "restored": [], "moved": [], "side_dir": None}
    if not jpath.exists():
        raise ResumptionRefused(f"{jpath}: the journal's log is missing")
    L = int(point["length"])
    cur = jpath.stat().st_size
    if cur < L:
        raise ResumptionRefused(f"{jpath}: the log is shorter ({cur}) than the recorded point ({L}); not the journal the progress file describes")
    h = prefix_sha256(jpath, L)
    if h != point["sha256"]:
        raise ResumptionRefused(f"{jpath}: the log's first {L} bytes hash {h}, the recorded point {point['sha256']}; not the journal the progress file describes")
    side = next_side_dir(jpath)
    if cur == L:
        rec["action"] = "plain reopening"
    else:
        side.mkdir(parents=True, exist_ok=True)
        with open(jpath, "rb") as fh:
            fh.seek(L); tail = fh.read()
        (side / "log-tail.jsonl").write_bytes(tail)
        rec.update({"action": "cut", "lines_moved": tail.count(b"\n") + (0 if tail.endswith(b"\n") or not tail else 1),
                    "tail_sha256": hashlib.sha256(tail).hexdigest(), "tail_bytes": len(tail), "side_dir": str(side)})
        with open(jpath, "r+b") as fh:
            fh.truncate(L)
    # the archive cut to its recorded length (its tail moved aside too)
    arec = archive.cut(int(point.get("archive_length", archive.length)), side)
    rec.update(arec)
    # the payload directory made exactly the set the retained log holds
    pdir = payload_dir_of(jpath)
    pdir.mkdir(parents=True, exist_ok=True)
    held = held_pointers(jpath, key, plaintext)
    present = {f.name: f for f in pdir.iterdir() if f.is_file()}
    for p in sorted(held):
        if p not in archive:
            raise ResumptionRefused(f"{jpath}: the retained log holds the payload {p} and the archive does not; refused")
        want = archive.get(p)
        f = pdir / p
        if not f.exists() or f.read_bytes() != want:
            why = "missing" if not f.exists() else "different"
            if f.exists():
                side.mkdir(parents=True, exist_ok=True); shutil.move(str(f), str(side / (p + ".replaced")))
            f.write_bytes(want)
            rec["restored"].append({"pointer": p, "why": why})
    for name, f in sorted(present.items()):
        if name in held:
            continue                                     # kept, or restored above (the different file already moved aside)
        side.mkdir(parents=True, exist_ok=True)          # an orphan or a `.tmp` of a write cut by the death: moved, never deleted
        shutil.move(str(f), str(side / name))
        rec["moved"].append(name)
    if rec["moved"] or rec["restored"] or rec["action"] == "cut" or arec.get("archive_tail_bytes"):
        rec["side_dir"] = str(side)
    else:
        rec["kind"] = "reopening"                       # nothing cut, restored or moved: a plain reopening, recorded as such
    return rec


def restore_live_index(jp: JournalPath, journal: Journal, touch_window: list[int] | None, touched: int | None) -> dict:
    """D6 B: the index as the live process held it. A fresh `CueIndex` of the path's geometry filled with every entry's
    cue in the log's order (the evicted entries included, whose vectors the live index kept), the demotion aliases set,
    the 256 touch window and the cumulative counter refilled from the recorded point; on the path's public attribute.
    Returns the counts."""
    live = jp.index
    fresh = CueIndex(dim=journal.cue_dim, n_bits=live.n_bits, n_tables=live.n_tables, seed=live.seed, rank=live.rank,
                     signature=live.signature)
    assert fresh.geometry() == live.geometry(), "the fresh index must draw the path's planes"
    added = 0
    for eid, e in journal._entries.items():                # dict order: the log's order
        fresh.add(eid, np.asarray(e.cue, dtype=np.float32)); added += 1
    for src, sid in journal.summary_of.items():
        fresh.set_alias(src, sid)
    fresh.touch_log = deque([int(x) for x in (touch_window or [])], maxlen=TOUCH_WINDOW)
    fresh.touched = int(touched or 0)
    jp.index = fresh
    absent_from_frozen_open = sum(1 for eid, e in journal._entries.items() if e.state == STATE_EVICTED and eid not in journal.summary_of)
    return {"vectors": added, "evicted_entries_in_index": sum(1 for e in journal._entries.values() if e.state == STATE_EVICTED),
            "absent_from_frozen_open": absent_from_frozen_open,      # the evicted, undemoted entries whose vectors the frozen open leaves out and the live index kept
            "touch_window": len(fresh.touch_log), "touched": fresh.touched,
            "what": "a fresh CueIndex of the path's geometry filled with every entry's cue in the log's order (the live index's add order), the aliases set, the touch state refilled"}


def index_state(jp: JournalPath) -> dict:
    """The index's touch window and cumulative counter, as a recorded point carries them (D6 B)."""
    return {"touch_window": [int(x) for x in jp.index.touch_log], "touched": int(jp.index.touched)}


# ---------------------------------------------------------------------------- #
# assembly of a final file (4 (b))                                               #
# ---------------------------------------------------------------------------- #
def assemble(progress: Progress) -> dict:
    """The final file's common part from the progress lines: the identity, the units in index order (kept once; a
    duplicate recomputed at a resumption compared and reported, a difference being a fault of the resumption), the
    session table, the cuts, the duplicate check, the progress file's own sha256."""
    units = progress.units()
    dups = progress.duplicates()
    sessions = progress.sessions()
    return {"identity": progress.identity(), "units": [units[i] for i in sorted(units)], "n_units": len(units),
            "sessions": sessions, "run_in_sessions": len(sessions),
            "duplicates_compared": [{"index": d["index"], "equal": d["equal"]} for d in dups],
            "resumption_fault": [d["index"] for d in dups if not d["equal"]],
            "progress_file": str(progress.path), "progress_sha256": sha256_file(progress.path) if progress.path.exists() else None}


def write_json(path: str | Path, obj: dict, indent: int = 1) -> str:
    """A file of the record: LF line endings, no NaN, UTF-8; returns the sha256 (first sixteen hexadecimal characters)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=indent, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
