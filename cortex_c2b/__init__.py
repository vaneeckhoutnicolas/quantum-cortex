"""cortex_c2b — the persistent tier (ADR-007), Slice A: the store.

The journal is the episodic organ: weights = semantic and procedural; journal =
episodic. It lives OUTSIDE the weights and survives the session boundary — the
thing the H.M. protocol measures. Slice A ships the store and its three laws:

  Size law   — an entry is ≤ 1 KB: (cue, pointer, salience, schema_id, timestamps,
               state). Content is NEVER duplicated into the journal: the payload
               lives in a content-addressed payload store (cortex_data hashing
               reused) and the entry carries only its hash.
  Read law   — retrieval is never a scan. Reads go through an index keyed by cue
               (a sub-linear structure); an O(N) scan over entries is prohibited
               BY TEST (test_no_scan fails if a read touches every entry).
  Lifecycle  — every entry carries its RES-17 (type,state) cell:
               live → consolidated → demoted → evicted. Slice A stores and
               transitions the state; the scheduling (RES-9) is Slice E.

Deliberately NOT in Slice A: the write path (DG separation, CA1 surprise, the
two-stage noise gate — Slice B), the router path (Slice C), the H.M. protocol
(Slice D), consolidation/eviction scheduling (Slice E). The store is the
substrate they all need; it is built first so each can be tested against it.

Persistence is a property PER COMPONENT (ADR-007 Decision 8, founder,
2026-09-12): "survives the session boundary" means after a full process stop
and relaunch, not within one process. Each component declares its lifecycle
(`lifecycle_declaration()`):
  structure          -- the append-only log (`journal.jsonl`): entries, states,
                        saliences, links, references, phases;
  bytes              -- the content-addressed payload directory next to the log
                        (`journal.payloads/<pointer>`), one file per payload,
                        released at the last reference;
  index              -- recomputed at replay from the log with a declared seed
                        (`JournalPath(journal, seed)`), aliases from the links;
  associative memory -- CA3 (write path) and the persistent memory (lifecycle)
                        rebuilt from their own events in the log today; checkpointed
                        with the weights once the C2 layer takes over.
At rest, log lines and payload files are encrypted per scope with the hub's
QJE1 framing (AES-256-GCM; `cortex_c2b.crypto`) when a key is given; the key is
an explicit argument, never read implicitly (key management deferred, hub 010).

Written from scratch (our filon). ANN over cues is Slice C's job; Slice A uses
an exact keyed index so the read law is enforced without a heavy dependency.
"""
from __future__ import annotations

import errno
import json
import shutil
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from cortex_data import _hash_obj  # content hashing reused (ADR-005)
from cortex_c2b.crypto import (seal, open_sealed, is_sealed, sealed_overhead, seal_line, open_line,
                               MAGIC)

# --- lifecycle states (RES-17 cell) -----------------------------------------
STATE_LIVE = "live"
STATE_CONSOLIDATED = "consolidated"
STATE_DEMOTED = "demoted"
STATE_EVICTED = "evicted"
_TRANSITIONS = {
    STATE_LIVE: {STATE_CONSOLIDATED, STATE_EVICTED},
    STATE_CONSOLIDATED: {STATE_DEMOTED, STATE_EVICTED},
    STATE_DEMOTED: {STATE_EVICTED},
    STATE_EVICTED: set(),
}

SCHEMA_SUMMARY = "summary"      # the schema_id of a demotion summary (Slice E)

# --- storage policy (ADR-007 Decision 9) --------------------------------------
POLICY_STOP = "stop"            # a storage failure raises StorageExhausted; the caller decides (measurement runs)
POLICY_READ_ONLY = "read_only"  # writes are refused (JournalReadOnly), reads continue, the state stays durable
POLICY_MEMORY = "memory"        # writes continue in a bounded spill buffer, flushed when the disk returns; declared not durable
POLICIES = (POLICY_STOP, POLICY_READ_ONLY, POLICY_MEMORY)
MODE_DURABLE, MODE_READ_ONLY, MODE_MEMORY = "durable", "read_only", "memory"


class StorageDegraded(OSError):
    """Base: the journal could not persist. Never silent (invariant 1)."""


class StorageExhausted(StorageDegraded):
    """Policy `stop`: the journal is consistent, nothing half written, the caller decides."""


class JournalReadOnly(StorageDegraded):
    """Policy `read_only` (or a full spill buffer): this write is refused; reads continue."""

ENTRY_MAX_BYTES = 1024          # the size law
CUE_DIM = 64                    # fixed-size cue (matryoshka-truncatable later)


def content_hash(payload: bytes) -> str:
    """Pointer = content hash of the payload (never the payload itself)."""
    import hashlib
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class Entry:
    """One journal entry — the ≤1 KB retention-law object."""
    cue: list[float]            # fixed-size address embedding (CUE_DIM floats)
    pointer: str                # content hash of the payload (in the payload store)
    salience: float
    schema_id: str              # span-contract tag (RES-8)
    t_written: float
    t_last_read: float
    state: str = STATE_LIVE
    entry_id: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    def size_bytes(self) -> int:
        return len(self.to_json().encode("utf-8"))


class PayloadStore:
    """Content-addressed payload storage. Same bytes → same hash → stored once.
    The journal never holds content; it holds pointers here.

    `directory` (optional): one file per payload, named by its pointer, next to
    the journal log -- the bytes' own lifecycle (Decision 8). With a `key` each
    file is sealed (QJE1 binary framing, the pointer as associated data, so a
    file cannot be swapped under another pointer). Without a directory the
    store is memory only -- and says so in `lifecycle_declaration()`."""

    def __init__(self, directory: str | Path | None = None, key: bytes | None = None):
        self._blobs: dict[str, bytes] = {}          # plaintext cache
        self._sizes: dict[str, int] = {}            # pointer -> plaintext size, known without reading
        self._pending: set[str] = set()             # registered in memory, not yet on disk (memory mode)
        self._dir = Path(directory) if directory else None
        self._key = key
        if self._dir:
            self._dir.mkdir(parents=True, exist_ok=True)                 # may raise: the journal applies its policy
            for f in self._dir.iterdir():
                if f.is_file() and len(f.name) == 16:
                    with open(f, "rb") as fh:
                        head = fh.read(len(MAGIC))
                    self._sizes[f.name] = f.stat().st_size - (sealed_overhead() if head == MAGIC else 0)

    def put(self, payload: bytes, durable: bool = True) -> str:
        """Durable first: the file reaches the disk before the pointer is registered,
        so a failure leaves nothing half registered (Decision 9, invariant 4).
        `durable=False` registers in memory only and marks the pointer pending
        (memory mode); `flush_pending()` writes it later."""
        h = content_hash(payload)
        if h in self._sizes:
            self._blobs.setdefault(h, payload)
            return h
        if self._dir and durable:
            self._write_file(h, payload)                                 # may raise OSError
        elif self._dir:
            self._pending.add(h)
        self._sizes[h] = len(payload)
        self._blobs[h] = payload
        return h

    def _write_file(self, h: str, payload: bytes) -> None:
        blob = seal(self._key, payload, h.encode()) if self._key else payload
        tmp = self._dir / (h + ".tmp")
        try:
            tmp.write_bytes(blob); tmp.replace(self._dir / h)           # never a half-written file
        except OSError:
            tmp.unlink(missing_ok=True); raise

    def flush_pending(self) -> int:
        """Write every pending payload (memory mode recovery). Raises on the first failure."""
        n = 0
        for h in sorted(self._pending):
            self._write_file(h, self._blobs[h]); self._pending.discard(h); n += 1
        return n

    @property
    def pending(self) -> int:
        return len(self._pending)

    def get(self, pointer: str) -> bytes:
        if pointer in self._blobs:
            return self._blobs[pointer]
        if pointer not in self._sizes or not self._dir:
            raise KeyError(pointer)
        blob = (self._dir / pointer).read_bytes()
        if is_sealed(blob):
            if self._key is None:
                raise ValueError(f"payload {pointer} is sealed and no key was given")
            payload = open_sealed(self._key, blob, pointer.encode())   # InvalidTag on tampering / wrong key
        else:
            payload = blob                                              # legacy clear file, coexists
        if content_hash(payload) != pointer:
            raise ValueError(f"integrity: payload {pointer} does not match its pointer")
        self._blobs[pointer] = payload
        return payload

    def release(self, pointer: str) -> bool:
        """Drop the bytes behind a pointer. Called by the journal only when the
        last non-evicted reference is gone (Slice E, reference counting)."""
        existed = pointer in self._sizes
        self._blobs.pop(pointer, None); self._sizes.pop(pointer, None); self._pending.discard(pointer)
        if self._dir:
            try:
                (self._dir / pointer).unlink(missing_ok=True)
            except OSError:
                pass                                                     # a release that cannot delete is retried at the next open
        return existed

    def __contains__(self, pointer: str) -> bool:
        return pointer in self._sizes

    def __len__(self):
        return len(self._sizes)

    def total_bytes(self) -> int:
        return sum(self._sizes.values())

    @property
    def persistent(self) -> bool:
        return self._dir is not None

    @property
    def sealed(self) -> bool:
        return self._key is not None


class Journal:
    """Append-only journal with a keyed index. Reads never scan.

    `path` (optional): an append-only JSONL file; every write appends one line,
    nothing is rewritten (the ledger discipline). State transitions are appended
    as events, so the file is a full provenance log (RES-17: the object's history).

    Decision 8 -- sealed by default: an on-disk journal needs a `key` (32 bytes);
    `plaintext=True` declares a test scope explicitly; a memory-only journal
    (no path) needs neither. Opening without either fails loudly.

    Decision 9 -- durable first, and a declared storage policy per scope:
    every mutation reaches the log (and the payload its file) BEFORE memory
    changes, so a failure leaves no half state. When the disk fails (no space,
    no permission, or free space under the declared floor), `policy` decides:
    `stop` raises StorageExhausted; `read_only` refuses the write and keeps
    reading; `memory` keeps writing into a bounded spill buffer, declared not
    durable, flushed with a `storage_restored` event when the disk returns.
    The policy, the floor, the sealing and the spill bound are the scope's
    configuration (`scope_config_hash()`).
    """

    def __init__(self, path: str | Path | None = None, cue_dim: int = CUE_DIM,
                 key: bytes | None = None, payload_dir: str | Path | None = None,
                 plaintext: bool = False, policy: str = POLICY_READ_ONLY,
                 disk_free_floor_bytes: int | None = None, spill_max_events: int = 10_000,
                 disk_check_every: int = 16):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        self.cue_dim = cue_dim
        self._path = Path(path) if path else None
        if self._path is not None and key is None and not plaintext:
            raise ValueError("an on-disk journal is sealed by default (ADR-007 D8): pass key=<32 bytes>, "
                             "or plaintext=True to declare a test scope explicitly")
        self._key = key
        self.plaintext = bool(plaintext) if self._path is not None else False
        self.policy = policy
        self.disk_free_floor_bytes = disk_free_floor_bytes
        self.spill_max_events = int(spill_max_events)
        self._disk_check_every = max(1, int(disk_check_every))
        self._persist_count = 0
        # ---- storage state (Decision 9) ----
        self.mode = MODE_DURABLE if self._path is None else MODE_DURABLE
        self.degraded_reason: str | None = None
        self.spill: deque[str] = deque()             # encoded lines not yet on disk (memory mode); a deque: O(1) at both ends
        self.refused_writes = 0                      # read_only mode: mutations refused since the failure
        self.gap_since: float | None = None          # logical time of the first failure of the current gap
        self._last_now: float | None = None
        self.restorations: list[dict] = []           # storage_restored records (this object and the log)
        # ---- the store ----
        if payload_dir is None and self._path is not None:
            payload_dir = self._path.with_name(self._path.stem + ".payloads")   # the bytes, next to the log
        self._entries: dict[str, Entry] = {}         # entry_id -> Entry
        self._by_cue_key: dict[str, list[str]] = {}  # cue key -> entry ids (keyed index)
        self._reads_touched = 0                      # instrumentation for the no-scan test
        # ---- what the associative memories and the scheduler rebuild from (Decision 8) ----
        self.ca3_order: list[tuple[str, dict]] = []  # (entry_id, meta) in write-path admission order
        self.consolidated_order: list[str] = []      # entry ids in consolidation order
        self.sentinels_recorded: list[tuple[int, str]] = []   # (fact index, entry_id)
        self.phase_count = 0
        self.refused_count = 0
        self.writes_since_last_phase: list[str] = []
        # ---- Slice E state, every item rebuilt from the log on replay ----
        self._refs: dict[str, set[str]] = {}         # pointer -> non-evicted entry ids referencing it
        self.summary_sources: dict[str, list[str]] = {}   # summary id -> its K source ids
        self.summary_of: dict[str, str] = {}         # source id -> summary id
        self.summary_level: dict[str, int] = {}      # summary id -> hierarchy level (1 = of entries)
        self.contracted: set[str] = set()            # ids credited under a contract (never decayed)
        # ---- open: the payload directory, the log, the writability probe ----
        try:
            self.payloads = PayloadStore(payload_dir, key)
        except OSError as e:
            self.payloads = PayloadStore(None, key)                    # memory only until the disk answers
            self._on_open_failure(e)
        else:
            if self._path is not None:
                try:
                    self._probe_writable()
                except OSError as e:
                    self._on_open_failure(e)                           # the files stay readable
        if self._path and self._path.exists():
            self._replay()
            self._collect_orphans()

    # ---- scope configuration -----------------------------------------------------
    def scope_config_hash(self) -> str:
        """Whatever changes the journal's behaviour on disk is in the hash (Decision 9)."""
        return _hash_obj({"policy": self.policy, "sealed": self._key is not None, "plaintext": self.plaintext,
                          "disk_free_floor_bytes": self.disk_free_floor_bytes,
                          "spill_max_events": self.spill_max_events, "cue_dim": self.cue_dim,
                          "on_disk": self._path is not None})

    # ---- keyed index: an exact bucket over a quantised cue (sub-linear reads) ----
    @staticmethod
    def _cue_key(cue) -> str:
        c = np.asarray(cue, dtype=np.float32)
        # coarse quantisation of the cue -> a bucket key; exact, cheap, sub-linear.
        # Slice C replaces this with ANN over cues; the contract (never scan) holds.
        return _hash_obj(np.round(c[:8], 1).tolist())

    # ---- write ----------------------------------------------------------------
    def write(self, cue, payload: bytes, salience: float, schema_id: str,
              now: float | None = None, ca3: dict | None = None) -> Entry:
        """Durable first: payload file, then the log line, then memory. `ca3` (the
        write path's separation meta) rides on the write event so CA3 is rebuilt
        from one line, never from two that could be split by a failure."""
        cue = np.asarray(cue, dtype=np.float32)
        assert cue.shape == (self.cue_dim,), f"cue must be ({self.cue_dim},)"
        now = time.time() if now is None else now
        self._last_now = now
        if self.mode == MODE_READ_ONLY:
            self._retry_or_refuse()                                       # the disk may be back
        pointer = self._put_payload(payload)                              # content stored once, by hash
        entry = Entry(cue=[round(float(x), 4) for x in cue], pointer=pointer,
                      salience=float(salience), schema_id=schema_id,
                      t_written=now, t_last_read=now, state=STATE_LIVE)
        entry.entry_id = _hash_obj({"p": pointer, "t": now, "s": schema_id})
        sz = entry.size_bytes()
        if sz > ENTRY_MAX_BYTES:
            raise ValueError(f"entry violates the size law: {sz} B > {ENTRY_MAX_BYTES} B")
        ev = {"ev": "write", "entry": asdict(entry)}
        if ca3 is not None:
            ev["ca3"] = dict(ca3)
        self._persist(ev)                                                 # may raise; memory untouched then
        self._entries[entry.entry_id] = entry
        self._by_cue_key.setdefault(self._cue_key(cue), []).append(entry.entry_id)
        self._refs.setdefault(pointer, set()).add(entry.entry_id)
        self.writes_since_last_phase.append(entry.entry_id)
        if ca3 is not None:
            self.ca3_order.append((entry.entry_id, dict(ca3)))
        return entry

    def _put_payload(self, payload: bytes) -> str:
        if self.mode == MODE_MEMORY:
            return self.payloads.put(payload, durable=False)
        try:
            self._check_disk_floor()
            return self.payloads.put(payload, durable=True)
        except OSError as e:
            self._on_storage_failure(e)                                   # raises for stop / read_only
            return self.payloads.put(payload, durable=False)             # memory policy: pending

    # ---- read (never a scan) --------------------------------------------------
    def read_by_cue(self, cue, now: float | None = None) -> list[Entry]:
        """Retrieve entries whose cue falls in the same bucket. Touches only that
        bucket -- never the whole journal. `_reads_touched` counts entries visited
        so the no-scan test can assert sub-linear behaviour."""
        key = self._cue_key(np.asarray(cue, dtype=np.float32))
        ids = self._by_cue_key.get(key, [])
        now = time.time() if now is None else now
        out = []
        for i in ids:
            e = self._entries[i]
            self._reads_touched += 1
            if e.state != STATE_EVICTED:
                e.t_last_read = now
                out.append(e)
        return out

    def get(self, entry_id: str) -> Entry:
        self._reads_touched += 1
        return self._entries[entry_id]

    # ---- lifecycle (RES-17 transitions) --------------------------------------
    def transition(self, entry_id: str, new_state: str) -> Entry:
        e = self._entries[entry_id]
        if new_state not in _TRANSITIONS[e.state]:
            raise ValueError(f"illegal transition {e.state} -> {new_state}")
        self._persist({"ev": "transition", "entry_id": entry_id, "to": new_state})
        e.state = new_state
        if new_state == STATE_CONSOLIDATED:
            self.consolidated_order.append(entry_id)
        if new_state == STATE_EVICTED:
            self._release_refs(entry_id)
            for src in self.summary_sources.get(entry_id, []):      # cascade: a summary takes its
                if self._entries[src].state != STATE_EVICTED:       # sources with it -- a demoted
                    self.transition(src, STATE_EVICTED)              # source is never left unreachable
        return e

    # ---- Slice E primitives: each one an event, so the log replays them ------
    def set_salience(self, entry_id: str, value: float, why: str = "credit",
                     now: float | None = None) -> Entry:
        """Salience is lifecycle state: every change is an event (invariant 1).
        `why == "contract"` marks the entry as contracted: scheduled decay never
        touches it."""
        e = self._entries[entry_id]
        value = float(min(1.0, max(0.0, value)))
        self._persist({"ev": "salience", "entry_id": entry_id, "to": value, "why": why})
        e.salience = value
        if why == "contract":
            self.contracted.add(entry_id)
        return e

    def decay(self, factor: float, states=(STATE_CONSOLIDATED, STATE_DEMOTED)) -> int:
        """Scheduled decay (ADR-007 D1) as ONE event; contracted entries exempt."""
        self._persist({"ev": "decay", "factor": float(factor), "states": list(states)})
        return self._apply_decay(factor, tuple(states))

    def _apply_decay(self, factor: float, states) -> int:
        n = 0
        for eid, e in self._entries.items():
            if e.state in states and eid not in self.contracted:
                e.salience = round(e.salience * factor, 6); n += 1
        return n

    def demote(self, summary_id: str, source_ids: list[str], level: int = 1) -> None:
        """Record the link summary <- sources (ADR-007 D4: keeping pointers). The
        summary holds a reference to every source payload, so those bytes live
        as long as the summary does. The state transitions of the sources are
        separate `transition` events (invariant 1)."""
        if summary_id not in self._entries:
            raise KeyError(summary_id)
        for sid in source_ids:
            if sid in self.summary_of:
                raise ValueError(f"{sid} already belongs to summary {self.summary_of[sid]}")
        self._persist({"ev": "demote", "summary": summary_id, "sources": list(source_ids),
                       "level": int(level)})
        self._link_summary(summary_id, list(source_ids), level)

    def _link_summary(self, summary_id: str, source_ids: list[str], level: int) -> None:
        self.summary_sources[summary_id] = source_ids
        self.summary_level[summary_id] = level
        for sid in source_ids:
            self.summary_of[sid] = summary_id
            self._refs.setdefault(self._entries[sid].pointer, set()).add(summary_id)

    def _release_refs(self, entry_id: str) -> None:
        """Reference counting on identity (the pointer), never on similarity.
        A payload is released when no non-evicted entry (or summary) points to it."""
        e = self._entries[entry_id]
        ptrs = [e.pointer] + [self._entries[s].pointer for s in self.summary_sources.get(entry_id, [])]
        for p in ptrs:
            holders = self._refs.get(p)
            if holders is None:
                continue
            holders.discard(entry_id)
            if not holders:
                del self._refs[p]
                self.payloads.release(p)

    def phase_event(self, record: dict) -> None:
        """Telemetry snapshot of a lifecycle phase, appended to the log. Replay only
        counts it (phases, refusals, writes since the last phase): it describes
        state, it does not change it."""
        self._persist({"ev": "phase", **record})
        self._count_phase(record)

    def _count_phase(self, record: dict) -> None:
        self.phase_count += 1
        if record.get("refused"):
            self.refused_count += 1
        self.writes_since_last_phase = []

    def mark_ca3(self, entry_id: str, meta: dict) -> None:
        """Kept for logs written before 2026-09-12; the write path now puts the
        meta on the write event itself (one line, never split by a failure)."""
        self._persist({"ev": "ca3", "entry_id": entry_id, **meta})
        self.ca3_order.append((entry_id, dict(meta)))

    def mark_sentinel(self, index: int, entry_id: str) -> None:
        self._persist({"ev": "sentinel", "i": int(index), "entry_id": entry_id})
        self.sentinels_recorded.append((int(index), entry_id))

    def references(self, pointer: str) -> set[str]:
        return set(self._refs.get(pointer, ()))

    def regulated_bytes(self) -> int:
        """The bytes the budget regulates: non-evicted entries + the payloads alive.
        Evicted entries are tombstones in memory; the append-only log is
        provenance and grows by design (its retention is a separate decision)."""
        return (sum(e.size_bytes() for e in self._entries.values() if e.state != STATE_EVICTED)
                + self.payloads.total_bytes())

    # ---- introspection (RES-17 snapshot) -------------------------------------
    def snapshot(self) -> dict:
        counts = {s: 0 for s in _TRANSITIONS}
        for e in self._entries.values():
            counts[e.state] += 1
        return {"entries": len(self._entries), "payloads": len(self.payloads),
                "payload_bytes": self.payloads.total_bytes(),
                "journal_bytes": sum(e.size_bytes() for e in self._entries.values()),
                "regulated_bytes": self.regulated_bytes(),
                "summaries": len(self.summary_sources), "contracted": len(self.contracted),
                "storage_mode": self.mode, "unpersisted_events": len(self.spill),
                **counts}

    def __len__(self):
        return len(self._entries)

    # ---- storage: durable first, then the declared policy (Decision 9) ----------
    def disk_free(self) -> int | None:
        if self._path is None:
            return None
        try:
            return int(shutil.disk_usage(self._path.parent).free)
        except OSError:
            return None

    def _check_disk_floor(self) -> None:
        """The proactive wall: under the declared floor, act as if the disk were full."""
        if self.disk_free_floor_bytes is None or self._path is None:
            return
        self._persist_count += 1
        if self._persist_count % self._disk_check_every != 1 and self._disk_check_every > 1:
            return
        free = self.disk_free()
        if free is not None and free < self.disk_free_floor_bytes:
            raise OSError(errno.ENOSPC, f"free space {free} B under the declared floor {self.disk_free_floor_bytes} B")

    def _probe_writable(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8"):
            pass

    def _encode(self, ev: dict) -> str:
        line = json.dumps(ev, separators=(",", ":"))
        return seal_line(self._key, line) if self._key is not None else line   # QJE1, per line, as on the hub

    def _write_line(self, line: str) -> None:
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _persist(self, ev: dict) -> None:
        """The line reaches the disk before memory changes. On failure the policy
        applies: stop raises, read_only refuses, memory spills (bounded)."""
        if self._path is None:
            return
        line = self._encode(ev)
        if self.mode == MODE_MEMORY:
            try:
                self._flush_spill()                                       # the disk may be back
            except OSError:
                self._spill(line)
                return
        try:
            self._check_disk_floor()
            self._write_line(line)
        except OSError as e:
            self._on_storage_failure(e)                                   # raises for stop / read_only
            self._spill(line)                                             # memory policy
            return
        if self.mode == MODE_READ_ONLY:
            self._restored(MODE_READ_ONLY)

    def _on_storage_failure(self, e: OSError) -> None:
        reason = f"{e.strerror or e}"
        if self.policy == POLICY_STOP:
            raise StorageExhausted(e.errno or errno.EIO, f"storage failed, policy stop: {reason}") from e
        if self.gap_since is None:
            self.gap_since = self._last_now
            self.degraded_reason = reason
        if self.policy == POLICY_READ_ONLY:
            self.mode = MODE_READ_ONLY
            self.refused_writes += 1
            raise JournalReadOnly(e.errno or errno.EIO, f"journal read only ({reason}); this write is refused") from e
        self.mode = MODE_MEMORY                                           # policy memory: the caller spills

    def _spill(self, line: str) -> None:
        if len(self.spill) >= self.spill_max_events:
            self.mode = MODE_READ_ONLY
            self.refused_writes += 1
            raise JournalReadOnly(errno.ENOSPC, f"spill buffer full ({self.spill_max_events} events); this write is refused")
        self.spill.append(line)

    def _retry_or_refuse(self) -> None:
        """read_only mode, a new write: probe the disk once; if it answers, resume."""
        try:
            self._check_disk_floor()
            self._probe_writable()
        except OSError as e:
            self.refused_writes += 1
            raise JournalReadOnly(e.errno or errno.EIO, f"journal read only ({self.degraded_reason}); this write is refused") from e

    def _flush_spill(self) -> None:
        """Memory mode recovery: pending payloads, then the spilled lines in order, then
        the restoration record. Raises on the first failure, nothing lost."""
        self._check_disk_floor()
        self.payloads.flush_pending()
        while self.spill:
            self._write_line(self.spill[0]); self.spill.popleft()   # O(1); a list.pop(0) here was O(n) per event
        self._restored(MODE_MEMORY)

    def _restored(self, from_mode: str) -> None:
        rec = {"ev": "storage_restored", "from_mode": from_mode, "reason": self.degraded_reason,
               "since": self.gap_since, "until": self._last_now,
               "refused_writes": self.refused_writes if from_mode == MODE_READ_ONLY else 0}
        self.mode = MODE_DURABLE
        self.degraded_reason, self.gap_since, self.refused_writes = None, None, 0
        try:
            self._write_line(self._encode(rec))
        except OSError:
            pass                                                          # the disk went again; the next write will see it
        self.restorations.append(rec)

    def _on_open_failure(self, e: OSError) -> None:
        reason = f"{e.strerror or e}"
        if self.policy == POLICY_STOP:
            raise StorageExhausted(e.errno or errno.EACCES, f"cannot open the journal on disk, policy stop: {reason}") from e
        self.degraded_reason = reason
        self.gap_since = None
        if self.policy == POLICY_READ_ONLY:
            if not (self._path.exists() and os_readable(self._path)):
                raise JournalReadOnly(e.errno or errno.EACCES, f"cannot open the journal and nothing to read: {reason}") from e
            self.mode = MODE_READ_ONLY
        else:
            self.mode = MODE_MEMORY

    def _collect_orphans(self) -> None:
        """A payload file no entry references (a write interrupted between the file
        and the line) is garbage, released at open."""
        for p in [p for p in list(self.payloads._sizes) if p not in self._refs]:
            self.payloads.release(p)

    # ---- append-only persistence ---------------------------------------------
    def _replay(self):
        """Rebuild in-memory state from the append-only log: entries, states,
        saliences, summary links, payload references, the CA3 and sentinel
        events, the phase counters. Payload BYTES are not in the log (the size
        law keeps content out of it): they live in the payload directory next
        to it, opened by `PayloadStore` (Decision 8). Before 2026-09-12 this
        docstring claimed the payloads were re-registered from the log; they
        were not -- recorded here so the correction stays visible."""
        for raw in self._path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            ev = json.loads(open_line(self._key, raw))                 # a clear line passes, a sealed one needs the key
            if ev["ev"] == "write":
                d = ev["entry"]
                e = Entry(**d)
                self._entries[e.entry_id] = e
                self._by_cue_key.setdefault(self._cue_key(e.cue), []).append(e.entry_id)
                self._refs.setdefault(e.pointer, set()).add(e.entry_id)
                self.writes_since_last_phase.append(e.entry_id)
                if "ca3" in ev:
                    self.ca3_order.append((e.entry_id, dict(ev["ca3"])))
            elif ev["ev"] == "transition":
                self._entries[ev["entry_id"]].state = ev["to"]
                if ev["to"] == STATE_CONSOLIDATED:
                    self.consolidated_order.append(ev["entry_id"])
                if ev["to"] == STATE_EVICTED:
                    self._release_refs(ev["entry_id"])
            elif ev["ev"] == "salience":
                self._entries[ev["entry_id"]].salience = float(ev["to"])
                if ev.get("why") == "contract":
                    self.contracted.add(ev["entry_id"])
            elif ev["ev"] == "decay":
                self._apply_decay(float(ev["factor"]), tuple(ev["states"]))
            elif ev["ev"] == "demote":
                self._link_summary(ev["summary"], list(ev["sources"]), int(ev.get("level", 1)))
            elif ev["ev"] == "phase":
                self._count_phase(ev)                     # telemetry: counted, not applied
            elif ev["ev"] == "ca3":
                self.ca3_order.append((ev["entry_id"], {k: v for k, v in ev.items() if k not in ("ev", "entry_id")}))
            elif ev["ev"] == "sentinel":
                self.sentinels_recorded.append((int(ev["i"]), ev["entry_id"]))
            elif ev["ev"] == "storage_restored":
                self.restorations.append({k: v for k, v in ev.items() if k != "ev"})


def os_readable(path: Path) -> bool:
    try:
        with open(path, "rb"):
            return True
    except OSError:
        return False


def lifecycle_declaration(journal: "Journal | None" = None) -> dict:
    """Decision 8: persistence is a property per component; each declares its own.
    With a journal, the declaration reflects that journal's actual setup."""
    has = journal is not None                                      # never `if journal`: an empty journal is falsy
    on_disk = has and journal._path is not None
    sealed = has and journal._key is not None
    durable = on_disk and journal.mode != MODE_MEMORY               # a memory-mode journal never claims (invariant 1)
    storage = {"mode": journal.mode if has else None, "policy": journal.policy if has else None,
               "degraded_reason": journal.degraded_reason if has else None,
               "unpersisted_events": len(journal.spill) if has else 0,
               "plaintext_scope": has and journal.plaintext}
    return {
        "storage": storage,
        "structure": {"what": "entries, states, saliences, links, references, phases",
                      "lifecycle": "append-only log, replayed at open",
                      "survives_restart": durable, "encrypted_at_rest": sealed},
        "bytes": {"what": "the payloads (content), one file per pointer",
                  "lifecycle": "content-addressed directory next to the log, released at the last reference",
                  "survives_restart": has and journal.payloads.persistent and durable,
                  "encrypted_at_rest": has and journal.payloads.sealed},
        "index": {"what": "the cue index (buckets) and the demotion aliases",
                  "lifecycle": "recomputed at open from the log, with a declared seed",
                  "survives_restart": durable, "encrypted_at_rest": None},
        "associative_memory": {"what": "CA3 codes (write path) and the persistent memory (lifecycle)",
                               "lifecycle": "rebuilt at open from their own events (ca3, consolidation order); "
                                            "checkpointed with the weights once the C2 layer takes over -- the "
                                            "language model arm re declares this component (ADR-007 D8, named boundary)",
                               "survives_restart": durable, "encrypted_at_rest": None},
    }
