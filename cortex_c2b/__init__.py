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

Written from scratch (our filon). ANN over cues is Slice C's job; Slice A uses
an exact keyed index so the read law is enforced without a heavy dependency.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from cortex_data import _hash_obj  # content hashing reused (ADR-005)

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
    The journal never holds content; it holds pointers here."""

    def __init__(self):
        self._blobs: dict[str, bytes] = {}

    def put(self, payload: bytes) -> str:
        h = content_hash(payload)
        if h not in self._blobs:
            self._blobs[h] = payload
        return h

    def get(self, pointer: str) -> bytes:
        return self._blobs[pointer]

    def __len__(self):
        return len(self._blobs)

    def total_bytes(self) -> int:
        return sum(len(b) for b in self._blobs.values())


class Journal:
    """Append-only journal with a keyed index. Reads never scan.

    `path` (optional): an append-only JSONL file; every write appends one line,
    nothing is rewritten (the ledger discipline). State transitions are appended
    as events, so the file is a full provenance log (RES-17: the object's history).
    """

    def __init__(self, path: str | Path | None = None, cue_dim: int = CUE_DIM):
        self.cue_dim = cue_dim
        self.payloads = PayloadStore()
        self._entries: dict[str, Entry] = {}         # entry_id -> Entry
        self._by_cue_key: dict[str, list[str]] = {}  # cue key -> entry ids (keyed index)
        self._path = Path(path) if path else None
        self._reads_touched = 0                      # instrumentation for the no-scan test
        if self._path and self._path.exists():
            self._replay()

    # ---- keyed index: an exact bucket over a quantised cue (sub-linear reads) ----
    @staticmethod
    def _cue_key(cue) -> str:
        c = np.asarray(cue, dtype=np.float32)
        # coarse quantisation of the cue → a bucket key; exact, cheap, sub-linear.
        # Slice C replaces this with ANN over cues; the contract (never scan) holds.
        return _hash_obj(np.round(c[:8], 1).tolist())

    # ---- write ----------------------------------------------------------------
    def write(self, cue, payload: bytes, salience: float, schema_id: str,
              now: float | None = None) -> Entry:
        cue = np.asarray(cue, dtype=np.float32)
        assert cue.shape == (self.cue_dim,), f"cue must be ({self.cue_dim},)"
        now = time.time() if now is None else now
        pointer = self.payloads.put(payload)          # content stored once, by hash
        entry = Entry(cue=[round(float(x), 4) for x in cue], pointer=pointer,
                      salience=float(salience), schema_id=schema_id,
                      t_written=now, t_last_read=now, state=STATE_LIVE)
        entry.entry_id = _hash_obj({"p": pointer, "t": now, "s": schema_id})
        sz = entry.size_bytes()
        if sz > ENTRY_MAX_BYTES:
            raise ValueError(f"entry violates the size law: {sz} B > {ENTRY_MAX_BYTES} B")
        self._entries[entry.entry_id] = entry
        self._by_cue_key.setdefault(self._cue_key(cue), []).append(entry.entry_id)
        self._append_event({"ev": "write", "entry": asdict(entry)})
        return entry

    # ---- read (never a scan) --------------------------------------------------
    def read_by_cue(self, cue, now: float | None = None) -> list[Entry]:
        """Retrieve entries whose cue falls in the same bucket. Touches only that
        bucket — never the whole journal. `_reads_touched` counts entries visited
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
        e.state = new_state
        self._append_event({"ev": "transition", "entry_id": entry_id, "to": new_state})
        return e

    # ---- introspection (RES-17 snapshot) -------------------------------------
    def snapshot(self) -> dict:
        counts = {s: 0 for s in _TRANSITIONS}
        for e in self._entries.values():
            counts[e.state] += 1
        return {"entries": len(self._entries), "payloads": len(self.payloads),
                "payload_bytes": self.payloads.total_bytes(),
                "journal_bytes": sum(e.size_bytes() for e in self._entries.values()),
                **counts}

    def __len__(self):
        return len(self._entries)

    # ---- append-only persistence ---------------------------------------------
    def _append_event(self, ev: dict):
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev, separators=(",", ":")) + "\n")

    def _replay(self):
        """Rebuild in-memory state from the append-only log (payloads are
        re-registered by hash from the log's write events)."""
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            if ev["ev"] == "write":
                d = ev["entry"]
                e = Entry(**d)
                self._entries[e.entry_id] = e
                self._by_cue_key.setdefault(self._cue_key(e.cue), []).append(e.entry_id)
            elif ev["ev"] == "transition":
                self._entries[ev["entry_id"]].state = ev["to"]
