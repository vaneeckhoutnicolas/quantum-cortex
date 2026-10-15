"""cortex_c2b.write_path — the hippocampal write path (ADR-007 D2), Slice B.

    separate (DG) → associate (CA3) → compare (CA1) → gate

Each stage is a flag (default on here; ablatable to off) so that interference
without DG, recall without CA3, and write precision without CA1 are each a row.

  DG  — pattern separation: expand-then-sparsify the cue so near-duplicates
        decorrelate before storage (the interference guard). A fixed random
        expansion to a larger space followed by k-winners-take-all: two cues that
        are close in input space land on mostly-different sparse codes.
  CA3 — pattern completion: reconstruct the input from what the memory already
        holds. Slice B uses the journal's own stored cues as the associative
        memory (nearest stored cue = the reconstruction); Slice C swaps in the
        C2 Hopfield/delta layer of ADR-006 (C2 *is* CA3) behind the same call.
  CA1 — the comparator: surprise = distance(reconstruction, input). Surprise is
        no longer a heuristic; it is a computed quantity at the memory interface
        (RES-2 made concrete). Feeds salience and the write gate.
  gate — the noise ruling (D4.8), TWO stages, never one detector:
        (a) admission: reject the redundant/already-known (surprise below a
            declared threshold) — but surprise alone is not signal: a corrupted
            input is maximally surprising and would be journaled enthusiastically;
            under the journal's identity rule a different source text is not
            redundancy;
        (b) the verdict is retrospective: an admitted entry that never gains
            salience, never consolidates, never earns outcome credit is NOISE and
            is evicted by the tribunal (`noise_tribunal`). Scheduled forgetting is
            the judge. Slice B implements the tribunal's rule; the schedule (RES-9)
            is Slice E.

The admission rule is a property of the journal, not of a neighbour (row 49).
`surprise` is the library's default and the inherited rule of every journal
already sealed. `identity` is asked for at construction: since Decision 12
(row 49, letter alpha, 2026-10-15) it is the rule of every new journal of the
record, declared by the run's configuration or argument and stamped in the
journal, the library's default staying `surprise` as ADR-008 D1 A kept the
index geometry's (ADR-007, the measured paragraph of 2026-10-15 says why). A
journal opened under another rule than the one it records is refused.

Written from scratch (our filon). No torch here — pure numpy over the store.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cortex_c2b import (Journal, Entry, STATE_LIVE, STATE_EVICTED, CUE_DIM, content_hash,
                        ADMISSION_SURPRISE, ADMISSION_IDENTITY)


# ---------------------------------------------------------------------------- #
# DG — pattern separation                                                      #
# ---------------------------------------------------------------------------- #
class DentateGyrus:
    """Expand-then-sparsify: project to a larger space with a fixed random map,
    keep the k largest activations (k-WTA), renormalise. Near-duplicates
    decorrelate; the separation is measurable as the drop in cosine similarity."""

    def __init__(self, in_dim: int = CUE_DIM, expand: int = 4, sparsity: float = 0.1, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.out_dim = in_dim * expand
        self.W = rng.standard_normal((in_dim, self.out_dim)).astype(np.float32) / np.sqrt(in_dim)
        self.k = max(1, int(self.out_dim * sparsity))

    def separate(self, cue: np.ndarray) -> np.ndarray:
        z = cue.astype(np.float32) @ self.W
        idx = np.argpartition(-np.abs(z), self.k)[: self.k]          # k winners
        out = np.zeros_like(z); out[idx] = z[idx]
        n = np.linalg.norm(out)
        return out / n if n > 0 else out


# ---------------------------------------------------------------------------- #
# CA3 — pattern completion (Slice B: from the journal's stored cues)            #
# ---------------------------------------------------------------------------- #
class CA3Completion:
    """Reconstruct from memory: return the stored code nearest to the input.
    Slice C replaces this with the C2 associative layer behind the same call.
    Since 2026-10-15 each stored code keeps the id of the entry it belongs to
    (`entry_ids`, None when unknown), so that a nearest code names its entry
    from this memory alone, whatever other write path shares the journal."""

    def __init__(self):
        self._codes: list[np.ndarray] = []
        self.entry_ids: list[str | None] = []

    def store(self, code: np.ndarray, entry_id: str | None = None):
        self._codes.append(code.astype(np.float32))
        self.entry_ids.append(entry_id)

    def nearest(self, code: np.ndarray) -> tuple[int, np.ndarray, float] | None:
        """The stored code closest to `code`: its index in this memory's admission order, the code,
        the cosine. None on an empty memory. `entry_ids[index]` is the entry the code belongs to."""
        if not self._codes:
            return None
        M = np.stack(self._codes)
        code = np.asarray(code, dtype=np.float32)
        sims = M @ code / (np.linalg.norm(M, axis=1) * (float(np.linalg.norm(code)) + 1e-8) + 1e-8)
        k = int(np.argmax(sims))
        return k, M[k], float(sims[k])

    def reconstruct(self, code: np.ndarray) -> np.ndarray | None:
        hit = self.nearest(code)
        return None if hit is None else hit[1]


# ---------------------------------------------------------------------------- #
# CA1 — the comparator: surprise as a computed quantity                        #
# ---------------------------------------------------------------------------- #
def ca1_surprise(reconstruction: np.ndarray | None, code: np.ndarray) -> float:
    """surprise = distance(reconstruction, input) in cosine terms ∈ [0, 1].
    No memory yet → maximal surprise (everything is new)."""
    if reconstruction is None:
        return 1.0
    a, b = reconstruction, code
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    return float(np.clip(1.0 - cos, 0.0, 1.0))


# ---------------------------------------------------------------------------- #
# The gated write path                                                          #
# ---------------------------------------------------------------------------- #
@dataclass
class WriteReport:
    admitted: bool
    surprise: float
    reason: str
    entry: Entry | None = None
    revised: bool = False       # ADR-007 amendment 2026-10-12: same address, different content, the entry revised in place
    versions: int = 0           # the number of versions the revised entry now carries
    admission: str = ADMISSION_SURPRISE   # the journal's rule, stamped on the write event
    # row 49 (ADR-007, amendment of 2026-10-14, evening): the comparator's quantities beside the decision
    nearest_entry_id: str | None = None   # the stored code nearest to the incoming one, before this write (None: empty memory)
    nearest_cosine: float | None = None   # its cosine with the incoming code
    near: dict | None = None              # {entry_id, surprise} stamped on the write event when an admitted sourced write
                                          # had a surprise under the legacy threshold; None otherwise
    legacy_surprise: float | None = None  # the shadow legacy gate's verdict (None when the shadow is off): the surprise
    legacy_admitted: bool | None = None   # against a memory fed with the codes the surprise rule admits, its decision,
    legacy_nearest_entry_id: str | None = None   # and the entry its nearest code belongs to (None: the shadow's memory was empty)


LEGACY_THRESHOLD = 0.15     # the legacy rule's threshold (the protocol's own since the first line of C2b): the shadow legacy gate judges
                            # at it and the `near` stamp is written under it, whatever threshold the live gate is given (committee of 2026-10-15)


class LegacyShadow:
    """The shadow legacy gate (ADR-007, amendment of 2026-10-14, evening, section 5, P3): a second completion
    memory fed only with the codes the surprise rule at 0.15 admits, judged with the same dentate on the same
    computed cue as the real gate and storing the code of the cue as journaled, so that a journal written
    under the identity rule records, write by write, what the legacy gate would have decided. The shadow
    lives in the process that plants: it is asked for on a fresh journal only (a journal that already holds
    write events refuses it, since the computed cues the legacy gate judged are not in the log, only the
    rounded ones), and the probes record its verdict in their own files at the write."""

    def __init__(self, threshold: float = LEGACY_THRESHOLD):
        self.thr = float(threshold)
        self.ca3 = CA3Completion()

    @property
    def entry_ids(self) -> list[str | None]:
        return self.ca3.entry_ids

    def judge(self, code: np.ndarray) -> tuple[float, bool, int | None]:
        """(surprise against the shadow's memory, the legacy decision, the index of the nearest stored code)."""
        hit = self.ca3.nearest(code)
        surprise = ca1_surprise(None if hit is None else hit[1], code)
        return surprise, bool(surprise >= self.thr), (None if hit is None else hit[0])

    def admit(self, stored_code: np.ndarray, entry_id: str | None) -> None:
        self.ca3.store(stored_code, entry_id)


class WritePath:
    """separate → associate → compare → gate, over a Journal.

    `admission` is the journal's rule, asked for here and refused if the journal
    already records another (the index geometry's precedent). None takes the rule the
    journal records, and the library's default, `surprise`, on a journal with no write yet.
    `identity` is asked for by the run: the record's rule for every new journal since
    Decision 12 (row 49), the library's default staying `surprise` (ADR-008 D1 A).

    `admission_threshold`: under `surprise`, a cue whose CA1 surprise is below it
    is REDUNDANT and is not admitted. Under `identity`, a write that carries a
    source text and is not the same address is admitted whatever the surprise.
    A write without a source text stays under the surprise rule on either
    journal: sentinels and summaries do not pass through this gate. Admitted
    entries start LIVE with salience = surprise. Stage (b) is `noise_tribunal()`.

    `shadow_legacy`: keep a `LegacyShadow` beside the gate and report its verdict on
    every write (the plantings of rows 49 and later); off by default, the gate's own
    decisions are the same either way; refused on a journal that already holds write
    events (the shadow judges computed cues, which the log does not keep).
    """

    def __init__(self, journal: Journal, admission_threshold: float = 0.15,
                 use_dg: bool = True, use_ca3: bool = True, use_ca1: bool = True, seed: int = 0,
                 admission: str | None = None, shadow_legacy: bool = False):
        if admission is None:                       # the journal's record, else its binding, else the code's default
            admission = journal.admission_record() or journal.admission_rule or ADMISSION_SURPRISE
        if admission not in (ADMISSION_SURPRISE, ADMISSION_IDENTITY):
            raise ValueError(f"admission must be {ADMISSION_SURPRISE!r} or {ADMISSION_IDENTITY!r}, got {admission!r}")
        self.j = journal
        self.thr = admission_threshold
        self.admission = admission
        self.use_dg, self.use_ca3, self.use_ca1 = use_dg, use_ca3, use_ca1
        self.dg = DentateGyrus(seed=seed) if use_dg else None
        self.ca3 = CA3Completion()
        if shadow_legacy and journal.ca3_order:
            raise ValueError("shadow_legacy: the shadow legacy gate is asked for on a fresh journal only; this journal already holds "
                             f"{len(journal.ca3_order)} admitted writes whose computed cues the log does not keep")
        self.shadow: LegacyShadow | None = LegacyShadow() if shadow_legacy else None
        # Decision 8 (persistence per component): CA3 is rebuilt from the journal's
        # `ca3` events, in admission order, with the same separation. A journal
        # written under another separation is refused, never silently re-coded.
        self._meta = {"dg": bool(use_dg), "seed": int(seed)}
        for eid, meta in journal.ca3_order:
            if {"dg": meta.get("dg"), "seed": meta.get("seed")} != self._meta:
                raise ValueError(f"CA3 rebuild: journal written with separation {meta}, this write path has {self._meta}")
            self.ca3.store(self._code(np.asarray(journal._entries[eid].cue, dtype=np.float32)), eid)
        journal.bind_admission(admission)          # the journal's rule, or a refusal: never a silent default

    def _code(self, cue: np.ndarray) -> np.ndarray:
        return self.dg.separate(cue) if self.dg else cue.astype(np.float32)

    def _entry_at(self, index: int | None) -> str | None:
        """The entry whose code sits at `index` of this write path's CA3."""
        if index is None or index >= len(self.ca3.entry_ids):
            return None
        return self.ca3.entry_ids[index]

    def write(self, cue, payload: bytes, schema_id: str, now: float | None = None,
              source: bytes | str | None = None) -> WriteReport:
        """`source` (ADR-009 B2): the text the cue was computed from; the store seals it
        beside the content. Same text: revision or redundancy, on either rule.
        Different text: admitted under `identity`, judged by the surprise under
        `surprise`. No text: the surprise rule, on either journal."""
        cue = np.asarray(cue, dtype=np.float32)
        same = self.j.at_address(cue, source)
        if same is not None:
            if content_hash(payload) == same.pointer:
                return WriteReport(admitted=False, surprise=0.0,
                                   reason="redundant (same address, same content)",
                                   entry=same, admission=self.j.admission_rule)
            e = self.j.revise(same.entry_id, payload, now=now)
            n = len(self.j.history(e.entry_id))
            return WriteReport(admitted=False, surprise=0.0,
                               reason=f"revised (same address, different content): version {n}",
                               entry=e, revised=True, versions=n, admission="revision")
        code = self._code(cue)
        hit = self.ca3.nearest(code) if self.use_ca3 else None
        recon = None if hit is None else hit[1]
        surprise = ca1_surprise(recon, code) if self.use_ca1 else 1.0
        nearest_id = self._entry_at(None if hit is None else hit[0])
        nearest_cos = None if hit is None else hit[2]
        rule = self.j.admission_rule
        if rule != self.admission:
            raise ValueError(f"admission: this write path was built under {self.admission!r}, the journal is bound to {rule!r}; refused")
        # the shadow legacy gate judges the same code against its own memory, before anything is stored
        legacy = None
        if self.shadow is not None:
            ls, lok, lk = self.shadow.judge(code)
            legacy = {"surprise": ls, "admitted": lok,
                      "nearest": (self.shadow.entry_ids[lk] if lk is not None else None)}
        # the rule is the journal's: a sourced write under identity is a different
        # address (the same address already returned above) and is admitted. The
        # nearest entry is not consulted.
        identity = rule == ADMISSION_IDENTITY and source is not None
        if not identity and surprise < self.thr:
            if legacy is not None and legacy["admitted"]:
                # the legacy gate would have stored this code (an unsourced write the live gate refuses against its larger memory):
                # the shadow stores the code of the cue as the journal would have kept it, with no entry behind it
                self.shadow.admit(self._code(np.asarray([round(float(x), 4) for x in cue], dtype=np.float32)), None)
            return WriteReport(admitted=False, surprise=surprise,
                               reason="redundant (below admission threshold)", admission=rule,
                               nearest_entry_id=nearest_id, nearest_cosine=nearest_cos,
                               **self._legacy_fields(legacy))
        # the comparator's verdict kept as provenance: an admitted write under the LEGACY threshold names its nearest code
        near = ({"entry_id": nearest_id, "surprise": float(surprise)}
                if (surprise < LEGACY_THRESHOLD and nearest_id is not None) else None)
        entry = self.j.write(cue, payload, salience=surprise, schema_id=schema_id, now=now,
                             ca3=self._meta, source=source, admission=rule, near=near)
        stored = self._code(np.asarray(entry.cue, dtype=np.float32))
        self.ca3.store(stored, entry.entry_id)
        if legacy is not None and legacy["admitted"]:
            self.shadow.admit(stored, entry.entry_id)        # the shadow stores the code AS JOURNALED, as CA3 does (Decision 8)
        why = "admitted (identity: different source text)" if identity else "admitted"
        return WriteReport(admitted=True, surprise=surprise, reason=why, entry=entry, admission=rule,
                           nearest_entry_id=nearest_id, nearest_cosine=nearest_cos, near=near,
                           **self._legacy_fields(legacy))

    @staticmethod
    def _legacy_fields(legacy: dict | None) -> dict:
        if legacy is None:
            return {}
        return {"legacy_surprise": float(legacy["surprise"]), "legacy_admitted": bool(legacy["admitted"]),
                "legacy_nearest_entry_id": legacy["nearest"]}

    # ---- outcome credit (RES-11 hook): salience earned after the fact ---------
    def credit(self, entry_id: str, amount: float, why: str = "credit"):
        e = self.j.get(entry_id)
        self.j.set_salience(entry_id, e.salience + amount, why=why)   # an event (Slice E, invariant 1)

    # ---- stage (b): the retrospective verdict — the noise tribunal -----------
    def noise_tribunal(self, salience_floor: float = 0.2, min_age: float = 0.0,
                       now: float | None = None) -> list[str]:
        """Evict LIVE entries that never earned salience above the floor and were
        never consolidated — the D4.8 definition: noise is what never improves
        prediction. A corrupted input (maximally surprising at admission) that
        never earns credit is exactly what this catches. Returns evicted ids."""
        import time as _t
        now = _t.time() if now is None else now
        evicted = []
        for eid in noise_candidates(self.j, salience_floor, min_age, now):
            self.j.transition(eid, STATE_EVICTED)
            evicted.append(eid)
        return evicted


def noise_candidates(journal: Journal, salience_floor: float, min_age: float, now: float) -> list[str]:
    """The D4.8 rule as a predicate, shared by the tribunal above and by the
    Slice E scheduler (which plans before it applies): LIVE entries that never
    earned salience above the floor and are old enough. Deterministic order."""
    return [eid for eid, e in sorted(journal._entries.items())
            if e.state == STATE_LIVE and e.salience < salience_floor and (now - e.t_written) >= min_age]
