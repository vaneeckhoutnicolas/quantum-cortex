"""cortex_c2b.lifecycle -- Slice E: consolidate, demote, evict on a rhythm (ADR-007 D4 and D6).

The journal was a store (A), a gated write path (B), a router path (C) and a
measured organ (D). Slice E makes it ALIVE: on a rhythm (RES-9, the sleep of the
model) live entries are replayed into the persistent associative memory,
consolidated entries are demoted into content-addressed summaries (RES-16),
and forgetting is scheduled (D4.8: noise is what never contributes). Budgets are
homeostatic setpoints, not limits. It is a RES-17 state machine, the same law
the router uses (`cortex_c2.consolidation`), at the memory scale.

Eight invariants, validated by the founder on 2026-09-12 before any code, each
one exercised by `tests/test_c2b_lifecycle.py`:

  1. No shortcut around the state machine. Every transition, every salience
     change (credit, contract, scheduled decay) and every demotion link is an
     event in the append only log; replaying the log rebuilds the lifecycle
     state exactly (states, saliences, links, references).
  2. Forward only, no resurrection. States move live -> consolidated ->
     demoted -> evicted; an evicted entry never scores, never reads, never
     returns. Payloads are released at the LAST pointer, by identity (the
     content hash) and never by similarity: an irreversible operation rests
     only on an exact relation.
  3. Consolidation is a replay, not a tick. An entry becomes `consolidated`
     only after it has been replayed into the persistent associative memory
     and its reconstruction verified under a declared error. Replay priority:
     salience, then primacy (RES-9).
  4. Demotion keeps access. K SIMILAR consolidated entries (same index bucket,
     pairwise cosine above a declared radius) become one summary: a content
     addressed payload holding the K pointers and a digest; the sources move
     to `demoted` and the summary is reachable from every source cue. A summary
     can be demoted in turn (the hierarchy of RES-16); no cycle is possible.
  5. Eviction is a verdict, never a size cut. Live entries are evicted only by
     the noise tribunal (Slice B rule, never modulated by pressure);
     consolidated entries only below a salience floor and old enough; demoted
     sources only with their summary (cascade). A live entry above the floor
     survives any pressure.
  6. Budgets are setpoints. Per scope, bytes (non-evicted entries + payloads
     alive) and read latency (entries touched per read, p95) are regulated
     variables; pressure in [0, 1] rises with the distance to the setpoint and
     acts on rhythms only (consolidation batch, eviction floor), never on the
     rules above.
  7. An explicit, deterministic rhythm. Phases fire on the injected logical
     clock every P admitted writes, from a hashed configuration and a seed;
     two runs with the same configuration and event sequence write identical
     logs. Every phase appends one telemetry snapshot.
  8. The H.M. check precedes the commit. A sentinel set of contracted facts
     (the frozen H.M. generator) lives in the journal; every phase is planned
     first, the sentinel probe is replayed on the plan, and the phase applies
     only if contracted recall holds and the negative control does not rise.
     Otherwise the phase is refused and recorded, pressure stays. No rollback
     exists (invariant 2), so the check comes first -- as the breaker re routes
     instead of killing. The protocol's thresholds are untouched.

  9. The journal survives a full process restart (ADR-007 Decision 8, founder,
     2026-09-12). A scheduler opened on a journal continues it: the persistent
     memory is rebuilt from the consolidation order, the sentinels are found
     again through their events, the phase and write counters resume. The test
     that counts runs session B in a new process started from the disk alone,
     sealed and clear (`test_the_journal_survives_a_full_process_restart`).

Design choices taken by the founder (2026-09-12): reference counting on
identity; the summary indexed under its source cues; refuse before commit; K,
P and the similarity radius in the hashed configuration; persistence declared
per component, sealed at rest per scope with the hub's QJE1 framing.

Pure Python over numpy; no torch. The persistent memory here is a stand-in
(`PersistentMemory`, a modern Hopfield store over cues); the C2 layer of
ADR-006 replaces it behind the same two calls when the journal is wired into
`train.py`. Written from scratch (our filon).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field

import numpy as np

from cortex_data import _hash_obj
from cortex_c2b import (Journal, Entry, STATE_LIVE, STATE_CONSOLIDATED, STATE_DEMOTED,
                        STATE_EVICTED, SCHEMA_SUMMARY, CUE_DIM, JournalReadOnly)
from cortex_c2b.write_path import WritePath, WriteReport, noise_candidates
from cortex_c2b.read_path import JournalPath
from cortex_c2b.hm_protocol import generate_facts, Fact


# ---------------------------------------------------------------------------- #
# Configuration -- hashed: whatever changes a result is in the hash (ADR-001/005) #
# ---------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LifecycleConfig:
    period_writes: int = 16          # P: one phase every P admitted writes
    consolidate_batch: int = 8       # entries replayed per phase at pressure 0 (x2 at pressure 1)
    replay_error_max: float = 0.05   # cosine distance reconstruction <-> cue that verifies a replay
    demote_k: int = 4                # K: sources per summary
    demote_min_cosine: float = 0.90  # similarity radius inside a group (pairwise cosine >= this)
    evict_floor: float = 0.20        # salience floor for consolidated entries at pressure 0
    evict_floor_max: float = 0.95    # the same floor at pressure 1
    evict_min_age: float = 0.0       # logical age (now - t_written) before a consolidated entry is a candidate
    tribunal_floor: float = 0.20     # Slice B rule for LIVE entries -- never modulated
    tribunal_min_age: float = 0.0
    decay: float = 0.90              # salience factor per phase on consolidated/demoted, contracted exempt
    bytes_setpoint: int | None = None      # regulated: non-evicted entries + payloads alive
    latency_setpoint: float | None = None  # regulated: p95 entries touched per read
    band: float = 0.25               # pressure 0 at setpoint*(1-band), 1 at the setpoint
    sentinel_facts: int = 20         # contracted facts planted at start (0 disables the check)
    sentinel_negctrl: int = 10       # never planted facts probed at every phase
    sentinel_seed: int = 0
    seed: int = 0
    withdraw_at_eviction: bool = True    # ADR-007 amendment 2026-10-01: an eviction withdraws the entry's pattern
                                         # from the persistent memory. Off until the withdrawal probe was read; on by
                                         # default since Decision 11 (2026-10-02, row 42: reading (b1) on three journals)

    def config_hash(self) -> str:
        d = asdict(self)
        if not d.get("withdraw_at_eviction"):
            d.pop("withdraw_at_eviction", None)   # the flag enters the hash only when on: the files of rows 36 to 41, produced
                                                  # with it off, keep their hash under withdraw_at_eviction=False
        return _hash_obj(d)


# ---------------------------------------------------------------------------- #
# The persistent associative memory -- the "weights" side, as a stand-in       #
# ---------------------------------------------------------------------------- #
class PersistentMemory:
    """Modern-Hopfield store over cues: replay adds a pattern, reconstruct returns
    the softmax weighted combination of stored patterns for a query. A replay
    is verified by reading back the cue itself. Interference between near
    duplicates makes a replay fail verification -- an honest capacity effect,
    reported, never hidden. The C2 layer takes over behind these two calls."""

    def __init__(self, dim: int = CUE_DIM, beta: float = 16.0):
        self.dim, self.beta = dim, beta
        self._K = np.zeros((0, dim), dtype=np.float32)
        self._ids: list[str | None] = []          # the entry behind each row, so that a withdrawal is by identity

    def __len__(self):
        return int(self._K.shape[0])

    @staticmethod
    def _unit(v) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    def reconstruct(self, cue) -> np.ndarray | None:
        if len(self) == 0:
            return None
        q = self._unit(cue)
        logits = self.beta * (self._K @ q)
        w = np.exp(logits - logits.max()); w /= w.sum()
        return w @ self._K

    def replay(self, cue) -> float:
        """Store the pattern and return the verification error (cosine distance
        between the reconstruction and the cue). The caller decides on the
        declared threshold; a failed replay is withdrawn from the store."""
        q = self._unit(cue)
        self._K = np.vstack([self._K, q[None, :]])
        self._ids.append(None)
        r = self.reconstruct(q)
        err = float(np.clip(1.0 - float(r @ q / (np.linalg.norm(r) + 1e-8)), 0.0, 1.0))
        return err

    def withdraw_last(self) -> None:
        self._K = self._K[:-1]
        self._ids.pop()

    def label_last(self, entry_id: str) -> None:
        """Name the entry behind the last replayed pattern (ADR-007 amendment 2026-10-01), so
        that a withdrawal can be by identity. The two call contract (replay, reconstruct) is
        untouched; a memory without this call keeps the v1.0 behaviour, no withdrawal."""
        if self._ids:
            self._ids[-1] = entry_id

    def withdraw(self, entry_id: str) -> bool:
        """Remove the pattern stored for this entry (ADR-007 amendment 2026-10-01). The
        store keeps its patterns explicitly, so the removal is exact by construction: after
        it, the memory holds the patterns of the other entries and nothing of this one.
        Returns whether a pattern was found; what the other patterns still reconstruct of
        the withdrawn address is the probe's measurement, not this method's claim."""
        rows = [i for i, x in enumerate(self._ids) if x == entry_id]
        if not rows:
            return False
        keep = np.ones(len(self._ids), dtype=bool); keep[rows] = False
        self._K = self._K[keep]
        self._ids = [x for i, x in enumerate(self._ids) if keep[i]]
        return True

    def ids(self) -> list[str | None]:
        return list(self._ids)


# ---------------------------------------------------------------------------- #
# A phase plan -- computed first, checked, then applied (or refused)           #
# ---------------------------------------------------------------------------- #
@dataclass
class PhasePlan:
    consolidate: list[str] = field(default_factory=list)
    demote: list[list[str]] = field(default_factory=list)
    evict: list[str] = field(default_factory=list)          # verdicts (tribunal + salience), cascade included
    pressure: float = 0.0
    floor_eff: float = 0.0
    batch_eff: int = 0

    def evict_set(self) -> set[str]:
        return set(self.evict)


class LifecycleScheduler:
    """The sleep of the journal. Wire it above a Journal, its WritePath and its
    JournalPath; route admitted writes through `write()` (or call `on_write`
    after your own). Phases fire every P admitted writes on the logical clock
    you pass as `now`; call `phase(now)` yourself for an explicit sleep.

    Nothing here touches `Entry.state` or `Entry.salience` directly: the store's
    primitives do, and each one is an event (invariant 1)."""

    def __init__(self, journal: Journal, write_path: WritePath, journal_path: JournalPath,
                 cfg: LifecycleConfig | None = None, memory: PersistentMemory | None = None,
                 now: float = 0.0):
        self.j, self.wp, self.jp = journal, write_path, journal_path
        self.cfg = cfg or LifecycleConfig()
        self.memory = memory if memory is not None else PersistentMemory(dim=journal.cue_dim)   # an empty memory is falsy
        self.replay_failed = 0                       # telemetry only; the log's phase snapshots carry the totals
        self.last_phase: dict | None = None
        # ---- Decision 8: a scheduler opened on a journal continues it, it does not restart it
        for eid in journal.consolidated_order:       # the persistent memory, in consolidation order
            e = journal._entries[eid]
            if self.cfg.withdraw_at_eviction and e.state == STATE_EVICTED:
                continue                             # withdrawn at its eviction (ADR-007 amendment 2026-10-01): the log says so
            self.memory.replay(e.cue)                # verified when it was consolidated
            self._label(eid)
        self.phases = journal.phase_count
        self.refused = journal.refused_count
        # the sentinel set: contracted facts from the frozen H.M. generator (invariant 8)
        self.sentinels: list[tuple[str, Fact]] = []
        self.negctrl: list[Fact] = []
        if self.cfg.sentinel_facts > 0:
            if journal.sentinels_recorded:
                facts, _ = generate_facts(self.cfg.sentinel_facts, seed=self.cfg.sentinel_seed)
                self.negctrl, _ = generate_facts(self.cfg.sentinel_negctrl, seed=self.cfg.sentinel_seed + 10_000)
                for i, eid in journal.sentinels_recorded:
                    if i >= len(facts) or eid not in journal._entries:
                        raise ValueError("sentinel events do not match this configuration's sentinel set")
                    self.sentinels.append((eid, facts[i]))
            else:
                self._plant_sentinels(now)
        sentinel_ids = {eid for eid, _ in self.sentinels}
        self.writes_since_phase = sum(1 for eid in journal.writes_since_last_phase
                                      if eid not in sentinel_ids
                                      and journal._entries[eid].schema_id != SCHEMA_SUMMARY)

    def _label(self, entry_id: str) -> None:
        """Name the last replayed pattern, when the memory can hold a name (optional call)."""
        label = getattr(self.memory, "label_last", None)
        if label is not None:
            label(entry_id)

    # ------------------------------------------------------------------ writes --
    def write(self, cue, payload: bytes, schema_id: str, now: float):
        """The one entry point in the lifecycle world: gated write, index, rhythm.
        A journal in read only mode refuses the write and says so in the report
        (Decision 9); a `stop` policy lets StorageExhausted propagate."""
        try:
            rep = self.wp.write(cue, payload, schema_id, now=now)
        except JournalReadOnly as e:
            return WriteReport(admitted=False, surprise=float("nan"), reason=f"refused, {e}")
        if rep.admitted:
            self.jp.on_write(rep.entry)
            self.on_write(rep.entry, now)
        return rep

    def on_write(self, entry: Entry, now: float) -> dict | None:
        """Count an admitted write; fire a phase every P (invariant 7)."""
        self.writes_since_phase += 1
        if self.writes_since_phase >= self.cfg.period_writes:
            self.writes_since_phase = 0
            return self.phase(now)
        return None

    def contract(self, entry_id: str, now: float | None = None) -> None:
        """Credit an entry to full salience under a contract: decay never touches it."""
        self.j.set_salience(entry_id, 1.0, why="contract", now=now)

    # -------------------------------------------------------------- sentinels --
    def _plant_sentinels(self, now: float) -> None:
        facts, _ = generate_facts(self.cfg.sentinel_facts, seed=self.cfg.sentinel_seed)
        self.negctrl, _ = generate_facts(self.cfg.sentinel_negctrl, seed=self.cfg.sentinel_seed + 10_000)
        for i, f in enumerate(facts):
            rep = self.wp.write(f.cue, f.statement.encode(), f.schema, now=now)
            if rep.admitted:
                self.jp.on_write(rep.entry)
                self.contract(rep.entry.entry_id, now)
                self.j.mark_sentinel(i, rep.entry.entry_id)   # an event: a restart finds them again
                self.sentinels.append((rep.entry.entry_id, f))

    def _hit(self, fact: Fact) -> tuple[str | None, bytes | None]:
        """The effective entry a probe of this fact lands on, and its payload."""
        hits = self.jp.retrieve(fact.cue, k=1)
        if not hits:
            return None, None
        e, payload, _ = hits[0]
        return e.entry_id, payload

    def sentinel_probe(self, plan: PhasePlan | None = None) -> tuple[float, float]:
        """Contracted recall and negative-control rate through the real read path,
        with a plan overlaid: a hit that the plan would evict counts as lost. A
        plan can only remove entries, so it can never raise the negative
        control; the guard is kept because the rule says so."""
        gone = plan.evict_set() if plan else set()
        if not self.sentinels:
            return 1.0, 0.0
        recalled = 0
        for eid, f in self.sentinels:
            hit, payload = self._hit(f)
            if hit is not None and hit not in gone and f.attr in payload.decode("utf-8", errors="ignore"):
                recalled += 1
        halluc = 0
        for f in self.negctrl:
            hit, payload = self._hit(f)
            if hit is not None and hit not in gone and f.attr in payload.decode("utf-8", errors="ignore"):
                halluc += 1
        return recalled / len(self.sentinels), (halluc / len(self.negctrl) if self.negctrl else 0.0)

    # --------------------------------------------------------------- pressure --
    @staticmethod
    def _pressure_of(usage: float, setpoint: float | None, band: float) -> float:
        if not setpoint or setpoint <= 0:
            return 0.0
        return float(np.clip((usage / setpoint - (1.0 - band)) / band, 0.0, 1.0))

    def pressure(self) -> dict:
        b = self.j.regulated_bytes()
        lat = self.jp.index.p95_touched()
        pb = self._pressure_of(b, self.cfg.bytes_setpoint, self.cfg.band)
        pl = self._pressure_of(lat, self.cfg.latency_setpoint, self.cfg.band)
        # Decision 9, invariant 2: free disk space is regulated too -- pressure rises as the
        # free space approaches the declared floor (0 at floor*(1+band), 1 at the floor)
        floor, free = self.j.disk_free_floor_bytes, self.j.disk_free()
        pd = 0.0
        if floor and free is not None:
            pd = float(np.clip((floor * (1.0 + self.cfg.band) - free) / (floor * self.cfg.band), 0.0, 1.0))
        return {"bytes": b, "latency_p95": lat, "disk_free": free, "pressure_bytes": pb,
                "pressure_latency": pl, "pressure_disk": pd, "pressure": max(pb, pl, pd)}

    # ------------------------------------------------------------------- plan --
    def plan(self, now: float) -> PhasePlan:
        cfg, J = self.cfg, self.j
        pr = self.pressure()["pressure"]
        p = PhasePlan(pressure=pr,
                      floor_eff=cfg.evict_floor + pr * (cfg.evict_floor_max - cfg.evict_floor),
                      batch_eff=int(math.ceil(cfg.consolidate_batch * (1.0 + pr))))
        # 3. consolidate: salience first, then primacy (oldest first), deterministic
        live = [(eid, e) for eid, e in J._entries.items() if e.state == STATE_LIVE]
        live.sort(key=lambda t: (-t[1].salience, t[1].t_written, t[0]))
        p.consolidate = [eid for eid, _ in live[: p.batch_eff]]
        # 4. demote: K similar consolidated entries per group -- similarity means the
        #    index neighbourhood (a bucket in any table) AND an exact pairwise cosine
        #    above the declared radius; deterministic order; never a scan of the store
        consolidated = [eid for eid, e in sorted(J._entries.items()) if e.state == STATE_CONSOLIDATED]
        cset, grouped = set(consolidated), set()
        for eid in consolidated:
            if eid in grouped:
                continue
            cue = np.asarray(J._entries[eid].cue, dtype=np.float32)
            scored = []
            for c in self.jp.index.candidates(cue):
                if c in cset and c not in grouped and c != eid:
                    v = np.asarray(J._entries[c].cue, dtype=np.float32)
                    cos = float(cue @ v / (np.linalg.norm(cue) * np.linalg.norm(v) + 1e-8))
                    if cos >= cfg.demote_min_cosine:
                        scored.append((-cos, c))
            members = [eid] + [c for _, c in sorted(scored)[: cfg.demote_k - 1]]
            if len(members) == cfg.demote_k and self._min_pairwise_cosine(members) >= cfg.demote_min_cosine:
                p.demote.append(members); grouped.update(members)
        # 5. evict: verdicts only
        evict = list(noise_candidates(J, cfg.tribunal_floor, cfg.tribunal_min_age, now))   # live, the tribunal
        for eid, e in sorted(J._entries.items()):
            if (e.state == STATE_CONSOLIDATED and eid not in grouped and e.salience < p.floor_eff
                    and (now - e.t_written) >= cfg.evict_min_age):
                evict.append(eid)
        for eid in list(evict):                                    # cascade, whatever the branch: a summary
            for src in J.summary_sources.get(eid, []):             # takes its sources with it (the store
                if J._entries[src].state != STATE_EVICTED and src not in evict:   # enforces the same rule)
                    evict.append(src)
        for src, sid in sorted(J.summary_of.items()):              # repair: a demoted source whose summary is
            if (J._entries[sid].state == STATE_EVICTED and J._entries[src].state == STATE_DEMOTED
                    and src not in evict):                         # already gone (a cascade cut by a storage
                evict.append(src)                                  # failure, policy stop) is evicted now
        p.evict = evict
        return p

    def _min_pairwise_cosine(self, ids: list[str]) -> float:
        V = np.stack([np.asarray(self.j._entries[i].cue, dtype=np.float32) for i in ids])
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
        S = V @ V.T
        return float(S[np.triu_indices(len(ids), k=1)].min()) if len(ids) > 1 else 1.0

    # ------------------------------------------------------------------ phase --
    def phase(self, now: float) -> dict:
        """One sleep: plan, check the sentinels on the plan, then apply or refuse."""
        cfg = self.cfg
        p = self.plan(now)
        recall_before, neg_before = self.sentinel_probe(None)
        recall_after, neg_after = self.sentinel_probe(p)
        ok = (recall_after >= 1.0) and (neg_after <= neg_before)
        rec = {"phase": self.phases + 1, "now": now, "config_hash": cfg.config_hash(),
               "pressure": round(p.pressure, 4), "floor_eff": round(p.floor_eff, 4),
               "batch_eff": p.batch_eff, "planned": {"consolidate": len(p.consolidate),
               "demote_groups": len(p.demote), "evict": len(p.evict)},
               "sentinel_recall": round(recall_after, 4), "sentinel_negctrl": round(neg_after, 4),
               "refused": not ok}
        if not ok:
            self.refused += 1
            rec["reason"] = ("contracted recall would fall to %.3f" % recall_after if recall_after < 1.0
                             else "negative control would rise")
            rec.update(self._log_telemetry())
            self.j.phase_event(rec)
            self.phases += 1
            self.last_phase = rec
            return rec
        applied = self._apply(p, now)
        self.j.decay(cfg.decay)                                    # one event; contracted exempt
        rec["applied"] = applied
        rec.update(self._log_telemetry())
        self.j.phase_event(rec)
        self.phases += 1
        self.last_phase = rec
        return rec

    def _apply(self, p: PhasePlan, now: float) -> dict:
        J, cfg = self.j, self.cfg
        consolidated, failed = 0, 0
        for eid in p.consolidate:                                  # 3. replay, verify, then transition
            err = self.memory.replay(J._entries[eid].cue)
            if err <= cfg.replay_error_max:
                J.transition(eid, STATE_CONSOLIDATED); consolidated += 1
                self._label(eid)
            else:
                self.memory.withdraw_last(); failed += 1
        self.replay_failed += failed
        demoted = 0
        for group in p.demote:                                     # 4. K similar -> one summary
            self._demote(group, now); demoted += 1
        evicted, withdrawn = 0, 0
        for eid in p.evict:                                        # 5. verdicts (cascade already listed)
            if J._entries[eid].state != STATE_EVICTED:
                J.transition(eid, STATE_EVICTED); evicted += 1
                if cfg.withdraw_at_eviction:                       # the address leaves with the content
                    withdraw = getattr(self.memory, "withdraw", None)
                    if withdraw is not None and withdraw(eid):
                        withdrawn += 1
        applied = {"consolidated": consolidated, "replay_failed": failed,
                   "demoted_groups": demoted, "evicted": evicted}
        if cfg.withdraw_at_eviction:
            applied["withdrawn"] = withdrawn                       # the phase event carries it only under the flag
        return applied

    def _demote(self, group: list[str], now: float) -> Entry:
        J = self.j
        cues = np.stack([np.asarray(J._entries[i].cue, dtype=np.float32) for i in group])
        centroid = cues.mean(axis=0); centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        level = 1 + max(J.summary_level.get(i, 0) for i in group)
        payload = json.dumps({"kind": SCHEMA_SUMMARY, "level": level,
                              "pointers": [J._entries[i].pointer for i in group],
                              "sources": list(group),
                              "digest": [round(float(x), 3) for x in centroid[:8]]},
                             separators=(",", ":")).encode()
        salience = max(J._entries[i].salience for i in group)
        summary = J.write(centroid, payload, salience=salience, schema_id=SCHEMA_SUMMARY, now=now)
        self.jp.on_write(summary)
        J.demote(summary.entry_id, list(group), level=level)      # the link (an event)
        for src in group:
            J.transition(src, STATE_DEMOTED)                       # each a transition event
            self.jp.index.set_alias(src, summary.entry_id)         # reachable from every source cue
        if any(i in J.contracted for i in group):
            self.contract(summary.entry_id, now)                   # a contract survives its demotion
        return summary

    # -------------------------------------------------------------- telemetry --
    def _telemetry(self) -> dict:
        snap = self.j.snapshot()
        pr = self.pressure()
        return {"bytes": pr["bytes"], "latency_p95": round(pr["latency_p95"], 2),
                "disk_free": pr["disk_free"], "pressure_disk": round(pr["pressure_disk"], 4),
                "storage_mode": self.j.mode, "unpersisted_events": len(self.j.spill),
                "scope_hash": self.j.scope_config_hash(),
                "pressure_bytes": round(pr["pressure_bytes"], 4),
                "pressure_latency": round(pr["pressure_latency"], 4),
                "live": snap[STATE_LIVE], "consolidated": snap[STATE_CONSOLIDATED],
                "demoted": snap[STATE_DEMOTED], "evicted": snap[STATE_EVICTED],
                "summaries": snap["summaries"], "contracted": snap["contracted"],
                "memory_patterns": len(self.memory), "phases_refused": self.refused,
                "replay_failed_total": self.replay_failed}

    def _log_telemetry(self) -> dict:
        """What goes into the phase event: everything but the raw free disk bytes, an
        environment reading that would break the byte identity of two identical
        runs (invariant 7). The disk pressure stays: it is a function of the
        declared floor, zero when none is declared."""
        t = self._telemetry(); t.pop("disk_free", None)
        return t

    def snapshot(self) -> dict:
        """The RES-17 snapshot for C6 telemetry / the run record's standard_suite."""
        return {"config_hash": self.cfg.config_hash(), "phases": self.phases, **self._telemetry()}
