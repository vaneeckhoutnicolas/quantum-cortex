"""cortex_c2b.revision_probe -- the revision at the same address, measured on a synthetic journal
(ADR-007, amendment of 2026-10-12, declared before any code and validated by the founder the same
day; the review's point 3, decision A3).

The protocol's generator at a seed plants N facts through the gate with their texts; M of them
(half, by a seeded draw) are revised through the same gate with a different attribute of the same
schema, and a third of those a second time. The journal is sealed on disk under a key generated for
the run (no key of the record is needed); a lifecycle scheduler consolidates every entry before the
revisions so that the memory's pattern count can be compared before and after. Read, top one, by
the planted address, then a second process from the disk alone that recomputes everything:

  current_returned   the share of revised entries whose read returns the current attribute
  past_returned      the share whose read returns a past attribute
  unrevised_intact   the share of unrevised entries whose read is unchanged
  history_exact      every history length equal to one plus the revisions made on that entry
  entries_exact      the entry count equal to N (no second entry)
  memory_unchanged   the pattern count equal before and after the revisions
  agreement          the second process equal to the first, number for number

Thresholds declared: 1.0, 0.0, 1.0, exact, exact, exact, exact. Readings written before any run:
(a) all hold; (b) a read returns a stale version; (c) the gate created an entry or lost a version;
(d) the processes disagree. Nothing here is a claim. The first number comes from the first file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cortex_c2b import Journal, POLICY_STOP, STATE_EVICTED, lifecycle_declaration
from cortex_c2b.crypto import generate_key, key_from_env
from cortex_c2b.hm_protocol import generate_facts, _ATTRS, _TEMPLATES, N_FACTS
from cortex_c2b.lifecycle import LifecycleConfig, LifecycleScheduler
from cortex_c2b.read_path import JournalPath
from cortex_c2b.write_path import WritePath

ADMISSION_THRESHOLD = 0.15     # the protocol's own
THRESHOLDS = {"current_returned": 1.0, "past_returned": 0.0, "unrevised_intact": 1.0, "history_exact": True,
              "entries_exact": True, "memory_unchanged": True, "agreement": True}


def _plan(facts, seed: int) -> dict:
    """Which facts are revised, with which attributes: a seeded draw, written once so both processes read it."""
    rng = np.random.default_rng(100_000 + seed)
    idx = sorted(rng.choice(len(facts), size=len(facts) // 2, replace=False).tolist())
    twice = set(sorted(rng.choice(idx, size=len(idx) // 3, replace=False).tolist())) if idx else set()
    plan = {}
    for i in idx:
        f = facts[i]
        others = [a for a in _ATTRS[f.schema] if a != f.attr]
        first = str(rng.choice(others))
        seq = [first]
        if i in twice:
            seq.append(str(rng.choice([a for a in others if a != first])))
        plan[i] = seq
    return plan


def _statement(f, attr: str) -> str:
    return _TEMPLATES[f.schema][0].format(e=f.entity, a=attr)


def _measure(journal: Journal, jp: JournalPath, facts, plan: dict, memory_patterns: int | None) -> dict:
    """The seven measures but the agreement, from a journal and its index (either process)."""
    current = past = intact = 0
    history_ok, versions = True, {}
    for i, f in enumerate(facts):
        hits = jp.retrieve(f.cue, k=1)
        payload = hits[0][1].decode("utf-8") if hits else ""
        if i in plan:
            seq = plan[i]
            current += int(payload == _statement(f, seq[-1]))
            past += int(payload in [f.statement] + [_statement(f, a) for a in seq[:-1]])
            eid = journal.at_address(f.cue, f.address_text)
            n = len(journal.history(eid.entry_id)) if eid is not None else 0
            versions[f.entity] = n
            history_ok = history_ok and (n == 1 + len(seq))
        else:
            intact += int(payload == f.statement)
            eid = journal.at_address(f.cue, f.address_text)
            n = len(journal.history(eid.entry_id)) if eid is not None else 0
            versions[f.entity] = n
            history_ok = history_ok and (n == 1)
    n_rev, n_unrev = len(plan), len(facts) - len(plan)
    return {"current_returned": current / n_rev if n_rev else None, "past_returned": past / n_rev if n_rev else None,
            "unrevised_intact": intact / n_unrev if n_unrev else None, "history_exact": history_ok,
            "entries_exact": len([e for e in journal._entries.values() if e.state != STATE_EVICTED]) == len(facts),
            "versions": versions, "memory_patterns": memory_patterns,
            "revisions_declared": lifecycle_declaration(journal)["structure"]["revisions"]}


def session_a(path: Path, key: bytes, n: int, seed: int) -> tuple[dict, dict, list]:
    facts, gen_hash = generate_facts(n, seed)
    cfg = LifecycleConfig(period_writes=10_000, consolidate_batch=n, sentinel_facts=0, decay=1.0)
    journal = Journal(path, key=key, policy=POLICY_STOP)
    wp, jp = WritePath(journal, admission_threshold=ADMISSION_THRESHOLD, seed=seed), JournalPath(journal, seed=seed)
    sched = LifecycleScheduler(journal, wp, jp, cfg, now=0.0)
    admitted = 0
    for i, f in enumerate(facts):
        rep = sched.write(f.cue, f.statement.encode("utf-8"), f.schema, now=float(i), source=f.address_text)
        admitted += int(rep.admitted)
    sched.phase(now=float(n) + 1.0)                                 # every entry consolidated: one pattern each
    patterns_before = len(sched.memory)
    plan = _plan(facts, seed)
    made, refused = 0, 0
    for i, seq in sorted(plan.items()):
        f = facts[i]
        for k, attr in enumerate(seq):
            rep = sched.write(f.cue, _statement(f, attr).encode("utf-8"), f.schema, now=1000.0 + 10 * i + k,
                              source=f.address_text)
            made += int(rep.revised); refused += int(not rep.revised)
    patterns_after = len(sched.memory)
    m = _measure(journal, jp, facts, plan, patterns_after)
    m["memory_unchanged"] = patterns_before == patterns_after
    cfg_out = {"n_facts": n, "seed": seed, "generator_hash": gen_hash, "admission_threshold": ADMISSION_THRESHOLD,
               "admitted": admitted, "revised": len(plan), "revised_twice": sum(1 for s in plan.values() if len(s) == 2),
               "revisions_made": made, "revisions_refused": refused, "lifecycle_hash": cfg.config_hash(),
               "index_seed": seed}
    return m, {"config": cfg_out, "memory": {"patterns_before": patterns_before, "patterns_after": patterns_after},
               "plan": {facts[i].entity: seq for i, seq in plan.items()},
               "storage": lifecycle_declaration(journal)["storage"]}, facts


def session_b(path: Path, key: bytes, n: int, seed: int) -> dict:
    facts, _ = generate_facts(n, seed)
    cfg = LifecycleConfig(period_writes=10_000, consolidate_batch=n, sentinel_facts=0, decay=1.0)
    journal = Journal(path, key=key, policy=POLICY_STOP)
    wp, jp = WritePath(journal, admission_threshold=ADMISSION_THRESHOLD, seed=seed), JournalPath(journal, seed=seed)
    sched = LifecycleScheduler(journal, wp, jp, cfg)                  # the memory rebuilt from the log alone
    m = _measure(journal, jp, facts, _plan(facts, seed), len(sched.memory))
    m["process"] = "new (nothing of session A but the disk)"
    m["memory_unchanged"] = None                                      # process B has no "before"; it carries the count
    return m


def _letter(m: dict) -> str:
    if not m["agreement"]:
        return "d: the two processes disagree; Decision 8 broken by the new event; attributed before anything else is read"
    if not m["entries_exact"] or not m["history_exact"]:
        return "c: the gate created an entry instead of revising, or lost a version; a construction fault, attributed"
    if m["past_returned"] > THRESHOLDS["past_returned"] or m["current_returned"] < THRESHOLDS["current_returned"] \
            or m["unrevised_intact"] < THRESHOLDS["unrevised_intact"]:
        return "b: a read returns a stale version (or an unrevised entry moved); a construction fault, attributed before any word"
    if not m["memory_unchanged"]:
        return "b: the memory's pattern count moved under the revisions, which must not touch it; attributed"
    return ("a: every revised entry reads at its current version, none at a past one, the unrevised are intact, the "
            "histories and the entry count are exact, the memory is untouched, and the second process agrees: the "
            "mechanism holds at the organ's level")


def run(n: int = N_FACTS, seed: int = 0, keep_dir: Path | None = None) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="revision-probe-")) if keep_dir is None else keep_dir
    key = generate_key()                                              # a key for this run only; no key of the record
    path = tmp / "journal.jsonl"
    try:
        m_a, meta, facts = session_a(path, key, n, seed)
        env = dict(os.environ, QUANTUM_CORTEX_JOURNAL_KEY=key.hex())
        proc = subprocess.run([sys.executable, "-m", "cortex_c2b.revision_probe", "--session-b", str(path),
                               "--n", str(n), "--seed", str(seed)],
                              capture_output=True, text=True, env=env, cwd=str(Path(__file__).resolve().parents[1]))
        if proc.returncode != 0:
            raise RuntimeError(f"session B failed:\n{proc.stderr[-3000:]}")
        m_b = json.loads(proc.stdout.strip().splitlines()[-1])
        compared = ("current_returned", "past_returned", "unrevised_intact", "history_exact", "entries_exact",
                    "versions", "memory_patterns", "revisions_declared")
        diffs = [k for k in compared if m_a[k] != m_b[k]]
        measures = {k: m_a[k] for k in ("current_returned", "past_returned", "unrevised_intact", "history_exact",
                                        "entries_exact", "memory_unchanged")}
        measures["agreement"] = not diffs
        letter = _letter(measures)
        return {"probe": "revision at the same address (ADR-007 amendment 2026-10-12; the review's point 3, decision A3)",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **meta, "thresholds": THRESHOLDS, "measures": measures,
                "process_a": {k: v for k, v in m_a.items()}, "process_b": m_b,
                "agreement": {"number_for_number": not diffs, "differences": diffs},
                "readings": {"letter": letter}, "verdict": letter}
    finally:
        if keep_dir is None:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


def summary_lines(out: dict) -> list[str]:
    c, m = out["config"], out["measures"]
    return [f"revision probe  n {c['n_facts']}  seed {c['seed']}  generator {c['generator_hash']}  admitted {c['admitted']}  "
            f"revised {c['revised']} (twice {c['revised_twice']})  revisions made {c['revisions_made']}  refused {c['revisions_refused']}",
            f"current returned {m['current_returned']:.3f}  past returned {m['past_returned']:.3f}  unrevised intact {m['unrevised_intact']:.3f}",
            f"history exact {m['history_exact']}  entries exact {m['entries_exact']}  memory unchanged {m['memory_unchanged']} "
            f"(patterns {out['memory']['patterns_before']} -> {out['memory']['patterns_after']})",
            f"two processes agree: {m['agreement']}",
            f"reading: {out['verdict']}"]


def _cli(argv=None):
    ap = argparse.ArgumentParser(description="the revision at the same address, measured on a synthetic journal")
    ap.add_argument("--n", type=int, default=N_FACTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="the directory for the retained file (revision-probe-seed<s>.json)")
    ap.add_argument("--session-b", default=None, help="(internal) reopen this journal in a new process and measure")
    args = ap.parse_args(argv)
    if args.session_b:
        key = key_from_env()
        if key is None:
            raise SystemExit("QUANTUM_CORTEX_JOURNAL_KEY missing for session B")
        print(json.dumps(session_b(Path(args.session_b), key, args.n, args.seed)))
        return
    p = Path(args.out) / f"revision-probe-seed{args.seed}.json" if args.out else None
    if p is not None and p.exists():
        raise SystemExit(f"revision probe: {p} exists; a file of the record is never overwritten, give another --out")
    out = run(n=args.n, seed=args.seed)
    for line in summary_lines(out):
        print(line)
    if p is not None:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"[record] retained: {p}  sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    _cli()
