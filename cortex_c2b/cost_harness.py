"""cortex_c2b.cost_harness -- the cost by stage of ADR-010 B.3, made exact by the amendment of 2026-10-15 on parts B and C
(graved 2026-10-29): a new module on the frozen system `c5b4818`, nothing of the organ changed.

A "configuration" of the interleaving is one of the three indexes (I0 the organ's live index; I1 the exhaustive cosine
over the same cues; I2 BM25). The read side stages up to the window are timed per index: the question's cue (the
encoder's pass), the index step, the marks (a stage of its own for I1 and I2, `marks_for`; included in the index step for
I0, whose scores come with the lookup, reported as included and not as zero), the window (`build_read_window`); the rule
stage (`policy_outcome`, V1, the oracle's rule) is timed per rule on the recorded outputs of the passes, which gives the
twenty one rows per checkpoint and family. The protocol: the harness is started by the launcher outside the package
(`tools/adr010_launcher.py --pin-threads`), which sets the four thread variables to 1 before the interpreter imports
anything; after the imports the harness calls `torch.set_num_threads(1)`, reads the four variables and torch's two counts
back, records them and refuses the run before any stage if a count is above one; the garbage collector is disabled
during every timed stage and re enabled between them, the collections that ran outside the stages counted; the timer is
`time.perf_counter_ns`, its own overhead measured on an empty stage (the median of one thousand calls) and reported
beside every stage whose median is under one hundred times it; six repetitions after one discarded warm up, the indexes
interleaved question by question in the six orders of three indexes, one order per repetition, so that every index
occupies every position twice; reported per stage and index the median, the 95th percentile and the spread across the
six repetitions (the range of the six medians), with the operation count (candidates scored by I0, cues compared by I1,
postings read by I2). A repetition whose median on a stage above one hundred times the timer's overhead is more than
twice, or less than half, the median of the other five repetitions' medians on that stage is reported and the whole
protocol run once more under another name; the row's numbers are the first protocol's and the second is reported beside
it; if it also holds such a repetition the stage is labelled "unstable on this processor" and both are printed.

Memory is measured in a run of its own with `tracemalloc` on, never in a timed run. The write side is timed on a scratch
replanting of the family's 200 statements with their recorded cues, six repetitions, never on the measured journal: the
cue (the encoder's pass); the gate by the frozen read only calls of the write path (the code, `CA3Completion.nearest` and
the surprise on the incoming cue against the write path's memory before the write), the organ's gate, which I1 and I2 do
not have (their stage reported as absent); the index insertion (`CueIndex.add` for I0, the reference's insertion for I1
and I2, I2's including its statistics); the whole store write by `Journal.write`, common; the sealing by the frozen `seal`
and `seal_line` on the same bytes; the append as the whole store write minus the sealing, labelled derived. The
durability is the organ's as frozen (no fsync), said in the file and not changed. The generation is reported once per
checkpoint as a common stage. The maintenance proxies are the tunable parameters per stack read from the configuration
objects and the lines of code per component, labelled as proxies. The 200 entry numbers carry the label "not a scale
result" (part C's is the scale comparison). Mains power, nothing else running, the processor and the versions recorded.
"""
from __future__ import annotations

import argparse
import gc
import inspect
import itertools
import json
import os
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

from cortex_c2b import Journal, POLICY_STOP, ADMISSION_IDENTITY, content_hash, record_tools
from cortex_c2b.crypto import seal, seal_line, generate_key
from cortex_c2b.hm_protocol import Fact, value_of_statement
from cortex_c2b.lm_bridge import JournalBridge, build_read_window, encode_query
from cortex_c2b.read_path import ADOPTED_GEOMETRY, CueIndex, JournalPath
from cortex_c2b.write_path import WritePath, ca1_surprise
from cortex_c2b.hm_lm import generate, policy_outcome
from cortex_c2b.near_duplicate_probe import load_model
from cortex_c2b import sealed_families as S
from cortex_c2b.references import ExhaustiveCosine, BM25Index, bm25_selection, v1_outcome, INDEXES
from cortex_c2b import resumable as RS
from cortex_c2b.resumable import THREAD_VARIABLES

REPETITIONS = 6
TIMER_FACTOR = 100            # a stage whose median is under this many timer overheads is reported beside the overhead
INSTABILITY = 2.0             # a repetition's median more than twice, or less than half, the median of the other five
ORDERS = list(itertools.permutations(INDEXES))       # the six orders of three indexes, one per repetition
NOT_A_SCALE_RESULT = "not a scale result: 200 entries, one processor, the build and the query separated; the scale comparison is part C's"
DURABILITY = ("the write's durability is the organ's as frozen: the sealed payload written whole to a temporary file and renamed into place, "
              "the sealed log line appended to a file opened and closed per write, neither synchronised to the disk (no fsync); "
              "'durable first' in the code means ordered before the memory changes; said here, not changed")
RULE_STAGES = ("plain", "mark-veto", "mark-veto+value", "V1", "V1+organ-value", "oracle", "oracle+organ-value")


def check_pinned() -> dict:
    """The thread counts read back after the imports; a count above one refuses the run (B.3). The harness calls
    `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)` first thing (the four variables pin the intra op pool
    only: torch's inter op pool defaults to the core count whatever they say, found by the build's test on 2026-10-30; the
    second call is refused by torch once parallel work has started, so it comes before any)."""
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as e:                                  # parallel work already started in this process: read back and refuse below
        interop_note = str(e)
    else:
        interop_note = None
    env = {k: os.environ.get(k) for k in THREAD_VARIABLES}
    counts = {"torch_threads": int(torch.get_num_threads()), "torch_interop_threads": int(torch.get_num_interop_threads())}
    problems = [k for k, v in env.items() if v != "1"] + [k for k, v in counts.items() if v != 1]
    return {"thread_variables": env, **counts, "pinned": not problems, "problems": problems, "launcher": os.environ.get("ADR010_LAUNCHER") == "1",
            "interop_set_note": interop_note}


def timer_overhead_ns(n: int = 1000) -> float:
    xs = []
    for _ in range(n):
        a = time.perf_counter_ns(); b = time.perf_counter_ns(); xs.append(b - a)
    return float(statistics.median(xs))


class Timed:
    """A timed stage: the garbage collector disabled during it, re enabled after; the collections run outside the stages
    counted (gc.get_count differences are not collections: `gc.callbacks` counts them)."""
    collections_outside = 0

    def __init__(self):
        self.ns = None

    def __enter__(self):
        gc.disable()
        self._t = time.perf_counter_ns()
        return self

    def __exit__(self, *a):
        self.ns = time.perf_counter_ns() - self._t
        gc.enable()
        return False


def _gc_callback(phase, info):
    if phase == "start" and gc.isenabled():
        Timed.collections_outside += 1


def quantiles(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=np.float64)
    return {"n": int(a.size), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}


# ---------------------------------------------------------------------------- #
# the read side                                                                  #
# ---------------------------------------------------------------------------- #
def read_stages_once(model, bridge, journal, organ_jp: JournalPath, index_name: str, reference, question: str, cue_cache: dict,
                     k: int, budget: int, mark: bool, rng_seed: tuple[int, int]) -> dict:
    """One question through the read side stages of one index, each timed: the cue, the index step, the marks, the window.
    Returns the times in ns and the operation count; the window and labels are returned too for the rule stage."""
    out = {}
    q_tokens = encode_query(question)
    with Timed() as t:
        cue = bridge.cues([q_tokens])[0]
    out["cue_ns"] = t.ns
    q = np.asarray(cue, dtype=np.float32)
    if index_name == "I0":
        before = organ_jp.index.touched
        with Timed() as t:
            hits = organ_jp.retrieve(q, k=k)
        out["index_ns"] = t.ns; out["operations"] = organ_jp.index.touched - before
        items = [(e.pointer, payload) for e, payload, _ in hits]; scores = [float(s) for _, _, s in hits]
        out["marks_ns"] = None            # included in the index step (the scores come with the lookup)
    else:
        with Timed() as t:
            r = reference.rank(question if index_name == "I2" else q, k=k)
        out["index_ns"] = t.ns; out["operations"] = int(r.get("postings_read", r.get("compared", 0)))
        pointers = [journal._entries[e].pointer for e in r["docs"] if journal._entries[e].state != "evicted"]
        with Timed() as t:
            scores = bridge.marks_for(q, pointers)
        out["marks_ns"] = t.ns
        items = [(p, journal.payloads.get(p)) for p in pointers]
    rng = np.random.default_rng(list(rng_seed))
    with Timed() as t:
        window, labels = build_read_window(items, budget, rng, marks=scores if mark else None)
    out["window_ns"] = t.ns
    out["window"] = window; out["labels"] = labels; out["pointers"] = [p for p, _ in items]; out["scores"] = scores
    return out


def rule_stages_once(rec: dict, labels: dict, marks: dict, payloads, threshold: float, question: str, expected: str) -> dict:
    """The rule stage per rule on recorded outputs (the frozen `policy_outcome`, V1, the oracle's rule), each timed."""
    out = {}
    kind, label, ans = rec["raw_kind"], rec["raw_label_byte"], rec["raw_answered"]
    with Timed() as t:
        pass
    out["plain"] = t.ns
    for pol in ("mark-veto", "mark-veto+value"):
        with Timed() as t:
            policy_outcome(kind, label, labels, marks, payloads, rec.get("own_pointer"), ans, expected, pol, threshold)
        out[pol] = t.ns
    cited_payload = payloads.get(labels.get(label)) if (kind == "cite" and label in labels) else None
    for name, organ in (("V1", False), ("V1+organ-value", True)):
        with Timed() as t:
            v1_outcome(question, cited_payload, ans, expected, organ_value=organ)
        out[name] = t.ns
    for name, organ in (("oracle", False), ("oracle+organ-value", True)):
        with Timed() as t:
            if marks:
                best = max(marks.values())
                top = [p for p, m in marks.items() if m == best][0] if best > threshold else None
                if top is not None and organ:
                    value_of_statement((payloads.get(top) or b"").decode("utf-8", errors="replace"))
        out[name] = t.ns
    return out


def summarise(times: dict[str, dict[str, list[list[float]]]], overhead: float) -> dict:
    """Per stage and index: the median, the 95th percentile and the spread across repetitions (the range of the six
    medians), the instability flags (a repetition's median more than twice or less than half the median of the others)."""
    out = {}
    for index, stages in times.items():
        out[index] = {}
        for stage, reps in stages.items():                   # reps: a list per repetition of per question times
            flat = [x for r in reps for x in r if x is not None]
            meds = [float(np.median(r)) for r in reps if r and r[0] is not None]
            if not flat:
                out[index][stage] = {"absent": True}
                continue
            unstable = []
            for i, m in enumerate(meds):
                others = [x for j, x in enumerate(meds) if j != i]
                if others:
                    mo = float(np.median(others))
                    if mo > 0 and (m > INSTABILITY * mo or m < mo / INSTABILITY):
                        unstable.append(i)
            q = quantiles(flat)
            out[index][stage] = {**q, "unit": "ns", "repetition_medians": meds, "spread_of_medians": (max(meds) - min(meds)) if meds else None,
                                 "under_100_timer_overheads": q["median"] < TIMER_FACTOR * overhead, "timer_overhead_ns": overhead,
                                 "unstable_repetitions": unstable, "flag": bool(unstable and q["median"] >= TIMER_FACTOR * overhead)}
    return out


def run_protocol(model, cfg, journal_path: Path, key: bytes, fam: dict, questions: list[dict], threshold: float, bm25_pair: tuple[float, float],
                 k: int, budget: int, mark: bool, device, repetitions: int = REPETITIONS, label: str = "protocol 1") -> dict:
    """The read side protocol on one journal: one discarded warm up, then `repetitions` repetitions in the six orders."""
    times: dict[str, dict[str, list]] = {ix: {"cue": [], "index": [], "marks": [], "window": [], **{r: [] for r in RULE_STAGES}} for ix in INDEXES}
    ops: dict[str, list] = {ix: [] for ix in INDEXES}
    with record_tools.opened_copy(journal_path, key, prefix="cost-read-") as (journal, _):
        bridge = JournalBridge(model, journal, k=k, budget_bytes=budget, seed=0, shuffle_seed=1, device=device, mark=mark,
                               index_geometry=ADOPTED_GEOMETRY, admission=journal.admission_record())
        organ_jp = bridge.jp
        refs = {"I1": ExhaustiveCosine(journal, organ_jp), "I2": None}
        from cortex_c2b.references import BM25Reference
        refs["I2"] = BM25Reference(journal, *bm25_pair)
        texts = {content_hash(f["statement"].encode("utf-8")): f["statement"].encode("utf-8") for f in fam["facts"]}
        payloads = dict(texts)
        import torch
        with torch.no_grad():
            for rep in range(-1, repetitions):                   # -1: the discarded warm up
                order = ORDERS[rep % len(ORDERS)] if rep >= 0 else INDEXES
                per_index = {ix: {s: [] for s in times[ix]} for ix in INDEXES}
                per_ops = {ix: [] for ix in INDEXES}
                for qi, q in enumerate(questions):
                    for ix in order:
                        r = read_stages_once(model, bridge, journal, organ_jp, ix, refs.get(ix), q["text"], {}, k, budget, mark, (1, qi))
                        per_index[ix]["cue"].append(r["cue_ns"]); per_index[ix]["index"].append(r["index_ns"])
                        per_index[ix]["marks"].append(r["marks_ns"]); per_index[ix]["window"].append(r["window_ns"])
                        per_ops[ix].append(r["operations"])
                        # the rule stage on a standing input: a citation of the first window line with the question's expected value
                        # stands in for the pass's recorded raw parse; what is timed is the cost of a rule on a surviving citation (the
                        # longest path: an abstention returns at the first line of `policy_outcome`, a vetoed citation skips the value
                        # reader), the veto's branch taken following the first line's mark against the threshold; stated in the file
                        first_label = next(iter(sorted(r["labels"]))) if r["labels"] else None
                        rec = {"raw_kind": "cite" if first_label is not None else "unknown", "raw_label_byte": first_label,
                               "raw_answered": (q["expected"] or "x"), "own_pointer": q["own_pointer"]}
                        marks = dict(zip(r["pointers"], r["scores"]))
                        rs = rule_stages_once(rec, r["labels"], marks, payloads, threshold, q["text"], q["expected"] or "(never planted)")
                        for name, ns in rs.items():
                            per_index[ix][name].append(ns)
                if rep >= 0:
                    for ix in INDEXES:
                        for s in times[ix]:
                            times[ix][s].append(per_index[ix][s])
                        ops[ix].append(per_ops[ix])
    overhead = timer_overhead_ns()
    summary = summarise(times, overhead)
    for ix in INDEXES:
        flat_ops = [o for r in ops[ix] for o in r]
        summary[ix]["operations"] = {**quantiles([float(o) for o in flat_ops]), "what": {"I0": "candidates scored (the index's touched increment)", "I1": "cues compared", "I2": "postings read"}[ix]}
        summary[ix]["marks"]["note"] = ("included in the index step for I0 (the scores come with the lookup), reported as included and not as zero" if ix == "I0"
                                        else "marks_for on the returned lines, a stage of its own")
    return {"label": label, "repetitions": repetitions, "orders": [list(o) for o in ORDERS[:repetitions]], "timer_overhead_ns": overhead,
            "stages": summary, "n_questions": len(questions), "collections_outside_stages": Timed.collections_outside,
            "rule_stage_note": ("timed on a standing input (a citation of the first window line with the question's expected value), not on the passes' recorded "
                                "outputs: the cost of a rule on a surviving citation, the longest path; an abstention returns at the first line of policy_outcome "
                                "and a vetoed citation skips the value reader, so the branch taken follows the first line's mark against the threshold; "
                                "the `plain` row is an empty stage, the timer's own cost per question, beside timer_overhead_ns")}


def write_side(model, cfg, fam: dict, planting: dict, bm25_pair: tuple[float, float], device, repetitions: int = REPETITIONS) -> dict:
    """The write side on a scratch replanting of the family's 200 statements with their recorded cues (the planting record's
    `cue_computed`), six repetitions in fresh sealed scratch journals: the cue, the gate (the organ's read only calls), the
    index insertion per index, the whole store write, the sealing, the append (derived)."""
    import torch
    from cortex_c2b.write_path import LEGACY_THRESHOLD
    cues = {w["entity"]: np.asarray(w["cue_computed"], dtype=np.float32) for w in planting["writes"] if w.get("admitted")}
    stages = {"cue": [], "gate": [], "insert_I0": [], "insert_I1": [], "insert_I2": [], "store_write": [], "seal_payload": [], "seal_line": []}
    cue_difference = 0.0; cue_compared = 0                                  # the recomputed cue against the planting record's cue_computed (B.3: "with their recorded cues")
    with torch.no_grad():
        for rep in range(-1, repetitions):
            per = {s: [] for s in stages}
            with tempfile.TemporaryDirectory(prefix="cost-write-") as tmp:
                key = generate_key()
                journal = Journal(Path(tmp) / "scratch.jsonl", key=key, policy=POLICY_STOP, index_geometry=ADOPTED_GEOMETRY)
                wp = WritePath(journal, seed=0, admission=ADMISSION_IDENTITY)
                jp = JournalPath(journal, geometry=ADOPTED_GEOMETRY)
                i1_store: list[np.ndarray] = []
                bm = BM25Index(*bm25_pair)
                bridge = JournalBridge(model, journal, write_path=wp, journal_path=jp, k=cfg.journal_k, budget_bytes=cfg.journal_read_bytes, seed=0,
                                       shuffle_seed=1, device=device, admission=ADMISSION_IDENTITY)
                for i, f in enumerate(fam["facts"]):
                    text = f["statement"]
                    with Timed() as t:
                        cue = bridge.cues([list(text.encode("utf-8"))])[0]
                    per["cue"].append(t.ns)
                    cue = np.asarray(cue, dtype=np.float32)
                    if rep == 0 and f["entity"] in cues:
                        cue_difference = max(cue_difference, float(np.max(np.abs(cue - cues[f["entity"]])))); cue_compared += 1
                    with Timed() as t:
                        code = wp._code(cue); hit = wp.ca3.nearest(code); surprise = ca1_surprise(None if hit is None else hit[1], code)
                    per["gate"].append(t.ns)
                    nearest_id = wp._entry_at(None if hit is None else hit[0])
                    near = ({"entry_id": nearest_id, "surprise": float(surprise)} if (surprise < LEGACY_THRESHOLD and nearest_id is not None) else None)
                    payload = text.encode("utf-8")
                    with Timed() as t:
                        entry = journal.write(cue, payload, salience=surprise, schema_id=f["schema"], now=float(i), ca3=wp._meta, source=text,
                                              admission=ADMISSION_IDENTITY, near=near)
                    per["store_write"].append(t.ns)
                    wp.ca3.store(wp._code(np.asarray(entry.cue, dtype=np.float32)), entry.entry_id)
                    with Timed() as t:
                        jp.index.add(entry.entry_id, np.asarray(entry.cue, dtype=np.float32))
                    per["insert_I0"].append(t.ns)
                    with Timed() as t:
                        i1_store.append(np.asarray(entry.cue, dtype=np.float32))
                    per["insert_I1"].append(t.ns)
                    with Timed() as t:
                        bm.insert(entry.entry_id, payload)
                    per["insert_I2"].append(t.ns)
                    ev = {"ev": "write", "entry": entry.to_dict(), "admission": ADMISSION_IDENTITY, "ca3": dict(wp._meta), "index": dict(journal.index_geometry or {})}
                    line = json.dumps(ev, separators=(",", ":"))
                    with Timed() as t:
                        seal(key, payload, entry.pointer.encode())
                    per["seal_payload"].append(t.ns)
                    with Timed() as t:
                        seal_line(key, line)
                    per["seal_line"].append(t.ns)
            if rep >= 0:
                for s in stages:
                    stages[s].append(per[s])
    overhead = timer_overhead_ns()
    out = summarise({"write": stages}, overhead)["write"]
    med_store = out["store_write"]["median"]; med_seal = out["seal_payload"]["median"] + out["seal_line"]["median"]
    out["append_derived"] = {"median": med_store - med_seal, "unit": "ns", "label": "derived: the whole store write minus the sealing (medians)"}
    out["gate"]["note"] = ("the organ's gate as the surprise's computation (the code, CA3Completion.nearest, the surprise); under the identity rule the admission's "
                           "decision is the address lookup (Journal.at_address), not timed here; I1 and I2 have no gate: absent, not zero")
    out["insert_I1"]["note"] = "I1 has no index structure: the cue appended to the store of vectors; its norms are computed at the first read and cached, so that cost sits in the read side's warm up, not here"
    out["insert_I2"]["note"] = "BM25's insertion with its statistics (N, the document frequencies, the lengths, avgdl)"
    return {"repetitions": repetitions, "stages": out, "durability": DURABILITY,
            "cues": {"what": "the cue stage recomputes the encoder's pass; the recomputed cue is compared with the planting record's cue_computed",
                     "largest_abs_difference": cue_difference, "compared": cue_compared},
            "write_event": "the harness's Journal.write carries the `near` stamp as the frozen write path computes it (the nearest code and the surprise under the legacy threshold), so the sealed line's bytes are the frozen write's",
            "n_statements": len(fam["facts"]), "timer_overhead_ns": overhead}


def memory_run(journal_path: Path, key: bytes, bm25_pair: tuple[float, float], ckpt_path: Path) -> dict:
    """The memory in a run of its own with tracemalloc: the journal's in memory state after open (the entries and the
    payloads read), then per stack the traced size after its read side build minus before; bytes with the entry count."""
    from cortex_c2b.references import BM25Reference
    tracemalloc.start()
    out = {"journal_on_disk_bytes": journal_path.stat().st_size + sum(f.stat().st_size for f in record_tools.payload_dir_of(journal_path).iterdir()),
           "checkpoint_bytes": Path(ckpt_path).stat().st_size, "note": "numpy's buffers are traced; torch's are not: the model's weights are reported from the checkpoint's size, common"}
    b_open = tracemalloc.get_traced_memory()[0]
    with record_tools.opened_copy(journal_path, key, prefix="cost-memory-") as (journal, _):
        after_open = tracemalloc.get_traced_memory()[0]
        for e in journal._entries.values():
            journal.payloads.get(e.pointer)
        after_payloads = tracemalloc.get_traced_memory()[0]
        out["journal_open_bytes"] = after_open - b_open                      # the frozen open: the log replayed into the entries (and the copy's bookkeeping)
        out["journal_payloads_read_bytes"] = after_payloads - after_open     # the payloads read into the store's cache
        out["journal_in_memory_bytes"] = after_payloads - b_open             # the journal's in memory state after open: the entries and the payloads read (B.3)
        out["entries"] = len(journal)
        stacks = {}
        b0 = tracemalloc.get_traced_memory()[0]
        jp = JournalPath(journal, geometry=ADOPTED_GEOMETRY)
        stacks["I0"] = tracemalloc.get_traced_memory()[0] - b0
        b1 = tracemalloc.get_traced_memory()[0]
        ref1 = ExhaustiveCosine(journal, jp); [ref1._norm(e) for e in ref1.live_ids()]
        stacks["I1"] = tracemalloc.get_traced_memory()[0] - b1
        b2 = tracemalloc.get_traced_memory()[0]
        ref2 = BM25Reference(journal, *bm25_pair)                           # the measured stack itself (the live entries, as the reference bridge builds it)
        stacks["I2"] = tracemalloc.get_traced_memory()[0] - b2
        bm = ref2.bm25
        out["read_side_build_bytes"] = stacks
        out["index_sizes"] = {"I0": {"vectors": len(jp.index.vecs), "tables": len(jp.index.tables), "buckets": sum(len(t) for t in jp.index.tables)},
                              "I1": {"vectors": len(ref1.live_ids())}, "I2": {"documents": bm.n_docs, "tokens": len(bm.postings), "postings": sum(len(p) for p in bm.postings.values())}}
    tracemalloc.stop()
    return out


def maintenance_proxies(bm25_pair: tuple[float, float]) -> dict:
    """Two proxies, labelled as such: the tunable parameters per stack read from its configuration objects, and the lines of
    code per component."""
    from cortex_c2b import read_path, references
    return {"label": "proxies, not measurements of maintenance",
            "tunable_parameters": {"I0": {"n_tables": ADOPTED_GEOMETRY["n_tables"], "n_bits": ADOPTED_GEOMETRY["n_bits"], "signature": ADOPTED_GEOMETRY["signature"], "seed": ADOPTED_GEOMETRY["seed"], "count": 4},
                                   "I1": {"count": 0}, "I2": {"k1": bm25_pair[0], "b": bm25_pair[1], "count": 2}},
            "lines_of_code": {"I0": len(inspect.getsource(read_path.CueIndex).splitlines()) + len(inspect.getsource(read_path.JournalPath).splitlines()),
                              "I1": len(inspect.getsource(references.ExhaustiveCosine).splitlines()),
                              "I2": len(inspect.getsource(references.BM25Index).splitlines()) + len(inspect.getsource(references.tokens).splitlines())}}


def generation_stage(model, cfg, journal_path: Path, key: bytes, questions: list[dict], device, n: int = 50) -> dict:
    """The generation reported once per checkpoint as a common stage: `generate` on the first `n` planted questions with the
    organ's window (I0), one repetition after one discarded warm up."""
    import torch
    with record_tools.opened_copy(journal_path, key, prefix="cost-gen-") as (journal, _):
        bridge = JournalBridge(model, journal, k=cfg.journal_k, budget_bytes=cfg.journal_read_bytes, seed=0, shuffle_seed=1, device=device,
                               mark=bool(getattr(cfg, "journal_familiarity_mark", False)), index_geometry=ADOPTED_GEOMETRY, admission=journal.admission_record())
        times = []
        with torch.no_grad():
            for rep in range(2):
                for q in questions[:n]:
                    tok = encode_query(q["text"]); cue = bridge.cues([tok])[0]
                    read = bridge.read(cue)
                    wt, wm = bridge.window_tensors([read.window], cfg.journal_read_bytes)
                    with Timed() as t:
                        generate(model, tok, wt, wm, gate=1.0, policy="plain", window_bytes=read.window)
                    if rep == 1:
                        times.append(t.ns)
    return {**quantiles([float(x) for x in times]), "unit": "ns", "n": len(times), "label": "the generation, a common stage, once per checkpoint; one repetition after a discarded warm up"}


def _cli(argv=None):
    import train
    from cortex_c2b.sealed_family import questions_of, seal_check
    ap = argparse.ArgumentParser(description="the cost by stage of ADR-010 B.3 on a planted sealed family (new module; start it through tools/adr010_launcher.py --pin-threads)")
    ap.add_argument("--family", required=True, choices=["T0", "A", "Av"]); ap.add_argument("--ckpt", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--journal", required=True, help="the family's planted journal (its key file and planting record beside it)")
    ap.add_argument("--out", default="metrics/mqar"); ap.add_argument("--repetitions", type=int, default=REPETITIONS)
    ap.add_argument("--generation-questions", type=int, default=50)
    args = ap.parse_args(argv)
    args.argv = list(argv) if argv is not None else sys.argv[1:]          # the arguments as given, for the recorded command
    pinned = check_pinned()
    if not pinned["pinned"]:
        raise SystemExit(f"cost harness: the thread counts are not pinned to one ({pinned['problems']}); start it through tools/adr010_launcher.py --pin-threads (B.3)")
    RS.require_launcher()
    if "metrics" in Path(args.out).parts and not (record_tools.REPO_ROOT / "metrics/mqar/FREEZE-v1.1.0.json").exists():
        raise SystemExit("cost harness: metrics/mqar/FREEZE-v1.1.0.json must exist before a run of the record (python -m cortex_c2b.freeze --write --commit c5b4818d3db29902f72d1b3078e01c4b3cb1e634); refused")
    gc.callbacks.append(_gc_callback)
    cfg = train.load_config(args.config)
    jpath = Path(args.journal); key_file = jpath.with_name(jpath.stem + ".key"); sidecar = jpath.with_name(jpath.stem + ".planting.json")
    if not (jpath.is_file() and key_file.is_file() and sidecar.is_file()):
        raise SystemExit("cost harness: the family's journal, its key file and its planting record must exist")
    key = bytes.fromhex(key_file.read_text(encoding="utf-8").strip())
    planting = json.loads(sidecar.read_text(encoding="utf-8"))
    model, ck, device = load_model(args.ckpt, cfg)
    run_id = ck.get("run_id")
    out_path = Path(args.out) / f"cost-{args.family}-{run_id}.json"
    if out_path.exists():
        raise SystemExit(f"cost harness: {out_path} exists; a file of the record is never overwritten")
    fams = S.generate_all(); fam = fams[args.family]
    seal_info = seal_check(fam)
    questions = [q for q in questions_of(fam, "direct", fams) if q["block"] in ("planted", "never_planted")]
    thr = float(planting["threshold"]["threshold"])
    bm = bm25_selection(); pair = (bm["pair"]["k1"], bm["pair"]["b"])
    mark = bool(getattr(cfg, "journal_familiarity_mark", False))
    started = RS.utc_now()
    p1 = run_protocol(model, cfg, jpath, key, fam, questions, thr, pair, cfg.journal_k, cfg.journal_read_bytes, mark, device, args.repetitions, "protocol 1")
    flagged = [(ix, st) for ix, stages in p1["stages"].items() for st, v in stages.items() if isinstance(v, dict) and v.get("flag")]
    p2 = None
    if flagged:
        p2 = run_protocol(model, cfg, jpath, key, fam, questions, thr, pair, cfg.journal_k, cfg.journal_read_bytes, mark, device, args.repetitions, "protocol 2 (after an unstable repetition)")
        for ix, st in flagged:
            v2 = p2["stages"][ix][st]
            p1["stages"][ix][st]["label"] = "unstable on this processor" if v2.get("flag") else "a repetition out of band in protocol 1; protocol 2 stable, reported beside"
    gen = generation_stage(model, cfg, jpath, key, [q for q in questions if q["block"] == "planted"], device, args.generation_questions)
    ws = write_side(model, cfg, fam, planting, pair, device, args.repetitions)
    mem = memory_run(jpath, key, pair, Path(args.ckpt))
    out = {"benchmark": "the cost by stage (ADR-010 B.3; the amendment of 2026-10-15 on parts B and C)", "family": args.family, "run_id": run_id,
           "label": NOT_A_SCALE_RESULT, "pinned": pinned, "command": RS.command_line("cortex_c2b.cost_harness", getattr(args, "argv", None)), "launcher": "tools/adr010_launcher.py --pin-threads",
           "read_side": {"protocol_1": p1, "protocol_2": p2, "the_row_reads": "protocol 1; protocol 2, when present, is reported beside it",
                         "configurations": list(INDEXES), "rules": list(RULE_STAGES)},
           "generation": gen, "write_side": ws, "memory": mem, "maintenance_proxies": maintenance_proxies(pair), "bm25": bm,
           "threshold": thr, "seal": seal_info, "planting_sha256_lf": RS.sha256_file(sidecar, lf=True),
           "checkpoint": {"path": str(args.ckpt), "run_id": run_id, "config_hash": ck.get("config_hash"), "sha256": RS.sha256_file(args.ckpt)},
           "device": record_tools.device_info(device), "versions": record_tools.versions(), "environment": RS.numeric_environment(),
           "code_commit": record_tools.git_state()[0], "started_utc": started, "timestamp_utc": RS.utc_now()}
    sha = RS.write_json(out_path, out)
    print(f"retained: {out_path}  sha256 {sha}")
    short = {ix: {st: round(v["median"]) for st, v in p1["stages"][ix].items() if isinstance(v, dict) and "median" in v} for ix in INDEXES}
    print(json.dumps({"read_side_medians_ns": short, "generation_median_ns": gen.get("median"), "flagged": flagged, "memory": mem.get("read_side_build_bytes")}, indent=1))


if __name__ == "__main__":
    _cli()
