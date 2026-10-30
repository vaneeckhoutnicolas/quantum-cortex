"""cortex_c2b.scale_probe -- the scale cost probe with the cascade (ADR-010 part C, made exact by the amendment of
2026-10-15 on parts B and C, graved 2026-10-29; D3 A, D4 A, D6 B taken by the founder). A new module on the frozen system
`c5b4818`; row 53 is read cold from its files by `read_c4` below; nothing here is a number of the record.

Per draw (the seal's scale plans: seeds 50 000 to 50 004 at ten thousand facts, 60 000 at one hundred thousand), encoder
(a v9 checkpoint) and gate (the identity rule, the system's; the surprise rule, the comparison ADR-007 section 4 (c)
asks for), in this order, each step resumable as section 4 declares:

  encode    the planned statements (the facts in the seal's written order, each E1 or E2 partner right after its fact)
            and the questions (the facts' direct queries, the never planted controls' queries) encoded once in batches of
            the recorded size, each batch's cues written to its own file as it completes (4 (c)); against single sequence
            encoding on the 200 protocol statements of seed 0 the largest absolute cue difference, the stored components
            whose four decimal rounding changes (of 12 800) and the signatures that flip under the adopted geometry (of
            3 200) are recorded (C.1 (c));
  c2        the first planting, through the write path and the index without a scheduler (every admitted entry live, no
            sentinel, no shadow legacy gate: "shadow: not asked for"), the unit the write, a recorded point every sixty
            four writes; then C.2's numbers: the candidate count per query and the own line in the window at the system's
            k on the sample of two thousand, I0's top four against I1's exact top four ("the own line in both"; "the same
            set of four"; "the same first line"), the three lever variants (sixteen tables of twelve bits; thirty two of
            ten; thirty two of twelve, the exact signature), I2 beside, the never planted control's maximal marks against
            the oracle's threshold attributed by the distance to the nearest planted name, the index sizes, the time per
            query by B.3's protocol interleaved and once more in blocked order, and the read side selection at scale without
            the model (C.2 (b): every E1 and E2 pair's original asked once through I0, the marks, the pair's raw cosine, R1
            and R4 by the frozen `select_citation` with the citation taken as the duplicate and as the own line);
  cascade   the second planting from the same cues through the frozen `LifecycleScheduler` (the frozen configuration with
            the adopted geometry, `sentinel_seed` 2, the withdrawal on), the unit the write and the recorded point the
            phase, the checks of C.3 (a) after every phase (the memory's identities against the states), the reads at the
            read phases (the sample's written and unevicted positions, every planted pair's original), the reconstruction
            reading of row 42 (b1 to b3) at the read phases, the sessions under the budget the probe enforces (fourteen
            hours from the session's first write for the hundred thousand draw, D3 A), the stop file, the payload archive,
            the cut rule and D6 B at every reopening (`cortex_c2b.resumable`).

Every journal of this module is opened with the storage policy `stop`, sealed under a key generated per journal and kept
beside it (`<journal>.key`); the record's key is never read (4 (g)). The founder's commands are written in REPRISE with
this build; every run goes through `tools/adr010_launcher.py`.
"""
from __future__ import annotations

import argparse
import gc
import os
import json
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path

import numpy as np

from cortex_c2b import (Journal, POLICY_STOP, ADMISSION_IDENTITY, ADMISSION_SURPRISE, STATE_EVICTED, STATE_CONSOLIDATED, STATE_DEMOTED,
                        STATE_LIVE, SCHEMA_SUMMARY, record_tools)
from cortex_c2b.crypto import generate_key
from cortex_c2b.hm_protocol import generate_facts, N_FACTS, _TEMPLATES
from cortex_c2b.lm_bridge import JournalBridge, encode_query
from cortex_c2b.read_path import ADOPTED_GEOMETRY, CueIndex, JournalPath
from cortex_c2b.write_path import WritePath
from cortex_c2b.lifecycle import LifecycleScheduler, LifecycleConfig, PersistentMemory
from cortex_c2b.hm_lm import mark_oracle
from cortex_c2b.near_duplicate_probe import load_model, parse_statement, cosine as frozen_cosine
from cortex_c2b.near_duplicate_reading import select_citation, wilson, COSINE_BINS
from cortex_c2b.reconstruction_probe import level_3_memory
from cortex_c2b import sealed_families as S
from cortex_c2b.references import ExhaustiveCosine, BM25Index, bm25_selection
from cortex_c2b import resumable as RS
from cortex_c2b.sealed_reading import clopper_pearson

GATES = (ADMISSION_IDENTITY, ADMISSION_SURPRISE)
LEVER_VARIANTS = (("16x12", 16, 12), ("32x10", 32, 10), ("32x12", 32, 12))
READ_PHASES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
READ_PHASES_100K = READ_PHASES + (1024, 2048, 4096)
C2_POINT_EVERY = 64
SENTINEL_SEED = 2
DEFAULT_BATCH = 64
DEFAULT_BUDGET_100K = 14 * 3600   # D3 A: fourteen hours from the session's first write, the hundred thousand cascade's default
FREEZE_PATH_REL = Path("metrics/mqar/FREEZE-v1.1.0.json")
FREEZE_COMMIT_FULL = "c5b4818d3db29902f72d1b3078e01c4b3cb1e634"
TIME_QUERIES = 200                # the time per query at C's sizes is measured on the first two hundred sample questions
SIX_ORDERS = [("I0", "I1", "I2"), ("I0", "I2", "I1"), ("I1", "I0", "I2"), ("I1", "I2", "I0"), ("I2", "I0", "I1"), ("I2", "I1", "I0")]
TIME_LABEL = "the time is the lifecycle's bookkeeping as frozen, not the index's or the memory's"
READ_SIDE_LABEL = "the read side at scale, without the model: the mark's ranking and the key's selection on fresh pairs; the model's choice is not here"
KEY_CONTRACT_LABEL = "the key's contract on an entity with two attributes, nine entities, no claim"
NEVER_PLANTED_LABEL = ("the share of never planted queries outside the veto's protection at this size, under the generator's alphabet, the training "
                       "family's threshold read at 200 entries and the first family's templates; not a claim rate of the system")
DECISION_13_SUCCESSOR = ("read out of sample on fresh pairs without the model (row 53): the mark ranks the duplicate first on n of N contested reads "
                         "per size and encoder")


def cascade_config() -> LifecycleConfig:
    """The frozen configuration with the adopted geometry, the sentinel seed 2 and its negatives at 10 002 (rows 41 and 42),
    the withdrawal on (Decision 11); named by its hash."""
    return LifecycleConfig(index_tables=ADOPTED_GEOMETRY["n_tables"], index_signature=ADOPTED_GEOMETRY["signature"],
                           sentinel_seed=SENTINEL_SEED, withdraw_at_eviction=True)


# ---------------------------------------------------------------------------- #
# the plan                                                                        #
# ---------------------------------------------------------------------------- #
def plan_of(seed: int, check_seal: bool = True, test_n: int | None = None) -> dict:
    """The draw's plan from the seal's constants, checked against the seal's content hash; the planned statements in order
    (each fact, then its partner), the questions, the never planted controls, the sample, the repeated names (C.1 (b)).
    `test_n`: a plan of that many facts under this seed, of size "test", never a draw of the seal (the tests' fixtures)."""
    if test_n is not None:
        return plan_from(S.scale_plan(seed, int(test_n), seed + 100), seed, seed + 100, "test", check_seal=False)
    spec = next((v for v in S.SCALE_DRAWS.values() if seed in v["seeds"]), None)
    if spec is None:
        raise ValueError(f"{seed} is not a scale draw of the seal")
    nseed = spec["never_planted_seeds"][spec["seeds"].index(seed)]
    return plan_from(S.scale_plan(seed, spec["n"], nseed), seed, nseed, next(k for k, v in S.SCALE_DRAWS.items() if seed in v["seeds"]), check_seal)


def plan_from(p: dict, seed: int, nseed: int, size: str, check_seal: bool = True) -> dict:
    sealed = None
    if check_seal and S.SEAL_PATH.exists():
        seal = json.loads(S.SEAL_PATH.read_text(encoding="utf-8"))
        sealed = (seal.get("scale_draws") or {}).get(str(seed), {}).get("content_sha256")
        if sealed is not None and sealed != p["content_sha256"]:
            raise ValueError(f"the draw {seed} regenerates to {p['content_sha256']}, the seal says {sealed}; refused")
    inner = p["_plan"]
    after: dict[int, list] = {}
    for q in inner["partners"]:
        if not q.get("dropped"):
            after.setdefault(q["after"], []).append(q)
    planned = []
    for i, f in enumerate(inner["facts"]):
        planned.append({"j": len(planned), "kind": "fact", "entity": f.entity, "schema": f.schema, "attribute": str(f.attr), "statement": f.statement,
                        "query": f.query, "position": i})
        for q in after.get(i, []):
            planned.append({"j": len(planned), "kind": q["kind"], "entity": (q.get("neighbour") or q["entity"]), "for_entity": q["entity"],
                            "schema": f.schema, "attribute": q["attribute"], "statement": q["statement"], "query": None, "position": i})
    names = [x["entity"] for x in planned if x["kind"] == "fact"]
    rep = sorted(n for n, c in Counter(names).items() if c > 1)
    return {"seed": seed, "n_facts": len(inner["facts"]), "never_seed": nseed, "size": size, "content_sha256": p["content_sha256"],
            "planned": planned, "n_planned": len(planned), "never": inner["never"], "sample": inner["sample"],
            "partners": {"E1": sum(1 for x in planned if x["kind"] == "E1"), "E2": sum(1 for x in planned if x["kind"] == "E2")},
            "repeated_names": rep, "repeated_detail": repeated_detail(planned, rep), "seal_checked": sealed is not None,
            "expected_phases_identity": len(planned) // 16}


def repeated_detail(planned: list[dict], repeated: list[str]) -> dict:
    """C.1 (b): per repeated name its facts (plan index, schema, attribute); cross schema ones keyed by (entity, schema); same
    schema ones with the first written statement as the own line, the later an E1 duplicate by chance; the partnered ones."""
    facts_by_name: dict[str, list] = {}
    for x in planned:
        if x["kind"] == "fact" and x["entity"] in repeated:
            facts_by_name.setdefault(x["entity"], []).append(x)
    partnered: dict[str, list] = {}
    for x in planned:
        if x["kind"] in ("E1", "E2") and x["for_entity"] in repeated:
            partnered.setdefault(x["for_entity"], []).append(x["j"])
    out = {}
    for name in repeated:
        facts = facts_by_name[name]; schemas = [x["schema"] for x in facts]
        cross = len(set(schemas)) == len(schemas)
        out[name] = {"facts": [{"j": x["j"], "schema": x["schema"], "attribute": x["attribute"]} for x in facts], "cross_schema": cross,
                     "own_line_rule": ("(entity, schema)" if cross else "the first written statement; the later one an E1 duplicate by chance"),
                     "partnered": partnered.get(name, [])}
    return out


def questions_of_plan(plan: dict) -> list[dict]:
    """The questions encoded: every fact's direct query, then the never planted controls' queries."""
    qs = []
    for x in plan["planned"]:
        if x["kind"] == "fact":
            qs.append({"q": len(qs), "kind": "fact", "j": x["j"], "entity": x["entity"], "schema": x["schema"], "text": x["query"]})
    for v in plan["never"]:
        qs.append({"q": len(qs), "kind": "never", "entity": v["entity"], "schema": v["schema"], "text": v["query"], "distance": v["distance"],
                   "nearest_planted": v["nearest_planted"]})
    return qs


def indexes_of_plan(plan: dict) -> dict:
    """Lookups the steps share: the plan index of a fact by its position, the question index of a fact's plan index, the
    pairs (the original's plan index, the partner's, the kind)."""
    planned = plan["planned"]
    j_of_position = {x["position"]: x["j"] for x in planned if x["kind"] == "fact"}
    q_of_j = {q["j"]: q["q"] for q in questions_of_plan(plan) if q["kind"] == "fact"}
    pairs = [(j_of_position[x["position"]], x["j"], x["kind"]) for x in planned if x["kind"] in ("E1", "E2")]
    return {"j_of_position": j_of_position, "q_of_j": q_of_j, "pairs": pairs, "sample_js": [j_of_position[p] for p in plan["sample"]]}


# ---------------------------------------------------------------------------- #
# paths and identity                                                              #
# ---------------------------------------------------------------------------- #
def run_dir(journals_root: Path, seed: int, run_id: str) -> Path:
    return Path(journals_root) / f"scale-{seed}-{run_id}"


def probe_config(args, cfg_hash: str, plan: dict, step: str, gate: str | None) -> dict:
    """The probe's configuration, hashed into the identity (4 (b)): the LifecycleConfig named by its hash, the batch size,
    the storage policy, the stop file's name (its name, not its path: a run moved to another folder and resumed there by
    the same command keeps its identity; the path as given is in the identity's recorded part)."""
    return {"step": step, "draw": plan["seed"], "size": plan["size"], "gate": gate, "batch_size": args.batch_size, "storage_policy": POLICY_STOP,
            "plaintext": bool(getattr(args, "plaintext", False)),
            "stop_file": Path(args.stop_file).name if args.stop_file else None, "lifecycle_config_hash": cascade_config().config_hash(),
            "read_phases": list(READ_PHASES_100K if plan["size"] == "hundred_thousand" else READ_PHASES) + ["last"], "c2_point_every": C2_POINT_EVERY,
            "config_hash": cfg_hash, "index_geometry": ADOPTED_GEOMETRY, "time_queries": TIME_QUERIES, "k": "journal_k of the configuration"}


def identity_for(args, run_id: str, plan: dict, step: str, gate: str | None, cfg_hash: str, extra_compared: dict | None = None) -> dict:
    root = record_tools.REPO_ROOT
    from cortex_c2b.freeze import FREEZE_PATH, tree_hash_of_list
    fz = root / FREEZE_PATH
    compared = {"kind": f"scale probe {step}", "draw": plan["seed"], "size": plan["size"], "gate": gate, "run_id": run_id,
                "probe_config_sha256": RS.sha256_text(probe_config(args, cfg_hash, plan, step, gate)),
                "plan_content_sha256": plan["content_sha256"], "checkpoint_sha256": RS.sha256_file(args.ckpt),
                "seal_sha256": RS.sha256_file(S.SEAL_PATH) if S.SEAL_PATH.exists() else None,
                "freeze_sha256": RS.sha256_file(fz) if fz.exists() else None,
                "frozen_tree_sha256": (tree_hash_of_list(json.loads(fz.read_text(encoding="utf-8")), root) if fz.exists() else None),
                "admission": gate, "environment": RS.numeric_environment(), **(extra_compared or {})}
    recorded = {"command": RS.command_line("cortex_c2b.scale_probe", getattr(args, "argv", None)), "budget_seconds": args.budget_seconds, "stop_file": args.stop_file, "out": str(args.out),
                "journals": str(args.journals)}
    return RS.make_identity(compared, recorded)


OPEN_KEYS = ("log_after_open", "index_state_after_open")


def check_identity(progress: RS.Progress, identity: dict, log_path: Path | None = None) -> None:
    """The recorded identity against the current one; the log after the scheduler's open compared as a prefix hash."""
    rec = progress.identity() or {}
    diffs = RS.compare_identity(rec, identity["compared"], ignore=OPEN_KEYS)
    want = rec.get("compared", {}).get("log_after_open")
    if log_path is not None and want:
        if not Path(log_path).exists() or RS.prefix_sha256(log_path, want["length"]) != want["sha256"]:
            diffs.append(f"log_after_open: the log's first {want['length']} bytes do not hash to the recorded {want['sha256']}")
    if diffs:
        raise RS.ResumptionRefused("the run identity differs; the resumption is refused (4 (b)):\n  " + "\n  ".join(diffs))


def cfg_hash_of(cfg) -> str:
    import train
    return train.config_hash(cfg)


# ---------------------------------------------------------------------------- #
# step 1: the encodings (C.1 (c))                                                  #
# ---------------------------------------------------------------------------- #
def encode_batched(bridge: JournalBridge, sequences: list[list[int]], batch_size: int) -> np.ndarray:
    out = []
    for b in range(0, len(sequences), batch_size):
        out.append(bridge.cues(sequences[b:b + batch_size]))
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


def batched_against_single(bridge: JournalBridge, batch_size: int) -> dict:
    """C.1 (c): the 200 protocol statements of seed 0 encoded one sequence per forward pass and in batches: the largest
    absolute cue difference, the stored components whose four decimal rounding changes, the signatures that flip."""
    facts, _ = generate_facts(N_FACTS, 0)
    seqs = [list(f.statement.encode("utf-8")) for f in facts]
    single = np.stack([bridge.cues([s])[0] for s in seqs])
    batched = encode_batched(bridge, seqs, batch_size)
    diff = float(np.max(np.abs(single - batched)))
    rs, rb = np.round(single, 4), np.round(batched, 4)
    comp = int((rs != rb).sum())
    idx = CueIndex(dim=int(single.shape[1]), n_bits=ADOPTED_GEOMETRY["n_bits"], n_tables=ADOPTED_GEOMETRY["n_tables"], seed=ADOPTED_GEOMETRY["seed"], signature=ADOPTED_GEOMETRY["signature"])
    flips = 0
    for a, b in zip(rs.astype(np.float32), rb.astype(np.float32)):
        flips += sum(1 for t in range(idx.n_tables) if idx._sig(a, t) != idx._sig(b, t))
    return {"batch_size": batch_size, "largest_abs_cue_difference": diff, "components_rounding_changed": comp, "components": int(rs.size),
            "signatures_flipped": flips, "signatures": int(rs.shape[0] * idx.n_tables), "index_geometry": ADOPTED_GEOMETRY}


def run_encode(args, model, cfg, device, plan: dict, run_id: str, rdir: Path) -> dict:
    """The statements and the questions encoded once, batch by batch, each batch to its own file (4 (c)); resumable at the
    batch. Writes the manifest `scale-encoding-<draw>-<run id>.json`."""
    import torch
    cdir = rdir / "cues"; cdir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) / f"scale-encoding-{plan['seed']}-{run_id}.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    identity = identity_for(args, run_id, plan, "encode", None, cfg_hash_of(cfg))
    progress = RS.Progress(cdir / "encoding.progress.jsonl")
    if progress.exists():
        check_identity(progress, identity)
    else:
        progress.write_identity(identity)
    bridge = JournalBridge(model, Journal(), k=cfg.journal_k, budget_bytes=cfg.journal_read_bytes, seed=0, shuffle_seed=1, device=device)
    B = args.batch_size
    sets = {"statements": [list(x["statement"].encode("utf-8")) for x in plan["planned"]],
            "questions": [encode_query(q["text"]) for q in questions_of_plan(plan)]}
    done = {(u["set"], int(u["batch"])): u for u in progress.units().values()}
    progress.start_session({"wall_clock": RS.utc_now(), "command": RS.command_line("cortex_c2b.scale_probe", getattr(args, "argv", None)), "resumed_from": (len(done) or None),
                            "torn_progress_tail": progress.torn_tail})
    stop = RS.StopFile(args.stop_file)
    files = {}
    unit_index = max((int(k) for k in progress.units()), default=-1) + 1
    with torch.no_grad():
        for name, seqs in sets.items():
            for b in range(0, len(seqs), B):
                bi = b // B
                f = cdir / f"{name}-b{bi}.npy"
                if (name, bi) in done and f.exists() and RS.sha256_file(f) == done[(name, bi)]["sha256"]:
                    files[f"{name}-b{bi}"] = done[(name, bi)]["sha256"]; continue
                if stop.requested():
                    progress.append({"kind": "session_end", "wall_clock": RS.utc_now(), "reason": "stop_file"}); progress.close()
                    print("scale probe: encoding stopped by the stop file; the same command resumes"); return {"stopped": True}
                cues = bridge.cues(seqs[b:b + B]).astype(np.float32)
                tmp = f.with_name(f.name + ".tmp")
                with open(tmp, "wb") as fh:                  # a handle: numpy would add its suffix to a bare name
                    np.save(fh, cues); fh.flush(); os.fsync(fh.fileno())
                tmp.replace(f)
                sha = RS.sha256_file(f)
                progress.append({"kind": "unit", "index": unit_index, "set": name, "batch": bi, "first": b, "n": int(cues.shape[0]), "sha256": sha, "wall_clock": RS.utc_now()})
                unit_index += 1
                files[f"{name}-b{bi}"] = sha
        comparison = batched_against_single(bridge, B)
    progress.append({"kind": "session_end", "wall_clock": RS.utc_now(), "reason": "complete"}); progress.close()
    manifest = {"benchmark": "the encodings of ADR-010 C (C.1 (c))", "draw": plan["seed"], "size": plan["size"], "run_id": run_id, "batch_size": B,
                "n_statements": len(sets["statements"]), "n_questions": len(sets["questions"]), "files": files, "batched_against_single": comparison,
                "plan_content_sha256": plan["content_sha256"], "repeated_names": plan["repeated_detail"], "identity": progress.identity(), "sessions": progress.sessions(),
                "device": record_tools.device_info(device), "versions": record_tools.versions(), "timestamp_utc": RS.utc_now()}
    RS.write_json(out_path, manifest)
    print(f"retained: {out_path}")
    return manifest


def load_cues(rdir: Path, plan: dict, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    cdir = rdir / "cues"
    n_s, n_q = plan["n_planned"], len(questions_of_plan(plan))
    st = np.concatenate([np.load(cdir / f"statements-b{bi}.npy") for bi in range((n_s + batch_size - 1) // batch_size)], axis=0)
    qs = np.concatenate([np.load(cdir / f"questions-b{bi}.npy") for bi in range((n_q + batch_size - 1) // batch_size)], axis=0)
    if st.shape[0] != n_s or qs.shape[0] != n_q:
        raise ValueError(f"the cue files hold {st.shape[0]} statements and {qs.shape[0]} questions; the plan has {n_s} and {n_q}")
    return st.astype(np.float32), qs.astype(np.float32)


# ---------------------------------------------------------------------------- #
# journals of the probe                                                            #
# ---------------------------------------------------------------------------- #
def new_journal(jdir: Path, plaintext: bool = False) -> tuple[Journal, bytes | None]:
    """A fresh journal of the probe under a key generated for it and kept beside it (4 (g)); `plaintext` is the tests'
    fixture (refused under metrics/). A folder that already holds a journal but no progress file (a death before the first
    recorded line) is moved aside whole, never deleted or reused."""
    if jdir.exists() and any(jdir.iterdir()):
        n = 1
        while jdir.with_name(jdir.name + f".aborted-{n}").exists():
            n += 1
        jdir.rename(jdir.with_name(jdir.name + f".aborted-{n}"))
    jdir.mkdir(parents=True, exist_ok=True)
    jpath = jdir / "journal.jsonl"
    if plaintext:
        (jdir / "journal.plaintext").write_text("the tests' fixture: an unsealed journal\n", encoding="utf-8", newline="\n")
        return Journal(jpath, plaintext=True, policy=POLICY_STOP, index_geometry=ADOPTED_GEOMETRY), None
    key = generate_key()
    jpath.with_name(jpath.stem + ".key").write_text(key.hex(), encoding="utf-8", newline="\n")
    return Journal(jpath, key=key, policy=POLICY_STOP, index_geometry=ADOPTED_GEOMETRY), key


def key_of(jdir: Path) -> bytes | None:
    """The journal's key from its file beside it; None for the tests' plaintext fixture."""
    if (jdir / "journal.plaintext").exists():
        return None
    return bytes.fromhex((jdir / "journal.key").read_text(encoding="utf-8").strip())


def key_sha_of(jdir: Path) -> str:
    return RS.sha256_file(jdir / "journal.key", lf=True) if (jdir / "journal.key").exists() else "plaintext (the tests' fixture)"


def open_journal(jdir: Path) -> Journal:
    key = key_of(jdir)
    return Journal(jdir / "journal.jsonl", key=key, plaintext=(key is None), policy=POLICY_STOP, index_geometry=ADOPTED_GEOMETRY)


def archive_pointers(journal: Journal, archive: RS.PayloadArchive, pointers) -> None:
    archive.append_from_store(record_tools.payload_dir_of(journal._path), [p for p in pointers if p])


def reopen(jdir: Path, progress: RS.Progress, archive: RS.PayloadArchive, gate: str) -> tuple:
    """A journal of the probe reopened at its last recorded point (4 (d), D6 B): the cut rule and the payload directory, the
    frozen open, the write path and the journal path, the live index restored. Returns (journal, key, wp, jp, start_j,
    tracker, cut_record, restored)."""
    jpath = jdir / "journal.jsonl"
    key = key_of(jdir)
    units = progress.units()
    ident = progress.identity()["compared"]
    if units:
        last = units[max(units)]
        point = {"length": last["log"]["length"], "sha256": last["log"]["sha256"], "archive_length": last["archive_length"]}
        start_j = int(last["next_j"]); touch = last.get("index_state") or {}
    else:
        rec0 = ident["log_after_open"]
        point = {"length": rec0["length"], "sha256": rec0["sha256"], "archive_length": rec0.get("archive_length", 0)}
        start_j = 0; touch = ident.get("index_state_after_open") or {}
    cut_record = RS.reconcile_journal(jpath, key, point, archive, plaintext=(key is None))
    journal = open_journal(jdir)
    wp = WritePath(journal, seed=0, admission=gate)
    jp = JournalPath(journal, geometry=ADOPTED_GEOMETRY)
    restored = RS.restore_live_index(jp, journal, touch.get("touch_window"), touch.get("touched"))
    tracker = RS.LogTracker(jpath, length=point["length"])
    return journal, key, wp, jp, start_j, tracker, cut_record, restored


# ---------------------------------------------------------------------------- #
# step 2: C.2, the planting without a scheduler, then its numbers                 #
# ---------------------------------------------------------------------------- #
def run_c2_planting(args, plan: dict, run_id: str, rdir: Path, gate: str, st_cues: np.ndarray, cfg_hash: str) -> dict:
    """The first planting (C.2): the write path and the index without a scheduler, no shadow ("shadow: not asked for"); the unit
    the write, a recorded point every sixty four writes (4 (c)); resumable at the recorded point (4 (d))."""
    jdir = rdir / f"c2-{gate}"; jpath = jdir / "journal.jsonl"
    if (jdir / "planting.json").exists():
        return json.loads((jdir / "planting.json").read_text(encoding="utf-8"))
    identity = identity_for(args, run_id, plan, "c2-planting", gate, cfg_hash)
    stop = RS.StopFile(args.stop_file); budget = RS.Budget(args.budget_seconds)
    t_open = time.monotonic()
    cut_record = restored = None
    if RS.Progress(jdir / "planting.progress.jsonl").exists():
        archive = RS.PayloadArchive(RS.PayloadArchive.path_for(jpath))
        progress = RS.Progress(jdir / "planting.progress.jsonl", sync_paths=[jpath], archive=archive)
        identity["compared"]["journal_key_sha256"] = key_sha_of(jdir)
        check_identity(progress, identity, jpath)
        journal, key, wp, jp, start_j, tracker, cut_record, restored = reopen(jdir, progress, archive, gate)
        resumed_from = start_j
    else:
        journal, key = new_journal(jdir, plaintext=bool(getattr(args, "plaintext", False)))
        archive = RS.PayloadArchive(RS.PayloadArchive.path_for(jpath))
        progress = RS.Progress(jdir / "planting.progress.jsonl", sync_paths=[jpath], archive=archive)
        wp = WritePath(journal, seed=0, admission=gate)
        jp = JournalPath(journal, geometry=ADOPTED_GEOMETRY)
        tracker = RS.LogTracker(jpath)
        identity["compared"]["journal_key_sha256"] = key_sha_of(jdir)
        identity["compared"]["log_after_open"] = {**tracker.update(), "archive_length": archive.length}
        identity["compared"]["index_state_after_open"] = RS.index_state(jp)
        progress.write_identity(identity)
        start_j = 0; resumed_from = None
    progress.start_session({"wall_clock": RS.utc_now(), "command": RS.command_line("cortex_c2b.scale_probe", getattr(args, "argv", None)), "resumed_from": resumed_from,
                            "reopening_seconds": round(time.monotonic() - t_open, 3), "restored": (cut_record or {}).get("restored"), "moved": (cut_record or {}).get("moved"),
                            "budget_seconds": args.budget_seconds, "torn_progress_tail": progress.torn_tail, "first_unit": (start_j // C2_POINT_EVERY) + 1})
    if cut_record:
        progress.append(cut_record)
    planned = plan["planned"]
    decisions = []
    reason = "complete"
    j = start_j
    try:
        while j < len(planned):
            budget.start()
            x = planned[j]
            rep = wp.write(st_cues[j], x["statement"].encode("utf-8"), x["schema"], now=float(j), source=x["statement"])
            if rep.admitted:
                jp.on_write(rep.entry)
                archive_pointers(journal, archive, [rep.entry.pointer, rep.entry.source_pointer])
            elif rep.revised and rep.entry is not None:
                archive_pointers(journal, archive, [rep.entry.pointer])
            decisions.append({"j": j, "kind": x["kind"], "admitted": bool(rep.admitted), "surprise": float(rep.surprise), "entry_id": (rep.entry.entry_id if rep.admitted else None),
                              "pointer": (rep.entry.pointer if rep.admitted else None), "nearest": rep.nearest_entry_id, "nearest_cosine": rep.nearest_cosine, "revised": bool(rep.revised)})
            j += 1
            if j % C2_POINT_EVERY == 0 or j == len(planned):
                progress.append({"kind": "unit", "index": (j + C2_POINT_EVERY - 1) // C2_POINT_EVERY, "next_j": j, "decisions": decisions, "log": tracker.update(),
                                 "archive_length": archive.length, "index_state": RS.index_state(jp), "wall_clock": RS.utc_now(), "entries": len(journal)})
                decisions = []
                if j < len(planned) and (stop.requested() or budget.exceeded()):     # a stop lands on a recorded point (4 (e))
                    reason = "stop_file" if stop.requested() else "budget"; break
    except Exception:
        f = RS.write_traceback(progress.path, f"plan index {j}")
        progress.append({"kind": "death", "wall_clock": RS.utc_now(), "unit": f"plan index {j}", "traceback_file": str(f)}); progress.close(); archive.close()
        raise
    progress.append({"kind": "session_end", "wall_clock": RS.utc_now(), "reason": reason, "last_unit": (j + C2_POINT_EVERY - 1) // C2_POINT_EVERY,
                     "resumption_point": (None if reason == "complete" else j)})
    progress.close(); archive.close()
    if reason != "complete":
        print(f"scale probe: C.2 planting stopped ({reason}) at plan index {j}; the same command resumes there")
        return {"stopped": reason}
    base = RS.assemble(progress)
    all_dec = [d for u in base["units"] for d in u["decisions"]]
    by_kind: dict = {}
    for d in all_dec:
        k = by_kind.setdefault(d["kind"], {"n": 0, "admitted": 0, "_s": []})
        k["n"] += 1; k["admitted"] += int(d["admitted"])
        if d["admitted"]:
            k["_s"].append(d["surprise"])
    for k in by_kind.values():
        s = np.asarray(k.pop("_s"), dtype=np.float64)
        k["salience"] = ({"median": float(np.median(s)), "min": float(s.min()), "max": float(s.max()), "under_floor_0.2": int((s < 0.2).sum())} if s.size else None)
    rec = {"step": "c2-planting", "gate": gate, "draw": plan["seed"], "size": plan["size"], "run_id": run_id, "shadow": "not asked for",
           "admission_record": journal.admission_record(), "entries": len(journal), "admitted": sum(1 for d in all_dec if d["admitted"]),
           "refused": sum(1 for d in all_dec if not d["admitted"]), "by_kind": by_kind, "phases": 0, "sentinels": 0,
           "decisions_sha256": RS.sha256_text(all_dec), "n_decisions": len(all_dec), "identity": base["identity"], "sessions": base["sessions"],
           "run_in_sessions": base["run_in_sessions"], "resumption_fault": base["resumption_fault"], "progress_sha256": base["progress_sha256"],
           "journal": {"path": str(jpath), "log": tracker.point(), "archive_length": archive.length}, "restored_index_at_last_reopening": restored, "timestamp_utc": RS.utc_now()}
    (jdir / "decisions.json").write_text(json.dumps(all_dec, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    (jdir / "planting.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    return rec


def threshold_for(model, cfg, device, admission: str = ADMISSION_IDENTITY) -> float:
    """The oracle's threshold as rows 47 and 49 read it (the training family, the legacy geometry for the scratch journal, the
    identity rule), by the frozen `mark_oracle` with no protocol read."""
    scratch_bridge = JournalBridge(model, Journal(), k=cfg.journal_k, budget_bytes=cfg.journal_read_bytes, seed=0, shuffle_seed=1, device=device,
                                   mark=bool(getattr(cfg, "journal_familiarity_mark", False)), index_geometry=ADOPTED_GEOMETRY, admission=admission)
    return float(mark_oracle(model, scratch_bridge, [], [], {}, admission=admission)["threshold"])


def raw_cosines_of_pairs(plan: dict, st_cues: np.ndarray, lookups: dict) -> dict[int, float]:
    """The pair's raw cosine by the partner's plan index: the two cues as encoded, before separation and rounding, by the
    frozen `near_duplicate_probe.cosine` (row 49's `cos_raw`: float64, each norm raised by 1e-12)."""
    return {pj: frozen_cosine(st_cues[oj], st_cues[pj]) for oj, pj, _ in lookups["pairs"]}


def c2_numbers(args, plan: dict, run_id: str, rdir: Path, gate: str, st_cues: np.ndarray, q_cues: np.ndarray, threshold: float, thr47: float | None,
               planting: dict, cfg_hash: str, k: int = 4) -> dict:
    """C.2's numbers on the first planting, without the model: the reads (the sample, the never planted controls, the pairs'
    originals), one line each and resumable (4 (c)); then the time per query (atomic), the index sizes, the aggregates."""
    jdir = rdir / f"c2-{gate}"
    out_path = Path(args.out) / f"scale-c2-{plan['seed']}-{run_id}-{gate}.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    journal = open_journal(jdir)
    jp = JournalPath(journal, geometry=ADOPTED_GEOMETRY)
    decisions = json.loads((jdir / "decisions.json").read_text(encoding="utf-8"))
    entry_of_j = {d["j"]: d["entry_id"] for d in decisions if d["admitted"]}
    pointer_of_j = {d["j"]: d["pointer"] for d in decisions if d["admitted"]}
    planned = plan["planned"]
    qs = questions_of_plan(plan)
    L = indexes_of_plan(plan)
    raw = raw_cosines_of_pairs(plan, st_cues, L)
    progress = RS.Progress(jdir / "reads.progress.jsonl")
    identity = identity_for(args, run_id, plan, "c2-reads", gate, cfg_hash, {"threshold": threshold, "planting_decisions_sha256": planting["decisions_sha256"], "k": k})
    if progress.exists():
        check_identity(progress, identity)
    else:
        progress.write_identity(identity)
    progress.start_session({"wall_clock": RS.utc_now(), "command": RS.command_line("cortex_c2b.scale_probe", getattr(args, "argv", None)), "torn_progress_tail": progress.torn_tail})
    stop = RS.StopFile(args.stop_file)
    live_ids = [eid for eid, e in journal._entries.items() if e.state != STATE_EVICTED]
    ex = ExhaustiveCosine(journal, jp); ex.hold(True)                      # a static journal: the live corpus held between reads
    bm_pair = bm25_selection()["pair"]
    bm = BM25Index(bm_pair["k1"], bm_pair["b"])
    for eid in live_ids:
        bm.insert(eid, journal.payloads.get(journal._entries[eid].pointer))
    levers = {}
    for name, nt, nb in LEVER_VARIANTS:
        ix = CueIndex(dim=journal.cue_dim, n_bits=nb, n_tables=nt, seed=0, signature="exact")
        for eid in live_ids:
            ix.add(eid, np.asarray(journal._entries[eid].cue, dtype=np.float32))
        levers[name] = ix
    repeated = set(plan["repeated_names"])
    other_line_of: dict[int, list] = {}                                     # a cross schema repeated name: the other line(s) of the name (C.1 (b), the key's contract)
    for name, det in plan["repeated_detail"].items():
        js = [f["j"] for f in det["facts"]]
        for j_ in js:
            other_line_of[j_] = [entry_of_j[o] for o in js if o != j_ and o in entry_of_j]
    reads = []
    for j in L["sample_js"]:
        x = planned[j]
        reads.append({"kind": "sample", "j": j, "position": x["position"], "q": L["q_of_j"][j], "entity": x["entity"], "schema": x["schema"], "text": x["query"],
                      "repeated_name": x["entity"] in repeated, "cross_schema_repeated": bool(x["entity"] in repeated and plan["repeated_detail"][x["entity"]]["cross_schema"])})
    for q in qs:
        if q["kind"] == "never":
            reads.append({"kind": "never", "q": q["q"], "entity": q["entity"], "schema": q["schema"], "distance": q["distance"], "nearest_planted": q["nearest_planted"], "text": q["text"]})
    for oj, pj, kind in L["pairs"]:
        reads.append({"kind": f"pair-{kind}", "j": oj, "partner_j": pj, "q": L["q_of_j"][oj], "entity": planned[oj]["entity"], "schema": planned[oj]["schema"],
                      "for_entity": planned[pj]["for_entity"], "partner_entity": planned[pj]["entity"], "text": planned[oj]["query"], "raw_cosine": raw[pj],
                      "repeated_name": planned[oj]["entity"] in repeated})
    for i, r in enumerate(reads):
        r["index"] = i
    done = progress.units()
    try:
        for r in reads:
            if r["index"] in done:
                continue
            if stop.requested():
                progress.append({"kind": "session_end", "wall_clock": RS.utc_now(), "reason": "stop_file"}); progress.close()
                print("scale probe: C.2 reads stopped by the stop file; the same command resumes"); return {"stopped": True}
            line = read_once(r, q_cues[r["q"]], journal, jp, ex, bm, levers, entry_of_j, pointer_of_j, threshold, k, other_line_of)
            line.update({"kind": "unit", "index": r["index"], "read": r, "wall_clock": RS.utc_now()})
            progress.append(line)
        timing = time_per_query(q_cues, [r for r in reads if r["kind"] == "sample"][:TIME_QUERIES], jp, ex, bm, k)
        sizes = index_sizes(journal, jp, bm, levers)
    except Exception:
        f = RS.write_traceback(progress.path, "the C.2 reads")
        progress.append({"kind": "death", "wall_clock": RS.utc_now(), "unit": "the C.2 reads", "traceback_file": str(f)}); progress.close()
        raise
    progress.append({"kind": "session_end", "wall_clock": RS.utc_now(), "reason": "complete"}); progress.close()
    base = RS.assemble(progress)
    agg = aggregate_c2(base["units"], plan)
    out = {"benchmark": "ADR-010 C.2 and C.2 (b): the cost at scale and the read side selection without the model", "draw": plan["seed"], "size": plan["size"],
           "run_id": run_id, "gate": gate, "entries": len(journal), "planting": {k_: v for k_, v in planting.items() if k_ not in ("sessions",)},
           "threshold": {"value": threshold, "row47": thr47, "within_1e-6": (abs(threshold - thr47) <= 1e-6) if thr47 is not None else None,
                         "read": "the training family, the legacy geometry for the scratch journal, the identity rule (rows 47 and 49)",
                         "boundary": "a never planted window counts at or above the threshold (the veto's boundary, >=); the frozen oracle claims strictly above (>)"},
           "k": k, "k_label": "the system's k (journal_k of the checkpoint's configuration)", "bm25": bm_pair, "lever_variants": [list(v) for v in LEVER_VARIANTS],
           "aggregates": agg, "time_per_query": timing, "index_sizes": sizes,
           "labels": {"read_side": READ_SIDE_LABEL, "key_contract": KEY_CONTRACT_LABEL, "never_planted": NEVER_PLANTED_LABEL, "decision_13_successor": DECISION_13_SUCCESSOR,
                      "levers": "the lever variants' curves choose nothing; a lever chosen on them later is labelled in sample to these journals and read on a later one",
                      "marks": "I0's marks are the index's own scores (float32), as the organ's bridge reads them; the frozen `marks_for` (float64) differs at the seventh decimal",
                      "mark_ranks_duplicate_first": "the raw comparison of the two marks at full precision; R1's selection under the tie band (1e-4) is beside it in `selection`"},
           "chance_near_twins": {"from_the_seal": "the draw's repeated names and its pairs at Hamming distance one are in the seal (SEALED-FAMILIES.json, scale_draws)",
                                 "repeated_names": plan["repeated_names"], "n_repeated_names": len(plan["repeated_names"])},
           "repeated_names": plan["repeated_detail"], "reads": base["units"], "n_reads": len(base["units"]), "n_units_what": "the reads (one recorded point each)",
           "identity": base["identity"], "sessions": base["sessions"], "progress_sha256": base["progress_sha256"],
           "device": record_tools.device_info("cpu"), "versions": record_tools.versions(), "environment": RS.numeric_environment(),
           "code_commit": record_tools.git_state()[0], "timestamp_utc": RS.utc_now()}
    sha = RS.write_json(out_path, out)
    print(f"retained: {out_path}  sha256 {sha}")
    return out


def read_once(r: dict, cue: np.ndarray, journal: Journal, jp: JournalPath, ex: ExhaustiveCosine, bm: BM25Index, levers: dict,
              entry_of_j: dict, pointer_of_j: dict, threshold: float, k: int, other_line_of: dict | None = None) -> dict:
    """One read of C.2 through I0 (the candidates, the top k by the index's cosine, a tie at the window's boundary reported), I1
    (the exact top k), I2 (BM25's top k), the lever variants; for a pair's original, C.2 (b)'s selection by R1 and R4 with the
    citation taken as the duplicate and as the own line; for a sample read of a cross schema repeated name, the key's
    contract (R4 on a window holding both lines of the name, the own line as the citation)."""
    q = np.asarray(cue, dtype=np.float32)
    own_eid = entry_of_j.get(r["j"]) if "j" in r else None
    own_ptr = pointer_of_j.get(r["j"]) if "j" in r else None
    cands = jp.index.candidates(q)                                        # not a read (not counted); the lookup below is the read
    before = jp.index.touched
    hits_all = jp.index.query(q, k=max(k, len(cands)))                    # one lookup, every candidate ranked: the window is its first k
    touched = jp.index.touched - before
    hits = hits_all[:k]
    i0 = [e for e, _ in hits]; i0_scores = [float(s) for _, s in hits]
    tie_at_boundary = bool(len(hits_all) > k and hits_all[k - 1][1] == hits_all[k][1])
    i1 = ex.rank(q, k=k)
    line = {"i0": {"candidates": len(cands), "touched": touched, "top": i0, "marks": i0_scores, "own_in_window": (own_eid in i0) if own_eid else None,
                   "own_in_candidates": (own_eid in cands) if own_eid else None, "max_mark": (max(i0_scores) if i0_scores else 0.0),
                   "max_mark_at_or_above_threshold": bool(i0_scores and max(i0_scores) >= threshold), "tie_at_boundary": tie_at_boundary},
            "i1": {"top": i1["docs"], "scores": i1["scores"], "compared": i1["compared"], "own_in_window": (own_eid in i1["docs"]) if own_eid else None},
            "agree": {"own_in_both": (bool(own_eid in i0 and own_eid in i1["docs"]) if own_eid else None), "same_set": set(i0) == set(i1["docs"]),
                      "same_first": bool(i0 and i1["docs"] and i0[0] == i1["docs"][0])},
            "levers": {}}
    for name, ix in levers.items():
        hb = ix.query(q, k=k)
        line["levers"][name] = {"candidates": len(ix.candidates(q)), "own_in_window": (own_eid in [e for e, _ in hb]) if own_eid else None}
    rb = bm.rank(r["text"], k=k)
    line["i2"] = {"top": rb["docs"], "scores": rb["scores"], "postings_read": rb["postings_read"], "own_in_window": (own_eid in rb["docs"]) if own_eid else None}
    marks = dict(zip(i0, i0_scores))

    def lines_of(ids):
        return [{"pointer": journal._entries[e].pointer, "mark": marks[e], "text": journal.payloads.get(journal._entries[e].pointer).decode("utf-8", errors="replace")} for e in ids]

    def cited_for(ptr, eid):
        return ptr if (ptr is not None and eid in marks and marks[eid] >= threshold) else None

    def selection(cited, lines):
        return {shape: {k_: v for k_, v in select_citation(shape, cited, lines, float(threshold), r["text"]).items()
                        if k_ in ("pointer", "changed", "abstain", "reason", "question_parsed", "candidates", "values", "unreadable", "tied")} for shape in ("R1", "R4")}
    if r["kind"].startswith("pair-"):
        dup_eid = entry_of_j.get(r["partner_j"]); dup_ptr = pointer_of_j.get(r["partner_j"])
        own_m, dup_m = marks.get(own_eid), marks.get(dup_eid)
        both = bool(own_m is not None and dup_m is not None and own_m >= threshold and dup_m >= threshold)
        lines = lines_of(i0)
        sel = {"duplicate": selection(cited_for(dup_ptr, dup_eid), lines), "own": selection(cited_for(own_ptr, own_eid), lines)}
        line["pair"] = {"kind": r["kind"][5:], "own_in_window": own_eid in i0, "duplicate_in_window": dup_eid in i0, "both_at_or_above_threshold": both,
                        "own_mark": own_m, "duplicate_mark": dup_m, "margin": ((own_m - dup_m) if (own_m is not None and dup_m is not None) else None),
                        "mark_ranks_duplicate_first": (bool(dup_m > own_m) if (own_m is not None and dup_m is not None) else None),
                        "raw_cosine": r["raw_cosine"], "selection": sel, "own_pointer": own_ptr, "duplicate_pointer": dup_ptr,
                        "lines_parse": [parse_statement(l["text"]) is not None for l in lines], "duplicate_admitted": dup_eid is not None}
    if r["kind"] == "sample" and r.get("cross_schema_repeated") and other_line_of:
        others = other_line_of.get(r["j"], [])
        both_in = bool(own_eid in i0 and any(o in i0 for o in others))
        line["key_contract"] = {"other_lines_of_the_name": len(others), "other_line_in_window": any(o in i0 for o in others), "both_lines_in_window": both_in,
                                "selection_own_as_citation": (selection(cited_for(own_ptr, own_eid), lines_of(i0)) if both_in else None), "label": KEY_CONTRACT_LABEL}
    return line


def time_per_query(q_cues: np.ndarray, sample_reads: list[dict], jp: JournalPath, ex: ExhaustiveCosine, bm: BM25Index, k: int) -> dict:
    """B.3's protocol at C's sizes on the first two hundred sample questions: the index step of each index timed by the cost
    harness's own discipline (the garbage collector disabled during the stage, `perf_counter_ns`, the timer overhead as the
    median of a thousand empty calls reported beside, one discarded warm up, six repetitions in the six orders interleaved
    question by question, the median, the 95th percentile, the spread of the six medians, the instability flags); then
    once more in blocked order (one index at a time); both reported. The thread state is read back and recorded, not
    enforced here (the encoder's step of the same run is not pinned)."""
    from cortex_c2b.cost_harness import Timed, timer_overhead_ns, summarise, check_pinned

    def run_index(ix, r):
        q = np.asarray(q_cues[r["q"]], dtype=np.float32)
        if ix == "I0":
            return jp.index.query(q, k=k)
        if ix == "I1":
            return ex.rank(q, k=k)
        return bm.rank(r["text"], k=k)
    inter = {ix: {"index": []} for ix in ("I0", "I1", "I2")}
    for rep in range(-1, 6):
        order = SIX_ORDERS[rep] if rep >= 0 else ("I0", "I1", "I2")
        per = {ix: [] for ix in inter}
        for r in sample_reads:
            for ix in order:
                with Timed() as t:
                    run_index(ix, r)
                per[ix].append(t.ns)
        if rep >= 0:
            for ix in inter:
                inter[ix]["index"].append(per[ix])
    blocked = {ix: {"index": []} for ix in inter}
    for ix in blocked:
        for rep in range(-1, 6):
            per = []
            for r in sample_reads:
                with Timed() as t:
                    run_index(ix, r)
                per.append(t.ns)
            if rep >= 0:
                blocked[ix]["index"].append(per)
    overhead = timer_overhead_ns()
    si = summarise(inter, overhead); sb = summarise(blocked, overhead)
    flagged = [ix for ix in si if si[ix]["index"].get("flag")]
    return {"n_queries": len(sample_reads), "repetitions": 6, "interleaved": {ix: v["index"] for ix, v in si.items()}, "blocked": {ix: v["index"] for ix, v in sb.items()},
            "timer_overhead_ns": overhead, "flagged_interleaved": flagged, "flagged_blocked": [ix for ix in sb if sb[ix]["index"].get("flag")],
            "instability_rule": "a repetition's median more than twice or less than half the median of the other five is flagged; this timing is run once (a second protocol under another name is the cost harness's rule, not applied at scale: the row says so)",
            "threads": check_pinned(), "threads_label": "read back, not enforced: the probe's steps share one run, and its encoder step is not pinned; the row names the state",
            "difference_label": "the difference between the interleaved and the blocked order is the cache effect of interleaving an exhaustive scan with bucket lookups",
            "orders": [list(o) for o in SIX_ORDERS], "label": "the index step's time per query at this size on one processor; the generation, the marks and the rules are not in it; the frozen reader's mark (`cue_similarity`, a scan of the entries per pointer) is not timed at scale"}


def index_sizes(journal: Journal, jp: JournalPath, bm: BM25Index, levers: dict) -> dict:
    """The index sizes: the structures' counts and the traced bytes of a fresh build of each under tracemalloc (a build of
    its own, never in a timed run)."""
    live = [eid for eid, e in journal._entries.items() if e.state != STATE_EVICTED]
    out = {"I0": {"vectors": len(jp.index.vecs), "tables": len(jp.index.tables), "buckets": sum(len(t) for t in jp.index.tables)},
           "I1": {"vectors": len(live)}, "I2": {"documents": bm.n_docs, "tokens": len(bm.postings), "postings": sum(len(p) for p in bm.postings.values())},
           **{name: {"vectors": len(ix.vecs), "tables": len(ix.tables), "buckets": sum(len(t) for t in ix.tables)} for name, ix in levers.items()}}
    tracemalloc.start()
    traced = {}
    b0 = tracemalloc.get_traced_memory()[0]
    ix0 = CueIndex(dim=journal.cue_dim, n_bits=ADOPTED_GEOMETRY["n_bits"], n_tables=ADOPTED_GEOMETRY["n_tables"], seed=0, signature="exact")
    for eid in live:
        ix0.add(eid, np.asarray(journal._entries[eid].cue, dtype=np.float32))
    traced["I0"] = tracemalloc.get_traced_memory()[0] - b0
    b1 = tracemalloc.get_traced_memory()[0]
    norms = {eid: float(np.linalg.norm(ix0.vecs[eid])) for eid in live}
    traced["I1"] = tracemalloc.get_traced_memory()[0] - b1
    b2 = tracemalloc.get_traced_memory()[0]
    bm2 = BM25Index(bm.k1, bm.b)
    for eid in live:
        bm2.insert(eid, journal.payloads.get(journal._entries[eid].pointer))
    traced["I2"] = tracemalloc.get_traced_memory()[0] - b2
    tracemalloc.stop()
    del ix0, norms, bm2
    out["traced_bytes"] = {**traced, "accounting": "tracemalloc over a fresh build of each structure from the live entries (numpy buffers and Python objects); the cues' vectors counted in I0, I1 adds its norms"}
    return out


def _dist(xs: list) -> dict | None:
    xs = [float(x) for x in xs if x is not None]
    return {"median": float(np.median(xs)), "p95": float(np.percentile(xs, 95)), "max": float(max(xs)), "n": len(xs)} if xs else None


def margins_by_cosine(pairs: list[dict]) -> dict:
    """The margins (the own mark minus the duplicate's) binned by the pair's raw cosine as row 50 bins them (`COSINE_BINS`)."""
    out = {}
    for lo, hi in COSINE_BINS:
        xs = [u["pair"]["margin"] for u in pairs if u["pair"]["margin"] is not None and lo <= float(u["pair"]["raw_cosine"]) < hi]
        out[f"[{lo}, {hi})"] = ({"n": len(xs), "median": float(np.median(xs)), "min": float(min(xs)), "max": float(max(xs)), "negative": sum(1 for x in xs if x < 0)} if xs else {"n": 0})
    return out


def pair_counts(us: list[dict]) -> dict:
    """The counts of C.2 (b) on a set of pair reads: the contested reads with their exact interval (Clopper and Pearson), the
    share where the mark ranks the duplicate first (R1's error) with its interval, the margins binned by the raw cosine, the
    reads where R4's outcome does not follow from the texts, the selection counts with the citation taken as the duplicate
    and as the own line."""
    admitted = [u for u in us if u["pair"]["duplicate_admitted"]]
    contested = [u for u in admitted if u["pair"]["both_at_or_above_threshold"]]
    dup_first = [u for u in contested if u["pair"]["mark_ranks_duplicate_first"]]
    follows = [u for u in contested if all(u["pair"]["lines_parse"]) and not u["pair"]["selection"]["duplicate"]["R4"].get("tied")]
    return {"n": len(us), "duplicate_admitted": len(admitted), "own_in_window": sum(1 for u in admitted if u["pair"]["own_in_window"]),
            "duplicate_in_window": sum(1 for u in admitted if u["pair"]["duplicate_in_window"]), "contested": len(contested),
            "contested_clopper_pearson_95": clopper_pearson(len(contested), len(admitted)) if admitted else None,
            "contested_wilson_95": wilson(len(contested), len(admitted)) if admitted else None,
            "mark_ranks_duplicate_first": len(dup_first),
            "r1_error_clopper_pearson_95": clopper_pearson(len(dup_first), len(contested)) if contested else None,
            "r1_error_wilson_95": wilson(len(dup_first), len(contested)) if contested else None,
            "intervals": "the exact interval (Clopper and Pearson) is the record's; Wilson's beside it for comparability with row 50",
            "margins_by_raw_cosine": margins_by_cosine(contested),
            "r4_outcome_by_construction": len(follows), "r4_not_following": len(contested) - len(follows),
            "selection_counts": {taken: {shape: {"abstain": sum(1 for u in contested if u["pair"]["selection"][taken][shape]["abstain"]),
                                                 "cites_own": sum(1 for u in contested if u["pair"]["selection"][taken][shape]["pointer"] == u["pair"]["own_pointer"]),
                                                 "cites_duplicate": sum(1 for u in contested if u["pair"]["selection"][taken][shape]["pointer"] == u["pair"]["duplicate_pointer"])}
                                         for shape in ("R1", "R4")} for taken in ("duplicate", "own")}}


def aggregate_c2(units: list[dict], plan: dict) -> dict:
    sample = [u for u in units if u["read"]["kind"] == "sample"]
    never = [u for u in units if u["read"]["kind"] == "never"]
    pairs = {kind: [u for u in units if u["read"]["kind"] == f"pair-{kind}"] for kind in ("E1", "E2")}
    out = {"sample": {"n": len(sample), "i0": {"candidates": _dist([u["i0"]["candidates"] for u in sample]), "own_in_window": sum(1 for u in sample if u["i0"]["own_in_window"]),
                                                  "own_in_candidates": sum(1 for u in sample if u["i0"]["own_in_candidates"]),
                                                  "ties_at_boundary": sum(1 for u in sample if u["i0"].get("tie_at_boundary"))},
                      "i1": {"own_in_window": sum(1 for u in sample if u["i1"]["own_in_window"]), "compared": _dist([u["i1"]["compared"] for u in sample])},
                      "i2": {"own_in_window": sum(1 for u in sample if u["i2"]["own_in_window"]), "postings_read": _dist([u["i2"]["postings_read"] for u in sample])},
                      "agree": {"own_in_both": sum(1 for u in sample if u["agree"]["own_in_both"]), "same_set": sum(1 for u in sample if u["agree"]["same_set"]),
                                "same_first": sum(1 for u in sample if u["agree"]["same_first"])},
                      "levers": {name: {"candidates": _dist([u["levers"][name]["candidates"] for u in sample]), "own_in_window": sum(1 for u in sample if u["levers"][name]["own_in_window"])}
                                 for name, _, _ in LEVER_VARIANTS},
                      "repeated_names_in_sample": sum(1 for u in sample if u["read"].get("repeated_name"))},
           "never_planted": {"n": len(never), "candidates": _dist([u["i0"]["candidates"] for u in never]),
                             "max_mark_at_or_above_threshold": sum(1 for u in never if u["i0"]["max_mark_at_or_above_threshold"]), "by_distance": {},
                             "ties_at_boundary": sum(1 for u in never if u["i0"].get("tie_at_boundary"))}}
    nn = out["never_planted"]
    nn["clopper_pearson_95"] = clopper_pearson(nn["max_mark_at_or_above_threshold"], len(never))
    for d in sorted({u["read"]["distance"] for u in never}):
        sub = [u for u in never if u["read"]["distance"] == d]
        nn["by_distance"][str(d)] = {"n": len(sub), "at_or_above": sum(1 for u in sub if u["i0"]["max_mark_at_or_above_threshold"])}
    nn["label"] = NEVER_PLANTED_LABEL
    for kind, us in pairs.items():
        plain = [u for u in us if not u["read"].get("repeated_name")]
        rep = [u for u in us if u["read"].get("repeated_name")]
        out[f"pairs_{kind}"] = {**pair_counts(plain), "repeated_name_pairs_apart": (pair_counts(rep) if rep else None),
                                "repeated_name_pairs": [u["read"]["partner_j"] for u in rep],
                                "what": "the pairs whose original is not a repeated name; the partnered repeated names counted apart (C.1 (b))"}
    kc = [u for u in sample if u.get("key_contract")]
    out["key_contract"] = {"label": KEY_CONTRACT_LABEL, "n_sample_reads_of_cross_schema_repeated_names": len(kc),
                           "both_lines_in_window": sum(1 for u in kc if u["key_contract"]["both_lines_in_window"]),
                           "r4_abstains_with_both_lines": sum(1 for u in kc if u["key_contract"]["both_lines_in_window"] and u["key_contract"]["selection_own_as_citation"]["R4"]["abstain"]),
                           "r4_keeps_own_with_both_lines": sum(1 for u in kc if u["key_contract"]["both_lines_in_window"] and u["key_contract"]["selection_own_as_citation"]["R4"]["pointer"] is not None),
                           "claimed_for": "nothing"}
    return out


# ---------------------------------------------------------------------------- #
# step 3: the cascade (C.3)                                                       #
# ---------------------------------------------------------------------------- #
def withdrawal_check(journal: Journal, memory: PersistentMemory) -> dict:
    """C.3 (a): the identities behind the memory's patterns against the entries in state consolidated or demoted; a withdrawn
    identity present, or a kept one absent, is a fault."""
    raw = memory.ids()
    unlabeled = sum(1 for x in raw if x is None)
    ids = {x for x in raw if x is not None}
    want = {eid for eid, e in journal._entries.items() if e.state in (STATE_CONSOLIDATED, STATE_DEMOTED)}
    present_withdrawn = sorted(ids - want); absent_kept = sorted(want - ids)
    return {"patterns": len(memory), "kept_states": len(want), "unlabeled_patterns": unlabeled, "fault": bool(present_withdrawn or absent_kept or unlabeled),
            "withdrawn_present": present_withdrawn[:20], "kept_absent": absent_kept[:20], "n_withdrawn_present": len(present_withdrawn), "n_kept_absent": len(absent_kept)}


def reconstruction_reading(journal: Journal, memory: PersistentMemory, cfg: LifecycleConfig) -> dict:
    """Row 42's reading (b1 to b3) at a read phase, cumulative over every address withdrawn so far, by the frozen
    `level_3_memory`: the defect against the bound 0.05, the nearest kept pattern against the radius 0.90; reported, never a
    fault; the count of reconstructions recorded."""
    r = level_3_memory(journal, memory, cfg.replay_error_max, None, radius=cfg.demote_min_cosine)
    return {"reconstructions": r.get("replayed"), "patterns": r.get("patterns"), "evicted_address_reading": r.get("evicted_address_reading"),
            "bound": cfg.replay_error_max, "radius": cfg.demote_min_cosine}


def window_of(jp: JournalPath, q: np.ndarray, k: int = 4) -> tuple[list[tuple], int]:
    """One lookup, as `JournalPath.retrieve` makes it (one `query`, every hit resolved through the aliases, the evicted
    dropped), returning the window and the holes (the hits dropped); one touch of the index per read, as the organ's."""
    window, holes = [], 0
    for eid, s in jp.index.query(q, k=k):
        e, pointer = jp._resolve(eid)
        if e.state == STATE_EVICTED:
            holes += 1
        else:
            window.append((e, pointer, float(s)))
    return window, holes


def cascade_reads(journal: Journal, jp: JournalPath, q_cues: np.ndarray, L: dict, j_written: int, threshold: float, entry_at_j: dict,
                  progress: RS.Progress | None = None, phase=None, k: int = 4) -> list[dict]:
    """The reads of a read phase, after the phase event and before the next write (and once more after the last plan write),
    in a fixed order: the sample's positions whose entries are written and not yet evicted (the seal's order), then every
    planted pair's original written so far (the plan's order); one lookup each (`window_of`) at the system's k; the window's
    pointers and holes recorded with the own line; each read's line appended to the progress file as it is made (4 (c): a
    session dead inside the reads re reads them whole at its resumption, the lines compared at assembly and kept once)."""
    out = []

    def emit(rec):
        out.append(rec)
        if progress is not None:
            progress.append({"kind": "read", "phase": phase, **rec})

    def one(j: int, what: str, partner_j: int | None = None, kind: str | None = None):
        q = np.asarray(q_cues[L["q_of_j"][j]], dtype=np.float32)
        cands = jp.index.candidates(q)                     # not a read: not counted in the latency telemetry (read_path.py)
        window, holes = window_of(jp, q, k=k)
        pointers = [p for _, p, _ in window]; marks = [s for _, _, s in window]
        own = entry_at_j.get(j)
        own_e = journal._entries.get(own) if own else None
        rec = {"what": what, "j": j, "own_in_window": (own_e.pointer in pointers) if own_e else None, "holes": holes, "candidates": len(cands),
               "pointers": pointers, "marks": marks, "max_mark_at_or_above_threshold": bool(marks and max(marks) >= threshold), "own_state": (own_e.state if own_e else "refused")}
        if partner_j is not None:
            dup = entry_at_j.get(partner_j); dup_e = journal._entries.get(dup) if dup else None
            rec.update({"kind": kind, "partner_j": partner_j, "duplicate_in_window": (dup_e.pointer in pointers) if dup_e else None,
                        "duplicate_state": (dup_e.state if dup_e else "refused")})
        return rec
    for j in L["sample_js"]:
        if j < j_written:
            own = entry_at_j.get(j)
            if own and journal._entries[own].state != STATE_EVICTED:
                emit(one(j, "sample"))
    for oj, pj, kind in L["pairs"]:
        if oj < j_written:
            emit(one(oj, "pair", pj, kind))
    return out


def read_lines_of(progress: RS.Progress) -> tuple[list[dict], list[dict]]:
    """The `read` lines of a progress file kept once by (phase, what, j, partner): a line recorded twice (a session dead
    inside the reads, the reads redone whole) is compared, a difference being a fault of the resumption."""
    seen: dict[tuple, dict] = {}; faults = []
    for l in progress.lines():
        if l.get("kind") != "read":
            continue
        k = (l.get("phase"), l.get("what"), l.get("j"), l.get("partner_j"))
        if k in seen:
            a = {x: y for x, y in seen[k].items() if x != "wall_clock"}; b = {x: y for x, y in l.items() if x != "wall_clock"}
            if a != b:
                faults.append(list(k))
            continue
        seen[k] = l
    return list(seen.values()), faults


def run_cascade(args, plan: dict, run_id: str, rdir: Path, gate: str, st_cues: np.ndarray, q_cues: np.ndarray, threshold: float, cfg_hash: str, k: int = 4) -> dict:
    """The second planting through the frozen scheduler (C.3): the unit the write, the recorded point the phase, the checks
    after every phase, the reads at the read phases and after the last plan write, the sessions under the budget, the stop
    file; the reopening by the cut rule, the payload archive and D6 B (section 4). Two times per phase: the lifecycle's (the
    period's writes with the phase, the frozen code alone) and the probe's own (the check, the reads, the synchronisation)."""
    jdir = rdir / f"c3-{gate}"; jpath = jdir / "journal.jsonl"
    out_path = Path(args.out) / f"scale-cascade-{plan['seed']}-{run_id}-{gate}.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    cfg = cascade_config()
    read_phases = set(READ_PHASES_100K if plan["size"] == "hundred_thousand" else READ_PHASES)
    identity = identity_for(args, run_id, plan, "cascade", gate, cfg_hash, {"lifecycle_config_hash": cfg.config_hash()})
    stop = RS.StopFile(args.stop_file); budget = RS.Budget(args.budget_seconds)
    planned = plan["planned"]
    L = indexes_of_plan(plan)
    t_open = time.monotonic()
    cut_record = restored = None
    if RS.Progress(jdir / "cascade.progress.jsonl").exists():
        archive = RS.PayloadArchive(RS.PayloadArchive.path_for(jpath))
        progress = RS.Progress(jdir / "cascade.progress.jsonl", sync_paths=[jpath], archive=archive)
        identity["compared"]["journal_key_sha256"] = key_sha_of(jdir)
        check_identity(progress, identity, jpath)
        journal, key, wp, jp, start_j, tracker, cut_record, restored = reopen(jdir, progress, archive, gate)
        sch = LifecycleScheduler(journal, wp, jp, cfg, now=float(start_j))      # Decision 8: continued, not restarted; the sentinels found
        units0 = progress.units()
        last_event = units0[max(units0)]["phase_event"] if units0 else None
        if last_event is not None and last_event.get("config_hash") != cfg.config_hash():
            raise RS.ResumptionRefused(f"the last phase event's configuration hash {last_event.get('config_hash')} differs from this run's {cfg.config_hash()}; refused")
        if sch.phases != (max(units0) if units0 else 0):
            raise RS.ResumptionRefused(f"the reopened journal counts {sch.phases} phases, the progress file {max(units0) if units0 else 0}; refused")
        resumed_from = start_j
    else:
        journal, key = new_journal(jdir, plaintext=bool(getattr(args, "plaintext", False)))
        archive = RS.PayloadArchive(RS.PayloadArchive.path_for(jpath))
        progress = RS.Progress(jdir / "cascade.progress.jsonl", sync_paths=[jpath], archive=archive)
        wp = WritePath(journal, seed=0, admission=gate)
        jp = JournalPath(journal, geometry=ADOPTED_GEOMETRY)
        sch = LifecycleScheduler(journal, wp, jp, cfg, now=0.0)              # the twenty sentinels planted at now 0.0
        archive_pointers(journal, archive, [e.pointer for e in journal._entries.values()])
        tracker = RS.LogTracker(jpath)
        identity["compared"]["journal_key_sha256"] = key_sha_of(jdir)
        identity["compared"]["log_after_open"] = {**tracker.update(), "archive_length": archive.length}
        identity["compared"]["index_state_after_open"] = RS.index_state(jp)
        progress.write_identity(identity)
        start_j = 0; resumed_from = None
    # the plan index of every admitted entry, from the recorded decisions (the writes after the last point are redone)
    entry_at_j: dict[int, str] = {}
    for u in progress.units().values():
        for d in u["decisions"]:
            if d["admitted"]:
                entry_at_j[int(d["j"])] = d["entry_id"]
    near_ids = {eid: (j_, planned[j_]["kind"]) for j_, eid in entry_at_j.items() if planned[j_]["kind"] in ("E1", "E2")}
    evicted_recorded = {eid for u in progress.units().values() for eid in (u.get("evicted_near_duplicates") or [])}
    progress.start_session({"wall_clock": RS.utc_now(), "command": RS.command_line("cortex_c2b.scale_probe", getattr(args, "argv", None)), "resumed_from": resumed_from,
                            "reopening_seconds": round(time.monotonic() - t_open, 3), "restored": (cut_record or {}).get("restored"), "moved": (cut_record or {}).get("moved"),
                            "budget_seconds": args.budget_seconds, "phases_at_start": sch.phases, "torn_progress_tail": progress.torn_tail})
    if cut_record:
        progress.append(cut_record)
    decisions = []
    reason = "complete"
    j = start_j
    n_seen = len(journal._entries)
    first_phase_of_session = None
    t_lifecycle = time.monotonic()                   # the frozen code's time: from the end of the probe's work to the end of the phase's write
    try:
        while j < len(planned):
            budget.start()
            x = planned[j]
            phases_before = sch.phases
            rep = sch.write(st_cues[j], x["statement"].encode("utf-8"), x["schema"], now=float(j), source=x["statement"])
            if rep.admitted:
                archive_pointers(journal, archive, [rep.entry.pointer, rep.entry.source_pointer])
                entry_at_j[j] = rep.entry.entry_id
                if x["kind"] in ("E1", "E2"):
                    near_ids[rep.entry.entry_id] = (j, x["kind"])
            elif rep.revised and rep.entry is not None:           # a revision holds a new pointer until its eviction (unreachable on the seal's plans: every text is distinct)
                archive_pointers(journal, archive, [rep.entry.pointer])
            decisions.append({"j": j, "kind": x["kind"], "admitted": bool(rep.admitted), "surprise": (float(rep.surprise) if rep.surprise == rep.surprise else None),
                              "entry_id": (rep.entry.entry_id if rep.admitted else None), "pointer": (rep.entry.pointer if rep.admitted else None), "revised": bool(rep.revised)})
            j += 1
            if sch.phases > phases_before:
                t_end_write = time.monotonic()
                lifecycle_seconds = t_end_write - t_lifecycle
                phase = sch.phases
                first_phase_of_session = first_phase_of_session or phase
                ents = list(journal._entries.values())
                new_entries = ents[n_seen:]; n_seen = len(ents)
                archive_pointers(journal, archive, [e.pointer for e in new_entries if e.schema_id == SCHEMA_SUMMARY])
                check = withdrawal_check(journal, sch.memory)
                newly = sorted(eid for eid in near_ids if journal._entries[eid].state == STATE_EVICTED and eid not in evicted_recorded)
                evicted_recorded.update(newly)
                line = {"kind": "unit", "index": phase, "phase": phase, "next_j": j, "decisions": decisions, "phase_event": sch.last_phase, "check": check,
                        "lifecycle_seconds": round(lifecycle_seconds, 4), "wall_clock": RS.utc_now(), "entries": len(journal),
                        "states": {s_: sum(1 for e in journal._entries.values() if e.state == s_) for s_ in (STATE_LIVE, STATE_CONSOLIDATED, STATE_DEMOTED, STATE_EVICTED)},
                        "summaries": len(journal.summary_sources), "memory_patterns": len(sch.memory),
                        "evicted_near_duplicates": newly, "evicted_near_duplicates_detail": [{"entry_id": e, "j": near_ids[e][0], "kind": near_ids[e][1]} for e in newly]}
                decisions = []
                if phase in read_phases:
                    reads = cascade_reads(journal, jp, q_cues, L, j, threshold, entry_at_j, progress, phase, k)
                    line["reads"] = reads_summary(reads)
                    line["reconstruction"] = reconstruction_reading(journal, sch.memory, cfg)
                line.update({"log": tracker.update(), "archive_length": archive.length, "index_state": RS.index_state(jp)})
                line["probe_seconds"] = round(time.monotonic() - t_end_write, 4)
                progress.append(line)
                t_lifecycle = time.monotonic()
                print(f"phase {phase}  plan index {j}  elapsed {budget.elapsed():.1f} s", flush=True)
                if stop.requested() or budget.exceeded():
                    reason = "stop_file" if stop.requested() else "budget"; break
        if reason == "complete":
            # the reads after the last plan write (C.3 (a): "and the last"), the tail's writes included, then the tail line (a session
            # dead here redoes the tail's writes and the reads; the lines are kept once)
            reads = cascade_reads(journal, jp, q_cues, L, j, threshold, entry_at_j, progress, "last", k)
            progress.append({"kind": "tail", "next_j": j, "decisions": decisions, "log": tracker.update(), "archive_length": archive.length, "wall_clock": RS.utc_now(),
                             "reads": reads_summary(reads), "reconstruction": reconstruction_reading(journal, sch.memory, cfg), "phases": sch.phases,
                             "note": "the writes after the last phase (fewer than a period, no phase fired), then the reads after the last plan write"})
    except Exception:
        f = RS.write_traceback(progress.path, f"plan index {j}, phase {sch.phases}")
        progress.append({"kind": "death", "wall_clock": RS.utc_now(), "unit": f"plan index {j}, phase {sch.phases}", "traceback_file": str(f)})
        progress.close(); archive.close()
        print(f"scale probe: the session died at plan index {j} (phase {sch.phases}); the traceback is in {f}", flush=True)
        raise
    progress.append({"kind": "session_end", "wall_clock": RS.utc_now(), "reason": reason, "first_unit": first_phase_of_session, "last_unit": sch.phases,
                     "resumption_point": (None if reason == "complete" else sch.phases)})
    progress.close(); archive.close()
    if reason != "complete":
        print(f"scale probe: cascade stopped ({reason}) at phase {sch.phases}; the same command resumes")
        return {"stopped": reason, "phase": sch.phases}
    base = RS.assemble(progress)
    units = base["units"]
    faults = [u["phase"] for u in units if u["check"]["fault"]]
    lines = progress.lines()
    reads_lines, read_faults = read_lines_of(progress)
    tail = next((l for l in reversed(lines) if l.get("kind") == "tail"), {})
    out = {"benchmark": "ADR-010 C.3: the cascade on a scale journal under the frozen lifecycle", "draw": plan["seed"], "size": plan["size"], "run_id": run_id, "gate": gate,
           "lifecycle_config": {"hash": cfg.config_hash(), "fields": {k_: getattr(cfg, k_) for k_ in cfg.__dataclass_fields__},
                                "compared_with": "the last phase event as the progress file copied it from the frozen scheduler (the log's prefix being hash checked at every resumption)"},
           "k": k, "k_label": "the system's k (journal_k of the checkpoint's configuration)",
           "phases": sch.phases, "expected_phases_identity": plan["expected_phases_identity"], "entries": len(journal), "sentinels": len(sch.sentinels),
           "states": units[-1]["states"] if units else None,
           "faults": {"phases": faults, "n": len(faults), "condition": ("the withdrawal under cascade holds on every phase" if not faults else "FAULT: attributed before anything else (C.4)")},
           "phase_times_seconds": [u["lifecycle_seconds"] for u in units], "phase_times_label": TIME_LABEL,
           "phase_times_what": "the frozen code's time per phase: the period's writes through the scheduler and the phase they fire, from the end of the probe's work after the previous phase to the end of the write that fired this one",
           "probe_times_seconds": [u["probe_seconds"] for u in units],
           "probe_times_what": "the probe's own time after each phase: the withdrawal check, the state counts, the reads and the reconstruction reading at a read phase, the recorded point's synchronisation; not in the phase times",
           "read_phases": sorted(read_phases) + ["last"],
           "reads_by_phase": {**{str(u["phase"]): u.get("reads") for u in units if "reads" in u}, "last": tail.get("reads")},
           "reconstruction_by_phase": {**{str(u["phase"]): u.get("reconstruction") for u in units if "reconstruction" in u}, "last": tail.get("reconstruction")},
           "n_read_lines": len(reads_lines), "n_units": len(units), "n_units_what": "the recorded points (the phases), not the writes",
           "phases_refused": sch.refused, "replay_failed_total": sch.replay_failed,
           "applied_totals": {k_: sum((u["phase_event"].get("applied") or {}).get(k_, 0) for u in units) for k_ in ("consolidated", "replay_failed", "demoted_groups", "evicted", "withdrawn")},
           "summaries": {"written": len(journal.summary_sources),
                         "evicted": sum(1 for sid in journal.summary_sources if journal._entries[sid].state == STATE_EVICTED),
                         "sources_of_evicted_summaries_evicted": sum(1 for sid, srcs in journal.summary_sources.items() if journal._entries[sid].state == STATE_EVICTED
                                                                     for src in srcs if journal._entries[src].state == STATE_EVICTED),
                         "sources_of_evicted_summaries": sum(len(srcs) for sid, srcs in journal.summary_sources.items() if journal._entries[sid].state == STATE_EVICTED)},
           "near_duplicates": near_duplicate_fate(units, reads_lines, planned, journal, near_ids, L, entry_at_j),
           "admitted": sum(d["admitted"] for u in units for d in u["decisions"]) + sum(d["admitted"] for d in tail.get("decisions", [])),
           "identity": base["identity"], "sessions": base["sessions"], "run_in_sessions": base["run_in_sessions"],
           "resumption_fault": base["resumption_fault"] + read_faults, "read_lines_recorded_twice_compared": len(lines) - len([l for l in lines if l.get("kind") != "read"]) - len(reads_lines),
           "progress_sha256": base["progress_sha256"], "progress_file": base["progress_file"], "restored_index_at_last_reopening": restored,
           "identity_labels": {"freeze_sha256_null": "the freeze file did not exist when the run started", "PYTHONHASHSEED_null": "the launcher was not used (refused since the build of 2026-10-30)"},
           "device": record_tools.device_info("cpu"), "versions": record_tools.versions(), "environment": RS.numeric_environment(), "code_commit": record_tools.git_state()[0],
           "timestamp_utc": RS.utc_now()}
    sha = RS.write_json(out_path, out)
    print(f"retained: {out_path}  sha256 {sha}")
    return out


def reads_summary(reads: list[dict]) -> dict:
    return {"n": len(reads), "sample_read": sum(1 for r_ in reads if r_["what"] == "sample"), "pairs_read": sum(1 for r_ in reads if r_["what"] == "pair"),
            "own_in_window": sum(1 for r_ in reads if r_.get("own_in_window")), "holes_total": sum(r_["holes"] for r_ in reads),
            "duplicates_in_window": sum(1 for r_ in reads if r_.get("duplicate_in_window"))}


def near_duplicate_fate(units: list[dict], reads: list[dict], planned: list[dict], journal: Journal, near_ids: dict, L: dict, entry_at_j: dict) -> dict:
    """Per near duplicate and per gate (ADR-007 section 4 (c)'s input): the phase of its eviction, whether it was read before
    it (its line returned in the window of any read at a read phase before the phase of its eviction, C.3 (a)), whether
    demotion grouped it with its partner."""
    evicted_at = {}
    for u in units:
        for d in u.get("evicted_near_duplicates_detail") or []:
            evicted_at[d["entry_id"]] = u["phase"]
    pointer_of = {eid: journal._entries[eid].pointer for eid in near_ids if eid in journal._entries}
    seen_before: dict[str, int] = {}                     # entry id -> the first read phase where its pointer was in any read's window
    for r in reads:
        ph = r.get("phase")
        if not isinstance(ph, int):
            continue                                     # the reads after the last plan write ("last") come after every eviction
        for eid, ptr in pointer_of.items():
            if ptr in (r.get("pointers") or []) and eid not in seen_before:
                seen_before[eid] = ph
    out = {"admitted": len(near_ids), "by_kind": {}, "read_before_eviction_what": "the duplicate's line in the window of any read (a sample read or a pair read) at a read phase before the phase of its eviction"}
    rows = []
    for eid, (j, kind) in sorted(near_ids.items(), key=lambda kv: kv[1][0]):
        e = journal._entries.get(eid)
        orig_j = L["j_of_position"][planned[j]["position"]]
        orig_eid = entry_at_j.get(orig_j)
        grouped = None
        if e is not None and eid in journal.summary_of:
            members = journal.summary_sources.get(journal.summary_of[eid], [])
            grouped = orig_eid is not None and orig_eid in members
        ep = evicted_at.get(eid)
        rows.append({"j": j, "kind": kind, "original_j": orig_j, "state": e.state if e else None, "evicted_at_phase": ep,
                     "read_before_eviction": (seen_before.get(eid) is not None and (ep is None or seen_before[eid] < ep)), "grouped_with_partner": grouped})
    for kind in ("E1", "E2"):
        sub = [r for r in rows if r["kind"] == kind]
        out["by_kind"][kind] = {"admitted": len(sub), "evicted": sum(1 for r in sub if r["state"] == STATE_EVICTED), "live": sum(1 for r in sub if r["state"] == STATE_LIVE),
                                "consolidated": sum(1 for r in sub if r["state"] == STATE_CONSOLIDATED), "demoted": sum(1 for r in sub if r["state"] == STATE_DEMOTED),
                                "read_before_eviction": sum(1 for r in sub if r["read_before_eviction"]), "grouped_with_partner": sum(1 for r in sub if r["grouped_with_partner"]),
                                "eviction_phase_counts": dict(Counter(str(r["evicted_at_phase"]) for r in sub if r["evicted_at_phase"] is not None))}
    out["rows"] = rows
    return out


# ---------------------------------------------------------------------------- #
# C.4's reading                                                                    #
# ---------------------------------------------------------------------------- #
def checks_from_progress(progress_path: str | Path) -> dict:
    """A cascade with a progress file and no final file (unfinished, or refused at a resumption and retained under its name):
    its recorded checks, read for C.4's fault condition like any session's (C.3 (b))."""
    pr = RS.Progress(progress_path)
    units = pr.units()
    ident = (pr.identity() or {}).get("compared", {})
    faults = [i for i, u in sorted(units.items()) if (u.get("check") or {}).get("fault")]
    return {"progress_file": str(progress_path), "draw": ident.get("draw"), "gate": ident.get("gate"), "run_id": ident.get("run_id"), "phases_recorded": len(units),
            "faults": {"phases": faults, "n": len(faults)}, "sessions": pr.sessions(), "final_file": None, "status": "unfinished or refused: read for the fault condition only"}


def read_c4(cascade_files: list[dict], c2_files: list[dict], progress_files: list[str | Path] = ()) -> dict:
    """C.4: two conditions. The withdrawal under cascade holds on every journal and phase (the final files and the progress
    files of unfinished or refused runs alike), or fails and is attributed before anything else; the count of never planted
    entities above the threshold at each size is the number at which the veto's guarantee is measured to end, with its
    interval and attribution, under its label."""
    faults = [{"draw": f["draw"], "gate": f["gate"], "run_id": f.get("run_id"), "phases": f["faults"]["phases"]} for f in cascade_files if f["faults"]["n"]]
    partial = [checks_from_progress(p) for p in progress_files]
    faults += [{"draw": c["draw"], "gate": c["gate"], "run_id": c["run_id"], "phases": c["faults"]["phases"], "from": c["progress_file"]} for c in partial if c["faults"]["n"]]
    never: dict = {}
    for f in c2_files:
        nn = f["aggregates"]["never_planted"]
        never.setdefault(f["size"], []).append({"draw": f["draw"], "gate": f["gate"], "run_id": f["run_id"], "n": nn["n"], "at_or_above": nn["max_mark_at_or_above_threshold"],
                                                "clopper_pearson_95": nn["clopper_pearson_95"], "by_distance": nn["by_distance"]})
    return {"withdrawal_under_cascade": {"holds_on_every_journal_and_phase": not faults, "faults": faults, "read_first": True,
                                         "runs_read": len(cascade_files) + len(partial), "unfinished_or_refused_runs": [c["progress_file"] for c in partial]},
            "never_planted_above_threshold": never, "label": NEVER_PLANTED_LABEL, "cost_curves": "reported with their construction; no threshold to pass",
            "decision_13_successor": DECISION_13_SUCCESSOR}


# ---------------------------------------------------------------------------- #
# command line                                                                     #
# ---------------------------------------------------------------------------- #
def _cli(argv=None):
    import train
    ap = argparse.ArgumentParser(description="the scale cost probe with the cascade (ADR-010 part C; a new module on the frozen system)")
    ap.add_argument("--draw", type=int, required=True, help="the seal's scale draw (50000 to 50004; 60000)")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--journals", default="runs/scale", help="where the probe's journals, keys, archives and cue files live (never committed)")
    ap.add_argument("--out", default="metrics/mqar")
    ap.add_argument("--step", default="all", choices=["encode", "c2", "cascade", "all"])
    ap.add_argument("--gate", default=None, choices=list(GATES), help="the gate as a parameter; both when omitted (identity first)")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--budget-seconds", type=float, default=None, help=f"4 (e): the session's wall clock budget from its first write; the hundred thousand cascade takes {DEFAULT_BUDGET_100K} s (D3 A) when none is given")
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--test-plan-n", type=int, default=None, help="the tests' fixtures only: a plan of N facts under --draw, of size 'test', never a draw of the seal; refused under metrics/")
    ap.add_argument("--plaintext", action="store_true", help="the tests' fixtures only: unsealed journals (4 (f)); refused under metrics/")
    args = ap.parse_args(argv)
    args.argv = list(argv) if argv is not None else sys.argv[1:]          # the arguments as given, for the recorded command
    RS.require_launcher()
    under_metrics = "metrics" in Path(args.out).parts
    if (args.test_plan_n is not None or args.plaintext) and under_metrics:
        raise SystemExit("scale probe: a test plan or a plaintext journal never writes under metrics/")
    if under_metrics and not (record_tools.REPO_ROOT / FREEZE_PATH_REL).exists():
        raise SystemExit(f"scale probe: {FREEZE_PATH_REL} must exist before a run of the record (python -m cortex_c2b.freeze --write --commit {FREEZE_COMMIT_FULL}); refused")
    cfg = train.load_config(args.config)
    plan = plan_of(args.draw, test_n=args.test_plan_n)
    if args.budget_seconds is None and plan["size"] == "hundred_thousand":
        args.budget_seconds = float(DEFAULT_BUDGET_100K)                   # D3 A: fourteen hours from the session's first write, the probe's own default
    k = int(cfg.journal_k)
    model, ck, device = load_model(args.ckpt, cfg)
    run_id = ck.get("run_id")
    if not run_id:
        raise SystemExit("scale probe: the checkpoint carries no run id")
    rdir = run_dir(Path(args.journals), args.draw, run_id)
    gates = [args.gate] if args.gate else list(GATES)
    h = train.config_hash(cfg)
    if args.step in ("encode", "all"):
        r = run_encode(args, model, cfg, device, plan, run_id, rdir)
        if r.get("stopped"):
            return
    if args.step == "encode":
        return
    st, q = load_cues(rdir, plan, args.batch_size)
    thr = threshold_for(model, cfg, device)
    thr47 = None
    r47 = Path(args.out) / f"hm-lm-{run_id}-session-b-index16exact.json"
    if r47.exists():
        thr47 = float(json.loads(r47.read_text(encoding="utf-8"))["hm_mark_oracle"]["threshold"])
        if abs(thr - thr47) > 1e-6:
            raise SystemExit(f"scale probe: the oracle's threshold {thr} differs from row 47's {thr47} by more than 1e-6; stopped")
    elif under_metrics:
        raise SystemExit(f"scale probe: row 47's plain file {r47} is needed to check the oracle's threshold (within 1e-6); refused")
    del model
    for gate in gates:
        if args.step in ("c2", "all"):
            p = run_c2_planting(args, plan, run_id, rdir, gate, st, h)
            if p.get("stopped"):
                return
            r = c2_numbers(args, plan, run_id, rdir, gate, st, q, thr, thr47, p, h, k)
            if r.get("stopped"):
                return
        if args.step in ("cascade", "all"):
            r = run_cascade(args, plan, run_id, rdir, gate, st, q, thr, h, k)
            if r.get("stopped"):
                return


if __name__ == "__main__":
    _cli()
