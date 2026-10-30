"""cortex_c2b.sealed_family -- the runner of ADR-010's sealed families (part A, A.5 and A.6; part B, B.1 and B.2), built as a
new module on the frozen system `c5b4818` after the amendment of 2026-10-15 on parts B and C (graved 2026-10-29). Rows 51
and 52 are read cold from its files by `cortex_c2b.sealed_reading`; nothing here is a number of the record.

What the runner does, in the amendment's words:

  the precondition (A.5)   before any sealed family is read, per checkpoint: the frozen session B replayed on the first
                           family's journal under `--index 16,exact` as row 47 ran it (one bridge, one stream from
                           `default_rng(1)` through the blocks in row 47's order), compared question by question with row
                           47's file; the oracle's threshold within 1e-6 of row 47's; BM25's ten selection counts
                           recomputed on the journal's own documents against the build's; a separate file,
                           `sealed-precondition-<run id>.json`, which every family run requires to say `ok`;
  the planting             the family's 200 statements written in generation order through the frozen `JournalBridge`
                           under `--admission identity` with the shadow legacy gate beside the rule (every write's rule,
                           surprise, decision, nearest stored code and the shadow's verdict recorded; a refusal never
                           stops the run and its questions stay in the denominators), into a fresh journal under a key
                           generated for the run and kept beside it (`<journal>.key`, the founder's to keep until rows 51
                           to 53 are graved, never committed); the oracle's threshold read once as rows 47 and 49 read it
                           (the training family, the legacy geometry for the scratch journal, the identity rule);
  the passes (B.1 (a))     one pass per form (the direct query; the paraphrase) and per index (I0 the frozen bridge; I1
                           and I2 the reference bridge of `cortex_c2b.references`), each in a new process from the disk
                           alone, each question through the frozen `_probe` under `plain` with `record_index`, the
                           window's permutation drawn for that question alone, `default_rng([shuffle_seed, i])` (D5 A;
                           1 for the direct form, 3 for the paraphrase), the blocks in the declared order (the 200
                           planted; the never planted of the form, the 50 then on A the 150; in the direct pass the fifty
                           cross schema questions then the near name block of 1 (c)); a short window reported, never a
                           stop; the rules applied to the one generation per question and index from the recorded output
                           by the frozen `policy_outcome` and `select_citation` (`plain`; `mark-veto`; `mark-veto+value`
                           with R4; V1; V1 with the organ's value; the mark oracle with the model's value and with the
                           organ's), twenty one rows per checkpoint and family;
  the resumption (4 (c))   the question is the unit; a progress file beside the pass's output carries one line per
                           question with the answer record and the reference's own fields; a resumption rebuilds
                           `pointer_of` from the planting record named in the identity, skips the questions whose lines
                           exist and continues at the first missing index; the planting, the oracle's threshold and the
                           precondition are atomic (discarded and run whole again when interrupted).

The founder's commands are written in `docs/working/REPRISE-2026-09-16.md` (the block of 2026-10-30) with this build; every run
goes through the launcher `tools/adr010_launcher.py` (PYTHONHASHSEED set before the interpreter starts; a run started
otherwise is refused), and a run of the record (an output under `metrics/`) needs the freeze file and row 47's plain file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from cortex_c2b import Journal, POLICY_STOP, ADMISSION_IDENTITY, content_hash, record_tools
from cortex_c2b.crypto import generate_key
from cortex_c2b.hm_protocol import Fact, value_of_statement, generate_facts, address_cue, N_FACTS, N_NEGCTRL, SCHEMAS
from cortex_c2b.lm_bridge import JournalBridge, LABELS
from cortex_c2b.read_path import ADOPTED_GEOMETRY, CueIndex, JournalPath
from cortex_c2b import sealed_families as S
from cortex_c2b.near_duplicate_probe import write_recorded, near_name_plan, load_model
from cortex_c2b.near_duplicate_reading import select_citation
from cortex_c2b.hm_lm import _probe, mark_oracle, policy_outcome, session_b_from_disk
from cortex_c2b.references import ReferenceBridge, bm25_selection, v1_outcome, INDEXES, TEXTBOOK_PAIR, entity_of_question, ExhaustiveCosine
from cortex_c2b import resumable as RS

FAMILIES_READ = ("T0", "A", "Av")                 # the three families read once before the tag (the development family is never read)
FORMS = ("direct", "paraphrase")
SHUFFLE_SEED = {"direct": 1, "paraphrase": 3}     # A.5: the record's index seed 0 plus one for the direct form, plus three for the paraphrase
RULES = ("plain", "mark-veto", "mark-veto+value", "V1", "V1+organ-value", "oracle", "oracle+organ-value")
BLOCKS_DIRECT = ("planted", "never_planted", "cross_schema", "near_name")
BLOCKS_PARAPHRASE = ("planted", "never_planted")
THRESHOLD_TOL = 1e-6
SEAL_PATH = Path("metrics/mqar/SEALED-FAMILIES.json")
FREEZE_PATH = Path("metrics/mqar/FREEZE-v1.1.0.json")
NO_HEAD_CLAIM = ("one generation per question and index: a claim conditional on the checkpoint carrying no matching head "
                 "(ADR-010 A.6); the policies are applied to the recorded output by the frozen functions")


# ---------------------------------------------------------------------------- #
# the family's questions, in the declared block order                           #
# ---------------------------------------------------------------------------- #
def family_data(name: str) -> dict:
    if name not in FAMILIES_READ:
        raise ValueError(f"the families read are {FAMILIES_READ}; the development family is never read as a result")
    fams = S.generate_all()
    return fams[name]


def seal_check(fam: dict, seal_path: Path = SEAL_PATH) -> dict:
    """The family regenerated against the seal's content hash (A.3): a difference refuses the run."""
    seal = json.loads(Path(seal_path).read_text(encoding="utf-8"))
    want = seal["families"][fam["name"]]["content_sha256"]
    have = S.content_hash_of(S.family_content(fam))
    if want != have:
        raise ValueError(f"the family {fam['name']} regenerates to {have}, the seal says {want}; refused (A.3)")
    return {"seal_path": str(seal_path), "seal_sha256": RS.sha256_file(seal_path), "family_content_sha256": have,
            "constants_sha256": seal["constants_sha256"]}


def facts_of(fam: dict) -> list[Fact]:
    """The family's planted facts as `Fact` objects (the address text and the hash cue as the protocol builds them)."""
    out = []
    for f in fam["facts"]:
        out.append(Fact(entity=f["entity"], schema=f["schema"], attr=f["attribute"], statement=f["statement"], query=f["query"],
                        paraphrase=f["paraphrase"], cue=address_cue(f"{f['schema']}|{f['entity']}"), address_text=f"{f['schema']}|{f['entity']}"))
    return out


def near_name_protected(fams: dict | None = None) -> set[str]:
    """The protected set of 1 (c): the seal's decontamination union together with the planted and never planted names of T0,
    A, Av and the development family."""
    fams = fams or S.generate_all()
    prot = set(S.protected_names())
    for fam in fams.values():
        prot |= {f["entity"] for f in fam["facts"]} | {v["entity"] for v in fam["never_planted"]}
    return prot


def near_name_block(fam: dict, fams: dict | None = None) -> dict:
    """The near name never planted block of a family (1 (c)): the frozen `near_name_plan` on the family's planted facts as
    `Fact` objects with the protected set of 1 (c); the question text the function's own `query` on T0 and Av (the first
    family's direct query), and on A the j-th kept name of each schema in A's Q1 for j even and Q2 for j odd."""
    plan = near_name_plan(facts_of(fam), near_name_protected(fams))
    questions = []
    per_schema = {s: 0 for s in SCHEMAS}
    for q in plan["questions"]:
        j = per_schema[q["schema"]]; per_schema[q["schema"]] += 1
        if fam["name"] == "A":
            tid = "Q1" if j % 2 == 0 else "Q2"
            text = S.A_TEMPLATES[q["schema"]][tid].format(e=q["entity"])
        else:
            tid = "Q"
            text = q["query"]
        questions.append({**q, "query_asked": text, "question_template": tid, "j": j})
    return {"questions": questions, "dropped": plan["dropped"], "rule": plan["rule"],
            "question_rule": ("A's direct query template Q1 for the j-th kept name of a schema with j even, Q2 for j odd" if fam["name"] == "A"
                              else "the first family's direct query of the schema (the function's own query field)")}


def questions_of(fam: dict, form: str, fams: dict | None = None) -> list[dict]:
    """The questions of a pass in the declared order (B.1 (a)), each with its block, its text, its entity, its expected
    value (None when nothing is planted) and its own pointer (None for the blocks without a planted line)."""
    if form not in FORMS:
        raise ValueError(f"form is one of {FORMS}")
    key = "query" if form == "direct" else "paraphrase"
    out = []
    for f in fam["facts"]:
        out.append({"block": "planted", "entity": f["entity"], "schema": f["schema"], "expected": f["attribute"], "text": f[key],
                    "own_pointer": content_hash(f["statement"].encode("utf-8")), "statement": f["statement"],
                    "template": f["query_template" if form == "direct" else "paraphrase_template"]})
    for v in fam["never_planted"]:
        out.append({"block": "never_planted", "entity": v["entity"], "schema": v["schema"], "expected": None, "text": v[key],
                    "own_pointer": None, "seed": v["seed"], "template": v["query_template" if form == "direct" else "paraphrase_template"]})
    if form == "direct":
        for c in fam["cross_schema"]:
            planted = next(f for f in fam["facts"] if f["entity"] == c["entity"])
            out.append({"block": "cross_schema", "entity": c["entity"], "schema": c["schema_asked"], "schema_planted": c["schema_planted"],
                        "expected": None, "text": c["question"], "own_pointer": content_hash(planted["statement"].encode("utf-8")),
                        "planted_attribute": planted["attribute"], "template": c["question_template"]})
        nn = near_name_block(fam, fams)
        for q in nn["questions"]:
            out.append({"block": "near_name", "entity": q["entity"], "schema": q["schema"], "expected": None, "text": q["query_asked"],
                        "own_pointer": None, "planted_neighbour": q["planted_neighbour"], "attr_of_neighbour": q["attr_of_neighbour"],
                        "template": q["question_template"]})
    for i, q in enumerate(out):
        q["index"] = i
    return out


FIRST_FAMILY = "first"
FIRST_FAMILY_LABEL = "in sample for the organ (geometry, row 45; policy, seed 1337); BM25's parameters as B.1 (b) labels them (B.2 (b))"
FIRST_NEVER_SEED = 10_000                           # the protocol's never planted facts: generate_facts(50, seed + 10 000) at seed 0 (hm_lm)


def questions_of_first(form: str) -> list[dict]:
    """The first family's questions for the paraphrase pass A.6 asks for (run once on the three checkpoints, on the record's
    journals as they are): the 200 planted facts of seed 0 and the protocol's 50 never planted of seed 10 000, the form's
    text; no cross schema and no near name block (the first family enters the readings as the in sample reference)."""
    if form not in FORMS:
        raise ValueError(f"form is one of {FORMS}")
    key = "query" if form == "direct" else "paraphrase"
    facts, _ = generate_facts(N_FACTS, 0)
    neg, _ = generate_facts(N_NEGCTRL, FIRST_NEVER_SEED)
    out = []
    for f in facts:
        out.append({"block": "planted", "entity": f.entity, "schema": f.schema, "expected": f.attr, "text": getattr(f, key),
                    "own_pointer": content_hash(f.statement.encode("utf-8")), "statement": f.statement, "template": None})
    for v in neg:
        out.append({"block": "never_planted", "entity": v.entity, "schema": v.schema, "expected": None, "text": getattr(v, key), "own_pointer": None,
                    "seed": FIRST_NEVER_SEED, "template": None})
    for i, q in enumerate(out):
        q["index"] = i
    return out


def pass_file_name(family: str, run_id: str, index: str, form: str) -> str:
    return f"sealed-{family}-{run_id}-{index}-{form}.pass.json"


def family_file_name(family: str, run_id: str) -> str:
    return f"sealed-{family}-{run_id}.json"


def precondition_file_name(run_id: str) -> str:
    return f"sealed-precondition-{run_id}.json"


def row47_plain_file(metrics_dir: Path, run_id: str) -> Path:
    return Path(metrics_dir) / f"hm-lm-{run_id}-session-b-index16exact.json"


# ---------------------------------------------------------------------------- #
# the permutation of D5 A                                                        #
# ---------------------------------------------------------------------------- #
def question_rng(shuffle_seed: int, index: int) -> np.random.Generator:
    """The generator made for one question alone: `numpy.random.default_rng([shuffle_seed, i])` (D5 A)."""
    return np.random.default_rng([int(shuffle_seed), int(index)])


def permutation_of_record(rec: dict) -> list[int] | None:
    """The permutation the window shows, recovered from a record: slot s (label A, B, ...) holds the item of retrieval index
    order[s]. None when the window is empty."""
    ro = rec.get("retrieval_order") or []
    win = rec.get("window") or []
    if not ro or not win:
        return None
    by_label = {w["label"]: w["pointer"] for w in win}
    out = []
    for s in range(len(win)):
        label = chr(LABELS[s])
        p = by_label.get(label)
        if p is None:
            return None
        out.append(ro.index(p))
    return out


# ---------------------------------------------------------------------------- #
# the rules on one generation (A.6, B.1): plain, the two veto policies, V1, the oracle #
# ---------------------------------------------------------------------------- #
def _label_byte(label: str | None) -> int | None:
    return ord(label) if isinstance(label, str) and len(label) == 1 else label


def apply_rules(rec: dict, texts: dict[str, bytes], threshold: float, expected: str | None, shape: str) -> dict:
    """The seven rules on the recorded output of one question (`_probe` under `plain` with `record_index`), by the frozen
    `policy_outcome` and `select_citation`, exactly as the live probe composes them under each policy. `texts`: pointer
    -> payload bytes (the planting record's statements). Returns {rule: {kind, cited_pointer, answered, outcome,
    cited_pointer_is_own, claim, ...}}."""
    labels = {_label_byte(w["label"]): w["pointer"] for w in rec.get("window", [])}
    marks = {w["pointer"]: float(w["mark"]) for w in rec.get("window", []) if w.get("mark") is not None}
    own = rec.get("own_pointer")
    raw_kind, raw_label, raw_ans = rec["raw_kind"], _label_byte(rec.get("raw_label")), rec.get("raw_answered")
    exp = expected if expected is not None else "(never planted)"

    payloads = dict(texts)

    def pack(kind, label, answered, outcome, **more):
        pointer = labels.get(label) if (kind == "cite" and label is not None) else None
        return {"kind": kind, "cited_pointer": pointer, "answered": answered, "outcome": outcome,
                "cited_pointer_is_own": ((pointer == own) if (pointer is not None and own) else None),
                "claim": kind != "unknown", **more}

    out = {}
    # plain: the probe's own outcome
    out["plain"] = pack(rec["kind"], _label_byte(rec.get("label")), rec.get("answered"), rec["outcome"])
    # mark-veto
    k, a, o = policy_outcome(raw_kind, raw_label, labels, marks, payloads, own, raw_ans, exp, "mark-veto", threshold)
    out["mark-veto"] = pack(k, raw_label if k == "cite" else None, a, o)
    # mark-veto+value with the shape (R4), the live composition of `_probe`
    k1, _, _ = policy_outcome(raw_kind, raw_label, labels, marks, payloads, own, raw_ans, exp, "mark-veto", threshold)
    veto_cited = labels.get(raw_label) if k1 == "cite" else None
    lines = [{"pointer": p, "mark": marks.get(p), "text": (texts.get(p) or b"").decode("utf-8", errors="replace")} for p in rec.get("retrieval_order", [])]
    shape_rec = select_citation(shape, veto_cited, lines, float(threshold), rec["query"])
    kind2, label2, ans2 = raw_kind, raw_label, raw_ans
    if veto_cited is not None:
        if shape_rec["pointer"] is None:
            kind2, label2, ans2 = "unknown", None, None
        elif shape_rec["pointer"] != veto_cited:
            label2 = next(b for b, p in labels.items() if p == shape_rec["pointer"])
    k, a, o = policy_outcome(kind2, label2, labels, marks, payloads, own, ans2, exp, "mark-veto+value", threshold)
    out["mark-veto+value"] = pack(k, label2 if k == "cite" else None, a, o, veto_cited_pointer=veto_cited, shape=shape_rec)
    # V1 on the decoder's own citation, with the model's value and with the organ's
    raw_pointer = labels.get(raw_label) if (raw_kind == "cite" and raw_label in labels) else None
    cited_payload = texts.get(raw_pointer) if raw_pointer is not None else None
    for name, organ in (("V1", False), ("V1+organ-value", True)):
        v = v1_outcome(rec["query"], cited_payload, raw_ans, exp, organ_value=organ)
        out[name] = pack(v["kind"], raw_label if v["kind"] == "cite" else None, v["answered"], v["outcome"], entity=v["entity"], reason=v["reason"])
    # the mark oracle (A.6: "the mark oracle (no model)"), as the frozen `mark_oracle` applies its rule: cite the highest marked line
    # when its mark is strictly above the threshold (the first maximum in retrieval order), else abstain; strict when that line is
    # the own line, the citation alone counted, no value (the frozen rule reads none); a claim on another line is a wrong claim
    # (`invalid`). With the organ's value: the value read from the cited line by the organ's reader, exact or not, or an abstention
    top = None
    if marks:
        best = max(marks.values())
        if best > threshold:
            top = [p for p in rec.get("retrieval_order", []) if marks.get(p) == best][0]
    for name, organ in (("oracle", False), ("oracle+organ-value", True)):
        if top is None:
            out[name] = pack("unknown", None, None, "abstain", reason="no line strictly above the threshold")
            continue
        payload = texts.get(top) or b""
        label_top = next(b for b, p in labels.items() if p == top)
        if not organ:
            out[name] = pack("cite", label_top, None, ("strict" if top == own else "invalid"), oracle_pointer=top,
                             value="none: the frozen mark_oracle's rule counts the citation alone (strict when the cited line is the own line)")
            continue
        v = value_of_statement(payload.decode("utf-8", errors="replace"))
        ans = v[2] if v else None
        if ans is None:
            out[name] = pack("unknown", None, None, "abstain", reason="the organ reads no value on the highest marked line", oracle_pointer=top)
            continue
        out[name] = pack("cite", label_top, ans, ("strict" if (ans == exp and top == own) else ("valid" if ans.encode("utf-8") in payload else "invalid")), oracle_pointer=top)
    return out


def partition_of(name: str, rule_rec: dict, rec: dict) -> str:
    """The full partition of a question under a rule (A.6): `strict_own` (strict citing the own line), `strict_other`,
    `valid_wrong_value`, `invalid`, `model_abstention` (the decoder abstained; never for the oracle's rules, which read no
    model output), `malformed` (a claim without the contract's shape), `veto_abstention` (a citation the veto refused or a
    label outside the window), `shape_abstention` (R4 abstained: no candidate or a conflict), `value_reader_abstention_own`
    and `_other` (the organ read no value on the final cited line, split by whether that line is the own line),
    `rule_abstention` (V1's refusal, the oracle's no line strictly above the threshold)."""
    outcome = rule_rec["outcome"]
    if outcome == "strict":
        return "strict_own" if rule_rec.get("cited_pointer_is_own") else "strict_other"
    if outcome == "valid":
        return "valid_wrong_value"
    if outcome == "invalid":
        return "invalid"
    own = rec.get("own_pointer")
    reason = str(rule_rec.get("reason") or "")
    if name in ("oracle", "oracle+organ-value"):                           # the oracle's rules read no model output: never the model's abstention
        if "reads no value" in reason:
            p = rule_rec.get("oracle_pointer")
            return "value_reader_abstention_own" if (p is not None and p == own) else "value_reader_abstention_other"
        return "rule_abstention"                                          # no line strictly above the threshold
    if rec["raw_kind"] == "unknown":
        return "model_abstention"
    if "veto_cited_pointer" in rule_rec:                                  # mark-veto+value
        vc = rule_rec["veto_cited_pointer"]
        if vc is None:
            return "veto_abstention"
        shape = rule_rec.get("shape") or {}
        if shape.get("pointer") is None:
            return "shape_abstention"
        return "value_reader_abstention_own" if shape.get("pointer") == own else "value_reader_abstention_other"
    if "reads no value" in reason:                                        # V1 with the organ's value: the decoder's own citation
        labels = {w["label"]: w["pointer"] for w in rec.get("window", [])}
        p = labels.get(rec.get("raw_label"))
        return "value_reader_abstention_own" if (p is not None and p == own) else "value_reader_abstention_other"
    if rec["raw_kind"] == "malformed":
        return "malformed"                                                # a claim without the contract's shape: counted against the model, under every rule
    return "veto_abstention" if name == "mark-veto" else "rule_abstention"


# ---------------------------------------------------------------------------- #
# the planting (atomic)                                                          #
# ---------------------------------------------------------------------------- #
def plant_family(model, cfg, fam: dict, journal_path: Path, key: bytes, device, admission: str = ADMISSION_IDENTITY) -> dict:
    """The family's 200 statements through the frozen bridge under the rule, with the shadow legacy gate; the planting
    record (every write's gate quantities, the shadow's verdict, the generator's truth), as rows 49 planted its arms."""
    import torch
    journal = Journal(journal_path, key=key, policy=POLICY_STOP, index_geometry=ADOPTED_GEOMETRY)
    mark = bool(getattr(cfg, "journal_familiarity_mark", False))
    bridge = JournalBridge(model, journal, k=cfg.journal_k, budget_bytes=cfg.journal_read_bytes, seed=0, shuffle_seed=1, device=device,
                           mark=mark, index_geometry=ADOPTED_GEOMETRY, admission=admission, shadow_legacy=True)
    entity_of_pointer: dict = {}
    writes = []
    with torch.no_grad():
        for i, f in enumerate(fam["facts"]):
            _, rec, _ = write_recorded(bridge, f["statement"], f["schema"], float(i), f["entity"], None, entity_of_pointer, "original",
                                       truth=(f["entity"], f["attribute"]))
            rec["statement_template"] = f["statement_template"]
            writes.append(rec)
    refused = [w for w in writes if not w["admitted"]]
    return {"family": fam["name"], "facts_seed": fam["seeds"]["facts"], "n_facts": len(fam["facts"]),
            "index_geometry": bridge.jp.index.geometry(), "journal_record": journal.index_record(),
            "admission": {"rule": journal.admission_record(), "source": journal.admission_record_source(), "asked": admission,
                          "threshold": bridge.wp.thr, "near_recorded": len(journal.near_of),
                          "shadow_legacy": {"judged": sum(1 for w in writes if w.get("legacy_decision") is not None),
                                            "legacy_refused": sum(1 for w in writes if w.get("legacy_decision") is False)}},
            "counts": {"written": len(writes), "admitted": sum(1 for w in writes if w["admitted"]), "refused": len(refused),
                       "refused_attributed": [{"entity": w["entity"], "nearest": w.get("nearest_before"), "surprise": w["surprise"]} for w in refused],
                       "entries": len(journal)},
            "pointer_of": {w["entity"]: w["pointer"] for w in writes if w["admitted"]},
            "writes": writes, "device": record_tools.device_info(device), "versions": record_tools.versions(), "timestamp_utc": RS.utc_now()}


def threshold_of(model, cfg, device, admission: str = ADMISSION_IDENTITY) -> dict:
    """The oracle's threshold read as rows 47 and 49 read it (the training family, the legacy geometry for the scratch
    journal, the rule given), with no protocol read: `mark_oracle` on empty lists through a bridge over a memory journal."""
    mark = bool(getattr(cfg, "journal_familiarity_mark", False))
    scratch_bridge = JournalBridge(model, Journal(), k=cfg.journal_k, budget_bytes=cfg.journal_read_bytes, seed=0, shuffle_seed=1,
                                   device=device, mark=mark, index_geometry=ADOPTED_GEOMETRY, admission=admission)
    o = mark_oracle(model, scratch_bridge, [], [], {}, admission=admission)
    return {"threshold": float(o["threshold"]), "training_accuracy": o["training_accuracy"], "training_pairs": o["training_pairs"],
            "threshold_index_geometry": o["threshold_index_geometry"], "scratch_admission": o["scratch_admission"],
            "scratch_admitted": o["scratch_admitted"], "scratch_facts": o["scratch_facts"]}


# ---------------------------------------------------------------------------- #
# the identity of a pass (4 (b))                                                 #
# ---------------------------------------------------------------------------- #
def frozen_tree_sha256(root: Path) -> str | None:
    """The code commit recomputed from the frozen files' hashes: the sha256 over the frozen set's (path, sha256_lf) pairs as
    present in the checkout, from `FREEZE-v1.1.0.json`'s list when it exists, else None."""
    from cortex_c2b.freeze import tree_hash_of_list
    fz = Path(root) / FREEZE_PATH
    if not fz.exists():
        return None
    return tree_hash_of_list(json.loads(fz.read_text(encoding="utf-8")), Path(root))


def pass_identity(args, run_id: str, ckpt_sha: str, planting_sha: str, threshold: float, key_sha: str, seal_info: dict,
                  bm25: dict, journal_hash: dict, oracle_record: dict | None = None) -> dict:
    """The compared part (4 (b)) and the recorded part; the oracle's whole record (its threshold is in the compared part)
    travels in the recorded part so that a resumption writes the same `oracle` block as a run in one session."""
    root = record_tools.REPO_ROOT
    compared = {"kind": "sealed family pass", "family": args.family, "index": args.index, "form": args.form, "run_id": run_id,
                "shuffle_seed": SHUFFLE_SEED[args.form], "admission": args.admission, "decode_shape": args.decode_shape,
                "oracle_threshold": threshold, "checkpoint_sha256": ckpt_sha, "planting_record_sha256": planting_sha,
                "journal_key_sha256": key_sha, "journal_log_sha256": journal_hash,
                "seal_sha256": seal_info.get("seal_sha256"), "family_content_sha256": seal_info.get("family_content_sha256"),
                "freeze_sha256": (RS.sha256_file(root / FREEZE_PATH) if (root / FREEZE_PATH).exists() else None),
                "frozen_tree_sha256": frozen_tree_sha256(root), "bm25": bm25["pair"], "k": args.k, "budget": args.budget,
                "environment": RS.numeric_environment()}
    recorded = {"command": RS.command_line("cortex_c2b.sealed_family", getattr(args, "argv", None)), "budget_seconds": args.budget_seconds, "stop_file": args.stop_file, "out": str(args.out),
                "oracle": oracle_record}
    return RS.make_identity(compared, recorded)


# ---------------------------------------------------------------------------- #
# one pass (a new process from the disk alone)                                   #
# ---------------------------------------------------------------------------- #
def run_pass(args) -> dict:
    """One form under one index on a planted family: resumable question by question (4 (c))."""
    import torch
    import train
    from cortex_c2b.crypto import key_from_env
    out_dir = Path(args.out)
    jpath = Path(args.journal)
    first = args.family == FIRST_FAMILY
    cfg = train.load_config(args.config)
    model, ck, device = load_model(args.ckpt, cfg)
    if getattr(model, "match_head", None) is not None:
        raise SystemExit("sealed family pass: the checkpoint carries a matching head; the one generation per question claim does not hold (A.6)")
    run_id = ck.get("run_id")
    if first:
        # A.6: the first family's paraphrase pass on the record's journal as it is, under the record's key (the environment), I0
        if args.form != "paraphrase" or args.index != "I0":
            raise SystemExit("sealed family pass: the first family is read on I0 under the paraphrase form only (A.6); its direct form is row 47's plain file")
        key = key_from_env()
        if key is None or not jpath.is_file():
            raise SystemExit("sealed family pass: the first family's journal is sealed under the record's key; set QUANTUM_CORTEX_JOURNAL_KEY and name the journal")
        row47_path = row47_plain_file(out_dir, run_id)
        if not row47_path.exists():
            raise SystemExit(f"sealed family pass: row 47's plain file {row47_path} is the first family's planting record and is not there")
        sidecar = row47_path; key_file = None
        planting = {"family": FIRST_FAMILY, "counts": {"entries": N_FACTS}, "admission": json.loads(row47_path.read_text(encoding="utf-8")).get("journal", {}).get("admission", "surprise (the record's)")}
        facts0, _ = generate_facts(N_FACTS, 0)
        pointer_of_planted = {f.entity: content_hash(f.statement.encode("utf-8")) for f in facts0}
        texts = {content_hash(f.statement.encode("utf-8")): f.statement.encode("utf-8") for f in facts0}
        questions = questions_of_first(args.form)
        seal_info = {"seal_path": None, "family_content_sha256": None, "note": "the first family is the protocol's generator at seed 0, not a sealed family"}
    else:
        key_file = jpath.with_name(jpath.stem + ".key")
        sidecar = jpath.with_name(jpath.stem + ".planting.json")
        if not (jpath.is_file() and key_file.is_file() and sidecar.is_file()):
            raise SystemExit(f"sealed family pass: {jpath}, its key file and its planting record must exist (the planting comes first)")
        key = bytes.fromhex(key_file.read_text(encoding="utf-8").strip())
        planting = json.loads(sidecar.read_text(encoding="utf-8"))
        fams = S.generate_all()
        fam = fams[args.family]
        seal_info = seal_check(fam)
        if planting.get("family") != args.family or planting.get("facts_seed") != fam["seeds"]["facts"]:
            raise SystemExit("sealed family pass: the planting record is not this family's")
        questions = questions_of(fam, args.form, fams)
        texts = {content_hash(f["statement"].encode("utf-8")): f["statement"].encode("utf-8") for f in fam["facts"]}
        pointer_of_planted = {e: p for e, p in planting["pointer_of"].items()}
    out_name = pass_file_name(args.family, run_id, args.index, args.form)
    final_path = out_dir / out_name
    progress = RS.Progress(out_dir / (out_name.replace(".json", ".progress.jsonl")))
    stop = RS.StopFile(args.stop_file)
    budget = RS.Budget(args.budget_seconds)
    if final_path.exists():
        raise SystemExit(f"sealed family pass: {final_path} exists; a file of the record is never overwritten (A.5)")
    bm25 = bm25_selection()
    journal_hash = record_tools.tree_hash(jpath, record_tools.payload_dir_of(jpath))
    row47_path = row47_plain_file(out_dir, run_id)
    thr47 = float(json.loads(row47_path.read_text(encoding="utf-8"))["hm_mark_oracle"]["threshold"]) if row47_path.exists() else None
    if progress.exists():
        # 4 (b): the oracle's threshold is read from the identity line at a resumption, not recomputed; the whole record of
        # the first session's reading (training accuracy, pairs, geometry) travels in the identity's recorded part, so that
        # a run in several sessions writes the same `oracle` block as a run in one
        ident0 = progress.identity() or {}
        thr_rec = dict((ident0.get("recorded") or {}).get("oracle") or {})
        thr = float((ident0.get("compared") or {}).get("oracle_threshold"))
        if thr_rec.get("threshold") != thr:
            raise SystemExit("sealed family pass: the progress file's identity line carries no oracle record, or one that differs from its compared threshold; the resumption is refused (4 (b))")
    else:
        thr_rec = threshold_of(model, cfg, device, admission=args.admission)
        thr = thr_rec["threshold"]
        thr_rec["row47"] = thr47
        if thr47 is not None:
            thr_rec["within_tolerance"] = abs(thr - thr47) <= THRESHOLD_TOL
            if not thr_rec["within_tolerance"]:
                raise SystemExit(f"sealed family pass: the oracle's threshold {thr} differs from row 47's {thr47} by more than {THRESHOLD_TOL}; stopped before any question, no rate (A.5)")
        elif "metrics" in out_dir.parts:
            raise SystemExit(f"sealed family pass: row 47's plain file {row47_path} is needed to check the oracle's threshold (A.5); refused")
        thr_rec["read_rule"] = "computed once, in the run's first session, and written to the identity line of the progress file; a resumption reads it there and does not recompute (4 (b))"
    identity = pass_identity(args, run_id, RS.sha256_file(args.ckpt), RS.sha256_file(sidecar, lf=True), thr,
                             (RS.sha256_file(key_file, lf=True) if key_file is not None else "the record's key (the environment), never written"),
                             seal_info, bm25, journal_hash, thr_rec)
    resumed_from = None
    if progress.exists():
        diffs = RS.compare_identity(progress.identity(), identity["compared"])
        if diffs:
            raise SystemExit("sealed family pass: the run identity differs, the resumption is refused (4 (b)):\n  " + "\n  ".join(diffs))
        done = progress.units()
        resumed_from = min((i for i in range(len(questions)) if i not in done), default=len(questions))
    else:
        progress.write_identity(identity)
    t0 = time.monotonic()
    progress.start_session({"wall_clock": RS.utc_now(), "resumed_from": resumed_from, "command": RS.command_line("cortex_c2b.sealed_family", getattr(args, "argv", None)),
                            "budget_seconds": args.budget_seconds, "reopening_seconds": None, "torn_progress_tail": progress.torn_tail})
    done = progress.units()
    mark = bool(getattr(cfg, "journal_familiarity_mark", False))
    first_unit = last_unit = None
    reason = "complete"
    with record_tools.opened_copy(jpath, key, prefix="sealed-pass-") as (journal, _copy):
        if first:
            admission_of_pass = journal.admission_record() or journal.admission_rule or "surprise"   # the record's journals: the inherited rule
        else:
            admission_of_pass = args.admission
            if journal.admission_record() != args.admission:
                raise SystemExit(f"sealed family pass: the journal records {journal.admission_record()!r}, this pass asks {args.admission!r}")
        if args.index == "I0":
            bridge = JournalBridge(model, journal, k=args.k, budget_bytes=args.budget, seed=0, shuffle_seed=SHUFFLE_SEED[args.form], device=device,
                                   mark=mark, index_geometry=ADOPTED_GEOMETRY, admission=admission_of_pass)
        else:
            bridge = ReferenceBridge(model, journal, args.index, k=args.k, budget_bytes=args.budget, seed=0, shuffle_seed=SHUFFLE_SEED[args.form],
                                     device=device, mark=mark, index_geometry=ADOPTED_GEOMETRY, admission=admission_of_pass,
                                     bm25_parameters=(bm25["pair"]["k1"], bm25["pair"]["b"]))
        organ_path = bridge.organ_path if args.index != "I0" else bridge.jp
        live_geometry = organ_path.index.geometry()
        whole = ExhaustiveCosine(journal, organ_path); whole.hold(True)      # the own line under the whole store (A.5), one scan per question
        i = None
        try:
            with torch.no_grad():
                for q in questions:
                    i = q["index"]
                    if i in done:
                        continue
                    if stop.requested() or budget.exceeded():
                        reason = "stop_file" if stop.requested() else "budget"
                        break
                    budget.start()
                    rec = probe_one(model, bridge, q, args, pointer_of_planted, texts, thr, whole, journal)
                    line = {"kind": "unit", "index": i, "block": q["block"], "wall_clock": RS.utc_now(), "record": rec}
                    progress.append(line)
                    first_unit = i if first_unit is None else first_unit
                    last_unit = i
        except Exception:
            f = RS.write_traceback(progress.path, f"question {i}")
            progress.append({"kind": "death", "wall_clock": RS.utc_now(), "unit": f"question {i}", "traceback_file": str(f)}); progress.close()
            raise
    progress.append({"kind": "session_end", "wall_clock": RS.utc_now(), "reason": reason, "first_unit": first_unit, "last_unit": last_unit,
                     "resumption_point": (None if reason == "complete" else min((i for i in range(len(questions)) if i not in progress.units()), default=None)),
                     "elapsed_seconds": round(time.monotonic() - t0, 3)})
    progress.close()
    if reason != "complete":
        nxt = min((i for i in range(len(questions)) if i not in progress.units()), default=None)
        print(f"sealed family pass: stopped ({reason}); progress {progress.path}; resumption point {nxt}; the same command resumes")
        return {"stopped": reason, "progress": str(progress.path), "resumption_point": nxt}
    base = RS.assemble(progress)
    units = base["units"]
    if base["resumption_fault"]:
        raise SystemExit(f"sealed family pass: a recomputed question differs from its earlier line at {base['resumption_fault']}; a fault of the resumption")
    out = {"benchmark": "sealed family pass (ADR-010 A.5, A.6, B.1; the amendment of 2026-10-15 on parts B and C)",
           "family": args.family, "index": args.index, "form": args.form, "shuffle_seed": SHUFFLE_SEED[args.form],
           "window_rule": "the permutation of question i's window is drawn from default_rng([shuffle_seed, i]) (D5 A)",
           "policy_of_generation": "plain", "rules": list(RULES), "decode_shape": args.decode_shape,
           "admission": admission_of_pass, "admission_asked": args.admission,
           "admission_label": ("the journal's inherited rule, the record's (the first family's journal as it is); the recipe's flag was asked and does not apply" if first
                               else "the journal's recorded rule, equal to the recipe's flag"),
           "one_generation_claim": NO_HEAD_CLAIM, "match_head": False,
           "texts": {ptr: t.decode("utf-8") for ptr, t in texts.items()}, "own_pointer_of": dict(pointer_of_planted),
           "field_labels": {"own_rank": ("the own line's rank among the bucket candidates the live index proposed (the frozen _probe with record_index)" if args.index == "I0"
                                         else "the own line's rank under the whole store by the organ's cosine (the reference bridge's candidates are every live entry; equal to whole_store.own_rank by construction)"),
                            "whole_store": "the own line's rank, cosine and best other cosine under every live entry by the index's own arithmetic (A.5), on every index",
                            "window.mark": ("the index's own score (float32), as the organ's bridge reads it" if args.index == "I0"
                                            else "the frozen marks_for (cue_similarity, float64) on the reference's lines; differs from the index's score at the seventh decimal"),
                            "touched": ("the live index's candidate count for this read (the organ's cost)" if args.index == "I0" else "the reference's operation count for this read (cues compared, or postings read)"),
                            "short_window": ("fewer lines than k in the organ's window: its buckets proposed fewer candidates" if args.index == "I0" else "zero by construction: the reference returns k lines"),
                            "rules.oracle": "the frozen mark_oracle's rule, the citation alone (no value); rules['oracle+organ-value'] reads the value by the organ",
                            "aggregates.strict_recall_own": "strict answers citing the own line over the block; the frozen gate's strict recall (any line) is `strict`"},
           "oracle": thr_rec, "bm25": bm25, "index_label": {"I0": "the organ's live index", "I1": "the organ's cues without the buckets, a component ablation", "I2": "BM25 over the stored payloads' bytes"}[args.index],
           "live_index_geometry": live_geometry, "index_bridge": ("the frozen JournalBridge" if args.index == "I0" else "the reference bridge of cortex_c2b.references"),
           "blocks": block_summary(units), "aggregates": aggregates(units, thr),
           "answers": [u["record"] for u in units], "n_questions": len(units),
           "planting": {"record_file": str(sidecar), "sha256_lf": RS.sha256_file(sidecar, lf=True), "counts": planting["counts"], "admission": planting["admission"],
                        "first_family": first, "first_family_label": (FIRST_FAMILY_LABEL if first else None)},
           "journal": {"path": str(jpath), "entries": planting["counts"]["entries"], "hash_before": journal_hash,
                       "hash_after": record_tools.tree_hash(jpath, record_tools.payload_dir_of(jpath)), "opened": "a copy in a temporary directory"},
           "seal": seal_info, "checkpoint": {"path": str(args.ckpt), "run_id": run_id, "config_hash": ck.get("config_hash"), "step": ck.get("step"), "sha256": identity["compared"]["checkpoint_sha256"]},
           "identity": base["identity"], "sessions": base["sessions"], "run_in_sessions": base["run_in_sessions"], "duplicates_compared": base["duplicates_compared"],
           "progress_file": base["progress_file"], "progress_sha256": base["progress_sha256"],
           "device": record_tools.device_info(device), "versions": record_tools.versions(), "code_commit": record_tools.git_state()[0],
           "timestamp_utc": RS.utc_now()}
    out["journal"]["unchanged"] = out["journal"]["hash_before"] == out["journal"]["hash_after"]
    sha = RS.write_json(final_path, out)
    print(f"retained: {final_path}  sha256 {sha}")
    return out


def whole_store_of(rec: dict, whole: ExhaustiveCosine, journal: Journal) -> dict | None:
    """A.5: the own line's rank under the whole store by the index's own cosine (every live entry, no bucket), its cosine and
    the best other cosine, from the recorded query cue; None when the question has no planted line."""
    own = rec.get("own_pointer")
    if own is None or rec.get("cue") is None:
        return None
    cue = np.asarray(rec["cue"], dtype=np.float32)
    r = whole.rank(cue, k=len(journal._entries))
    own_ids = {eid for eid, e in journal._entries.items() if e.pointer == own}
    own_pos = [i for i, d in enumerate(r["docs"]) if d in own_ids]
    if not own_pos:
        return {"own_rank": None, "own_cosine": None, "best_other_cosine": (max(r["scores"]) if r["scores"] else None), "compared": r["compared"], "own_live": False}
    k = own_pos[0]
    others = [s for d, s in zip(r["docs"], r["scores"]) if d not in own_ids]
    return {"own_rank": k + 1, "own_cosine": r["scores"][k], "best_other_cosine": (max(others) if others else None), "compared": r["compared"], "own_live": True}


def probe_one(model, bridge, q: dict, args, pointer_of_planted: dict, texts: dict, thr: float, whole: ExhaustiveCosine | None = None,
              journal: Journal | None = None) -> dict:
    """One question through the frozen `_probe` under `plain` with `record_index`, the window's permutation drawn for it
    alone (D5 A), then the seven rules on the recorded output; the own line under the whole store beside (A.5)."""
    i = q["index"]
    bridge.rng = question_rng(SHUFFLE_SEED[args.form], i)
    if hasattr(bridge, "set_query"):
        bridge.set_query(q["text"])
    f = Fact(entity=q["entity"], schema=q["schema"], attr=(q["expected"] if q["expected"] is not None else "(never planted)"),
             statement=q.get("statement", ""), query=q["text"], paraphrase="", cue=np.zeros(64, dtype=np.float32))
    pointer_of = {q["entity"]: pointer_of_planted[q["entity"]]} if (q["own_pointer"] is not None and q["entity"] in pointer_of_planted) else {}
    r = _probe(model, bridge, [f], True, args.window_len, pointer_of, policy="plain", veto_threshold=None, record_index=True)
    rec = dict(r["answers"][0])
    rec.update({"index": i, "block": q["block"], "form": args.form, "pass_index": args.index, "question_template": q.get("template"),
                "own_pointer": q["own_pointer"], "expected": q["expected"], "window_size": len(rec.get("window", [])),
                "short_window": len(rec.get("window", [])) < args.k, "permutation": permutation_of_record(rec),
                "shuffle_seed": SHUFFLE_SEED[args.form]})
    for k_ in ("planted_neighbour", "attr_of_neighbour", "schema_planted", "planted_attribute", "seed"):
        if k_ in q:
            rec[k_] = q[k_]
    if q["block"] == "near_name":
        npnt = pointer_of_planted.get(q["planted_neighbour"])
        marks = {w["pointer"]: w["mark"] for w in rec.get("window", [])}
        rec["neighbour_pointer"] = npnt; rec["neighbour_in_window"] = npnt in marks; rec["neighbour_mark"] = marks.get(npnt)
        rec["neighbour_mark_at_or_above_threshold"] = bool(npnt in marks and marks[npnt] is not None and marks[npnt] >= thr)
    ref = bridge.last_reference() if hasattr(bridge, "last_reference") else None
    rec["reference"] = ref
    rec["whole_store"] = whole_store_of(rec, whole, journal) if (whole is not None and journal is not None) else None
    rec["rules"] = apply_rules(rec, texts, thr, q["expected"], args.decode_shape)
    rec["partition"] = {name: partition_of(name, rr, rec) for name, rr in rec["rules"].items()}
    return rec


def block_summary(units: list[dict]) -> dict:
    out: dict = {}
    for u in units:
        b = u["block"]; r = u["record"]
        d = out.setdefault(b, {"n": 0, "short_windows": 0, "empty_windows": 0, "window_sizes": {}})
        d["n"] += 1; d["short_windows"] += int(r["short_window"]); d["empty_windows"] += int(r["window_size"] == 0)
        d["window_sizes"][str(r["window_size"])] = d["window_sizes"].get(str(r["window_size"]), 0) + 1
    return out


def aggregates(units: list[dict], thr: float) -> dict:
    """Per block and rule: the counts of the partition, coverage, strict recall on the own line (planted), claims (the
    other blocks); the own line in the window; the record's four conditions are recomputed by the reading module."""
    out: dict = {}
    for u in units:
        b = u["block"]; r = u["record"]
        for rule, rr in r["rules"].items():
            d = out.setdefault(b, {}).setdefault(rule, {"n": 0, "claims": 0, "strict_own": 0, "strict": 0, "valid": 0, "invalid": 0, "abstain": 0, "partition": {}})
            d["n"] += 1; d["claims"] += int(rr["claim"]); d["strict"] += int(rr["outcome"] == "strict"); d["valid"] += int(rr["outcome"] == "valid")
            d["invalid"] += int(rr["outcome"] == "invalid"); d["abstain"] += int(rr["outcome"] == "abstain")
            d["strict_own"] += int(rr["outcome"] == "strict" and rr.get("cited_pointer_is_own") is True)
            p = r["partition"][rule]; d["partition"][p] = d["partition"].get(p, 0) + 1
    for b in out:
        n_own = sum(1 for u in units if u["block"] == b and u["record"].get("own_in_window"))
        for rule, d in out[b].items():
            d["coverage"] = d["claims"] / d["n"] if d["n"] else None
            d["strict_recall_own"] = d["strict_own"] / d["n"] if d["n"] else None
            d["own_in_window"] = n_own
    return out


# ---------------------------------------------------------------------------- #
# the precondition (A.5), a separate file per checkpoint                          #
# ---------------------------------------------------------------------------- #
ANSWER_KEYS = ("entity", "kind", "label", "answered", "outcome", "own_in_window", "cited_pointer", "touched", "signatures_query",
               "retrieval_order", "own_in_candidates", "own_rank", "mark_own", "mark_max")


def compare_with_row47(live: dict, row47: dict) -> dict:
    """The replayed I0 arm against row 47's plain file question by question (the answer fields above and the window's lines
    with their labels and marks), the counts, the threshold within 1e-6."""
    diffs = []
    if row47.get("checkpoint", {}).get("run_id") != live["checkpoint"]["run_id"]:
        diffs.append("the row 47 file is of another checkpoint")
    for arm in ("on", "off", "negctrl"):
        a, b = live["answers"][arm], row47["answers"][arm]
        if len(a) != len(b):
            diffs.append(f"{arm}: {len(a)} live answers against {len(b)}"); continue
        for i, (x, y) in enumerate(zip(a, b)):
            for k in ANSWER_KEYS:
                if k in y and x.get(k) != y.get(k):
                    diffs.append(f"{arm} {i} {x.get('entity')}: {k} live {x.get(k)!r} against {y.get(k)!r}")
            wx = [(w["label"], w["pointer"], w.get("mark")) for w in x.get("window", [])]
            wy = [(w["label"], w["pointer"], w.get("mark")) for w in y.get("window", [])]
            if wx != wy:
                diffs.append(f"{arm} {i} {x.get('entity')}: the window differs")
    for k in ("hm_recall_on", "hm_recall_off", "hm_false_abstention_on", "hm_invalid_citation_on", "hm_valid_citation_on", "hm_retrieval_hit", "hm_negctrl_rate"):
        if live.get(k) != row47.get(k):
            diffs.append(f"count {k}: live {live.get(k)} against {row47.get(k)}")
    thr_live, thr_47 = float(live["hm_mark_oracle"]["threshold"]), float(row47["hm_mark_oracle"]["threshold"])
    within = abs(thr_live - thr_47) <= THRESHOLD_TOL
    if not within:
        diffs.append(f"the oracle's threshold: live {thr_live} against row 47's {thr_47} (difference {thr_live - thr_47}, tolerance {THRESHOLD_TOL})")
    return {"ok": not diffs, "differences": diffs[:100], "n_differences": len(diffs),
            "threshold": {"live": thr_live, "row47": thr_47, "difference": thr_live - thr_47, "tolerance": THRESHOLD_TOL, "within": within},
            "convention": "one bridge, one stream from default_rng(1) through the blocks in row 47's order (on, off, negctrl); the frozen session_b_from_disk under plain and --index 16,exact, as row 47 ran it"}


def bm25_counts_on_journal(journal: Journal) -> dict:
    """BM25's ten selection counts recomputed on the journal's own live documents in write order against the first
    family's 200 direct queries (the documents must be the generator's 200 statements of seed 0), compared with the build's."""
    docs = [(eid, journal.payloads.get(e.pointer)) for eid, e in journal._entries.items() if e.state != "evicted"]
    facts, _ = generate_facts(N_FACTS, 0)
    statements = [d.decode("utf-8") for _, d in docs]
    pointer_index = {journal._entries[eid].pointer: j for j, (eid, _) in enumerate(docs)}
    queries = [f.query for f in facts]
    owns = [pointer_index.get(content_hash(f.statement.encode("utf-8"))) for f in facts]
    if any(o is None for o in owns):
        return {"ok": False, "reason": "the journal does not hold every statement of the first family", "n_docs": len(docs)}
    sel = bm25_selection(statements, queries, owns)
    build = bm25_selection()
    return {"ok": sel["counts"] == build["counts"] and len(docs) == N_FACTS, "n_docs": len(docs), "on_journal": sel, "build": build}


def run_precondition(args) -> dict:
    import train
    from cortex_c2b.crypto import key_from_env
    cfg = train.load_config(args.config)
    key = key_from_env()
    if key is None:
        raise SystemExit("precondition: the first family's journal is sealed under the record's key; set QUANTUM_CORTEX_JOURNAL_KEY in the window")
    out_dir = Path(args.out)
    ck = __import__("torch").load(args.ckpt, map_location="cpu", weights_only=False)
    run_id = ck.get("run_id"); del ck
    row47_path = row47_plain_file(out_dir, run_id)
    if not row47_path.exists():
        raise SystemExit(f"precondition: row 47's plain file {row47_path} is not there")
    final = out_dir / precondition_file_name(run_id)
    if final.exists():
        raise SystemExit(f"precondition: {final} exists; a file of the record is never overwritten")
    row47 = json.loads(row47_path.read_text(encoding="utf-8"))
    live = session_b_from_disk(args.ckpt, args.journal, cfg, n_facts=N_FACTS, n_negctrl=N_NEGCTRL, seed=0, k=cfg.journal_k,
                               budget=cfg.journal_read_bytes, window_len=cfg.journal_read_bytes, policy="plain",
                               index_override={"n_tables": 16, "n_bits": 10, "signature": "exact"})
    cmp = compare_with_row47(live, row47)
    with record_tools.opened_copy(Path(args.journal), key, prefix="sealed-precondition-") as (journal, _):
        bm = bm25_counts_on_journal(journal)
    out = {"benchmark": "the precondition of ADR-010 A.5 (the amendment of 2026-10-15 on parts B and C, D5 A)", "checkpoint": live["checkpoint"],
           "row47_file": {"path": str(row47_path), "sha256_lf": RS.sha256_file(row47_path, lf=True)},
           "comparison": cmp, "bm25_selection": bm, "ok": bool(cmp["ok"] and bm["ok"]),
           "live_replay": {k: live[k] for k in ("hm_recall_on", "hm_recall_off", "hm_false_abstention_on", "hm_invalid_citation_on", "hm_valid_citation_on",
                                                "hm_retrieval_hit", "hm_negctrl_rate", "cost", "index", "journal", "code_commit")},
           "live_replay_threshold": live["hm_mark_oracle"]["threshold"],
           "freeze": {"path": str(FREEZE_PATH), "sha256": (RS.sha256_file(record_tools.REPO_ROOT / FREEZE_PATH) if (record_tools.REPO_ROOT / FREEZE_PATH).exists() else None)},
           "device": live["device"], "versions": live["versions"], "code_commit": live["code_commit"], "timestamp_utc": RS.utc_now()}
    sha = RS.write_json(final, out)
    print(f"retained: {final}  sha256 {sha}")
    print(json.dumps({"ok": out["ok"], "threshold": cmp["threshold"], "n_differences": cmp["n_differences"], "bm25_ok": bm["ok"]}, indent=2))
    return out


# ---------------------------------------------------------------------------- #
# the family run: planting, then the six passes in new processes, then the file  #
# ---------------------------------------------------------------------------- #
def run_family(args) -> dict:
    import torch
    import train
    out_dir = Path(args.out)
    jpath = Path(args.journal)
    key_file = jpath.with_name(jpath.stem + ".key")
    sidecar = jpath.with_name(jpath.stem + ".planting.json")
    cfg = train.load_config(args.config)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    run_id = ck.get("run_id")
    if not run_id:
        raise SystemExit("sealed family: the checkpoint carries no run id")
    del ck
    pre = out_dir / precondition_file_name(run_id)
    if not pre.exists() or not json.loads(pre.read_text(encoding="utf-8")).get("ok"):
        raise SystemExit(f"sealed family: the precondition file {pre} must exist and say ok before any family is read (A.5)")
    final = out_dir / family_file_name(args.family, run_id)
    if final.exists():
        raise SystemExit(f"sealed family: {final} exists; a file of the record is never overwritten")
    commit, clean = record_tools.git_state()
    if commit is None or clean is not True:
        raise SystemExit("sealed family: the code's commit is unknown or the tracked tree is not clean; commit the code that runs, then run")
    fams = S.generate_all()
    fam = fams[args.family]
    seal_info = seal_check(fam)
    # the planting, atomic: a journal without its planting record is removed and planted again
    if jpath.exists() and not (sidecar.exists() and key_file.exists()):
        raise SystemExit(f"sealed family: {jpath} exists without its planting record or key; an interrupted planting is removed by hand before the run")
    if not jpath.exists():
        model, ck, device = load_model(args.ckpt, cfg)
        if getattr(model, "match_head", None) is not None:
            raise SystemExit("sealed family: the checkpoint carries a matching head (A.6's claim does not hold)")
        key = generate_key()
        jpath.parent.mkdir(parents=True, exist_ok=True)
        tmp_key = key_file.with_suffix(".key.tmp")
        tmp_key.write_text(key.hex(), encoding="utf-8", newline="\n")
        record = plant_family(model, cfg, fam, jpath, key, device, admission=args.admission)
        record.update({"code_commit": commit, "code_tree_clean": clean, "checkpoint": {"path": str(args.ckpt), "run_id": run_id, "config_hash": ck.get("config_hash")},
                       "seal": seal_info, "threshold": threshold_of(model, cfg, device, admission=args.admission)})
        row47_path = row47_plain_file(out_dir, run_id)
        if row47_path.exists():
            thr47 = float(json.loads(row47_path.read_text(encoding="utf-8"))["hm_mark_oracle"]["threshold"])
            record["threshold"]["row47"] = thr47; record["threshold"]["within_tolerance"] = abs(record["threshold"]["threshold"] - thr47) <= THRESHOLD_TOL
            if not record["threshold"]["within_tolerance"]:
                sidecar.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
                raise SystemExit(f"sealed family: the oracle's threshold {record['threshold']['threshold']} differs from row 47's {thr47} by more than 1e-6; stopped, no rate shown (A.5)")
        sidecar.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        tmp_key.replace(key_file)
        del model
        print(json.dumps({"planted": args.family, "counts": record["counts"], "admission": record["admission"]["rule"], "journal": str(jpath)}, indent=2))
    # the six passes, each in a new process from the disk alone
    results = {}
    for form in FORMS:
        for index in INDEXES:
            name = out_dir / pass_file_name(args.family, run_id, index, form)
            if name.exists():
                results[(index, form)] = name; continue
            argv = [sys.executable, "-m", "cortex_c2b.sealed_family", "--pass", "--family", args.family, "--index", index, "--form", form,
                    "--ckpt", str(args.ckpt), "--config", str(args.config), "--journal", str(jpath), "--out", str(out_dir),
                    "--admission", args.admission, "--decode-shape", args.decode_shape]
            if args.budget_seconds is not None:
                argv += ["--budget-seconds", str(args.budget_seconds)]
            if args.stop_file:
                argv += ["--stop-file", str(args.stop_file)]
            proc = subprocess.run([sys.executable, "-m", "cortex_c2b.sealed_family", *argv[3:]], text=True, cwd=str(record_tools.REPO_ROOT), encoding="utf-8", errors="replace")
            if proc.returncode != 0:
                raise SystemExit(f"sealed family: the pass {index} {form} failed ({proc.returncode}); the progress file is the attribution")
            if not name.exists():
                print(f"sealed family: the pass {index} {form} stopped before its end; the same command resumes it")
                return {"stopped": True, "pass": [index, form]}
            results[(index, form)] = name
    passes = {f"{i}-{f}": json.loads(results[(i, f)].read_text(encoding="utf-8")) for (i, f) in results}
    planting = json.loads(sidecar.read_text(encoding="utf-8"))
    out = {"benchmark": "sealed family (ADR-010 part A, B; the amendment of 2026-10-15 on parts B and C)", "family": args.family, "what": fam["what"],
           "run_id": run_id, "admission": args.admission, "decode_shape": args.decode_shape, "rules": list(RULES), "indexes": list(INDEXES), "forms": list(FORMS),
           "seal": seal_info, "planting": {k: v for k, v in planting.items() if k != "writes"}, "planting_writes": planting["writes"],
           "threshold": planting.get("threshold"),
           "precondition": {"path": str(pre), "sha256_lf": RS.sha256_file(pre, lf=True)},
           "passes": {k: {"file": str(results[tuple(k.split("-"))]), "sha256_lf": RS.sha256_file(results[tuple(k.split("-"))], lf=True),
                          "aggregates": v["aggregates"], "blocks": v["blocks"], "n_questions": v["n_questions"], "run_in_sessions": v["run_in_sessions"]} for k, v in passes.items()},
           "near_name_block": {k: v for k, v in near_name_block(fam, fams).items() if k != "questions"},
           "label": fam.get("what"), "code_commit": commit, "timestamp_utc": RS.utc_now()}
    sha = RS.write_json(final, out)
    print(f"retained: {final}  sha256 {sha}")
    return out


def run_first_family(args) -> dict:
    """A.6: the first family's paraphrase pass, run once per checkpoint on the record's journal as it is (I0, the record's
    key in the environment), in a new process from the disk alone; its file is `sealed-first-<run id>-I0-paraphrase.pass.json`."""
    import torch
    out_dir = Path(args.out)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    run_id = ck.get("run_id"); del ck
    if not run_id:
        raise SystemExit("sealed family: the checkpoint carries no run id")
    pre = out_dir / precondition_file_name(run_id)
    if not pre.exists() or not json.loads(pre.read_text(encoding="utf-8")).get("ok"):
        raise SystemExit(f"sealed family: the precondition file {pre} must exist and say ok before any pass is read (A.5)")
    commit, clean = record_tools.git_state()
    if commit is None or clean is not True:
        raise SystemExit("sealed family: the code's commit is unknown or the tracked tree is not clean; commit the code that runs, then run")
    name = out_dir / pass_file_name(FIRST_FAMILY, run_id, "I0", "paraphrase")
    if name.exists():
        print(f"sealed family: {name} exists; a file of the record is never overwritten")
        return {"exists": str(name)}
    argv = ["--pass", "--family", FIRST_FAMILY, "--index", "I0", "--form", "paraphrase", "--ckpt", str(args.ckpt), "--config", str(args.config),
            "--journal", str(args.journal), "--out", str(out_dir), "--admission", args.admission, "--decode-shape", args.decode_shape]
    if args.budget_seconds is not None:
        argv += ["--budget-seconds", str(args.budget_seconds)]
    if args.stop_file:
        argv += ["--stop-file", str(args.stop_file)]
    proc = subprocess.run([sys.executable, "-m", "cortex_c2b.sealed_family", *argv], text=True, cwd=str(record_tools.REPO_ROOT), encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SystemExit(f"sealed family: the first family's paraphrase pass failed ({proc.returncode}); the progress file is the attribution")
    if not name.exists():
        print("sealed family: the pass stopped before its end; the same command resumes it")
        return {"stopped": True}
    return {"retained": str(name)}


# ---------------------------------------------------------------------------- #
# command line                                                                   #
# ---------------------------------------------------------------------------- #
def _cli(argv=None):
    import train
    ap = argparse.ArgumentParser(description="the runner of ADR-010's sealed families (new module, the frozen system at c5b4818)")
    ap.add_argument("--precondition", action="store_true", help="A.5: the I0 arm replayed on the first family's journal as row 47 ran it, compared with row 47's file; the key in QUANTUM_CORTEX_JOURNAL_KEY")
    ap.add_argument("--family", default=None, choices=list(FAMILIES_READ) + [FIRST_FAMILY], help="T0 | A | Av; `first` is the first family's paraphrase pass on the record's journal (A.6)")
    ap.add_argument("--pass", dest="one_pass", action="store_true", help="(internal) one form under one index, a new process from the disk alone")
    ap.add_argument("--index", default=None, choices=list(INDEXES)); ap.add_argument("--form", default=None, choices=list(FORMS))
    ap.add_argument("--ckpt", required=True); ap.add_argument("--config", required=True); ap.add_argument("--journal", required=True)
    ap.add_argument("--out", default="metrics/mqar")
    ap.add_argument("--admission", default=None, choices=["surprise", "identity"], help="the rule passed to the frozen functions; the record's is identity (a run that omits it is refused)")
    ap.add_argument("--decode-shape", default=None, choices=["none", "R1", "R2", "R3", "R4"], help="the shape passed to the frozen select_citation under mark-veto+value; the record's is R4 (a run that omits it is refused)")
    ap.add_argument("--budget-seconds", type=float, default=None, help="4 (e): the session's wall clock budget, counted from its first unit")
    ap.add_argument("--stop-file", default=None, help="4 (e): the stop request, a file whose appearance stops the session at the next recorded point")
    args = ap.parse_args(argv)
    args.argv = list(argv) if argv is not None else sys.argv[1:]          # the arguments as given, for the recorded command
    cfg = train.load_config(args.config)
    args.k = cfg.journal_k; args.budget = cfg.journal_read_bytes; args.window_len = cfg.journal_read_bytes
    if args.precondition:
        RS.require_launcher()
        if "metrics" in Path(args.out).parts and not (record_tools.REPO_ROOT / FREEZE_PATH).exists():
            raise SystemExit(f"sealed family: {FREEZE_PATH} must exist before a run of the record (python -m cortex_c2b.freeze --write --commit c5b4818d3db29902f72d1b3078e01c4b3cb1e634); refused")
        return run_precondition(args)
    if args.admission != ADMISSION_IDENTITY or args.decode_shape != "R4":
        raise SystemExit("sealed family: the system's recipe is declared and not a code default: --admission identity and --decode-shape R4 are required (the amendment, section 0)")
    RS.require_launcher()
    if "metrics" in Path(args.out).parts and not (record_tools.REPO_ROOT / FREEZE_PATH).exists():
        raise SystemExit(f"sealed family: {FREEZE_PATH} must exist before a run of the record (python -m cortex_c2b.freeze --write --commit c5b4818d3db29902f72d1b3078e01c4b3cb1e634); refused")
    if args.family is None:
        ap.error("--family is required (T0 | A | Av | first)")
    if args.one_pass:
        if args.index is None or args.form is None:
            ap.error("--pass needs --index and --form")
        return run_pass(args)
    if args.family == FIRST_FAMILY:
        return run_first_family(args)
    return run_family(args)


if __name__ == "__main__":
    _cli()
