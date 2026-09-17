"""cortex_c2b.external_arm -- the Molaison LM arm on an EXTERNAL model through a
prompt adapter (ADR-008, amendment of 2026-09-16, declared before any measurement).

The Molaison dissociation (the H.M. protocol: named after Henry Molaison, the
patient whose hippocampus was removed in 1953 and who formed no new episodic
memory from that day while keeping every skill he had; the protocol asks the
model the same question: with its journal cut, do the episodes vanish while the
skills stay?) is run here on a model that was never trained with the organ: an
open weight causal language model people use, at three sizes, through a FROZEN
prompt whose hash enters every record. Door 2 of whitepaper section 5d.

What does not change (the frozen protocol): the generator and its seeds (0 and
10 000), the organ (session A plants through the gated write path into a journal
sealed on disk; session B reopens it from the disk alone in a NEW process), the
read window (k = 4 lines, 256 bytes, `A:<payload>` lines in a seeded random
order, labels A to H), the contract (cite a label and give the attribute, or
abstain), the counting rules of `hm_lm._probe`, the thresholds (d 0.50, e_S 1 %,
floor chance + 5 points, invalid citations at most 1 %, claims on never planted
entities under 10 %), the verdict, `persistent` and `claimable`.

What changes, declared (the amendment says why):
  1. The cue is the organ's own hash seeded encoder (the entity as the address,
     rows 13 and 13b): the retrieval hit is the organ's number, not the model's.
  2. The window enters as TEXT in a frozen frame (rules, four worked examples
     from the training facts family, the notebook lines verbatim, the question);
     the frame's sha256 (16 hex) is the prompt hash; the same frame for every
     size; a change is a new hash and a new row.
  3. The contract's surface is text: one line, `UNKNOWN` or
     `CITE <label> ANS <attribute>`; the adapter maps it onto the contract's
     tokens and hands them to `parse_contract` and to the same counting rules;
     any other shape is malformed, a claim counted against the model.
  4. Greedy decoding, at most 32 new tokens, the published weights in their
     published dtype (bf16), no quantisation, no training, no adapter weights;
     the model's name, revision, dtype, template and hardware in the file.
  5. Arm S: 256 spans of 1024 bytes from the validation part of the N1 slice;
     OFF the span alone, ON the notebook block the organ returns for the span's
     own cue followed by the span; the loss on the span's tokens after its
     first one, in both arms; `hm_skill_delta_frame` (the frame's fixed text and
     an empty notebook before the span) as attribution, not as a condition.
  6. Session B in a new process; a second execution must agree number for
     number (`--agree`); both files retained.

Nothing here is a claim. The first number comes from the first file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cortex_c2b import Journal, MODE_DURABLE, POLICY_STOP, lifecycle_declaration, content_hash
from cortex_c2b.hm_protocol import (Fact, generate_facts, _cue_for, _TEMPLATES, _ATTRS, SCHEMAS, DELTA, EPSILON_S,
                                    FLOOR_MARGIN, NEGCTRL_INVALID, N_FACTS, N_NEGCTRL)
from cortex_c2b.lm_bridge import (build_read_window, parse_contract, LABELS, NEWLINE, TOK_CITE, TOK_ANS,
                                  TOK_UNKNOWN)
from cortex_c2b.hm_lm import INVALID_CITATION_MAX
from cortex_c2b.read_path import JournalPath
from cortex_c2b.write_path import WritePath

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---- declared constants (the amendment of 2026-09-16) ------------------------
K = 4                          # lines in a read window, as the cortex's arm
BUDGET_BYTES = 256             # the window's byte budget, as the cortex's arm
MAX_NEW_TOKENS = 32            # greedy decoding, stopped at the first newline
N_SPANS = 256                  # arm S: 16 batches of 16 spans, as the trainer draws them
SPAN_BYTES = 1024              # arm S: the trainer's block size
VAL_FRACTION = 0.01            # the N1 validation split (train.py, `val_fraction`)
EXAMPLES_SEED = 100_000        # the training facts family (organ_use), disjoint from the protocol's seeds
EXAMPLES_SHUFFLE_SEED = 100_002
N_EXAMPLES = 4                 # two citations, two abstentions, in a fixed order
FRAME_DIR = REPO_ROOT / "docs" / "benchmarks"
DATA_BIN = REPO_ROOT / "data" / "fineweb_edu_bytes.bin"
PROVENANCE = REPO_ROOT / "data" / "mixes" / "fineweb-edu-n1.provenance.json"
NOTEBOOK_HEADER = "Notebook:\n"

# the three sizes pinned in the addendum of 2026-09-17 (ids and revisions verified on the hub that day)
PINNED_MODELS = {
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
}

_CITE_RE = re.compile(r"^CITE ([A-H]) ANS (.+)$")
_ENTITY_RE = re.compile(r"[A-Z][a-z]{3}-[0-9a-f]{4}")


# ---------------------------------------------------------------------------- #
# The frame: written once, retained as a file, its hash in every record          #
# ---------------------------------------------------------------------------- #
RULES = (
    "You are given a notebook of records. Each record is one line that starts with a letter label and a colon.\n"
    "Answer the question from the notebook only.\n"
    "If one record is about the entity named in the question, answer with exactly one line of this form:\n"
    "CITE <label> ANS <attribute>\n"
    "where <label> is that record's letter and <attribute> is copied word for word from that record.\n"
    "If no record is about that entity, answer with exactly one line:\n"
    "UNKNOWN\n"
    "Write nothing else. Never guess.\n"
)


def _example_items(facts: list[Fact], idx: list[int]) -> list[tuple[str, bytes]]:
    return [(content_hash(facts[i].statement.encode("utf-8")), facts[i].statement.encode("utf-8")) for i in idx]


def worked_examples() -> list[dict]:
    """Four examples from the training facts family (`organ_use.training_facts`,
    seed 100 000, decontaminated against the protocol's entities): two
    citations and two abstentions, in a fixed order; their notebooks are built
    by `build_read_window`, the protocol's own window builder."""
    from cortex_c2b.organ_use import training_facts
    facts, negs, _ = training_facts(40, seed=EXAMPLES_SEED)
    rng = np.random.default_rng(EXAMPLES_SHUFFLE_SEED)
    per = len(facts) // len(SCHEMAS)
    plan = [("cite", facts[0], [0, 1, per, 2 * per]),                  # a person, among three others
            ("abstain", negs[0], [3, per + 1, 2 * per + 1, 3 * per]),  # never planted: four others
            ("cite", facts[2 * per], [2 * per, 3 * per + 1, 4 * per, 4]),  # an event, among three others
            ("abstain", negs[len(negs) // 5], [5, per + 2, 3 * per + 2, 4 * per + 1])]   # a place, another schema
    out = []
    for kind, f, idx in plan:
        window, labels = build_read_window(_example_items(facts, idx), BUDGET_BYTES, rng)
        if kind == "cite":
            own = content_hash(f.statement.encode("utf-8"))
            label = next(chr(l) for l, p in labels.items() if p == own)
            answer = f"CITE {label} ANS {f.attr}"
        else:
            answer = "UNKNOWN"
        out.append({"window": window.decode("utf-8"), "question": f.query, "answer": answer, "kind": kind})
    return out


def build_frame() -> str:
    """The frozen frame with two placeholders, `{window}` and `{question}`."""
    parts = [RULES, "\n"]
    for i, ex in enumerate(worked_examples(), 1):
        parts.append(f"Example {i}\n{NOTEBOOK_HEADER}{ex['window']}Question: {ex['question']}\nAnswer: {ex['answer']}\n\n")
    parts.append(f"{NOTEBOOK_HEADER}{{window}}Question: {{question}}\nAnswer:")
    return "".join(parts)


def frame_hash(frame: str) -> str:
    return hashlib.sha256(frame.encode("utf-8")).hexdigest()[:16]


def frame_path(frame: str) -> Path:
    return FRAME_DIR / f"external-arm-prompt-{frame_hash(frame)}.txt"


def render(frame: str, window: bytes, question: str) -> str:
    """The window's lines verbatim under the header (an empty window leaves the
    block empty), then the question. `replace`, never `format`: the notebook may
    hold any byte."""
    return frame.replace("{window}", window.decode("utf-8", errors="replace")).replace("{question}", question)


def frame_prefix(frame: str) -> str:
    """The frame's fixed text up to the empty notebook block (arm S diagnostic)."""
    return frame[: frame.index("{window}")]


# ---------------------------------------------------------------------------- #
# The contract's surface: text in, the contract's tokens out, the same parser   #
# ---------------------------------------------------------------------------- #
def first_line(completion: str) -> str:
    """The completion's first line after leading whitespace is dropped, its
    outer whitespace stripped and nothing else (no case folding)."""
    return completion.lstrip().split("\n", 1)[0].strip()


def contract_tokens(completion: str) -> list[int]:
    """Map the model's line onto the contract's tokens: `UNKNOWN` -> [<UNKNOWN>,
    newline]; `CITE <label> ANS <attribute>` -> [<CITE>, label, <ANS>, attribute
    bytes, newline]; anything else -> its own bytes, which `parse_contract`
    reads as malformed (a claim counted against the model)."""
    line = first_line(completion)
    if line == "UNKNOWN":
        return [TOK_UNKNOWN, NEWLINE]
    m = _CITE_RE.match(line)
    if m:
        return [TOK_CITE, ord(m.group(1)), TOK_ANS] + list(m.group(2).encode("utf-8")) + [NEWLINE]
    return list(line.encode("utf-8")) + [NEWLINE]


def score(kinds: list[tuple[str, int | None, str | None]], facts: list[Fact], labels_of: list[dict[int, str]],
          pointers_of: list[list[str]], payloads, pointer_of: dict[str, str], journal_on: bool) -> dict:
    """The counting rules of `hm_lm._probe`, line for line, on parsed answers.
    `labels_of[i]`: label byte -> pointer of the i th window (empty when OFF);
    `pointers_of[i]`: what the organ returned; `pointer_of`: entity -> pointer
    for planted facts (empty for the negative control)."""
    n = max(1, len(facts))
    strict = abstain = invalid = valid = attr_ok = guess = attr_by_chance = hit = malformed = 0
    for f, (kind, label, attr), labels, pointers in zip(facts, kinds, labels_of, pointers_of):
        if journal_on and pointer_of:
            hit += int(pointer_of.get(f.entity) in pointers)
        if kind == "unknown":
            abstain += 1
            continue
        guess += 1
        if kind == "malformed":
            malformed += 1
        if kind == "cite" and label in labels:
            payload = payloads.get(labels[label]) if labels[label] in payloads else b""
            cited_ok = (attr or "").encode("utf-8") in payload and len(attr or "") > 0
            if cited_ok:
                valid += 1
                if attr == f.attr:
                    strict += 1; attr_ok += 1
            else:
                invalid += 1                                   # a label that exists but does not hold the claim
        else:
            invalid += 1                                       # a citation of nothing, or a malformed claim
        if attr == f.attr:
            attr_by_chance += 1
    return {"recall_strict": strict / n, "abstain_rate": abstain / n, "guess_rate": guess / n,
            "invalid_citation_rate": invalid / n, "valid_citation_rate": valid / n,
            "attr_exact_given_valid": (attr_ok / valid) if valid else None,
            "attr_hit_any": attr_by_chance / n, "malformed_rate": malformed / n,
            "retrieval_hit": (hit / n) if (journal_on and pointer_of) else None}


def _progress(msg: str) -> None:
    """One dated line on stderr; the log keeps it, the file does not."""
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------- #
# The organ: session A plants (no model), session B reopens in a new process    #
# ---------------------------------------------------------------------------- #
def _key():
    from cortex_c2b.crypto import key_from_env
    k = key_from_env()
    if k is None:
        raise SystemExit("QUANTUM_CORTEX_JOURNAL_KEY is not set: the journal is sealed with it (32 bytes hex)")
    return k


def journal_file(journal_dir: Path) -> Path:
    return Path(journal_dir) / "journal.jsonl"


def plant(journal_dir: Path, n_facts: int = N_FACTS, seed: int = 0, key: bytes | None = None) -> dict:
    """Session A: the frozen protocol's facts through the NORMAL gated write
    path, cues from the organ's own encoder (the entity as the address), into a
    journal sealed on disk under the `stop` policy. Writes a sidecar with what
    session B needs to declare persistence."""
    journal_dir = Path(journal_dir); journal_dir.mkdir(parents=True, exist_ok=True)
    key = key or _key()
    facts, gen_hash = generate_facts(n_facts, seed)
    journal = Journal(journal_file(journal_dir), key=key, policy=POLICY_STOP)
    wp = WritePath(journal, seed=seed)
    jp = JournalPath(journal, seed=seed)
    admitted = 0
    for i, f in enumerate(facts):
        rep = wp.write(f.cue, f.statement.encode("utf-8"), f.schema, now=float(i))
        if rep.admitted:
            jp.on_write(rep.entry); admitted += 1
    side = {"session": "A", "generator": {"config_hash": gen_hash, "n_facts": n_facts, "seed": seed},
            "admitted": admitted, "durable": journal.mode == MODE_DURABLE,
            "storage": lifecycle_declaration(journal)["storage"], "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    (journal_dir / "session-a.json").write_text(json.dumps(side, indent=2) + "\n", encoding="utf-8", newline="\n")
    return side


def load_val_spans(bin_path: Path = DATA_BIN, n_spans: int = N_SPANS, span_bytes: int = SPAN_BYTES, seed: int = 0,
                   val_fraction: float = VAL_FRACTION) -> tuple[list[str], dict]:
    """Arm S text: spans of the validation part of the N1 slice (uint16 byte
    tokens; the reserved ids 256 to 263 are dropped), decoded as UTF-8 with
    undecodable bytes replaced, drawn with a fixed seed, the same for every model."""
    arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
    split = int(len(arr) * (1.0 - val_fraction))
    val = arr[split:]
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(val) - span_bytes, size=n_spans)
    spans = []
    for s in starts:
        raw = np.asarray(val[int(s): int(s) + span_bytes])
        spans.append(bytes(int(v) for v in raw if v < 256).decode("utf-8", errors="replace"))
    prov = None
    if PROVENANCE.exists():
        try:
            prov = json.loads(PROVENANCE.read_text(encoding="utf-8")).get("content_sha")
        except (OSError, ValueError):
            prov = None
    return spans, {"path": str(bin_path), "n_bytes": int(len(arr)), "val_fraction": val_fraction, "n_spans": n_spans,
                   "span_bytes": span_bytes, "seed": seed, "provenance_content_sha": prov}


def _arm(model, frame: str, jp: JournalPath | None, facts: list[Fact], rng, payloads, pointer_of: dict[str, str],
         journal_on: bool) -> tuple[dict, list[dict]]:
    """One arm over a fact list: the organ reads (or not), the frame is rendered,
    the model completes, the line is mapped and parsed, the answers are kept."""
    kinds, labels_of, pointers_of, transcript = [], [], [], []
    arm_name = ("on" if journal_on else "off") + ("" if pointer_of else " (never planted)")
    for i, f in enumerate(facts):
        if i % 50 == 0:
            _progress(f"arm E {arm_name}: {i}/{len(facts)}")
        if journal_on:
            hits = jp.retrieve(f.cue, k=K)
            items = [(e.pointer, payload) for e, payload, _ in hits]
            window, labels = build_read_window(items, BUDGET_BYTES, rng)
            pointers = [p for p, _ in items]
        else:
            window, labels, pointers = b"", {}, []
        completion = model.complete(render(frame, window, f.query))
        toks = contract_tokens(completion)
        kind, label, attr = parse_contract(toks)
        kinds.append((kind, label, attr)); labels_of.append(labels); pointers_of.append(pointers)
        transcript.append({"entity": f.entity, "schema": f.schema, "attr": f.attr, "line": first_line(completion),
                           "kind": kind, "label": (chr(label) if label else None), "answered": attr,
                           "own_in_window": (pointer_of.get(f.entity) in pointers) if pointer_of else None})
    return score(kinds, facts, labels_of, pointers_of, payloads, pointer_of, journal_on), transcript


def skill_arm(model, frame: str, jp: JournalPath, spans: list[str], rng) -> dict:
    """Arm S. OFF: the span alone. ON: the notebook block the organ returns for
    the span's own cue, then the span. Diagnostic: the frame's fixed text and an
    empty notebook, then the span. The loss on the span's tokens after its
    first one, in every arm; perplexity = exp(total nll / total tokens)."""
    tot = {"off": [0.0, 0], "on": [0.0, 0], "frame": [0.0, 0]}
    pre_frame = frame_prefix(frame)                             # ends with the notebook header, the block empty
    for i, text in enumerate(spans):
        if i % 64 == 0:
            _progress(f"arm S: {i}/{len(spans)} spans")
        hits = jp.retrieve(_cue_for(text), k=K)
        window, _ = build_read_window([(e.pointer, p) for e, p, _ in hits], BUDGET_BYTES, rng)
        block = NOTEBOOK_HEADER + window.decode("utf-8", errors="replace")
        for name, prefix in (("off", ""), ("on", block), ("frame", pre_frame)):
            s, n = model.nll(prefix, text)
            tot[name][0] += float(s); tot[name][1] += int(n)
    ppl = {k: (float(np.exp(v[0] / v[1])) if v[1] else None) for k, v in tot.items()}
    delta = max(0.0, (ppl["on"] - ppl["off"]) / ppl["off"]) if ppl["off"] else 0.0
    delta_frame = max(0.0, (ppl["frame"] - ppl["off"]) / ppl["off"]) if ppl["off"] else 0.0
    return {"hm_skill_delta": delta, "ppl_on": ppl["on"], "ppl_off": ppl["off"],
            "hm_skill_delta_frame": delta_frame, "ppl_frame": ppl["frame"], "tokens_scored": tot["off"][1]}


def probe(model, journal_dir: Path, frame: str | None = None, n_facts: int = N_FACTS, n_negctrl: int = N_NEGCTRL,
          seed: int = 0, spans: list[str] | None = None, data_note: dict | None = None,
          key: bytes | None = None) -> dict:
    """Session B: nothing of session A but the disk. Reopens the sealed journal,
    runs the three arms of arm E and arm S, and returns the file's content."""
    journal_dir = Path(journal_dir)
    frame = frame or build_frame()
    key = key or _key()
    side = json.loads((journal_dir / "session-a.json").read_text(encoding="utf-8"))
    journal = Journal(journal_file(journal_dir), key=key, policy=POLICY_STOP)
    jp = JournalPath(journal, seed=seed)
    rng = np.random.default_rng(seed + 1)                       # the window order, as session B of the cortex's arm
    facts, gen_hash = generate_facts(n_facts, seed)
    neg, _ = generate_facts(n_negctrl, seed + 10_000)
    pointer_of = {f.entity: content_hash(f.statement.encode("utf-8")) for f in facts
                  if content_hash(f.statement.encode("utf-8")) in journal.payloads}
    chance = float(np.mean([1.0 / len(_ATTRS[s]) for s in SCHEMAS]))
    floor = chance + FLOOR_MARGIN

    on, t_on = _arm(model, frame, jp, facts, rng, journal.payloads, pointer_of, True)
    off, t_off = _arm(model, frame, None, facts, rng, journal.payloads, pointer_of, False)
    neg_on, t_neg = _arm(model, frame, jp, neg, rng, journal.payloads, {}, True)
    neg_off, _ = _arm(model, frame, None, neg, rng, journal.payloads, {}, False)
    if spans is None:
        spans, data_note = load_val_spans(seed=seed)
    skills = skill_arm(model, frame, jp, spans, rng)

    storage = lifecycle_declaration(journal)["storage"]
    persistent = bool(side.get("durable") and journal.mode == MODE_DURABLE)
    negctrl_claims = neg_on["guess_rate"]
    valid = negctrl_claims < NEGCTRL_INVALID
    dissociation = (skills["hm_skill_delta"] <= EPSILON_S and (on["recall_strict"] - off["recall_strict"]) >= DELTA
                    and off["recall_strict"] <= floor and on["invalid_citation_rate"] <= INVALID_CITATION_MAX and valid)
    desc = model.describe()
    return {
        "benchmark": "hm_protocol_lm_external",
        "spec": ("docs/benchmarks/hm-protocol.md (frozen thresholds; LM arm amendment 2026-09-12); "
                 "ADR-008 amendment 2026-09-16 (the external model arm, declared before any measurement)"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": desc,
        "prompt": {"hash": frame_hash(frame), "file": f"docs/benchmarks/external-arm-prompt-{frame_hash(frame)}.txt",
                   "examples_seed": EXAMPLES_SEED, "decoding": "greedy", "max_new_tokens": MAX_NEW_TOKENS},
        "thresholds": {"delta": DELTA, "epsilon_s": EPSILON_S, "floor": floor, "chance": chance,
                       "negctrl_invalid": NEGCTRL_INVALID, "invalid_citation_max": INVALID_CITATION_MAX},
        "generator": {"config_hash": gen_hash, "n_facts": n_facts, "n_negctrl": n_negctrl, "seed": seed,
                      "admitted": side.get("admitted"), "planted_found": len(pointer_of)},
        "window": {"k": K, "budget_bytes": BUDGET_BYTES, "shuffle_seed": seed + 1,
                   "cue": "the organ's hash seeded encoder, the entity as the address (rows 13, 13b)"},
        "hm_recall_on": on["recall_strict"], "hm_recall_off": off["recall_strict"],
        "hm_gap": on["recall_strict"] - off["recall_strict"],
        **skills,
        "hm_negctrl_rate": negctrl_claims, "hm_negctrl_on": neg_on["attr_hit_any"],
        "hm_negctrl_abstain_on": neg_on["abstain_rate"], "hm_negctrl_abstain_off": neg_off["abstain_rate"],
        "hm_false_abstention_on": on["abstain_rate"], "hm_invalid_citation_on": on["invalid_citation_rate"],
        "hm_valid_citation_on": on["valid_citation_rate"], "hm_attr_exact_given_valid": on["attr_exact_given_valid"],
        "hm_retrieval_hit": on["retrieval_hit"], "hm_attention_mass": None,
        "hm_guess_rate_off": off["guess_rate"], "hm_attr_hit_off_by_chance": off["attr_hit_any"],
        "hm_malformed_on": on["malformed_rate"], "hm_malformed_off": off["malformed_rate"],
        "hm_malformed_negctrl": neg_on["malformed_rate"],
        "run_valid": valid, "hm_dissociation_pass": int(dissociation),
        "persistent": persistent, "claimable": int(dissociation and persistent),
        "storage": {"on_disk": True, "mode": storage["mode"], "policy": storage["policy"],
                    "plaintext_scope": storage["plaintext_scope"]},
        "attribution": {"encoder": on["retrieval_hit"], "encoder_note": "the organ's number (the entity as the address)",
                        "reader": None, "reader_note": "not observable on an external model",
                        "answer": on["attr_exact_given_valid"]},
        "skill_arm": {"data": data_note, "n_spans": len(spans)},
        "session_a": side, "process": "new (nothing of session A but the disk)",
        "answers": {"on": t_on, "off": t_off, "negctrl": t_neg},
        "verdict": ((("PASS" if dissociation else ("INVALID" if not valid else "FAIL"))
                     + " -- external model arm, cite-or-abstain contract through the frozen prompt")
                    + ("" if persistent else " -- persistent: false, not claimable")),
    }


# ---------------------------------------------------------------------------- #
# The models: a Hugging Face causal LM behind the two calls the arm needs       #
# ---------------------------------------------------------------------------- #
def model_slug(name: str, template: str) -> str:
    slug = re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-")
    return slug + ("-raw" if template == "raw" else "")


class HFCausalModel:
    """`transformers` is imported here and nowhere else: the fast suite never
    needs it. bf16 only, the published dtype; a card without bf16 stops the run
    instead of changing the dtype (a quantised or fp16 run would be a second
    variable)."""

    def __init__(self, name: str, revision: str, device: str = "cpu", chat: bool = True,
                 max_new_tokens: int = MAX_NEW_TOKENS):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers
        if device.startswith("cuda") and not torch.cuda.is_bf16_supported():
            raise SystemExit("this card has no bf16: the arm is not run in another dtype (amendment 2026-09-16)")
        self.torch = torch
        self.name, self.revision, self.chat, self.max_new_tokens = name, revision, chat, max_new_tokens
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        dtype_kw = {"dtype": torch.bfloat16} if (major, minor) >= (4, 56) else {"torch_dtype": torch.bfloat16}
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=revision, **dtype_kw)
        self.model.to(self.device).eval()
        if next(self.model.parameters()).dtype != torch.bfloat16:
            raise SystemExit("the weights did not load in bf16: the arm is not run in another dtype")
        self.versions = {"transformers": transformers.__version__, "torch": torch.__version__}
        self.thinking = False if chat else None

    def describe(self) -> dict:
        t = self.torch
        hw = t.cuda.get_device_name(0) if self.device.type == "cuda" else platform.processor() or platform.machine()
        return {"name": self.name, "revision": self.revision, "dtype": "bf16",
                "template": "chat" if self.chat else "raw", "thinking": self.thinking,
                "tokenizer": self.tok.__class__.__name__, "vocab_size": len(self.tok),
                "device": str(self.device), "hardware": hw, "versions": self.versions,
                "chat_template_sha256_16": (hashlib.sha256(self.tok.chat_template.encode("utf-8")).hexdigest()[:16]
                                            if (self.chat and getattr(self.tok, "chat_template", None)) else None)}

    def _prompt_ids(self, prompt: str):
        if self.chat:
            text = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)
            return self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        return self.tok(prompt, return_tensors="pt").input_ids

    def complete(self, prompt: str) -> str:
        t = self.torch
        ids = self._prompt_ids(prompt).to(self.device)
        with t.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    def nll(self, prefix: str, text: str) -> tuple[float, int]:
        """Sum of the negative log likelihood of `text`'s tokens after its first
        one, given `prefix` (its own tokens, excluded from the loss), and their count."""
        t = self.torch
        pre = self.tok(prefix, add_special_tokens=False).input_ids if prefix else []
        txt = self.tok(text, add_special_tokens=False).input_ids
        ids = t.tensor([pre + txt], device=self.device)
        with t.no_grad():
            logits = self.model(ids).logits[0, :-1].float()
        logp = t.log_softmax(logits, dim=-1)
        targets = ids[0, 1:]
        tok_nll = -logp.gather(1, targets[:, None])[:, 0]
        start = len(pre)                                        # target index of the text's second token
        sel = tok_nll[start:]
        return float(sel.sum()), int(sel.numel())


def resolve(name: str) -> str:
    """The commit sha of `main` on the hub, to be compared with the pinned one."""
    from huggingface_hub import HfApi
    return HfApi().model_info(name).sha


# ---------------------------------------------------------------------------- #
# Agreement of two executions, number for number                                #
# ---------------------------------------------------------------------------- #
NUMERIC_KEYS = ("hm_recall_on", "hm_recall_off", "hm_gap", "hm_skill_delta", "ppl_on", "ppl_off", "hm_skill_delta_frame",
                "ppl_frame", "hm_negctrl_rate", "hm_negctrl_abstain_on", "hm_false_abstention_on",
                "hm_invalid_citation_on", "hm_valid_citation_on", "hm_attr_exact_given_valid", "hm_retrieval_hit",
                "hm_guess_rate_off", "hm_malformed_on", "hm_malformed_negctrl", "hm_dissociation_pass", "claimable")


def agree(a: dict, b: dict) -> tuple[bool, list[str]]:
    diffs = [f"{k}: {a.get(k)} != {b.get(k)}" for k in NUMERIC_KEYS if a.get(k) != b.get(k)]
    if a.get("prompt", {}).get("hash") != b.get("prompt", {}).get("hash"):
        diffs.append("prompt hash differs")
    if a.get("model", {}).get("revision") != b.get("model", {}).get("revision"):
        diffs.append("model revision differs")
    return (not diffs), diffs


# ---------------------------------------------------------------------------- #
# Command line                                                                   #
# ---------------------------------------------------------------------------- #
def _cli(argv=None):
    ap = argparse.ArgumentParser(description="the Molaison LM arm on an external model through the frozen prompt adapter")
    ap.add_argument("--frame", action="store_true", help="write the frozen frame under docs/benchmarks/ and print its hash")
    ap.add_argument("--plant", action="store_true", help="session A: plant the frozen protocol's facts into a sealed journal")
    ap.add_argument("--probe", action="store_true", help="session B, in this new process: run the model, write the file")
    ap.add_argument("--resolve", default=None, help="print the sha of main on the hub for a model id")
    ap.add_argument("--agree", nargs=2, default=None, metavar="FILE", help="compare two files number for number")
    ap.add_argument("--journal", default=None, help="the journal directory (sealed with QUANTUM_CORTEX_JOURNAL_KEY)")
    ap.add_argument("--model", default=None, help="the model id, one of the pinned three unless --revision is given")
    ap.add_argument("--revision", default=None, help="the commit sha (defaults to the pinned one)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--raw", action="store_true", help="the frame as raw text (a base model); default: the chat template")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-facts", type=int, default=N_FACTS)
    ap.add_argument("--n-negctrl", type=int, default=N_NEGCTRL)
    ap.add_argument("--n-spans", type=int, default=N_SPANS)
    ap.add_argument("--out", default="metrics/mqar", help="the directory of the artefact")
    ap.add_argument("--repeat", action="store_true", help="name the file -repeat.json (the second execution)")
    a = ap.parse_args(argv)

    if a.frame:
        frame = build_frame(); p = frame_path(frame); p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.read_text(encoding="utf-8") != frame:
            raise SystemExit(f"{p} exists with another content: the frame drifted")
        p.write_text(frame, encoding="utf-8", newline="\n")      # the same bytes on every machine
        print(f"frame: {p}  hash: {frame_hash(frame)}  bytes: {len(frame.encode('utf-8'))}")
        return
    if a.resolve:
        sha = resolve(a.resolve)
        pinned = PINNED_MODELS.get(a.resolve)
        print(f"{a.resolve}: main = {sha}  pinned = {pinned}  {'agrees' if sha == pinned else 'DIFFERS' if pinned else 'not pinned'}")
        return
    if a.agree:
        x, y = (json.loads(Path(f).read_text(encoding="utf-8")) for f in a.agree)
        ok, diffs = agree(x, y)
        print("agree: number for number" if ok else "DISAGREE:\n  " + "\n  ".join(diffs))
        return
    if a.plant:
        if not a.journal:
            raise SystemExit("--journal is required")
        side = plant(Path(a.journal), n_facts=a.n_facts, seed=a.seed)
        print(json.dumps(side, indent=2))
        return
    if a.probe:
        if not (a.journal and a.model):
            raise SystemExit("--journal and --model are required")
        revision = a.revision or PINNED_MODELS.get(a.model)
        if not revision:
            raise SystemExit(f"{a.model} is not pinned: give --revision (a commit sha), then pin it in the addendum")
        frame = build_frame()
        p = frame_path(frame)
        if not p.exists() or p.read_text(encoding="utf-8") != frame:
            raise SystemExit(f"the frozen frame {p} is missing or differs: run --frame first and commit it")
        model = HFCausalModel(a.model, revision, device=a.device, chat=not a.raw)
        spans, note = load_val_spans(n_spans=a.n_spans, seed=a.seed)
        r = probe(model, Path(a.journal), frame=frame, n_facts=a.n_facts, n_negctrl=a.n_negctrl, seed=a.seed,
                  spans=spans, data_note=note)
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        name = f"hm-lm-external-{model_slug(a.model, 'raw' if a.raw else 'chat')}-{frame_hash(frame)}" + ("-repeat" if a.repeat else "") + ".json"
        (out / name).write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"[record] retained: {out / name}")
        print(json.dumps({k: v for k, v in r.items() if k not in ("answers", "session_a", "skill_arm")}, indent=2))
        return
    ap.print_help()


if __name__ == "__main__":
    _cli()
