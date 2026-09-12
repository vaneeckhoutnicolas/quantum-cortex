"""cortex_c2b.hm_lm -- the H.M. protocol on the MODEL (ADR-008; spec amendment 2026-09-12).

The organ-level protocol (`hm_protocol.py`) reads payloads with a hash-seeded
cue encoder and a reference skill probe. This module runs the same two arms on
the language model itself, under the cite-or-abstain contract:

  Arm E, journal ON : for each planted fact, the query's cue comes from the
    model, the organ returns a labelled read window, the model generates, and
    the answer is parsed against the contract. STRICT recall = a valid citation
    (a label that exists, whose episode holds the attribute) AND the exact
    attribute. Reported next to it: abstention rate on the negative control,
    false abstention on planted facts (the cost of prudence), invalid citation
    rate (a claim with a fake provenance, the worst case), attribute exactness
    given a valid citation, retrieval hit rate (the encoder), attention mass on
    the retrieved bytes (the reader). A failure is attributable (invariant 8).
  Arm E, journal OFF: the reader is absent (no window, gate 0). A citation is
    impossible, so the strict recall is zero by construction -- the parametric
    floor in its cleanest form; the guess rate and the by-chance attribute hits
    are reported so the floor keeps its meaning.
  Arm S: ordinary language modelling with the reader ACTIVE on whatever the
    journal returns for each span (journal ON) against the reader absent (OFF):
    `hm_skill_delta` = relative perplexity degradation. This is the real e_S:
    retrieval noise must not hurt the model.

Verdict (amended spec): pass = skill delta <= e_S AND strict recall ON - strict
recall OFF >= d AND strict recall OFF <= floor AND invalid citation <= 1% AND
claims on the negative control < 10%. `persistent` / `claimable` as in the organ
protocol: a claim needs the journal on disk, durable, and session B reopened
from the disk alone -- the strict version runs session B in another process
(`tests/test_c2b_lm_bridge.py`). Thresholds d, e_S, floor untouched.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from cortex_c2b import Journal, MODE_DURABLE, POLICY_STOP, lifecycle_declaration
from cortex_c2b.hm_protocol import (Fact, generate_facts, SCHEMAS, _ATTRS, DELTA, EPSILON_S,
                                    FLOOR_MARGIN, NEGCTRL_INVALID, N_FACTS, N_NEGCTRL)
from cortex_c2b.lm_bridge import (JournalBridge, encode_query, parse_contract, NEWLINE, TOK_EPI)
from cortex_c2b.write_path import WritePath
from cortex_c2b.read_path import JournalPath

INVALID_CITATION_MAX = 0.01      # declared 2026-09-12, before any run on the model


@torch.no_grad()
def generate(model, prefix: list[int], window_tokens=None, window_mask=None, gate: float = 1.0,
             max_new: int = 48) -> tuple[list[int], float | None]:
    """Greedy byte-level decoding until a newline. Returns (new tokens, the
    reader's attention mass at the first decoding step)."""
    device = next(model.parameters()).device
    idx = torch.tensor([prefix], dtype=torch.long, device=device)
    out, mass = [], None
    for step in range(max_new):
        logits, _ = model(idx[:, -model.cfg.block_size:], window_tokens=window_tokens,
                          window_mask=window_mask, gate=gate)
        if step == 0:
            mass = model.read_mass()
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        if nxt == NEWLINE:
            break
        idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)
    return out, mass


def _probe(model, bridge: JournalBridge, facts: list[Fact], journal_on: bool, window_len: int,
           pointer_of: dict[str, str]) -> dict:
    """One arm over a fact list. `pointer_of`: entity -> pointer for planted facts
    (empty for the negative control)."""
    n = max(1, len(facts))
    strict = abstain = invalid = valid = attr_ok = guess = attr_by_chance = hit = 0
    masses = []
    for f in facts:
        query = encode_query(f.query)
        if journal_on:
            cue = bridge.cues([query])[0]
            read = bridge.read(cue)
            wt, wm = bridge.window_tensors([read.window], window_len)
            toks, mass = generate(model, query, wt, wm, gate=1.0)
            if mass is not None:
                masses.append(mass)
            hit += int(pointer_of.get(f.entity) in read.pointers) if pointer_of else 0
            labels = read.labels
        else:
            toks, _ = generate(model, query, None, None, gate=0.0)
            labels = {}
        kind, label, attr = parse_contract(toks)
        if kind == "unknown":
            abstain += 1
            continue
        guess += 1
        if kind == "cite" and label in labels:
            payload = bridge.j.payloads.get(labels[label]) if labels[label] in bridge.j.payloads else b""
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
            "attr_hit_any": attr_by_chance / n,
            "retrieval_hit": (hit / n) if (journal_on and pointer_of) else None,
            "attention_mass": (float(np.mean(masses)) if masses else None)}


@torch.no_grad()
def skill_delta(model, bridge: JournalBridge, val_batches: list[tuple[torch.Tensor, torch.Tensor]],
                window_len: int) -> tuple[float, float, float]:
    """Arm S: val loss with the reader active on whatever the journal returns for
    each span (ON) against the reader absent (OFF). Returns (delta, ppl_on, ppl_off)."""
    on, off = [], []
    for x, y in val_batches:
        _, l_off = model(x, y, gate=0.0)
        cues = bridge.cues([row.tolist() for row in x])
        windows = [bridge.read(c).window for c in cues]
        wt, wm = bridge.window_tensors(windows, window_len)
        _, l_on = model(x, y, window_tokens=wt, window_mask=wm, gate=1.0)
        on.append(float(l_on)); off.append(float(l_off))
    ppl_on, ppl_off = math.exp(float(np.mean(on))), math.exp(float(np.mean(off)))
    return max(0.0, (ppl_on - ppl_off) / ppl_off), ppl_on, ppl_off


def run_hm_lm(model, journal_factory, n_facts: int = N_FACTS, n_negctrl: int = N_NEGCTRL, seed: int = 0,
              k: int = 4, budget: int = 256, window_len: int = 256, val_batches=None,
              on_disk: bool = False, reopen: bool = True) -> dict:
    """`journal_factory()` returns the run's journal (a fresh one on the first
    call; the same on-disk scope reopened on the second). Session A plants the
    frozen protocol's facts through the model's cues; session B reopens the
    journal (a new object, the index recomputed) and probes."""
    facts, gen_hash = generate_facts(n_facts, seed)
    neg, _ = generate_facts(n_negctrl, seed + 10_000)
    chance = float(np.mean([1.0 / len(_ATTRS[s]) for s in SCHEMAS]))
    floor = chance + FLOOR_MARGIN
    device = next(model.parameters()).device
    model.eval()

    # --- session A: plant through the NORMAL gated path, cues from the model ---
    journal = journal_factory()
    bridge = JournalBridge(model, journal, k=k, budget_bytes=budget, seed=seed, device=device)
    admitted = 0
    for i, f in enumerate(facts):
        admitted += int(bridge.write(f.statement, f.schema, now=float(i), entity=f.entity).admitted)
    pointer_of = dict(bridge.pointer_of)
    durable_a = journal.mode == MODE_DURABLE

    # --- session B: nothing of A but the journal (reopened when on disk) ---
    if on_disk and reopen:
        del bridge
        journal = journal_factory()
    bridge = JournalBridge(model, journal, k=k, budget_bytes=budget, seed=seed, shuffle_seed=seed + 1, device=device)
    storage = lifecycle_declaration(journal)["storage"]
    persistent = bool(on_disk and reopen and durable_a and journal.mode == MODE_DURABLE)

    on = _probe(model, bridge, facts, True, window_len, pointer_of)
    off = _probe(model, bridge, facts, False, window_len, pointer_of)
    neg_on = _probe(model, bridge, neg, True, window_len, {})
    neg_off = _probe(model, bridge, neg, False, window_len, {})
    delta, ppl_on, ppl_off = skill_delta(model, bridge, val_batches, window_len) if val_batches else (0.0, None, None)

    negctrl_claims = neg_on["guess_rate"]                     # a claim on a never-planted entity
    valid = negctrl_claims < NEGCTRL_INVALID
    dissociation = (delta <= EPSILON_S and (on["recall_strict"] - off["recall_strict"]) >= DELTA
                    and off["recall_strict"] <= floor and on["invalid_citation_rate"] <= INVALID_CITATION_MAX
                    and valid)
    return {
        "benchmark": "hm_protocol_lm",
        "spec": "docs/benchmarks/hm-protocol.md (frozen thresholds; LM arm amendment 2026-09-12)",
        "thresholds": {"delta": DELTA, "epsilon_s": EPSILON_S, "floor": floor, "chance": chance,
                       "negctrl_invalid": NEGCTRL_INVALID, "invalid_citation_max": INVALID_CITATION_MAX},
        "generator": {"config_hash": gen_hash, "n_facts": n_facts, "n_negctrl": n_negctrl, "seed": seed,
                      "admitted": admitted},
        "hm_recall_on": on["recall_strict"], "hm_recall_off": off["recall_strict"],
        "hm_gap": on["recall_strict"] - off["recall_strict"],
        "hm_skill_delta": delta, "ppl_on": ppl_on, "ppl_off": ppl_off,
        "hm_negctrl_rate": negctrl_claims, "hm_negctrl_on": neg_on["attr_hit_any"],
        "hm_negctrl_abstain_on": neg_on["abstain_rate"], "hm_negctrl_abstain_off": neg_off["abstain_rate"],
        "hm_false_abstention_on": on["abstain_rate"], "hm_invalid_citation_on": on["invalid_citation_rate"],
        "hm_valid_citation_on": on["valid_citation_rate"], "hm_attr_exact_given_valid": on["attr_exact_given_valid"],
        "hm_retrieval_hit": on["retrieval_hit"], "hm_attention_mass": on["attention_mass"],
        "hm_guess_rate_off": off["guess_rate"], "hm_attr_hit_off_by_chance": off["attr_hit_any"],
        "run_valid": valid, "hm_dissociation_pass": int(dissociation),
        "persistent": persistent, "claimable": int(dissociation and persistent),
        "storage": {"on_disk": on_disk, "mode": storage["mode"], "policy": storage["policy"],
                    "plaintext_scope": storage["plaintext_scope"]},
        "attribution": {"encoder": on["retrieval_hit"], "reader": on["attention_mass"],
                        "answer": on["attr_exact_given_valid"]},
        "verdict": (("PASS" if dissociation else ("INVALID" if not valid else "FAIL"))
                    + " -- LM arm, cite-or-abstain contract"
                    + ("" if persistent else " -- persistent: false, not claimable")),
    }
