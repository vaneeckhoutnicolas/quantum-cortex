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
from cortex_c2b.lm_bridge import (JournalBridge, encode_query, parse_contract, build_read_window, NEWLINE, TOK_EPI,
                                  TOK_CITE, TOK_ANS, TOK_UNKNOWN, LABELS)
from cortex_c2b.write_path import WritePath
from cortex_c2b.read_path import JournalPath, STATE_EVICTED

INVALID_CITATION_MAX = 0.01      # declared 2026-09-12, before any run on the model


POLICIES = ("plain", "head", "head+pointer", "mark-veto", "mark-veto+value")


def policy_outcome(kind: str, label, labels: dict, marks_by_pointer: dict, payloads, own_pointer, attr: str | None,
                   expected_attr: str, policy: str, threshold: float | None):
    """ADR-008 amendment 2026-09-18, the two organ side decode policies, one function used
    live (in the probe) and in replay (on an answers file), so the two cannot disagree.
    `mark-veto`: the model's citation stands unless the cited line's mark is below the
    oracle's threshold (or the label is not in the window); then it is an abstention.
    `mark-veto+value`: the veto, and the attribute answered is the cited line's own
    value, read by the organ from the line's statement by the generator's templates,
    never generated; a vetoed or missing line is an abstention.
    Returns (kind, attr, outcome) with outcome in strict | valid | invalid | abstain."""
    from cortex_c2b.hm_protocol import value_of_statement
    if kind == "unknown":
        return kind, None, "abstain"
    if policy in ("mark-veto", "mark-veto+value"):
        pointer = labels.get(label) if kind == "cite" else None
        mark = marks_by_pointer.get(pointer) if pointer is not None else None
        if pointer is None or mark is None or (threshold is not None and mark < threshold):
            return "unknown", None, "abstain"                       # the organ vetoes: no line, or a low mark
        if policy == "mark-veto+value":
            payload = payloads.get(pointer) if pointer in payloads else b""
            v = value_of_statement(payload.decode("utf-8", errors="replace")) if payload else None
            attr = v[2] if v else None
            if attr is None:
                return "unknown", None, "abstain"
    if kind == "cite" and label in labels:
        payload = payloads.get(labels[label]) if labels[label] in payloads else b""
        cited_ok = (attr or "").encode("utf-8") in payload and len(attr or "") > 0
        if not cited_ok:
            return kind, attr, "invalid"
        return kind, attr, ("strict" if attr == expected_attr else "valid")
    return kind, attr, "invalid"


def pointer_label(model, window_bytes: bytes) -> int | None:
    """The pointer policy's citation: the label of the window line that received the
    most of the reader's attention from the decision position (heads averaged, the
    null slot excluded), read from the reader's last attention map. None when the
    window has no labelled line."""
    attn = None
    for blk in model.blocks:
        r = getattr(blk, "reader", None)
        if r is not None and getattr(r, "last_attn", None) is not None:
            attn = r.last_attn[0, :, -1, 1:].mean(dim=0)         # (r,): the decision position, real bytes only
    if attn is None:
        return None
    best, best_mass = None, -1.0
    start = 0
    n = len(window_bytes)
    while start < n:
        end = window_bytes.find(bytes([NEWLINE]), start)
        end = n if end < 0 else end
        label = window_bytes[start] if end > start else None
        if label in LABELS:
            stop = min(end + 1, attn.numel())
            m = float(attn[start:stop].sum()) if stop > start else -1.0
            if m > best_mass:
                best, best_mass = label, m
        start = end + 1
    return best


@torch.no_grad()
def generate(model, prefix: list[int], window_tokens=None, window_mask=None, gate: float = 1.0,
             max_new: int = 48, policy: str = "plain", window_bytes: bytes | None = None) -> tuple[list[int], float | None]:
    """Greedy byte-level decoding until a newline. Returns (new tokens, the
    reader's attention mass at the first decoding step).

    `policy` (ADR-008 amendment of 2026-09-14, declared before any measurement):
      plain        -- every token from the model's argmax, the head at most a bias (v7 as trained);
      head         -- the matching head decides: m <= 0 emits the abstention, m > 0 forces <CITE>
                      and the model generates the label and the attribute;
      head+pointer -- the head decides, and the label is read off the reader's attention
                      (the line with the most attention from the decision position); the
                      model generates only the attribute.
    Without a window, or without a head, every policy is the plain one."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; one of {POLICIES}")
    device = next(model.parameters()).device
    idx = torch.tensor([prefix], dtype=torch.long, device=device)
    out, mass, match = [], None, None
    has_head = getattr(model, "match_head", None) is not None
    forced: list[int] = []
    for step in range(max_new):
        # RES-21: the matching head fires at the decision position only, i.e. on the first step
        dpos = (torch.tensor([idx.size(1) - 1], device=device)
                if (step == 0 and has_head and window_tokens is not None) else None)
        logits, _ = model(idx[:, -model.cfg.block_size:], window_tokens=window_tokens,
                          window_mask=window_mask, gate=gate, decision_pos=dpos)
        if step == 0:
            mass = model.read_mass()
            match = getattr(model, "last_match_logit", None)   # the head's judgment at the decision step
            if policy != "plain" and has_head and window_tokens is not None and match is not None:
                if float(match[0]) <= 0.0:
                    out = [TOK_UNKNOWN, NEWLINE]
                    break
                forced = [TOK_CITE]
                if policy == "head+pointer":
                    label = pointer_label(model, window_bytes or b"")
                    if label is not None:
                        forced += [label, TOK_ANS]
        if forced:
            nxt = forced.pop(0)
        else:
            nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        if nxt == NEWLINE:
            break
        idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)
    if has_head:
        model.last_match_logit = match                          # later steps reset it; the probe reads the decision step's
    return out, mass


def _probe(model, bridge: JournalBridge, facts: list[Fact], journal_on: bool, window_len: int,
           pointer_of: dict[str, str], policy: str = "plain", veto_threshold: float | None = None) -> dict:
    """One arm over a fact list. `pointer_of`: entity -> pointer for planted facts
    (empty for the negative control)."""
    n = max(1, len(facts))
    strict = abstain = invalid = valid = attr_ok = guess = attr_by_chance = hit = 0
    masses = []
    match_right = match_n = 0                                   # RES-21: the head's own judgment, against the read
    mark_own, mark_other_max, mark_max = [], [], []             # RES-23: the organ's similarities per read
    answers = []                                                # Rev66: every answer kept per line, a recording, not a variable
    for f in facts:
        query = encode_query(f.query)
        read = None
        if journal_on:
            cue = bridge.cues([query])[0]
            read = bridge.read(cue)
            if read.scores:
                own_p = pointer_of.get(f.entity) if pointer_of else None
                others = [sc for pt, sc in zip(read.pointers, read.scores) if pt != own_p]
                if own_p in read.pointers:
                    mark_own.append(float(read.scores[read.pointers.index(own_p)]))
                if others:
                    mark_other_max.append(float(max(others)))
                mark_max.append(float(max(read.scores)))
            wt, wm = bridge.window_tensors([read.window], window_len)
            toks, mass = generate(model, query, wt, wm, gate=1.0, policy=policy, window_bytes=read.window)
            if mass is not None:
                masses.append(mass)
            present = (pointer_of.get(f.entity) in read.pointers) if pointer_of else False
            hit += int(present) if pointer_of else 0
            m = getattr(model, "last_match_logit", None)
            if m is not None:
                match_right += int((float(m[0]) > 0) == present); match_n += 1
            labels = read.labels
        else:
            toks, _ = generate(model, query, None, None, gate=0.0)
            labels = {}
        kind, label, attr = parse_contract(toks)
        own_p = pointer_of.get(f.entity) if pointer_of else None
        mbp = dict(zip(read.pointers, [float(x) for x in read.scores])) if (read is not None and read.scores) else {}
        if policy in ("mark-veto", "mark-veto+value"):
            kind, attr, _ = policy_outcome(kind, label, labels, mbp, bridge.j.payloads, own_p, attr, f.attr, policy, veto_threshold)
        rec = {"entity": f.entity, "schema": f.schema, "attr": f.attr, "kind": kind,
               "label": (chr(label) if isinstance(label, int) else label) if label is not None else None, "answered": attr,
               "outcome": None, "own_in_window": (own_p in read.pointers) if (read is not None and own_p) else None}
        if read is not None and read.scores:
            marks_by_pointer = dict(zip(read.pointers, [float(x) for x in read.scores]))
            rec["mark_own"] = marks_by_pointer.get(own_p) if own_p else None
            rec["mark_max"] = max(marks_by_pointer.values())
            rec["cited_pointer_is_own"] = (labels.get(label) == own_p) if (kind == "cite" and label in labels and own_p) else None
            rec["mark_cited"] = marks_by_pointer.get(labels[label]) if (kind == "cite" and label in labels) else None
            rec["cited_is_highest_mark"] = (labels.get(label) is not None and marks_by_pointer.get(labels[label]) == rec["mark_max"]) if (kind == "cite" and label in labels) else None
        answers.append(rec)
        if kind == "unknown":
            abstain += 1; rec["outcome"] = "abstain"
            continue
        guess += 1
        if kind == "cite" and label in labels:
            payload = bridge.j.payloads.get(labels[label]) if labels[label] in bridge.j.payloads else b""
            cited_ok = (attr or "").encode("utf-8") in payload and len(attr or "") > 0
            if cited_ok:
                valid += 1; rec["outcome"] = "valid"
                if attr == f.attr:
                    strict += 1; attr_ok += 1; rec["outcome"] = "strict"
            else:
                invalid += 1; rec["outcome"] = "invalid"       # a label that exists but does not hold the claim
        else:
            invalid += 1; rec["outcome"] = "invalid"           # a citation of nothing, or a malformed claim
        if attr == f.attr:
            attr_by_chance += 1
    return {"recall_strict": strict / n, "abstain_rate": abstain / n, "guess_rate": guess / n,
            "invalid_citation_rate": invalid / n, "valid_citation_rate": valid / n,
            "attr_exact_given_valid": (attr_ok / valid) if valid else None,
            "attr_hit_any": attr_by_chance / n,
            "retrieval_hit": (hit / n) if (journal_on and pointer_of) else None,
            "attention_mass": (float(np.mean(masses)) if masses else None),
            "match_acc": (match_right / match_n) if match_n else None,
            "marks": {"own": _stats(mark_own), "other_max": _stats(mark_other_max), "max": _stats(mark_max)},
            "answers": answers}


@torch.no_grad()
def retrieval_probe(model, journal, facts: list[Fact], neg: list[Fact], pointer_of: dict[str, str],
                    k: int = 4, seed: int = 0) -> dict:
    """ADR-008 amendment 2026-09-19, the retrieval policy probe: the first factor of the
    decomposition (retrieval) read on its own, with no generation and no training, on a
    checkpoint and a sealed journal that already exist. For each protocol question the
    query cue is the model's own; the same entries are indexed twice, ranked by the dense
    cosine of every run to date and by the expand and sparsify tag of ADR-003 line 6
    (the dentate gyrus in the map, the mushroom body in Dasgupta, Stevens and Navlakha
    2017). Reports, per ranking: whether the own episode is among the candidates the
    buckets return at all, and whether it is in the top k. A miss in the first is the
    bucketing's, a miss in the second only is the ranking's; the two are never mixed."""
    from cortex_c2b.read_path import CueIndex, JournalPath
    device = next(model.parameters()).device
    out = {"probe": "retrieval policy (ADR-003 line 6): dense cosine against expand and sparsify",
           "k": k, "n_facts": len(facts), "n_negctrl": len(neg), "rankings": {}}
    for rank in ("cosine", "fly"):
        jp = JournalPath(journal, index=CueIndex(seed=seed, rank=rank), seed=seed)
        for eid, e in journal._entries.items():
            if e.state != STATE_EVICTED:
                jp.index.add(eid, np.asarray(e.cue, dtype=np.float32))
        bridge = JournalBridge(model, journal, journal_path=jp, k=k, seed=seed, device=device)
        in_cand = in_topk = 0; ranks = []; margins = []; neg_best = []
        for f in facts:
            cue = bridge.cues([encode_query(f.query)])[0]
            own = pointer_of.get(f.entity)
            scored = jp.index.query(np.asarray(cue, dtype=np.float32), k=len(jp.index))
            by_pointer = []
            for eid, sc in scored:
                e, pointer = jp._resolve(eid)
                by_pointer.append((pointer, sc))
            pos = next((i for i, (pt, _) in enumerate(by_pointer) if pt == own), None)
            if pos is None:
                continue
            in_cand += 1; ranks.append(pos + 1)
            if pos < k:
                in_topk += 1
            others = [sc for i, (pt, sc) in enumerate(by_pointer) if pt != own]
            margins.append(float(by_pointer[pos][1] - max(others))) if others else None
        for f in neg:
            cue = bridge.cues([encode_query(f.query)])[0]
            sc = jp.index.query(np.asarray(cue, dtype=np.float32), k=1)
            neg_best.append(float(sc[0][1]) if sc else float("nan"))
        n = len(facts)
        out["rankings"][rank] = {
            "own_in_candidates": in_cand / n, "own_in_top_k": in_topk / n,
            "own_rank": _stats([float(r) for r in ranks]), "own_minus_best_other": _stats(margins),
            "negctrl_best_score": _stats([x for x in neg_best if x == x])}
    c, f_ = out["rankings"]["cosine"], out["rankings"]["fly"]
    out["reading"] = {"top_k_cosine": c["own_in_top_k"], "top_k_fly": f_["own_in_top_k"],
                      "candidate_misses_cosine": round(1 - c["own_in_candidates"], 4),
                      "candidate_misses_fly": round(1 - f_["own_in_candidates"], 4)}
    return out


def replay_policy(answers_file: dict, policy: str, threshold: float | None = None) -> dict:
    """Post hoc: apply an organ side policy to the answers a session B kept (Rev66 files), with
    the oracle's threshold of that file. The rates it returns are what the live policy would
    have produced on the same checkpoint and reads, because both use `policy_outcome`;
    a replay on the seed the policy was conceived on is a post hoc reading, never a result."""
    thr = answers_file["hm_mark_oracle"]["threshold"] if threshold is None else float(threshold)   # a sweep reads the sensitivity
    def outcome(a):
        kind, label = a["kind"], a.get("label")
        labels = {label: "cited"} if (kind == "cite" and a.get("mark_cited") is not None) else {}
        mbp = {"cited": a["mark_cited"]} if a.get("mark_cited") is not None else {}
        own = "cited" if a.get("cited_pointer_is_own") else "own"
        # the value read from the cited line is the fact's attribute iff the cited line is the own one
        payloads = {"cited": (f"x took place in {a['attr']}" if a.get("cited_pointer_is_own") else "x took place in other").encode()}
        if policy == "mark-veto+value":
            k, attr, o = policy_outcome(kind, label, labels, mbp, payloads, own, a.get("answered"), a["attr"], "mark-veto", thr)
            if o == "abstain":
                return "abstain"
            return "strict" if a.get("cited_pointer_is_own") else "invalid"
        k, attr, o = policy_outcome(kind, label, labels, mbp, {"cited": ((a.get("answered") or "") if a["outcome"] in ("strict", "valid") else "").encode()}, own, a.get("answered"), a["attr"], policy, thr)
        if policy == "mark-veto" and o != "abstain":
            return a["outcome"]                                    # the model's own outcome stands when not vetoed
        return o
    on = [outcome(a) for a in answers_file["answers"]["on"]]; n = len(on)
    neg = [outcome(a) for a in answers_file["answers"]["negctrl"]]
    strict, valid, invalid, abst = on.count("strict"), on.count("strict") + on.count("valid"), on.count("invalid"), on.count("abstain")
    negc = sum(1 for o in neg if o != "abstain")
    return {"policy": policy, "post_hoc": True, "threshold": thr, "hm_recall_on": strict / n, "hm_valid_citation_on": valid / n,
            "hm_invalid_citation_on": invalid / n, "hm_false_abstention_on": abst / n, "hm_negctrl_rate": negc / len(neg),
            "gate_conditions_without_skill_arm": int(strict / n >= DELTA and negc / len(neg) < NEGCTRL_INVALID and invalid / n <= INVALID_CITATION_MAX)}


def _stats(xs: list[float]) -> dict | None:
    if not xs:
        return None
    a = np.asarray(xs, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
            "q10": float(np.quantile(a, 0.10)), "q50": float(np.quantile(a, 0.50)), "q90": float(np.quantile(a, 0.90))}


@torch.no_grad()
def mark_oracle(model, bridge: JournalBridge, facts: list[Fact], neg: list[Fact], pointer_of: dict[str, str],
                train_seed: int = 100_000, train_n: int = 200, k: int | None = None) -> dict:
    """RES-23, declared before v9 ran: the ceiling the organ's marks allow, with no
    model output. A rule: cite the line with the highest mark when that mark clears
    a threshold, else abstain. The threshold is read from the training facts family
    only (a scratch journal of `train_n` training facts planted with this model's
    cues, seed `train_seed`, and their never planted negatives), never from the
    protocol's facts; then the rule is applied to the protocol's own reads. Strict
    recall counts the own episode cited; a claim on a never planted entity counts
    against; the skill arm does not apply (no model). Deterministic from the
    checkpoint and the sealed journal."""
    from cortex_c2b.organ_use import training_facts, plant_pool
    k = k or bridge.k
    tf, tneg, _ = training_facts(train_n, train_seed)
    scratch = JournalBridge(model, Journal(), k=k, budget_bytes=bridge.budget, seed=0, shuffle_seed=1,
                            device=bridge.device, mark=bridge.mark)
    plant_pool(scratch, tf)
    pos, negs = [], []
    for f in tf:
        r = scratch.read(scratch.cues([encode_query(f.query)])[0])
        own = scratch.pointer_of.get(f.entity)
        if r.scores and own in r.pointers:
            pos.append((max(r.scores), r.pointers[int(np.argmax(r.scores))] == own))
        elif r.scores:
            pos.append((max(r.scores), False))
    for f in tneg:
        r = scratch.read(scratch.cues([encode_query(f.query)])[0])
        negs.append(max(r.scores) if r.scores else 0.0)
    cands = sorted({m for m, _ in pos} | set(negs))
    best_thr, best_acc = 0.0, -1.0
    for thr in cands:                                           # claim iff max mark > thr; the best on the training family
        acc = (sum(1 for m, right in pos if m > thr and right) + sum(1 for m in negs if m <= thr)) / max(1, len(pos) + len(negs))
        if acc > best_acc:
            best_thr, best_acc = float(thr), float(acc)
    def apply(fs, planted: bool):
        claims = strict = 0
        for f in fs:
            r = bridge.read(bridge.cues([encode_query(f.query)])[0])
            if not r.scores:
                continue
            top = int(np.argmax(r.scores))
            if r.scores[top] > best_thr:
                claims += 1
                if planted and r.pointers[top] == pointer_of.get(f.entity):
                    strict += 1
        return claims / max(1, len(fs)), strict / max(1, len(fs))
    on_claims, on_strict = apply(facts, True)
    neg_claims, _ = apply(neg, False)
    gap = on_strict - 0.0                                       # off is zero by construction: no window, no citation
    passes = (gap >= DELTA) and (neg_claims < NEGCTRL_INVALID)
    return {"oracle": "mark oracle (RES-23): cite the highest marked line above a threshold read from the training family",
            "threshold": best_thr, "training_accuracy": best_acc, "training_pairs": len(pos) + len(negs),
            "recall_on_strict": on_strict, "claims_on": on_claims, "negctrl_claims": neg_claims,
            "gap": gap, "passes_without_skill_arm": bool(passes), "encoder": "this checkpoint's own cue encoder"}


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
              on_disk: bool = False, reopen: bool = True, mark: bool = False) -> dict:
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
    bridge = JournalBridge(model, journal, k=k, budget_bytes=budget, seed=seed, device=device, mark=mark)
    admitted = 0
    for i, f in enumerate(facts):
        admitted += int(bridge.write(f.statement, f.schema, now=float(i), entity=f.entity).admitted)
    pointer_of = dict(bridge.pointer_of)
    durable_a = journal.mode == MODE_DURABLE

    # --- session B: nothing of A but the journal (reopened when on disk) ---
    if on_disk and reopen:
        del bridge
        journal = journal_factory()
    bridge = JournalBridge(model, journal, k=k, budget_bytes=budget, seed=seed, shuffle_seed=seed + 1, device=device,
                           mark=mark)
    storage = lifecycle_declaration(journal)["storage"]
    persistent = bool(on_disk and reopen and durable_a and journal.mode == MODE_DURABLE)

    on = _probe(model, bridge, facts, True, window_len, pointer_of)
    off = _probe(model, bridge, facts, False, window_len, pointer_of)
    neg_on = _probe(model, bridge, neg, True, window_len, {})
    neg_off = _probe(model, bridge, neg, False, window_len, {})
    delta, ppl_on, ppl_off = skill_delta(model, bridge, val_batches, window_len) if val_batches else (0.0, None, None)
    oracle = mark_oracle(model, bridge, facts, neg, pointer_of)

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
        "hm_marks": {"on": on["marks"], "negctrl": neg_on["marks"], "written_in_window": bool(mark)},
        "hm_mark_oracle": oracle,
        "hm_guess_rate_off": off["guess_rate"], "hm_attr_hit_off_by_chance": off["attr_hit_any"],
        "hm_match_acc_on": on["match_acc"], "hm_match_acc_negctrl": neg_on["match_acc"],   # RES-21 diagnostics (None without the head)
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


# ---------------------------------------------------------------------------- #
# Session B in a NEW process: the model from its checkpoint, the journal from    #
# the sealed disk, nothing else (ADR-007 D8, the named boundary on the model)    #
# ---------------------------------------------------------------------------- #
@torch.no_grad()
def shape_probe(model, bridge: JournalBridge, facts: list[Fact], neg: list[Fact], window_len: int,
                pointer_of: dict[str, str], seed: int = 0) -> dict:
    """A diagnostic outside the frozen protocol (like the memorisation probe): does the
    cite or abstain decision follow the NUMBER OF LINES in the window? Run v5
    (`db028e6a4262`) trained on paired negatives whose withheld window had k - 1 lines,
    while every protocol window has k. Four claim rates, greedy decoding, gate on:
      planted_withheld_short : the planted fact, its own episode withheld, k - 1 lines (the v5 negative as trained)
      planted_withheld_kept  : the same, k lines (the k + 1 th match fills the slot: the v6 negative)
      absent_k_lines         : a never planted entity, k lines (the protocol's negative control)
      absent_short           : the same, the best match dropped, k - 1 lines
    A decision that reads the window claims alike at k - 1 and k lines; a decision that
    counts lines claims at k and abstains at k - 1."""
    rng = np.random.default_rng(seed + 3)
    model.eval()

    def claims(items, query):
        window, labels = build_read_window(items, bridge.budget, rng)
        wt, wm = bridge.window_tensors([window], window_len)
        toks, _ = generate(model, query, wt, wm, gate=1.0)
        kind, _, _ = parse_contract(toks)
        return int(kind != "unknown"), len(labels)

    from cortex_c2b.organ_use import withheld_items
    out = {"planted_withheld_short": [0, 0], "planted_withheld_kept": [0, 0],
           "absent_k_lines": [0, 0], "absent_short": [0, 0]}
    lines = {k: [] for k in out}
    own_retrieved = 0
    bridge.pointer_of.update(pointer_of)                                # the fill of the kept shape draws among the planted
    for f in facts:
        own = pointer_of.get(f.entity)
        if own is None:
            continue
        query = encode_query(f.query)
        cue = bridge.cues([query])[0]
        if own not in bridge.read(cue).pointers:
            continue                                                   # only the pairs whose halves differ in shape
        own_retrieved += 1
        short = withheld_items(bridge, cue, own, rng, keep_shape=False)  # as v5 built it
        kept = withheld_items(bridge, cue, own, rng, keep_shape=True)    # as v6 builds it
        for name, items in (("planted_withheld_short", short), ("planted_withheld_kept", kept)):
            c, n_lines = claims(items, query)
            out[name][0] += c; out[name][1] += 1; lines[name].append(n_lines)
    for f in neg:
        query = encode_query(f.query)
        cue = bridge.cues([query])[0]
        hits = bridge.jp.retrieve(cue, k=bridge.k)
        top = [(e.pointer, payload) for e, payload, _ in hits]
        for name, items in (("absent_k_lines", top), ("absent_short", top[1:])):
            c, n_lines = claims(items, query)
            out[name][0] += c; out[name][1] += 1; lines[name].append(n_lines)
    return {"probe": "shape (lines in the window) -- a diagnostic, not a protocol measure",
            "k": bridge.k, "pairs_with_own_retrieved": own_retrieved,
            **{name: {"claim_rate": (c / n if n else None), "n": n,
                      "lines_mean": (float(np.mean(lines[name])) if lines[name] else None)}
               for name, (c, n) in out.items()}}


@torch.no_grad()
def _head_inputs(model, bridge: JournalBridge, window, window_len: int, query: list[int]):
    """The two vectors the matching head receives at the decision position, for one query
    and one window: the residual stream before the reader and the reader's attended values."""
    wt, wm = bridge.window_tensors([window], window_len)
    idx = torch.tensor([query], dtype=torch.long, device=next(model.parameters()).device)
    model(idx[:, -model.cfg.block_size:], window_tokens=wt, window_mask=wm, gate=1.0)
    blk = model.blocks[model.cfg.journal_block]
    pre, att = blk.last_pre[0, -1], blk.last_attended[0, -1]
    return torch.cat([pre, att]).float().cpu().numpy()


def representation_probe(model, bridge: JournalBridge, facts: list[Fact], window_len: int,
                         pointer_of: dict[str, str], seed: int = 0, steps: int = 400, lr: float = 0.5) -> dict:
    """A post hoc linear probe (logistic regression, numpy) on the two vectors the matching
    head receives, over minimal pairs of the given facts: the plain read (present, when the
    own episode was retrieved) against the same read with the own episode withheld and the
    shape kept (absent). Half the pairs train the probe, the other half test it. Reads: a
    test accuracy well above 0.5 means the comparison is linearly extractable from those
    vectors and the trained head's constant answer is a fault of the coupled training; an
    accuracy near 0.5 means the reader's representation does not carry it."""
    from cortex_c2b.organ_use import withheld_items
    rng = np.random.default_rng(seed + 5)
    model.eval()
    bridge.pointer_of.update(pointer_of)
    X, y = [], []
    for f in facts:
        own = pointer_of.get(f.entity)
        if own is None:
            continue
        query = encode_query(f.query)
        cue = bridge.cues([query])[0]
        read = bridge.read(cue)
        if own not in read.pointers:
            continue
        X.append(_head_inputs(model, bridge, read.window, window_len, query)); y.append(1.0)
        items = withheld_items(bridge, cue, own, rng, keep_shape=True)
        window, _ = build_read_window(items, bridge.budget, rng)
        X.append(_head_inputs(model, bridge, window, window_len, query)); y.append(0.0)
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 8:
        return {"probe": "representation (linear, post hoc)", "n_pairs": n // 2, "note": "too few pairs"}
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    order = rng.permutation(n // 2)                                   # split by pair, never by example
    train_pairs, test_pairs = order[: len(order) // 2], order[len(order) // 2:]
    tr = np.concatenate([[2 * i, 2 * i + 1] for i in train_pairs]); te = np.concatenate([[2 * i, 2 * i + 1] for i in test_pairs])
    w = np.zeros(Xn.shape[1]); b = 0.0
    for _ in range(steps):                                            # plain gradient descent with a small ridge
        z = Xn[tr] @ w + b; p = 1.0 / (1.0 + np.exp(-z))
        g = p - y[tr]
        w -= lr * (Xn[tr].T @ g / len(tr) + 1e-3 * w); b -= lr * g.mean()
    def acc(ix):
        return float((((Xn[ix] @ w + b) > 0).astype(float) == y[ix]).mean())
    return {"probe": "representation (linear, post hoc): logistic regression on the head's two vectors over minimal pairs",
            "n_pairs": n // 2, "train_pairs": len(train_pairs), "test_pairs": len(test_pairs),
            "train_accuracy": acc(tr), "test_accuracy": acc(te), "chance": 0.5,
            "reading": ("the comparison is linearly extractable from the head's vectors: the trained head's constant is a fault of the coupled training"
                        if acc(te) >= 0.8 else
                        "the comparison is not linearly extractable from the head's vectors at this size: the reader's representation does not carry it"
                        if acc(te) <= 0.6 else "partially extractable; neither reading is clean")}


def representation_probe_from_disk(ckpt_path, journal_path, cfg, n_facts: int = N_FACTS, seed: int = 0,
                                   k: int = 4, budget: int = 256, window_len: int = 256) -> dict:
    import train
    from cortex_c2b import content_hash
    from cortex_c2b.crypto import key_from_env
    device = torch.device("cpu")
    model = train.VanillaGPT(cfg).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    journal = Journal(journal_path, key=key_from_env(), policy=POLICY_STOP)
    bridge = JournalBridge(model, journal, k=k, budget_bytes=budget, seed=seed, shuffle_seed=seed + 1, device=device)
    facts, gen_hash = generate_facts(n_facts, seed)
    pointer_of = {f.entity: content_hash(f.statement.encode("utf-8")) for f in facts
                  if content_hash(f.statement.encode("utf-8")) in journal.payloads}
    r = representation_probe(model, bridge, facts, window_len, pointer_of, seed=seed)
    r["checkpoint"] = {"path": str(ckpt_path), "run_id": ck.get("run_id"), "config_hash": ck.get("config_hash"), "step": ck.get("step")}
    r["generator"] = {"config_hash": gen_hash, "n_facts": n_facts, "seed": seed}
    return r


def shape_probe_from_disk(ckpt_path, journal_path, cfg, n_facts: int = N_FACTS, n_negctrl: int = N_NEGCTRL,
                          seed: int = 0, k: int = 4, budget: int = 256, window_len: int = 256) -> dict:
    """The shape probe from the disk alone (checkpoint, sealed journal, key in the environment)."""
    import train
    from cortex_c2b import content_hash
    from cortex_c2b.crypto import key_from_env
    device = torch.device("cpu")
    model = train.VanillaGPT(cfg).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    journal = Journal(journal_path, key=key_from_env(), policy=POLICY_STOP)
    bridge = JournalBridge(model, journal, k=k, budget_bytes=budget, seed=seed, shuffle_seed=seed + 1, device=device)
    facts, gen_hash = generate_facts(n_facts, seed)
    neg, _ = generate_facts(n_negctrl, seed + 10_000)
    pointer_of = {f.entity: content_hash(f.statement.encode("utf-8")) for f in facts
                  if content_hash(f.statement.encode("utf-8")) in journal.payloads}
    r = shape_probe(model, bridge, facts, neg, window_len, pointer_of, seed=seed)
    r["checkpoint"] = {"path": str(ckpt_path), "run_id": ck.get("run_id"), "config_hash": ck.get("config_hash"),
                       "step": ck.get("step")}
    r["generator"] = {"config_hash": gen_hash, "n_facts": n_facts, "n_negctrl": n_negctrl, "seed": seed}
    r["journal"] = {"path": str(journal_path), "entries": len(journal), "planted_found": len(pointer_of)}
    return r


def _verdict(on: dict, off: dict, neg_on: dict, floor: float) -> dict:
    """The four gate conditions a new process can evaluate without the skill arm (the skill
    delta is a property of the model on ordinary text and does not depend on the decode
    policy; the in process value stands): gap >= delta, off <= floor, invalid citations
    <= 0.01, claims on never planted entities < 0.10."""
    gap = on["recall_strict"] - off["recall_strict"]
    valid = neg_on["guess_rate"] < NEGCTRL_INVALID
    four = (gap >= DELTA and off["recall_strict"] <= floor and on["invalid_citation_rate"] <= INVALID_CITATION_MAX and valid)
    return {"hm_gap": gap, "run_valid": valid, "gate_conditions_without_skill_arm": int(four),
            "verdict_without_skill_arm": ("PASS" if four else ("INVALID" if not valid else "FAIL")) + " -- the skill delta is the in process arm's"}


def session_b_from_disk(ckpt_path, journal_path, cfg, n_facts: int = N_FACTS, n_negctrl: int = N_NEGCTRL,
                        seed: int = 0, k: int = 4, budget: int = 256, window_len: int = 256,
                        train_pool_seed: int | None = None, train_pool_n: int = 200, policy: str = "plain") -> dict:
    """Reopen everything from the disk alone and probe. The planted facts are found
    by content addressing (a statement's pointer is the hash of its bytes), so no
    state of session A is needed beyond the journal file and the checkpoint."""
    import train
    from cortex_c2b import content_hash
    from cortex_c2b.crypto import key_from_env
    device = torch.device("cpu")
    model = train.VanillaGPT(cfg).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    journal = Journal(journal_path, key=key_from_env(), policy=POLICY_STOP)
    mark = bool(getattr(cfg, "journal_familiarity_mark", False))
    bridge = JournalBridge(model, journal, k=k, budget_bytes=budget, seed=seed, shuffle_seed=seed + 1, device=device,
                           mark=mark)
    facts, gen_hash = generate_facts(n_facts, seed)
    neg, _ = generate_facts(n_negctrl, seed + 10_000)
    pointer_of = {f.entity: content_hash(f.statement.encode("utf-8")) for f in facts
                  if content_hash(f.statement.encode("utf-8")) in journal.payloads}
    thr = mark_oracle(model, bridge, facts, neg, pointer_of)["threshold"] if policy.startswith("mark-veto") else None
    on = _probe(model, bridge, facts, True, window_len, pointer_of, policy=policy, veto_threshold=thr)
    off = _probe(model, bridge, facts, False, window_len, pointer_of, policy=policy, veto_threshold=thr)
    neg_on = _probe(model, bridge, neg, True, window_len, {}, policy=policy, veto_threshold=thr)
    storage = lifecycle_declaration(journal)["storage"]
    # the memorisation probe (run dc34fcf000aa): the same contract on entities the model SAW during training
    # (the first training pool), planted into a memory scope; a model that reads its window scores alike on
    # seen and unseen entities, a model that memorised entity -> attribute scores far higher on the seen ones
    train_probe = None
    if train_pool_seed is not None:
        from cortex_c2b.organ_use import training_facts, plant_pool
        tf, _, _ = training_facts(train_pool_n, seed=train_pool_seed)
        jb2 = JournalBridge(model, Journal(), k=k, budget_bytes=budget, seed=seed, shuffle_seed=seed + 2, device=device)
        plant_pool(jb2, tf)
        tp = _probe(model, jb2, tf, True, window_len, dict(jb2.pointer_of), policy=policy)
        train_probe = {"seed": train_pool_seed, "n": len(tf), "recall_strict": tp["recall_strict"],
                       "valid_citation": tp["valid_citation_rate"], "invalid_citation": tp["invalid_citation_rate"],
                       "abstain_rate": tp["abstain_rate"], "retrieval_hit": tp["retrieval_hit"]}
    chance = float(np.mean([1.0 / len(_ATTRS[s]) for s in SCHEMAS])); floor = chance + FLOOR_MARGIN
    verdict_fields = _verdict(on, off, neg_on, floor)
    oracle = mark_oracle(model, bridge, facts, neg, pointer_of)
    return {"benchmark": "hm_protocol_lm_session_b_new_process", "policy": policy, **verdict_fields,
            "hm_marks": {"on": on["marks"], "negctrl": neg_on["marks"], "written_in_window": mark},
            "hm_mark_oracle": oracle,
            "answers": {"on": on["answers"], "off": off["answers"], "negctrl": neg_on["answers"]},
            "training_pool_probe": train_probe,
            "checkpoint": {"path": str(ckpt_path), "run_id": ck.get("run_id"), "config_hash": ck.get("config_hash"),
                           "step": ck.get("step")},
            "journal": {"path": str(journal_path), "entries": len(journal), "mode": storage["mode"],
                        "policy": storage["policy"], "planted_found": len(pointer_of)},
            "generator": {"config_hash": gen_hash, "n_facts": n_facts, "n_negctrl": n_negctrl, "seed": seed},
            "hm_recall_on": on["recall_strict"], "hm_recall_off": off["recall_strict"],
            "hm_false_abstention_on": on["abstain_rate"], "hm_invalid_citation_on": on["invalid_citation_rate"],
            "hm_valid_citation_on": on["valid_citation_rate"], "hm_attr_exact_given_valid": on["attr_exact_given_valid"],
            "hm_retrieval_hit": on["retrieval_hit"], "hm_attention_mass": on["attention_mass"],
            "hm_negctrl_rate": neg_on["guess_rate"], "hm_negctrl_abstain_on": neg_on["abstain_rate"],
            "hm_match_acc_on": on["match_acc"], "hm_match_acc_negctrl": neg_on["match_acc"],
            "persistent": storage["mode"] == MODE_DURABLE, "process": "new (nothing of session A but the disk)"}


def _cli():
    import argparse, json
    from pathlib import Path
    import train
    ap = argparse.ArgumentParser(description="the H.M. protocol's LM arm, session B in a new process")
    ap.add_argument("--session-b", action="store_true")
    ap.add_argument("--retrieval-probe", action="store_true", help="ADR-008 amendment 2026-09-19: the retrieval policy probe on a checkpoint and its sealed journal (no generation, no training)")
    ap.add_argument("--replay", default=None, help="post hoc: apply --policy to a session B answers file (Rev66) and print the rates; never a result on the seed the policy was conceived on")
    ap.add_argument("--threshold", type=float, default=None, help="with --replay: override the oracle's veto threshold (a sensitivity sweep)")
    ap.add_argument("--shape-probe", action="store_true",
                    help="the shape probe alone (claims at k - 1 against k lines), a diagnostic: metrics/mqar/shape-probe-<run_id>.json")
    ap.add_argument("--representation-probe", action="store_true",
                    help="a post hoc linear probe on the matching head's two vectors over minimal pairs: metrics/mqar/representation-probe-<run_id>.json")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--journal", required=True)
    ap.add_argument("--config", required=True, help="the run's config json (the model's shape)")
    ap.add_argument("--out", default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-facts", type=int, default=N_FACTS); ap.add_argument("--n-negctrl", type=int, default=N_NEGCTRL)
    ap.add_argument("--probe-training-pool", action="store_true",
                    help="also probe the contract on the run's FIRST training pool (entities seen in training): memorisation evidence")
    ap.add_argument("--policy", default="plain", choices=list(POLICIES),
                    help="decode policy: plain | head | head+pointer (ADR-008 amendment 2026-09-14) | mark-veto | mark-veto+value (amendment 2026-09-18)")
    args = ap.parse_args()
    if not (args.session_b or args.shape_probe or args.representation_probe or args.retrieval_probe):
        if args.replay:
            r = replay_policy(json.loads(Path(args.replay).read_text(encoding="utf-8")), args.policy, threshold=args.threshold)
            print(json.dumps(r, indent=2)); return
        ap.error("one of --session-b / --shape-probe / --representation-probe / --retrieval-probe / --replay is required")
    cfg = train.load_config(args.config)
    if args.retrieval_probe:                                  # ADR-008 amendment 2026-09-19 (RES-24)
        from cortex_c2b.crypto import key_from_env
        from cortex_c2b import content_hash
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = train.VanillaGPT(cfg).to(device)
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); model.eval()
        journal = Journal(args.journal, key=key_from_env(), policy=POLICY_STOP)
        facts, _ = generate_facts(args.n_facts, args.seed)
        neg, _ = generate_facts(args.n_negctrl, args.seed + 10_000)
        pointer_of = {f.entity: content_hash(f.statement.encode("utf-8")) for f in facts
                      if content_hash(f.statement.encode("utf-8")) in journal.payloads}
        r = retrieval_probe(model, journal, facts, neg, pointer_of, k=cfg.journal_k, seed=args.seed)
        r["checkpoint"] = {"path": str(args.ckpt), "run_id": ck.get("run_id"), "config_hash": ck.get("config_hash")}
        r["journal"] = {"path": str(args.journal), "entries": len(journal._entries)}
        out = Path(args.out) if args.out else Path("metrics/mqar") / f"retrieval-probe-{ck.get('run_id')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
        print(f"retained: {out}")
        print(json.dumps({"reading": r["reading"], "rankings": {k2: {kk: v2[kk] for kk in ("own_in_candidates", "own_in_top_k")} for k2, v2 in r["rankings"].items()}}, indent=2))
        return
    if args.representation_probe:
        r = representation_probe_from_disk(args.ckpt, args.journal, cfg, n_facts=args.n_facts, seed=args.seed,
                                           k=cfg.journal_k, budget=cfg.journal_read_bytes, window_len=cfg.journal_read_bytes)
        out = Path(args.out) if args.out else Path("metrics/mqar") / f"representation-probe-{r['checkpoint']['run_id']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
        print(f"retained: {out}")
        print(json.dumps(r, indent=2))
        return
    if args.shape_probe:
        r = shape_probe_from_disk(args.ckpt, args.journal, cfg, n_facts=args.n_facts, n_negctrl=args.n_negctrl, seed=args.seed,
                                  k=cfg.journal_k, budget=cfg.journal_read_bytes, window_len=cfg.journal_read_bytes)
        out = Path(args.out) if args.out else Path("metrics/mqar") / f"shape-probe-{r['checkpoint']['run_id']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
        print(f"retained: {out}")
        print(json.dumps(r, indent=2))
        return
    r = session_b_from_disk(args.ckpt, args.journal, cfg, n_facts=args.n_facts, n_negctrl=args.n_negctrl,
                            seed=args.seed, k=cfg.journal_k, budget=cfg.journal_read_bytes, window_len=cfg.journal_read_bytes,
                            train_pool_seed=(cfg.journal_train_seed if args.probe_training_pool else None),
                            train_pool_n=cfg.journal_pool_facts, policy=args.policy)
    suffix = "" if args.policy == "plain" else "-" + args.policy.replace("+", "-")
    out = Path(args.out) if args.out else Path("metrics/mqar") / f"hm-lm-{r['checkpoint']['run_id']}-session-b{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    print(f"retained: {out}")
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    _cli()
