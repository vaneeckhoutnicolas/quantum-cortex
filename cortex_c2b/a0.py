"""cortex_c2b.a0 -- A0, the construction's ceiling: the magnetism of a line under byte keys (ADR-008, amendment of
2026-10-02, part A, validated B1; delivered with ADR-010's build as its part E says; row 54).

For every read of a session B file, each line of the window is scored by its byte overlap with the queried entity, in the
two declared variants: the bag (the multiset of the bytes of the line's entity against the multiset of the queried
entity's bytes, as a Jaccard index) and the aligned form (the count of positions where the line's entity and the queried
entity carry the same byte at the same offset of the name). ADR-010 E's dated change of the definition: each line's
entity is read from the planting record's text by the generator's name pattern (a test shows it equals the template
reading wherever a template parses on rows 48 and 49's files; it is the only source on A and on T0's paraphrase forms);
the queried entity from the question's text by the same pattern. Recorded per read: the oracle's score per line, the
rank of the own line under each variant, whether the oracle's top line is the own line; A0's informative output, its top
line against the decoder's own citation (`raw_cited_pointer` where the file carries it, the decoder's output before any
policy; `cited_pointer` on the files written before row 49, where it is the decoder's citation after the veto only) with
the mark oracle beside, binned by the bag score of the nearest other line (under 0.5; 0.5 to under 0.8; 0.8 and above); a
tie at the top counts as not decisive. Aggregate: the share of reads where the oracle's top line is the own line, per
variant. No training, no model, no key: seconds on the committed files.

The readings, declared in ADR-008 before any number and fixed there: the own line on top "rarely" is a share at or
below 0.5 (A0 (i): the construction bounds the selectivity), "almost always" a share at or above 0.9 (A0 (ii) when the
model's top line is not the own line: the limit is in the reading, not in the keys; A0 (iii) when the model's selection
follows the oracle without the decision following), anything between a partial ceiling reported as such. Nothing here is
a number of the record before the founder runs it.
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from cortex_c2b import content_hash as _ch
from cortex_c2b.hm_protocol import generate_facts, N_FACTS
from cortex_c2b.near_duplicate_probe import parse_statement, parse_question

NAME_PATTERN = re.compile(r"[BDFGKLMNPRSTVZ][aeiou][bdfgklmnprstvz][aeiou]-[0-9a-f]{4}")
BINS = (("under_0.5", 0.0, 0.5), ("0.5_to_under_0.8", 0.5, 0.8), ("0.8_and_above", 0.8, 1.0000001))
RARELY, ALMOST_ALWAYS = 0.5, 0.9


def entity_in(text: str) -> str | None:
    """The entity by the generator's name pattern: the first match, None when none (a summary, an empty line)."""
    m = NAME_PATTERN.search(text or "")
    return m.group(0) if m else None


def bag_score(a: str | None, b: str | None) -> float:
    """The Jaccard index of the two byte multisets."""
    if not a or not b:
        return 0.0
    ca, cb = Counter(a.encode("utf-8")), Counter(b.encode("utf-8"))
    inter = sum((ca & cb).values()); union = sum((ca | cb).values())
    return inter / union if union else 0.0


def aligned_score(a: str | None, b: str | None) -> int:
    """The count of positions where the two names carry the same byte at the same offset."""
    if not a or not b:
        return 0
    xa, xb = a.encode("utf-8"), b.encode("utf-8")
    return sum(1 for i in range(min(len(xa), len(xb))) if xa[i] == xb[i])


def texts_of_file(f: dict) -> dict[str, str]:
    """Pointer -> text, from the planting record the file embeds (rows 48 and 49: `near_duplicates.planting.writes`; a
    sealed family pass file: its `texts`, the family's statements by pointer, or `planting_writes` on a family file); a
    file of the record written on the first family with none of these (row 47): the generator's first family at the file's
    seed. A pass file without `texts` is refused rather than read against the first family's statements."""
    texts: dict[str, str] = {}
    if isinstance(f.get("texts"), dict):
        return {str(k): str(v) for k, v in f["texts"].items()}
    nd = (f.get("near_duplicates") or {}).get("planting") or {}
    for w in nd.get("writes") or []:
        if w.get("pointer") and w.get("text"):
            texts[w["pointer"]] = w["text"]
    for w in f.get("planting_writes") or []:
        if w.get("pointer") and w.get("text"):
            texts[w["pointer"]] = w["text"]
    if not texts:
        if isinstance(f.get("answers"), list):
            raise ValueError("a0: a sealed family pass file without its `texts` (written by the runner since the build of 2026-10-30); refused rather than read against the first family")
        for fact in generate_facts(N_FACTS, (f.get("generator") or {}).get("seed", 0))[0]:
            texts[_ch(fact.statement.encode("utf-8"))] = fact.statement
    return texts


def own_pointer_map(f: dict) -> dict[str, str]:
    """Entity -> the pointer of its own planted line: the planting record's originals (rows 48, 49), the pass file's
    `own_pointer_of` (the sealed families), else the generator's first family."""
    out: dict[str, str] = {}
    if isinstance(f.get("own_pointer_of"), dict):
        return {str(k): str(v) for k, v in f["own_pointer_of"].items()}
    nd = (f.get("near_duplicates") or {}).get("planting") or {}
    for w in nd.get("writes") or []:
        if w.get("kind") in ("original", "E0d") and w.get("pointer") and w.get("entity"):
            out.setdefault(w["entity"], w["pointer"])
    for w in f.get("planting_writes") or []:
        if w.get("pointer") and w.get("entity"):
            out.setdefault(w["entity"], w["pointer"])
    if not out:
        for fact in generate_facts(N_FACTS, (f.get("generator") or {}).get("seed", 0))[0]:
            out[fact.entity] = _ch(fact.statement.encode("utf-8"))
    return out


def _bin_of(x: float | None) -> str | None:
    if x is None:
        return None
    for name, lo, hi in BINS:
        if lo <= x < hi:
            return name
    return BINS[-1][0]


def oracle_cited_pointer(answer: dict, threshold: float | None) -> str | None:
    """The mark oracle's citation on this read: the highest marked line when its mark is strictly above the threshold."""
    win = [w for w in answer.get("window") or [] if w.get("mark") is not None]
    if not win or threshold is None:
        return None
    order = answer.get("retrieval_order")
    if order:                                                    # the frozen oracle's tie rule: the first maximum in retrieval order
        win = sorted(win, key=lambda w: order.index(w["pointer"]) if w["pointer"] in order else len(order))
    hi = max(win, key=lambda w: w["mark"])                       # max keeps the first of equal marks: retrieval order when carried, the window's slot order otherwise
    return hi["pointer"] if hi["mark"] > threshold else None


def decoder_citation(answer: dict) -> tuple[str | None, str]:
    """The decoder's own citation and which field gave it (`raw_cited_pointer` before any policy, else `cited_pointer`)."""
    if "raw_cited_pointer" in answer:
        return answer.get("raw_cited_pointer"), "raw_cited_pointer"
    return answer.get("cited_pointer"), "cited_pointer"


def score_read(answer: dict, texts: dict[str, str], threshold: float | None, own_of: dict[str, str] | None = None) -> dict | None:
    """A0 on one read: None when the window is empty or the question names no entity."""
    win = answer.get("window") or []
    if not win:
        return None
    q_entity = entity_in(answer.get("query") or "")
    if q_entity is None:
        pq = parse_question(answer.get("query") or "")
        q_entity = pq[1] if pq else None
    if q_entity is None and not answer.get("query"):
        q_entity = entity_in(answer.get("entity") or "")      # a record written before row 49 carries no question text: its entity field
    if q_entity is None:
        return None
    own = answer.get("own_pointer")
    if own is None and own_of is not None and answer.get("block", "on") not in ("never_planted", "near_name", "negctrl"):
        own = own_of.get(answer.get("entity"))
    if own is None and answer.get("cited_pointer_is_own"):
        own = answer.get("cited_pointer")
    if any(w["pointer"] not in texts for w in win):
        return {"skipped": True, "why": "a window line's pointer has no text in the file; not scored", "pointers_without_text": [w["pointer"] for w in win if w["pointer"] not in texts]}
    lines = []
    for w in win:
        text = texts.get(w["pointer"], "")
        ent = entity_in(text)
        lines.append({"label": w.get("label"), "pointer": w["pointer"], "entity": ent, "bag": bag_score(ent, q_entity),
                      "aligned": aligned_score(ent, q_entity), "mark": w.get("mark"), "is_own": (w["pointer"] == own) if own else None})
    out = {"query_entity": q_entity, "own_pointer": own, "lines": lines}
    for variant in ("bag", "aligned"):
        order = sorted(lines, key=lambda l: -l[variant])
        top = order[0][variant]
        tied = [l for l in order if l[variant] == top]
        own_rank = next((i + 1 for i, l in enumerate(order) if l["is_own"]), None) if own else None
        out[variant] = {"top_pointer": tied[0]["pointer"] if len(tied) == 1 else None, "tie": len(tied) > 1, "top_score": top,
                        "own_rank": own_rank, "top_is_own": (len(tied) == 1 and tied[0]["is_own"] is True),
                        "own_score": next((l[variant] for l in lines if l["is_own"]), None) if own else None}
    others = [l["bag"] for l in lines if not l["is_own"]] if own else [l["bag"] for l in lines]
    nearest_other = max(others) if others else None
    out["nearest_other_bag"] = nearest_other
    out["bin"] = _bin_of(nearest_other)
    cited, field = decoder_citation(answer)
    out["decoder_cited_pointer"] = cited; out["decoder_field"] = field
    out["decoder_cites_own"] = (cited == own) if (cited is not None and own) else None
    out["decoder_agrees_with_bag_top"] = (cited is not None and out["bag"]["top_pointer"] is not None and cited == out["bag"]["top_pointer"])
    out["decoder_agrees_with_aligned_top"] = (cited is not None and out["aligned"]["top_pointer"] is not None and cited == out["aligned"]["top_pointer"])
    oc = oracle_cited_pointer(answer, threshold)
    out["mark_oracle_pointer"] = oc; out["mark_oracle_cites_own"] = (oc == own) if (oc is not None and own) else None
    return out


def read_file(f: dict, file_name: str = "") -> dict:
    """A0 over a session B file (or a sealed family pass file): every read with its lines and the aggregates per block."""
    texts = texts_of_file(f)
    own_of = own_pointer_map(f)
    thr = (f.get("hm_mark_oracle") or {}).get("threshold")
    if thr is None and isinstance(f.get("oracle"), dict):
        thr = f["oracle"].get("threshold")
    blocks = {}
    if isinstance(f.get("answers"), dict):
        for arm, answers in f["answers"].items():
            if arm == "off":
                continue
            blocks[arm] = answers
        for name, blk in (f.get("reported_blocks") or {}).items():
            blocks[f"reported:{name}"] = blk.get("answers") or []
    elif isinstance(f.get("answers"), list):                     # a sealed family pass file: the records carry their block
        for a in f["answers"]:
            blocks.setdefault(a.get("block", "answers"), []).append(a)
    per_block = {}
    reads = {}
    for name, answers in blocks.items():
        scored = [(i, score_read(a, texts, thr, own_of if name in ("on", "planted", "cross_schema") or name.startswith("reported:neighbour") else None)) for i, a in enumerate(answers)]
        kept = [s for _, s in scored if s is not None and not s.get("skipped")]
        without_text = sum(1 for _, s in scored if s is not None and s.get("skipped"))
        with_own = [s for s in kept if s["own_pointer"]]
        agg = {"n_answers": len(answers), "n_scored": len(kept), "n_with_own": len(with_own), "n_not_scored_no_text": without_text}
        for v in ("bag", "aligned"):
            agg[v] = {"top_is_own": sum(1 for s in with_own if s[v]["top_is_own"]), "ties_at_top": sum(1 for s in kept if s[v]["tie"]),
                      "share_top_is_own": (sum(1 for s in with_own if s[v]["top_is_own"]) / len(with_own)) if with_own else None,
                      "own_rank_counts": dict(Counter(str(s[v]["own_rank"]) for s in with_own))}
        agg["decoder"] = {"cites_own": sum(1 for s in with_own if s["decoder_cites_own"]), "agrees_with_bag_top": sum(1 for s in kept if s["decoder_agrees_with_bag_top"]),
                          "agrees_with_aligned_top": sum(1 for s in kept if s["decoder_agrees_with_aligned_top"]),
                          "field": (kept[0]["decoder_field"] if kept else None)}
        agg["mark_oracle"] = {"cites_own": sum(1 for s in with_own if s["mark_oracle_cites_own"])}
        bins = {}
        for bname, _, _ in BINS:
            sub = [s for s in with_own if s["bin"] == bname]
            bins[bname] = {"n": len(sub), "bag_top_is_own": sum(1 for s in sub if s["bag"]["top_is_own"]), "aligned_top_is_own": sum(1 for s in sub if s["aligned"]["top_is_own"]),
                           "decoder_cites_own": sum(1 for s in sub if s["decoder_cites_own"]), "mark_oracle_cites_own": sum(1 for s in sub if s["mark_oracle_cites_own"]),
                           "decoder_agrees_with_bag_top": sum(1 for s in sub if s["decoder_agrees_with_bag_top"])}
        agg["by_nearest_other_bag"] = bins
        per_block[name] = agg
        reads[name] = [{"i": i, **(s or {"skipped": True})} for i, s in scored]
    return {"file": file_name, "threshold": thr, "n_texts": len(texts), "blocks": per_block, "reads": reads,
            "definition": {"entity_of_line": "the planting record's text read by the generator's name pattern (ADR-010 E)",
                           "entity_of_question": "the question's text by the generator's name pattern (the template reading where the pattern finds none; the record's entity field on the files written before row 49, which carry no question text)",
                           "bag": "Jaccard index of the two byte multisets", "aligned": "same byte at the same offset of the name, counted",
                           "decisive": "a tie at the top is not decisive", "bins": "by the bag score of the nearest other line: under 0.5; 0.5 to under 0.8; 0.8 and above",
                           "mark_oracle_tie": "the first maximum in retrieval order where the file carries it (rows 49 and later, the passes), the window's slot order on the files before",
                           "not_scored": "a read whose window holds a pointer without a text in the file is counted apart, never scored at zero"}}


def letter_of(share: float | None, decoder_share: float | None) -> str:
    """ADR-008 part A's readings on the aggregate share (the bag variant, the planted block)."""
    if share is None:
        return "unread"
    if share <= RARELY:
        return "A0 (i): the oracle's top line is the own line rarely; the construction bounds the selectivity"
    if share >= ALMOST_ALWAYS:
        if decoder_share is not None and decoder_share < ALMOST_ALWAYS:
            return "A0 (ii): the oracle's top line is the own line almost always and the model's top line is not; the limit is in the reading, not in the keys"
        return "A0 (iii) or the oracle high with the model following it: read the decision apart (R2 of ADR-008 part A)"
    return f"a partial ceiling: the share {share:.3f} lies between {RARELY} and {ALMOST_ALWAYS}, reported as such"


def template_entity_agrees(f: dict) -> dict:
    """The test ADR-010 E asks for: the entity by the name pattern equals the template reading wherever a template parses."""
    texts = texts_of_file(f)
    agree = disagree = parsed = 0
    for text in texts.values():
        ps = parse_statement(text)
        if ps is None:
            continue
        parsed += 1
        if entity_in(text) == ps[1]:
            agree += 1
        else:
            disagree += 1
    return {"parsed": parsed, "agree": agree, "disagree": disagree}


def _cli(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="A0, the construction's ceiling under byte keys, on committed session B files (no model, no key)")
    ap.add_argument("files", nargs="+", help="session B files (rows 48, 49) or sealed family pass files")
    ap.add_argument("--out", default="metrics/mqar")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    names: list[str] = []
    for name in args.files:                                 # a pattern is expanded here (PowerShell passes it as written)
        if any(c in name for c in "*?["):
            hits = sorted(glob.glob(name))
            if not hits:
                raise SystemExit(f"a0: {name} matches no file")
            names.extend(hits)
        else:
            names.append(name)
    for name in names:
        p = Path(name)
        f = json.loads(p.read_text(encoding="utf-8"))
        r = read_file(f, p.name)
        r["source_sha256_lf"] = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        r["template_entity_agreement"] = template_entity_agrees(f)
        planted = r["blocks"].get("on") or r["blocks"].get("planted") or {}
        share = (planted.get("bag") or {}).get("share_top_is_own")
        dec = planted.get("decoder", {}).get("cites_own"); nwo = planted.get("n_with_own")
        r["letter"] = letter_of(share, (dec / nwo) if (dec is not None and nwo) else None)
        r["share_top_is_own_planted"] = share; r["decoder_cites_own_share_planted"] = ((dec / nwo) if (dec is not None and nwo) else None)
        target = out_dir / f"a0-{p.stem}.json"
        if target.exists():
            raise SystemExit(f"a0: {target} exists; a file of the record is never overwritten")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(r, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        print(f"retained: {target}  sha256 {hashlib.sha256(target.read_bytes()).hexdigest()[:16]}")
        print(json.dumps({"file": p.name, "blocks": {k: {"n_with_own": v["n_with_own"], "bag_share_top_is_own": v["bag"]["share_top_is_own"],
                                                         "aligned_share_top_is_own": v["aligned"]["share_top_is_own"], "decoder_cites_own": v["decoder"]["cites_own"]}
                                                     for k, v in r["blocks"].items()}, "letter": r["letter"]}, indent=1))


if __name__ == "__main__":
    _cli()
