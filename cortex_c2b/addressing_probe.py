"""The addressing probe (ADR-007, amendment of 2026-10-08, declared before any number; RESULTS row 13c).

What the organ alone recalls when it is addressed three ways: by the planted address (the entity's
hash seeded cue, which rows 13 and 13b use at the write and at the read, by the protocol's design),
by the hash seeded cue of the question's text, and by that of the paraphrase; against two chance
levels, a random unit cue and the attribute of a random line. The organ's own encoder is a hash,
not a language: the probe states that as a number, so that rows 13 and 13b can be read for what they
validate (storage indexed by key, and its persistence) and continuity by the text of a question for
what it is (the model arm's measurement, §4.7). No key, no checkpoint, no training: seconds of a
processor. Declared readings in the amendment; the file is the record."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cortex_c2b import Journal
from cortex_c2b.hm_protocol import generate_facts, _cue_for, N_FACTS, CUE_DIM
from cortex_c2b.read_path import JournalPath
from cortex_c2b.write_path import WritePath

ADMISSION_THRESHOLD = 0.15     # the protocol's own (run_hm_protocol)
CHANCE_BAND = 0.05             # "at chance": within this of the random cue rate, declared before any number


def run(n: int = N_FACTS, seed: int = 0, admission_threshold: float = ADMISSION_THRESHOLD) -> dict:
    facts, generator_hash = generate_facts(n, seed)
    j = Journal()
    w, r = WritePath(j, admission_threshold=admission_threshold, seed=seed), JournalPath(j)
    admitted = 0
    for i, f in enumerate(facts):
        rep = w.write(f.cue, f.statement.encode(), f.schema, now=float(i),
                      source=f.address_text)                    # the text of the address, sealed beside the content (ADR-009 B2,
        if rep.admitted:                                        # 2026-10-12); the rates read the cues alone and do not move
            r.on_write(rep.entry); admitted += 1

    def rate(cue_of) -> float:
        ok = 0
        for f in facts:
            hits = r.retrieve(cue_of(f), k=1)
            ok += int(bool(hits) and f.attr in hits[0][1].decode("utf-8", errors="ignore"))
        return ok / len(facts)

    rng = np.random.default_rng(10_000 + seed)                 # the chance draws, seeded apart from the facts

    def random_cue(_f):
        v = rng.standard_normal(CUE_DIM).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    by_address = rate(lambda f: f.cue)
    by_question = rate(lambda f: _cue_for(f.query))
    by_paraphrase = rate(lambda f: _cue_for(f.paraphrase))
    by_random_cue = rate(random_cue)
    by_random_line = float(np.mean([f.attr in facts[int(rng.integers(len(facts)))].statement for f in facts]))
    chance = max(by_random_cue, by_random_line)
    at_chance = (abs(by_question - chance) <= CHANCE_BAND) and (abs(by_paraphrase - chance) <= CHANCE_BAND)
    if by_address == 1.0 and at_chance:
        letter = ("a: by the planted address the organ returns every attribute; by the text of the question and by the "
                  "paraphrase it is at chance: the organ's encoder is an index, rows 13 and 13b validate storage and "
                  "persistence, and continuity by the text is the model arm's measurement")
    elif by_address < 1.0:
        letter = "c: the planted address does not return every attribute: attributed before anything else is read"
    else:
        letter = ("b: the text of the question or the paraphrase is above chance under the hash seeded encoder: "
                  "attributed before any word is written")
    return {"probe": "addressing (ADR-007 amendment 2026-10-08; row 13c)",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": {"n_facts": n, "seed": seed, "admission_threshold": admission_threshold,
                       "generator_hash": generator_hash, "chance_band": CHANCE_BAND, "k": 1,
                       "encoder": "the protocol's hash seeded cue (_cue_for); the entity is the address at the write"},
            "admitted": admitted,
            "rates": {"by_planted_address": by_address, "by_question_text": by_question, "by_paraphrase": by_paraphrase,
                      "by_random_cue": by_random_cue, "by_random_line_attribute": by_random_line},
            "readings": {"address_exact": by_address == 1.0, "text_at_chance": at_chance, "letter": letter},
            "verdict": letter}


def summary_lines(out: dict) -> list[str]:
    r = out["rates"]
    return [f"addressing probe  n {out['config']['n_facts']}  seed {out['config']['seed']}  generator {out['config']['generator_hash']}  admitted {out['admitted']}",
            f"by the planted address {r['by_planted_address']:.3f}  by the question's text {r['by_question_text']:.3f}  "
            f"by the paraphrase {r['by_paraphrase']:.3f}  chance: random cue {r['by_random_cue']:.3f}, random line {r['by_random_line_attribute']:.3f}",
            f"reading: {out['verdict']}"]


def _cli(argv=None):
    ap = argparse.ArgumentParser(description="the addressing probe on the organ alone (row 13c)")
    ap.add_argument("--n", type=int, default=N_FACTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="the directory for the retained file")
    args = ap.parse_args(argv)
    out = run(n=args.n, seed=args.seed)
    for line in summary_lines(out):
        print(line)
    if args.out:
        o = Path(args.out); o.mkdir(parents=True, exist_ok=True)
        p = o / f"addressing-probe-seed{args.seed}.json"
        p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"[record] retained: {p}  sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    _cli()
