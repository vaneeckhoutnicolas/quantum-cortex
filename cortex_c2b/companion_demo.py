"""The organ as a component, on use case 1 of the whitepaper (§1.4): the long running
project companion. A practitioner works for months on one matter; a fresh session
months later must find what was decided, cite the record it rests on, and say
"no record" for what was never written.

This demonstration runs on the protocol's own code path (the gated write path, the
sealed on disk journal, the cue index, session B in a NEW PROCESS that knows nothing
of session A but the disk) with the organ level cue convention of the Molaison
protocol: the entity is the address (`_cue_for(f"{schema}|{entity}")`). What it shows
is the organ's contract: an answer is a citation of a stored record with its pointer,
or an abstention when the journal's best candidate scores under the declared line.
What it does not show is the learned addressing of paraphrases: that is the language
model arm's job, measured separately (RESULTS rows 21 onward).

    python -m cortex_c2b.companion_demo --out metrics/mqar/companion-demo-<date>.json

The declared line: the citation threshold on the journal's cosine score, CITE_MIN = 0.5.
A stored address scores 1.0 by construction; a never written address, a random unit
vector against a 64 dimensional index, scores well under 0.5 (the demonstration
records the observed maximum so the margin is a measurement, not a promise).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cortex_c2b import Journal, POLICY_STOP, lifecycle_declaration
from cortex_c2b.hm_protocol import _cue_for
from cortex_c2b.read_path import JournalPath
from cortex_c2b.write_path import WritePath

CITE_MIN = 0.5                       # the declared citation line on the journal's cosine score


@dataclass
class Record:
    date: str
    schema: str
    entity: str                      # the address
    text: str                        # the payload as written that day


# A fictional intellectual property matter, January to May: what a practitioner writes down.
CASE: list[Record] = [
    Record("12 Jan", "decision", "argument:originality",
           "12 Jan: the originality argument is set aside on the strength of ruling 2024/AR/512 (Brussels Court of Appeal, 8 Nov 2023), which denies originality to template based layouts"),
    Record("12 Jan", "object", "ruling:2024/AR/512",
           "12 Jan: ruling 2024/AR/512, Brussels Court of Appeal, 8 Nov 2023: analysed in full; holds that a layout produced from a vendor template carries no author's own choices"),
    Record("19 Jan", "decision", "argument:database-right",
           "19 Jan: the database right is retained as the main ground: the platform's listing base is a substantial investment in obtaining and verifying the data"),
    Record("2 Feb", "person", "expert:Barbieux",
           "2 Feb: the court expert Barbieux is to be challenged on method: his report compares source trees without dating the commits"),
    Record("16 Feb", "event", "deadline:conclusions",
           "16 Feb: the deadline for our conclusions is 30 April; the opponent's are due 15 April"),
    Record("3 Mar", "object", "exhibit:git-history",
           "3 Mar: exhibit 12, the git history 2008 to 2009, is retained as the primary proof of authorship; certified copy ordered"),
    Record("21 Mar", "decision", "settlement:offer-1",
           "21 Mar: the first settlement offer (licence at 40 percent) is declined; the client wants recognition of authorship before any licence"),
    Record("8 Apr", "place", "venue:hearing",
           "8 Apr: the hearing is fixed at the Brussels Enterprise Court, chamber 17, room B, 4 May at 9:30"),
    Record("8 Apr", "event", "hearing:4-May",
           "8 Apr: hearing of 4 May: pleadings limited to the database right and the authorship of exhibit 12; the originality argument stays out"),
    Record("22 Apr", "person", "witness:Moreau",
           "22 Apr: witness Moreau (the hosting provider) confirms in writing the deployment dates of March 2009; statement filed as exhibit 15"),
    Record("30 Apr", "decision", "conclusions:filed",
           "30 Apr: our conclusions are filed: two grounds, the database right and the authorship of exhibit 12; originality is not pleaded"),
    Record("4 May", "event", "hearing:outcome",
           "4 May: the case is taken under advisement; judgment expected within six weeks"),
]

# What a fresh session asks months later: planted addresses, then addresses that were never written.
QUESTIONS_PLANTED = [
    ("argument:originality", "what was decided about the originality argument, and on what ground?"),
    ("hearing:4-May", "what is the scope of the 4 May hearing?"),
    ("settlement:offer-1", "what happened to the first settlement offer?"),
    ("exhibit:git-history", "what is our primary proof of authorship?"),
    ("expert:Barbieux", "what is our line on the court expert?"),
]
QUESTIONS_NEVER_WRITTEN = [
    ("witness:Dupont", "what did witness Dupont say?"),
    ("invoice:2019-042", "was invoice 2019-042 paid?"),
    ("ruling:2022/AR/77", "what does ruling 2022/AR/77 hold?"),
]


def cue(schema: str, entity: str):
    return _cue_for(f"{schema}|{entity}")


def _schema_of(entity: str) -> str:
    for r in CASE:
        if r.entity == entity:
            return r.schema
    return "decision"                # a never written address is asked as if it were a decision


def session_a(journal_path: Path, key: bytes) -> dict:
    """Months of work: every record written through the gated write path into a sealed journal."""
    journal = Journal(journal_path, key=key, policy=POLICY_STOP)
    wp = WritePath(journal, admission_threshold=0.15, seed=0)
    jp = JournalPath(journal)
    written = []
    for i, r in enumerate(CASE):
        rep = wp.write(cue(r.schema, r.entity), r.text.encode("utf-8"), r.schema, now=float(i))
        if rep.admitted:
            jp.on_write(rep.entry)
            written.append({"date": r.date, "entity": r.entity, "pointer": rep.entry.pointer})
    decl = lifecycle_declaration(journal)["storage"]
    return {"records_written": len(written), "records": written, "storage": decl,
            "durable": journal.mode == "durable"}


def session_b(journal_path: Path, key: bytes) -> dict:
    """A fresh session, months later, in a new process: the journal reopened from the disk
    alone, the index recomputed; every answer is a citation or an abstention."""
    journal = Journal(journal_path, key=key, policy=POLICY_STOP)
    jp = JournalPath(journal)
    transcript = []
    for entity, question in QUESTIONS_PLANTED + QUESTIONS_NEVER_WRITTEN:
        hits = jp.retrieve(cue(_schema_of(entity), entity), k=1)
        score = float(hits[0][2]) if hits else 0.0
        if hits and score >= CITE_MIN:
            entry, payload, _ = hits[0]
            transcript.append({"question": question, "address": entity, "answer": "cite",
                               "pointer": entry.pointer, "record": payload.decode("utf-8", errors="replace"),
                               "score": round(score, 4)})
        else:
            transcript.append({"question": question, "address": entity, "answer": "abstain",
                               "pointer": None, "record": None, "score": round(score, 4)})
    return {"entries_reopened": len(journal), "transcript": transcript,
            "storage": lifecycle_declaration(journal)["storage"]}


def run(journal_path: Path, key: bytes) -> dict:
    a = session_a(journal_path, key)
    env = dict(os.environ, QUANTUM_CORTEX_JOURNAL_KEY=key.hex())
    proc = subprocess.run([sys.executable, "-m", "cortex_c2b.companion_demo", "--session-b", str(journal_path)],
                          capture_output=True, text=True, env=env, cwd=str(Path(__file__).resolve().parents[1]))
    if proc.returncode != 0:
        raise RuntimeError(f"session B failed:\n{proc.stderr[-2000:]}")
    b = json.loads(proc.stdout.strip().splitlines()[-1])
    planted = [t for t in b["transcript"] if t["address"] in dict(QUESTIONS_PLANTED)]
    never = [t for t in b["transcript"] if t["address"] in dict(QUESTIONS_NEVER_WRITTEN)]
    pointer_of = {r["entity"]: r["pointer"] for r in a["records"]}
    cited_right = sum(t["answer"] == "cite" and t["pointer"] == pointer_of.get(t["address"]) for t in planted)
    abstained = sum(t["answer"] == "abstain" for t in never)
    return {
        "demonstration": "the organ as a component: use case 1, the long running project companion",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_path": "gated write path, sealed on disk journal, cue index; session B in a new process from the disk alone",
        "cue_convention": "organ level, the Molaison protocol's: the entity is the address (hash seeded cue); paraphrase addressing is the model arm's job",
        "cite_min": CITE_MIN,
        "session_a": {"records_written": a["records_written"], "storage": a["storage"]},
        "session_b": {"process": "new (nothing of session A but the disk)", "entries_reopened": b["entries_reopened"],
                      "storage": b["storage"], "transcript": b["transcript"]},
        "summary": {"planted_questions": len(planted), "cited_with_the_right_pointer": cited_right,
                    "never_written_questions": len(never), "abstained": abstained,
                    "max_score_on_never_written": max(t["score"] for t in never),
                    "min_score_on_planted": min(t["score"] for t in planted)},
    }


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="the organ as a component: the long running project companion")
    ap.add_argument("--session-b", default=None, help="(internal) reopen this journal in a new process and answer")
    ap.add_argument("--journal", default=None, help="journal path (default: a temporary sealed directory)")
    ap.add_argument("--out", default=None, help="retain the demonstration as a file")
    args = ap.parse_args()
    if args.session_b:
        from cortex_c2b.crypto import key_from_env
        key = key_from_env()
        if key is None:
            raise SystemExit("QUANTUM_CORTEX_JOURNAL_KEY missing")
        print(json.dumps(session_b(Path(args.session_b), key)))
        return
    from cortex_c2b.crypto import generate_key
    key = generate_key()
    if args.journal:
        r = run(Path(args.journal), key)
    else:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            r = run(Path(d) / "companion-journal.jsonl", key)
    for t in r["session_b"]["transcript"]:
        if t["answer"] == "cite":
            print(f"Q: {t['question']}\n   cite {t['pointer'][:12]} (score {t['score']}): {t['record']}\n")
        else:
            print(f"Q: {t['question']}\n   no record for {t['address']} (best score {t['score']} < {CITE_MIN}): abstain\n")
    print(json.dumps(r["summary"], indent=1))
    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
        print(f"retained: {out}")


if __name__ == "__main__":
    _cli()
