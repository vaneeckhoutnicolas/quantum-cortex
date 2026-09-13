"""The organ as a component (use case 1): every planted address is cited with its own
pointer from a NEW process that reopened the sealed journal; every never written
address is an abstention under the declared line; the storage is sealed and durable."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cortex_c2b.companion_demo import run, CASE, QUESTIONS_PLANTED, QUESTIONS_NEVER_WRITTEN, CITE_MIN   # noqa: E402
from cortex_c2b.crypto import generate_key                                                              # noqa: E402


def test_companion_demo_cites_the_planted_and_abstains_on_the_never_written(tmp_path):
    r = run(tmp_path / "companion-journal.jsonl", generate_key())
    s = r["summary"]
    assert r["session_a"]["records_written"] == len(CASE)
    assert r["session_b"]["entries_reopened"] == len(CASE)
    assert r["session_b"]["process"].startswith("new")
    assert s["planted_questions"] == len(QUESTIONS_PLANTED) == s["cited_with_the_right_pointer"]
    assert s["never_written_questions"] == len(QUESTIONS_NEVER_WRITTEN) == s["abstained"]
    assert s["min_score_on_planted"] >= CITE_MIN > s["max_score_on_never_written"]
    assert r["session_a"]["storage"]["mode"] == "durable" and r["session_b"]["storage"]["mode"] == "durable"
    assert not r["session_a"]["storage"]["plaintext_scope"]                       # sealed at rest
    for t in r["session_b"]["transcript"]:
        assert t["answer"] in ("cite", "abstain")
        if t["answer"] == "cite":
            assert t["record"] and t["pointer"]
