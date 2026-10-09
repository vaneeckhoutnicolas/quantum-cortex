"""The addressing probe (ADR-007 amendment 2026-10-08; row 13c): the organ alone, addressed by the
planted address, by the text of the question and by the paraphrase, against two chance levels.
What these tests pin: the planted address returns every attribute (a construction fact of rows 13
and 13b); the text rates sit at the chance level under the hash seeded encoder; the reading is the
amendment's letter (a) on the protocol's generator; the file carries the generator's hash and is
reproducible. Nothing here is a number of the record."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cortex_c2b import addressing_probe as ap                                    # noqa: E402
from cortex_c2b.hm_protocol import generate_facts                                # noqa: E402


def test_the_planted_address_is_exact_and_the_text_is_at_chance():
    out = ap.run(n=100, seed=0)
    r = out["rates"]
    assert out["admitted"] == 100 and r["by_planted_address"] == 1.0
    chance = max(r["by_random_cue"], r["by_random_line_attribute"])
    assert abs(r["by_question_text"] - chance) <= ap.CHANCE_BAND
    assert abs(r["by_paraphrase"] - chance) <= ap.CHANCE_BAND
    assert out["readings"] == {"address_exact": True, "text_at_chance": True, "letter": out["verdict"]}
    assert out["verdict"].startswith("a:")


def test_the_file_is_reproducible_and_carries_the_generator(tmp_path):
    a, b = ap.run(n=50, seed=3), ap.run(n=50, seed=3)
    assert a["rates"] == b["rates"] and a["config"]["generator_hash"] == generate_facts(50, 3)[1]
    ap._cli(["--n", "50", "--seed", "3", "--out", str(tmp_path)])
    d = json.loads((tmp_path / "addressing-probe-seed3.json").read_text(encoding="utf-8"))
    assert d["rates"] == a["rates"] and len(ap.summary_lines(d)) == 3
