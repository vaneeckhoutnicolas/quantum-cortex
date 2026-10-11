"""RESULTS row 13c: the addressing probe on the organ alone, run by the founder on 2026-10-09 at the
four seeds of row 13b (ADR-007 amendment 2026-10-08, declared before any number), read cold from the
files on 2026-10-11. Every scalar the row states is recomputed here from the committed files: the
five rates of each seed, the generator hashes, the admission counts, the margin of the text and of
the paraphrase to the chance level, the amendment's letter recomputed from the rates alone, the
provenance hashes; and the probe is run again at each seed, in process, and compared to its file
rate for rate, since it needs no key, no checkpoint and no training. A change to a file breaks this
test; a change to the row without the file cannot pass it."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cortex_c2b import addressing_probe as ap                                    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (0, 1, 2, 3)
N, THRESHOLD, BAND = 200, 0.15, 0.05
GENERATOR = {0: "a5e22c70a3275ed4", 1: "82bd2ff094d9b030", 2: "80bb909b049846de", 3: "0c37aac8c10a178f"}
ROW13C = {   # per seed: by the planted address, by the question's text, by the paraphrase, random cue, random line attribute
    0: (1.0, 0.035, 0.035, 0.020, 0.030),
    1: (1.0, 0.030, 0.030, 0.050, 0.025),
    2: (1.0, 0.025, 0.020, 0.030, 0.015),
    3: (1.0, 0.015, 0.015, 0.015, 0.035),
}
LF_SHA = {0: "5c2339b23fabcb80", 1: "cd1edece0a88efc8", 2: "7323f8537e4a9d1a", 3: "464a09c234e60234"}
TIMESTAMPS = {0: "2026-10-09T19:23:21+00:00", 1: "2026-10-09T19:28:01+00:00",
              2: "2026-10-09T19:28:02+00:00", 3: "2026-10-09T19:28:04+00:00"}
RATE_KEYS = ("by_planted_address", "by_question_text", "by_paraphrase", "by_random_cue", "by_random_line_attribute")


def _path(seed: int) -> Path:
    return ROOT / "metrics/mqar" / f"addressing-probe-seed{seed}.json"


def _load(seed: int) -> dict:
    return json.loads(_path(seed).read_text(encoding="utf-8"))


def _lf_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


@pytest.mark.parametrize("seed", SEEDS)
def test_each_file_carries_the_graved_rates_configuration_and_counts(seed):
    out = _load(seed)
    c = out["config"]
    assert (c["n_facts"], c["seed"], c["admission_threshold"], c["chance_band"], c["k"]) == (N, seed, THRESHOLD, BAND, 1)
    assert c["generator_hash"] == GENERATOR[seed]
    assert out["admitted"] == N
    assert out["timestamp_utc"] == TIMESTAMPS[seed]
    assert tuple(out["rates"][k] for k in RATE_KEYS) == pytest.approx(ROW13C[seed], abs=1e-12)
    assert out["readings"]["address_exact"] is True and out["readings"]["text_at_chance"] is True
    assert out["verdict"].startswith("a:") and out["verdict"] == out["readings"]["letter"]


@pytest.mark.parametrize("seed", SEEDS)
def test_the_letter_recomputed_from_the_rates_alone_is_a(seed):
    address, text, paraphrase, cue, line = ROW13C[seed]
    chance = max(cue, line)
    assert address == 1.0
    assert abs(text - chance) <= BAND and abs(paraphrase - chance) <= BAND


def test_the_margins_to_chance_are_at_most_one_fact_in_two_hundred_above_and_below_on_three_seeds():
    margins = {s: (ROW13C[s][1] - max(ROW13C[s][3], ROW13C[s][4]), ROW13C[s][2] - max(ROW13C[s][3], ROW13C[s][4])) for s in SEEDS}
    assert max(max(m) for m in margins.values()) == pytest.approx(0.005, abs=1e-12)          # seed 0, the text and the paraphrase
    assert sum(1 for s in SEEDS if max(margins[s]) < 0) == 3                                   # seeds 1, 2, 3 below the chance level
    assert all(m[0] <= 0.005 + 1e-12 and m[1] <= 0.005 + 1e-12 for m in margins.values())


def test_the_provenance_file_names_the_four_files_and_their_hashes():
    prov = json.loads((ROOT / "metrics/mqar/PROVENANCE-addressing-probe.json").read_text(encoding="utf-8"))
    assert prov["row"] == "13c" and prov["code_commit"] == "92a3a40"
    for seed in SEEDS:
        entry = prov["files"][f"addressing-probe-seed{seed}.json"]
        assert entry["sha256_lf"] == LF_SHA[seed] == _lf_sha(_path(seed))
        assert entry["timestamp_utc"] == TIMESTAMPS[seed] and entry["generator_hash"] == GENERATOR[seed]


@pytest.mark.parametrize("seed", SEEDS)
def test_the_probe_run_again_at_the_seed_reproduces_its_file_rate_for_rate(seed):
    out = ap.run(n=N, seed=seed)
    assert out["config"]["generator_hash"] == GENERATOR[seed] and out["admitted"] == N
    assert tuple(out["rates"][k] for k in RATE_KEYS) == pytest.approx(ROW13C[seed], abs=1e-12)
    assert out["verdict"] == _load(seed)["verdict"]
