"""Row 46: the external model arm, Qwen3-1.7B through the frozen prompt, two executions on the founder's
processor (2026-09-17 23:11 UTC and 2026-10-13 03:09 UTC), the agreement between them, S′ with its companions
(ADR-008, amendments of 2026-10-08 and 2026-10-12). Everything the row states is recomputed here from the two
files: the shared numbers equal, the numbers measured once named, S, S′, S_adm, q and the identity between
them from the 256 per span records, the band of chance subsets from the same records with its fixed seed,
the veto threshold from the organ alone, the first file's numbers pinned, the provenance hashes."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from cortex_c2b import external_arm as ea

ROOT = Path(__file__).resolve().parent.parent
FIRST = ROOT / "metrics/mqar/hm-lm-external-qwen-qwen3-1.7b-7fda43aced410d65.json"
REPEAT = ROOT / "metrics/mqar/hm-lm-external-qwen-qwen3-1.7b-7fda43aced410d65-repeat.json"
PROV = ROOT / "metrics/mqar/PROVENANCE-external-arm.json"
S_PRIME_KEYS = ["hm_skill_delta_prime", "ppl_prime", "hm_skill_delta_prime_signed", "hm_prime_window_share", "prime_spans_with_window",
                "prime_token_share", "hm_skill_delta_admitted", "veto_threshold_used", "hm_skill_delta_admitted_within_band"]


@pytest.fixture(scope="module")
def first():
    return json.loads(FIRST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def repeat():
    return json.loads(REPEAT.read_text(encoding="utf-8"))


def test_row_46_the_two_executions_agree_number_for_number_on_the_shared_numbers(first, repeat):
    ok, lines = ea.agree(first, repeat)
    assert ok and lines == []
    assert ea.measured_once(first, repeat) == S_PRIME_KEYS
    shared = [k for k in ea.NUMERIC_KEYS if k in first and k in repeat]
    assert len(shared) == 20 and all(first[k] == repeat[k] for k in shared)
    assert first["model"] == repeat["model"] and first["prompt"] == repeat["prompt"] and first["thresholds"] == repeat["thresholds"]
    assert first["generator"] == repeat["generator"] and first["window"] == repeat["window"]
    assert first["model"]["revision"] == repeat["model"]["revision"] == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert first["prompt"]["hash"] == repeat["prompt"]["hash"] == "7fda43aced410d65"
    assert first["model"]["device"] == repeat["model"]["device"] == "cpu" and first["model"]["dtype"] == "bf16"
    assert first["timestamp_utc"].startswith("2026-09-17T23:11") and repeat["timestamp_utc"].startswith("2026-10-13T03:09")
    assert first["verdict"] == repeat["verdict"] and first["verdict"].startswith("FAIL")
    assert first["run_valid"] is True and repeat["run_valid"] is True and repeat["persistent"] is True and repeat["claimable"] == 0


def test_row_46_the_first_execution_s_numbers_are_the_ones_the_row_quotes(first):
    assert first["hm_recall_on"] == 0.73 and first["hm_recall_off"] == 0.0 and first["hm_gap"] == 0.73
    assert first["hm_negctrl_rate"] == 0.0 and first["hm_invalid_citation_on"] == 0.005 and first["hm_valid_citation_on"] == 0.995
    assert first["hm_false_abstention_on"] == 0.0 and first["hm_retrieval_hit"] == 1.0
    assert first["hm_attr_exact_given_valid"] == pytest.approx(0.7336683417085427)
    assert first["hm_skill_delta"] == pytest.approx(0.18686101600240324) and first["hm_skill_delta_frame"] == pytest.approx(0.4897301782719774)
    assert first["ppl_on"] == pytest.approx(24.524043397634383) and first["ppl_off"] == pytest.approx(20.66294457984348)
    assert first["tokens_scored"] == 58443 and first["hm_dissociation_pass"] == 0 and first["claimable"] == 0
    assert "skill_arm" not in first or "spans" not in first.get("skill_arm", {})      # the first file holds no per span record


def test_row_46_s_s_prime_s_adm_and_q_recompute_from_the_256_per_span_records(repeat):
    sp = repeat["skill_arm"]["spans"]
    assert len(sp) == 256 == repeat["skill_arm"]["n_spans"]
    tok = sum(s["tokens"] for s in sp)
    assert tok == repeat["tokens_scored"] == 58443
    off, on, frame = (sum(s[k] for s in sp) for k in ("nll_off", "nll_on", "nll_frame"))
    assert math.exp((on - off) / tok) - 1 == pytest.approx(repeat["hm_skill_delta"], abs=1e-12)
    assert math.exp((frame - off) / tok) - 1 == pytest.approx(repeat["hm_skill_delta_frame"], abs=1e-12)
    assert math.exp(on / tok) == pytest.approx(repeat["ppl_on"], abs=1e-9) and math.exp(off / tok) == pytest.approx(repeat["ppl_off"], abs=1e-9)
    adm = [s for s in sp if s["admitted"]]
    assert len(adm) == 13 == repeat["prime_spans_with_window"] and len(adm) / 256 == repeat["hm_prime_window_share"]
    q = sum(s["tokens"] for s in adm) / tok
    assert q == pytest.approx(repeat["prime_token_share"], abs=1e-12)
    on_prime = sum(s["nll_on"] if s["admitted"] else s["nll_off"] for s in sp)
    s_prime = math.exp((on_prime - off) / tok) - 1
    assert s_prime == pytest.approx(repeat["hm_skill_delta_prime"], abs=1e-12) == pytest.approx(repeat["hm_skill_delta_prime_signed"], abs=1e-12)
    assert math.exp(on_prime / tok) == pytest.approx(repeat["ppl_prime"], abs=1e-9)
    s_adm = math.exp((sum(s["nll_on"] for s in adm) - sum(s["nll_off"] for s in adm)) / sum(s["tokens"] for s in adm)) - 1
    assert s_adm == pytest.approx(repeat["hm_skill_delta_admitted"], abs=1e-12)
    assert (1 + s_adm) ** q - 1 == pytest.approx(s_prime, abs=1e-12)              # the construction fact of the second round
    assert s_prime > 0.01 and s_adm > (1.01) ** (1 / q) - 1                        # S′ above the 1 % line, as S_adm above its equivalent
    thr = repeat["veto_threshold_used"]
    assert all(s["top_mark"] > thr and s["lines"] > 0 for s in adm) and all(not (s["top_mark"] > thr and s["lines"] > 0) for s in sp if not s["admitted"])
    assert sum(1 for s in sp if s["lines"] == 0) == 7 and all(not s["admitted"] for s in sp if s["lines"] == 0)
    assert min(s["top_mark"] for s in adm) == pytest.approx(0.3419995903968811) and max(s["top_mark"] for s in sp if not s["admitted"]) == pytest.approx(0.3359510898590088)


def test_row_46_the_band_of_chance_subsets_recomputes_from_the_records_and_reads_a(repeat):
    band = ea.admitted_cost_band(repeat["skill_arm"]["spans"])
    fb = repeat["hm_skill_delta_admitted_band"]
    for key in ("observed", "band_low", "band_high", "share_at_or_under"):
        assert band[key] == pytest.approx(fb[key], abs=1e-12), key
    assert band["within_band"] is fb["within_band"] is True and repeat["hm_skill_delta_admitted_within_band"] is True
    assert (band["admitted"], band["spans"], band["draws"], band["seed"]) == (13, 256, 2000, 20261012) == (fb["admitted"], fb["spans"], fb["draws"], fb["seed"])
    assert fb["band_low"] < repeat["hm_skill_delta_admitted"] < fb["band_high"]            # reading (a) of the amendment of 2026-10-12
    assert 0.13 < fb["band_low"] < 0.14 and 0.24 < fb["band_high"] < 0.25 and fb["share_at_or_under"] == 0.6905


def test_row_46_the_veto_threshold_recomputes_from_the_organ_alone(repeat):
    thr = ea.organ_mark_threshold(seed=0)
    assert thr["threshold"] == repeat["veto_threshold"]["threshold"] == repeat["veto_threshold_used"] == pytest.approx(0.3389076888561249)
    assert thr["training_accuracy"] == 1.0 and thr["training_pairs"] == 250 and thr["planted"] == 200
    assert repeat["veto_threshold"]["train_seed"] == 100000 and repeat["veto_threshold"]["index_seed"] == 0
    assert repeat["veto_threshold"]["max_negative_mark"] == repeat["veto_threshold"]["threshold"]


def test_row_46_the_provenance_names_the_two_files_by_their_lf_hashes():
    prov = json.loads(PROV.read_text(encoding="utf-8"))
    assert prov["row"] == "46"
    for name, e in prov["files"].items():
        raw = (ROOT / "metrics/mqar" / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(raw).hexdigest()[:16] == e["sha256_lf"], name
        d = json.loads(raw)
        assert d["timestamp_utc"] == e["timestamp_utc"] and d["prompt"]["hash"] == e["prompt_hash"] and d["model"]["revision"] == e["model_revision"]
    assert set(prov["files"]) == {FIRST.name, REPEAT.name}
    assert prov["reading"] == "a"
