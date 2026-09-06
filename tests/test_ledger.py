"""Ledger integrity: every committed record must validate against run-v1.

Mirrors the CI `validate-ledger` job as a real pytest. The ledger law
(CONTRIBUTING): a run without its committed, schema-valid record does not exist.
"""
import json
from pathlib import Path

import pytest
from jsonschema import validate

REPO = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((REPO / "metrics" / "schema" / "run-v1.schema.json").read_text())
LEDGER = REPO / "metrics" / "runs.jsonl"


def _records():
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def test_schema_itself_is_valid_json():
    assert SCHEMA.get("required"), "run-v1 schema must declare required fields"


def test_ledger_lines_are_json_and_schema_valid():
    recs = _records()
    if not recs:
        pytest.skip("no ledger yet (expected pre-N1)")
    for rec in recs:
        validate(rec, SCHEMA)  # raises on any malformed record


def test_ledger_run_ids_are_unique():
    recs = _records()
    if not recs:
        pytest.skip("no ledger yet (expected pre-N1)")
    ids = [r["run_id"] for r in recs]
    assert len(ids) == len(set(ids)), f"duplicate run_id in ledger: {ids}"


def test_completed_runs_have_finite_perplexity():
    for rec in _records():
        if rec["run"]["status"] == "completed":
            ppl = rec["results"]["val_perplexity"]
            assert ppl is None or (ppl == ppl and ppl < float("inf")), rec["run_id"]
