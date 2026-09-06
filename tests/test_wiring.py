"""Wiring tests: cortex_data ↔ train.py ↔ ledger (NOW-7 Slice 1 integration)."""
import json
from pathlib import Path

import numpy as np
from jsonschema import validate

from cortex_data import load_manifest, from_manifest, mix_hash

REPO = Path(__file__).resolve().parents[1]


def test_first_manifest_loads_and_hashes():
    man = REPO / "data" / "mixes" / "fineweb-edu-n1.json"
    dag, mh = load_manifest(man)
    assert isinstance(mh, str) and len(mh) == 16
    # deterministic: reloading gives the same hash
    _, mh2 = load_manifest(man)
    assert mh == mh2


def test_manifest_compiles_to_readable_bin(tmp_path):
    # a tiny in-line manifest with an ArraySource-equivalent is not on disk, so
    # build a source manifest over a small .bin we write here
    toks = np.arange(0, 256, dtype=np.uint16).repeat(50)  # 12800 tokens
    src = tmp_path / "s.bin"
    src.write_bytes(toks.tobytes())
    man = {"op": "mix",
           "inputs": [{"op": "source", "path": str(src), "dataset_id": "t",
                       "license": "test", "slice": "s"}],
           "weights": [1.0]}
    dag = from_manifest(man)
    out = tmp_path / "out.bin"
    n = dag.to_bin(out, seed=1337)
    arr = np.memmap(out, dtype=np.uint16, mode="r")   # exactly how train.py reads
    assert len(arr) == n == len(toks)
    assert arr.min() >= 0 and arr.max() <= 255


def test_n1_record_carries_mix_hash_matching_manifest():
    ledger = REPO / "metrics" / "runs.jsonl"
    if not ledger.exists():
        return
    rec = json.loads(ledger.read_text().splitlines()[0])
    _, mh = load_manifest(REPO / "data" / "mixes" / "fineweb-edu-n1.json")
    assert rec["training"].get("mix_hash") == mh, \
        "the N1 record's mix_hash must match its representative manifest"


def test_schema_accepts_mix_hash():
    schema = json.loads((REPO / "metrics" / "schema" / "run-v1.schema.json").read_text())
    assert "mix_hash" in schema["properties"]["training"]["properties"]
    ledger = REPO / "metrics" / "runs.jsonl"
    if ledger.exists():
        validate(json.loads(ledger.read_text().splitlines()[0]), schema)
