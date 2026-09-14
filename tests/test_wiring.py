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
    """The N1 record carries the manifest hash as it was computed at the time, with the slice
    absent from the manifest's path ('unresolved' content sha). When the slice IS present on
    disk, the manifest resolves on its content and the hash changes: that resolved hash is
    checked against the provenance note written the first time it was seen (2026-09-14)."""
    ledger = REPO / "metrics" / "runs.jsonl"
    if not ledger.exists():
        return
    rec = json.loads(ledger.read_text().splitlines()[0])
    manifest = REPO / "data" / "mixes" / "fineweb-edu-n1.json"
    doc = json.loads(manifest.read_text())
    dag = from_manifest(doc)
    spec = dag.inputs[0]._spec()
    prov = json.loads((REPO / "data" / "mixes" / "fineweb-edu-n1.provenance.json").read_text())
    if str(spec["content_sha"]).startswith("unresolved:"):
        assert rec["training"].get("mix_hash") == dag.node_hash() == prov["mix_hash_as_recorded_in_the_n1_record"], \
            "the N1 record's mix_hash must match its representative manifest, as recorded (slice absent)"
    else:
        assert spec["content_sha"] == prov["content_sha_16"], \
            "the slice on disk is not the one the provenance note resolved: a different data file"
        assert dag.node_hash() == prov["mix_hash_resolved"]
        assert rec["training"].get("mix_hash") == prov["mix_hash_as_recorded_in_the_n1_record"]


def test_schema_accepts_mix_hash():
    schema = json.loads((REPO / "metrics" / "schema" / "run-v1.schema.json").read_text())
    assert "mix_hash" in schema["properties"]["training"]["properties"]
    ledger = REPO / "metrics" / "runs.jsonl"
    if ledger.exists():
        validate(json.loads(ledger.read_text().splitlines()[0]), schema)
