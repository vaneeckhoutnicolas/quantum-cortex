"""Stage A of the recurrent base arms (RESULTS rows 30 and 31): the committed
record is reproducible from its unit files alone with the runner's own
functions, and the units carry the arm hashes of the code. A different unit
file, or a change to an arm's configuration, fails loudly here (the lesson of
Rev52: run the tests where the record lives)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cortex_eval.mqar import MQARTier
from cortex_eval.multiseed import _mean_ci
from cortex_eval.recurrent_base import ARM_BY_NAME
from cortex_eval.resumable_ladder import load_done
from cortex_eval.resumable_rb import aggregate

ROOT = Path(__file__).resolve().parent.parent / "metrics" / "mqar"
REF, RB, SIG = ROOT / "ref3000-ckpt", ROOT / "rb3000-ckpt", ROOT / "sigma-l3-3000-ckpt"
SEEDS = [1337, 2024, 7, 42, 99, 3, 11, 2026]
TIERS = [MQARTier(kv_pairs=8, seq_len=128)]


def test_the_references_at_3000_steps_are_thirteen_new_units_plus_the_six_of_the_sigma():
    assert len(list(REF.glob("unit-*.json"))) == 13 and len(list(SIG.glob("unit-*.json"))) == 6
    done = load_done([SIG, REF])
    assert len(done) == 19 and all(u["status"] == "done" and u["steps"] == 3000 for u in done.values())
    committed = json.loads((REF / "SUBSET-L3-local-conv-ARCH-none-3000steps.json").read_text())
    for path in ("L3-+local-conv", "ARCH-none"):
        vals = [float(np.mean([done[f"{path}__s{s}__kv8_seq128"]["accuracy"] for _ in TIERS])) for s in SEEDS]
        st = _mean_ci(vals)
        c = committed["stats"][path]
        assert (st["mean"], st["sd"], st["ci95"], st["n"]) == (c["mean"], c["sd"], c["ci95"], c["n"])
        assert vals == c["per_seed"] == c["per_tier"]["kv8"]
    assert committed["complete"] and committed["subset"] == ["L3-+local-conv", "ARCH-none"]


def test_the_arms_aggregate_is_reproducible_from_the_units_and_the_references():
    done = load_done([RB])
    assert len(done) == 16 and all(u["status"] == "done" and u["steps"] == 3000 for u in done.values())
    refs = load_done([SIG, REF])
    arms = [ARM_BY_NAME["RB-bare"], ARM_BY_NAME["RB-critical"]]
    rec = aggregate(done, refs, arms, SEEDS, TIERS)
    committed = json.loads((RB / "LATEST-rb.json").read_text())
    for key in ("stats", "per_seed", "references", "paired_tests", "gate_readout"):
        assert rec[key] == committed[key], key
    assert committed["references"]["missing"] == []


def test_every_arm_unit_carries_the_code_s_arm_hash_and_its_declared_readout():
    for p in RB.glob("unit-*.json"):
        u = json.loads(p.read_text())
        assert u["arm_hash"] == ARM_BY_NAME[u["arm"]].config_hash(), p.name
        assert {"accuracy", "accuracy_trunk_only", "consolidation_gap", "took_off", "gate_final", "opened_at",
                "retractions"} <= set(u)
        assert abs(u["consolidation_gap"] - round(u["accuracy"] - u["accuracy_trunk_only"], 4)) < 1e-9
        if u["arm"] == "RB-bare":
            assert u["opened_at"] is None                       # the gate is open from step zero
        else:
            assert u["opened_at"] is not None                    # the critical period opened, and when is on record


def test_the_provenance_hashes_match_the_committed_files():
    import hashlib
    prov = json.loads((ROOT / "PROVENANCE-stage-a.json").read_text())
    for rel, sha in prov["sha256_16"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:16] == sha, rel
    log = ROOT / "logs" / "recurrent-base-stage-a.log"
    assert hashlib.sha256(log.read_bytes()).hexdigest()[:16] == prov["logs_retained_in_repo"]["stage_a"]["sha256_16"]
