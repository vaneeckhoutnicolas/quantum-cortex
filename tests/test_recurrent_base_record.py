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
    stage_a = [p for p in REF.glob("unit-*.json") if not p.name.startswith("unit-ARCH-none+local-conv")]
    assert len(stage_a) == 13 and len(list(SIG.glob("unit-*.json"))) == 6
    done = load_done([SIG, REF])
    assert len(done) == 27 and all(u["status"] == "done" and u["steps"] == 3000 for u in done.values())
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


def test_the_convolved_control_row_32_is_reproducible_from_its_units():
    """Rev60 (declared before the run): eight units of the control plus a local
    convolution, their subset aggregate and the declared readout."""
    from cortex_eval.resumable_rb import REF_CONTROL_CONV, control_conv_readout
    conv = sorted(REF.glob("unit-ARCH-none+local-conv__*.json"))
    assert len(conv) == 8
    refs = load_done([SIG, REF])
    vals = [float(np.mean([refs[f"{REF_CONTROL_CONV}__s{s}__kv8_seq128"]["accuracy"] for _ in TIERS])) for s in SEEDS]
    sub = json.loads((REF / "SUBSET-ARCH-nonelocal-conv-3000steps.json").read_text())
    st = _mean_ci(vals); c = sub["stats"][REF_CONTROL_CONV]
    assert (st["mean"], st["sd"], st["ci95"], st["n"]) == (c["mean"], c["sd"], c["ci95"], c["n"]) and vals == c["per_seed"]
    assert sub["complete"] and sub["subset"] == [REF_CONTROL_CONV]
    rec = control_conv_readout(refs, load_done([RB]), [ARM_BY_NAME["RB-bare"], ARM_BY_NAME["RB-critical"]], SEEDS, TIERS)
    committed = json.loads((REF / "READOUT-control-conv-3000steps.json").read_text())
    for key in ("stats", "per_seed", "paired_tests", "gate_readout"):
        assert rec[key] == committed[key], key
    assert sum(v == 1.0 for v in vals) == 4 and sum(v < 0.11 for v in vals) == 4     # the record's bimodality, as read


def test_the_convolved_control_provenance_hashes_match_the_committed_files():
    import hashlib
    prov = json.loads((ROOT / "PROVENANCE-control-conv.json").read_text())
    for rel, sha in prov["sha256_16"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:16] == sha, rel
    log = ROOT / "logs" / "control-conv-3000.log"
    assert hashlib.sha256(log.read_bytes()).hexdigest()[:16] == prov["logs_retained_in_repo"]["control_conv"]["sha256_16"]


def test_the_sixteen_seed_record_rows_33_to_35_is_reproducible_from_its_units():
    """Rev62 (declared before the run): the horizon, the eight new seeds of the three references
    and the two decisive arms, the sixteen seed aggregates and readout."""
    from cortex_eval.resumable_rb import REF_CONTROL_CONV, control_conv_readout
    S16 = SEEDS + [100, 200, 300, 400, 500, 600, 700, 800]
    for d, steps in (("conv6000-ckpt", 6000), ("conv12000-ckpt", 12000)):
        done = load_done([ROOT / d]); assert len(done) == 4 and all(u["steps"] == steps for u in done.values())
        sub = json.loads(next((ROOT / d).glob("SUBSET-*.json")).read_text())
        vals = [done[f"{REF_CONTROL_CONV}__s{s}__kv8_seq128"]["accuracy"] for s in (1337, 7, 42, 11)]
        assert vals == sub["stats"][REF_CONTROL_CONV]["per_seed"] and _mean_ci(vals)["mean"] == sub["stats"][REF_CONTROL_CONV]["mean"]
    s16 = ROOT / "ref3000-s16-ckpt"; rb16 = ROOT / "rb3000-s16-ckpt"
    refs_new = load_done([s16]); assert len(refs_new) == 24
    sub = json.loads(next(s16.glob("SUBSET-*.json")).read_text())
    for path in ("L3-+local-conv", "ARCH-none", REF_CONTROL_CONV):
        vals = [refs_new[f"{path}__s{s}__kv8_seq128"]["accuracy"] for s in S16[8:]]
        assert vals == sub["stats"][path]["per_seed"]
    refs_all = load_done([SIG, REF, s16]); assert len(refs_all) == 51
    arms = [ARM_BY_NAME["RB-bare"], ARM_BY_NAME["RB-critical"]]
    done8 = load_done([rb16]); assert len(done8) == 16
    for name, done, seeds in (("LATEST-rb.json", done8, S16[8:]), ("LATEST-rb-16seeds.json", load_done([RB, rb16]), S16)):
        rec = aggregate(done, refs_all, arms, seeds, TIERS); committed = json.loads((rb16 / name).read_text())
        for key in ("stats", "per_seed", "references", "paired_tests", "gate_readout"):
            assert rec[key] == committed[key], (name, key)
    rr = control_conv_readout(refs_all, load_done([RB, rb16]), arms, S16, TIERS)
    committed = json.loads((s16 / "READOUT-control-conv-3000steps-16seeds.json").read_text())
    for key in ("stats", "per_seed", "paired_tests", "gate_readout"):
        assert rr[key] == committed[key], key
    for p in rb16.glob("unit-*.json"):
        u = json.loads(p.read_text()); assert u["arm_hash"] == ARM_BY_NAME[u["arm"]].config_hash()


def test_the_sixteen_seed_provenance_hashes_match_the_committed_files():
    import hashlib
    prov = json.loads((ROOT / "PROVENANCE-s16.json").read_text())
    for rel, sha in prov["sha256_16"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:16] == sha, rel
    log = ROOT / "logs" / "recurrent-base-s16.log"
    assert hashlib.sha256(log.read_bytes()).hexdigest()[:16] == prov["logs_retained_in_repo"]["s16"]["sha256_16"]
