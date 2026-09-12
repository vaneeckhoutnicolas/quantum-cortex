"""The recurrent base hybrid family (Rev38): the no op law against L3 to the bit, the
critical period's closed gate, the consolidation term, the retraction, the phi arm,
determinism, and the resumable runner's declared readout. Mechanics only: no result."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cortex_eval.mqar import MQARTier, make_batch                       # noqa: E402
from cortex_eval.recurrent_ladder import _build                         # noqa: E402
from cortex_eval.recurrent_base import (_build_rb, run_rb_unit, ARMS, ARM_BY_NAME, L3, PHI)   # noqa: E402
from cortex_eval import recurrent_base                                  # noqa: E402

TIER = MQARTier(kv_pairs=4, seq_len=32)


def test_no_op_law_the_hybrid_at_init_is_l3_to_the_bit():
    torch.manual_seed(11); l3 = _build(L3, 32, 2)
    X, _ = make_batch(TIER, 4, seed=1); idx = torch.from_numpy(X)
    for name in ("RB-bare", "RB-critical", "RB-cls", "RB-cls-phi"):
        rb = _build_rb(ARM_BY_NAME[name], 32, 2, seed=11)
        with torch.no_grad():
            a = l3(idx); full, trunk = rb(idx)
        assert torch.equal(a, full) and torch.equal(a, trunk), name
    assert float(_build_rb(ARM_BY_NAME["RB-critical"], 32, 2, 11).gate) == 0.0
    assert float(_build_rb(ARM_BY_NAME["RB-bare"], 32, 2, 11).gate) == 1.0


def test_closed_gate_lets_no_gradient_into_the_attention():
    rb = _build_rb(ARM_BY_NAME["RB-critical"], 32, 2, seed=2)
    X, Y = make_batch(TIER, 4, seed=1)
    full, _ = rb(torch.from_numpy(X))
    torch.nn.functional.cross_entropy(full.view(-1, full.size(-1)), torch.from_numpy(Y).view(-1), ignore_index=-100).backward()
    assert rb.qkv.weight.grad is None and rb.proj.weight.grad is None      # the arm IS L3 while closed
    assert rb.trunk.qkv.weight.grad is not None
    rb.gate.fill_(1.0); rb.zero_grad()
    full, _ = rb(torch.from_numpy(X))
    torch.nn.functional.cross_entropy(full.view(-1, full.size(-1)), torch.from_numpy(Y).view(-1), ignore_index=-100).backward()
    assert rb.proj.weight.grad is not None and rb.proj.weight.grad.abs().sum() > 0


def test_critical_period_opens_on_a_measured_plateau():
    arm = ARM_BY_NAME["RB-critical"]
    fast = type(arm)(**{**arm.__dict__, "name": "t", "plateau_window": 5, "plateau_min_steps": 10, "plateau_tol": 1.0})
    r = run_rb_unit(fast, TIER, steps=20, seed=3, d_model=32, eval_batch=16)    # tolerance 1.0: opens at the first check
    assert r["opened_at"] == 10 and r["gate_final"] == 1.0
    never = type(arm)(**{**arm.__dict__, "name": "n", "plateau_window": 5, "plateau_min_steps": 10, "plateau_tol": -1.0})
    r = run_rb_unit(never, TIER, steps=20, seed=3, d_model=32, eval_batch=16)   # negative tolerance: never opens
    assert r["opened_at"] is None and r["gate_final"] == 0.0


def test_consolidation_term_pulls_the_trunk_toward_the_full_model():
    rb = _build_rb(ARM_BY_NAME["RB-cls"], 32, 2, seed=4)
    with torch.no_grad():
        rb.proj.weight.normal_(0, 0.5)                                      # a live attention: full != trunk only
    X, Y = make_batch(TIER, 8, seed=2); idx = torch.from_numpy(X); m = torch.from_numpy(Y) != -100
    opt = torch.optim.Adam(rb.trunk.parameters(), lr=3e-3)
    def kl():
        full, trunk = rb(idx)
        return torch.nn.functional.kl_div(trunk[m].log_softmax(-1), full.detach()[m].softmax(-1), reduction="batchmean")
    k0 = float(kl().detach())
    for _ in range(40):
        loss = kl(); opt.zero_grad(); loss.backward(); opt.step()
    assert float(kl().detach()) < 0.5 * k0


def test_retraction_fades_the_gate_only_after_a_take_off(monkeypatch):
    arm = type(ARM_BY_NAME["RB-cls"])(**{**ARM_BY_NAME["RB-cls"].__dict__, "name": "r", "retract_every": 5})
    monkeypatch.setattr(recurrent_base, "_acc", lambda model, tier, seed, batch, which: 0.6 if which == "full" else 0.59)
    r = run_rb_unit(arm, TIER, steps=10, seed=5, d_model=32, eval_batch=8)
    assert r["retractions"] == [5, 10] and abs(r["gate_final"] - 0.25) < 1e-6      # 1/2 twice
    phi = type(ARM_BY_NAME["RB-cls-phi"])(**{**ARM_BY_NAME["RB-cls-phi"].__dict__, "name": "p", "retract_every": 5})
    r = run_rb_unit(phi, TIER, steps=5, seed=5, d_model=32, eval_batch=8)
    assert abs(r["gate_final"] - 1 / PHI) < 1e-4
    monkeypatch.setattr(recurrent_base, "_acc", lambda model, tier, seed, batch, which: 0.2 if which == "full" else 0.19)
    r = run_rb_unit(arm, TIER, steps=10, seed=5, d_model=32, eval_batch=8)
    assert r["retractions"] == [] and r["gate_final"] == 1.0                   # no take off: the fast trace stays


def test_phi_enters_two_places_only_and_the_arms_are_hashed():
    a, b = ARM_BY_NAME["RB-cls"], ARM_BY_NAME["RB-cls-phi"]
    assert abs(b.retraction - 1 / PHI) < 1e-9 and abs(b.attn_frac - 1 / PHI ** 2) < 1e-9
    assert a.retraction == 0.5 and a.attn_frac == 0.5
    assert {k: v for k, v in a.__dict__.items() if k not in ("name", "retraction", "attn_frac")} == \
           {k: v for k, v in b.__dict__.items() if k not in ("name", "retraction", "attn_frac")}
    assert len({arm.config_hash() for arm in ARMS}) == 4
    rb_a, rb_b = _build_rb(a, 32, 2, 1), _build_rb(b, 32, 2, 1)
    assert rb_a.proj.weight.shape[1] == 16 and rb_b.proj.weight.shape[1] == 12       # attention widths 1/2 and 0.382


def test_units_are_deterministic():
    r1 = run_rb_unit(ARM_BY_NAME["RB-bare"], TIER, steps=12, seed=7, d_model=32, eval_batch=16)
    r2 = run_rb_unit(ARM_BY_NAME["RB-bare"], TIER, steps=12, seed=7, d_model=32, eval_batch=16)
    assert r1 == r2 and r1["status"] == "done"


def test_runner_reads_the_ladder_references_and_writes_the_declared_readout(tmp_path):
    ref, rb = tmp_path / "ref", tmp_path / "rb"
    run = subprocess.run([sys.executable, "-m", "cortex_eval.resumable_ladder", "--quick", "--paths", "L3-+local-conv",
                          "ARCH-none", "--ckpt-dir", str(ref)], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr
    assert any(p.name.startswith("SUBSET-") for p in ref.iterdir())
    run = subprocess.run([sys.executable, "-m", "cortex_eval.resumable_rb", "--quick", "--ckpt-dir", str(rb),
                          "--reference-from", str(ref)], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr
    r = json.loads((rb / "LATEST-rb.json").read_text())
    assert r["complete"] and r["references"]["missing"] == [] and len(list(rb.glob("unit-*.json"))) == 8
    for k in ("RB-bare_full_vs_L3", "RB-critical_trunk_vs_L3", "RB-cls_full_vs_control", "RB-cls-phi_vs_RB-cls"):
        assert k in r["paired_tests"] and r["gate_readout"][k] == "held"      # two seeds: never gated
    assert set(r["stats"]) == {"RB-bare", "RB-critical", "RB-cls", "RB-cls-phi"}
