"""C2b Slice D — the H.M. protocol (the milestone).

The generator is leakage-proof and config-hashed; the journal never hallucinates
(negctrl ON = 0); the dissociation holds against the C2b organ (PASS); and the
frozen thresholds are exactly those of the spec.
"""
from cortex_c2b.hm_protocol import (run_hm_protocol, generate_facts, _entity_name,
                                     DELTA, EPSILON_S, FLOOR_MARGIN, NEGCTRL_INVALID, SCHEMAS)
import numpy as np, re


def test_thresholds_are_the_frozen_spec():
    assert DELTA == 0.50 and EPSILON_S == 0.01 and FLOOR_MARGIN == 0.05 and NEGCTRL_INVALID == 0.10


def test_generator_is_leakage_proof_and_balanced():
    facts, cfg = generate_facts(200, seed=0)
    assert len(facts) == 200 and len(cfg) == 16
    # names follow {trigrams}-{4 hex}: never a dictionary word
    assert all(re.fullmatch(r"[A-Z][a-z]{3}-[0-9a-f]{4}", f.entity) for f in facts)
    # balanced across the 5 schemas
    for s in SCHEMAS:
        assert sum(f.schema == s for f in facts) == 40
    # seeded: same seed → same facts and hash
    f2, cfg2 = generate_facts(200, seed=0)
    assert cfg == cfg2 and [f.entity for f in facts] == [f.entity for f in f2]


def test_journal_never_hallucinates_never_planted_facts():
    r = run_hm_protocol(seed=0)
    assert r["hm_negctrl_on"] == 0.0


def test_dissociation_passes_against_the_organ():
    r = run_hm_protocol(seed=0)
    assert r["run_valid"] is True
    assert r["hm_recall_on"] >= 0.95
    assert r["hm_recall_off"] <= r["thresholds"]["floor"]
    assert r["hm_gap"] >= DELTA
    assert r["hm_skill_delta"] <= EPSILON_S
    assert r["hm_dissociation_pass"] == 1


def test_pass_is_stable_across_seeds():
    for s in (1, 2, 3):
        assert run_hm_protocol(seed=s)["hm_dissociation_pass"] == 1
