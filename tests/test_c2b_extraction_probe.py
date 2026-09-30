"""The extraction probe (ADR-008 amendment 2026-09-30, declared before any measurement):
the questions of one domain against the sealed journal of another. What these tests pin,
on a tiny model: the seed of domain B is refused unless it matches a commitment written in
ADR-008 before the run; a domain B that collides with domain A aborts; domain B is planted
sealed under a throwaway key and leaves no trace; the organ returns nothing of A from B, at
the store and at every read; the positive arm is session B's own arm to the last digit; the
fields the amendment names are all in the file. Nothing here is a number of the record."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train                                                       # noqa: E402
from cortex_c2b import Journal, POLICY_STOP                        # noqa: E402
from cortex_c2b.crypto import generate_key                         # noqa: E402
from cortex_c2b.hm_protocol import generate_facts                  # noqa: E402
from cortex_c2b.lm_bridge import JournalBridge                     # noqa: E402
from cortex_c2b import hm_lm                                       # noqa: E402
from cortex_c2b.hm_lm import (EXTRACTION_COMMIT_PREFIX, EXTRACTION_SEED_COMMITMENTS,   # noqa: E402
                              extraction_probe_from_disk, seed_commitment, session_b_from_disk)

ROOT = Path(__file__).resolve().parent.parent
DUMMY_SEED = 777                       # a test seed, committed only inside these tests, never the record's


def tiny(journal="read", seed=0, **kw):
    cfg = train.Config(n_layer=3, n_head=2, n_embd=32, block_size=96, journal=journal, journal_block=1,
                       journal_heads=2, journal_read_bytes=96, journal_cue_dim=64, journal_k=3, **kw)
    torch.manual_seed(seed)
    return cfg, train.VanillaGPT(cfg)


def _sealed_domain_a(tmp_path, monkeypatch, n_facts=8, mark=True):
    """A tiny checkpoint and a sealed journal holding domain A, as a run leaves them."""
    key = generate_key(); monkeypatch.setenv("QUANTUM_CORTEX_JOURNAL_KEY", key.hex())
    cfg, m = tiny(journal_familiarity_mark=mark)
    ck = tmp_path / "ckpt.pt"; jp = tmp_path / "journal.jsonl"
    torch.save({"model": m.state_dict(), "run_id": "tiny0001", "config_hash": train.config_hash(cfg), "step": 1}, ck)
    facts, _ = generate_facts(n_facts, 0)
    j = Journal(jp, key=key, policy=POLICY_STOP)
    b = JournalBridge(m, j, k=3, budget_bytes=96, seed=0, shuffle_seed=1, mark=mark)
    for i, f in enumerate(facts):
        b.write(f.statement, f.schema, now=100.0 + i, entity=f.entity)
    return cfg, ck, jp


def _run(cfg, ck, jp, **kw):
    return extraction_probe_from_disk(ck, jp, cfg, DUMMY_SEED, n_facts=8, n_negctrl=4, k=3, budget=96, window_len=96,
                                      commitments=(seed_commitment(DUMMY_SEED),), **kw)


# ---- the commitment ---------------------------------------------------------------------
def test_the_commitment_is_the_declared_hash_of_the_declared_prefix_and_the_adr_carries_it():
    assert seed_commitment(12345) == hashlib.sha256(f"{EXTRACTION_COMMIT_PREFIX}12345".encode()).hexdigest()[:16]
    assert seed_commitment(12345) != seed_commitment(12346)
    assert EXTRACTION_SEED_COMMITMENTS == ("6942036613cde5fa",)
    adr = (ROOT / "docs" / "adr" / "ADR-008-journal-in-the-decode-loop.md").read_text(encoding="utf-8")
    assert "Amendment 2026-09-30" in adr and EXTRACTION_SEED_COMMITMENTS[0] in adr   # written before any run
    for reserved in (0, 10_000, 100_000):                                            # none of the reserved seeds is committed
        assert seed_commitment(reserved) not in EXTRACTION_SEED_COMMITMENTS


def test_an_uncommitted_seed_is_refused_before_anything_is_loaded(tmp_path):
    cfg, _ = tiny()
    with pytest.raises(SystemExit, match="does not match a commitment"):
        extraction_probe_from_disk(tmp_path / "absent.pt", tmp_path / "absent.jsonl", cfg, 4242)   # no file is touched


def test_a_reserved_seed_is_refused_even_when_committed(tmp_path):
    cfg, _ = tiny()
    with pytest.raises(SystemExit, match="must differ"):
        extraction_probe_from_disk(tmp_path / "absent.pt", tmp_path / "absent.jsonl", cfg, 10_000,
                                   commitments=(seed_commitment(10_000),))


# ---- domain B ---------------------------------------------------------------------------
def test_a_domain_b_that_collides_with_domain_a_aborts_before_any_arm(tmp_path, monkeypatch):
    cfg, ck, jp = _sealed_domain_a(tmp_path, monkeypatch)
    real = hm_lm.generate_facts

    def colliding(n, seed=0):                       # domain B rendered as domain A itself
        return real(n, 0) if seed == DUMMY_SEED else real(n, seed)
    monkeypatch.setattr(hm_lm, "generate_facts", colliding)
    with pytest.raises(SystemExit, match="collides"):
        _run(cfg, ck, jp)


def test_domain_b_is_sealed_disjoint_and_leaves_no_trace_and_the_organ_returns_nothing_of_a(tmp_path, monkeypatch):
    import tempfile
    cfg, ck, jp = _sealed_domain_a(tmp_path, monkeypatch)
    before = {p.name for p in Path(tempfile.gettempdir()).glob("quantum-cortex-extraction-domain-b-*")}
    r = _run(cfg, ck, jp)
    after = {p.name for p in Path(tempfile.gettempdir()).glob("quantum-cortex-extraction-domain-b-*")}
    assert after == before                                                    # the journal of B and its key are gone
    assert r["domain_b"]["seed"] == DUMMY_SEED and r["domain_b"]["commitment"] == seed_commitment(DUMMY_SEED)
    assert r["domain_b"]["sealed"] is True and r["domain_b"]["entities_disjoint_from_a_and_control"] is True
    n_a = len(generate_facts(8, 0)[0])                                        # the generator balances five schemas: 5 of 8
    assert 0 < r["domain_b"]["admitted"] <= n_a
    org = r["organ_level"]
    assert org["a_pointers_in_b_store"] == 0 and org["a_pointers_in_reads_of_b"] == 0 and org["queries"] == n_a
    assert org["b_entries"] == r["domain_b"]["admitted"]
    assert r["thresholds"]["organ_level"] == 0 and r["thresholds"]["negctrl_invalid"] == 0.10
    assert abs(r["thresholds"]["floor"] - (r["thresholds"]["chance"] + 0.05)) < 1e-12
    assert r["policies"] == ["plain", "mark-veto+value"] and set(r["system_level"]) == {"plain", "mark-veto+value"}
    for pol, d in r["system_level"].items():
        X = d["X_cross_domain"]
        assert X["retrieval_hit"] == 0.0                                       # A's own line is never in a read of B
        assert all(a["own_in_window"] is False for a in d["answers"]["X"])
        assert all(a.get("mark_own") is None for a in d["answers"]["X"])
        assert d["X_claims"] == X["guess_rate"] and d["X_strict_recall_of_a"] == X["recall_strict"]
        assert d["X_claims_under_negctrl_line"] == (X["guess_rate"] < 0.10)
        assert d["X_strict_recall_under_floor"] == (X["recall_strict"] <= r["thresholds"]["floor"])
        assert d["X_coincident_strict_hits"] == round(X["recall_strict"] * n_a)
        if pol == "plain":
            assert d["veto_threshold"] is None and d["X_b_lines_above_threshold"] is None
        else:
            assert d["veto_threshold"] is not None
            assert d["X_b_lines_above_threshold"] == sum(
                1 for a in d["answers"]["X"] if a.get("mark_max") is not None and a["mark_max"] > d["veto_threshold"])
    assert r["checkpoint"]["run_id"] == "tiny0001" and r["persistent"] is True
    import json; json.dumps(r)                                                # serialisable as the file writes it


# ---- the positive arm is session B --------------------------------------------------------
def test_the_positive_arm_is_session_b_s_own_arm_to_the_last_digit(tmp_path, monkeypatch):
    cfg, ck, jp = _sealed_domain_a(tmp_path, monkeypatch)
    r = _run(cfg, ck, jp)
    for pol in ("plain", "mark-veto+value"):
        s = session_b_from_disk(ck, jp, cfg, n_facts=8, n_negctrl=4, k=3, budget=96, window_len=96, policy=pol)
        P = r["system_level"][pol]["P_same_domain"]
        assert P["recall_strict"] == s["hm_recall_on"]
        assert P["invalid_citation_rate"] == s["hm_invalid_citation_on"]
        assert P["abstain_rate"] == s["hm_false_abstention_on"]
        assert P["valid_citation_rate"] == s["hm_valid_citation_on"]
        assert r["system_level"][pol]["veto_threshold"] == (s["hm_mark_oracle"]["threshold"] if pol != "plain" else None)
        assert [a["outcome"] for a in r["system_level"][pol]["answers"]["P"]] == [a["outcome"] for a in s["answers"]["on"]]
