"""ADR-009, Decision B2 (the founder, 2026-10-05; graved 2026-10-08): every journal entry keeps the
text its address was computed from, sealed like the content, as a payload of its own referenced by
a second pointer in the entry (`source_pointer`). The five tests below were declared in the ADR
before any code was written and the code follows them, in their order:

  1. the round trip      a write with a source text seals it; a read returns it decoded, equal to
                         the byte; without the key it is unreadable, like the content;
  2. the size law        an entry with a source pointer stays under 1024 bytes at the protocol's
                         sizes; a source text of any length is accepted, since it lives in the
                         payload store; a text put inside the entry is refused as today;
  3. persistence         every existing persistence test passes without a change (the suite); a
     unchanged           journal written before this decision opens, declares `source_text: absent`,
                         reads as before, and its rebuild on opening is the one of today;
  4. the Molaison rows   the hash addresses do not change, so the protocol's files (rows 13, 13b)
                         recompute to the digit with the source text present; the record tests run
                         unchanged (the suite);
  5. the lifecycle       an eviction releases the source payload with the content; the phase event
                         carries the count; a reopened scheduler agrees number for number.

Two kinds of entries carry no source pointer by construction, as the dated line of 2026-10-12 under
B2 states: a demotion summary (its address is a centroid of cues, not a text) and a sentinel planted
by the lifecycle (its text is a function of the hashed configuration, regenerated at every open);
section 3 pins that, since it is what keeps the controls of rows 41 and 42 reproducing their files.
Nothing here is a number of the record."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cortex_c2b import (Journal, Entry, ENTRY_MAX_BYTES, CUE_DIM, POLICY_STOP, SCHEMA_SUMMARY,   # noqa: E402
                        STATE_EVICTED, STATE_LIVE, content_hash, lifecycle_declaration)
from cortex_c2b.crypto import generate_key, MAGIC, LINE_PREFIX                                   # noqa: E402
from cortex_c2b.write_path import WritePath                                                       # noqa: E402
from cortex_c2b.read_path import JournalPath                                                      # noqa: E402
from cortex_c2b.lifecycle import LifecycleScheduler, LifecycleConfig                              # noqa: E402
from cortex_c2b import hm_protocol                                                                # noqa: E402
from cortex_c2b.hm_protocol import generate_facts, run_hm_protocol, value_of_statement            # noqa: E402
from cortex_c2b import addressing_probe                                                           # noqa: E402

MQAR = ROOT / "metrics" / "mqar"
# the eight fields an entry had from Slice A (2026-09-08) to the tag v1.0.0: the serialisation of an entry
# without a source text must stay exactly this, so that the logs and sizes of earlier journals do not move
V1_ENTRY_KEYS = ("cue", "pointer", "salience", "schema_id", "t_written", "t_last_read", "state", "entry_id")
V1_APPLIED_KEYS = {"consolidated", "replay_failed", "demoted_groups", "evicted", "withdrawn"}
# the line the v1.0.0 code (`4bb6031`, 2026-10-11) writes for this entry, computed before this decision's code
V1_FIXED_ENTRY_JSON = ('{"cue":[' + ",".join(["0.1"] * 64) + '],"pointer":"0123456789abcdef","salience":0.5,'
                       '"schema_id":"person","t_written":1.0,"t_last_read":1.0,"state":"live","entry_id":"fedcba9876543210"}')


def unit(rng, dim=CUE_DIM):
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def near(base, rng, eps=0.02):
    v = base + eps * rng.standard_normal(base.shape).astype(np.float32)
    return v / np.linalg.norm(v)


def events(path: Path, key=None):
    from cortex_c2b.crypto import open_line
    return [json.loads(open_line(key, l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ============================================================================ 1. the round trip
def test_1_a_write_with_a_source_text_seals_it_and_a_read_returns_it_to_the_byte(tmp_path):
    key = generate_key()
    path = tmp_path / "journal.jsonl"
    j = Journal(path, key=key, policy=POLICY_STOP)
    rng = np.random.default_rng(1)
    source = "person|Kalo-3f2a"                                        # the text the address was computed from
    e = j.write(unit(rng), b"Kalo-3f2a works as a glazier", 0.9, "person", now=1.0, source=source)
    assert e.source_pointer == content_hash(source.encode("utf-8")) and len(e.source_pointer) == 16
    assert e.source_pointer != e.pointer                               # a payload of its own, next to the content
    assert j.source_text(e.entry_id) == source.encode("utf-8")        # decoded, equal to the byte
    assert j.payloads.get(e.pointer) == b"Kalo-3f2a works as a glazier"   # the content, as before
    # sealed at rest, like the content: the file exists under its pointer, framed, and never in clear
    f = tmp_path / "journal.payloads" / e.source_pointer
    assert f.exists() and f.read_bytes().startswith(MAGIC) and source.encode() not in f.read_bytes()
    assert source.encode() not in path.read_bytes() and path.read_text(encoding="utf-8").startswith(LINE_PREFIX)
    # a new object, the disk alone, the key given: the same bytes
    j2 = Journal(path, key=key, policy=POLICY_STOP)
    assert j2.source_text(e.entry_id) == source.encode("utf-8")
    assert j2.get(e.entry_id).source_pointer == e.source_pointer
    assert lifecycle_declaration(j2)["bytes"]["source_text"]["present"] == 1
    # without the key: unreadable, loud, like the content
    with pytest.raises(ValueError):
        Journal(path, plaintext=True)
    with pytest.raises(Exception):
        Journal(path, key=generate_key())


def test_1_bytes_and_str_sources_are_accepted_and_the_write_path_and_the_scheduler_carry_them():
    j = Journal()
    rng = np.random.default_rng(2)
    a = j.write(unit(rng), b"content a", 0.9, "note", now=1.0, source=b"address a")
    b = j.write(unit(rng), b"content b", 0.9, "note", now=2.0, source="address b")
    assert j.source_text(a.entry_id) == b"address a" and j.source_text(b.entry_id) == b"address b"
    c = j.write(unit(rng), b"content c", 0.9, "note", now=3.0)        # no source given: absent, declared
    assert c.source_pointer is None and j.source_text(c.entry_id) is None
    assert j.source_text_counts() == {"present": 2, "absent": 1}
    # through the gated write path and through the scheduler, the same contract
    wp = WritePath(j, seed=0)
    rep = wp.write(unit(rng), b"content d", "note", now=4.0, source="address d")
    assert rep.admitted and j.source_text(rep.entry.entry_id) == b"address d"
    jp = JournalPath(j)
    sch = LifecycleScheduler(j, wp, jp, LifecycleConfig(sentinel_facts=0), now=5.0)
    rep = sch.write(unit(rng), b"content e", "note", now=6.0, source="address e")
    assert rep.admitted and j.source_text(rep.entry.entry_id) == b"address e"


def test_1_when_the_source_is_the_content_itself_the_two_pointers_coincide_and_the_bytes_are_stored_once():
    """The model arm's case: the cue is computed from the statement and the statement is the content."""
    j = Journal()
    rng = np.random.default_rng(3)
    text = "Vorel-3f2a lives in Tallinn"
    e = j.write(unit(rng), text.encode(), 0.9, "person", now=1.0, source=text)
    assert e.source_pointer == e.pointer and len(j.payloads) == 1
    assert j.source_text(e.entry_id) == text.encode() == j.payloads.get(e.pointer)
    assert j.references(e.pointer) == {e.entry_id}


def test_1_the_bridge_of_the_model_arm_keeps_the_text_it_computed_the_cue_from():
    import torch
    import train
    from cortex_c2b.lm_bridge import JournalBridge
    cfg = train.Config(n_layer=2, n_head=2, n_embd=32, block_size=96, journal="read", journal_block=1,
                       journal_heads=2, journal_read_bytes=96, journal_cue_dim=CUE_DIM, journal_k=3)
    torch.manual_seed(0)
    model = train.VanillaGPT(cfg)
    j = Journal()
    bridge = JournalBridge(model, j, k=3, budget_bytes=96, seed=0)
    facts, _ = generate_facts(10, seed=7)
    for i, f in enumerate(facts):
        rep = bridge.write(f.statement, f.schema, now=float(i), entity=f.entity)
        if rep.admitted:
            e = rep.entry
            assert e.source_pointer == e.pointer                       # the cue came from the statement itself
            assert j.source_text(e.entry_id) == f.statement.encode("utf-8")
    assert j.source_text_counts()["absent"] == 0 and j.source_text_counts()["present"] == len(j)


# ============================================================================ 2. the size law
def test_2_an_entry_with_a_source_pointer_stays_under_the_size_law_at_the_protocols_sizes():
    j = Journal()
    wp = WritePath(j, seed=0)
    facts, _ = generate_facts(200, seed=0)
    sizes = []
    for i, f in enumerate(facts):
        rep = wp.write(f.cue, f.statement.encode(), f.schema, now=float(i), source=f.address_text)
        assert rep.admitted
        e = rep.entry
        assert e.source_pointer is not None and len(e.source_pointer) == 16
        assert f.address_text.encode() not in e.to_json().encode()      # never inside the entry
        sizes.append(e.size_bytes())
    assert len(sizes) == 200 and max(sizes) <= ENTRY_MAX_BYTES, max(sizes)
    assert max(sizes) < 720                                             # 618 to 661 bytes with one pointer; a second pointer adds 37


def test_2_a_source_text_of_any_length_is_accepted_and_the_entry_does_not_grow_with_it():
    j = Journal()
    c = unit(np.random.default_rng(4))
    short = j.write(c, b"x", 0.5, "note", now=1.0, source="short")
    long_text = ("a very long source text " * 5000).encode("utf-8")       # 120 000 bytes, in the payload store
    long = j.write(c, b"y", 0.5, "note", now=2.0, source=long_text)
    assert j.source_text(long.entry_id) == long_text
    assert long.size_bytes() == short.size_bytes()                      # the same cue: the entry holds a 16 character pointer, whatever the text
    assert long.size_bytes() <= ENTRY_MAX_BYTES


def test_2_a_text_put_inside_the_entry_is_refused_by_the_size_law_as_today():
    j = Journal()
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError, match="size law"):
        j.write(unit(rng), b"x", 0.5, schema_id="the text inside the entry " * 60, now=1.0, source="fine")
    assert len(j) == 0                                                   # no entry: the refusal came before the log and memory, as today


# ============================================================================ 3. persistence unchanged
def test_3_an_entry_without_a_source_text_serialises_exactly_as_at_v1_0_0():
    e = Entry(cue=[0.1] * CUE_DIM, pointer="0123456789abcdef", salience=0.5, schema_id="person",
              t_written=1.0, t_last_read=1.0, entry_id="fedcba9876543210")
    assert e.source_pointer is None
    assert e.to_json() == V1_FIXED_ENTRY_JSON                           # byte for byte the line of 2026-10-11
    assert tuple(e.to_dict()) == V1_ENTRY_KEYS and e.size_bytes() == 409
    assert Entry(**json.loads(e.to_json())) == e                        # a v1.0.0 line replays into an entry
    s = Entry(cue=[0.1] * CUE_DIM, pointer="0123456789abcdef", salience=0.5, schema_id="person",
              t_written=1.0, t_last_read=1.0, entry_id="fedcba9876543210", source_pointer="abcdef0123456789")
    assert tuple(s.to_dict()) == V1_ENTRY_KEYS + ("source_pointer",) and s.size_bytes() == 409 + len(',"source_pointer":"abcdef0123456789"')


def test_3_a_journal_written_before_this_decision_opens_declares_absent_and_reads_as_before(tmp_path):
    path = tmp_path / "journal.jsonl"
    j = Journal(path, plaintext=True)                                   # a declared test scope (D8)
    rng = np.random.default_rng(6)
    cues = [unit(rng) for _ in range(5)]
    ids = [j.write(c, f"episode {i}".encode(), 0.7, "note", now=float(i)).entry_id for i, c in enumerate(cues)]
    raw = path.read_text(encoding="utf-8")
    assert "source_pointer" not in raw                                  # the log of a journal written without source texts is the v1.0.0 log
    for ev in events(path):
        assert tuple(ev["entry"]) == V1_ENTRY_KEYS
    r = Journal(path, plaintext=True)                                   # opened as an existing journal
    assert lifecycle_declaration(r)["bytes"]["source_text"] == {
        "what": "the text each address was computed from, one sealed payload per entry, released with the content (ADR-009, B2)",
        "present": 0, "absent": 5}
    assert r.source_text_counts() == {"present": 0, "absent": 5}
    for i, (eid, c) in enumerate(zip(ids, cues)):
        assert r.source_text(eid) is None and r.get(eid).source_pointer is None
        assert r.payloads.get(r.get(eid).pointer) == f"episode {i}".encode()   # reads as before
        assert eid in {h.entry_id for h in r.read_by_cue(c)}
    assert {k: (e.state, e.salience) for k, e in j._entries.items()} == {k: (e.state, e.salience) for k, e in r._entries.items()}
    assert {p: set(h) for p, h in j._refs.items()} == {p: set(h) for p, h in r._refs.items()}


@pytest.mark.parametrize("withdraw", [False, True])                     # the configuration of rows 36 to 41, and the default since Decision 11
def test_3_sentinels_and_summaries_carry_no_source_pointer_and_a_rebuild_writes_what_it_wrote_before(tmp_path, withdraw):
    key = generate_key()
    path = tmp_path / "journal.jsonl"
    cfg = LifecycleConfig(period_writes=6, demote_k=3, demote_min_cosine=0.85, bytes_setpoint=20_000,
                          withdraw_at_eviction=withdraw)
    j = Journal(path, key=key)
    wp, jp = WritePath(j, seed=0), JournalPath(j)
    sch = LifecycleScheduler(j, wp, jp, cfg, now=0.0)
    assert len(sch.sentinels) == cfg.sentinel_facts
    for eid, _ in sch.sentinels:                                         # the sentinel: a function of the hashed configuration
        assert j.get(eid).source_pointer is None
    rng = np.random.default_rng(1)                                        # the drive of invariant 1: the cluster demotes
    base = unit(rng)
    for i in range(60):                                                  # writes without a source text, as every run of the record
        c = near(base, rng, eps=0.05) if i % 2 == 0 else unit(rng)
        sch.write(c, f"payload {i} {rng.integers(1_000_000)}".encode(), "note", now=100.0 + i)
    summaries = [e for e in j._entries.values() if e.schema_id == SCHEMA_SUMMARY]
    assert summaries, "the near duplicate cluster must have been demoted"
    assert all(e.source_pointer is None for e in summaries)             # the summary: a centroid of cues, no text
    assert j.source_text_counts()["present"] == 0 and j.sourced_entries == 0
    for ev in events(path, key):
        if ev["ev"] == "write":
            assert tuple(ev["entry"]) == V1_ENTRY_KEYS                   # the log holds the v1.0.0 lines
        if ev["ev"] == "phase" and not ev.get("refused"):
            assert set(ev["applied"]) <= V1_APPLIED_KEYS                  # no count of released source texts: nothing to count
            assert "source_released" not in ev["applied"]
    before = {"memory": len(sch.memory), "phases": sch.phases, "snapshot": j.snapshot(), "ca3": len(wp.ca3._codes)}
    r = Journal(path, key=key)                                           # the rebuild on opening, as today
    wp2, jp2 = WritePath(r, seed=0), JournalPath(r)
    sch2 = LifecycleScheduler(r, wp2, jp2, cfg)
    assert len(sch2.memory) == before["memory"] and sch2.phases == before["phases"]
    assert r.snapshot() == before["snapshot"] and len(wp2.ca3._codes) == before["ca3"]
    assert lifecycle_declaration(r)["bytes"]["source_text"]["present"] == 0
    rec = sch2.phase(now=500.0)
    assert not rec["refused"] and "source_released" not in rec["applied"]   # a continued journal of the record stays silent on it


# ============================================================================ 4. the Molaison rows
ROW13 = MQAR / "hm-protocol-2026-09-08.json"
ROW13B = {0: MQAR / "hm-protocol-persistent-2026-09-12.json",
          1: MQAR / "hm-protocol-persistent-2026-09-14-seed1.json",
          2: MQAR / "hm-protocol-persistent-2026-09-14-seed2.json",
          3: MQAR / "hm-protocol-persistent-2026-09-14-seed3.json"}
SCALARS = ("hm_recall_on", "hm_recall_off", "hm_gap", "hm_negctrl_rate", "hm_negctrl_on", "hm_negctrl_off_raw",
           "hm_negctrl_off_above_chance", "hm_skill_delta", "run_valid", "hm_dissociation_pass")


def _spy_on_writes(monkeypatch) -> list:
    seen: list = []
    original = WritePath.write

    def spy(self, cue, payload, schema_id, now=None, source=None):
        seen.append((payload, source))
        return original(self, cue, payload, schema_id, now=now, source=source)
    monkeypatch.setattr(WritePath, "write", spy)
    return seen


def test_4_the_protocol_plants_every_fact_with_its_address_text_and_row_13_recomputes_to_the_digit(monkeypatch):
    seen = _spy_on_writes(monkeypatch)
    r = run_hm_protocol(seed=0)
    facts, _ = generate_facts(200, seed=0)
    assert len(seen) == 200
    for f, (payload, source) in zip(facts, seen):
        assert payload == f.statement.encode() and source == f.address_text == f"{f.schema}|{f.entity}"
    file = json.loads(ROW13.read_text(encoding="utf-8"))
    for k in SCALARS:
        assert r[k] == file[k], k
    assert r["generator"]["config_hash"] == file["generator"]["config_hash"] == "a5e22c70a3275ed4"
    assert r["generator"]["admitted"] == file["generator"]["admitted"] == 200
    assert r["thresholds"] == file["thresholds"]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_4_row_13b_recomputes_to_the_digit_with_the_source_text_present_on_disk(tmp_path, seed):
    key = generate_key()
    path = tmp_path / "hm-journal.jsonl"
    r = run_hm_protocol(seed=seed, journal_path=path, key=key)
    file = json.loads(ROW13B[seed].read_text(encoding="utf-8"))
    for k in SCALARS + ("persistent", "claimable"):
        assert r[k] == file[k], k
    assert r["storage"] == file["storage"]
    assert r["generator"]["config_hash"] == file["generator"]["config_hash"] and r["generator"]["admitted"] == 200
    # the source text is present, sealed, for every planted fact: reopened from the disk alone
    j = Journal(path, key=key, policy=POLICY_STOP)
    assert lifecycle_declaration(j)["bytes"]["source_text"] == {
        "what": "the text each address was computed from, one sealed payload per entry, released with the content (ADR-009, B2)",
        "present": 200, "absent": 0}
    for e in j._entries.values():
        schema, entity, _ = value_of_statement(j.payloads.get(e.pointer).decode("utf-8"))
        assert j.source_text(e.entry_id) == f"{schema}|{entity}".encode("utf-8")


def test_4_the_addressing_probe_plants_with_the_address_text_and_its_rates_do_not_move(monkeypatch):
    seen = _spy_on_writes(monkeypatch)
    out = addressing_probe.run(n=200, seed=0)
    assert len(seen) == 200 and all(source == f"{value_of_statement(p.decode())[0]}|{value_of_statement(p.decode())[1]}" for p, source in seen)
    assert (out["rates"]["by_planted_address"], out["rates"]["by_question_text"], out["rates"]["by_paraphrase"],
            out["rates"]["by_random_cue"], out["rates"]["by_random_line_attribute"]) == pytest.approx((1.0, 0.035, 0.035, 0.020, 0.030), abs=1e-12)


# ============================================================================ 5. the lifecycle
def _sourced_journal(path: Path, key: bytes, cfg: LifecycleConfig, n: int = 120, seed: int = 8):
    j = Journal(path, key=key)
    wp, jp = WritePath(j, seed=0), JournalPath(j)
    sch = LifecycleScheduler(j, wp, jp, cfg, now=0.0)
    rng = np.random.default_rng(seed)
    sources = {}
    for i in range(n):
        rep = sch.write(unit(rng), f"payload {i} {rng.integers(1_000_000)}".encode(), "note", now=100.0 + i,
                        source=f"note|{i}")
        if rep.admitted:
            sources[rep.entry.entry_id] = f"note|{i}".encode()
    return j, wp, jp, sch, sources


def test_5_an_eviction_releases_the_source_payload_with_the_content_and_the_phase_event_carries_the_count(tmp_path):
    key = generate_key()
    path = tmp_path / "journal.jsonl"
    cfg = LifecycleConfig(period_writes=8, bytes_setpoint=8_000, decay=0.7)       # the drive of invariant 2: the budget evicts
    j, wp, jp, sch, sources = _sourced_journal(path, key, cfg)
    evicted = [eid for eid in sources if j.get(eid).state == STATE_EVICTED]
    alive = [eid for eid in sources if j.get(eid).state != STATE_EVICTED]
    assert evicted and alive, "the budget must have evicted some sourced entries and kept others"
    for eid in evicted:
        e = j.get(eid)
        assert e.source_pointer is not None and e.source_pointer not in j.payloads       # the bytes left with the content
        assert e.pointer not in j.payloads
        assert not (tmp_path / "journal.payloads" / e.source_pointer).exists()
        assert j.source_text(eid) is None                                                  # declared loss: the address text too
    for eid in alive:
        assert j.source_text(eid) == sources[eid]
    phases = [ev for ev in events(path, key) if ev["ev"] == "phase" and not ev.get("refused")]
    assert phases
    assert all("source_released" in p["applied"] for p in phases)                          # the journal holds source texts: every phase says
    assert sum(p["applied"]["source_released"] for p in phases) == len(evicted)           # one distinct text per entry here
    assert sum(p["applied"]["evicted"] for p in phases) >= len(evicted)                   # the sentinels and summaries evicted count no text
    assert lifecycle_declaration(j)["bytes"]["source_text"]["present"] == len(alive)
    assert j.source_text_counts()["absent"] == sum(1 for e in j._entries.values()
                                                   if e.state != STATE_EVICTED and e.source_pointer is None)   # sentinels, summaries


def test_5_a_reopened_scheduler_agrees_number_for_number_under_decision_11(tmp_path):
    key = generate_key()
    path = tmp_path / "journal.jsonl"
    cfg = LifecycleConfig(period_writes=8, bytes_setpoint=8_000, decay=0.7)
    assert cfg.withdraw_at_eviction is True                                                # Decision 11, the default
    j, wp, jp, sch, sources = _sourced_journal(path, key, cfg)
    before = {"memory": len(sch.memory), "phases": sch.phases, "snapshot": j.snapshot(),
              "declaration": lifecycle_declaration(j)["bytes"]["source_text"], "counts": j.source_text_counts(),
              "refs": {p: set(h) for p, h in j._refs.items()}, "sourced": j.sourced_entries,
              "texts": {eid: j.source_text(eid) for eid in sources}, "sentinel": sch.sentinel_probe(),
              "writes_since_phase": sch.writes_since_phase}
    r = Journal(path, key=key)                                                             # the disk alone
    wp2, jp2 = WritePath(r, seed=0), JournalPath(r)
    sch2 = LifecycleScheduler(r, wp2, jp2, cfg)
    assert len(sch2.memory) == before["memory"] and sch2.phases == before["phases"]
    assert r.snapshot() == before["snapshot"]
    assert lifecycle_declaration(r)["bytes"]["source_text"] == before["declaration"] and r.source_text_counts() == before["counts"]
    assert {p: set(h) for p, h in r._refs.items()} == before["refs"] and r.sourced_entries == before["sourced"]
    assert {eid: r.source_text(eid) for eid in sources} == before["texts"]
    assert sch2.sentinel_probe() == before["sentinel"] and sch2.writes_since_phase == before["writes_since_phase"]
    rec = sch2.phase(now=10_000.0)                                                         # continuity: the next sleep applies and counts
    assert not rec["refused"] and "source_released" in rec["applied"]


def test_5_a_shared_source_text_is_released_at_its_last_reference_only():
    j = Journal()
    rng = np.random.default_rng(9)
    a = j.write(unit(rng), b"first content", 0.9, "note", now=1.0, source="same address text")
    b = j.write(unit(rng), b"second content", 0.9, "note", now=2.0, source="same address text")
    assert a.source_pointer == b.source_pointer and len(j.payloads) == 3                    # two contents, one text
    assert j.references(a.source_pointer) == {a.entry_id, b.entry_id}
    j.transition(a.entry_id, STATE_EVICTED)
    assert a.pointer not in j.payloads and a.source_pointer in j.payloads                   # the content left; the text is still held
    assert j.source_text(a.entry_id) is None and j.source_text(b.entry_id) == b"same address text"
    j.transition(b.entry_id, STATE_EVICTED)
    assert a.source_pointer not in j.payloads and j.references(a.source_pointer) == set()
    # the model arm's case: one payload for the content and the text, released once, at the eviction
    text = "Vorel-3f2a lives in Tallinn"
    c = j.write(unit(rng), text.encode(), 0.9, "person", now=3.0, source=text)
    assert len(j.payloads) == 1
    j.transition(c.entry_id, STATE_EVICTED)
    assert len(j.payloads) == 0 and j.source_text(c.entry_id) is None
