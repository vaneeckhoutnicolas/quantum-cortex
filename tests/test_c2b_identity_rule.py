"""The identity rule at the write and the read side selection (ADR-007, amendment of 2026-10-14, evening; rows 49 and 50),
tests written before the founder's runs, from the amendment's list (section 8). Everything here runs on synthetic cues or a
tiny untrained model: no number below is a number of the record, and none is asserted as one; the tests pin the rule's
mechanics (what it admits, what it refuses, what it stamps), the shadow legacy gate against the legacy gate, the fidelity
of the identity plantings to row 48's on the controls, the decode shapes' cases, the replay's validity conditions, and the
cold reading's exits on constructed files."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train                                                            # noqa: E402
from cortex_c2b import (Journal, POLICY_STOP, content_hash, lifecycle_declaration, ADMISSION_SURPRISE, ADMISSION_IDENTITY,
                        ADMISSION_RULES)                                 # noqa: E402
from cortex_c2b.crypto import generate_key                              # noqa: E402
from cortex_c2b.hm_protocol import generate_facts, value_of_statement, _TEMPLATES, SCHEMAS   # noqa: E402
from cortex_c2b.lm_bridge import JournalBridge                          # noqa: E402
from cortex_c2b.read_path import ADOPTED_GEOMETRY                       # noqa: E402
from cortex_c2b.write_path import WritePath, LegacyShadow               # noqa: E402
from cortex_c2b import near_duplicate_probe as P                        # noqa: E402
from cortex_c2b import near_duplicate_reading as R                      # noqa: E402
from cortex_c2b import hm_lm, record_tools                              # noqa: E402

V9_HASHES = {"configs/local_journal_v9_seed2.json": "d0c0d93ba24555a8", "configs/kaggle_t4_journal_v9_seed3.json": "a727d1cb445afa92",
             "configs/kaggle_t4_journal_v9.json": "36a3dd10bbff84eb"}


def unit(rng, dim=64):
    v = rng.standard_normal(dim).astype(np.float32); return v / np.linalg.norm(v)


def near(v, rng, cos=0.9995):
    return P.pair_at(v, cos, rng)


def tiny(seed=9):
    cfg = train.Config(n_layer=3, n_head=2, n_embd=32, block_size=96, journal="read", journal_block=1,
                       journal_heads=2, journal_read_bytes=96, journal_cue_dim=64, journal_k=3)
    torch.manual_seed(seed)
    m = train.VanillaGPT(cfg); m.eval()
    return cfg, m


def checkpoint(tmp_path, cfg, model):
    ck = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict(), "run_id": "tiny", "config_hash": train.config_hash(cfg), "step": 0}, ck)
    return ck


# ---------------------------------------------------------------- the rule as code
def test_identity_admits_a_near_duplicate_the_surprise_rule_refuses():
    rng = np.random.default_rng(1)
    v = unit(rng); w = near(v, rng, 0.999997)                                # a near duplicate of surprise under 1e-6 after the dentate
    j_s = Journal(policy=POLICY_STOP); wp_s = WritePath(j_s, admission=ADMISSION_SURPRISE)
    j_i = Journal(policy=POLICY_STOP); wp_i = WritePath(j_i, admission=ADMISSION_IDENTITY)
    for wp in (wp_s, wp_i):
        assert wp.write(v, b"a", "s", now=0.0, source="text a").admitted
    r_s = wp_s.write(w, b"b", "s", now=1.0, source="text b")
    r_i = wp_i.write(w, b"b", "s", now=1.0, source="text b")
    assert not r_s.admitted and r_s.reason == "redundant (below admission threshold)" and 0.0 < r_s.surprise < 1e-6
    assert r_i.admitted and r_i.reason == "admitted (identity: different source text)" and abs(r_i.surprise - r_s.surprise) < 1e-9
    assert r_i.admission == ADMISSION_IDENTITY and r_s.admission == ADMISSION_SURPRISE
    # the comparator's verdict kept as provenance on the admitted write, absent under the surprise rule by construction
    assert r_i.near == {"entry_id": r_i.nearest_entry_id, "surprise": r_i.surprise} and r_i.nearest_entry_id == j_i.ca3_order[0][0]
    assert j_i.near_of[r_i.entry.entry_id] == r_i.near and len(j_s.near_of) == 0
    decl = lifecycle_declaration(j_i)
    assert decl["structure"]["admission"]["near_recorded"] == 1 and decl["structure"]["admission"]["rule"] == ADMISSION_IDENTITY
    assert set(decl) == {"storage", "structure", "bytes", "index", "associative_memory"}       # the four components, the rule inside the log's
    # the salience is the surprise on both journals (D4.8 unchanged)
    assert abs(r_i.entry.salience - r_i.surprise) < 1e-12
    # a write without a source text stays under the surprise rule on the identity journal
    r_u = wp_i.write(near(v, rng, 0.99999), b"c", "s", now=2.0)
    assert not r_u.admitted and r_u.reason == "redundant (below admission threshold)"
    # a distinct text whose surprise clears the threshold is admitted without a near stamp (the reason names the rule that applied)
    r_far = wp_i.write(unit(rng), b"d", "s", now=3.0, source="text d")
    assert r_far.admitted and r_far.near is None and r_far.reason == "admitted (identity: different source text)"
    assert wp_s.write(unit(rng), b"e", "s", now=4.0, source="text e").reason == "admitted"


def test_same_text_stays_redundant_and_the_revision_unchanged_under_identity():
    rng = np.random.default_rng(2)
    v = unit(rng)
    j = Journal(policy=POLICY_STOP); wp = WritePath(j, admission=ADMISSION_IDENTITY)
    r1 = wp.write(v, b"content", "s", now=0.0, source="the text")
    r2 = wp.write(v, b"content", "s", now=1.0, source="the text")
    assert r1.admitted and not r2.admitted and r2.reason == "redundant (same address, same content)" and r2.entry.entry_id == r1.entry.entry_id
    r3 = wp.write(near(v, rng), b"other content", "s", now=2.0, source="the text")      # the same text, another cue: the same address
    assert not r3.admitted and r3.revised and r3.versions == 2 and r3.entry.entry_id == r1.entry.entry_id and r3.admission == "revision"
    assert len(j) == 1 and j.get(r1.entry.entry_id).pointer == content_hash(b"other content")
    # no normalisation: a trailing space and a letter case make a second address, admitted whatever the surprise
    r4 = wp.write(v, b"content", "s", now=3.0, source="the text ")
    r5 = wp.write(v, b"content", "s", now=4.0, source="The text")
    assert r4.admitted and r5.admitted and len(j) == 3 and len({e.source_pointer for e in j._entries.values()}) == 3
    # the same two writes under the surprise rule are refused (the identical cue is under the threshold)
    j2 = Journal(policy=POLICY_STOP); wp2 = WritePath(j2, admission=ADMISSION_SURPRISE)
    wp2.write(v, b"content", "s", now=0.0, source="the text")
    assert not wp2.write(v, b"content", "s", now=3.0, source="the text ").admitted
    assert not wp2.write(v, b"content", "s", now=4.0, source="The text").admitted


def test_the_stamp_is_written_kept_and_checked(tmp_path):
    rng = np.random.default_rng(3)
    key = generate_key()
    j = Journal(tmp_path / "j.jsonl", key=key, policy=POLICY_STOP)
    assert j.admission_record() is None and j.admission_record_source() is None
    wp = WritePath(j, admission=ADMISSION_IDENTITY)
    assert j.admission_rule == ADMISSION_IDENTITY and j.admission_record() is None     # bound, not yet recorded: the first write stamps
    v = unit(rng)
    wp.write(v, b"a", "s", now=0.0, source="a")
    wp.write(near(v, rng, 0.99999), b"b", "s", now=1.0, source="b")
    assert j.admission_record() == ADMISSION_IDENTITY and j.admission_record_source() == "stamp"
    lines = [json.loads(__import__("cortex_c2b").open_line(key, raw)) for raw in (tmp_path / "j.jsonl").read_text(encoding="utf-8").splitlines() if raw.strip()]
    writes = [ev for ev in lines if ev["ev"] == "write"]
    assert all(ev["admission"] == ADMISSION_IDENTITY for ev in writes) and "near" in writes[1] and "near" not in writes[0]
    # reopened: the record is read back, None takes it, the same rule binds, the other rule is refused
    j2 = Journal(tmp_path / "j.jsonl", key=key, policy=POLICY_STOP)
    assert j2.admission_record() == ADMISSION_IDENTITY and j2.admission_record_source() == "stamp" and len(j2.near_of) == 1
    assert WritePath(j2, admission=None).admission == ADMISSION_IDENTITY
    assert WritePath(Journal(tmp_path / "j.jsonl", key=key, policy=POLICY_STOP), admission=ADMISSION_IDENTITY).admission == ADMISSION_IDENTITY
    with pytest.raises(ValueError, match="refused"):
        WritePath(Journal(tmp_path / "j.jsonl", key=key, policy=POLICY_STOP), admission=ADMISSION_SURPRISE)
    # a bound journal refuses a write under the other rule
    with pytest.raises(ValueError, match="refused"):
        j2.write(unit(rng), b"x", salience=1.0, schema_id="s", now=5.0, admission=ADMISSION_SURPRISE)
    # an unstamped journal (every journal of the record to date) opens as legacy and refuses identity: one write event,
    # its stamp removed, the log sealed again under the key
    from cortex_c2b import seal_line
    j3 = Journal(tmp_path / "legacy3.jsonl", key=key, policy=POLICY_STOP)
    e = j3.write(unit(rng), b"l", salience=1.0, schema_id="s", now=0.0)
    evs = [json.loads(__import__("cortex_c2b").open_line(key, l)) for l in (tmp_path / "legacy3.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert evs[0]["admission"] == ADMISSION_SURPRISE                                  # the code's default stamps surprise
    evs[0].pop("admission")
    (tmp_path / "legacy3.jsonl").write_text("".join(seal_line(key, json.dumps(ev, separators=(",", ":"))) + "\n" for ev in evs), encoding="utf-8")
    j4 = Journal(tmp_path / "legacy3.jsonl", key=key, policy=POLICY_STOP)
    assert j4.admission_record() == ADMISSION_SURPRISE and j4.admission_record_source() == "legacy_by_default"
    assert WritePath(j4).admission == ADMISSION_SURPRISE
    with pytest.raises(ValueError, match="refused"):
        WritePath(Journal(tmp_path / "legacy3.jsonl", key=key, policy=POLICY_STOP), admission=ADMISSION_IDENTITY)
    with pytest.raises(ValueError):
        WritePath(Journal(policy=POLICY_STOP), admission="similarity")
    assert ADMISSION_RULES == (ADMISSION_SURPRISE, ADMISSION_IDENTITY)
    # a log carrying two rules is refused at open
    evs = [json.loads(__import__("cortex_c2b").open_line(key, l)) for l in (tmp_path / "j.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    evs[1]["admission"] = ADMISSION_SURPRISE
    (tmp_path / "mixed.jsonl").write_text("".join(seal_line(key, json.dumps(ev, separators=(",", ":"))) + "\n" for ev in evs), encoding="utf-8")
    import shutil
    shutil.copytree(tmp_path / "j.payloads", tmp_path / "mixed.payloads")
    with pytest.raises(ValueError, match="two admission rules"):
        Journal(tmp_path / "mixed.jsonl", key=key, policy=POLICY_STOP)


def test_the_three_v9_configuration_hashes_are_unchanged_and_identity_is_named():
    for path, h in V9_HASHES.items():
        cfg = train.load_config(ROOT / path)
        assert cfg.journal_admission == ADMISSION_SURPRISE and train.config_hash(cfg) == h and train.admission_of(cfg) == ADMISSION_SURPRISE
    cfg = train.load_config(ROOT / "configs/local_journal_v9_seed2.json")
    cfg.journal_admission = ADMISSION_IDENTITY
    assert train.config_hash(cfg) != V9_HASHES["configs/local_journal_v9_seed2.json"] and train.admission_of(cfg) == ADMISSION_IDENTITY
    cfg.journal_admission = "similarity"
    with pytest.raises(ValueError):
        train.admission_of(cfg)


def test_a_run_refuses_to_continue_a_journal_recorded_under_the_other_rule(tmp_path, monkeypatch):
    key = generate_key()
    j = Journal(tmp_path / "run.jsonl", key=key, policy=POLICY_STOP, index_geometry=ADOPTED_GEOMETRY)
    WritePath(j, admission=ADMISSION_IDENTITY).write(unit(np.random.default_rng(4)), b"a", "s", now=0.0, source="a")
    cfg = train.Config(journal="read", journal_path=str(tmp_path / "run.jsonl"), journal_index_tables=16, journal_index_signature="exact")
    monkeypatch.setattr(train, "REPO_ROOT", Path("/"))
    with pytest.raises(SystemExit, match="another rule"):
        train.check_existing_journal_geometry(cfg, key)
    cfg.journal_admission = ADMISSION_IDENTITY
    assert train.check_existing_journal_geometry(cfg, key)["n_tables"] == 16


def test_b0_and_e0d_are_identical_under_both_rules_on_synthetic_cues():
    rng = np.random.default_rng(5)
    cues = [unit(rng) for _ in range(60)]                                        # distinct facts: every surprise clears the threshold
    results = {}
    for rule in ADMISSION_RULES:
        j = Journal(policy=POLICY_STOP); wp = WritePath(j, admission=rule, shadow_legacy=True)
        reps = [wp.write(c, f"c{i}".encode(), "s", now=float(i), source=f"text {i}") for i, c in enumerate(cues)]
        results[rule] = [(r.admitted, round(r.surprise, 12), r.entry.pointer, r.entry.cue, r.near, r.legacy_admitted) for r in reps]
    assert results[ADMISSION_SURPRISE] == results[ADMISSION_IDENTITY]
    assert all(a and lg for a, _, _, _, _, lg in results[ADMISSION_IDENTITY]) and all(nr is None for *_, nr, _ in results[ADMISSION_IDENTITY])


def test_the_shadow_gate_reproduces_the_legacy_gate_write_by_write():
    rng = np.random.default_rng(6)
    base = [unit(rng) for _ in range(40)]
    writes = []
    for i, v in enumerate(base):                                                 # originals and near duplicates interleaved, as the arms plant
        writes.append((v, f"o{i}")); writes.append((near(v, rng, 0.9995 if i % 3 else 0.99), f"d{i}"))
    j_s = Journal(policy=POLICY_STOP); wp_s = WritePath(j_s, admission=ADMISSION_SURPRISE)
    j_i = Journal(policy=POLICY_STOP); wp_i = WritePath(j_i, admission=ADMISSION_IDENTITY, shadow_legacy=True)
    legacy, shadow = [], []
    for t, (c, text) in enumerate(writes):
        rs = wp_s.write(c, text.encode(), "s", now=float(t), source=text)
        ri = wp_i.write(c, text.encode(), "s", now=float(t), source=text)
        legacy.append((rs.admitted, rs.surprise, rs.nearest_entry_id)); shadow.append((ri.legacy_admitted, ri.legacy_surprise, ri.legacy_nearest_entry_id))
        assert ri.admitted                                                       # every distinct text admitted under identity
    assert [a for a, _, _ in legacy] == [a for a, _, _ in shadow]
    assert max(abs(x - y) for (_, x, _), (_, y, _) in zip(legacy, shadow)) < 1e-6
    # the shadow's nearest names the same write as the legacy gate's nearest (the entries differ by journal, the texts agree)
    for (_, _, le), (_, _, se) in zip(legacy, shadow):
        lt = j_s.source_text(le) if le else None; st = j_i.source_text(se) if se else None
        assert lt == st
    assert sum(1 for a, _, _ in legacy if not a) > 0                             # the construction did produce refusals
    assert len(j_i) == len(writes) and len(j_s) == sum(1 for a, _, _ in legacy if a)
    # the shadow is asked for on a fresh journal only: the log keeps the rounded cues, not the computed ones the legacy gate judged
    with pytest.raises(ValueError, match="fresh journal"):
        WritePath(j_i, admission=ADMISSION_IDENTITY, shadow_legacy=True)
    assert WritePath(j_i, admission=ADMISSION_IDENTITY).shadow is None                 # the rebuild without the shadow is fine
    # the shadow is off by default and the gate's decisions do not depend on it
    j_n = Journal(policy=POLICY_STOP); wp_n = WritePath(j_n, admission=ADMISSION_IDENTITY)
    for t, (c, text) in enumerate(writes[:10]):
        r = wp_n.write(c, text.encode(), "s", now=float(t), source=text)
        assert r.legacy_admitted is None and r.admitted
    assert wp_n.shadow is None
    # an unsourced write the live gate refuses against its larger memory and the legacy gate admits: the shadow stores its code,
    # so that the next write is judged as the legacy gate would (committee of 2026-10-15, finding 2)
    rng2 = np.random.default_rng(66)
    a = unit(rng2); b = near(a, rng2, 0.9995); c = near(a, rng2, 0.995); d = near(c, rng2, 0.99995)
    j_l = Journal(policy=POLICY_STOP); wp_l = WritePath(j_l, admission=ADMISSION_SURPRISE)
    j_x = Journal(policy=POLICY_STOP); wp_x = WritePath(j_x, admission=ADMISSION_IDENTITY, shadow_legacy=True)
    seq = ((a, "a", "a"), (b, "b", "b"), (c, "c", None), (d, "d", "d"))
    for t, (cue, text, src) in enumerate(seq):
        rl = wp_l.write(cue, text.encode(), "s", now=float(t), source=src); rx = wp_x.write(cue, text.encode(), "s", now=float(t), source=src)
        assert rx.legacy_admitted == rl.admitted and abs(rx.legacy_surprise - rl.surprise) < 1e-6, (t, rl.admitted, rx.legacy_admitted)
    assert len(wp_x.shadow.entry_ids) == len(j_l)
    # the `near` stamp is written under the legacy threshold whatever the live gate's own (finding 10)
    j_t = Journal(policy=POLICY_STOP); wp_t = WritePath(j_t, admission=ADMISSION_IDENTITY, admission_threshold=0.05)
    wp_t.write(a, b"a", "s", now=0.0, source="a"); r = wp_t.write(c, b"c", "s", now=1.0, source="c")
    assert r.admitted and (r.near is not None) == (r.surprise < 0.15)
    # two write paths on one journal with no write yet: the second cannot change the rule (finding 3)
    j_two = Journal(policy=POLICY_STOP); first = WritePath(j_two, admission=ADMISSION_IDENTITY)
    with pytest.raises(ValueError, match="bound to"):
        WritePath(j_two, admission=ADMISSION_SURPRISE)
    assert WritePath(j_two).admission == ADMISSION_IDENTITY                            # None takes the binding
    first.write(a, b"a", "s", now=0.0, source="a")
    assert j_two.admission_record() == ADMISSION_IDENTITY


# ---------------------------------------------------------------- the template sets and the question reader
def test_template_sets_precedence_and_the_question_reader():
    facts, _ = generate_facts(200, 0)
    for f in facts:
        assert P.parse_statement(f.statement) == (f.schema, f.entity, f.attr, "protocol")
        e4 = P.e4_statement(f)
        assert P.parse_statement(e4) == (f.schema, f.entity, f.attr, "E4") and value_of_statement(e4) is None
        assert P.parse_question(f.query) == (f.schema, f.entity, "query") and P.parse_question(f.paraphrase) == (f.schema, f.entity, "paraphrase")
    for seed in (20_000, 100_000):
        for f in generate_facts(200, seed)[0]:
            assert P.parse_statement(f.statement) == (f.schema, f.entity, f.attr, "protocol")
            assert P.parse_statement(P.e4_statement(f)) == (f.schema, f.entity, f.attr, "E4") and value_of_statement(P.e4_statement(f)) is None
    assert P.parse_statement("Tone-4010 spends the week as weaver") is None and P.parse_question("Tone-4010 earns a living how?") is None
    assert set(P.E4_TEMPLATES) == set(SCHEMAS) and all(P.E4_TEMPLATES[s] != _TEMPLATES[s][0] for s in SCHEMAS)
    assert P.move_name("Tone-4010", -1) == "Tone-401f" and P.move_name("Tone-401f", 1) == "Tone-4010" and P.bump_name("Tone-4010") == P.move_name("Tone-4010", 1)


def test_the_near_name_block_and_the_neighbour_questions_are_planned_from_the_names():
    facts, _ = generate_facts(200, 0); neg, _ = generate_facts(50, 10_000)
    protected = {f.entity for f in facts} | {f.entity for f in neg}
    e2 = P.e2_plan(facts, protected)
    plan = P.near_name_plan(facts, protected | {nb for nb in e2.values() if nb})
    assert len(plan["questions"]) + len(plan["dropped"]) == 50 and all(q["entity"] not in protected for q in plan["questions"])
    for q in plan["questions"]:
        assert P.move_name(q["entity"], 1) == q["planted_neighbour"] and P.parse_question(q["query"]) == (q["schema"], q["entity"], "query")
    per = {s: sum(1 for q in plan["questions"] if q["schema"] == s) + sum(1 for d in plan["dropped"] if next(f for f in facts if f.entity == d["planted"]).schema == s) for s in SCHEMAS}
    assert all(v == 10 for v in per.values())
    nq = P.neighbour_questions(facts, e2)
    assert len(nq) == sum(1 for v in e2.values() if v) and all(P.parse_question(q.query) == (q.schema, q.entity, "query") for q in nq)
    assert all(content_hash(q.statement.encode("utf-8")) == content_hash(P.duplicate_statement(f, "E2", e2).encode("utf-8"))
               for q, f in zip(nq, [f for f in facts if e2.get(f.entity)]))


# ---------------------------------------------------------------- the shapes
def _lines(*rows):
    return [{"pointer": p, "mark": m, "text": t} for p, m, t in rows]


def test_r1_four_cases_and_r4_three_outcomes_and_the_fallback():
    q = "what is Tone-4010's job?"
    own = ("own", 0.980, "Tone-4010 works as a cartographer"); dup = ("dup", 0.985, "Tone-4011 works as a cartographer")
    low = ("low", 0.300, "Zuzu-0000 works as a glazier")
    # R1 (1) the cited line is the highest: stands
    assert R.select_citation("R1", "dup", _lines(dup, own, low), 0.5, q)["pointer"] == "dup"
    # (2) another line is higher: changed to it
    r = R.select_citation("R1", "own", _lines(dup, own, low), 0.5, q); assert r["pointer"] == "dup" and r["changed"]
    # (3) a tie within the band with the cited line among the tied: stands
    tie = ("tie", 0.98005, "Zuzu-0001 works as a cooper")
    r = R.select_citation("R1", "own", _lines(tie, own, low), 0.5, q); assert r["pointer"] == "own" and r["tied"] and not r["changed"]
    # (4) a tie among others, the cited line not among them: the first tied in retrieval order
    r = R.select_citation("R1", "low", _lines(tie, own, low), 0.1, q); assert r["pointer"] == "tie" and r["tied"] and r["changed"]
    # R4 (a) no line about the entity: abstain
    r = R.select_citation("R4", "dup", _lines(dup, low), 0.5, q); assert r["pointer"] is None and r["abstain"] and r["candidates"] == 0
    # (b) one value among the candidates: the highest marked candidate
    r = R.select_citation("R4", "dup", _lines(dup, own, low), 0.5, q); assert r["pointer"] == "own" and r["values"] == 1 and r["changed"]
    # (c) two values: a conflict exposed, abstain
    v2 = ("v2", 0.97, "Tone-4010 works as a glazier")
    r = R.select_citation("R4", "own", _lines(dup, own, v2), 0.5, q); assert r["pointer"] is None and r["values"] == 2
    # the question parses under no template: R1
    r = R.select_citation("R4", "own", _lines(dup, own, low), 0.5, "Tone-4010 earns a living how?"); assert r["pointer"] == "dup" and r["fallback"] == "R1"
    # R2 and R3
    assert R.select_citation("R2", "dup", _lines(dup, own, low), 0.5, q)["pointer"] == "own"
    assert R.select_citation("R2", "own", _lines(dup, own, v2), 0.5, q)["pointer"] == "own"           # several match: the model's stands
    assert R.select_citation("R3", "own", _lines(dup, own, low), 0.5, q)["pointer"] is None
    assert R.select_citation("R3", "dup", _lines(dup, own, low), 0.5, q)["pointer"] == "dup"
    # no shape ever creates a claim: an abstention, a malformed output, a vetoed citation keep their outcome
    for sh in R.SHAPES:
        r = R.select_citation(sh, None, _lines(dup, own, low), 0.5, q); assert r["pointer"] is None and not r["changed"]
        r = R.select_citation(sh, "low", _lines(dup, own, low), 0.5, q); assert r["pointer"] == "low" and not r["changed"]   # under the threshold: the veto's domain
    with pytest.raises(ValueError):
        R.select_citation("R9", "own", _lines(own), 0.5, q)


def test_the_entity_is_read_from_the_question_text_with_the_label_blanked():
    f = generate_facts(5, 0)[0][0]
    texts = {"own": f.statement, "dup": P.statement_of(f, f.attr, entity=P.bump_name(f.entity))}
    answer = {"entity": "", "attr": f.attr, "query": f.query, "raw_answered": f.attr, "veto_cited_pointer": "dup", "cited_pointer": "dup",
              "window": [{"label": "A", "pointer": "dup", "mark": 0.99}, {"label": "B", "pointer": "own", "mark": 0.98}], "retrieval_order": ["dup", "own"]}
    r = R.replay_answer(answer, "R4", 0.5, texts, "own", f.attr)
    assert r["cited_pointer"] == "own" and r["cited_pointer_is_own"] is True and r["outcome"] == "strict" and r["label"] == "B"
    r0 = R.replay_answer(answer, "none", 0.5, texts, "own", f.attr)
    assert r0["cited_pointer"] == "dup" and r0["cited_pointer_is_own"] is False and r0["outcome"] == "strict"   # the value is right, the provenance wrong (row 48)


def test_the_replay_mirrors_the_live_label_convention_on_constructed_answers():
    """The live probe keeps the model's label on an abstention by the veto or by the value reader and sets it to None when a
    shape abstains (committee of 2026-10-15, finding 1 of two readers): a vetoed citation, a malformed output with a label and a
    label outside the window keep their outcome and their label under every shape; a shape that abstains gives None; a shape
    that changes the citation gives the new line's label."""
    f = generate_facts(5, 0)[0][0]
    texts = {"own": f.statement, "dup": P.statement_of(f, f.attr, entity=P.bump_name(f.entity)), "low": P.statement_of(generate_facts(5, 0)[0][1], "glazier")}
    window = [{"label": "A", "pointer": "dup", "mark": 0.99}, {"label": "B", "pointer": "own", "mark": 0.98}, {"label": "C", "pointer": "low", "mark": 0.2}]
    base = {"entity": f.entity, "schema": f.schema, "attr": f.attr, "query": f.query, "window": window, "retrieval_order": ["dup", "own", "low"]}
    vetoed = {**base, "kind": "unknown", "label": "C", "answered": None, "outcome": "abstain", "cited_pointer": None, "veto_cited_pointer": None,
              "raw_kind": "cite", "raw_label": "C", "raw_answered": "x", "raw_cited_pointer": "low"}
    malformed = {**base, "kind": "unknown", "label": "B", "answered": None, "outcome": "abstain", "cited_pointer": None, "veto_cited_pointer": None,
                 "raw_kind": "malformed", "raw_label": "B", "raw_answered": None, "raw_cited_pointer": None}
    outside = {**base, "kind": "unknown", "label": "H", "answered": None, "outcome": "abstain", "cited_pointer": None, "veto_cited_pointer": None,
               "raw_kind": "cite", "raw_label": "H", "raw_answered": "x", "raw_cited_pointer": None}
    for a in (vetoed, malformed, outside):
        for sh in R.SHAPES:
            r = R.replay_answer(a, sh, 0.5, texts, "own", f.attr)
            assert (r["kind"], r["label"], r["answered"], r["outcome"], r["cited_pointer"]) == ("unknown", a["label"], None, "abstain", None), (sh, a["label"], r)
    cited = {**base, "kind": "cite", "label": "A", "answered": f.attr, "outcome": "strict", "cited_pointer": "dup", "veto_cited_pointer": "dup",
             "raw_kind": "cite", "raw_label": "A", "raw_answered": f.attr, "raw_cited_pointer": "dup"}
    r = R.replay_answer(cited, "none", 0.5, texts, "own", f.attr); assert (r["kind"], r["label"], r["cited_pointer"], r["outcome"]) == ("cite", "A", "dup", "strict")
    r = R.replay_answer(cited, "R4", 0.5, texts, "own", f.attr); assert (r["kind"], r["label"], r["cited_pointer"], r["cited_pointer_is_own"]) == ("cite", "B", "own", True)
    r = R.replay_answer(cited, "R3", 0.5, texts, "own", f.attr); assert (r["kind"], r["label"], r["cited_pointer"]) == ("cite", "A", "dup")   # the top mark: R3 keeps it
    cited_own = {**cited, "label": "B", "cited_pointer": "own", "veto_cited_pointer": "own", "raw_label": "B", "raw_cited_pointer": "own"}
    r = R.replay_answer(cited_own, "R3", 0.5, texts, "own", f.attr); assert (r["kind"], r["label"], r["cited_pointer"]) == ("unknown", None, None)
    r = R.replay_answer(cited_own, "R1", 0.5, texts, "own", f.attr); assert (r["kind"], r["label"], r["cited_pointer"]) == ("cite", "A", "dup")
    # a file of row 48: no raw fields, no query; the veto surviving citation is the final one and the question is rebuilt from the generator
    old = {k: v for k, v in cited.items() if not k.startswith("raw_") and k not in ("veto_cited_pointer", "query", "retrieval_order")}
    r = R.replay_answer(old, "R4", 0.5, texts, "own", f.attr)
    assert r["cited_pointer"] == "own" and r["question_rebuilt"] and r["retrieval_order_flagged"]
    assert R.question_text_of(old) == (f.query, True)


def test_the_exact_statistics():
    assert abs(R.mcnemar_one_sided_p(6, 0) - 1 / 64) < 1e-12 and R.mcnemar_one_sided_p(5, 0) > 0.025 and R.mcnemar_one_sided_p(0, 0) == 1.0
    lo, hi = R.tango_interval(6, 6, 400); assert hi < 0.02 and abs(lo + hi) < 1e-9
    assert R.tango_interval(7, 7, 400)[1] > 0.02                                    # twelve balanced discordant pairs are the most that close
    assert R.tango_interval(0, 0, 400)[1] < 0.01 and R.tango_interval(10, 0, 400)[0] > 0
    w = R.wilson(91, 92); assert abs(w[0] - 0.941) < 0.001 and abs(w[1] - 0.998) < 0.001
    assert R.wilson(0, 0) is None
    assert R.ROW49_ARMS == ("B0", "E0d", "E1", "E2", "E4") and R.ARMS == ("B0", "E0d", "E1", "E2")


# ---------------------------------------------------------------- the plantings, session B, the replay and the cold reading on a tiny model
@pytest.fixture(scope="module")
def tiny_rows(tmp_path_factory):
    """Five identity arms and four legacy arms planted and read on a tiny model (12 facts, 6 never planted), plus the live R4
    run on E2 identity: the inputs of the readings, built once for the module."""
    tmp_path = tmp_path_factory.mktemp("rows")
    cfg, model = tiny(2)                   # a tiny seed whose untrained encoder admits every distinct fact of B0 and E0d under the surprise
    ck = checkpoint(tmp_path, cfg, model)  # rule (as the record's checkpoints did on row 48) and refuses near duplicates of E1 and E2
    idx = {"n_tables": 16, "n_bits": 10, "signature": "exact"}
    orig = record_tools.git_state
    record_tools.git_state = lambda: ("test0000", True)
    import os
    files = {"identity": {}, "legacy": {}, "live": None, "plantings": {}}
    try:
        for rule, arms in ((ADMISSION_IDENTITY, P.ARMS), (ADMISSION_SURPRISE, ("B0", "E0d", "E1", "E2"))):
            for arm in arms:
                key = generate_key(); jp = tmp_path / f"{rule}-{arm}.jsonl"
                rec = P.plant_arm(model, cfg, jp, key, arm, seed=0, n_facts=12, n_negctrl=6, admission=rule)
                os.environ["QUANTUM_CORTEX_JOURNAL_KEY"] = key.hex()
                r = hm_lm.session_b_from_disk(ck, jp, cfg, n_facts=12, n_negctrl=6, seed=0, k=3, budget=96, window_len=96, policy="mark-veto+value",
                                              index_override=idx, near_duplicate_arm=arm, planting_record=rec, admission=rule)
                r["_sha256"] = "tiny"
                files["identity" if rule == ADMISSION_IDENTITY else "legacy"][arm] = r
                files["plantings"][(rule, arm)] = (jp, key)
        jp, key = files["plantings"][(ADMISSION_IDENTITY, "E2")]
        os.environ["QUANTUM_CORTEX_JOURNAL_KEY"] = key.hex()
        rec = files["identity"]["E2"]["near_duplicates"]["planting"]
        files["live"] = hm_lm.session_b_from_disk(ck, jp, cfg, n_facts=12, n_negctrl=6, seed=0, k=3, budget=96, window_len=96, policy="mark-veto+value",
                                                  index_override=idx, near_duplicate_arm="E2", planting_record=rec, admission=ADMISSION_IDENTITY, decode_shape="R4")
    finally:
        record_tools.git_state = orig
        os.environ.pop("QUANTUM_CORTEX_JOURNAL_KEY", None)
    return files


def test_plantings_under_identity_admit_every_distinct_text_and_record_the_shadow(tiny_rows):
    facts, _ = generate_facts(12, 0)
    for arm, f in tiny_rows["identity"].items():
        plant = f["near_duplicates"]["planting"]
        assert plant["admission"]["rule"] == ADMISSION_IDENTITY and plant["admission"]["source"] == "stamp" and f["admission"]["record"] == ADMISSION_IDENTITY
        ws = [w for w in plant["writes"] if not w.get("dropped")]
        assert all(w["admitted"] for w in ws) and all(w["admission"] == ADMISSION_IDENTITY for w in ws)
        assert all(w["legacy_decision"] is not None and w["legacy_surprise"] is not None for w in ws)
        assert all(w["truth"]["entity"] and w["truth"]["value"] for w in ws)
        assert all(len(w["cue_computed"]) == 64 and len(w["cue_journaled"]) == 64 for w in ws)
        for w in ws:
            assert (w["near"] is not None) == (w["surprise"] < P.ADMISSION_THRESHOLD and w["nearest_entry_id"] is not None)
        assert plant["counts"]["entries"] == len(ws) and plant["admission"]["near_recorded"] == sum(1 for w in ws if w["near"])
        assert plant["admission"]["shadow_legacy"]["judged"] == len(ws)
        if arm in ("E1", "E2"):
            assert plant["counts"]["duplicates_refused"] == 0 and plant["counts"]["duplicates_admitted"] == plant["counts"]["duplicates"]
        if arm == "E4":
            assert plant["counts"]["paraphrases"] == len(facts) and plant["counts"]["paraphrases_refused"] == 0 and plant["counts"]["duplicates"] == 0
            assert all(P.parse_statement(w["text"])[3] == "E4" and w["truth"] == {"entity": w["for_entity"], "value": next(x.attr for x in facts if x.entity == w["for_entity"])}
                       for w in ws if w["kind"] == "E4")
            assert f["near_duplicates"]["partner"] == "paraphrase" and f["near_duplicates"]["duplicates_found"] == len(facts) and f["near_duplicates"]["consistent"]
        assert f["hm_mark_oracle"]["scratch_admission"] == ADMISSION_IDENTITY and f["hm_mark_oracle"]["scratch_admitted"] == f["hm_mark_oracle"]["scratch_facts"]
    # the shadow against the legacy plantings: the same decisions and surprises write by write (P3), all admitted on the controls
    for arm in ("E1", "E2"):
        wi = [w for w in tiny_rows["identity"][arm]["near_duplicates"]["planting"]["writes"] if not w.get("dropped")]
        wl = [w for w in tiny_rows["legacy"][arm]["near_duplicates"]["planting"]["writes"] if not w.get("dropped")]
        assert [w["text"] for w in wi] == [w["text"] for w in wl]
        assert [w["legacy_decision"] for w in wi] == [w["admitted"] for w in wl]
        assert max(abs(a["legacy_surprise"] - b["surprise"]) for a, b in zip(wi, wl)) < 1e-6
    for arm in ("B0", "E0d"):
        assert all(w["legacy_decision"] for w in tiny_rows["identity"][arm]["near_duplicates"]["planting"]["writes"])


def test_session_b_identity_files_carry_the_fields_the_readings_depend_on(tiny_rows):
    f = tiny_rows["identity"]["E2"]
    assert f["admission"]["record"] == ADMISSION_IDENTITY and f["admission"]["argument"] == ADMISSION_IDENTITY and f["decode_shape"] is None
    for arm in ("on", "off", "negctrl"):
        for a in f["answers"][arm]:
            assert set(a) >= {"query", "raw_kind", "raw_label", "raw_answered", "raw_cited_pointer", "veto_cited_pointer"}
            assert P.parse_question(a["query"]) is not None
            if arm != "off":                                                       # an empty window (no candidate) has an empty order
                assert set(a["retrieval_order"]) == {w["pointer"] for w in a["window"]} and len(a["retrieval_order"]) == len(a["window"])
            # the citation surviving the veto is the final one unless the value reader abstained (the off arm has no window)
            if a.get("cited_pointer") is not None:
                assert a["veto_cited_pointer"] == a["cited_pointer"]
    for a in f["answers"]["on"]:
        assert "citation_support" in a and "citation_support_final" in a and "cited_line_kind" in a or a.get("citation_support") is None
    rb = f["reported_blocks"]
    assert set(rb) == {"near_name_never_planted", "neighbour_questions"}
    nn = rb["near_name_never_planted"]
    assert nn["plan"]["n"] == len(nn["answers"]) and nn["plan"]["n"] + len(nn["plan"]["dropped"]) == 10     # 12 asked, the generator balances to 10
    for a in nn["answers"]:
        assert P.move_name(a["entity"], 1) == a["planted_neighbour"] and a["attr"] == "(never planted)"
        assert set(a) >= {"neighbour_in_window", "neighbour_mark", "neighbour_mark_at_or_above_threshold", "cited_neighbour", "window"}
    nq = rb["neighbour_questions"]
    assert nq["n"] == f["near_duplicates"]["duplicates_found"] and len(nq["answers"]) == nq["n"]
    for a in nq["answers"]:
        assert a["original_entity"] and set(a) >= {"original_in_window", "cited_is_original", "both_marks_at_or_above_threshold"}
    # E4's file: the paraphrase is the partner, its citation is correct provenance
    f4 = tiny_rows["identity"]["E4"]
    for a in f4["answers"]["on"]:
        if a.get("cited_is_duplicate"):
            assert a["citation_support"] is True and a["cited_pointer_is_own"] is False
    assert "neighbour_questions" not in f4["reported_blocks"] and "near_name_never_planted" in f4["reported_blocks"]
    # the file name
    assert hm_lm.session_b_file_name("4cfd6f6e1406", "mark-veto+value", {"n_tables": 16, "n_bits": 10, "signature": "exact"}, control=False, near_dup="E4", admission="identity") \
        == "hm-lm-4cfd6f6e1406-near-dup-E4-identity-session-b-index16exact-mark-veto-value.json"
    assert hm_lm.session_b_file_name("4cfd6f6e1406", "mark-veto+value", {"n_tables": 16, "n_bits": 10, "signature": "exact"}, control=False, near_dup="E2", admission="identity", shape="R4") \
        == "hm-lm-4cfd6f6e1406-near-dup-E2-identity-R4-session-b-index16exact-mark-veto-value.json"
    assert hm_lm.session_b_file_name("x", "mark-veto+value", None, control=False, near_dup="E2") == "hm-lm-x-near-dup-E2-session-b-mark-veto-value.json"


def test_the_two_blocks_leave_the_declared_blocks_untouched_and_the_controls_reproduce_the_legacy_files(tiny_rows):
    # P2 on the controls: the identity files reproduce the legacy ones question by question, the entries first
    for arm in ("B0", "E0d"):
        e = R.compare_entries(tiny_rows["identity"][arm], tiny_rows["legacy"][arm]); assert e["ok"], e["differences"][:3]
        q = R.compare_questions(tiny_rows["identity"][arm], tiny_rows["legacy"][arm]); assert q["ok"], q["differences"][:3]
    # the declared blocks of a file with the two reported blocks equal those of the legacy file of the same arm when the gate agrees
    # (B0: the same journal under both rules), so the blocks, which run after, touched nothing of the three
    f_i, f_l = tiny_rows["identity"]["B0"], tiny_rows["legacy"]["B0"]
    for arm in ("on", "off", "negctrl"):
        assert [(a["kind"], a["label"], a["outcome"], a.get("cited_pointer")) for a in f_i["answers"][arm]] == \
               [(a["kind"], a["label"], a["outcome"], a.get("cited_pointer")) for a in f_l["answers"][arm]]
        assert [[(w["label"], w["pointer"]) for w in a.get("window", [])] for a in f_i["answers"][arm]] == \
               [[(w["label"], w["pointer"]) for w in a.get("window", [])] for a in f_l["answers"][arm]]


def test_the_two_blocks_run_after_the_declared_blocks_and_touch_nothing_of_them(tiny_rows, monkeypatch):
    """The same journal read with the two reported blocks patched away: the declared blocks' answers equal field by field those of
    the file read with the blocks (committee of 2026-10-15, finding 6 of the systems reader)."""
    import os
    cfg, model = tiny(2)
    jp, key = tiny_rows["plantings"][(ADMISSION_IDENTITY, "E2")]
    ck = jp.parent / "ckpt.pt"
    monkeypatch.setattr(record_tools, "git_state", lambda: ("test0000", True))
    monkeypatch.setenv("QUANTUM_CORTEX_JOURNAL_KEY", key.hex())
    from cortex_c2b import near_duplicate_probe as probe
    monkeypatch.setattr(probe, "near_name_plan", lambda facts, protected: {"questions": [], "dropped": [], "rule": "patched away"})
    monkeypatch.setattr(probe, "neighbour_questions", lambda facts, plan: [])
    rec = tiny_rows["identity"]["E2"]["near_duplicates"]["planting"]
    r = hm_lm.session_b_from_disk(ck, jp, cfg, n_facts=12, n_negctrl=6, seed=0, k=3, budget=96, window_len=96, policy="mark-veto+value",
                                  index_override={"n_tables": 16, "n_bits": 10, "signature": "exact"}, near_duplicate_arm="E2", planting_record=rec,
                                  admission=ADMISSION_IDENTITY)
    assert r["reported_blocks"]["near_name_never_planted"]["plan"]["n"] == 0 and r["reported_blocks"]["neighbour_questions"]["n"] == 0
    f = tiny_rows["identity"]["E2"]
    for arm in ("on", "off", "negctrl"):
        assert len(r["answers"][arm]) == len(f["answers"][arm])
        for a, b in zip(r["answers"][arm], f["answers"][arm]):
            assert {k: v for k, v in a.items()} == {k: v for k, v in b.items()}, (arm, a["entity"])
    assert r["hm_mark_oracle"]["threshold"] == f["hm_mark_oracle"]["threshold"]


def test_the_replay_validity_conditions_and_the_live_r4_run(tiny_rows):
    files = tiny_rows["identity"]
    for arm, f in files.items():
        v = R.validity_none_reproduces(f); assert v["ok"], (arm, v["differences"][:3])
    live = tiny_rows["live"]
    assert live["decode_shape"] == "R4" and all(a.get("shape") is not None for a in live["answers"]["on"])
    # the generation unchanged: the raw parse and the windows equal the file's without the shape, question by question
    for arm in ("on", "negctrl"):
        for a, b in zip(live["answers"][arm], files["E2"]["answers"][arm]):
            assert (a["raw_kind"], a["raw_label"], a["raw_answered"], a["veto_cited_pointer"]) == (b["raw_kind"], b["raw_label"], b["raw_answered"], b["veto_cited_pointer"])
            assert [(w["label"], w["pointer"]) for w in a["window"]] == [(w["label"], w["pointer"]) for w in b["window"]]
    for a in live["answers"]["on"]:
        sh = a["shape"]
        if a["veto_cited_pointer"] is not None and sh["pointer"] is not None:
            assert a["cited_pointer"] == sh["pointer"] or a["cited_pointer"] is None
        if a["veto_cited_pointer"] is None:
            assert a["cited_pointer"] is None
    # the one function: the live R4 run reproduces the replay's R4 question by question, the inputs compared first
    cmp = R.live_r4_against_replay(live, files["E2"]); assert cmp["ok"], cmp["differences"][:3]
    assert cmp["n_compared"] > 0 and set(cmp["inputs_sha256"]) == {"live", "identity_E2"}


def test_the_cold_reading_of_row_49_and_the_replay_of_row_50_on_the_tiny_files(tiny_rows, monkeypatch):
    monkeypatch.setattr(R, "RUN_IDS", ("tiny",)); monkeypatch.setattr(R, "LIVE_SEEDS", {"tiny": 2}); monkeypatch.setattr(R, "CONCEIVING_SEED", {})
    files = {"tiny": dict(tiny_rows["identity"])}; row48 = {"tiny": dict(tiny_rows["legacy"])}; row47 = {"tiny": tiny_rows["legacy"]["B0"]}
    rd = R.read_row49(files, row48, row47, expected_commit="test0000")
    assert rd["P0"]["ok"] and rd["P1"]["ok"] and rd["P2"]["ok"] and rd["P3"]["ok"], (rd["P0"]["problems"], rd["P1"]["problems"], rd["P2"]["problems"], rd["P3"]["problems"])
    assert rd["F1"]["zero"] and rd["letter"] == "alpha" and rd["decision_12"] == R.DECISION_12
    assert rd["P3"]["per_file"]["tiny:E1"]["ok"] and rd["P3"]["per_file"]["tiny:E2"]["ok"]
    assert rd["E4"]["per_run"]["tiny"]["paraphrases"] == 10 and set(rd["E4"]["per_run"]["tiny"]["per_schema"]) == set(SCHEMAS)
    rep = rd["reported"]
    assert set(rep["tests"]) == {"E1", "E2", "E4"} and "p_two_sided" in rep["tests"]["E2"] and rep["tests"]["E1"]["p"] is None and rep["tests"]["E4"]["tango_95"] is not None
    assert all(k in rep["tribunal_static"] for k in ("tiny:E1", "tiny:E4")) and rep["live_cost"]["tiny:E2"]["n"] == 10
    # the exits: a threshold moved stops at P1; a refused duplicate in the planting stops at F1 (beta); a missing file stops at P0
    import copy
    bad = copy.deepcopy(files); bad["tiny"]["E1"]["hm_mark_oracle"]["threshold"] += 0.01
    rd1 = R.read_row49(bad, row48, row47); assert rd1["letter"] == "gamma" and rd1["stopped_at"] == "P1" and rd1["F1"] is None
    bad = copy.deepcopy(files); w = next(x for x in bad["tiny"]["E2"]["near_duplicates"]["planting"]["writes"] if x["kind"] == "E2"); w["admitted"] = False; w["reason"] = "redundant (below admission threshold)"
    rd2 = R.read_row49(bad, row48, row47); assert rd2["letter"] == "beta" and rd2["stopped_at"] == "F1" and rd2["F1"]["near_duplicates_refused"] == 1
    bad = copy.deepcopy(files); del bad["tiny"]["E4"]
    rd3 = R.read_row49(bad, row48, row47); assert rd3["letter"] == "gamma" and rd3["stopped_at"] == "P0" and any("E4: missing" in p for p in rd3["P0"]["problems"])
    bad = copy.deepcopy(files); bad["tiny"]["B0"]["answers"]["on"][0]["outcome"] = "invalid"
    rd4 = R.read_row49(bad, row48, row47); assert rd4["letter"] == "gamma" and rd4["stopped_at"] == "P2"
    bad = copy.deepcopy(files); bad["tiny"]["E1"]["near_duplicates"]["planting"]["writes"][1]["legacy_decision"] = not bad["tiny"]["E1"]["near_duplicates"]["planting"]["writes"][1]["legacy_decision"]
    rd5 = R.read_row49(bad, row48, row47); assert rd5["letter"] == "gamma" and rd5["stopped_at"] == "P3"
    # row 50: the replay on the identity files, every shape, Decision 13 computed, the post hoc replay on the legacy files
    assert "unread" in R.replay_row50(files, row49_letter="gamma")                 # nothing read before row 49 is alpha
    r50 = R.replay_row50(files, row49_letter=rd["letter"])
    assert r50["valid"] and set(r50["per_file"]) == {f"tiny:{a}" for a in P.ARMS} and r50["decision_13"] is not None
    d13 = r50["decision_13"]
    assert set(d13) >= {"superiority", "no_loss", "adopted", "closure", "per_seed"} and d13["closure"]["n_expected"] == 10
    assert d13["closure"]["n_paired"] + sum(len(v) for v in d13["closure"]["split_per_seed"]["tiny"]["without_window"].values()) >= d13["closure"]["n_paired"]
    assert d13["adopted"] == (d13["superiority"]["holds"] and d13["no_loss"]["holds"])
    assert d13["closure"]["F2_persists_under_R4"] in (True, False)                 # computed whatever the adoption
    for key, per in r50["per_file"].items():
        for sh in R.SHAPES:
            assert per[sh]["own_cited"] <= 10 and per[sh]["conditions"]["hold"] in (True, False) and 0.0 <= per[sh]["coverage"] <= 1.0
        assert "R4_against_R1" in per and "reported:near_name_never_planted" in per and "margins_by_raw_cosine" in per
        assert set(per["R4_against_R1"]["attributed"]) >= {"ranking_error_of_the_mark", "parse_failure"}
    nqb = r50["per_file"]["tiny:E2"]["reported:neighbour_questions"]                 # the answers with a window, the same under every shape
    assert len({v["n"] for v in nqb.values()}) == 1 and 0 < nqb["none"]["n"] <= 10
    nnb = r50["per_file"]["tiny:E2"]["reported:near_name_never_planted"]
    for sh in R.SHAPES:                                                           # no shape creates a claim from a model abstention or a vetoed citation
        assert nnb[sh]["claims"] <= nnb["none"]["claims"]
    post = R.replay_row50(row48, post_hoc=True)
    assert post["post_hoc"] and post["valid"] and post["label"].startswith("post hoc") and post["decision_13"] is None
    # a window pointer that resolves to no planting text is a defect of the file and withholds the replay
    bad = copy.deepcopy(files); bad["tiny"]["E2"]["answers"]["on"][0]["window"][0]["pointer"] = "0000000000000000"
    bad["tiny"]["E2"]["answers"]["on"][0]["retrieval_order"][0] = "0000000000000000"
    r_bad = R.replay_row50(bad, row49_letter="alpha"); assert not r_bad["valid"] and "tiny:E2" in r_bad["unread"]
    # a file missing withholds the replay; a corrupted endpoint field fails the validity condition
    bad = copy.deepcopy(files); del bad["tiny"]["E1"]
    assert "missing" in R.replay_row50(bad, row49_letter="alpha")["unread"]
    for field in ("cited_pointer_is_own", "both_marks_at_or_above_threshold", "label"):
        bad = copy.deepcopy(files); a0 = next(a for a in bad["tiny"]["E2"]["answers"]["on"] if a.get("window"))
        a0[field] = (not a0[field]) if isinstance(a0[field], bool) else "Z"
        assert not R.validity_none_reproduces(bad["tiny"]["E2"])["ok"], field
