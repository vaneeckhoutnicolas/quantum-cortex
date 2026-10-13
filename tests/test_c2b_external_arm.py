"""The external model arm (ADR-008 amendment 2026-09-16): the text surface of the
contract maps onto the same parser; the frame is frozen and leakage free; the
verdict follows the frozen rules on stub models; the skill arm and the
agreement check do what they declare. No external model here: `transformers`
is never imported."""
from __future__ import annotations

import copy
import json
import re

import numpy as np
import pytest

from cortex_c2b import external_arm as ea
from cortex_c2b.crypto import generate_key
from cortex_c2b.hm_protocol import generate_facts, _TEMPLATES, N_FACTS, N_NEGCTRL
from cortex_c2b.lm_bridge import encode_target, parse_contract


# ---- stub models: the two calls the arm needs, scripted -----------------------
def _attr_of(payload: str) -> str | None:
    for st, _, _ in _TEMPLATES.values():
        m = re.match("^" + re.escape(st).replace(r"\{e\}", "(.+?)").replace(r"\{a\}", "(.+)") + "$", payload)
        if m:
            return m.group(2)
    return None


class Oracle:
    """Reads the notebook perfectly: cites the line about the queried entity,
    abstains when there is none. Its skill is untouched by any prefix."""
    name = "stub-oracle"

    def __init__(self, noisy_skill: bool = False):
        self.noisy = noisy_skill

    def complete(self, prompt: str) -> str:
        block = prompt.split("Notebook:\n")[-1].split("Question:")[0]
        question = prompt.rsplit("Question: ", 1)[1].split("\n")[0]
        entity = ea._ENTITY_RE.search(question).group(0)
        for line in block.splitlines():
            label, _, payload = line.partition(":")
            if entity in payload:
                return f"CITE {label} ANS {_attr_of(payload)}"
        return "UNKNOWN"

    def nll(self, prefix: str, text: str) -> tuple[float, int]:
        n = max(1, len(text) - 1)
        return ((1.0 if (self.noisy and prefix) else 0.5) * n, n)

    def describe(self) -> dict:
        return {"name": self.name, "revision": "stub", "dtype": "none", "template": "raw"}


class Constant(Oracle):
    def __init__(self, line: str):
        super().__init__(); self.line = line; self.name = f"stub-constant({line})"

    def complete(self, prompt: str) -> str:
        return self.line


# ---- the mapper: the text surface onto the contract's tokens, the same parser --
def test_mapper_agrees_with_parse_contract_on_the_contract_cases():
    assert ea.contract_tokens("UNKNOWN") == encode_target(None, None)
    assert ea.contract_tokens("CITE B ANS glazier") == encode_target(ord("B"), "glazier")
    assert ea.contract_tokens("  \n CITE C ANS the guild council \nsecond line") == encode_target(ord("C"), "the guild council")
    for text, label, attr in (("UNKNOWN", None, None), ("CITE A ANS 1931", ord("A"), "1931"),
                              ("CITE H ANS the ferry syndic", ord("H"), "the ferry syndic")):
        assert parse_contract(ea.contract_tokens(text)) == parse_contract(encode_target(label, attr))


@pytest.mark.parametrize("line", ["cite b ans glazier", "CITE Z ANS x", "", "CITE B ANS", "UNKNOWN.",
                                  "I think it is A", "CITE B: glazier", "Cite B ANS glazier"])
def test_any_other_shape_is_malformed_never_folded(line):
    kind, label, attr = parse_contract(ea.contract_tokens(line))
    assert kind == "malformed" and label is None and attr is None


def test_the_attribute_is_matched_byte_for_byte_after_the_first_line():
    kind, label, attr = parse_contract(ea.contract_tokens("CITE D ANS Glazier"))
    assert (kind, chr(label), attr) == ("cite", "D", "Glazier")          # no case folding: the scorer decides


# ---- the frame: frozen, retained, leakage free ------------------------------------
def test_frame_is_deterministic_and_its_hash_moves_with_any_byte():
    f1, f2 = ea.build_frame(), ea.build_frame()
    assert f1 == f2 and len(ea.frame_hash(f1)) == 16
    assert ea.frame_hash(f1) != ea.frame_hash(f1 + " ")
    assert f1.count("{window}") == 1 and f1.count("{question}") == 1
    assert f1.count("Example ") == ea.N_EXAMPLES and f1.count("Answer: UNKNOWN") == 2 and f1.count("Answer: CITE ") == 2


def test_the_retained_frame_file_is_the_code_s_frame():
    frame = ea.build_frame()
    p = ea.frame_path(frame)
    assert p.exists(), f"run `python -m cortex_c2b.external_arm --frame` and commit {p.name}"
    assert p.read_text(encoding="utf-8") == frame


def test_frame_holds_no_protocol_entity():
    frame = ea.build_frame()
    for seed, n in ((0, N_FACTS), (10_000, N_NEGCTRL)):
        for f in generate_facts(n, seed)[0]:
            assert f.entity not in frame


def test_render_puts_the_window_verbatim_and_an_empty_window_leaves_the_block_empty():
    frame = ea.build_frame()
    r = ea.render(frame, b"A:Vore-3f2a works as a glazier\nB:Kalu-1a2b is a town in Vilkas\n", "what is Vore-3f2a's job?")
    assert r.endswith("Notebook:\nA:Vore-3f2a works as a glazier\nB:Kalu-1a2b is a town in Vilkas\nQuestion: what is Vore-3f2a's job?\nAnswer:")
    assert ea.render(frame, b"", "q").endswith("Notebook:\nQuestion: q\nAnswer:")
    assert ea.frame_prefix(frame).endswith("Notebook:\n")


# ---- the verdict on stubs: the frozen rules, session B from the disk ---------------
def _plant_and_probe(tmp_path, model, n_facts=25, n_negctrl=10):
    key = generate_key()
    jdir = tmp_path / "journal"
    side = ea.plant(jdir, n_facts=n_facts, seed=0, key=key)
    assert side["admitted"] == n_facts and side["durable"]
    spans = [f"span {i}: " + "the quick brown fox jumps over the lazy dog " * (6 + i % 3) for i in range(6)]
    return ea.probe(model, jdir, n_facts=n_facts, n_negctrl=n_negctrl, seed=0, spans=spans, data_note={"stub": True}, key=key)


def test_a_perfect_reader_passes_persistent_and_claimable(tmp_path):
    r = _plant_and_probe(tmp_path, Oracle())
    assert r["hm_recall_on"] == 1.0 and r["hm_recall_off"] == 0.0 and r["hm_gap"] == 1.0
    assert r["hm_retrieval_hit"] == 1.0 and r["hm_attention_mass"] is None
    assert r["hm_negctrl_rate"] == 0.0 and r["hm_invalid_citation_on"] == 0.0 and r["hm_malformed_on"] == 0.0
    assert r["hm_skill_delta"] == 0.0 and r["hm_skill_delta_frame"] == 0.0
    assert r["run_valid"] and r["hm_dissociation_pass"] == 1 and r["persistent"] and r["claimable"] == 1
    assert r["verdict"].startswith("PASS") and r["process"].startswith("new")
    assert r["prompt"]["hash"] == ea.frame_hash(ea.build_frame())
    assert len(r["answers"]["on"]) == 25 and r["answers"]["on"][0]["kind"] == "cite"
    # ADR-008, 2026-10-13: the file records the index geometry from the live object and the journal's record (the planted journal
    # of this arm is written through a path, so it is stamped legacy; the journal of 2026-09-17 reads legacy by default)
    assert r["index"]["live"]["signature"] == "collapsed" and r["index"]["live"]["n_tables"] == 4 and r["index"]["live"]["planes_sha256"] == "680896ab07072738"
    assert r["index"]["overridden"] is False and r["index"]["record"]["signature"] == "collapsed" and r["index"]["record_source"] == "stamp"


def test_a_constant_citation_is_invalid_on_the_negative_control(tmp_path):
    r = _plant_and_probe(tmp_path, Constant("CITE A ANS glazier"))
    assert r["hm_negctrl_rate"] == 1.0 and not r["run_valid"] and r["hm_dissociation_pass"] == 0
    assert r["verdict"].startswith("INVALID") and r["hm_malformed_on"] == 0.0
    assert r["hm_invalid_citation_on"] + r["hm_valid_citation_on"] == pytest.approx(1.0)


def test_a_constant_abstention_fails_on_the_gap(tmp_path):
    r = _plant_and_probe(tmp_path, Constant("UNKNOWN"))
    assert r["hm_recall_on"] == 0.0 and r["hm_false_abstention_on"] == 1.0 and r["hm_negctrl_rate"] == 0.0
    assert r["run_valid"] and r["hm_dissociation_pass"] == 0 and r["verdict"].startswith("FAIL")


def test_a_model_that_ignores_the_contract_is_malformed_and_counted_as_claims(tmp_path):
    r = _plant_and_probe(tmp_path, Constant("I think it is A"))
    assert r["hm_malformed_on"] == 1.0 and r["hm_malformed_negctrl"] == 1.0
    assert r["hm_invalid_citation_on"] == 1.0 and r["hm_negctrl_rate"] == 1.0 and not r["run_valid"]


def test_retrieval_noise_that_hurts_the_skill_fails_arm_s_and_is_attributed(tmp_path):
    r = _plant_and_probe(tmp_path, Oracle(noisy_skill=True))
    assert r["hm_recall_on"] == 1.0 and r["hm_negctrl_rate"] == 0.0
    assert r["hm_skill_delta"] > ea.EPSILON_S and r["hm_skill_delta_frame"] > ea.EPSILON_S
    assert r["hm_dissociation_pass"] == 0 and r["verdict"].startswith("FAIL")


# ---- the skill arm's text and the agreement check -----------------------------------
def test_val_spans_come_from_the_validation_part_and_drop_reserved_ids(tmp_path):
    arr = np.concatenate([np.full(4000, 65, dtype=np.uint16), np.full(400, 66, dtype=np.uint16)])
    arr[4000:4400:7] = 259                                    # a reserved id inside the validation part
    p = tmp_path / "bytes.bin"; arr.tofile(p)
    spans, note = ea.load_val_spans(p, n_spans=3, span_bytes=64, seed=0, val_fraction=0.1)
    assert len(spans) == 3 and all(set(s) == {"B"} for s in spans) and all(len(s) < 64 for s in spans)
    assert note["n_bytes"] == 4400 and note["val_fraction"] == 0.1


def test_two_executions_agree_number_for_number_or_the_difference_is_named(tmp_path):
    r = _plant_and_probe(tmp_path, Oracle())
    ok, diffs = ea.agree(r, copy.deepcopy(r))
    assert ok and diffs == []
    r2 = copy.deepcopy(r); r2["hm_recall_on"] = 0.5
    ok, diffs = ea.agree(r, r2)
    assert not ok and any(d.startswith("hm_recall_on") for d in diffs)
    assert json.loads(json.dumps(r))["model"]["name"] == "stub-oracle"


def test_the_pinned_models_are_three_sizes_of_one_family_with_full_shas():
    assert list(ea.PINNED_MODELS) == ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]
    assert all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in ea.PINNED_MODELS.values())
    assert ea.model_slug("Qwen/Qwen3-1.7B", "chat") == "qwen-qwen3-1.7b" and ea.model_slug("Qwen/Qwen3-8B", "raw").endswith("-raw")


def test_the_frame_is_written_with_unix_line_endings_on_every_machine(tmp_path, monkeypatch):
    monkeypatch.setattr(ea, "FRAME_DIR", tmp_path)
    ea._cli(["--frame"])
    p = next(tmp_path.glob("external-arm-prompt-*.txt"))
    assert b"\r" not in p.read_bytes() and p.read_text(encoding="utf-8") == ea.build_frame()


# ---- S′, the skill arm under the organ's own judgment (ADR-008 amendment 2026-10-08) ----
# Declared before the code, 2026-10-12, and hardened the same day by the committee's second round: the same
# spans as S, the window attached only where the organ returns at least one line for the span and the top
# mark is strictly above the veto threshold of the run; the threshold read from the training family alone by
# the mark oracle's rule (RES-23), planted and read by the address cue as this arm does; the share of spans
# and of tokens with a window recorded, the per span records retained so that S and S′ recompute from the
# file; the verdict untouched (S's); two executions compared on the numbers both files carry.
import subprocess
import sys as _sys
from cortex_c2b.organ_use import training_facts


class WindowSensitive(Oracle):
    """A stub whose loss sees the run's own notebook block: 0.5 per token alone, plus
    0.1 per token and per line of the block attached (the frame's fixed text, whose
    last notebook is empty, costs nothing). Charging a span with another prefix than
    the one the rule names changes its loss."""
    name = "stub-window-sensitive"

    def nll(self, prefix: str, text: str) -> tuple[float, int]:
        n = max(1, len(text) - 1)
        block = prefix.rsplit("Notebook:\n", 1)[-1] if prefix else ""
        return ((0.5 + 0.1 * block.count("\n")) * n, n)


class Directional(Oracle):
    """A stub for which a window HURTS the spans named and HELPS every other one:
    S can sit at zero while S′, which keeps only the admitted windows, is above the line."""
    name = "stub-directional"

    def __init__(self, hurt: set[str]):
        super().__init__(); self.hurt = hurt

    def nll(self, prefix: str, text: str) -> tuple[float, int]:
        n = max(1, len(text) - 1)
        block = prefix.rsplit("Notebook:\n", 1)[-1] if prefix else ""
        if not block:
            return 0.5 * n, n
        return ((0.8 if text in self.hurt else 0.2) * n, n)


def _journal_and_path(tmp_path, n_facts=25):
    key = generate_key()
    jdir = tmp_path / "journal"
    ea.plant(jdir, n_facts=n_facts, seed=0, key=key)
    journal = ea.Journal(ea.journal_file(jdir), key=key, policy=ea.POLICY_STOP)
    return journal, ea.JournalPath(journal, seed=0)


def _spans_and_marks(jp, n=12):
    spans = [f"span number {i}: " + "the quick brown fox jumps over the lazy dog " * (4 + i % 3) for i in range(n)]
    hits = [jp.retrieve(ea._cue_for(t), k=ea.K) for t in spans]
    marks = [max([s for _, _, s in h], default=0.0) for h in hits]
    return spans, marks, [len(h) for h in hits]


def _recompute(detail: list[dict], threshold: float) -> dict:
    """S, S′ and their companions from the per span records alone."""
    import math
    tok = sum(d["tokens"] for d in detail)
    ppl = {k: math.exp(sum(d[f"nll_{k}"] for d in detail) / tok) for k in ("off", "on", "frame")}
    adm = [d for d in detail if d["admitted"]]
    assert all(d["admitted"] == (d["lines"] > 0 and d["top_mark"] > threshold) for d in detail)
    prime_nll = sum(d["nll_on"] if d["admitted"] else d["nll_off"] for d in detail)
    ppl["prime"] = math.exp(prime_nll / tok)
    out = {"ppl_off": ppl["off"], "ppl_on": ppl["on"], "ppl_frame": ppl["frame"], "ppl_prime": ppl["prime"],
           "hm_skill_delta": max(0.0, ppl["on"] / ppl["off"] - 1), "hm_skill_delta_prime": max(0.0, ppl["prime"] / ppl["off"] - 1),
           "hm_skill_delta_prime_signed": ppl["prime"] / ppl["off"] - 1,
           "hm_prime_window_share": len(adm) / len(detail), "prime_spans_with_window": len(adm),
           "prime_token_share": sum(d["tokens"] for d in adm) / tok}
    if adm:
        t = sum(d["tokens"] for d in adm)
        out["hm_skill_delta_admitted"] = math.exp(sum(d["nll_on"] for d in adm) / t) / math.exp(sum(d["nll_off"] for d in adm) / t) - 1
    else:
        out["hm_skill_delta_admitted"] = None
    return out


def test_the_threshold_is_read_from_the_training_family_alone_by_the_oracle_s_rule_and_is_deterministic(tmp_path):
    a = ea.organ_mark_threshold(seed=0)
    assert a == ea.organ_mark_threshold(seed=0)
    _, _, report = training_facts(ea.TRAIN_N, ea.TRAIN_SEED)
    assert a["generator_hash"] == report["generator_hash"] and a["train_seed"] == 100_000 == ea.TRAIN_SEED    # that family, no other
    assert a["training_pairs"] == a["n_training_facts"] + a["n_training_negatives"] == 250
    assert a["training_accuracy"] == 1.0 and a["planted"] == a["n_training_facts"]
    assert a["threshold"] == a["max_negative_mark"] < a["min_positive_mark"]                 # the lowest point of the plateau
    assert 0.0 <= a["threshold"] < 1.0 and a["min_positive_mark"] > 0.999                     # the address cue reads its own line
    assert a["rule"].startswith("the window is attached") and "strictly above" in a["rule"] and "empty read" in a["rule"]
    assert a["cue_convention"].startswith("the training family is planted and read by the entity's address cue")
    # the same number from another process: the run's session B is one
    code = ("import json, sys; sys.path.insert(0, %r); from cortex_c2b import external_arm as ea; "
            "print(json.dumps(ea.organ_mark_threshold(seed=0)))" % str(ea.REPO_ROOT))
    other = json.loads(subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip())
    assert other == a


def test_the_threshold_rule_is_the_mark_oracle_s_own_function():
    from cortex_c2b.hm_lm import best_mark_threshold
    pos, negs = [(1.0, True), (0.9, True), (0.6, False)], [0.3, 0.5, 0.2]
    thr, acc = best_mark_threshold(pos, negs)
    assert thr == 0.5 and acc == 5 / 6                                  # claim iff mark > thr: 0.6 (a wrong top) is the one miss
    assert best_mark_threshold([(1.0, True)], [0.4, 0.4]) == (0.4, 1.0)   # a tie leaves the lowest threshold


def test_s_prime_attaches_the_window_only_where_the_organ_returns_a_line_above_the_threshold(tmp_path):
    _, jp = _journal_and_path(tmp_path)
    spans, marks, lines = _spans_and_marks(jp)
    thr = float(sorted(marks)[len(marks) // 2])                  # a mixed share by construction
    admitted = [l > 0 and m > thr for m, l in zip(marks, lines)]
    assert 0 < sum(admitted) < len(spans)
    r = ea.skill_arm(WindowSensitive(), ea.build_frame(), jp, spans, np.random.default_rng(1), threshold=thr)
    d = r["spans_detail"]
    assert [x["admitted"] for x in d] == admitted and [x["lines"] for x in d] == lines
    assert [x["top_mark"] for x in d] == pytest.approx(marks)
    for x in d:                                                 # the loss charged to S′ is ON's where admitted and OFF's elsewhere
        n = x["tokens"]
        assert x["nll_off"] == pytest.approx(0.5 * n) and x["nll_frame"] == pytest.approx(0.5 * n)
        assert x["nll_on"] == pytest.approx((0.5 + 0.1 * x["lines"]) * n)
    rec = _recompute(d, thr)
    for k, v in rec.items():
        assert r[k] == pytest.approx(v), k
    assert r["veto_threshold_used"] == thr and 0.0 < r["hm_skill_delta_prime"] < r["hm_skill_delta"]
    assert r["hm_skill_delta_admitted"] > 0.0 and 0.0 < r["prime_token_share"] < 1.0


def test_s_prime_equals_s_when_every_span_with_a_line_is_admitted_and_is_zero_when_none_is(tmp_path):
    _, jp = _journal_and_path(tmp_path)
    spans, marks, lines = _spans_and_marks(jp, n=16)
    spans = [t for t, l in zip(spans, lines) if l > 0][:6]      # spans the organ answers
    assert len(spans) >= 3
    model = Oracle(noisy_skill=True)
    every = ea.skill_arm(model, ea.build_frame(), jp, spans, np.random.default_rng(1), threshold=-1.0)
    assert every["hm_prime_window_share"] == 1.0 == every["prime_token_share"] and every["ppl_prime"] == every["ppl_on"]
    assert every["hm_skill_delta_prime"] == every["hm_skill_delta"] > ea.EPSILON_S
    assert every["hm_skill_delta_admitted"] == pytest.approx(every["hm_skill_delta"])
    none = ea.skill_arm(model, ea.build_frame(), jp, spans, np.random.default_rng(1), threshold=2.0)
    assert none["hm_prime_window_share"] == 0.0 == none["prime_token_share"] and none["ppl_prime"] == none["ppl_off"]
    assert none["hm_skill_delta_prime"] == 0.0 and none["hm_skill_delta_admitted"] is None and none["hm_skill_delta"] > ea.EPSILON_S
    # an empty read never attaches a window, whatever the threshold
    _, _, lines_all = _spans_and_marks(jp, n=16)
    empty = [t for t, l in zip(_spans_and_marks(jp, n=16)[0], lines_all) if l == 0]
    if empty:
        r = ea.skill_arm(model, ea.build_frame(), jp, empty[:2], np.random.default_rng(1), threshold=-1.0)
        assert r["prime_spans_with_window"] == 0 and r["hm_skill_delta_prime"] == 0.0


def test_s_prime_is_reported_next_to_s_and_the_verdict_reads_s_in_both_directions(tmp_path, monkeypatch):
    r = _plant_and_probe(tmp_path, Oracle(noisy_skill=True))
    for k in ("hm_skill_delta_prime", "ppl_prime", "hm_skill_delta_prime_signed", "hm_prime_window_share",
              "prime_spans_with_window", "prime_token_share", "hm_skill_delta_admitted", "veto_threshold_used"):
        assert k in r
    assert r["veto_threshold"]["threshold"] == r["veto_threshold_used"] and r["veto_threshold"] == ea.organ_mark_threshold(seed=0)
    assert r["software"]["numpy"] == np.__version__ and len(r["skill_arm"]["spans"]) == 6
    assert r["hm_skill_delta"] > ea.EPSILON_S and r["hm_dissociation_pass"] == 0            # S fails
    # S fails while S′ sits at zero (the organ admits nothing): the verdict does not move
    monkeypatch.setattr(ea, "organ_mark_threshold",
                        lambda seed, **kw: {"threshold": 2.0, "source": "the training family (forced for the test)"})
    forced = _plant_and_probe(tmp_path / "f", Oracle(noisy_skill=True))
    assert forced["hm_skill_delta_prime"] == 0.0 and forced["hm_prime_window_share"] == 0.0
    assert forced["hm_skill_delta"] > ea.EPSILON_S and forced["hm_dissociation_pass"] == 0 and forced["verdict"].startswith("FAIL")
    monkeypatch.undo()
    # S passes while S′ is above the line (only the hurting windows are admitted): the verdict does not move either
    _, jp = _journal_and_path(tmp_path / "d")
    spans, marks, lines = _spans_and_marks(jp, n=20)
    spans = [t for t, l in zip(spans, lines) if l > 0][:6]      # spans the organ answers: a window exists to help or hurt
    marks = [max(s for _, _, s in jp.retrieve(ea._cue_for(t), k=ea.K)) for t in spans]
    thr = float(sorted(marks)[3])
    hurt = {t for t, m in zip(spans, marks) if m > thr}
    assert 1 <= len(hurt) <= 2
    monkeypatch.setattr(ea, "organ_mark_threshold", lambda seed, **kw: {"threshold": thr, "source": "forced for the test"})
    key = generate_key(); jdir = tmp_path / "d2" / "journal"
    ea.plant(jdir, n_facts=25, seed=0, key=key)
    both = ea.probe(Directional(hurt), jdir, n_facts=25, n_negctrl=10, seed=0, spans=spans, data_note={"stub": True}, key=key)
    assert both["hm_skill_delta"] == 0.0 and both["hm_skill_delta_prime"] > ea.EPSILON_S
    assert both["hm_skill_delta_admitted"] > 0.0 and both["prime_spans_with_window"] == len(hurt)
    assert both["hm_dissociation_pass"] == 1 and both["verdict"].startswith("PASS")
    monkeypatch.undo()
    perfect = _plant_and_probe(tmp_path / "b", Oracle())
    assert perfect["hm_skill_delta_prime"] == 0.0 == perfect["hm_skill_delta"] and perfect["hm_dissociation_pass"] == 1


def test_s_and_s_prime_recompute_from_the_per_span_records_of_the_file(tmp_path):
    r = _plant_and_probe(tmp_path, WindowSensitive())
    rec = _recompute(r["skill_arm"]["spans"], r["veto_threshold_used"])
    for k, v in rec.items():
        assert r[k] == pytest.approx(v) if v is not None else r[k] is None, k
    assert json.loads(json.dumps(r))["skill_arm"]["spans"][0]["i"] == 0


def test_two_executions_agree_on_the_numbers_both_carry_and_name_what_was_measured_once(tmp_path):
    r = _plant_and_probe(tmp_path, Oracle())
    r2 = copy.deepcopy(r); r2["hm_skill_delta_prime"] = 0.5
    ok, diffs = ea.agree(r, r2)
    assert not ok and any(d.startswith("hm_skill_delta_prime") for d in diffs)
    assert not ea.agree(r, {**copy.deepcopy(r), "hm_prime_window_share": 0.99})[0]
    assert not ea.agree(r, {**copy.deepcopy(r), "veto_threshold_used": 0.9})[0]
    # the file of 2026-09-17 carries no S′: the shared numbers decide, the others are named as measured once
    old = {k: v for k, v in copy.deepcopy(r).items()
           if k not in ("hm_skill_delta_prime", "ppl_prime", "hm_skill_delta_prime_signed", "hm_prime_window_share",
                        "prime_spans_with_window", "prime_token_share", "hm_skill_delta_admitted", "veto_threshold_used",
                        "veto_threshold", "software")}
    ok, diffs = ea.agree(old, r)
    assert ok and diffs == []
    assert set(ea.measured_once(old, r)) == {"hm_skill_delta_prime", "ppl_prime", "hm_skill_delta_prime_signed",
                                             "hm_prime_window_share", "prime_spans_with_window", "prime_token_share",
                                             "hm_skill_delta_admitted", "veto_threshold_used"}
    old["hm_skill_delta"] = 0.3
    assert not ea.agree(old, r)[0]                                 # a shared number still decides
    assert all(k in ea.NUMERIC_KEYS for k in ("hm_skill_delta_prime", "ppl_prime", "hm_prime_window_share",
                                              "veto_threshold_used", "prime_spans_with_window"))


# ---- the second round's additions (2026-10-12): the identity, the comparator, the shared rng ---------------
def test_s_prime_obeys_the_identity_one_plus_s_prime_equals_one_plus_s_adm_to_the_token_share(tmp_path):
    """Before the clipping at zero, 1 + S′ = (1 + S_adm) ** q with q the admitted spans' share of tokens:
    S′ is arithmetic on S_adm and q, which is why S_adm and its comparator carry the information."""
    _, jp = _journal_and_path(tmp_path)
    spans, marks, lines = _spans_and_marks(jp)
    thr = float(sorted(marks)[len(marks) // 2])
    r = ea.skill_arm(WindowSensitive(), ea.build_frame(), jp, spans, np.random.default_rng(1), threshold=thr)
    q = r["prime_token_share"]
    assert 0.0 < q < 1.0
    assert 1.0 + r["hm_skill_delta_prime_signed"] == pytest.approx((1.0 + r["hm_skill_delta_admitted"]) ** q)
    assert r["hm_skill_delta_prime"] == max(0.0, r["hm_skill_delta_prime_signed"])
    assert sum(d["tokens"] for d in r["spans_detail"]) == r["tokens_scored"]


def test_the_admitted_cost_band_recomputes_from_the_per_span_records_and_reads_as_declared(tmp_path):
    import math
    _, jp = _journal_and_path(tmp_path)
    spans, marks, lines = _spans_and_marks(jp)
    thr = float(sorted(marks)[len(marks) // 2])
    r = ea.skill_arm(WindowSensitive(), ea.build_frame(), jp, spans, np.random.default_rng(1), threshold=thr)
    d, band = r["spans_detail"], r["hm_skill_delta_admitted_band"]
    m = sum(1 for x in d if x["admitted"])
    assert band["admitted"] == m == r["prime_spans_with_window"] and band["spans"] == len(d) and band["draws"] == ea.BAND_DRAWS
    assert band["observed"] == pytest.approx(r["hm_skill_delta_admitted"])
    # recomputed here from the records, with the same declared draw
    rng = np.random.default_rng(ea.BAND_SEED)
    costs = []
    for _ in range(ea.BAND_DRAWS):
        idx = rng.choice(len(d), size=m, replace=False)
        on = sum(d[i]["nll_on"] for i in idx); off = sum(d[i]["nll_off"] for i in idx); tok = sum(d[i]["tokens"] for i in idx)
        costs.append(math.exp(on / tok) / math.exp(off / tok) - 1.0)
    lo, hi = np.quantile(costs, [0.025, 0.975])
    assert band["band_low"] == pytest.approx(float(lo)) and band["band_high"] == pytest.approx(float(hi))
    assert band["within_band"] == (lo <= band["observed"] <= hi) == r["hm_skill_delta_admitted_within_band"]
    assert band["share_at_or_under"] == pytest.approx(float(np.mean(np.asarray(costs) <= band["observed"])))
    # a window that hurts only the admitted spans puts S_adm outside the band, above every chance subset
    hurt = {t for t, mk, l in zip(spans, marks, lines) if l > 0 and mk > thr}
    r2 = ea.skill_arm(Directional(hurt), ea.build_frame(), jp, spans, np.random.default_rng(1), threshold=thr)
    b2 = r2["hm_skill_delta_admitted_band"]
    assert b2["observed"] > b2["band_high"] and not b2["within_band"] and b2["share_at_or_under"] == 1.0
    # nothing admitted, or everything: no comparator, by construction
    none = ea.skill_arm(WindowSensitive(), ea.build_frame(), jp, spans, np.random.default_rng(1), threshold=2.0)
    assert none["hm_skill_delta_admitted_band"] is None and none["hm_skill_delta_admitted_within_band"] is None
    answered = [t for t, l in zip(spans, lines) if l > 0]
    every = ea.skill_arm(WindowSensitive(), ea.build_frame(), jp, answered, np.random.default_rng(1), threshold=-1.0)
    assert every["hm_skill_delta_admitted_band"] is None
    assert "hm_skill_delta_admitted_within_band" in ea.NUMERIC_KEYS


def test_reading_the_threshold_leaves_the_run_s_window_order_untouched(tmp_path):
    """The defect class of 2026-09-20: a read made for the oracle must not advance the
    shuffle the windows are drawn with, or two runs see different windows."""
    rng = np.random.default_rng(1)
    before = rng.bit_generator.state
    ea.organ_mark_threshold(seed=0)
    assert rng.bit_generator.state == before
    _, jp = _journal_and_path(tmp_path)
    spans, _, _ = _spans_and_marks(jp, n=6)
    a = ea.skill_arm(WindowSensitive(), ea.build_frame(), jp, spans, np.random.default_rng(7), threshold=0.0)
    ea.organ_mark_threshold(seed=0)
    b = ea.skill_arm(WindowSensitive(), ea.build_frame(), jp, spans, np.random.default_rng(7), threshold=0.0)
    assert a["spans_detail"] == b["spans_detail"] and a["ppl_on"] == b["ppl_on"]
