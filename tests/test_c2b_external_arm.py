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
    spans = ["the quick brown fox jumps over the lazy dog " * 8] * 3
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
