"""cortex_c2b.organ_use -- teaching the model to use its own organ (NOW-8; ADR-008 invariant 6).

The rule that combines abstention and citation, decided by the founder on
2026-09-12: **the target of every example is computed from what the journal
actually returned, never from the ground truth alone.** If the episode that
holds the answer is among the retrieved ones, the target cites its local label
and gives the attribute; otherwise the target is `<UNKNOWN>`, even though the
fact exists in the journal. The model is never rewarded for citing what it was
not shown, so no invented citation enters the signal; while the cue encoder is
poor the model learns silence first (one token), and as the contrastive loss
improves retrieval, citations replace abstentions by themselves.

Three mitigations, all declared and hashed in the run's configuration:
  - an asymmetric weight on the decision token of an abstention target
    (answering instead of abstaining costs more than the reverse);
  - negative examples (entities never planted) from the first step;
  - a bootstrap for the cold start: on a missed retrieval, with a declared
    probability that decays to zero, the right episode is FORCED into the
    window and the example is marked `forced` -- counted in the diagnostics,
    never as recall.

Decontamination (invariant 5): the training facts come from the same generator
family as the frozen protocol, with disjoint seeds, and every entity name is
checked against the protocol's facts and its negative control; a collision is
removed and reported. Written from scratch (our filon).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cortex_c2b.hm_protocol import Fact, generate_facts, N_FACTS, N_NEGCTRL
from cortex_c2b.lm_bridge import (JournalBridge, encode_query, encode_target, build_read_window,
                                  TOK_UNKNOWN)

EVAL_SEEDS = (0, 10_000)                  # the frozen protocol's facts and its negative control


# ---------------------------------------------------------------------------- #
# Facts: disjoint seeds, checked entities                                       #
# ---------------------------------------------------------------------------- #
def decontaminate(train: list[Fact], eval_facts: list[Fact]) -> tuple[list[Fact], list[str]]:
    """Remove every training fact whose entity appears in the evaluation facts."""
    banned = {f.entity for f in eval_facts}
    kept = [f for f in train if f.entity not in banned]
    removed = [f.entity for f in train if f.entity in banned]
    return kept, removed


def training_facts(n: int, seed: int = 100_000, n_negatives: int = 50) -> tuple[list[Fact], list[Fact], dict]:
    """Planted facts and never-planted negatives for training, both disjoint
    from the frozen protocol's sets (seeds 0 and 10_000). Returns
    (facts, negatives, report)."""
    if seed in EVAL_SEEDS or seed + 1 in EVAL_SEEDS:
        raise ValueError("training seeds must differ from the frozen protocol's seeds (0 and 10_000)")
    eval_facts = [f for s in EVAL_SEEDS for f in generate_facts(N_FACTS if s == 0 else N_NEGCTRL, s)[0]]
    facts, gen_hash = generate_facts(n, seed)
    negs, _ = generate_facts(n_negatives, seed + 1)
    facts, removed_f = decontaminate(facts, eval_facts)
    negs, removed_n = decontaminate(negs, eval_facts + facts)
    report = {"generator_hash": gen_hash, "seed": seed, "n_facts": len(facts), "n_negatives": len(negs),
              "removed_collisions": removed_f + removed_n}
    return facts, negs, report


# ---------------------------------------------------------------------------- #
# The pool: facts planted through the normal gated path under the CURRENT encoder
# ---------------------------------------------------------------------------- #
def plant_pool(bridge: JournalBridge, facts: list[Fact], now: float = 0.0) -> dict:
    """Write every fact as an episode (statement bytes, schema, cue from the
    model). Returns admission statistics; `bridge.pointer_of` maps entities to
    their pointers, which is how a target knows whether the right episode was
    among the retrieved ones."""
    admitted = 0
    for i, f in enumerate(facts):
        rep = bridge.write(f.statement, f.schema, now=now + i, entity=f.entity)
        admitted += int(rep.admitted)
    return {"planted": len(facts), "admitted": admitted}


# ---------------------------------------------------------------------------- #
# Examples                                                                       #
# ---------------------------------------------------------------------------- #
@dataclass
class Example:
    kind: str                          # answer | abstain | negative | forced
    query: list[int]                   # <EPI> + question bytes
    target: list[int]                  # the contract
    window: bytes
    labels: dict[int, str] = field(default_factory=dict)
    retrieved_ok: bool = False         # the right episode was among the retrieved (the encoder's success)
    entity: str = ""
    attr: str = ""
    statement: list[int] = field(default_factory=list)   # the episode's bytes, for the contrastive pair

    @property
    def tokens(self) -> list[int]:
        return self.query + self.target

    def weights(self, asym: float) -> list[float]:
        """Loss on the target only; the decision token of an abstention target
        carries the asymmetric weight."""
        w = [0.0] * len(self.query) + [1.0] * len(self.target)
        if self.target and self.target[0] == TOK_UNKNOWN:
            w[len(self.query)] = float(asym)
        return w


def _label_of(labels: dict[int, str], pointer: str) -> int | None:
    for label, p in labels.items():
        if p == pointer:
            return label
    return None


def make_example(bridge: JournalBridge, fact: Fact, negative: bool, forced_frac: float,
                 rng: np.random.Generator, paraphrase: bool = False) -> Example:
    """One example, its target computed from the actual read (the rule above)."""
    question = fact.paraphrase if paraphrase else fact.query
    query = encode_query(question)
    cue = bridge.cues([query])[0]
    read = bridge.read(cue)
    statement = list(fact.statement.encode("utf-8"))
    if negative:
        return Example("negative", query, encode_target(None, None), read.window, read.labels,
                       False, fact.entity, fact.attr, statement)
    pointer = bridge.pointer_of.get(fact.entity)
    label = _label_of(read.labels, pointer) if pointer else None
    if label is not None:
        return Example("answer", query, encode_target(label, fact.attr), read.window, read.labels,
                       True, fact.entity, fact.attr, statement)
    if pointer and forced_frac > 0 and rng.random() < forced_frac:
        # the bootstrap: the right episode is forced into the window (replacing the last slot)
        items = [(p, bridge.j.payloads.get(p)) for p in read.pointers][: max(0, bridge.k - 1)]
        items.append((pointer, bridge.j.payloads.get(pointer)))
        window, labels = build_read_window(items, bridge.budget, rng)
        return Example("forced", query, encode_target(_label_of(labels, pointer), fact.attr), window, labels,
                       False, fact.entity, fact.attr, statement)
    return Example("abstain", query, encode_target(None, None), read.window, read.labels,
                   False, fact.entity, fact.attr, statement)


def make_paired_negative(bridge: JournalBridge, fact: Fact, rng: np.random.Generator,
                         paraphrase: bool = False, keep_shape: bool = False) -> Example:
    """The minimal pair (v5 lever, run 6f10e8cbfebf): the SAME planted fact, its own episode
    WITHHELD from the window; target abstain. Four runs showed the decision to cite or abstain
    following the share of negatives seen in training (a class prior) far more than the
    window's content; with pairs the prior earns nothing, and the only way down the loss is
    to check whether the queried entity is in the read.

    `keep_shape=False` is the pair AS RUN in v5 (`db028e6a4262`, INVALID): the own pointer
    removed from the k retrieved, the window built from what is left, so it has k - 1 lines
    whenever the episode had been retrieved (over 90 % of the time late in the run). That is
    a shape cue, cheaper than a comparison, and absent from the frozen protocol's windows,
    which always have k lines. The pair was not minimal; kept as written so the v5 run stays
    reproducible under its hash. `keep_shape=True` is the v6 lever: k + 1 retrieved, the own
    dropped, k lines kept, so the pair's two halves have the same shape and differ only by the
    episode's presence."""
    question = fact.paraphrase if paraphrase else fact.query
    query = encode_query(question)
    cue = bridge.cues([query])[0]
    own = bridge.pointer_of.get(fact.entity)
    items = withheld_items(bridge, cue, own, rng, keep_shape=keep_shape)
    window, labels = build_read_window(items, bridge.budget, rng)
    return Example("negative", query, encode_target(None, None), window, labels, False,
                   fact.entity, fact.attr, list(fact.statement.encode("utf-8")))


def withheld_items(bridge: JournalBridge, cue: np.ndarray, own: str | None, rng: np.random.Generator,
                   keep_shape: bool) -> list[tuple[str, bytes]]:
    """The window items of a paired negative. `keep_shape=False`: the read's items minus the own
    pointer, one line fewer than the read whenever the own was retrieved (v5 as run). `keep_shape=True`:
    the same number of lines as the read would have shown: the k + 1 th candidate fills the slot when the
    index returns one, else another planted episode drawn at random (the index is a locality sensitive
    hash and may return fewer than k candidates, so the fill is by construction, not by luck)."""
    if not keep_shape:
        read = bridge.read(cue)
        return [(p, bridge.j.payloads.get(p)) for p in read.pointers if p != own]
    hits = bridge.jp.retrieve(cue, k=bridge.k + 1)
    top = [(e.pointer, payload) for e, payload, _ in hits]
    shown = min(len(top), bridge.k)                                     # what a plain read shows
    items = [(p, pl) for p, pl in top if p != own][: bridge.k]
    if len(items) < shown:                                              # the own was shown and no k + 1 th candidate exists
        seen = {p for p, _ in top}
        others = sorted(p for p in bridge.pointer_of.values() if p != own and p not in seen)
        if others:
            p = others[int(rng.integers(len(others)))]
            items.append((p, bridge.j.payloads.get(p)))
    return items


def make_batch(bridge: JournalBridge, facts: list[Fact], negatives: list[Fact], n: int,
               rng: np.random.Generator, negatives_frac: float = 0.25, forced_frac: float = 0.5,
               paraphrase_frac: float = 0.5, paired: bool = False, keep_shape: bool = False) -> list[Example]:
    """`paired`: negatives are minimal pairs of the planted facts (the episode withheld)
    instead of separate never planted entities; `keep_shape`: the pair keeps k lines (v6)."""
    out = []
    for _ in range(n):
        if rng.random() < negatives_frac and (paired or negatives):
            if paired:
                f = facts[int(rng.integers(len(facts)))]
                out.append(make_paired_negative(bridge, f, rng, paraphrase=rng.random() < paraphrase_frac,
                                                keep_shape=keep_shape))
            else:
                f = negatives[int(rng.integers(len(negatives)))]
                out.append(make_example(bridge, f, True, 0.0, rng, paraphrase=rng.random() < paraphrase_frac))
        else:
            f = facts[int(rng.integers(len(facts)))]
            out.append(make_example(bridge, f, False, forced_frac, rng, paraphrase=rng.random() < paraphrase_frac))
    return out


def batch_stats(examples: list[Example]) -> dict:
    n = max(1, len(examples))
    kinds = {k: sum(e.kind == k for e in examples) for k in ("answer", "abstain", "negative", "forced")}
    planted = [e for e in examples if e.kind != "negative"]
    return {**{f"n_{k}": v for k, v in kinds.items()},
            "retrieval_hit": (sum(e.retrieved_ok for e in planted) / len(planted)) if planted else 0.0,
            "abstain_target_rate": sum(e.target[0] == TOK_UNKNOWN for e in examples) / n}


def collate(bridge: JournalBridge, examples: list[Example], asym: float, window_len: int):
    """Tensors for one training step: (idx, targets, weights, window, window_mask,
    query_sequences, statement_sequences). Targets are the next token; the
    weight of a position is the weight of the token it predicts."""
    import torch
    seqs = [e.tokens for e in examples]
    idx, _ = bridge.tokens([s[:-1] for s in seqs])
    tgt, _ = bridge.tokens([s[1:] for s in seqs])
    n = idx.size(1)
    w = torch.zeros(len(examples), n)
    for i, e in enumerate(examples):
        ww = e.weights(asym)[1:]                                   # aligned with the predicted token
        w[i, : len(ww)] = torch.tensor(ww)
    window, wmask = bridge.window_tensors([e.window for e in examples], window_len)
    return idx, tgt, w.to(idx.device), window, wmask, [e.query for e in examples], [e.statement for e in examples]
