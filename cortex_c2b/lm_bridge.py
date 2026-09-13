"""cortex_c2b.lm_bridge -- the journal in the decode loop (ADR-008; REPRISE step 3).

Until 2026-09-12 the journal was measured as an ORGAN: a hash-seeded cue encoder
and a reference skill probe stood in for the model. This module wires the organ
into the language model itself, under the nine invariants validated by the
founder before any code was written (ADR-008):

  1. N1 intact at step zero: the reader's output projection is zero-initialised,
     so a model with the journal on produces N1's logits at initialisation,
     window or no window; a parent checkpoint loads its weights unchanged.
  2. One read path, the organ's: cues go to `JournalPath.retrieve`; nothing is
     duplicated on the model side, nothing scans.
  3. The journal enters as CONTENT, through a zero-init cross attention over
     the retrieved bytes embedded with the model's own byte embeddings -- not
     as prompt tokens. Journal OFF = the reader absent = N1 exactly.
  4. The router decides the read, on the floor: the reader's contribution is
     multiplied by the path 4 gate; with no score the gate is zero.
  5. The cue encoder is learned (a projection of the mean pooled residual
     stream to a unit vector), trained by contrast: the cue of a query must be
     nearest to the cue of its own statement among the batch. One encoder for
     writing and reading; the DG/CA3/CA1 write gate is unchanged.
  6. Organ use is taught, not assumed: `cortex_c2b.organ_use` builds the
     curriculum; the target of every example is computed from what the journal
     actually returned (cite what was shown, else abstain).
  7. Two processes, persistence declared: session A writes, session B reopens
     the sealed journal and the checkpoint from the disk alone.
  8. A failure is attributable: retrieval hit (the encoder), attention mass on
     the retrieved bytes (the reader), attribute exactness given a valid
     citation (the answer) are reported next to the verdict.
  9. Everything is a flag, a ledger record, and a CPU smoke test.

The contract of an episodic answer (ADR-008, point d, decided 2026-09-12):
  `<EPI>` marks an episodic query; the answer is `<CITE>` + a one byte LOCAL
  LABEL of the retrieved episode (the harness maps it back to the pointer and
  verifies it) + `<ANS>` + the attribute bytes + newline, or `<UNKNOWN>` +
  newline. Four of the eight reserved oracle tokens (NOW-1) are taken; the
  vocabulary size does not change, so the N1 checkpoint loads as is.

Pure torch over the numpy organ. Written from scratch (our filon).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex_c2b import Journal, CUE_DIM
from cortex_c2b.write_path import WritePath, WriteReport
from cortex_c2b.read_path import JournalPath

# ---- the contract tokens: 4 of the 8 reserved oracle tokens (ids 256..263) ----
TOK_EPI, TOK_CITE, TOK_ANS, TOK_UNKNOWN = 256, 257, 258, 259
CONTRACT_TOKENS = {"EPI": TOK_EPI, "CITE": TOK_CITE, "ANS": TOK_ANS, "UNKNOWN": TOK_UNKNOWN}
LABELS = b"ABCDEFGH"                     # local labels of the retrieved episodes, at most 8 per read
NEWLINE = 10


# ---------------------------------------------------------------------------- #
# The cue encoder -- learned, one for writing and reading (invariant 5)        #
# ---------------------------------------------------------------------------- #
class CueEncoder(nn.Module):
    """Mean pooled residual stream -> linear -> unit vector of `cue_dim`."""

    def __init__(self, n_embd: int, cue_dim: int = CUE_DIM):
        super().__init__()
        self.proj = nn.Linear(n_embd, cue_dim, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            pooled = h.mean(dim=1)
        else:
            m = mask.to(h.dtype).unsqueeze(-1)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return F.normalize(self.proj(pooled), dim=-1, eps=1e-6)


def contrastive_loss(query_cues: torch.Tensor, statement_cues: torch.Tensor,
                     temperature: float = 0.1) -> torch.Tensor:
    """InfoNCE in both directions: row i of the queries must pick row i of the
    statements among the batch (and back). The signal that teaches the encoder
    to put a question and its own episode at the same address."""
    logits = query_cues @ statement_cues.t() / temperature
    target = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


# ---------------------------------------------------------------------------- #
# The reader -- zero-init cross attention over the read window (invariant 3)   #
# ---------------------------------------------------------------------------- #
class JournalReader(nn.Module):
    """Queries from the residual stream, keys and values from the embedded bytes
    of the read window (plus a learned null slot: "nothing to read"). The output
    projection starts at zero, so the model begins as N1 and learns to read.
    `forward` also returns the attention mass on the real bytes (excluding the
    null slot): the reader's diagnostic (invariant 8)."""

    def __init__(self, n_embd: int, n_head: int, window_len: int):
        super().__init__()
        self.h = n_head
        self.ln_x = nn.LayerNorm(n_embd)
        self.ln_w = nn.LayerNorm(n_embd)
        self.q = nn.Linear(n_embd, n_embd, bias=False)
        self.kv = nn.Linear(n_embd, 2 * n_embd, bias=False)
        self.out = nn.Linear(n_embd, n_embd, bias=False)
        self.null = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.wpe = nn.Embedding(window_len, n_embd)          # order within the window
        nn.init.zeros_(self.out.weight)                       # the no-op law
        self.scale = (n_embd // n_head) ** -0.5

    def forward(self, x: torch.Tensor, window_emb: torch.Tensor, window_mask: torch.Tensor):
        b, t, c = x.shape
        r = window_emb.size(1)
        pos = torch.arange(r, device=x.device)
        w = torch.cat([self.null.expand(b, 1, c), window_emb + self.wpe(pos)[None]], dim=1)   # (b, r+1, c)
        m = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=x.device), window_mask.bool()], dim=1)
        hq = self.q(self.ln_x(x)).view(b, t, self.h, c // self.h).transpose(1, 2)               # (b,h,t,d)
        k, v = self.kv(self.ln_w(w)).split(c, dim=2)
        k = k.view(b, r + 1, self.h, c // self.h).transpose(1, 2)
        v = v.view(b, r + 1, self.h, c // self.h).transpose(1, 2)
        att = (hq @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(~m[:, None, None, :], float("-inf"))
        p = torch.softmax(att, dim=-1)
        read = (p @ v).transpose(1, 2).contiguous().view(b, t, c)
        mass = p[..., 1:].sum(dim=-1).mean()                   # attention on real bytes, null slot excluded
        self.last_attended = read                              # RES-21: what was read, before the zero init projection
        self.last_attn = p.detach()                            # (b, h, t, r + 1): the pointer policy reads it at decode
        return self.out(read), mass


class MatchHead(nn.Module):
    """RES-21 (ADR-008, amendment of 2026-09-14): a small head that answers ONE question at
    the decision position, "is the queried entity's episode in the window?", from what
    the model had before reading (the residual stream at the journal block) and what it
    read (the reader's attended values, before the zero init projection). Trained
    directly on that label, which the harness knows exactly for every curriculum example
    (the own pointer among the window's labels, never the ground truth alone), and
    coupled to the decision token's logits: + coupling * m on <CITE>, - coupling * m on
    <UNKNOWN>. The last layer starts at zero, so the model begins as its parent (m = 0,
    no bias) and learns the comparison from there."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.ln_pre = nn.LayerNorm(n_embd)
        self.ln_read = nn.LayerNorm(n_embd)
        self.up = nn.Linear(2 * n_embd, n_embd)
        self.out = nn.Linear(n_embd, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # the no-op law

    def forward(self, pre: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.ln_pre(pre), self.ln_read(attended)], dim=-1)
        return self.out(torch.nn.functional.gelu(self.up(z))).squeeze(-1)  # (b,) a logit: present > 0


# ---------------------------------------------------------------------------- #
# The router's gate on the read (invariant 4)                                  #
# ---------------------------------------------------------------------------- #
PATH_JOURNAL = 4                          # the journal is path 4 of the RES-18 router (ADR-007)


def journal_gate(route: str, decision=None) -> float:
    """The multiplier on the reader's contribution. `pinned_on` / `pinned_off`
    are the two arms of the protocol (declared, recorded); `router` takes the
    path 4 weight of a RouteDecision -- a hard decision for another path, or no
    decision at all, gives zero: the floor guarantee at the model scale."""
    if route == "pinned_on":
        return 1.0
    if route == "pinned_off":
        return 0.0
    if route != "router":
        raise ValueError(f"unknown journal_route {route!r}")
    if decision is None:
        return 0.0
    if decision.weights is not None:
        return float(decision.weights[PATH_JOURNAL]) if len(decision.weights) > PATH_JOURNAL else 0.0
    return float(decision.confidence) if decision.path == PATH_JOURNAL else 0.0


# ---------------------------------------------------------------------------- #
# The read window -- labelled, budgeted, in a seeded random order              #
# ---------------------------------------------------------------------------- #
def build_read_window(items: list[tuple[str, bytes]], budget: int, rng: np.random.Generator
                      ) -> tuple[bytes, dict[int, str]]:
    """`items` = (pointer, payload) as retrieved. Returns the window bytes
    (`A:<payload>\\n` per item, order shuffled with `rng` so a position is never a
    shortcut, each payload cut to its share of `budget`) and the label byte ->
    pointer map the harness uses to verify a citation."""
    items = list(items)[: len(LABELS)]
    if not items:
        return b"", {}
    order = rng.permutation(len(items))
    per = max(1, budget // len(items) - 3)                     # 'X:' and the newline
    out, labels = [], {}
    for slot, i in enumerate(order):
        pointer, payload = items[i]
        label = LABELS[slot]
        labels[label] = pointer
        out.append(bytes([label]) + b":" + payload[:per] + b"\n")
    return b"".join(out), labels


def encode_query(text: str) -> list[int]:
    return [TOK_EPI] + list(text.encode("utf-8"))


def encode_target(label: int | None, attr: str | None) -> list[int]:
    """The contract: cite a label and answer, or abstain."""
    if label is None:
        return [TOK_UNKNOWN, NEWLINE]
    return [TOK_CITE, label, TOK_ANS] + list(attr.encode("utf-8")) + [NEWLINE]


def parse_contract(tokens: list[int]) -> tuple[str, int | None, str | None]:
    """('unknown'|'cite'|'malformed', label, attribute). A malformed answer is a
    claim without the contract's shape -- counted against the model, never for it."""
    toks = list(tokens)
    if toks and toks[0] == TOK_UNKNOWN:
        return "unknown", None, None
    if len(toks) >= 3 and toks[0] == TOK_CITE and toks[2] == TOK_ANS and toks[1] in LABELS:
        body = toks[3:]
        if NEWLINE in body:
            body = body[: body.index(NEWLINE)]
        try:
            return "cite", toks[1], bytes(b for b in body if b < 256).decode("utf-8")
        except UnicodeDecodeError:
            return "malformed", toks[1], None
    return "malformed", None, None


# ---------------------------------------------------------------------------- #
# The bridge -- the torch model above, the numpy organ below (invariant 2)     #
# ---------------------------------------------------------------------------- #
@dataclass
class ReadResult:
    window: bytes
    labels: dict[int, str]            # label byte -> pointer
    pointers: list[str]               # what the organ returned, in retrieval order
    scores: list[float]


class JournalBridge:
    """Holds the organ (journal, write path, router path) and talks to the model
    through two calls it must expose: `journal_hidden(idx, mask)` (the residual
    stream at the journal block, before the reader) and `cue_encoder`.

    The model never sees a pointer or a payload except through the read window;
    the organ never sees a tensor except the cue."""

    def __init__(self, model, journal: Journal, write_path: WritePath | None = None,
                 journal_path: JournalPath | None = None, k: int = 4, budget_bytes: int = 256,
                 seed: int = 0, shuffle_seed: int | None = None, device=None):
        """`seed` is the scope's separation seed (DG) and the index seed -- it must
        be the same at every open of a journal (Decision 8: a journal written under
        another separation is refused). `shuffle_seed` only drives the order of
        the read window and may differ between sessions."""
        self.model = model
        self.j = journal
        if model.cue_encoder is None or model.cue_encoder.proj.out_features != journal.cue_dim:
            raise ValueError("the model's cue encoder must produce the journal's cue dimension "
                             f"({journal.cue_dim}); set journal_cue_dim accordingly")
        self.wp = write_path or WritePath(journal, seed=seed)
        self.jp = journal_path or JournalPath(journal, seed=seed)
        self.k, self.budget = k, budget_bytes
        self.rng = np.random.default_rng(seed if shuffle_seed is None else shuffle_seed)
        self.device = device or next(model.parameters()).device
        self.pointer_of: dict[str, str] = {}                   # entity -> pointer, for the curriculum's targets

    # ---- tensors ----------------------------------------------------------------
    def tokens(self, sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        n = max(len(s) for s in sequences)
        idx = torch.zeros(len(sequences), n, dtype=torch.long)
        mask = torch.zeros(len(sequences), n, dtype=torch.bool)
        for i, s in enumerate(sequences):
            idx[i, : len(s)] = torch.tensor(s, dtype=torch.long); mask[i, : len(s)] = True
        return idx.to(self.device), mask.to(self.device)

    def cue_tensor(self, sequences: list[list[int]]) -> torch.Tensor:
        """Cues with gradient (for the contrastive loss)."""
        idx, mask = self.tokens(sequences)
        h = self.model.journal_hidden(idx)
        return self.model.cue_encoder(h, mask)

    def cues(self, sequences: list[list[int]]) -> np.ndarray:
        """Cues for the organ (no gradient)."""
        with torch.no_grad():
            return self.cue_tensor(sequences).float().cpu().numpy()

    # ---- the organ ---------------------------------------------------------------
    def write(self, text: str, schema: str, now: float, entity: str | None = None) -> WriteReport:
        cue = self.cues([list(text.encode("utf-8"))])[0]
        rep = self.wp.write(cue, text.encode("utf-8"), schema, now=now)
        if rep.admitted:
            self.jp.on_write(rep.entry)
            if entity is not None:
                self.pointer_of[entity] = rep.entry.pointer
        return rep

    def read(self, cue: np.ndarray) -> ReadResult:
        hits = self.jp.retrieve(cue, k=self.k)
        items = [(e.pointer, payload) for e, payload, _ in hits]
        window, labels = build_read_window(items, self.budget, self.rng)
        return ReadResult(window=window, labels=labels, pointers=[p for p, _ in items],
                          scores=[float(s) for _, _, s in hits])

    def window_tensors(self, windows: list[bytes], window_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Byte windows -> (tokens, mask), each cut to `window_len`; an empty
        window is one masked position (the reader keeps its null slot)."""
        r = max(1, min(window_len, max((len(w) for w in windows), default=1)))
        tok = torch.zeros(len(windows), r, dtype=torch.long)
        mask = torch.zeros(len(windows), r, dtype=torch.bool)
        for i, w in enumerate(windows):
            w = w[:r]
            if w:
                tok[i, : len(w)] = torch.tensor(list(w), dtype=torch.long); mask[i, : len(w)] = True
        return tok.to(self.device), mask.to(self.device)
