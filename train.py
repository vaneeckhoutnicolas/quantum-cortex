"""quantum-cortex — N1 training scaffold (single-file trainer, feature F2).

Vanilla GPT-2-style control model (the ONLY legitimate baseline per hub decision 019),
byte-level tokenizer with reserved oracle token ids (register NOW-1),
device-agnostic per ADR-002 (CUDA bf16/fp16, CPU fp32 for smoke tests only),
checkpoint/resume (EuroHPC plan §4), and first-class metrics per ADR-001:
every completed run appends one schema-v1 record to metrics/runs.jsonl and
regenerates metrics/LATEST.md. A run without its committed record does not exist.

Usage:
  python train.py --config configs/smoke_cpu.json
  python train.py --config configs/kaggle_t4.json [--resume]
  python train.py --regen-latest
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = REPO_ROOT / "metrics" / "schema" / "run-v1.schema.json"
LEDGER_PATH = REPO_ROOT / "metrics" / "runs.jsonl"
LATEST_PATH = REPO_ROOT / "metrics" / "LATEST.md"

# ----------------------------------------------------------------------------- config


@dataclass
class Config:
    # run
    kind: str = "control"                # control | ablation | combo
    component_under_test: str | None = None  # null for control; C1..C4 otherwise
    notes: str | None = None
    provider: str = "other"              # kaggle | colab | local | eurohpc | trc | rented | other
    # model (vanilla GPT-2-style control: LayerNorm, GELU, learned pos emb, tied embeddings)
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 1024
    vocab_bytes: int = 256
    reserved_oracle_tokens: int = 8      # NOW-1: token space reserved from day one
    # data
    data_mode: str = "synthetic"         # synthetic | bin
    bin_path: str = "data/train_bytes.bin"
    dataset_id: str = "synthetic-v0"
    data_slice: str = "seeded pattern, 5% noise"
    val_fraction: float = 0.01
    # training
    max_tokens: int = 200_000
    batch_size: int = 8
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1337
    optimizer: str = "adamw"             # F1 (NorMuon-class) arrives only via its own ablation
    c2_variant: str = "none"             # none | hopfield | delta  (ADR-006: C2 associative layer, default off)
    c2_mem_slots: int = 64               # hopfield: number of stored key/value patterns
    c2_heads: int = 4                    # delta: number of recurrent heads
    c2_delta_l2_keys: bool = True        # delta: L2-normalise q,k before the write (KDA-style stability)
    c2_delta_state_clip: float = 0.0     # delta: optional |S| clip (0 = off); belt-and-braces guard
    # journal in the decode loop (ADR-008), default off -- the model starts as N1 and learns to read
    journal: str = "none"                # none | read
    journal_block: int = 5               # the block that carries the reader (one hippocampal port)
    journal_k: int = 4                   # episodes retrieved per read
    journal_read_bytes: int = 256        # byte budget of the read window
    journal_cue_dim: int = 64            # the organ's cue dimension (CUE_DIM)
    journal_heads: int = 4               # heads of the reader's cross attention
    journal_route: str = "pinned_on"     # pinned_on | pinned_off | router  (the H.M. arms pin; the router is measured apart)
    journal_asym_weight: float = 2.0     # loss weight on the decision token of an abstention target (answering instead of abstaining costs more)
    journal_neg_frac: float = 0.25       # share of curriculum examples on never planted entities (target <UNKNOWN>); v4 lever (run ce008745375a)
    journal_paired_negatives: bool = False   # v5 lever (run 6f10e8cbfebf): negatives are minimal pairs, the same planted fact with its
                                             # episode withheld from the window, so the decision must depend on the read, not on a class prior
    journal_paired_keep_shape: bool = False  # v6 lever (run db028e6a4262, INVALID): the pair keeps k lines (k + 1 retrieved, the own
                                             # dropped); as run in v5 the withheld window had k - 1 lines, a shape cue the protocol never shows
    journal_forced_frac: float = 0.5     # bootstrap: share of missed retrievals where the right episode is forced into the window, decays to 0
    journal_curriculum_ratio: float = 0.5  # share of training steps spent on organ use examples (the rest: ordinary language modelling)
    journal_contrastive_weight: float = 1.0
    journal_pool_facts: int = 200        # planted facts per curriculum epoch
    journal_pool_refresh: int = 100      # steps between two replantings under the current encoder
    journal_train_seed: int = 100_000    # the training facts' generator seed (disjoint from the frozen protocol's 0 and 10_000)
    journal_fresh_pool: bool = False     # v2 lever (run dc34fcf000aa, INVALID): a NEW set of facts at every replant, so the only way
                                         # down the loss is to read the window -- a fixed pool let the model memorise entity -> attribute
    journal_lm_window: bool = False      # v2 lever: language modelling steps also carry a retrieved window (gate on), so the model
                                         # learns to ignore an irrelevant read on ordinary text (the skill arm measures exactly that)
    journal_path: str | None = None      # the sealed on-disk journal of the run (key from QUANTUM_CORTEX_JOURNAL_KEY); None = a memory scope
    journal_hm_arm: bool = True          # run the frozen protocol's LM arm at the end of the run
    journal_hm_facts: int = 200          # the protocol's sizes (200 / 50 are the frozen defaults; smaller only for a CPU smoke)
    journal_hm_negctrl: int = 50
    parent_run_id: str | None = None     # the checkpoint this run resumes from (a fine tune is a new run)
    parent_ckpt: str | None = None       # path of that checkpoint
    # circuit breaker (ADR-006 D7): aggressive early-abort, thresholds declared pre-run
    cb_enabled: bool = True              # NaN/Inf abort is always on; divergence check needs a baseline
    cb_divergence_mult: float = 2.0      # abort if loss > mult × baseline-at-step over a window
    cb_window: int = 3                   # consecutive divergent evals before abort
    cb_baseline_losses: str | None = None  # optional path to control's per-eval losses (jsonl or json list)
    # io
    out_dir: str = "runs/smoke"
    log_every: int = 10
    eval_every: int = 50
    eval_batches: int = 8
    ckpt_every: int = 200

    @property
    def vocab_size(self) -> int:
        return self.vocab_bytes + self.reserved_oracle_tokens


CONFIG_HASH_EXCLUDE = {"provider", "notes", "out_dir", "log_every", "parent_ckpt"}


def load_config(path: str) -> Config:
    cfg = Config()
    overrides = json.loads(Path(path).read_text())
    valid = {f.name for f in dataclasses.fields(Config)}
    for k, v in overrides.items():
        if k not in valid:
            raise SystemExit(f"unknown config key: {k!r} (typo safety — honest data)")
        setattr(cfg, k, v)
    return cfg


# Levers added after runs were recorded: absent from the hash while at their default, so a
# configuration written before the lever existed keeps its hash (run dc34fcf000aa stays
# 6785ba1f8e213dce, 5f0a6d3ff4e8 stays b9e10b3e0a9e5938); moved, the lever enters the hash
# like any other field.
HASH_TRANSPARENT_AT_DEFAULT = {"journal_fresh_pool": False, "journal_lm_window": False, "journal_neg_frac": 0.25,
                               "journal_paired_negatives": False, "journal_paired_keep_shape": False}


def config_hash(cfg: Config) -> str:
    d = {k: v for k, v in dataclasses.asdict(cfg).items()
         if k not in CONFIG_HASH_EXCLUDE and not (k in HASH_TRANSPARENT_AT_DEFAULT and v == HASH_TRANSPARENT_AT_DEFAULT[k])}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- data


def make_synthetic(cfg: Config, n_tokens: int) -> np.ndarray:
    """Deterministic learnable byte stream: repeated seeded pattern + 5% noise."""
    rng = np.random.default_rng(cfg.seed)
    pattern = rng.integers(0, cfg.vocab_bytes, size=4096, dtype=np.uint16)
    reps = int(np.ceil(n_tokens / pattern.size)) + 1
    arr = np.tile(pattern, reps)[: n_tokens]
    noise = rng.random(arr.size) < 0.05
    arr[noise] = rng.integers(0, cfg.vocab_bytes, size=int(noise.sum()), dtype=np.uint16)
    return arr


def load_data(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    if cfg.data_mode == "synthetic":
        total = int(cfg.max_tokens * 1.05) + cfg.block_size + 1
        arr = make_synthetic(cfg, total)
    elif cfg.data_mode == "bin":
        p = (REPO_ROOT / cfg.bin_path).resolve()
        if not p.exists():
            raise SystemExit(f"bin dataset missing: {p} (build it — see notebooks/n1_kaggle.ipynb)")
        arr = np.memmap(p, dtype=np.uint16, mode="r")
    else:
        raise SystemExit(f"unknown data_mode: {cfg.data_mode}")
    split = int(len(arr) * (1.0 - cfg.val_fraction))
    return arr[:split], arr[split:]


def get_batch(arr: np.ndarray, cfg: Config, gen: torch.Generator, device: torch.device):
    hi = len(arr) - cfg.block_size - 1
    ix = torch.randint(0, hi, (cfg.batch_size,), generator=gen)
    x = torch.stack([torch.from_numpy(np.asarray(arr[i : i + cfg.block_size], dtype=np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(np.asarray(arr[i + 1 : i + 1 + cfg.block_size], dtype=np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# ----------------------------------------------------------------------------- model


class HopfieldMemory(nn.Module):
    """Variant H (RES-1): modern-Hopfield content-addressable retrieval.
    A learned bank of key/value patterns read by softmax attention (energy
    descent). Zero-initialised output projection ⇒ at init this is a no-op,
    so a fresh model with the flag on starts identical to control and learns
    the memory from there. Third residual sub-block; default off (ADR-006).
    """
    def __init__(self, cfg: "Config"):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.n_embd)
        self.q = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.keys = nn.Parameter(torch.randn(cfg.c2_mem_slots, cfg.n_embd) * 0.02)
        self.vals = nn.Parameter(torch.randn(cfg.c2_mem_slots, cfg.n_embd) * 0.02)
        self.out = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        nn.init.zeros_(self.out.weight)  # no-op at init
        self.scale = cfg.n_embd ** -0.5

    def forward(self, x):
        h = self.ln(x)
        q = self.q(h)                                   # (b,t,c)
        att = torch.softmax((q @ self.keys.t()) * self.scale, dim=-1)  # (b,t,slots)
        read = att @ self.vals                          # (b,t,c)
        return self.out(read)


class DeltaMemory(nn.Module):
    """Variant D (K1, KDA-class): a per-head recurrent state S updated by an
    error-correcting rank-one write S += beta * k (v - S^T k)^T with a
    per-channel forget gate. The Widrow-Hoff / delta-rule family. Chunk-free
    sequential scan (correctness-first; a parallel scan is a later slice).
    Zero-initialised output ⇒ no-op at init. Default off (ADR-006).
    """
    def __init__(self, cfg: "Config"):
        super().__init__()
        self.h = cfg.c2_heads
        self.dk = cfg.n_embd // cfg.c2_heads
        self.ln = nn.LayerNorm(cfg.n_embd)
        self.to_qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.to_beta = nn.Linear(cfg.n_embd, cfg.c2_heads, bias=True)
        self.to_gate = nn.Linear(cfg.n_embd, cfg.c2_heads * self.dk, bias=True)
        self.out = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        nn.init.zeros_(self.out.weight)  # no-op at init
        self.l2_keys = cfg.c2_delta_l2_keys
        self.state_clip = cfg.c2_delta_state_clip

    def forward(self, x):
        b, t, c = x.shape
        z = self.ln(x)
        q, k, v = self.to_qkv(z).split(c, dim=2)
        q = q.view(b, t, self.h, self.dk)
        k = k.view(b, t, self.h, self.dk)
        v = v.view(b, t, self.h, self.dk)
        # KDA-style stability: L2-normalise q,k so the rank-one write stays bounded
        # (unnormalised keys let |S| grow unboundedly across the scan → the v1 divergence).
        if getattr(self, "l2_keys", True):
            q = torch.nn.functional.normalize(q, dim=-1, eps=1e-6)
            k = torch.nn.functional.normalize(k, dim=-1, eps=1e-6)
        beta = torch.sigmoid(self.to_beta(z)).view(b, t, self.h, 1)        # write rate
        gate = torch.sigmoid(self.to_gate(z)).view(b, t, self.h, self.dk)  # per-channel forget
        S = torch.zeros(b, self.h, self.dk, self.dk, device=x.device, dtype=x.dtype)
        clip = getattr(self, "state_clip", 0.0)
        outs = []
        for i in range(t):
            ki = k[:, i]                          # (b,h,dk)
            vi = v[:, i]
            qi = q[:, i]
            gi = gate[:, i]
            bi = beta[:, i]
            # channel-wise forget on the key dimension (columns of S)
            S = S * gi.unsqueeze(2)
            u = torch.einsum('bhij,bhi->bhj', S, ki)      # read S^T k
            err = vi - u                                  # prediction error
            S = S + bi.unsqueeze(-1) * torch.einsum('bhi,bhj->bhij', ki, err)
            if clip > 0.0:                                 # optional belt-and-braces guard
                S = torch.clamp(S, -clip, clip)
            oi = torch.einsum('bhij,bhi->bhj', S, qi)     # read with query
            outs.append(oi.reshape(b, c))
        o = torch.stack(outs, dim=1)                      # (b,t,c)
        return self.out(o)


class Block(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int = -1):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp_up = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.mlp_down = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        # C2 associative layer (ADR-006), default off — a third residual sub-block
        if cfg.c2_variant == "hopfield":
            self.c2 = HopfieldMemory(cfg)
        elif cfg.c2_variant == "delta":
            self.c2 = DeltaMemory(cfg)
        else:
            self.c2 = None
        # the journal reader (ADR-008), one block only, default off
        self.reader = None
        if cfg.journal == "read" and layer_idx == cfg.journal_block:
            from cortex_c2b.lm_bridge import JournalReader
            self.reader = JournalReader(cfg.n_embd, cfg.journal_heads, cfg.journal_read_bytes)

    def forward(self, x, window=None, window_mask=None, gate: float = 1.0):
        b, t, c = x.shape
        h = self.ln1(x)
        q, k, v = self.attn(h).split(c, dim=2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.proj(a)
        if self.c2 is not None:
            x = x + self.c2(x)          # associative memory residual (no-op at init)
        self.last_read_mass = None
        if self.reader is not None and window is not None and gate != 0.0:
            read, mass = self.reader(x, window, window_mask)
            x = x + gate * read         # the journal as content, on the router's gate (no-op at init)
            self.last_read_mass = mass
        x = x + self.mlp_down(F.gelu(self.mlp_up(self.ln2(x))))
        return x


class VanillaGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.cue_encoder = None
        if cfg.journal == "read":
            from cortex_c2b.lm_bridge import CueEncoder
            self.cue_encoder = CueEncoder(cfg.n_embd, cfg.journal_cue_dim)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight  # tied embeddings (vanilla)
        self.apply(self._init)
        for name, p in self.named_parameters():  # scaled residual init
            if name.endswith("proj.weight") or name.endswith("mlp_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        # ADR-006: restore the C2 no-op AFTER global init overwrote it — the
        # associative residual must start at ~zero so the model begins as the
        # transformer and learns the memory from there. (A test asserts this.)
        for blk in self.blocks:
            if getattr(blk, "c2", None) is not None:
                nn.init.zeros_(blk.c2.out.weight)
            if getattr(blk, "reader", None) is not None:      # ADR-008: the reader starts as a no-op too
                nn.init.zeros_(blk.reader.out.weight)
                nn.init.zeros_(blk.reader.null)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def journal_hidden(self, idx):
        """The residual stream at the journal block, before the reader: what the
        cue encoder pools. Without a journal, the stream at the last block."""
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        stop = self.cfg.journal_block if self.cfg.journal == "read" else len(self.blocks)
        for blk in self.blocks[:stop]:
            x = blk(x)
        return x

    def read_mass(self):
        """The reader's attention mass on the retrieved bytes at the last forward (invariant 8)."""
        for blk in self.blocks:
            if getattr(blk, "reader", None) is not None and blk.last_read_mass is not None:
                return float(blk.last_read_mass.detach())
        return None

    def forward(self, idx, targets=None, window_tokens=None, window_mask=None, gate: float = 1.0,
                loss_weights=None):
        """`window_tokens` (b, r): the read window's bytes, embedded with the model's
        own `wte`; `gate`: the router's path 4 weight; `loss_weights` (b, t): per
        token weights (the curriculum's answer-only, asymmetric loss)."""
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        window = self.wte(window_tokens) if window_tokens is not None else None
        for blk in self.blocks:
            x = blk(x, window, window_mask, gate) if blk.reader is not None else blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
            if loss_weights is None:
                loss = ce.mean()
            else:
                w = loss_weights.reshape(-1).to(ce.dtype)
                loss = (ce * w).sum() / w.sum().clamp(min=1.0)
        return logits, loss


def load_parent(model: "VanillaGPT", ckpt_path: str | Path, device) -> tuple[str, list[str]]:
    """A fine tune starts from a parent checkpoint: every parent weight loads
    unchanged (strict on the parent's keys); only the journal's own modules may
    be new. Returns (parent run id, the keys that were initialised fresh)."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    result = model.load_state_dict(ck["model"], strict=False)
    if result.unexpected_keys:
        raise SystemExit(f"parent checkpoint has keys this model lacks: {result.unexpected_keys[:5]}")
    allowed = ("cue_encoder.", ".reader.")
    bad = [k for k in result.missing_keys if not any(a in k for a in allowed)]
    if bad:
        raise SystemExit(f"parent checkpoint lacks non-journal weights: {bad[:5]}")
    return ck.get("run_id", "unknown"), list(result.missing_keys)


# ----------------------------------------------------------------------------- ledger (ADR-001)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "uncommitted"


def build_record(cfg: Config, chash: str, run_id: str, started_iso: str, params: int,
                 tokens_seen: int, steps: int, wall_s: float, final_loss: float | None,
                 val_ppl: float | None, status: str, anomalies: str | None, device: torch.device) -> dict:
    lr_schedule = f"warmup{cfg.warmup_steps}+cosine(max={cfg.lr},min={cfg.min_lr_ratio}x)"
    if device.type == "cuda":
        hardware = torch.cuda.get_device_name(0)
        precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        gpu_hours = round(wall_s / 3600.0, 4)
    else:
        hardware = platform.processor() or platform.machine() or "cpu"
        precision = "fp32"
        gpu_hours = None
    return {
        "schema_version": "run-v1",
        "run_id": run_id,
        "timestamp_utc": started_iso,
        "git_commit": git_commit(),
        "config_hash": chash,
        "run": {"kind": cfg.kind, "component_under_test": cfg.component_under_test,
                "status": status, "anomalies": anomalies, "notes": cfg.notes},
        "model": {"params_total": params,
                  "architecture_id": (("journal-" if cfg.journal == "read" else "")
                                      + (f"vanilla-{max(1, round(params / 1e6))}m-byte" if cfg.c2_variant == "none"
                                         else f"c2{cfg.c2_variant}-{max(1, round(params / 1e6))}m-byte")),
                  "tokenizer_id": f"byte-v0+{cfg.reserved_oracle_tokens}oracle"},
        "training": {"tokens_seen": tokens_seen, "dataset_id": cfg.dataset_id,
                     "data_slice": cfg.data_slice, "seed": cfg.seed, "steps": steps,
                     "batch_size": cfg.batch_size, "lr_schedule": lr_schedule,
                     "precision": precision, "parent_run_id": cfg.parent_run_id},
        "compute": {"provider": cfg.provider, "hardware": hardware, "gpu_hours": gpu_hours},
        "results": {"final_train_loss": final_loss, "val_perplexity": val_ppl,
                    "benchmarks": {"routing_specialization_mi": None, "mqar_accuracy": None,
                                   "oracle_shift_recovery": None,
                                   "invariant_compliance_rate": None, "standard_suite": None}},
        "comparison": None,
    }


def validate_record(record: dict) -> None:
    try:
        import jsonschema
    except ImportError:
        print("[ledger] jsonschema not installed — CI will validate this record")
        return
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(record, schema)
    print("[ledger] record valid against run-v1")


def append_record(record: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"[ledger] appended run {record['run_id']} -> {LEDGER_PATH.relative_to(REPO_ROOT)}")


def regen_latest() -> None:
    records = []
    if LEDGER_PATH.exists():
        for line in LEDGER_PATH.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    lines = ["# LATEST — run ledger state (generated by train.py — never hand-edit)", "",
             f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')} · records: {len(records)}", ""]
    if not records:
        lines.append("No completed runs yet (expected pre-N1 execution).")
    else:
        lines += ["| run_id | date (UTC) | kind | comp | arch | tokens | steps | val_ppl | status | provider |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in records:
            ppl = r["results"]["val_perplexity"]
            lines.append("| {} | {} | {} | {} | {} | {:,} | {} | {} | {} | {} |".format(
                r["run_id"], r["timestamp_utc"][:10], r["run"]["kind"],
                r["run"]["component_under_test"] or "-", r["model"]["architecture_id"],
                r["training"]["tokens_seen"], r["training"]["steps"],
                f"{ppl:.3f}" if isinstance(ppl, (int, float)) else "null",
                r["run"]["status"], r["compute"]["provider"]))
        done = [r for r in records if r["run"]["status"] == "completed"
                and isinstance(r["results"]["val_perplexity"], (int, float))]
        if done:
            best = min(done, key=lambda r: r["results"]["val_perplexity"])
            lines += ["", f"Best val_perplexity: **{best['results']['val_perplexity']:.3f}** (run {best['run_id']})"]
    LATEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ledger] regenerated {LATEST_PATH.relative_to(REPO_ROOT)}")


# ----------------------------------------------------------------------------- train


def lr_at(step: int, total: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    p = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, help="path to config json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--regen-latest", action="store_true")
    ap.add_argument("--time-budget-min", type=float, default=None,
                    help="stop cleanly after N minutes of training: checkpoint + exit 0, no record; rerun with --resume (session-wall / preemption survival)")
    args = ap.parse_args()
    if args.regen_latest:
        regen_latest()
        return
    if not args.config:
        raise SystemExit("--config required (or --regen-latest)")

    cfg = load_config(args.config)
    chash = config_hash(cfg)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device.type == "cuda" and not use_bf16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    train_arr, val_arr = load_data(cfg)
    model = VanillaGPT(cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    # ---- ADR-008: a fine tune starts from a parent checkpoint (strict on the parent's keys)
    if cfg.parent_ckpt:
        parent_id, fresh = load_parent(model, REPO_ROOT / cfg.parent_ckpt, device)
        if cfg.parent_run_id and cfg.parent_run_id != parent_id:
            raise SystemExit(f"parent_run_id {cfg.parent_run_id} does not match the checkpoint's {parent_id}")
        cfg.parent_run_id = parent_id
        print(f"[parent] {parent_id}: parent weights loaded unchanged; {len(fresh)} journal tensors fresh (zero output)")
    # ---- ADR-008: the journal in the decode loop (default off)
    journal_ctx = None
    if cfg.journal == "read":
        from cortex_c2b import Journal, POLICY_STOP
        from cortex_c2b.crypto import key_from_env
        from cortex_c2b.lm_bridge import JournalBridge, contrastive_loss, journal_gate
        from cortex_c2b.organ_use import training_facts, plant_pool, make_batch, batch_stats, collate
        if cfg.journal_path and key_from_env() is None:
            raise SystemExit("journal_path is set but QUANTUM_CORTEX_JOURNAL_KEY is not in the environment: the run's "
                             "journal is sealed by default (ADR-007 D8); set the key before training, not after")
        facts, negatives, rep_facts = training_facts(cfg.journal_pool_facts, cfg.journal_train_seed)
        print(f"[journal] curriculum facts {rep_facts['n_facts']} negatives {rep_facts['n_negatives']} "
              f"generator {rep_facts['generator_hash']} collisions removed {len(rep_facts['removed_collisions'])}"
              f" | fresh pool per replant: {cfg.journal_fresh_pool} | window on LM steps: {cfg.journal_lm_window}"
              f" | negatives {cfg.journal_neg_frac} paired: {cfg.journal_paired_negatives}"
              f" shape kept: {cfg.journal_paired_keep_shape}")
        def run_journal():
            """The run's own scope: sealed on disk (key from the environment), policy stop --
            a measurement run never continues in memory unnoticed (ADR-007 D9)."""
            if cfg.journal_path:
                return Journal(REPO_ROOT / cfg.journal_path, key=key_from_env(), policy=POLICY_STOP)
            return Journal()
        journal_ctx = {"facts": facts, "negatives": negatives, "rng": np.random.default_rng(cfg.seed + 17),
                       "bridge": None, "planted_at": -1, "run_journal": run_journal, "refresh": 0}
    if cfg.optimizer != "adamw":
        raise SystemExit("only adamw in the control scaffold — F1 (NorMuon-class) arrives via its own ablation")
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, 0.95))

    tokens_per_step = cfg.batch_size * cfg.block_size
    total_steps = max(1, cfg.max_tokens // tokens_per_step)
    out_dir = REPO_ROOT / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"

    start_step, tokens_seen = 0, 0
    started_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_id = hashlib.sha256(f"{chash}{started_iso}".encode()).hexdigest()[:12]
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck["config_hash"] != chash:
            raise SystemExit("resume refused: config_hash mismatch (honest data — change config, change run)")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_step, tokens_seen = ck["step"] + 1, ck["tokens_seen"]
        started_iso, run_id = ck["started_iso"], ck["run_id"]
        print(f"[resume] step {start_step}/{total_steps} run {run_id}")
        if start_step >= total_steps:
            print(f"[resume] run {run_id} already complete — nothing to do (its record is in the ledger)")
            return
    elif args.resume:
        print("[resume] no checkpoint found — starting fresh")

    print(f"[run {run_id}] device={device.type} params={params:,} steps={total_steps} "
          f"tokens/step={tokens_per_step:,} config_hash={chash}")

    gen = torch.Generator().manual_seed(cfg.seed + start_step)

    def evaluate() -> float:
        model.eval()
        with torch.no_grad():
            losses = []
            for _ in range(cfg.eval_batches):
                x, y = get_batch(val_arr, cfg, gen, device)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    _, l = model(x, y)
                losses.append(l.item())
        model.train()
        return float(np.mean(losses))

    t0, loss_val, first_loss = time.time(), None, None
    val_loss = None
    anomalies = None
    status = "completed"
    cb_divergent = 0
    # build a baseline lookup for the divergence check (ADR-006 D7), if provided
    baseline_at = None
    if cfg.cb_baseline_losses:
        try:
            bp = (REPO_ROOT / cfg.cb_baseline_losses).resolve()
            raw = bp.read_text().strip()
            if raw.startswith("["):
                _bvals = json.loads(raw)                     # [{"step":s,"val_loss":v}, ...] or [v, ...]
            else:
                _bvals = [json.loads(l) for l in raw.splitlines() if l.strip()]
            _bmap = {}
            for i, item in enumerate(_bvals):
                if isinstance(item, dict):
                    _bmap[int(item["step"])] = float(item["val_loss"])
                else:
                    _bmap[(i + 1) * cfg.eval_every] = float(item)
            _bsteps = sorted(_bmap)
            def baseline_at(s, _m=_bmap, _ss=_bsteps):
                # nearest recorded baseline step <= s
                cand = [k for k in _ss if k <= s]
                return _m[cand[-1]] if cand else (_m[_ss[0]] if _ss else None)
            print(f"[circuit-breaker] divergence check armed against {len(_bmap)} baseline points")
        except Exception as e:
            print(f"[circuit-breaker] baseline load failed ({e}); NaN/Inf abort still active")
            baseline_at = None
    try:
        cur_step = start_step
        for step in range(start_step, total_steps):
            cur_step = step
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, cfg)
            curriculum_step = (journal_ctx is not None
                               and journal_ctx["rng"].random() < cfg.journal_curriculum_ratio)
            if curriculum_step:
                # ADR-008 invariant 6: organ use is taught; the pool is replanted under the
                # current encoder every `journal_pool_refresh` steps (a memory scope)
                if journal_ctx["bridge"] is None or step - journal_ctx["planted_at"] >= cfg.journal_pool_refresh:
                    if cfg.journal_fresh_pool and journal_ctx["refresh"] > 0:
                        # v2: never the same facts twice -- a new seed per replant, decontaminated like the first
                        journal_ctx["facts"], journal_ctx["negatives"], _ = training_facts(
                            cfg.journal_pool_facts, cfg.journal_train_seed + 2 * journal_ctx["refresh"])
                    journal_ctx["refresh"] += 1
                    jb = JournalBridge(model, Journal(), k=cfg.journal_k, budget_bytes=cfg.journal_read_bytes,
                                       seed=cfg.seed, shuffle_seed=cfg.seed + step, device=device)
                    model.eval(); plant_pool(jb, journal_ctx["facts"], now=float(step)); model.train()
                    journal_ctx["bridge"], journal_ctx["planted_at"] = jb, step
                jb = journal_ctx["bridge"]
                progress = step / max(1, total_steps)
                forced = cfg.journal_forced_frac * max(0.0, 1.0 - progress)      # the bootstrap decays to zero
                model.eval()
                examples = make_batch(jb, journal_ctx["facts"], journal_ctx["negatives"], cfg.batch_size,
                                      journal_ctx["rng"], negatives_frac=cfg.journal_neg_frac, forced_frac=forced,
                                      paired=cfg.journal_paired_negatives, keep_shape=cfg.journal_paired_keep_shape)
                model.train()
                x, y, w, wt, wm, qseq, sseq = collate(jb, examples, cfg.journal_asym_weight, cfg.journal_read_bytes)
                gate = journal_gate(cfg.journal_route) if cfg.journal_route != "router" else 1.0   # a learned router is measured apart (D9 held)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    _, loss_ce = model(x, y, window_tokens=wt, window_mask=wm, gate=gate, loss_weights=w)
                    loss_c = contrastive_loss(jb.cue_tensor(qseq), jb.cue_tensor(sseq))
                    loss = loss_ce + cfg.journal_contrastive_weight * loss_c
                journal_ctx["last_stats"] = {**batch_stats(examples), "loss_ce": float(loss_ce.detach()), "loss_contrastive": float(loss_c.detach())}
            else:
                x, y = get_batch(train_arr, cfg, gen, device)
                wt = wm = None
                if journal_ctx is not None and cfg.journal_lm_window and journal_ctx["bridge"] is not None:
                    # v2: the reader stays on during language modelling, fed with whatever the journal returns
                    # for each span, so an irrelevant read costs nothing after training (the e_S arm)
                    jb = journal_ctx["bridge"]
                    model.eval()
                    cues = jb.cues([row.tolist() for row in x])
                    windows = [jb.read(c).window for c in cues]
                    model.train()
                    wt, wm = jb.window_tensors(windows, cfg.journal_read_bytes)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    _, loss = model(x, y, window_tokens=wt, window_mask=wm, gate=1.0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            loss_val = loss.item()
            # circuit breaker — hard criterion: NaN/Inf is instant death (ADR-006 D7)
            if cfg.cb_enabled and (math.isnan(loss_val) or math.isinf(loss_val)):
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                            "tokens_seen": tokens_seen, "config_hash": chash,
                            "started_iso": started_iso, "run_id": run_id}, ckpt_path)
                anomalies = f"NaN/Inf loss at step {step} — circuit breaker (hard abort)"
                print(f"[circuit-breaker] {anomalies}")
                status = "aborted"
                cur_step = step
                break
            if first_loss is None:
                first_loss = loss_val
            tokens_seen += tokens_per_step
            if step % cfg.log_every == 0 or step == total_steps - 1:
                tps = tokens_seen / max(1e-9, time.time() - t0)
                print(f"step {step}/{total_steps} loss {loss_val:.4f} lr {opt.param_groups[0]['lr']:.2e} tok/s {tps:,.0f}")
                if journal_ctx is not None and journal_ctx.get("last_stats"):
                    st = journal_ctx["last_stats"]
                    print(f"  journal: hit {st['retrieval_hit']:.2f} abstain-target {st['abstain_target_rate']:.2f} "
                          f"forced {st['n_forced']} neg {st['n_negative']} ce {st['loss_ce']:.3f} contrastive {st['loss_contrastive']:.3f}")
            if step and step % cfg.eval_every == 0:
                val_loss = evaluate()
                print(f"  eval: val_loss {val_loss:.4f} ppl {math.exp(val_loss):.3f}")
                # circuit breaker — divergence criterion vs a control baseline (ADR-006 D7)
                if cfg.cb_enabled and baseline_at is not None:
                    b = baseline_at(step)
                    if b is not None and val_loss > cfg.cb_divergence_mult * b:
                        cb_divergent += 1
                        print(f"[circuit-breaker] divergent eval {cb_divergent}/{cfg.cb_window} "
                              f"(val {val_loss:.4f} > {cfg.cb_divergence_mult}× baseline {b:.4f} at step {step})")
                    else:
                        cb_divergent = 0
                    if cb_divergent >= cfg.cb_window:
                        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                                    "tokens_seen": tokens_seen, "config_hash": chash,
                                    "started_iso": started_iso, "run_id": run_id}, ckpt_path)
                        anomalies = (f"diverged at step {step}: val {val_loss:.4f} > "
                                     f"{cfg.cb_divergence_mult}× baseline {b:.4f} over {cfg.cb_window} evals "
                                     f"— circuit breaker")
                        print(f"[circuit-breaker] {anomalies}")
                        status = "aborted"
                        cur_step = step
                        break
            if step and step % cfg.ckpt_every == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                            "tokens_seen": tokens_seen, "config_hash": chash,
                            "started_iso": started_iso, "run_id": run_id}, ckpt_path)
            if args.time_budget_min is not None and (time.time() - t0) / 60.0 >= args.time_budget_min:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                            "tokens_seen": tokens_seen, "config_hash": chash,
                            "started_iso": started_iso, "run_id": run_id}, ckpt_path)
                print(f"[budget] {args.time_budget_min:.0f} min reached at step {step}/{total_steps} — "
                      f"checkpoint saved, exiting cleanly (no record); rerun with --resume to continue")
                return
    except KeyboardInterrupt:
        status, anomalies = "aborted", "KeyboardInterrupt — checkpoint holds last saved step"
    val_loss = evaluate() if status == "completed" else val_loss
    wall = time.time() - t0
    if status == "completed" and first_loss is not None and loss_val is not None and loss_val >= first_loss:
        anomalies = f"train loss did not decrease ({first_loss:.4f} -> {loss_val:.4f})"
    val_ppl = round(math.exp(val_loss), 4) if isinstance(val_loss, float) else None
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": (total_steps - 1) if status == "completed" else cur_step,
                "tokens_seen": tokens_seen, "config_hash": chash,
                "started_iso": started_iso, "run_id": run_id}, ckpt_path)
    record = build_record(cfg, chash, run_id, started_iso, params, tokens_seen,
                          total_steps if status == "completed" else start_step,
                          wall, round(loss_val, 4) if loss_val is not None else None,
                          val_ppl, status, anomalies, device)
    if journal_ctx is not None and cfg.journal_hm_arm and status == "completed":
        # ADR-008 invariant 9: the frozen protocol's LM arm, its result in the record and as a file
        from cortex_c2b.hm_lm import run_hm_lm
        vb = [get_batch(val_arr, cfg, gen, device) for _ in range(cfg.eval_batches)]
        hm = run_hm_lm(model, journal_ctx["run_journal"], n_facts=cfg.journal_hm_facts, n_negctrl=cfg.journal_hm_negctrl,
                       k=cfg.journal_k, budget=cfg.journal_read_bytes, window_len=cfg.journal_read_bytes,
                       val_batches=vb, on_disk=bool(cfg.journal_path))
        (out_dir / "hm-lm.json").write_text(json.dumps(hm, indent=2) + "\n", encoding="utf-8")
        kept = REPO_ROOT / "metrics" / "mqar" / f"hm-lm-{run_id}.json"          # runs/ is never committed; this is
        kept.parent.mkdir(parents=True, exist_ok=True)                          # the artefact a RESULTS row cites
        kept.write_text(json.dumps(hm, indent=2) + "\n", encoding="utf-8")
        suite = {k: (float(v) if isinstance(v, bool) else v) for k, v in hm.items()
                 if k.startswith("hm_") and isinstance(v, (int, float)) and v is not None}
        suite["hm_persistent"] = float(hm["persistent"]); suite["hm_claimable"] = float(hm["claimable"])
        record["results"]["benchmarks"]["standard_suite"] = suite
        print(f"[hm-lm] {hm['verdict']} | recall on {hm['hm_recall_on']:.3f} off {hm['hm_recall_off']:.3f} "
              f"skill delta {hm['hm_skill_delta']:.4f} invalid citation {hm['hm_invalid_citation_on']:.3f} "
              f"| retained: {kept}")
    validate_record(record)
    append_record(record)
    regen_latest()
    print("\n[record — copy this line into metrics/runs.jsonl on the machine that commits]")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
