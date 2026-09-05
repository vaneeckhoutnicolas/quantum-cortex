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
    # io
    out_dir: str = "runs/smoke"
    log_every: int = 10
    eval_every: int = 50
    eval_batches: int = 8
    ckpt_every: int = 200

    @property
    def vocab_size(self) -> int:
        return self.vocab_bytes + self.reserved_oracle_tokens


CONFIG_HASH_EXCLUDE = {"provider", "notes", "out_dir", "log_every"}


def load_config(path: str) -> Config:
    cfg = Config()
    overrides = json.loads(Path(path).read_text())
    valid = {f.name for f in dataclasses.fields(Config)}
    for k, v in overrides.items():
        if k not in valid:
            raise SystemExit(f"unknown config key: {k!r} (typo safety — honest data)")
        setattr(cfg, k, v)
    return cfg


def config_hash(cfg: Config) -> str:
    d = {k: v for k, v in dataclasses.asdict(cfg).items() if k not in CONFIG_HASH_EXCLUDE}
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


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp_up = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.mlp_down = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head

    def forward(self, x):
        b, t, c = x.shape
        h = self.ln1(x)
        q, k, v = self.attn(h).split(c, dim=2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.proj(a)
        x = x + self.mlp_down(F.gelu(self.mlp_up(self.ln2(x))))
        return x


class VanillaGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight  # tied embeddings (vanilla)
        self.apply(self._init)
        for name, p in self.named_parameters():  # scaled residual init
            if name.endswith("proj.weight") or name.endswith("mlp_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


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
                  "architecture_id": f"vanilla-{max(1, round(params / 1e6))}m-byte",
                  "tokenizer_id": f"byte-v0+{cfg.reserved_oracle_tokens}oracle"},
        "training": {"tokens_seen": tokens_seen, "dataset_id": cfg.dataset_id,
                     "data_slice": cfg.data_slice, "seed": cfg.seed, "steps": steps,
                     "batch_size": cfg.batch_size, "lr_schedule": lr_schedule,
                     "precision": precision},
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
    try:
        for step in range(start_step, total_steps):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, cfg)
            x, y = get_batch(train_arr, cfg, gen, device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            loss_val = loss.item()
            if first_loss is None:
                first_loss = loss_val
            tokens_seen += tokens_per_step
            if step % cfg.log_every == 0 or step == total_steps - 1:
                tps = tokens_seen / max(1e-9, time.time() - t0)
                print(f"step {step}/{total_steps} loss {loss_val:.4f} lr {opt.param_groups[0]['lr']:.2e} tok/s {tps:,.0f}")
            if step and step % cfg.eval_every == 0:
                val_loss = evaluate()
                print(f"  eval: val_loss {val_loss:.4f} ppl {math.exp(val_loss):.3f}")
            if step and step % cfg.ckpt_every == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                            "tokens_seen": tokens_seen, "config_hash": chash,
                            "started_iso": started_iso, "run_id": run_id}, ckpt_path)
    except KeyboardInterrupt:
        status, anomalies = "aborted", "KeyboardInterrupt — checkpoint holds last saved step"
    val_loss = evaluate() if status == "completed" else val_loss
    wall = time.time() - t0
    if status == "completed" and first_loss is not None and loss_val is not None and loss_val >= first_loss:
        anomalies = f"train loss did not decrease ({first_loss:.4f} -> {loss_val:.4f})"
    val_ppl = round(math.exp(val_loss), 4) if isinstance(val_loss, float) else None
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": total_steps - 1,
                "tokens_seen": tokens_seen, "config_hash": chash,
                "started_iso": started_iso, "run_id": run_id}, ckpt_path)
    record = build_record(cfg, chash, run_id, started_iso, params, tokens_seen,
                          total_steps if status == "completed" else start_step,
                          wall, round(loss_val, 4) if loss_val is not None else None,
                          val_ppl, status, anomalies, device)
    validate_record(record)
    append_record(record)
    regen_latest()
    print("\n[record — copy this line into metrics/runs.jsonl on the machine that commits]")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
