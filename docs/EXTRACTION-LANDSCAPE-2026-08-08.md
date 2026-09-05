# EXTRACTION — the August 2026 landscape — 2026-08-08

**Input:** Nicolas's curated list (Top 10 relevant + Top 5 original). **Pipeline applied to every entry:** LICENSE first, classify, load beyond the README where the code carries the idea, extract **knowledge only** (our words, equations, file:line evidence), filter by complementarity with continuity (D15) and C1–C6. **Legal verdict up front: PASS by construction** — zero code copied anywhere; two entries caught by the protocol and ruled read-only.

## 1. License table (verified live via raw, 2026-08-08)

| Repo | License | Ruling |
|---|---|---|
| karpathy/nanochat | MIT (verified) | dependency/study |
| allenai/OLMo | Apache-2.0 (verified) | dependency/study |
| deepseek-ai/DeepSeek-V3 | MIT (verified) — **code**; model weights carry separate licenses, verify at graft | study |
| QwenLM/Qwen3 | **not found at standard paths** | **read-only pending clarification** (weights commonly Apache-2.0 on HF — verify at graft) |
| MoonshotAI/Kimi-K3 | read in full earlier | cleared (see EXTRACTION-K3) |
| BlinkDL/RWKV-LM | Apache-2.0 (verified earlier) | study |
| state-spaces/mamba | Apache-2.0 (verified earlier) | study |
| microsoft/BitNet | MIT (verified earlier) | dependency/study (RES-7) |
| facebookresearch/blt | CC-class (verified earlier) | **read-only** — paper is the source |
| vllm-project/vllm | Apache-2.0 (verified earlier) | dependency (NOW-4) |
| SamsungSAILMontreal/TinyRecursiveModels | MIT (verified) | study |
| ML-GSAI/LLaDA | **not found at standard paths** | **read-only pending clarification** |
| goombalab/hnet | MIT (verified) | study |
| lucidrains/titans-pytorch | MIT (verified) | study (unofficial impl; papers are truth) |
| NX-AI/xlstm | Apache-2.0 (verified) | study |

## 2. Loaded and read (five idea-dense repos, code evidence)

- **TRM (TinyRecursiveModels).** A tiny core recursing in latent space with deep supervision; evidence: `models/recursive_reasoning/trm.py`, `trm_singlez.py` — and notably `transformers_baseline.py`: **they ship their own control**, a culture cousin of ours. **Retained → RES-13** (latent recursive refinement as an inference-time compute dial). Skeptic (from Nicolas's own brief, kept): the 1000× per-example augmentation regime explains part of the ARC gap; transfer to language modeling unproven — that is precisely what our ablation would test.
- **H-Net.** Learned dynamic chunking: `hnet/modules/dc.py` computes `boundary_prob`/`boundary_mask` from hidden states — segmentation itself is a gradient-trained decision, one level more radical than BLT's entropy heuristic. **Retained → third tokenizer candidate** for the open question of hub decision 019 (and MIT, unlike BLT's CC — the permissive one of the pair). Resonance graved: boundary-by-information is the same gesture as **D4.8's admission-by-surprise**, applied to segmentation.
- **titans-pytorch.** Test-time neural memory: `neural_memory.py` implements surprise-driven updates with momentum orders and — line 158 — **spectral norming of the surprise update via Newton-Schulz iteration**: the same orthogonalization family as Muon (F1). Cross-pollination fact recorded. **Positioning claim graved (the analyst's third resonance, now ours in writing): no test-time-memory work ships an episodic/semantic dissociation protocol — the H.M. protocol is our claimed angle** for RES-2/C2b. Papers remain the source of truth; this repo is a comprehension aid.
- **xLSTM.** Matrix-state mLSTM with exponential gating: `xlstm/blocks/mlstm/{cell,layer,backends}.py`. Same fast-weight family as KDA/DeltaNet — **industrial citation for C2's positioning**, plus a EuroHPC-narrative neighbor (Linz) worth one line in the Playground application.
- **nanochat.** Full pipeline in ~8k lines, MIT, `scripts/ tasks/ tests/` structure confirmed — the N1-adjacent reference stands. **Honest flag:** the claimed `autoresearch` extension (March 2026) is **absent from the main-branch tarball** — unverified as stated; possibly a branch, fork, or separate repo. Not used until located and licensed.

## 3. Assessed (established knowledge; targeted loads deferred to need)

- **OLMo — the open-data North Star.** Radical openness (weights *and* data): its released pretraining/instruction mixes are **NOW-7 source candidates** (license per dataset, verified at ingestion). The provenance culture matches ours.
- **DeepSeek V3/V4 — MLA.** Latent KV compression = serving-memory economics; noted for NOW-4/serving, already met in the wild inside k3-in-c's MLA layers.
- **Qwen3-Next — gated DeltaNet at production scale.** The delta rule industrialized: with KDA (K3) and xLSTM, three independent industrial confirmations that **C2's family is the winning stream** — and that hybrids (linear + periodic full attention) dominate, which is exactly the stackable-citizen constraint. Their own caution kept: pure linear (Mamba/RWKV alone) lags on short contexts — we never bet on pure.
- **RWKV-7 / Mamba.** The constant-memory lineage and the SSM foundation; same hybrid caution; RWKV also remains the community-compute precedent (D7).
- **BitNet & TRM — the two poles of the thesis, now citable baselines:** capability **per bit** (RES-7) and capability **per parameter** (RES-13) both have existing extreme incumbents; our register cites them instead of pretending a void.
- **LLaDA / Dream — diffusion LLMs.** As a *generation paradigm*: **set aside** — our decode contracts (C4) and span references (RES-8) assume autoregressive semantics. Retained narrowly: **masked-diffusion infilling as a candidate *edit mode* for RES-8's typed diffs** (infilling is diffusion's home turf) — noted in RES-8's orbit, no entry.
- **vLLM / llama.cpp.** Already the two pillars of NOW-4; vLLM's 2026 AMD/Intel/TPU spread reinforces ADR-002's portability bet.

## 4. Set aside, honestly

Diffusion as the core paradigm; pure-linear architectures (their own weakness on short contexts, per the brief); nanochat-autoresearch methodology (unverified); everything GPL/CC/unlocated (BLT code, Qwen3 pending, LLaDA pending) — papers only.

## 5. Ledger of this pass

Fifteen entries processed; five loaded with file:line evidence; two read-only rulings issued; one new register entry (RES-13); one positioning claim graved (the H.M. angle vs Titans); three citations added to C2's positioning (KDA, DeltaNet/Qwen3-Next, mLSTM/xLSTM); one tokenizer candidate added (H-Net); one NOW-7 source lead (OLMo data); zero lines of code copied. Promotion beyond this document follows the register rule: dated ADR, ledger runs.
