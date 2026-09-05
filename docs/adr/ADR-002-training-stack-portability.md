# ADR 002 — Training stack, and where portability actually applies

- **Status:** Proposed
- **Date:** 2026-08-07
- **Deciders:** Nicolas Van Eeckhout

## Context

Question raised: should the model be trained "on an abstraction layer" (ggml-class, hardware-agnostic) rather than directly on GPU — and is the portability concern relevant to training, to serving, or both? Constraint: the 2–10× performance tax mentioned for naive abstraction layers is unacceptable — no adoption without performance.

## Decision

1. **ggml / ONNX / GGUF are inference runtimes, never training targets.** The training abstraction layer already exists and is the de-facto standard: **PyTorch** (reference stack; JAX/XLA variant only if a TPU Research Cloud grant materializes). Training code is written **device-agnostic** (`device` parameterized: cuda / rocm-as-cuda / cpu / mps).

2. **Transparency is about correctness, not wall-clock.** Across backends the result is the same up to floating-point non-determinism (kernel order, seeds — the config hash pins what can be pinned). Wall-clock is *not* transparent: CPU pretraining is 2–3 orders of magnitude slower. Therefore CPU/MPS paths exist **for unit tests and CI smoke only**; pretraining runs on GPU (or TPU under TRC), full stop.

3. **Training-side portability reduces to one precise rule, dictated by our own compute plan.** The free/granted compute spans three silicons: NVIDIA (Kaggle, rented H100), **AMD/ROCm (EuroHPC LUMI-class systems)**, TPU (TRC). Rule: **pure-framework ops only; any custom kernel must be Triton** (portable NVIDIA/AMD) **and budgeted** — CUDA-only ops are prohibited, otherwise a granted allocation on AMD or TPU hardware would be unusable.

4. **Serving-side portability is everything** — that is NOW-4's runtime matrix (GGUF/llama.cpp, vLLM, ONNX, HF remote code), unchanged.

5. **Performance clause.** The 2–10× tax belongs to *naive or immature* abstraction layers. Mature layers (PyTorch/cuDNN and Triton at training; llama.cpp and vLLM at serving) already embed hand-tuned kernels: consuming them costs approximately nothing. What would cost adoption is inventing our own compute layer or shipping exotic unbudgeted ops — prohibited here and in NOW-4.

## Consequences

- N1's notebook is written device-agnostic, runs CUDA on Kaggle, and its CPU path is exercised only as a correctness smoke test.
- Benchmark tooling is **installed as a dependency** where licensing allows (e.g. zoology, Apache-2.0), not copied into the repo.
- Any future custom kernel proposal must arrive with: a Triton implementation, a fallback pure-framework path, and a measured speedup in the ledger justifying it.
