# OSS SURVEY — relevant open-source repositories — 2026-08-07

**Purpose:** curated preselection of open-source repositories worth loading *beyond the README* — structure, licenses, and specific files inspected — to extract ideas that strengthen quantum-cortex, challenged by the expert committee. Zero plagiarism: the protocol below is binding.

## Anti-plagiarism protocol (binding — legal role)

1. **Default mode = read-only inspiration.** Understand the mechanism, close the file, re-implement from understanding and from the papers, **optimize for our context, and stay critical** about what is integrated — the skeptic validates every integration through its own ablation. Ideas with citation are science; copied code is plagiarism.
2. **Core rule (hardened 2026-08-07, same day):** the implementations of the core — C1–C6 and anything on the model's critical path — are **always written from scratch**. No verbatim reuse in the core, whatever the license. Borrowed elements can only ever be **additional and peripheral** (tooling, benchmarks, utilities), never the heart of what we build.
3. **Preference order for peripheral needs:** pip **dependency** (cleanest provenance, never vendored) > **re-implementation from the paper** > **adaptation as a last resort** — and adaptation only under a permissive, compatible license (MIT / BSD / Apache-2.0), with (a) attribution in `NOTICE`, (b) a provenance header in the file (source repo, commit, license), (c) an entry in this survey.
4. **GPL / AGPL / custom / unclear licenses:** inspiration only, zero code reuse. When a LICENSE file is non-standard, read it in full before deciding — until then the repo is read-only.
5. **Datasets** carry their own licenses (e.g. FineWeb-Edu: ODC-By → attribution required).
6. Every borrowed *idea* is cited in the whitepaper's related-work section — the positioning section hub decision 019 already mandates.

## Loaded and verified today (tarball, license read, structure inspected)

### 1. KellerJordan/modded-nanogpt — **MIT** (verified)
The GPT-2 speedrun: community-iterated record of *training efficiency per dollar* — Muon optimizer, architectural and schedule tricks, Triton kernels. Inspected: `train_gpt.py` (2 382 lines, single-file trainer), `triton_kernels.py`, `dc_triton_kernels.py`, `records/` (dated speedrun history — a ledger culture cousin). **Idea for cortex:** the N1 control should start from these training-efficiency lessons, not vanilla-2019 hyperparameters — on a free-tier budget, optimizer efficiency is compute. Triton-only kernels also model ADR-002's rule. **Skeptic:** speedrun tricks are tuned for one benchmark; re-validate each on our ladder before adoption. Maps to N1, ADR-002.

### 2. HazyResearch/zoology — **Apache-2.0** (verified)
Synthetic-task framework for memory/recall in sequence models; **MQAR confirmed present**: `zoology/experiments/mqar_example_configs/` (original, composition, forgetting variants) + `zoology/config.py`. **Idea for cortex:** our C2 benchmark installs as a dependency (protocol rule 4), never copied; the forgetting/composition variants extend our declared capability suite. **Skeptic:** synthetic wins must be paired with the perplexity guard — the advancement rule already enforces it. Maps to C2, ADR-001, N2.

### 3. ml-jku/hopfield-layers — **custom JKU notice** (non-standard LICENSE header verified)
The reference implementation of modern Hopfield networks (Ramsauer et al.), `hflayers/` (`functional.py`, `activation.py`). **Ruling per protocol rule 3: read-only inspiration** until the full license text is read and cleared — implement C2 from the paper's math, not from this code. **Skeptic:** the paper is the source of truth anyway; the repo is a comprehension aid. Maps to C2, RES-1.

## Preselection to load next (same discipline: tarball → LICENSE → structure → targeted files)

| Repo | License (to confirm on load) | Study | Idea for cortex | Maps to |
|---|---|---|---|---|
| karpathy/nanoGPT | MIT | `train.py`, `model.py` | the minimal-trainer skeleton discipline for N1 | N1 |
| karpathy/nanochat | MIT | full pipeline layout | the ~$100 pretrain→SFT budget blueprint | N5, ladder |
| pytorch/torchtitan | BSD-3 | parallelism configs | clean modern multi-GPU pretraining for the 1–3B step | post-N5 |
| huggingface/nanotron | Apache-2.0 | 3D-parallel internals | EuroHPC-scale training path | post-N5 |
| ggml-org/llama.cpp | MIT | op set, GGUF format | the portability contract our ops must fit | NOW-4 |
| vllm-project/vllm | Apache-2.0 | model integration API | day-one datacenter serving of a custom arch | NOW-4 |
| EleutherAI/lm-evaluation-harness | MIT | task registry | the `standard_suite` slot of run-v1 | ADR-001 |
| state-spaces/mamba | Apache-2.0 | kernel + fallback-path packaging | how a custom-arch model shipped adoption-wide | NOW-4, ADR-002 |
| fla-org/flash-linear-attention | MIT | Triton linear-attention zoo | portable Triton patterns near C2's math | C2, ADR-002 |
| lucidrains/titans-pytorch | MIT | unofficial Titans modules | test-time memory mechanics (paper remains the truth source) | RES-2 |
| BlinkDL/RWKV-LM | Apache-2.0 | community-training history | the "compute follows demo" precedent, structurally | D7, N5 |
| geoopt/geoopt | Apache-2.0 | Lorentz manifold optimizers | Riemannian optimization tooling for mixed curvature | RES-3 |
| HuggingFaceFW/fineweb-edu (dataset) | ODC-By | slice tooling | the open pretraining corpus, attribution required | N1 |

## Amendment — GitOps tooling (2026-08-07, same day)

Question addressed: which repositories are good sources for the **GitOps discipline** (repo as source of truth for metrics, state, pipelines) in our context — and what about GitLab?

### Loaded and verified today
1. **iterative/dvc — Apache-2.0 (verified).** Data Version Control: metrics **diffable in git** (`dvc/commands/metrics.py` inspected) — the direct cousin of our run ledger; and the clean way to version dataset slices and checkpoints *outside* git while keeping their hashes *inside* it. **Idea:** adopt as a dependency for data/checkpoint versioning at N1. **Skeptic:** DVC's own metrics feature must not replace `runs.jsonl` — one source of truth, ours.
2. **iterative/cml — Apache-2.0 (verified).** Continuous Machine Learning: CI posts metrics reports on commits/PRs; native on **GitHub Actions and GitLab CI** (`bin/cml`, `src/` inspected). **Idea:** the CI job that validates every ledger line against `run-v1.schema.json` and regenerates `LATEST.md` is this pattern — plain CI first; CML itself only if PR-comment reports prove useful. Maps to ADR-001.
3. **facebookresearch/hydra — MIT (verified).** Config composition (`hydra/compose.py`) → disciplined, hashable resolved configs feeding `config_hash`. **Skeptic:** a dataclass + YAML may suffice at 30M — decide at N1, complexity measured first.

### Assessed without loading (positioning)
- **MLflow / Weights & Biases** — server/SaaS-centric tracking: the **anti-pattern** for a git-as-truth ledger. At most an optional exporter later, never a dependency.
- **Argo CD / Flux** — GitOps proper (Kubernetes reconciliation from git): relevant **only at the serving/deployment stage**; out of scope until then.
- **nteract/papermill** — parameterized notebook execution; candidate for driving the N1 Kaggle notebook from a config. Load before N1 if adopted.

### Hosting note (GitHub vs GitLab)
The open ML ecosystem lives overwhelmingly on GitHub — no flagship ML repository on GitLab justifies moving; discovery and contribution gravity decide: **quantum-cortex stays on GitHub**. GitLab keeps two real roles: (a) a CI platform with first-class DVC/CML support; (b) a **self-hosted option for the private `quantum-*` repositories** if sovereignty is ever wanted. Some HPC centers run their own GitLab instances — check per-center at EuroHPC Stage 1. Note, not a decision; promotion follows the register rule (dated ADR).

## Amendment — Frontier engines & the K3 wave (2026-08-08)

Question addressed: repositories of the kimi-k3-in-c genre (transparent hardware mobilization: disk/RAM/VRAM/device pools), loaded or assessed to re-challenge the architecture.

### Loaded and verified
- **FareedKhan-dev/kimi-k3-in-c — Apache-2.0 (verified).** Portable C99 inference of the real Kimi K3 (2.78T): routed experts (1.45 TB) never resident, multiplied from packed 4-bit form via async disk streaming (`src/io/k3_trunk.c`); dense trunk resident to a chosen depth. Same model in 8 GB or 224 GB, byte-identical output — **memory is a dial, not a floor**. Study target: `src/core/k3_ops.c` (KDA/MLA hybrid, MoE 16/896, AttnRes). Empirical exhibit for RES-5/RES-6 and the bandwidth-bound thesis (ADR-002 §5). Author-measured numbers; single-author repo — claims until reproduced.

### Assessed (search + cards; licenses verified via raw where noted)
- **MoonshotAI/Kimi-K3** — model card of the 2.8T open-weights frontier MoE (KDA + AttnRes + Stable LatentMoE, 16/896, 1M context, MXFP4/MXFP8 QAT). Role for us: **default C5 frontier graft** via its OpenAI-compatible API; study sources: KDA and AttnRes papers for the C2 positioning section. License of the weights: verify at graft time (K2 was modified-MIT).
- **Lizonghang/prima.cpp** — distributed on-device inference on heterogeneous *home* clusters (mixed CPU/GPU, low RAM, Wi-Fi, mixed OSs); pipelined-ring parallelism overlapping disk I/O; **Halda**, a heterogeneity-aware scheduler co-optimizing per-device workloads. **RES-5's closest living relative.** License file not found at standard paths — read-only until clarified (protocol rule 4).
- **SJTU-IPADS/PowerInfer — MIT (verified).** Hot/cold-neuron sparsity split across CPU/GPU; PowerInfer-2 decomposes matrices into **fine-grained neuron-cluster computations** with segmented caching and cluster-level pipelining — fragmentation at sub-layer granularity, directly relevant to RES-6/SPEC-3.
- **exo-explore/exo — Apache (verified).** Automatic model partitioning across a pool of everyday heterogeneous devices — the consumer-cluster expression of the resource-pool idea.
- **Mozilla-Ocho/llamafile — Apache-2.0 (verified).** One cross-OS executable (Cosmopolitan libc): NOW-4 portability pushed to its extreme.
- **b4rtaz/distributed-llama — MIT (verified)** and **bigscience-workshop/petals — MIT (verified).** Tensor-parallel over cheap links, and volunteer-distributed serving of large models — two more pool topologies.
- **tinygrad/tinygrad** (license header custom-styled — confirm on load) and **karpathy/llm.c — MIT (verified).** The lazy multi-backend planner philosophy, and the genre's origin.
- Ecosystem signals: MLX out-of-core feature request ("storage bandwidth grows faster than RAM capacity"); a pure-C GLM-5.2 744B engine streaming experts in 16 GB — the dial pattern is generalizing.

### Addendum (same day) — the numerical-base lead
- **microsoft/BitNet (bitnet.cpp) — MIT (verified).** The reference stack for **ternary {-1, 0, +1} weights at 1.58 bits/weight**, trained low-bit from scratch (QAT with high-precision latent weights), with CPU/GPU kernels and open 2B-class weights. The strongest existing answer to "play with the numerical base": weight multiplications collapse to additions/subtractions, memory shrinks ~10× vs fp16. Study source and kernel reference for RES-7; dependency-class license.

### Addendum (same day) — decoding & tokenization leads
- **dottxt-ai/outlines — Apache-2.0 (verified)** and **mlc-ai/xgrammar — Apache-2.0 (verified).** Runtime-switchable constrained decoding (grammars/JSON schemas at decode time): the dependency-class tooling for C4's dynamic span contracts (RES-8).
- **facebookresearch/blt — Creative Commons license (verified: CC-class, non-commercial family).** Byte Latent Transformer: dynamic entropy-based patches instead of fixed tokens. **Protocol rule 4 catch: read-only, never a dependency** — incompatible with an Apache-2.0 ecosystem; the paper is the study source. Recorded as a tokenizer-direction candidate for the open question of hub decision 019, alongside hierarchical RVQ-style semantic codes (papers).

### Addendum (2026-08-08, evening) — the full landscape pass
Fifteen repositories from the founder's curated list processed through the binding pipeline (LICENSE first, load beyond README, knowledge-only extraction): see **`EXTRACTION-LANDSCAPE-2026-08-08.md`** — five loaded with file:line evidence (TRM, H-Net, titans-pytorch, xLSTM, nanochat), two read-only rulings issued (Qwen3 and LLaDA: LICENSE not found at standard paths), zero code copied.

**Committee verdict:** the execution plane (how weights and fragments move across tiers) is where the industry is innovating fastest — our weights architecture C1–C6 is unchanged by this wave; RES-6 and SPEC-3 capture it on our side. Core stays from scratch; these repos are study material and, where permissive, peripheral dependencies at most.

## Process

- Loading discipline = today's: tarball via codeload, LICENSE first, structure, targeted files; findings appended here as **dated amendments** — never silent edits.
- This survey feeds N1/N2 preparation; it is not a roadmap step of its own. Promotion of any extracted idea follows the register rule: dated ADR amendment citing sources.

## Committee sign-off

Architecture: the paper is always the source of truth; repos accelerate comprehension. Training: modded-nanogpt lessons enter N1 or the free tier is wasted. Systems: mamba's packaging (kernel + fallback) is the template for any custom op we ever ship. Product: dependencies over vendoring keeps the repo small and the provenance clean. Red-team: license column is verified on load, never assumed — one repo already required a read-only ruling today.
