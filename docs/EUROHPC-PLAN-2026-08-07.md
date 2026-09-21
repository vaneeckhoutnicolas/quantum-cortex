> **Historical document, not the current state.** True when written; kept because the sequence of decisions is part of the method. For where the project stands now, read `docs/RESULTS.md`, `docs/roadmap.md` and `docs/WHITEPAPER.md`; for the labels, `docs/NAVIGATION.md`.

# EUROHPC PLAN — eligibility, two-stage access, trainability guarantees — 2026-08-07

**Goal:** be certain that if the EuroHPC route is taken, quantum-cortex actually trains in that environment — before depending on it. This document amends the earlier plan ("applications gated on N5") with a two-stage strategy; the amendment is dated, the earlier statement is kept in the REPRISE with its amendment marker.

## 1. Eligibility — Win2Win SRL

- The **AI Factories Industrial Innovation track** (Playground / Fast Lane / Large Scale) is **open and free of charge to EU AI SMEs and startups** for innovation purposes. Belgium is an EU member and EuroHPC participating state; Win2Win SRL applies as an SME (EU definition: < 250 FTE, ≤ €50M turnover — declared in the application).
- Playground: FIFO, **no competitive selection**, eligibility + technical assessment, access can be granted within **~2 working days**; allocations of 1–3 months; **one system per application**; guidance provided by the hosting AI Factory.
- Fast Lane: for users familiar with HPC, **up to 50 000 GPU-hours** over max 3 months, approval within ~4 working days, FIFO.
- Obligations to expect (per EuroHPC AI access documentation): a **final report** at the end of the allocation, publishable — fully aligned with our open ledger; failure to report can disqualify future proposals.
- **Platform note (verify at submission):** the call pages went through a platform transition during 2026; proposals were accepted via `access.eurohpc-ju.europa.eu` in the interim. Check the live EuroHPC JU AI Factories access-calls page for the current portal on the day of submission.

## 2. Two-stage strategy (amendment to D7 / REPRISE §7)

- **Stage 1 — Playground, applied EARLY (before N5).** Purpose: **environment validation**, which is exactly the assurance sought. Cheap to apply (short templated proposal, FIFO), worst case is a refusal costing nothing. Stage-1 deliverable: run the N1 control **and at least one ablation on the EuroHPC system itself**, producing ledger records with `compute.provider = "eurohpc"`, and validating the full operational chain (container, scheduler, offline data staging, checkpoint/resume).
- **Stage 2 — Fast Lane, gated on N5** as before — now with the strongest possible application: an ablation report whose `run_id`s already include runs executed on EuroHPC hardware ("we already run on your systems").

## 3. System choice (one system per application)

- **Prefer NVIDIA partitions** (A100/H100-class systems such as Leonardo, MareNostrum 5, MeluXina, Karolina, Vega — availability of partitions varies; pick from the live application form): zero-friction PyTorch.
- **LUMI (AMD MI250X / ROCm) is acceptable** — *because* ADR-002 already prohibits CUDA-only ops (pure-framework ops; custom kernels Triton-only and budgeted). That rule **is** the portability guarantee for this route: PyTorch/ROCm exposes the `torch.cuda` API, and Triton targets AMD. If LUMI is chosen, Stage 1 doubles as the ROCm smoke test.

## 4. Trainability checklist (what the environment imposes — anticipated now)

- **Slurm batch scheduling with wall-time limits** → the N1 trainer must implement **checkpoint/resume** from day one (added to N1's deliverable).
- **Containers are Apptainer/Singularity, not Docker** → N1 ships a container recipe (Dockerfile → `apptainer build`) as an artifact.
- **Compute nodes are typically offline** → datasets, tokenizer and packages staged in advance (`HF_HUB_OFFLINE=1`, pre-downloaded FineWeb-Edu slice, no runtime pip).
- **Storage quotas / scratch** → checkpoint rotation; ledger records are tiny by design.
- **Use the hosting AI Factory's included guidance** — onboarding support is part of the access mode.

## 5. Sizing honesty (orders of magnitude; assume 30–40% MFU; unknowns stay unknowns)

| Run | ≈ FLOPs (6·N·D) | ≈ GPU-hours |
|---|---|---|
| 60M × 1.2B tokens | ~4×10^17 | ~1–2 A100-h |
| 125M × 2.5B tokens | ~2×10^18 | ~5–8 A100-h |
| 1B × 20B tokens | ~1.2×10^20 | ~300–400 A100-h (or ~80–120 H100-h) |
| 3B × 60B tokens | ~1.1×10^21 | ~3 000 A100-h |

Conclusion: the **entire ablation ladder fits inside a Playground allocation**; the validated 1–3B run fits comfortably inside Fast Lane's 50 000 GPU-hour ceiling. These estimates go into the requested-resources field of the applications, labeled as estimates.

## 6. Application checklist

1. Verify current portal on the EuroHPC JU AI Factories access-calls page (platform transition — see §1).
2. Playground application: templated short proposal — project relevance (open-source LLM research, ablation protocol, public ledger), requested system + partition, GPU-hours from §5, duration (start with 2 months), SME declaration for Win2Win SRL.
3. On grant: execute Stage-1 deliverable; append records to the ledger; write the final report from the ledger (it already is the report).
4. Fast Lane application: after N5, citing `run_id`s including the `provider = "eurohpc"` ones.
