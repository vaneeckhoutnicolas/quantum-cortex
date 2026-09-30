> **Mirror.** This page is a copy of `docs/decisions/019-quantum-cortex-architecture.md` in the `quantum-meridian` hub repository, which is private at the v1 edition of quantum-cortex. Copied on 2026-09-30 at the hub's state that carries the D17 amendment of that day; the hub remains the authority and this copy is refreshed at each tag of this repository. Licence of this page: CC BY 4.0 (`LICENSE-DOCS.md`), by the same author.

# quantum-cortex — a trained kernel model, not an orchestration shell

**Status:** decided (2026-08-07, founding day; recorded here at E1, 2026-09-05) — creates the
sibling repository [`quantum-cortex`](https://github.com/vaneeckhoutnicolas/quantum-cortex)
(Apache-2.0, open); no change to the existing repositories. Deciders: Nicolas Van Eeckhout.
Reference page: [quantum-cortex — the pure-LLM member](../reference/quantum-cortex.md). The
ablation ladder has not started: the cortex ledger is empty (its `metrics/` and `docs/roadmap.md`
carry the status).

> **Numbering note, as written on 2026-08-07 (honest data):** "018 assumes ADR 017 stays reserved
> for the compression-order decision. If 017 was never committed, renumber this file — do not leave
> a silent gap."
>
> **Amendment 2026-09-05:** the hub keeps its records in `docs/decisions/` (not `docs/adr/`), and both
> 017 (reduction ordering) and 018 (predictive frontier) exist — so this record, never committed
> before E1, is **019**. Every cortex-side reference to "ADR 018" was rewritten to *hub decision 019*
> in the same delivery; the cortex's own series stays local, in its `docs/adr/`.

## Context

Quantum Meridian's commercial model separates a free agent layer from a proprietary control plane. The ecosystem needs an open-source traction engine that is (a) genuinely novel — not another orchestration shell over third-party models — and (b) QM-native out of the box, so that adopting the open model pulls users toward the QM ecosystem.

Budget reality, stated plainly: no individual budget competes with frontier labs on general capability. That game is closed. The winnable, measurable game is **capability per parameter at equal scale**: at identical parameter count, token budget, data, tokenizer, and compute, demonstrate capabilities that a vanilla transformer baseline does not have.

Architectural thesis (founding discussion, 2026-08): a brain is not one uniform network. It has specialized zones, an associative memory with attractor dynamics ("magnetism"), exogenous perturbation that forces recontextualization ("volatility", gray states), and hard invariants. These concepts must live **in the weights and in the decode loop**, not in orchestration code around the model.

## Decision

Build `quantum-cortex`: a small language model (ablation ladder 30M → 60M → 125M parameters; 1–3B only if validated) whose architecture implements six components.

### C1. Zones — semantic mixture-of-experts
MoE where expert routing is trained toward semantic specialization (interpretable domains), not pure load balancing. The router is part of the model, learned with it.

### C2. Magnetism — associative memory layer
A modern-Hopfield-style associative layer: retrieval by energy descent toward attractors. Two tiers:
- (a) **in-weights** associative layer — the ablatable research object;
- (b) **persistent tier** backed by the QM semantic journal (agent-side, encrypted at rest per invariant I13), through journal-compatible read/write events.

A positioning section against the test-time-memory literature (e.g. Google Titans) is mandatory in the whitepaper before any public claim.

### C3. Volatility — oracle channel
A dedicated input stream of exogenous events, injected mid-generation, that modulates routing temperature and expert gating. Intended effect: distributional bifurcation ("gray states") instead of binary branching. The oracle is an interface; sources are pluggable (news feed, sensor, controlled randomness, QM events).

### C4. Invariants — typed decode contracts (Kern legacy)
Typed I/O contracts enforced at decode time via constrained decoding. An output violating its declared contract is rejected at the source, not filtered afterwards. Every module manifest declares a determinism class: `deterministic | heuristic | stochastic`.

### C5. Grafted zones — external models through the same router
External LLMs attach through the **same routing interface** as internal experts, via an adapter contract (OpenAI-compatible transport by default). Standalone = the small cortex alone; plugged = the same brain with larger zones. One mechanism, two regimes.

`quantum-cortex` re-implements **neither** MCP federation **nor** the control plane: it consumes Quantum Meridian for both.

### C6. QM-native surface
By default the model server exposes:
- an OpenAI-compatible endpoint;
- native telemetry matching QM proposal-scoring inputs — cache affinity, cost, latency, quota (weights 0.4 / 0.25 / 0.2 / 0.15);
- journal-compatible events for C2's persistent tier.

Trust boundary (corrected formulation): only the inference path — prompts, completions, provider keys — never crosses the proxy.

*Amendment 2026-09-30 (the cortex's security section, whitepaper §4.9).* The division is restated from the cortex's side: the proxy is the boundary of the inference path, the agent is the actor that holds the tools and the journal, and the cortex's persistent memory sits on the agent's side of that boundary. The organ's guarantees are stated in two registers, by design (separation by domain, content authenticated per line and per payload, the sequence not yet, fail closed, the rule of silence) and by measure (poisoning reread from rows 37 and 39, extraction measured in row 40: nothing at the organ level by construction, no claim at the system level on three seeds and 600 cross domain queries). What the cortex does not do and this hub must, named in the cortex's §1.6: external content as data and never as an instruction, output never an action until an actor with the least privilege performs it in a sandbox, an action journal that cannot be altered after the fact, the custody of keys on the deployment side.

## Ablation protocol (normative)

- **Control:** vanilla transformer at equal parameters, tokens, data, tokenizer, and compute. Pinned seed policy; a config hash is logged for every run.
- **One component per ablation.** Combinations only after individual wins.
- **Declared capability benchmarks:**
  - C1 — routing specialization metrics (expert/domain mutual information);
  - C2 — multi-query associative recall (MQAR-style) at long range;
  - C3 — oracle-shift adaptation (mid-sequence context change, recovery speed and quality);
  - C4 — invariant compliance rate on typed outputs;
  - plus perplexity and one small standard eval suite for regression tracking.
- **Advancement rule:** a component ships only if it beats control on ≥ 1 declared capability with perplexity degradation ≤ 2%. The tolerance is revisable, but must be stated **before** the run.
- **Honest data:** all results published, including losses. Errors are noted, never erased.
- **Scale ladder:** 30M → 60M → 125M on free-tier compute; 1–3B only for the validated combination, on granted or rented compute.

## Compute plan (zero-budget start)

1. **Free tier:** Kaggle (~30 GPU-hours/week, 2×T4) as the ablation workhorse; Colab free for smoke tests. A 30–60M run at ~1B tokens is a few hours on Kaggle — the full ablation ladder is feasible at €0.
2. **Applications to prepare:** EuroHPC AI Factories access — free of charge for EU SMEs/startups (Playground: FIFO, access within ~2 working days; Fast Lane: up to 50 000 GPU-hours), applicant Win2Win SRL; TPU Research Cloud — free TPUs against published results, aligned with the open plan.
3. **Community training** ("let others train it") follows credibility, not the reverse: it becomes realistic only once a reproducible ablation table exists.
4. **Hardware purchase is a gate, not a calendar item:** decided on ablation results plus utilization math, never before.

## Consequences

`quantum-cortex` is **not**: a frontier competitor; an orchestration shell; an MCP federation; a control plane.

Risks, accepted:
- each component may lose to its control — that is the purpose of the protocol;
- the field moves fast (Titans, MoE literature): the whitepaper positioning section is mandatory before any public claim;
- single-developer bus factor: mitigated by REPRISE documents in the repository.

## Open questions

Tokenizer choice; data mix (open corpora, FineWeb-Edu class); framework (PyTorch by default, JAX if TRC is granted); exact journal event schema for C2's persistent tier; interplay with QM feature gates (none expected — the cortex is fully open).
