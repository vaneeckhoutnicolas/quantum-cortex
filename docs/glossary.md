# Glossary — quantum-cortex coined terms

- **Zone (C1)** — a semantically specialized expert in the mixture-of-experts; routing is trained toward interpretable domains, not pure load balancing.
- **Magnetism (C2)** — associative retrieval by energy descent toward attractors (modern-Hopfield-style); links emerge by semantic proximity, not logical implication.
- **Attractor** — a stable point in the associative layer's energy landscape toward which related concepts converge.
- **Persistent tier (C2b)** — the journal-backed long-term memory of the cortex; agent-side, encrypted at rest (QM invariant I13).
- **Oracle channel (C3)** — a dedicated input stream of exogenous events injected mid-generation, modulating routing temperature and gating.
- **Gray state** — the intended effect of the oracle channel: a distributional bifurcation of the decision path instead of a binary 0/1 branch.
- **Invariant contract (C4)** — a typed I/O contract enforced at decode time; violating outputs are rejected at the source (Kern/Skald legacy).
- **Determinism class** — declared behavior of a module: `deterministic | heuristic | stochastic`.
- **Grafted zone (C5)** — an external LLM attached through the same router interface as internal experts. Standalone = small cortex; plugged = same brain, larger zones.
- **QM-native surface (C6)** — the default exposure: OpenAI-compatible endpoint + telemetry matching QM proposal scoring (cache affinity / cost / latency / quota) + journal-compatible events.
- **Control run** — vanilla transformer at equal params / tokens / data / tokenizer / compute; the only legitimate baseline of an ablation.
- **Ablation ladder** — the scale sequence 30M → 60M → 125M (→ 1–3B only for a validated combination).
- **Run ledger** — `metrics/runs.jsonl`, append-only, git-versioned; a run without its committed record does not exist (ADR-001).
- **LATEST** — `metrics/LATEST.md`, regenerated after every run; the always-current state of the numbers.
- **Advancement rule** — a component ships only if it beats control on ≥ 1 declared capability with perplexity degradation ≤ 2%, tolerance stated before the run (hub decision 019).
