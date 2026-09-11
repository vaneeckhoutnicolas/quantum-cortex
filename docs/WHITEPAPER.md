# WHITEPAPER — "Continuity as a first-class capability" — living skeleton

**Status:** deliverable specified 2026-09-07 (D19). This file is the paper's **living skeleton**: it
grows by consolidation (RES-19) — a section is filled only with claims that have passed the
**admission gate** below. Everything else lives in *Open questions*, explicitly labelled. The
skeleton is versioned like everything else: dated additions, prior text kept.

**Why a paper, and why now:** D19 says the method is as much the contribution as the model — the
demonstration that an individual with an AI alter-ego can build something significant *by rigour,
not force*. A paper is how that demonstration reaches the world. Specifying it early keeps every
chantier honest about what it must produce to be *publishable*, not just *interesting*.

## The admission gate (a claim enters a results section only if ALL hold)

1. **Statistical** — measured on ≥ 3 seeds, the gap significant at 95% (paired test, `cortex_eval/multiseed.py`). A single seed is an anecdote, not a claim.
2. **Anchored** — situated against the external references (`cortex_eval/external_baseline.py`, the recurrent ladder) at equal size, tiers, steps.
3. **Real, not toy** — demonstrated on the real task (real MQAR spans / real language-model runs), not only a synthetic surrogate. (Level 1b showed a 93% toy result collapsing to 25% on real spans — this rule exists because of that.)
4. **Reproducible** — a committed artefact (`metrics/`), a config hash, a run_id; the code that produced it is in the repo.
5. **Honestly bounded** — reserves stated (seeds, scale, single capability); losses and retrogradations published alongside.
6. **Complete** (the completeness test) — the result is regressed on the declared context (data hash, regime, scale, held-out composition); unexplained inter-seed or inter-holdout variance is treated as the signature of an *omitted factor*, not noise, and blocks the claim until the factor is found or declared as a limit. (Transposed from the author's parallel physics program [Van Eeckhout, 2026], where thermodynamic "volatility" vanished exactly under complete accounting; first applied to our own router result, where it exposed an ill-posed ratio metric and sharpened a retrogradation.)

A finding that fails any rule is an **open question**, not a result — and open questions are a legitimate, valued part of the paper (they are what a solo founder can *pose* to the field).

## Skeleton

### 0. Abstract
*(written last)*

### 1. Thesis — continuity as a capability

**1.1 Scope.** This paper does not compete on general language-modeling capability. At the scale accessible to a single researcher (tens to hundreds of millions of parameters, commodity or shared-HPC compute), that contest is settled by resources rather than ideas. We instead isolate one capability that current evaluation practice does not measure, and ask whether it can be specified, measured, and improved reproducibly at small scale.

**1.2 The capability.** We call it *continuity*: the property of a model that (i) retains episodic content across the boundary of a session or context window, (ii) revises its predictions when an exogenous signal indicates that the world has changed, and (iii) does not regenerate content it already holds, referencing it instead. We deliberately treat continuity as a **triad** rather than a scalar. The three components are dissociable — a system may persist without revising, or revise without persisting — and a blended score would hide exactly the trade-offs of interest. Each component receives its own protocol:

- *Episodic persistence* — an H.M.-style dissociation protocol, after the neuropsychological case in which episodic memory was lost while skills were retained [Scoville & Milner, 1957]. Synthetic facts introduced in a session are probed after the session boundary, with a matched control that must show no persistence; leakage controls use synthetic entities absent from any pretraining corpus.
- *Adaptive revision* — an oracle-shift protocol: a first-class input channel carries exogenous events, and the model's predictive distribution is measured before and after a controlled shift.
- *Non-regeneration* — a reference-ratio protocol: the fraction of output tokens that cite existing spans rather than re-emit them.

The protocols are specified before any model exists that scores well on them, and are published as an open, versioned suite. We consider defining the measurement to be a contribution independent of the model.

**1.3 Evaluation stance.** All results are read as a Pareto frontier over declared axes — the three continuity components plus two efficiency denominators (capability per parameter, capability per bit) — never as a single scalar. A perplexity tolerance of 2% relative to the control is a *guard* that a candidate must clear to be considered, not an axis on which it can win. No composite score is computed. Optimizations across components are not assumed additive; joint effects are recorded next to isolated effects.

**1.4 Positioning: who continuity is for.** The intended contribution is threefold: a specification of continuity as a measurable capability; an architecture in which memory components are arbitrated on a guaranteed non regression floor (§3); and a method by which a single researcher, working with an AI system as a design and engineering collaborator, can produce reproducible results under a strict evidentiary discipline (§2). We make no claim to state of the art general capability. The model is intended for settings in which continuity is the decisive property, and we name five such settings, each tied to the component of the triad it requires and to the organ that measures it. Where continuity is not decisive, frontier models are the appropriate tool.

- *The long running project companion.* A practitioner who works for months on one matter: a lawyer on an intellectual property case decides in January to set aside an argument on the strength of a ruling; in a fresh session in May, asked for a hearing strategy, the model recalls that decision and its ground, cites the ruling already analysed rather than summarising it again, and does not resurface the discarded argument. Component: episodic persistence and non regeneration. Organ: the persistent tier C2b (§3) and typed span references. This is the setting the H.M. protocol measures, and the one the Quantum Meridian orchestration layer already serves around third party models.
- *The assistant whose world changes.* A compliance officer following European banking regulation: a directive enters into force on a Tuesday; the oracle channel receives the event, and from Wednesday the model's answers on reporting obligations change, visibly and datably, without retraining. A model frozen at its training cutoff would answer as the world stood then. Component: adaptive revision. Organ: the oracle channel C3 (not yet built).
- *Private, local, encrypted memory.* A general practitioner keeping patient follow up notes on her own machine, sending no record to any API: the journal is encrypted per patient, the model never sees another patient's record, and recall works across consultations three months apart. A model with genuine associative memory that fits on a local device occupies a position that cloud economy laboratories cannot take. Component: episodic persistence under a trust boundary. Requires scale and ternary weights (§5).
- *The local model of an agent.* A Quantum Meridian agent managing a team's infrastructure: for everything that touches the project's history (why a service was migrated, what was decided after an incident) it queries the cortex, which carries that memory in its journal; for writing new code it calls a frontier model; the cortex signals, through its decode contracts, when a question leaves its domain. Component: persistence plus contracted decoding. This is the non regression router of §3 applied at the scale of models rather than of memories.
- *The measurement itself, for other laboratories.* A research group that has published a long memory model and wishes to show that it retains episodes rather than reconstructs them downloads the open suite (the H.M. protocol, oracle shift, the reference ratio), runs it with its frozen thresholds on their own model, and publishes a result comparable to ours. Here the product is not the model but the measurement; whoever defines the benchmark frames the question.

### 2. Method — an evidentiary discipline for single-researcher work

The project was carried out by one researcher in continuous collaboration with a large language model acting as a design, implementation and review partner. We describe the discipline that governed this collaboration, because we consider it to be the primary reason the results in §4 can be trusted, and because it is reproducible independently of the architecture.

**2.1 Measure-first.** No architectural component enters the model on the strength of an argument. Every component is introduced behind a configuration flag that defaults to off, so that the control configuration is bit-identical to the baseline (a test asserts this). A component advances only if an ablation, at equal parameters, data, seed and schedule, extends the Pareto frontier of §1.3. Components that do not clear the bar are recorded as losses.

**2.2 The ledger.** A training run that has not been committed to an append-only ledger does not exist. Each record follows a versioned JSON schema (`run-v1`) validated in continuous integration, and carries a configuration hash and a content hash of the data mix (`mix_hash`). Two runs are comparable only if their `mix_hash` coincide; "at equal data" is thereby a machine-checkable property rather than a claim.

**2.3 Data as a first-class object.** The data mix is a directed acyclic graph of typed operators (sources, transforms, combinators), evaluated lazily and hashed per node. Identical sources are folded by content signature before sampling, so that a mix of a source with itself is a multiset-equivalent of the source — a property enforced by an invariant test. Decontamination is a first-class operator: evaluation *n*-grams may not appear in training data, and a quantified contamination report is archived per mix.

**2.4 Tests from the first day.** Structural invariants were written before, not after, the components they protect. Five defects were caught by these tests prior to any ablation: two in the sampling semantics of the data mixer, one in which a global weight initializer silently overwrote the zero-initialization that guarantees a memory layer starts as a no-op, one ordering error in a routing-hysteresis counter, and one numerical divergence of a recurrent memory under unnormalized keys. Each would have invalidated an ablation without producing a visible failure. We report them because the discipline that catches such defects is part of the method, not an incident.

**2.5 The circuit breaker.** Training runs are subject to an aggressive early-abort rule with a bounded refine-and-retry loop: NaN or Inf losses abort immediately; divergence beyond a declared multiple of the control's loss over a window aborts; each abort may trigger at most two pre-declared refinements (tighter gradient clipping, then a reduced learning rate) before the configuration is retired. Thresholds are fixed before the run and never adjusted post hoc. Every abort is a ledger record. In the ablation of §4 the rule fired twelve times on a divergent configuration, cleanly, before the defect of §2.4 was corrected.

**2.6 Admission of claims.** A finding enters a results section of this paper only if it is (i) measured on at least three seeds with the relevant gap significant at the 95% level under a paired test, (ii) situated against external references at equal size, (iii) demonstrated on the real task rather than a synthetic surrogate, (iv) reproducible from a committed artefact, and (v) stated with its reserves. Findings that fail any criterion are reported as open questions. This rule was adopted after a result obtained on a synthetic routing task (a learned router capturing ~93% of an oracle ceiling) was found to shrink to ~25% on the real task with the same code; we regard the retrogradation as a result of the method rather than a failure of the model.

**2.7 The same discipline, twice.** The author maintains a parallel research program in classical gravitation (mirror-constrained black-hole thermodynamics [Van Eeckhout, 2026], six preprints in preparation) with no subject-matter relation to language models. That program independently converged on the same apparatus: a theorem register, per-claim novelty pre-gates against the literature, a pre-submission checklist, dated follow-up notes, an archive kept for provenance, forty-digit cross-validation against published closed forms, and the explicit recording of falsified hypotheses. That two unrelated fields, worked by one researcher with a language model as collaborator, produced the same evidentiary discipline suggests the discipline is a property of the method rather than of either subject. One result crossed over: the physics program's *complete-ledger* observation — that apparent volatility of an invariant is the signature of an omitted variable and vanishes under complete accounting — became rule 6 of the admission gate above, and its first application to our own router result replaced an ill-posed ratio metric with an absolute one.

**2.8 Decisions and errors.** Every design decision is a dated record; amendments are appended, never substituted, so that the record of what was believed and when is preserved. Nineteen such decisions and twenty dated revisions of the idea register underlie this paper. We note that the collaboration with the language model was itself subject to this discipline: proposals from either party were treated as hypotheses to be tested, not as conclusions.

**2.9 The collaboration as design thinking.** The working method between the author and the language model was not improvised for this project. It follows a design thinking discipline the author formalised in 2018 for his consultancy (Cogarius, *Design Thinking Snapshot*, June 2018): understand before defining the problem, define the problem before designing the solution; observe, contextualise, structure, empower, co create; alternate divergence and convergence in loops (co understand, then co design); and hold desirability, viability and feasibility together rather than optimising one. Each of these maps onto a rule of this repository. The idea register precedes any code, and the continuity protocols were frozen before a model existed that could pass them (understand before solving). The landscape reading of the recurrent family, then the ladder that situates our components inside it, is observation followed by contextualisation. The decision records and the domain map are structure. The admission gate and the circuit breaker are the empowering tools. The memory router of §3 emerged from three dictated messages, refined in dialogue: co creation. And the Pareto reading of §1.3 is the three way tension of desirability, viability and feasibility applied to a model.

The two participants are wired differently, and the collaboration was designed to make that difference productive rather than to erase it. The author supplies the upstream structure: which variables exist, which invariants must hold before code is written, what a result must consolidate to count. The language model supplies rapid divergence and execution: variants, drafts, implementations, the literature at hand, and a first pass at every document. The structure frames the generation; the generation feeds the structure; the ledger arbitrates. When the model over claimed, the author's structural checks caught it (five retrogradations in one week, §2.6); when the author's intuition needed a form, the model gave it one in minutes. Neither side alone would have produced a repository in which the absence of a result is itself a recorded result. We report this not as a claim about either participant but as a reproducible pattern: a researcher with a strong upstream discipline and a generative partner with a broad reach, bound by a shared evidentiary gate, can do research that is faster than the researcher alone and more honest than the model alone.

### 3. Architecture — a memory router on a non-regression floor

**3.1 Base model.** The base model is a byte-level decoder-only transformer of the GPT-2 lineage, trained from scratch, with a reserved block of token identifiers for exogenous (oracle) events. We keep the base deliberately conventional so that every departure from it is attributable to a single ablated component. The design draws on a mapping from brain systems to model components — associative memory, a thalamic-style gating channel, basal-ganglia-style action vetoing, a neuromodulatory bus — recorded as a validated feature map; we use the mapping as a source of hypotheses, not as a claim of biological fidelity.

**3.2 Associative memory (C2).** Two memory layers are implemented as an additional residual sub-block in each transformer block, each zero-initialized at the output projection so that the model begins as the control and learns the memory: a modern-Hopfield layer (a learned bank of key/value patterns read by softmax attention — a static, content-addressable memory) [Ramsauer et al., 2020], and a gated delta-rule layer (a per-head recurrent state updated by an error-correcting rank-one write with a per-channel forget gate, keys L2-normalized) [Schlag et al., 2021; Yang et al., 2024]. The two belong to the same fast-weight family and differ in whether the memory is static or written online.

**3.3 The router.** The central architectural proposal is that C2 should not be one memory, nor a fixed choice between memories, but a **router** over paths — the control (no memory), the Hopfield layer, the delta layer — subject to three constraints:

1. *A guaranteed floor.* The control path is always available. A memory path is selected on a span only where its predicted score exceeds the control's; if no memory path does, the control is used. Consequently the router cannot degrade the model relative to the control: at worst it equals it. Falling to the control on a span is recorded as a result, not a failure.
2. *Routing regime follows consolidation.* Where the router has learned, over repeated passes, which path wins for a span type — confidently and stably — it routes *hard* (one path, selected before computation, at minimal cost). Where it has not, it routes *soft* (paths computed and mixed), which is more expensive but generates the signal from which hardening is later decided. The system migrates from soft to hard as it learns; average cost falls as routing precision rises. Hardening is deliberately conservative: an uncertain decision is never hardened.
3. *Fallback with hysteresis.* When a selected path degrades or diverges, the breaker of §2.5 re-routes to the best remaining path above the control, or to the control, and re-softens the decision. A declared cooldown prevents oscillation between paths.

The router is versioned behind a stable interface: a v1 oracle (which reads the true per-path scores and therefore computes the ceiling any learned router may reach) and a v2 learned gate (which predicts per-path scores from the span context alone). State is serialized with a version tag and dispatched by version, so later versions extend the interface without breaking earlier ones.

**3.4 One mechanism at three scales.** We observe that the same consolidation law — *what has been computed and remains valid is not recomputed; what changes recomposes only its delta* — governs the memory hierarchy inside the model, the soft-to-hard migration of the router, and the accumulation of tested components and dated decisions in the project itself. We do not claim this as a theorem; we record it as the organizing principle that made the three levels mutually consistent, and as the reason efficiency appears in §4 as a consequence rather than a trade-off.

### 4. Results — read as a domain map, not a ranking

*(Only gate-passing claims enter as results; each with artefact, seeds, CI. Cells are filled from `metrics/mqar/domain-map.json` as evidence accrues.)*

**4.1 Baseline.** Control `be1fa8139f59`: byte-level GPT-2-style, 25.9M parameters, val_ppl 2.601 on 500M FineWeb-Edu byte-tokens. Reproducible; anchored; a baseline, not a comparison (rule 1 does not apply).

**4.2 The reading rule (why a map).** The first ablation (12 tiers, one seed) ranked the memories by whole-curve AUC: Hopfield +14%, delta +2% over the control. The confirmation run, restricted to the hard/short tiers where the paths actually separate, ranked them the other way: delta first, Hopfield at the control. Under rule 6 this is not a contradiction; it is the omitted factor Σ — *which tiers are held*. The honest object is therefore a **domain map**: for each tier, the winning path and its margin, with the evidence basis. Of the 12 tiers, only 6 are separable (margin ≥ 0.01); on the other 6 no path can be distinguished, and a whole-curve ranking silently averages over them.

**4.3 The domain map** *(single-seed basis unless marked; multi-seed rows carry a 95% paired test).*

| regime | separable tiers | winner | evidence |
|:--|:--|:--|:--|
| easy / dense recall (kv 4, seq 128–512) | 3 | **Hopfield** (margins 0.016–0.051) | 12-tier, 1 seed — *pending multi-seed* |
| hard / short (kv 8–16, seq 128) | 1 + confirmation tiers | **delta** (0.033 at kv 8/128; leads on both completed confirmation seeds) | 12-tier 1 seed + confirmation *(3 seeds, in progress)* |
| long sequences (kv 8–16, seq 512) | 1 | **control** — memory hurts (0.188 vs 0.150) | 12-tier, 1 seed — *pending multi-seed* |
| saturated (kv 32, seq 256–512) | 0 | none — noise-level (all ≈ 0.01) | not a result |

**4.4 What passes the gate — the confirmation run (3 seeds, 1000 steps, tiers kv 16 and 8 at seq 128; verified by its retained log and its committed Kaggle artefact).**

| comparison | mean gap | paired t (df = 2) | t_crit 95% | verdict |
|:--|--:|--:|--:|:--|
| delta vs control | +0.0107 (+13%) | 1.76 | 4.30 | does not pass |
| Hopfield vs control | −0.0031 (−4%) | −0.54 | 4.30 | does not pass |
| Hopfield-hybrid vs best pure recurrent (L4) | **+0.048** | **3.78** | 4.30 | does not pass |
| delta-hybrid vs best pure recurrent (L4) | **+0.062** | **4.09** | 4.30 | does not pass |

Nothing enters as a claim. The two rows that matter are the last two: against the best rung of the recurrent ladder (the gated delta rule, L4), the hybrids reach three to four times its accuracy with t statistics of 3.78 and 4.09, held only because three seeds set the critical value at 4.30. This is a design-limited hold, not a variance-limited one. The next run is the ladder at eight seeds (t_crit 2.37 at df 7); whether the effect clears that bar is for the run to decide. One candidate finding is withdrawn by this run: "the delta rule regresses when pure" (L4 − L3 = +0.008, t = 0.90) was a single-seed artefact of an earlier mini-run.

*A second run at five seeds was reported during preparation of this paper and is withdrawn: its numbers were read from a log that was not retained as a file and could not be located afterwards. Under §2.6 an unverifiable source yields no result; the withdrawn figures are kept in the repository's artefact history so the error is visible.*

**4.5 Continuity, first measurement — the H.M. dissociation.** The episodic-persistence component of the triad (§1.2) has its first number. Against the C2b organ (§3; store, hippocampal write path, journal as router path), the H.M. protocol — 200 leakage-proof synthetic facts planted through the gated write path, probed in a fresh context with the journal on and off, a 50-fact negative control, thresholds frozen on 2026-08-08 — returns `hm_dissociation_pass = 1`, stable across four seeds: recall 1.000 with the journal, 0.125 without (the chance floor; threshold 0.175), a gap of 0.875 against the required 0.50, a journal negative-control rate of 0.000 (the store never returns an attribute for an entity it does not hold), and a skill delta of 0. *Cut the journal: episodic recall collapses; skills hold.* The founding prediction — weights carry the semantic and procedural, the journal carries the episodic — is thereby a measurement rather than a design premise. **Scope (rule 5):** this is a measurement of the organ, with a hash-seeded cue encoder and a reference skill probe; it establishes that the journal stores, retrieves and refuses to confabulate, and that the two stores are separable. The language-model arm — real skill suites, a learned encoder, the model reading the journal in generation — is the v1 requirement (§5). **Method note (rule 5):** the first run was invalid on the negative control (0.12); the journal was clean (0.000) and the fault lay in the reference floor simulator and in a specification that read guessing as hallucination; the simulator was corrected and the specification amended, dated before the rerun, with δ, ε_S and the floor untouched (`docs/benchmarks/hm-protocol.md`, amendment of 2026-09-08).

**4.6 The result that the map itself constitutes.** No single memory dominates: Hopfield, delta and the control each own a regime. This is the empirical content behind the architecture of §3: a router on a control floor is not an elegance, it is what a per-domain result *requires*. The upper envelope of the map (best path per tier) exceeds the best single path by +6% and the control by +21% on the 12-tier basis — the ceiling a router may reach (§5 records how far the learned router currently is from it).

### 5. Open questions (valued, explicitly not results)
- **The learned router on real spans** — 93% of the oracle ceiling on a synthetic task, ~25% on real MQAR spans (12 tiers, under-sampled). Needs thousands of real spans (router wired into the LM). [Rev20]
- ~~Does the delta-rule pay only when hybridised?~~ **Withdrawn (2026-09-09):** the single-seed mini-run suggested L4 regresses when pure; at three seeds L4 − L3 = +0.008 (t = 0.90) — a seed artefact. What remains, and is held only by the quota: hybrids reach 3–4× the best pure recurrent (§4.4).
- **The continuity triad — two of three components still unmeasured.** Episodic persistence now has its first measurement (§4.5, against the organ). Adaptive revision (oracle-shift) needs C3, not yet built; non-regeneration (reference ratio) needs the typed span-reference decode of RES-8, not yet built. And the persistence measurement itself must be repeated with the language model wired to the journal (learned cue encoder, real skill suites) before it counts as a measurement of the *model* rather than of the organ.
- **Scale** — everything is 25M / MQAR-scale; a 100M+ run (EuroHPC) is required before any capability claim generalises.

### 6. Limitations & honest reserves
Single seeds where noted; one capability (associative recall) measured so far; synthetic-vs-real gap demonstrated on our own router; no external pre-trained comparison yet (a non-equalised Mamba/RWKV reference is a separate, license-checked job).

### 7. Reproducibility
`docs/GETTING-STARTED.md`; `pytest tests/` (the CI runs it); the ledger `metrics/runs.jsonl`; artefacts `metrics/mqar/`; every decision dated in `docs/adr/` and the hub decision log D1–D19.

## Gates for the paper itself
- **v0 (internal)** may be drafted once §4 holds ≥ 1 gate-passing comparison (the first candidate: the recurrent ladder at 8 seeds).
- **v1 (submittable)** requires the continuity triad measured on at least one organ (C2b or C3 built) **and** a 100M+ run — i.e. the capability the paper is named after must exist as a measurement, not a specification.
- Target: arXiv (cs.LG), then a workshop; open-results (D17): the paper, ledger and benchmarks are public; the recipe stays the workshop's.

## References (verified 2026-09-08 against arXiv / publisher pages; completed at v0)
- Scoville, W.B., Milner, B. (1957). Loss of recent memory after bilateral hippocampal lesions. *J. Neurol. Neurosurg. Psychiatry* 20(1):11–21. doi:10.1136/jnnp.20.1.11.
- Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Adler, T., et al. (2021). Hopfield Networks is All You Need. *ICLR 2021.* arXiv:2008.02217.
- Schlag, I., Irie, K., Schmidhuber, J. (2021). Linear Transformers Are Secretly Fast Weight Programmers. *ICML 2021.* arXiv:2102.11174.
- Yang, S., Wang, B., Shen, Y., Panda, R., Kim, Y. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. *ICML 2024.* arXiv:2312.06635. *(GLA — the canonical rung L0 of the recurrent ladder)*
- Yang, S., Wang, B., Zhang, Y., Shen, Y., Kim, Y. (2024). Parallelizing Linear Transformers with the Delta Rule over Sequence Length. *NeurIPS 2024.* arXiv:2406.06484. *(DeltaNet)*
- Yang, S., Kautz, J., Hatamizadeh, A. (2024). Gated Delta Networks: Improving Mamba2 with Delta Rule. arXiv:2412.06464. *(Gated DeltaNet — the family of our DeltaMemory; rung L4)*
- Kimi Team, Zhang, Y., Lin, Z., Yao, X., Hu, J., Meng, F., et al. (2025). Kimi Linear: An Expressive, Efficient Attention Architecture. arXiv:2510.26692. *(KDA extends Gated DeltaNet with channel-wise gating; rung L2)*
- Arora, S., Eyuboglu, S., Timalsina, A., Johnson, I., Poli, M., Zou, J., Rudra, A., Ré, C. (2024). Zoology: Measuring and Improving Recall in Efficient Language Models. *ICLR 2024.* arXiv:2312.04927. *(MQAR)*
- Gu, A., Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752. *(selective gate — rung L1; local conv — rung L3)*
- Ma, S., Wang, H., Ma, L., Wang, L., Wang, W., Huang, S., Dong, L., Wang, R., Xue, J., Wei, F. (2024). The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. arXiv:2402.17764. *(BitNet b1.58)*
- Penedo, G., Kydlíček, H., Ben Allal, L., Lozhkov, A., Mitchell, M., Raffel, C., von Werra, L., Wolf, T. (2024). The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. *NeurIPS 2024 Datasets & Benchmarks.* arXiv:2406.17557. *(FineWeb-Edu, ODC-By)*
- Van Eeckhout, N. (2026). Mirror-Constrained Black Hole Thermodynamics: A Two-Horizon Companion. Preprint draft with reproducible code, https://github.com/vaneeckhoutnicolas/mirror-bh-thermodynamics (ORCID 0000-0002-5256-3185). *Not yet submitted to arXiv; cited per the repository's CITATION.cff; to be replaced by the arXiv identifier at submission (at final pass).* *(the complete-ledger / context-beta observation transposed as admission-gate rule 6; §2.7)*

*Verification note (method §2.7 applied to the bibliography): every entry above was checked against its arXiv abstract page or publisher record before being kept. The first draft had one missing identifier (Kimi Linear), two missing venues (Ramsauer: ICLR 2021; Zoology: ICLR 2024), and two missing lineage papers (GLA; Gated DeltaNet) — corrected here. Language-model-generated citations are treated as hypotheses until verified.*
