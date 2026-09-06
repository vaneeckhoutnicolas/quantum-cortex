# ADR 005 — The data composition layer (NOW-7)

- **Status:** Accepted (Nicolas Van Eeckhout, 2026-09-06)
- **Deciders:** Nicolas Van Eeckhout
- **Context:** NOW-7 is the mandatory prerequisite before N2 (SEQUENCE). Before comparing *architectures*, the *data terrain* must be a declared, hashed, reproducible object — otherwise "at equal data" is a claim, not a fact. The N1 control (`be1fa8139f59`) trained on a single corpus; this ADR generalises data to a composition of sources.
- **Cites:** hub-019 (advancement rule), ADR-001 (metrics ledger), ADR-004/D16 (Pareto evaluation), the anti-plagiarism protocol (`docs/OSS-SURVEY`), and register entries NOW-7, RES-12, **RES-14** (data-portfolio valuation).

## Context — the two bottlenecks named

The founder identified that a composition layer can bottleneck **both** processing time **and** results, and that these are different problems requiring different answers:
- **Processing bottleneck** → solved by design (lazy streaming + node-hash memoisation). A performance concern; additive to fix.
- **Results bottleneck** ("does this mix produce a better model?") → *not* solved by any engine cleverness. It is an empirical question answered only by the ledger and the Pareto frontier (D16). The engine makes each mix **cheap to produce and compare**; it never *chooses* the winning mix. This ADR keeps that line sharp, exactly as D16 exists to prevent the speed/quality confusion.

## Decision 1 — A composition is a DAG of typed operators, not a weighted list

The mix is a **declarative directed acyclic graph** of composable, typed operators — not a `[{source, weight}]` list. This is what makes an arbitrary combinatorial space of variables expressible without changing the engine.

- **Sources** (`source`): a licensed corpus/shard, content-addressed, with a declared `license`. e.g. FineWeb-Edu (ODC-By), an OLMo shard (license verified at ingestion).
- **Unary transforms** (`filter`, `dedup`, `decontaminate`, `normalize`): actions on one stream.
- **Intra-source distribution** (landscape-driven, 2026-09-06): `repeat` (deliberate repetition, **bounded ≤2 epochs** — OLMo/Kimi report diminishing returns beyond) and `rephrase` (amplification by *synthesis* — Kimi K2's token-utility approach, superior to naive repetition; a teacher-LLM rewrites in varied styles under **RES-12** provenance, Kimi's factuality caveat kept). `upsample` (quality-aware, OLMo 3). Distinct from `mix`: **mixing sets the distribution *across* sources; upsample/repeat/rephrase set it *within* a source** (OLMo 3 distinction, credited).
- **N-ary combinators** (`mix`, `concat`, `interleave`, `sample`): actions that fuse streams. Future: `schedule`/`curriculum` — a mix that changes over training (OLMo/Kimi late-stage annealing).
- **Sink**: the final byte-stream, hashed.

A one-variable run = a DAG with one source + decontaminate. An n-variable run = the *same engine*, a richer DAG. Complexity lives in the manifest (data), never in the engine (code). This mirrors RES-8's typed, composable decode contracts — architectural coherence, not coincidence.

## Decision 2 — Every node is content-addressed; the manifest is the recipe

- Each node carries a **deterministic hash** = f(operator, params, input-hashes). The graph's sink hash is the **`mix_hash`**.
- **Two runs are comparable iff they share `mix_hash`.** "At equal data" becomes machine-checkable, not declarative — the foundation of every honest ablation and every point on the Pareto frontier.
- Two DAGs differing by exactly one operator are automatically comparable on that single difference: the composition layer *natively produces* D16's ablation axes. **Composition provenance (D16 corollary 3) falls out for free: the DAG is the provenance.**
- The manifest lives at `data/mixes/<name>.json` — committed, human-readable, reusable. Heavy slices go under DVC (W2); the manifest stays in git. The manifest *is* the reproducible data recipe.

## Decision 3 — Lazy, streaming execution; memoisation specified, built when exercised

- **Lazy + streaming:** each operator is a pull-based iterator; the sink pulls bytes on demand. **No intermediate materialisation** — same philosophy as the K3 streaming extraction (K6) and RES-6's cache-obliviousness. The processing bottleneck is removed by construction.
- **Node memoisation (content-addressed cache):** a node computed once is never recomputed; DAGs sharing a subtree reuse its hash — the modern build-system pattern (Bazel/Nix-class), credited, written from scratch. **Specified now, implemented when a *second* mix actually shares a subtree** (today there is one mix — building the cache now would be a speculative build system the ledger does not yet exercise; measure-first).

## Decision 4 — Decontamination as a first-class operator, with a Bloom-filter algebra

Decontamination generalises the H.M. protocol's leakage controls (the `Vorel-3f2a` anti-leak entities + negative control) to *all* evaluation sets (MQAR, H.M., future benchmarks): eval-set n-grams must not appear in training data; a **quantified contamination report** is produced and archived, and a mix that fails decontamination is not comparable.

Mechanism, honestly scoped — the founder's bit-vector insight is **literal**, not metaphor, at the set-signature level. Represent each source's n-gram set as a **Bloom filter** (a bit vector); then set operations *are* bitwise operations:
- **OR** (`|`) = union of sources (the mix)
- **AND** (`&`) = cross-source intersection → instant contamination detection (`bloom_train & bloom_eval ≠ 0`)
- **XOR / AND-NOT** = difference → "in train, not in eval" (contamination subtraction)

The decontamination operator is thus a Bloom-filter algebra. **Specified now; the large-scale Bloom cross-check is implemented when there are *n* sizeable sources to cross** — W1's single-source decontamination can be exact and simple.

## Decision 5 — The mix weight is a pluggable resolver (the RES-14 seam)

`mix` consumes a **resolved weight**; *how* it resolves is behind a clean seam. **Today: a declared constant.** **Later (RES-14, infrastructure-gated): the output of a data-β estimator** measured from the Pareto frontier. This ADR implements only the constant resolver and the seam — never the estimator (a data-β is a *result read from the frontier*, not an input declared before it; implementing it now would invert measure-first). Weight semantics (Decision 6) are chosen so the seam accepts a β unchanged later.

## Decision 6 — Weight semantics: signature-aware multiset redistribution

A `mix` weight means **proportional contribution of a source's tokens to a bounded output budget** (default = weighted mean of source lengths — mixing *redistributes*, it does not inflate corpus size). The founder's plumbing test (`mix([S,S]) ≡ S`) exposed that naive without-replacement block sampling over overlapping sources front-loads the shared prefix — so the honest semantics is **signature-aware**: identical sources are **folded by content hash** before sampling (the founder's bit-vector insight — the signature *identifies* redundancy, so overlapping halves are never re-sampled), and each real source is then covered **uniformly** (even stride). Result: `mix([S,S], budget=|S|)` is a multiset-equivalent of S because the duplicate is *recognised, not re-drawn*. Deliberate repetition is the separate, bounded `repeat` operator. This folding is RES-16 (content-addressed consolidation) inside the combinator. The semantics accepts a future data-β resolver (Decision 5, RES-14) unchanged.

*Design note (resolved 2026-09-06):* three semantics were weighed — strict positional equality (forbids legitimate repetition), naive weighted concatenation (inflates to 2|S|), signature-aware multiset redistribution (chosen: makes the duplication invariant true *and* keeps repetition explicit/bounded — validated against OLMo mixing≠upsampling and Kimi rephrase>repeat).

## Decision 7 — The invariant harness (built from the start — the founder's plumbing test)

The engine's interface and its invariants are **structural** (getting them wrong later means painful refactors and invalidated ablations), so — unlike the performance features — they are built **d'emblée**, with tests in `tests/` (kern-style) enforced by CI. The degenerate cases are the best plumbing test that exists:

- **Identity:** `mix([S])` ≡ `S`.
- **Duplication idempotence (the founder's "run the same thing twice" test):** `mix([S, S], equal weights)` ≡ `S` at both byte-stream and `mix_hash` level. This also *forces* the Decision-6 weight semantics to be explicit and verified.
- **Commutativity:** `mix([A, B])` ≡ `mix([B, A])` at equal seed.
- **Determinism:** same DAG + same seed → same `mix_hash`, twice running.
- **Decontamination:** a planted trap-token in training is detected by the operator.

These five invariants are the "obvious errors" caught up front. The performance features (Decisions 3–4's memoisation and large-scale Bloom) are additive and deferred; the interface and invariants are not.

## Decision 8 — Legality: our own filon, built from scratch, over licensed sources, on credited recipes

Per the anti-plagiarism protocol, three distinct things are kept distinct:
- **The composition engine** is written **from scratch** — our filon; zero borrowed lines.
- **The data** (FineWeb-Edu, OLMo shards) are **licensed corpora**: using them is licensed use, not plagiarism, *provided* each source declares its `license` in the manifest and attribution is honoured (FineWeb-Edu: ODC-By; OLMo: verify per shard at ingestion).
- **The recipes** (OLMo's proportions, its decontamination method) are **ideas/methods** — unprotectable, reusable, credited, re-implemented never copied.

"Consolidate our own filon on the basis of FineWeb and OLMo" = our original engine orchestrating licensed sources via credited, rewritten recipes. Within the rules.

## Decision 9 — Schema extension + N1 migration

The record's `training` block gains `mix_hash` and a `data_mix` summary (the manifest or its digest). This is a schema change → dated (this ADR) and additive. **Soft migration:** the one existing record (`be1fa8139f59`) receives a retroactive `mix_hash` as a single-source mix — no retraining; the baseline stays valid. `additionalProperties: false` on `training` means the two new fields are added explicitly to the schema in the same change.

## Consequences & implementation plan (sliced, each slice with its test)

Abstraction is graved in full here; **implementation is incremental, wave-driven** (build wide, execute narrow, widen per wave):

1. **Slice 1 (W1-enabling) — IMPLEMENTED & GREEN 2026-09-06:** `cortex_data/` — operator interface (`Op`), hashed DAG manifest, lazy pull-based `BinSource`/`ArraySource`/`Decontaminate`/`Mix` (signature-aware multiset), compilation to a memmap-readable uint16 `.bin` (train.py-compatible), and the **five-invariant harness** in `tests/test_invariants.py` (all passing). Integration verified: engine → `.bin` → trainer memmap. Wiring `mix_hash` into `train.py`'s record + N1 retro-migration remains (next).
2. **Slice 2 (when a 2nd mix exists):** node memoisation cache; `concat`/`interleave`.
3. **Slice 3 (when n sizeable sources cross):** the Bloom-filter algebra at scale.
4. **Slice 4 (post-W1, RES-14):** the data-β estimator behind the Decision-5 seam, once the frontier has points.

N1 is untouched. Nothing here chooses a mix; it makes mixes cheap to produce, hash-comparable, and honest — so the Pareto frontier (D16) can do the choosing. The winning mix is always a ledger result, never an engine decision.
