# How this repository is organised, and what every label means

The record uses several numbering schemes on purpose: an idea, a decision, a measurement and a plan are different objects and must not be confused. This page is the key. It is the second thing to read after the README, and the first if a label in some other document is unfamiliar.

## The five labels

| Label | Example | What it is | Where it lives | Can it change? |
|---|---|---|---|---|
| **RES-n** | RES-23, the familiarity mark | **An idea**, with its brain source, what it would test and its reserve. An idea is not evidence | `docs/IDEAS-REGISTER-2026-08-07.md`, one section per entry | Its status changes (named → declared → built → measured); its number never |
| **Rev-n** | Rev72 | **A dated revision of the register**: what was decided, declared, measured or discarded on that day, and why. The narrative spine of the project | same file, the list at the top, newest first | Never rewritten. A later revision supersedes an earlier one and says so |
| **ADR-n** | ADR-008 | **An architecture decision**, and the place where a run's readings are declared **before** it is run. An amendment to an ADR is how a new variable enters | `docs/adr/` | Amended, never silently edited. Each amendment is dated and carries its outcome once measured |
| **Row n** | row 39 | **A measurement**, one line per run or probe: what was measured, the numbers, the artefacts, the test that recomputes them, and the reserve | `docs/RESULTS.md` | Never. A row whose artefact is missing is treated as absent |
| **D-n** | D17, the public switch | **A hub decision**, taken in `quantum-meridian` and mirrored here when it governs this repository | the hub's `docs/decisions/` | By a later hub decision |

**NOW-n** and **W1 to W4** appear in the older planning documents: NOW-n are the early work packages of August, the waves W1 to W4 are the order in which brain derived features may enter, defined in ADR-003. Neither is a result.

## The path an idea takes

```
RES-n written in the register          (an idea, with its brain source and its reserve)
   ↓  a dated revision says why it is worth a run
Rev-n                                  (the decision, in the open)
   ↓  its readings are declared before anything runs
ADR-n amendment                        (what each possible outcome will mean, including the ones that sink it)
   ↓  the run, its log retained as a file before any number is read
Row n in RESULTS.md                    (the numbers, the artefacts, the test, the reserve)
   ↓  what it changed
the whitepaper, and the ADR's outcome paragraph
```

A number that skipped a step is not in the record. That is the whole discipline, and `docs/FALSIFY.md` is the twenty minute way to check that it was followed.

## Where to look for what

- **The result** → `docs/WHITEPAPER.md`, and `docs/RESULTS.md` for every number with its artefact.
- **How to attack it** → `docs/FALSIFY.md`.
- **What was decided and when** → `docs/adr/` for the architecture, the revision list in the register for the narrative, the hub's decision log for what governs both repositories.
- **What is planned** → `docs/roadmap.md` (the steps and their gates, current), `docs/SEQUENCE.md` (how the build order was decided, superseded for the current state).
- **The brain side** → `docs/adr/ADR-003-brain-feature-map.md` (the transposition map, twenty eight lines, audited in whitepaper §5c bis), `docs/concepts/` (one page per structure).
- **The protocol** → `docs/benchmarks/hm-protocol.md`, whose four thresholds have been frozen since 2026-09-12.
- **The artefacts** → `metrics/runs.jsonl` (one line per training run), `metrics/ARTEFACTS.md` (where each checkpoint and sealed journal lives), `metrics/mqar/` (the measurement files and the retained logs).
- **How the work is run** → `docs/working/`, method and not evidence.

## Documents kept for the history, not for the current state

These carry their date in their name and were true when written. They are not updated, and nothing in them should be read as the current state: `EUROHPC-PLAN-2026-08-07`, `EXECUTION-2026-08-07`, `FEATURES-2026-08-07`, `OSS-SURVEY-2026-08-07`, `EXTRACTION-K3-2026-08-08`, `EXTRACTION-LANDSCAPE-2026-08-08`, `HANDOFF-2026-08-07-ecosystem-context`, `STATUS-2026-09-13`, `ONE-PAGER-2026-09-14`, and `SEQUENCE.md`. For the current state, the three files that are kept current are `RESULTS.md`, `roadmap.md` and `WHITEPAPER.md`.
