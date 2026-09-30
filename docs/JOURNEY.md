# A long haul: what this step means, what it does not, and what comes next

*This page is not part of the whitepaper and claims nothing the record does not. It exists so that a reader understands the approach before the numbers, and the implications after them. Written 2026-09-30, at the first edition.*

## What this project is doing, and at what pace

quantum-cortex asks one question and refuses to answer it faster than the evidence allows: can a language model be given a memory that survives the end of a session, in a way that can be specified, measured and improved, at a scale one person can afford? The question is old. What is new here is the discipline around it: thresholds frozen before any result existed, the readings of every run written down before it ran, a log kept as a file before any number in it is read, a failure treated as a result and attributed before the next attempt, and the whole record public, including the runs that did not work.

Six weeks separate the first entry of the register from this edition. In that time the record accumulated forty measured rows, twelve training runs, about forty six GPU hours on a free tier and a laptop, seventy eight dated revisions of the register and eight architecture decisions. Most of the model measurements in it are INVALID and say so in the same table as the rest. That is not a confession; it is the point. A record that shows only what worked cannot be trusted about what worked.

The pace is the pace of one researcher with a language model as a working partner. The partner writes most of the code and reads every log; the researcher sets the discipline, decides what enters and what waits, and refuses a good part of what is proposed. Neither could have produced this record alone at this speed, and neither is allowed to grave a number the other has not read cold from a file.

## Why the humility is structural, not a tone

Research on memory in language models is full of results that do not survive a second seed, a real task or a re reading. This project has withdrawn one claim already (a five seed run whose log was never retained), retrograded another (a router that reached 93 % of its ceiling on a synthetic task and 25 % on real spans), caught a construction error by re reading its own code, and settled a disagreement between two AI instances by going back to the source. Each of these is in the register with its date. The seven rule gate of the whitepaper exists because of them.

So when this edition says that something holds, it says on how many seeds, against which frozen threshold, with which reserve written next to it. When it says that something does not hold, it says why, as precisely as the measurements allow, and what would have to be true for it to hold later. The reader who wants to attack the work has a twenty minute page for it (`docs/FALSIFY.md`). Humility here is not a posture; it is what is left when every claim has to survive that page.

## What this step means

Three things are established at this edition, each with a test that can fail.

The organ exists and holds what the weights should not: cut the journal and the episodes vanish while the skills stay, on four seeds, in a new process, from the disk alone. The model, at 27 million parameters, retrieves the right episode and reads it, and does not learn from the bytes alone to judge whether what it read answers the question; that limit is stated with its size attached and its attribution measured three ways. And the system, the model deciding and the organ answering, holds the four frozen thresholds on two independent seeds once the organ's own judgment is given to the model as a number and the organ is allowed to veto what it cannot back.

To that, the edition adds what the organ guarantees under attack, in two registers kept apart: what the code does by construction, and what has been measured. A reader holding another domain's key obtains nothing from the organ, by construction, and no claim from the system, on three seeds and six hundred queries; the margin by which that zero holds is written next to it, because it is thin on one seed.

## What this step does not mean

This is not a product, not a model anyone should deploy, and not a claim of general capability. The model is small; the facts it stores are synthetic and leakage proof by design; one of the three components of continuity is measured, two are specified and not built; the recurrent hybrids are negative at this size; the external benchmark on open models has one execution and no agreement yet; and the journal's integrity covers its content but not yet its sequence, a flaw the edition declares rather than hides.

It is also not a proof that the two systems thesis, a cortex that does not rebuild what its hippocampus already holds, is right. It is the first setting in which that thesis can fail on a table, at a scale one person can afford, and it has not failed there yet.

## What comes next, in order

The next edition (v1.1) is a list, not a promise, and each item faces a residual this record measured. The index's buckets, where the remaining retrieval misses live. A wider negative control, because the veto's margin rests on fifty entities per seed and was measured thin once. The write side separation of near duplicates, with its threshold set on the organ's marks. The detectors as code: a refusal written to the journal with its cause, a status carried by the answer, a two arm suite that freezes their false positive rate. The consolidation arms of the recurrent base, the one measurement that could make the trunk carry the regime. A reader with a local convolution. The agreement of the external arm on one hardware, then larger open models. And the reconstruction test on the organ's layer of numbers, declared in the paper's last door.

Beyond that, the question that decides the model arm's future is a matter of size and needs a larger run than a free tier allows: does the reader's selectivity to the entity emerge at 150 million, 300 million or a billion parameters? That run is the declared next step, in the European high performance computing environment, and it will be declared before it is measured, like everything here.

## What it implies, if it holds

For practitioners: a measurement you can run on your own model today, with frozen thresholds and a negative control, that answers whether the model holds an episode or reconstructs it. For the field: a small, public, falsifiable setting for the separation of skills and episodes, with a record that can be audited to the digit. For users, eventually: a memory that is sealed per subject, stays on the machine, and can be shown to be a memory rather than a prompt. None of these is delivered by this edition; each is made testable by it.

## How to read the record

Start with the README, then `docs/NAVIGATION.md` for the labels, then `docs/RESULTS.md` for any number, then the whitepaper. Everything else is method, and it is public because the method is one of the contributions.
