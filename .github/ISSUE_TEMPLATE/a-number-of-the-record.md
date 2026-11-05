---
name: A number of the record
about: You recomputed a number of docs/RESULTS.md and found something else, or a test of the record fails on your clone
title: "Row <n>: <the number> does not recompute"
labels: record
---

**The row and the number.** Row `<n>` of `docs/RESULTS.md`, the number as the row prints it: `<…>`.

**The file and the command.** The committed file the row names (`metrics/mqar/<…>`) and the command you ran (`docs/FALSIFY.md` names the one that recomputes each number), with its output pasted whole:

```
<the output>
```

**What you found.** The number you obtained, and the difference. If a test fails, the test's name and the last lines of `pytest`'s output.

**Your environment.** The commit (`git rev-parse HEAD`), the Python version, the operating system; whether the run was on a fresh clone.

A failure of a number is a result, attributed before anything changes: the row is never rewritten, a dated addition records what was found.
