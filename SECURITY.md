# Security policy

## Reporting a vulnerability

**Please do not open a public issue for security problems.** Report privately:

1. **Preferred** — GitHub's private vulnerability reporting: open this repository's **Security** tab and click **"Report a vulnerability"** (enabled on the public repository since the v1.0.0 edition).
2. Or email the maintainer at <vaneeckhoutnicolas@gmail.com>.

Please include what you found, where (file, version/commit), how to reproduce it, and the impact you foresee.

## Rules of this repository

- No secrets, tokens, or private keys are ever committed. Development keypairs do not belong in git — production keys are generated and stored outside the repository. The C2b journal is sealed at rest by default (AES-256-GCM, the hub's QJE1 framing): an on-disk journal requires a key, `plaintext=True` declares a test scope explicitly; the key is an explicit argument and is never written to disk by this code (ADR-007 D8, D9); a training run reads it from the environment variable `QUANTUM_CORTEX_JOURNAL_KEY` (a Kaggle secret), never from a file in the repository.
- The run ledger (`metrics/runs.jsonl`) is append-only and CI-validated (ADR-001): anything that lets a record be altered or dropped silently is reported as a security matter.
- The package `quantum-cortex` on the Python index is published by the maintainer only, from the public copy at a stated commit (`docs/WORKSHOP.md`), from an account protected by two factor authentication and with no standing upload token; a version on the index is never re uploaded, a correction is a new version. A file that does not match the index's own hashes for a version is not this project's.
- This is pre-audit research software, provided "as is" without warranty — see `DISCLAIMER.md`.
