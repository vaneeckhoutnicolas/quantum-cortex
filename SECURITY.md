# Security policy

## Reporting a vulnerability

**Please do not open a public issue for security problems.** Report privately:

1. **Preferred** — GitHub's private vulnerability reporting: open this repository's **Security** tab and click **"Report a vulnerability"** (enable it under *Settings → Code security and analysis* if it is not already on).
2. Or email the maintainer at <vaneeckhoutnicolas@gmail.com>.

Please include what you found, where (file, version/commit), how to reproduce it, and the impact you foresee.

## Rules of this repository

- No secrets, tokens, or private keys are ever committed. Development keypairs do not belong in git — production keys are generated and stored outside the repository. The C2b journal is sealed at rest (AES-256-GCM, the hub's QJE1 framing) when a key is given; that key is an explicit argument and is never written to disk by this code (ADR-007 D8).
- The run ledger (`metrics/runs.jsonl`) is append-only and CI-validated (ADR-001): anything that lets a record be altered or dropped silently is reported as a security matter.
- This is pre-audit research software, provided "as is" without warranty — see `DISCLAIMER.md`.
