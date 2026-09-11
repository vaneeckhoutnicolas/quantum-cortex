"""cortex_c2b.crypto -- encryption at rest for the journal and its payloads (ADR-007 D6, D8).

Same concept on both planes: the Quantum Meridian daemon encrypts its journal
per line with AES-256-GCM in the framing `QJE1:<base64(iv|tag|ciphertext)>`
(hub decision 010, layer 1). The cortex reuses that framing byte for byte for
its log lines, and a binary variant for payload files (`b"QJE1" + iv + tag +
ciphertext`), so a tampered line or file fails loudly instead of yielding
garbage, and a legacy clear line can coexist with encrypted ones in one file
(migration falls out for free, as on the hub).

Scope: one key per journal, one journal per scope (hub decision 009). Key
management (where the key lives: DPAPI on the hub, secret store, HSM) is a
separate problem, deferred as in hub decision 010 -- here the key is an
explicit 32 byte argument, never read implicitly, never written to disk.

Dependency: `cryptography` (Apache-2.0 OR BSD-3-Clause, license checked at
ingestion on 2026-09-12), imported lazily: a journal without a key never
imports it. Written from scratch (our filon) over that primitive; we do not
implement ciphers ourselves.
"""
from __future__ import annotations

import base64
import os
import secrets

MAGIC = b"QJE1"
LINE_PREFIX = "QJE1:"
IV_LEN, TAG_LEN, KEY_LEN = 12, 16, 32
AAD_LINE = b"quantum-cortex journal line"


def generate_key() -> bytes:
    """A fresh 256 bit key. Where it is kept is the caller's decision."""
    return secrets.token_bytes(KEY_LEN)


def key_from_env(var: str = "QUANTUM_CORTEX_JOURNAL_KEY") -> bytes | None:
    """Optional helper: a hex key in the environment, or None. Explicit call only."""
    v = os.environ.get(var)
    if not v:
        return None
    k = bytes.fromhex(v)
    if len(k) != KEY_LEN:
        raise ValueError(f"{var} must be {KEY_LEN} bytes hex, got {len(k)}")
    return k


def _aesgcm(key: bytes):
    if len(key) != KEY_LEN:
        raise ValueError(f"key must be {KEY_LEN} bytes, got {len(key)}")
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:                                  # pragma: no cover
        raise ImportError("encryption at rest needs the `cryptography` package "
                          "(pip install cryptography)") from e
    return AESGCM(key)


def seal(key: bytes, plaintext: bytes, aad: bytes) -> bytes:
    """Binary framing for payload files: MAGIC | iv | tag | ciphertext."""
    iv = os.urandom(IV_LEN)
    ct_tag = _aesgcm(key).encrypt(iv, plaintext, aad)
    return MAGIC + iv + ct_tag[-TAG_LEN:] + ct_tag[:-TAG_LEN]


def open_sealed(key: bytes, blob: bytes, aad: bytes) -> bytes:
    if not blob.startswith(MAGIC):
        raise ValueError("not a sealed blob")
    body = blob[len(MAGIC):]
    iv, tag, ct = body[:IV_LEN], body[IV_LEN:IV_LEN + TAG_LEN], body[IV_LEN + TAG_LEN:]
    return _aesgcm(key).decrypt(iv, ct + tag, aad)          # raises InvalidTag on tampering / wrong key


def is_sealed(blob: bytes) -> bool:
    return blob.startswith(MAGIC)


def sealed_overhead() -> int:
    return len(MAGIC) + IV_LEN + TAG_LEN


def seal_line(key: bytes, text: str) -> str:
    """Text framing for log lines, identical to the hub's: QJE1:<base64(iv|tag|ct)>."""
    iv = os.urandom(IV_LEN)
    ct_tag = _aesgcm(key).encrypt(iv, text.encode("utf-8"), AAD_LINE)
    return LINE_PREFIX + base64.b64encode(iv + ct_tag[-TAG_LEN:] + ct_tag[:-TAG_LEN]).decode("ascii")


def open_line(key: bytes | None, line: str) -> str:
    """A clear line passes through; a sealed line needs the key and fails loudly."""
    if not line.startswith(LINE_PREFIX):
        return line
    if key is None:
        raise ValueError("encrypted journal line and no key given")
    raw = base64.b64decode(line[len(LINE_PREFIX):])
    iv, tag, ct = raw[:IV_LEN], raw[IV_LEN:IV_LEN + TAG_LEN], raw[IV_LEN + TAG_LEN:]
    return _aesgcm(key).decrypt(iv, ct + tag, AAD_LINE).decode("utf-8")


def is_sealed_line(line: str) -> bool:
    return line.startswith(LINE_PREFIX)
