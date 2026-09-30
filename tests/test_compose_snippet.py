"""The README's "Use it with your own model" blocks, run as the README prints them
(whitepaper 3.6, way (i): the organ as a library, any model decoding). Session A writes
one record through the gated write path into a journal sealed on disk; session B, in a
NEW process that knows nothing of session A but the disk, cites it with its pointer and
abstains on an address never written; a second session A is refused as redundant; nothing
at rest is plaintext. The blocks are extracted from README.md, so a change to the README
that breaks the snippet breaks this test, and a change to the code that breaks the README
does too. Nothing here is a number of the record."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RECORD = b"12 Jan: the originality argument is set aside on the strength of ruling 2024/AR/512"


def _block(marker: str) -> str:
    """The fenced python block that follows `<!-- compose: <marker> -->` in the README."""
    text = README.read_text(encoding="utf-8")
    m = re.search(rf"<!-- compose: {re.escape(marker)} -->\s*```python\n(.*?)```", text, flags=re.S)
    assert m, f"README block '{marker}' missing"
    return m.group(1)


def _run(code: str, cwd: Path, key_hex: str) -> str:
    """One block, in its own process, with the key in the environment and the repository on the path."""
    env = dict(os.environ, QUANTUM_CORTEX_JOURNAL_KEY=key_hex,
               PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(cwd), env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout


def test_readme_compose_blocks_run_as_printed(tmp_path):
    from cortex_c2b.crypto import generate_key
    key_hex = generate_key().hex()
    a, b = _block("session A"), _block("session B")

    out_a = _run(a, tmp_path, key_hex)
    assert out_a.startswith("written "), out_a
    pointer12 = out_a.split()[1]
    assert re.fullmatch(r"[0-9a-f]{12}", pointer12), out_a

    out_b = _run(b, tmp_path, key_hex)                    # a new process: the disk alone
    lines = out_b.strip().splitlines()
    assert lines[0].startswith(f"cite {pointer12} "), out_b
    assert RECORD.decode() in lines[0], out_b
    assert lines[1] == "abstain: no record for decision|argument:database-right", out_b

    out_a2 = _run(a, tmp_path, key_hex)                   # the gate: what the journal already predicts is refused
    assert out_a2.startswith("refused:"), out_a2

    journal_dir = tmp_path / "runs" / "my-scope"
    files = [journal_dir / "journal.jsonl"] + sorted((journal_dir / "journal.payloads").iterdir())
    assert len(files) == 2, files                          # the log and one payload
    for f in files:
        raw = f.read_bytes()
        assert RECORD not in raw and b"originality" not in raw, f"plaintext at rest in {f.name}"


def test_readme_names_the_test_and_the_public_cue():
    text = README.read_text(encoding="utf-8")
    assert "tests/test_compose_snippet.py" in text
    from cortex_c2b.hm_protocol import address_cue, _cue_for
    assert address_cue is _cue_for
