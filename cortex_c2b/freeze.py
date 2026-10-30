"""cortex_c2b.freeze -- the frozen set of ADR-010 A.5 as a file (the amendment of 2026-10-15 on parts B and C, section 0;
graved 2026-10-29).

The v1.1.0 system is frozen at `c5b4818d3db29902f72d1b3078e01c4b3cb1e634`. The frozen set is every file matching
`cortex_c2b/**/*.py`, `cortex_c2/**/*.py`, `train.py`, `configs/**` and `tests/**` present at that commit. The first build
of the ADR writes `metrics/mqar/FREEZE-v1.1.0.json` from the git objects of that commit (never from a working tree): the
commit's full identifier, the file count, and per file, sorted by path, the sha256 of its bytes with `\\r\\n` read as
`\\n` (the record's `sha256_lf` convention, so that a Windows checkout and the git objects hash alike).

Two tests use it: the first compares the frozen files of the checkout under test with the list and runs everywhere; the
second reads the git objects of the commit that produced a row's files, runs when they are present and is reported as
"not run: no git objects" otherwise, never as passed. A frozen file that differs fails every record test of the ADR; new
files (this module among them) are outside the list by construction.

The founder's command, from `C:\\git\\quantum-cortex` (the git history is on his machine):
    python -m cortex_c2b.freeze --write --commit c5b4818d3db29902f72d1b3078e01c4b3cb1e634
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

FREEZE_COMMIT = "c5b4818d3db29902f72d1b3078e01c4b3cb1e634"
FREEZE_PUBLIC_COMMIT = "e1488c0c76e4075ff0e9e9e89ba2dadaca4d5d7b"
FREEZE_PATH = Path("metrics/mqar/FREEZE-v1.1.0.json")
FROZEN_PATTERNS = ("cortex_c2b/**/*.py", "cortex_c2/**/*.py", "train.py", "configs/**", "tests/**")
CONVENTION = "sha256_lf: the sha256 of the file's bytes with \\r\\n read as \\n, in full (sixty four hexadecimal characters)"
NOT_RUN = "not run: no git objects"


def sha256_lf(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def is_frozen_path(path: str) -> bool:
    """Whether a repository relative path (forward slashes) falls in the frozen set's patterns."""
    p = path.replace("\\", "/")
    for pat in FROZEN_PATTERNS:
        if pat.endswith("/**"):
            if p.startswith(pat[:-3] + "/"):
                return True
        elif "/**/" in pat:
            head, tail = pat.split("/**/", 1)
            if p.startswith(head + "/") and fnmatch.fnmatch(p.rsplit("/", 1)[-1], tail):
                return True
        elif p == pat:
            return True
    return False


def _git(args: list[str], root: Path, binary: bool = False):
    r = subprocess.run(["git", *args], cwd=str(root), capture_output=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.decode('utf-8', 'replace').strip()}")
    return r.stdout if binary else r.stdout.decode("utf-8")


def has_git_objects(root: Path, commit: str = FREEZE_COMMIT) -> bool:
    """Whether the checkout at `root` holds the git objects of the commit."""
    try:
        out = _git(["cat-file", "-t", commit], root).strip()
        return out == "commit"
    except Exception:
        return False


def files_at_commit(root: Path, commit: str) -> dict[str, str]:
    """The frozen files at the commit with their sha256_lf, from the git objects alone."""
    full = _git(["rev-parse", f"{commit}^{{commit}}"], root).strip()
    if len(commit) == 40 and full != commit:
        raise RuntimeError(f"the commit {commit} resolves to {full}")
    listing = _git(["ls-tree", "-r", "--name-only", full], root).splitlines()
    out = {}
    for path in sorted(p.strip() for p in listing if p.strip()):
        if is_frozen_path(path):
            blob = _git(["cat-file", "-p", f"{full}:{path}"], root, binary=True)
            out[path] = sha256_lf(blob)
    return out


def write_freeze(root: Path, commit: str = FREEZE_COMMIT, out: Path | None = None) -> dict:
    """The freeze file from the git objects of the commit (never from a working tree). Refused when the file exists."""
    out = Path(out) if out else (Path(root) / FREEZE_PATH)
    if out.exists():
        raise FileExistsError(f"{out} exists; a file of the record is never overwritten")
    full = _git(["rev-parse", f"{commit}^{{commit}}"], Path(root)).strip()
    if len(commit) != 40 or full != commit:
        raise RuntimeError(f"the freeze is written from the full identifier of the commit; {commit!r} resolves to {full}")
    files = files_at_commit(Path(root), full)
    doc = {"what": "the frozen set of ADR-010 A.5: every file matching the patterns at the freeze commit, with its sha256_lf, read from the git objects of that commit",
           "commit": full, "public_copy_commit": FREEZE_PUBLIC_COMMIT, "patterns": list(FROZEN_PATTERNS), "convention": CONVENTION,
           "source": "git objects (ls-tree and cat-file at the commit), never a working tree",
           "file_count": len(files), "files": files, "tree_sha256": tree_sha256(files),
           "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    return doc


def tree_sha256(files: dict[str, str]) -> str:
    """One sha256 over the sorted (path, sha256_lf) pairs."""
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(p.encode("utf-8") + b"\0" + files[p].encode("ascii") + b"\n")
    return h.hexdigest()


def load_freeze(root: Path, path: Path | None = None) -> dict:
    p = Path(path) if path else (Path(root) / FREEZE_PATH)
    return json.loads(p.read_text(encoding="utf-8"))


def check_tree(freeze: dict, root: Path) -> dict:
    """The first test: the frozen files of the checkout under test against the list (runs everywhere)."""
    missing, differing, equal = [], [], 0
    for path, want in sorted(freeze["files"].items()):
        f = Path(root) / path
        if not f.is_file():
            missing.append(path); continue
        have = sha256_lf(f.read_bytes())
        if have != want:
            differing.append({"path": path, "frozen": want, "checkout": have})
        else:
            equal += 1
    return {"ok": not missing and not differing, "equal": equal, "missing": missing, "differing": differing,
            "file_count": len(freeze["files"]), "commit": freeze["commit"]}


def tree_hash_of_list(freeze: dict, root: Path) -> str:
    """The code commit recomputed from the frozen files' hashes as present in the checkout (the run identity of 4 (b))."""
    files = {}
    for path in sorted(freeze["files"]):
        f = Path(root) / path
        files[path] = sha256_lf(f.read_bytes()) if f.is_file() else "missing"
    return tree_sha256(files)


def check_git_objects(freeze: dict, root: Path, commit: str | None = None) -> dict:
    """The second test: the git objects of the commit against the list; "not run: no git objects" when they are absent,
    never passed by default."""
    commit = commit or freeze["commit"]
    if not has_git_objects(Path(root), commit):
        return {"status": NOT_RUN, "ok": None, "commit": commit}
    files = files_at_commit(Path(root), commit)
    diffs = [p for p in set(files) | set(freeze["files"]) if files.get(p) != freeze["files"].get(p)]
    return {"status": "run", "ok": not diffs, "commit": commit, "differing": sorted(diffs), "file_count_git": len(files),
            "file_count_list": len(freeze["files"])}


def _cli(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="the frozen set of ADR-010 A.5 as a file, from the git objects of the freeze commit")
    ap.add_argument("--write", action="store_true"); ap.add_argument("--check", action="store_true")
    ap.add_argument("--commit", default=FREEZE_COMMIT, help="the full identifier of the freeze commit")
    ap.add_argument("--root", default=None, help="the repository root (the current directory by default)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else Path.cwd()
    if args.write:
        try:
            doc = write_freeze(root, args.commit, Path(args.out) if args.out else None)
        except (RuntimeError, FileExistsError) as e:
            raise SystemExit(f"freeze: {e}")
        out = Path(args.out) if args.out else (root / FREEZE_PATH)
        print(f"retained: {out}  files {doc['file_count']}  tree_sha256 {doc['tree_sha256'][:16]}  sha256 {hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")
        return
    if args.check:
        fz = load_freeze(root, Path(args.out) if args.out else None)
        t = check_tree(fz, root); g = check_git_objects(fz, root)
        print(json.dumps({"tree": {k: t[k] for k in ("ok", "equal", "missing", "file_count")} | {"differing": [d["path"] for d in t["differing"]]}, "git_objects": g}, indent=2))
        raise SystemExit(0 if (t["ok"] and g["ok"] in (None, True)) else 1)
    ap.error("one of --write / --check")


if __name__ == "__main__":
    _cli()
