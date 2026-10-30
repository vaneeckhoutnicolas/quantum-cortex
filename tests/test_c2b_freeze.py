"""The freeze of ADR-010 A.5 (the amendment of 2026-10-15 on parts B and C, section 0): the frozen set written from the git
objects of the freeze commit, never from a working tree; the two tests the amendment names, on a temporary repository
here (the record's file is written by the founder's command and checked by the record tests of rows 51 to 54)."""
import json
import subprocess
from pathlib import Path

import pytest

from cortex_c2b import freeze as F
from cortex_c2b import record_tools


def git(args, root):
    r = subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A small repository with files inside and outside the patterns, one of them with CRLF line endings, committed twice."""
    root = tmp_path / "repo"
    (root / "cortex_c2b" / "sub").mkdir(parents=True)
    (root / "cortex_c2").mkdir(); (root / "configs").mkdir(); (root / "tests").mkdir(); (root / "docs").mkdir(); (root / "metrics").mkdir()
    (root / "cortex_c2b" / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")                # CRLF: hashed as LF
    (root / "cortex_c2b" / "sub" / "b.py").write_bytes(b"z = 3\n")
    (root / "cortex_c2b" / "data.txt").write_bytes(b"not a module\n")               # outside: cortex_c2b/**/*.py only
    (root / "cortex_c2" / "c.py").write_bytes(b"c = 1\n")
    (root / "train.py").write_bytes(b"train\n")
    (root / "configs" / "x.json").write_bytes(b"{}\n")
    (root / "tests" / "test_x.py").write_bytes(b"def test(): pass\n")
    (root / "docs" / "note.md").write_bytes(b"outside\n")
    git(["init", "-q"], root); git(["config", "user.email", "t@t"], root); git(["config", "user.name", "t"], root)
    git(["config", "core.autocrlf", "false"], root)
    git(["config", "commit.gpgsign", "false"], root)         # the founder's global config signs commits with a key under a passphrase; the fixture's commits are not signed
    git(["add", "-A"], root); git(["commit", "-q", "-m", "one"], root)
    first = git(["rev-parse", "HEAD"], root)
    (root / "cortex_c2b" / "a.py").write_bytes(b"x = 1\r\ny = 22\r\n")
    git(["add", "-A"], root); git(["commit", "-q", "-m", "two"], root)
    second = git(["rev-parse", "HEAD"], root)
    return root, first, second


def test_the_patterns():
    assert F.is_frozen_path("cortex_c2b/a.py") and F.is_frozen_path("cortex_c2b/sub/b.py") and F.is_frozen_path("cortex_c2/c.py")
    assert F.is_frozen_path("train.py") and F.is_frozen_path("configs/x.json") and F.is_frozen_path("tests/test_x.py") and F.is_frozen_path("tests\\sub\\t.py")
    assert not F.is_frozen_path("cortex_c2b/data.txt") and not F.is_frozen_path("docs/note.md") and not F.is_frozen_path("tools/adr010_launcher.py")
    assert not F.is_frozen_path("train.pyc") and not F.is_frozen_path("metrics/mqar/x.json")
    assert F.sha256_lf(b"a\r\nb\r\n") == F.sha256_lf(b"a\nb\n")


def test_write_freeze_reads_the_git_objects_of_the_commit_not_the_working_tree(repo):
    root, first, second = repo
    out = root / "metrics" / "FREEZE.json"
    doc = F.write_freeze(root, first, out)
    assert doc["commit"] == first and doc["file_count"] == 6 and set(doc["files"]) == {"cortex_c2b/a.py", "cortex_c2b/sub/b.py", "cortex_c2/c.py", "train.py", "configs/x.json", "tests/test_x.py"}
    assert doc["files"]["cortex_c2b/a.py"] == F.sha256_lf(b"x = 1\ny = 2\n")        # the first commit's content, LF, although the tree holds the second's
    assert doc["source"].startswith("git objects") and doc["convention"].startswith("sha256_lf")
    assert doc["tree_sha256"] == F.tree_sha256(doc["files"]) and len(doc["tree_sha256"]) == 64
    with pytest.raises(FileExistsError):
        F.write_freeze(root, first, out)                                            # a file of the record is never overwritten
    with pytest.raises(RuntimeError):
        F.write_freeze(root, first[:7], root / "metrics" / "other.json")             # the full identifier, never a short one
    with pytest.raises(RuntimeError):
        F.write_freeze(root, "0" * 40, root / "metrics" / "other2.json")
    loaded = F.load_freeze(root, out)
    assert loaded == json.loads(out.read_text(encoding="utf-8")) and loaded["files"] == doc["files"]


def test_check_tree_and_check_git_objects(repo, tmp_path):
    root, first, second = repo
    out = root / "metrics" / "FREEZE.json"
    doc = F.write_freeze(root, second, out)
    t = F.check_tree(doc, root)
    assert t["ok"] and t["equal"] == 6 and t["missing"] == [] and t["differing"] == []
    (root / "cortex_c2b" / "sub" / "b.py").write_bytes(b"z = 4\n")
    t = F.check_tree(doc, root)
    assert not t["ok"] and [d["path"] for d in t["differing"]] == ["cortex_c2b/sub/b.py"]
    (root / "train.py").unlink()
    t = F.check_tree(doc, root)
    assert t["missing"] == ["train.py"] and t["equal"] == 4
    assert F.tree_hash_of_list(doc, root) != doc["tree_sha256"]
    g = F.check_git_objects(doc, root)
    assert g["status"] == "run" and g["ok"] and g["file_count_git"] == 6               # the objects of the commit, whatever the tree
    doc1 = F.write_freeze(root, first, root / "metrics" / "F1.json")
    g1 = F.check_git_objects(doc1, root, commit=second)
    assert g1["status"] == "run" and not g1["ok"] and g1["differing"] == ["cortex_c2b/a.py"]
    elsewhere = tmp_path / "no-git"; elsewhere.mkdir()
    g2 = F.check_git_objects(doc, elsewhere)
    assert g2["status"] == F.NOT_RUN and g2["ok"] is None                             # never passed by default
    assert not F.has_git_objects(elsewhere, second) and F.has_git_objects(root, second) and not F.has_git_objects(root, "0" * 40)


def test_the_record_freeze_file_when_present_names_the_frozen_commit_and_matches_this_checkout():
    """The founder's file, once written: its commit is the freeze commit, the convention is sha256_lf, and the checkout under
    test holds every frozen file as listed (the record test the rows 51 to 54 run)."""
    path = record_tools.REPO_ROOT / F.FREEZE_PATH
    if not path.exists():
        pytest.skip("the freeze file is written by the founder's command (python -m cortex_c2b.freeze --write --commit c5b4818d3db29902f72d1b3078e01c4b3cb1e634)")
    doc = F.load_freeze(record_tools.REPO_ROOT)
    assert doc["commit"] == F.FREEZE_COMMIT and doc["public_copy_commit"] == F.FREEZE_PUBLIC_COMMIT
    assert doc["patterns"] == list(F.FROZEN_PATTERNS) and doc["tree_sha256"] == F.tree_sha256(doc["files"])
    t = F.check_tree(doc, record_tools.REPO_ROOT)
    assert t["ok"], {"missing": t["missing"], "differing": [d["path"] for d in t["differing"]]}
    g = F.check_git_objects(doc, record_tools.REPO_ROOT)
    assert g["status"] == F.NOT_RUN or g["ok"], g


def test_the_cli_refuses_a_short_commit(repo, tmp_path):
    root, first, second = repo
    with pytest.raises(SystemExit):
        F._cli(["--write", "--commit", second[:12], "--root", str(root), "--out", str(tmp_path / "f.json")])
    F._cli(["--write", "--commit", second, "--root", str(root), "--out", str(tmp_path / "f.json")])
    assert (tmp_path / "f.json").exists()
    with pytest.raises(SystemExit) as e:
        F._cli(["--check", "--root", str(root), "--out", str(tmp_path / "f.json")])
    assert e.value.code == 0
