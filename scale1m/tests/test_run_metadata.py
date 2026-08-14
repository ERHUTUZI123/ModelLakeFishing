"""Tests for the run-provenance half of scale1m/train_rung.py.

These exist because of a specific failure, not a hypothetical one. Every run of
the T6 workflow of 2026-08-11 recorded `git_head: ""`, because the git commands
were issued one directory above the repository root. Nothing raised: git simply
returned non-zero, the empty stdout was stored, and the "tree is dirty, save a
patch" branch never fired -- while the tree was in fact carrying the entire
uncommitted T0+T6 changeset. The result is six runs whose code state cannot be
recovered. See docs/1M/T6more.md P1-2.

The lesson these tests encode: provenance that fails silently is worse than
provenance that is absent, because it looks the same as success.
"""

import os
import re

import pytest

from scale1m import train_rung as TR


# ── git root ─────────────────────────────────────────────────────────────────

def test_git_root_is_the_repository_not_its_parent():
    """_REPO_ROOT is the import root (one above the package); _GIT_ROOT is the
    package directory, which is where .git actually lives. Conflating them is
    the original bug."""
    assert TR._GIT_ROOT == os.path.dirname(os.path.dirname(
        os.path.abspath(TR.__file__)))
    assert TR._REPO_ROOT == os.path.dirname(TR._GIT_ROOT)
    assert TR._GIT_ROOT != TR._REPO_ROOT


@pytest.mark.skipif(not os.path.isdir(os.path.join(TR._GIT_ROOT, ".git")),
                    reason="not a git checkout (source tarball / export)")
def test_git_head_is_recorded_when_running_inside_the_checkout():
    head = TR._git("rev-parse", "HEAD")
    assert head is not None, "git failed inside its own checkout"
    assert re.fullmatch(r"[0-9a-f]{40}", head), head


@pytest.mark.skipif(not os.path.isdir(os.path.join(TR._GIT_ROOT, ".git")),
                    reason="not a git checkout (source tarball / export)")
def test_untracked_files_alone_do_not_count_as_a_dirty_tree():
    """`status --short` lists untracked paths too. On the cluster logs/ and a
    stray slurm-*.out are always there, so every run of 2026-08-13 recorded a
    dirty tree and wrote a zero-byte uncommitted.patch. Dirtiness has to mean
    "tracked files differ from HEAD" -- the thing a patch can capture."""
    short = TR._git("status", "--short")
    tracked = TR._git("status", "--porcelain", "--untracked-files=no")
    assert short is not None and tracked is not None
    for line in tracked.splitlines():
        assert not line.startswith("??"), line


def test_git_failure_is_distinguishable_from_a_clean_result():
    """A clean `status --short` returns "". A broken git must NOT also return
    "", or an unusable repo is indistinguishable from a pristine one -- which
    is exactly how the empty git_head passed unnoticed."""
    assert TR._git("definitely-not-a-git-subcommand") is None


# ── run directory reuse ──────────────────────────────────────────────────────

def test_fresh_run_dir_reports_no_prior_contents(tmp_path):
    out, prior = TR.make_run_dir(str(tmp_path / "R_fresh"))
    assert prior == []
    for sub in ("ckpt", "exports", "metrics", "stdout", "metadata"):
        assert os.path.isdir(os.path.join(out, sub))


def test_reused_run_dir_reports_the_leftover_log(tmp_path):
    """stdout/train.log is opened in append mode by the job script, so a failed
    attempt's traceback becomes part of the next attempt's delivered log. The
    manifest has to say the directory was reused."""
    out = str(tmp_path / "R_reused")
    TR.make_run_dir(out)
    with open(os.path.join(out, "stdout", "train.log"), "w",
              encoding="utf-8") as fh:
        fh.write("ModuleNotFoundError: No module named 'matplotlib'\n")

    _, prior = TR.make_run_dir(out)
    assert "stdout/train.log" in prior


def test_reused_run_dir_reports_leftover_checkpoints(tmp_path):
    out = str(tmp_path / "R_ckpt")
    TR.make_run_dir(out)
    open(os.path.join(out, "ckpt", "last.pt"), "wb").close()

    _, prior = TR.make_run_dir(out)
    assert "ckpt/*.pt" in prior


def test_an_empty_leftover_log_does_not_count_as_reuse(tmp_path):
    """The job script creates stdout/ before python starts, so a zero-byte log
    is the normal fresh case and must not raise a false alarm."""
    out = str(tmp_path / "R_empty")
    TR.make_run_dir(out)
    open(os.path.join(out, "stdout", "train.log"), "wb").close()

    _, prior = TR.make_run_dir(out)
    assert prior == []
