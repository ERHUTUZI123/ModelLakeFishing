import os
import re

import pytest

from scale1m import train_rung as TR


def test_git_root_is_the_repository_not_its_parent():
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
    short = TR._git("status", "--short")
    tracked = TR._git("status", "--porcelain", "--untracked-files=no")
    assert short is not None and tracked is not None
    for line in tracked.splitlines():
        assert not line.startswith("??"), line


def test_git_failure_is_distinguishable_from_a_clean_result():
    assert TR._git("definitely-not-a-git-subcommand") is None


def test_fresh_run_dir_reports_no_prior_contents(tmp_path):
    out, prior = TR.make_run_dir(str(tmp_path / "R_fresh"))
    assert prior == []
    for sub in ("ckpt", "exports", "metrics", "stdout", "metadata"):
        assert os.path.isdir(os.path.join(out, sub))


def test_reused_run_dir_reports_the_leftover_log(tmp_path):
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
    out = str(tmp_path / "R_empty")
    TR.make_run_dir(out)
    open(os.path.join(out, "stdout", "train.log"), "wb").close()

    _, prior = TR.make_run_dir(out)
    assert prior == []
