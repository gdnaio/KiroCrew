"""A step that edits a file that is not UTF-8 keeps its diff for review.

``git diff`` prints the bytes of a text file as they are stored, so a Latin-1
source file makes a strict decode of git's output raise ``UnicodeDecodeError``.
``get_step_diff`` swallows that error and answers ``""``, which ``self_review``
reads as "no diff" on a git run. Output that only feeds a prompt is decoded with
replacement; output that is compared or reused as a path stays strict.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import requires_git
from kiro_crew import git_coord
from kiro_crew.taskrunner import Step, TaskRun

pytestmark = requires_git


@pytest.fixture(autouse=True)
def _passthrough_sandbox(_floor_monkeypatch):
    """Run the real git unwrapped: the tests exercise decoding, not the sandbox."""
    _floor_monkeypatch.setattr(
        git_coord,
        "sandboxed_spawn_argv",
        lambda argv, *a, **k: (list(argv), dict(os.environ), None),
    )


async def _latin1_run(tmp_path: Path, task_id: str) -> TaskRun:
    """A run whose first step rewrote a committed Latin-1 file."""
    repo = tmp_path / "proj"
    repo.mkdir()
    await git_coord._git(str(repo), "init", "-b", "main")
    await git_coord._git(str(repo), "config", "user.email", "t@example.com")
    await git_coord._git(str(repo), "config", "user.name", "t")
    (repo / "legacy.txt").write_bytes("caf\xe9 au lait\n".encode("latin-1"))
    await git_coord._git(str(repo), "add", "-A")
    await git_coord._git(str(repo), "commit", "-m", "initial")
    run = TaskRun(spec_path="/s.md", spec_content="s")
    run.task_id = task_id
    run.work_dir = str(repo)
    await git_coord.init_workspace(run)
    assert run.git_enabled is True
    (Path(run.work_dir) / "legacy.txt").write_bytes("caf\xe9 au lait, cr\xe8me\n".encode("latin-1"))
    sha = await git_coord.commit_step(
        run, Step(index=1, title="edit caf\u00e9 file", description="d")
    )
    assert sha
    return run


class TestPromptOnlyReadsKeepNonUtf8Output:
    @pytest.mark.asyncio
    async def test_step_diff_names_the_non_utf8_file(self, tmp_path: Path) -> None:
        run = await _latin1_run(tmp_path, "latin1_diff")

        diff = await git_coord.get_step_diff(run)

        assert diff != ""
        assert "legacy.txt" in diff
        assert "caf" in diff

    @pytest.mark.asyncio
    async def test_state_summary_survives_a_non_utf8_log_encoding(self, tmp_path: Path) -> None:
        """A repository that prints its log in Latin-1 still gives the run its summary."""
        run = await _latin1_run(tmp_path, "latin1_summary")
        await git_coord._git(run.work_dir, "config", "i18n.logOutputEncoding", "ISO-8859-1")
        with pytest.raises(UnicodeDecodeError):
            await git_coord._git(run.work_dir, "log", "--oneline", "-1")

        summary = await git_coord.get_state_summary(run)

        assert "## Git Log (changes so far)" in summary
        assert "step 1: edit caf" in summary
        assert "legacy.txt" in summary


class TestGitReadsOutsideThePromptStayStrict:
    @pytest.mark.asyncio
    async def test_default_decode_still_raises_on_non_utf8_output(self, tmp_path: Path) -> None:
        run = await _latin1_run(tmp_path, "latin1_strict")

        with pytest.raises(UnicodeDecodeError):
            await git_coord._git(run.work_dir, "diff", "HEAD~1")

    @pytest.mark.asyncio
    async def test_replace_is_opt_in_per_call(self, tmp_path: Path) -> None:
        run = await _latin1_run(tmp_path, "latin1_opt_in")

        text = await git_coord._git(run.work_dir, "diff", "HEAD~1", errors="replace")

        assert isinstance(text, str)
        assert "\ufffd" in text

    @pytest.mark.asyncio
    async def test_utf8_output_is_unchanged(self, tmp_path: Path) -> None:
        run = await _latin1_run(tmp_path, "utf8_same")
        (Path(run.work_dir) / "note.txt").write_text("café ☕\n", encoding="utf-8")
        await git_coord.commit_step(run, Step(index=2, title="add note", description="d"))

        strict = await git_coord._git(run.work_dir, "diff", "HEAD~1")

        assert "+café ☕" in strict
        assert await git_coord.get_step_diff(run) == strict

    @pytest.mark.asyncio
    async def test_a_failing_git_still_raises_runtime_error(self, tmp_path: Path) -> None:
        run = await _latin1_run(tmp_path, "latin1_fail")

        with pytest.raises(RuntimeError, match="git rev-parse --verify nope failed"):
            await git_coord._git(run.work_dir, "rev-parse", "--verify", "nope", errors="replace")
