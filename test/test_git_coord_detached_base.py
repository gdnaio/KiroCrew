"""A run started on a detached checkout compares itself with its start commit.

``git rev-parse --abbrev-ref HEAD`` prints the word ``HEAD`` when the checkout
is detached. Stored as the run's base, ``get_state_summary`` then ran
``git log HEAD..HEAD`` and ``git diff --stat HEAD`` inside the run's worktree,
which compares the worktree with itself, so every later step got no record of
what the earlier steps changed.
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
    """Run the real git unwrapped: these tests exercise git logic, not the
    sandbox backend a host may lack."""
    _floor_monkeypatch.setattr(
        git_coord,
        "sandboxed_spawn_argv",
        lambda argv, *a, **k: (list(argv), dict(os.environ), None),
    )


async def _repo_with_one_commit(root: Path, name: str) -> Path:
    repo = root / name
    repo.mkdir()
    await git_coord._git(str(repo), "init", "-b", "main")
    (repo / "README.md").write_text("# p\n", encoding="utf-8")
    await git_coord._git(str(repo), "add", "-A")
    await git_coord._git(str(repo), "commit", "-m", "initial")
    return repo


async def _start_and_commit_one_step(repo: Path, task_id: str) -> TaskRun:
    run = TaskRun(spec_path="/s.md", spec_content="s")
    run.task_id = task_id
    run.work_dir = str(repo)
    await git_coord.init_workspace(run)
    (Path(run.work_dir) / "models.py").write_text("class User: pass\n", encoding="utf-8")
    await git_coord.commit_step(run, Step(index=1, title="Create models", description="d"))
    return run


async def _expected_summary(run: TaskRun) -> str:
    """The two blocks a step's context carries, built from git's own log line."""
    line = (await git_coord._git(run.work_dir, "log", "--oneline", "-1")).strip()
    return (
        f"## Git Log (changes so far)\n```\n{line}\n```\n\n"
        "## Files Changed\n```\nmodels.py | 1 +\n 1 file changed, 1 insertion(+)\n```"
    )


class TestDetachedBase:
    @pytest.mark.asyncio
    async def test_detached_start_records_the_start_commit(self, tmp_path: Path) -> None:
        repo = await _repo_with_one_commit(tmp_path, "detached")
        start = (await git_coord._git(str(repo), "rev-parse", "HEAD")).strip()
        await git_coord._git(str(repo), "checkout", "--detach")

        run = await _start_and_commit_one_step(repo, "detached_run")

        assert run.git_enabled is True
        assert run.base_branch == start
        assert len(run.base_branch) == 40

    @pytest.mark.asyncio
    async def test_detached_start_summary_names_the_steps_changes(self, tmp_path: Path) -> None:
        repo = await _repo_with_one_commit(tmp_path, "detached2")
        await git_coord._git(str(repo), "checkout", "--detach")

        run = await _start_and_commit_one_step(repo, "detached_summary")
        summary = await git_coord.get_state_summary(run)

        assert "Git Log (changes so far)" in summary
        assert "step 1: Create models" in summary
        assert "models.py" in summary
        assert summary == await _expected_summary(run)

    @pytest.mark.asyncio
    async def test_branch_start_keeps_the_branch_name(self, tmp_path: Path) -> None:
        repo = await _repo_with_one_commit(tmp_path, "onbranch")

        run = await _start_and_commit_one_step(repo, "branch_run")
        summary = await git_coord.get_state_summary(run)

        assert run.base_branch == "main"
        assert "Git Log (changes so far)" in summary
        assert "step 1: Create models" in summary
        assert "models.py" in summary
        assert summary == await _expected_summary(run)
