"""An agent spec's own PreToolUse guard on a workflow agent step.

On a backend where Kiro Crew, not the harness, fires the agent spec's ``hooks`` (KAS),
a workflow ``ctx.agent()`` step runs its agent with nobody watching, through
``stream_and_collect`` under ``AUTO_APPROVE``. Both executors (``agent_exec`` and the
pool's ``_run_step``) hand it ``channel_spec_hooks.SessionSpecGate``, so for a step that
runs on a session of its own, a guard that exits 2 stops the call before the step's
automatic approval.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import kiro_crew.dashboard.handlers.usage as usage_mod
import kiro_crew.hooks as hooks_mod
from kiro_crew.acp.types import AcpEvent
from kiro_crew.agent_sdk import spec_hooks
from kiro_crew.agent_sdk.backends import ACP_BACKEND_KAS, ACP_BACKEND_KIRO
from kiro_crew.agent_sdk.capabilities import capabilities_for
from kiro_crew.hooks import HOOK_EVENT_PRE_TOOL_USE, ScriptHook, ScriptHookResult
from kiro_crew.providers.base import EVENT_COMPLETE, EVENT_PERMISSION_REQUEST
from kiro_crew.sel import SecurityEventLog
from kiro_crew.workflows.agent_exec import build_agent_fn
from kiro_crew.workflows.agent_pool import _run_step


class _Session:
    """A step's session on *backend* whose agent asks to run one shell command.

    It reports its backend's capabilities, as a dedicated session's ``AcpProvider`` does.
    """

    def __init__(self, backend: str) -> None:
        self.capabilities = capabilities_for(backend)
        self._agent = "a1"
        self.session_id = "acp-1"
        self.cwd = "/w"
        self.approved: list[str] = []
        self.rejected: list[str] = []

    async def stream(self, message, **_kw):
        yield AcpEvent(kind=EVENT_PERMISSION_REQUEST, request_id="rq1", title="Running: echo hi")
        yield AcpEvent(kind=EVENT_COMPLETE, stop_reason="end_turn")

    async def approve_tool(self, request_id, **_kw):
        self.approved.append(request_id)
        return True

    async def reject_tool(self, request_id):
        self.rejected.append(request_id)


class _Sessions:
    """A session manager double that hands out one session."""

    def __init__(self, session: _Session) -> None:
        self.get_or_create = AsyncMock(return_value=(session, False, False))
        self.release = MagicMock()


@pytest.fixture
def guard(monkeypatch) -> dict:
    """Agent ``a1``'s spec: one PreToolUse guard on every tool; ``exit`` is its exit code.

    The spec read always returns the guard, so whether Crew asks it on a backend is
    decided by the code under test, never by this fixture.
    """
    record: dict = {"exit": 2, "runs": [], "agents": []}

    async def fake_turn_spec_hooks(provider, agent, **_kw):
        record["agents"].append(agent)
        hook = ScriptHook(
            id="spec:a1:preToolUse:0",
            name="guard.sh",
            event=HOOK_EVENT_PRE_TOOL_USE,
            matcher="*",
            command="guard.sh",
            spec_origin="user",
        )
        return spec_hooks.TurnSpecHooks([hook], "/w", False, True)

    async def fake_run(hook, context="", hook_event=None, cwd=None, **_kw):
        record["runs"].append((hook.command, (hook_event or {}).get("session_key")))
        return ScriptHookResult(
            hook_id=hook.id, hook_name=hook.name, event=hook.event, exit_code=record["exit"]
        )

    monkeypatch.setattr(spec_hooks, "turn_spec_hooks", fake_turn_spec_hooks)
    monkeypatch.setattr(hooks_mod, "run_script_hook", fake_run)
    # No Hooks-page store in this process: only the spec's own guard may run.
    monkeypatch.setattr(hooks_mod, "get_global_hook_store", lambda: None)
    # A step writes its usage row; keep it off disk.
    monkeypatch.setattr(usage_mod, "_write_token_record", lambda record, now: None)
    return record


@pytest.fixture
def spec_hook_denials(tmp_path, monkeypatch):
    """A synchronous security log under ``tmp_path``; returns its spec-hook denials."""
    monkeypatch.setattr(SecurityEventLog, "_instance", None)
    log = SecurityEventLog(base_dir=tmp_path / "sel", sync=True)
    assert log is SecurityEventLog()

    def _rows() -> list[tuple[str, str]]:
        path = tmp_path / "sel" / "security_events.jsonl"
        if not path.exists():
            return []
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return [
            (e["outcome"], e["request_id"])
            for e in events
            if e.get("event_type") == "tool_invocation"
            and (e.get("metadata") or {}).get("mechanism") == "spec_hook"
        ]

    return _rows


async def _own_session_step(session: _Session) -> str:
    fn = build_agent_fn(_Sessions(session), run_id="wf1", default_agent="a1")
    await fn("do work", {})
    return "wf:wf1:0"


async def _pooled_step(session: _Session) -> str:
    await _run_step(session, "do work", key="wf-pool:wf1:0")
    return "wf-pool:wf1:0"


@pytest.mark.parametrize("step", [_own_session_step, _pooled_step], ids=["own-session", "pool"])
@pytest.mark.parametrize(
    ("backend", "exit_code", "refused", "runs"),
    [
        (ACP_BACKEND_KAS, 2, True, 1),
        (ACP_BACKEND_KAS, 0, False, 1),
        (ACP_BACKEND_KIRO, 2, False, 0),
    ],
)
def test_a_workflow_step_asks_its_agents_spec_guard(
    guard, spec_hook_denials, step, backend, exit_code, refused, runs
):
    guard["exit"] = exit_code
    session = _Session(backend)
    key = asyncio.run(step(session))
    # Exit 2 refuses the call before the step's automatic approval; exit 0 lets it
    # run as before; kiro-cli runs the spec's hooks itself, so Crew never asks them.
    assert (session.rejected, session.approved) == ((["rq1"], []) if refused else ([], ["rq1"]))
    assert guard["runs"] == [("guard.sh", key)] * runs
    assert guard["agents"] == ["a1"] * runs
    assert spec_hook_denials() == [("denied", "rq1")] * refused
