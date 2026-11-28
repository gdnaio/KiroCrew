"""An idle chat's saved queue never holds a restart.

The queue is written to the transcript's first line and restored at boot, so a
slot that runs no turn, has every entry on disk and leaves every entry for a
person to send loses nothing to a restart. Counting it kept
``POST /api/restart/when-idle`` at ``idle: false`` for as long as the chat
stayed idle, and the count came back after every restart. A queue the restart
would shorten (a save still owed, an entry past the durable bounds) or hand
back changed (an entry that is not a restored card nobody chose, such as a
paused crew's send) still counts.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kiro_crew import quiescence, work_holders
from kiro_crew.dashboard.handlers import restart_gate
from kiro_crew.dashboard.slot_queue_repository import (
    MAX_DURABLE_QUEUE_BYTES,
    MAX_DURABLE_QUEUE_ENTRIES,
    RESTORED_QUEUE_KEY,
    RESTORED_SEND_CHOSEN_KEY,
)


def _prompts(n: int, *, restored: bool = True) -> list[dict[str, object]]:
    return [
        {
            "id": f"q{i}",
            "content": f"prompt {i}",
            **({RESTORED_QUEUE_KEY: True} if restored else {}),
        }
        for i in range(n)
    ]


def _slot(queue: list[dict[str, object]], **overrides: object) -> SimpleNamespace:
    fields: dict[str, object] = {
        "running": False,
        "queue_persist_pending": False,
        "_queue": queue,
        "durable_source": lambda: list(queue),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _orchestrator(*slots: object) -> SimpleNamespace:
    return SimpleNamespace(
        sessions=SimpleNamespace(),
        dashboard_state=SimpleNamespace(_slots={f"s{i}": s for i, s in enumerate(slots)}),
        _in_flight_work_counts=lambda: (0, 0),
    )


def test_an_idle_slot_with_a_saved_queue_is_quiescent():
    orch = _orchestrator(_slot(_prompts(6)))

    census = quiescence.assess(orch, quiet_window_secs=0.0, now=100.0)

    assert census.queued_messages == 0
    assert census.busy == 0
    assert census.quiescent is True
    assert quiescence.queued_message_count(orch) == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"running": True},
        {"queue_persist_pending": True},
    ],
    ids=["turn-running", "save-owed"],
)
def test_a_queue_behind_a_turn_or_not_yet_saved_still_counts(overrides):
    orch = _orchestrator(_slot(_prompts(3), **overrides))

    census = quiescence.assess(orch, quiet_window_secs=0.0, now=100.0)

    assert census.queued_messages == 3
    assert census.quiescent is False


def test_a_queue_held_by_a_paused_crew_still_counts():
    # Sent while the crew was paused: Resume would send it, and a restart
    # would turn it into restored cards, which changes what Resume sends.
    orch = _orchestrator(_slot(_prompts(2, restored=False)))

    assert quiescence.queued_message_count(orch) == 2


def test_a_restored_card_a_person_chose_still_counts():
    queue = _prompts(2)
    queue[0][RESTORED_SEND_CHOSEN_KEY] = True
    orch = _orchestrator(_slot(queue))

    assert quiescence.queued_message_count(orch) == 2


def test_a_slot_that_cannot_say_whether_it_saved_still_counts():
    slot = SimpleNamespace(running=False, _queue=_prompts(1))

    holders = quiescence.queued_message_holders(_orchestrator(slot))

    assert [(h.kind, h.count) for h in holders] == [(work_holders.QUEUED, 1)]


def test_an_idle_queue_past_the_durable_cap_still_counts():
    over = MAX_DURABLE_QUEUE_ENTRIES + 1
    orch = _orchestrator(_slot(_prompts(over)))

    assert quiescence.queued_message_count(orch) == over


def test_an_idle_queue_past_the_durable_byte_budget_still_counts():
    queue = _prompts(2)
    for entry in queue:
        entry["content"] = "x" * (MAX_DURABLE_QUEUE_BYTES * 3 // 5)
    orch = _orchestrator(_slot(queue))

    assert quiescence.queued_message_count(orch) == 2
    queue.pop()
    assert quiescence.queued_message_count(orch) == 0


def test_a_queue_behind_a_held_peer_steer_still_counts():
    # The turn's end would send the held steer; a restart would bring it back
    # as a restored card, which changes who sends it.
    queue = _prompts(1)
    held = [{"id": "steer", "content": "held peer steer"}]
    slot = _slot(queue, durable_source=lambda: [*held, *queue])

    holders = quiescence.queued_message_holders(_orchestrator(slot))

    assert [(h.kind, h.count) for h in holders] == [(work_holders.QUEUED, 1)]


def _request(probe) -> MagicMock:
    req = MagicMock()
    req.remote = "127.0.0.1"
    req.transport = None
    req.headers = {}
    req.can_read_body = False
    req.json = AsyncMock(return_value=None)
    req.get = {"internal_auth": True}.get
    req.app = {"state": SimpleNamespace(_quiescence_probe=probe)}
    return req


@pytest.mark.asyncio
async def test_the_idle_check_answers_idle_once_only_the_idle_queue_is_left():
    orch = _orchestrator(_slot(_prompts(6)), _slot(_prompts(1), running=True))

    def _probe(**kw: float) -> quiescence.Quiescence:
        return quiescence.assess(orch, **kw)

    resp = await restart_gate.api_restart_when_idle(_request(_probe))
    body = json.loads(resp.body)
    assert (body["idle"], body["background"]) == (False, 1)

    orch.dashboard_state._slots.pop("s1")
    resp = await restart_gate.api_restart_when_idle(_request(_probe))
    body = json.loads(resp.body)
    assert (body["idle"], body["background"]) == (True, 0)
