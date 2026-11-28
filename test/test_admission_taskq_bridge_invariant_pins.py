"""Mutation pins for the durable-queue bridge lines in spawn admission that carried no test.

These lines in ``kiro_crew.subagent_manager.admission`` (the one spelling of "a run
exists", ``taskq_bridge_reads._live_run_ids``, and the refill of the bounded
dispatch window over the task store's queued rows, ``taskq_bridge_window``) carry
invariants that no other test observes: delete any one of them and every test that
covers the bridge stays green. This file pins them, because their silent removal
hydrates a second copy of a run that is still resident, places one row in the window
twice, or parks the pump for as long as a far deferral lasts. The window-entry guard
that keeps a new spawn behind an older row waiting on disk (``taskq_should_window``
and its async twin) is left to the queue-depth reconcile tests, which already red its
deletion in both copies.

``AUTOSDE.yaml``'s ``an-invariant-line-needs-a-mutation-pin`` requires each pin to
red under three mutations, not one:

1. the line DELETED,
2. the line LOOSENED to the next-weakest thing a reader might write,
3. the HEALTHY side asserted -- under a satisfied fence the entry joins the window,
   or the wake arms at its real time.

Every class below carries all three. Each is verified by applying the mutations its
docstring names to its line and observing this file red; the counts are in the PR
description.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Iterator
from unittest.mock import MagicMock

import pytest
from overload_fakes import Clock, ManagerHarness, task_record

from kiro_crew.subagent import SubagentInfo
from kiro_crew.subagent_manager.admission import FairnessSettings, SpawnAdmissionCoordinator
from kiro_crew.subagent_manager.admission.types import MIN_RECHECK_DELAY_SECS
from kiro_crew.taskq import KIND_SUBAGENT, model

pytestmark = [pytest.mark.usefixtures("healthy_host_memory"), pytest.mark.timeout(30)]

T0 = 1_000_000.0
PARENT = "dash:pins"


@pytest.fixture
def h(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., Any]]:
    """A factory for a real manager with its task store open; each is closed after the test."""
    import kiro_crew.subagent as subagent_mod

    monkeypatch.setattr(subagent_mod, "Stats", MagicMock())
    monkeypatch.setattr(subagent_mod, "sel", MagicMock())
    made: list[ManagerHarness] = []

    async def make(max_concurrent: int = 1, window: int = 4) -> ManagerHarness:
        hz = ManagerHarness(max_concurrent, FairnessSettings())
        made.append(hz)
        await hz.mgr.wait_taskq_ready()
        hz.store._window = window
        return hz

    try:
        yield make
    finally:
        for hz in made:
            hz.close()
            hz.mgr.close()


def _window(hz: ManagerHarness) -> list[str]:
    return [str(p.get("_preassigned_id") or "") for p in hz.mgr._queue]


async def _store_only_head(hz: ManagerHarness) -> tuple[SubagentInfo, SubagentInfo]:
    """A running blocker, and the oldest waiting row back on disk only.

    The row joins the window when it is spawned (nothing older waits), and the
    eviction that makes room for another lane drops it back to store-only, the
    way ``_evict_for_lanes`` gives up a lane head.
    """
    blocker = hz.spawn("blocker", parent=PARENT)
    await hz.settle()
    assert hz.state(blocker) == model.RUNNING
    older = hz.spawn("older", parent=PARENT)
    assert _window(hz) == [older.id], "the first waiting spawn joins the window"
    assert hz.mgr._admission._evict_for_lanes(1) == 1
    assert _window(hz) == [] and hz.state(older) == model.QUEUED
    return blocker, older


class TestAResidentRunsRowIsNeverHydratedAgain:
    """``_live_run_ids``'s ``[aid for aid, info in ... if not info.done]``.

    A live run's row can be claimable for a moment (a wake lands in ``retry_wait``
    until the pump grants the slot back). The refill's
    exclusions and the queued listing leave it out because the registry answers
    for it; hydrating it would start a second copy of a run that is resident.

    - deleted (``return []``): the refill hydrates the resident run's row into the
      window, and the queued listing reports the running run as waiting;
    - loosened (drop the ``not info.done`` filter, every registered id counts as
      live): a finished run whose record is still registered keeps its id in
      every exclusion set. A retry or a continuation runs under a new id, so no
      later row needs that id back; what the loosening breaks is the set itself,
      which then grows with every retained record (``test_taskq_refusal_settle.py``
      asserts the same for a refused row whose settle has landed);
    - healthy: under the same refill, a store-only row with no run joins the
      window, and once the resident run finishes its id leaves the exclusions.
    """

    @staticmethod
    async def _resident_with_claimable_row(hz: ManagerHarness) -> tuple[SubagentInfo, SubagentInfo]:
        live = hz.spawn("resident", parent="dash:resident")
        await hz.settle()
        assert hz.state(live) == model.RUNNING
        other = hz.spawn("other", parent=PARENT)
        assert hz.mgr._admission._evict_for_lanes(1) == 1
        assert _window(hz) == [] and hz.state(other) == model.QUEUED
        row = hz.store.get(live.id)
        assert row is not None
        assert hz.store.transition(
            live.id, model.RETRY_WAIT, generation=row.generation, next_run_at=None
        )
        assert not hz.live(live).done
        return live, other

    @pytest.mark.asyncio
    async def test_the_refill_never_hydrates_a_resident_runs_row(
        self, h: Callable[..., Any]
    ) -> None:
        hz = await h(max_concurrent=1)
        live, other = await self._resident_with_claimable_row(hz)

        assert hz.mgr._admission.taskq_refill_window() == 1
        assert _window(hz) == [other.id]
        assert hz.mgr._admission.taskq_overflow() == 0, "the chip counts no waiting row"

    @pytest.mark.asyncio
    async def test_the_queued_listing_never_reports_a_resident_run(
        self, h: Callable[..., Any]
    ) -> None:
        hz = await h(max_concurrent=1)
        live, other = await self._resident_with_claimable_row(hz)

        listing = await hz.mgr._admission.taskq_queued_runs_async(PARENT)
        everyone = await hz.mgr._admission.taskq_queued_runs_async()
        assert [run.id for run in listing.runs] == [other.id]
        assert [run.id for run in everyone.runs] == [other.id]

    @pytest.mark.asyncio
    async def test_the_healthy_side_hydrates_a_row_with_no_run(self, h: Callable[..., Any]) -> None:
        hz = await h(max_concurrent=1)
        live = hz.spawn("resident", parent="dash:resident")
        await hz.settle()
        other = hz.spawn("other", parent=PARENT)
        assert hz.mgr._admission._evict_for_lanes(1) == 1
        assert hz.mgr._admission.taskq_overflow() == 1

        assert hz.mgr._admission.taskq_refill_window() == 1
        assert _window(hz) == [other.id]
        assert live.id in hz.mgr._admission.taskq_excluded_ids()
        await hz.end(hz.live(live))
        assert hz.started == [live.id, other.id]
        # The finished record stays registered, but it is not a live run, so
        # the id leaves every exclusion set.
        assert hz.mgr._agents[live.id].done
        assert live.id not in hz.mgr._admission.taskq_excluded_ids()
        assert live.id not in hz.mgr._admission.taskq_dispatch_excluded_ids()


class TestARowAConcurrentStepPlacedJoinsTheWindowOnce:
    """``_refill_apply``'s ``if rec.id not in present:``.

    The async refill fetches on the writer thread and applies on the loop; a step
    that runs on the loop between the two (another refill pass) may already have
    placed a fetched row in the window. ``present`` is read when the apply starts,
    so that row is skipped.

    - deleted (append unconditionally): the row is in the window twice;
    - loosened (``present`` starts empty and only this pass's appends are added):
      a row placed before the pass is still placed again;
    - healthy: a fetched row that no step placed joins the window, once.
    """

    @staticmethod
    def _interpose_a_refill(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
        """Run one sync refill pass on the loop between the async pass's fetch and apply."""
        real = SpawnAdmissionCoordinator._report_retired_stage_rows
        ran = {"passes": 0}

        def _report(self: Any, store: Any, retired: Any, stranded: bool) -> None:
            real(self, store, retired, stranded)
            if ran["passes"] == 0:
                ran["passes"] += 1
                assert self.taskq_refill_window() == 1

        monkeypatch.setattr(SpawnAdmissionCoordinator, "_report_retired_stage_rows", _report)
        return ran

    @pytest.mark.asyncio
    async def test_a_row_placed_mid_pass_is_not_placed_again(
        self, h: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hz = await h()
        _blocker, older = await _store_only_head(hz)
        ran = self._interpose_a_refill(monkeypatch)

        await hz.mgr._admission.taskq_refill_window_async()
        assert ran["passes"] == 1
        assert _window(hz) == [older.id]

    @pytest.mark.asyncio
    async def test_the_healthy_side_places_a_fetched_row_once(self, h: Callable[..., Any]) -> None:
        hz = await h()
        _blocker, older = await _store_only_head(hz)

        assert await hz.mgr._admission.taskq_refill_window_async() == 1
        assert _window(hz) == [older.id]
        assert await hz.mgr._admission.taskq_refill_window_async() == 0
        assert _window(hz) == [older.id]


class TestAnIdleWakeIsCappedAtTheAdmitWait:
    """``_refill_schedule_wake``'s ``min(due_in, self.taskq_admit_wait_secs())``.

    When a refill finds nothing claimable and the window is empty, the pump arms
    its own wake for the earliest deferred row. ``due_in`` is measured on the
    STORE's clock and handed to the LOOP's ``call_later``; the cap bounds any one
    sleep at the admit wait, so a far deferral (or a store clock that has jumped)
    never parks the pump for the whole distance with nothing re-reading the store.
    The floor half is held by ``test_queue_depth_reconcile_accepts.py``'s
    ``test_an_overdue_wake_is_never_rearmed_at_zero``.

    - deleted (``max(floor, due_in)``): a row deferred far ahead arms its full
      distance;
    - loosened (``min(due_in, 10 * admit_wait)``): the cap is ten admit waits;
    - healthy: a row due sooner than the admit wait arms at its real time, not at
      the cap and not at the floor.
    """

    @staticmethod
    async def _armed_delays(
        hz: ManagerHarness, monkeypatch: pytest.MonkeyPatch, wait: float
    ) -> list[float]:
        clock = Clock(T0)
        monkeypatch.setattr(hz.store, "_clock", clock)
        hz.store.accept_one(task_record("deferred", session_key=PARENT))
        assert hz.store.defer("deferred", wait=wait, reason="test")
        loop = asyncio.get_running_loop()
        delays: list[float] = []

        def _call_later(delay: float, callback: Any, *a: Any, **k: Any) -> None:
            if callback == hz.mgr._drain_queue:
                delays.append(delay)

        # Scoped: the loop's own ``call_later`` is back before teardown runs.
        with monkeypatch.context() as timers:
            timers.setattr(loop, "call_later", _call_later)
            assert hz.mgr._admission.taskq_refill_window() == 0
        assert hz.store.next_eligible_at(KIND_SUBAGENT) == T0 + wait
        return delays

    @pytest.mark.asyncio
    async def test_a_far_deferral_arms_no_later_than_the_admit_wait(
        self, h: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hz = await h()
        admit_wait = hz.mgr._admission.taskq_admit_wait_secs()
        assert admit_wait > MIN_RECHECK_DELAY_SECS

        delays = await self._armed_delays(hz, monkeypatch, wait=100 * admit_wait)
        assert delays == [admit_wait]

    @pytest.mark.asyncio
    async def test_the_healthy_side_arms_a_near_wake_at_its_real_time(
        self, h: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hz = await h()
        admit_wait = hz.mgr._admission.taskq_admit_wait_secs()
        due_in = admit_wait / 6
        assert MIN_RECHECK_DELAY_SECS < due_in < admit_wait

        delays = await self._armed_delays(hz, monkeypatch, wait=due_in)
        assert delays == [due_in]
