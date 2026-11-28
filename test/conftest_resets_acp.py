"""Per-test resets of process-global ``kiro_crew.acp`` state.

Part of ``test/conftest.py``, which imports and re-binds these names so pytest registers
each fixture as the suite conftest's own; see the import block there. Plain module,
never collected.
"""

from __future__ import annotations

import sys

from conftest_guards import per_test_guard


@per_test_guard
def _reset_kas_agent_config_error_reports():
    """No agent config error reported yet: ``kas_agent_config_error.report`` warns
    once per report and once at its cap, so leftover state silences the next WARNING."""

    def clear():
        module = sys.modules.get("kiro_crew.acp.kas_agent_config_error")
        if module is not None:
            module._reported.clear()
            module._cap_warned = False

    clear()
    try:
        yield
    finally:
        clear()


@per_test_guard
def _isolate_kas_model_catalog_recording():
    """Start each test with no recorded KAS catalog and no recorded hidden models.

    ``agent_sdk.kas_model_catalog`` keeps the last ``model`` select and the
    account's hidden-model row as module values stamped with a vault identity;
    the bundle-route picker serves whatever they hold. Left set, a test that
    recorded them would mark rows ``hidden_by_account`` in whichever bundle-route
    test runs next on the worker. Only a module already imported can hold them.
    """

    def clear():
        module = sys.modules.get("kiro_crew.agent_sdk.kas_model_catalog")
        if module is not None:
            module._recorded = None
            module._recorded_identity = ""
            module._recorded_hidden = None
            module._recorded_hidden_identity = ""
            module._hidden_models_changed = None
            module._recorded_hidden_seq = 0
            module._hidden_models_seq = 0

    clear()
    try:
        yield
    finally:
        clear()


@per_test_guard
def _reset_kas_secret_storage_unknown_kind_warnings():
    """No unknown engine key kind warned about yet: ``kas_secret_storage`` warns once
    per kind per process, so leftover state silences the next test's WARNING."""

    def clear():
        module = sys.modules.get("kiro_crew.acp.kas_secret_storage")
        if module is not None:
            module._WARNED_UNKNOWN_KINDS.clear()

    clear()
    try:
        yield
    finally:
        clear()


@per_test_guard
def _reset_kas_skill_facts():
    """No skill file's facts kept yet: ``kas_skill_scope._skill_facts`` keeps them
    across session starts by path, so a test's files must not answer a later test's."""

    def clear():
        module = sys.modules.get("kiro_crew.acp.kas_skill_scope")
        if module is not None:
            module._SKILL_FACTS.clear()

    clear()
    try:
        yield
    finally:
        clear()


@per_test_guard
def _isolate_kas_approve_for_me():
    """Drop the approve-for-me owner check and engine answer a test left behind.

    A gateway boot registers an owner check bound to its own ``DashboardState``
    and may cache the engine's answer; both are process state. Only a module
    already imported can hold them, so a run that never imports it pays nothing.
    """
    module = sys.modules.get("kiro_crew.acp.kas_approve_for_me")
    if module is not None:
        module.reset_owner_check()
    yield
    module = sys.modules.get("kiro_crew.acp.kas_approve_for_me")
    if module is not None:
        module.reset_owner_check()


@per_test_guard
def _reset_kas_parent_notes():
    """Start each test with no noted sub-agent run parents.

    Every manager run notes the session that started it, keyed by run key. Tests
    reuse run keys, so a parent one test noted would join the next test's run and
    hold it to an unrelated agent's denies. The directory noted for a parent goes
    with it: it would anchor the next test's limits in a folder of this one.
    """

    def clear():
        module = sys.modules.get("kiro_crew.acp.kas_parent_limits")
        if module is not None:
            module._NOTED.clear()
            module._NOTED_DIRS.clear()
            module._INCOMPLETE.clear()

    clear()
    try:
        yield
    finally:
        clear()
