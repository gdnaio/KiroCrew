"""A deleted wheel copy's own agent placeholders, carried by its pin row (app-kit-platform.md § 33).

A built-in may ship an ``agent_placeholders`` module computing agent-template names
beyond the host set, and a tree adopted at an earlier pin can still name them. The
release that deletes the copy freezes the module's ``HOST_SPELLINGS`` table into the
row's ``deletedCopy`` (``builtin_adoption_baseline.freeze_deleted_copy``), the loader
holds every spelling to host placeholders and relative segments, and once the copy is
gone the gateway renders those names through the host set for a record adopted under
pin trust, and for nothing else (``no_copy_placeholder_values``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from builtin_adoption_helpers import (
    NAME,
    PACKAGE,
    REPOSITORY,
    VERSION,
    install_pins,
    pin_row,
    repository_manifest,
    seed_agent,
    shipped_manifest,
    write_tree,
)

from kiro_crew.apps import agent_template_values, bridges
from kiro_crew.apps import builtin_adoption_baseline as baseline
from kiro_crew.apps import builtin_adoption_pins as pins_mod
from kiro_crew.apps import execution as exec_mod
from kiro_crew.apps import manager

KEY = pins_mod.PLACEHOLDER_SPELLINGS_KEY
SPELLINGS = {
    "{LEGACY_ROOT}": "{APP_DATA}/vendor/engine",
    "{LEGACY_PATH}": "{HOST_PATH}{PATH_SEP}{APP_DATA}/vendor/tools/bin",
}
TEMPLATE = {
    "name": "legacy",
    "mcpServers": {
        "engine": {
            "command": "python",
            "args": ["{LEGACY_ROOT}/server.py"],
            "env": {"PATH": "{LEGACY_PATH}"},
        }
    },
    "resources": ["file://{APP_ROOT}/prompts/guide.md"],
}


@pytest.fixture(autouse=True)
def _fresh_reads():
    """Each test reads its own pin table and wheel scan, and leaves both empty."""
    pins_mod.load_adoption_pins.cache_clear()
    exec_mod.invalidate_shipped_builtin_apps()
    yield
    pins_mod.load_adoption_pins.cache_clear()
    exec_mod.invalidate_shipped_builtin_apps()


def _row(spellings: object = None) -> dict:
    """The fixture app's row with a deleted copy, carrying *spellings* when given."""
    frozen: dict = {"permissions": {}, "grants": {}}
    if spellings is not None:
        frozen[KEY] = spellings
    return pin_row("a" * 40, "b" * 64, "c" * 64, deletedCopy=frozen)


# --------------------------------------------------------------------------- #
# The row shape
# --------------------------------------------------------------------------- #


def test_the_restated_host_set_is_the_host_set():
    """The light loader restates the host placeholders; it must never drift from them."""
    assert pins_mod.HOST_PLACEHOLDER_NAMES == frozenset(agent_template_values.HOST_PLACEHOLDERS)


def test_a_row_carrying_spellings_loads_them():
    pin = pins_mod.parse_adoption_pins({"schemaVersion": 1, "pins": [_row(SPELLINGS)]})[NAME]
    assert pins_mod.frozen_limit_input(pin, KEY) == SPELLINGS


@pytest.mark.parametrize(
    "spelling",
    [
        "{APP_DATA}",
        "{APP_ROOT}/prompts",
        "{APP_DATA}/vendor/sdpm/mcp-local",
        "{UV_BIN}",
        "{HOST_PATH}{PATH_SEP}{APP_DATA}/vendor/preview-tools/bin",
        "{APP_DATA}/a.b_c-d",
    ],
)
def test_a_spelling_in_host_placeholders_and_relative_segments_is_accepted(spelling):
    assert pins_mod.placeholder_spellings_error({"{LEGACY}": spelling}) == ""


@pytest.mark.parametrize(
    "spellings",
    [
        pytest.param({}, id="empty"),
        pytest.param(["{APP_DATA}"], id="not-an-object"),
        pytest.param({"{APP_DATA}": "{APP_ROOT}/x"}, id="key-in-the-host-set"),
        pytest.param({"LEGACY": "{APP_DATA}"}, id="key-not-a-placeholder"),
        pytest.param({"{LEGACY}": 1}, id="value-not-a-string"),
        pytest.param({"{LEGACY}": ""}, id="value-empty"),
        pytest.param({"{LEGACY}": "{ENGINE_ROOT}/x"}, id="other-placeholder"),
        pytest.param({"{LEGACY}": "{APP_DATA}/../x"}, id="parent-segment"),
        pytest.param({"{LEGACY}": "{APP_DATA}/a..b"}, id="double-dot"),
        pytest.param({"{LEGACY}": "{APP_DATA}/.hidden"}, id="dot-segment"),
        pytest.param({"{LEGACY}": "/usr/bin"}, id="absolute"),
        pytest.param({"{LEGACY}": "C:\\tools"}, id="absolute-windows"),
        pytest.param({"{LEGACY}": "{HOST_PATH}{PATH_SEP}/usr/bin"}, id="absolute-entry"),
        pytest.param({"{LEGACY}": "{APP_DATA}\\x"}, id="backslash"),
        pytest.param({"{LEGACY}": "{APP_DATA}x"}, id="glued-literal"),
        pytest.param({"{LEGACY}": "{APP_DATA}/"}, id="empty-segment"),
        pytest.param({"{LEGACY}": "{APP_DATA}//x"}, id="double-slash"),
        pytest.param({"{LEGACY}": "{HOST_PATH}/x"}, id="path-under-a-whole-placeholder"),
        pytest.param({"{LEGACY}": "{PATH_SEP}"}, id="empty-entries"),
        pytest.param({"{LEGACY}": "{APP_DATA}{PATH_SEP}"}, id="trailing-separator"),
    ],
)
def test_any_other_spelling_drops_the_row(spellings):
    assert pins_mod.placeholder_spellings_error(spellings) != ""
    assert pins_mod.parse_adoption_pins({"schemaVersion": 1, "pins": [_row(spellings)]}) == {}


# --------------------------------------------------------------------------- #
# The freeze: the module's table, read from the source
# --------------------------------------------------------------------------- #


def _copy(tmp_path: Path, module: str | None) -> Path:
    """A wheel copy of the fixture app, with *module* as its ``agent_placeholders.py``."""
    copy = write_tree(tmp_path / PACKAGE, shipped_manifest())
    if module is not None:
        (copy / baseline.PLACEHOLDERS_MODULE).write_text(module, encoding="utf-8")
    return copy


@pytest.mark.parametrize(
    "module",
    [
        f"HOST_SPELLINGS: dict[str, str] = {json.dumps(SPELLINGS)}\n",
        f"import os\n\nHOST_SPELLINGS = {json.dumps(SPELLINGS)}\n\ndef placeholder_values():\n"
        "    return {}\n",
    ],
    ids=["annotated", "plain"],
)
def test_a_copy_with_a_placeholder_module_freezes_its_table(tmp_path, module):
    frozen = baseline.freeze_deleted_copy(_copy(tmp_path, module))
    assert frozen[KEY] == SPELLINGS
    assert pins_mod.deleted_copy_error(frozen) == ""


def test_a_copy_without_a_placeholder_module_freezes_no_spellings(tmp_path):
    assert KEY not in baseline.freeze_deleted_copy(_copy(tmp_path, None))


@pytest.mark.parametrize(
    "module",
    [
        "def placeholder_values():\n    return {}\n",
        f"HOST_SPELLINGS = dict({json.dumps(SPELLINGS)})\n",
        'HOST_SPELLINGS = {"{APP_DATA}": "{APP_ROOT}"}\n',
        'HOST_SPELLINGS = {"{LEGACY}": "/opt/engine"}\n',
    ],
    ids=["no-table", "not-a-literal", "host-set-key", "absolute-value"],
)
def test_a_placeholder_module_whose_table_cannot_be_frozen_refuses_the_freeze(tmp_path, module):
    """A copy whose names a pin row could not render is never frozen without them."""
    with pytest.raises(ValueError):
        baseline.freeze_deleted_copy(_copy(tmp_path, module))


def test_a_spelling_renders_with_this_hosts_separators():
    host = {"{APP_DATA}": "D", "{HOST_PATH}": "P", "{PATH_SEP}": os.pathsep}
    assert baseline.render_spelling("{APP_DATA}/vendor/engine", host) == os.path.join(
        "D", "vendor", "engine"
    )
    assert baseline.render_spelling(SPELLINGS["{LEGACY_PATH}"], host) == (
        "P" + os.pathsep + os.path.join("D", "vendor", "tools", "bin")
    )
    assert baseline.render_spelling("{APP_ROOT}/prompts", host) is None


# --------------------------------------------------------------------------- #
# The reader: an adopted record's agents, with the copy gone
# --------------------------------------------------------------------------- #


@pytest.fixture
def host_path(tmp_path, monkeypatch) -> str:
    """A fixed ``{HOST_PATH}``, so the rendered ``PATH`` does not depend on this host."""
    value = str(tmp_path / "bin")
    monkeypatch.setattr(agent_template_values, "_host_path", lambda: value)
    return value


def _adopted(*, trusted: bool = True) -> Path:
    """The fixture app's record adopted from its pinned repository; its root."""
    root = manager.app_dir(NAME)
    write_tree(root, repository_manifest())
    manager._write_installed(
        NAME,
        manager.InstalledApp(
            name=NAME,
            version=VERSION,
            displayName="Fixture Former",
            source=f"registry:{NAME}",
            origin="registry",
            lifecycle="gateway",
            sourceUrl=REPOSITORY,
            sourceCommit="a" * 40,
        ),
    )
    if trusted:
        seed_agent(
            Path(os.environ["KIROCREW_HOME"]),
            {"apps_trusted": [NAME], "apps_trusted_repositories": {NAME: REPOSITORY}},
        )
    return root


def _render(root: Path, template: dict) -> dict | None:
    """Render *template* as the adopted record's one agent; the written config, or None."""
    agent = root / "agents" / "legacy.json"
    agent.parent.mkdir(parents=True, exist_ok=True)
    agent.write_text(json.dumps(template), encoding="utf-8")
    out = bridges._render_shipped_agent(NAME, agent, app_root=root)
    return None if out is None else json.loads(out.read_text(encoding="utf-8"))


def test_an_adopted_tree_naming_the_deleted_copys_placeholders_renders(
    tmp_path, monkeypatch, host_path
):
    """The repro: the package is gone, and the adopted agents still register."""
    install_pins(monkeypatch, tmp_path / "adoption-pins.json", [_row(SPELLINGS)])
    root = _adopted()

    rendered = _render(root, TEMPLATE)

    assert rendered is not None
    engine = rendered["mcpServers"]["engine"]
    assert Path(engine["args"][0]) == root / "data" / "vendor" / "engine" / "server.py"
    assert engine["env"]["PATH"] == (
        host_path + os.pathsep + str(root / "data" / "vendor" / "tools" / "bin")
    )
    assert Path(rendered["resources"][0].removeprefix("file://")) == (root / "prompts" / "guide.md")


def test_a_record_not_adopted_under_pin_trust_registers_nothing(tmp_path, monkeypatch, host_path):
    install_pins(monkeypatch, tmp_path / "adoption-pins.json", [_row(SPELLINGS)])
    root = _adopted(trusted=False)

    assert _render(root, TEMPLATE) is None


def test_spellings_beside_a_shipped_copy_are_never_read(tmp_path, monkeypatch, host_path):
    """One source of truth: while the copy ships, its module answers, never the row."""
    install_pins(monkeypatch, tmp_path / "adoption-pins.json", [_row(SPELLINGS)])
    monkeypatch.setattr(baseline, "wheel_copy_ships", lambda name: True)
    root = _adopted()

    assert _render(root, TEMPLATE) is None


def test_a_row_that_froze_no_spellings_registers_nothing(tmp_path, monkeypatch, host_path):
    install_pins(monkeypatch, tmp_path / "adoption-pins.json", [_row()])
    root = _adopted()

    assert _render(root, TEMPLATE) is None


def test_a_name_the_row_does_not_spell_stays_unregistered(tmp_path, monkeypatch, host_path):
    install_pins(monkeypatch, tmp_path / "adoption-pins.json", [_row(SPELLINGS)])
    root = _adopted()

    assert _render(root, {**TEMPLATE, "prompt": "file://{UNSPELLED}/x.md"}) is None


def test_a_spelling_whose_host_value_is_missing_is_left_out(tmp_path, monkeypatch):
    """No ``{HOST_PATH}`` here: the name spelled through it fails closed, the others render."""
    monkeypatch.setattr(agent_template_values, "_host_path", lambda: "")
    install_pins(monkeypatch, tmp_path / "adoption-pins.json", [_row(SPELLINGS)])
    root = _adopted()

    assert _render(root, TEMPLATE) is None
    rendered = _render(root, {"name": "legacy", "prompt": "file://{LEGACY_ROOT}/x.md"})
    assert rendered is not None
    assert Path(rendered["prompt"].removeprefix("file://")) == (
        root / "data" / "vendor" / "engine" / "x.md"
    )
