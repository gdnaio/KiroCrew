"""PPTX Maker: ``HOST_SPELLINGS`` says, in host placeholders, what ``placeholder_values()`` computes.

The release that deletes this wheel copy freezes the table into the pin row
(``adoption_pin.py freeze`` reads it from the source), and with the module gone the
gateway renders a tree adopted at a pin whose agents still name ``{ENGINE_ROOT}`` and the
rest from that row (app-kit-platform.md § 33, "The row carries the copy's own agent
placeholders"). What is pinned here is that the frozen spellings render to exactly the
values this module computes, on this host, for the root an adopted record registers
from: its install directory under the data home.

The ``uv`` resolver, ``PATH`` and the LibreOffice lookup are stubbed, so the comparison
runs identically on every shard whatever the host has installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kiro_crew import soffice_discovery
from kiro_crew.apps import agent_template_values
from kiro_crew.apps.builtin_adoption_baseline import host_spellings, render_spelling
from kiro_crew.apps.builtins.pptx_maker import agent_placeholders
from kiro_crew.apps.builtins.pptx_maker.backend import paths, provision
from kiro_crew.apps.manager import app_dir


@pytest.fixture(params=[None, "LibreOffice"], ids=["no-fixed-root-soffice", "fixed-root-soffice"])
def host(request, tmp_path, monkeypatch):
    """A host with ``uv``, a fixed ``PATH`` and, in one case, a fixed-root ``soffice``."""
    monkeypatch.setattr(provision, "resolve_uv", lambda: str(tmp_path / "uv"))
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    fixed = None if request.param is None else str(tmp_path / request.param / "soffice.exe")
    monkeypatch.setattr(soffice_discovery, "windows_install_path", lambda: fixed)
    return agent_template_values.host_placeholder_values(
        agent_placeholders.APP_NAME, app_dir(agent_placeholders.APP_NAME)
    )


def _computed() -> dict[str, str]:
    """What the module adds to the host set."""
    return {
        name: value
        for name, value in agent_placeholders.placeholder_values().items()
        if name not in agent_template_values.HOST_PLACEHOLDERS
    }


def _spelled(host: dict[str, str]) -> dict[str, str | None]:
    return {
        name: render_spelling(spelling, host)
        for name, spelling in agent_placeholders.HOST_SPELLINGS.items()
    }


def test_every_spelling_renders_the_value_the_module_computes(host):
    paths.preview_tools_bin().mkdir(parents=True)

    assert _spelled(host) == _computed()


def test_the_tools_path_names_the_managed_directory_before_it_exists(host):
    """The one difference: the module appends that directory only once it exists."""
    computed = _computed()
    spelled = _spelled(host)

    assert spelled.pop("{TOOLS_PATH}") == (
        computed.pop("{TOOLS_PATH}") + os.pathsep + str(paths.preview_tools_bin())
    )
    assert spelled == computed


def test_the_freeze_reads_this_table_from_the_source():
    package = Path(agent_placeholders.__file__).parent

    assert host_spellings(package) == agent_placeholders.HOST_SPELLINGS
