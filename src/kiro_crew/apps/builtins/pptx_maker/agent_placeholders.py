"""PPTX Maker's agent-template placeholder values, computed in the gateway.

Read by the core's ``apps.bridges_agents._placeholder_values`` when it renders this
app's shipped agent configs: the engine's MCP server lives under the provisioned engine
root and is launched with ``uv``, neither of which has a fixed location, so the packaged
template names them as placeholders and this module supplies the values. Every value is
computed from the data home and the installed package, never read back from disk.
"""

from __future__ import annotations

from kiro_crew.apps.builtins.pptx_maker.backend import paths, provision
from kiro_crew.apps.manager import app_dir

APP_NAME = "pptx-maker"

#: Each name :func:`placeholder_values` adds to the host set, spelled in host placeholders
#: alone, as this app's own repository spells them in its agent templates. The release
#: that deletes this copy freezes the table into the pin row (``adoption_pin.py freeze``
#: reads it from the source), and the gateway then renders a tree adopted at a pin whose
#: agents still name these from the row. ``tests/test_host_spellings.py`` holds each
#: spelling equal to the value computed below, for the root an adopted record registers
#: from. ``{TOOLS_PATH}`` names the managed tool directory whether or not it exists yet.
HOST_SPELLINGS: dict[str, str] = {
    "{ENGINE_ROOT}": "{APP_DATA}/vendor/sdpm",
    "{ENGINE_MCP_DIR}": "{APP_DATA}/vendor/sdpm/mcp-local",
    "{APP_PROMPTS}": "{APP_ROOT}/prompts",
    "{TOOLS_PATH}": "{HOST_PATH}{PATH_SEP}{APP_DATA}/vendor/preview-tools/bin",
}


def placeholder_values() -> dict[str, str]:
    """``{"{PLACEHOLDER}": value}``, or ``{}`` while ``uv`` cannot be resolved."""
    uv_bin = provision.resolve_uv()
    if not uv_bin:
        return {}
    return {
        "{UV_BIN}": uv_bin,
        "{ENGINE_ROOT}": str(paths.engine_root()),
        "{ENGINE_MCP_DIR}": str(paths.engine_mcp_dir()),
        "{APP_PROMPTS}": str(app_dir(APP_NAME) / "prompts"),
        # The engine invokes `pdftoppm`/`soffice` BY NAME from inside its MCP
        # server, which kiro-cli spawns from this config — not from any gateway
        # subprocess. So the app's managed tool dir has to be on the PATH declared
        # here or the managed install is invisible at the one moment it matters.
        # Appended, so a real system poppler still wins the engine's own lookup.
        "{TOOLS_PATH}": provision.mcp_tools_path(),
    }
