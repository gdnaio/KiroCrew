"""The KAS Node watch script on fixture data, and one run the way the workflow runs it.

Fixture data only: no network and no `gh`. Each fixture case writes a pin file, an
index.json and an existing-titles file under `tmp_path`, then reads what the
script printed and whether it wrote a body. The fixture cases call the script's
`main(argv)` in process; only the real-pin-file case runs it as a subprocess,
the way `kas-node-watch.yml` runs it. The last test reads that workflow's
`concurrency` block.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from skill_script_helpers import load_skill_script

from kiro_crew.acp.kas_pin import KAS_PIN_VERSION
from kiro_crew.testing.wait import FALLBACK_TIMEOUT_SECS

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / ".github" / "scripts" / "kas_node_watch.py"
_REAL_PIN = _REPO_ROOT / "src" / "kiro_crew" / "acp" / "kas_pin.py"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "kas-node-watch.yml"

_WATCH = load_skill_script("kas_node_watch_under_test", _SCRIPT)

_PIN_SOURCE = '"""Fixture pin."""\n\nKAS_PIN_VERSION = "0.66.117-node22.23.4"\n'
_BEHIND_TITLE = "KAS pin bundles Node 22.23.4; Node 22.23.5 is out"


def _release(version: str) -> dict[str, object]:
    return {"version": version, "date": "2026-11-11", "lts": "Jod", "security": True}


def _args(
    tmp_path: Path,
    index: list[dict[str, object]],
    existing: str = "",
    pin_file: Path | None = None,
) -> tuple[list[str], Path]:
    if pin_file is None:
        pin_file = tmp_path / "kas_pin.py"
        pin_file.write_text(_PIN_SOURCE, encoding="utf-8")
    index_file = tmp_path / "index.json"
    index_file.write_text(json.dumps(index), encoding="utf-8")
    existing_file = tmp_path / "existing.txt"
    existing_file.write_text(existing, encoding="utf-8")
    body = tmp_path / "body.md"
    argv = [
        "--pin-file",
        str(pin_file),
        "--index",
        str(index_file),
        "--existing",
        str(existing_file),
        "--body-out",
        str(body),
    ]
    return argv, body


def _run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    index: list[dict[str, object]],
    existing: str = "",
) -> tuple[int, str, str, Path]:
    argv, body = _args(tmp_path, index, existing)
    code = _WATCH.main(argv)
    out, err = capsys.readouterr()
    return code, out, err, body


def test_a_pin_behind_the_newest_of_its_major_files_one_issue(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err, body = _run(tmp_path, capsys, [_release("v22.23.5"), _release("v22.23.4")])

    assert code == 0, err
    assert out == _BEHIND_TITLE + "\n"
    text = body.read_text(encoding="utf-8")
    assert "22.23.4" in text and "22.23.5" in text
    assert "`KAS_PIN_VERSION` in `src/kiro_crew/acp/kas_pin.py`" in text
    assert '"Node patch release" row' in text and "`packaging/kas-al2/README.md`" in text
    assert "closes this issue itself once the pin is current" in text
    assert "until the next 22.x release" in text
    assert text.rstrip().endswith("<sub>Filed by `.github/workflows/kas-node-watch.yml`.</sub>")
    assert "security" not in text.lower()
    assert "@" not in text


def test_a_pin_on_the_newest_release_files_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err, body = _run(tmp_path, capsys, [_release("v22.23.4"), _release("v22.23.3")])

    assert code == 0, err
    assert out == ""
    assert not body.exists()


def test_a_title_already_filed_open_or_closed_is_named_but_not_filed_again(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The workflow keeps an open issue with this title and closes the others.
    existing = f"Some other issue\n{_BEHIND_TITLE}\n"
    code, out, err, body = _run(
        tmp_path, capsys, [_release("v22.23.5"), _release("v22.23.4")], existing
    )

    assert code == 0, err
    assert out == _BEHIND_TITLE + "\n"
    assert not body.exists()


def test_only_the_pins_major_counts_whatever_the_index_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `22.9.0` sorts after `22.23.6` as a string and the true newest is not first.
    index = [
        _release("v27.0.0-alpha"),
        _release("v24.11.1"),
        _release("v22.9.0"),
        _release("v22.23.4"),
        _release("v22.23.6"),
        _release("v20.19.9"),
    ]

    code, out, err, body = _run(tmp_path, capsys, index)

    assert code == 0, err
    assert out == "KAS pin bundles Node 22.23.4; Node 22.23.6 is out\n"
    assert "24.11.1" not in body.read_text(encoding="utf-8")


def test_a_malformed_index_is_an_error_not_a_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err, body = _run(tmp_path, capsys, [{"date": "2026-11-11"}])

    assert code != 0
    assert out == ""
    assert err.startswith("kas_node_watch: ")
    assert not body.exists()


def test_the_real_pin_file_parses_to_its_bundled_node(tmp_path: Path) -> None:
    # The module's one real interpreter child: the script run as the workflow runs it.
    pinned = KAS_PIN_VERSION.split("-node", 1)[1]
    major = pinned.split(".", 1)[0]
    newer = f"v{major}.999.0"
    argv, body = _args(tmp_path, [_release(newer), _release(f"v{pinned}")], pin_file=_REAL_PIN)

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), *argv],
        capture_output=True,
        encoding="utf-8",
        timeout=FALLBACK_TIMEOUT_SECS,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"KAS pin bundles Node {pinned}; Node {newer[1:]} is out\n"
    assert body.exists()


def test_runs_go_one_at_a_time_and_the_running_one_is_never_cancelled() -> None:
    # Two runs side by side both list the titles before either files the due one.
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))

    assert workflow["concurrency"] == {"group": "kas-node-watch", "cancel-in-progress": False}
