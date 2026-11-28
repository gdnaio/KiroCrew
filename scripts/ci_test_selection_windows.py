"""The `windows` block of the SHADOW verdict: the test files a diff needs on Windows.

SHADOW ONLY: `scripts/ci_test_selection.py` writes the block into its verdict and
`scripts/ci_test_selection_ledger.py` scores it against what failed on Windows.
No lane runs fewer tests because of it.

    "windows": {"verdict": "select" | "full", "reason": str | null,
                "run": [test file, ...], "would_run_share": float}

`run`, under `select`, is the union of
1. the test files the diff changes;
2. the bundle's `windows_targeting` test files the scorer put in tiers A to D;
3. the whole guard tier: it is there for what the map cannot see reach a test,
   and an OS effect is of that kind;
4. the test files that read a changed prose doc or frontend file
   (`scripts/ci_test_data_readers.py`): what they read is the same on Windows;
5. the test files the map has no entry for: it cannot say what they run;
6. tiers A to D of the OS-sensitive changed files scored as a diff of their own:
   a changed `.py` file the bundle's `os_sensitive` table names, one whose
   added lines hit a pattern of `ci_test_os_tables` (W1, W2 or W3), or one
   whose added lines call into a file the table names (`calls_into`). The
   added lines are matched as they are, comments included, and with no added
   lines to read every changed `.py` file counts: both err toward more tests;
minus the files Windows never collects (`test/windows-collect-ignore.txt`).

`calls_into`. A file can act differently on Windows with nothing in its own text
saying so: it calls a function that branches on the OS in another file. A
changed file counts when the lines the diff adds to it use, outside a comment,

* a name it imports from a module whose file the `os_sensitive` table names,
  where that file defines the name as a function or a class
  (`from kiro_crew.lock import held`, then `held()`); or
* an attribute of such a module it imports, defined there the same way
  (`from kiro_crew import lock` or `import kiro_crew.lock`, then
  `lock.held()`).

The imports are read from the file at the head and the definitions from the
imported file there. A name is followed one step, to the module the import
names: a name that module only passes on from a third is not followed.

The verdict is `full`, first match wins, when
* the Linux verdict is `full` (`linux_full`);
* a changed path is in the `windows` bucket of ci.yml's `changes` job, read from
  that file so the list lives in one place (`windows_path`);
* the bundle lacks `os_sensitive` or `windows_targeting` (`table_absent`);
* the bundle's `built_at` is more than `STALE_DAYS` days before the head commit
  (now, for a diff with no head), or is missing or unreadable (`bundle_stale`);
* anything here raises (`error`).

`would_run_share` weighs each test file Windows collects as the Linux share
does (`ci_test_map.node_weights`).
"""

from __future__ import annotations

import ast
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import ci_test_map as ctm
import ci_test_os_tables as ost

# A bundle older than this against the diff's head scores nothing: the tables
# describe a tree the diff no longer sits on. The nightly rebuild makes 7 a
# week of failed builds.
STALE_DAYS = 7
RUN_TIERS = ("A", "B", "C", "D")
TABLES = ("os_sensitive", "windows_targeting")
CI_WORKFLOW = ".github/workflows/ci.yml"
GIT_TIMEOUT_SECS = 30.0

Changes = list[tuple[str, list[str]]]


def full(reason: str) -> dict[str, Any]:
    """The block that runs every Windows test, and why."""
    return {"verdict": "full", "reason": reason, "run": [], "would_run_share": 1.0}


def path_bucket(root: Path, bucket: str = "windows") -> list[str]:
    """The globs of one bucket (`windows`) in ci.yml's `changes` job path filter.

    Read line by line (the scorer is stdlib only): the bucket's key, then its
    quoted list items up to the next key. Anything else there raises, so a
    bucket this cannot read is never taken for an empty one.
    """
    globs: list[str] = []
    indent: int | None = None
    for line in (root / CI_WORKFLOW).read_text(encoding="utf-8").splitlines():
        body = line.strip()
        if indent is None:
            if body == f"{bucket}:":
                indent = len(line) - len(line.lstrip())
            continue
        if not body or body.startswith("#"):
            continue
        if len(line) - len(line.lstrip()) <= indent:
            break
        item = re.fullmatch(r"-\s+'([^']+)'", body)
        if item is None:
            raise ValueError(f"{CI_WORKFLOW}: cannot read the {bucket} filter line {body!r}")
        globs.append(item.group(1))
    if not globs:
        raise ValueError(f"{CI_WORKFLOW}: no {bucket} path filter")
    return globs


def glob_pattern(glob: str) -> re.Pattern[str]:
    """A path filter glob as a regex: `**/` any folders, `**` anything, `*` within one name."""
    parts: list[str] = []
    at = 0
    while at < len(glob):
        for token, regex in (("**/", r"(?:.*/)?"), ("**", r".*"), ("*", r"[^/]*")):
            if glob.startswith(token, at):
                parts.append(regex)
                at += len(token)
                break
        else:
            parts.append(re.escape(glob[at]))
            at += 1
    return re.compile("".join(parts))


def added_lines(diff: str) -> dict[str, str]:
    """`git diff -U0` text -> {path: the lines the diff adds to it}."""
    found: dict[str, list[str]] = {}
    path: str | None = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("+") and path is not None:
            found.setdefault(path, []).append(line[1:])
    return {name: "\n".join(lines) + "\n" for name, lines in found.items()}


def git_added_lines(base: str, head: str, root: Path) -> dict[str, str]:
    """`added_lines` of the `.py` files `base..head` changes."""
    out = subprocess.run(
        ["git", "diff", "-U0", "--no-color", "--no-ext-diff", f"{base}..{head}", "--", "*.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=GIT_TIMEOUT_SECS,
    ).stdout
    return added_lines(out)


def commit_time(ref: str, root: Path) -> datetime:
    """When the commit `ref` names was committed."""
    out = subprocess.run(
        ["git", "show", "-s", "--format=%cI", f"{ref}^{{commit}}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=GIT_TIMEOUT_SECS,
    ).stdout
    return datetime.fromisoformat(out.strip())


def is_stale(built_at: object, at: datetime) -> bool:
    """Whether a bundle built at `built_at` is too old for a diff whose head is `at`.

    A bundle that does not say when it was built, in a form with a UTC offset,
    is stale: its age is unknown.
    """
    if not isinstance(built_at, str):
        return True
    try:
        built = datetime.fromisoformat(built_at)
        return (at - built).total_seconds() > STALE_DAYS * 86400
    except (ValueError, TypeError):
        return True


Source = Callable[[str], "str | None"]
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _module_files(module: str) -> tuple[str, str]:
    """The two files a dotted module of the source tree can be."""
    stem = "src/" + module.replace(".", "/")
    return stem + ".py", stem + "/__init__.py"


def _defined(text: str | None) -> set[str]:
    """The functions and classes a file's text defines at its top level."""
    try:
        tree = ast.parse(text or "")
    except (SyntaxError, ValueError):
        return set()
    kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    return {node.name for node in tree.body if isinstance(node, kinds)}


def calls_into(path: str, added: str, table: Mapping[str, Any], source: Source) -> bool:
    """Whether the lines a diff adds to `path` use a function or class of an OS-sensitive file.

    `source` gives a file's text at the head, None when it is not there. A file
    that does not parse, the changed one or an imported one, brings nothing.
    """
    code = "\n".join(line.split("#", 1)[0] for line in added.splitlines())
    words = set(_WORD.findall(code))
    if not words:
        return False
    try:
        tree = ast.parse(source(path) or "")
    except (SyntaxError, ValueError):
        return False
    package = (
        path[len("src/") :].rsplit("/", 1)[0].replace("/", ".") if path.startswith("src/") else ""
    )
    defined: dict[str, set[str]] = {}

    def has(file: str, name: str) -> bool:
        if file not in table:
            return False
        if file not in defined:
            defined[file] = _defined(source(file))
        return name in defined[file]

    def attribute(local: str, files: tuple[str, str]) -> bool:
        named = re.findall(r"(?<![\w.])" + re.escape(local) + r"\.([A-Za-z_][A-Za-z0-9_]*)", code)
        return any(has(file, name) for file in files for name in named)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if attribute(alias.asname or alias.name, _module_files(alias.name)):
                    return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                parts = package.split(".") if package else []
                parts = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                module = ".".join([*parts, module] if module else parts)
            if not module:
                continue
            for alias in node.names:
                local = alias.asname or alias.name
                if local in words and any(has(f, alias.name) for f in _module_files(module)):
                    return True
                if attribute(local, _module_files(f"{module}.{alias.name}")):
                    return True
    return False


def os_sensitive_changes(
    changes: Changes,
    table: Mapping[str, Any],
    added: Mapping[str, str] | None,
    source: Source | None = None,
) -> Changes:
    """The changed `.py` files, tests apart, that can act differently on Windows.

    `source` gives a file's text at the head; without it no file counts for
    calling into another (`calls_into`).
    """
    picked: Changes = []
    for status, paths in changes:
        path = paths[-1] if paths else ""
        if not path.endswith(".py") or ctm.is_test_file(path):
            continue
        lines = None if added is None else added.get(path, "")
        if (
            path in table
            or lines is None
            or ost.tier_of(lines) is not None
            or (source is not None and calls_into(path, lines, table, source))
        ):
            picked.append((status, paths))
    return picked


def block(
    changes: Changes,
    linux: Mapping[str, Any],
    bundle: Mapping[str, Any],
    universe: Iterable[str],
    guard: Iterable[str],
    *,
    added: Mapping[str, str] | None,
    bucket: Iterable[str],
    ignored: set[str],
    at: datetime,
    score: Callable[[Changes], Mapping[str, Any]],
    readers: Iterable[str] = (),
    unknown: Iterable[str] = (),
    source: Source | None = None,
) -> dict[str, Any]:
    """The `windows` block for one diff the scorer gave the verdict body `linux`.

    `guard` is the guard tier the scorer used. `score` tiers a diff as the scorer
    did this one; `added` is `added_lines` of the diff, None when there are none
    to read; `at` is the head's commit time; `readers` is the test files that
    read a changed doc or frontend file (`ci_test_map.data_rows`); `unknown` is
    the test files the map has no entry for; `source` gives a file's text at
    the head, for `calls_into`.
    """
    if linux["verdict"] != "select":
        return full(f"linux_full: {linux['reason']}")
    patterns = [(glob, glob_pattern(glob)) for glob in bucket]
    for _status, paths in changes:
        for path in paths:
            hit = next((glob for glob, pattern in patterns if pattern.fullmatch(path)), None)
            if hit is not None:
                return full(f"windows_path: {path} is in the bucket ({hit})")
    for table in TABLES:
        if not bundle.get(table):
            return full(f"table_absent: {table}")
    if is_stale(bundle.get("built_at"), at):
        return full(f"bundle_stale: built_at is {bundle.get('built_at')!r}, the head is {at}")

    def run_tiers(body: Mapping[str, Any]) -> set[str]:
        return {test for tier in RUN_TIERS for test in body["tiers"][tier]}

    run = {paths[-1] for _status, paths in changes if paths and ctm.is_test_file(paths[-1])}
    run |= set(bundle["windows_targeting"]) & run_tiers(linux)
    run |= set(guard)
    run |= set(readers)
    # The map cannot say what a test file it has no entry for runs, on any system.
    run |= set(unknown)
    sensitive = os_sensitive_changes(changes, bundle["os_sensitive"], added, source)
    if sensitive:
        scored = score(sensitive)
        if scored["verdict"] != "select":
            return full(f"linux_full: {scored['reason']}")
        run |= run_tiers(scored)
    run -= ignored
    counts: Mapping[str, int] = bundle.get("tests", {})
    listed = bundle.get(ctm.UNRECORDED)
    weigh = ctm.node_weights(counts, listed if isinstance(listed, list) else ())
    collected = (set(universe) | run) - ignored
    total = sum(weigh(test) for test in collected)
    share = sum(weigh(test) for test in run) / total if total else 1.0
    return {"verdict": "select", "reason": None, "run": sorted(run), "would_run_share": share}
