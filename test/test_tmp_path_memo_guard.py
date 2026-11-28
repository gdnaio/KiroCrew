"""Guard: no memo in a test module is keyed on a path under a test's ``tmp_path``.

``setup.cfg`` sets ``tmp_path_retention_policy = failed``, so pytest removes a passing test's
``tmp_path`` at its teardown. It names a ``tmp_path`` from the first 30 characters of the test's
name and a number one past the highest still on disk, so once a passing test's directory is
gone, the next test whose name shares those characters can get the same path, as can every case
of a parametrized test with a long name. A memo kept for the process under that path answers
the later test from the earlier test's files: a scan reports a tree that is gone, a "checked
once" set skips the check, a "last value written" table carries one test's numbers into the
next.

A memo is a function under ``functools.lru_cache`` or ``functools.cache`` (decorated, or wrapped
by a call), or a function that reads and writes a module-level dict or set under a key built
from one of its parameters. A call feeds it a ``tmp_path`` when the argument bound to a key
parameter is a path built from a ``tmp_path`` or ``tmpdir`` parameter, from a ``self`` attribute
set from one, or from a fixture returning one: directly, through a function that hands one of
its own parameters to the memo's key, or through a module global its own module patches to such
a path. A ``tmp_path_factory.mktemp`` directory is never handed out twice in one run, so a memo
may be keyed on one; so may the checkout.

shortcut: a path handed to an object's constructor and reached through its methods is not
followed, nor is a cache decorator imported under another name; follow constructor arguments
into ``self`` attributes, and resolve ``functools`` import aliases, once a memo is reached
either way.
"""

from __future__ import annotations

import ast
import re
import textwrap
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import pytest
import source_corpus

pytestmark = pytest.mark.xdist_group(name="tree_scan_tmp_path_memo_guard")

_TMP_PARAMS = frozenset({"tmp_path", "tmpdir"})
_MEMO_DECORATORS = frozenset({"cache", "lru_cache"})
_CONTAINER_CALLS = frozenset(
    {"dict", "set", "defaultdict", "OrderedDict", "WeakKeyDictionary", "WeakValueDictionary"}
)
#: Methods whose result is still a path under the receiver. Any other method (``read_text``,
#: ``stat``, ``exists``) yields what the path holds or says, which keys nothing on the path.
_PATH_METHODS = frozenset(
    {
        "absolute",
        "as_posix",
        "expanduser",
        "glob",
        "iterdir",
        "joinpath",
        "relative_to",
        "resolve",
        "rglob",
        "with_name",
        "with_stem",
        "with_suffix",
    }
)
#: Calls whose result is not a path even when an argument is one.
_NOT_A_PATH = frozenset({"bool", "hash", "id", "int", "len", "repr"})
_PATCH_CALLS = frozenset({"object", "patch", "setattr"})
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
#: A filter for speed in front of the parse in :func:`scan`: a module joins the scan by holding one
#: of these patterns, or by naming a module the scan found a memo in. What it promises is that
#: every form :data:`_SPELLINGS` pins reaches the parse: ``lru_cache``; ``cache`` after an ``@`` on
#: its line, or called however it is parenthesised; and a binding, on the line its name starts, to
#: ``{}`` or to a call of a :data:`_CONTAINER_CALLS` name, bare or inside the parentheses ``black``
#: wraps a long one in. A memo written another way, such as with its ``=`` on a later line than its
#: name or a comment inside its value, can be skipped: give that form a case in :data:`_SPELLINGS`
#: and a pattern here. The patterns are searched apart, so one that opens with a literal is found
#: by a scan for that literal instead of by an alternation that tries every branch at every
#: position.
_DEFINES = (
    re.compile(r"lru_cache"),
    re.compile(r"@[^\n]*\bcache\b"),
    re.compile(r"cache(?<!\wcache)\b[\s)]*\("),
    re.compile(
        r"^\w[^\n]*=\s*(?:\(\s*)*(?:\{\s*\}|(?:\w+\.)*(?:"
        + "|".join(sorted(_CONTAINER_CALLS))
        + r")\b[\s)]*\()",
        re.M,
    ),
)

Function = ast.FunctionDef | ast.AsyncFunctionDef
Binding = tuple[list[ast.expr], ast.AST | None]


@dataclass(frozen=True)
class Memo:
    """A memo's parameters, and the ones its key is built from."""

    params: tuple[str, ...]
    key: frozenset[str]


@dataclass
class Module:
    """One parsed test module: its keyed memos, the names it imports, and its functions."""

    rel: str
    stem: str
    tree: ast.Module
    memos: dict[str, Memo] = field(default_factory=dict)
    imported: dict[str, tuple[str, str]] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    functions: list[Function] = field(default_factory=list)
    calls: dict[int, list[ast.Call]] = field(default_factory=dict)


def _params(fn: Function | ast.Lambda) -> tuple[str, ...]:
    args = fn.args
    return tuple(a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs))


def _call_name(func: ast.expr) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _decorators(fn: Function) -> set[str]:
    return {_call_name(d.func if isinstance(d, ast.Call) else d) for d in fn.decorator_list}


def _positional(index: ast.expr) -> bool:
    """``paths[0]`` or ``paths[1:]`` still names a path; ``record["id"]`` names a field."""
    if isinstance(index, ast.Slice):
        return True
    if isinstance(index, ast.UnaryOp) and isinstance(index.op, ast.USub):
        index = index.operand
    return isinstance(index, ast.Constant) and type(index.value) is int


def builds_on(node: ast.AST | None, names: Iterable[str], attrs: Iterable[str] = ()) -> bool:
    """Whether *node* is a path built from one of *names*, or a ``self`` attribute in *attrs*."""
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return node.attr in attrs
        return node.attr in ("parent", "parents") and builds_on(node.value, names, attrs)
    if isinstance(node, ast.Subscript):
        return _positional(node.slice) and builds_on(node.value, names, attrs)
    if isinstance(node, ast.Starred):
        return builds_on(node.value, names, attrs)
    if isinstance(node, ast.BinOp):
        return builds_on(node.left, names, attrs) or builds_on(node.right, names, attrs)
    if isinstance(node, ast.JoinedStr):
        return any(
            isinstance(v, ast.FormattedValue) and builds_on(v.value, names, attrs)
            for v in node.values
        )
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(builds_on(e, names, attrs) for e in node.elts)
    if isinstance(node, ast.Call):
        func = node.func
        arguments = [*node.args, *(k.value for k in node.keywords)]
        if isinstance(func, ast.Attribute) and func.attr != "join":
            if func.attr not in _PATH_METHODS:
                return False
            return builds_on(func.value, names, attrs) or any(
                builds_on(a, names, attrs) for a in arguments
            )
        if _call_name(func) in _NOT_A_PATH:
            return False
        # str(p), Path(p), os.fspath(p), os.path.join(p, ...), make_tree(tmp_path)
        return any(builds_on(a, names, attrs) for a in arguments)
    return False


def _bindings(fn: Function) -> list[Binding]:
    """``(targets, value)`` for every binding in *fn*; a loop or ``with`` binds a NAME only."""
    out: list[Binding] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            target, value = node.targets[0], node.value
            pairs = (
                isinstance(target, (ast.Tuple, ast.List))
                and isinstance(value, (ast.Tuple, ast.List))
                and len(target.elts) == len(value.elts)
                and len(node.targets) == 1
            )
            if pairs:  # ``a, b = x, y`` binds each name to its own value
                out.extend(([t], v) for t, v in zip(target.elts, value.elts))
            else:
                out.append((node.targets, node.value))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            out.append(([node.target], node.value))
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            if isinstance(node.target, ast.Name):
                out.append(([node.target], node.iter))
        elif isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name):
            out.append(([node.optional_vars], node.context_expr))
    return out


class Scan:
    """Every keyed memo across *modules*, and every call that keys one on a ``tmp_path``.

    *fixture_modules* are read only for the fixtures they hand the scanned modules.
    """

    def __init__(self, modules: Iterable[Module], fixture_modules: Iterable[Module] = ()) -> None:
        self.modules = {m.rel: m for m in modules}
        self._stems: dict[str, list[Module]] = {}
        for module in self.modules.values():
            self._stems.setdefault(module.stem, []).append(module)
        self._fixture_sources = {m.rel: m for m in (*fixture_modules, *self.modules.values())}
        self._bound: dict[int, list[Binding]] = {}
        self._flowed: dict[tuple[int, frozenset[str], frozenset[str]], set[str]] = {}
        self.forward()
        self._fixtures: dict[str, set[str]] | None = None

    def flows(self, fn: Function, seeds: Iterable[str], attrs: Iterable[str] = ()) -> set[str]:
        """*seeds* plus every name *fn* binds to a path built from them."""
        names, kept = set(seeds), frozenset(attrs)
        if not names and not kept:
            return names
        cache_key = (id(fn), frozenset(names), kept)
        if cache_key in self._flowed:
            return self._flowed[cache_key]
        bindings = self._bound.get(id(fn))
        if bindings is None:
            bindings = self._bound[id(fn)] = _bindings(fn)
        grew = True
        while grew:
            grew = False
            for targets, value in bindings:
                if not builds_on(value, names, kept):
                    continue
                for target in targets:
                    for leaf in ast.walk(target):
                        if isinstance(leaf, ast.Name) and leaf.id not in names:
                            names.add(leaf.id)
                            grew = True
        self._flowed[cache_key] = names
        return names

    def target(self, call: ast.Call, here: Module) -> tuple[str, Memo, bool] | None:
        """The memo *call* reaches: ``(qualified name, memo, called on self or cls)``."""
        func = call.func
        if isinstance(func, ast.Name):
            if func.id in here.memos:
                return f"{here.stem}.{func.id}", here.memos[func.id], False
            stem, name = here.imported.get(func.id, ("", ""))
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            owner = func.value.id
            if owner in ("self", "cls") and func.attr in here.memos:
                return f"{here.stem}.{func.attr}", here.memos[func.attr], True
            stem, name = here.aliases.get(owner, ""), func.attr
        else:
            return None
        for source in self._stems.get(stem, ()):
            if name in source.memos:
                return f"{stem}.{name}", source.memos[name], False
        return None

    def key_arguments(self, call: ast.Call, here: Module) -> tuple[str, list[ast.expr]] | None:
        """The memo *call* reaches, and the arguments it binds to that memo's key."""
        reached = self.target(call, here)
        if reached is None:
            return None
        qualified, memo, method = reached
        params = (
            memo.params[1:] if method and memo.params[:1] in (("self",), ("cls",)) else memo.params
        )
        bound = [arg for param, arg in zip(params, call.args) if param in memo.key]
        bound += [k.value for k in call.keywords if k.arg in memo.key]
        return qualified, bound

    def _memo_calls(
        self, module: Module, fn: Function
    ) -> Iterator[tuple[ast.Call, str, list[ast.expr]]]:
        """Each call in *fn* that reaches a memo, its name, and the arguments keying it."""
        names = self._memo_names | {
            local for local, (_stem, name) in module.imported.items() if name in self._memo_names
        }
        for call in module.calls.get(id(fn), ()):
            if _call_name(call.func) in names and (reached := self.key_arguments(call, module)):
                yield call, reached[0], reached[1]

    def forward(self) -> None:
        """Make each function that hands one of its parameters to a memo's key a memo too."""
        grew = True
        while grew:
            grew = False
            self._memo_names = {name for m in self.modules.values() for name in m.memos}
            for module in self.modules.values():
                for fn in module.functions:
                    params = [p for p in _params(fn) if p not in ("self", "cls")]
                    known = module.memos.get(fn.name)
                    key = set(known.key) if known else set()
                    for _call, _qualified, arguments in self._memo_calls(module, fn):
                        for arg in arguments:
                            key.update(p for p in params if builds_on(arg, self.flows(fn, {p})))
                    if key and (known is None or not key <= known.key):
                        module.memos[fn.name] = Memo(_params(fn), frozenset(key))
                        grew = True

    @property
    def fixtures(self) -> dict[str, set[str]]:
        """Per module, the fixture names that hand a test a path under its ``tmp_path``."""
        if self._fixtures is None:
            self._fixtures = self._tainted_fixtures()
        return self._fixtures

    def _tainted_fixtures(self) -> dict[str, set[str]]:
        names: dict[tuple[str, str], str] = {}
        tainted: set[tuple[str, str]] = set()
        grew = True
        while grew:
            grew = False
            for module in self._fixture_sources.values():
                for fn in module.functions:
                    if "fixture" not in _decorators(fn):
                        continue
                    ident = (module.rel, fn.name)
                    names[ident] = _fixture_name(fn)
                    if ident in tainted:
                        continue
                    visible = _TMP_PARAMS | self._visible(module, names, tainted)
                    flowed = self.flows(fn, set(_params(fn)) & visible)
                    if flowed and any(
                        isinstance(node, (ast.Return, ast.Yield)) and builds_on(node.value, flowed)
                        for node in ast.walk(fn)
                    ):
                        tainted.add(ident)
                        grew = True
        return {rel: self._visible(m, names, tainted) for rel, m in self.modules.items()}

    def _visible(
        self, module: Module, names: dict[tuple[str, str], str], tainted: set[tuple[str, str]]
    ) -> set[str]:
        """The tainted fixtures *module* can request: its own, a conftest's, and ones it imports."""
        imported = set(module.imported.values())
        out: set[str] = set()
        for rel, fn_name in tainted:
            source = self._fixture_sources[rel]
            if (
                rel == module.rel
                or rel.endswith("conftest.py")
                or (source.stem, fn_name) in imported
            ):
                out.add(names[(rel, fn_name)])
        return out

    def offenders(self) -> list[str]:
        """``path:line: memo is keyed on a tmp_path value (argument)`` for each offending call."""
        found: set[str] = set()
        for module in self.modules.values():
            fixtures = self.fixtures.get(module.rel, set())
            patched = self._patched_globals(module, fixtures)
            for fn, attrs in self._scopes(module, fixtures):
                names = self.flows(fn, set(_params(fn)) & (_TMP_PARAMS | fixtures), attrs)
                if not (names or attrs or patched):
                    continue
                for call, qualified, arguments in self._memo_calls(module, fn):
                    for arg in arguments:
                        if builds_on(arg, names, attrs) or builds_on(arg, patched):
                            found.add(
                                f"{module.rel}:{call.lineno}: {qualified} is keyed on a "
                                f"tmp_path value ({ast.unparse(arg)})"
                            )
        return sorted(found)

    def _scopes(self, module: Module, fixtures: set[str]) -> Iterator[tuple[Function, set[str]]]:
        """Each function of *module*, with the ``self`` attributes its class sets to a tmp path."""
        in_classes: set[int] = set()
        for cls in ast.walk(module.tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            methods = [fn for fn in cls.body if isinstance(fn, _FUNCS)]
            attrs: set[str] = set()
            for fn in methods:
                flowed = self.flows(fn, set(_params(fn)) & (_TMP_PARAMS | fixtures))
                if not flowed:
                    continue
                for targets, value in self._bound[id(fn)]:
                    if builds_on(value, flowed):
                        attrs.update(
                            t.attr
                            for t in targets
                            if isinstance(t, ast.Attribute)
                            and isinstance(t.value, ast.Name)
                            and t.value.id == "self"
                        )
            for fn in methods:
                in_classes.add(id(fn))
                yield fn, attrs
        for fn in module.functions:
            if id(fn) not in in_classes:
                yield fn, set()

    def _patched_globals(self, module: Module, fixtures: set[str]) -> set[str]:
        """Module globals *module* patches, in one of its tests, to a path under ``tmp_path``."""
        patched: set[str] = set()
        for fn in module.functions:
            names = self.flows(fn, set(_params(fn)) & (_TMP_PARAMS | fixtures))
            if not names:
                continue
            for call in module.calls.get(id(fn), ()):
                if _call_name(call.func) not in _PATCH_CALLS:
                    continue
                for index, arg in enumerate(call.args):
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        later = [*call.args[index + 1 :], *(k.value for k in call.keywords)]
                        if any(builds_on(value, names) for value in later):
                            patched.add(arg.value.rsplit(".", 1)[-1])
        return patched


def _fixture_name(fn: Function) -> str:
    for decorator in fn.decorator_list:
        for keyword in getattr(decorator, "keywords", ()):
            if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                return str(keyword.value.value)
    return fn.name


def _empty_containers(tree: ast.Module) -> set[str]:
    """Module-level names bound to ``{}``, or to a dict or set a constructor call makes."""
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        empty = (isinstance(value, ast.Dict) and not value.keys) or (
            isinstance(value, ast.Call) and _call_name(value.func) in _CONTAINER_CALLS
        )
        if empty:
            out.update(t.id for t in targets if isinstance(t, ast.Name))
    return out


def _container_key(fn: Function, boxes: set[str]) -> frozenset[str]:
    """The parameters *fn* both reads and writes one module-level dict or set under."""
    params = set(_params(fn))
    reads: dict[str, set[str]] = {}
    writes: dict[str, set[str]] = {}

    def note(table: dict[str, set[str]], box: str, key: ast.AST) -> None:
        used = {p for p in params if builds_on(key, {p})}
        if used:
            table.setdefault(box, set()).update(used)

    for node in ast.walk(fn):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id in boxes:
                note(
                    writes if isinstance(node.ctx, ast.Store) else reads, node.value.id, node.slice
                )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args:
            owner = node.func.value
            if isinstance(owner, ast.Name) and owner.id in boxes:
                if node.func.attr in ("get", "setdefault"):
                    note(reads, owner.id, node.args[0])
                if node.func.attr in ("add", "setdefault"):
                    note(writes, owner.id, node.args[0])
        elif isinstance(node, ast.Compare) and len(node.ops) == 1:
            box = node.comparators[0]
            if isinstance(node.ops[0], (ast.In, ast.NotIn)) and isinstance(box, ast.Name):
                if box.id in boxes:
                    note(reads, box.id, node.left)
    key: set[str] = set()
    for box in reads.keys() & writes.keys():
        key |= reads[box] & writes[box]
    return frozenset(key)


def _is_own_module(value: ast.expr) -> bool:
    """``sys.modules[__name__]``."""
    return (
        isinstance(value, ast.Subscript)
        and isinstance(value.value, ast.Attribute)
        and value.value.attr == "modules"
        and isinstance(value.slice, ast.Name)
        and value.slice.id == "__name__"
    )


def collect(rel: str, source: str) -> Module | None:
    """*source*'s keyed memos, imports and functions, or ``None`` when it does not parse."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    module = Module(rel, rel.rsplit("/", 1)[-1].removesuffix(".py"), tree)
    boxes = _empty_containers(tree)
    module.functions = [node for node in ast.walk(tree) if isinstance(node, _FUNCS)]
    defs = {fn.name: fn for fn in module.functions}
    for fn in module.functions:
        module.calls[id(fn)] = [node for node in ast.walk(fn) if isinstance(node, ast.Call)]
        if _decorators(fn) & _MEMO_DECORATORS:
            params = _params(fn)
            module.memos[fn.name] = Memo(params, frozenset(params) - {"self", "cls"})
        elif boxes and (key := _container_key(fn, boxes)):
            module.memos[fn.name] = Memo(_params(fn), key)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                stem = node.module.rsplit(".", 1)[-1]
                module.imported[alias.asname or alias.name] = (stem, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                stem = alias.name.rsplit(".", 1)[-1]
                module.aliases[alias.asname or stem] = stem
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
            if _is_own_module(value):
                module.aliases.update((name, module.stem) for name in names)
            if not isinstance(value, ast.Call) or not value.args:
                continue
            wrapper = value.func.func if isinstance(value.func, ast.Call) else value.func
            first = value.args[0]
            if _call_name(wrapper) in _MEMO_DECORATORS:
                wrapped = defs.get(first.id) if isinstance(first, ast.Name) else first
                if isinstance(wrapped, (*_FUNCS, ast.Lambda)):
                    params = _params(wrapped)
                    module.memos.update((name, Memo(params, frozenset(params))) for name in names)
            elif _call_name(value.func) == "import_module" and isinstance(first, ast.Constant):
                stem = str(first.value).rsplit(".", 1)[-1]
                module.aliases.update((name, stem) for name in names)
    module.memos = {name: memo for name, memo in module.memos.items() if memo.key}
    return module


def _is_test_module(rel: str) -> bool:
    if rel == "conftest.py":
        return True
    if rel.startswith("test/"):
        return not rel.startswith("test/workflows/malicious/")
    return rel.startswith("src/kiro_crew/apps/builtins/") and ("/tests/" in rel or "_tests/" in rel)


def scan(texts: dict[str, str]) -> Scan:
    """The scan of *texts* (``{repo-relative path: source}``), parsing only what can matter.

    A module joins when it defines a memo, or names the module of one, which every way of
    reaching a memo from another module (``from m import f``, ``import m``,
    ``import_module("m")``) needs. Every conftest, and each module a member imports from, is
    read for the fixtures it hands the members: every module of that name, since which one an
    import binds depends on the path a run puts first.
    """
    parsed: dict[str, Module] = {}

    def parse(rel: str) -> bool:
        if rel not in parsed and (module := collect(rel, texts[rel])) is not None:
            parsed[rel] = module
        return rel in parsed

    members = {
        rel
        for rel, text in texts.items()
        if any(pattern.search(text) for pattern in _DEFINES) and parse(rel)
    }
    seen: set[str] = set()
    while True:
        Scan(parsed[rel] for rel in members)
        stems = {parsed[rel].stem for rel in members if parsed[rel].memos} - seen
        if not stems:
            break
        seen |= stems
        word = re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, sorted(stems))))
        for rel, text in texts.items():
            if rel not in members and any(stem in text for stem in stems):
                if word.search(text) and parse(rel):
                    members.add(rel)
    by_stem: dict[str, list[str]] = {}
    for rel in texts:
        by_stem.setdefault(rel.rsplit("/", 1)[-1].removesuffix(".py"), []).append(rel)
    suppliers = {
        supplier
        for rel in members
        for stem, _name in parsed[rel].imported.values()
        for supplier in by_stem.get(stem, ())
    }
    suppliers |= {rel for rel in texts if rel.endswith("conftest.py")}
    fixture_modules = [parsed[rel] for rel in suppliers - members if parse(rel)]
    return Scan((parsed[rel] for rel in members), fixture_modules)


def tree_texts() -> dict[str, str]:
    """Every test module of both testpaths, as ``{repo-relative path: source}``."""
    root = source_corpus.repo_root()
    texts: dict[str, str] = {}
    for path in source_corpus.repo_files_named(".py"):
        rel = path.relative_to(root).as_posix()
        if _is_test_module(rel):
            try:
                texts[rel] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
    return texts


# ---------------------------------------------------------------------------
# the rule, on planted modules: every way in, and the shapes that are not one
# ---------------------------------------------------------------------------

_HELPERS = textwrap.dedent("""
    import functools

    @functools.lru_cache(maxsize=None)
    def scan_root(root):
        return tuple(root.rglob("*"))

    def scan(root=REAL_ROOT):
        return list(scan_root(root))

    kept = functools.lru_cache(maxsize=None)(lambda root: root)

    _SEEN = {}

    def stamp(path):
        if path not in _SEEN:
            _SEEN[path] = len(_SEEN)
        return _SEEN[path]

    _DONE = set()

    def copy(template, dest):
        if template not in _DONE:
            _DONE.add(template)
        return dest
    """)

_PLANTED = [
    ("a-joined-path", "def test_a(tmp_path):\n    scan_root(tmp_path / 'repo')\n", 1),
    (
        "an-assigned-path",
        "def test_a(tmp_path):\n    root = tmp_path / 'repo'\n    tree = root / 'x'\n"
        "    scan_root(tree)\n",
        1,
    ),
    ("a-helper-built-path", "def test_a(tmp_path):\n    scan_root(make_tree(tmp_path))\n", 1),
    (
        "a-string-spelling",
        "def test_a(tmp_path):\n    scan_root(str(tmp_path))\n    scan_root(f'{tmp_path}/x')\n",
        2,
    ),
    ("a-tmpdir", "def test_a(tmpdir):\n    scan_root(tmpdir)\n", 1),
    ("through-a-forwarder", "def test_a(tmp_path):\n    scan(tmp_path)\n", 1),
    ("through-a-module-alias", "def test_a(tmp_path):\n    helpers.scan_root(tmp_path)\n", 1),
    ("a-wrapped-lambda", "def test_a(tmp_path):\n    helpers.kept(tmp_path)\n", 1),
    ("a-dict-memo", "def test_a(tmp_path):\n    stamp(tmp_path / 'f')\n", 1),
    ("a-set-memo-key", "def test_a(tmp_path):\n    copy(tmp_path / 't', 'dest')\n", 1),
    (
        "one-name-of-an-unpacked-pair",
        "def test_a(tmp_path):\n    template, dest = tmp_path / 't', ROOT\n"
        "    copy(template, dest)\n",
        1,
    ),
    (
        "a-fixture",
        "import pytest\n\n@pytest.fixture\ndef repo(tmp_path):\n    return tmp_path / 'r'\n\n"
        "def test_a(repo):\n    scan_root(repo)\n",
        1,
    ),
    (
        "a-self-attribute",
        "import pytest\n\nclass TestA:\n    @pytest.fixture(autouse=True)\n"
        "    def _root(self, tmp_path):\n        self.root = tmp_path / 'r'\n\n"
        "    def test_a(self):\n        scan_root(self.root)\n",
        1,
    ),
    (
        "a-patched-global",
        "import functools\nimport sys\n\nROOT = None\n\n@functools.lru_cache(maxsize=None)\n"
        "def _facts(root):\n    return root\n\ndef facts():\n    return _facts(ROOT)\n\n"
        "def test_a(tmp_path, monkeypatch):\n"
        "    monkeypatch.setattr(sys.modules[__name__], 'ROOT', tmp_path / 'fake')\n"
        "    facts()\n",
        1,
    ),
    (
        "a-mktemp-directory",
        "def test_a(tmp_path_factory):\n    scan_root(tmp_path_factory.mktemp('x'))\n",
        0,
    ),
    (
        "what-a-file-holds",
        "def test_a(tmp_path):\n    scan_root((tmp_path / 'f').read_text())\n",
        0,
    ),
    ("an-identity", "def test_a(tmp_path):\n    stamp(id(tmp_path))\n", 0),
    ("the-unwrapped-function", "def test_a(tmp_path):\n    scan_root.__wrapped__(tmp_path)\n", 0),
    (
        "a-field-of-a-record",
        "def test_a(tmp_path):\n    record = {'id': 'x', 'path': tmp_path}\n"
        "    stamp(record['id'])\n",
        0,
    ),
    (
        "a-parameter-outside-the-key",
        "def test_a(tmp_path):\n    copy(TEMPLATE, tmp_path / 'd')\n",
        0,
    ),
    (
        "the-other-name-of-an-unpacked-pair",
        "def test_a(tmp_path):\n    template, dest = ROOT / 't', tmp_path / 'd'\n"
        "    copy(template, dest)\n",
        0,
    ),
    ("the-real-tree", "def test_a(tmp_path):\n    scan_root(REAL_ROOT)\n", 0),
]


def _offenders_in(body: str) -> list[str]:
    head = "import helpers\nfrom helpers import copy, scan, scan_root, stamp\n\n"
    return scan({"test/helpers.py": _HELPERS, "test/test_case.py": head + body}).offenders()


@pytest.mark.parametrize(
    ("body", "expected"), [(b, e) for _id, b, e in _PLANTED], ids=[i for i, _b, _e in _PLANTED]
)
def test_a_memo_keyed_on_a_tmp_path_is_found_by_every_route_in(body: str, expected: int) -> None:
    offenders = _offenders_in(body)
    assert len(offenders) == expected, offenders


_DICT_MEMO = (
    "\n\ndef kept(root):\n    if root not in _SEEN:\n        _SEEN[root] = len(_SEEN)\n"
    "    return _SEEN[root]\n"
)

_SPELLINGS = [
    (
        "a-cache-decorator",
        "import functools\n\n@functools.cache\ndef kept(root):\n    return root\n",
    ),
    (
        "an-aliased-module",
        "import functools as ft\n\n@ft.cache\ndef kept(root):\n    return root\n",
    ),
    (
        "a-wrapped-function",
        "import functools\n\ndef _kept(root):\n    return root\n\n"
        "kept = functools.lru_cache(maxsize=None)(_kept)\n",
    ),
    (
        "a-cache-call",
        "import functools\n\ndef _kept(root):\n    return root\n\nkept = functools.cache(_kept)\n",
    ),
    ("a-dict-literal", "_SEEN = {}" + _DICT_MEMO),
    ("an-annotated-dict", "_SEEN: dict[str, int] = {}" + _DICT_MEMO),
    ("a-chained-binding", "_SEEN = _ALSO = {}" + _DICT_MEMO),
    ("an-ordered-dict", "import collections\n\n_SEEN = collections.OrderedDict()" + _DICT_MEMO),
    ("a-weak-dict", "import weakref\n\n_SEEN = weakref.WeakValueDictionary()" + _DICT_MEMO),
    ("a-default-dict", "import collections\n\n_SEEN = collections.defaultdict(int)" + _DICT_MEMO),
    (
        "a-set",
        "_SEEN = set()\n\ndef kept(root):\n    if root not in _SEEN:\n        _SEEN.add(root)\n"
        "    return root\n",
    ),
    ("a-binding-black-wraps", "_SEEN: dict[str, int] = (\n    {}\n)" + _DICT_MEMO),
    (
        "a-constructor-black-wraps",
        "import collections\n\n_SEEN: dict[str, int] = (\n    collections.OrderedDict()\n)"
        + _DICT_MEMO,
    ),
    (
        "a-parenthesised-decorator",
        "import functools\n\n@(functools.cache)\ndef kept(root):\n    return root\n",
    ),
    (
        "a-decorator-reached-through-a-call",
        "@registry.get().cache\ndef kept(root):\n    return root\n",
    ),
    (
        "a-parenthesised-wrapper",
        "import functools\n\ndef _kept(root):\n    return root\n\n"
        "kept = (functools.cache)(_kept)\n",
    ),
]


@pytest.mark.parametrize("memo", [m for _id, m in _SPELLINGS], ids=[i for i, _m in _SPELLINGS])
def test_a_memo_in_each_pinned_spelling_reaches_the_scan(memo: str) -> None:
    """The text filter in front of the parse lets each of these memo spellings through."""
    case = memo + "\n\ndef test_a(tmp_path):\n    kept(tmp_path / 'tree')\n"
    assert len(scan({"test/test_alone.py": case}).offenders()) == 1


_TMP_PATH_REPO = (
    "import pytest\n\n@pytest.fixture\ndef repo(tmp_path):\n    return tmp_path / 'r'\n"
)
_PLAIN_REPO = "import pytest\n\n@pytest.fixture\ndef repo():\n    return 'r'\n"


@pytest.mark.parametrize("holder", ["test/a", "test/b"], ids=["the-first", "the-last"])
def test_a_fixture_from_either_of_two_modules_of_one_name_is_read(holder: str) -> None:
    """Two helper modules in different directories can share a file name, and the fixtures of
    both are read: which one ``from repo_fixtures import repo`` binds depends on the run. The
    ``tmp_path`` fixture sits in the first module of the name in one case and in the last in the
    other, so a scan that reads only one of them fails a case."""
    texts = {
        f"{root}/repo_fixtures.py": _TMP_PATH_REPO if root == holder else _PLAIN_REPO
        for root in ("test/a", "test/b")
    }
    texts["test/helpers.py"] = _HELPERS
    texts["test/test_case.py"] = (
        "from helpers import scan_root\nfrom repo_fixtures import repo\n\n"
        "def test_a(repo):\n    scan_root(repo)\n"
    )
    assert len(scan(texts).offenders()) == 1


# ---------------------------------------------------------------------------
# the tree
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tree_scan() -> Scan:
    with source_corpus.untraced():
        return scan(tree_texts())


def test_no_memo_in_a_test_module_is_keyed_on_a_tmp_path(tree_scan: Scan) -> None:
    with source_corpus.untraced():
        offenders = tree_scan.offenders()
    assert not offenders, (
        "a memo in a test module is keyed on a path under a test's tmp_path, which pytest "
        "hands to a later test once a passing one is removed. Plant the tree under a "
        "tmp_path_factory.mktemp directory, or keep no table by path:\n  " + "\n  ".join(offenders)
    )


def test_the_scan_reads_the_memos_of_the_tree(tree_scan: Scan) -> None:
    """An empty scan and a broken one pass the guard alike, so known memos must show."""
    memos = {f"{m.stem}.{name}" for m in tree_scan.modules.values() for name in m.memos}
    assert {
        "git_repo_templates.copy_repo",
        "test_lazy_data_home_paths._path_factories_for",
    } <= memos
