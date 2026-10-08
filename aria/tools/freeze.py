"""
Frozen tool arguments
---------------------
``freeze_args`` takes the arguments of a tool call, checks that they are plain
JSON-like data and returns a deeply immutable COPY. The approved call and the
executed call must be the same thing, so everything after this point (policy,
approval, audit digest, execution) works from the frozen copy, and the caller's
original dict can change without any effect.

Accepted types, matched EXACTLY (a subclass is rejected, because a subclass can
override methods and run code while we read it):

  dict with str keys, list, tuple (treated as a list), str, int (at most 100
  digits), float (finite only), bool, None.

Strings (values and keys) must be valid Unicode text: an unpaired surrogate is
rejected. Otherwise two different strings could get the same ASCII escape in the
canonical JSON, and so the same digest.

Everything else is rejected with ``FreezeError``: bytes, sets, any other
object, non-str keys, NaN and infinity, cycles, data that is too deep or too
big. The checks never call ``repr``, ``str``, ``==`` or ``hash`` on a rejected
object, and error messages show only its type name and a path such as
``$.items[2].url``. Argument VALUES never appear in an error message.

The size limits are checked while reading, and a container whose length alone
would exceed the node budget is refused before its items are snapshotted. This
bounds the work and the copy to roughly the limits. It is not a guarantee against
every resource problem: the caller's own data is already in memory, and a string
is checked after it exists.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterator, Mapping, NoReturn, Optional

__all__ = [
    "MAX_INT_DIGITS",
    "FreezeError",
    "FrozenArgs",
    "FrozenMap",
    "freeze_args",
]

MAX_INT_DIGITS = 100
_MAX_INT_ABS = 10**MAX_INT_DIGITS
_MAX_INT_BITS = 333  # anything longer than this has more than 100 digits
_HARD_MAX_DEPTH = 100  # keeps every recursive walk far below the interpreter limit
_PATH_CHARS = 200
_KEY_CHARS = 40

_TOKEN = object()


class FreezeError(ValueError):
    """The arguments are not plain JSON-like data, or they exceed a limit."""


def _safe_type_name(value: object) -> str:
    """Name of the type of ``value`` without running code from the value or its metaclass."""
    try:
        t = type(value)
        name = type.__dict__["__name__"].__get__(t)
        return ascii(name[:60]) if type(name) is str else "<unknown>"
    except Exception:
        return "<unknown>"


class FrozenMap(Mapping[str, Any]):
    """A read-only mapping with str keys. Nested values are immutable too:
    mappings are ``FrozenMap``, lists are tuples, the rest are scalars.

    Equal to a plain dict with the same content (a tuple matches a list).
    Built only by ``freeze_args``. Not hashable, not picklable. ``repr`` shows
    the size only, so argument values do not leak into logs."""

    __slots__ = ("_d",)
    _d: dict[str, Any]

    def __init__(self, token: object, data: dict[str, Any]) -> None:
        if token is not _TOKEN:
            raise TypeError("FrozenMap is created by freeze_args, not directly")
        object.__setattr__(self, "_d", data)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("FrozenMap cannot be subclassed")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("FrozenMap is read-only")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("FrozenMap is read-only")

    def __getitem__(self, key: str) -> Any:
        if type(key) is not str:  # never hash a foreign object; makes get(unhashable) safe
            raise KeyError("key must be a str")
        return self._d[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __contains__(self, key: object) -> bool:
        # Only str keys can be present; do not hash anything else.
        return type(key) is str and key in self._d

    def __repr__(self) -> str:
        return f"<FrozenMap with {len(self._d)} keys>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        try:
            return _equal(self, other)
        except RecursionError:
            return False

    __hash__ = None  # type: ignore[assignment]

    def __copy__(self) -> FrozenMap:
        return self

    def __deepcopy__(self, memo: Any) -> FrozenMap:
        return self

    def __reduce_ex__(self, protocol: Any) -> NoReturn:
        raise TypeError("FrozenMap cannot be pickled; use FrozenArgs.thaw() for plain data")

    def thaw(self) -> dict[str, Any]:
        """A fresh plain deep copy (dicts and lists)."""
        return {k: _thaw(v) for k, v in self._d.items()}


def _scalar_equal(a: Any, b: Any) -> bool:
    # Python says True == 1 == 1.0 and 0.0 == -0.0, but their canonical JSON and
    # digest differ. Equality must agree with the digest, so the exact types must
    # match (this also keeps a foreign object's __eq__ from ever being called) and
    # a negative zero is not a zero.
    ta = type(a)
    if ta is not type(b):
        return False
    if ta is float:
        return bool(a == b and math.copysign(1.0, a) == math.copysign(1.0, b))
    return bool(a == b)


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, Mapping):
        if not isinstance(b, Mapping) or len(a) != len(b):
            return False
        for key, value in a.items():
            if key not in b or not _equal(value, b[key]):
                return False
        return True
    if type(a) is tuple:
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return False
        return all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(b, (Mapping, list, tuple)):
        return False
    return _scalar_equal(a, b)


def _thaw(value: Any) -> Any:
    if type(value) is FrozenMap:
        return {k: _thaw(v) for k, v in value._d.items()}
    if type(value) is tuple:
        return [_thaw(v) for v in value]
    return value


class FrozenArgs(Mapping[str, Any]):
    """
    The frozen arguments of one tool call.

    Behaves as a read-only mapping (``[]``, ``get``, ``in``, iteration, ``len``,
    ``==`` with a dict). ``thaw()`` gives a fresh plain deep copy,
    ``canonical_json()`` the canonical text and ``sha256()`` its plain SHA-256
    in hex. The text and digest are computed once, when the arguments are
    frozen, from the frozen copy.
    """

    __slots__ = ("_root", "_json", "_digest")
    _root: FrozenMap
    _json: str
    _digest: str

    def __init__(self, token: object, root: FrozenMap) -> None:
        if token is not _TOKEN:
            raise TypeError("FrozenArgs is created by freeze_args, not directly")
        text = json.dumps(
            root.thaw(), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_json", text)
        object.__setattr__(self, "_digest", hashlib.sha256(text.encode("ascii")).hexdigest())

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("FrozenArgs cannot be subclassed")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("FrozenArgs is read-only")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("FrozenArgs is read-only")

    def __getitem__(self, key: str) -> Any:
        return self._root[key]  # FrozenMap refuses non-str keys with KeyError

    def __iter__(self) -> Iterator[str]:
        return iter(self._root)

    def __len__(self) -> int:
        return len(self._root)

    def __contains__(self, key: object) -> bool:
        return key in self._root

    def __repr__(self) -> str:
        return f"<FrozenArgs with {len(self._root)} keys, sha256={self._digest[:12]}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        try:
            return _equal(self._root, other)
        except RecursionError:
            return False

    __hash__ = None  # type: ignore[assignment]

    def __copy__(self) -> FrozenArgs:
        return self

    def __deepcopy__(self, memo: Any) -> FrozenArgs:
        return self

    def __reduce_ex__(self, protocol: Any) -> NoReturn:
        raise TypeError("FrozenArgs cannot be pickled; use thaw() for plain data")

    def thaw(self) -> dict[str, Any]:
        """A fresh plain deep copy (dicts and lists). Changing it changes nothing else."""
        return self._root.thaw()

    def canonical_json(self) -> str:
        """Canonical JSON: sorted keys, no whitespace, ASCII only."""
        return str(self._json)

    def sha256(self) -> str:
        """Plain SHA-256 (hex) of the canonical JSON. Not keyed; a keyed digest for
        audit records comes from the redaction module."""
        return str(self._digest)


class _Walker:
    def __init__(self, max_depth: int, max_nodes: int, max_string_chars: int, max_total_chars: int):
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.max_string_chars = max_string_chars
        self.max_total_chars = max_total_chars
        self.nodes = 0
        self.chars = 0
        self.path: list[str] = ["$"]
        self.active: set[int] = set()

    def fail(self, message: str) -> NoReturn:
        where = "".join(self.path)
        if len(where) > _PATH_CHARS:
            where = where[: _PATH_CHARS - 3] + "..."
        raise FreezeError(f"arguments rejected at {where}: {message}")

    def add_string(self, text: str, what: str) -> None:
        n = len(text)
        if n > self.max_string_chars:
            self.fail(f"{what} is longer than {self.max_string_chars} characters")
        if not text.isascii():
            try:
                text.encode("utf-8")
            except UnicodeEncodeError:
                # An unpaired surrogate would be written as the same \uXXXX escape as
                # a real character (U+1F600 and the pair of surrogates that encode it),
                # so two different values would get the same canonical JSON and digest.
                self.fail(f"{what} contains an unpaired surrogate")
        self.chars += n
        if self.chars > self.max_total_chars:
            self.fail(f"total text is longer than {self.max_total_chars} characters")

    def count_node(self) -> None:
        self.nodes += 1
        if self.nodes > self.max_nodes:
            self.fail(f"more than {self.max_nodes} values")

    def walk(self, value: Any, depth: int) -> Any:
        self.count_node()
        t = type(value)
        if value is None:
            return None
        if t is bool:
            return value
        if t is str:
            self.add_string(value, "string")
            return value
        if t is int:
            if value.bit_length() > _MAX_INT_BITS or abs(value) >= _MAX_INT_ABS:
                self.fail(f"integer has more than {MAX_INT_DIGITS} digits")
            return value
        if t is float:
            if not math.isfinite(value):
                self.fail("float is not finite (NaN and infinity are not allowed)")
            return value
        if t is dict or t is list or t is tuple:
            if depth > self.max_depth:
                self.fail(f"nested deeper than {self.max_depth} levels")
            ident = id(value)
            if ident in self.active:
                self.fail("the data refers to itself (cycle)")
            self.active.add(ident)
            try:
                if t is dict:
                    return self.walk_dict(value, depth)
                return self.walk_seq(value, depth)
            finally:
                self.active.discard(ident)
        self.fail(f"unsupported type {_safe_type_name(value)}")

    def walk_dict(self, value: dict[Any, Any], depth: int) -> FrozenMap:
        if len(value) > self.max_nodes - self.nodes:
            self.fail(f"more than {self.max_nodes} values")
        items = list(value.items())  # one snapshot; str keys hash without user code
        out: dict[str, Any] = {}
        for key, item in items:
            if type(key) is not str:
                self.fail(f"dict key of type {_safe_type_name(key)} (keys must be str)")
            self.add_string(key, "key")
            self.path.append("." + ascii(key[:_KEY_CHARS]))
            try:
                out[key] = self.walk(item, depth + 1)
            finally:
                self.path.pop()
        return FrozenMap(_TOKEN, out)

    def walk_seq(self, value: Any, depth: int) -> tuple[Any, ...]:
        if len(value) > self.max_nodes - self.nodes:
            self.fail(f"more than {self.max_nodes} values")
        items = tuple(value)  # snapshot of an exact list or tuple
        out = []
        for index, item in enumerate(items):
            self.path.append(f"[{index}]")
            try:
                out.append(self.walk(item, depth + 1))
            finally:
                self.path.pop()
        return tuple(out)


def _check_limit(name: str, value: Any, upper: Optional[int] = None) -> None:
    if type(value) is not int or value < 1:
        raise FreezeError(f"{name} must be a positive int")
    if upper is not None and value > upper:
        raise FreezeError(f"{name} must be at most {upper}")


def freeze_args(
    arguments: Any,
    *,
    max_depth: int = 16,
    max_nodes: int = 10_000,
    max_string_chars: int = 1_000_000,
    max_total_chars: int = 4_000_000,
) -> FrozenArgs:
    """
    Check ``arguments`` and return a deeply immutable copy as ``FrozenArgs``.

    ``arguments`` must be a plain ``dict`` with str keys holding plain JSON-like
    data (see the module text). The top level counts as depth 1, so the default
    ``max_depth=16`` allows 15 levels of nesting below it. ``max_depth`` can be
    raised to at most 100. Raises ``FreezeError`` on anything else.
    """
    _check_limit("max_depth", max_depth, _HARD_MAX_DEPTH)
    _check_limit("max_nodes", max_nodes)
    _check_limit("max_string_chars", max_string_chars)
    _check_limit("max_total_chars", max_total_chars)
    if type(arguments) is not dict:
        raise FreezeError(
            f"arguments must be a plain dict, got {_safe_type_name(arguments)}"
        )
    walker = _Walker(max_depth, max_nodes, max_string_chars, max_total_chars)
    try:
        root = walker.walk(arguments, 1)
    except RecursionError:
        raise FreezeError("arguments are nested too deeply to read safely") from None
    except FreezeError:
        raise
    except Exception as exc:  # e.g. RuntimeError: dict changed size during iteration
        raise FreezeError(
            f"arguments could not be read safely ({_safe_type_name(exc)})"
        ) from None
    return FrozenArgs(_TOKEN, root)
