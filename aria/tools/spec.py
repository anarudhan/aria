"""
ARIA Tool Specs
---------------
The declaration layer of the guarded tool pipeline (issue #42, story #59).

A ``ToolSpec`` says what a tool needs before it may run: which capabilities,
over which scope, how long it may take, whether calls may overlap and whether
a human must approve each call. A ``Need`` is one capability plus the scope it
is needed over.

This module only describes and validates. Nothing here checks a permission or
runs a tool. The later pipeline stories (#60 and #63) read these objects and
enforce them.

Everything fails closed: a bad value raises ``ToolSpecError`` when the spec is
built, and a scope that cannot be worked out raises ``ToolSpecError`` when it
is resolved. A caller that gets ``ToolSpecError`` from ``Need.resolve_scope``
MUST deny the call. It must never fall back to "*" or to "no scope".
"""

from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Union

from aria.permissions.engine import RISK_LEVEL, Capability

from .freeze import FrozenArgs, FrozenMap

__all__ = [
    "APPROVAL_MODES",
    "MAX_NEEDS",
    "MAX_SCOPE_CHARS",
    "MAX_TIMEOUT_S",
    "RISK_ORDER",
    "TOOL_NAME_PATTERN",
    "Need",
    "ToolSpec",
    "ToolSpecError",
    "validate_tool_name",
]

# \Z, not $: "$" also matches before a trailing newline, so "abc\n" would pass re.match.
TOOL_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_.\-]{0,63}\Z"
_TOOL_NAME_RE = re.compile(TOOL_NAME_PATTERN)

APPROVAL_MODES = ("auto", "always")
MAX_TIMEOUT_S = 300.0
MAX_NEEDS = 32
MAX_SCOPE_CHARS = 4096
MAX_DESCRIPTION_CHARS = 2000

# Risk levels from lowest to highest. The values are the ones RISK_LEVEL uses.
RISK_ORDER = ("low", "medium", "high", "critical")
_RISK_RANK = {name: rank for rank, name in enumerate(RISK_ORDER)}

ScopeFrom = Union[str, Callable[[Mapping[str, Any]], str]]


class ToolSpecError(ValueError):
    """A tool declaration is invalid, or a scope could not be resolved.

    From ``Need.resolve_scope`` this means DENY the call."""


def _is_exact_str(value: object) -> bool:
    # Exact type, not isinstance: a str subclass can override methods.
    return type(value) is str


def _type_name(value: object) -> str:
    return type(value).__name__


def validate_tool_name(name: object) -> str:
    """Return ``name`` if it is a valid tool name, else raise ToolSpecError."""
    if type(name) is not str:
        raise ToolSpecError(f"tool name must be a str, got {_type_name(name)}")
    if _TOOL_NAME_RE.fullmatch(name) is None:
        raise ToolSpecError(
            f"tool name {name[:70]!r} is not valid. Use 1 to 64 characters "
            "from A-Z a-z 0-9 _ . - and start with a letter or underscore "
            f"(pattern {TOOL_NAME_PATTERN})"
        )
    return name


def _check_scope_text(value: str, label: str) -> None:
    if not value or not value.strip():
        raise ToolSpecError(f"{label} must not be empty")
    if len(value) > MAX_SCOPE_CHARS:
        raise ToolSpecError(f"{label} is longer than {MAX_SCOPE_CHARS} characters")


@dataclass(frozen=True, slots=True)
class Need:
    """
    One capability a tool needs, and the scope it needs it over.

    ``scope`` is a fixed scope string and defaults to ``"*"`` (the wildcard:
    only a grant for ``*`` covers it). ``scope_from`` is the other way to give
    a scope: it names the ARGUMENT that holds the scope (a string), or it is a
    callable that computes the scope from the frozen arguments. Use one or the
    other. Setting both raises ``ToolSpecError``.

    The resolved scope is only a string here. Whether it is a safe path, URL
    or id is decided by ``aria.permissions.scope`` in the pipeline stage (it
    must call ``canonical_scope`` on every resolved scope; this layer only
    checks that the text is non-empty and at most 4096 characters).

    A ``scope_from`` callable must be pure, fast and deterministic. It is not
    copied (the spec holds the same object), not timed and not sandboxed. The
    pipeline must resolve each Need exactly once per call and pass that same
    string through every later stage; it must never call the callable again.
    """

    capability: Capability
    scope: str = "*"
    scope_from: Optional[ScopeFrom] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("Need cannot be subclassed")

    def __post_init__(self) -> None:
        if not isinstance(self.capability, Capability):
            raise ToolSpecError(
                f"Need.capability must be a Capability member, got {_type_name(self.capability)}"
            )
        if not _is_exact_str(self.scope):
            raise ToolSpecError(f"Need.scope must be a str, got {_type_name(self.scope)}")
        _check_scope_text(self.scope, "Need.scope")
        source = self.scope_from
        if source is None:
            return
        if self.scope != "*":
            raise ToolSpecError(
                "Need takes either a fixed scope or scope_from, not both "
                f"(scope={self.scope[:40]!r} and scope_from set)"
            )
        if isinstance(source, str):
            if not _is_exact_str(source) or not source.strip():
                raise ToolSpecError("Need.scope_from must name an argument (a non-empty str)")
            if len(source) > 256:
                raise ToolSpecError("Need.scope_from argument name is too long (max 256)")
        elif callable(source):
            if inspect.iscoroutinefunction(source):
                raise ToolSpecError("Need.scope_from must be a plain function, not async")
        else:
            raise ToolSpecError(
                "Need.scope_from must be an argument name (str) or a callable, "
                f"got {_type_name(source)}"
            )

    def resolve_scope(self, arguments: Mapping[str, Any]) -> str:
        """
        Return the scope for a call with these frozen arguments.

        ``arguments`` must be the ``FrozenArgs`` (or ``FrozenMap``) built by
        ``freeze_args``. A plain dict is refused, because a callable could
        change it. Raises ``ToolSpecError`` when the named argument is missing, is not a
        str, is empty, or when the callable raises or returns a non-str or an
        empty str. THE CALLER MUST DENY THE CALL ON ToolSpecError. Messages
        carry argument names and type names, never argument values.
        """
        source = self.scope_from
        if source is None:
            return self.scope
        if type(arguments) is not FrozenArgs and type(arguments) is not FrozenMap:
            raise ToolSpecError(
                "cannot resolve scope: arguments must be the FrozenArgs from freeze_args, "
                f"got {_type_name(arguments)}"
            )
        value: Any
        if isinstance(source, str):
            try:
                value = arguments[source]
            except KeyError:
                raise ToolSpecError(
                    f"cannot resolve scope: argument {source!r} is missing"
                ) from None
            except Exception as exc:
                raise ToolSpecError(
                    f"cannot resolve scope: reading argument {source!r} failed "
                    f"({_type_name(exc)})"
                ) from None
            where = f"argument {source!r}"
        else:
            try:
                value = source(arguments)
            except Exception as exc:
                raise ToolSpecError(
                    f"cannot resolve scope: scope_from callable raised {_type_name(exc)}"
                ) from None
            if inspect.iscoroutine(value):
                value.close()
                raise ToolSpecError(
                    "cannot resolve scope: scope_from callable returned a coroutine"
                )
            where = "scope_from callable result"
        if not _is_exact_str(value):
            raise ToolSpecError(
                f"cannot resolve scope: {where} must be a str, got {_type_name(value)}"
            )
        try:
            _check_scope_text(value, where)
        except ToolSpecError as exc:
            raise ToolSpecError(f"cannot resolve scope: {exc}") from None
        return str(value)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """
    What a tool declares about itself. Immutable and validated on creation.

    ``needs`` is a tuple of ``Need``. It may be empty only when
    ``internal=True``: a tool that touches the outside world must say what it
    needs. ``timeout_s`` is a finite number above 0 and at most 300.
    ``approval`` is ``"auto"`` or ``"always"`` (a human approves every call).

    ``internal`` cannot be set by user code: ``ToolSpec(internal=True)`` raises.
    The registry sets it only for the exact callables it was given in ``internal_tools``.
    INTERNAL IS NOT A PERMISSION EXEMPTION. It only says the tool is part of
    the framework itself (for example the spill read-back). A pipeline must
    never treat it as authority to skip a permission check.
    """

    name: str
    needs: tuple[Need, ...]
    timeout_s: float = 30.0
    concurrency_safe: bool = False
    approval: str = "auto"
    internal: bool = False
    description: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("ToolSpec cannot be subclassed")

    def __post_init__(self) -> None:
        _validate_spec(self, allow_internal=False)

    @property
    def risk(self) -> str:
        """The highest risk among the needed capabilities ("low" if there are none).

        A capability with no known risk level counts as "critical", so a gap in
        RISK_LEVEL makes a tool look more dangerous, never less."""
        unknown = len(RISK_ORDER) - 1
        highest = 0
        for need in self.needs:
            level = RISK_LEVEL.get(need.capability)
            rank = _RISK_RANK.get(level, unknown) if isinstance(level, str) else unknown
            highest = max(highest, rank)
        return RISK_ORDER[highest]


def _validate_spec(spec: ToolSpec, allow_internal: bool) -> None:
    """Check every field of ``spec``. ``__post_init__`` calls this with
    allow_internal=False; only the registry's ``_build_spec`` passes True.

    There is deliberately no flag or window that user code could reach while the
    registry builds a spec: the permission to be internal is an argument, not state."""
    validate_tool_name(spec.name)

    if type(spec.needs) is not tuple:
        raise ToolSpecError(f"ToolSpec.needs must be a tuple, got {_type_name(spec.needs)}")
    if len(spec.needs) > MAX_NEEDS:
        raise ToolSpecError(f"ToolSpec.needs has more than {MAX_NEEDS} entries")
    for index, need in enumerate(spec.needs):
        if type(need) is not Need:
            raise ToolSpecError(
                f"ToolSpec.needs[{index}] must be a Need, got {_type_name(need)}"
            )

    if type(spec.internal) is not bool:
        raise ToolSpecError(
            f"ToolSpec.internal must be a bool, got {_type_name(spec.internal)}"
        )
    if type(spec.concurrency_safe) is not bool:
        raise ToolSpecError(
            f"ToolSpec.concurrency_safe must be a bool, got {_type_name(spec.concurrency_safe)}"
        )
    if spec.internal and not allow_internal:
        raise ToolSpecError(
            "a tool cannot declare itself internal. Internal status is granted only by "
            "the registry (ToolRegistry(internal_tools=...)) and is not a permission exemption"
        )
    if not spec.needs and not spec.internal:
        raise ToolSpecError(
            f"tool {spec.name!r} declares no needs. A tool that is not internal must "
            "declare at least one capability it needs"
        )

    timeout = spec.timeout_s
    if type(timeout) not in (int, float):  # also rejects bool, Decimal, numpy scalars
        raise ToolSpecError(
            f"ToolSpec.timeout_s must be an int or float, got {_type_name(timeout)}"
        )
    try:
        finite = math.isfinite(timeout)
    except OverflowError:  # an int too large for a float
        finite = False
    if not finite:
        raise ToolSpecError("ToolSpec.timeout_s must be finite")
    if not 0 < timeout <= MAX_TIMEOUT_S:
        raise ToolSpecError(
            f"ToolSpec.timeout_s must be above 0 and at most {MAX_TIMEOUT_S:g}"
        )
    object.__setattr__(spec, "timeout_s", float(timeout))

    if not _is_exact_str(spec.approval):
        raise ToolSpecError(f"ToolSpec.approval must be a str, got {_type_name(spec.approval)}")
    if spec.approval not in APPROVAL_MODES:
        raise ToolSpecError(
            f"ToolSpec.approval must be one of {APPROVAL_MODES}, got {spec.approval[:20]!r}"
        )

    if not _is_exact_str(spec.description):
        raise ToolSpecError(
            f"ToolSpec.description must be a str, got {_type_name(spec.description)}"
        )
    if len(spec.description) > MAX_DESCRIPTION_CHARS:
        raise ToolSpecError(f"ToolSpec.description is longer than {MAX_DESCRIPTION_CHARS}")


def _refuse(*_args: Any, **_kwargs: Any) -> Any:
    raise TypeError("this object cannot be pickled or rebuilt from state; construct it normally")


def _same(self: Any, *_args: Any) -> Any:
    return self  # immutable, so a copy is the object itself


def _harden(cls: type) -> None:
    """Copying returns the same (immutable) object; pickling and __setstate__ are refused.

    ``__setstate__`` and ``__getstate__`` are assigned after the class is built because
    ``dataclass(slots=True, frozen=True)`` installs its own and would replace ones written
    in the class body. Without this, ``copy.copy(spec).__setstate__(...)`` or a pickle
    round trip could build a spec that skipped validation."""
    cls.__copy__ = _same  # type: ignore[attr-defined]
    cls.__deepcopy__ = _same  # type: ignore[attr-defined]
    cls.__reduce__ = _refuse  # type: ignore[assignment]
    cls.__reduce_ex__ = _refuse  # type: ignore[assignment]
    cls.__getstate__ = _refuse  # type: ignore[attr-defined]
    cls.__setstate__ = _refuse  # type: ignore[attr-defined]


_harden(Need)
_harden(ToolSpec)


def _build_spec(internal: object, **fields: Any) -> ToolSpec:
    """Registry-private constructor that may set ``internal=True`` (and then allows no needs).

    Builds the object without ``__init__`` and validates it with ``allow_internal=True``.
    User code cannot declare a tool internal: ``ToolSpec(internal=True)`` raises. This is a
    guard against mistakes, not against code that calls this private function."""
    spec = object.__new__(ToolSpec)
    values: dict[str, Any] = {
        "name": None,
        "needs": (),
        "timeout_s": 30.0,
        "concurrency_safe": False,
        "approval": "auto",
        "internal": internal,
        "description": "",
    }
    unknown = set(fields) - set(values)
    if unknown:
        raise ToolSpecError(f"unknown ToolSpec fields: {sorted(unknown)}")
    values.update(fields)
    for key, value in values.items():
        object.__setattr__(spec, key, value)
    _validate_spec(spec, allow_internal=True)
    return spec
