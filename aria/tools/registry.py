"""
The tool registry
-----------------
Holds the tools an agent may be given, each with its own immutable copy of its
spec. The pipeline reads specs from here, never from the function, so changing
``fn.__aria_tool_spec__`` or the original spec object after registration has
no effect.

A tool registered without any spec is stored as UNDECLARED. The registry still
returns its callable from ``get`` (so a later stage can look it up and deny it
by name) but ``spec`` returns ``None`` for it and ``undeclared()`` lists it.
Nothing here denies anything; that is the job of the pipeline stories.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from .decorator import declared_spec
from .spec import Need, ToolSpec, ToolSpecError, _build_spec, validate_tool_name

__all__ = ["ToolRegistry"]


def _copy_spec(spec: Any, name: Optional[str] = None) -> ToolSpec:
    """Build a brand new ToolSpec (and new Needs) from ``spec``, validating again.

    The copy is made from the fields, so a spec that was tampered with after
    its creation (for example through ``object.__setattr__``) is rejected here
    instead of being trusted."""
    if type(spec) is not ToolSpec:
        raise ToolSpecError(f"spec must be a ToolSpec, got {type(spec).__name__}")
    needs = spec.needs
    if type(needs) is not tuple:
        raise ToolSpecError("spec.needs must be a tuple")
    copied = []
    for need in needs:
        if type(need) is not Need:
            raise ToolSpecError(f"spec.needs holds a {type(need).__name__}, expected Need")
        copied.append(Need(need.capability, need.scope, need.scope_from))
    return _build_spec(
        spec.internal,
        name=spec.name if name is None else name,
        needs=tuple(copied),
        timeout_s=spec.timeout_s,
        concurrency_safe=spec.concurrency_safe,
        approval=spec.approval,
        description=spec.description,
    )


class ToolRegistry:
    """Maps tool names to callables and to their (immutable) specs."""

    def __init__(self, internal_names: Optional[frozenset[str]] = None) -> None:
        """
        ``internal_names`` is a frozenset of exact tool names that the registry
        reports as ``internal=True`` when they are registered WITH a declared
        spec. This is the only way a tool becomes internal. Internal means "part
        of the framework itself"; it is NOT a permission exemption. A name in
        the set that is registered without a spec stays undeclared.
        """
        if internal_names is None:
            internal_names = frozenset()
        if type(internal_names) is not frozenset:
            raise ToolSpecError(
                f"internal_names must be a frozenset, got {type(internal_names).__name__}"
            )
        self._internal_names: frozenset[str] = frozenset(
            validate_tool_name(n) for n in internal_names
        )
        self._lock = threading.Lock()
        self._tools: dict[str, Callable[..., Any]] = {}
        self._specs: dict[str, Optional[ToolSpec]] = {}

    def register(
        self,
        fn: Callable[..., Any],
        spec: Optional[ToolSpec] = None,
        *,
        name: Optional[str] = None,
    ) -> Optional[ToolSpec]:
        """
        Register ``fn``. Returns the registry's copy of the spec, or ``None``
        when the tool is undeclared.

        Which spec: an explicit ``spec`` argument wins over the function's
        ``__aria_tool_spec__``. If neither exists the tool is undeclared. A spec
        found on ``fn`` is accepted only if ``fn`` owns it (see
        ``aria.tools.decorator``): a ``functools.wraps`` wrapper or a subclass
        that inherited the attribute raises, and must declare its own needs.
        Which name: ``name`` if given, otherwise the spec's name, otherwise the
        function's ``__name__``. When ``name`` differs from the spec's name the
        copy is stored under ``name`` with ``spec.name`` set to match, so the
        registry key and ``spec.name`` are always the same.

        Raises ``ToolSpecError`` for a non-callable, a bad or duplicate name,
        a spec that is not a ``ToolSpec`` (or was tampered with), or a
        malformed ``__aria_tool_spec__`` attribute.
        """
        if not callable(fn):
            raise ToolSpecError(f"a tool must be callable, got {type(fn).__name__}")

        if spec is None:
            # Raises for a wrapper or subclass that inherited a spec it does not own.
            spec = declared_spec(fn)

        if name is not None and type(name) is not str:
            raise ToolSpecError(f"name must be a str, got {type(name).__name__}")

        stored: Optional[ToolSpec]
        if spec is not None:
            key = name if name is not None else getattr(spec, "name", None)
            if type(key) is not str:
                raise ToolSpecError("spec.name must be a str")
            if getattr(spec, "internal", None) is not False:
                raise ToolSpecError(
                    "a spec cannot be internal. Internal status is granted by the registry "
                    "(internal_names) and is not a permission exemption"
                )
            stored = _copy_spec(spec, key)
            if key in self._internal_names:
                stored = _build_spec(
                    True,
                    name=stored.name,
                    needs=stored.needs,
                    timeout_s=stored.timeout_s,
                    concurrency_safe=stored.concurrency_safe,
                    approval=stored.approval,
                    description=stored.description,
                )
        else:
            if name is not None:
                key = name
            else:
                try:
                    key = getattr(fn, "__name__", None)
                except Exception:
                    key = None
                if type(key) is not str:
                    raise ToolSpecError("cannot work out a tool name; pass name=...")
            validate_tool_name(key)  # same rule for an undeclared tool's name
            stored = None

        with self._lock:
            if key in self._tools:
                raise ToolSpecError(f"a tool named {key!r} is already registered")
            self._tools[key] = fn
            self._specs[key] = stored
        return None if stored is None else _copy_spec(stored)

    def get(self, name: object) -> Optional[Callable[..., Any]]:
        """The callable registered under ``name``, declared or not. ``None`` if unknown."""
        if type(name) is not str:
            return None
        return self._tools.get(name)

    def spec(self, name: object) -> Optional[ToolSpec]:
        """A copy of the spec, or ``None`` for an undeclared OR unknown tool.

        Use ``name in registry`` or ``get`` to tell those two apart."""
        if type(name) is not str:
            return None
        stored = self._specs.get(name)
        return None if stored is None else _copy_spec(stored)

    def is_declared(self, name: object) -> bool:
        return type(name) is str and self._specs.get(name) is not None

    def names(self) -> list[str]:
        """All registered names, in registration order."""
        with self._lock:
            return list(self._tools)

    def undeclared(self) -> list[str]:
        """Names of tools that have no spec, in registration order."""
        with self._lock:
            return [n for n, s in self._specs.items() if s is None]

    def __contains__(self, name: object) -> bool:
        return type(name) is str and name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def describe(self) -> list[dict[str, Any]]:
        """A JSON-safe description of every tool, in registration order."""
        with self._lock:
            items = list(self._specs.items())
        out: list[dict[str, Any]] = []
        for name, spec in items:
            if spec is None:
                out.append(
                    {
                        "name": name,
                        "declared": False,
                        "risk": "unknown",
                        "needs": [],
                        "timeout_s": None,
                        "concurrency_safe": None,
                        "approval": None,
                        "internal": None,
                        "description": "",
                    }
                )
            else:
                out.append(
                    {
                        "name": name,
                        "declared": True,
                        "risk": spec.risk,
                        "needs": [n.capability.value for n in spec.needs],
                        "timeout_s": spec.timeout_s,
                        "concurrency_safe": spec.concurrency_safe,
                        "approval": spec.approval,
                        "internal": spec.internal,
                        "description": spec.description,
                    }
                )
        return out
