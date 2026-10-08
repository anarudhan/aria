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

Internal tools
--------------
``ToolRegistry(internal_tools={"read_spilled": the_callable})`` is how the
framework says which callables are part of itself. A tool becomes ``internal``
only when the object registered under that name IS (identity, ``is``) the
callable the framework supplied. A different object that uses a reserved name
is refused with ``ToolSpecError``, whichever is registered first. ``internal``
is NOT a permission exemption; see ``ToolSpec``.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Mapping
from typing import Any, Callable, Optional

from .decorator import declared_spec
from .spec import Need, ToolSpec, ToolSpecError, _build_spec, validate_tool_name

__all__ = ["ToolRegistry"]


def _copy_spec(spec: Any, name: Optional[str] = None) -> ToolSpec:
    """Build a brand new ToolSpec (and new Needs) from ``spec``, validating again.

    The copy is made from the fields, so a spec that was tampered with after
    its creation (for example through ``object.__setattr__``) is rejected here
    instead of being trusted. A non-internal spec with no needs is rejected."""
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


def _check_scope_from_parameters(fn: Callable[..., Any], spec: ToolSpec) -> None:
    """A string ``scope_from`` must name a parameter the tool really receives.

    Otherwise the scope would come from one argument while the tool uses another.
    The tool is called by keyword, so the name must be a parameter that accepts a
    keyword, or the tool must take ``**kwargs``. If the signature cannot be read
    this fails closed. The check covers only the NAME; it cannot prove the tool
    uses that argument for what the scope says."""
    names = [n.scope_from for n in spec.needs if isinstance(n.scope_from, str)]
    if not names:
        return
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        raise ToolSpecError(
            f"cannot read the signature of tool {spec.name!r}, so scope_from "
            f"{names[0]!r} cannot be checked. Refusing to register it"
        ) from None
    parameters = signature.parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return
    accepted = {
        p.name
        for p in parameters
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    for name in names:
        if name not in accepted:
            raise ToolSpecError(
                f"scope_from {name!r} is not a keyword parameter of tool {spec.name!r} "
                f"(it has {sorted(accepted) or 'none'}). The scope would come from an "
                "argument the tool does not use"
            )


class ToolRegistry:
    """Maps tool names to callables and to their (immutable) specs."""

    def __init__(self, internal_tools: Optional[Mapping[str, Callable[..., Any]]] = None) -> None:
        """
        ``internal_tools`` maps reserved tool names to the exact callables that
        are part of the framework. Rules:

        - A tool is ``internal`` only if the object registered under a reserved
          name IS the callable given here (``is``). Another object using that
          name raises ``ToolSpecError``, in either registration order.
        - An internal tool is always declared. With no spec it gets an internal
          spec with no needs; with a spec it keeps the needs it declares.
          Only internal tools may have no needs.
        - Names are validated like tool names and the values must be callable.
          The mapping is copied.
        - Internal is NOT a permission exemption and the registry's creator
          decides the callables; a registry built by user code can mark its own
          callables internal, so the pipeline must be built with the framework's
          own registry.
        """
        internal: dict[str, Callable[..., Any]] = {}
        if internal_tools is not None:
            if not isinstance(internal_tools, Mapping):
                raise ToolSpecError(
                    f"internal_tools must be a mapping, got {type(internal_tools).__name__}"
                )
            for key, value in list(internal_tools.items()):
                validate_tool_name(key)
                if not callable(value):
                    raise ToolSpecError(f"internal tool {key!r} must be callable")
                internal[key] = value
        self._internal_tools = internal
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
        reserved name of an internal tool, otherwise the function's ``__name__``.
        When ``name`` differs from the spec's name the copy is stored under
        ``name`` with ``spec.name`` set to match.

        A string ``scope_from`` must name a keyword parameter of ``fn`` (or ``fn``
        must take ``**kwargs``), otherwise ``ToolSpecError``.

        Raises ``ToolSpecError`` for a non-callable, a bad or duplicate name,
        a reserved internal name used by another object, a spec that is not a
        ``ToolSpec`` (or was tampered with), or a malformed ``__aria_tool_spec__``.
        """
        if not callable(fn):
            raise ToolSpecError(f"a tool must be callable, got {type(fn).__name__}")

        if spec is None:
            # Raises for a wrapper or subclass that inherited a spec it does not own.
            spec = declared_spec(fn)

        if name is not None and type(name) is not str:
            raise ToolSpecError(f"name must be a str, got {type(name).__name__}")

        reserved_for_fn = next((k for k, v in self._internal_tools.items() if v is fn), None)

        key: Any
        if name is not None:
            key = name
        elif spec is not None:
            key = getattr(spec, "name", None)
        elif reserved_for_fn is not None:
            key = reserved_for_fn
        else:
            try:
                key = getattr(fn, "__name__", None)
            except Exception:
                key = None
        if type(key) is not str:
            raise ToolSpecError("cannot work out a tool name; pass name=...")
        validate_tool_name(key)

        is_internal = False
        if key in self._internal_tools:
            if self._internal_tools[key] is not fn:
                raise ToolSpecError(
                    f"the name {key!r} is reserved for an internal tool and this is a "
                    "different object"
                )
            is_internal = True

        stored: Optional[ToolSpec]
        if spec is not None:
            if getattr(spec, "internal", None) is not False:
                raise ToolSpecError(
                    "a spec cannot be internal. Internal status is granted by the registry "
                    "(internal_tools) and is not a permission exemption"
                )
            stored = _copy_spec(spec, key)
            if is_internal:
                stored = _build_spec(
                    True,
                    name=stored.name,
                    needs=stored.needs,
                    timeout_s=stored.timeout_s,
                    concurrency_safe=stored.concurrency_safe,
                    approval=stored.approval,
                    description=stored.description,
                )
        elif is_internal:
            stored = _build_spec(True, name=key, needs=())
        else:
            stored = None

        if stored is not None:
            _check_scope_from_parameters(fn, stored)

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
