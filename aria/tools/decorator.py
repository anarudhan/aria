"""
The ``@tool`` decorator
-----------------------
Attaches a frozen ``ToolSpec`` to a function as ``__aria_tool_spec__`` and
returns the SAME function. The signature and the behaviour of the function do
not change, so it works on sync and async functions and on methods defined in
a class (the attribute goes on the function object).

    @tool(Capability.WEB_BROWSE, scope_from="url")
    async def fetch(url: str) -> str: ...

    @tool(Capability.FILE_READ, Capability.MEMORY_WRITE, timeout_s=10)
    def summarise(path: str) -> str: ...

Declaring does not enforce anything. The registry and the pipeline read the
spec later.

A spec belongs to the object it was declared on
-----------------------------------------------
``functools.wraps`` copies ``__dict__``, and a subclass inherits class
attributes, so a wrapper or a subclass would otherwise carry the attribute of
the tool it wraps and run under that tool's (narrower) declared needs. To stop
that the decorator also records the OWNER of each spec in a private, weak
table keyed by object identity. ``functools.wraps`` cannot copy that table.
``declared_spec`` accepts a spec only when the owner is the object itself, the
function of a bound method, or the class of an instance. A wrapper or subclass
must declare its own needs with its own ``@tool`` (or an explicit spec).
"""

from __future__ import annotations

import threading
import types
import weakref
from typing import Any, Callable, Optional, TypeVar, Union

from aria.permissions.engine import Capability

from .spec import Need, ScopeFrom, ToolSpec, ToolSpecError

__all__ = ["SPEC_ATTRIBUTE", "declared_spec", "tool"]

SPEC_ATTRIBUTE = "__aria_tool_spec__"

F = TypeVar("F", bound=Callable[..., Any])

_MISSING = object()

# id(owner) -> (weak reference to owner, spec). Keyed by id so that hashing or
# comparing a foreign object never runs its code; the weak reference proves the
# id still belongs to the same object.
_OWNERS: dict[int, tuple[weakref.ref[Any], ToolSpec]] = {}
_OWNERS_LOCK = threading.Lock()


def _owned_spec(obj: object) -> Optional[ToolSpec]:
    entry = _OWNERS.get(id(obj))
    if entry is not None and entry[0]() is obj:
        return entry[1]
    return None


def _bind_owner(owner: object, spec: ToolSpec) -> None:
    key = id(owner)

    def forget(ref: weakref.ref[Any], key: int = key) -> None:
        with _OWNERS_LOCK:
            entry = _OWNERS.get(key)
            if entry is not None and entry[0] is ref:
                del _OWNERS[key]

    ref = weakref.ref(owner, forget)
    with _OWNERS_LOCK:
        _OWNERS[key] = (ref, spec)


def declared_spec(fn: object) -> Optional[ToolSpec]:
    """
    The spec declared for ``fn`` with ``@tool``, or ``None`` if it has none.

    Raises ``ToolSpecError`` when ``fn`` carries a spec attribute that it does
    not own (a ``functools.wraps`` wrapper or a subclass of a declared tool), or
    when the attribute does not match what ``@tool`` recorded.
    """
    try:
        attached = getattr(fn, SPEC_ATTRIBUTE, _MISSING)
    except Exception:
        raise ToolSpecError(f"reading {SPEC_ATTRIBUTE} from the tool failed") from None

    owned: Optional[ToolSpec] = _owned_spec(fn)
    if owned is None and type(fn) is types.MethodType:
        owned = _owned_spec(fn.__func__)
    if owned is None:
        owned = _owned_spec(type(fn))  # an instance of a decorated class

    if attached is _MISSING:
        if owned is not None:
            raise ToolSpecError(
                f"the {SPEC_ATTRIBUTE} attribute was removed from a declared tool; "
                "refusing to guess"
            )
        return None
    if owned is None:
        raise ToolSpecError(
            f"this object carries a tool spec ({SPEC_ATTRIBUTE}) that it does not own. "
            "It looks like a functools.wraps wrapper, a subclass, or a hand-set attribute. "
            "A wrapper or subclass must declare its own needs: apply @tool to it, or pass "
            "an explicit spec when registering"
        )
    if attached is not owned:
        raise ToolSpecError(
            f"the {SPEC_ATTRIBUTE} attribute was replaced after @tool set it; refusing to use it"
        )
    return owned


def tool(
    *needs_or_capabilities: Union[Capability, Need],
    name: Optional[str] = None,
    scope: str = "*",
    scope_from: Optional[ScopeFrom] = None,
    timeout_s: float = 30.0,
    concurrency_safe: bool = False,
    approval: str = "auto",
    internal: bool = False,
    description: str = "",
) -> Callable[[F], F]:
    """
    Declare what a tool needs.

    Positional arguments are ``Capability`` members (each becomes a ``Need``
    using ``scope`` or ``scope_from``) and/or ready made ``Need`` objects
    (used exactly as given). ``scope`` and ``scope_from`` apply only to the
    bare capabilities; passing them with no bare capability is an error
    because they would silently do nothing.

    The spec is built and validated when ``tool(...)`` is called, so a bad
    declaration fails at import time, not at the first call.

    ``internal=True`` always raises ``ToolSpecError``: a tool cannot declare
    itself internal. Only the registry grants that (``internal_names``), and
    it is not a permission exemption.
    """
    if len(needs_or_capabilities) == 1 and callable(needs_or_capabilities[0]) and not isinstance(
        needs_or_capabilities[0], (Capability, Need)
    ):
        raise ToolSpecError("use @tool(...) with parentheses, for example @tool(Capability.FILE_READ)")

    needs: list[Need] = []
    bare = 0
    for item in needs_or_capabilities:
        if isinstance(item, Capability):
            needs.append(Need(item, scope=scope, scope_from=scope_from))
            bare += 1
        elif type(item) is Need:
            needs.append(item)
        else:
            raise ToolSpecError(
                f"@tool takes Capability or Need arguments, got {type(item).__name__}"
            )
    if bare == 0 and (scope != "*" or scope_from is not None):
        raise ToolSpecError(
            "scope and scope_from apply to bare Capability arguments, but none was given. "
            "Put the scope on the Need instead"
        )

    spec_args: dict[str, Any] = dict(
        needs=tuple(needs),
        timeout_s=timeout_s,
        concurrency_safe=concurrency_safe,
        approval=approval,
        internal=internal,
        description=description,
    )
    # Validate everything except the default name now, so a bad value fails here.
    ToolSpec(name=name if name is not None else "_", **spec_args)

    def decorate(fn: F) -> F:
        if not callable(fn):
            raise ToolSpecError(
                f"@tool must be applied to a callable, got {type(fn).__name__}. "
                "With @classmethod or @staticmethod put @tool below them."
            )
        if _owned_spec(fn) is not None:
            raise ToolSpecError("this function already has a tool spec; do not apply @tool twice")
        try:
            weakref.ref(fn)
        except TypeError:
            raise ToolSpecError(
                f"@tool cannot bind a spec to a {type(fn).__name__} object. Apply @tool to the "
                "function itself, below @classmethod or @staticmethod"
            ) from None
        final_name = name if name is not None else getattr(fn, "__name__", None)
        if type(final_name) is not str:
            raise ToolSpecError("cannot work out a tool name; pass name=... to @tool")
        spec = ToolSpec(name=final_name, **spec_args)
        try:
            setattr(fn, SPEC_ATTRIBUTE, spec)  # replaces a copy made by functools.wraps
        except (AttributeError, TypeError):
            raise ToolSpecError(
                "cannot attach a tool spec to this object. Decorate the function itself, "
                "not a bound method or a builtin"
            ) from None
        _bind_owner(fn, spec)
        return fn

    return decorate
