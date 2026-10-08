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
attribute later.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar, Union

from aria.permissions.engine import Capability

from .spec import Need, ScopeFrom, ToolSpec, ToolSpecError

__all__ = ["SPEC_ATTRIBUTE", "tool"]

SPEC_ATTRIBUTE = "__aria_tool_spec__"

F = TypeVar("F", bound=Callable[..., Any])


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
        if SPEC_ATTRIBUTE in getattr(fn, "__dict__", {}):
            raise ToolSpecError("this function already has a tool spec; do not apply @tool twice")
        final_name = name if name is not None else getattr(fn, "__name__", None)
        if type(final_name) is not str:
            raise ToolSpecError("cannot work out a tool name; pass name=... to @tool")
        spec = ToolSpec(name=final_name, **spec_args)
        try:
            setattr(fn, SPEC_ATTRIBUTE, spec)
        except (AttributeError, TypeError):
            raise ToolSpecError(
                "cannot attach a tool spec to this object. Decorate the function itself, "
                "not a bound method or a builtin"
            ) from None
        return fn

    return decorate
