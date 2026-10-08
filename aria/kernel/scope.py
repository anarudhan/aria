"""
ARIA Kernel Scopes
------------------
A ``Scope`` owns the things one part of ARIA registers (event listeners, child
scopes, background tasks, anything with an "undo") and undoes all of them, in a
safe order, when that part is unloaded. It is inspired by the Cordis "fibers
and effects" model, with its known problems fixed. Standard library only.

This gives CLEANUP, not isolation or sandboxing. Code in a scope runs with the
full privileges of the process, and nothing here limits what it can do.

Rules
  * ``scope.effect(fn)`` runs ``fn()`` now and keeps the "undo" it returns. The
    undo may be a callable (sync, or returning an awaitable), a list or tuple
    of callables, or an awaitable that resolves to one of those.
  * Disposal is strictly sequential and in REVERSE registration order. Async
    undos are awaited one by one. (Cordis runs them together, which breaks
    ordering.)
  * Child scopes are effects of their parent. Disposing a parent disposes its
    children first (newest first), then its own effects (newest first). A child
    is a dependent of its parent, so it goes before what the parent set up.
  * Every undo runs on its own. If one raises, the others still run, and the
    error is recorded on ``scope.errors``. Nothing is swallowed silently.
    (Cordis stops at the first error and loses the rest.)
  * Removers and disposers are bound to the effect that was registered, never
    to a shared variable, so removing effect A can never remove effect B.
  * ``asyncio.CancelledError`` is never recorded as a failure and never
    swallowed. Cancellation does not stop an unwinding in progress: the
    remaining undos still run, and then the cancellation is raised.
  * After ``dispose()``, ``effect``, ``on`` and ``child`` raise
    ``ScopeDisposedError``.

A scope is meant to be used from the event loop thread. It is not thread safe.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import time
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
)

from .events import Events, _label_of

DEFAULT_MAX_ERRORS = 1000
STATE_EVENT = "scope/state"

# How many recent errors describe() shows, and how deep it follows children.
_DESCRIBE_ERRORS = 5
_DESCRIBE_DEPTH = 16

Setup = Callable[["Scope"], Any]


class ScopeState(Enum):
    """Lifecycle of a scope."""

    PENDING = "pending"  # created, setup not finished
    ACTIVE = "active"  # setup finished
    FAILED = "failed"  # setup failed, effects already undone
    DISPOSED = "disposed"  # finished for good


class ScopeDisposedError(RuntimeError):
    """The scope is disposed (or being disposed or unwound) and takes no new effects."""


@dataclass(frozen=True)
class ScopeError:
    """One failure inside a scope: where, what was raised, and when (``time.time()``)."""

    scope: str
    label: str
    exception: Exception
    when: float

    def __str__(self) -> str:
        return f"{self.label}: {type(self.exception).__name__}: {self.exception}"


class ChildScopeErrors(Exception):
    """Recorded on a parent when a child scope finished with recorded errors.

    The child is gone after disposal, so this keeps its errors visible: they
    are in ``errors``.
    """

    def __init__(self, scope: str, errors: List[ScopeError], dropped: int = 0) -> None:
        self.scope = scope
        self.errors: Tuple[ScopeError, ...] = tuple(errors)
        self.dropped = dropped
        total = len(self.errors) + dropped
        super().__init__(f"child scope {scope!r} finished with {total} recorded error(s)")


# Scopes whose disposal is running in the current context. A disposer that
# calls dispose() on its own scope would wait for itself; this lets it return
# instead. Tasks started inside a disposer inherit the marker.
_DISPOSING: ContextVar[FrozenSet[int]] = ContextVar("aria_scope_disposing", default=frozenset())


class _Done:
    """An already finished awaitable. Awaiting it does nothing."""

    __slots__ = ()

    def __await__(self) -> Generator[Any, None, None]:
        return
        yield  # pragma: no cover - makes this a generator


_DONE = _Done()


class _Effect:
    __slots__ = ("id", "label", "disposers", "task", "child")

    def __init__(self, ident: int, label: str, disposers: List[Callable[[], Any]]) -> None:
        self.id = ident
        self.label = label
        self.disposers = disposers
        # Set while an async effect has not resolved yet.
        self.task: Optional["asyncio.Future[None]"] = None
        self.child: Optional["Scope"] = None


class Remover:
    """Undoes one effect early. Returned by ``effect`` and ``on``.

    Calling it removes the effect from the scope and runs its undo now. The
    return value is awaitable. If the undo was sync (listeners always are), it
    has already finished and awaiting is optional. If the undo is async, or
    the effect was still resolving, you must ``await`` the result to wait for
    it; otherwise Python warns that a coroutine was never awaited.

    Calling it again does nothing and returns an already finished awaitable.
    It only ever touches the effect it was created for.
    """

    __slots__ = ("_scope", "_effect")

    def __init__(self, scope: "Scope", effect: _Effect) -> None:
        self._scope = scope
        self._effect = effect

    @property
    def active(self) -> bool:
        """True while the effect is still registered."""
        return self._effect.id in self._scope._effects

    def __call__(self) -> Awaitable[None]:
        scope = self._scope
        if scope._effects.pop(self._effect.id, None) is None:
            return _DONE
        return scope._undo(self._effect)


async def _exhaust_asyncgen(agen: Any) -> None:
    async for _ in agen:
        pass


def _discard_awaitable(obj: Any) -> None:
    close = getattr(obj, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # pragma: no cover - best effort only
            pass


class Scope:
    """See the module docstring. Public API:

    ``Scope(events=None, *, name="scope", parent=None, max_errors=1000)``
    ``scope.effect(fn, label="") -> Remover``
    ``scope.on(event, fn, *, prepend=False, once=False) -> Remover``
    ``scope.child(name, *, events=None) -> Scope``
    ``await scope.start(setup=None) -> Scope``
    ``await scope.restart(setup=None) -> Scope``
    ``await scope.dispose() -> None``
    ``scope.state``, ``scope.errors``, ``scope.describe()``
    ``async with scope:`` (start on enter, dispose on exit)
    """

    events: Optional[Events]
    parent: Optional["Scope"]

    def __init__(
        self,
        events: Optional[Events] = None,
        *,
        name: str = "scope",
        parent: Optional["Scope"] = None,
        max_errors: int = DEFAULT_MAX_ERRORS,
    ) -> None:
        if max_errors < 1:
            raise ValueError("max_errors must be at least 1")
        if parent is not None:
            parent._check_open("create a child scope")
            if events is None:
                events = parent.events
        self.name = str(name)
        self.events = events
        self.parent = parent
        self._state = ScopeState.PENDING
        self._ids = itertools.count(1)
        # Insertion ordered: popitem() gives the newest entry first. Child
        # scopes are kept apart from the other effects: a disposal undoes all
        # children (newest first) before the scope's own effects (newest first).
        self._effects: Dict[int, _Effect] = {}
        self._children: Dict[int, _Effect] = {}
        self._errors: Deque[ScopeError] = deque(maxlen=max_errors)
        self._dropped_errors = 0
        self._pending: Set["asyncio.Future[None]"] = set()
        self._starting = False
        self._closing = False  # dispose() was requested
        self._unwinding = False  # an unwind task exists and has not finished
        self._task: Optional["asyncio.Task[None]"] = None
        self._async_failure: Optional[Exception] = None
        self._link: Optional[Tuple["Scope", int]] = None
        if parent is not None:
            parent._attach(self)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def state(self) -> ScopeState:
        return self._state

    @property
    def disposed(self) -> bool:
        return self._state is ScopeState.DISPOSED

    @property
    def effect_count(self) -> int:
        """Live effects of this scope, not counting child scopes."""
        return len(self._effects)

    @property
    def children(self) -> Tuple["Scope", ...]:
        """Live child scopes, oldest first."""
        return tuple(e.child for e in self._children.values() if e.child is not None)

    def __repr__(self) -> str:
        return f"<Scope {self.name!r} {self._state.value} effects={self.effect_count}>"

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    @property
    def errors(self) -> List[ScopeError]:
        """A copy of the recorded errors, oldest first."""
        return list(self._errors)

    def take_errors(self) -> List[ScopeError]:
        """Return the recorded errors and clear them."""
        taken = list(self._errors)
        self._errors.clear()
        return taken

    def clear_errors(self) -> None:
        self._errors.clear()

    def _record(self, label: str, exc: Exception) -> None:
        if len(self._errors) == self._errors.maxlen:
            self._dropped_errors += 1
        self._errors.append(ScopeError(self.name, label, exc, time.time()))

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _set_state(self, new: ScopeState) -> None:
        if self._state is new:
            return
        self._state = new
        if self.events is not None:
            try:
                # Plain emit: observer failures are recorded by Events and
                # never reach the scope.
                self.events.emit(STATE_EVENT, self.name, new)
            except Exception as exc:
                self._record(STATE_EVENT, exc)

    def _check_open(self, what: str) -> None:
        if self._state is ScopeState.DISPOSED or self._closing:
            raise ScopeDisposedError(f"scope {self.name!r} is disposed; cannot {what}")
        if self._unwinding:
            raise ScopeDisposedError(
                f"scope {self.name!r} is being unwound; cannot {what} until it has finished"
            )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _add(self, disposers: List[Callable[[], Any]], label: str) -> _Effect:
        eff = _Effect(next(self._ids), label, disposers)
        self._effects[eff.id] = eff
        return eff

    def _attach(self, child: "Scope") -> None:
        eff = _Effect(next(self._ids), f"scope {child.name}", [child.dispose])
        self._children[eff.id] = eff
        eff.child = child
        child._link = (self, eff.id)

    @staticmethod
    def _normalize(value: Any) -> List[Callable[[], Any]]:
        if value is None:
            return []
        if callable(value):
            return [value]
        if isinstance(value, (list, tuple)):
            out: List[Callable[[], Any]] = []
            for item in value:
                if item is None:
                    continue
                if not callable(item):
                    raise TypeError(
                        f"a disposer list may only hold callables, got {type(item).__name__}"
                    )
                out.append(item)
            return out
        raise TypeError(
            "an effect must return None, a callable, a list or tuple of callables, or an "
            f"awaitable of those, got {type(value).__name__}"
        )

    def effect(self, fn: Callable[[], Any], label: str = "") -> Remover:
        """Run ``fn()`` now and keep the undo it returns.

        ``fn`` may return None, a callable, a list or tuple of callables (run
        newest first), or an awaitable that resolves to one of those. An
        awaitable is started as a task on the running loop, so this needs a
        running event loop in that case. If the awaitable fails, the error is
        recorded on the scope.

        If ``fn`` raises, nothing is registered and the exception propagates.
        If it returns something that is not a valid undo, ``TypeError`` is
        raised; whatever ``fn`` did is then NOT undone, so return a callable.

        Raises ``ScopeDisposedError`` once the scope is disposed or unwinding.
        """
        self._check_open("register an effect")
        if not callable(fn):
            raise TypeError(f"effect needs a callable, got {type(fn).__name__}")
        label = label or _label_of(fn)
        result = fn()
        if inspect.isawaitable(result):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                _discard_awaitable(result)
                raise RuntimeError(
                    f"effect {label!r} returned an awaitable but there is no running event "
                    "loop; register async effects from async code"
                ) from None
            eff = self._add([], label)
            task = loop.create_task(self._resolve(eff, result))
            eff.task = task
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)
        else:
            eff = self._add(self._normalize(result), label)
        return Remover(self, eff)

    async def _resolve(self, eff: _Effect, aw: Awaitable[Any]) -> None:
        try:
            value = await aw
            eff.disposers = self._normalize(value)
        except Exception as exc:
            self._record(eff.label, exc)
            if self._starting and self._async_failure is None:
                self._async_failure = exc

    def on(
        self,
        event: str,
        fn: Callable[..., Any],
        *,
        prepend: bool = False,
        once: bool = False,
    ) -> Remover:
        """Register an ``Events`` listener that is removed when the scope is disposed.

        Needs the scope to have an ``Events`` instance. Returns a ``Remover``.
        A ``once`` listener also drops its scope entry when it fires, so
        one-shot listeners do not pile up.
        """
        self._check_open("register a listener")
        events = self.events
        if events is None:
            raise RuntimeError(f"scope {self.name!r} has no Events instance; cannot register on()")
        if not callable(fn):
            raise TypeError(f"listener must be callable, got {type(fn).__name__}")
        label = _label_of(fn)
        holder: List[_Effect] = []

        def run_once(*args: Any) -> Any:
            # Forget the scope entry first: the listener is gone after this call.
            if holder:
                self._effects.pop(holder[0].id, None)
            return fn(*args)

        listener = run_once if once else fn
        add = events.once if once else events.on
        disposer = add(event, listener, prepend=prepend, label=label)
        eff = self._add([disposer], f"{event}: {label}")
        holder.append(eff)
        return Remover(self, eff)

    def child(self, name: str, *, events: Optional[Events] = None) -> "Scope":
        """Create a child scope. It shares this scope's events unless given its own.

        The child is an effect of this scope: disposing this scope disposes the
        child first. Disposing the child on its own detaches it.
        """
        self._check_open("create a child scope")
        return Scope(events if events is not None else self.events, name=name, parent=self)

    # ------------------------------------------------------------------
    # Undoing one effect
    # ------------------------------------------------------------------

    def _advance(self, label: str, rest: List[Callable[[], Any]]) -> Optional[Awaitable[Any]]:
        """Run disposers from ``rest`` (newest last) until one returns something to await.

        Returns that awaitable, or None when ``rest`` is used up. Errors are
        recorded. Anything that is not awaitable is ignored, except that sync
        and async generators are run to the end: calling a generator function
        does nothing by itself, and silently skipping it would hide the bug.
        """
        while rest:
            disposer = rest.pop()
            try:
                result = disposer()
                if inspect.isawaitable(result):
                    return result
                if inspect.isasyncgen(result):
                    return _exhaust_asyncgen(result)
                if inspect.isgenerator(result):
                    for _ in result:
                        pass
            except Exception as exc:
                self._record(label, exc)
        return None

    def _undo(self, eff: _Effect) -> Awaitable[None]:
        """Undo a claimed effect (already out of ``_effects``). Sync part runs now."""
        task = eff.task
        if task is not None and not task.done():
            return self._undo_after_resolve(eff)
        rest = list(eff.disposers)
        eff.disposers = []
        pending = self._advance(eff.label, rest)
        if pending is None:
            return _DONE
        return self._drain(eff.label, pending, rest)

    async def _drain(
        self, label: str, pending: Optional[Awaitable[Any]], rest: List[Callable[[], Any]]
    ) -> None:
        cancelled: Optional[BaseException] = None
        while pending is not None:
            try:
                await pending
            except asyncio.CancelledError as exc:
                # Cleanup must finish. Remember the cancellation, raise it last.
                cancelled = cancelled or exc
            except Exception as exc:
                self._record(label, exc)
            pending = self._advance(label, rest)
        if cancelled is not None:
            raise cancelled

    async def _undo_after_resolve(self, eff: _Effect) -> None:
        task = eff.task
        assert task is not None
        cancelled: Optional[BaseException] = None
        while not task.done():
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError as exc:
                cancelled = cancelled or exc
        rest = list(eff.disposers)
        eff.disposers = []
        pending = self._advance(eff.label, rest)
        if pending is not None:
            try:
                await self._drain(eff.label, pending, rest)
            except asyncio.CancelledError as exc:
                cancelled = cancelled or exc
        if cancelled is not None:
            raise cancelled

    # ------------------------------------------------------------------
    # Unwinding and disposal
    # ------------------------------------------------------------------

    async def _unwind_loop(self) -> None:
        cancelled: Optional[BaseException] = None
        while self._effects or self._children:
            # Children first, then own effects, each newest first.
            _, eff = (self._children if self._children else self._effects).popitem()
            try:
                aw = self._undo(eff)
                if aw is not _DONE:
                    await aw
            except asyncio.CancelledError as exc:
                cancelled = cancelled or exc
        if cancelled is not None:
            raise cancelled

    async def _run(self) -> None:
        token = _DISPOSING.set(_DISPOSING.get() | {id(self)})
        try:
            await self._unwind_loop()
        finally:
            _DISPOSING.reset(token)
            self._unwinding = False
            if self._closing and not self._has_work():
                self._finish_dispose()

    def _ensure_task(self) -> "asyncio.Task[None]":
        task = self._task
        if task is None or task.done():
            self._unwinding = True
            task = self._task = asyncio.get_running_loop().create_task(self._run())
        return task

    async def _join_unwind(self) -> None:
        """Undo every effect (without disposing), or wait for a running unwind."""
        if id(self) in _DISPOSING.get():
            return
        task = self._task
        if (task is None or task.done()) and not self._has_work():
            return
        while True:
            await asyncio.shield(self._ensure_task())
            task = self._task
            if (task is None or task.done()) and not self._has_work():
                return

    def _has_work(self) -> bool:
        return bool(self._effects or self._children)

    def _finish_dispose(self) -> None:
        link, self._link = self._link, None
        if link is not None:
            parent, ident = link
            parent._children.pop(ident, None)
            if self._errors or self._dropped_errors:
                parent._record(
                    f"scope {self.name}",
                    ChildScopeErrors(self.name, list(self._errors), self._dropped_errors),
                )
        self._set_state(ScopeState.DISPOSED)

    async def dispose(self) -> None:
        """Undo everything, newest first, then mark the scope DISPOSED.

        Idempotent: later calls return at once if it is finished, and wait for
        it if it is still running. Errors from undos do not raise; they are in
        ``errors``. A disposer that calls ``dispose()`` on its own scope gets an
        immediate return instead of a deadlock.

        The unwinding runs as its own task. If the caller is cancelled (for
        example by ``asyncio.wait_for``), it stops waiting, but the unwinding
        carries on until it is done, so nothing is left behind.
        """
        if self._state is ScopeState.DISPOSED or id(self) in _DISPOSING.get():
            return
        self._closing = True
        task = self._task
        if (task is None or task.done()) and not self._has_work():
            self._finish_dispose()
            return
        while self._state is not ScopeState.DISPOSED:
            await asyncio.shield(self._ensure_task())

    # ------------------------------------------------------------------
    # Start, restart
    # ------------------------------------------------------------------

    async def start(self, setup: Optional[Setup] = None) -> "Scope":
        """Run ``setup(scope)`` (sync or async) and make the scope ACTIVE.

        If ``setup`` raises, the effects registered so far are undone, the
        error is recorded on ``errors``, and the scope becomes FAILED. The
        exception is NOT re-raised, so a failing part cannot take down its
        siblings. Check ``scope.state`` or ``scope.errors``. A FAILED scope
        never retries by itself; call ``restart``.

        Async effects registered during setup are awaited before the scope
        turns ACTIVE; if one of them failed, the scope is FAILED too.

        ``asyncio.CancelledError`` is not a failure: the effects registered so
        far are undone, the scope goes back to PENDING and the cancellation
        propagates.
        """
        self._check_open("start")
        if self._starting or self._state is not ScopeState.PENDING:
            raise RuntimeError(
                f"scope {self.name!r} is {self._state.value}; start() needs a PENDING scope "
                "(use restart() after a failure)"
            )
        self._starting = True
        return await self._run_setup(setup)

    async def restart(self, setup: Optional[Setup] = None) -> "Scope":
        """Undo all effects, then run ``setup`` again. Allowed from FAILED or ACTIVE."""
        self._check_open("restart")
        if self._starting or self._state not in (ScopeState.FAILED, ScopeState.ACTIVE):
            raise RuntimeError(
                f"scope {self.name!r} is {self._state.value}; restart() needs a FAILED or "
                "ACTIVE scope"
            )
        self._starting = True
        try:
            await self._join_unwind()
        except BaseException:
            self._starting = False
            raise
        if self._closing:
            self._starting = False
            await self.dispose()
            return self
        self._set_state(ScopeState.PENDING)
        return await self._run_setup(setup)

    async def _settle(self) -> None:
        while self._pending:
            await asyncio.wait(list(self._pending))

    async def _run_setup(self, setup: Optional[Setup]) -> "Scope":
        self._async_failure = None
        failed = False
        try:
            try:
                if setup is not None:
                    result = setup(self)
                    if inspect.isawaitable(result):
                        await result
                await self._settle()
                failed = self._async_failure is not None
            except Exception as exc:
                self._record("setup", exc)
                failed = True
            except GeneratorExit:
                raise
            except BaseException:
                # Cancellation (or Ctrl-C): not a failure, but do not leak.
                await self._join_unwind()
                raise
            if self._closing:
                self._starting = False
                await self.dispose()
                return self
            if failed:
                await self._join_unwind()
                if self._closing:
                    self._starting = False
                    await self.dispose()
                    return self
                self._set_state(ScopeState.FAILED)
            else:
                self._set_state(ScopeState.ACTIVE)
        finally:
            self._starting = False
        return self

    async def __aenter__(self) -> "Scope":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.dispose()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """JSON-safe health report.

        ``effects`` counts this scope's own effects; child scopes are listed
        under ``children``. ``recent_errors`` holds the last few errors as text.
        """
        return self._describe(_DESCRIBE_DEPTH)

    def _describe(self, depth: int) -> Dict[str, Any]:
        children = self.children
        report: Dict[str, Any] = {
            "name": self.name,
            "state": self._state.value,
            "effects": self.effect_count,
            "error_count": len(self._errors),
            "dropped_errors": self._dropped_errors,
            "recent_errors": [str(e) for e in list(self._errors)[-_DESCRIBE_ERRORS:]],
        }
        if depth > 0:
            report["children"] = [c._describe(depth - 1) for c in children]
        else:
            report["children"] = []
            report["children_truncated"] = len(children)
        return report
