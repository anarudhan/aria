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
import functools
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
    Coroutine,
    Deque,
    Dict,
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
        try:
            detail = str(self.exception)
        except Exception:  # a broken __str__ must not break error reporting
            detail = "<unprintable>"
        return f"{self.label}: {type(self.exception).__name__}: {detail}"


# How many error texts a parent keeps from a finished child.
_CHILD_ERROR_TEXTS = 3


class ChildScopeErrors(Exception):
    """Recorded on a parent when a child scope finished with recorded errors.

    The child is gone after disposal, so this keeps a summary: ``count`` is how
    many errors the child had recorded or dropped, and ``recent`` holds the text
    of the last few. It deliberately does not keep the child's error objects
    (and their tracebacks), so a parent's memory stays bounded.
    """

    def __init__(self, scope: str, count: int, recent: Tuple[str, ...] = ()) -> None:
        self.scope = scope
        self.count = count
        self.recent = tuple(recent)
        super().__init__(f"child scope {scope!r} finished with {count} recorded error(s)")


# Unwinding work is numbered: every unwind pass of a scope, and every remover
# task, gets a fresh epoch number. The epochs of the work that is running are
# kept, outermost first, in a context variable, so every task started from inside
# an undo (helper tasks such as gather() or create_task(), fire and forget tasks,
# eager tasks) inherits them from the first step on. ``_LIVE_EPOCHS`` says which
# epochs are still running and for which scope; an epoch is never reused and
# stops counting when its work ends. Two things use this:
#  * A dispose() call whose context holds a live epoch of that same scope is made
#    from inside the scope's own unwinding and returns at once instead of waiting
#    for itself.
#  * The innermost live epoch in the context names the scope on whose behalf a
#    dispose() call waits (see ``_WAITS``).
# A leftover task from an earlier unwind, or a new scope that happens to get the
# same ``id()``, can never be mistaken for either, because its epochs are dead.
_EPOCHS = itertools.count(1)
_SERIALS = itertools.count(1)
_DISPOSING: ContextVar[Tuple[int, ...]] = ContextVar("aria_scope_disposing", default=())
_LIVE_EPOCHS: Dict[int, int] = {}  # live epoch -> serial number of its scope

# Who is waiting for whose disposal: waiter serial -> {target serial: how many
# waits}. A wait is recorded when a dispose() call is made from inside some
# scope's live unwinding, directly or from any task that unwinding started. It
# lets two scopes that dispose each other at the same time notice the ring and
# return instead of waiting for each other forever. The price: a background task
# an undo started counts as part of that undo while the undo is live, so a
# dispose() it makes may return early when waiting would close a ring.
_WAITS: Dict[int, Dict[int, int]] = {}


def _waits_reach(start: int, goal: int) -> bool:
    seen: Set[int] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node == goal:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(_WAITS.get(node, ()))
    return False


def _add_wait(waiter: int, target: int) -> None:
    edges = _WAITS.setdefault(waiter, {})
    edges[target] = edges.get(target, 0) + 1


def _drop_wait(waiter: int, target: int) -> None:
    edges = _WAITS.get(waiter)
    if edges is None:
        return
    left = edges.get(target, 0) - 1
    if left > 0:
        edges[target] = left
    else:
        edges.pop(target, None)
        if not edges:
            del _WAITS[waiter]


def _current_owner() -> Optional[int]:
    """Serial of the scope whose live unwinding the running code belongs to, if any."""
    for epoch in reversed(_DISPOSING.get()):
        serial = _LIVE_EPOCHS.get(epoch)
        if serial is not None:
            return serial
    return None


class _Done(Coroutine[Any, Any, None]):
    """An already finished awaitable. Awaiting it does nothing.

    It is a coroutine in the ``collections.abc`` sense, so ``asyncio.run`` and
    ``asyncio.create_task`` accept it, but not a native coroutine object, so it
    never triggers Python's "coroutine was never awaited" warning when ignored.
    It holds no state and can be shared.
    """

    __slots__ = ()

    def __await__(self) -> Generator[Any, None, None]:
        return
        yield  # pragma: no cover - makes this a generator

    def send(self, value: Any) -> Any:
        raise StopIteration

    def throw(self, *args: Any) -> Any:  # type: ignore[override]
        exc = args[0]
        raise exc if isinstance(exc, BaseException) else exc(*args[1:2])

    def close(self) -> None:
        return None


_DONE = _Done()


class _Waiter(Coroutine[Any, Any, None]):
    """What a remover hands back while work is scheduled: waits for it to finish.

    Like ``_Done`` it is a ``collections.abc`` coroutine, so ``await``,
    ``asyncio.run`` and ``asyncio.create_task`` all work, and ignoring it is
    harmless (the scheduled work happens either way).

    It keeps no state of its own: ``send`` does what ``Future.__await__`` does,
    looking at the future each time. So the same object may be awaited, wrapped
    in several tasks one after the other, or in several tasks at once.
    """

    __slots__ = ("_fut",)

    def __init__(self, fut: "asyncio.Future[Any]") -> None:
        self._fut = fut

    def __await__(self) -> Generator[Any, None, None]:
        return self._fut.__await__()

    def send(self, value: Any) -> Any:
        fut = self._fut
        if not fut.done():
            fut._asyncio_future_blocking = True  # type: ignore[attr-defined]
            return fut  # the task waits for it and calls send again
        raise StopIteration(fut.result())

    def throw(self, *args: Any) -> Any:  # type: ignore[override]
        exc = args[0]
        raise exc if isinstance(exc, BaseException) else exc(*args[1:2])

    def close(self) -> None:
        return None


class _Effect:
    __slots__ = ("id", "label", "disposers", "task", "child", "cancel_sent")

    def __init__(self, ident: int, label: str, disposers: List[Callable[[], Any]]) -> None:
        self.id = ident
        self.label = label
        self.disposers = disposers
        # Set while an async effect has not resolved yet.
        self.task: Optional["asyncio.Future[None]"] = None
        self.child: Optional["Scope"] = None
        # Cancelling a resolving effect is asked once only: an effect that
        # chooses to finish anyway must not be hit by a second cancel.
        self.cancel_sent = False

    def cancel_resolving(self) -> None:
        task = self.task
        if task is not None and not task.done() and not self.cancel_sent:
            self.cancel_sent = True
            task.cancel()


class Remover:
    """Undoes one effect early. Returned by ``effect`` and ``on``.

    Calling it removes the effect from the scope and starts its undo now. The
    return value is awaitable.

    * Sync undos (listeners always are) have already finished when the call
      returns; awaiting the result is optional.
    * If an undo is async, or the effect is still resolving, then with a running
      event loop the rest of the work is scheduled as a task. It happens even if
      you never await the result, and ``dispose()`` waits for it. Await the
      result to wait for it yourself, and to see a failure of the undo
      (failures are recorded in ``scope.errors`` either way). Awaiting is safe
      to cancel: the undo carries on.
    * With NO running loop, the rest of the work is returned as an awaitable that
      you must run yourself (for example with ``asyncio.run``).

    Calling it again does nothing and returns an already finished awaitable.
    It only ever touches the effect it was created for. If an undo run this way
    calls ``dispose()`` on its own scope, that call returns at once (the same
    no-deadlock rule as during disposal).
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
        return scope._undo_detached(self._effect)


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


def _discard_cancelled(task: "asyncio.Future[Any]", *, aw: Any) -> None:
    if task.cancelled():
        _discard_awaitable(aw)


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
        self._serial = next(_SERIALS)
        self._live: Set[int] = set()  # epochs of this scope's running unwind work
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
            # A task cancelled before its first step never starts the awaitable;
            # close it so Python does not warn that it was never awaited.
            task.add_done_callback(functools.partial(_discard_cancelled, aw=result))
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

    def _advance(
        self, label: str, rest: List[Callable[[], Any]], bag: List[BaseException]
    ) -> Optional[Awaitable[Any]]:
        """Run disposers from ``rest`` (newest last) until one returns something to await.

        Returns that awaitable, or None when ``rest`` is used up. Errors are
        recorded. A ``CancelledError`` raised by a disposer is that disposer's
        own and is dropped. Any other ``BaseException`` (``SystemExit`` and the
        like) goes into ``bag`` and is raised once everything has run. Anything
        that is not awaitable is ignored, except that sync and async generators
        are run to the end: calling a generator function does nothing by
        itself, and silently skipping it would hide the bug.
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
            except asyncio.CancelledError:
                pass
            except BaseException as exc:
                bag.append(exc)
        return None

    def _undo(self, eff: _Effect, bag: Optional[List[BaseException]] = None) -> Awaitable[None]:
        """Undo a claimed effect (already out of its dict). The sync part runs now.

        With ``bag=None`` the caller wants the result raised: a ``BaseException``
        that a disposer raised, or a genuine cancellation of the awaiting task,
        is raised after all disposers have run. With a shared ``bag`` (the
        unwinding loop) they are collected there instead.
        """
        own = bag is None
        if bag is None:
            bag = []
        task = eff.task
        if task is not None and not task.done():
            # Still resolving: ask it to stop, so a hanging effect cannot hold the
            # undo (and with it dispose()) forever. One that ignores the request
            # and resolves anyway is still undone, once.
            eff.cancel_resolving()
            return self._undo_late(eff, bag, own)
        rest = list(eff.disposers)
        eff.disposers = []
        pending = self._advance(eff.label, rest, bag)
        if pending is None:
            if own and bag:
                raise bag[0]
            return _DONE
        return self._undo_async(eff, pending, rest, bag, own)

    async def _undo_async(
        self,
        eff: _Effect,
        pending: Awaitable[Any],
        rest: List[Callable[[], Any]],
        bag: List[BaseException],
        own: bool,
    ) -> None:
        await self._drain(eff.label, pending, rest, bag, keep=eff.child is not None)
        if own and bag:
            raise bag[0]

    async def _undo_late(self, eff: _Effect, bag: List[BaseException], own: bool) -> None:
        """Wait for an async effect to resolve, then undo it. Never skips the undo."""
        task = eff.task
        assert task is not None
        while not task.done():
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError as exc:
                bag.append(exc)
        rest = list(eff.disposers)
        eff.disposers = []
        pending = self._advance(eff.label, rest, bag)
        if pending is not None:
            await self._drain(eff.label, pending, rest, bag, keep=False)
        if own and bag:
            raise bag[0]

    async def _drain(
        self,
        label: str,
        pending: Optional[Awaitable[Any]],
        rest: List[Callable[[], Any]],
        bag: List[BaseException],
        keep: bool,
    ) -> None:
        while pending is not None:
            await self._await_one(label, pending, bag, keep)
            pending = self._advance(label, rest, bag)

    async def _await_one(
        self, label: str, aw: Awaitable[Any], bag: List[BaseException], keep: bool
    ) -> None:
        """Run one async disposer as its own task and wait for it.

        Being a task of its own, the disposer's cancellation (for example it
        stops another task and awaits it, or cancels itself) is not confused
        with a cancellation of the task that is unwinding. Only a real
        cancellation of the waiter is remembered in ``bag``. Then the disposer's
        task is cancelled and awaited (unless ``keep``, used for child scopes
        whose own unwinding must finish), so nothing outlives the call.
        """
        try:
            task = asyncio.ensure_future(aw)
        except Exception as exc:
            _discard_awaitable(aw)
            self._record(label, exc)
            return
        while not task.done():
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError as exc:
                bag.append(exc)
                if not keep:
                    task.cancel()
        if task.cancelled():
            return  # the disposer's own cancellation, or ours after a real one
        exc2 = task.exception()
        if exc2 is None:
            return
        if isinstance(exc2, Exception):
            self._record(label, exc2)
        else:
            bag.append(exc2)

    def _undo_detached(self, eff: _Effect) -> Awaitable[None]:
        """Undo for a remover: the work is scheduled so it happens even if not awaited."""
        aw = self._undo(eff)
        if aw is _DONE:
            return _DONE
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return aw  # no loop: the caller has to run it
        task = loop.create_task(self._marked(aw))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return _Waiter(asyncio.shield(task))

    async def _marked(self, aw: Awaitable[Any]) -> None:
        epoch = next(_EPOCHS)
        self._live.add(epoch)
        _LIVE_EPOCHS[epoch] = self._serial
        token = _DISPOSING.set(_DISPOSING.get() + (epoch,))
        try:
            await aw
        finally:
            self._live.discard(epoch)
            _LIVE_EPOCHS.pop(epoch, None)
            try:
                _DISPOSING.reset(token)
            except ValueError:  # pragma: no cover - context changed under us
                pass

    def _inside_own_work(self) -> bool:
        return bool(self._live.intersection(_DISPOSING.get()))

    def _live_pending(self) -> List["asyncio.Future[None]"]:
        return [t for t in self._pending if not t.done()]

    def _cancel_unresolved(self) -> None:
        """Cancel async effects that have not resolved (used when setup is abandoned)."""
        for eff in list(self._effects.values()):
            eff.cancel_resolving()

    # ------------------------------------------------------------------
    # Unwinding and disposal
    # ------------------------------------------------------------------

    async def _unwind_loop(self) -> None:
        bag: List[BaseException] = []
        while True:
            if self._effects or self._children:
                # Children first, then own effects, each newest first.
                _, eff = (self._children if self._children else self._effects).popitem()
                try:
                    aw = self._undo(eff, bag)
                    if aw is not _DONE:
                        await aw
                except asyncio.CancelledError as exc:
                    bag.append(exc)
                continue
            live = self._live_pending()
            if not live:
                break
            try:
                await asyncio.wait(live)
            except asyncio.CancelledError as exc:
                bag.append(exc)
        if bag:
            raise bag[0]

    async def _run(self) -> None:
        epoch = next(_EPOCHS)
        self._live.add(epoch)
        _LIVE_EPOCHS[epoch] = self._serial
        token = _DISPOSING.set(_DISPOSING.get() + (epoch,))
        try:
            self._cancel_unresolved()
            await self._unwind_loop()
        finally:
            self._live.discard(epoch)
            _LIVE_EPOCHS.pop(epoch, None)
            try:
                _DISPOSING.reset(token)
            except ValueError:
                # "Token was created in a different Context": can happen when an
                # unwind task that never finished is collected. Nothing to undo.
                pass
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
        if self._inside_own_work():
            return  # called from inside this scope's own unwinding
        task = self._task
        if (task is None or task.done()) and not self._has_work():
            return
        while True:
            await asyncio.shield(self._ensure_task())
            task = self._task
            if (task is None or task.done()) and not self._has_work():
                return

    def _has_work(self) -> bool:
        return bool(self._effects or self._children or self._live_pending())

    def _finish_dispose(self) -> None:
        link, self._link = self._link, None
        if link is not None:
            parent, ident = link
            parent._children.pop(ident, None)
            if self._errors or self._dropped_errors:
                recent = tuple(str(e) for e in list(self._errors)[-_CHILD_ERROR_TEXTS:])
                parent._record(
                    f"scope {self.name}",
                    ChildScopeErrors(self.name, len(self._errors) + self._dropped_errors, recent),
                )
        self._set_state(ScopeState.DISPOSED)

    async def dispose(self) -> None:
        """Undo everything, newest first, then mark the scope DISPOSED.

        Idempotent: later calls return at once if it is finished, and wait for
        it if it is still running. Errors from undos do not raise; they are in
        ``errors``. A disposer that calls ``dispose()`` on its own scope gets an
        immediate return instead of a deadlock. So does a disposer that waits for
        another scope whose disposal is (directly or through others) waiting for
        this one: the call that would close the cycle returns without waiting.

        If a disposer raised a ``BaseException`` that is not ``CancelledError``
        (``SystemExit``, ``KeyboardInterrupt``), the remaining disposers still
        run, the scope ends DISPOSED, and then that exception is raised here.

        The unwinding runs as its own task. If the caller is cancelled (for
        example by ``asyncio.wait_for``), it stops waiting, but the unwinding
        carries on until it is done, so nothing is left behind.
        """
        if self._state is ScopeState.DISPOSED or self._inside_own_work():
            return  # finished, or called from inside this scope's own unwinding
        waiter = _current_owner()
        if waiter is not None and _waits_reach(self._serial, waiter):
            return  # waiting would close a cycle: two disposals waiting for each other
        self._closing = True
        task = self._task
        if (task is None or task.done()) and not self._has_work():
            self._finish_dispose()
            return
        if waiter is not None:
            _add_wait(waiter, self._serial)
        try:
            while self._state is not ScopeState.DISPOSED:
                await asyncio.shield(self._ensure_task())
        finally:
            if waiter is not None:
                _drop_wait(waiter, self._serial)

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
        # PENDING first: if the restart is cancelled while the old effects are
        # being undone, the scope is left PENDING (empty, may be started again)
        # and not ACTIVE with nothing in it.
        self._set_state(ScopeState.PENDING)
        try:
            await self._join_unwind()
        except BaseException:
            self._starting = False
            raise
        if self._closing:
            self._starting = False
            await self.dispose()
            return self
        return await self._run_setup(setup)

    async def _settle(self) -> None:
        while True:
            live = self._live_pending()
            if not live:
                return
            await asyncio.wait(live)

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
                # Cancellation (or Ctrl-C): not a failure, but do not leak. Async
                # effects that have not resolved are cancelled first, otherwise a
                # hanging one would keep the unwinding (and a timeout) waiting.
                self._cancel_unresolved()
                await self._join_unwind()
                raise
            if self._closing:
                self._starting = False
                await self.dispose()
                return self
            if failed:
                self._cancel_unresolved()
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
            "closing": self._closing,  # dispose() was requested
            "unwinding": self._unwinding,  # undos are running right now
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
