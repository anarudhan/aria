"""
ARIA Kernel Events
------------------
A small event system with five dispatch modes. It is inspired by the Cordis
event system, with its known problems fixed. Standard library only.

Modes
  emit       sync, fire and forget. Every listener runs, errors are collected.
  bail       sync, first result that is not None and not False wins.
  serial     async, listeners awaited one by one, first bailed result wins.
  parallel   async, all listeners run together, failures are gathered.
  waterfall  onion chain, each listener decides whether to call ``next``.

This module gives clean extension points. It does NOT isolate or sandbox
listeners: a listener runs with the full privileges of the process.

Rules that hold for every mode
  * A dispatch works on a snapshot of the listeners. Adding or removing
    listeners during a dispatch does not change that dispatch, except that a
    listener removed before its turn comes is skipped.
  * Listeners are removed by a unique token (through their disposer), never by
    comparing functions. The same function may be registered twice.
  * ``asyncio.CancelledError`` is never swallowed and never recorded as a
    listener failure.
  * Event names are plain strings and all strings are equal. ``__proto__``,
    ``toString`` and ``__class__`` behave like any other name.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Set, Tuple

Listener = Callable[..., Any]

DEFAULT_MAX_ERRORS = 1000


@dataclass(frozen=True)
class ListenerError:
    """One failed listener call: which event, which listener, what was raised."""

    event: str
    listener: str
    exception: Exception

    def __str__(self) -> str:
        return f"{self.event}: {self.listener}: {type(self.exception).__name__}: {self.exception}"


class ListenerErrors(Exception):
    """Raised by ``parallel`` when one or more listeners failed.

    ``errors`` holds one ``ListenerError`` per failed listener, in listener order.
    """

    def __init__(self, event: str, errors: List[ListenerError]) -> None:
        self.event = event
        self.errors: Tuple[ListenerError, ...] = tuple(errors)
        detail = "; ".join(str(e) for e in self.errors)
        super().__init__(f"{len(self.errors)} listener(s) failed for event {event!r}: {detail}")

    @property
    def exceptions(self) -> Tuple[Exception, ...]:
        return tuple(e.exception for e in self.errors)


class NextAlreadyCalledError(RuntimeError):
    """A waterfall listener called ``next`` more than once."""


class _Entry:
    __slots__ = ("token", "event", "fn", "label", "once", "active")

    def __init__(self, token: int, event: str, fn: Listener, label: str, once: bool) -> None:
        self.token = token
        self.event = event
        self.fn = fn
        self.label = label
        self.once = once
        self.active = True


class Disposer:
    """Removes one registration. Calling it more than once is harmless.

    Calling returns True if this call removed the listener, False if it was
    already gone. ``token`` is the unique id of the registration.
    """

    __slots__ = ("_events", "_entry")

    def __init__(self, events: "Events", entry: _Entry) -> None:
        self._events = events
        self._entry = entry

    @property
    def token(self) -> int:
        return self._entry.token

    @property
    def active(self) -> bool:
        return self._entry.active

    def __call__(self) -> bool:
        return self._events._remove(self._entry)


def _label_of(fn: Listener) -> str:
    name = getattr(fn, "__qualname__", None)
    return name if isinstance(name, str) else repr(fn)


def _discard_awaitable(obj: Any) -> None:
    """Close a coroutine we will never await, to avoid 'never awaited' warnings."""
    close = getattr(obj, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # pragma: no cover - best effort only
            pass


async def _await(aw: Awaitable[Any]) -> Any:
    return await aw


class Events:
    """Event registry and dispatcher. See the module docstring for the rules."""

    def __init__(self, *, max_errors: int = DEFAULT_MAX_ERRORS) -> None:
        if max_errors < 1:
            raise ValueError("max_errors must be at least 1")
        # Copy on write: each event maps to an immutable tuple, so a dispatch
        # snapshot is just the tuple that was current when it started.
        self._listeners: Dict[str, Tuple[_Entry, ...]] = {}
        self._tokens = itertools.count(1)
        self._errors: Deque[ListenerError] = deque(maxlen=max_errors)
        self._dropped_errors = 0
        self._tasks: Set["asyncio.Future[Any]"] = set()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def on(
        self,
        name: str,
        listener: Listener,
        *,
        prepend: bool = False,
        label: Optional[str] = None,
    ) -> Disposer:
        """Register ``listener`` for ``name``. Returns a disposer.

        ``prepend=True`` puts it before the existing listeners.
        """
        return self._add(name, listener, prepend=prepend, label=label, once=False)

    def once(
        self,
        name: str,
        listener: Listener,
        *,
        prepend: bool = False,
        label: Optional[str] = None,
    ) -> Disposer:
        """Like ``on`` but the listener removes itself BEFORE it runs.

        It therefore runs at most once, even if it raises or is re-entered.
        """
        return self._add(name, listener, prepend=prepend, label=label, once=True)

    def _add(
        self,
        name: str,
        listener: Listener,
        *,
        prepend: bool,
        label: Optional[str],
        once: bool,
    ) -> Disposer:
        self._check_name(name)
        if not callable(listener):
            raise TypeError(f"listener must be callable, got {type(listener).__name__}")
        entry = _Entry(next(self._tokens), name, listener, label or _label_of(listener), once)
        current = self._listeners.get(name, ())
        self._listeners[name] = (entry,) + current if prepend else current + (entry,)
        return Disposer(self, entry)

    def _remove(self, entry: _Entry) -> bool:
        if not entry.active:
            return False
        entry.active = False
        remaining = tuple(e for e in self._listeners.get(entry.event, ()) if e.token != entry.token)
        if remaining:
            self._listeners[entry.event] = remaining
        else:
            self._listeners.pop(entry.event, None)
        return True

    @staticmethod
    def _check_name(name: Any) -> None:
        if not isinstance(name, str):
            raise TypeError(f"event name must be a str, got {type(name).__name__}")

    def listener_count(self, name: str) -> int:
        self._check_name(name)
        return len(self._listeners.get(name, ()))

    # ------------------------------------------------------------------
    # Errors and health
    # ------------------------------------------------------------------

    @property
    def errors(self) -> List[ListenerError]:
        """A copy of the collected listener errors (from ``emit``), oldest first."""
        return list(self._errors)

    def take_errors(self) -> List[ListenerError]:
        """Return the collected errors and clear them."""
        taken = list(self._errors)
        self._errors.clear()
        return taken

    def clear_errors(self) -> None:
        self._errors.clear()

    def _record(self, event: str, label: str, exc: Exception) -> None:
        if len(self._errors) == self._errors.maxlen:
            self._dropped_errors += 1
        self._errors.append(ListenerError(event, label, exc))

    def describe(self) -> Dict[str, Any]:
        """Health report: listeners per event and the collected error count."""
        return {
            "listeners": {name: len(entries) for name, entries in self._listeners.items()},
            "error_count": len(self._errors),
            "dropped_errors": self._dropped_errors,
            "pending_async_listeners": len(self._tasks),
        }

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    def _snapshot(self, name: str) -> Tuple[_Entry, ...]:
        self._check_name(name)
        return self._listeners.get(name, ())

    def _claim(self, entry: _Entry) -> bool:
        """Decide whether this listener's turn should run.

        A listener removed earlier is skipped. A ``once`` listener is removed
        here, before it runs.
        """
        if not entry.active:
            return False
        if entry.once:
            self._remove(entry)
        return True

    # ------------------------------------------------------------------
    # emit
    # ------------------------------------------------------------------

    def emit(self, name: str, *args: Any) -> None:
        """Call every listener in order and ignore return values.

        A listener that raises does not stop the others; the error is recorded
        (see ``errors``). ``CancelledError`` and other non-``Exception``
        exceptions are not caught.

        Async listeners: ``emit`` cannot await. If a listener returns an
        awaitable and a loop is running, the awaitable is scheduled as a task
        on that loop and any failure is recorded in ``errors`` when it
        happens. If no loop is running the awaitable is closed unawaited and a
        ``TypeError`` is recorded for that listener (the other listeners
        still run). Use ``serial`` or ``parallel`` if you need to wait.
        """
        for entry in self._snapshot(name):
            if not self._claim(entry):
                continue
            try:
                result = entry.fn(*args)
                if inspect.isawaitable(result):
                    self._schedule(name, entry, result)
            except Exception as exc:
                self._record(name, entry.label, exc)

    def _schedule(self, name: str, entry: _Entry, aw: Awaitable[Any]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _discard_awaitable(aw)
            raise TypeError(
                f"listener {entry.label!r} for event {name!r} is async but emit() was called "
                "with no running event loop; use serial() or parallel() from async code"
            ) from None
        task = loop.create_task(_await(aw))
        self._tasks.add(task)
        task.add_done_callback(lambda t: self._task_done(name, entry.label, t))

    def _task_done(self, name: str, label: str, task: "asyncio.Future[Any]") -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if isinstance(exc, Exception):
            self._record(name, label, exc)

    # ------------------------------------------------------------------
    # bail
    # ------------------------------------------------------------------

    def bail(self, name: str, *args: Any) -> Any:
        """Return the first result that is not ``None`` and not ``False``.

        ``0``, ``""`` and other falsy values DO count as results. Returns
        ``None`` if no listener bails. An exception propagates and stops the
        dispatch (fail closed). A listener that returns an awaitable raises
        ``TypeError``: a coroutine object must never be mistaken for a
        decision. Use ``serial`` for async listeners.
        """
        for entry in self._snapshot(name):
            if not self._claim(entry):
                continue
            result = entry.fn(*args)
            if inspect.isawaitable(result):
                _discard_awaitable(result)
                raise TypeError(
                    f"listener {entry.label!r} for event {name!r} returned an awaitable; "
                    "bail() is sync, use serial() for async listeners"
                )
            if result is not None and result is not False:
                return result
        return None

    # ------------------------------------------------------------------
    # serial
    # ------------------------------------------------------------------

    async def serial(self, name: str, *args: Any) -> Any:
        """Await listeners one by one; return the first bailed result.

        Sync and async listeners both work. A bailed result is anything that
        is not ``None`` and not ``False``. An exception propagates.
        """
        for entry in self._snapshot(name):
            if not self._claim(entry):
                continue
            result = entry.fn(*args)
            if inspect.isawaitable(result):
                result = await result
            if result is not None and result is not False:
                return result
        return None

    # ------------------------------------------------------------------
    # parallel
    # ------------------------------------------------------------------

    async def parallel(self, name: str, *args: Any) -> None:
        """Run all listeners together. All run even if some fail.

        If any failed, raise one ``ListenerErrors`` holding every error, in
        listener order. Returns nothing on success. If the caller is
        cancelled, the running listeners are cancelled and ``CancelledError``
        propagates.
        """
        entries = self._snapshot(name)
        if not entries:
            return
        failures: List[Optional[ListenerError]] = [None] * len(entries)

        async def run(index: int, entry: _Entry) -> None:
            if not self._claim(entry):
                return
            try:
                result = entry.fn(*args)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                failures[index] = ListenerError(name, entry.label, exc)

        tasks = [asyncio.ensure_future(run(i, e)) for i, e in enumerate(entries)]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise
        errors = [f for f in failures if f is not None]
        if errors:
            raise ListenerErrors(name, errors)

    # ------------------------------------------------------------------
    # waterfall
    # ------------------------------------------------------------------

    def waterfall(self, name: str, *args: Any, inner: Callable[[], Any]) -> Any:
        """Run an onion chain and return whatever the first layer returns.

        Each listener is called as ``listener(*args, next)``. ``next()`` runs
        the next listener, or ``inner()`` after the last one, and returns its
        result. A listener that does not call ``next`` ends the chain and its
        return value is the result. Calling ``next`` twice raises
        ``NextAlreadyCalledError``, also after an await. Errors propagate.

        Sync and async: the return value is passed through untouched. If any
        layer is async, or ``inner`` is async, the value you get back may be
        awaitable, so check with ``inspect.isawaitable`` and await it. If you
        are in async code and do not want to check, use ``awaterfall``, which
        always returns a coroutine.
        """
        chain = self._snapshot(name)

        def run(start: int) -> Any:
            index = start
            while index < len(chain):
                entry = chain[index]
                if self._claim(entry):
                    break
                index += 1
            else:
                return inner()
            called = False

            def next_() -> Any:
                nonlocal called
                if called:
                    raise NextAlreadyCalledError(
                        f"next() called more than once by listener {entry.label!r} "
                        f"for event {name!r}"
                    )
                called = True
                return run(index + 1)

            return entry.fn(*args, next_)

        return run(0)

    async def awaterfall(self, name: str, *args: Any, inner: Callable[[], Any]) -> Any:
        """Same as ``waterfall`` but always a coroutine: awaitable results are awaited."""
        result = self.waterfall(name, *args, inner=inner)
        while inspect.isawaitable(result):
            result = await result
        return result
