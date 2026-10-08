"""Tests for aria.kernel.events (issue #38)."""

from __future__ import annotations

import asyncio
import functools
import gc
import inspect
import sys
import threading
import time
import warnings
import weakref
from typing import Any, List

import pytest

from aria.kernel import (
    Disposer,
    Events,
    ListenerError,
    ListenerErrors,
    NextAlreadyCalledError,
)


@pytest.fixture
def ev() -> Events:
    return Events()


# ----------------------------------------------------------------------
# Registration, disposers, naming
# ----------------------------------------------------------------------


class TestRegistration:
    def test_on_returns_callable_disposer(self, ev: Events) -> None:
        d = ev.on("a", lambda: None)
        assert isinstance(d, Disposer)
        assert callable(d)
        assert ev.listener_count("a") == 1

    def test_disposer_removes_and_is_idempotent(self, ev: Events) -> None:
        calls: List[int] = []
        d = ev.on("a", lambda: calls.append(1))
        assert d() is True
        assert d() is False
        assert d() is False
        ev.emit("a")
        assert calls == []
        assert ev.listener_count("a") == 0

    def test_tokens_are_unique(self, ev: Events) -> None:
        f = lambda: None  # noqa: E731
        assert ev.on("a", f).token != ev.on("a", f).token

    def test_same_function_twice_then_remove_one(self, ev: Events) -> None:
        calls: List[str] = []

        def f() -> None:
            calls.append("f")

        d1 = ev.on("a", f)
        d2 = ev.on("a", f)
        ev.emit("a")
        assert calls == ["f", "f"]
        d1()
        calls.clear()
        ev.emit("a")
        assert calls == ["f"]
        assert d2.active and not d1.active
        d1()  # a stale disposer must not remove the other registration
        ev.emit("a")
        assert calls == ["f", "f"]
        d2()
        assert ev.listener_count("a") == 0

    def test_prepend_ordering(self, ev: Events) -> None:
        order: List[str] = []
        ev.on("a", lambda: order.append("1"))
        ev.on("a", lambda: order.append("2"))
        ev.on("a", lambda: order.append("p1"), prepend=True)
        ev.on("a", lambda: order.append("p2"), prepend=True)
        ev.emit("a")
        assert order == ["p2", "p1", "1", "2"]

    def test_once_prepend(self, ev: Events) -> None:
        order: List[str] = []
        ev.on("a", lambda: order.append("on"))
        ev.once("a", lambda: order.append("once"), prepend=True)
        ev.emit("a")
        ev.emit("a")
        assert order == ["once", "on", "on"]

    def test_rejects_non_callable_and_non_str_name(self, ev: Events) -> None:
        with pytest.raises(TypeError):
            ev.on("a", "nope")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            ev.on(1, lambda: None)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            ev.emit(None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("name", ["__proto__", "toString", "__class__", "constructor", "", "__dict__"])
    def test_odd_names_work_like_any_other(self, ev: Events, name: str) -> None:
        got: List[Any] = []
        assert ev.listener_count(name) == 0
        assert ev.emit(name) is None  # nothing registered, no error
        ev.on(name, lambda x: got.append(x))
        ev.emit(name, 5)
        assert got == [5]
        assert ev.describe()["listeners"] == {name: 1}

    def test_partial_gets_a_label(self, ev: Events) -> None:
        ev.on("a", functools.partial(lambda x, y: 1 / 0, 1))
        ev.emit("a", 2)
        assert ev.errors[0].listener.startswith("functools.partial")

    def test_custom_label(self, ev: Events) -> None:
        ev.on("a", lambda: 1 / 0, label="my-listener")
        ev.emit("a")
        assert ev.errors[0].listener == "my-listener"


# ----------------------------------------------------------------------
# emit
# ----------------------------------------------------------------------


class TestEmit:
    def test_empty(self, ev: Events) -> None:
        assert ev.emit("nothing", 1, 2) is None
        assert ev.errors == []

    def test_order_args_and_return_values_ignored(self, ev: Events) -> None:
        seen: List[Any] = []
        ev.on("a", lambda *a: seen.append(("one", a)) or "ignored")
        ev.on("a", lambda *a: seen.append(("two", a)) or 42)
        assert ev.emit("a", 1, "x") is None
        assert seen == [("one", (1, "x")), ("two", (1, "x"))]

    def test_raising_listener_does_not_stop_others_and_is_recorded(self, ev: Events) -> None:
        order: List[str] = []

        def boom() -> None:
            order.append("boom")
            raise ValueError("bad")

        ev.on("a", lambda: order.append("first"))
        ev.on("a", boom)
        ev.on("a", lambda: order.append("last"))
        ev.emit("a")
        assert order == ["first", "boom", "last"]
        errs = ev.errors
        assert len(errs) == 1
        assert isinstance(errs[0], ListenerError)
        assert errs[0].event == "a"
        assert errs[0].listener.endswith("boom")
        assert isinstance(errs[0].exception, ValueError)
        assert "bad" in str(errs[0])

    def test_errors_read_and_clear(self, ev: Events) -> None:
        ev.on("a", lambda: 1 / 0)
        ev.emit("a")
        ev.emit("a")
        assert len(ev.errors) == 2
        assert len(ev.errors) == 2  # reading does not clear
        ev.errors.clear()  # the property returns a copy
        assert len(ev.errors) == 2
        taken = ev.take_errors()
        assert len(taken) == 2
        assert ev.errors == []
        ev.emit("a")
        ev.clear_errors()
        assert ev.errors == []

    def test_error_buffer_is_bounded(self) -> None:
        ev = Events(max_errors=3)
        ev.on("a", lambda: 1 / 0)
        for _ in range(5):
            ev.emit("a")
        assert len(ev.errors) == 3
        assert ev.describe()["dropped_errors"] == 2
        with pytest.raises(ValueError):
            Events(max_errors=0)

    def test_base_exception_is_not_swallowed(self, ev: Events) -> None:
        ev.on("a", lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            ev.emit("a")
        assert ev.errors == []

    def test_cancelled_error_is_not_swallowed_or_recorded(self, ev: Events) -> None:
        def cancel() -> None:
            raise asyncio.CancelledError()

        ev.on("a", cancel)
        with pytest.raises(asyncio.CancelledError):
            ev.emit("a")
        assert ev.errors == []

    def test_async_listener_without_loop_records_type_error_and_others_run(
        self, ev: Events
    ) -> None:
        ran: List[str] = []

        async def alistener() -> None:
            ran.append("async")  # pragma: no cover - must never run

        ev.on("a", lambda: ran.append("before"))
        ev.on("a", alistener)
        ev.on("a", lambda: ran.append("after"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a 'never awaited' warning would fail the test
            ev.emit("a")
            import gc

            gc.collect()
        assert ran == ["before", "after"]
        assert len(ev.errors) == 1
        assert isinstance(ev.errors[0].exception, TypeError)
        assert "no running event loop" in str(ev.errors[0].exception)

    async def test_async_listener_with_loop_is_scheduled(self, ev: Events) -> None:
        done = asyncio.Event()
        ran: List[str] = []

        async def alistener(x: int) -> None:
            await asyncio.sleep(0)
            ran.append(f"async{x}")
            done.set()

        ev.on("a", alistener)
        ev.on("a", lambda x: ran.append(f"sync{x}"))
        ev.emit("a", 1)
        assert ran == ["sync1"]  # emit did not wait
        assert ev.describe()["pending_async_listeners"] == 1
        await asyncio.wait_for(done.wait(), 1)
        await asyncio.sleep(0)
        assert ran == ["sync1", "async1"]
        assert ev.describe()["pending_async_listeners"] == 0

    async def test_scheduled_async_failure_is_recorded(self, ev: Events) -> None:
        async def bad() -> None:
            raise RuntimeError("late")

        ev.on("a", bad)
        ev.emit("a")
        for _ in range(5):
            await asyncio.sleep(0)
        assert len(ev.errors) == 1
        assert isinstance(ev.errors[0].exception, RuntimeError)

    async def test_scheduled_async_cancellation_is_not_recorded(self, ev: Events) -> None:
        started = asyncio.Event()

        async def slow() -> None:
            started.set()
            await asyncio.sleep(10)

        ev.on("a", slow)
        ev.emit("a")
        await started.wait()
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert ev.errors == []
        assert ev.describe()["pending_async_listeners"] == 0


# ----------------------------------------------------------------------
# Snapshot, removal and addition during dispatch
# ----------------------------------------------------------------------


class TestSnapshot:
    def test_listener_added_during_dispatch_not_run_in_that_dispatch(self, ev: Events) -> None:
        order: List[str] = []

        def adder() -> None:
            order.append("adder")
            ev.on("a", lambda: order.append("late"))

        ev.on("a", adder)
        ev.emit("a")
        assert order == ["adder"]
        order.clear()
        ev.emit("a")
        assert order == ["adder", "late"]  # the next dispatch sees the new listener

    def test_prepended_during_dispatch_not_run(self, ev: Events) -> None:
        order: List[str] = []

        def adder() -> None:
            order.append("adder")
            ev.on("a", lambda: order.append("p"), prepend=True)

        ev.on("a", adder)
        ev.emit("a")
        assert order == ["adder"]

    def test_listener_removed_before_its_turn_is_skipped(self, ev: Events) -> None:
        order: List[str] = []
        disposers: List[Disposer] = []

        def first() -> None:
            order.append("first")
            disposers[0]()

        ev.on("a", first)
        disposers.append(ev.on("a", lambda: order.append("second")))
        ev.on("a", lambda: order.append("third"))
        ev.emit("a")
        assert order == ["first", "third"]

    def test_listener_removing_itself_still_finishes_its_call(self, ev: Events) -> None:
        order: List[str] = []
        holder: List[Disposer] = []

        def me() -> None:
            holder[0]()
            order.append("me")

        holder.append(ev.on("a", me))
        ev.on("a", lambda: order.append("other"))
        ev.emit("a")
        assert order == ["me", "other"]

    def test_removed_listener_already_run_is_not_affected(self, ev: Events) -> None:
        order: List[str] = []
        d_first = ev.on("a", lambda: order.append("first"))
        ev.on("a", lambda: (order.append("second"), d_first()))
        ev.emit("a")
        assert order == ["first", "second"]
        order.clear()
        ev.emit("a")
        assert order == ["second"]

    def test_listener_removed_during_bail_is_skipped(self, ev: Events) -> None:
        disposers: List[Disposer] = []
        ev.on("a", lambda: disposers[0]() and None)
        disposers.append(ev.on("a", lambda: "should-not-run"))
        assert ev.bail("a") is None

    async def test_removal_during_serial_is_skipped(self, ev: Events) -> None:
        order: List[str] = []
        disposers: List[Disposer] = []

        async def first() -> None:
            order.append("first")
            disposers[0]()

        ev.on("a", first)
        disposers.append(ev.on("a", lambda: order.append("second")))
        ev.on("a", lambda: order.append("third"))
        await ev.serial("a")
        assert order == ["first", "third"]

    async def test_removal_during_parallel_before_turn_is_skipped(self, ev: Events) -> None:
        order: List[str] = []
        disposers: List[Disposer] = []

        def first() -> None:
            order.append("first")
            disposers[0]()

        ev.on("a", first)
        disposers.append(ev.on("a", lambda: order.append("second")))
        await ev.parallel("a")
        assert order == ["first"]

    def test_removal_during_waterfall_is_skipped(self, ev: Events) -> None:
        order: List[str] = []
        disposers: List[Disposer] = []

        def first(nxt: Any) -> Any:
            order.append("first")
            disposers[0]()
            return nxt()

        ev.on("a", first)
        disposers.append(ev.on("a", lambda nxt: order.append("second") or nxt()))
        ev.on("a", lambda nxt: order.append("third") or nxt())
        ev.waterfall("a", inner=lambda: order.append("inner"))
        assert order == ["first", "third", "inner"]

    def test_additions_during_waterfall_not_run(self, ev: Events) -> None:
        order: List[str] = []

        def first(nxt: Any) -> Any:
            ev.on("a", lambda n: order.append("late") or n())
            return nxt()

        ev.on("a", first)
        ev.waterfall("a", inner=lambda: order.append("inner"))
        assert order == ["inner"]


# ----------------------------------------------------------------------
# once
# ----------------------------------------------------------------------


class TestOnce:
    def test_runs_once(self, ev: Events) -> None:
        calls: List[int] = []
        ev.once("a", lambda: calls.append(1))
        ev.emit("a")
        ev.emit("a")
        assert calls == [1]
        assert ev.listener_count("a") == 0

    def test_removes_itself_before_running(self, ev: Events) -> None:
        seen: List[int] = []
        ev.once("a", lambda: seen.append(ev.listener_count("a")))
        ev.emit("a")
        assert seen == [0]

    def test_removed_even_if_it_raises(self, ev: Events) -> None:
        calls: List[int] = []

        def boom() -> None:
            calls.append(1)
            raise ValueError("x")

        ev.once("a", boom)
        ev.emit("a")
        ev.emit("a")
        assert calls == [1]
        assert len(ev.errors) == 1

    def test_removed_even_if_it_raises_in_bail(self, ev: Events) -> None:
        calls: List[int] = []

        def boom() -> None:
            calls.append(1)
            raise ValueError("x")

        ev.once("a", boom)
        with pytest.raises(ValueError):
            ev.bail("a")
        assert ev.bail("a") is None
        assert calls == [1]

    def test_reentrant_emit_runs_once_only(self, ev: Events) -> None:
        calls: List[int] = []

        def f() -> None:
            calls.append(1)
            ev.emit("a")  # re-entrant dispatch must not run it again

        ev.once("a", f)
        ev.emit("a")
        assert calls == [1]

    def test_second_dispatch_with_older_snapshot_does_not_rerun(self, ev: Events) -> None:
        calls: List[str] = []
        ev.on("a", lambda: ev.emit("a") if not calls else None)
        ev.once("a", lambda: calls.append("once"))
        ev.emit("a")
        assert calls == ["once"]

    def test_disposer_before_first_run_cancels_it(self, ev: Events) -> None:
        calls: List[int] = []
        d = ev.once("a", lambda: calls.append(1))
        d()
        ev.emit("a")
        assert calls == []
        assert d() is False

    async def test_once_in_serial_and_parallel(self, ev: Events) -> None:
        calls: List[str] = []
        ev.once("s", lambda: calls.append("s"))
        ev.once("p", lambda: calls.append("p"))
        await ev.serial("s")
        await ev.serial("s")
        await ev.parallel("p")
        await ev.parallel("p")
        assert calls == ["s", "p"]

    def test_once_in_waterfall(self, ev: Events) -> None:
        calls: List[str] = []
        ev.once("w", lambda nxt: calls.append("w") or nxt())
        ev.waterfall("w", inner=lambda: None)
        ev.waterfall("w", inner=lambda: None)
        assert calls == ["w"]


# ----------------------------------------------------------------------
# bail
# ----------------------------------------------------------------------


class TestBail:
    def test_empty(self, ev: Events) -> None:
        assert ev.bail("none") is None

    def test_first_result_wins_and_stops(self, ev: Events) -> None:
        order: List[str] = []
        ev.on("a", lambda: order.append("1"))  # returns None
        ev.on("a", lambda: order.append("2") or "win")
        ev.on("a", lambda: order.append("3") or "lose")
        assert ev.bail("a") == "win"
        assert order == ["1", "2"]

    def test_false_and_none_do_not_bail(self, ev: Events) -> None:
        ev.on("a", lambda: False)
        ev.on("a", lambda: None)
        ev.on("a", lambda: "yes")
        assert ev.bail("a") == "yes"

    def test_all_pass_returns_none(self, ev: Events) -> None:
        ev.on("a", lambda: None)
        ev.on("a", lambda: False)
        assert ev.bail("a") is None

    def test_zero_counts_as_result(self, ev: Events) -> None:
        ev.on("a", lambda: 0)
        ev.on("a", lambda: 99)
        result = ev.bail("a")
        assert result == 0 and result is not False

    def test_empty_string_counts_as_result(self, ev: Events) -> None:
        ev.on("a", lambda: "")
        ev.on("a", lambda: "later")
        assert ev.bail("a") == ""

    def test_empty_containers_count(self, ev: Events) -> None:
        ev.on("a", lambda: [])
        assert ev.bail("a") == []

    def test_exception_propagates_fails_closed(self, ev: Events) -> None:
        order: List[str] = []

        def boom() -> None:
            raise PermissionError("deny")

        ev.on("a", boom)
        ev.on("a", lambda: order.append("never") or "ok")
        with pytest.raises(PermissionError):
            ev.bail("a")
        assert order == []
        assert ev.errors == []  # propagated, not swallowed into the error list

    def test_async_listener_raises_type_error(self, ev: Events) -> None:
        async def alistener() -> bool:
            return True  # pragma: no cover - never runs

        ev.on("a", alistener)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(TypeError, match="awaitable"):
                ev.bail("a")
            import gc

            gc.collect()


# ----------------------------------------------------------------------
# serial
# ----------------------------------------------------------------------


class TestSerial:
    async def test_empty(self, ev: Events) -> None:
        assert await ev.serial("none") is None

    async def test_order_is_strictly_sequential(self, ev: Events) -> None:
        order: List[str] = []

        async def slow() -> None:
            order.append("slow-start")
            await asyncio.sleep(0.01)
            order.append("slow-end")

        async def fast() -> None:
            order.append("fast")

        ev.on("a", slow)
        ev.on("a", fast)
        ev.on("a", lambda: order.append("sync"))
        assert await ev.serial("a") is None
        assert order == ["slow-start", "slow-end", "fast", "sync"]

    async def test_mixed_sync_async_first_bailed_result(self, ev: Events) -> None:
        order: List[str] = []

        async def a1() -> None:
            order.append("a1")

        async def a2() -> str:
            order.append("a2")
            return "from-async"

        ev.on("a", a1)
        ev.on("a", lambda: order.append("s") or None)
        ev.on("a", a2)
        ev.on("a", lambda: order.append("never") or "never")
        assert await ev.serial("a") == "from-async"
        assert order == ["a1", "s", "a2"]

    async def test_zero_and_empty_string_count(self, ev: Events) -> None:
        async def zero() -> int:
            return 0

        ev.on("a", zero)
        ev.on("a", lambda: "later")
        assert await ev.serial("a") == 0
        ev2 = Events()
        ev2.on("a", lambda: "")
        assert await ev2.serial("a") == ""

    async def test_false_does_not_bail(self, ev: Events) -> None:
        async def f() -> bool:
            return False

        ev.on("a", f)
        assert await ev.serial("a") is None

    async def test_exception_propagates_and_stops(self, ev: Events) -> None:
        order: List[str] = []

        async def boom() -> None:
            raise KeyError("k")

        ev.on("a", boom)
        ev.on("a", lambda: order.append("never"))
        with pytest.raises(KeyError):
            await ev.serial("a")
        assert order == []

    async def test_cancellation_propagates(self, ev: Events) -> None:
        started = asyncio.Event()

        async def hang() -> None:
            started.set()
            await asyncio.sleep(10)

        ev.on("a", hang)
        task = asyncio.ensure_future(ev.serial("a"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ----------------------------------------------------------------------
# parallel
# ----------------------------------------------------------------------


class TestParallel:
    async def test_empty(self, ev: Events) -> None:
        assert await ev.parallel("none") is None

    async def test_runs_together_and_returns_nothing(self, ev: Events) -> None:
        started: List[str] = []
        gate = asyncio.Event()

        async def waiter(tag: str) -> None:
            started.append(tag)
            await gate.wait()

        ev.on("a", lambda: waiter("a"))
        ev.on("a", lambda: waiter("b"))
        ev.on("a", lambda: waiter("c"))
        task = asyncio.ensure_future(ev.parallel("a"))
        for _ in range(5):
            await asyncio.sleep(0)
        assert started == ["a", "b", "c"]  # all started before any finished
        assert not task.done()
        gate.set()
        assert await task is None

    async def test_sync_listeners_work(self, ev: Events) -> None:
        got: List[int] = []
        ev.on("a", lambda: got.append(1))
        ev.on("a", lambda: got.append(2))
        await ev.parallel("a")
        assert got == [1, 2]

    async def test_one_failure_all_still_run(self, ev: Events) -> None:
        ran: List[str] = []

        async def ok(tag: str) -> None:
            await asyncio.sleep(0.01)
            ran.append(tag)

        def boom() -> None:
            raise ValueError("one")

        ev.on("a", lambda: ok("x"))
        ev.on("a", boom)
        ev.on("a", lambda: ok("y"))
        with pytest.raises(ListenerErrors) as info:
            await ev.parallel("a")
        assert sorted(ran) == ["x", "y"]
        exc = info.value
        assert exc.event == "a"
        assert len(exc.errors) == 1
        assert isinstance(exc.exceptions[0], ValueError)
        assert "one" in str(exc)

    async def test_several_failures_all_reported_in_listener_order(self, ev: Events) -> None:
        async def slow_fail() -> None:
            await asyncio.sleep(0.02)
            raise ValueError("slow")

        async def fast_fail() -> None:
            raise KeyError("fast")

        ev.on("a", slow_fail)
        ev.on("a", fast_fail)
        ev.on("a", lambda: None)
        with pytest.raises(ListenerErrors) as info:
            await ev.parallel("a")
        kinds = [type(e) for e in info.value.exceptions]
        assert kinds == [ValueError, KeyError]  # listener order, not finish order
        assert all(isinstance(e, ListenerError) for e in info.value.errors)

    async def test_failures_not_double_recorded_in_events_errors(self, ev: Events) -> None:
        ev.on("a", lambda: 1 / 0)
        with pytest.raises(ListenerErrors):
            await ev.parallel("a")
        assert ev.errors == []

    async def test_outer_cancellation_cancels_listeners_and_propagates(self, ev: Events) -> None:
        started = asyncio.Event()
        cancelled: List[str] = []

        async def hang() -> None:
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append("hang")
                raise

        ev.on("a", hang)
        task = asyncio.ensure_future(ev.parallel("a"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled == ["hang"]

    async def test_listener_raising_cancelled_error_is_not_a_listener_failure(
        self, ev: Events
    ) -> None:
        async def cancel() -> None:
            raise asyncio.CancelledError()

        ev.on("a", cancel)
        with pytest.raises(asyncio.CancelledError):
            await ev.parallel("a")


# ----------------------------------------------------------------------
# waterfall
# ----------------------------------------------------------------------


class TestWaterfall:
    def test_empty_calls_inner(self, ev: Events) -> None:
        assert ev.waterfall("none", 1, inner=lambda: "inner") == "inner"

    def test_inner_is_required_keyword(self, ev: Events) -> None:
        with pytest.raises(TypeError):
            ev.waterfall("a")  # type: ignore[call-arg]

    def test_onion_order_and_args(self, ev: Events) -> None:
        order: List[str] = []

        def outer(x: int, y: int, nxt: Any) -> Any:
            order.append(f"outer-in {x} {y}")
            r = nxt()
            order.append("outer-out")
            return f"outer({r})"

        def middle(x: int, y: int, nxt: Any) -> Any:
            order.append("middle-in")
            r = nxt()
            order.append("middle-out")
            return f"middle({r})"

        ev.on("a", outer)
        ev.on("a", middle)
        result = ev.waterfall("a", 1, 2, inner=lambda: order.append("inner") or "core")
        assert result == "outer(middle(core))"
        assert order == ["outer-in 1 2", "middle-in", "inner", "middle-out", "outer-out"]

    def test_prepend_wraps_outside(self, ev: Events) -> None:
        ev.on("a", lambda nxt: f"b({nxt()})")
        ev.on("a", lambda nxt: f"a({nxt()})", prepend=True)
        assert ev.waterfall("a", inner=lambda: "x") == "a(b(x))"

    def test_listener_that_never_calls_next_ends_chain(self, ev: Events) -> None:
        order: List[str] = []
        ev.on("a", lambda nxt: "short-circuit")
        ev.on("a", lambda nxt: order.append("never") or nxt())
        result = ev.waterfall("a", inner=lambda: order.append("inner-never"))
        assert result == "short-circuit"
        assert order == []

    def test_listener_can_change_result_and_ignore_inner_value(self, ev: Events) -> None:
        ev.on("a", lambda nxt: (nxt(), "replaced")[1])
        assert ev.waterfall("a", inner=lambda: "orig") == "replaced"

    def test_double_next_raises(self, ev: Events) -> None:
        def twice(nxt: Any) -> Any:
            nxt()
            return nxt()

        ev.on("a", twice)
        with pytest.raises(NextAlreadyCalledError, match="more than once"):
            ev.waterfall("a", inner=lambda: 1)

    def test_double_next_does_not_run_inner_twice(self, ev: Events) -> None:
        inner_calls: List[int] = []

        def twice(nxt: Any) -> Any:
            nxt()
            try:
                nxt()
            except NextAlreadyCalledError:
                pass
            return "done"

        ev.on("a", twice)
        assert ev.waterfall("a", inner=lambda: inner_calls.append(1)) == "done"
        assert inner_calls == [1]

    def test_next_error_is_a_runtime_error(self) -> None:
        assert issubclass(NextAlreadyCalledError, RuntimeError)

    def test_exception_after_next_propagates(self, ev: Events) -> None:
        order: List[str] = []

        def after(nxt: Any) -> Any:
            nxt()
            order.append("after-next")
            raise ValueError("post")

        ev.on("a", after)
        with pytest.raises(ValueError, match="post"):
            ev.waterfall("a", inner=lambda: order.append("inner"))
        assert order == ["inner", "after-next"]

    def test_exception_before_next_and_in_inner_propagate(self, ev: Events) -> None:
        def boom(nxt: Any) -> Any:
            raise KeyError("pre")

        ev.on("a", boom)
        with pytest.raises(KeyError):
            ev.waterfall("a", inner=lambda: 1)

        ev2 = Events()
        ev2.on("a", lambda nxt: nxt())

        def bad_inner() -> None:
            raise OSError("inner")

        with pytest.raises(OSError):
            ev2.waterfall("a", inner=bad_inner)

    def test_inner_exception_can_be_handled_by_listener(self, ev: Events) -> None:
        def guard(nxt: Any) -> Any:
            try:
                return nxt()
            except ZeroDivisionError:
                return "recovered"

        ev.on("a", guard)
        assert ev.waterfall("a", inner=lambda: 1 / 0) == "recovered"

    def test_sync_waterfall_with_async_listener_returns_awaitable(self, ev: Events) -> None:
        async def alistener(nxt: Any) -> Any:
            return f"a({nxt()})"

        ev.on("a", alistener)
        result = ev.waterfall("a", inner=lambda: "core")
        assert inspect.isawaitable(result)
        assert asyncio.run(_await_it(result)) == "a(core)"

    async def test_async_listener_awaits_next(self, ev: Events) -> None:
        order: List[str] = []

        async def a1(nxt: Any) -> Any:
            order.append("a1-in")
            r = await nxt()
            order.append("a1-out")
            return f"a1({r})"

        async def a2(nxt: Any) -> Any:
            order.append("a2-in")
            await asyncio.sleep(0)
            r = nxt()  # inner is sync here
            order.append("a2-out")
            return f"a2({r})"

        ev.on("a", a1)
        ev.on("a", a2)
        result = await ev.awaterfall("a", inner=lambda: "core")
        assert result == "a1(a2(core))"
        assert order == ["a1-in", "a2-in", "a2-out", "a1-out"]

    async def test_mixed_sync_and_async_layers(self, ev: Events) -> None:
        async def async_inner() -> str:
            await asyncio.sleep(0)
            return "core"

        def sync_layer(nxt: Any) -> Any:
            return nxt()  # passes the awaitable through untouched

        async def async_layer(nxt: Any) -> Any:
            return f"A({await nxt()})"

        ev.on("a", sync_layer)
        ev.on("a", async_layer)
        assert await ev.awaterfall("a", inner=async_inner) == "A(core)"

    async def test_awaterfall_with_sync_everything(self, ev: Events) -> None:
        ev.on("a", lambda nxt: nxt() + 1)
        assert await ev.awaterfall("a", inner=lambda: 1) == 2
        assert await Events().awaterfall("none", inner=lambda: "i") == "i"

    async def test_awaterfall_flattens_returned_but_unawaited_next(self, ev: Events) -> None:
        async def passthrough(nxt: Any) -> Any:
            return nxt()  # forgot to await: returns the inner coroutine

        async def async_inner() -> str:
            return "core"

        ev.on("a", passthrough)
        assert await ev.awaterfall("a", inner=async_inner) == "core"

    async def test_next_after_an_await_raises(self, ev: Events) -> None:
        inner_calls: List[int] = []

        async def sneaky(nxt: Any) -> Any:
            first = nxt()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            second = nxt()  # called again after an await
            return first, second  # pragma: no cover

        ev.on("a", sneaky)
        with pytest.raises(NextAlreadyCalledError):
            await ev.awaterfall("a", inner=lambda: inner_calls.append(1))
        assert inner_calls == [1]

    async def test_awaited_next_then_second_next_raises(self, ev: Events) -> None:
        async def twice(nxt: Any) -> Any:
            async def get() -> Any:
                return nxt()

            await get()
            await asyncio.sleep(0)
            return nxt()

        ev.on("a", twice)
        with pytest.raises(NextAlreadyCalledError):
            await ev.awaterfall("a", inner=lambda: 1)

    async def test_async_never_calling_next(self, ev: Events) -> None:
        async def stop(nxt: Any) -> str:
            return "stopped"

        ev.on("a", stop)
        ev.on("a", lambda nxt: nxt())
        called: List[int] = []
        assert await ev.awaterfall("a", inner=lambda: called.append(1)) == "stopped"
        assert called == []

    async def test_async_exception_after_next(self, ev: Events) -> None:
        async def after(nxt: Any) -> Any:
            await nxt()
            raise ValueError("late")

        ev.on("a", after)

        async def inner() -> int:
            return 1

        with pytest.raises(ValueError, match="late"):
            await ev.awaterfall("a", inner=inner)

    async def test_cancellation_propagates_through_chain(self, ev: Events) -> None:
        started = asyncio.Event()

        async def hang_inner() -> None:
            started.set()
            await asyncio.sleep(10)

        async def layer(nxt: Any) -> Any:
            return await nxt()

        ev.on("a", layer)
        task = asyncio.ensure_future(ev.awaterfall("a", inner=hang_inner))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    def test_waterfall_errors_not_recorded(self, ev: Events) -> None:
        ev.on("a", lambda nxt: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            ev.waterfall("a", inner=lambda: 1)
        assert ev.errors == []


async def _await_it(aw: Any) -> Any:
    return await aw


# ----------------------------------------------------------------------
# parallel: no sibling outlives the call
# ----------------------------------------------------------------------


class TestParallelCleanup:
    async def _run_and_check(self, ev: Events, bad: Any, expected: Any) -> None:
        """Run parallel with a failing listener plus hanging siblings."""
        loop = asyncio.get_running_loop()
        loop_problems: List[Any] = []
        loop.set_exception_handler(lambda _loop, ctx: loop_problems.append(ctx))
        finished: List[str] = []

        async def sibling(tag: str) -> None:
            try:
                await asyncio.sleep(10)
            finally:
                finished.append(tag)

        ev.on("a", lambda: sibling("s1"))
        ev.on("a", bad)
        ev.on("a", lambda: sibling("s2"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(expected):
                await ev.parallel("a")
            # The siblings are already finished here, not merely cancelled.
            assert sorted(finished) == ["s1", "s2"]
            leftover = [
                t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()
            ]
            assert leftover == []
            gc.collect()
            await asyncio.sleep(0)
            gc.collect()
        assert loop_problems == []

    async def test_listener_raising_cancelled_error_cancels_and_finishes_siblings(
        self, ev: Events
    ) -> None:
        async def cancel() -> None:
            await asyncio.sleep(0)  # let the siblings start first
            raise asyncio.CancelledError()

        await self._run_and_check(ev, cancel, asyncio.CancelledError)

    async def test_listener_raising_base_exception_cancels_and_finishes_siblings(
        self, ev: Events
    ) -> None:
        class Stop(BaseException):
            pass

        async def stop() -> None:
            await asyncio.sleep(0)
            raise Stop()

        await self._run_and_check(ev, stop, Stop)

    async def test_outer_cancel_waits_for_siblings(self, ev: Events) -> None:
        finished: List[str] = []
        started = asyncio.Event()

        async def sibling(tag: str) -> None:
            started.set()
            try:
                await asyncio.sleep(10)
            finally:
                await asyncio.sleep(0)  # slow cleanup must be waited for
                finished.append(tag)

        ev.on("a", lambda: sibling("s1"))
        ev.on("a", lambda: sibling("s2"))
        task = asyncio.ensure_future(ev.parallel("a"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sorted(finished) == ["s1", "s2"]

    async def test_ordinary_failures_still_collected_with_siblings_complete(
        self, ev: Events
    ) -> None:
        finished: List[str] = []

        async def slow_ok() -> None:
            await asyncio.sleep(0.01)
            finished.append("ok")

        ev.on("a", slow_ok)
        ev.on("a", lambda: 1 / 0)
        with pytest.raises(ListenerErrors):
            await ev.parallel("a")
        assert finished == ["ok"]


# ----------------------------------------------------------------------
# Concurrent next()
# ----------------------------------------------------------------------


class TestConcurrentNext:
    async def test_two_tasks_calling_next_get_exactly_one_error(self, ev: Events) -> None:
        inner_calls: List[int] = []

        async def inner() -> str:
            inner_calls.append(1)
            await asyncio.sleep(0)
            return "core"

        async def layer(nxt: Any) -> Any:
            async def call() -> Any:
                await asyncio.sleep(0)
                return await nxt()

            return await asyncio.gather(call(), call(), return_exceptions=True)

        ev.on("a", layer)
        results = await ev.awaterfall("a", inner=inner)
        errors = [r for r in results if isinstance(r, NextAlreadyCalledError)]
        values = [r for r in results if r == "core"]
        assert len(errors) == 1
        assert len(values) == 1
        assert inner_calls == [1]


# ----------------------------------------------------------------------
# Memory: removal releases the listener
# ----------------------------------------------------------------------


def _make_listener(seen: List[Any]) -> Any:
    def listener() -> None:
        seen.append(1)

    return listener


class TestRelease:
    def test_disposed_listener_is_not_kept_alive_by_the_disposer(self, ev: Events) -> None:
        fn = _make_listener([])
        ref = weakref.ref(fn)
        d = ev.on("a", fn)
        del fn
        gc.collect()
        assert ref() is not None  # still registered
        d()
        gc.collect()
        assert ref() is None
        assert d() is False  # the disposer itself is still alive and harmless

    def test_once_listener_released_after_its_call(self, ev: Events) -> None:
        seen: List[Any] = []
        fn = _make_listener(seen)
        ref = weakref.ref(fn)
        d = ev.once("a", fn)
        del fn
        ev.emit("a")
        gc.collect()
        assert seen == [1]
        assert ref() is None
        assert d.active is False

    def test_once_function_survives_until_its_call_completes(self, ev: Events) -> None:
        alive_during_call: List[bool] = []
        holder: List[Any] = []

        def make() -> Any:
            def listener() -> None:
                gc.collect()
                alive_during_call.append(holder[0]() is not None)

            return listener

        fn = make()
        holder.append(weakref.ref(fn))
        ev.once("a", fn)
        del fn
        ev.emit("a")
        assert alive_during_call == [True]


# ----------------------------------------------------------------------
# Scale and compaction
# ----------------------------------------------------------------------

N_SCALE = 20_000
TIME_BOUND = 2.0


class TestScale:
    def test_many_listeners_register_emit_dispose(self) -> None:
        ev = Events()
        calls = [0]

        def listener() -> None:
            calls[0] += 1

        t0 = time.perf_counter()
        disposers = [ev.on("a", listener) for _ in range(N_SCALE)]
        assert time.perf_counter() - t0 < TIME_BOUND
        assert ev.listener_count("a") == N_SCALE
        assert ev.describe()["listeners"] == {"a": N_SCALE}

        t0 = time.perf_counter()
        ev.emit("a")
        assert time.perf_counter() - t0 < TIME_BOUND
        assert calls[0] == N_SCALE

        t0 = time.perf_counter()
        for d in disposers:
            assert d() is True
        assert time.perf_counter() - t0 < TIME_BOUND
        assert ev.listener_count("a") == 0
        assert ev.describe()["listeners"] == {}
        ev.emit("a")
        assert calls[0] == N_SCALE

    def test_many_once_listeners(self) -> None:
        ev = Events()
        calls = [0]

        def listener() -> None:
            calls[0] += 1

        t0 = time.perf_counter()
        for _ in range(N_SCALE):
            ev.once("a", listener)
        assert time.perf_counter() - t0 < TIME_BOUND
        assert ev.listener_count("a") == N_SCALE

        t0 = time.perf_counter()
        ev.emit("a")
        assert time.perf_counter() - t0 < TIME_BOUND
        assert calls[0] == N_SCALE
        assert ev.listener_count("a") == 0
        ev.emit("a")
        assert calls[0] == N_SCALE

    def test_many_prepends_keep_order(self) -> None:
        ev = Events()
        order: List[int] = []
        t0 = time.perf_counter()
        for i in range(N_SCALE):
            ev.on("a", functools.partial(order.append, i), prepend=True)
        assert time.perf_counter() - t0 < TIME_BOUND
        assert ev.listener_count("a") == N_SCALE
        ev.emit("a")
        assert order == list(range(N_SCALE - 1, -1, -1))

    def test_live_count_and_order_survive_compaction(self, ev: Events) -> None:
        order: List[str] = []
        disposers: List[Disposer] = []
        for i in range(200):
            disposers.append(ev.on("a", functools.partial(order.append, f"b{i}")))
        for i in range(200):
            disposers.append(
                ev.on("a", functools.partial(order.append, f"f{i}"), prepend=True)
            )
        # Keep every 10th registration; removing the rest compacts several times.
        kept = [idx for idx in range(len(disposers)) if idx % 10 == 0]
        for idx, d in enumerate(disposers):
            if idx % 10 != 0:
                d()
        assert ev.listener_count("a") == len(kept) == 40
        ev.emit("a")
        expected_front = [f"f{i}" for i in range(199, -1, -1) if (200 + i) % 10 == 0]
        expected_back = [f"b{i}" for i in range(200) if i % 10 == 0]
        assert order == expected_front + expected_back
        # Adding after compaction still works and keeps ordering.
        order.clear()
        ev.on("a", functools.partial(order.append, "new"))
        ev.on("a", functools.partial(order.append, "newfront"), prepend=True)
        ev.emit("a")
        assert order[0] == "newfront" and order[-1] == "new"
        assert ev.listener_count("a") == 42

    def test_snapshot_taken_before_compaction_is_unaffected(self, ev: Events) -> None:
        order: List[int] = []
        disposers: List[Disposer] = []

        def remover() -> None:
            # Removing many listeners mid dispatch compacts storage; the running
            # dispatch must still skip exactly the removed ones.
            for d in disposers[:75]:
                d()

        ev.on("a", remover)
        for i in range(100):
            disposers.append(ev.on("a", functools.partial(order.append, i)))
        ev.emit("a")
        assert order == list(range(75, 100))
        assert ev.listener_count("a") == 26


# ----------------------------------------------------------------------
# Threads
# ----------------------------------------------------------------------


def _run_threads(target: Any, count: int) -> None:
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=target, args=(k,)) for k in range(count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old)


class TestThreads:
    def test_concurrent_registration_loses_nothing(self, ev: Events) -> None:
        threads_n, per_thread = 4, 2000
        tokens: List[List[int]] = [[] for _ in range(threads_n)]
        barrier = threading.Barrier(threads_n)
        failures: List[BaseException] = []

        def work(k: int) -> None:
            try:
                barrier.wait()
                for _ in range(per_thread):
                    tokens[k].append(ev.on("a", lambda: None, prepend=(k % 2 == 0)).token)
            except BaseException as exc:  # pragma: no cover - only on failure
                failures.append(exc)

        _run_threads(work, threads_n)
        assert failures == []
        assert ev.listener_count("a") == 8000
        assert ev.describe()["listeners"] == {"a": 8000}
        all_tokens = [tok for ts in tokens for tok in ts]
        assert len(set(all_tokens)) == 8000
        ev.emit("a")
        assert ev.errors == []

    def test_every_registered_listener_runs(self, ev: Events) -> None:
        threads_n, per_thread = 4, 500
        seen: List[int] = []
        barrier = threading.Barrier(threads_n)

        def work(k: int) -> None:
            barrier.wait()
            for i in range(per_thread):
                ev.on("a", functools.partial(seen.append, k * per_thread + i))

        _run_threads(work, threads_n)
        ev.emit("a")
        assert sorted(seen) == list(range(threads_n * per_thread))

    def test_no_stale_snapshot_when_emitting_while_registering(self, ev: Events) -> None:
        registrars, per_thread = 3, 1500
        done = threading.Event()
        seen: List[int] = []

        def work(k: int) -> None:
            if k == registrars:  # the emitting thread
                while not done.is_set():
                    ev.emit("a")
                return
            for i in range(per_thread):
                ev.on("a", functools.partial(seen.append, 1))

        old = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            threads = [threading.Thread(target=work, args=(k,)) for k in range(registrars + 1)]
            for t in threads:
                t.start()
            for t in threads[:registrars]:
                t.join()
            done.set()
            threads[registrars].join()
        finally:
            sys.setswitchinterval(old)
        seen.clear()
        ev.emit("a")
        assert len(seen) == registrars * per_thread == ev.listener_count("a")

    def test_concurrent_register_and_dispose(self, ev: Events) -> None:
        threads_n, per_thread = 4, 1000
        barrier = threading.Barrier(threads_n)

        def work(_k: int) -> None:
            barrier.wait()
            ds = [ev.on("a", lambda: None) for _ in range(per_thread)]
            for d in ds[::2]:
                d()

        _run_threads(work, threads_n)
        assert ev.listener_count("a") == threads_n * per_thread // 2


# ----------------------------------------------------------------------
# describe
# ----------------------------------------------------------------------


class TestDescribe:
    def test_empty(self, ev: Events) -> None:
        assert ev.describe() == {
            "listeners": {},
            "error_count": 0,
            "dropped_errors": 0,
            "pending_async_listeners": 0,
        }

    def test_counts_listeners_and_errors(self, ev: Events) -> None:
        ev.on("a", lambda: 1 / 0)
        d = ev.on("a", lambda: None)
        ev.on("b", lambda: None)
        ev.once("c", lambda: None)
        ev.emit("a")
        info = ev.describe()
        assert info["listeners"] == {"a": 2, "b": 1, "c": 1}
        assert info["error_count"] == 1
        d()
        ev.emit("c")
        info = ev.describe()
        assert info["listeners"] == {"a": 1, "b": 1}  # empty events are not listed
        ev.take_errors()
        assert ev.describe()["error_count"] == 0

    def test_describe_is_a_copy(self, ev: Events) -> None:
        ev.on("a", lambda: None)
        info = ev.describe()
        info["listeners"]["a"] = 99
        assert ev.describe()["listeners"]["a"] == 1
