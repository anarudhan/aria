"""Tests for aria.kernel.scope (issue #39).

No test depends on timing. Waiting is done with ``asyncio.Event`` and bare
``await asyncio.sleep(0)`` turns of the loop.
"""

from __future__ import annotations

import asyncio
import gc
import json
import random
import sys
import warnings
import weakref
from typing import Any, Callable, Dict, List, Optional

import pytest

from aria.kernel import (
    ChildScopeErrors,
    Events,
    Remover,
    Scope,
    ScopeDisposedError,
    ScopeError,
    ScopeState,
)
from aria.kernel import scope as scope_module
from aria.kernel.scope import _DISPOSING


def rec(log: List[Any], name: Any) -> Callable[[], Callable[[], None]]:
    """An effect that, when undone, appends ``name`` to ``log``."""

    def effect() -> Callable[[], None]:
        return lambda: log.append(name)

    return effect


async def turns(n: int = 5) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


# ----------------------------------------------------------------------
# Effects and disposers
# ----------------------------------------------------------------------


def test_state_enum_has_the_four_states() -> None:
    assert [s.name for s in ScopeState] == ["PENDING", "ACTIVE", "FAILED", "DISPOSED"]


async def test_effect_may_return_none() -> None:
    ran: List[str] = []
    s = Scope(name="s")
    s.effect(lambda: ran.append("fn"))  # returns None
    assert ran == ["fn"]
    assert s.effect_count == 1
    await s.dispose()
    assert s.errors == []


async def test_effect_runs_immediately_and_disposer_runs_on_dispose() -> None:
    log: List[str] = []
    s = Scope(name="s")
    s.effect(rec(log, "a"))
    assert log == []
    await s.dispose()
    assert log == ["a"]
    assert s.state is ScopeState.DISPOSED


async def test_list_and_tuple_of_disposers_run_newest_first() -> None:
    log: List[str] = []
    s = Scope(name="s")
    s.effect(lambda: [lambda: log.append("l1"), lambda: log.append("l2")])
    s.effect(lambda: (lambda: log.append("t1"), lambda: log.append("t2")))
    await s.dispose()
    assert log == ["t2", "t1", "l2", "l1"]


async def test_awaitable_effect_resolving_to_a_disposer() -> None:
    log: List[str] = []
    s = Scope(name="s")

    async def connect() -> Callable[[], None]:
        await asyncio.sleep(0)
        return lambda: log.append("closed")

    s.effect(connect)
    await turns()  # let it resolve (an effect still running at dispose is cancelled)
    await s.dispose()
    assert log == ["closed"]


async def test_awaitable_effect_resolving_to_list_or_none() -> None:
    log: List[str] = []
    s = Scope(name="s")

    async def many() -> Any:
        await asyncio.sleep(0)
        return [lambda: log.append("x"), lambda: log.append("y")]

    async def nothing() -> None:
        await asyncio.sleep(0)

    s.effect(many)
    s.effect(nothing)
    await turns()
    await s.dispose()
    assert log == ["y", "x"]
    assert s.errors == []


async def test_async_disposer_is_awaited() -> None:
    log: List[str] = []
    s = Scope(name="s")

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            await asyncio.sleep(0)
            log.append("async-undo")

        return undo

    s.effect(effect)
    await s.dispose()
    assert log == ["async-undo"]


async def test_disposal_is_strictly_sequential_and_reverse() -> None:
    running = 0
    peak = 0
    order: List[int] = []

    def make(i: int) -> Callable[[], Callable[[], Any]]:
        def effect() -> Callable[[], Any]:
            async def undo() -> None:
                nonlocal running, peak
                running += 1
                peak = max(peak, running)
                await turns(3)  # yields to the loop several times
                order.append(i)
                running -= 1

            return undo

        return effect

    s = Scope(name="s")
    for i in range(6):
        s.effect(make(i))
    await s.dispose()
    assert order == [5, 4, 3, 2, 1, 0]
    assert peak == 1  # Cordis would run them together


async def test_children_are_disposed_before_own_effects_newest_first() -> None:
    log: List[str] = []
    root = Scope(name="root")
    root.effect(rec(log, "r1"))
    c1 = root.child("c1")
    c1.effect(rec(log, "c1-a"))
    c1.effect(rec(log, "c1-b"))
    root.effect(rec(log, "r2"))
    c2 = root.child("c2")
    c2.effect(rec(log, "c2-a"))
    gc1 = c1.child("c1-g")
    gc1.effect(rec(log, "g"))
    await root.dispose()
    assert log == ["c2-a", "g", "c1-b", "c1-a", "r2", "r1"]
    assert all(s.state is ScopeState.DISPOSED for s in (root, c1, c2, gc1))


async def test_errors_do_not_stop_other_disposers() -> None:
    log: List[str] = []
    s = Scope(name="s")
    s.effect(rec(log, "first"))

    def bad_sync() -> Callable[[], None]:
        def undo() -> None:
            raise ValueError("sync boom")

        return undo

    def bad_async() -> Callable[[], Any]:
        async def undo() -> None:
            await asyncio.sleep(0)
            raise KeyError("async boom")

        return undo

    s.effect(bad_sync, label="bad-sync")
    s.effect(rec(log, "middle"))
    s.effect(bad_async, label="bad-async")
    s.effect(rec(log, "last"))
    await s.dispose()  # must not raise
    assert log == ["last", "middle", "first"]
    errors = s.errors
    assert [e.label for e in errors] == ["bad-async", "bad-sync"]
    assert isinstance(errors[0].exception, KeyError)
    assert isinstance(errors[1].exception, ValueError)
    assert all(isinstance(e, ScopeError) and e.scope == "s" for e in errors)
    assert all(isinstance(e.when, float) and e.when > 0 for e in errors)
    assert errors[0].when <= errors[1].when
    assert "bad-sync" in str(errors[1]) and "ValueError" in str(errors[1])


async def test_error_inside_a_disposer_list_does_not_skip_the_rest() -> None:
    log: List[str] = []

    def boom() -> None:
        raise RuntimeError("x")

    s = Scope(name="s")
    s.effect(lambda: [lambda: log.append("a"), boom, lambda: log.append("c")])
    await s.dispose()
    assert log == ["c", "a"]
    assert len(s.errors) == 1


async def test_errors_are_bounded_and_can_be_taken() -> None:
    s = Scope(name="s", max_errors=3)

    def bad() -> Callable[[], None]:
        def undo() -> None:
            raise RuntimeError("x")

        return undo

    for _ in range(5):
        s.effect(bad)
    await s.dispose()
    assert len(s.errors) == 3
    assert s.describe()["dropped_errors"] == 2
    assert len(s.take_errors()) == 3
    assert s.errors == []
    with pytest.raises(ValueError):
        Scope(max_errors=0)


# ----------------------------------------------------------------------
# Removers
# ----------------------------------------------------------------------


async def test_remover_undoes_one_effect_early_and_is_idempotent() -> None:
    log: List[str] = []
    s = Scope(name="s")
    s.effect(rec(log, "a"))
    remove_b = s.effect(rec(log, "b"))
    s.effect(rec(log, "c"))
    assert isinstance(remove_b, Remover) and remove_b.active
    await remove_b()
    assert log == ["b"] and not remove_b.active
    await remove_b()  # second call does nothing
    remove_b()  # not awaited: nothing pending, so nothing to warn about
    assert log == ["b"]
    await s.dispose()
    assert log == ["b", "c", "a"]


async def test_sync_remover_has_already_run_without_awaiting() -> None:
    log: List[str] = []
    s = Scope(name="s")
    remove = s.effect(rec(log, "a"))
    remove()
    assert log == ["a"]


async def test_removing_a_never_removes_b_even_after_other_removals() -> None:
    # Cordis List.push deleted the *current* index at dispose time. Removers
    # here are bound to their own effect.
    log: List[str] = []
    s = Scope(name="s")
    removers = [s.effect(rec(log, name)) for name in "abcde"]
    removers[1]()  # b
    removers[3]()  # d
    removers[0]()  # a
    removers[1]()  # b again, must not touch anything
    removers[4]()  # e
    assert log == ["b", "d", "a", "e"]
    await s.dispose()
    assert log == ["b", "d", "a", "e", "c"]


async def test_late_binding_trap_with_loop_variables() -> None:
    log: List[int] = []
    s = Scope(name="s")
    removers = []
    for i in range(5):
        removers.append(s.effect(lambda i=i: (lambda: log.append(i))))
    for i in (3, 1, 3, 4, 0):
        removers[i]()
    assert log == [3, 1, 4, 0]
    await s.dispose()
    assert log == [3, 1, 4, 0, 2]


async def test_same_function_registered_twice_is_two_effects() -> None:
    log: List[str] = []

    def undo() -> None:
        log.append("undo")

    def effect() -> Callable[[], None]:
        return undo

    s = Scope(name="s")
    r1 = s.effect(effect)
    r2 = s.effect(effect)
    assert s.effect_count == 2
    r1()
    assert log == ["undo"] and s.effect_count == 1 and r2.active
    await s.dispose()
    assert log == ["undo", "undo"]


async def test_remover_after_dispose_is_a_noop() -> None:
    log: List[str] = []
    s = Scope(name="s")
    remove = s.effect(rec(log, "a"))
    await s.dispose()
    await remove()
    assert log == ["a"] and s.errors == []


async def test_remover_with_async_disposer_must_be_awaited() -> None:
    log: List[str] = []
    s = Scope(name="s")

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            await asyncio.sleep(0)
            log.append("done")

        return undo

    remove = s.effect(effect)
    aw = remove()
    assert log == []
    await aw
    assert log == ["done"]
    await s.dispose()
    assert log == ["done"]


async def test_remover_of_a_pending_async_effect_waits_for_it() -> None:
    log: List[str] = []
    gate = asyncio.Event()
    s = Scope(name="s")

    async def slow() -> Callable[[], None]:
        try:
            await gate.wait()
        except asyncio.CancelledError:
            await gate.wait()  # ignores the request to stop and finishes anyway
        return lambda: log.append("undone")

    remove = s.effect(slow)
    await turns()  # the effect is running (a cancel before its first step would end it)
    t = asyncio.ensure_future(remove())
    await turns()
    assert not t.done() and log == []
    gate.set()
    await t
    assert log == ["undone"]
    await s.dispose()
    assert log == ["undone"]


async def test_removed_effects_release_their_closures() -> None:
    class Payload:
        pass

    s = Scope(name="s")
    p = Payload()
    ref = weakref.ref(p)
    remove = s.effect(lambda p=p: (lambda: p))
    del p
    gc.collect()
    assert ref() is not None  # the scope holds it
    remove()
    gc.collect()
    assert ref() is None
    p2 = Payload()
    ref2 = weakref.ref(p2)
    s.effect(lambda p2=p2: (lambda: p2))
    del p2
    await s.dispose()
    gc.collect()
    assert ref2() is None


async def test_many_effects_removed_in_random_order() -> None:
    log: List[int] = []
    s = Scope(name="s")
    n = 5000
    removers = [s.effect(lambda i=i: (lambda: log.append(i))) for i in range(n)]
    order = list(range(n))
    random.Random(7).shuffle(order)
    for i in order[: n // 2]:
        removers[i]()
    assert log == order[: n // 2]
    await s.dispose()
    kept = sorted(order[n // 2 :], reverse=True)
    assert log[n // 2 :] == kept


# ----------------------------------------------------------------------
# Events integration
# ----------------------------------------------------------------------


async def test_on_registers_a_listener_removed_with_the_scope() -> None:
    events = Events()
    s = Scope(events, name="s")
    got: List[int] = []
    s.on("tick", got.append)
    events.emit("tick", 1)
    assert events.listener_count("tick") == 1
    await s.dispose()
    events.emit("tick", 2)
    assert got == [1]
    assert events.listener_count("tick") == 0


async def test_on_returns_a_remover_and_same_listener_twice_is_two_effects() -> None:
    events = Events()
    s = Scope(events, name="s")
    got: List[int] = []
    r1 = s.on("tick", got.append)
    s.on("tick", got.append)
    assert events.listener_count("tick") == 2
    r1()
    assert events.listener_count("tick") == 1
    events.emit("tick", 1)
    assert got == [1]  # the second registration is still alive
    await s.dispose()
    assert events.listener_count("tick") == 0


async def test_on_prepend_and_once() -> None:
    events = Events()
    s = Scope(events, name="s")
    order: List[str] = []
    s.on("e", lambda: order.append("normal"))
    s.on("e", lambda: order.append("first"), prepend=True)
    s.on("e", lambda: order.append("once"), once=True)
    events.emit("e")
    events.emit("e")
    assert order == ["first", "normal", "once", "first", "normal"]


async def test_once_listener_drops_its_scope_entry_when_it_fires() -> None:
    events = Events()
    s = Scope(events, name="s")
    for _ in range(50):
        s.on("e", lambda: None, once=True)
    assert s.effect_count == 50
    events.emit("e")
    assert s.effect_count == 0
    assert events.listener_count("e") == 0
    await s.dispose()


async def test_once_listener_removed_before_firing_never_fires() -> None:
    events = Events()
    s = Scope(events, name="s")
    got: List[str] = []
    remove = s.on("e", lambda: got.append("x"), once=True)
    remove()
    events.emit("e")
    assert got == []


async def test_on_without_events_is_a_clear_error() -> None:
    s = Scope(name="s")
    with pytest.raises(RuntimeError, match="no Events"):
        s.on("e", lambda: None)
    with pytest.raises(TypeError):
        Scope(Events(), name="x").on("e", 42)  # type: ignore[arg-type]


async def test_child_shares_events_unless_given_its_own() -> None:
    events = Events()
    other = Events()
    root = Scope(events, name="root")
    shared = root.child("shared")
    own = root.child("own", events=other)
    assert shared.events is events and own.events is other
    shared.on("e", lambda: None)
    own.on("e", lambda: None)
    assert events.listener_count("e") == 1 and other.listener_count("e") == 1
    await root.dispose()
    assert events.listener_count("e") == 0 and other.listener_count("e") == 0


async def test_scope_state_events_are_emitted_on_each_change() -> None:
    events = Events()
    seen: List[Any] = []
    events.on("scope/state", lambda name, state: seen.append((name, state)))
    s = Scope(events, name="plug")
    await s.start()
    await s.dispose()
    assert seen == [("plug", ScopeState.ACTIVE), ("plug", ScopeState.DISPOSED)]


async def test_scope_state_events_for_failure_and_restart() -> None:
    events = Events()
    seen: List[ScopeState] = []
    events.on("scope/state", lambda name, state: seen.append(state))
    s = Scope(events, name="plug")

    def bad(_: Scope) -> None:
        raise RuntimeError("no")

    await s.start(bad)
    await s.restart(None)
    assert seen == [ScopeState.FAILED, ScopeState.PENDING, ScopeState.ACTIVE]


async def test_failing_observer_does_not_affect_the_scope() -> None:
    events = Events()

    def broken(name: str, state: ScopeState) -> None:
        raise RuntimeError("observer bug")

    async def broken_async(name: str, state: ScopeState) -> None:
        raise RuntimeError("async observer bug")

    events.on("scope/state", broken)
    events.on("scope/state", broken_async)
    log: List[str] = []
    s = Scope(events, name="s")
    await s.start(lambda sc: sc.effect(rec(log, "a")))
    assert s.state is ScopeState.ACTIVE
    await s.dispose()
    await turns()
    assert s.state is ScopeState.DISPOSED and log == ["a"]
    assert s.errors == []
    assert len(events.errors) >= 2  # the observer failures are recorded by Events


# ----------------------------------------------------------------------
# Dispose rules
# ----------------------------------------------------------------------


async def test_dispose_is_idempotent() -> None:
    log: List[str] = []
    s = Scope(name="s")
    s.effect(rec(log, "a"))
    await s.dispose()
    await s.dispose()
    assert log == ["a"] and s.state is ScopeState.DISPOSED


async def test_concurrent_dispose_calls_wait_for_the_first() -> None:
    log: List[str] = []
    gate = asyncio.Event()
    s = Scope(name="s")

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            await gate.wait()
            log.append("slow")

        return undo

    s.effect(effect)
    d1 = asyncio.ensure_future(s.dispose())
    d2 = asyncio.ensure_future(s.dispose())
    await turns()
    assert not d1.done() and not d2.done()
    gate.set()
    await asyncio.gather(d1, d2)
    assert log == ["slow"] and s.state is ScopeState.DISPOSED


async def test_registering_after_dispose_raises_a_clear_error() -> None:
    events = Events()
    s = Scope(events, name="gone")
    await s.dispose()
    with pytest.raises(ScopeDisposedError, match="gone"):
        s.effect(lambda: None)
    with pytest.raises(ScopeDisposedError):
        s.on("e", lambda: None)
    with pytest.raises(ScopeDisposedError):
        s.child("c")
    with pytest.raises(ScopeDisposedError):
        Scope(events, name="c", parent=s)
    with pytest.raises(ScopeDisposedError):
        await s.start()
    assert isinstance(ScopeDisposedError("x"), RuntimeError)


async def test_dispose_without_effects_is_immediate() -> None:
    s = Scope(name="s")
    await s.dispose()
    assert s.state is ScopeState.DISPOSED and s.disposed


async def test_child_disposed_on_its_own_detaches_from_parent() -> None:
    log: List[str] = []
    root = Scope(name="root")
    c = root.child("c")
    c.effect(rec(log, "c"))
    assert root.children == (c,)
    await c.dispose()
    assert root.children == () and log == ["c"]
    await root.dispose()
    assert log == ["c"]


async def test_child_errors_stay_visible_on_the_parent() -> None:
    root = Scope(name="root")
    c = root.child("kid")

    def bad() -> Callable[[], None]:
        def undo() -> None:
            raise RuntimeError("kid failed")

        return undo

    c.effect(bad)
    await root.dispose()
    assert len(c.errors) == 1
    (err,) = root.errors
    assert isinstance(err.exception, ChildScopeErrors)
    assert err.exception.scope == "kid"
    assert err.exception.count == 1
    assert len(err.exception.recent) == 1
    assert "kid failed" in err.exception.recent[0]


async def test_deeply_nested_scopes_dispose_without_recursion_trouble() -> None:
    log: List[int] = []
    root = Scope(name="root")
    cur = root
    for i in range(300):
        cur = cur.child(f"n{i}")
        cur.effect(rec(log, i))
    await root.dispose()
    assert log == list(range(299, -1, -1))
    d = Scope(name="x")
    cur = d
    for i in range(100):
        cur = cur.child(f"n{i}")
    report = d.describe()
    assert json.loads(json.dumps(report))["name"] == "x"  # truncated, not recursive forever
    await d.dispose()


# ----------------------------------------------------------------------
# Re-entrancy
# ----------------------------------------------------------------------


async def test_disposer_calling_dispose_again_does_not_deadlock() -> None:
    log: List[str] = []
    s = Scope(name="s")

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            await s.dispose()  # waits for itself without the guard
            log.append("undo")

        return undo

    s.effect(effect)
    await asyncio.wait_for(s.dispose(), 5)  # the timeout is only a deadlock guard
    assert log == ["undo"] and s.state is ScopeState.DISPOSED


async def test_task_started_by_a_disposer_may_call_dispose_too() -> None:
    log: List[str] = []
    s = Scope(name="s")

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            await asyncio.ensure_future(s.dispose())
            log.append("undo")

        return undo

    s.effect(effect)
    await asyncio.wait_for(s.dispose(), 5)
    assert log == ["undo"]


async def test_child_disposer_calling_parent_dispose_does_not_deadlock() -> None:
    log: List[str] = []
    root = Scope(name="root")
    kid = root.child("kid")

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            await root.dispose()
            log.append("kid-undo")

        return undo

    kid.effect(effect)
    root.effect(rec(log, "root"))
    await asyncio.wait_for(root.dispose(), 5)
    assert log == ["kid-undo", "root"]


async def test_disposer_registering_new_effects_is_refused_and_recorded() -> None:
    log: List[str] = []
    s = Scope(name="s")

    def effect() -> Callable[[], None]:
        def undo() -> None:
            s.effect(rec(log, "sneaky"))

        return undo

    s.effect(effect, label="registers-late")
    s.effect(rec(log, "other"))
    await s.dispose()
    assert log == ["other"]
    (err,) = s.errors
    assert err.label == "registers-late"
    assert isinstance(err.exception, ScopeDisposedError)


async def test_disposer_can_remove_a_sibling_effect() -> None:
    log: List[str] = []
    s = Scope(name="s")
    remove_a = s.effect(rec(log, "a"))

    def effect() -> Callable[[], None]:
        return lambda: (remove_a(), log.append("b"))[1]

    s.effect(effect)
    await s.dispose()
    assert log == ["a", "b"]  # a ran once, not twice


# ----------------------------------------------------------------------
# start, setup, failure, restart
# ----------------------------------------------------------------------


async def test_start_success_makes_scope_active_and_passes_the_scope() -> None:
    s = Scope(name="s")
    assert s.state is ScopeState.PENDING
    got: List[Scope] = []
    result = await s.start(got.append)
    assert result is s and got == [s] and s.state is ScopeState.ACTIVE
    t = Scope(name="t")
    await t.start()  # no setup at all
    assert t.state is ScopeState.ACTIVE


async def test_start_accepts_async_setup_and_waits_for_async_effects() -> None:
    log: List[str] = []
    s = Scope(name="s")

    async def setup(sc: Scope) -> None:
        await asyncio.sleep(0)
        sc.effect(rec(log, "a"))

        async def late() -> Callable[[], None]:
            await turns(3)
            return lambda: log.append("late")

        sc.effect(late)  # not awaited by setup itself

    await s.start(setup)
    assert s.state is ScopeState.ACTIVE
    assert s.errors == []
    await s.dispose()
    assert log == ["late", "a"]


async def test_failing_setup_unwinds_marks_failed_and_does_not_raise() -> None:
    log: List[str] = []
    s = Scope(name="plugin")

    def setup(sc: Scope) -> None:
        sc.effect(rec(log, "a"))
        sc.effect(rec(log, "b"))
        raise ValueError("bad config")

    result = await s.start(setup)
    assert result is s
    assert s.state is ScopeState.FAILED
    assert log == ["b", "a"]
    assert s.effect_count == 0
    (err,) = s.errors
    assert err.label == "setup" and isinstance(err.exception, ValueError)


async def test_failing_async_setup_with_children_unwinds_them_too() -> None:
    log: List[str] = []
    s = Scope(name="plugin")

    async def setup(sc: Scope) -> None:
        sc.effect(rec(log, "own"))
        c = sc.child("c")
        c.effect(rec(log, "child"))
        await asyncio.sleep(0)
        raise RuntimeError("late failure")

    await s.start(setup)
    assert s.state is ScopeState.FAILED
    assert log == ["child", "own"]
    assert s.children == ()


async def test_failing_plugin_does_not_crash_its_siblings() -> None:
    root = Scope(name="root")
    good = root.child("good")
    bad = root.child("bad")

    def boom(_: Scope) -> None:
        raise RuntimeError("boom")

    results = await asyncio.gather(good.start(), bad.start(boom))
    assert [r.state for r in results] == [ScopeState.ACTIVE, ScopeState.FAILED]
    assert root.state is ScopeState.PENDING
    await root.dispose()


async def test_failed_scope_never_retries_and_start_is_refused() -> None:
    calls: List[int] = []

    def bad(_: Scope) -> None:
        calls.append(1)
        raise RuntimeError("x")

    s = Scope(name="s")
    await s.start(bad)
    await turns()
    assert calls == [1] and s.state is ScopeState.FAILED
    with pytest.raises(RuntimeError, match="restart"):
        await s.start(bad)
    with pytest.raises(RuntimeError):
        await (await _active()).start()


async def _active() -> Scope:
    s = Scope(name="a")
    await s.start()
    return s


async def test_restart_recovers_from_failed() -> None:
    attempts: List[int] = []
    log: List[str] = []

    def flaky(sc: Scope) -> None:
        attempts.append(1)
        sc.effect(rec(log, f"try{len(attempts)}"))
        if len(attempts) == 1:
            raise RuntimeError("first time")

    s = Scope(name="s")
    await s.start(flaky)
    assert s.state is ScopeState.FAILED and log == ["try1"]
    await s.restart(flaky)
    assert s.state is ScopeState.ACTIVE
    assert len(s.errors) == 1  # the first failure is still on record
    await s.dispose()
    assert log == ["try1", "try2"]


async def test_restart_can_fail_again_and_restart_from_active_undoes_first() -> None:
    log: List[str] = []

    def always(sc: Scope) -> None:
        sc.effect(rec(log, "x"))
        raise RuntimeError("again")

    s = Scope(name="s")
    await s.start(always)
    await s.restart(always)
    assert s.state is ScopeState.FAILED and len(s.errors) == 2
    await s.restart(lambda sc: sc.effect(rec(log, "ok")))
    assert s.state is ScopeState.ACTIVE
    await s.restart(lambda sc: sc.effect(rec(log, "ok2")))  # from ACTIVE
    assert log == ["x", "x", "ok"]
    await s.dispose()
    assert log == ["x", "x", "ok", "ok2"]


async def test_restart_is_refused_from_pending_and_disposed_and_while_starting() -> None:
    s = Scope(name="s")
    with pytest.raises(RuntimeError):
        await s.restart(None)
    gate = asyncio.Event()

    async def slow(_: Scope) -> None:
        await gate.wait()

    t = asyncio.ensure_future(s.start(slow))
    await turns()
    with pytest.raises(RuntimeError):
        await s.start(None)
    gate.set()
    await t
    await s.dispose()
    with pytest.raises(ScopeDisposedError):
        await s.restart(None)


async def test_async_effect_failure_during_start_fails_the_scope() -> None:
    log: List[str] = []
    s = Scope(name="s")

    async def broken() -> Callable[[], None]:
        await asyncio.sleep(0)
        raise ConnectionError("db down")

    def setup(sc: Scope) -> None:
        sc.effect(rec(log, "ok"))
        sc.effect(broken, label="connect-db")

    await s.start(setup)
    assert s.state is ScopeState.FAILED and log == ["ok"]
    assert [e.label for e in s.errors] == ["connect-db"]  # recorded once, not twice


async def test_async_effect_failure_after_start_is_recorded() -> None:
    s = Scope(name="s")
    await s.start()

    async def broken() -> None:
        raise ConnectionError("later")

    s.effect(broken, label="late")
    await turns()
    assert s.state is ScopeState.ACTIVE
    assert [e.label for e in s.errors] == ["late"]
    await s.dispose()


async def test_invalid_value_from_async_effect_is_recorded() -> None:
    s = Scope(name="s")

    async def wrong() -> int:
        return 42

    s.effect(wrong, label="wrong")
    await s.dispose()
    (err,) = s.errors
    assert err.label == "wrong" and isinstance(err.exception, TypeError)


# ----------------------------------------------------------------------
# Dispose while things are pending
# ----------------------------------------------------------------------


async def test_dispose_during_async_setup_undoes_everything() -> None:
    log: List[str] = []
    s = Scope(name="s")
    gate = asyncio.Event()
    entered = asyncio.Event()

    async def setup(sc: Scope) -> None:
        sc.effect(rec(log, "a"))
        entered.set()
        await gate.wait()
        sc.effect(rec(log, "late"))  # refused: the scope is being disposed

    t = asyncio.ensure_future(s.start(setup))
    await entered.wait()
    await s.dispose()
    assert log == ["a"] and s.state is ScopeState.DISPOSED
    gate.set()
    assert await t is s
    assert log == ["a"] and s.state is ScopeState.DISPOSED
    assert any(isinstance(e.exception, ScopeDisposedError) for e in s.errors)


async def test_dispose_during_setup_that_then_finishes_normally() -> None:
    log: List[str] = []
    s = Scope(name="s")
    gate = asyncio.Event()
    entered = asyncio.Event()

    async def setup(sc: Scope) -> None:
        sc.effect(rec(log, "a"))
        entered.set()
        await gate.wait()

    t = asyncio.ensure_future(s.start(setup))
    await entered.wait()
    await s.dispose()
    gate.set()
    await t
    assert s.state is ScopeState.DISPOSED and log == ["a"] and s.errors == []


async def test_async_effect_resolving_after_dispose_began_is_undone_in_order() -> None:
    log: List[str] = []
    gate = asyncio.Event()
    s = Scope(name="s")

    async def slow() -> Callable[[], None]:
        try:
            await gate.wait()
        except asyncio.CancelledError:
            await gate.wait()  # ignores the request to stop and finishes anyway
        return lambda: log.append("slow")

    s.effect(rec(log, "first"))
    s.effect(slow)
    s.effect(rec(log, "last"))
    d = asyncio.ensure_future(s.dispose())
    await turns()
    assert log == ["last"] and not d.done()
    gate.set()
    await d
    assert log == ["last", "slow", "first"]


async def test_cancelled_dispose_waiter_does_not_stop_the_unwinding() -> None:
    log: List[str] = []
    gate = asyncio.Event()
    s = Scope(name="s")
    s.effect(rec(log, "a"))

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            await gate.wait()
            log.append("slow")

        return undo

    s.effect(effect)
    d = asyncio.ensure_future(s.dispose())
    await turns()
    d.cancel()
    with pytest.raises(asyncio.CancelledError):
        await d
    assert not s.disposed and log == []
    gate.set()
    await s.dispose()  # joins the unwinding that is still running
    assert log == ["slow", "a"] and s.disposed and s.errors == []


async def test_wait_for_timeout_on_dispose_leaves_nothing_behind() -> None:
    log: List[str] = []
    gate = asyncio.Event()
    s = Scope(name="s")
    s.effect(rec(log, "a"))

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            await gate.wait()
            log.append("slow")

        return undo

    s.effect(effect)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(s.dispose(), 0.01)
    gate.set()
    await s.dispose()
    assert log == ["slow", "a"]


async def test_cancelling_the_unwinding_task_still_runs_remaining_disposers() -> None:
    log: List[str] = []
    gate = asyncio.Event()
    s = Scope(name="s")
    s.effect(rec(log, "a"))

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            await gate.wait()

        return undo

    s.effect(effect)
    s.effect(rec(log, "c"))
    d = asyncio.ensure_future(s.dispose())
    await turns()
    assert s._task is not None
    s._task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await d
    assert log == ["c", "a"]
    assert s.disposed
    assert s.errors == []  # cancellation is not a failure


# ----------------------------------------------------------------------
# Cancellation
# ----------------------------------------------------------------------


async def test_cancelled_setup_unwinds_and_propagates() -> None:
    log: List[str] = []
    gate = asyncio.Event()
    entered = asyncio.Event()
    s = Scope(name="s")

    async def setup(sc: Scope) -> None:
        sc.effect(rec(log, "a"))
        sc.effect(rec(log, "b"))
        entered.set()
        await gate.wait()

    t = asyncio.ensure_future(s.start(setup))
    await entered.wait()
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert log == ["b", "a"]
    assert s.errors == []  # not a failure
    assert s.state is ScopeState.PENDING  # may be started again
    await s.start(lambda sc: sc.effect(rec(log, "again")))
    assert s.state is ScopeState.ACTIVE


async def test_setup_raising_cancelled_error_is_not_swallowed() -> None:
    log: List[str] = []
    s = Scope(name="s")

    def setup(sc: Scope) -> None:
        sc.effect(rec(log, "a"))
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await s.start(setup)
    assert log == ["a"] and s.errors == [] and s.state is ScopeState.PENDING


async def test_cancelled_error_raised_by_a_sync_disposer_is_dropped() -> None:
    # A disposer's own CancelledError is not a real cancellation of the unwinding.
    log: List[str] = []
    s = Scope(name="s")
    s.effect(rec(log, "a"))

    def effect() -> Callable[[], None]:
        def undo() -> None:
            raise asyncio.CancelledError()

        return undo

    s.effect(effect)
    await s.dispose()
    assert log == ["a"] and s.errors == [] and s.disposed


# ----------------------------------------------------------------------
# async with, describe
# ----------------------------------------------------------------------


async def test_async_with_starts_and_disposes() -> None:
    log: List[str] = []
    s = Scope(name="s")
    async with s as inner:
        assert inner is s and s.state is ScopeState.ACTIVE
        s.effect(rec(log, "a"))
    assert s.state is ScopeState.DISPOSED and log == ["a"]


async def test_async_with_disposes_on_error_and_does_not_hide_it() -> None:
    log: List[str] = []
    s = Scope(name="s")
    with pytest.raises(KeyError):
        async with s:
            s.effect(rec(log, "a"))
            raise KeyError("body")
    assert s.disposed and log == ["a"]


async def test_describe_is_json_safe_and_complete() -> None:
    events = Events()
    root = Scope(events, name="root")
    kid = root.child("kid")
    root.effect(lambda: None)
    root.effect(lambda: None)
    kid.effect(lambda: None)

    def bad() -> Callable[[], None]:
        def undo() -> None:
            raise RuntimeError("oops ☃")

        return undo

    for _ in range(8):
        kid.effect(bad)
    report = root.describe()
    assert report["name"] == "root" and report["state"] == "pending"
    assert report["effects"] == 2 and report["error_count"] == 0
    (kid_report,) = report["children"]
    assert kid_report["name"] == "kid" and kid_report["effects"] == 9
    json.dumps(report)
    await kid.dispose()
    report = root.describe()
    assert report["children"] == []
    assert report["error_count"] == 1  # the child's errors are summarised on the parent
    after = kid.describe()
    assert after["state"] == "disposed" and after["error_count"] == 8
    assert len(after["recent_errors"]) == 5  # only the last few
    assert all(isinstance(x, str) for x in after["recent_errors"])
    assert "RuntimeError" in after["recent_errors"][0]
    assert json.loads(json.dumps(after)) == after
    await root.dispose()


# ----------------------------------------------------------------------
# Misuse and Python traps
# ----------------------------------------------------------------------


async def test_effect_input_validation() -> None:
    s = Scope(name="s")
    with pytest.raises(TypeError):
        s.effect(42)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        s.effect(lambda: 42)
    with pytest.raises(TypeError):
        s.effect(lambda: [lambda: None, "nope"])
    with pytest.raises(TypeError):
        s.effect(lambda: {"a": 1})
    assert s.effect_count == 0  # nothing half registered


async def test_effect_that_raises_registers_nothing() -> None:
    s = Scope(name="s")

    def bad() -> None:
        raise ValueError("fn failed")

    with pytest.raises(ValueError):
        s.effect(bad)
    assert s.effect_count == 0 and s.errors == []


def test_awaitable_effect_without_running_loop_is_refused_cleanly() -> None:
    s = Scope(name="s")

    async def work() -> None:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeError, match="running event loop"):
            s.effect(work)
        gc.collect()  # a never-awaited coroutine would warn (and fail) here
    assert s.effect_count == 0


async def test_disposer_returning_truthy_non_awaitable_is_ignored() -> None:
    s = Scope(name="s")
    ran: List[str] = []

    def effect() -> Callable[[], Any]:
        def undo() -> Any:
            ran.append("x")
            return "truthy"

        return undo

    s.effect(effect)
    s.effect(lambda: (lambda: 123))
    s.effect(lambda: (lambda: lambda: ran.append("never")))  # returns a callable: not called
    await s.dispose()
    assert ran == ["x"] and s.errors == []


async def test_disposer_returning_a_task_or_future_is_awaited() -> None:
    log: List[str] = []
    s = Scope(name="s")

    def effect() -> Callable[[], Any]:
        async def work() -> None:
            await asyncio.sleep(0)
            log.append("task done")

        return lambda: asyncio.ensure_future(work())

    def failing() -> Callable[[], Any]:
        async def work() -> None:
            raise RuntimeError("task failed")

        return lambda: asyncio.ensure_future(work())

    s.effect(effect)
    s.effect(failing, label="failing-task")
    await s.dispose()
    assert log == ["task done"]
    assert [e.label for e in s.errors] == ["failing-task"]


async def test_async_generator_and_generator_disposers_are_run_to_the_end() -> None:
    log: List[str] = []
    s = Scope(name="s")

    def agen_effect() -> Callable[[], Any]:
        async def undo():  # type: ignore[no-untyped-def]
            log.append("agen start")
            yield 1
            await asyncio.sleep(0)
            log.append("agen end")

        return undo

    def gen_effect() -> Callable[[], Any]:
        def undo():  # type: ignore[no-untyped-def]
            log.append("gen start")
            yield 1
            log.append("gen end")

        return undo

    s.effect(agen_effect)
    s.effect(gen_effect)
    await s.dispose()
    assert log == ["gen start", "gen end", "agen start", "agen end"]


async def test_no_coroutine_is_left_unawaited() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = Scope(Events(), name="s")

        def effect() -> Callable[[], Any]:
            async def undo() -> None:
                await asyncio.sleep(0)

            return undo

        for _ in range(5):
            s.effect(effect)
        r = s.effect(effect)
        await r()
        await s.start(lambda sc: sc.effect(effect))
        await s.restart(lambda sc: sc.effect(effect))
        await s.dispose()
        c = Scope(name="c")
        c.effect(effect)
        await c.dispose()
        gc.collect()
    bad = [w for w in caught if issubclass(w.category, (RuntimeWarning, ResourceWarning))]
    assert bad == []


async def test_listeners_are_removed_by_token_not_equality() -> None:
    class Eq:
        def __call__(self) -> None:
            pass

        def __eq__(self, other: object) -> bool:
            return True

        __hash__ = None  # type: ignore[assignment]

    events = Events()
    s = Scope(events, name="s")
    a, b = Eq(), Eq()
    ra = s.on("e", a)
    s.on("e", b)
    ra()
    assert events.listener_count("e") == 1
    await s.dispose()
    assert events.listener_count("e") == 0


async def test_scope_names_are_plain_strings() -> None:
    s = Scope(name="__proto__")
    assert s.describe()["name"] == "__proto__"
    t = Scope(name=123)  # type: ignore[arg-type]
    assert t.name == "123"


# ----------------------------------------------------------------------
# Randomized differential test against a simple reference model
# ----------------------------------------------------------------------


class _Model:
    """The simplest possible statement of the rules, with no asyncio in it."""

    def __init__(self, parent: Optional["_Model"], log: List[int]) -> None:
        self.parent = parent
        self.log = log
        self.effects: List[Any] = []  # [id, fails]
        self.children: List["_Model"] = []
        self.disposed = False
        self.errors = 0
        if parent is not None:
            parent.children.append(self)

    def run(self, eff: Any) -> None:
        self.log.append(eff[0])
        if eff[1]:
            self.errors += 1

    def dispose(self) -> None:
        if self.disposed:
            return
        for kid in reversed(list(self.children)):
            kid.dispose()
        for eff in reversed(self.effects):
            self.run(eff)
        self.effects = []
        self.disposed = True
        if self.parent is not None:
            self.parent.children.remove(self)
            if self.errors:
                self.parent.errors += 1

    def remove(self, eff: Any) -> None:
        if eff in self.effects:
            self.effects.remove(eff)
            self.run(eff)


async def _differential(seed: int) -> None:
    rng = random.Random(seed)
    real_log: List[int] = []
    model_log: List[int] = []
    root = Scope(name="root")
    scopes: List[Scope] = [root]
    models: List[_Model] = [_Model(None, model_log)]
    handles: List[Any] = []  # (scope index, model effect, remover)
    next_id = 0

    def make(ident: int, kind: str) -> Callable[[], Callable[[], Any]]:
        fails = kind.endswith("fail")
        is_async = kind.startswith("async")

        def effect() -> Callable[[], Any]:
            if is_async:

                async def undo_async() -> None:
                    await asyncio.sleep(0)
                    real_log.append(ident)
                    if fails:
                        raise RuntimeError(f"fail {ident}")

                return undo_async

            def undo() -> None:
                real_log.append(ident)
                if fails:
                    raise RuntimeError(f"fail {ident}")

            return undo

        return effect

    for _ in range(rng.randint(10, 60)):
        op = rng.choice(["eff", "eff", "eff", "rm", "rm", "child", "disp", "disp2"])
        i = rng.randrange(len(scopes))
        scope, model = scopes[i], models[i]
        if op == "eff":
            kind = rng.choice(["sync", "async", "sync-fail", "async-fail"])
            next_id += 1
            if model.disposed:
                with pytest.raises(ScopeDisposedError):
                    scope.effect(make(next_id, kind))
            else:
                remover = scope.effect(make(next_id, kind))
                eff = [next_id, kind.endswith("fail")]
                model.effects.append(eff)
                handles.append((model, eff, remover))
        elif op == "rm" and handles:
            model, eff, remover = rng.choice(handles)
            await remover()
            model.remove(eff)
        elif op == "child":
            if model.disposed:
                with pytest.raises(ScopeDisposedError):
                    scope.child("c")
            else:
                scopes.append(scope.child(f"c{len(scopes)}"))
                models.append(_Model(model, model_log))
        elif op == "disp":
            await scope.dispose()
            model.dispose()
        elif op == "disp2":
            await asyncio.gather(scope.dispose(), scope.dispose())
            model.dispose()
        assert real_log == model_log, f"seed {seed} after {op}"

    await root.dispose()
    models[0].dispose()
    assert real_log == model_log, f"seed {seed} final"
    for scope, model in zip(scopes, models):
        want = ScopeState.DISPOSED
        assert scope.state is want and model.disposed, f"seed {seed} {scope.name}"
        assert len(scope.errors) == model.errors, f"seed {seed} {scope.name} errors"
        assert scope.effect_count == 0 and scope.children == ()


async def test_randomized_differential_against_reference_model() -> None:
    for seed in range(400):
        await _differential(seed)


async def test_randomized_state_before_final_dispose_matches_model() -> None:
    # Check live (not yet disposed) scopes too: effect counts and children.
    for seed in range(100):
        rng = random.Random(10_000 + seed)
        root = Scope(name="root")
        live = [root]
        counts = {id(root): 0}
        for _ in range(40):
            s = rng.choice(live)
            if rng.random() < 0.3:
                c = s.child("c")
                live.append(c)
                counts[id(c)] = 0
            else:
                s.effect(lambda: (lambda: None))
                counts[id(s)] += 1
        for s in live:
            assert s.effect_count == counts[id(s)]
        await root.dispose()
        assert all(s.disposed for s in live)


# ----------------------------------------------------------------------
# Review round: cancellation inside disposers
# ----------------------------------------------------------------------


async def test_task_stop_disposer_does_not_raise_cancelled_error_at_the_caller() -> None:
    async def work() -> None:
        await asyncio.Event().wait()

    t = asyncio.ensure_future(work())
    await turns()

    def effect() -> Callable[[], Any]:
        async def stop() -> None:
            t.cancel()
            await t  # raises CancelledError inside the disposer

        return stop

    s = Scope(name="s")
    s.effect(effect)
    await s.dispose()  # must not raise: the caller was not cancelled
    assert s.disposed and s.errors == [] and t.cancelled()


async def test_disposer_cancelling_its_own_task_is_dropped_not_recorded() -> None:
    log: List[str] = []
    s = Scope(name="s")
    s.effect(rec(log, "a"))

    def effect() -> Callable[[], Any]:
        async def stop() -> None:
            current = asyncio.current_task()
            assert current is not None
            current.cancel()
            await asyncio.sleep(0)
            log.append("not reached")

        return stop

    s.effect(effect)
    s.effect(rec(log, "c"))
    await s.dispose()
    assert log == ["c", "a"] and s.errors == [] and s.disposed


async def test_task_stop_disposer_through_start_restart_and_async_with() -> None:
    tasks: List["asyncio.Future[None]"] = []

    def stopper(sc: Scope) -> None:
        async def work() -> None:
            await asyncio.Event().wait()

        t = asyncio.ensure_future(work())
        tasks.append(t)

        def effect() -> Callable[[], Any]:
            async def stop() -> None:
                t.cancel()
                await t

            return stop

        sc.effect(effect)

    s = Scope(name="s")
    await s.start(stopper)
    await s.restart(stopper)  # undoes the first one
    assert tasks[0].cancelled() and s.state is ScopeState.ACTIVE
    await s.dispose()
    assert tasks[1].cancelled() and s.errors == []

    async with Scope(name="w") as w:
        stopper(w)
    assert tasks[2].cancelled() and w.errors == []

    r = Scope(name="r")
    stopper(r)
    await r.dispose()
    assert tasks[3].cancelled() and r.errors == []


async def test_scoped_usage_in_sibling_tasks_sees_no_spurious_cancellation() -> None:
    async def part(name: str) -> str:
        async def work() -> None:
            await asyncio.Event().wait()

        t = asyncio.ensure_future(work())
        async with Scope(name=name) as sc:

            def effect() -> Callable[[], Any]:
                async def stop() -> None:
                    t.cancel()
                    await t

                return stop

            sc.effect(effect)
            await turns()
        return name

    assert await asyncio.gather(part("a"), part("b"), part("c")) == ["a", "b", "c"]


async def test_genuine_cancellation_still_propagates_runs_the_rest_and_leaks_nothing() -> None:
    log: List[str] = []
    s = Scope(name="s")
    s.effect(rec(log, "a"))

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                log.append("b-cancelled")

        return undo

    s.effect(effect)
    s.effect(rec(log, "c"))
    before = asyncio.all_tasks()
    d = asyncio.ensure_future(s.dispose())
    await turns()
    assert s._task is not None
    s._task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await d
    assert log == ["c", "b-cancelled", "a"]
    assert s.disposed and s.errors == []
    await turns()
    assert asyncio.all_tasks() <= before | {d}  # the disposer's task is gone


# ----------------------------------------------------------------------
# Review round: start() and timeouts
# ----------------------------------------------------------------------


async def test_wait_for_start_times_out_when_an_async_effect_never_resolves() -> None:
    s = Scope(name="s")
    log: List[str] = []

    def setup(sc: Scope) -> None:
        sc.effect(rec(log, "first"))

        async def hang() -> None:
            await asyncio.Event().wait()

        sc.effect(hang, label="hang")

    loop = asyncio.get_running_loop()
    began = loop.time()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(s.start(setup), 0.2)
    assert loop.time() - began < 1.5
    assert log == ["first"] and s.state is ScopeState.PENDING and s.errors == []
    assert s.effect_count == 0 and not s._live_pending()


async def test_slow_undo_after_a_start_timeout_is_waited_for_only_as_long_as_itself() -> None:
    s = Scope(name="s")
    log: List[str] = []

    def setup(sc: Scope) -> None:
        def effect() -> Callable[[], Any]:
            async def undo() -> None:
                await turns(30)
                log.append("slow undo done")

            return undo

        sc.effect(effect)

        async def hang() -> None:
            await asyncio.Event().wait()

        sc.effect(hang)

    loop = asyncio.get_running_loop()
    began = loop.time()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(s.start(setup), 0.2)
    assert loop.time() - began < 1.5
    assert log == ["slow undo done"]


async def test_effect_resolving_late_after_cancellation_is_undone_exactly_once() -> None:
    s = Scope(name="s")
    log: List[str] = []

    def setup(sc: Scope) -> None:
        async def stubborn() -> Callable[[], None]:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass  # swallows the cancellation and still produces something
            await asyncio.sleep(0)
            return lambda: log.append("undo")

        sc.effect(stubborn)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(s.start(setup), 0.2)
    assert log == ["undo"]
    await s.dispose()
    assert log == ["undo"]  # not run a second time


async def test_failed_setup_with_a_hanging_effect_does_not_hang() -> None:
    s = Scope(name="s")

    async def setup(sc: Scope) -> None:
        async def hang() -> None:
            await asyncio.Event().wait()

        sc.effect(hang)
        raise RuntimeError("setup failed")

    unraisable: List[Any] = []
    old_hook = sys.unraisablehook
    sys.unraisablehook = unraisable.append
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            await asyncio.wait_for(s.start(setup), 5)  # 5 s is only a hang guard
            gc.collect()
    finally:
        sys.unraisablehook = old_hook
    assert s.state is ScopeState.FAILED and len(s.errors) == 1
    assert unraisable == []  # the never-started effect coroutine was closed, not leaked


# ----------------------------------------------------------------------
# Review round: removers whose result is not awaited
# ----------------------------------------------------------------------


async def test_unawaited_remover_still_runs_async_and_sync_disposers() -> None:
    log: List[str] = []
    s = Scope(name="s")

    def effect() -> Any:
        async def async_a() -> None:
            await asyncio.sleep(0)
            log.append("async_a")

        return [async_a, lambda: log.append("sync_b")]

    r = s.effect(effect)
    r()  # NOT awaited
    assert log == ["sync_b"]  # the list runs newest first: sync_b, then async_a
    await s.dispose()  # waits for the scheduled work
    assert log == ["sync_b", "async_a"]


async def test_unawaited_remover_of_a_resolving_effect_does_not_leak() -> None:
    log: List[str] = []
    gate = asyncio.Event()
    s = Scope(name="s")

    async def slow() -> Callable[[], None]:
        try:
            await gate.wait()
        except asyncio.CancelledError:
            await gate.wait()  # ignores the request to stop and finishes anyway
        return lambda: log.append("undone")

    remove = s.effect(slow)
    await turns()  # the effect is running (a cancel before its first step would end it)
    remove()  # not awaited, effect still resolving
    assert s.effect_count == 0
    d = asyncio.ensure_future(s.dispose())
    await turns()
    assert not d.done()  # dispose waits for the remover's work
    unwind = s._task
    await turns(20)
    assert s._task is unwind  # it waits quietly, it does not start unwind pass after pass
    gate.set()
    await d
    assert log == ["undone"]


async def test_unawaited_remover_work_is_done_after_the_loop_runs() -> None:
    log: List[str] = []
    s = Scope(name="s")

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            await asyncio.sleep(0)
            log.append("done")

        return undo

    s.effect(effect)()
    await turns()
    assert log == ["done"] and s.errors == []


async def test_remover_errors_are_recorded_even_when_not_awaited() -> None:
    s = Scope(name="s")

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            raise ValueError("bad undo")

        return undo

    s.effect(effect, label="bad")()
    await turns()
    assert [e.label for e in s.errors] == ["bad"]


async def test_cancelling_the_waiter_of_a_remover_does_not_stop_the_undo() -> None:
    log: List[str] = []
    gate = asyncio.Event()
    s = Scope(name="s")

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            await gate.wait()
            log.append("done")

        return undo

    remove = s.effect(effect)
    waiter = asyncio.ensure_future(_run(remove()))
    await turns()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    gate.set()
    await s.dispose()
    assert log == ["done"]


async def _run(aw: Any) -> None:
    await aw


def test_remover_without_a_running_loop_returns_work_to_run_by_hand() -> None:
    log: List[str] = []
    s = Scope(name="s")

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            log.append("async")

        return undo

    remove = s.effect(effect)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        aw = remove()
        assert log == []
        asyncio.run(_run(aw))
        gc.collect()
    assert log == ["async"]


async def test_disposer_run_by_a_remover_may_call_dispose_without_deadlock() -> None:
    log: List[str] = []
    s = Scope(name="s")

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            await s.dispose()  # returns at once, like during a disposal
            log.append("undo")

        return undo

    s.effect(effect)()
    await asyncio.wait_for(s.dispose(), 5)
    assert log == ["undo"] and s.disposed


# ----------------------------------------------------------------------
# Review round: disposals that wait for each other
# ----------------------------------------------------------------------


def _dispose_other(log: List[str], name: str, other: Scope) -> Callable[[], Callable[[], Any]]:
    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            await other.dispose()
            log.append(name)

        return undo

    return effect


async def test_two_scopes_disposing_each_other_do_not_deadlock() -> None:
    log: List[str] = []
    a = Scope(name="a")
    b = Scope(name="b")
    a.effect(_dispose_other(log, "a", b))
    b.effect(_dispose_other(log, "b", a))
    await asyncio.wait_for(asyncio.gather(a.dispose(), b.dispose()), 5)
    assert a.disposed and b.disposed and sorted(log) == ["a", "b"]
    assert scope_module._WAITS == {}  # the bookkeeping is cleaned up


async def test_three_scopes_in_a_dispose_cycle_do_not_deadlock() -> None:
    log: List[str] = []
    a, b, c = Scope(name="a"), Scope(name="b"), Scope(name="c")
    a.effect(_dispose_other(log, "a", b))
    b.effect(_dispose_other(log, "b", c))
    c.effect(_dispose_other(log, "c", a))
    await asyncio.wait_for(asyncio.gather(a.dispose(), b.dispose(), c.dispose()), 5)
    assert all(s.disposed for s in (a, b, c)) and sorted(log) == ["a", "b", "c"]
    assert scope_module._WAITS == {}


async def test_a_disposer_still_waits_for_another_scope_that_is_not_in_a_cycle() -> None:
    log: List[str] = []
    gate = asyncio.Event()
    other = Scope(name="other")

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            await gate.wait()
            log.append("other-undone")

        return undo

    other.effect(effect)
    a = Scope(name="a")
    a.effect(_dispose_other(log, "a", other))
    d = asyncio.ensure_future(a.dispose())
    await turns()
    assert not d.done()
    gate.set()
    await d
    assert log == ["other-undone", "a"]


# ----------------------------------------------------------------------
# Review round: health report, memory, odd exceptions
# ----------------------------------------------------------------------


async def test_describe_shows_when_a_disposal_is_in_progress() -> None:
    s = Scope(name="s")
    seen: List[Any] = []
    before = s.describe()
    assert before["closing"] is False and before["unwinding"] is False

    def effect() -> Callable[[], None]:
        return lambda: seen.append(s.describe())

    s.effect(effect)
    await s.dispose()
    (during,) = seen
    assert during["closing"] is True and during["unwinding"] is True
    assert during["state"] == "pending"  # not DISPOSED until it is finished
    after = s.describe()
    assert after["state"] == "disposed" and after["unwinding"] is False
    json.dumps(after)


async def test_describe_survives_an_exception_whose_str_raises() -> None:
    class Nasty(Exception):
        def __str__(self) -> str:
            raise RuntimeError("no str for you")

    s = Scope(name="s")

    def effect() -> Callable[[], None]:
        def undo() -> None:
            raise Nasty()

        return undo

    s.effect(effect, label="nasty")
    await s.dispose()
    report = s.describe()
    assert report["error_count"] == 1
    assert "Nasty" in report["recent_errors"][0] and "nasty" in report["recent_errors"][0]
    parent = Scope(name="p")
    kid = parent.child("k")
    kid.effect(effect, label="nasty")
    await parent.dispose()  # summarising the child's errors must not raise either
    assert "child scope" in parent.describe()["recent_errors"][0]


async def test_parent_keeps_only_a_bounded_summary_of_a_childs_errors() -> None:
    root = Scope(name="root")
    kid = root.child("kid")

    def bad() -> Callable[[], None]:
        def undo() -> None:
            raise RuntimeError("x" * 50)

        return undo

    for _ in range(200):
        kid.effect(bad)
    await root.dispose()
    (err,) = root.errors
    summary = err.exception
    assert isinstance(summary, ChildScopeErrors)
    assert summary.count == 200
    assert len(summary.recent) == 3 and all(isinstance(t, str) for t in summary.recent)
    for value in vars(summary).values():
        assert not (isinstance(value, (list, tuple)) and value and isinstance(value[0], ScopeError))


def test_pending_unwind_task_collected_with_its_loop_does_not_blow_up() -> None:
    # Closing a loop that still has an unwind task in flight finalizes its
    # coroutine in another context; resetting the context variable must not raise.
    caught: List[Any] = []
    old_hook = sys.unraisablehook
    sys.unraisablehook = caught.append
    try:
        loop = asyncio.new_event_loop()
        keep: List[Any] = [Scope(name="s")]

        async def go() -> None:
            def effect() -> Callable[[], Any]:
                async def undo() -> None:
                    await asyncio.Event().wait()

                return undo

            keep[0].effect(effect)
            keep.append(asyncio.ensure_future(keep[0].dispose()))
            await turns()

        loop.run_until_complete(go())
        loop.close()
        del loop
        keep[0]._task = None
        keep.clear()
        gc.collect()
    finally:
        sys.unraisablehook = old_hook
    assert caught == []


class _Fatal(BaseException):
    """Stands in for SystemExit/KeyboardInterrupt without stopping the test run."""


async def test_base_exception_from_a_disposer_finishes_the_rest_then_reraises() -> None:
    log: List[str] = []
    s = Scope(name="s")
    s.effect(rec(log, "a"))

    def effect() -> Callable[[], None]:
        def undo() -> None:
            raise _Fatal("stop")

        return undo

    s.effect(effect)
    s.effect(rec(log, "c"))
    with pytest.raises(_Fatal):
        await s.dispose()
    assert log == ["c", "a"]
    assert s.state is ScopeState.DISPOSED and s.errors == []
    await s.dispose()  # already finished, quiet


async def test_base_exception_from_an_async_disposer_finishes_the_rest_then_reraises() -> None:
    log: List[str] = []
    s = Scope(name="s")
    s.effect(rec(log, "a"))

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            await asyncio.sleep(0)
            raise _Fatal("stop")

        return undo

    s.effect(effect)
    s.effect(rec(log, "c"))
    with pytest.raises(_Fatal):
        await s.dispose()
    assert log == ["c", "a"] and s.disposed


async def test_base_exception_from_a_remover_is_raised_after_the_other_disposers() -> None:
    log: List[str] = []
    s = Scope(name="s")

    def effect() -> Any:
        def boom() -> None:
            raise _Fatal("x")

        return [lambda: log.append("first"), boom, lambda: log.append("last")]

    remove = s.effect(effect)
    with pytest.raises(_Fatal):
        remove()
    assert log == ["last", "first"]


# ----------------------------------------------------------------------
# Review round: tests aimed at mutants that used to survive
# ----------------------------------------------------------------------


async def test_failing_async_disposer_in_a_list_does_not_stop_the_others() -> None:
    log: List[str] = []

    async def ok_one() -> None:
        await asyncio.sleep(0)
        log.append("ok1")

    async def ok_two() -> None:
        await asyncio.sleep(0)
        log.append("ok2")

    async def bad() -> None:
        await asyncio.sleep(0)
        raise ValueError("bad async")

    s = Scope(name="s")
    s.effect(lambda: [ok_one, ok_two, bad], label="list")  # runs bad, ok_two, ok_one
    await s.dispose()
    assert log == ["ok2", "ok1"]
    assert [e.label for e in s.errors] == ["list"]
    # and through a remover
    s2 = Scope(name="s2")
    remove = s2.effect(lambda: [ok_one, bad, ok_two])
    log.clear()
    await remove()
    assert log == ["ok2", "ok1"] and len(s2.errors) == 1


async def test_restart_unwind_refuses_new_registrations() -> None:
    log: List[str] = []
    s = Scope(name="s")

    def setup(sc: Scope) -> None:
        def effect() -> Callable[[], None]:
            def undo() -> None:
                sc.effect(rec(log, "sneaky"))  # must be refused during the unwind

            return undo

        sc.effect(effect, label="registers-during-unwind")

    await s.start(setup)
    await s.restart(None)
    assert s.state is ScopeState.ACTIVE
    (err,) = s.errors
    assert err.label == "registers-during-unwind"
    assert isinstance(err.exception, ScopeDisposedError)
    assert "unwound" in str(err.exception)
    assert s.effect_count == 0
    await s.dispose()
    assert log == []


async def test_failed_setup_unwind_refuses_new_registrations() -> None:
    s = Scope(name="s")

    def setup(sc: Scope) -> None:
        def effect() -> Callable[[], None]:
            return lambda: sc.effect(lambda: None)

        sc.effect(effect, label="late")
        raise RuntimeError("fail")

    await s.start(setup)
    assert s.state is ScopeState.FAILED and s.effect_count == 0
    assert any(isinstance(e.exception, ScopeDisposedError) for e in s.errors)


async def test_cancelling_the_unwind_still_waits_for_a_child_scope_to_finish() -> None:
    log: List[str] = []
    gate = asyncio.Event()
    root = Scope(name="root")
    root.effect(rec(log, "root-effect"))
    kid = root.child("kid")

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            await gate.wait()
            log.append("kid-undo")

        return undo

    kid.effect(effect)
    d = asyncio.ensure_future(root.dispose())
    await turns()
    assert root._task is not None
    root._task.cancel()
    await turns()
    assert log == []  # the parent's own effect must not run before the child is done
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await d
    assert log == ["kid-undo", "root-effect"]
    assert root.disposed and kid.disposed


async def test_dispose_requested_during_restart_wins() -> None:
    log: List[str] = []
    gate = asyncio.Event()
    s = Scope(name="s")

    def first(sc: Scope) -> None:
        def effect() -> Callable[[], Any]:
            async def undo() -> None:
                await gate.wait()
                log.append("undo")

            return undo

        sc.effect(effect)

    await s.start(first)
    ran: List[str] = []
    r = asyncio.ensure_future(s.restart(lambda sc: ran.append("setup")))
    await turns()
    d = asyncio.ensure_future(s.dispose())
    await turns()
    gate.set()
    await asyncio.gather(r, d)
    assert s.state is ScopeState.DISPOSED
    assert ran == [] and log == ["undo"]


async def test_join_waits_for_a_second_unwind_pass() -> None:
    # A registration that sneaks in right after the first unwind pass ends (a
    # call_soon callback queued by the last disposer) must also be undone before
    # restart() runs the new setup.
    log: List[str] = []
    s = Scope(name="s")
    loop = asyncio.get_running_loop()

    def first(sc: Scope) -> None:
        def effect() -> Callable[[], None]:
            def undo() -> None:
                loop.call_soon(lambda: sc.effect(rec(log, "sneaked")))

            return undo

        sc.effect(effect)

    await s.start(first)
    await s.restart(lambda sc: log.append(f"setup-after-{log}"))
    assert log == ["sneaked", "setup-after-['sneaked']"]
    assert s.effect_count == 0


# ----------------------------------------------------------------------
# Review round: no loop errors and no warnings in the main scenarios
# ----------------------------------------------------------------------


async def test_main_scenarios_leave_no_loop_errors_unraisables_or_warnings() -> None:
    loop = asyncio.get_running_loop()
    loop_errors: List[Any] = []
    unraisable: List[Any] = []
    old_handler = loop.get_exception_handler()
    old_hook = sys.unraisablehook
    loop.set_exception_handler(lambda lp, ctx: loop_errors.append(ctx))
    sys.unraisablehook = unraisable.append
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for scenario in (
                test_effect_runs_immediately_and_disposer_runs_on_dispose,
                test_disposal_is_strictly_sequential_and_reverse,
                test_children_are_disposed_before_own_effects_newest_first,
                test_errors_do_not_stop_other_disposers,
                test_concurrent_dispose_calls_wait_for_the_first,
                test_failing_async_setup_with_children_unwinds_them_too,
                test_restart_can_fail_again_and_restart_from_active_undoes_first,
                test_cancelled_setup_unwinds_and_propagates,
                test_cancelled_dispose_waiter_does_not_stop_the_unwinding,
                test_cancelling_the_unwinding_task_still_runs_remaining_disposers,
                test_task_stop_disposer_does_not_raise_cancelled_error_at_the_caller,
                test_disposer_cancelling_its_own_task_is_dropped_not_recorded,
                test_genuine_cancellation_still_propagates_runs_the_rest_and_leaks_nothing,
                test_wait_for_start_times_out_when_an_async_effect_never_resolves,
                test_effect_resolving_late_after_cancellation_is_undone_exactly_once,
                test_unawaited_remover_still_runs_async_and_sync_disposers,
                test_unawaited_remover_of_a_resolving_effect_does_not_leak,
                test_two_scopes_disposing_each_other_do_not_deadlock,
                test_base_exception_from_an_async_disposer_finishes_the_rest_then_reraises,
                test_disposer_calling_dispose_again_does_not_deadlock,
                test_task_started_by_a_disposer_may_call_dispose_too,
                test_async_with_starts_and_disposes,
                test_no_coroutine_is_left_unawaited,
            ):
                await scenario()
            await turns()
            gc.collect()
            await turns()
    finally:
        loop.set_exception_handler(old_handler)
        sys.unraisablehook = old_hook
    assert loop_errors == []
    assert unraisable == []


# ----------------------------------------------------------------------
# Second review round: stale markers, wait edges, hanging effects
# ----------------------------------------------------------------------


async def test_leftover_task_from_an_earlier_unwind_cannot_swallow_a_later_dispose() -> None:
    gate = asyncio.Event()
    log: List[str] = []
    leftovers: List["asyncio.Future[None]"] = []
    s = Scope(name="s")

    def first(sc: Scope) -> None:
        def effect() -> Callable[[], None]:
            def undo() -> None:
                async def background() -> None:
                    await gate.wait()
                    await s.dispose()  # a real request, made long after the unwind

                leftovers.append(asyncio.ensure_future(background()))

            return undo

        sc.effect(effect)

    def second(sc: Scope) -> None:
        sc.effect(rec(log, "e1"))
        sc.effect(rec(log, "e2"))

    await s.start(first)
    await s.restart(second)  # the first unwind (and its marker) is over
    assert s.state is ScopeState.ACTIVE and s.effect_count == 2
    gate.set()
    await asyncio.wait_for(leftovers[0], 5)
    assert s.state is ScopeState.DISPOSED
    assert log == ["e2", "e1"] and s.effect_count == 0


async def test_recycled_id_of_a_disposed_scope_cannot_make_dispose_a_noop() -> None:
    gate = asyncio.Event()
    log: List[str] = []
    holder: List[Scope] = []
    leftovers: List["asyncio.Future[None]"] = []

    async def background() -> None:
        await gate.wait()
        await holder[0].dispose()

    old = Scope(name="old")

    def effect() -> Callable[[], None]:
        return lambda: leftovers.append(asyncio.ensure_future(background()))

    old.effect(effect)
    await old.dispose()  # the leftover task is born inside this unwind
    old_id = id(old)
    del old, effect
    gc.collect()
    keep: List[Scope] = []
    fresh = Scope(name="fresh")
    for _ in range(5000):  # try to get the dead scope's id back
        if id(fresh) == old_id:
            break
        keep.append(fresh)
        fresh = Scope(name="fresh")
    # (If this interpreter never recycles the id, the check below still holds.)
    fresh.effect(rec(log, "fresh-undo"))
    holder.append(fresh)
    gate.set()
    await asyncio.wait_for(leftovers[0], 5)
    assert fresh.state is ScopeState.DISPOSED and log == ["fresh-undo"]


async def test_markers_naming_a_scope_but_not_its_live_work_are_ignored() -> None:
    log: List[str] = []
    c = Scope(name="c")
    c.effect(rec(log, "undo"))
    token = _DISPOSING.set((id(c), c._serial, 1, 2, 3))
    try:
        await asyncio.wait_for(c.dispose(), 5)
    finally:
        _DISPOSING.reset(token)
    assert c.disposed and log == ["undo"]


async def test_background_task_of_a_live_undo_counts_as_part_of_it() -> None:
    # Documented trade-off: a task an undo started inherits "inside this unwind"
    # while that unwind is live, so a dispose() it makes can be seen as a wait of
    # the undo. Here B's wait for A would close a ring A -> B -> A, so it returns
    # early instead of hanging. Nothing hangs and every undo still runs once.
    a_gate = asyncio.Event()
    log: List[str] = []
    a = Scope(name="a")
    b = Scope(name="b")
    background: List["asyncio.Future[None]"] = []

    def a_effect() -> Callable[[], Any]:
        async def undo() -> None:
            async def bg() -> None:
                await b.dispose()

            background.append(asyncio.ensure_future(bg()))
            await a_gate.wait()
            log.append("a-undone")

        return undo

    def b_effect() -> Callable[[], Any]:
        async def undo() -> None:
            await turns(5)  # let A's background task make its call first
            await a.dispose()  # would close a ring: returns without waiting
            log.append("b-after-a")

        return undo

    a.effect(a_effect)
    b.effect(b_effect)
    both = asyncio.gather(a.dispose(), b.dispose())
    await turns(10)
    assert log == ["b-after-a"]  # returned early, A is still unwinding
    assert a.state is ScopeState.PENDING
    a_gate.set()
    await asyncio.wait_for(both, 5)
    await asyncio.wait_for(asyncio.gather(*background), 5)
    assert log == ["b-after-a", "a-undone"]
    assert a.disposed and b.disposed and scope_module._WAITS == {}


async def test_the_owner_of_a_task_stops_counting_when_its_unwind_is_over() -> None:
    gate = asyncio.Event()
    seen: List[Any] = []
    leftovers: List["asyncio.Future[None]"] = []
    a = Scope(name="a")

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            async def bg() -> None:
                seen.append(scope_module._current_owner())  # during the unwind
                await gate.wait()
                seen.append(scope_module._current_owner())  # after it

            leftovers.append(asyncio.ensure_future(bg()))
            await turns()

        return undo

    a.effect(effect)
    await a.dispose()
    gate.set()
    await asyncio.wait_for(leftovers[0], 5)
    assert seen == [a._serial, None]
    assert scope_module._current_owner() is None


_FORMS = ["direct", "gather", "create_task", "shield", "wait_for", "ensure_future"]


def _dispose_via(form: str, scope: Scope) -> Callable[[], Any]:
    async def call() -> None:
        if form == "direct":
            await scope.dispose()
        elif form == "gather":
            await asyncio.gather(scope.dispose())
        elif form == "create_task":
            await asyncio.create_task(scope.dispose())
        elif form == "shield":
            await asyncio.shield(scope.dispose())
        elif form == "wait_for":
            await asyncio.wait_for(scope.dispose(), 5)
        else:
            await asyncio.ensure_future(scope.dispose())

    return call


@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("size", [2, 3])
async def test_rings_complete_whatever_helper_task_the_wait_goes_through(
    form: str, size: int
) -> None:
    counts = [0] * size
    scopes = [Scope(name=f"s{i}") for i in range(size)]

    def make(i: int) -> Callable[[], Callable[[], Any]]:
        def effect() -> Callable[[], Any]:
            async def undo() -> None:
                counts[i] += 1
                await _dispose_via(form, scopes[(i + 1) % size])()

            return undo

        return effect

    for i in range(size):
        scopes[i].effect(make(i))
    await asyncio.wait_for(asyncio.gather(*(sc.dispose() for sc in scopes)), 5)
    assert all(sc.disposed for sc in scopes) and counts == [1] * size
    assert scope_module._WAITS == {}


async def _ring_fuzz(first_seed: int, last_seed: int) -> None:
    for seed in range(first_seed, last_seed):
        rng = random.Random(seed)
        n = rng.randint(2, 6)
        scopes: List[Scope] = [Scope(name="s0")]
        for i in range(1, n):
            if rng.random() < 0.5:
                scopes.append(rng.choice(scopes).child(f"s{i}"))
            else:
                scopes.append(Scope(name=f"s{i}"))
        counts = [0] * n
        violations: List[str] = []
        tasks: List["asyncio.Future[None]"] = []

        def make(i: int, targets: List[int], form: str) -> Callable[[], Callable[[], Any]]:
            def effect() -> Callable[[], Any]:
                async def undo() -> None:
                    counts[i] += 1
                    if scopes[i].children:
                        violations.append(f"s{i} undone while a child is alive")
                    for j in targets:
                        if form == "spawn":
                            tasks.append(asyncio.ensure_future(scopes[j].dispose()))
                        else:
                            await _dispose_via(form, scopes[j])()
                    await asyncio.sleep(0)

                return undo

            return effect

        for i in range(n):
            targets = [j for j in range(n) if rng.random() < 0.4]
            scopes[i].effect(make(i, targets, rng.choice(_FORMS + ["spawn"])))
        order = list(range(n))
        rng.shuffle(order)
        chosen = order[: rng.randint(1, n)]
        await asyncio.wait_for(asyncio.gather(*(scopes[i].dispose() for i in chosen)), 5)
        await asyncio.wait_for(asyncio.gather(*(sc.dispose() for sc in scopes)), 5)
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks), 5)
        assert all(sc.disposed for sc in scopes), f"seed {seed}"
        assert counts == [1] * n, f"seed {seed}"
        assert violations == [], f"seed {seed}: {violations}"
        assert scope_module._WAITS == {}, f"seed {seed}"


async def test_randomized_dispose_rings_never_deadlock_and_undo_exactly_once() -> None:
    await _ring_fuzz(20_000, 20_300)


# Note for anyone adding checks here: under the eager task factory a finished task
# stays in ``Scope._pending`` until its done-callback runs on the next loop turn,
# so a check of "no pending work" right after the last await needs a short sleep.
_EAGER = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="asyncio.eager_task_factory needs Python 3.12"
)


@_EAGER
async def test_plain_ring_completes_under_the_eager_task_factory() -> None:
    # With eager tasks the first synchronous part of a task runs inside
    # create_task(), before the caller could register anything about the task.
    loop = asyncio.get_running_loop()
    loop.set_task_factory(asyncio.eager_task_factory)  # type: ignore[attr-defined]
    try:
        a, b = Scope(name="a"), Scope(name="b")

        def eff_a() -> Callable[[], Any]:
            async def undo() -> None:
                await asyncio.sleep(0)  # A suspends first
                await b.dispose()

            return undo

        def eff_b() -> Callable[[], Any]:
            async def undo() -> None:
                await a.dispose()  # B's very first synchronous step calls A.dispose

            return undo

        a.effect(eff_a)
        b.effect(eff_b)
        ta = asyncio.ensure_future(a.dispose())
        tb = asyncio.ensure_future(b.dispose())
        await asyncio.wait_for(asyncio.gather(ta, tb), 5)
        assert a.disposed and b.disposed and scope_module._WAITS == {}
    finally:
        loop.set_task_factory(None)


@_EAGER
async def test_gather_ring_completes_under_the_eager_task_factory() -> None:
    loop = asyncio.get_running_loop()
    loop.set_task_factory(asyncio.eager_task_factory)  # type: ignore[attr-defined]
    try:
        scopes = [Scope(name=f"s{i}") for i in range(3)]
        counts = [0, 0, 0]

        def make(i: int) -> Callable[[], Callable[[], Any]]:
            def effect() -> Callable[[], Any]:
                async def undo() -> None:
                    counts[i] += 1
                    await asyncio.gather(scopes[(i + 1) % 3].dispose())

                return undo

            return effect

        for i in range(3):
            scopes[i].effect(make(i))
        await asyncio.wait_for(asyncio.gather(*(sc.dispose() for sc in scopes)), 5)
        assert all(sc.disposed for sc in scopes) and counts == [1, 1, 1]
    finally:
        loop.set_task_factory(None)


@_EAGER
async def test_ring_fuzz_under_the_eager_task_factory() -> None:
    loop = asyncio.get_running_loop()
    loop.set_task_factory(asyncio.eager_task_factory)  # type: ignore[attr-defined]
    try:
        await _ring_fuzz(30_000, 30_150)
    finally:
        loop.set_task_factory(None)


async def test_dispose_completes_with_an_async_effect_that_never_resolves() -> None:
    s = Scope(name="s")
    log: List[str] = []
    s.effect(rec(log, "earlier"))

    async def hang() -> None:
        await asyncio.Event().wait()

    s.effect(hang, label="hang")
    await turns()
    await asyncio.wait_for(s.dispose(), 5)  # 5 s is only a hang guard
    assert s.disposed and log == ["earlier"] and s.errors == []
    assert not s._live_pending()


async def test_remover_of_an_async_effect_that_never_resolves_completes() -> None:
    s = Scope(name="s")

    async def hang() -> None:
        await asyncio.Event().wait()

    remove = s.effect(hang)
    await turns()
    await asyncio.wait_for(remove(), 5)
    assert s.effect_count == 0 and not s._live_pending()
    await s.dispose()


async def test_effect_that_ignores_the_cancel_request_is_undone_exactly_once() -> None:
    log: List[str] = []
    gate = asyncio.Event()
    attempts: List[int] = []
    s = Scope(name="s")

    async def stubborn() -> Callable[[], None]:
        try:
            await gate.wait()
        except asyncio.CancelledError:
            attempts.append(1)
            await gate.wait()  # a second cancel request would end it here
        return lambda: log.append("undo")

    s.effect(stubborn)
    await turns()  # it is running
    d = asyncio.ensure_future(s.dispose())
    await turns()
    assert not d.done()  # still waiting for the effect, which ignored the request
    gate.set()
    await d
    assert log == ["undo"] and attempts == [1]  # asked once, undone once


async def test_effect_cancelled_before_its_first_step_leaves_no_warning() -> None:
    unraisable: List[Any] = []
    old_hook = sys.unraisablehook
    sys.unraisablehook = unraisable.append
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            s = Scope(name="s")

            async def never_started() -> None:
                await asyncio.sleep(0)

            s.effect(never_started)
            await s.dispose()
            gc.collect()
    finally:
        sys.unraisablehook = old_hook
    assert unraisable == [] and s.errors == [] and s.disposed


# ----------------------------------------------------------------------
# Second review round: cancelled restart, remover results
# ----------------------------------------------------------------------


async def test_cancelled_restart_leaves_the_scope_pending_not_active_and_empty() -> None:
    gate = asyncio.Event()
    s = Scope(name="s")

    def first(sc: Scope) -> None:
        def effect() -> Callable[[], Any]:
            async def undo() -> None:
                await gate.wait()

            return undo

        sc.effect(effect)

    await s.start(first)
    ran: List[str] = []
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(s.restart(lambda sc: ran.append("setup")), 0.05)
    assert s.state is ScopeState.PENDING and ran == []
    gate.set()
    await turns()
    assert s.state is ScopeState.PENDING and s.effect_count == 0
    await s.start(lambda sc: ran.append("again"))  # it can be started again
    assert s.state is ScopeState.ACTIVE and ran == ["again"]


def test_remover_result_works_with_asyncio_run_when_there_is_no_loop() -> None:
    log: List[str] = []
    s = Scope(name="s")
    asyncio.run(s.effect(rec(log, "sync"))())  # already finished, still accepted
    assert log == ["sync"]

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            log.append("async")

        return undo

    remove = s.effect(effect)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        asyncio.run(remove())  # a plain coroutine: no wrapper needed
        gc.collect()
    assert log == ["sync", "async"]


async def test_remover_result_works_with_create_task() -> None:
    log: List[str] = []
    s = Scope(name="s")
    await asyncio.create_task(s.effect(rec(log, "sync"))())

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            await asyncio.sleep(0)
            log.append("async")

        return undo

    await asyncio.create_task(s.effect(effect)())
    again = s.effect(rec(log, "again"))
    again()
    await asyncio.create_task(again())  # second call: finished awaitable
    assert log == ["sync", "async", "again"]


async def test_cancelling_a_task_that_wraps_a_remover_result_does_not_stop_the_undo() -> None:
    log: List[str] = []
    gate = asyncio.Event()
    s = Scope(name="s")

    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            await gate.wait()
            log.append("done")

        return undo

    t = asyncio.create_task(s.effect(effect)())
    await turns()
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    gate.set()
    await s.dispose()
    assert log == ["done"]


# ----------------------------------------------------------------------
# Third review round: a remover result may be used more than once
# ----------------------------------------------------------------------


def _slow_undo_effect(log: List[str], gate: Optional[asyncio.Event] = None) -> Any:
    def effect() -> Callable[[], Any]:
        async def undo() -> None:
            if gate is not None:
                await gate.wait()
            else:
                await asyncio.sleep(0)
            log.append("undone")

        return undo

    return effect


async def test_remover_result_can_be_wrapped_in_a_task_twice_in_a_row() -> None:
    log: List[str] = []
    s = Scope(name="s")
    result = s.effect(_slow_undo_effect(log))()
    await asyncio.create_task(result)
    await asyncio.create_task(result)  # already finished: returns at once
    await result
    assert log == ["undone"]


async def test_remover_result_can_be_awaited_by_several_tasks_at_once() -> None:
    log: List[str] = []
    gate = asyncio.Event()
    s = Scope(name="s")
    result = s.effect(_slow_undo_effect(log, gate))()
    both = asyncio.gather(
        asyncio.create_task(result), asyncio.create_task(result), _run(result)
    )
    await turns()
    assert not both.done()
    gate.set()
    await asyncio.wait_for(both, 5)
    assert log == ["undone"]


async def test_remover_result_works_with_wait_for_shield_and_gather() -> None:
    log: List[str] = []
    s = Scope(name="s")
    result = s.effect(_slow_undo_effect(log))()
    await asyncio.wait_for(result, 5)
    await asyncio.shield(result)
    await asyncio.gather(result, result)
    assert log == ["undone"]


async def test_cancelled_before_start_task_on_a_remover_result_keeps_the_undo() -> None:
    log: List[str] = []
    s = Scope(name="s")
    result = s.effect(_slow_undo_effect(log))()
    t = asyncio.create_task(result)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    await s.dispose()  # waits for the scheduled undo
    assert log == ["undone"]


async def test_shared_awaitable_is_cancelled_by_dispose_unless_shielded() -> None:
    async def work() -> None:
        await asyncio.Event().wait()

    shared = asyncio.ensure_future(work())
    protected = asyncio.ensure_future(work())
    s = Scope(name="s")
    s.effect(lambda: shared)
    s.effect(lambda: asyncio.shield(protected))
    await turns()
    await s.dispose()
    assert shared.cancelled()  # the scope owns what it was handed
    assert not protected.cancelled()  # only the scope's wait was cancelled
    protected.cancel()
    with pytest.raises(asyncio.CancelledError):
        await protected


# ----------------------------------------------------------------------
# Fourth review round: a parent never runs its undos before its children
# ----------------------------------------------------------------------


async def _children_first_with_background_wait() -> List[str]:
    # P has a child C; X is independent. C's undo starts a fire-and-forget task
    # that awaits X.dispose() (C's undo does not wait for it) and then waits on a
    # gate. X's undo really awaits P.dispose(). Before P.dispose() is called, the
    # wait graph holds C -> X -> P from the inherited marker.
    p = Scope(name="P")
    c = p.child("C")
    x = Scope(name="X")
    order: List[str] = []
    gate = asyncio.Event()

    async def background() -> None:
        await x.dispose()

    def eff_c() -> Callable[[], Any]:
        async def undo() -> None:
            order.append("C undo start")
            asyncio.ensure_future(background())
            await gate.wait()
            order.append("C undo end")

        return undo

    def eff_x() -> Callable[[], Any]:
        async def undo() -> None:
            await p.dispose()

        return undo

    def eff_p() -> Callable[[], Any]:
        return lambda: order.append(f"P own undo; C is {c.state.value}")

    c.effect(eff_c)
    x.effect(eff_x)
    p.effect(eff_p)
    tc = asyncio.ensure_future(c.dispose())
    await turns()
    tp = asyncio.ensure_future(p.dispose())
    await turns(10)
    assert order == ["C undo start"]  # P is waiting for its child, whatever the graph says
    gate.set()
    await asyncio.wait_for(asyncio.gather(tc, tp), 5)
    await turns()
    assert scope_module._WAITS == {}
    return order


async def test_parent_never_runs_its_undos_before_a_child_that_is_unwinding() -> None:
    order = await _children_first_with_background_wait()
    assert order == ["C undo start", "C undo end", "P own undo; C is disposed"]


@_EAGER
async def test_parent_waits_for_its_child_under_the_eager_task_factory_too() -> None:
    loop = asyncio.get_running_loop()
    loop.set_task_factory(asyncio.eager_task_factory)  # type: ignore[attr-defined]
    try:
        order = await _children_first_with_background_wait()
    finally:
        loop.set_task_factory(None)
    assert order == ["C undo start", "C undo end", "P own undo; C is disposed"]


@pytest.mark.parametrize("child_first", [True, False])
async def test_a_real_ring_through_a_child_is_opened_at_the_childs_call(
    child_first: bool,
) -> None:
    # The child's undo awaits parent.dispose() while the parent waits for the
    # child: a real ring. The child's call must be the one that returns early.
    p = Scope(name="P")
    c = p.child("C")
    log: List[str] = []
    seen: List[str] = []

    def eff_c() -> Callable[[], Any]:
        async def undo() -> None:
            log.append("C undo start")
            await p.dispose()  # returns early: the parent waits for this very undo
            seen.append(f"P is {p.state.value} when C's call returned")
            await turns(3)
            log.append("C undo end")

        return undo

    c.effect(eff_c)
    p.effect(lambda: (lambda: log.append("P own undo")))
    if child_first:
        first = asyncio.ensure_future(c.dispose())
        await turns()
        second = asyncio.ensure_future(p.dispose())
    else:
        first = asyncio.ensure_future(p.dispose())
        await turns()
        second = asyncio.ensure_future(c.dispose())
    await asyncio.wait_for(asyncio.gather(first, second), 5)
    assert log == ["C undo start", "C undo end", "P own undo"]
    assert seen and "disposed" not in seen[0]  # it did not wait for the parent to finish
    assert p.disposed and c.disposed and scope_module._WAITS == {}


async def test_ring_through_a_child_via_helper_task_completes() -> None:
    p = Scope(name="P")
    c = p.child("C")
    log: List[str] = []

    def eff_c() -> Callable[[], Any]:
        async def undo() -> None:
            await asyncio.gather(p.dispose())
            log.append("C undo end")

        return undo

    c.effect(eff_c)
    p.effect(lambda: (lambda: log.append("P own undo")))
    await asyncio.wait_for(asyncio.gather(c.dispose(), p.dispose()), 5)
    assert log == ["C undo end", "P own undo"]


# ----------------------------------------------------------------------
# Fourth review round: every awaiter of a remover result has its own shield
# ----------------------------------------------------------------------


async def test_second_await_after_a_wait_for_timeout_is_not_cancelled() -> None:
    log: List[str] = []
    gate = asyncio.Event()
    s = Scope(name="s")
    result = s.effect(_slow_undo_effect(log, gate))()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(result, 0.02)
    assert log == []  # the undo is still running
    waiter = asyncio.ensure_future(_run(result))  # later wait, by a task never cancelled
    await turns()
    assert not waiter.done()
    gate.set()
    await asyncio.wait_for(waiter, 5)  # no spurious CancelledError
    assert log == ["undone"]


async def test_cancelling_one_of_two_tasks_on_one_waiter_leaves_the_other_alone() -> None:
    log: List[str] = []
    gate = asyncio.Event()
    s = Scope(name="s")
    result = s.effect(_slow_undo_effect(log, gate))()
    t1 = asyncio.create_task(result)
    t2 = asyncio.create_task(result)
    await turns()
    t1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t1
    await turns()
    assert not t2.done() and log == []
    gate.set()
    await asyncio.wait_for(t2, 5)  # got the result, not a cancellation
    assert log == ["undone"] and not t2.cancelled()


async def test_gather_of_the_same_waiter_twice_and_cancel_before_first_step() -> None:
    log: List[str] = []
    s = Scope(name="s")
    result = s.effect(_slow_undo_effect(log))()
    await asyncio.wait_for(asyncio.gather(result, result), 5)
    assert log == ["undone"]
    again = s.effect(_slow_undo_effect(log))()
    t = asyncio.create_task(again)
    t.cancel()  # before the task ever ran
    with pytest.raises(asyncio.CancelledError):
        await t
    await asyncio.wait_for(again, 5)  # a later await still sees the finished undo
    assert log == ["undone", "undone"]


async def test_waiter_send_from_a_plain_callback_without_a_task() -> None:
    log: List[str] = []
    gate = asyncio.Event()
    s = Scope(name="s")
    result = s.effect(_slow_undo_effect(log, gate))()
    loop = asyncio.get_running_loop()
    got: List[Any] = []
    loop.call_soon(lambda: got.append(result.send(None)))  # no current task here
    await turns()
    assert len(got) == 1 and not got[0].done()
    gate.set()
    await asyncio.wait_for(_run(result), 5)

    def finish() -> None:
        try:
            result.send(None)
        except StopIteration:
            got.append("stopped")

    loop.call_soon(finish)
    await turns()
    assert got[1:] == ["stopped"] and log == ["undone"]


# ----------------------------------------------------------------------
# Fifth review round: a ring may close through more than one path
# ----------------------------------------------------------------------


async def _two_paths_to_the_parent() -> List[str]:
    # No background task, every wait is real. P has a child C. C's undo awaits
    # P.dispose() and Y.dispose() together; Y's undo awaits P.dispose() too. So
    # there are two paths from C back to P (C -> P and C -> Y -> P), and waking
    # one outside wait is not enough to open the ring.
    p = Scope(name="P")
    c = p.child("C")
    y = Scope(name="Y")
    log: List[str] = []
    gate = asyncio.Event()

    def eff_c() -> Callable[[], Any]:
        async def undo() -> None:
            await gate.wait()
            await asyncio.gather(p.dispose(), y.dispose())
            log.append("C done")

        return undo

    def eff_y() -> Callable[[], Any]:
        async def undo() -> None:
            await gate.wait()
            await p.dispose()
            log.append("Y done")

        return undo

    p.effect(lambda: (lambda: log.append(f"P own undo, C is {c.state.value}")))
    c.effect(eff_c)
    y.effect(eff_y)
    tc = asyncio.ensure_future(c.dispose())
    ty = asyncio.ensure_future(y.dispose())
    await turns()
    gate.set()
    await asyncio.wait_for(asyncio.gather(tc, ty), 5)
    await asyncio.wait_for(p.dispose(), 5)
    assert p.disposed and c.disposed and y.disposed
    assert scope_module._WAITS == {}
    return log


async def test_ring_closing_through_two_paths_is_fully_opened() -> None:
    log = await _two_paths_to_the_parent()
    assert sorted(log) == ["C done", "P own undo, C is disposed", "Y done"]  # each once
    assert log.index("C done") < log.index("P own undo, C is disposed")


@_EAGER
async def test_ring_closing_through_two_paths_under_the_eager_task_factory() -> None:
    loop = asyncio.get_running_loop()
    loop.set_task_factory(asyncio.eager_task_factory)  # type: ignore[attr-defined]
    try:
        log = await _two_paths_to_the_parent()
    finally:
        loop.set_task_factory(None)
    assert sorted(log) == ["C done", "P own undo, C is disposed", "Y done"]
    assert log.index("C done") < log.index("P own undo, C is disposed")


async def _second_path_from_a_background_task(second_path_last: bool) -> List[str]:
    # The ring P -> C -> Z -> P is real. A fire-and-forget task of C's undo adds a
    # second path C -> Y -> P.
    p = Scope(name="P")
    c = p.child("C")
    z = Scope(name="Z")
    y = Scope(name="Y")
    log: List[str] = []
    gate = asyncio.Event()

    def eff_c() -> Callable[[], Any]:
        async def undo() -> None:
            if second_path_last:
                asyncio.ensure_future(y.dispose())
            await z.dispose()
            log.append("C done")

        return undo

    def waits_for_parent(name: str) -> Callable[[], Callable[[], Any]]:
        def effect() -> Callable[[], Any]:
            async def undo() -> None:
                await gate.wait()
                await p.dispose()
                log.append(f"{name} done")

            return undo

        return effect

    p.effect(lambda: (lambda: log.append(f"P own undo, C is {c.state.value}")))
    c.effect(eff_c)
    z.effect(waits_for_parent("Z"))
    y.effect(waits_for_parent("Y"))
    tc = asyncio.ensure_future(c.dispose())
    await turns()
    gate.set()
    await asyncio.wait_for(tc, 5)
    await asyncio.wait_for(asyncio.gather(z.dispose(), y.dispose(), p.dispose()), 5)
    assert p.disposed and c.disposed and y.disposed and z.disposed
    assert scope_module._WAITS == {}
    return log


@pytest.mark.parametrize("second_path_last", [True, False])
async def test_ring_with_a_second_spurious_path_from_a_background_task(
    second_path_last: bool,
) -> None:
    log = await _second_path_from_a_background_task(second_path_last)
    assert sorted(log) == sorted(
        ["C done", "Z done", "Y done", "P own undo, C is disposed"]
    )  # every undo ran once
    assert log.index("C done") < log.index("P own undo, C is disposed")


@_EAGER
async def test_ring_with_a_second_spurious_path_under_the_eager_task_factory() -> None:
    loop = asyncio.get_running_loop()
    loop.set_task_factory(asyncio.eager_task_factory)  # type: ignore[attr-defined]
    try:
        log = await _second_path_from_a_background_task(True)
    finally:
        loop.set_task_factory(None)
    assert log.index("C done") < log.index("P own undo, C is disposed")


async def _background_fuzz(first_seed: int, last_seed: int) -> None:
    # "Start a background task, then await another scope" undos, with parents and
    # children, staggered starts (counted in loop turns, so no real time passes).
    for seed in range(first_seed, last_seed):
        rng = random.Random(seed)
        n = rng.randint(2, 6)
        scopes: List[Scope] = [Scope(name="s0")]
        for i in range(1, n):
            if rng.random() < 0.5:
                scopes.append(rng.choice(scopes).child(f"s{i}"))
            else:
                scopes.append(Scope(name=f"s{i}"))
        runs: Dict[Any, int] = {}
        violations: List[str] = []
        background: List["asyncio.Future[None]"] = []

        def make(
            owner: Scope, key: Any, kind: str, tgt: Scope, tgt2: Scope
        ) -> Callable[[], Callable[[], Any]]:
            def effect() -> Callable[[], Any]:
                async def undo() -> None:
                    runs[key] = runs.get(key, 0) + 1
                    if owner.children:
                        violations.append(f"{owner.name} undone while a child is alive")
                    if kind == "turns":
                        await turns(rng.randint(0, 3))
                    elif kind == "bg_then_other":
                        background.append(asyncio.ensure_future(tgt2.dispose()))
                        await tgt.dispose()
                    elif kind == "other":
                        await tgt.dispose()
                    elif kind == "gather":
                        await asyncio.gather(tgt.dispose(), turns(2))

                return undo

            return effect

        expected = 0
        for owner in scopes:
            for k in range(rng.randint(0, 3)):
                kind = rng.choice(["turns", "other", "other", "gather", "bg_then_other"] * 2)
                owner.effect(
                    make(owner, (owner.name, k), kind, rng.choice(scopes), rng.choice(scopes))
                )
                expected += 1
        order = list(scopes)
        rng.shuffle(order)
        started = []
        for sc in order:
            started.append(asyncio.ensure_future(sc.dispose()))
            await turns(rng.choice([0, 0, 2]))
        await asyncio.wait_for(asyncio.gather(*started), 5)
        if background:
            await asyncio.wait_for(asyncio.gather(*background), 5)
        assert all(sc.disposed for sc in scopes), f"seed {seed}"
        assert sum(runs.values()) == expected and set(runs.values()) <= {1}, f"seed {seed}"
        assert violations == [], f"seed {seed}: {violations[:3]}"
        assert scope_module._WAITS == {}, f"seed {seed}"


async def test_randomized_background_then_wait_rings_never_hang() -> None:
    await _background_fuzz(40_000, 40_400)


@_EAGER
async def test_randomized_background_then_wait_rings_under_the_eager_task_factory() -> None:
    loop = asyncio.get_running_loop()
    loop.set_task_factory(asyncio.eager_task_factory)  # type: ignore[attr-defined]
    try:
        await _background_fuzz(41_000, 41_250)
    finally:
        loop.set_task_factory(None)
