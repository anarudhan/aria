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
from typing import Any, Callable, List, Optional

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
    await s.dispose()  # waits for the effect to resolve, then undoes it
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
        await gate.wait()
        return lambda: log.append("undone")

    remove = s.effect(slow)
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
        await gate.wait()
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

    await asyncio.wait_for(s.start(setup), 5)  # 5 s is only a hang guard
    assert s.state is ScopeState.FAILED and len(s.errors) == 1


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
        await gate.wait()
        return lambda: log.append("undone")

    remove = s.effect(slow)
    remove()  # not awaited, effect still resolving
    assert s.effect_count == 0
    d = asyncio.ensure_future(s.dispose())
    await turns()
    assert not d.done()  # dispose waits for the remover's work
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
