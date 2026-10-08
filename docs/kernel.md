# Kernel events

`aria.kernel.Events` is a small event system. Code announces that something
happened under a plain string name, and any number of listeners react. It uses
only the Python standard library.

**What this is and is not.** This gives clean extension points. It is NOT
isolation or sandboxing. A listener runs inside your process with all of its
privileges, and nothing here limits what it can do. Do not use it to contain
untrusted code.

## Registering listeners

```python
from aria.kernel import Events

events = Events()

dispose = events.on("task.done", handler)           # runs after existing listeners
events.on("task.done", early, prepend=True)         # runs before existing listeners
events.once("task.done", one_shot)                  # runs at most once

dispose()   # removes this registration. Calling it again does nothing.
```

- Event names are plain strings. `"__proto__"`, `"toString"` and `"__class__"`
  work like any other name.
- Every registration gets a unique token. Removal uses that token (through the
  disposer), never a comparison of functions. You can register the same
  function twice and remove only one of the two.
- `once` removes the listener BEFORE it runs, so it runs at most once even if
  it raises, or if the same event is emitted again from inside it.

## The five dispatch modes

| Mode | Sync or async | Use it when | Result |
|---|---|---|---|
| `emit(name, *args)` | sync | You are announcing something and do not need answers | nothing |
| `bail(name, *args)` | sync | The first listener with an answer decides (for example a veto) | first result that is not `None` and not `False` |
| `await serial(name, *args)` | async | Same as `bail`, listeners may be async and must run one at a time | first such result |
| `await parallel(name, *args)` | async | Independent work that may run together | nothing, or `ListenerErrors` |
| `waterfall(name, *args, inner=...)` | sync or async | Listeners wrap a core operation, like middleware | what the outermost layer returns |

### emit

Calls every listener in order and ignores return values. A listener that
raises does not stop the others. The failure is recorded and never silent:

```python
events.errors         # copy of the recorded ListenerError records (oldest first)
events.take_errors()  # read and clear
events.clear_errors()
```

Each record has `event`, `listener` (a label, or the one you pass with
`label=`) and `exception`. The record list is bounded (default 1000, set with
`Events(max_errors=...)`); `describe()` reports how many were dropped.

Note that a record holds the exception object, and with it the traceback and
every local variable and payload the traceback refers to. These stay alive
until you call `take_errors()` or `clear_errors()` (or the bound pushes the
record out), so drain the list regularly in long running programs.

**Async listeners in `emit`.** `emit` cannot wait. If a listener returns an
awaitable and an event loop is running, it is scheduled as a task and any
later failure is recorded in `errors`. If no loop is running, the coroutine is
closed without running and a `TypeError` is recorded for that listener, while
the remaining listeners still run. If you need to wait for async listeners,
use `serial` or `parallel`.

### bail

Returns the first result that is not `None` and not `False`, then stops.
`0`, `""` and `[]` DO count as results. If nobody bails the result is `None`.
An exception from a listener propagates and stops the dispatch, so a failing
guard fails closed. `bail` is sync: a listener that returns an awaitable
raises `TypeError` (a coroutine object must never be mistaken for a decision).

### serial

Like `bail`, but awaits each listener in turn. Sync and async listeners both
work. An exception propagates.

### parallel

Starts all listeners together. All of them run even if some fail. If any
failed, one `ListenerErrors` is raised holding every error in listener order
(`exc.errors`, `exc.exceptions`). It does not use `ExceptionGroup`, so it works
on Python 3.10. Returns nothing on success.

No listener outlives the call. If the caller is cancelled, or a listener
raises `CancelledError` (or another `BaseException`), the other listeners are
cancelled and awaited to completion before the exception propagates.

### waterfall

An onion chain. Each listener is called as `listener(*args, next)`:

```python
def audit(task, next):
    log("before")
    result = next()          # runs the next listener, or inner() at the end
    log("after")
    return result

def cache(task, next):
    if task in cache_store:
        return cache_store[task]   # never calls next: the chain ends here
    return next()

events.on("run", audit)
events.on("run", cache)
result = events.waterfall("run", task, inner=lambda: do_work(task))
```

- A listener that does not call `next` ends the chain. Its return value is the
  result.
- Calling `next` twice raises `NextAlreadyCalledError`, also when the second
  call happens after an `await`.
- Errors propagate through the chain, so a listener can catch what `next()`
  raises.

**Sync and async.** `waterfall` returns exactly what the outermost listener
returns. If any layer is a coroutine function, or `inner` is async, that value
is awaitable, so check with `inspect.isawaitable` and await it. From async
code, use `await events.awaterfall(...)`: it is a coroutine, it awaits any
awaitable result, and it gives you the final value.

```python
result = await events.awaterfall("run", task, inner=do_work_async)
```

An async layer should `await next()`. If it forgets and returns `next()`
directly, `awaterfall` still resolves it.

Two things the library cannot fix for you:

- A sync layer that calls `next()` and then throws the result away, while
  `inner` (or a later layer) is async, creates a coroutine that nobody awaits.
  Python then prints "coroutine was never awaited", and the work never ran.
  Return or await the result of `next()`.
- `awaterfall` awaits any awaitable that the chain returns, even one that is
  legitimately meant as data (for example a future you want to hand back
  unawaited). If your result is an awaitable on purpose, wrap it in a plain
  object, or use `waterfall` and handle the result yourself.

## Rules for every mode

- **Snapshot.** A dispatch works on the listeners as they were when it
  started. Listeners added during a dispatch do not run in it. A listener
  removed during a dispatch is skipped if its turn has not come yet.
- **Cancellation.** `asyncio.CancelledError` is never swallowed and is never
  recorded as a listener failure.
- **Fail closed.** `bail`, `serial` and `waterfall` let exceptions propagate.
  Only `emit` collects them (it is the "tell everyone" mode), and `parallel`
  gathers them into `ListenerErrors`.

## Threads and scale

- Registration and removal (`on`, `once`, disposers, `listener_count`,
  `describe`) are thread safe: one internal lock protects them, and it is
  never held while a listener runs.
- Dispatch (`emit`, `bail`, `serial`, `parallel`, `waterfall`) is meant to run
  on the event loop thread. `serial`, `parallel` and `awaterfall` need that
  loop; the sync modes can be called from any single thread you choose.
- Registration is O(1) and removal is O(1) on average: a removed listener is
  only marked as removed, and the storage is cleaned up in batches. A removed
  listener releases its function straight away, so a disposer you keep does not
  keep the listener alive. Tens of thousands of listeners on one event are fine.
  A dispatch takes its snapshot in O(n), once per change to the listener list.

## Health report

`events.describe()` returns a plain dict for status pages and health checks:

```python
{
    "listeners": {"task.done": 2, "run": 1},   # events with no listeners are omitted
    "error_count": 0,                          # errors currently collected by emit
    "dropped_errors": 0,                       # errors discarded because the list was full
    "pending_async_listeners": 0,              # async listeners started by emit still running
}
```

# Kernel scopes

`aria.kernel.Scope` makes sure that everything one part of ARIA registers is
undone when that part is unloaded. A plugin, a synthesized agent or a
background service registers things as it loads: event listeners, child
parts, tasks, connections. When it is unloaded, the scope undoes all of them
automatically, in a safe order, so nothing is left behind. If an undo fails,
the failure is recorded and visible. It uses only the Python standard library.
The idea comes from the Cordis "fibers and effects" model.

**What this is and is not.** A scope gives CLEANUP. It is NOT isolation or
sandboxing. Code that runs inside a scope has all the privileges of the
process, and a scope does not stop it from doing anything. It only makes sure
that what the code registered through the scope is removed again. Do not use it
to contain untrusted code.

## Example

```python
from aria.kernel import Events, Scope, ScopeState

events = Events()
root = Scope(events, name="app")

async def load_plugin(scope: Scope) -> None:
    scope.on("task.done", on_done)                 # listener, removed with the scope

    def open_log():
        f = open("plugin.log", "a")
        return f.close                             # the "undo" is returned
    scope.effect(open_log, label="log file")

    async def connect():
        conn = await db.connect()
        return conn.close                          # sync or async undo
    scope.effect(connect, label="database")

    helper = scope.child("helper")                 # a part inside the part
    helper.on("task.done", helper_handler)

plugin = root.child("plugin")
await plugin.start(load_plugin)
if plugin.state is ScopeState.FAILED:
    print(plugin.errors)                           # why it failed

await plugin.dispose()   # helper first, then database, log file, listener
```

`async with scope:` starts the scope when you enter and disposes it when you
leave, also when the block raises.

## Effects

`scope.effect(fn, label="")` runs `fn()` right away and keeps what it returns
as the undo. `fn` may return:

- `None`: there is nothing to undo.
- A callable: the undo. It may be sync, or return an awaitable (an `async def`,
  or a function that returns a task).
- A list or tuple of callables: all of them are run, last one first.
- An awaitable that resolves to one of the above. It is started as a task, so
  this needs a running event loop.

`effect` returns a **remover**. Calling it undoes that one effect early and
takes it out of the scope. The result is awaitable: if the undo is async, or the
effect is still resolving, `await` it. For a sync undo (all `scope.on`
listeners) it is already finished when the call returns. Calling a remover again
does nothing. A remover only ever touches the effect it was made for, so
removing effect A can never remove effect B, whatever was removed before.

`scope.on(event, fn, *, prepend=False, once=False)` registers an `Events`
listener as an effect. It needs the scope to have an `Events` instance
(`Scope(events, ...)`). Child scopes use their parent's events unless you pass
`scope.child(name, events=other)`. The same function registered twice is two
effects. A `once` listener that has fired no longer takes up space in the scope.

## Rules

1. **Reverse order, one at a time.** Undos run strictly one after the other,
   last registered first. An async undo is awaited before the next one starts.
   (Cordis starts async undos together, which breaks the order.)
2. **Children first.** A child scope is an effect of its parent. Disposing a
   parent disposes its children first, newest child first, and then the
   parent's own effects, newest first. A child depends on the parent, so it goes
   before what the parent set up.
3. **One failure never stops the rest.** Every undo runs on its own. If one
   raises, the others still run. (Cordis stops at the first error and loses the
   rest.)
4. **Failures are recorded, never swallowed.** `scope.errors` is a list of
   `ScopeError` records with `scope`, `label`, `exception` and `when`
   (`time.time()`). `scope.take_errors()` reads and clears them, and the list is
   bounded (`Scope(max_errors=...)`, default 1000). When a child scope ends with
   errors, its parent records a `ChildScopeErrors` entry, so they stay visible
   after the child is gone. Like `Events.errors`, a record keeps the exception
   and its traceback alive until you take or clear it.
5. **Dispose is safe to repeat.** `await scope.dispose()` a second time does
   nothing. Calls made while the first is running wait for it. After disposal,
   `effect`, `on` and `child` raise `ScopeDisposedError`. They also raise it
   while the scope is being unwound.
6. **No deadlock.** An undo may call `await scope.dispose()` on its own scope (or
   on a parent): it returns at once. An undo that tries to register a new effect
   gets `ScopeDisposedError`, which is recorded. No lock is held while your code
   runs.
7. **Nothing leaks while something is still pending.** If you dispose a scope
   while an async effect has not resolved, disposal waits for it in its place in
   the order and then undoes it. If you dispose while an async setup is still
   running, what the setup already registered is undone, and its next attempt to
   register raises `ScopeDisposedError`.
8. **Cancellation is not failure.** `asyncio.CancelledError` is never recorded
   and never swallowed. Cancelling the task that waits for `dispose()` (for
   example `asyncio.wait_for` timing out) stops the waiting only: the
   unwinding runs on as its own task until it is finished. Cancelling the
   unwinding task itself does not skip the remaining undos either; they run, and
   then the cancellation is raised.

Two small conveniences: a sync or async generator returned as an undo is run to
the end (calling a generator function does nothing by itself, and silently
skipping it would hide a bug), and any other non-awaitable value an undo
returns, such as `True`, is ignored.

## Lifecycle

| State | Meaning |
|---|---|
| `PENDING` | Created, setup has not finished. You can already register effects. |
| `ACTIVE` | `start` finished without error. |
| `FAILED` | Setup raised. Everything it registered has been undone. |
| `DISPOSED` | Finished for good. |

`await scope.start(setup=None)` runs an optional `setup(scope)` (sync or async).
Async effects registered during setup are awaited first.

- On success the scope is `ACTIVE`.
- If setup raises, the effects registered so far are undone, the error is
  recorded in `scope.errors`, and the scope is `FAILED`. **The exception is
  not raised again.** One failing plugin must not crash its siblings, so you
  check `scope.state` (or `scope.errors`) instead. A `FAILED` scope never tries
  again by itself.
- `await scope.restart(setup)` tries again from `FAILED` (or from `ACTIVE`, which
  undoes everything first).
- If setup is cancelled, what it registered is undone, the scope goes back to
  `PENDING`, and the `CancelledError` propagates.

If the scope has an `Events` instance, each state change is announced with a
plain `emit`: `events.emit("scope/state", scope_name, new_state)`. A listener
that fails is recorded by `Events` and does not affect the scope.

## Health report

`scope.describe()` returns plain data (safe for `json.dumps`):

```python
{
    "name": "plugin",
    "state": "active",
    "effects": 3,                 # this scope's own effects, children not counted
    "error_count": 0,
    "dropped_errors": 0,          # errors discarded because the list was full
    "recent_errors": [],          # the last few errors as text
    "children": [ {...same shape...} ],   # live child scopes
}
```

Very deep trees are cut off after 16 levels (`children_truncated` tells how
many children were left out).

## Limits

- A scope is meant to be used from the event loop thread. It is not thread
  safe.
- The sync part of a remover and of `effect` needs no loop, but an async effect,
  `start`, `restart` and `dispose` do.
- An undo that is waiting forever makes `dispose()` wait forever. Wrap the call
  in `asyncio.wait_for` if you need a time limit: the unwinding continues in the
  background and the scope still ends up clean.
- Out of scope for now: service injection, reactive reloads and context
  isolation.
