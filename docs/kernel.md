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
