# Tools: specs, registry and frozen arguments

This page describes the declaration layer of the guarded tool pipeline (issue #42, story #59). It lives in the package `aria/tools/` and uses only the standard library.

**Read this first: this story only adds the declaration layer. Nothing enforces it yet.** Declaring a tool does not check a permission, does not stop an undeclared tool from running and does not apply a timeout. `LLMAgent` still runs any callable it is given. The enforcement comes in the later stories, mainly #60 (decision stages) and #63 (wiring into `LLMAgent`). Until then, treat the specs as documentation that the pipeline will read.

## What a tool spec is

A tool spec is a small, immutable record in which a tool says what it needs before it may run:

| Field | Meaning | Default |
|---|---|---|
| `name` | 1 to 64 characters from `A-Z a-z 0-9 _ . -`, starting with a letter or underscore | the function name |
| `needs` | a tuple of `Need` objects: which capability, over which scope | required |
| `timeout_s` | longest run time in seconds: a finite number above 0, at most 300 | `30.0` |
| `concurrency_safe` | may this call run at the same time as others | `False` |
| `approval` | `"auto"` or `"always"` (a human approves every call) | `"auto"` |
| `internal` | a tool of the framework itself that needs no capability | `False` |
| `description` | free text, at most 2000 characters | empty |

`spec.risk` is the highest level among the needed capabilities, taken from `RISK_LEVEL` in the permission engine (`low`, `medium`, `high`, `critical`). A tool with no needs is `low`. A capability the engine has no level for counts as `critical`, so a gap makes a tool look more dangerous, never less.

A tool that is not `internal` must declare at least one need. A tool that touches files, the network or anything else outside itself has to say so.

### Needs and scopes

A `Need` is one capability plus the scope it is needed over. The scope is the resource the permission applies to (see `aria/permissions/scope.py`).

- `Need(Capability.FILE_READ)` has the fixed scope `"*"`. Only a grant for `*` covers a request for `*`.
- `Need(Capability.FILE_READ, scope="/data/reports")` has a fixed scope.
- `Need(Capability.WEB_BROWSE, scope_from="url")` takes the scope from the argument named `url`.
- `Need(Capability.FILE_READ, scope_from=lambda args: args["dir"] + "/" + args["name"])` computes it from the frozen arguments.

Use a fixed scope or `scope_from`, not both. `need.resolve_scope(arguments)` returns the scope string. It raises `ToolSpecError` when the named argument is missing, is not a string, is empty or is longer than 4096 characters, or when the callable raises or returns something that is not a string. **A caller that gets `ToolSpecError` from `resolve_scope` must deny the call.** It must never fall back to `*` or to "no scope". Error messages carry argument names and type names, never argument values.

`resolve_scope` returns only the string. Whether it is a safe path, URL or id is checked later by the pipeline with the functions in `aria.permissions.scope`.

## How to declare a tool

With the decorator. It attaches the spec to the function as `__aria_tool_spec__` and returns the same function, so the signature and behaviour do not change. It works on sync and async functions and on methods in classes.

```python
from aria.permissions import Capability
from aria.tools import tool

@tool(Capability.WEB_BROWSE, scope_from="url", timeout_s=20)
async def fetch_page(url: str) -> str:
    ...

@tool(Capability.FILE_READ, Capability.MEMORY_WRITE, scope="/data/reports")
def summarise(path: str) -> str:
    ...

class Mailer:
    @tool(Capability.EMAIL_SEND, approval="always", name="mailer.send")
    async def send(self, to: str, body: str) -> None:
        ...
```

Bare `Capability` members are turned into `Need` objects using `scope` or `scope_from`. You can also pass ready-made `Need` objects. `scope` and `scope_from` apply only to the bare capabilities, and passing them with none is an error. A bad value (a timeout of 0, an unknown approval mode, a bad name) raises `ToolSpecError` when the decorator line runs, so it fails at import time. Apply `@tool` below `@staticmethod` and `@classmethod`, to the function itself and not to a bound method. Applying it twice to one function raises.

For a callable you do not own (a third-party function, a `functools.partial`) build the spec yourself and pass it when you register:

```python
from aria.tools import Need, ToolRegistry, ToolSpec

registry = ToolRegistry()
registry.register(
    third_party_search,
    ToolSpec(name="search", needs=(Need(Capability.WEB_SEARCH),)),
)
```

An explicit spec wins over `__aria_tool_spec__`.

## The registry and undeclared tools

`ToolRegistry.register(fn, spec=None, *, name=None)` stores the callable and an immutable copy of its spec. The copy is made at registration and checked again, so changing `fn.__aria_tool_spec__` later, or changing the original spec object (even with `object.__setattr__`), has no effect on what the registry holds. `registry.spec(name)` also hands out copies. Duplicate names and non-callables raise `ToolSpecError`. A name given with `name=` is used as the registry key and as the stored spec's name, so the two always match.

A tool registered with no spec at all is **undeclared**:

- `registry.undeclared()` lists such names.
- `registry.spec(name)` returns `None` for them.
- `registry.get(name)` still returns the callable, so a later stage can find it and deny it by name.
- A malformed `__aria_tool_spec__` (not a `ToolSpec`) is refused, not treated as undeclared.

The design decision for version 1 (see `docs/design/0042-guarded-pipeline.md`) is that **the pipeline denies undeclared tools**. An opt-in compatibility mode that allows and logs them is planned, off by default. **None of this is enforced yet**; today the registry only reports which tools are undeclared.

`registry.describe()` returns a JSON-safe list with name, declared flag, risk, needs (capability value strings), timeout, `concurrency_safe`, approval, `internal` and description. For an undeclared tool the risk is `"unknown"` and the other fields are empty.

## How arguments are frozen

`freeze_args(arguments)` checks the arguments of a tool call and returns a deeply immutable copy, a `FrozenArgs`. The idea is that the call a person approves and the call that runs are the same thing: policy, approval, the audit digest and the tool itself all work from the frozen copy, and later changes to the caller's dict cannot reach them.

Accepted: a plain `dict` with `str` keys, and inside it `dict`, `list`, `tuple` (treated as a list), `str`, `int`, `float`, `bool` and `None`. Types are matched exactly, so a subclass (a `str` subclass, an `OrderedDict`, a namedtuple, an `IntEnum`) is rejected, because a subclass can override methods and run code while it is read.

Rejected with `FreezeError` (a `ValueError`): non-string keys, bytes, sets, any other object, NaN and infinity, integers with more than 100 digits, cycles (an object that contains itself, found by tracking the objects on the current path), and data over a limit. The same structure used twice without a cycle is allowed but counted each time, so a small structure that expands to a huge tree hits the size limit.

`freeze_args` never calls `repr`, `str`, `==` or `hash` on an object it rejects. Error messages show a path such as `$.'items'[2].'url'` and the type name, never the value.

### Limits

| Limit | Default | Notes |
|---|---|---|
| `max_depth` | 16 | the top-level dict is depth 1; at most 100 may be asked for |
| `max_nodes` | 10 000 | every value counts, containers included |
| `max_string_chars` | 1 000 000 | per string, keys included |
| `max_total_chars` | 4 000 000 | all strings and keys together |
| integer size | 100 digits | fixed |

Limits are checked while reading, before a copy is made, so a hostile input cannot use much memory or time. The limits are keyword arguments of `freeze_args`.

### What you get

`FrozenArgs` is a read-only mapping: `[]`, `get`, `in`, iteration, `len` and `==` with a dict all work. Nested dicts are read-only `FrozenMap` objects and lists are tuples. A list in a dict compares equal to a tuple of the same items, and `True` is not equal to `1`.

- `frozen.thaw()` returns a fresh plain deep copy (dicts and lists) that a tool may change freely.
- `frozen.canonical_json()` returns sorted keys, no whitespace, ASCII only (`ensure_ascii`).
- `frozen.sha256()` returns the plain SHA-256 of that text, in hex. A keyed digest for audit records comes later from the redaction module.
- `repr` shows only the number of keys and the start of the digest, so values do not end up in logs.
- It cannot be pickled or built directly, and `copy.copy` and `copy.deepcopy` return the same object.

The canonical text and the digest are computed once at freeze time from the frozen copy.

Honest limits of the immutability: it is deep and stops mutation through the normal interface, but Python cannot stop code that reaches into private attributes or uses `object.__setattr__`. It guards against mistakes and careless code, not against hostile code running in the same process.

## Still to come

This section is extended by the later stories.

- Decision stages: policy hooks, approval, guards, the deny-wins fold (#60).
- The run stage: timeouts, cancellation, concurrency, one audit record per call.
- Wiring into `LLMAgent`, the executor and the orchestrator, so that undeclared tools are denied and specs are enforced (#63).
- Redacted argument previews and the keyed argument digest for audit records.
- The `allow_logged` compatibility mode for undeclared tools.
- Spill handling and the internal `read_spilled` tool.
