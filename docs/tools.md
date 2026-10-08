# Tools: specs, registry and frozen arguments

This page describes the declaration layer of the guarded tool pipeline (issue #42, story #59). It lives in the package `aria/tools/` and uses only the standard library.

**Read this first: this story only adds the declaration layer. Nothing enforces it yet.** Declaring a tool does not check a permission, does not stop an undeclared tool from running and does not apply a timeout. `LLMAgent` still runs any callable it is given. The enforcement comes in the later stories, mainly #60 (decision stages) and #63 (wiring into `LLMAgent`). Until then, treat the specs as documentation that the pipeline will read.

## What a tool spec is

A tool spec is a small, immutable record in which a tool says what it needs before it may run:

| Field | Meaning | Default |
|---|---|---|
| `name` | 1 to 64 characters from `A-Z a-z 0-9 _ . -`, starting with a letter or underscore (pattern `TOOL_NAME_PATTERN`, which ends in `\Z` so a trailing newline is refused by `re.match` too) | the function name |
| `needs` | a tuple of `Need` objects: which capability, over which scope | required |
| `timeout_s` | longest run time in seconds: a finite number above 0, at most 300 | `30.0` |
| `concurrency_safe` | may this call run at the same time as others | `False` |
| `approval` | `"auto"` or `"always"` (a human approves every call) | `"auto"` |
| `internal` | the tool is part of the framework itself. Set only by the registry, never by the tool | `False` |
| `description` | free text, at most 2000 characters | empty |

`spec.risk` is the highest level among the needed capabilities, taken from `RISK_LEVEL` in the permission engine (`low`, `medium`, `high`, `critical`). A capability the engine has no level for counts as `critical`, so a gap makes a tool look more dangerous, never less.

A `Need` or `ToolSpec` is validated exactly when it is built: `__init__` checks every value first and assigns only if all are valid, and it marks the object ready last. Calling `__init__` again on an existing object raises and changes nothing, and a failed `__init__` leaves a blank object that nothing accepts (`risk`, `resolve_scope` and `register()` refuse an object that never finished `__init__`, such as one made with `copyreg.__newobj__`). Copying returns the same object and pickling is refused, so there is no other way to build one. Immutability is the normal-interface kind described under "Honest limits".

A tool that is not `internal` must declare at least one need. A tool that touches files, the network or anything else outside itself has to say so.

**`internal` is NOT a permission exemption.** A pipeline must never treat it as authority to skip a permission check. It only says that the tool is part of the framework itself, for example so that the spill read-back is not spilled again. A tool cannot declare itself internal: `ToolSpec(internal=True)` and `@tool(internal=True)` raise `ToolSpecError`. Only the registry grants it, and **by identity, not by name**: the framework builds the registry with `ToolRegistry(internal_tools={"read_spilled": the_callable})`, and a tool is internal only if the object registered under that name *is* (`is`) that callable. A different object that uses a reserved name is refused with `ToolSpecError`, whichever is registered first, so a user tool cannot take the name and the exemption that goes with it. Registering the internal callable under another name makes it an ordinary tool only when a spec is given (and then it needs at least one need); with no spec it is simply undeclared. Bound methods are new objects on every attribute access, so pass the very same bound-method object to both `ToolRegistry(internal_tools=...)` and `register()`; a fresh `obj.method` is a different object and its registration is refused. Under its reserved name, an internal tool is always declared: with no spec they get an internal spec with no needs (the only kind of spec allowed to have none), and with a spec they keep the needs it declares. Whoever builds the registry decides the callables, so the pipeline must be given the framework's own registry. This is a guard against mistakes, not against code in the same process.

### Needs and scopes

A `Need` is one capability plus the scope it is needed over. The scope is the resource the permission applies to (see `aria/permissions/scope.py`).

- `Need(Capability.FILE_READ)` has the fixed scope `"*"`. Only a grant for `*` covers a request for `*`.
- `Need(Capability.FILE_READ, scope="/data/reports")` has a fixed scope.
- `Need(Capability.WEB_BROWSE, scope_from="url")` takes the scope from the argument named `url`.
- `Need(Capability.FILE_READ, scope_from=lambda args: args["dir"] + "/" + args["name"])` computes it from the frozen arguments.

Use a fixed scope or `scope_from`, not both. `need.resolve_scope(arguments)` returns the scope string. For `scope_from`, `arguments` must be the `FrozenArgs` (or a nested `FrozenMap`) built by `freeze_args`; a plain dict, or anything else, raises `ToolSpecError`, because a callable could change a plain dict. It raises `ToolSpecError` when the named argument is missing, is not a string, is empty or is longer than 4096 characters, or when the callable raises or returns something that is not a string. **A caller that gets `ToolSpecError` from `resolve_scope` must deny the call.** It must never fall back to `*` or to "no scope". Error messages carry argument names and type names, never argument values.

`resolve_scope` returns only the string. Whether it is a safe path, URL or id is checked later by the pipeline with the functions in `aria.permissions.scope`; this layer only checks that the text is non-empty and at most 4096 characters.

A `scope_from` callable must be **pure, fast and deterministic**. The spec holds the same callable object (it cannot be copied), nothing times it and nothing sandboxes it. The pipeline must resolve each `Need` exactly once per call and pass that same string through every later stage.

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

A spec belongs to the object it was declared on. `functools.wraps` copies `__dict__`, and a subclass inherits class attributes, so a wrapper or a subclass would otherwise carry the spec of the tool it wraps and run under that tool's narrower needs. The decorator therefore also records the owner in a private weak table that `functools.wraps` cannot copy. The registry accepts a spec only when the owner is the object being registered, the function of a bound method, or the class of an instance. A wrapper or subclass raises `ToolSpecError` and **must declare its own needs** (this includes a class created by a decorated metaclass, which is not allowed to inherit the metaclass's declaration; only an instance of a decorated, non-metaclass class is accepted through its class): apply `@tool` to it (this works on a `wraps` wrapper) or pass an explicit spec. A hand-set, replaced or removed `__aria_tool_spec__` is refused for the same reason.

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

**A string `scope_from` is checked against the tool's signature at registration.** `Need(FILE_READ, scope_from="path")` is accepted only if the tool has a parameter `path` that can receive a keyword argument (or the tool takes `**kwargs`); a positional-only or `*args` parameter does not count. Otherwise the scope would be taken from one argument while the tool reads another, for example `def reader(file)` with the arguments `{"path": "/allowed/x", "file": "/etc/passwd"}`. For a bound method `self` is ignored, for a callable object the signature of `__call__` is used, and if the signature cannot be read the registration fails. The signature is whatever `inspect.signature` reports, which follows `__wrapped__` and `__signature__`, so a tool author who deliberately lies there can pass the check. A tool with `**kwargs` accepts any `scope_from` name. This only checks the NAME. **A `scope_from` callable can look at one argument while the tool uses another, and a tool with `**kwargs` can ignore the named one; this layer cannot prevent that.** The pipeline must reject arguments the tool does not declare (see Still to come).

## The registry and undeclared tools

`ToolRegistry.register(fn, spec=None, *, name=None)` stores the callable and an immutable copy of its spec. The copy is made at registration and checked again, so changing `fn.__aria_tool_spec__` later, or changing the original spec object (even with `object.__setattr__`), has no effect on what the registry holds. `registry.spec(name)` also hands out copies. Duplicate names and non-callables raise `ToolSpecError`. A name given with `name=` is used as the registry key and as the stored spec's name, so the two always match.

A tool registered with no spec at all is **undeclared**:

- `registry.undeclared()` lists such names.
- `registry.spec(name)` returns `None` for them.
- `registry.get(name)` still returns the callable, so a later stage can find it and deny it by name.
- A malformed, hand-set or inherited `__aria_tool_spec__` is refused, not treated as undeclared.

The design decision for version 1 (see `docs/design/0042-guarded-pipeline.md`) is that **the pipeline denies undeclared tools**. An opt-in compatibility mode that allows and logs them is planned, off by default. **None of this is enforced yet**; today the registry only reports which tools are undeclared.

`registry.describe()` returns a JSON-safe list with name, declared flag, risk, needs (capability value strings), timeout, `concurrency_safe`, approval, `internal` and description. For an undeclared tool the risk is `"unknown"` and the other fields are empty.

## How arguments are frozen

`freeze_args(arguments)` checks the arguments of a tool call and returns a deeply immutable copy, a `FrozenArgs`. The idea is that the call a person approves and the call that runs are the same thing: policy, approval, the audit digest and the tool itself all work from the frozen copy, and later changes to the caller's dict cannot reach them.

Accepted: a plain `dict` with `str` keys, and inside it `dict`, `list`, `tuple` (treated as a list), `str`, `int`, `float`, `bool` and `None`. Types are matched exactly, so a subclass (a `str` subclass, an `OrderedDict`, a namedtuple, an `IntEnum`) is rejected, because a subclass can override methods and run code while it is read.

Rejected with `FreezeError` (a `ValueError`): non-string keys, bytes, sets, any other object, NaN and infinity, strings or keys with an unpaired surrogate (see below), integers with more than 100 digits, cycles (an object that contains itself, found by tracking the objects on the current path), and data over a limit. The same structure used twice without a cycle is allowed but counted each time, so a small structure that expands to a huge tree hits the size limit.

`freeze_args` never calls `repr`, `str`, `==` or `hash` on an object it rejects. Error messages show a path such as `$.'items'[2].'url'` and the type name, never the value.

### Limits

| Limit | Default | Notes |
|---|---|---|
| `max_depth` | 16 | the top-level dict is depth 1; at most 100 may be asked for |
| `max_nodes` | 10 000 | every value counts, containers included |
| `max_string_chars` | 1 000 000 | per string, keys included |
| `max_total_chars` | 4 000 000 | all strings and keys together |
| integer size | 100 digits | fixed |

Limits are checked while reading. A container whose length alone would exceed the remaining node budget is refused before its items are snapshotted, and each string is checked as it is reached. This bounds the work and the copy to roughly the limits. It is not a guarantee against every resource problem: the caller's own data is already in memory, and a string is measured after it exists. The limits are keyword arguments of `freeze_args`.

**Unpaired surrogates are rejected.** A Python string can hold half of a surrogate pair. With `ensure_ascii` the emoji U+1F600 and the two lone surrogates `chr(0xd83d) + chr(0xde00)` would give the same text `\ud83d\ude00`, so two different values would share one canonical JSON and one digest. Strings and keys must therefore be valid Unicode text (they must encode to UTF-8), and the emoji itself is accepted.

### What you get

`FrozenArgs` is a read-only mapping: `[]`, `get`, `in`, iteration, `len` and `==` with a dict all work. Nested dicts are read-only `FrozenMap` objects and lists are tuples. A list in a dict compares equal to a tuple of the same items. Whole-structure `==` is strict about scalar types, so that it agrees with the digest: `freeze_args({"a": 1}) != freeze_args({"a": 1.0})`, `0.0` and `-0.0` differ, and `True` differs from `1`. Reading a single value is plain Python: `frozen["flag"] == 1` is `True` when the stored value is the bool `True`. **The digest is the identity of the arguments**; compare digests, not individual values.

- `frozen.thaw()` returns a fresh plain deep copy (dicts and lists) that a tool may change freely.
- `frozen.canonical_json()` returns sorted keys, no whitespace, ASCII only (`ensure_ascii`).
- `frozen.sha256()` returns the plain SHA-256 of that text, in hex. A keyed digest for audit records comes later from the redaction module.
- `repr` shows only the number of keys and the start of the digest, so values do not end up in logs.
- It cannot be pickled or built directly, and `copy.copy` and `copy.deepcopy` return the same object.

The canonical text and the digest are computed once at freeze time from the frozen copy. Looking up a key that is not a `str` (for example `frozen.get([1])`) returns the default, or raises `KeyError` for `[]`; it never hashes the foreign object.

Honest limits of the immutability: it is deep and stops mutation through the normal interface, but Python cannot stop code that reaches into private attributes or uses `object.__setattr__`. The registry's private dicts, `FrozenMap._d`, the owner table and the module-private tokens are all reachable by code in the same process. If someone changes them, for example by editing `_d` in place, the stored canonical text and digest go stale and no longer describe the data. The identity rule for internal tools has the same limit: code that can build a `ToolRegistry`, call the private `_build_spec`, or edit the registry's internal mapping can make anything internal. Copying a spec returns the same object, and pickling or rebuilding one from state is refused, so a copy cannot skip validation. This layer guards against mistakes and careless code, not against hostile code running in the same process.

## Still to come

This section is extended by the later stories.

- Decision stages: policy hooks, approval, guards, the deny-wins fold (#60).
- The run stage: timeouts, cancellation, concurrency, one audit record per call.
- Wiring into `LLMAgent`, the executor and the orchestrator, so that undeclared tools are denied and specs are enforced (#63).
- Redacted argument previews and the keyed argument digest for audit records.
- The `allow_logged` compatibility mode for undeclared tools.
- Spill handling and the internal `read_spilled` tool.
- The pipeline must call `canonical_scope` (`aria/permissions/scope.py`) on every resolved scope, because this layer only checks that the scope is non-empty and at most 4096 characters.
- Model output must be parsed with an `object_pairs_hook` that rejects duplicate keys, before `freeze_args` is called. A plain `json.loads` silently keeps the last duplicate, so the approved text and the executed call could differ.
- `scope_from` contract for the pipeline: resolve each `Need` exactly once per call, with the `FrozenArgs` of that call, and pass that same string through policy, approval, the commit check and the audit record. Never call the callable a second time. Callables must be pure, fast and deterministic; the pipeline decides how to bound them.
- `internal` is not an exemption. The pipeline must check permissions for internal tools like any other.
- The pipeline must reject arguments that the tool does not declare as parameters (unless the tool takes `**kwargs`). Registration only checks that a string `scope_from` names a real parameter; it cannot know which argument the tool body actually uses.
