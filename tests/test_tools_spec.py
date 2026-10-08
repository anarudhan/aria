"""Tests for Need, ToolSpec and the @tool decorator (issue #59)."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import math
from fractions import Fraction

import pytest

from aria.permissions import RISK_LEVEL, Capability
from aria.tools import (
    RISK_ORDER,
    SPEC_ATTRIBUTE,
    FrozenArgs,
    Need,
    ToolSpec,
    ToolSpecError,
    freeze_args,
    tool,
)

C = Capability


def make(**overrides):
    base = dict(name="t", needs=(Need(C.FILE_READ),))
    base.update(overrides)
    return ToolSpec(**base)


# ─── Need ────────────────────────────────────────────────────────────────────


class TestNew:
    def test_defaults_to_wildcard(self):
        n = Need(C.FILE_READ)
        assert n.scope == "*" and n.scope_from is None
        assert n.resolve_scope(freeze_args({})) == "*"

    def test_fixed_scope(self):
        assert Need(C.FILE_READ, scope="/tmp/x").resolve_scope(freeze_args({"a": 1})) == "/tmp/x"

    def test_scope_from_argument_name(self):
        n = Need(C.WEB_BROWSE, scope_from="url")
        assert n.resolve_scope(freeze_args({"url": "https://example.com/a"})) == "https://example.com/a"

    def test_scope_from_frozen_arguments(self):
        n = Need(C.WEB_BROWSE, scope_from="url")
        assert n.resolve_scope(freeze_args({"url": "https://example.com"})) == "https://example.com"

    def test_scope_from_callable(self):
        n = Need(C.FILE_READ, scope_from=lambda a: a["dir"] + "/" + a["file"])
        assert n.resolve_scope(freeze_args({"dir": "/d", "file": "f"})) == "/d/f"

    def test_both_fixed_and_scope_from_rejected(self):
        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope="/tmp", scope_from="path")

    @pytest.mark.parametrize("bad", [Capability.FILE_READ.value, None, 3, "file_read", object()])
    def test_capability_must_be_member(self, bad):
        with pytest.raises(ToolSpecError):
            Need(bad)

    @pytest.mark.parametrize("bad", ["", "   ", None, 5, b"x", "x" * 5000])
    def test_bad_fixed_scope(self, bad):
        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope=bad)

    @pytest.mark.parametrize("bad", ["", "  ", 5, b"x", object(), ["a"], "x" * 300])
    def test_bad_scope_from(self, bad):
        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope_from=bad)

    def test_async_scope_from_rejected(self):
        async def f(a):
            return "x"

        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope_from=f)

    def test_str_subclass_scope_rejected(self):
        class S(str):
            pass

        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope=S("/x"))
        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope_from=S("path"))

    def test_immutable(self):
        n = Need(C.FILE_READ)
        with pytest.raises(dataclasses.FrozenInstanceError):
            n.scope = "/x"  # type: ignore[misc]

    def test_cannot_subclass(self):
        with pytest.raises(TypeError):

            class Sub(Need):
                pass


class TestResolveScopeFailsClosed:
    def test_missing_argument(self):
        with pytest.raises(ToolSpecError, match="missing"):
            Need(C.FILE_READ, scope_from="path").resolve_scope(freeze_args({}))

    @pytest.mark.parametrize("value", [None, 3, 1.5, ["/x"], {"a": 1}, True, ("a",)])
    def test_non_string_value(self, value):
        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope_from="path").resolve_scope(freeze_args({"path": value}))

    @pytest.mark.parametrize("value", ["", "   ", "\n"])
    def test_empty_value(self, value):
        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope_from="path").resolve_scope(freeze_args({"path": value}))

    def test_overlong_value(self):
        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope_from="path").resolve_scope(freeze_args({"path": "x" * 5000}))

    def test_str_subclass_value_rejected(self):
        class S(str):
            pass

        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope_from=lambda a: S("/x")).resolve_scope(freeze_args({}))

    def test_callable_raises(self):
        def boom(a):
            raise RuntimeError("secret-token-123")

        with pytest.raises(ToolSpecError) as info:
            Need(C.FILE_READ, scope_from=boom).resolve_scope(freeze_args({"a": 1}))
        # The callable's message is not copied; only its type.
        assert "secret-token-123" not in str(info.value)
        assert "RuntimeError" in str(info.value)

    @pytest.mark.parametrize("result", [None, 5, b"x", ["x"], ""])
    def test_callable_returns_bad_value(self, result):
        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope_from=lambda a: result).resolve_scope(freeze_args({}))

    def test_callable_returning_coroutine_is_closed(self):
        async def co():
            return "x"

        n = Need(C.FILE_READ, scope_from=lambda a: co())
        with pytest.raises(ToolSpecError):
            n.resolve_scope(freeze_args({}))

    @pytest.mark.parametrize("bad", [{"path": "/x"}, ["path"], None, "path", (("path", "/x"),)])
    def test_arguments_must_be_frozen_args(self, bad):
        # A plain dict could be changed by a scope_from callable, so it is refused,
        # and the callable is never run on it.
        calls = []
        by_name = Need(C.FILE_READ, scope_from="path")
        by_call = Need(C.FILE_READ, scope_from=lambda a: calls.append(a) or "/x")
        with pytest.raises(ToolSpecError, match="FrozenArgs"):
            by_name.resolve_scope(bad)
        with pytest.raises(ToolSpecError, match="FrozenArgs"):
            by_call.resolve_scope(bad)
        assert calls == []

    def test_mapping_lookalikes_are_refused(self):
        from collections.abc import Mapping

        class Fake(Mapping):
            def __getitem__(self, k):
                return "/x"

            def __iter__(self):
                return iter(("path",))

            def __len__(self):
                return 1

        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope_from="path").resolve_scope(Fake())  # type: ignore[arg-type]

    def test_frozen_map_is_accepted(self):
        frozen = freeze_args({"n": {"path": "/x"}})
        assert Need(C.FILE_READ, scope_from="path").resolve_scope(frozen["n"]) == "/x"

    def test_callable_cannot_change_the_frozen_arguments(self):
        frozen = freeze_args({"path": "/x", "l": [1]})

        def evil(a):
            a["path"] = "/etc"  # raises TypeError on a FrozenArgs

        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope_from=evil).resolve_scope(frozen)
        assert frozen["path"] == "/x"

    def test_fixed_scope_does_not_read_the_arguments(self):
        assert Need(C.FILE_READ, scope="/d").resolve_scope(freeze_args({})) == "/d"

    def test_error_never_contains_argument_values(self):
        with pytest.raises(ToolSpecError) as info:
            Need(C.FILE_READ, scope_from="path").resolve_scope(freeze_args({"path": 12345678}))
        assert "12345678" not in str(info.value)

    def test_hostile_mapping_getitem(self):
        from collections.abc import Mapping

        class Hostile(Mapping):
            def __getitem__(self, k):
                raise ZeroDivisionError("x")

            def __iter__(self):
                return iter(())

            def __len__(self):
                return 0

        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope_from="path").resolve_scope(Hostile())

    def test_keyboard_interrupt_is_not_swallowed(self):
        def boom(a):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            Need(C.FILE_READ, scope_from=boom).resolve_scope(freeze_args({}))


# ─── ToolSpec ────────────────────────────────────────────────────────────────


class TestToolSpec:
    def test_defaults(self):
        s = make()
        assert s.timeout_s == 30.0 and s.concurrency_safe is False
        assert s.approval == "auto" and s.internal is False and s.description == ""

    def test_immutable(self):
        s = make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.timeout_s = 1.0  # type: ignore[misc]

    def test_cannot_subclass(self):
        with pytest.raises(TypeError):

            class Sub(ToolSpec):
                pass

    @pytest.mark.parametrize("name", ["a", "_x", "Tool-1.v2", "a" * 64, "web.fetch", "A_b-c.d"])
    def test_good_names(self, name):
        assert make(name=name).name == name

    @pytest.mark.parametrize(
        "name",
        ["", "1a", "-a", ".a", "a b", "a/b", "a" * 65, "é", "a\n", "a\x00", "tool!", " a", "a ", None, 5, b"a"],
    )
    def test_bad_names(self, name):
        with pytest.raises(ToolSpecError):
            make(name=name)

    def test_trailing_newline_name_rejected(self):
        # "$" in a regex would accept this with re.match; fullmatch must not.
        with pytest.raises(ToolSpecError):
            make(name="ok\n")

    @pytest.mark.parametrize("name", ["abc\n", "abc\r\n", "a\n\n", "\nabc"])
    def test_exported_pattern_rejects_trailing_newline_with_match_and_fullmatch(self, name):
        import re

        from aria.tools import TOOL_NAME_PATTERN

        assert re.match(TOOL_NAME_PATTERN, name) is None
        assert re.fullmatch(TOOL_NAME_PATTERN, name) is None
        assert re.search(TOOL_NAME_PATTERN, name) is None

    @pytest.mark.parametrize("name", ["abc", "_", "a.b-c", "A" * 64])
    def test_exported_pattern_accepts_good_names(self, name):
        import re

        from aria.tools import TOOL_NAME_PATTERN

        assert re.match(TOOL_NAME_PATTERN, name)

    def test_name_str_subclass_rejected(self):
        class S(str):
            pass

        with pytest.raises(ToolSpecError):
            make(name=S("ok"))

    @pytest.mark.parametrize("needs", [[Need(C.FILE_READ)], None, "x", (C.FILE_READ,), (None,), {Need(C.FILE_READ)}])
    def test_needs_must_be_tuple_of_need(self, needs):
        with pytest.raises(ToolSpecError):
            make(needs=needs)

    def test_needs_subclass_instance_rejected(self):
        with pytest.raises(ToolSpecError):
            make(needs=(object(),))

    def test_too_many_needs(self):
        with pytest.raises(ToolSpecError):
            make(needs=tuple(Need(C.FILE_READ) for _ in range(33)))

    def test_non_internal_needs_at_least_one(self):
        with pytest.raises(ToolSpecError, match="at least one"):
            ToolSpec(name="t", needs=())

    def test_user_code_cannot_declare_internal(self):
        with pytest.raises(ToolSpecError, match="internal"):
            ToolSpec(name="t", needs=(), internal=True)
        with pytest.raises(ToolSpecError, match="internal"):
            ToolSpec(name="t", needs=(Need(C.FILE_READ),), internal=True)

    def test_replace_cannot_make_a_spec_internal(self):
        with pytest.raises(ToolSpecError):
            dataclasses.replace(make(), internal=True)

    def test_internal_gate_is_closed_after_a_registry_style_build(self):
        from aria.tools.spec import _build_spec

        built = _build_spec(True, name="t", needs=(Need(C.FILE_READ),))
        assert built.internal is True
        with pytest.raises(ToolSpecError):
            ToolSpec(name="t", needs=(Need(C.FILE_READ),), internal=True)
        # even when the build itself fails the gate closes again
        with pytest.raises(ToolSpecError):
            _build_spec(True, name="bad name", needs=())
        with pytest.raises(ToolSpecError):
            ToolSpec(name="t", needs=(Need(C.FILE_READ),), internal=True)

    @pytest.mark.parametrize("t", [0, 0.0, -1, -0.5, 300.01, 301, 10**9, math.nan, math.inf, -math.inf, 10**400])
    def test_bad_timeouts_numeric(self, t):
        with pytest.raises(ToolSpecError):
            make(timeout_s=t)

    @pytest.mark.parametrize("t", [True, False, "5", None, b"5", Fraction(1, 2), [1], 1 + 2j])
    def test_bad_timeouts_types(self, t):
        with pytest.raises(ToolSpecError):
            make(timeout_s=t)

    @pytest.mark.parametrize("t", [0.001, 1, 30, 299.9, 300, 300.0])
    def test_good_timeouts(self, t):
        s = make(timeout_s=t)
        assert type(s.timeout_s) is float and s.timeout_s == float(t)

    @pytest.mark.parametrize("v", [1, 0, "yes", None, 1.0])
    def test_concurrency_safe_must_be_bool(self, v):
        with pytest.raises(ToolSpecError):
            make(concurrency_safe=v)

    @pytest.mark.parametrize("v", [1, 0, "yes", None, 1.0])
    def test_internal_must_be_bool(self, v):
        with pytest.raises(ToolSpecError):
            make(internal=v)

    @pytest.mark.parametrize("v", ["", "never", "Auto", "ALWAYS", "ask", None, 1, True, b"auto", "auto "])
    def test_bad_approval(self, v):
        with pytest.raises(ToolSpecError):
            make(approval=v)

    @pytest.mark.parametrize("v", ["auto", "always"])
    def test_good_approval(self, v):
        assert make(approval=v).approval == v

    @pytest.mark.parametrize("v", [None, 5, b"x", ["x"], "x" * 2001])
    def test_bad_description(self, v):
        with pytest.raises(ToolSpecError):
            make(description=v)

    def test_tool_spec_error_is_value_error(self):
        assert issubclass(ToolSpecError, ValueError)


class TestRisk:
    def test_every_risk_value_has_a_rank(self):
        assert set(RISK_LEVEL.values()) <= set(RISK_ORDER)

    def test_every_capability_maps_to_its_own_level(self):
        for cap in Capability:
            assert make(needs=(Need(cap),)).risk == RISK_LEVEL[cap]

    def test_highest_wins(self):
        s = make(needs=(Need(C.FILE_READ), Need(C.EMAIL_SEND), Need(C.FILE_WRITE)))
        assert s.risk == "high"
        s = make(needs=(Need(C.EMAIL_SEND), Need(C.SELF_MODIFY), Need(C.WEB_BROWSE)))
        assert s.risk == "critical"
        s = make(needs=(Need(C.FILE_READ), Need(C.FILE_WRITE)))
        assert s.risk == "medium"

    def test_unknown_risk_counts_as_critical(self, monkeypatch):
        patched = dict(RISK_LEVEL)
        del patched[C.FILE_READ]
        monkeypatch.setattr("aria.tools.spec.RISK_LEVEL", patched)
        assert make(needs=(Need(C.FILE_READ),)).risk == "critical"
        patched[C.FILE_READ] = "bizarre"
        assert make(needs=(Need(C.FILE_READ),)).risk == "critical"


# ─── @tool ───────────────────────────────────────────────────────────────────


class TestDecorator:
    def test_returns_same_function_and_attaches_spec(self):
        def f(x):
            return x + 1

        g = tool(C.FILE_READ)(f)
        assert g is f
        assert f(1) == 2
        spec = getattr(f, SPEC_ATTRIBUTE)
        assert type(spec) is ToolSpec and spec.name == "f"
        assert spec.needs == (Need(C.FILE_READ),)

    def test_attribute_name(self):
        assert SPEC_ATTRIBUTE == "__aria_tool_spec__"

    def test_signature_unchanged(self):
        def f(a: int, b: str = "x") -> str:
            return b * a

        before = inspect.signature(f)
        tool(C.MEMORY_WRITE)(f)
        assert inspect.signature(f) == before

    def test_async_function(self):
        @tool(C.WEB_BROWSE, scope_from="url")
        async def fetch(url):
            return url

        assert inspect.iscoroutinefunction(fetch)
        assert asyncio.run(fetch("u")) == "u"
        assert fetch.__aria_tool_spec__.needs[0].scope_from == "url"

    def test_methods_in_classes(self):
        class Agent:
            @tool(C.FILE_READ, name="agent.read")
            def read(self, path):
                return path

            @tool(C.NETWORK)
            async def ping(self):
                return "pong"

        a = Agent()
        assert a.read("p") == "p"
        assert asyncio.run(a.ping()) == "pong"
        assert Agent.read.__aria_tool_spec__.name == "agent.read"
        # A bound method exposes the function's attribute.
        assert a.read.__aria_tool_spec__.name == "agent.read"
        assert a.ping.__aria_tool_spec__.name == "ping"

    def test_staticmethod_below(self):
        class A:
            @staticmethod
            @tool(C.FILE_READ)
            def s(x):
                return x

        assert A.s("v") == "v"
        assert A.s.__aria_tool_spec__.name == "s"

    def test_name_override(self):
        @tool(C.FILE_READ, name="custom-name")
        def f():
            pass

        assert f.__aria_tool_spec__.name == "custom-name"

    def test_options_flow_to_spec(self):
        @tool(
            C.FILE_WRITE,
            timeout_s=5,
            concurrency_safe=True,
            approval="always",
            description="writes a file",
        )
        def f():
            pass

        s = f.__aria_tool_spec__
        assert (s.timeout_s, s.concurrency_safe, s.approval, s.description) == (
            5.0,
            True,
            "always",
            "writes a file",
        )

    def test_internal_cannot_be_declared_with_the_decorator(self):
        with pytest.raises(ToolSpecError, match="internal"):
            tool(internal=True)
        with pytest.raises(ToolSpecError, match="internal"):
            tool(C.FILE_READ, internal=True)

    def test_non_internal_without_needs_rejected_at_decoration(self):
        with pytest.raises(ToolSpecError):
            tool()

    def test_scope_applies_to_bare_capabilities(self):
        @tool(C.FILE_READ, C.FILE_WRITE, scope="/data")
        def f():
            pass

        assert [n.scope for n in f.__aria_tool_spec__.needs] == ["/data", "/data"]

    def test_scope_from_applies_and_need_objects_kept(self):
        own = Need(C.NETWORK, scope="https://example.com")

        @tool(C.FILE_READ, own, scope_from="path")
        def f(path):
            pass

        needs = f.__aria_tool_spec__.needs
        assert needs[0] == Need(C.FILE_READ, scope_from="path")
        assert needs[1] is own

    def test_scope_without_bare_capability_rejected(self):
        with pytest.raises(ToolSpecError):
            tool(Need(C.FILE_READ), scope="/x")
        with pytest.raises(ToolSpecError):
            tool(scope_from="p")

    def test_both_scope_and_scope_from_rejected(self):
        with pytest.raises(ToolSpecError):
            tool(C.FILE_READ, scope="/x", scope_from="p")

    @pytest.mark.parametrize("bad", ["file_read", 3, None, object(), [C.FILE_READ]])
    def test_bad_positional_rejected(self, bad):
        with pytest.raises(ToolSpecError):
            tool(bad)

    def test_bare_decorator_gives_clear_error(self):
        with pytest.raises(ToolSpecError, match="parentheses"):

            @tool
            def f():
                pass

    def test_bad_values_fail_at_decoration_time(self):
        with pytest.raises(ToolSpecError):
            tool(C.FILE_READ, timeout_s=0)
        with pytest.raises(ToolSpecError):
            tool(C.FILE_READ, approval="maybe")
        with pytest.raises(ToolSpecError):
            tool(C.FILE_READ, name="bad name")

    def test_lambda_needs_name(self):
        with pytest.raises(ToolSpecError):
            tool(C.FILE_READ)(lambda: 1)
        f = tool(C.FILE_READ, name="lam")(lambda: 1)
        assert f.__aria_tool_spec__.name == "lam"

    def test_not_callable_rejected(self):
        with pytest.raises(ToolSpecError):
            tool(C.FILE_READ)(42)  # type: ignore[type-var]

    def test_bound_method_rejected_with_clear_error(self):
        class A:
            def m(self):
                pass

        with pytest.raises(ToolSpecError, match="Decorate the function"):
            tool(C.FILE_READ)(A().m)

    def test_double_decoration_rejected(self):
        def f():
            pass

        tool(C.FILE_READ)(f)
        with pytest.raises(ToolSpecError, match="twice"):
            tool(C.FILE_WRITE)(f)

    def test_spec_on_each_function_is_independent(self):
        deco = tool(C.FILE_READ, timeout_s=7)

        def a():
            pass

        def b():
            pass

        deco(a)
        deco(b)
        assert a.__aria_tool_spec__.name == "a" and b.__aria_tool_spec__.name == "b"

    def test_frozen_args_feed_resolve_scope_end_to_end(self):
        @tool(C.FILE_READ, scope_from="path")
        def read(path):
            pass

        frozen = freeze_args({"path": "/data/x.txt"})
        assert isinstance(frozen, FrozenArgs)
        need = read.__aria_tool_spec__.needs[0]
        assert need.resolve_scope(frozen) == "/data/x.txt"


class TestDecoratorOwnership:
    def test_wraps_wrapper_is_not_an_owner_but_can_be_decorated(self):
        import functools

        @tool(C.FILE_READ)
        def inner():
            pass

        @functools.wraps(inner)
        def wrapper(*a, **k):
            return inner(*a, **k)

        # wraps copied the attribute, but the wrapper does not own it
        assert wrapper.__aria_tool_spec__ is inner.__aria_tool_spec__
        # applying @tool to the wrapper works (it is not "already declared")
        tool(C.NETWORK, name="wrapped")(wrapper)
        assert wrapper.__aria_tool_spec__.name == "wrapped"
        assert wrapper.__aria_tool_spec__.needs[0].capability is C.NETWORK
        assert inner.__aria_tool_spec__.name == "inner"  # the original is untouched

    def test_double_decoration_of_the_same_object_still_rejected(self):
        def f():
            pass

        tool(C.FILE_READ)(f)
        with pytest.raises(ToolSpecError, match="twice"):
            tool(C.FILE_READ)(f)

    def test_subclass_can_be_decorated(self):
        @tool(C.FILE_READ, name="base")
        class Base:
            def __call__(self):
                pass

        @tool(C.NETWORK, name="child")
        class Child(Base):
            pass

        assert Base.__aria_tool_spec__.name == "base" and Child.__aria_tool_spec__.name == "child"

    @pytest.mark.parametrize("target", ["staticmethod", "classmethod"])
    def test_decorating_a_descriptor_object_is_refused_clearly(self, target):
        def f(*a):
            pass

        wrapper = staticmethod(f) if target == "staticmethod" else classmethod(f)
        with pytest.raises(ToolSpecError, match="below"):
            tool(C.FILE_READ)(wrapper)  # type: ignore[type-var]

    def test_owner_table_forgets_dead_objects(self):
        import gc

        from aria.tools import decorator

        def make_one():
            def f():
                pass

            tool(C.FILE_READ)(f)
            return id(f)

        key = make_one()
        gc.collect()
        assert key not in decorator._OWNERS


class TestInternalCannotBeForged:
    def test_gc_hook_during_a_registry_build_cannot_forge_an_internal_spec(self, monkeypatch):
        import gc

        import aria.tools.spec as spec_module
        from aria.tools import ToolRegistry

        forged = []
        attempts = []

        def hook(phase, info):
            if phase != "start":
                return
            attempts.append(1)
            for needs in ((), (Need(C.FILE_READ),)):
                try:
                    forged.append(ToolSpec(name="forged", needs=needs, internal=True))
                except ToolSpecError:
                    pass

        real_validate = spec_module.validate_tool_name

        def validate_and_collect(name):
            gc.collect()  # runs the hook while the registry is building a spec
            return real_validate(name)

        def real():
            pass

        monkeypatch.setattr(spec_module, "validate_tool_name", validate_and_collect)
        gc.callbacks.append(hook)
        try:
            reg = ToolRegistry(internal_tools={"real": real})
            reg.register(real)
            reg.register(plain_fn, ToolSpec(name="other", needs=(Need(C.FILE_READ),)))
        finally:
            gc.callbacks.remove(hook)
        assert attempts, "the gc hook never ran, so the test proves nothing"
        assert forged == []
        assert reg.spec("real").internal is True and reg.spec("other").internal is False

    def test_there_is_no_module_state_that_enables_internal(self):
        import aria.tools.spec as spec_module

        assert not hasattr(spec_module, "_internal_gate")
        # a validate call without the permission fails whatever else is going on
        spec = spec_module._build_spec(True, name="ok", needs=())
        with pytest.raises(ToolSpecError):
            spec_module._validate_spec(spec, allow_internal=False)


def plain_fn():
    return 1


class TestCopyAndPickleCannotBypassValidation:
    def test_setstate_cannot_forge_a_spec(self):
        import copy

        spec = make()
        clone = copy.copy(spec)
        with pytest.raises(TypeError):
            clone.__setstate__(("evil", (), 30.0, False, "auto", True, ""))
        with pytest.raises(TypeError):
            spec.__setstate__(("evil", (), 30.0, False, "auto", True, ""))
        assert spec.internal is False and spec.name == "t"

    def test_setstate_cannot_forge_a_need(self):
        import copy

        need = Need(C.FILE_READ)
        with pytest.raises(TypeError):
            copy.copy(need).__setstate__(("not a capability", "", None))
        with pytest.raises(TypeError):
            need.__setstate__((C.FILE_READ, "/x", None))

    @pytest.mark.parametrize(
        "obj", [make(), Need(C.NETWORK, scope="https://x.example")], ids=["spec", "need"]
    )
    def test_pickle_is_refused(self, obj):
        import pickle

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with pytest.raises(TypeError):
                pickle.dumps(obj, protocol)

    @pytest.mark.parametrize(
        "obj", [make(), Need(C.NETWORK, scope="https://x.example")], ids=["spec", "need"]
    )
    def test_copy_and_deepcopy_return_the_same_object(self, obj):
        import copy

        assert copy.copy(obj) is obj
        assert copy.deepcopy(obj) is obj
        assert copy.deepcopy({"k": [obj]})["k"][0] is obj

    def test_replace_still_validates(self):
        with pytest.raises(ToolSpecError):
            dataclasses.replace(make(), timeout_s=-1)
        assert dataclasses.replace(make(), timeout_s=5).timeout_s == 5.0

    def test_raw_reduce_protocol_is_closed(self):
        spec = make()
        with pytest.raises(TypeError):
            spec.__reduce__()
        with pytest.raises(TypeError):
            spec.__reduce_ex__(2)
        with pytest.raises(TypeError):
            spec.__getstate__()
