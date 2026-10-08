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
        assert n.resolve_scope({}) == "*"

    def test_fixed_scope(self):
        assert Need(C.FILE_READ, scope="/tmp/x").resolve_scope({"a": 1}) == "/tmp/x"

    def test_scope_from_argument_name(self):
        n = Need(C.WEB_BROWSE, scope_from="url")
        assert n.resolve_scope({"url": "https://example.com/a"}) == "https://example.com/a"

    def test_scope_from_frozen_arguments(self):
        n = Need(C.WEB_BROWSE, scope_from="url")
        assert n.resolve_scope(freeze_args({"url": "https://example.com"})) == "https://example.com"

    def test_scope_from_callable(self):
        n = Need(C.FILE_READ, scope_from=lambda a: a["dir"] + "/" + a["file"])
        assert n.resolve_scope({"dir": "/d", "file": "f"}) == "/d/f"

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
            Need(C.FILE_READ, scope_from="path").resolve_scope({})

    @pytest.mark.parametrize("value", [None, 3, 1.5, b"/x", ["/x"], {"a": 1}, True, ("a",)])
    def test_non_string_value(self, value):
        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope_from="path").resolve_scope({"path": value})

    @pytest.mark.parametrize("value", ["", "   ", "\n"])
    def test_empty_value(self, value):
        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope_from="path").resolve_scope({"path": value})

    def test_overlong_value(self):
        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope_from="path").resolve_scope({"path": "x" * 5000})

    def test_str_subclass_value_rejected(self):
        class S(str):
            pass

        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope_from="path").resolve_scope({"path": S("/x")})

    def test_callable_raises(self):
        def boom(a):
            raise RuntimeError("secret-token-123")

        with pytest.raises(ToolSpecError) as info:
            Need(C.FILE_READ, scope_from=boom).resolve_scope({"a": 1})
        # The callable's message is not copied; only its type.
        assert "secret-token-123" not in str(info.value)
        assert "RuntimeError" in str(info.value)

    @pytest.mark.parametrize("result", [None, 5, b"x", ["x"], ""])
    def test_callable_returns_bad_value(self, result):
        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope_from=lambda a: result).resolve_scope({})

    def test_callable_returning_coroutine_is_closed(self):
        async def co():
            return "x"

        n = Need(C.FILE_READ, scope_from=lambda a: co())
        with pytest.raises(ToolSpecError):
            n.resolve_scope({})

    def test_arguments_must_be_a_mapping(self):
        with pytest.raises(ToolSpecError):
            Need(C.FILE_READ, scope_from="path").resolve_scope(["path"])  # type: ignore[arg-type]

    def test_error_never_contains_argument_values(self):
        with pytest.raises(ToolSpecError) as info:
            Need(C.FILE_READ, scope_from="path").resolve_scope({"path": 12345678})
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
            Need(C.FILE_READ, scope_from=boom).resolve_scope({})


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

    def test_internal_may_have_no_needs(self):
        s = ToolSpec(name="t", needs=(), internal=True)
        assert s.needs == () and s.risk == "low"

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

    def test_internal_without_needs(self):
        @tool(internal=True)
        def f():
            pass

        assert f.__aria_tool_spec__.internal and f.__aria_tool_spec__.needs == ()

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
            tool(internal=True, scope_from="p")

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
