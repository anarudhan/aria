"""Tests for ToolRegistry (issue #59)."""

from __future__ import annotations

import functools
import json

import pytest

from aria.permissions import Capability
from aria.tools import SPEC_ATTRIBUTE, Need, ToolRegistry, ToolSpec, ToolSpecError, tool

C = Capability


def plain(x=0):
    return x


class TestRegisterDeclared:
    def test_decorated_tool(self):
        @tool(C.FILE_READ, timeout_s=5)
        def read(path):
            return path

        reg = ToolRegistry()
        stored = reg.register(read)
        assert reg.get("read") is read
        assert reg.spec("read") == stored == read.__aria_tool_spec__
        assert reg.names() == ["read"]
        assert reg.undeclared() == []
        assert reg.is_declared("read")
        assert "read" in reg and len(reg) == 1

    def test_explicit_spec_wins_over_attribute(self):
        @tool(C.FILE_READ, name="decorated")
        def f():
            pass

        explicit = ToolSpec(name="explicit", needs=(Need(C.EMAIL_SEND),))
        reg = ToolRegistry()
        reg.register(f, explicit)
        assert reg.names() == ["explicit"]
        assert reg.spec("explicit").needs[0].capability is C.EMAIL_SEND
        assert reg.get("explicit") is f

    def test_explicit_spec_for_third_party_callable(self):
        reg = ToolRegistry()
        spec = ToolSpec(name="partial_tool", needs=(Need(C.NETWORK),))
        reg.register(functools.partial(plain, 3), spec)
        assert reg.get("partial_tool")() == 3

    def test_name_argument_renames_and_syncs_spec(self):
        reg = ToolRegistry()
        spec = ToolSpec(name="orig", needs=(Need(C.NETWORK),), timeout_s=9)
        stored = reg.register(plain, spec, name="renamed")
        assert stored.name == "renamed" and reg.spec("renamed").name == "renamed"
        assert reg.spec("renamed").timeout_s == 9.0
        assert "orig" not in reg
        assert spec.name == "orig"  # the caller's object is untouched

    def test_name_argument_on_decorated_function(self):
        @tool(C.FILE_READ)
        def f():
            pass

        reg = ToolRegistry()
        reg.register(f, name="other")
        assert reg.spec("other").name == "other"

    def test_bound_method(self):
        class A:
            @tool(C.MEMORY_WRITE)
            def remember(self, k):
                return k

        reg = ToolRegistry()
        reg.register(A().remember)
        assert reg.spec("remember").needs[0].capability is C.MEMORY_WRITE

    def test_internal_tool(self):
        reg = ToolRegistry()
        reg.register(plain, ToolSpec(name="read_spilled", needs=(), internal=True))
        assert reg.spec("read_spilled").internal is True
        assert reg.undeclared() == []


class TestUndeclared:
    def test_no_spec_means_undeclared(self):
        reg = ToolRegistry()
        assert reg.register(plain) is None
        assert reg.undeclared() == ["plain"]
        assert reg.spec("plain") is None
        assert reg.get("plain") is plain  # a later stage can still find it and deny it
        assert not reg.is_declared("plain")
        assert "plain" in reg

    def test_undeclared_with_name(self):
        reg = ToolRegistry()
        reg.register(lambda: 1, name="adhoc")
        assert reg.undeclared() == ["adhoc"]

    def test_lambda_without_name_rejected(self):
        with pytest.raises(ToolSpecError):
            ToolRegistry().register(lambda: 1)

    def test_partial_without_name_rejected(self):
        with pytest.raises(ToolSpecError):
            ToolRegistry().register(functools.partial(plain, 1))

    @pytest.mark.parametrize("bad", ["", "bad name", "x" * 65, "1x", "a\n"])
    def test_undeclared_names_are_validated_too(self, bad):
        with pytest.raises(ToolSpecError):
            ToolRegistry().register(plain, name=bad)

    def test_unknown_name(self):
        reg = ToolRegistry()
        assert reg.get("nope") is None and reg.spec("nope") is None
        assert "nope" not in reg and not reg.is_declared("nope")

    def test_non_str_lookup_returns_none_without_hashing(self):
        class Hostile:
            def __hash__(self):
                raise RuntimeError("hash ran")

            def __eq__(self, other):
                raise RuntimeError("eq ran")

        reg = ToolRegistry()
        reg.register(plain)
        h = Hostile()
        assert reg.get(h) is None and reg.spec(h) is None
        assert h not in reg and not reg.is_declared(h)
        assert reg.get(None) is None and reg.get(["plain"]) is None

    def test_non_str_name_argument(self):
        with pytest.raises(ToolSpecError):
            ToolRegistry().register(plain, name=5)  # type: ignore[arg-type]


class TestRejections:
    @pytest.mark.parametrize("bad", [None, 5, "tool", [], {}, object()])
    def test_non_callable_rejected(self, bad):
        with pytest.raises(ToolSpecError):
            ToolRegistry().register(bad, name="x")

    def test_duplicate_names_rejected(self):
        reg = ToolRegistry()
        reg.register(plain)
        with pytest.raises(ToolSpecError, match="already registered"):
            reg.register(plain)
        with pytest.raises(ToolSpecError):
            reg.register(lambda: 2, name="plain")
        assert reg.get("plain") is plain

    def test_duplicate_between_declared_and_undeclared(self):
        reg = ToolRegistry()
        reg.register(plain)
        with pytest.raises(ToolSpecError):
            reg.register(plain, ToolSpec(name="plain", needs=(Need(C.FILE_READ),)))
        assert reg.undeclared() == ["plain"]  # the first registration stands

    @pytest.mark.parametrize("bad", ["spec", 5, {"name": "x"}, object()])
    def test_spec_must_be_toolspec(self, bad):
        with pytest.raises(ToolSpecError):
            ToolRegistry().register(plain, bad)

    def test_malformed_attribute_refuses_instead_of_treating_as_undeclared(self):
        def f():
            pass

        setattr(f, SPEC_ATTRIBUTE, {"name": "f", "needs": []})
        reg = ToolRegistry()
        with pytest.raises(ToolSpecError, match="expected a ToolSpec"):
            reg.register(f)
        assert "f" not in reg

    def test_attribute_that_raises_on_read(self):
        class Weird:
            def __call__(self):
                pass

            def __getattr__(self, name):
                raise RuntimeError("boom")

        with pytest.raises(ToolSpecError):
            ToolRegistry().register(Weird(), name="weird")

    def test_tampered_spec_rejected_at_registration(self):
        spec = ToolSpec(name="t", needs=(Need(C.FILE_READ),))
        object.__setattr__(spec, "timeout_s", 99999.0)
        with pytest.raises(ToolSpecError):
            ToolRegistry().register(plain, spec)

        spec2 = ToolSpec(name="t", needs=(Need(C.FILE_READ),))
        object.__setattr__(spec2, "needs", [Need(C.FILE_READ)])
        with pytest.raises(ToolSpecError):
            ToolRegistry().register(plain, spec2)

        spec3 = ToolSpec(name="t", needs=(Need(C.FILE_READ),))
        object.__setattr__(spec3, "approval", "never")
        with pytest.raises(ToolSpecError):
            ToolRegistry().register(plain, spec3)

    def test_failed_registration_leaves_registry_unchanged(self):
        reg = ToolRegistry()
        with pytest.raises(ToolSpecError):
            reg.register(plain, "not a spec")  # type: ignore[arg-type]
        assert reg.names() == [] and len(reg) == 0


class TestCopySemantics:
    def test_changing_function_attribute_later_has_no_effect(self):
        @tool(C.FILE_READ, timeout_s=5)
        def f():
            pass

        reg = ToolRegistry()
        reg.register(f)
        before = reg.spec("f")
        setattr(f, SPEC_ATTRIBUTE, ToolSpec(name="f", needs=(Need(C.SELF_MODIFY),), timeout_s=300))
        assert reg.spec("f") == before
        assert reg.spec("f").needs[0].capability is C.FILE_READ
        delattr(f, SPEC_ATTRIBUTE)
        assert reg.spec("f") == before

    def test_mutating_original_spec_object_has_no_effect(self):
        spec = ToolSpec(name="f", needs=(Need(C.FILE_READ),), timeout_s=5)
        reg = ToolRegistry()
        reg.register(plain, spec, name="f")
        object.__setattr__(spec, "timeout_s", 250.0)
        object.__setattr__(spec, "approval", "always")
        object.__setattr__(spec, "needs", (Need(C.SELF_MODIFY),))
        got = reg.spec("f")
        assert got.timeout_s == 5.0 and got.approval == "auto"
        assert got.needs[0].capability is C.FILE_READ

    def test_registry_stores_a_distinct_copy(self):
        spec = ToolSpec(name="f", needs=(Need(C.FILE_READ),))
        reg = ToolRegistry()
        stored = reg.register(plain, spec, name="f")
        assert stored is not spec and stored == spec
        assert stored.needs[0] is not spec.needs[0]
        assert reg.spec("f") is not reg.spec("f")  # callers get copies, never the internal one

    def test_tampering_with_returned_spec_does_not_reach_the_registry(self):
        reg = ToolRegistry()
        stored = reg.register(plain, ToolSpec(name="f", needs=(Need(C.FILE_READ),)), name="f")
        object.__setattr__(stored, "timeout_s", 299.0)
        got = reg.spec("f")
        object.__setattr__(got, "approval", "always")
        assert reg.spec("f").timeout_s == 30.0 and reg.spec("f").approval == "auto"
        assert reg.describe()[0]["timeout_s"] == 30.0

    def test_registering_does_not_modify_function_or_spec(self):
        @tool(C.FILE_READ)
        def f():
            pass

        original = f.__aria_tool_spec__
        ToolRegistry().register(f)
        assert f.__aria_tool_spec__ is original


class TestDescribe:
    def test_json_safe_and_complete(self):
        reg = ToolRegistry()
        reg.register(
            plain,
            ToolSpec(
                name="mail",
                needs=(Need(C.EMAIL_SEND, scope_from=lambda a: "x"), Need(C.FILE_READ)),
                timeout_s=12,
                concurrency_safe=True,
                approval="always",
                description="sends mail",
            ),
        )
        reg.register(lambda: 0, name="mystery")
        reg.register(plain, ToolSpec(name="spill", needs=(), internal=True), name="spill")
        out = json.loads(json.dumps(reg.describe()))
        assert [d["name"] for d in out] == ["mail", "mystery", "spill"]
        mail = out[0]
        assert mail == {
            "name": "mail",
            "declared": True,
            "risk": "high",
            "needs": ["email_send", "file_read"],
            "timeout_s": 12.0,
            "concurrency_safe": True,
            "approval": "always",
            "internal": False,
            "description": "sends mail",
        }
        assert out[1]["declared"] is False and out[1]["risk"] == "unknown"
        assert out[1]["needs"] == []
        assert out[2]["internal"] is True and out[2]["risk"] == "low"

    def test_empty_registry(self):
        assert ToolRegistry().describe() == []
        assert ToolRegistry().names() == []
        assert ToolRegistry().undeclared() == []

    def test_names_keep_registration_order(self):
        reg = ToolRegistry()
        for n in ["zeta", "alpha", "mid"]:
            reg.register(plain, name=n)
        assert reg.names() == ["zeta", "alpha", "mid"]
        assert reg.undeclared() == ["zeta", "alpha", "mid"]
