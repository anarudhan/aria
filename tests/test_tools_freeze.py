"""Tests for freeze_args and FrozenArgs (issue #59)."""

from __future__ import annotations

import collections
import copy
import enum
import hashlib
import json
import math
import pickle
import random
from collections.abc import Mapping

import pytest

from aria.tools import FreezeError, FrozenArgs, FrozenMap, freeze_args


def nest_lists(depth, leaf=1):
    value = leaf
    for _ in range(depth):
        value = [value]
    return value


def nest_dicts(depth):
    value = {"x": 1}
    for _ in range(depth - 1):
        value = {"k": value}
    return value


# ─── accepted data ───────────────────────────────────────────────────────────


class TestAccepts:
    def test_all_json_types(self):
        data = {
            "s": "text",
            "i": 7,
            "f": 1.5,
            "t": True,
            "n": None,
            "l": [1, "a", None],
            "tu": (1, 2),
            "d": {"x": {"y": [{"z": 1}]}},
        }
        frozen = freeze_args(data)
        assert isinstance(frozen, FrozenArgs)
        assert frozen["s"] == "text" and frozen["i"] == 7 and frozen["t"] is True
        assert frozen["n"] is None
        assert frozen["l"] == (1, "a", None)
        assert frozen["tu"] == (1, 2)  # a tuple is treated as a list
        assert frozen["d"]["x"]["y"][0]["z"] == 1

    def test_empty(self):
        f = freeze_args({})
        assert len(f) == 0 and f == {} and f.canonical_json() == "{}"

    def test_mapping_behaviour(self):
        f = freeze_args({"a": 1, "b": [1]})
        assert len(f) == 2 and list(f) == ["a", "b"]
        assert "a" in f and "z" not in f
        assert f.get("a") == 1 and f.get("z") is None and f.get("z", 5) == 5
        assert dict(f.items())["a"] == 1
        assert sorted(f.keys()) == ["a", "b"]
        with pytest.raises(KeyError):
            f["z"]
        assert isinstance(f, Mapping) and isinstance(f["b"], tuple)

    def test_equality_with_dict(self):
        data = {"a": 1, "b": [1, 2, {"c": None}], "d": {"e": "x"}}
        f = freeze_args(data)
        assert f == data
        assert data == f
        assert f != {"a": 1}
        assert f != {"a": 1, "b": [1, 2, {"c": 0}], "d": {"e": "x"}}
        assert f != {"a": 1, "b": [1, 2], "d": {"e": "x"}}
        assert f != "a"
        assert (f == 5) is False
        assert freeze_args({"a": 1}) == freeze_args({"a": 1})
        assert freeze_args({"a": 1}) != freeze_args({"a": 2})

    def test_equality_distinguishes_bool_from_int(self):
        assert freeze_args({"a": True}) != {"a": 1}
        assert freeze_args({"a": 1}) != {"a": True}
        assert freeze_args({"a": [False]}) != {"a": [0]}

    def test_large_but_in_limits(self):
        f = freeze_args({"s": "x" * 1_000_000})
        assert len(f["s"]) == 1_000_000

    def test_int_limits(self):
        big = 10**100 - 1
        assert freeze_args({"i": big})["i"] == big
        assert freeze_args({"i": -big})["i"] == -big
        assert freeze_args({"i": 0})["i"] == 0

    def test_float_edge_values(self):
        f = freeze_args({"a": -0.0, "b": 5e-324, "c": 1.7976931348623157e308})
        assert f["c"] == 1.7976931348623157e308

    def test_depth_exactly_at_limit(self):
        d = nest_dicts(16)
        assert freeze_args(d) == d
        assert freeze_args(nest_dicts(3), max_depth=3) == nest_dicts(3)

    def test_shared_non_cyclic_references_are_fine(self):
        shared = [1, 2]
        f = freeze_args({"a": shared, "b": shared, "c": [shared, shared]})
        assert f["a"] == (1, 2) and f["c"] == ((1, 2), (1, 2))

    def test_unicode(self):
        data = {"ключ": "значение", "emoji": "\U0001f600", "nul": "a\x00b", "ké": " "}
        f = freeze_args(data)
        assert f.thaw() == data
        assert f.canonical_json().isascii()


# ─── rejected data ───────────────────────────────────────────────────────────


class Point:
    pass


class Color(enum.Enum):
    RED = 1


class IntColor(enum.IntEnum):
    ONE = 1


class MyStr(str):
    pass


class MyInt(int):
    pass


class MyDict(dict):
    pass


class MyList(list):
    pass


Pair = collections.namedtuple("Pair", "a b")


REJECTED = [
    pytest.param(b"bytes", id="bytes"),
    pytest.param(bytearray(b"x"), id="bytearray"),
    pytest.param({1, 2}, id="set"),
    pytest.param(frozenset({1}), id="frozenset"),
    pytest.param(Point(), id="object"),
    pytest.param(Point, id="class"),
    pytest.param(len, id="builtin"),
    pytest.param(lambda: 1, id="lambda"),
    pytest.param(1 + 2j, id="complex"),
    pytest.param(Color.RED, id="enum"),
    pytest.param(IntColor.ONE, id="intenum"),
    pytest.param(MyStr("x"), id="str-subclass"),
    pytest.param(MyInt(3), id="int-subclass"),
    pytest.param(MyDict(a=1), id="dict-subclass"),
    pytest.param(MyList([1]), id="list-subclass"),
    pytest.param(Pair(1, 2), id="namedtuple"),
    pytest.param(collections.OrderedDict(a=1), id="ordereddict"),
    pytest.param(collections.defaultdict(int), id="defaultdict"),
    pytest.param(range(3), id="range"),
    pytest.param((i for i in range(3)), id="generator"),
    pytest.param(memoryview(b"x"), id="memoryview"),
    pytest.param(math.nan, id="nan"),
    pytest.param(math.inf, id="inf"),
    pytest.param(-math.inf, id="-inf"),
    pytest.param(float("nan"), id="nan2"),
    pytest.param(10**100, id="int-101-digits"),
    pytest.param(-(10**100), id="neg-int-101-digits"),
    pytest.param(10**5000, id="int-5000-digits"),
    pytest.param({1: "a"}, id="int-key"),
    pytest.param({None: "a"}, id="none-key"),
    pytest.param({("a",): "a"}, id="tuple-key"),
    pytest.param({True: "a"}, id="bool-key"),
    pytest.param({b"k": "a"}, id="bytes-key"),
    pytest.param({1.5: "a"}, id="float-key"),
    pytest.param({MyStr("k"): "a"}, id="str-subclass-key"),
    pytest.param({Color.RED: "a"}, id="enum-key"),
]


class TestRejects:
    @pytest.mark.parametrize("value", REJECTED)
    def test_as_value(self, value):
        with pytest.raises(FreezeError):
            freeze_args({"v": value})

    @pytest.mark.parametrize("value", REJECTED)
    def test_nested_in_list(self, value):
        with pytest.raises(FreezeError):
            freeze_args({"v": [1, [2, {"w": [value]}]]})

    @pytest.mark.parametrize("value", [None, [], [("a", 1)], "text", 5, b"{}", MyDict(), collections.OrderedDict()])
    def test_top_level_must_be_plain_dict(self, value):
        with pytest.raises(FreezeError, match="plain dict"):
            freeze_args(value)

    def test_frozen_args_cannot_be_refrozen(self):
        with pytest.raises(FreezeError):
            freeze_args(freeze_args({"a": 1}))

    def test_nested_frozen_map_rejected(self):
        inner = freeze_args({"a": 1})
        with pytest.raises(FreezeError):
            freeze_args({"v": inner})

    def test_mixed_int_and_str_keys(self):
        with pytest.raises(FreezeError, match="keys must be str"):
            freeze_args({1: "a", "1": "b"})
        with pytest.raises(FreezeError):
            freeze_args({"1": "b", 1: "a"})

    def test_error_is_value_error(self):
        assert issubclass(FreezeError, ValueError)

    def test_error_names_the_path_and_type(self):
        with pytest.raises(FreezeError) as info:
            freeze_args({"items": [1, {"url": b"x"}]})
        msg = str(info.value)
        assert "$.'items'[1].'url'" in msg and "bytes" in msg

    def test_error_never_contains_values(self):
        for bad in ({"k": ["super-secret-value", b"super-secret-bytes"]}, {"k": {"x": {1, 2, 3}}}):
            with pytest.raises(FreezeError) as info:
                freeze_args(bad)
            assert "super-secret" not in str(info.value)

    def test_error_path_is_bounded_and_ascii_safe(self):
        key = "line1\nline2\x1b[31m" + "k" * 500
        with pytest.raises(FreezeError) as info:
            freeze_args({key: {key: b"x"}})
        msg = str(info.value)
        assert "\n" not in msg and "\x1b" not in msg and len(msg) < 400


class TestLimits:
    def test_too_deep_dicts(self):
        with pytest.raises(FreezeError, match="deeper"):
            freeze_args(nest_dicts(17))

    def test_too_deep_lists(self):
        with pytest.raises(FreezeError, match="deeper"):
            freeze_args({"v": nest_lists(20)})

    def test_extremely_deep_does_not_recurse_into_the_interpreter_limit(self):
        deep = nest_lists(100_000)
        with pytest.raises(FreezeError):
            freeze_args({"v": deep})

    def test_extremely_deep_with_the_maximum_allowed_depth(self):
        deep = {"v": nest_lists(5000)}
        with pytest.raises(FreezeError):
            freeze_args(deep, max_depth=100)

    def test_custom_depth(self):
        with pytest.raises(FreezeError):
            freeze_args(nest_dicts(4), max_depth=3)

    def test_too_many_nodes(self):
        with pytest.raises(FreezeError, match="values"):
            freeze_args({"v": list(range(10_001))})

    def test_node_count_exact_boundary(self):
        # root dict (1) + list (1) + 9_998 ints = 10_000
        assert len(freeze_args({"v": list(range(9_998))})["v"]) == 9_998
        with pytest.raises(FreezeError):
            freeze_args({"v": list(range(9_999))})

    def test_shared_references_count_every_time(self):
        # A small structure that expands to a huge tree must hit the node limit.
        node = [1, 2]
        for _ in range(14):
            node = [node, node]
        with pytest.raises(FreezeError):
            freeze_args({"v": node})

    def test_string_too_long(self):
        with pytest.raises(FreezeError, match="longer"):
            freeze_args({"s": "x" * 1_000_001})

    def test_key_too_long(self):
        with pytest.raises(FreezeError):
            freeze_args({"k" * 1_000_001: 1})

    def test_total_chars(self):
        chunk = "x" * 900_000
        ok = {f"k{i}": chunk for i in range(4)}  # 3.6M plus keys
        freeze_args(ok)
        with pytest.raises(FreezeError, match="total"):
            freeze_args({f"k{i}": chunk for i in range(5)})

    def test_keys_count_toward_total(self):
        with pytest.raises(FreezeError, match="total"):
            freeze_args({"a" * 10: "b" * 10}, max_total_chars=19)
        freeze_args({"a" * 10: "b" * 10}, max_total_chars=20)

    def test_custom_limits(self):
        with pytest.raises(FreezeError):
            freeze_args({"a": [1, 2, 3]}, max_nodes=4)
        with pytest.raises(FreezeError):
            freeze_args({"a": "xxxx"}, max_string_chars=3)

    @pytest.mark.parametrize("name", ["max_depth", "max_nodes", "max_string_chars", "max_total_chars"])
    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, "5", None])
    def test_bad_limit_values(self, name, bad):
        with pytest.raises(FreezeError):
            freeze_args({"a": 1}, **{name: bad})

    def test_max_depth_has_a_hard_ceiling(self):
        with pytest.raises(FreezeError):
            freeze_args({"a": 1}, max_depth=101)
        freeze_args({"a": 1}, max_depth=100)

    def test_huge_int_rejected_without_converting_to_text(self):
        # Converting this to str would raise ValueError on Python 3.11+ and be slow before.
        with pytest.raises(FreezeError, match="digits"):
            freeze_args({"i": 1 << 1_000_000})


class TestCycles:
    def test_dict_self_reference(self):
        d: dict = {}
        d["self"] = d
        with pytest.raises(FreezeError, match="itself"):
            freeze_args(d)

    def test_list_self_reference(self):
        lst: list = []
        lst.append(lst)
        with pytest.raises(FreezeError, match="itself"):
            freeze_args({"v": lst})

    def test_indirect_cycle(self):
        a: dict = {}
        b: list = [a]
        a["b"] = b
        with pytest.raises(FreezeError, match="itself"):
            freeze_args(a)

    def test_cycle_through_tuple(self):
        lst: list = []
        tup = (lst,)
        lst.append(tup)
        with pytest.raises(FreezeError, match="itself"):
            freeze_args({"v": tup})

    def test_cycle_at_depth_still_caught(self):
        inner: list = [1]
        inner.append(inner)
        with pytest.raises(FreezeError):
            freeze_args({"a": {"b": {"c": inner}}})

    def test_failed_walk_leaves_no_state_behind(self):
        # The path set is per call: a second call with the same data behaves the same.
        d: dict = {}
        d["self"] = d
        for _ in range(3):
            with pytest.raises(FreezeError):
                freeze_args(d)
        ok = [1]
        assert freeze_args({"a": ok, "b": ok}) == {"a": [1], "b": [1]}


# ─── hostile objects ─────────────────────────────────────────────────────────


class Hostile:
    """Any use of these methods means user code ran during freezing."""

    calls: list = []

    def __repr__(self):
        Hostile.calls.append("repr")
        raise RuntimeError("repr ran")

    def __str__(self):
        Hostile.calls.append("str")
        raise RuntimeError("str ran")

    def __eq__(self, other):
        Hostile.calls.append("eq")
        raise RuntimeError("eq ran")

    def __hash__(self):
        Hostile.calls.append("hash")
        raise RuntimeError("hash ran")

    def __bool__(self):
        Hostile.calls.append("bool")
        raise RuntimeError("bool ran")

    def __len__(self):
        Hostile.calls.append("len")
        raise RuntimeError("len ran")

    def __iter__(self):
        Hostile.calls.append("iter")
        raise RuntimeError("iter ran")

    def __getattribute__(self, name):
        Hostile.calls.append(f"getattr {name}")
        raise RuntimeError("getattr ran")

    def __class_getitem__(cls, item):
        Hostile.calls.append("cgi")
        raise RuntimeError


class HostileMeta(type):
    def __repr__(cls):
        Hostile.calls.append("meta repr")
        raise RuntimeError("meta repr ran")

    @property
    def __name__(cls):  # type: ignore[override]
        Hostile.calls.append("meta name")
        raise RuntimeError("meta name ran")

    def __eq__(cls, other):
        Hostile.calls.append("meta eq")
        raise RuntimeError

    def __hash__(cls):
        Hostile.calls.append("meta hash")
        raise RuntimeError


class WithMeta(metaclass=HostileMeta):
    def __str__(self):
        Hostile.calls.append("str2")
        raise RuntimeError

    def __repr__(self):
        Hostile.calls.append("repr2")
        raise RuntimeError


class HostileStr(str):
    def __eq__(self, other):
        Hostile.calls.append("streq")
        raise RuntimeError

    def __hash__(self):
        Hostile.calls.append("strhash")
        raise RuntimeError

    def __len__(self):
        Hostile.calls.append("strlen")
        raise RuntimeError

    def __str__(self):
        Hostile.calls.append("strstr")
        raise RuntimeError

    def __repr__(self):
        Hostile.calls.append("strrepr")
        raise RuntimeError


class HostileInt(int):
    def __abs__(self):
        Hostile.calls.append("abs")
        raise RuntimeError

    def __eq__(self, other):
        Hostile.calls.append("inteq")
        raise RuntimeError

    def __hash__(self):
        Hostile.calls.append("inthash")
        raise RuntimeError


class HostileDict(dict):
    def items(self):
        Hostile.calls.append("items")
        raise RuntimeError

    def __iter__(self):
        Hostile.calls.append("dictiter")
        raise RuntimeError

    def __len__(self):
        Hostile.calls.append("dictlen")
        raise RuntimeError

    def __getitem__(self, k):
        Hostile.calls.append("getitem")
        raise RuntimeError


class HostileList(list):
    def __iter__(self):
        Hostile.calls.append("listiter")
        raise RuntimeError

    def __len__(self):
        Hostile.calls.append("listlen")
        raise RuntimeError

    def __getitem__(self, i):
        Hostile.calls.append("listgetitem")
        raise RuntimeError


class TestHostileObjects:
    def setup_method(self):
        Hostile.calls.clear()

    @pytest.mark.parametrize(
        "make",
        [
            lambda: Hostile(),
            lambda: WithMeta(),
            lambda: HostileStr("x"),
            lambda: HostileInt(5),
            lambda: HostileDict(a=1),
            lambda: HostileList([1]),
        ],
        ids=["object", "metaclass", "str", "int", "dict", "list"],
    )
    @pytest.mark.parametrize("where", ["value", "nested", "top"])
    def test_foreign_objects_rejected_without_running_their_code(self, make, where):
        obj = make()
        if where == "value":
            data = {"v": obj}
        elif where == "nested":
            data = {"v": [{"w": [obj]}]}
        else:
            data = obj
        with pytest.raises(FreezeError) as info:
            freeze_args(data)
        assert Hostile.calls == []
        # The type name is shown and nothing else about the object.
        assert "ran" not in str(info.value)

    def test_hostile_key_rejected_without_running_its_code(self):
        class HostileKey:
            def __hash__(self):
                return 1  # hashable so it can be a key; everything else raises

            def __eq__(self, other):
                Hostile.calls.append("keyeq")
                raise RuntimeError("eq ran")

            def __repr__(self):
                Hostile.calls.append("keyrepr")
                raise RuntimeError("repr ran")

            __str__ = __repr__

        with pytest.raises(FreezeError, match="HostileKey"):
            freeze_args({HostileKey(): 1, "ok": 2})
        assert Hostile.calls == []

    def test_hostile_str_subclass_key_rejected_without_running_its_code(self):
        class SneakyStr(str):
            def __len__(self):
                Hostile.calls.append("len")
                raise RuntimeError

            def __str__(self):
                Hostile.calls.append("str")
                raise RuntimeError

        with pytest.raises(FreezeError):
            freeze_args({SneakyStr("k"): 1})
        assert Hostile.calls == []

    def test_hostile_value_next_to_good_data(self):
        with pytest.raises(FreezeError):
            freeze_args({"good": [1, 2, 3], "bad": Hostile(), "also_good": {"x": 1}})
        assert Hostile.calls == []

    def test_hostile_object_is_never_touched_even_for_cycle_check(self):
        obj = Hostile()
        with pytest.raises(FreezeError):
            freeze_args({"a": obj, "b": obj})
        assert Hostile.calls == []

    def test_metaclass_name_property_not_used(self):
        with pytest.raises(FreezeError) as info:
            freeze_args({"v": WithMeta()})
        assert "WithMeta" in str(info.value)
        assert Hostile.calls == []


# ─── immutability and copy semantics ─────────────────────────────────────────


class TestImmutability:
    def test_cannot_mutate_mapping(self):
        f = freeze_args({"a": 1, "n": {"b": 2}})
        with pytest.raises(TypeError):
            f["a"] = 2  # type: ignore[index]
        with pytest.raises(TypeError):
            del f["a"]  # type: ignore[attr-defined]
        with pytest.raises(TypeError):
            f["n"]["b"] = 3
        for method in ("update", "pop", "clear", "setdefault", "popitem"):
            assert not hasattr(f, method)
            assert not hasattr(f["n"], method)

    def test_nested_lists_are_tuples(self):
        f = freeze_args({"l": [1, [2, 3], {"x": [4]}]})
        assert type(f["l"]) is tuple and type(f["l"][1]) is tuple
        assert type(f["l"][2]) is FrozenMap and type(f["l"][2]["x"]) is tuple
        with pytest.raises(AttributeError):
            f["l"].append(1)  # type: ignore[attr-defined]
        with pytest.raises(TypeError):
            f["l"][0] = 9  # type: ignore[index]

    def test_attributes_cannot_be_set(self):
        f = freeze_args({"a": 1})
        for obj in (f, f["a"] and freeze_args({"n": {"b": 1}})["n"]):
            with pytest.raises(AttributeError):
                obj.extra = 1
            with pytest.raises(AttributeError):
                obj._d = {}
            with pytest.raises(AttributeError):
                del obj._d
        with pytest.raises(AttributeError):
            f._json = "{}"
        with pytest.raises(AttributeError):
            f._digest = "0"

    def test_no_instance_dict(self):
        f = freeze_args({"a": {"b": 1}})
        assert not hasattr(f, "__dict__") and not hasattr(f["a"], "__dict__")

    def test_cannot_be_built_directly(self):
        with pytest.raises(TypeError):
            FrozenArgs(object(), None)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            FrozenMap(object(), {"a": []})  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            FrozenArgs({"a": 1})  # type: ignore[call-arg]

    def test_cannot_subclass(self):
        with pytest.raises(TypeError):

            class A(FrozenArgs):
                pass

        with pytest.raises(TypeError):

            class B(FrozenMap):
                pass

    def test_not_hashable(self):
        with pytest.raises(TypeError):
            hash(freeze_args({"a": 1}))

    def test_not_picklable_but_copy_is_safe(self):
        f = freeze_args({"a": [1], "b": {"c": 2}})
        with pytest.raises(TypeError):
            pickle.dumps(f)
        with pytest.raises(TypeError):
            pickle.dumps(f["b"])
        assert copy.copy(f) is f and copy.deepcopy(f) is f
        assert copy.deepcopy(f["b"]) is f["b"]

    def test_repr_hides_values(self):
        f = freeze_args({"password": "hunter2-secret", "n": {"token": "abc-secret"}})
        for text in (repr(f), str(f), repr(f["n"])):
            assert "secret" not in text and "hunter2" not in text


class TestCopySemantics:
    def test_mutating_original_after_freezing_changes_nothing(self):
        original = {"a": [1, 2], "b": {"c": "x"}, "d": "s"}
        frozen = freeze_args(original)
        digest = frozen.sha256()
        text = frozen.canonical_json()
        snapshot = frozen.thaw()

        original["a"].append(3)
        original["b"]["c"] = "CHANGED"
        original["d"] = "other"
        original["new"] = 1
        del original["b"]["c"]

        assert frozen.thaw() == snapshot
        assert frozen["a"] == (1, 2) and frozen["b"]["c"] == "x" and "new" not in frozen
        assert frozen.canonical_json() == text and frozen.sha256() == digest

    def test_original_tuple_content_and_shared_lists(self):
        inner = [1]
        original = {"a": inner, "b": inner}
        frozen = freeze_args(original)
        inner.append(2)
        assert frozen["a"] == (1,) and frozen["b"] == (1,)

    def test_mutating_thawed_copy_changes_nothing(self):
        frozen = freeze_args({"a": [1, {"x": 2}], "b": {"c": [3]}})
        digest, text = frozen.sha256(), frozen.canonical_json()
        thawed = frozen.thaw()
        thawed["a"].append(99)
        thawed["a"][1]["x"] = "BAD"
        thawed["b"]["c"].clear()
        thawed["z"] = 1
        assert frozen == {"a": [1, {"x": 2}], "b": {"c": [3]}}
        assert frozen.sha256() == digest and frozen.canonical_json() == text

    def test_thaw_returns_fresh_copy_every_time(self):
        frozen = freeze_args({"a": [1], "b": {"c": 1}})
        t1, t2 = frozen.thaw(), frozen.thaw()
        assert t1 == t2 and t1 is not t2 and t1["a"] is not t2["a"] and t1["b"] is not t2["b"]

    def test_thaw_gives_plain_types(self):
        t = freeze_args({"a": [1, (2, 3)], "b": {"c": [4]}}).thaw()
        assert type(t) is dict and type(t["b"]) is dict
        assert type(t["a"]) is list and type(t["a"][1]) is list and type(t["b"]["c"]) is list

    def test_nested_frozen_map_thaw(self):
        frozen = freeze_args({"n": {"m": [1, 2]}})
        inner = frozen["n"]
        assert inner.thaw() == {"m": [1, 2]}

    def test_digest_and_json_computed_from_frozen_copy_at_freeze_time(self):
        d = {"a": 1}
        f = freeze_args(d)
        d["a"] = 2
        assert f.canonical_json() == '{"a":1}'
        assert f.sha256() == freeze_args({"a": 1}).sha256() != freeze_args({"a": 2}).sha256()


# ─── canonical JSON and digest ───────────────────────────────────────────────


class TestCanonical:
    def test_sorted_keys_no_whitespace_ascii(self):
        f = freeze_args({"b": 1, "a": [True, None, 1.5, "é\U0001f600"], "c": {"z": 1, "y": 2}})
        assert f.canonical_json() == (
            '{"a":[true,null,1.5,"\\u00e9\\ud83d\\ude00"],"b":1,"c":{"y":2,"z":1}}'
        )

    def test_key_order_does_not_change_the_digest(self):
        a = freeze_args({"x": 1, "y": {"p": 1, "q": 2}})
        b = freeze_args({"y": {"q": 2, "p": 1}, "x": 1})
        assert a.canonical_json() == b.canonical_json() and a.sha256() == b.sha256()

    def test_digest_is_plain_sha256_of_the_text(self):
        f = freeze_args({"a": [1, 2], "b": "é"})
        expect = hashlib.sha256(f.canonical_json().encode("utf-8")).hexdigest()
        assert f.sha256() == expect and len(f.sha256()) == 64

    def test_list_order_matters(self):
        assert freeze_args({"a": [1, 2]}).sha256() != freeze_args({"a": [2, 1]}).sha256()

    def test_types_are_distinguished(self):
        digests = {
            freeze_args({"a": v}).sha256() for v in (1, 1.0, True, "1", None, [1], {"a": 1}, "true")
        }
        assert len(digests) == 8

    def test_tuple_and_list_are_the_same(self):
        assert freeze_args({"a": (1, 2)}).sha256() == freeze_args({"a": [1, 2]}).sha256()

    def test_lone_surrogate_is_escaped_not_crashing(self):
        f = freeze_args({"a": "\ud800"})
        assert f.canonical_json() == '{"a":"\\ud800"}'


# ─── randomized round trip ───────────────────────────────────────────────────

_ALPHABET = "abcXYZ019 _-.\"'\\/\n\té中\U0001f600\x00\x7f"


def _rand_str(rng: random.Random) -> str:
    return "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, 12)))


def _rand_value(rng: random.Random, depth: int):
    kinds = ["str", "int", "float", "bool", "none"]
    if depth < 5:
        kinds += ["list", "dict", "list", "dict"]
    kind = rng.choice(kinds)
    if kind == "str":
        return _rand_str(rng)
    if kind == "int":
        return rng.choice([0, 1, -1, rng.randint(-(10**6), 10**6), rng.randint(-(10**99), 10**99)])
    if kind == "float":
        return rng.choice(
            [0.0, -0.0, 1.5, rng.uniform(-1e6, 1e6), rng.uniform(-1, 1) * 10 ** rng.randint(-300, 300)]
        )
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "none":
        return None
    if kind == "list":
        return [_rand_value(rng, depth + 1) for _ in range(rng.randint(0, 5))]
    return {_rand_str(rng): _rand_value(rng, depth + 1) for _ in range(rng.randint(0, 5))}


@pytest.mark.parametrize("block", range(5))
def test_random_round_trip(block):
    """freeze -> thaw equals the original, and the canonical JSON matches json.dumps."""
    for seed in range(block * 100, block * 100 + 100):
        rng = random.Random(seed)
        original = {_rand_str(rng): _rand_value(rng, 0) for _ in range(rng.randint(0, 6))}
        expected_json = json.dumps(original, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        before = copy.deepcopy(original)

        frozen = freeze_args(original)

        assert original == before, f"seed {seed}: freeze changed its input"
        assert frozen.thaw() == original, f"seed {seed}"
        assert frozen == original, f"seed {seed}"
        assert frozen.canonical_json() == expected_json, f"seed {seed}"
        assert frozen.sha256() == hashlib.sha256(expected_json.encode()).hexdigest()
        # Freezing the thawed copy gives the same bytes.
        again = freeze_args(frozen.thaw())
        assert again.canonical_json() == expected_json and again == frozen
        # The thawed copy has the same container types as the input.
        assert json.dumps(frozen.thaw(), sort_keys=True) == json.dumps(original, sort_keys=True)


def test_random_mutation_after_freeze_never_leaks():
    for seed in range(200):
        rng = random.Random(10_000 + seed)
        original = {_rand_str(rng): _rand_value(rng, 0) for _ in range(rng.randint(1, 6))}
        frozen = freeze_args(original)
        text, digest = frozen.canonical_json(), frozen.sha256()
        _scramble(original, rng)
        assert frozen.canonical_json() == text and frozen.sha256() == digest
        thawed = frozen.thaw()
        _scramble(thawed, rng)
        assert frozen.canonical_json() == text and frozen.sha256() == digest


def _scramble(node, rng):
    if isinstance(node, dict):
        for key in list(node):
            _scramble(node[key], rng)
        node["__added__"] = rng.random()
        if node and rng.random() < 0.5:
            node.pop(next(iter(node)))
    elif isinstance(node, list):
        for item in node:
            _scramble(item, rng)
        node.append("__added__")
