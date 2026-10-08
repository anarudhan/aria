"""
Tests for aria.security.redact (issue #58): redact_args, ArgsDigester, and the move of
sanitize_error_text out of aria.plugins.retry.

These tests try to break the redaction: hostile objects, cycles, huge and deep inputs.
Redaction is best effort; the tests describe what IS covered, not a guarantee.
"""

from __future__ import annotations

import ast
import collections
import copy
import hashlib
import hmac
import json
import pickle
import random
import threading
import time
from pathlib import Path

import pytest

import aria.plugins.retry as retry_module
import aria.security.redact as redact_module
from aria.security.redact import (
    MAX_ERROR_TEXT,
    REDACTED,
    ArgsDigester,
    default_digester,
    redact_args,
    sanitize_error_text,
)

TIME_LIMIT = 2.0


def jsonable(value) -> str:
    """Strict JSON: no nan/inf, only plain types. Raises if the value is not JSON-safe."""
    return json.dumps(value, allow_nan=False, ensure_ascii=True)


# ─── The move out of retry.py ─────────────────────────────────────────────────

class TestMove:
    def test_retry_re_exports_the_same_objects(self):
        assert retry_module.sanitize_error_text is redact_module.sanitize_error_text
        assert retry_module.MAX_ERROR_TEXT is redact_module.MAX_ERROR_TEXT
        assert MAX_ERROR_TEXT == 500

    def test_retry_no_longer_holds_its_own_patterns(self):
        assert not hasattr(retry_module, "_SECRET_PATTERNS")

    def test_redact_module_imports_only_the_standard_library(self):
        # Import-light: nothing from aria, so no import cycle is possible.
        tree = ast.parse(Path(redact_module.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0, "relative import"
                assert node.module is not None and not node.module.startswith("aria")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("aria")

    def test_sanitize_still_works_from_the_new_place(self):
        assert sanitize_error_text("password=hunter2 fail") == "password=[REDACTED] fail"
        assert sanitize_error_text("x" * 600).endswith("original length 600 characters]")


# ─── Key names ────────────────────────────────────────────────────────────────

SECRET_KEYS = [
    "password", "Password", "PASSWORD", "passwd", "passphrase", "secret", "token",
    "api_key", "apikey", "apiKey", "ApiKey", "API_KEY", "X-Api-Key", "x-api-key",
    "api-key", "api key", "api.key", "access_key", "accessKey", "AccessKey",
    "aws_secret_access_key", "AWSSecretAccessKey", "private_key", "privateKey",
    "authorization", "Authorization", "Proxy-Authorization", "auth", "Auth", "oauth",
    "cookie", "Set-Cookie", "cookies", "credential", "credentials", "Credentials",
    "session", "sessionId", "session_id", "bearer", "client_secret", "clientSecret",
    "access_token", "accessToken", "refresh_token", "x-auth-token", "id_token",
    "tokens", "db_password", "dbPassword", "DB_PASSWORD", "signing_key", "ssh_key",
    "my_custom_key", "service_key", "stripe_secret", "webhook_secret", "github_token",
    "password1", "token2", "api_key_1", "secret_2", "pwd", "pass", "passcode", "jwt",
    "ＰＡＳＳＷＯＲＤ",  # full-width letters, normalised by NFKC
    "a" * 2000 + "_secret",
]

PLAIN_KEYS = [
    "path", "query", "command", "url", "name", "description", "limit", "key", "Key",
    "monkey", "turkey", "author", "authority", "tokenizer", "keyboard", "content", "id",
    "file", "mode", "recursive", "timeout", "", "x", "passage", "compass", "bypass",
    "password_hint_text_is_not_here",  # contains "password": covered, see below
]
PLAIN_KEYS.remove("password_hint_text_is_not_here")


class TestKeyNames:
    @pytest.mark.parametrize("key", SECRET_KEYS)
    def test_secret_key_hides_the_value(self, key):
        out = redact_args({key: "hunter2-value"})
        assert list(out.values()) == [REDACTED]

    @pytest.mark.parametrize("key", PLAIN_KEYS)
    def test_other_keys_are_kept(self, key):
        out = redact_args({key: "visible-value"})
        assert list(out.values()) == ["visible-value"]

    @pytest.mark.parametrize("value", [
        "text", 0, 12345, 1.5, True, False, None, b"bytes", [1, 2], {"a": 1}, (1, 2),
        {1, 2}, object(), float("nan"), 10 ** 500, {"nested": {"deep": ["x"]}},
    ])
    def test_value_type_does_not_matter(self, value):
        assert redact_args({"password": value}) == {"password": REDACTED}

    def test_secret_value_is_not_even_looked_at(self):
        class Boom:
            def __getattribute__(self, name):
                raise AssertionError("touched")

        assert redact_args({"token": Boom()}) == {"token": REDACTED}

    def test_key_names_in_nested_structures(self):
        out = redact_args({"headers": [{"Authorization": "Basic abc", "Accept": "x"}]})
        assert out == {"headers": [{"Authorization": REDACTED, "Accept": "x"}]}

    def test_the_key_name_itself_stays_visible(self):
        assert "password" in redact_args({"password": "x"})

    def test_secret_inside_a_key_name_is_redacted(self):
        # A key that IS a secret (some APIs use ids as keys) goes through sanitize_error_text.
        out = redact_args({"sk-abcdefghijklmnop123456": 1})
        assert "abcdefghijklmnop123456" not in jsonable(out)

    def test_key_collisions_do_not_lose_values(self):
        out = redact_args({1: "a", "1": "b", 2.5: "c", "2.5": "d"})
        assert sorted(out.values()) == ["a", "b", "c", "d"]
        assert len(out) == 4

    def test_non_string_keys_become_text(self):
        out = redact_args({1: "a", None: "b", (1, 2): "d", b"k": "e", 2.5: "f"})
        assert all(isinstance(k, str) for k in out)
        assert out["1"] == "a" and out["None"] == "b"
        assert out["<tuple>"] == "d" and out["<bytes 1>"] == "e" and out["2.5"] == "f"
        jsonable(out)

    def test_object_key_is_never_stringified(self):
        class Key:
            def __str__(self):
                raise RuntimeError("str")

            def __repr__(self):
                raise RuntimeError("repr")

        out = redact_args({Key(): "v"})
        assert out == {"<Key>": "v"}


# ─── Structure ────────────────────────────────────────────────────────────────

class TestStructure:
    def test_nested_dicts_lists_and_tuples(self):
        value = {"a": {"b": [1, {"c": ("x", "y")}]}, "t": (1, (2, 3))}
        assert redact_args(value) == {"a": {"b": [1, {"c": ["x", "y"]}]}, "t": [1, [2, 3]]}

    def test_tuple_and_set_become_lists(self):
        out = redact_args({"t": (1, 2), "s": {3}, "f": frozenset({4})})
        assert out == {"t": [1, 2], "s": [3], "f": [4]}
        assert all(type(v) is list for v in out.values())

    def test_top_level_scalars(self):
        assert redact_args(None) is None
        assert redact_args(5) == 5
        assert redact_args("password=hunter2") == "password=[REDACTED]"
        assert redact_args([]) == []
        assert redact_args({}) == {}

    def test_input_is_not_changed(self):
        value = {"password": "p", "list": [1, {"token": "t"}], "s": "token=abc"}
        before = copy.deepcopy(value)
        redact_args(value)
        assert value == before

    def test_result_is_a_copy(self):
        inner = [1, 2]
        out = redact_args({"x": inner})
        out["x"].append(3)
        assert inner == [1, 2]

    def test_subclasses_of_builtins_become_plain_types(self):
        class MyDict(dict):
            pass

        class MyList(list):
            pass

        out = redact_args(MyDict(a=MyList([1])))
        assert type(out) is dict and type(out["a"]) is list

    def test_ordered_and_default_dicts(self):
        assert redact_args(collections.OrderedDict(a=1)) == {"a": 1}
        assert redact_args(collections.defaultdict(list, a=[1])) == {"a": [1]}

    def test_other_containers_are_not_walked(self):
        assert redact_args(collections.deque([1])) == "<deque>"
        assert redact_args(range(3)) == "<range>"
        assert redact_args(iter([1])) == "<list_iterator>"
        assert redact_args((x for x in [1])) == "<generator>"


# ─── Strings ──────────────────────────────────────────────────────────────────

class TestStrings:
    def test_string_leaves_go_through_sanitize(self):
        out = redact_args({"cmd": "curl -H 'Authorization: Bearer abc.def' --password=hunter2"})
        text = jsonable(out)
        assert "abc.def" not in text and "hunter2" not in text

    def test_known_shapes_in_strings(self):
        samples = {
            "sk-abcdefghijklmnopqrstuv": "abcdefghijklmnopqrstuv",
            "AKIAABCDEFGHIJKLMNOP": "AKIAABCDEFGHIJKLMNOP",
            "ghp_" + "a" * 30: "a" * 30,
            "postgres://user:s3cretpw@host/db": "s3cretpw",
            "api_key=zzz999": "zzz999",
        }
        for text, secret in samples.items():
            assert secret not in jsonable(redact_args([text, {"k": text}]))

    def test_long_strings_are_cut_to_max_string(self):
        out = redact_args("x" * 1000)
        assert out.startswith("x" * 200 + "...")
        assert "original length 1000" in out
        out = redact_args("x" * 1000, max_string=10)
        assert out.startswith("x" * 10 + "...")

    def test_short_strings_are_unchanged(self):
        assert redact_args("hello") == "hello"
        assert redact_args("y" * 200) == "y" * 200

    def test_secret_on_the_cut_boundary_is_redacted_before_cutting(self):
        text = "x" * 190 + "password=hunter2supersecret"
        out = redact_args(text)
        assert "hunter2" not in out and "hunter" not in out

    def test_strings_are_redacted_at_every_depth(self):
        out = redact_args({"a": {"b": {"c": ["token=abc123xyz"]}}})
        assert "abc123xyz" not in jsonable(out)

    def test_str_subclass_with_hostile_methods(self):
        class Evil(str):
            def __str__(self):
                raise RuntimeError("str")

            def __repr__(self):
                raise RuntimeError("repr")

            def __len__(self):
                raise RuntimeError("len")

            def __getitem__(self, item):
                raise RuntimeError("getitem")

            def __iter__(self):
                raise RuntimeError("iter")

        out = redact_args({"k": Evil("password=hunter2"), Evil("name"): 1})
        assert out == {"k": "password=[REDACTED]", "name": 1}

    def test_unpaired_surrogates_stay_json_safe(self):
        out = redact_args({"s": "bad \ud800 text", "\udfff": 1})
        jsonable(out)
        json.loads(jsonable(out))

    def test_nul_and_control_characters(self):
        out = redact_args("a\x00b\x1fc password=x\x00y")
        jsonable(out)

    def test_huge_string_is_not_scanned_and_not_leaked(self):
        text = "password=hunter2 " + "x" * 300_000
        out = redact_args({"s": text})
        assert out["s"].startswith("<string, ")
        assert "hunter2" not in jsonable(out)

    def test_total_character_budget(self):
        out = redact_args(["x" * 150_000 for _ in range(20)])
        assert any(item.startswith("<string, ") for item in out)
        jsonable(out)


# ─── Non-string values ────────────────────────────────────────────────────────

class TestScalars:
    def test_ints_floats_bools_none_are_kept(self):
        value = {"i": 7, "neg": -3, "f": 2.5, "t": True, "n": None, "z": 0, "fz": 0.0}
        assert redact_args(value) == value
        assert redact_args({"t": True})["t"] is True

    @pytest.mark.parametrize("number, expected", [
        (float("nan"), "nan"), (float("inf"), "inf"), (float("-inf"), "-inf"),
    ])
    def test_non_finite_floats_become_strings(self, number, expected):
        assert redact_args([number, {"x": number}]) == [expected, {"x": expected}]

    def test_huge_ints_are_summarised(self):
        out = redact_args([10 ** 5000, 2 ** 255, -(2 ** 300)])
        assert out[1] == 2 ** 255
        assert out[0].startswith("<int,") and out[2].startswith("<int,")
        jsonable(out)

    def test_bytes(self):
        assert redact_args(b"secret bytes") == "<bytes 12>"
        assert redact_args(b"") == "<bytes 0>"
        assert redact_args(bytearray(b"abc")) == "<bytes 3>"
        assert redact_args({"k": b"x" * 100_000}) == {"k": "<bytes 100000>"}

    def test_other_objects_become_their_type_name(self):
        class Thing:
            pass

        assert redact_args(Thing()) == "<Thing>"
        assert redact_args(object()) == "<object>"
        assert redact_args(len) == "<builtin_function_or_method>"
        assert redact_args(lambda: 1) == "<function>"
        assert redact_args(Thing) == "<type>"
        assert redact_args(1 + 2j) == "<complex>"
        assert redact_args(memoryview(b"abc")) == "<memoryview>"
        assert redact_args(Path("/etc/passwd")) == "<PosixPath>"

    def test_int_and_float_subclasses_with_hostile_methods(self):
        class Evil(int):
            def __repr__(self):
                raise RuntimeError

            __str__ = __int__ = __index__ = __hash__ = __repr__

        class EvilFloat(float):
            def __repr__(self):
                raise RuntimeError

            __str__ = __float__ = __repr__

        out = redact_args([Evil(5), EvilFloat(2.5), EvilFloat("nan")])
        assert out == [5, 2.5, "nan"]

    def test_bool_is_not_turned_into_an_int(self):
        out = redact_args([True, False])
        assert out[0] is True and out[1] is False


# ─── Hostile inputs never raise ───────────────────────────────────────────────

class RaisesEverything:
    def __str__(self):
        raise RuntimeError("str")

    def __repr__(self):
        raise RuntimeError("repr")

    def __getattr__(self, name):
        raise RuntimeError("getattr " + name)

    def __len__(self):
        raise RuntimeError("len")

    def __iter__(self):
        raise RuntimeError("iter")

    def __bool__(self):
        raise RuntimeError("bool")

    def __eq__(self, other):
        raise RuntimeError("eq")

    def __hash__(self):
        return 1

    def __format__(self, spec):
        raise RuntimeError("format")

    def __reduce__(self):
        raise RuntimeError("reduce")

    def __getitem__(self, item):
        raise RuntimeError("getitem")


class WithSlots:
    __slots__ = ("secret", "other")

    def __init__(self):
        self.secret = "password=hunter2"


class BadClass:
    @property
    def __class__(self):  # type: ignore[override]
        raise RuntimeError("class")


class TestHostileInputs:
    def test_object_that_raises_everywhere(self):
        value = {"a": RaisesEverything(), "b": [RaisesEverything()], "c": (RaisesEverything(),)}
        out = redact_args(value)
        assert out == {"a": "<RaisesEverything>", "b": ["<RaisesEverything>"],
                       "c": ["<RaisesEverything>"]}

    def test_object_as_dict_key(self):
        assert redact_args({RaisesEverything(): 1}) == {"<RaisesEverything>": 1}

    def test_slots_object_is_not_read(self):
        out = redact_args({"x": WithSlots()})
        assert out == {"x": "<WithSlots>"}
        assert "hunter2" not in jsonable(out)

    def test_object_with_raising_class_property(self):
        assert redact_args([BadClass()]) == ["<BadClass>"]

    def test_object_with_secret_in_repr_is_not_repr_ed(self):
        class Leaky:
            def __repr__(self):
                return "Leaky(password='hunter2')"

            __str__ = __repr__

        assert "hunter2" not in jsonable(redact_args([Leaky(), {"k": Leaky()}]))

    def test_exception_instances_and_classes(self):
        out = redact_args([ValueError("password=hunter2"), ValueError, KeyError])
        assert out == ["<ValueError>", "<type>", "<type>"]

    def test_dict_subclass_with_hostile_methods(self):
        class EvilDict(dict):
            def __iter__(self):
                raise RuntimeError

            def items(self):
                raise RuntimeError

            def keys(self):
                raise RuntimeError

            def values(self):
                raise RuntimeError

            def __len__(self):
                raise RuntimeError

            def __getitem__(self, key):
                raise RuntimeError

            def get(self, *args):
                raise RuntimeError

        assert redact_args(EvilDict(a=1, password="x")) == {"a": 1, "password": REDACTED}

    def test_list_subclass_with_hostile_methods(self):
        class EvilList(list):
            def __iter__(self):
                raise RuntimeError

            def __len__(self):
                raise RuntimeError

            def __getitem__(self, item):
                raise RuntimeError

        assert redact_args(EvilList([1, 2])) == [1, 2]

    def test_set_and_tuple_subclasses(self):
        class EvilSet(set):
            def __iter__(self):
                raise RuntimeError

        class EvilTuple(tuple):
            def __iter__(self):
                raise RuntimeError

        assert redact_args(EvilSet({1})) == [1]
        assert redact_args(EvilTuple((1,))) == [1]

    def test_dict_changed_by_another_thread_does_not_raise(self):
        value = {str(i): i for i in range(100)}
        stop = threading.Event()

        def mutate():
            n = 1000
            while not stop.is_set():
                value[str(n)] = n
                value.pop(str(n), None)
                n += 1

        thread = threading.Thread(target=mutate)
        thread.start()
        try:
            for _ in range(200):
                jsonable(redact_args(value))
        finally:
            stop.set()
            thread.join()

    def test_weakref_proxy_and_mock_like_objects(self):
        import weakref

        class Ref(list):
            pass

        assert redact_args(weakref.proxy(Ref([1, 2]))) == "<ProxyType>"

    def test_getattr_that_returns_anything(self):
        class Anything:
            def __getattr__(self, name):
                return self

            def __call__(self, *a, **k):
                return self

        assert redact_args(Anything()) == "<Anything>"

    def test_bad_parameters_do_not_raise(self):
        value = {"a": [1, {"b": "x"}]}
        for kwargs in (
            {"max_depth": -5}, {"max_items": -1}, {"max_string": -1},
            {"max_depth": 10 ** 9}, {"max_items": 10 ** 9}, {"max_string": 10 ** 9},
            {"max_depth": "6"}, {"max_items": None}, {"max_string": 2.5},
            {"max_depth": True}, {"max_depth": 0, "max_items": 0, "max_string": 0},
        ):
            jsonable(redact_args(value, **kwargs))

    def test_never_raises_on_random_junk(self):
        rng = random.Random(1234)
        junk_leaves = [None, True, 0, -1, 2 ** 70, 1e308, float("nan"), "", "x", "token=abc",
                       b"b", object(), RaisesEverything(), WithSlots(), len, 1j, Path("p")]

        def build(depth):
            if depth == 0 or rng.random() < 0.3:
                return rng.choice(junk_leaves)
            kind = rng.choice(["list", "dict", "tuple", "set"])
            n = rng.randint(0, 5)
            if kind == "list":
                return [build(depth - 1) for _ in range(n)]
            if kind == "tuple":
                return tuple(build(depth - 1) for _ in range(n))
            if kind == "set":
                return {rng.choice([1, "a", None, (1, 2), 2.5]) for _ in range(n)}
            keys = ["password", "a", "Token", 1, None, (1,), "x-api-key", "k"]
            return {rng.choice(keys): build(depth - 1) for _ in range(n)}

        for _ in range(300):
            value = build(8)
            jsonable(redact_args(value))
            jsonable(redact_args(value, max_depth=2, max_items=2, max_string=3))


# ─── Cycles, depth and size ───────────────────────────────────────────────────

class TestCyclesAndLimits:
    def test_self_referencing_list(self):
        a: list = [1]
        a.append(a)
        assert redact_args(a) == [1, "<cycle>"]

    def test_self_referencing_dict(self):
        d: dict = {"a": 1}
        d["me"] = d
        assert redact_args(d) == {"a": 1, "me": "<cycle>"}

    def test_mutual_cycle(self):
        a: list = []
        b: dict = {"a": a}
        a.append(b)
        assert redact_args(a) == [{"a": "<cycle>"}]

    def test_cycle_through_tuple_and_set_member(self):
        a: list = []
        t = (a, 1)
        a.append(t)
        assert redact_args(a) == [["<cycle>", 1]]

    def test_the_same_object_twice_is_not_a_cycle(self):
        shared = [1, 2]
        out = redact_args({"a": shared, "b": shared, "c": [shared, shared]})
        assert out == {"a": [1, 2], "b": [1, 2], "c": [[1, 2], [1, 2]]}

    def test_max_depth(self):
        value = [[[[[[[["deep"]]]]]]]]       # 8 levels
        out = redact_args(value)
        assert "deep" not in jsonable(out)
        assert "<max depth>" in jsonable(out)
        assert redact_args([[["x"]]], max_depth=3) == [[["x"]]]
        assert redact_args([[["x"]]], max_depth=2) == [[["<max depth>"]]]
        assert redact_args({"a": {"b": 1}}, max_depth=0) == {"a": "<max depth>"}
        assert redact_args(5, max_depth=0) == 5

    def test_exactly_max_depth_is_kept(self):
        value: object = "leaf"
        for _ in range(6):
            value = [value]
        assert "leaf" in jsonable(redact_args(value))
        assert "leaf" not in jsonable(redact_args([value]))

    def test_max_items_list(self):
        out = redact_args(list(range(150)))
        assert len(out) == 101
        assert out[:100] == list(range(100))
        assert out[-1] == "<50 more items omitted>"

    def test_max_items_dict(self):
        out = redact_args({f"k{i}": i for i in range(130)}, max_items=10)
        assert len(out) == 11
        assert out["<omitted>"] == "<120 more items omitted>"

    def test_exactly_max_items_has_no_marker(self):
        assert redact_args(list(range(100))) == list(range(100))
        assert len(redact_args({str(i): i for i in range(100)})) == 100

    def test_max_items_set_and_tuple(self):
        assert len(redact_args(tuple(range(20)), max_items=5)) == 6
        assert redact_args(set(range(20)), max_items=5)[-1] == "<15 more items omitted>"

    def test_max_items_zero(self):
        assert redact_args([1, 2], max_items=0) == ["<2 more items omitted>"]

    def test_omitted_marker_does_not_overwrite_a_real_key(self):
        value = {"<omitted>": "real", **{f"k{i}": i for i in range(5)}}
        out = redact_args(value, max_items=3)
        assert out["<omitted>"] == "real"
        assert len(out) == 4

    def test_wide_and_deep_structure_is_bounded(self):
        # 40 x 40 x 40 x 100 = 6.4 million values without the work budget
        value = [[[list(range(100)) for _ in range(40)] for _ in range(40)] for _ in range(40)]
        start = time.perf_counter()
        out = redact_args(value, max_depth=10)
        assert time.perf_counter() - start < TIME_LIMIT
        assert "<budget exceeded>" in jsonable(out)

    def test_exponential_sharing_is_bounded(self):
        node: object = [1]
        for _ in range(60):
            node = [node, node]
        start = time.perf_counter()
        jsonable(redact_args(node, max_depth=64))
        assert time.perf_counter() - start < TIME_LIMIT


class TestTimeBounds:
    def test_deep_list_nesting_10000(self):
        root: list = []
        cur = root
        for _ in range(10_000):
            nxt: list = []
            cur.append(nxt)
            cur = nxt
        start = time.perf_counter()
        out = redact_args(root)
        assert time.perf_counter() - start < TIME_LIMIT
        assert "<max depth>" in jsonable(out)

    def test_deep_dict_nesting_10000(self):
        root: dict = {}
        cur = root
        for _ in range(10_000):
            nxt: dict = {}
            cur["a"] = nxt
            cur = nxt
        start = time.perf_counter()
        out = redact_args(root, max_depth=64)
        assert time.perf_counter() - start < TIME_LIMIT
        assert "<max depth>" in jsonable(out)

    def test_deep_mixed_nesting_with_a_huge_max_depth(self):
        value: object = "x"
        for i in range(10_000):
            value = {"k": value} if i % 2 else [value]
        out = redact_args(value, max_depth=10 ** 9)
        jsonable(out)

    @pytest.mark.parametrize("text", [
        "a" * 100_000,
        "password=" * 11_000,
        '"' * 100_000,
        'token="' * 14_000,
        "authorization: " * 6_600,
        "key=" * 25_000,
        '\\"' * 50_000,
        "password:" + " " * 100_000,
        "http://a:" + "b" * 100_000,
        "bearer " + "a" * 100_000,
        "cookie:" + "a" * 100_000,
        "é" * 100_000,
        "ﬁ" * 100_000,   # NFKC expands this ligature to two characters
    ])
    def test_100k_character_strings(self, text):
        start = time.perf_counter()
        out = redact_args({"s": text})
        assert time.perf_counter() - start < TIME_LIMIT
        jsonable(out)

    def test_many_big_strings(self):
        start = time.perf_counter()
        out = redact_args(["password=" * 11_000 for _ in range(50)])
        assert time.perf_counter() - start < TIME_LIMIT
        jsonable(out)

    def test_many_colliding_keys(self):
        value = {object(): i for i in range(5000)}
        start = time.perf_counter()
        out = redact_args(value, max_items=10_000)
        assert time.perf_counter() - start < TIME_LIMIT
        assert len(out) == 5000

    def test_many_small_strings(self):
        value = [f"token=abc{i} and password={i}" for i in range(10_000)]
        start = time.perf_counter()
        out = redact_args(value, max_items=10_000)
        assert time.perf_counter() - start < TIME_LIMIT
        assert "abc5" not in jsonable(out)

    def test_huge_key(self):
        start = time.perf_counter()
        jsonable(redact_args({"k" * 1_000_000: 1}))
        assert time.perf_counter() - start < TIME_LIMIT


# ─── Secrets never survive: a broad leak check ───────────────────────────────

class TestNoLeaks:
    SECRET = "Zx9-very-secret-1234"

    def test_secret_under_many_keys_and_shapes(self):
        s = self.SECRET
        value = {
            "password": s, "Authorization": f"Bearer {s}", "x-api-key": s,
            "nested": {"client_secret": [s, (s,), {"a": s}]},
            "list": [{"token": s}, {"cookie": s}, {"session": {"id": s}}],
            "headers": {"Set-Cookie": s, "Proxy-Authorization": s},
            "url": f"https://user:{s}@example.com/x",
            "cmd": f"tool --token={s} --password={s}",
            "env": {"AWS_SECRET_ACCESS_KEY": s, "DB_PASSWORD": s, "SERVICE_KEY": s},
            "bytes": s.encode(),
            "obj": type("Holder", (), {"v": s})(),
            1: {"api_key": s},
        }
        text = jsonable(redact_args(value))
        assert s not in text

    def test_secret_as_set_member_or_tuple_under_secret_key(self):
        assert self.SECRET not in jsonable(redact_args({"secrets": {self.SECRET}}))
        assert self.SECRET not in jsonable(redact_args({"passwords": (self.SECRET,)}))


# ─── ArgsDigester ─────────────────────────────────────────────────────────────

class TestArgsDigester:
    def test_digest_is_64_hex_chars(self):
        digest = ArgsDigester().digest({"a": 1})
        assert len(digest) == 64
        int(digest, 16)
        assert digest == digest.lower()

    def test_same_input_same_digest(self):
        d = ArgsDigester()
        assert d.digest({"a": [1, 2], "b": "x"}) == d.digest({"a": [1, 2], "b": "x"})

    def test_key_order_does_not_matter(self):
        d = ArgsDigester()
        assert d.digest({"a": 1, "b": {"c": 2, "d": 3}}) == d.digest({"b": {"d": 3, "c": 2}, "a": 1})

    def test_different_input_different_digest(self):
        d = ArgsDigester()
        values = [{}, [], None, 0, "0", {"a": 1}, {"a": 2}, {"b": 1}, [1, 2], [2, 1], "a", True,
                  False, 1.5, {"a": [1]}, {"a": "1"}]
        digests = [d.digest(v) for v in values]
        assert len(set(digests)) == len(values)

    def test_two_digesters_differ(self):
        assert ArgsDigester().digest({"a": 1}) != ArgsDigester().digest({"a": 1})

    def test_canonical_form_is_exact(self):
        d = ArgsDigester()
        expected = hmac.new(
            d._key, b'{"a":1,"b":[1,2],"c":"\\u00e9"}', hashlib.sha256
        ).hexdigest()
        assert d.digest({"c": "é", "b": (1, 2), "a": 1}) == expected

    def test_it_is_a_keyed_digest_not_a_plain_hash(self):
        d = ArgsDigester()
        plain = hashlib.sha256(b'{"pin":"1234"}').hexdigest()
        assert d.digest({"pin": "1234"}) != plain
        # an attacker who guesses the PIN cannot confirm it without the key
        assert ArgsDigester().digest({"pin": "1234"}) != d.digest({"pin": "1234"})

    def test_key_is_32_bytes_and_not_printed(self):
        d = ArgsDigester()
        assert isinstance(d._key, bytes) and len(d._key) == 32
        for text in (repr(d), str(d), f"{d}", format(d)):
            assert d._key.hex() not in text
            assert str(d._key) not in text
        assert "key" not in repr(d).lower().replace("argsdigester", "")
        assert not hasattr(d, "__dict__")

    def test_key_is_not_reachable_by_pickling_or_copying(self):
        d = ArgsDigester()
        with pytest.raises(TypeError):
            pickle.dumps(d)
        with pytest.raises(TypeError):
            copy.copy(d)
        with pytest.raises(TypeError):
            copy.deepcopy(d)

    @pytest.mark.parametrize("value", [
        b"bytes", {1, 2}, object(), float("nan"), float("inf"), -float("inf"), {1: "a"},
        {None: 1}, {(1, 2): 1}, [object()], {"a": {b"x"}}, 1j, ArgsDigester, Path("x"),
        frozenset(), {"a": {1: 2}}, [[{2: 3}]],
    ])
    def test_non_json_values_are_rejected_with_type_error(self, value):
        with pytest.raises(TypeError):
            ArgsDigester().digest(value)

    def test_int_too_large_to_print_is_rejected_with_type_error(self):
        with pytest.raises(TypeError):
            ArgsDigester().digest(10 ** 5000)

    def test_cycles_are_rejected_with_type_error(self):
        a: list = []
        a.append(a)
        d: dict = {}
        d["me"] = d
        digester = ArgsDigester()
        with pytest.raises(TypeError):
            digester.digest(a)
        with pytest.raises(TypeError):
            digester.digest(d)

    def test_very_deep_nesting_is_rejected_with_type_error(self):
        root: list = []
        cur = root
        for _ in range(10_000):
            nxt: list = []
            cur.append(nxt)
            cur = nxt
        start = time.perf_counter()
        with pytest.raises(TypeError):
            ArgsDigester().digest(root)
        assert time.perf_counter() - start < TIME_LIMIT

    def test_hostile_objects_give_type_error_not_other_errors(self):
        digester = ArgsDigester()
        for value in (RaisesEverything(), BadClass(), [RaisesEverything()], {"a": WithSlots()}):
            with pytest.raises(TypeError):
                digester.digest(value)

    def test_error_message_does_not_contain_the_value(self):
        with pytest.raises(TypeError) as info:
            ArgsDigester().digest({"password": object(), "x": "hunter2"})
        assert "hunter2" not in str(info.value)

    def test_unicode_and_ascii_escaping(self):
        d = ArgsDigester()
        assert d.digest({"k": "é中"}) == d.digest({"k": "é中"})
        assert d.digest({"k": "é"}) != d.digest({"k": "e"})
        d.digest({"k": "\ud800"})   # a lone surrogate is escaped, not an error

    def test_str_and_int_keys_are_not_confused(self):
        d = ArgsDigester()
        with pytest.raises(TypeError):
            d.digest({1: "a"})
        assert d.digest({"1": "a"}) != d.digest({"2": "a"})

    def test_json_scalars_and_big_inputs(self):
        d = ArgsDigester()
        for value in (None, True, 0, -1, 2 ** 100, 1.5, "", "x" * 1_000_000, [], {}):
            assert len(d.digest(value)) == 64
        start = time.perf_counter()
        d.digest({str(i): list(range(50)) for i in range(2000)})
        assert time.perf_counter() - start < TIME_LIMIT

    def test_bool_and_int_differ(self):
        d = ArgsDigester()
        assert d.digest(True) != d.digest(1)
        assert d.digest(1) != d.digest(1.0)

    def test_default_digester_is_lazy_and_shared(self, monkeypatch):
        monkeypatch.setattr(redact_module, "_default_digester", None)
        first = default_digester()
        assert isinstance(first, ArgsDigester)
        assert default_digester() is first
        assert first.digest({"a": 1}) == default_digester().digest({"a": 1})

    def test_default_digester_is_created_once_across_threads(self, monkeypatch):
        monkeypatch.setattr(redact_module, "_default_digester", None)
        seen: list = []
        barrier = threading.Barrier(8)

        def grab():
            barrier.wait()
            seen.append(default_digester())

        threads = [threading.Thread(target=grab) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(seen) == 8 and all(item is seen[0] for item in seen)

    def test_default_digester_differs_from_a_fresh_one(self):
        assert default_digester().digest({"a": 1}) != ArgsDigester().digest({"a": 1})


# ─── Public API ───────────────────────────────────────────────────────────────

def test_public_api():
    assert set(redact_module.__all__) == {
        "MAX_ERROR_TEXT", "REDACTED", "ArgsDigester", "default_digester",
        "redact_args", "sanitize_error_text",
    }
    assert REDACTED == "[REDACTED]"
    for name in redact_module.__all__:
        assert hasattr(redact_module, name)
