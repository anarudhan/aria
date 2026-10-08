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
import unicodedata
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
        assert out["<tuple>"] == REDACTED          # a key we cannot read: fail closed
        assert out["<bytes 1>"] == "e" and out["2.5"] == "f"
        jsonable(out)

    def test_object_key_is_never_stringified(self):
        class Key:
            def __str__(self):
                raise RuntimeError("str")

            def __repr__(self):
                raise RuntimeError("repr")

        out = redact_args({Key(): "v"})
        assert out == {"<Key>": REDACTED}


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
        assert redact_args({RaisesEverything(): 1}) == {"<RaisesEverything>": REDACTED}

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
            d._key, b'\x00\x00\x00\x04args{"a":1,"b":[1,2],"c":"\\u00e9"}', hashlib.sha256
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


# ═══════════════════════════════════════════════════════════════════════════════
# Review round (issue #58): gaps found by the independent security review
# ═══════════════════════════════════════════════════════════════════════════════

ZWSP = chr(0x200B)          # zero width space (category Cf)
SHY = chr(0xAD)             # soft hyphen (category Cf)
CYR_O = chr(0x043E)         # Cyrillic small o, looks like Latin o
LONE_HIGH = chr(0xD83D)     # the two halves of a surrogate pair, built with chr() on purpose
LONE_LOW = chr(0xDE00)
EMOJI = chr(0x1F600)
FDFA = chr(0xFDFA)          # NFKC turns this one character into 18


# ─── Item 1: digest work budget ───────────────────────────────────────────────

class TestDigestBudget:
    def test_exponential_sharing_stops_at_the_budget(self):
        x: object = ["a"]
        for _ in range(25):
            x = [x, x]
        start = time.perf_counter()
        with pytest.raises(TypeError):
            ArgsDigester().digest(x)
        assert time.perf_counter() - start < TIME_LIMIT

    def test_dict_sharing_stops_at_the_budget(self):
        x: object = {"a": 1}
        for _ in range(40):
            x = {"l": x, "r": x}
        with pytest.raises(TypeError):
            ArgsDigester().digest(x)

    def test_moderate_sharing_still_works(self):
        x: object = ["a"]
        for _ in range(10):
            x = [x, x]
        assert len(ArgsDigester().digest(x)) == 64

    def test_every_visit_is_counted(self, monkeypatch):
        monkeypatch.setattr(redact_module, "_DIGEST_MAX_NODES", 50)
        shared = [1, 2, 3]
        ArgsDigester().digest([shared] * 10)             # 1 + 10 + 30 = 41 values
        with pytest.raises(TypeError):
            ArgsDigester().digest([shared] * 20)        # 1 + 20 + 60 = 81 values

    def test_character_budget_counts_every_visit(self, monkeypatch):
        monkeypatch.setattr(redact_module, "_DIGEST_MAX_CHARS", 1000)
        text = "x" * 400
        ArgsDigester().digest([text, text])
        with pytest.raises(TypeError):
            ArgsDigester().digest([text, text, text])
        with pytest.raises(TypeError):
            ArgsDigester().digest({"k" * 1001: 1})

    def test_depth_limit(self):
        ok: object = 1
        for _ in range(redact_module._DIGEST_MAX_DEPTH):
            ok = [ok]
        assert len(ArgsDigester().digest(ok)) == 64
        with pytest.raises(TypeError):
            ArgsDigester().digest([ok])

    def test_large_honest_content_is_accepted(self):
        content = {"path": "/tmp/a.txt", "content": "line of text\n" * 400_000}   # about 5 MB
        start = time.perf_counter()
        assert len(ArgsDigester().digest(content)) == 64
        assert time.perf_counter() - start < TIME_LIMIT

    def test_a_huge_flat_list_is_rejected_quickly(self):
        start = time.perf_counter()
        with pytest.raises(TypeError):
            ArgsDigester().digest([0] * 300_000)
        assert time.perf_counter() - start < TIME_LIMIT

    def test_message_has_no_value(self):
        x: object = ["hunter2-marker"]
        for _ in range(25):
            x = [x, x]
        with pytest.raises(TypeError) as info:
            ArgsDigester().digest(x)
        assert "hunter2" not in str(info.value)

    def test_limits_are_documented_constants(self):
        assert redact_module._DIGEST_MAX_NODES == 200_000
        assert redact_module._DIGEST_MAX_CHARS == 16_000_000
        assert redact_module._DIGEST_MAX_DEPTH == 100
        text = (Path(__file__).resolve().parent.parent / "docs" / "redaction.md").read_text(
            encoding="utf-8")
        for number in ("200,000", "16,000,000", "100 levels"):
            assert number in text


# ─── Item 2: only exact types are hashed ──────────────────────────────────────

class TestDigestExactTypes:
    def test_dict_subclass_with_its_own_items(self):
        class D2(dict):
            def items(self):
                return [(1, "a")]

        digester = ArgsDigester()
        with pytest.raises(TypeError):
            digester.digest(D2(x=1))

    def test_list_subclass_with_its_own_iter(self):
        class L2(list):
            def __iter__(self):
                return iter([9, 9, 9])

        with pytest.raises(TypeError):
            ArgsDigester().digest(L2([1, 2]))

    @pytest.mark.parametrize("make", [
        lambda: type("S", (str,), {})("x"),
        lambda: type("I", (int,), {})(1),
        lambda: type("F", (float,), {})(1.5),
        lambda: type("T", (tuple,), {})((1,)),
        lambda: type("D", (dict,), {})(),
        lambda: type("L", (list,), {})(),
        lambda: collections.OrderedDict(a=1),
        lambda: collections.defaultdict(list),
        lambda: __import__("enum").IntEnum("E", "A")["A"],
    ])
    def test_subclasses_are_rejected_everywhere(self, make):
        digester = ArgsDigester()
        with pytest.raises(TypeError):
            digester.digest(make())
        with pytest.raises(TypeError):
            digester.digest([make()])
        with pytest.raises(TypeError):
            digester.digest({"k": make()})

    def test_str_subclass_as_a_key_is_rejected(self):
        class K(str):
            pass

        with pytest.raises(TypeError):
            ArgsDigester().digest({K("a"): 1})

    def test_no_user_code_runs(self):
        calls = []

        class Spy(dict):
            def __getattribute__(self, name):
                calls.append(name)
                return dict.__getattribute__(self, name)

            def __iter__(self):
                calls.append("iter")
                return dict.__iter__(self)

            def __len__(self):
                calls.append("len")
                return 0

        with pytest.raises(TypeError):
            ArgsDigester().digest({"a": [Spy(x=1)]})
        assert calls == []

    def test_exact_types_still_work(self):
        digester = ArgsDigester()
        value = {"a": 1, "b": 2.5, "c": True, "d": None, "e": "x", "f": [1, (2, 3)]}
        assert len(digester.digest(value)) == 64

    def test_digest_is_of_the_inspected_copy(self):
        class Sneaky(list):
            pass

        digester = ArgsDigester()
        with pytest.raises(TypeError):
            digester.digest(Sneaky([1]))
        assert digester.digest([1]) == digester.digest((1,))


# ─── Item 3: surrogates ───────────────────────────────────────────────────────

class TestSurrogates:
    def test_pair_and_emoji_no_longer_collide(self):
        digester = ArgsDigester()
        pair = LONE_HIGH + LONE_LOW
        assert len(digester.digest(EMOJI)) == 64
        with pytest.raises(TypeError):
            digester.digest(pair)

    @pytest.mark.parametrize("bad", [LONE_HIGH, LONE_LOW, "a" + LONE_HIGH + "b",
                                     LONE_HIGH + LONE_LOW])
    def test_digest_rejects_surrogates_in_text_and_keys(self, bad):
        digester = ArgsDigester()
        with pytest.raises(TypeError):
            digester.digest(bad)
        with pytest.raises(TypeError):
            digester.digest({"k": [bad]})
        with pytest.raises(TypeError):
            digester.digest({bad: 1})

    def test_digest_message_has_no_value(self):
        with pytest.raises(TypeError) as info:
            ArgsDigester().digest("secret-" + LONE_HIGH)
        assert "secret" not in str(info.value)

    def test_redact_args_replaces_lone_surrogates(self):
        bad = "x" + LONE_HIGH + "y" + LONE_LOW
        out = redact_args({"v": bad, bad: 1, "list": [bad]})
        text = jsonable(out)
        json.dumps(out, ensure_ascii=False).encode("utf-8")      # must not raise
        assert out["v"] == "x�y�"
        assert "x�y�" in out
        assert LONE_HIGH not in text

    def test_real_emoji_is_kept(self):
        assert redact_args(EMOJI) == EMOJI

    def test_redact_output_can_be_written_as_utf8(self):
        value = [LONE_HIGH, {LONE_LOW: LONE_HIGH + LONE_LOW}, "ok"]
        json.dumps(redact_args(value), ensure_ascii=False).encode("utf-8")


# ─── Item 4: NFKC expansion ───────────────────────────────────────────────────

class TestNfkcExpansion:
    def test_redact_args_with_expanding_strings_is_fast(self):
        start = time.perf_counter()
        out = redact_args([FDFA * 200_000] * 5)
        assert time.perf_counter() - start < TIME_LIMIT
        assert all(item.startswith("<string,") for item in out)
        jsonable(out)

    def test_expanding_string_with_a_secret_does_not_leak(self):
        for text in (FDFA * 200_000 + "password=hunter2",
                     "password=hunter2 " + FDFA * 200_000,
                     FDFA * 5000 + "password=hunter2"):
            out = redact_args({"s": text, "t": [text, text]})
            assert "hunter2" not in jsonable(out)

    def test_small_expanding_text_is_still_scanned(self):
        out = redact_args(FDFA * 3 + " password=hunter2")
        assert "hunter2" not in out

    def test_sanitize_error_text_is_bounded_too(self):
        for text in (FDFA * 200_000, FDFA * 1_000_000, "a" * 5_000_000):
            start = time.perf_counter()
            out = sanitize_error_text(text)
            assert time.perf_counter() - start < TIME_LIMIT
            assert out.startswith("<text,") and "not scanned" in out

    def test_sanitize_error_text_normal_text_is_unchanged(self):
        assert sanitize_error_text("plain text") == "plain text"
        text = "café 中文 " * 1000
        assert sanitize_error_text(text, limit=10 ** 7) == text
        big = "x" * 900_000 + " password=hunter2"
        assert "hunter2" not in sanitize_error_text(big)

    def test_chunked_normalisation_equals_plain_normalisation(self):
        import unicodedata

        # a combining mark exactly where a slice would end must still compose
        for prefix in (9_999, 10_000, 10_001, 19_999, 20_000):
            text = "a" * prefix + "é" + "Ａ" * 5 + "b" * 30_000
            assert (sanitize_error_text(text, limit=10 ** 7)
                    == unicodedata.normalize("NFKC", text))

    def test_many_expanding_strings_share_the_budget(self):
        start = time.perf_counter()
        out = redact_args([FDFA * 10_000 for _ in range(100)])
        assert time.perf_counter() - start < TIME_LIMIT
        jsonable(out)


# ─── Item 5 and 11: key classification fails closed ──────────────────────────

REVIEW_SECRET_KEYS = [
    "accesstoken", "ACCESSTOKEN", "authtoken", "refreshtoken", "csrftoken", "idtoken",
    "dbpwd", "setcookie", "xauth", "to" + ZWSP + "ken",
    "to" + SHY + "ken", "t" + CYR_O + "ken", "pass" + ZWSP + "word", "api" + SHY + "_key",
    "t\x00oken", "se‍cret", "tóken",
    "key_data", "key_material", "keyData", "KeyMaterial", "privkey", "private_pem",
    "privatePem", "signature", "x-amz-signature", "X-Amz-Signature", "sig", "otp", "pin",
    "seed", "mnemonic", "totp", "x-amz-security-token", "pincode",
    # Greek look-alikes
    "τοken", "pαssword", "sεcret",
    "k" * 1001, "a" * 1500 + "_x", "x" * 5000,
]


class TestKeyClassificationFailsClosed:
    @pytest.mark.parametrize("key", REVIEW_SECRET_KEYS, ids=lambda k: ascii(k)[:40])
    def test_secret_spellings_are_redacted(self, key):
        assert list(redact_args({key: "value-123"}).values()) == [REDACTED]

    @pytest.mark.parametrize("key", ["monkey", "author", "authority", "tokenizer", "bypass",
                                     "passage", "compass", "path", "query", "keyboard",
                                     "tokenize", "tokenizer_name", "Author", "design",
                                     "spin", "pinned", "seeded", "signal", "single",
                                     "été", "中文"])
    def test_plain_keys_are_kept(self, key):
        assert list(redact_args({key: "visible"}).values()) == ["visible"]

    def test_long_key_is_secret_not_truncated(self):
        # the old code kept the first and last 500 characters and dropped the middle
        key = "a" * 600 + "token" + "b" * 600
        assert list(redact_args({key: "v"}).values()) == [REDACTED]
        key = "a" * 600 + "visible" + "b" * 600     # over 1000 characters: secret as well
        assert list(redact_args({key: "v"}).values()) == [REDACTED]

    def test_exactly_1000_chars_is_still_classified(self):
        assert list(redact_args({"k" * 1000: "v"}).values()) == ["v"]
        assert list(redact_args({"k" * 1001: "v"}).values()) == [REDACTED]

    def test_many_long_keys_run_out_of_budget_and_fail_closed(self):
        start = time.perf_counter()
        out = redact_args({("k%d" % i) + "x" * 900: "visible" for i in range(2000)},
                          max_items=2000)
        assert time.perf_counter() - start < TIME_LIMIT
        assert REDACTED in out.values()          # no budget left to classify: secret


# ─── Item 6: name/value and pair shapes ───────────────────────────────────────

class TestNameValueShapes:
    def test_env_list(self):
        out = redact_args({"env": [{"name": "DB_PASSWORD", "value": "hunter2"},
                                   {"name": "LOG_LEVEL", "value": "debug"}]})
        assert out == {"env": [{"name": "DB_PASSWORD", "value": REDACTED},
                               {"name": "LOG_LEVEL", "value": "debug"}]}

    @pytest.mark.parametrize("entry", [
        {"Name": "Authorization", "Value": "Basic abc"},
        {"NAME": "x-api-key", "VALUE": "abc"},
        {"key": "client_secret", "value": "abc"},
        {"Key": "session_id", "Value": "abc"},
        {"value": "abc", "name": "PASSWORD"},
        {"name": "ACCESSTOKEN", "value": {"nested": "abc"}},
        {"name": "token", "value": ["abc"], "other": 1},
    ])
    def test_name_entry_hides_value_entry(self, entry):
        out = redact_args(entry)
        assert [v for k, v in out.items() if k.lower() == "value"] == [REDACTED]
        assert "abc" not in jsonable(out)

    def test_other_entries_are_kept(self):
        out = redact_args({"name": "DB_PASSWORD", "value": "x", "description": "visible"})
        assert out["description"] == "visible"

    def test_plain_name_keeps_value(self):
        assert redact_args({"name": "colour", "value": "blue"}) == {"name": "colour",
                                                                    "value": "blue"}

    def test_value_without_a_secret_name_is_kept(self):
        assert redact_args({"value": "x"}) == {"value": "x"}
        assert redact_args({"label": "password", "value": "x"}) == {"label": "password",
                                                                    "value": "x"}

    def test_value_is_never_read(self):
        class Boom:
            def __getattribute__(self, name):
                raise AssertionError("touched")

        out = redact_args({"name": "TOKEN", "value": Boom()})
        assert out["value"] == REDACTED

    def test_header_pairs(self):
        out = redact_args({"headers": [("Authorization", "Basic dXNlcjpwYXNz"),
                                       ("Cookie", "sid=abc"), ("Accept", "text/html")]})
        assert out == {"headers": [["Authorization", REDACTED], ["Cookie", REDACTED],
                                   ["Accept", "text/html"]]}

    def test_pair_as_list_and_top_level(self):
        assert redact_args(["password", "hunter2"]) == ["password", REDACTED]
        assert redact_args(("X-Api-Key", {"deep": "hunter2"})) == ["X-Api-Key", REDACTED]
        assert redact_args(["--password", "hunter2"]) == ["--password", REDACTED]

    def test_pair_second_item_is_never_read(self):
        class Boom:
            def __getattribute__(self, name):
                raise AssertionError("touched")

        assert redact_args([("token", Boom())]) == [["token", REDACTED]]

    def test_longer_lists_are_not_pairs(self):
        # documented limit: an argv style list is not recognised
        assert redact_args(["tool", "--password", "hunter2"]) == ["tool", "--password", "hunter2"]
        assert redact_args(["password"]) == ["password"]
        assert redact_args({"password", "x"}) in (["password", "x"], ["x", "password"])

    def test_pair_with_non_secret_first_item(self):
        assert redact_args([("Accept", "x"), ("a", "b")]) == [["Accept", "x"], ["a", "b"]]

    def test_pair_with_long_first_item_is_not_a_name(self):
        text = "password " * 100
        assert redact_args([text, "visible"], max_string=2000)[1] == "visible"


# ─── Item 7: text shapes ──────────────────────────────────────────────────────

class TestTextShapes:
    PEM = ("-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA7hunter2xyz\nabcdef\n"
           "-----END RSA PRIVATE KEY-----")

    @pytest.mark.parametrize("name", ["RSA ", "EC ", "OPENSSH ", "ENCRYPTED ", ""])
    def test_pem_private_keys(self, name):
        pem = f"-----BEGIN {name}PRIVATE KEY-----\nMIIEhunter2body\n-----END {name}PRIVATE KEY-----"
        out = sanitize_error_text(f"key: {pem} done")
        assert "hunter2" not in out and out.endswith("done")
        assert "MIIE" not in out

    def test_pem_without_end_goes_to_the_end(self):
        out = sanitize_error_text("x -----BEGIN PRIVATE KEY-----\nMIIEhunter2 and more text")
        assert "hunter2" not in out and "more text" not in out

    def test_pem_in_redact_args_within_the_first_200_chars(self):
        out = redact_args({"note": "ssh: " + self.PEM})
        assert "hunter2" not in jsonable(out)

    def test_pem_escaped_newlines(self):
        text = "-----BEGIN PRIVATE KEY-----\\nMIIEhunter2\\n-----END PRIVATE KEY-----"
        assert "hunter2" not in sanitize_error_text(text)

    def test_public_key_is_not_touched(self):
        text = "-----BEGIN PUBLIC KEY-----\nMIIB\n-----END PUBLIC KEY-----"
        assert sanitize_error_text(text) == text

    @pytest.mark.parametrize("text, secret", [
        ("x-amz-signature=abcdef123456&y=1", "abcdef123456"),
        ("X-Amz-Security-Token=zzzTOKENzzz&y=1", "zzzTOKENzzz"),
        ("https://b.s3.amazonaws.com/o?Signature=abcdef123456&Expires=1", "abcdef123456"),
        ("?sig=abcdef123456&se=2030", "abcdef123456"),
        ('{"signature": "abcdef 123456"}', "abcdef"),
        ("DB_PASS=hunter2 port=1", "hunter2"),
        ("db_pass: hunter2", "hunter2"),
        ("pwd=hunter2&x=1", "hunter2"),
        ("PWD: hunter2", "hunter2"),
        ("redis://:hunter2@host:6379/0", "hunter2"),
        ("amqp://user:hunter2@host", "hunter2"),
    ])
    def test_more_shapes(self, text, secret):
        assert secret not in sanitize_error_text(text)

    def test_whole_word_labels_do_not_overreach(self):
        for text in ("bypass=1", "design=blue", "compass=north", "assign=x", "sigma=2",
                     "http://host:8080/x", "http://user@host"):
            assert sanitize_error_text(text) == text

    @pytest.mark.parametrize("text, secret", [
        ('password="ab\\"cd ef" tail', "cd ef"),
        ("password='ab\\'cd ef' tail", "cd ef"),
        (json.dumps({"password": 'p"w d'}) + " tail", "w d"),
        (json.dumps({"api_key": 'a"b c"d e'}), "d e"),
        ('secret="x\\\\" y="z" tail', "x"),
        ('{\\"password\\": \\"p w\\"} tail', "p w"),
        ('cookie: "a\\"b c"', "b c"),
        ('Authorization: "Bearer a\\"b c"', "b c"),
    ])
    def test_quoted_values_with_escapes(self, text, secret):
        out = sanitize_error_text(text)
        assert secret not in out
        assert "tail" in out or "tail" not in text

    def test_text_after_the_quoted_value_is_kept(self):
        assert sanitize_error_text('password="a\\"b c" next=1') == "password=[REDACTED] next=1"

    def test_still_stable_when_applied_twice(self):
        for text in (self.PEM, 'password="ab\\"cd ef" tail', "redis://:hunter2@h", "pwd=x sig=y"):
            once = sanitize_error_text(text)
            assert sanitize_error_text(once) == once

    @pytest.mark.parametrize("unit", [
        'token="a\\\\', "token='\\'", 'password="' + "\\a", "-----BEGIN PRIVATE KEY-----",
        "-----BEGIN ", "pass=", 'sig="\\"', "Signature:\\", "redis://:", "cookie: \\\"",
        "authorization: \\\"a b ", "x\\\\\\\"y",
    ])
    def test_token_soup_is_fast(self, unit):
        text = unit * (200_000 // len(unit))
        for _ in range(5):
            start = time.perf_counter()
            sanitize_error_text(text)
            assert time.perf_counter() - start < TIME_LIMIT
        start = time.perf_counter()
        redact_args([text] * 5)
        assert time.perf_counter() - start < TIME_LIMIT


# ─── Item 8: hostile metaclass ────────────────────────────────────────────────

class TestHostileMetaclass:
    def test_metaclass_name_property_is_not_run(self):
        calls = []

        class Meta(type):
            @property
            def __name__(cls):  # type: ignore[override]
                calls.append("name")
                return "ATTACKER-password=hunter2"

            def __getattribute__(cls, name):
                if name == "__name__":
                    calls.append("getattr")
                    raise RuntimeError("attacker code")
                return type.__getattribute__(cls, name)

            def __instancecheck__(cls, obj):
                calls.append("instancecheck")
                raise RuntimeError("attacker code")

            def __subclasscheck__(cls, sub):
                calls.append("subclasscheck")
                raise RuntimeError("attacker code")

        class Victim(metaclass=Meta):
            pass

        out = redact_args({"x": Victim(), "y": [Victim()], Victim(): 1})
        assert calls == []
        assert out["x"] == "<Victim>" and out["y"] == ["<Victim>"]
        assert "ATTACKER" not in jsonable(out) and "hunter2" not in jsonable(out)

    def test_class_with_a_secret_in_its_real_name_is_redacted(self):
        odd = type("password=hunter2", (), {})
        out = redact_args(odd())
        assert "hunter2" not in out

    def test_very_long_class_name_is_cut(self):
        odd = type("N" * 5000, (), {})
        assert len(redact_args(odd())) < 100


# ─── Item 9: non-string keys ──────────────────────────────────────────────────

class TestNonStringKeys:
    @pytest.mark.parametrize("key", [b"password", b"API_KEY", type("B", (bytes,), {})(b"token"),
                                     b"x-auth-token", b"pass\x00word"],
                             ids=lambda k: type(k).__name__ + repr(bytes(k)))
    def test_bytes_keys_are_classified(self, key):
        out = redact_args({key: "hunter2"})
        assert list(out.values()) == [REDACTED]
        assert "hunter2" not in jsonable(out)

    def test_plain_bytes_key_is_kept(self):
        assert redact_args({b"name": "x"}) == {"<bytes 4>": "x"}

    def test_huge_bytes_key_fails_closed(self):
        assert list(redact_args({b"k" * 2000: "x"}).values()) == [REDACTED]

    @pytest.mark.parametrize("key", [("password",), frozenset({"a"}), (1, 2), object(),
                                     RaisesEverything(), WithSlots(), len, 1 + 2j],
                             ids=lambda k: type(k).__name__)
    def test_other_keys_redact_the_value(self, key):
        out = redact_args({key: "hunter2"})
        assert list(out.values()) == [REDACTED]

    @pytest.mark.parametrize("key", [1, 2.5, True, None, 10 ** 400])
    def test_number_and_none_keys_keep_their_value(self, key):
        out = redact_args({key: "visible"})
        assert list(out.values()) == ["visible"]

    def test_key_object_methods_never_run(self):
        calls = []

        class K:
            def __str__(self):
                calls.append("str")
                return "password"

            def __repr__(self):
                calls.append("repr")
                return "password"

            def __hash__(self):
                return 1

            def __eq__(self, other):
                return self is other

        redact_args({K(): "x"})
        assert calls == []


# ─── Item 12 and 13: purpose and matches ──────────────────────────────────────

class TestPurposeAndMatches:
    def test_purpose_separates_digests(self):
        d = ArgsDigester()
        assert d.digest({"a": 1}) == d.digest({"a": 1}, "args")
        assert d.digest({"a": 1}, "args") != d.digest({"a": 1}, "result")
        assert d.digest({"a": 1}, purpose="result") == d.digest({"a": 1}, "result")

    def test_length_prefix_makes_the_encoding_unambiguous(self):
        # without a length prefix "ab" + "c" and "a" + "bc" would be the same message
        d = ArgsDigester()
        assert d.digest("c", "ab") != d.digest("bc", "a")
        assert d.digest(1, "a1") != d.digest(11, "a")

    def test_known_answer_with_purpose(self):
        d = ArgsDigester()
        expected = hmac.new(d._key, b'\x00\x00\x00\x06result[1,2]', hashlib.sha256).hexdigest()
        assert d.digest([1, 2], "result") == expected

    @pytest.mark.parametrize("bad", ["", "1abc", "a b", "a" * 33, "café", "a\n", "-a", "a/b"])
    def test_bad_purpose_values(self, bad):
        with pytest.raises(ValueError):
            ArgsDigester().digest({}, bad)

    @pytest.mark.parametrize("bad", [None, 5, b"args", ["args"]])
    def test_non_string_purpose(self, bad):
        with pytest.raises(TypeError):
            ArgsDigester().digest({}, bad)

    def test_good_purposes(self):
        d = ArgsDigester()
        for good in ("args", "result", "a", "tool.args-v1", "A_b-c.d", "x" * 32):
            assert len(d.digest({}, good)) == 64

    def test_matches(self):
        d = ArgsDigester()
        value = {"a": [1, 2]}
        digest = d.digest(value)
        assert d.matches(value, digest) is True
        assert d.matches(value, digest.upper()) is True
        assert d.matches({"a": [1, 3]}, digest) is False
        assert d.matches(value, digest, "result") is False
        assert d.matches(value, d.digest(value, "result"), "result") is True
        assert ArgsDigester().matches(value, digest) is False

    @pytest.mark.parametrize("bad", [None, 5, b"x" * 64, "", "abc", "z" * 64, "a" * 63,
                                     "a" * 65, "é" * 64, ["a"] * 64, 1.5, object()])
    def test_matches_never_raises_on_malformed_digests(self, bad):
        assert ArgsDigester().matches({"a": 1}, bad) is False

    def test_matches_with_values_digest_rejects(self):
        d = ArgsDigester()
        for value in (b"x", {1}, object(), float("nan"), RaisesEverything()):
            assert d.matches(value, "a" * 64) is False
        assert d.matches({}, "a" * 64, "bad purpose!") is False

    def test_matches_uses_constant_time_compare(self, monkeypatch):
        calls = []
        real = hmac.compare_digest
        monkeypatch.setattr(redact_module.hmac, "compare_digest",
                            lambda a, b: calls.append(1) or real(a, b))
        d = ArgsDigester()
        d.matches({}, d.digest({}))
        assert calls == [1]


# ─── Item 14: tests that kill surviving mutants ───────────────────────────────

class TestHardLimitsAndFailureHandling:
    def test_max_depth_is_capped_at_64(self):
        def nest(levels):
            value: object = "leaf"
            for _ in range(levels):
                value = [value]
            return value

        assert "leaf" in jsonable(redact_args(nest(64), max_depth=10 ** 9))
        assert "leaf" not in jsonable(redact_args(nest(65), max_depth=10 ** 9))
        assert "<max depth>" in jsonable(redact_args(nest(65), max_depth=10 ** 9))
        assert "<max depth>" in jsonable(redact_args(nest(100), max_depth=65))

    def test_max_items_is_capped(self):
        out = redact_args(list(range(20_000)), max_items=10 ** 9)
        assert len(out) == 10_001

    def test_a_container_that_fails_becomes_unreadable(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("attacker-text")

        monkeypatch.setattr(redact_module, "_key_info", boom)
        out = redact_args({"outer": [{"a": 1}], "b": 2})
        assert out == "<dict, unreadable>"
        assert "attacker" not in jsonable(out)

    def test_a_failing_list_becomes_unreadable_inside_a_dict(self, monkeypatch):
        real = redact_module._clean_str

        def flaky(text, limit, budget):
            if text == "boom":
                raise RuntimeError("x")
            return real(text, limit, budget)

        monkeypatch.setattr(redact_module, "_clean_str", flaky)
        out = redact_args({"ok": ["fine"], "bad": ["boom"]})
        assert out == {"ok": ["fine"], "bad": "<list, unreadable>"}

    def test_top_level_failure_returns_a_marker(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("attacker-text")

        monkeypatch.setattr(redact_module, "_Budget", boom)
        assert redact_args({"a": 1}) == "<redaction failed>"
        monkeypatch.undo()
        assert redact_args({"a": 1}) == {"a": 1}

    def test_top_level_failure_from_a_bad_parameter_path(self, monkeypatch):
        monkeypatch.setattr(redact_module, "_redact", lambda *a, **k: 1 / 0)
        assert redact_args([1]) == "<redaction failed>"

    def test_dict_mutated_while_it_is_copied(self, monkeypatch):
        victim: dict = {"a": "one", "b": "two", "c": "three"}
        real = redact_module._clean_str
        state = {"done": False}

        def mutating(text, limit, budget):
            if not state["done"] and text == "one":
                state["done"] = True
                victim["new-key"] = 1            # changes the size of the dict being copied
            return real(text, limit, budget)

        monkeypatch.setattr(redact_module, "_clean_str", mutating)
        out = redact_args(victim)
        jsonable(out)                            # no exception, JSON safe

    def test_set_and_list_mutated_while_copied(self, monkeypatch):
        items = {"one", "two", "three"}
        real = redact_module._clean_str
        state = {"done": False}

        def mutating(text, limit, budget):
            if not state["done"]:
                state["done"] = True
                items.add("added-while-iterating")
            return real(text, limit, budget)

        monkeypatch.setattr(redact_module, "_clean_str", mutating)
        jsonable(redact_args({"s": items}))


# ═══════════════════════════════════════════════════════════════════════════════
# Final review round (issue #58)
# ═══════════════════════════════════════════════════════════════════════════════

COMBINING_ACUTE = chr(0x301)
DOT_ABOVE = chr(0x307)
CYR_A = chr(0x0430)
CYR_E = chr(0x0435)
GR_O = chr(0x03BF)
S_DOT = chr(0x1E61)          # s with dot above, what NFKC makes of "s" + DOT_ABOVE
HELLO_RU = "".join(chr(c) for c in (0x41F, 0x440, 0x438, 0x432, 0x435, 0x442))   # Cyrillic word


# ─── S1: split and look-alike labels in text ─────────────────────────────────

SPLIT_LABELS = [
    "pa" + ZWSP + "ssword=hunter2",
    "secret" + ZWSP + "=hunter2",
    "pas" + S_DOT + "word=hunter2",
    "pas" + "s" + DOT_ABOVE + "word=hunter2",
    "tok" + CYR_E + "n=hunter2",
    "p" + CYR_A + "ssword=hunter2",
    "t" + GR_O + "ken: hunter2",
    "pa" + SHY + "ss" + chr(0x2060) + "word=hunter2",
    "pa" + chr(0) + "ssword=hunter2",
    "pa" + chr(1) + "ssword=hunter2",
    "api" + ZWSP + "_key=hunter2",
    "Author" + ZWSP + "ization: Bearer hunter2",
    "Coo" + ZWSP + "kie: sid=hunter2",
    "postgres://user:hunter2@host" + ZWSP,
    "x-api-key" + COMBINING_ACUTE + ": hunter2",
    "Set-Cookie: sid" + CYR_E + "=1; password=hunter2",
    "pass" + ZWSP + "word=" + '"hunter2 more"',
]


class TestSplitAndLookAlikeLabels:
    @pytest.mark.parametrize("text", SPLIT_LABELS, ids=lambda t: ascii(t)[:50])
    def test_sanitize_hides_the_secret(self, text):
        assert "hunter2" not in sanitize_error_text(text)

    @pytest.mark.parametrize("text", SPLIT_LABELS, ids=lambda t: ascii(t)[:50])
    def test_redact_args_hides_the_secret(self, text):
        out = redact_args({"note": text, "list": [text], "deep": {"x": [text]}})
        assert "hunter2" not in jsonable(out)

    def test_split_label_in_quoted_value_with_spaces(self):
        out = sanitize_error_text("pa" + ZWSP + 'ssword="hunter2 and more" tail')
        assert "hunter2" not in out and "and more" not in out and out.endswith("tail")

    def test_text_without_a_secret_is_returned_unchanged(self):
        for text in (HELLO_RU + " ok", "café " + ZWSP + "x", "t" + CYR_E + "xt " + SHY + "y",
                     "pa" + ZWSP + "ss word", chr(0x4E2D) * 10):
            assert sanitize_error_text(text) == text

    def test_plain_secret_keeps_the_rest_of_the_text(self):
        # the first run hides it, the second finds nothing: the text is not folded
        text = HELLO_RU + " password=hunter2 " + HELLO_RU + " t" + CYR_E + "xt"
        assert sanitize_error_text(text) == (HELLO_RU + " password=[REDACTED] " + HELLO_RU
                                             + " t" + CYR_E + "xt")

    def test_secret_found_only_by_the_shadow_keeps_the_rest_of_the_text(self):
        text = "p" + CYR_A + "ssword=hunter2 " + HELLO_RU + " café"
        out = sanitize_error_text(text)
        assert out == "p" + CYR_A + "ssword=[REDACTED] " + HELLO_RU + " café"

    def test_stable_when_applied_twice(self):
        for text in SPLIT_LABELS + [HELLO_RU + " p" + CYR_A + "ssword=x"]:
            once = sanitize_error_text(text)
            assert sanitize_error_text(once) == once

    def test_white_space_is_not_removed_from_the_shadow(self):
        text = "pa" + ZWSP + "ssword: hunter2\tx\r\ncookie: a b"
        out = sanitize_error_text(text)
        assert "hunter2" not in out and "\r\n" in out and "\t" in out

    def test_shadow_is_bounded(self):
        for text in (CYR_A * 1_000_000, ZWSP * 1_000_000, chr(0x4E2D) * 1_000_000,
                     (ZWSP + "x") * 400_000):
            start = time.perf_counter()
            out = sanitize_error_text(text)
            assert time.perf_counter() - start < TIME_LIMIT
            assert isinstance(out, str)

    def test_cap_applies_to_the_input(self, monkeypatch):
        monkeypatch.setattr(redact_module, "_MAX_SANITIZE_CHARS", 1000)
        text = "x" * 900 + " " + "pa" + ZWSP + "ssword=hunter2"      # 917 characters: scanned
        assert (sanitize_error_text(text, limit=5000)
                == "x" * 900 + " pa" + ZWSP + "ssword=[REDACTED]")
        text = "x" * 990 + " pa" + ZWSP + "ssword=hunter2"           # over the cap: not scanned
        assert sanitize_error_text(text).startswith("<text,")

    def test_redaction_growth_never_triggers_the_cap(self):
        # 900,001 characters with one non-ASCII letter and many short secrets: each secret
        # grows to "[REDACTED]", so the redacted text is longer than 1,000,000 characters
        unit = "a=1 token=b "
        text = chr(0xE9) + unit * (900_000 // len(unit))
        out = sanitize_error_text(text, limit=10 ** 8)
        assert not out.startswith("<text,")
        assert out.count("token=[REDACTED]") == 900_000 // len(unit)
        assert len(out) > 1_000_000
        # and the same through the shadow path
        text = chr(0xE9) + ("a=1 t" + ZWSP + "oken=b ") * (800_000 // 13)
        out = sanitize_error_text(text, limit=10 ** 8)
        assert not out.startswith("<text,") and "=b " not in out

    def test_shadow_time_on_soup(self):
        text = ("token=" + chr(34) + CYR_A + ZWSP) * 60_000
        start = time.perf_counter()
        sanitize_error_text(text)
        redact_args([text] * 5)
        assert time.perf_counter() - start < TIME_LIMIT


# ─── S2: long runs of combining marks ────────────────────────────────────────

class TestCombiningMarkRuns:
    @pytest.mark.parametrize("size", [100_000, 600_000, 3_000_000])
    def test_long_run_is_not_scanned_and_is_fast(self, size):
        text = "a" + COMBINING_ACUTE * size
        start = time.perf_counter()
        out = sanitize_error_text(text)
        assert time.perf_counter() - start < TIME_LIMIT
        assert out == f"<text, {size + 1} characters, not scanned>"

    def test_long_run_hides_a_secret_in_front_of_it(self):
        text = "password=hunter2 a" + COMBINING_ACUTE * 3_000_000
        out = sanitize_error_text(text)
        assert "hunter2" not in out and "not scanned" in out

    def test_long_run_through_redact_args(self):
        start = time.perf_counter()
        out = redact_args({"s": "a" + COMBINING_ACUTE * 150_000, "t": ["password=x"]})
        assert time.perf_counter() - start < TIME_LIMIT
        assert out["s"].startswith("<string,") and out["t"] == ["password=[REDACTED]"]

    def test_run_at_a_slice_boundary(self):
        short = "a" * 10_000 + COMBINING_ACUTE * 20            # inside the allowance
        assert sanitize_error_text(short, limit=10 ** 7) == unicodedata.normalize("NFKC", short)
        long_run = "a" * 10_000 + COMBINING_ACUTE * 40
        assert sanitize_error_text(long_run).startswith("<text,")

    def test_ordinary_marks_are_normalised_as_before(self):
        text = ("e" + COMBINING_ACUTE) * 20_000
        assert sanitize_error_text(text, limit=10 ** 7) == unicodedata.normalize("NFKC", text)

    def test_error_text_via_retry_with_a_hostile_message(self):
        start = time.perf_counter()
        out = retry_module._error_text(ValueError("a" + COMBINING_ACUTE * 3_000_000))
        assert time.perf_counter() - start < TIME_LIMIT
        assert out.startswith("ValueError: <text,")


# ─── M1, M2: PGP blocks, sessionid ───────────────────────────────────────────

class TestPgpAndSessionId:
    def test_pgp_private_key_block(self):
        pgp = ("-----BEGIN PGP PRIVATE KEY BLOCK-----\nVersion: x\n\nlQOYBFhunter2body\n=abcd\n"
               "-----END PGP PRIVATE KEY BLOCK-----")
        out = sanitize_error_text("key " + pgp + " after")
        assert "hunter2" not in out and "lQOY" not in out and out.endswith("after")

    def test_pgp_block_without_end(self):
        out = sanitize_error_text("-----BEGIN PGP PRIVATE KEY BLOCK-----\nlQOYhunter2")
        assert "hunter2" not in out

    def test_pgp_public_block_is_not_touched(self):
        text = "-----BEGIN PGP PUBLIC KEY BLOCK-----\nabc\n-----END PGP PUBLIC KEY BLOCK-----"
        assert sanitize_error_text(text) == text

    @pytest.mark.parametrize("key", ["sessionid", "SessionID", "sessionId", "session_id",
                                     "SESSIONID", "x-sessionid", "phpsessionid", "JSESSIONID"])
    def test_session_id_spellings(self, key):
        assert list(redact_args({key: "abc"}).values()) == [REDACTED]


# ─── M5, M6 ───────────────────────────────────────────────────────────────────

class TestErrorTextAndDigestMemory:
    def test_error_text_keeps_the_type_name_for_a_huge_message(self):
        out = retry_module._error_text(ValueError("password=hunter2 " + "x" * 2_000_000))
        assert out.startswith("ValueError: <text, ")
        assert "hunter2" not in out and out.endswith("not scanned>")

    def test_error_text_normal_message_is_unchanged(self):
        out = retry_module._error_text(ValueError("password=hunter2 failed"))
        assert out == "ValueError: password=[REDACTED] failed"

    def test_huge_containers_are_rejected_before_they_are_copied(self):
        import tracemalloc

        limit = redact_module._DIGEST_MAX_NODES
        values = [
            list(range(limit + 100_000)),
            tuple(range(limit + 100_000)),
            dict.fromkeys((str(i) for i in range(limit + 100_000)), 0),
        ]
        digester = ArgsDigester()
        for value in values:
            tracemalloc.start()
            try:
                with pytest.raises(TypeError):
                    digester.digest(value)
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
            assert peak < 200_000, peak

    def test_nested_huge_container_is_rejected(self):
        with pytest.raises(TypeError):
            ArgsDigester().digest({"a": [list(range(300_000))]})


# ═══════════════════════════════════════════════════════════════════════════════
# Third review round (issue #58): invisible value starts, more invisible characters,
# the size cap, speed, and a differential fuzz
# ═══════════════════════════════════════════════════════════════════════════════

NUL = chr(0)
WORD_JOINER = chr(0x2060)
ENCLOSING = chr(0x20DD)          # Me
SPACING_MARK = chr(0x0903)       # Mc
HANGUL_CHOSEONG_FILLER = chr(0x115F)
HANGUL_JUNGSEONG_FILLER = chr(0x1160)
HANGUL_FILLER = chr(0x3164)
BRAILLE_BLANK = chr(0x2800)
PRIVATE_USE = chr(0xE000)        # Co
UNASSIGNED = chr(0x0378)         # Cn
LONE_SURROGATE = chr(0xD800)     # Cs
TATWEEL = chr(0x0640)
MODIFIER_APOSTROPHE = chr(0x02BC)
RATIO = chr(0x2236)
MODIFIER_COLON = chr(0xA789)
COLON_EQUALS = chr(0x2254)
EQUALS_COLON = chr(0x2255)
MODIFIER_EQUALS = chr(0xA78A)


# ─── N1: an invisible character at the start of a value ──────────────────────

N1_INPUTS = [
    "password:" + ZWSP + " hunter2",
    "password= " + ZWSP + " hunter2",
    "token=" + SHY + " hunter2",
    "api_key=" + ZWSP + " hunter2",
    "Authorization:" + ZWSP + " Bearer hunter2",
    "password=" + NUL + " hunter2",
    "password=" + ZWSP + "\nhunter2",
    "password=" + ZWSP + "\thunter2",
    "password:" + ZWSP + ZWSP + " hunter2",
    "password=" + WORD_JOINER + " hunter2",
    "cookie:" + ZWSP + " hunter2; a=b",
    "password=" + COMBINING_ACUTE + " hunter2",
    "password=" + HANGUL_FILLER + " hunter2",
    "secret" + ZWSP + ":" + ZWSP + " hunter2",
    "x " + ZWSP + "password: " + ZWSP + " hunter2 tail",
]


class TestInvisibleValueStart:
    @pytest.mark.parametrize("text", N1_INPUTS, ids=lambda t: ascii(t)[:50])
    def test_sanitize_hides_the_secret(self, text):
        assert "hunter2" not in sanitize_error_text(text)

    @pytest.mark.parametrize("text", N1_INPUTS, ids=lambda t: ascii(t)[:50])
    def test_redact_args_hides_the_secret(self, text):
        assert "hunter2" not in jsonable(redact_args({"note": text, "l": [text]}))

    def test_only_the_value_is_replaced(self):
        out = sanitize_error_text("pre " + HELLO_RU + " password:" + ZWSP + " hunter2 post " + ZWSP)
        assert out == "pre " + HELLO_RU + " password:" + ZWSP + " [REDACTED] post " + ZWSP

    def test_invisible_characters_inside_the_value_go_with_it(self):
        out = sanitize_error_text("password=hu" + ZWSP + "nter2 tail")
        assert out == "password=[REDACTED] tail"

    def test_invariant_a_second_run_hides_nothing_more(self):
        for text in N1_INPUTS + SPLIT_LABELS:
            out = sanitize_error_text(text, limit=10 ** 6)
            assert redact_module._redact_text(out) == out
            built = redact_module._build_shadow(out)
            if built is not None:
                assert redact_module._apply_patterns(built[0]) == built[0]
            assert sanitize_error_text(out, limit=10 ** 6) == out


# ─── N2: more invisible characters and separator look-alikes ─────────────────

INVISIBLE_FOR_LABELS = [
    ("Me", ENCLOSING), ("Mc", SPACING_MARK), ("choseong filler", HANGUL_CHOSEONG_FILLER),
    ("jungseong filler", HANGUL_JUNGSEONG_FILLER), ("hangul filler", HANGUL_FILLER),
    ("braille blank", BRAILLE_BLANK), ("private use", PRIVATE_USE), ("unassigned", UNASSIGNED),
    ("surrogate", LONE_SURROGATE), ("tatweel", TATWEEL), ("modifier apostrophe",
                                                           MODIFIER_APOSTROPHE),
    ("ZWSP", ZWSP), ("SHY", SHY), ("NUL", NUL), ("DEL", chr(0x7F)), ("C1 control", chr(0x80)),
    ("FS", chr(0x1C)), ("combining", COMBINING_ACUTE), ("variation selector", chr(0xFE0F)),
    ("tag", chr(0xE0041)), ("LRM", chr(0x200E)), ("BOM", chr(0xFEFF)),
]


class TestMoreInvisibleCharacters:
    @pytest.mark.parametrize("label", ["password", "token", "api_key", "Authorization"])
    @pytest.mark.parametrize("name, char", INVISIBLE_FOR_LABELS, ids=[n for n, _ in INVISIBLE_FOR_LABELS])
    def test_label_split_by_the_character(self, label, name, char):
        for position in (1, len(label) // 2, len(label) - 1):
            split = label[:position] + char + label[position:]
            for text in (f"{split}=hunter2", f"x {split}: hunter2 y",
                         '{"' + split + '": "hunter2"}'):
                out = sanitize_error_text(text)
                assert "hunter2" not in out, (name, text, out)

    @pytest.mark.parametrize("name, char", INVISIBLE_FOR_LABELS, ids=[n for n, _ in INVISIBLE_FOR_LABELS])
    def test_value_starting_with_the_character(self, name, char):
        out = sanitize_error_text("password=" + char + " hunter2")
        assert "hunter2" not in out

    @pytest.mark.parametrize("name, char", INVISIBLE_FOR_LABELS, ids=[n for n, _ in INVISIBLE_FOR_LABELS])
    def test_key_names_split_by_the_character(self, name, char):
        for key in ("pa" + char + "ssword", "to" + char + "ken", "api" + char + "_key"):
            assert list(redact_args({key: "v"}).values()) == [REDACTED], (name, key)

    @pytest.mark.parametrize("sep", [RATIO, MODIFIER_COLON, COLON_EQUALS, EQUALS_COLON,
                                     MODIFIER_EQUALS], ids=lambda s: hex(ord(s)))
    def test_separator_look_alikes(self, sep):
        for text in ("password" + sep + "hunter2", "token " + sep + " hunter2 tail",
                     "api_key" + sep + '"hunter2 x"'):
            out = sanitize_error_text(text)
            assert "hunter2" not in out and "x\"" not in out, (text, out)

    def test_look_alike_separator_text_is_kept_when_there_is_no_label(self):
        for sep in (RATIO, COLON_EQUALS):
            text = "ratio 3" + sep + "4 and x" + sep + "y"
            assert sanitize_error_text(text) == text

    def test_visible_characters_in_a_label_are_a_known_gap(self):
        # documented under "not covered": the shadow drops invisible characters only
        for filler in (" ", ".", "'", "\t"):
            text = "pa" + filler + "ssword=hunter2"
            assert "hunter2" in sanitize_error_text(text)

    def test_whitespace_controls_are_kept_exactly(self):
        text = "a\tb\nc\x0bd\x0ce\rf" + chr(0x85) + "g"
        assert sanitize_error_text(text) == unicodedata.normalize("NFKC", text)


# ─── N4: speed ────────────────────────────────────────────────────────────────

class TestShadowSpeed:
    def test_hangul_text_with_a_secret_at_the_end(self):
        text = chr(0xD55C) * 990_000 + " pass" + ZWSP + "word=hunter2"
        start = time.perf_counter()
        out = sanitize_error_text(text)
        assert time.perf_counter() - start < TIME_LIMIT
        assert "hunter2" not in out

    def test_redact_args_on_hangul_strings(self):
        start = time.perf_counter()
        redact_args([chr(0xD55C) * 66_000] * 15)
        assert time.perf_counter() - start < TIME_LIMIT

    def test_dense_secrets_with_look_alike_labels(self):
        text = ("p" + CYR_A + "ssword=x ") * 40_000            # about 360,000 characters
        start = time.perf_counter()
        out = sanitize_error_text(text, limit=10 ** 8)
        assert time.perf_counter() - start < TIME_LIMIT
        assert out == ("p" + CYR_A + "ssword=[REDACTED] ") * 40_000

    def test_timing_does_not_depend_on_what_the_process_saw_before(self, monkeypatch):
        # 21,000+ distinct characters used to fill a global cache; afterwards every later
        # occurrence of an uncached character was worked out again (two seconds per million)
        seen = "".join(chr(code) for code in range(0x20000, 0x20000 + 21_500))
        sanitize_error_text(seen, limit=10)
        sanitize_error_text(seen + " p" + CYR_A + "ssword=x", limit=10)
        calls = []
        real = redact_module._fold_char
        monkeypatch.setattr(redact_module, "_fold_char", lambda char: calls.append(1) or real(char))
        for text in (chr(0x9FA0) * 1_000_000,                         # never folded
                     chr(0xAC00 + 7000) * 1_000_000,                  # decomposes (Hangul)
                     chr(0xE9) * 500_000 + " password=hunter2"):      # decomposes (accent)
            start = time.perf_counter()
            out = sanitize_error_text(text)
            assert time.perf_counter() - start < TIME_LIMIT
            assert "hunter2" not in out
        assert len(calls) <= 5, len(calls)        # once per distinct character, not per occurrence

    def test_redact_args_after_the_process_saw_many_characters(self):
        sanitize_error_text("".join(chr(code) for code in range(0x20000, 0x20000 + 21_500)))
        start = time.perf_counter()
        redact_args([chr(0x9FA0) * 60_000] * 16)
        assert time.perf_counter() - start < TIME_LIMIT

    def test_a_text_of_300000_different_characters_is_bounded(self):
        text = "".join(chr(code) for code in range(0x10000, 0x10000 + 300_000))
        start = time.perf_counter()
        sanitize_error_text(text)
        assert time.perf_counter() - start < TIME_LIMIT

    def test_a_text_of_all_different_characters_is_bounded(self):
        text = "".join(chr(code) for code in range(0x4E00, 0x4E00 + 20_000)) * 5
        start = time.perf_counter()
        sanitize_error_text(text)
        assert time.perf_counter() - start < TIME_LIMIT

    def test_pure_ascii_and_cjk_texts_use_no_shadow(self):
        assert redact_module._build_shadow("plain ascii password=x") is None
        assert redact_module._build_shadow(chr(0x4E2D) * 10 + " text") is None
        assert redact_module._build_shadow("a" + ZWSP + "b") is not None


# ─── Differential fuzz (fixed seeds) ──────────────────────────────────────────

FUZZ_LABELS = ["password", "secret", "token", "api_key", "cookie", "pwd", "sig", "passwd",
               "access_token", "client_secret"]
FUZZ_INVISIBLE = [ZWSP, SHY, WORD_JOINER, NUL, COMBINING_ACUTE, ENCLOSING, SPACING_MARK,
                  HANGUL_FILLER, BRAILLE_BLANK, PRIVATE_USE, UNASSIGNED, TATWEEL,
                  MODIFIER_APOSTROPHE, chr(0x200D), chr(0xFE0F)]
FUZZ_LOOKALIKES = {"a": CYR_A, "e": CYR_E, "o": CYR_O, "p": chr(0x0440), "c": chr(0x0441),
                   "s": chr(0x0455), "y": chr(0x0443), "i": chr(0x0456)}
FUZZ_FILL = [" ", "\n", " x ", " " + HELLO_RU + " ", " caf" + chr(0xE9) + " ", " , ", " ; ",
             " " + chr(0x4E2D) + chr(0x6587) + " "]
FUZZ_SEPS = ["=", ": ", ":", "= ", "=\"", ": \"", RATIO, " : "]


def fuzz_item(rng):
    """One "label sep value" item with some obfuscation; returns (text, expects_hidden)."""
    label = rng.choice(FUZZ_LABELS)
    sep = rng.choice(FUZZ_SEPS)
    quoted = sep.endswith("\"")
    value = "hunter2 more" if quoted else "hunter2"
    letters = list(label)
    for _ in range(rng.randint(0, 2)):
        position = rng.randint(0, len(letters))
        letters.insert(position, rng.choice(FUZZ_INVISIBLE))
    if rng.random() < 0.5:
        indexes = [i for i, c in enumerate(letters) if c in FUZZ_LOOKALIKES]
        if indexes:
            index = rng.choice(indexes)
            letters[index] = FUZZ_LOOKALIKES[letters[index]]
    spelled = "".join(letters)
    start = rng.choice(["", "", ZWSP, " " + ZWSP, SHY, NUL, COMBINING_ACUTE, HANGUL_FILLER,
                        "[REDACTED]", ZWSP + "[REDACTED]"])
    if rng.random() < 0.3 and "[REDACTED]" not in start:    # marker + space means "done"
        start += " "
    end = "\"" if quoted else ""
    return spelled + sep + start + value + end


class TestDifferentialFuzz:
    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    def test_secret_never_appears_and_second_pass_changes_nothing(self, seed):
        rng = random.Random(seed)
        for _ in range(75):
            items = [fuzz_item(rng) for _ in range(rng.randint(1, 4))]
            text = "".join(rng.choice(FUZZ_FILL) + item for item in items) + rng.choice(FUZZ_FILL)
            out = sanitize_error_text(text, limit=10 ** 6)
            assert "hunter2" not in out, (text, out)
            assert "more" not in out.replace("[REDACTED]", ""), (text, out)
            assert sanitize_error_text(out, limit=10 ** 6) == out, (text, out)
            args = redact_args({"note": text, "nested": [text]}, max_string=10 ** 5)
            assert "hunter2" not in jsonable(args), text

    @pytest.mark.parametrize("seed", [5, 6])
    def test_same_number_of_redactions_as_the_clean_spelling(self, seed):
        rng = random.Random(seed)
        for _ in range(60):
            label = rng.choice(FUZZ_LABELS)
            sep = rng.choice(["=", ": "])
            clean = f"{label}{sep}hunter2"
            split = label[:2] + rng.choice(FUZZ_INVISIBLE) + label[2:]
            dirty = f"x {split}{sep}{rng.choice(FUZZ_INVISIBLE)} hunter2 y {HELLO_RU}"
            clean_text = f"x {clean} y {HELLO_RU}"
            assert (sanitize_error_text(dirty).count("[REDACTED]")
                    == sanitize_error_text(clean_text).count("[REDACTED]") == 1), dirty

    def test_the_fuzz_covers_every_obfuscation(self):
        rng = random.Random(1)
        seen = {char for _ in range(300) for char in fuzz_item(rng) if char in FUZZ_INVISIBLE}
        assert len(seen) >= 12


# ─── M3: a value that starts with the marker ─────────────────────────────────

class TestMarkerPrefix:
    @pytest.mark.parametrize("text, expected", [
        ("password=[REDACTED]hunter2", "password=[REDACTED]"),
        ("token: [REDACTED]hunter2 tail", "token: [REDACTED] tail"),
        ("pwd=[REDACTED]hunter2", "pwd=[REDACTED]"),
        ("key=[REDACTED]hunter2&x=1", "key=[REDACTED]&x=1"),
        ("api_key=[REDACTED]hun ter2", "api_key=[REDACTED] ter2"),
        ("Authorization: [REDACTED]hunter2", "Authorization: [REDACTED]"),
        ("cookie: [REDACTED]hunter2; a=b", "cookie: [REDACTED]"),
        ('password="abc"hunter2', "password=[REDACTED]"),
        ('{"password": "abc"hunter2}', '{"password": [REDACTED]}'),
    ])
    def test_marker_followed_by_letters_is_an_ordinary_value(self, text, expected):
        assert sanitize_error_text(text) == expected

    @pytest.mark.parametrize("text", [
        "password=[REDACTED]", "password=[REDACTED] tail", "password: [REDACTED]\nnext line",
        "password=[REDACTED]&x=1", "password=[REDACTED],x=1", "password=[REDACTED];x",
        '{"password": [REDACTED]}', "(password=[REDACTED])", "[password=[REDACTED]]",
        "Authorization: [REDACTED]", "Authorization: [REDACTED] second", "cookie: [REDACTED]",
        "token=[REDACTED]'", 'token=[REDACTED]"', "key=[REDACTED]", "secret=[REDACTED]\\",
        "http://user:[REDACTED]@host/path",
    ])
    def test_an_already_redacted_value_is_left_alone(self, text):
        assert sanitize_error_text(text) == text

    def test_a_url_password_keeps_the_host(self):
        assert (sanitize_error_text("http://user:hunter2@host/x")
                == "http://user:[REDACTED]@host/x")

    def test_a_url_user_that_is_a_label_word_loses_the_host_too(self):
        # known over-redaction (docs/redaction.md): the URL pattern leaves "[REDACTED]@host", and
        # a marker followed by "@" is an ordinary value for the label "secret:" in front of it
        assert (sanitize_error_text("http://secret:hunter2@host/x")
                == "http://secret:[REDACTED]")

    def test_redact_args(self):
        out = redact_args({"note": "password=[REDACTED]hunter2", "done": "password=[REDACTED]",
                           "list": ["token: [REDACTED]hunter2"]})
        assert out == {"note": "password=[REDACTED]", "done": "password=[REDACTED]",
                       "list": ["token: [REDACTED]"]}

    def test_with_invisible_characters(self):
        out = sanitize_error_text("pa" + ZWSP + "ssword=" + ZWSP + "[REDACTED]hunter2 tail")
        assert "hunter2" not in out and out.endswith("tail")

    def test_applying_it_twice_changes_nothing(self):
        for text in ("password=[REDACTED]hunter2", "Authorization: [REDACTED]hunter2 x",
                     'password="abc"def', "cookie: [REDACTED]a\nb",
                     "key=[REDACTED]k&key=[REDACTED]z", "password=[REDACTED][REDACTED]hunter2",
                     "token=[REDACTED]]hunter2", "p" + CYR_A + "ssword=[REDACTED]hunter2"):
            once = sanitize_error_text(text)
            assert sanitize_error_text(once) == once, text

    def test_marker_rule_is_linear(self):
        for unit in ("password=[REDACTED]", "token=[REDACTED]x", "cookie: [REDACTED]a\n",
                     "Authorization: [REDACTED] [REDACTED]"):
            text = unit * (200_000 // len(unit))
            start = time.perf_counter()
            sanitize_error_text(text)
            assert time.perf_counter() - start < TIME_LIMIT


# ═══════════════════════════════════════════════════════════════════════════════
# Follow-up #74: the marker is one value character; the fold work is per call
# ═══════════════════════════════════════════════════════════════════════════════

FORGED_MARKER_FORMS = [
    ("password=[REDACTED]@hunter2", "password=[REDACTED]"),
    ("Cookie: [REDACTED] sid=hunter2", "Cookie: [REDACTED]"),
    ("Cookie: [REDACTED]; sid=hunter2", "Cookie: [REDACTED]"),
    ("Authorization: Bearer [REDACTED]hunter2", "Authorization: [REDACTED]"),
    ("password=[REDACTED][REDACTED]hunter2", "password=[REDACTED]"),
    ("Bearer [REDACTED]hunter2", "Bearer [REDACTED]"),
    ("sk-[REDACTED]hunter2", "sk-[REDACTED]"),
    ("secret=[REDACTED]@host", "secret=[REDACTED]"),
    ('password="abc"[REDACTED]x', "password=[REDACTED]"),
    ("token=abc[REDACTED]hunter2", "token=[REDACTED]"),
    ("key=[REDACTED]&hunter2", "key=[REDACTED]&hunter2"),    # "&" ends a value, marked or not
]


class TestForgedMarker:
    @pytest.mark.parametrize("text, expected", FORGED_MARKER_FORMS, ids=lambda t: ascii(t)[:40])
    def test_a_forged_marker_hides_like_an_unmarked_value(self, text, expected):
        assert sanitize_error_text(text) == expected
        assert sanitize_error_text(expected) == expected

    @pytest.mark.parametrize("text, expected", FORGED_MARKER_FORMS, ids=lambda t: ascii(t)[:40])
    def test_through_redact_args(self, text, expected):
        assert redact_args({"note": text, "list": [text]}) == {"note": expected,
                                                                 "list": [expected]}

    @pytest.mark.parametrize("text, expected", [
        ("password=[REDACTED]@hunter2", "password=[REDACTED]"),
        ("Cookie: [REDACTED] sid=hunter2", "Cookie: [REDACTED]"),
        ("Authorization: Bearer [REDACTED]hunter2", "Authorization: [REDACTED]"),
    ])
    def test_with_invisible_characters(self, text, expected):
        split = text[:2] + ZWSP + text[2:]
        out = sanitize_error_text(split)
        assert "hunter2" not in out
        assert sanitize_error_text(out) == out

    def test_an_unmarked_value_behaves_the_same(self):
        for marked, plain in (("password=[REDACTED]@hunter2", "password=x@hunter2"),
                              ("Cookie: [REDACTED] sid=hunter2", "Cookie: x sid=hunter2"),
                              ("Bearer [REDACTED]hunter2", "Bearer xhunter2")):
            assert "hunter2" not in sanitize_error_text(plain)
            assert "hunter2" not in sanitize_error_text(marked)

    def test_authorization_two_words_stay_consistent(self):
        # "Basic abc extra" leaves "[REDACTED] extra"; running again must not hide "extra"
        once = sanitize_error_text("Authorization: Basic abc extra")
        assert once == "Authorization: [REDACTED] extra"
        assert sanitize_error_text(once) == once
        assert sanitize_error_text("Authorization: [REDACTED] hunter2") == (
            "Authorization: [REDACTED] hunter2")           # documented: looks like the above
        # a third word after a marker in the second place is the same gap as for plain text
        assert sanitize_error_text("Authorization: Basic [REDACTED] hunter2") == (
            "Authorization: [REDACTED] hunter2")
        assert sanitize_error_text("Authorization: Basic abc hunter2") == (
            "Authorization: [REDACTED] hunter2")

    def test_cookie_json_value_stops_at_its_end_but_a_header_does_not(self):
        assert (sanitize_error_text('{"cookie": "sid=1", "x": 2}')
                == '{"cookie": [REDACTED], "x": 2}')
        assert (sanitize_error_text('{"cookie": [REDACTED], "x": 2}')
                == '{"cookie": [REDACTED], "x": 2}')
        assert sanitize_error_text('Cookie: "a=b"; sid=hunter2') == "Cookie: [REDACTED]"
        assert sanitize_error_text("Cookie: 'a\nb' tail=hunter2") == "Cookie: [REDACTED]"
        assert sanitize_error_text("Cookie: [REDACTED]\nnext=1") == "Cookie: [REDACTED]\nnext=1"

    def test_a_forged_marker_never_hides_less_than_plain_text(self):
        rng = random.Random(74)
        labels = ["password", "token", "api_key", "secret", "pwd", "sig", "Authorization",
                  "Cookie", "key"]
        tails = ["", "@hunter2", "hunter2", " hunter2", ";hunter2", "x@hunter2", "[REDACTED]hunter2"]
        for _ in range(400):
            label = rng.choice(labels)
            text = f"{label}{rng.choice(['=', ': '])}[REDACTED]{rng.choice(tails)} end"
            out = sanitize_error_text(text, limit=10 ** 5)
            assert sanitize_error_text(out, limit=10 ** 5) == out, text


class TestFoldWorkPerCall:
    def test_many_strings_of_one_call_share_the_work(self):
        decomposable = "".join(
            chr(code) for code in range(0x80, 0x3000)
            if unicodedata.normalize("NFKC", chr(code)) == chr(code)
            and unicodedata.normalize("NFKD", chr(code)) != chr(code))
        decomposable += "".join(chr(code) for code in range(0xAC00, 0xD7A4))
        assert len(set(decomposable)) > 10_000
        count = 1_000_000 // len(decomposable)
        start = time.perf_counter()
        redact_args([decomposable] * count)
        assert time.perf_counter() - start < TIME_LIMIT

    def test_fold_char_runs_once_per_distinct_character_in_a_call(self, monkeypatch):
        calls = []
        real = redact_module._fold_char
        monkeypatch.setattr(redact_module, "_fold_char", lambda char: calls.append(char) or real(char))
        text = "caf" + chr(0xE9) + " " + chr(0xE0) + chr(0xE8) + chr(0xEC) + " password=x"
        redact_args([text] * 50)
        distinct = len(set(calls))
        assert 1 <= distinct <= 4 and len(calls) == distinct      # once per distinct character
        calls.clear()
        redact_args([text])
        redact_args([text])
        assert len(calls) == 2 * distinct                          # nothing is kept between calls

    def test_the_memo_is_not_shared_between_calls_and_is_cleared(self):
        assert redact_module._FOLD_CONTEXT.get() is None
        redact_args([chr(0xE9) * 10])
        assert redact_module._FOLD_CONTEXT.get() is None
        sanitize_error_text(chr(0xE9) + chr(0x200B) + "password=x")
        assert redact_module._FOLD_CONTEXT.get() is None

    def test_the_memo_is_bounded(self, monkeypatch):
        monkeypatch.setattr(redact_module, "_MAX_CONTEXT_CHARS", 1000)
        strings = ["".join(chr(code) for code in range(start, start + 600))
                   for start in range(0x4E00, 0x4E00 + 6000, 600)]
        out = redact_args(strings + ["p" + CYR_A + "ssword=hunter2"])
        assert out[-1] == "p" + CYR_A + "ssword=[REDACTED]"

    def test_many_small_calls_stay_cheap(self):
        rng = random.Random(2)
        hangul = [chr(code) for code in range(0xAC00, 0xD7A4)]
        strings = ["".join(rng.choice(hangul) for _ in range(500)) for _ in range(2000)]
        start = time.perf_counter()
        for text in strings:
            sanitize_error_text(text)
        assert time.perf_counter() - start < TIME_LIMIT     # about 0.6 s here, 0.4 s on main

    def test_random_code_points_across_many_strings(self):
        rng = random.Random(1)
        points = [chr(code) for code in range(0x80, 0x2FA1E) if not 0xD800 <= code <= 0xDFFF]
        strings = ["".join(rng.choice(points) for _ in range(40_000)) for _ in range(10)]
        start = time.perf_counter()
        redact_args(strings)
        assert time.perf_counter() - start < 2 * TIME_LIMIT   # about 0.9 s here; main 1.9 s

    def test_other_letters_never_change_or_hide(self):
        # the shortcut in _fold_table: category Lo (Hangul, CJK, kana, Indic ...) is skipped
        for code in range(0x80, 0x30000):
            if 0xD800 <= code <= 0xDFFF:
                continue
            char = chr(code)
            if (unicodedata.category(char) == "Lo" and unicodedata.normalize("NFKC", char) == char
                    and char not in redact_module._EXTRA_INVISIBLE):
                assert redact_module._fold_char(char) == char, hex(code)

    def test_decomposition_filter_agrees_with_fold_char(self):
        chars = {chr(code) for code in range(0x80, 0x3000)
                 if unicodedata.normalize("NFKC", chr(code)) == chr(code)}
        context = redact_module._fold_table(chars)
        for char in chars:
            folded = redact_module._fold_char(char)
            if folded == "":
                assert char in context.hidden, hex(ord(char))
            elif folded != char:
                assert context.table.get(ord(char)) == folded, hex(ord(char))
            else:
                assert ord(char) not in context.table and char not in context.hidden
