"""
Tests for aria.plugins.retry and for the retry behaviour of the LLM plugins (issue #41).

Nothing here really waits: every policy gets a recording fake sleep, a fake clock and a
fixed random source. The vendor SDKs are replaced by fakes injected into sys.modules.
"""
import asyncio
import copy
import errno
import pickle
import socket
import ssl
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

from aria.plugins import AnthropicPlugin, OpenAIPlugin
from aria.plugins.retry import (
    EmptyResponseError,
    LLMError,
    LLMErrorCode,
    RetryPolicy,
    call_with_retry,
    classify,
    get_retry_after,
    sanitize_error_text,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────


class Sleeper:
    """Fake asyncio.sleep that records the requested waits and returns at once."""

    def __init__(self):
        self.delays = []

    async def __call__(self, seconds):
        self.delays.append(seconds)


class Ticker:
    """Fake clock: every reading is one second later than the last."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        self.now += 1.0
        return self.now


def make_policy(rng_value=0.0, **kwargs):
    """Policy with a fake sleep (``policy.sleeper``), fake clock and fixed random value."""
    sleeper = Sleeper()
    kwargs.setdefault("sleep", sleeper)
    kwargs.setdefault("clock", Ticker())
    kwargs.setdefault("rng", lambda: rng_value)
    policy = RetryPolicy(**kwargs)
    policy.sleeper = sleeper
    return policy


class Flaky:
    """An async callable that fails with the given errors in order, then returns a value."""

    def __init__(self, errors, result="ok"):
        self.errors = list(errors)
        self.result = result
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


# Fake vendor errors. Only the class NAME and the attributes matter to classify().
class RateLimitError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class APITimeoutError(APIConnectionError):
    pass


class AuthenticationError(Exception):
    pass


class PermissionDeniedError(Exception):
    pass


class BadRequestError(Exception):
    pass


class NotFoundError(Exception):
    pass


class ProxyError(Exception):
    pass


class FakeAPIStatusError(Exception):
    """Like the real SDK errors: required keyword-only arguments in __init__."""

    def __init__(self, message, *, response, body):
        super().__init__(message)
        self.response = response
        self.body = body


class InternalServerError(Exception):
    pass


class APIStatusError(Exception):
    def __init__(self, message="", status_code=None, response=None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        if response is not None:
            self.response = response


def with_status(status, **kwargs):
    return APIStatusError(f"status {status}", status_code=status, **kwargs)


async def _raising_sleep(seconds):
    raise RuntimeError("clock broke")


# ─── classify ─────────────────────────────────────────────────────────────────

class TestClassify:

    @pytest.mark.parametrize("status,code", [
        (429, LLMErrorCode.RATE_LIMIT),
        (500, LLMErrorCode.SERVER),
        (502, LLMErrorCode.SERVER),
        (503, LLMErrorCode.SERVER),
        (529, LLMErrorCode.SERVER),
        (599, LLMErrorCode.SERVER),
        (408, LLMErrorCode.TIMEOUT),
        (409, LLMErrorCode.SERVER),      # conflict: both vendor SDKs retry it
        (401, LLMErrorCode.AUTH),
        (403, LLMErrorCode.AUTH),
        (400, LLMErrorCode.BAD_REQUEST),
        (404, LLMErrorCode.BAD_REQUEST),
        (413, LLMErrorCode.BAD_REQUEST),
        (422, LLMErrorCode.BAD_REQUEST),
    ])
    def test_status_code_attribute(self, status, code):
        assert classify(with_status(status)) is code

    def test_status_attribute(self):
        exc = Exception("x")
        exc.status = 429
        assert classify(exc) is LLMErrorCode.RATE_LIMIT

    def test_response_status_code(self):
        exc = Exception("x")
        exc.response = SimpleNamespace(status_code=503, headers={})
        assert classify(exc) is LLMErrorCode.SERVER

    def test_status_beats_class_name(self):
        # A connection-looking name with a 400 status is still a bad request.
        exc = APIConnectionError("x")
        exc.status_code = 400
        assert classify(exc) is LLMErrorCode.BAD_REQUEST

    @pytest.mark.parametrize("value", ["429", True, 99, 600, None, 12.5])
    def test_non_http_status_values_are_ignored(self, value):
        exc = Exception("x")
        exc.status = value
        assert classify(exc) is LLMErrorCode.UNKNOWN

    @pytest.mark.parametrize("exc,code", [
        (RateLimitError("slow down"), LLMErrorCode.RATE_LIMIT),
        (InternalServerError("oops"), LLMErrorCode.SERVER),
        (APITimeoutError("t"), LLMErrorCode.TIMEOUT),
        (APIConnectionError("c"), LLMErrorCode.TRANSPORT),
        (AuthenticationError("a"), LLMErrorCode.AUTH),
        (PermissionDeniedError("p"), LLMErrorCode.AUTH),
        (BadRequestError("b"), LLMErrorCode.BAD_REQUEST),
        (NotFoundError("model not found"), LLMErrorCode.BAD_REQUEST),
        (ProxyError("proxy refused"), LLMErrorCode.TRANSPORT),
    ])
    def test_class_names_without_any_status(self, exc, code):
        assert classify(exc) is code

    def test_subclass_of_a_named_class_is_found_through_the_hierarchy(self):
        class MyLimit(RateLimitError):
            pass
        assert classify(MyLimit("x")) is LLMErrorCode.RATE_LIMIT

    @pytest.mark.parametrize("exc,code", [
        (TimeoutError("t"), LLMErrorCode.TIMEOUT),
        (asyncio.TimeoutError(), LLMErrorCode.TIMEOUT),
        (socket.timeout("t"), LLMErrorCode.TIMEOUT),
        (ConnectionError("c"), LLMErrorCode.TRANSPORT),
        (ConnectionResetError("c"), LLMErrorCode.TRANSPORT),
        (ConnectionRefusedError("c"), LLMErrorCode.TRANSPORT),
        (BrokenPipeError("c"), LLMErrorCode.TRANSPORT),
        (socket.gaierror(-2, "name or service not known"), LLMErrorCode.TRANSPORT),
        (OSError(errno.ECONNRESET, "reset"), LLMErrorCode.TRANSPORT),
        (OSError(errno.ENETUNREACH, "unreachable"), LLMErrorCode.TRANSPORT),
        (OSError(errno.ETIMEDOUT, "timed out"), LLMErrorCode.TIMEOUT),
    ])
    def test_builtin_network_errors(self, exc, code):
        assert classify(exc) is code

    @pytest.mark.parametrize("exc", [
        RuntimeError("boom"),
        ValueError("v"),
        KeyError("k"),
        OSError("plain"),
        FileNotFoundError(errno.ENOENT, "missing"),
    ])
    def test_everything_else_is_unknown(self, exc):
        assert classify(exc) is LLMErrorCode.UNKNOWN

    def test_bad_certificate_is_not_retryable_but_other_tls_errors_are(self):
        cert = ssl.SSLCertVerificationError(1, "certificate verify failed")
        assert classify(cert) is LLMErrorCode.AUTH
        assert not classify(cert).retryable
        assert classify(ssl.SSLError(1, "handshake failure")) is LLMErrorCode.TRANSPORT

    def test_plain_programming_errors_are_not_classified_by_name(self):
        class TimeoutConfigError(ValueError):
            pass

        class ConnectionStringError(ValueError):
            pass

        class RateLimitConfigError(KeyError):
            pass

        for exc in (TimeoutConfigError("x"), ConnectionStringError("x"),
                    RateLimitConfigError("x"), TypeError("x")):
            assert classify(exc) is LLMErrorCode.UNKNOWN

    def test_a_programming_error_with_an_http_status_is_still_classified(self):
        class TimeoutConfigError(ValueError):
            pass

        exc = TimeoutConfigError("x")
        exc.status_code = 503
        assert classify(exc) is LLMErrorCode.SERVER

    def test_an_unmapped_status_does_not_turn_name_matching_back_on(self):
        class TimeoutConfigError(ValueError):
            pass

        exc = TimeoutConfigError("x")
        exc.status_code = 302      # a redirect: not a provider error code
        assert classify(exc) is LLMErrorCode.UNKNOWN

    @pytest.mark.parametrize("name", ["ReadError", "ConnectionStringInvalid", "WriteError"])
    def test_non_programming_errors_with_transport_like_names_are_transport(self, name):
        # Documented behaviour: matching is by name, so look-alike names are caught too.
        assert classify(type(name, (Exception,), {})("x")) is LLMErrorCode.TRANSPORT

    def test_classify_never_raises_on_hostile_attributes(self):
        class Hostile(Exception):
            @property
            def status_code(self):
                raise RuntimeError("boom")

            @property
            def status(self):
                raise RuntimeError("boom")

            @property
            def response(self):
                raise RuntimeError("boom")

            @property
            def retry_after(self):
                raise RuntimeError("boom")

        exc = Hostile("x")
        assert classify(exc) is LLMErrorCode.UNKNOWN
        assert get_retry_after(exc) is None

    def test_a_raising_response_headers_property_is_survived(self):
        class Response:
            status_code = 429

            @property
            def headers(self):
                raise RuntimeError("boom")

        exc = Exception("x")
        exc.response = Response()
        assert classify(exc) is LLMErrorCode.RATE_LIMIT
        assert get_retry_after(exc) is None

    def test_empty_response_error(self):
        assert classify(EmptyResponseError("nothing")) is LLMErrorCode.EMPTY

    def test_llm_error_keeps_its_code(self):
        err = LLMError("x", code=LLMErrorCode.AUTH)
        assert classify(err) is LLMErrorCode.AUTH

    def test_retryable_flags(self):
        retryable = {c for c in LLMErrorCode if c.retryable}
        assert retryable == {
            LLMErrorCode.RATE_LIMIT, LLMErrorCode.SERVER, LLMErrorCode.TIMEOUT,
            LLMErrorCode.TRANSPORT, LLMErrorCode.EMPTY,
        }
        for code in (LLMErrorCode.AUTH, LLMErrorCode.BAD_REQUEST, LLMErrorCode.UNKNOWN):
            assert not code.retryable

    def test_all_codes_exist(self):
        assert {c.name for c in LLMErrorCode} == {
            "RATE_LIMIT", "SERVER", "TIMEOUT", "TRANSPORT", "EMPTY",
            "AUTH", "BAD_REQUEST", "UNKNOWN",
        }

    def test_classify_does_not_need_the_vendor_sdks(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "anthropic", None)  # import would fail
        monkeypatch.setitem(sys.modules, "openai", None)
        assert classify(RateLimitError("x")) is LLMErrorCode.RATE_LIMIT


# ─── retry-after ──────────────────────────────────────────────────────────────

class TestRetryAfter:

    def test_attribute(self):
        exc = RateLimitError("x")
        exc.retry_after = 7
        assert get_retry_after(exc) == 7.0

    def test_header_on_response(self):
        exc = with_status(429, response=SimpleNamespace(headers={"retry-after": "12"}))
        assert get_retry_after(exc) == 12.0

    def test_header_lookup_is_case_insensitive_for_plain_dicts(self):
        exc = with_status(429, response=SimpleNamespace(headers={"Retry-After": "3"}))
        assert get_retry_after(exc) == 3.0

    def test_millisecond_header(self):
        exc = with_status(429, response=SimpleNamespace(headers={"retry-after-ms": "1500"}))
        assert get_retry_after(exc) == 1.5

    @pytest.mark.parametrize("value", ["Wed, 21 Oct 2026 07:28:00 GMT", "-5", "nan", "inf", "x", None])
    def test_unusable_values_are_ignored(self, value):
        exc = with_status(429, response=SimpleNamespace(headers={"retry-after": value}))
        assert get_retry_after(exc) is None

    def test_no_information(self):
        assert get_retry_after(RuntimeError("x")) is None

    def test_delay_uses_retry_after(self):
        assert make_policy().delay_for(1, retry_after=7.0) == 7.0

    def test_delay_is_capped_at_max_delay(self):
        assert make_policy(max_delay=30.0).delay_for(1, retry_after=600.0) == 30.0

    async def test_retry_after_is_waited_instead_of_the_backoff(self):
        policy = make_policy()
        exc = with_status(429, response=SimpleNamespace(headers={"retry-after": "9"}))
        assert await call_with_retry(Flaky([exc]), policy) == "ok"
        assert policy.sleeper.delays == [9.0]
        assert policy.history[0].retry_after == 9.0

    async def test_retry_after_larger_than_the_cap_is_capped(self):
        policy = make_policy(max_delay=30.0)
        exc = RateLimitError("x")
        exc.retry_after = 3600
        await call_with_retry(Flaky([exc]), policy)
        assert policy.sleeper.delays == [30.0]


# ─── backoff and jitter ───────────────────────────────────────────────────────

class TestBackoff:

    def test_defaults(self):
        policy = RetryPolicy()
        assert (policy.max_retries, policy.base_delay, policy.max_delay) == (3, 0.5, 30.0)

    def test_exponential_growth(self):
        policy = make_policy(jitter=0.0)
        assert [policy.delay_for(n) for n in (1, 2, 3, 4, 5)] == [0.5, 1.0, 2.0, 4.0, 8.0]

    def test_growth_stops_at_the_cap(self):
        policy = make_policy(jitter=0.0, max_delay=30.0)
        assert policy.delay_for(7) == 30.0   # 0.5 * 64 = 32 -> capped
        assert policy.delay_for(50) == 30.0

    def test_huge_retry_numbers_do_not_overflow(self):
        policy = make_policy(jitter=0.0, max_delay=30.0)
        assert policy.delay_for(5000) == 30.0

    def test_custom_multiplier(self):
        policy = make_policy(jitter=0.0, base_delay=1.0, multiplier=3.0)
        assert [policy.delay_for(n) for n in (1, 2, 3)] == [1.0, 3.0, 9.0]

    def test_jitter_only_shortens_and_stays_in_bounds(self):
        # rng 0 -> no reduction, rng just under 1 -> maximum reduction (25 percent)
        low = make_policy(rng_value=0.0).delay_for(3)
        high = make_policy(rng_value=0.999999).delay_for(3)
        assert low == 2.0
        assert 1.5 <= high < 1.5001
        assert high < low

    def test_jitter_never_exceeds_the_cap_over_many_random_draws(self):
        import random
        rnd = random.Random(1234)
        policy = RetryPolicy(max_delay=5.0, base_delay=2.0, jitter=1.0, rng=rnd.random)
        for n in range(1, 40):
            delay = policy.delay_for(n)
            assert 0.0 <= delay <= 5.0

    def test_jitter_is_random_between_calls(self):
        values = iter([0.0, 0.5, 1.0])
        policy = make_policy(rng=lambda: next(values), jitter=0.5, base_delay=4.0)
        assert [policy.delay_for(1) for _ in range(3)] == [4.0, 3.0, 2.0]

    @pytest.mark.parametrize("kwargs", [
        {"max_retries": -1}, {"base_delay": -0.1}, {"max_delay": -1},
        {"multiplier": 0.5}, {"jitter": -0.1}, {"jitter": 1.1}, {"max_history": 0},
        {"base_delay": float("nan")}, {"base_delay": float("inf")},
        {"max_delay": float("nan")}, {"max_delay": float("inf")},
        {"multiplier": float("nan")}, {"multiplier": float("inf")},
        {"jitter": float("nan")}, {"jitter": float("inf")},
        {"base_delay": 10 ** 400},
    ])
    def test_out_of_range_settings_are_rejected(self, kwargs):
        with pytest.raises(ValueError, match=next(iter(kwargs))):
            RetryPolicy(**kwargs)

    @pytest.mark.parametrize("kwargs", [
        {"max_retries": True}, {"max_retries": 2.0}, {"max_retries": 1.5},
        {"max_retries": "3"}, {"max_retries": None},
        {"max_retries": float("nan")}, {"max_retries": float("inf")},
        {"max_history": True}, {"max_history": 5.0}, {"max_history": "5"},
        {"base_delay": "1"}, {"base_delay": None}, {"base_delay": True},
        {"max_delay": "30"}, {"multiplier": "2"}, {"jitter": "0.1"}, {"jitter": True},
    ])
    def test_wrongly_typed_settings_are_rejected(self, kwargs):
        with pytest.raises(TypeError, match=next(iter(kwargs))):
            RetryPolicy(**kwargs)

    @pytest.mark.parametrize("kwargs", [
        {"max_retries": 101}, {"max_retries": 10 ** 30}, {"max_history": 100_001},
        {"max_history": 10 ** 30},
    ])
    def test_upper_bounds_give_a_clear_value_error(self, kwargs):
        with pytest.raises(ValueError, match=next(iter(kwargs))):
            RetryPolicy(**kwargs)

    def test_upper_bounds_themselves_are_accepted(self):
        RetryPolicy(max_retries=100, max_history=100_000)

    @pytest.mark.parametrize("kwargs", [
        {"sleep": 5}, {"sleep": "fast"}, {"clock": 1.0}, {"clock": "now"},
        {"rng": 0.5}, {"rng": []},
    ])
    def test_sleep_clock_and_rng_must_be_callable(self, kwargs):
        with pytest.raises(TypeError, match=next(iter(kwargs))):
            RetryPolicy(**kwargs)

    def test_none_for_sleep_clock_and_rng_means_the_default(self):
        policy = RetryPolicy(sleep=None, clock=None, rng=None)
        assert callable(policy.clock) and callable(policy.rng)
        assert policy.sleep is None
        assert 0.0 <= policy.delay_for(1) <= 0.5

    def test_boundary_values_are_accepted(self):
        RetryPolicy(max_retries=0, base_delay=0, max_delay=0, multiplier=1, jitter=0,
                    max_history=1)
        RetryPolicy(jitter=1, base_delay=1, max_delay=2.5)

    async def test_default_sleep_is_looked_up_when_used(self, monkeypatch):
        waits = []

        async def fake_sleep(seconds):
            waits.append(seconds)

        policy = RetryPolicy(jitter=0.0)          # no sleep injected
        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        await call_with_retry(Flaky([RateLimitError("x")]), policy)
        assert waits == [0.5]

    async def test_waits_grow_between_real_retries(self):
        policy = make_policy(jitter=0.0)
        await call_with_retry(Flaky([RateLimitError("a")] * 3), policy)
        assert policy.sleeper.delays == [0.5, 1.0, 2.0]


# ─── call_with_retry ──────────────────────────────────────────────────────────

class TestCallWithRetry:

    async def test_first_call_succeeds_without_waiting(self):
        policy = make_policy()
        fn = Flaky([], result={"a": 1})
        assert await call_with_retry(fn, policy) == {"a": 1}
        assert fn.calls == 1
        assert policy.sleeper.delays == []

    @pytest.mark.parametrize("failures", [1, 2, 3])
    async def test_success_after_n_failures(self, failures):
        policy = make_policy()
        fn = Flaky([InternalServerError("down")] * failures, result="finally")
        assert await call_with_retry(fn, policy) == "finally"
        assert fn.calls == failures + 1
        assert len(policy.sleeper.delays) == failures

    async def test_policy_is_optional(self):
        assert await call_with_retry(Flaky([])) == "ok"

    @pytest.mark.parametrize("error,code", [
        (RateLimitError("r"), LLMErrorCode.RATE_LIMIT),
        (InternalServerError("s"), LLMErrorCode.SERVER),
        (TimeoutError("t"), LLMErrorCode.TIMEOUT),
        (ConnectionError("c"), LLMErrorCode.TRANSPORT),
        (EmptyResponseError("e"), LLMErrorCode.EMPTY),
    ])
    async def test_every_retryable_code_is_retried(self, error, code):
        policy = make_policy()
        fn = Flaky([error])
        assert await call_with_retry(fn, policy) == "ok"
        assert fn.calls == 2
        assert policy.history[0].code is code

    @pytest.mark.parametrize("error,code", [
        (AuthenticationError("bad key"), LLMErrorCode.AUTH),
        (with_status(401), LLMErrorCode.AUTH),
        (with_status(403), LLMErrorCode.AUTH),
        (PermissionDeniedError("no"), LLMErrorCode.AUTH),
        (BadRequestError("bad"), LLMErrorCode.BAD_REQUEST),
        (with_status(400), LLMErrorCode.BAD_REQUEST),
        (with_status(422), LLMErrorCode.BAD_REQUEST),
        (RuntimeError("???"), LLMErrorCode.UNKNOWN),
        (ValueError("???"), LLMErrorCode.UNKNOWN),
    ])
    async def test_non_retryable_errors_are_never_retried(self, error, code):
        policy = make_policy()
        fn = Flaky([error, error, error, error])
        with pytest.raises(LLMError) as info:
            await call_with_retry(fn, policy)
        assert fn.calls == 1
        assert policy.sleeper.delays == []
        assert info.value.code is code
        assert info.value.attempts == 1
        assert info.value.cause is error
        assert info.value.__cause__ is error
        assert "not retried" in str(info.value)
        assert policy.history[-1].outcome == "not_retryable"

    async def test_exhaustion_raises_llm_error_with_attempt_count(self):
        policy = make_policy()  # 3 retries -> 4 attempts
        errors = [RateLimitError(f"limit {i}") for i in range(10)]
        fn = Flaky(errors)
        with pytest.raises(LLMError) as info:
            await call_with_retry(fn, policy)
        err = info.value
        assert fn.calls == 4
        assert err.attempts == 4
        assert err.code is LLMErrorCode.RATE_LIMIT
        assert err.cause is errors[3]
        assert err.__cause__ is errors[3]
        assert "4 attempt(s)" in str(err) and "rate_limit" in str(err) and "limit 3" in str(err)
        assert len(policy.sleeper.delays) == 3          # no wait after the last failure
        assert [r.outcome for r in err.history] == ["retry", "retry", "retry", "exhausted"]

    async def test_zero_retries_means_one_attempt(self):
        policy = make_policy(max_retries=0)
        fn = Flaky([InternalServerError("x")] * 3)
        with pytest.raises(LLMError) as info:
            await call_with_retry(fn, policy)
        assert fn.calls == 1 and info.value.attempts == 1
        assert policy.sleeper.delays == []

    async def test_llm_error_is_a_plain_error_that_keeps_the_original_as_cause(self):
        original = ConnectionError("down")
        with pytest.raises(LLMError) as info:
            await call_with_retry(Flaky([original] * 9), make_policy())
        err = info.value
        assert type(err) is LLMError
        assert not isinstance(err, ConnectionError)
        assert err.cause is original and err.__cause__ is original

    async def test_original_exception_is_left_untouched(self):
        original = OSError(104, "reset")
        with pytest.raises(LLMError):
            await call_with_retry(Flaky([original] * 9), make_policy())
        assert (original.errno, original.strerror) == (104, "reset")
        assert str(original) == "[Errno 104] reset"

    @pytest.mark.parametrize("original", [
        FakeAPIStatusError("kw-only", response=SimpleNamespace(status_code=500, headers={}),
                           body={"error": "x"}),
        FakeAPIStatusError("kw-only 400", response=SimpleNamespace(status_code=400, headers={}),
                           body=None),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
        OSError(104, "reset"),
        KeyError("k"),
    ])
    async def test_awkward_exception_types_can_be_wrapped(self, original):
        with pytest.raises(LLMError) as info:
            await call_with_retry(Flaky([original] * 9), make_policy())
        assert info.value.cause is original
        assert info.value.__cause__ is original
        assert type(original).__name__ in str(info.value)

    async def test_vendor_attributes_are_not_overwritten(self):
        original = with_status(500)
        original.code = "vendor_code"
        with pytest.raises(LLMError) as info:
            await call_with_retry(Flaky([original] * 9), make_policy())
        assert original.code == "vendor_code"
        assert info.value.code is LLMErrorCode.SERVER

    async def test_llm_error_pickles_without_its_cause(self):
        policy = make_policy()
        with pytest.raises(LLMError) as info:
            await call_with_retry(Flaky([ConnectionError("down")] * 9), policy)
        clone = pickle.loads(pickle.dumps(info.value))
        assert type(clone) is LLMError
        assert str(clone) == str(info.value)
        assert (clone.code, clone.attempts) == (LLMErrorCode.TRANSPORT, 4)
        assert clone.cause is None
        assert clone.cause_type == "ConnectionError" and clone.cause_text == "down"
        assert [r.outcome for r in clone.history] == ["retry", "retry", "retry", "exhausted"]
        assert clone.history == info.value.history

    async def test_llm_error_wrapping_a_kw_only_exception_round_trips(self):
        original = FakeAPIStatusError(
            "server broke", response=SimpleNamespace(status_code=500, headers={}),
            body={"error": "x"})
        with pytest.raises(LLMError) as info:
            await call_with_retry(Flaky([original] * 9), make_policy())
        clone = pickle.loads(pickle.dumps(info.value))
        assert clone.code is LLMErrorCode.SERVER and clone.attempts == 4
        assert clone.cause is None
        assert clone.cause_type == "FakeAPIStatusError"
        assert clone.cause_text == "server broke"
        assert str(clone) == str(info.value)

    async def test_llm_error_wrapping_an_unpicklable_exception_round_trips(self):
        original = ConnectionError("down")
        original.lock = threading.Lock()                  # cannot be pickled
        original.callback = lambda: None                  # nor can this
        with pytest.raises(LLMError) as info:
            await call_with_retry(Flaky([original] * 9), make_policy())
        clone = pickle.loads(pickle.dumps(info.value))
        assert clone.cause is None and clone.cause_type == "ConnectionError"

    async def test_pickled_bytes_do_not_contain_secrets_from_the_cause(self):
        original = FakeAPIStatusError(
            "bad call api_key=BODYSECRET111 sk-ant-api03-PICKLESECRET222",
            response=SimpleNamespace(status_code=400, headers={"x-api-key": "HEADERSECRET333"}),
            body={"api_key": "BODYSECRET111", "password": "PICKLESECRET444"})
        with pytest.raises(LLMError) as info:
            await call_with_retry(Flaky([original]), make_policy())
        data = pickle.dumps(info.value)
        for secret in (b"BODYSECRET111", b"PICKLESECRET222", b"PICKLESECRET444",
                       b"HEADERSECRET333"):
            assert secret not in data
        assert b"FakeAPIStatusError" in data           # the type name is kept

    async def test_copy_and_deepcopy_keep_working_and_keep_the_cause(self):
        original = ConnectionError("down")
        with pytest.raises(LLMError) as info:
            await call_with_retry(Flaky([original] * 9), make_policy())
        for clone in (copy.copy(info.value), copy.deepcopy(info.value)):
            assert type(clone) is LLMError and clone is not info.value
            assert clone.cause is original
            assert (clone.code, clone.attempts, str(clone)) == (
                info.value.code, info.value.attempts, str(info.value))
            assert clone.history == info.value.history

    async def test_cause_text_is_redacted_and_cause_is_untouched(self):
        original = AuthenticationError("bad api_key=CAUSESECRET555 " + "z" * 900)
        with pytest.raises(LLMError) as info:
            await call_with_retry(Flaky([original]), make_policy())
        assert "CAUSESECRET555" not in info.value.cause_text
        assert len(info.value.cause_text) < 700
        assert "CAUSESECRET555" in str(original)

    async def test_many_different_exception_types_do_not_create_new_error_classes(self):
        before = LLMError.__subclasses__()
        for i in range(5):
            cls = type(f"Err{i}", (Exception,), {})
            with pytest.raises(LLMError):
                await call_with_retry(Flaky([cls("x")]), make_policy())
        assert LLMError.__subclasses__() == before

    async def test_keyboard_interrupt_is_not_caught(self):
        with pytest.raises(KeyboardInterrupt):
            await call_with_retry(Flaky([KeyboardInterrupt()]), make_policy())

    async def test_cancelled_error_from_the_call_is_not_retried_but_recorded(self):
        policy = make_policy()
        fn = Flaky([asyncio.CancelledError()])
        with pytest.raises(asyncio.CancelledError):
            await call_with_retry(fn, policy)
        assert fn.calls == 1
        assert policy.sleeper.delays == []
        assert [r.outcome for r in policy.history] == ["cancelled"]
        assert policy.history[0].attempt == 1
        assert "call was running" in policy.history[0].error
        stats = policy.stats()
        assert stats["cancelled"] == 1 and stats["failed"] == 0 and stats["attempts"] == 0

    async def test_task_cancelled_while_the_call_is_running_is_recorded(self):
        started = asyncio.Event()

        async def slow_call():
            started.set()
            await asyncio.sleep(3600)

        policy = make_policy()
        task = asyncio.ensure_future(call_with_retry(slow_call, policy))
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert [r.outcome for r in policy.history] == ["cancelled"]
        assert policy.stats()["cancelled"] == 1

    @pytest.mark.parametrize("make_sleep", [
        lambda: _raising_sleep,
        lambda: (lambda seconds: None),          # returns a non-awaitable
    ])
    async def test_a_broken_sleep_gives_a_final_recorded_error(self, make_sleep):
        policy = make_policy(sleep=make_sleep())
        fn = Flaky([RateLimitError("limit")] * 5)
        with pytest.raises(LLMError) as info:
            await call_with_retry(fn, policy)
        err = info.value
        assert fn.calls == 1
        assert err.code is LLMErrorCode.RATE_LIMIT and err.attempts == 1
        assert isinstance(err.cause, RateLimitError)
        assert err.__cause__ is not err.cause            # chained to the sleep failure
        assert "wait before the next retry also failed" in str(err)
        assert [r.outcome for r in policy.history] == ["retry", "exhausted"]
        assert "retry wait failed" in policy.history[-1].error
        stats = policy.stats()
        assert stats["attempts"] == 1 and stats["failed"] == 1
        assert stats["failures_by_code"] == {"rate_limit": 1}

    async def test_cancellation_during_the_wait_stops_everything(self):
        started = asyncio.Event()

        async def blocking_sleep(seconds):
            started.set()
            await asyncio.sleep(3600)      # really would block, if not cancelled

        policy = make_policy(sleep=blocking_sleep)
        fn = Flaky([RateLimitError("r")] * 5)
        task = asyncio.ensure_future(call_with_retry(fn, policy))
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fn.calls == 1                           # not retried after cancel
        assert task.cancelled()
        assert [r.outcome for r in policy.history] == ["retry", "cancelled"]
        assert policy.stats()["cancelled"] == 1
        assert policy.stats()["failed"] == 0

    async def test_the_default_sleep_is_cancellable_too(self):
        policy = RetryPolicy(base_delay=3600.0, max_delay=3600.0, jitter=0.0)
        task = asyncio.ensure_future(call_with_retry(Flaky([RateLimitError("r")]), policy))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)

    async def test_original_text_is_in_the_message(self):
        with pytest.raises(LLMError, match="quota exceeded"):
            await call_with_retry(Flaky([BadRequestError("quota exceeded")]), make_policy())


# ─── attempt records and stats ────────────────────────────────────────────────

class TestRecords:

    async def test_each_attempt_is_recorded_with_code_delay_and_error(self):
        policy = make_policy(jitter=0.0)
        fn = Flaky([RateLimitError("slow down"), InternalServerError("oops")])
        await call_with_retry(fn, policy, operation="test.op")

        first, second, third = list(policy.history)
        assert (first.attempt, first.outcome, first.code, first.delay) == (
            1, "retry", LLMErrorCode.RATE_LIMIT, 0.5)
        assert "RateLimitError" in first.error and "slow down" in first.error
        assert (second.attempt, second.outcome, second.code, second.delay) == (
            2, "retry", LLMErrorCode.SERVER, 1.0)
        assert (third.attempt, third.outcome, third.code, third.delay, third.error) == (
            3, "success", None, 0.0, "")
        assert {r.operation for r in policy.history} == {"test.op"}

    async def test_timestamps_come_from_the_injected_clock(self):
        policy = make_policy()
        await call_with_retry(Flaky([RateLimitError("x")]), policy)
        assert [r.timestamp for r in policy.history] == [1001.0, 1002.0]

    async def test_records_are_bounded(self):
        policy = make_policy(max_history=5, max_retries=0)
        for _ in range(20):
            await call_with_retry(Flaky([]), policy)
        assert len(policy.history) == 5
        assert policy.stats()["calls"] == 20            # counters are not bounded
        assert policy.stats()["succeeded"] == 20

    async def test_on_attempt_receives_every_record(self):
        seen = []
        policy = make_policy()
        await call_with_retry(Flaky([RateLimitError("x")]), policy, on_attempt=seen.append)
        assert [r.outcome for r in seen] == ["retry", "success"]

    async def test_on_attempt_may_be_a_coroutine_function(self):
        seen = []

        async def hook(record):
            seen.append(record.outcome)

        await call_with_retry(Flaky([RateLimitError("x")]), make_policy(), on_attempt=hook)
        assert seen == ["retry", "success"]

    async def test_a_broken_hook_does_not_break_the_call(self):
        def hook(record):
            raise RuntimeError("hook bug")

        assert await call_with_retry(Flaky([]), make_policy(), on_attempt=hook) == "ok"

    async def test_final_failure_reaches_the_hook(self):
        seen = []
        with pytest.raises(LLMError):
            await call_with_retry(
                Flaky([AuthenticationError("k")]), make_policy(), on_attempt=seen.append)
        assert seen[-1].outcome == "not_retryable"
        assert seen[-1].code is LLMErrorCode.AUTH

    async def test_stats(self):
        policy = make_policy(max_retries=1)
        await call_with_retry(Flaky([RateLimitError("a")]), policy)           # ok on 2nd
        with pytest.raises(LLMError):
            await call_with_retry(Flaky([AuthenticationError("b")]), policy)  # fails at once
        with pytest.raises(LLMError):
            await call_with_retry(Flaky([InternalServerError("c")] * 2), policy)  # exhausted

        stats = policy.stats()
        assert stats["calls"] == 3
        assert stats["succeeded"] == 1
        assert stats["failed"] == 2
        assert stats["attempts"] == 2 + 1 + 2
        assert stats["retries"] == 2
        assert stats["failures_by_code"] == {"rate_limit": 1, "auth": 1, "server": 2}
        assert stats["last_error"]["code"] == "server"
        assert "InternalServerError: c" in stats["last_error"]["error"]

    async def test_describe_is_plain_data(self):
        import json
        policy = make_policy()
        await call_with_retry(Flaky([RateLimitError("x")]), policy)
        info = policy.describe()
        json.dumps(info)                                  # must be serialisable
        assert info["max_retries"] == 3
        assert info["retryable_codes"] == ["empty", "rate_limit", "server", "timeout", "transport"]
        assert [r["outcome"] for r in info["recent_attempts"]] == ["retry", "success"]
        assert info["stats"]["retries"] == 1

    async def test_reset(self):
        policy = make_policy()
        await call_with_retry(Flaky([RateLimitError("x")]), policy)
        policy.reset()
        assert len(policy.history) == 0
        assert policy.stats()["calls"] == 0 and policy.stats()["last_error"] is None


# ─── Error text safety ────────────────────────────────────────────────────────

class TestErrorTextSafety:

    @pytest.mark.parametrize("text,secret", [
        ("invalid key sk-ant-api03-FAKEFAKEFAKEFAKE1234 rejected", "FAKEFAKEFAKEFAKE1234"),
        ("bad sk-proj-fakefakefake5678", "fakefakefake5678"),
        ("Authorization: Bearer eyJfake.bearer-token_123", "eyJfake.bearer-token_123"),
        ("sent Bearer eyJfake.bearer-token_123 to server", "eyJfake.bearer-token_123"),
        ("authorization=Basic hunter2hunter2", "hunter2hunter2"),
        ("{'x-api-key': 'hunter2hunter2', 'content-type': 'json'}", "hunter2hunter2"),
        ('headers: {"X-Api-Key": "hunter2hunter2"}', "hunter2hunter2"),
        ("GET /v1/models?key=AIzaFakeQueryKey999&alt=json", "AIzaFakeQueryKey999"),
        ("url ?alt=json&api_key=AIzaFakeQueryKey999", "AIzaFakeQueryKey999"),
        ('request api_key="hunter2hunter2" failed', "hunter2hunter2"),
        ("api-key: hunter2hunter2", "hunter2hunter2"),
        ("Proxy-Authorization: Basic hunter2hunter2", "hunter2hunter2"),
    ])
    def test_secrets_are_redacted(self, text, secret):
        cleaned = sanitize_error_text(text)
        assert secret not in cleaned
        assert "[REDACTED]" in cleaned

    def test_text_around_a_secret_is_kept(self):
        cleaned = sanitize_error_text("call failed: x-api-key: abc123 (status 401)")
        assert cleaned.startswith("call failed: x-api-key:")
        assert cleaned.endswith("(status 401)")

    @pytest.mark.parametrize("text", ["plain failure", "monkey=banana", "task-skill-1", ""])
    def test_harmless_text_is_unchanged(self, text):
        assert sanitize_error_text(text) == text

    def test_long_text_is_cut_with_an_ellipsis_and_the_original_length(self):
        cleaned = sanitize_error_text("x" * 1200)
        assert cleaned.startswith("x" * 500 + "...")
        assert "original length 1200" in cleaned
        assert len(cleaned) < 600

    def test_text_at_the_limit_is_not_cut(self):
        assert sanitize_error_text("y" * 500) == "y" * 500

    def test_a_secret_on_the_cut_boundary_is_redacted_before_cutting(self):
        # An AWS key id (20 characters) starts at position 490. Only its first 10 characters
        # are left before the cut at 500, which is too short to be recognised as a key. If
        # cutting ran BEFORE redaction, that fragment ("AKIAABCDEF") would survive.
        text = "f" * 489 + " AKIAABCDEFGHIJKLMNOP" + " tail" * 40
        assert len(text) > 500 and text[490:500] == "AKIAABCDEF"
        cleaned = sanitize_error_text(text)
        assert "ABCDEF" not in cleaned and "AKIA" not in cleaned
        assert "[REDACTED]" in cleaned
        assert "truncated" in cleaned

    def test_truncation_marker_reports_the_length_before_redaction(self):
        text = ("password=" + "x" * 50 + " " + "y" * 40 + " ") * 20      # 2020 characters
        assert len(text) == 2020
        cleaned = sanitize_error_text(text)
        assert "xxxxx" not in cleaned
        assert "original length 2020 characters" in cleaned

    @pytest.mark.parametrize("text,secret", [
        ("password=PWSECRET1 ok", "PWSECRET1"),
        ("passwd=PWSECRET2 ok", "PWSECRET2"),
        ('{"password": "PWSECRET3 with spaces"}', "PWSECRET3"),
        ("client_secret=CSSECRET4&grant=1", "CSSECRET4"),
        ("secret=CSSECRET5 ok", "CSSECRET5"),
        ("token=TOKSECRET6 ok", "TOKSECRET6"),
        ("access_token=TOKSECRET7 ok", "TOKSECRET7"),
        ('{"access_token": "TOKSECRET8"}', "TOKSECRET8"),
        ("Cookie: session=COOKSECRET9; theme=dark", "COOKSECRET9"),
        ("Cookie: session=COOKSECRET9; theme=dark", "theme=dark"),
        ("Set-Cookie: sid=COOKSECRET10; HttpOnly; Secure", "COOKSECRET10"),
        ("{'cookie': 'sid=COOKSECRET11; a=b'}", "COOKSECRET11"),
        ("X-Auth-Token: AUTHSECRET12", "AUTHSECRET12"),
        ("OpenAI-Organization: org-ORGSECRET13", "ORGSECRET13"),
        ("Authorization: Basic BASICSECRET14", "BASICSECRET14"),
        ("Authorization: Token TOKENSECRET15", "TOKENSECRET15"),
        ("Authorization: Bearer BEARSECRET16 rejected", "BEARSECRET16"),
        ("Authorization: CustomScheme CUSTSECRET17", "CUSTSECRET17"),
        ("Authorization: JUSTASECRET18", "JUSTASECRET18"),
        ('Authorization: Digest username="u", nonce="NONCESECRET19", '
         'response="RESPSECRET20", opaque="x"', "NONCESECRET19"),
        ('Authorization: Digest username="u", nonce="NONCESECRET19", '
         'response="RESPSECRET20", opaque="x"', "RESPSECRET20"),
        ("AWS id AKIAABCDEFGHIJKLMNOP end", "AKIAABCDEFGHIJKLMNOP"),
        ("temporary ASIAABCDEFGHIJKLMNOP end", "ASIAABCDEFGHIJKLMNOP"),
        ("aws_secret_access_key=AWSSECRET21/abc+def", "AWSSECRET21"),
        ("key AIza" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r" + " end",
         "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r"),
        ("gh ghp_abcdefghijklmnopqrstuvwxyz012345 end", "abcdefghijklmnopqrstuvwxyz012345"),
        ("gh gho_abcdefghijklmnopqrstuvwxyz012345 end", "abcdefghijklmnopqrstuvwxyz012345"),
        ("gh ghs_abcdefghijklmnopqrstuvwxyz012345 end", "abcdefghijklmnopqrstuvwxyz012345"),
        ("gh github_pat_11ABCDEFG0abcdefghijklmnopqrstuv end",
         "11ABCDEFG0abcdefghijklmnopqrstuv"),
        ("slack xoxb-123456789012-ABCDEFGHIJKL end", "ABCDEFGHIJKL"),
        ("slack xoxa-123456789012-ABCDEFGHIJKL end", "ABCDEFGHIJKL"),
        ("slack xoxp-123456789012-ABCDEFGHIJKL end", "ABCDEFGHIJKL"),
        ("slack xoxr-123456789012-ABCDEFGHIJKL end", "ABCDEFGHIJKL"),
        ("slack xoxs-123456789012-ABCDEFGHIJKL end", "ABCDEFGHIJKL"),
        ("key sk-abc.def.DOTSECRET22 end", "DOTSECRET22"),
        ("key sk-proj-abc-def-DASHSECRET23 end", "DASHSECRET23"),
        ('{\\"api_key\\": \\"ESCSECRET24\\"}', "ESCSECRET24"),
        ("api%5Fkey=URLSECRET25&x=1", "URLSECRET25"),
        ("api%5fkey=URLSECRET26&x=1", "URLSECRET26"),
        ("?a=1&key%3DURLSECRET27", "URLSECRET27"),
        ("?a=1&password%3DURLSECRET28", "URLSECRET28"),
        ('api_key = "SPACEA SPACEB" after', "SPACEA"),
        ('api_key = "SPACEA SPACEB" after', "SPACEB"),
        ("api_key = 'SPACEC SPACED' after", "SPACED"),
        ("\uff50\uff41\uff53\uff53\uff57\uff4f\uff52\uff44=FULLSECRET29", "FULLSECRET29"),
        ("ｔｏｋｅｎ＝FULLSECRET30", "FULLSECRET30"),
        ("ａｐｉ＿ｋｅｙ: FULLSECRET31", "FULLSECRET31"),
        ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMSJ9.JWTSIGNATURE32", "JWTSIGNATURE32"),
        ('bad "password":"NOSPACESECRET33"', "NOSPACESECRET33"),
    ])
    def test_more_secret_shapes_are_redacted(self, text, secret):
        cleaned = sanitize_error_text(text)
        assert secret not in cleaned
        assert "[REDACTED]" in cleaned

    @pytest.mark.parametrize("text", [
        "plain failure", "monkey=banana", "task-skill-1", "max_tokens=100", "tokens used 5",
        "status 429 rate limited", "ask-me-later", "the key is missing",
    ])
    def test_harmless_text_survives(self, text):
        assert sanitize_error_text(text) == text

    def test_redaction_is_stable_when_applied_twice(self):
        text = ("Authorization: Digest nonce=\"N1\" Cookie: a=b password = \"p q\" "
                "key=K1 aws_secret_access_key=ABC sk-abcdefgh12345")
        once = sanitize_error_text(text)
        assert sanitize_error_text(once) == once

    def test_every_pattern_is_fast_on_100k_character_inputs(self):
        size = 100_000

        def rep(unit):
            return (unit * (size // len(unit) + 1))[:size]

        inputs = {
            "labels": rep("api_key="), "password": rep("password="),
            "password+spaces": "password" + " " * size,
            "password+open quote": 'password="' + "a" * size,
            "password+quoted spaces": 'password="' + "a b " * (size // 4),
            "password+unclosed": rep('password="a '), "escaped quotes": rep('\\"'),
            "quotes": '"' * size, "apostrophes": "'" * size,
            "auth+spaces": "Authorization:" + " " * size,
            "auth words": "Authorization: " + "a " * (size // 2),
            "auth repeat": rep("Authorization: x y "),
            "digest": "Authorization: Digest " + "a" * size, "digest repeat": rep("authorization:digest "),
            "cookie": "Cookie: " + "a" * size, "cookie repeat": rep("cookie:"),
            "cookie lines": rep("cookie: a\n"),
            "sk": "sk-" + "a" * size, "sk repeat": rep("sk-"), "sk dots": "sk-" + "a." * (size // 2),
            "akia": rep("AKIA"), "akia long": "AKIA" + "A" * size,
            "aiza": rep("AIza"), "aiza long": "AIza" + "a" * size,
            "ghp": rep("ghp_"), "ghp long": "ghp_" + "a" * size, "pat": rep("github_pat_"),
            "xox": rep("xoxb-"), "xox long": "xoxb-" + "a" * size,
            "%5F": rep("%5F"), "%3D": rep("%3D"), "key%3D": rep("key%3D"),
            "api%5Fkey": rep("api%5Fkey"), "token=====": "token" + "=" * size,
            "token quote repeat": rep('token="'), "json escaped": rep('{\\"api_key\\": \\"'),
            "bearer repeat": rep("bearer "), "bearer+spaces": "bearer" + " " * size,
            "bearer long": "bearer " + "a" * size,
            "jwt": "eyJ" + "a" * size, "jwt dots": rep("eyJaaaaaaaaaa."),
            "full-width": "ａ" * size, "full-width label": rep("ｐａｓｓｗｏｒｄ＝"),
            "combining": rep("e\u0301"), "backslashes": "\\" * size,
            "key repeat": rep("key="), "monkey": rep("monkey"), "plain": "a" * size,
            "partial words": "secre" * (size // 5), "token repeat": rep("token"),
            "label with space sep": rep("token : "),
        }
        slowest = 0.0
        for name, text in inputs.items():
            started = time.perf_counter()
            sanitize_error_text(text)
            took = time.perf_counter() - started
            slowest = max(slowest, took)
            assert took < 2.0, f"{name} took {took:.2f}s"
        assert slowest < 2.0

    async def test_records_stats_and_message_are_sanitised_but_cause_is_untouched(self):
        secret_text = ("401 for sk-ant-api03-FAKEFAKEFAKEFAKE1234, "
                       "Authorization: Bearer eyJfake.bearer-token_123 " + "z" * 800)
        original = AuthenticationError(secret_text)
        policy = make_policy()
        seen = []
        with pytest.raises(LLMError) as info:
            await call_with_retry(Flaky([original]), policy, on_attempt=seen.append)

        err = info.value
        assert err.cause is original
        assert "FAKEFAKEFAKEFAKE1234" in str(original)          # cause keeps everything
        everything = [str(err), err.history[0].error, seen[0].error,
                      policy.history[0].error, policy.stats()["last_error"]["error"],
                      str(policy.describe())]
        for text in everything:
            assert "FAKEFAKEFAKEFAKE1234" not in text
            assert "eyJfake.bearer-token_123" not in text
        assert "original length" in err.history[0].error
        assert len(err.history[0].error) < 700
        assert len(str(err)) < 800

    async def test_retry_record_text_is_sanitised_too(self):
        policy = make_policy()
        exc = RateLimitError(f"?key=AIzaFakeQueryKey999 {'q' * 900}")
        await call_with_retry(Flaky([exc]), policy)
        assert "AIzaFakeQueryKey999" not in policy.history[0].error
        assert len(policy.history[0].error) < 700

    async def test_error_whose_class_name_raises_is_still_handled(self):
        class Meta(type):
            @property
            def __name__(cls):
                raise RuntimeError("no name")

        class Odd(Exception, metaclass=Meta):
            pass

        with pytest.raises(LLMError) as info:
            await call_with_retry(Flaky([Odd("odd failure")]), make_policy())
        assert "odd failure" in str(info.value)

    async def test_error_whose_str_raises_is_still_handled(self):
        class BadStr(Exception):
            def __str__(self):
                raise RuntimeError("no str")

        with pytest.raises(LLMError) as info:
            await call_with_retry(Flaky([BadStr()]), make_policy())
        assert "BadStr" in str(info.value)


# ─── Fake SDKs for the plugin tests ───────────────────────────────────────────

class ScriptedSDK:
    """
    A fake SDK client whose create() plays back a script, one item per call.
    An item that is an exception is raised, anything else is returned.
    """

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return item


def install_anthropic(monkeypatch, script):
    sdk = ScriptedSDK(script)
    module = types.ModuleType("anthropic")
    module.AsyncAnthropic = lambda api_key=None: SimpleNamespace(messages=sdk)
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return sdk


def install_openai(monkeypatch, script):
    sdk = ScriptedSDK(script)
    module = types.ModuleType("openai")
    module.AsyncOpenAI = lambda api_key=None: SimpleNamespace(
        chat=SimpleNamespace(completions=sdk))
    monkeypatch.setitem(sys.modules, "openai", module)
    return sdk


def a_text(text):
    return SimpleNamespace(type="text", text=text)


def a_tool(name="search", args=None):
    return SimpleNamespace(type="tool_use", name=name, input=args or {}, id="toolu_1")


def a_response(blocks, tokens=(10, 5)):
    return SimpleNamespace(
        content=blocks, usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]))


def o_response(content="hello", tool_calls=None, total=42, choices=True):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)] if choices else [],
        usage=SimpleNamespace(total_tokens=total),
    )


USER = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _no_env_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


# ─── AnthropicPlugin ──────────────────────────────────────────────────────────

class TestAnthropicRetry:

    async def test_success_on_first_call_is_unchanged(self, monkeypatch):
        sdk = install_anthropic(monkeypatch, [a_response([a_text("hello")])])
        policy = make_policy()
        plugin = AnthropicPlugin(api_key="k", retry_policy=policy)

        assert await plugin.complete("p") == "hello"
        assert len(sdk.requests) == 1
        assert policy.sleeper.delays == []
        assert policy.stats()["succeeded"] == 1

    async def test_complete_retries_a_rate_limit_then_succeeds(self, monkeypatch):
        sdk = install_anthropic(monkeypatch, [RateLimitError("429"), a_response([a_text("ok")])])
        policy = make_policy()
        plugin = AnthropicPlugin(api_key="k", retry_policy=policy)

        assert await plugin.complete("p") == "ok"
        assert len(sdk.requests) == 2
        assert sdk.requests[0] == sdk.requests[1]          # same request is repeated
        assert len(policy.sleeper.delays) == 1

    async def test_chat_retries_a_server_error_then_succeeds(self, monkeypatch):
        sdk = install_anthropic(monkeypatch, [
            with_status(529), with_status(503), a_response([a_text("hi")], (7, 3))])
        plugin = AnthropicPlugin(api_key="k", retry_policy=make_policy())

        result = await plugin.chat(USER)

        assert result == {"content": "hi", "tool_calls": [], "tokens_used": 10}
        assert len(sdk.requests) == 3

    async def test_complete_exhaustion_gives_llm_error(self, monkeypatch):
        sdk = install_anthropic(monkeypatch, [with_status(500)])
        plugin = AnthropicPlugin(api_key="k", retry_policy=make_policy())

        with pytest.raises(LLMError) as info:
            await plugin.complete("p")

        assert info.value.code is LLMErrorCode.SERVER
        assert info.value.attempts == 4
        assert len(sdk.requests) == 4

    async def test_chat_exhaustion_gives_llm_error(self, monkeypatch):
        install_anthropic(monkeypatch, [APITimeoutError("slow")])
        plugin = AnthropicPlugin(api_key="k", retry_policy=make_policy(max_retries=1))
        with pytest.raises(LLMError) as info:
            await plugin.chat(USER)
        assert info.value.code is LLMErrorCode.TIMEOUT and info.value.attempts == 2

    @pytest.mark.parametrize("method", ["complete", "chat"])
    @pytest.mark.parametrize("error,code", [
        (AuthenticationError("bad key"), LLMErrorCode.AUTH),
        (BadRequestError("bad"), LLMErrorCode.BAD_REQUEST),
    ])
    async def test_auth_and_bad_request_are_not_retried(self, monkeypatch, method, error, code):
        sdk = install_anthropic(monkeypatch, [error])
        plugin = AnthropicPlugin(api_key="k", retry_policy=make_policy())
        arg = "p" if method == "complete" else USER

        with pytest.raises(LLMError) as info:
            await getattr(plugin, method)(arg)

        assert len(sdk.requests) == 1
        assert info.value.code is code and info.value.attempts == 1

    async def test_empty_content_in_complete_is_retried_then_a_clear_error(self, monkeypatch):
        sdk = install_anthropic(monkeypatch, [a_response([])])
        plugin = AnthropicPlugin(api_key="k", retry_policy=make_policy())

        with pytest.raises(LLMError) as info:
            await plugin.complete("p")

        assert not isinstance(info.value, IndexError)
        assert info.value.code is LLMErrorCode.EMPTY
        assert info.value.attempts == 4 and len(sdk.requests) == 4
        assert "no content blocks" in str(info.value)

    async def test_empty_content_can_recover_on_retry(self, monkeypatch):
        install_anthropic(monkeypatch, [a_response([]), a_response([a_text("now it works")])])
        plugin = AnthropicPlugin(api_key="k", retry_policy=make_policy())
        assert await plugin.complete("p") == "now it works"

    async def test_empty_chat_response_is_retried_then_a_clear_error(self, monkeypatch):
        sdk = install_anthropic(monkeypatch, [a_response([])])
        plugin = AnthropicPlugin(api_key="k", retry_policy=make_policy())
        with pytest.raises(LLMError) as info:
            await plugin.chat(USER)
        assert info.value.code is LLMErrorCode.EMPTY and len(sdk.requests) == 4

    async def test_chat_with_only_a_tool_call_is_not_empty(self, monkeypatch):
        sdk = install_anthropic(monkeypatch, [a_response([a_tool("search", {"q": "x"})])])
        result = await AnthropicPlugin(api_key="k", retry_policy=make_policy()).chat(USER)
        assert result["tool_calls"] == [{"name": "search", "arguments": {"q": "x"}, "id": "toolu_1"}]
        assert len(sdk.requests) == 1

    async def test_complete_with_a_non_text_first_block_still_returns_empty_string(
            self, monkeypatch):
        install_anthropic(monkeypatch, [a_response([a_tool()])])
        assert await AnthropicPlugin(api_key="k", retry_policy=make_policy()).complete("p") == ""

    async def test_a_refusal_in_chat_is_returned_not_retried(self, monkeypatch):
        refusal = a_response([])
        refusal.stop_reason = "refusal"
        sdk = install_anthropic(monkeypatch, [refusal])
        plugin = AnthropicPlugin(api_key="k", retry_policy=make_policy())

        result = await plugin.chat(USER)

        assert result == {"content": "", "tool_calls": [], "tokens_used": 15}
        assert len(sdk.requests) == 1
        assert plugin.retry_stats()["retries"] == 0

    async def test_a_refusal_in_complete_returns_an_empty_string(self, monkeypatch):
        refusal = a_response([])
        refusal.stop_reason = "refusal"
        sdk = install_anthropic(monkeypatch, [refusal])
        plugin = AnthropicPlugin(api_key="k", retry_policy=make_policy())
        assert await plugin.complete("p") == ""
        assert len(sdk.requests) == 1

    async def test_other_stop_reasons_with_no_content_are_still_empty(self, monkeypatch):
        cut = a_response([])
        cut.stop_reason = "max_tokens"
        sdk = install_anthropic(monkeypatch, [cut])
        plugin = AnthropicPlugin(api_key="k", retry_policy=make_policy())
        with pytest.raises(LLMError) as info:
            await plugin.chat(USER)
        assert info.value.code is LLMErrorCode.EMPTY and len(sdk.requests) == 4

    async def test_message_errors_before_the_call_are_not_retried(self, monkeypatch):
        sdk = install_anthropic(monkeypatch, [a_response([a_text("x")])])
        plugin = AnthropicPlugin(api_key="k", retry_policy=make_policy())
        with pytest.raises(KeyError):
            await plugin.chat([{"content": "no role"}])
        assert sdk.requests == []

    async def test_attempts_are_recorded_on_the_plugin(self, monkeypatch):
        install_anthropic(monkeypatch, [RateLimitError("429"), a_response([a_text("ok")])])
        seen = []
        plugin = AnthropicPlugin(
            api_key="k", retry_policy=make_policy(), on_attempt=seen.append)

        await plugin.complete("p")

        history = plugin.retry_history()
        assert [(r.attempt, r.outcome) for r in history] == [(1, "retry"), (2, "success")]
        assert history[0].code is LLMErrorCode.RATE_LIMIT
        assert history[0].operation == "anthropic.complete"
        assert seen == history
        assert plugin.retry_stats()["retries"] == 1
        info = plugin.describe()
        assert info["plugin"] == "AnthropicPlugin"
        assert info["retry"]["stats"]["succeeded"] == 1

    def test_each_plugin_gets_its_own_default_policy(self, monkeypatch):
        install_anthropic(monkeypatch, [a_response([a_text("x")])])
        one, two = AnthropicPlugin(api_key="k"), AnthropicPlugin(api_key="k")
        assert one.retry_policy is not two.retry_policy
        assert one.retry_policy.max_retries == 3

    async def test_retries_can_be_turned_off(self, monkeypatch):
        sdk = install_anthropic(monkeypatch, [with_status(500)])
        plugin = AnthropicPlugin(api_key="k", retry_policy=make_policy(max_retries=0))
        with pytest.raises(LLMError):
            await plugin.complete("p")
        assert len(sdk.requests) == 1

    async def test_cancellation_during_the_wait_propagates(self, monkeypatch):
        started = asyncio.Event()

        async def blocking_sleep(seconds):
            started.set()
            await asyncio.sleep(3600)

        sdk = install_anthropic(monkeypatch, [RateLimitError("429")])
        plugin = AnthropicPlugin(api_key="k", retry_policy=make_policy(sleep=blocking_sleep))
        task = asyncio.ensure_future(plugin.complete("p"))
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(sdk.requests) == 1

    async def test_temperature_handling_is_untouched(self, monkeypatch):
        sdk = install_anthropic(monkeypatch, [a_response([a_text("x")])])
        plugin = AnthropicPlugin(api_key="k", temperature=0.9, retry_policy=make_policy())
        await plugin.complete("p")
        await plugin.chat(USER)
        assert plugin.temperature == 0.9
        assert all("temperature" not in request for request in sdk.requests)


# ─── OpenAIPlugin ─────────────────────────────────────────────────────────────

class TestOpenAIRetry:

    async def test_success_on_first_call_is_unchanged(self, monkeypatch):
        sdk = install_openai(monkeypatch, [o_response("hello")])
        policy = make_policy()
        plugin = OpenAIPlugin(api_key="k", retry_policy=policy)

        assert await plugin.complete("p") == "hello"
        assert len(sdk.requests) == 1 and policy.sleeper.delays == []

    async def test_complete_retries_then_succeeds(self, monkeypatch):
        sdk = install_openai(monkeypatch, [RateLimitError("429"), o_response("ok")])
        plugin = OpenAIPlugin(api_key="k", retry_policy=make_policy())
        assert await plugin.complete("p") == "ok"
        assert len(sdk.requests) == 2

    async def test_chat_retries_a_connection_error_then_succeeds(self, monkeypatch):
        sdk = install_openai(monkeypatch, [
            APIConnectionError("net"), with_status(502), o_response("hi", total=33)])
        plugin = OpenAIPlugin(api_key="k", retry_policy=make_policy())

        result = await plugin.chat(USER)

        assert result == {"content": "hi", "tool_calls": [], "tokens_used": 33}
        assert len(sdk.requests) == 3

    async def test_complete_exhaustion_gives_llm_error(self, monkeypatch):
        sdk = install_openai(monkeypatch, [ConnectionError("down")])
        plugin = OpenAIPlugin(api_key="k", retry_policy=make_policy())
        with pytest.raises(LLMError) as info:
            await plugin.complete("p")
        assert info.value.code is LLMErrorCode.TRANSPORT
        assert info.value.attempts == 4 and len(sdk.requests) == 4

    async def test_chat_exhaustion_gives_llm_error(self, monkeypatch):
        install_openai(monkeypatch, [with_status(503)])
        plugin = OpenAIPlugin(api_key="k", retry_policy=make_policy(max_retries=2))
        with pytest.raises(LLMError) as info:
            await plugin.chat(USER)
        assert info.value.code is LLMErrorCode.SERVER and info.value.attempts == 3

    @pytest.mark.parametrize("method", ["complete", "chat"])
    @pytest.mark.parametrize("error,code", [
        (with_status(401), LLMErrorCode.AUTH),
        (with_status(400), LLMErrorCode.BAD_REQUEST),
    ])
    async def test_auth_and_bad_request_are_not_retried(self, monkeypatch, method, error, code):
        sdk = install_openai(monkeypatch, [error])
        plugin = OpenAIPlugin(api_key="k", retry_policy=make_policy())
        arg = "p" if method == "complete" else USER

        with pytest.raises(LLMError) as info:
            await getattr(plugin, method)(arg)

        assert len(sdk.requests) == 1
        assert info.value.code is code

    async def test_no_choices_in_complete_is_empty_not_index_error(self, monkeypatch):
        sdk = install_openai(monkeypatch, [o_response(choices=False)])
        plugin = OpenAIPlugin(api_key="k", retry_policy=make_policy())
        with pytest.raises(LLMError) as info:
            await plugin.complete("p")
        assert not isinstance(info.value, IndexError)
        assert info.value.code is LLMErrorCode.EMPTY and len(sdk.requests) == 4

    async def test_no_choices_in_chat_is_empty(self, monkeypatch):
        install_openai(monkeypatch, [o_response(choices=False)])
        plugin = OpenAIPlugin(api_key="k", retry_policy=make_policy())
        with pytest.raises(LLMError) as info:
            await plugin.chat(USER)
        assert info.value.code is LLMErrorCode.EMPTY

    async def test_chat_with_no_text_and_no_tool_calls_is_empty_and_can_recover(
            self, monkeypatch):
        install_openai(monkeypatch, [o_response(None), o_response("recovered")])
        plugin = OpenAIPlugin(api_key="k", retry_policy=make_policy())
        result = await plugin.chat(USER)
        assert result["content"] == "recovered"
        assert plugin.retry_history()[0].code is LLMErrorCode.EMPTY

    async def test_a_refusal_in_chat_is_returned_not_retried(self, monkeypatch):
        response = o_response(None)
        response.choices[0].message.refusal = "I cannot help with that."
        sdk = install_openai(monkeypatch, [response])
        plugin = OpenAIPlugin(api_key="k", retry_policy=make_policy())

        result = await plugin.chat(USER)

        assert result == {"content": "", "tool_calls": [], "tokens_used": 42}
        assert len(sdk.requests) == 1
        assert plugin.retry_stats()["retries"] == 0

    async def test_a_content_filter_stop_is_returned_not_retried(self, monkeypatch):
        response = o_response(None)
        response.choices[0].finish_reason = "content_filter"
        sdk = install_openai(monkeypatch, [response])
        plugin = OpenAIPlugin(api_key="k", retry_policy=make_policy())
        assert (await plugin.chat(USER))["content"] == ""
        assert len(sdk.requests) == 1

    async def test_an_empty_refusal_field_does_not_count_as_a_refusal(self, monkeypatch):
        response = o_response(None)
        response.choices[0].message.refusal = None
        sdk = install_openai(monkeypatch, [response])
        plugin = OpenAIPlugin(api_key="k", retry_policy=make_policy(max_retries=1))
        with pytest.raises(LLMError) as info:
            await plugin.chat(USER)
        assert info.value.code is LLMErrorCode.EMPTY and len(sdk.requests) == 2

    async def test_chat_with_only_tool_calls_is_not_empty(self, monkeypatch):
        call = SimpleNamespace(
            id="c1", function=SimpleNamespace(name="search", arguments='{"q": 1}'))
        sdk = install_openai(monkeypatch, [o_response(None, [call])])
        result = await OpenAIPlugin(api_key="k", retry_policy=make_policy()).chat(USER)
        assert result["tool_calls"] == [{"name": "search", "arguments": {"q": 1}, "id": "c1"}]
        assert len(sdk.requests) == 1

    async def test_complete_with_none_content_still_returns_empty_string(self, monkeypatch):
        install_openai(monkeypatch, [o_response(None)])
        assert await OpenAIPlugin(api_key="k", retry_policy=make_policy()).complete("p") == ""

    async def test_attempts_are_recorded_on_the_plugin(self, monkeypatch):
        install_openai(monkeypatch, [with_status(429), o_response("ok")])
        plugin = OpenAIPlugin(api_key="k", retry_policy=make_policy())

        await plugin.chat(USER)

        history = plugin.retry_history()
        assert [(r.attempt, r.outcome, r.code) for r in history] == [
            (1, "retry", LLMErrorCode.RATE_LIMIT), (2, "success", None)]
        assert history[0].operation == "openai.chat"
        assert plugin.describe()["retry"]["stats"]["failures_by_code"] == {"rate_limit": 1}

    async def test_cancellation_during_the_wait_propagates(self, monkeypatch):
        started = asyncio.Event()

        async def blocking_sleep(seconds):
            started.set()
            await asyncio.sleep(3600)

        sdk = install_openai(monkeypatch, [with_status(500)])
        plugin = OpenAIPlugin(api_key="k", retry_policy=make_policy(sleep=blocking_sleep))
        task = asyncio.ensure_future(plugin.chat(USER))
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(sdk.requests) == 1

    async def test_temperature_handling_is_untouched(self, monkeypatch):
        sdk = install_openai(monkeypatch, [o_response("x")])
        plugin = OpenAIPlugin(api_key="k", temperature=0.9, retry_policy=make_policy())
        await plugin.complete("p")
        await plugin.chat(USER)
        assert plugin.temperature == 0.9
        assert all("temperature" not in request for request in sdk.requests)


# ─── Passwords inside URLs (found by an independent check of the redaction filter) ─


@pytest.mark.parametrize("text,secret", [
    ("https://user:hunter2@api.example.com/v1", "hunter2"),
    ("proxy error via http://svc:S3cr3tPass@proxy.internal:8080", "S3cr3tPass"),
    ("postgres://admin:p%40ss@db.internal/app failed", "p%40ss"),
    ("HTTPS://User:TopSecret@Host/", "TopSecret"),
    ("connect to socks5://u:pw-with-dashes_123@10.0.0.1:1080", "pw-with-dashes_123"),
])
def test_passwords_inside_urls_are_redacted(text, secret):
    out = sanitize_error_text(text)
    assert secret not in out
    assert "[REDACTED]" in out
    assert "@" in out  # the rest of the URL is kept so the error is still useful


@pytest.mark.parametrize("text", [
    "https://example.com:8080/path",
    "mailto:someone@example.com",
    "contact user@example.com about https://example.com/a:b",
    "http://host/path?x=a:b@c",
    "ssh://git@github.com/org/repo.git",
    "plain text with a colon: and an at @ sign",
])
def test_urls_without_a_password_are_left_alone(text):
    assert sanitize_error_text(text) == text


def test_url_password_redaction_is_stable_and_fast():
    once = sanitize_error_text("https://user:hunter2@api.example.com/v1")
    assert sanitize_error_text(once) == once
    import time

    for nasty in ["a://" * 25000, "http://" + "u" * 100000 + ":" + "p" * 100000,
                  "http://" + "u:" * 50000 + "@", "x://:" * 25000]:
        started = time.perf_counter()
        sanitize_error_text(nasty)
        assert time.perf_counter() - started < 2.0
