# Retrying LLM calls

Calls to an LLM provider fail for many reasons. Most are short-lived: the provider is busy, the network drops, the answer comes back empty. `AnthropicPlugin` and `OpenAIPlugin` now try again for these, wait a little longer each time, and give every failure a clear code. The code is in `aria/plugins/retry.py` and uses only the Python standard library.

When the first call works, nothing changes. You get the same result as before.

## What gets retried

Every failure is sorted into one of eight codes. The plugin does not need the provider's own library to do this. It looks at the HTTP status, at the name of the error, and at the standard Python network errors.

| Code | Typical cause | Retried? |
| --- | --- | --- |
| `RATE_LIMIT` | HTTP 429, "RateLimitError" | yes |
| `SERVER` | HTTP 500 to 599 (including 529 "overloaded"), HTTP 409 (conflict), "InternalServerError" | yes |
| `TIMEOUT` | HTTP 408, "APITimeoutError", Python `TimeoutError` | yes |
| `TRANSPORT` | dropped connection, DNS failure, proxy error, "APIConnectionError", `ConnectionError` | yes |
| `EMPTY` | the provider answered but there was no usable text or content | yes |
| `AUTH` | HTTP 401 or 403, wrong key, no permission, a TLS certificate that fails verification | never |
| `BAD_REQUEST` | any other HTTP 4xx (409 and 429 excepted, see above), "BadRequestError", model not found | never |
| `UNKNOWN` | anything we do not recognise, including bugs in our own code | never |

Retrying a wrong API key or a malformed request only repeats the same mistake, so those fail at once. A bad TLS certificate will not fix itself either, so it is treated like an auth problem. An error we cannot recognise also fails at once, so a bug is never hidden behind several slow retries.

Notes on how errors are recognised:

- An HTTP status (on the error, or on its `response`) wins over the error's class name.
- Both vendor SDKs retry a 409 conflict, so it is retried here too. There is no "conflict" code, so it is filed under `SERVER`, the closest one.
- Class names are only used for errors that look like they come from a provider. Plain programming errors (`ValueError`, `TypeError`, `KeyError`, `IndexError`, `AttributeError`, `AssertionError`, `LookupError`) are never sorted by name, so a `TimeoutConfigError(ValueError)` is `UNKNOWN`, not a timeout. If such an error carries an HTTP status, the name is used as before.
- Looking at an error never raises. If reading an attribute fails, the error is `UNKNOWN`.

An answer counts as `EMPTY` when:

- Anthropic returns no content blocks (this used to crash with an `IndexError`)
- OpenAI returns no choices
- a chat call gets neither text nor tool calls back

A completion that returns a non-text block (Anthropic) or `None` content (OpenAI) still returns an empty string, exactly as before.

## How long it waits

By default a call is tried once, then retried up to 3 times (4 attempts in total). The wait before each retry doubles:

| Retry | Wait (before jitter) |
| --- | --- |
| 1 | 0.5 seconds |
| 2 | 1 second |
| 3 | 2 seconds |

The wait never goes above 30 seconds. A little random jitter makes each wait up to 25 percent shorter, so many callers that fail together do not all retry at the same moment. Jitter only ever shortens a wait, so it can never push it past the cap.

If the provider says how long to wait (a `retry_after` value on the error, or a `retry-after` or `retry-after-ms` header), ARIA waits that long instead, up to the 30 second cap. Only plain numbers of seconds are understood. A date in the header is ignored and the normal backoff is used.

## When it gives up

If the retries run out, or the error is one that is never retried, the plugin raises `LLMError`. It carries:

- `code`: the `LLMErrorCode` of the last failure
- `attempts`: how many calls were made
- `cause`: the original exception (it is also chained, so the traceback shows both)
- `history`: the attempt records for this one call

The message says what happened, for example `LLM call failed after 4 attempt(s) [rate_limit]: RateLimitError: slow down`.

`LLMError` is a plain exception. It is not a subclass of the original error, so code that used to catch `ConnectionError` or `TimeoutError` around a plugin call must now catch `LLMError` and look at `.cause` or `.code`. The original exception is never changed. `LLMError` can be pickled as long as its `cause` can be.

Cancelling is always respected. If the task is cancelled during a wait, `asyncio.CancelledError` goes straight up and nothing is retried.

## Seeing what happened

Nothing is hidden. Every attempt is recorded, including the ones that worked.

```python
plugin = AnthropicPlugin(api_key="...")
await plugin.complete("hello")

plugin.retry_history()   # list of AttemptRecord, newest last
plugin.retry_stats()     # counters
plugin.describe()        # settings, counters and the last few attempts as plain data
```

Each `AttemptRecord` has: `operation` (for example `anthropic.chat`), `attempt` (1, 2, 3 ...), `outcome`, `code`, `delay` (seconds waited before the next try), `error` (the error text), `timestamp` and `retry_after`. The outcome is one of `success`, `retry`, `exhausted` (retries ran out, or the wait between attempts failed), `not_retryable`, or `cancelled` (the task was cancelled, while the call was running or while waiting).

The list keeps the last 200 records so it cannot grow without limit. The counters (`calls`, `succeeded`, `failed`, `cancelled`, `attempts`, `retries`, `failures_by_code`, `last_error`) are never trimmed. ### What is and is not stored

Stored in the records, the counters' `last_error`, and the `LLMError` message: the error type name and the error text, with obvious secrets hidden and cut to 500 characters (when cut, an ellipsis and the original length are added). The secrets that are hidden are `sk-...` style keys, `Bearer` tokens, `Authorization` and `Proxy-Authorization` values, `x-api-key` and `api_key` values, and `key=`, `api_key=` and `token=` values in a URL. They are replaced with `[REDACTED]`.

Not stored by this code: prompts, responses and request bodies. But error text is written by the provider and can quote part of a request, and the filter only knows the shapes listed above. It does not find other secrets or personal data. Treat the records as possibly sensitive and do not publish them without looking.

`LLMError.cause` is the original exception, untouched, with its full text. If you log the traceback, Python prints the original message too, unfiltered.

To send each attempt to the audit trail as it happens, pass a function:

```python
plugin = AnthropicPlugin(api_key="...", on_attempt=lambda record: audit.log(record.to_dict()))
```

A hook that raises an error is logged and ignored. It cannot break the LLM call.

## Changing the settings

```python
from aria.plugins.retry import RetryPolicy

plugin = OpenAIPlugin(retry_policy=RetryPolicy(max_retries=5, base_delay=1.0, max_delay=60.0))
plugin = OpenAIPlugin(retry_policy=RetryPolicy(max_retries=0))   # no retries
```

Other settings: `multiplier` (default 2.0), `jitter` (0 to 1, default 0.25) and `max_history` (default 200).

Settings are checked when the policy is created. `max_retries` and `max_history` must be whole numbers (`True`, `2.0` and `nan` are rejected), with `max_retries` 0 or more and `max_history` 1 or more. `base_delay` and `max_delay` must be finite numbers 0 or more, `multiplier` 1 or more, and `jitter` between 0 and 1. A wrong type raises `TypeError`, a wrong value raises `ValueError`, and the message names the setting.

For tests you can replace the waiting and the randomness: `RetryPolicy(sleep=fake_sleep, clock=fake_clock, rng=lambda: 0.0)`, so a test never really waits. If you do not pass `sleep`, `asyncio.sleep` is looked up at the moment of waiting.

Your own plugin can use the same helper:

```python
from aria.plugins.retry import call_with_retry, RetryPolicy

policy = RetryPolicy()
text = await call_with_retry(lambda: my_client.generate(prompt), policy, operation="my.complete")
```

If your call can come back empty, raise `EmptyResponseError` inside the function and it will be retried as `EMPTY`.

## Behaviour changes

Things that behave differently from before this change:

- `chat` can now raise `LLMError(EMPTY)` after the retries where it used to return an empty answer. This happens when the reply has no text and no tool calls: for example a reply with only a "thinking" block, a refusal with no content, or an answer cut off by the length limit before any text.
- `complete` still returns an empty string when the first block is not text (Anthropic) or the content is `None` (OpenAI). Only a response with no content blocks or no choices at all counts as `EMPTY`.
- Errors that used to escape as the provider's own exception (or `IndexError`) now arrive as `LLMError`. See "When it gives up" for how to read the original.
- Tokens spent on attempts that were discarded as empty (or that failed) are not counted in the `tokens_used` that `chat` returns. Only the successful attempt is counted, so the contract does not charge for them.
- An OpenAI 429 with `insufficient_quota` is retried like any other 429, even though waiting will not add credit. It fails after the retries run out (about 3.5 seconds with the defaults) with `code=RATE_LIMIT`.

## Not covered here

- The `temperature` setting is not part of this change (see issue #33).
- Streaming responses are not supported by the plugins, so there is nothing to retry half-way through a stream.
- Time spent waiting adds to the time a call takes. With the defaults a call that keeps failing takes about 3.5 seconds of waiting before it gives up, or much more if the provider asks for longer waits.
