# Retrying LLM calls

Calls to an LLM provider fail for many reasons. Most are short-lived: the provider is busy, the network drops, the answer comes back empty. `AnthropicPlugin` and `OpenAIPlugin` now try again for these, wait a little longer each time, and give every failure a clear code. The code is in `aria/plugins/retry.py` and uses only the Python standard library.

When the first call works, nothing changes. You get the same result as before.

## What gets retried

Every failure is sorted into one of eight codes. The plugin does not need the provider's own library to do this. It looks at the HTTP status, at the name of the error, and at the standard Python network errors.

| Code | Typical cause | Retried? |
| --- | --- | --- |
| `RATE_LIMIT` | HTTP 429, "RateLimitError" | yes |
| `SERVER` | HTTP 500 to 599 (including 529 "overloaded"), "InternalServerError" | yes |
| `TIMEOUT` | HTTP 408, "APITimeoutError", Python `TimeoutError` | yes |
| `TRANSPORT` | dropped connection, DNS failure, "APIConnectionError", `ConnectionError` | yes |
| `EMPTY` | the provider answered but there was no usable text or content | yes |
| `AUTH` | HTTP 401 or 403, wrong key, no permission | never |
| `BAD_REQUEST` | any other HTTP 4xx, "BadRequestError", model not found | never |
| `UNKNOWN` | anything we do not recognise, including bugs in our own code | never |

Retrying a wrong API key or a malformed request only repeats the same mistake, so those fail at once. An error we cannot recognise also fails at once, so a bug is never hidden behind several slow retries.

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

`LLMError` is also an instance of the original error type when there is one. Code that already catches `ConnectionError` or `TimeoutError` around a plugin call keeps working.

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

Each `AttemptRecord` has: `operation` (for example `anthropic.chat`), `attempt` (1, 2, 3 ...), `outcome`, `code`, `delay` (seconds waited before the next try), `error` (the error text), `timestamp` and `retry_after`. The outcome is one of `success`, `retry`, `exhausted` (retries ran out), `not_retryable`, or `cancelled`.

The list keeps the last 200 records so it cannot grow without limit. The counters (`calls`, `succeeded`, `failed`, `cancelled`, `attempts`, `retries`, `failures_by_code`, `last_error`) are never trimmed. Error text can come from the provider, but prompts and responses are not stored.

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

Other settings: `multiplier` (default 2.0), `jitter` (0 to 1, default 0.25) and `max_history` (default 200). For tests you can replace the waiting and the randomness: `RetryPolicy(sleep=fake_sleep, clock=fake_clock, rng=lambda: 0.0)`, so a test never really waits.

Your own plugin can use the same helper:

```python
from aria.plugins.retry import call_with_retry, RetryPolicy

policy = RetryPolicy()
text = await call_with_retry(lambda: my_client.generate(prompt), policy, operation="my.complete")
```

If your call can come back empty, raise `EmptyResponseError` inside the function and it will be retried as `EMPTY`.

## Not covered here

- The `temperature` setting is not part of this change (see issue #33).
- Streaming responses are not supported by the plugins, so there is nothing to retry half-way through a stream.
- Time spent waiting adds to the time a call takes. With the defaults a call that keeps failing takes about 3.5 seconds of waiting before it gives up, or much more if the provider asks for longer waits.
