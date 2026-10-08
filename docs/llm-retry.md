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
- Names are matched as fragments on errors that are not plain programming errors, so a class named like `ReadError` or `ConnectionStringInvalid` is treated as `TRANSPORT`. A rare false match costs a few retries, nothing more.
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

If the provider says how long to wait (a `retry_after` value on the error, or a `retry-after` or `retry-after-ms` header), ARIA waits that long instead, up to the 30 second cap. Only plain numbers of seconds are understood. A date in the header is ignored and the normal backoff is used. A value of `0` means "retry at once": the retry happens immediately, and still stops after `max_retries`.

## When it gives up

If the retries run out, or the error is one that is never retried, the plugin raises `LLMError`. It carries:

- `code`: the `LLMErrorCode` of the last failure
- `attempts`: how many calls were made
- `cause`: the original exception (it is also chained, so the traceback shows both)
- `history`: the attempt records for this one call

The message says what happened, for example `LLM call failed after 4 attempt(s) [rate_limit]: RateLimitError: slow down`.

`LLMError` is a plain exception. It is not a subclass of the original error, so code that used to catch `ConnectionError` or `TimeoutError` around a plugin call must now catch `LLMError` and look at `.code` (and `.cause` if you need the original). The original exception is never changed.

`LLMError` can be pickled and copied, but pickling drops the cause. Real SDK errors often cannot be loaded back, and pickling them would copy their unredacted text. After pickling, `err.cause` is `None`, and what is kept is `code`, `attempts`, the message, `cause_type` (the original class name), `cause_text` (the original message, redacted) and `history`. `copy.copy` and `copy.deepcopy` keep the same `cause` object.

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

The list keeps the last 200 records so it cannot grow without limit. The counters (`calls`, `succeeded`, `failed`, `cancelled`, `attempts`, `retries`, `failures_by_code`, `last_error`) are never trimmed. The counters are plain integers and are not thread safe. That is fine inside one asyncio event loop, which is how the plugins run. Do not share one policy between threads.

### What is and is not stored

Stored in the records, the counters' `last_error`, `cause_text` and the `LLMError` message: the error type name and the error text, with common secrets hidden and cut to 500 characters. When cut, an ellipsis is added with the length of the text as it was before any redaction or cutting.

The redaction code (`sanitize_error_text`) lives in `aria/security/redact.py` and is re-exported from `aria/plugins/retry.py`, so both import paths work. What it covers, what it does not, and the matching tool for tool arguments are described in [redaction.md](redaction.md).

**Redaction is best effort. It is a list of known shapes, not a guarantee.** Anything not on the list below goes through unchanged, and error text written by a provider can quote part of your request. Treat the records as possibly sensitive and look before you publish them.

The text is first normalised with NFKC (so full-width look-alike letters become normal ones), then these shapes are replaced with `[REDACTED]`:

- Values after `password`, `passwd`, `passphrase`, `secret` (so `client_secret` too), `token` (so `access_token` and `x-auth-token` too), `api_key` (and `x-api-key`), `access_key` (so `aws_secret_access_key` too), `private_key`, `openai-organization` and `signature` (so `x-amz-signature` too), and the whole words `pwd`, `pass` (so `DB_PASS`) and `sig`, written with `=` or `:`. This covers plain text, JSON, backslash-escaped JSON (`{\"api_key\": \"X\"}`), URL-encoded labels (`api%5Fkey=X`, `key%3DX`), and quoted values that contain spaces and backslash escapes (`password="ab\"cd ef"`). A bare `key=` (for example in a URL query) is covered too.
- `Authorization` and `Proxy-Authorization` values with any scheme (Basic, Bearer, Token, Digest and others). For Digest the whole header value, including `nonce` and `response`, is hidden.
- `Cookie` and `Set-Cookie` header values.
- The password in `scheme://user:password@host` URLs (proxy and connection errors often print these). The user name may be empty (`redis://:password@host`). A URL with only a user name, or only a port, is left alone.
- PEM and PGP private keys (`-----BEGIN ... PRIVATE KEY-----`, also `PRIVATE KEY BLOCK`), from the BEGIN line to the END line, or to the end of the text when there is no END line.
- `Bearer` tokens anywhere, `sk-` keys (also ones with dots or dashes), JSON web tokens (`eyJ...`), AWS key ids (`AKIA...`, `ASIA...`), Google keys (`AIza...`), GitHub tokens (`ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_`, `github_pat_`) and Slack tokens (`xoxa-`, `xoxb-`, `xoxp-`, `xoxr-`, `xoxs-`).

Labels split by invisible characters (a zero-width space, a soft hyphen, a control character, a combining mark, a Hangul filler inside `password`), written with accents, or with Cyrillic or Greek look-alike letters are read too, and so are look-alikes of `:` and `=` (U+2236, U+A789, U+2254, U+2255, U+A78A), and a value that starts with an invisible character. Only the secret is replaced; the rest of the text, invisible characters and accents included, stays as it was. A label split by a visible character (a space, a dot, a tab) is not read.

A text of more than 1,000,000 characters (counted after NFKC, which can make one character up to 18, and counted before any redaction), or one with a run of more than 32 combining marks at a slice boundary, is not scanned. Every text up to the limit is scanned. The marker `[REDACTED]` counts as one ordinary value character, so `password=[REDACTED]@hunter2`, `Cookie: [REDACTED] sid=x` and `Bearer [REDACTED]hunter2` are hidden whole like the same text without a marker, and a value that already is the marker is left as it is. The error text then becomes `<text, N characters, not scanned>`, and `ExceptionType: ` is kept in front of it in the attempt records. Nothing from such a text is kept.

Not covered: secrets with no label and no known prefix, secrets split across lines or by markup, an opening quote that is never closed (only the first word after it is hidden), personal data, and anything not listed above.

Not stored by this code: prompts, responses and request bodies.

**Tracebacks and `cause` still hold the original text.** `LLMError.cause` and `LLMError.__cause__` are the original exception, untouched, with its full unredacted message and attributes (for an SDK error that can include the response body and headers). Python prints that chained exception when it prints a traceback, so `logger.exception(...)` or `traceback.print_exc()` on an `LLMError` can leak a secret into your logs. To log a failure safely, log `str(err)` together with `err.code` (and `err.attempts`). Do not log `err.cause` or the traceback unless you know the destination is safe.

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

Settings are checked when the policy is created. `max_retries` and `max_history` must be whole numbers (`True`, `2.0` and `nan` are rejected), with `max_retries` 0 or more and `max_history` 1 or more. `base_delay` and `max_delay` must be finite numbers 0 or more, `multiplier` 1 or more, and `jitter` between 0 and 1. `max_retries` may be at most 100 and `max_history` at most 100000. `sleep`, `clock` and `rng` must be callables (`clock` and `rng` may also be `None`, meaning the default). A wrong type raises `TypeError`, a wrong value raises `ValueError`, and the message names the setting.

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

- `chat` can now raise `LLMError(EMPTY)` after the retries where it used to return an empty answer. This happens when the reply has no text and no tool calls: for example a reply with only a "thinking" block, or an answer cut off by the length limit before any text. (A refusal is not one of these, see below.)
- `complete` still returns an empty string when the first block is not text (Anthropic) or the content is `None` (OpenAI). Only a response with no content blocks or no choices at all counts as `EMPTY`.
- Errors that used to escape as the provider's own exception (or `IndexError`) now arrive as `LLMError`. See "When it gives up" for how to read the original.
- Tokens spent on attempts that were discarded as empty (or that failed) are not counted in the `tokens_used` that `chat` returns. Only the successful attempt is counted, so the contract does not charge for them.
- Refusals are the exception to the `EMPTY` rule. A refusal is a final answer and retrying it only costs money, so it is not retried and does not become an error. OpenAI `chat` returns what it always returned (empty content, no tool calls) when the message has a `refusal`, or when the finish reason is `content_filter`. Anthropic `chat` and `complete` do the same when the `stop_reason` is `refusal`. An empty reply for any other reason (for example `max_tokens`) is still `EMPTY`.
- An OpenAI 429 with `insufficient_quota` is retried like any other 429, even though waiting will not add credit. It fails after the retries run out (about 3.5 seconds with the defaults) with `code=RATE_LIMIT`.

## Not covered here

- The `temperature` setting is not part of this change (see issue #33).
- Streaming responses are not supported by the plugins, so there is nothing to retry half-way through a stream.
- Time spent waiting adds to the time a call takes. With the defaults a call that keeps failing takes about 3.5 seconds of waiting before it gives up, or much more if the provider asks for longer waits.
