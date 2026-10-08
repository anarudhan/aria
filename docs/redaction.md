# Redaction

ARIA hides secrets before text or tool arguments are written to a log, an audit record or an error report. The code is in `aria/security/redact.py`. It uses only the Python standard library and imports nothing else from ARIA, so any part of the framework can use it without causing an import cycle.

**Redaction is best effort. It is not a guarantee.** It knows a list of secret shapes and key names. A secret in any other shape goes through unchanged. Treat stored records as possibly sensitive and look before you publish them.

## What is in the module

| Name | What it does |
| --- | --- |
| `sanitize_error_text(text, limit=500)` | Hides common secret shapes in text, then cuts it to `limit` characters. |
| `MAX_ERROR_TEXT` | The default limit, 500. |
| `redact_args(value, *, max_depth=6, max_items=100, max_string=200)` | Returns a redacted, JSON-safe copy of tool arguments, for audit records. |
| `ArgsDigester` | A keyed digest of tool arguments (HMAC-SHA256). |
| `default_digester()` | The shared `ArgsDigester` of this process, made on first use. |
| `REDACTED` | The marker text, `[REDACTED]`. |

`sanitize_error_text` and `MAX_ERROR_TEXT` used to live in `aria/plugins/retry.py`. They moved without any change in behaviour, and `retry.py` still exports them, so `from aria.plugins.retry import sanitize_error_text` keeps working. The shapes it hides (passwords in URLs, `Authorization` and `Cookie` headers, `Bearer` tokens, `sk-` keys, JSON web tokens, AWS, Google, GitHub and Slack keys, and values after labels such as `password=`, `token:` and `api_key=`) are listed in [llm-retry.md](llm-retry.md).

## Redacting tool arguments

```python
from aria.security.redact import redact_args

safe = redact_args({"url": "https://x.test", "headers": {"Authorization": "Bearer abc"}, "retries": 3})
# {"url": "https://x.test", "headers": {"Authorization": "[REDACTED]"}, "retries": 3}
```

`redact_args` never changes what you give it and never raises. The result holds only plain JSON types (dict with text keys, list, text, int, finite float, bool, None), so it can be stored as it is.

### What is covered

- **Secret key names.** A dict value is replaced by `[REDACTED]` when its key names a secret, whatever the type of the value (text, number, list, a whole nested dict). The value is not even looked at. Keys are compared case-insensitively, after splitting on `-`, `_`, spaces, dots and camelCase, and after Unicode NFKC. So `apiKey`, `API_KEY` and `X-Api-Key` are the same key. A key counts as secret when:
  - it contains `password`, `passwd`, `passphrase`, `secret`, `credential`, `authorization`, `apikey`, `accesskey`, `privatekey`, `signingkey`, `encryptionkey`, `masterkey`, `sshkey`, `authkey`, `sessionkey` or `clientkey` (so `aws_secret_access_key`, `clientSecret`, `DB_PASSWORD` and `credentials` are covered), or
  - it has one of these as a whole word: `token`, `auth`, `oauth`, `cookie`, `session`, `bearer`, `pwd`, `pass`, `passcode`, `jwt`, with an optional plural "s" (so `access_token`, `refreshToken`, `x-auth-token`, `session_id` and `cookies` are covered), or
  - it is made of several words and the last one is `key` (`service_key`, `my_custom_key`).

  A number at the end of a word is ignored (`token2`, `api_key_1`).
- **Text values.** Every text value goes through `sanitize_error_text`, cut to `max_string` characters (200). The text is redacted first and cut after, so a secret at the cut point is not left half visible.
- **Text keys.** Dict keys go through the same text redaction.
- **Numbers, bools and None** are kept. `nan` and `inf` become the text `"nan"`, `"inf"` and `"-inf"`. An integer of more than 256 bits becomes a short marker.
- **Bytes** become `<bytes N>` (the length only). Their content is never copied.
- **Any other object** becomes `<TypeName>`. Its `repr`, `str` and attributes are never used, because they could print a secret or raise an error.
- **Tuples, sets and frozensets** become lists. A set keeps its own iteration order.
- **Limits.** Below `max_depth` levels (6) the value becomes `<max depth>`. A list, tuple, set or dict with more than `max_items` entries (100) is cut and ends with `<N more items omitted>`. A container that contains itself becomes `<cycle>`. Dict keys that are not text are turned into text without calling the key's own methods (`1` becomes `"1"`, an object becomes `"<TypeName>"`). Two keys that give the same text are kept apart as `name` and `name#2`.
- **Bounded work.** A hostile or huge input cannot make the call slow: `max_depth` is at most 64, one call looks at no more than 10,000 values and 1,000,000 characters in total, and one text over 200,000 characters is replaced by `<string, N characters, not scanned>` without being read. When the budget runs out the rest becomes `<budget exceeded>`. If anything unexpected goes wrong inside a container, that container becomes `<TypeName, unreadable>`.

### What is not covered

- **A secret under a harmless key, in a shape nobody knows.** `{"note": "my pin is 4821"}` goes through unchanged. So does `{"data": "k3j2h4k2j3h4"}`.
- **A secret next to its label, not inside the same text.** `["--password", "hunter2"]` is two separate texts, and the second has no label. `tool --password hunter2` (a space and no `=` or `:`) is not recognised either.
- **Bare `key`.** A key named only `key` is not treated as secret (cache and database keys are common). A key named `max_tokens` or `sort_key` is treated as secret (over-redaction is the safe side).
- **Other mapping types.** Only `dict`, `list`, `tuple`, `set` and `frozenset` (and their subclasses) are walked. A `deque`, `range`, generator or custom mapping becomes `<TypeName>`.
- **Personal data** such as names, e-mail addresses and phone numbers.
- **Secrets split by markup or by line breaks**, and secrets that only appear after decoding (base64, URL escapes beyond the few listed).
- **Strings over 200,000 characters** are not scanned at all. They are replaced, so nothing leaks, but the content is lost.

If a secret must never be stored, do not pass it to a tool as an argument. Use a reference (a name or an id) and look the secret up inside the tool.

## The keyed digest

An audit record needs a way to tell whether two calls had the same arguments, and an approval needs to be tied to exactly the arguments the person saw. A digest does that without storing the arguments.

A plain hash is not safe for this. If the arguments hold a weak secret (a short password, a four digit PIN), anyone who can read the stored hash can try every likely value, hash each one, and compare. Hashing does not stop that, because it needs no secret.

`ArgsDigester` uses HMAC-SHA256 with a random 32 byte key made when the object is created (`secrets.token_bytes(32)`). The key lives only in that object in memory. It is not printed by `repr`, the object cannot be pickled or copied, and each digester has its own key. Without the key, a guess cannot be checked against the digest, so the stored value gives an attacker nothing to test against.

```python
from aria.security.redact import default_digester

digest = default_digester().digest({"path": "/tmp/x", "mode": "w"})   # 64 hex characters
```

- The arguments are first written as canonical JSON: sorted keys, no spaces, ASCII only. Equal arguments give the same digest whatever the key order. A tuple and a list are the same in JSON, so they give the same digest.
- One digester gives the same digest for the same input every time. Two digesters (or two runs of the program) give different digests, so a digest is only meaningful inside the process that made it. Do not compare digests between runs.
- Values JSON cannot hold are rejected with `TypeError` and are never turned into text: bytes, sets, arbitrary objects, `nan`, `inf`, dict keys that are not text, cycles, nesting too deep for the JSON encoder, and integers too large to print. Pass the original arguments, not the output of `redact_args`, because the redacted copy would give the same digest for different secrets.
- The digest hides the secret from someone who only has the stored digest. It does not help against someone who can read the memory of the running process, because the key is there too.
