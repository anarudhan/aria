# Redaction

ARIA hides secrets before text or tool arguments are written to a log, an audit record or an error report. The code is in `aria/security/redact.py`. It uses only the Python standard library.

**Redaction is best effort. It is not a guarantee.** It knows a list of secret shapes and key names. A secret in any other shape goes through unchanged. Treat stored records as possibly sensitive and look before you publish them.

## What is in the module

| Name | What it does |
| --- | --- |
| `sanitize_error_text(text, limit=500)` | Hides common secret shapes in text, then cuts it to `limit` characters. |
| `MAX_ERROR_TEXT` | The default limit, 500. |
| `redact_args(value, *, max_depth=6, max_items=100, max_string=200)` | Returns a redacted, JSON-safe copy of tool arguments, for audit records. |
| `ArgsDigester` | A keyed digest of tool arguments (HMAC-SHA256), with `digest(obj, purpose="args")` and `matches(obj, digest, purpose="args")`. |
| `default_digester()` | The shared `ArgsDigester` of this process, made on first use. |
| `REDACTED` | The marker text, `[REDACTED]`. |

`sanitize_error_text` and `MAX_ERROR_TEXT` used to live in `aria/plugins/retry.py`. They moved here from `aria/plugins/retry.py`, which still exports them, so `from aria.plugins.retry import sanitize_error_text` keeps working.

### Imports and cycles

`redact.py` itself imports only the standard library. But `import aria.security.redact` still runs `aria/__init__.py` and `aria/security/__init__.py`, and that loads the patcher and the scanner. There is no import cycle today only because `aria/security/patcher.py` imports the permission engine under `TYPE_CHECKING` only. **That rule must stay**: if the patcher (or the scanner) imports `aria.permissions.engine` at run time, then an engine that imports `aria.security.redact` would create a cycle. Keep run-time imports of the permission engine out of `aria/security/`.

## Text: `sanitize_error_text`

It first normalises the text with NFKC (so full-width look-alike letters become plain ones), then replaces these shapes with `[REDACTED]`, and last cuts the text. The marker of a cut text reports the length before redaction.

- Values after `password`, `passwd`, `passphrase`, `secret`, `token`, `api_key`, `access_key`, `private_key`, `openai-organization`, `signature` (so `client_secret`, `access_token`, `x-amz-signature` and `x-amz-security-token` too), a bare `key=`, and the whole words `pwd`, `pass` (so `DB_PASS=`) and `sig`, written with `=` or `:`. This covers plain text, JSON, backslash-escaped JSON and URL-encoded labels. Quoted values may contain spaces and backslash escapes (`password="ab\"cd ef"` is hidden whole).
- `Authorization` and `Proxy-Authorization` values with any scheme, and `Cookie` and `Set-Cookie` values.
- The password in `scheme://user:password@host` URLs, also with an empty user (`redis://:password@host`).
- PEM and PGP private keys (`-----BEGIN ... PRIVATE KEY-----`, also `... PRIVATE KEY BLOCK-----`, up to the END line, or up to the end of the text when there is none).
- `Bearer` tokens, `sk-` keys, JSON web tokens, AWS key ids, Google keys, GitHub tokens and Slack tokens.

**Split and look-alike labels.** The patterns are also run on a "shadow" of the text, and what they find there is hidden in the original text. The shadow is the text with these characters removed: format characters (Cf), marks (Mn, Mc, Me), private use (Co), unassigned (Cn), lone surrogates (Cs), control characters (Cc) except tab, line feed, vertical tab, form feed, carriage return and U+0085, and also the Hangul fillers U+115F, U+1160, U+3164 and U+FFA0, the Braille blank U+2800, the Arabic tatweel U+0640 and the modifier apostrophe U+02BC. Cyrillic and Greek look-alike letters (the same small list the key rules use) are folded to Latin, and the look-alikes of `:` and `=` (U+2236, U+A789 for `:`; U+2254, U+2255, U+A78A for `=`) are folded to those. So `pa<zero width space>ssword=x`, `pasṡword=x`, `p<Cyrillic а>ssword=x`, `password<U+2236>x` and `password=<zero width space> x` are all hidden.

Every shadow character remembers which character of the original it came from. Only the secret (the value, and any characters inside it) is replaced in the original text. Everything else stays as it was: invisible characters, accents, Cyrillic prose and the spelling of the label are not changed. The sweep over all patterns is repeated, at most 3 times, until a sweep hides nothing more.

**Two paths, and what holds on each.** A text that has something to fold takes the *mapped path* described above. On the mapped path the output is a fixed point in the cases tested: running the function again changes nothing. A text that has nothing to fold (all ASCII with no control characters, or text such as Chinese or Korean whose shadow is the text itself) takes the *plain path*: one pass of the patterns, as it worked before the shadow was added (the marker rule below is the only deliberate difference). The plain path is not always idempotent. In rare pathological shapes made of several URL-like pieces, a second run hides more than the first. One example is `",user:pw@hostBasic}=http://pass://user:pw@hostDigestsig,eyJ pwdpwd &abc123def456]\n,, '"`. In one random test, about 1 in 1,800 texts glued together from label words, quotes, `@` and `://` pieces behaved like that.

**One more character can change the result.** Because the two paths run the patterns on different strings, adding a single non-ASCII character (or a control character) to a text can make the mapped path hide a secret that the plain path leaves, in those same rare URL-like shapes. It hides more, never less.

**The marker rule.** Our own marker `[REDACTED]` counts as ONE ordinary value character. A value that starts with, ends with or contains the marker is hidden whole and replaced by one marker, exactly like the same value written with any other character: `password=[REDACTED]hunter2`, `password=[REDACTED]@hunter2`, `password=[REDACTED][REDACTED]hunter2`, `Cookie: [REDACTED] sid=hunter2`, `Cookie: [REDACTED]; sid=hunter2`, `Authorization: Bearer [REDACTED]hunter2`, `Bearer [REDACTED]hunter2`, `sk-[REDACTED]hunter2` and `password="abc"[REDACTED]x` all come out with the value replaced by one `[REDACTED]`. A forged marker therefore protects almost nothing that the same text without a marker would not protect. There are two exceptions, both listed under "What is not covered": a `Cookie` line whose marker is followed by `,`, `]` or `}` (the redactor's own JSON-style output `{"cookie": [REDACTED], "x": 1}` looks the same, so the line is left alone to keep a second run stable), and the two-word `Authorization` case. A value that already is the marker (`password=[REDACTED]`) is replaced by itself, so redacting twice changes nothing; a replacement that gives back the same text is not counted as a change. A quoted value takes the characters stuck to its closing quote with it: `password="abc"hunter2` becomes `password=[REDACTED]`.

`Cookie:` takes the rest of the line, except that a quoted value (or our marker) ends the match when it is a whole JSON-style value, that is when it is followed by `,`, `}`, `]` or the end of the line. So `{"cookie": "sid=1", "x": 2}` keeps `"x": 2}`, while `Cookie: "a=b"; sid=hunter2` hides everything after the colon. `Authorization:` takes one or two words: `Authorization: Basic abc extra` becomes `Authorization: [REDACTED] extra`, and a value that is only markers (`Authorization: [REDACTED] extra`) is left as it is, because that is exactly what the first run leaves behind.

The size cap is applied to the input: a text of more than 1,000,000 characters after NFKC normalisation (one character can become up to 18, for example U+FDFA) is not scanned. It is replaced by `<text, N characters, not scanned>`, so nothing leaks and the work stays bounded. Redaction making a text longer (a short secret becomes `[REDACTED]`) can never trigger the cap. The same marker is returned for a text with a run of more than 32 combining marks at a slice boundary (a run like that is not written by anyone; following it to its end would make the work unbounded). Every text up to the cap is scanned.

**Timing.** The shadow is built per text from its distinct characters, so the time depends on the text and not on what the process has seen before. `redact_args` keeps one small memo for the length of a single call (never shared with another call, emptied when it holds more than 50,000 characters), so the strings of one call do not repeat the work. Characters that decompose without leaving an ASCII character or a look-alike (Hangul syllables, Indic composites, precomposed non-Latin letters) and the "other letters" category (Hangul, CJK, kana and so on) are skipped in bulk and never go through the slow per-character path.

Measured on one machine, one run each, "measured, not proven"; a slower machine will be slower. "Main" is the version before this follow-up. Without secrets: 1,000,000 characters of one CJK, Korean, Cyrillic or accented character, 0.3 to 0.7 s, the same cold and after the process saw 25,000 other characters; 1,000,000 zero-width spaces, 0.15 s; `redact_args` on 16 strings of 60,000 CJK characters or 15 of 66,000 Korean characters, 0.3 to 0.4 s; 82 strings made of the 12,101 different characters that NFKD changes but NFKC keeps (the one-million budget, the worst case for the decomposition check), 0.75 s (main 0.6 s); 2,000 calls of `sanitize_error_text` on 500 random Korean characters, 0.6 s (main 0.4 s); `redact_args` on 16 strings of 60,000 random characters from all of Unicode, 1.4 to 2.0 s depending on the random draw (main 2.8 to 3.1 s). The slowest texts found are texts made of very many different characters: one string of 900,000 different, mostly unassigned code points takes 1.2 s (main 0.8 s), and `redact_args` on 5 strings of 180,000 different code points of all kinds takes 1.8 to 2.0 s (main 1.8 s), which is the slowest case I know and close to the 2 s limit used by the tests. With dense secrets (one every 7 to 11 characters), 1,000,000 characters take 1.0 to 1.9 s when the labels are split or look-alike (the mapped path) and 0.35 s in plain ASCII. The tests in `tests/test_redact.py` assert less than 2 s for inputs of 200,000 to 1,000,000 characters of these shapes; a few of them are about half of that limit on this machine.

## Redacting tool arguments

```python
from aria.security.redact import redact_args

safe = redact_args({"url": "https://x.test", "headers": {"Authorization": "Bearer abc"}, "retries": 3})
# {"url": "https://x.test", "headers": {"Authorization": "[REDACTED]"}, "retries": 3}
```

`redact_args` never changes what you give it and never raises. The result holds only plain JSON types (dict with text keys, list, text, int, finite float, bool, None), and every text in it can be written as UTF-8 (a lone surrogate becomes U+FFFD).

### What is covered

- **Secret key names.** A dict value is replaced by `[REDACTED]` when its key names a secret, whatever the type of the value (text, number, list, a whole nested dict). The value is not even looked at. Before the key is compared it is normalised with Unicode NFKD, the characters that the text shadow drops (see above: all control characters, format characters, marks, private use, unassigned, the fillers, tatweel and the modifier apostrophe) are dropped, a small explicit list of Cyrillic and Greek look-alike letters is folded to Latin (so a Cyrillic `о` in `tоken` does not help), and the key is lower-cased and split on non-letters and camelCase. So `apiKey`, `API_KEY` and `X-Api-Key` are the same key. A key counts as secret when:
  - it contains one of these (separators ignored): `password`, `passwd`, `passphrase`, `secret`, `credential`, `authorization`, `token`, `pwd`, `cookie`, `jwt`, `bearer`, `signature`, `mnemonic`, `apikey`, `accesskey`, `privatekey`, `privkey`, `privatepem`, `signingkey`, `encryptionkey`, `masterkey`, `sshkey`, `authkey`, `sessionkey`, `sessionid`, `clientkey`, `keydata`, `keymaterial`, `pincode`. So `accesstoken`, `csrftoken`, `setcookie`, `dbpwd`, `aws_secret_access_key`, `key_data` and `private_pem` are covered. `tokenizer` and `tokenize` are the one exception for `token`;
  - it has one of these as a whole word, with an optional plural "s": `token`, `auth`, `oauth`, `cookie`, `session`, `pwd`, `pass`, `passcode`, `jwt`, `credential`, `bearer`, `signature`, `sig`, `otp`, `totp`, `pin`, `seed`, `mnemonic`, `privkey`;
  - it ends in `auth` (`xauth`, `proxyauth`), or it is made of several words and the last one is `key` (`service_key`, `my_custom_key`);
  - it is longer than 1,000 characters. A key that long cannot be checked cheaply, so it is treated as secret instead of looking only at part of it.

  A number at the end of a word is ignored (`token2`, `api_key_1`).
- **Keys that are not text.** A bytes key is read as Latin-1 and classified like text. A number, bool or `None` key is kept as text (`"1"`). Any other key (a tuple, an object, a frozenset) cannot be read safely, so its value is hidden.
- **Name and value pairs.** `{"name": "DB_PASSWORD", "value": "hunter2"}`: when a dict has a `name` or `key` entry (any case) whose text is a secret name, its `value` entry is hidden. `("Authorization", "Basic abc")` and `["Cookie", "sid=abc"]`: a list or tuple of exactly two items whose first item is a secret name has its second item hidden. The hidden value is never read. The name text itself stays visible. This is how environment variables and header lists are usually written.
- **Text values.** Every text value goes through `sanitize_error_text`, cut to `max_string` characters (200). The text is redacted first and cut after, so a secret at the cut point is not left half visible. Text keys go through the same redaction.
- **Numbers, bools and None** are kept. `nan` and `inf` become the text `"nan"`, `"inf"` and `"-inf"`. An integer of more than 256 bits becomes a short marker.
- **Bytes** become `<bytes N>` (the length only).
- **Any other object** becomes `<TypeName>`. Its `repr`, `str`, attributes and even the `__name__` property of its class are never used (a metaclass could run code there). The name is read straight from the class and goes through text redaction.
- **Tuples, sets and frozensets** become lists. A set keeps its own iteration order.
- **Limits.** Below `max_depth` levels (6) the value becomes `<max depth>`. A list, tuple, set or dict with more than `max_items` entries (100) is cut and ends with `<N more items omitted>`. A container that contains itself becomes `<cycle>`. Two keys that give the same text are kept apart as `name` and `name#2`.
- **Bounded work.** `max_depth` is at most 64. One call looks at no more than 10,000 values, and the characters it scans add up to no more than 1,000,000, counted after NFKC normalisation (which can turn one character into up to 18). One text over 200,000 characters (after normalisation) is replaced by `<string, N characters, not scanned>` without being scanned. When the budget runs out the rest becomes a marker. If anything unexpected goes wrong inside a container, that container becomes `<TypeName, unreadable>`. The timing was measured with the inputs in `tests/test_redact.py` (100,000-character strings, 10,000 levels of nesting, expanding characters, shared sub-structures); it is not a proof for every input.

### What is not covered

- **A `Cookie` line whose marker is followed by `,`, `]` or `}`** (also with white space before that character), for example `Cookie: [REDACTED], sid=hunter2`, keeps the text after the marker. The same line without a marker is hidden whole. The redactor's own JSON-style output looks identical, so this is the price of a stable second run.
- **`Authorization: [REDACTED] hunter2` and `Authorization: Basic [REDACTED] hunter2`** keep the last word. A value that is only markers counts as done, because that is exactly what `Authorization: Basic abc extra` leaves behind; hiding more on a second run would break idempotence. This is a trade-off, not the same as the unmarked text.
- **Text spelled with Unicode tag characters** (U+E0020 to U+E007F) is not read as content by the shadow, so a value hidden in them is kept.
- **A secret under a harmless key, in an unknown shape.** `{"note": "my pin is 4821"}` goes through unchanged. So does `{"data": "k3j2h4k2j3h4"}`.
- **A secret next to its label in a longer list.** The pair rule needs exactly two items. `["tool", "--password", "hunter2"]` (an argument list) is not recognised, and neither is `tool --password hunter2` (a space and no `=` or `:`) inside one text. A pair whose first item is longer than 256 characters is not taken as a name.
- **Other name and value spellings.** Only the entries called `name` or `key` followed by `value` are linked. `{"ParameterKey": "DbPassword", "ParameterValue": "x"}` is not.
- **Bare `key`.** A key named only `key` is not treated as secret (cache and database keys are common). Other names are over-redacted on purpose: `max_tokens`, `sort_key`, `pass_through`, `random_seed`, `session_id` and `pin` are all hidden.
- **Visible characters inside a label.** The shadow drops invisible characters only. A label split by a visible character is not read: a space, dot or apostrophe (`pa ssword`, `p.assword`), a no-break or ideographic space, a line or paragraph separator, a modifier letter such as `ʰ`, or a control character that counts as white space (tab, line feed, vertical tab, form feed, carriage return, U+0085). `pa<tab>ssword=x` is therefore not hidden, and neither is a Greek transliteration such as `ρασσωορδ`.
- **Other separator look-alikes.** Only the five look-alikes listed above, and the ones NFKC already turns into `:` or `=` (full-width and small forms), are read as separators.
- **Other look-alike letters and spelling tricks.** Only Cyrillic and Greek look-alikes are folded. Digit-for-letter spellings (`p4ssw0rd`) and other scripts are not.
- **Other mapping types.** Only `dict`, `list`, `tuple`, `set` and `frozenset` (and their subclasses) are walked. A `deque`, `range`, generator or custom mapping becomes `<TypeName>`.
- **Personal data** such as names, e-mail addresses and phone numbers.
- **Secrets split by markup or by line breaks**, and secrets that only appear after decoding (base64, URL escapes beyond the few listed).
- **Public-looking labels in text.** The text label `pass:` is treated as a password, so ordinary text such as `pass: ok` loses its value. That is over-redaction on purpose.
- **Double-escaped quotes.** A value that is JSON-encoded twice (`json.dumps(json.dumps({"password": 'p"w d'}))`) still leaks the part after the escaped quote.
- **Letter-spaced and other-script spellings.** `p.i.n` and `s_e_s_s_i_o_n` (letters separated by punctuation) are not read as the word. Small-capital letters and look-alikes from scripts other than Cyrillic and Greek are not folded.
- **Over-hiding by the name/value rule.** `{"name": "token", "value": "rm -rf /"}` and `["token", "count"]` have their second entry hidden, because the first says `token`. That is over-redaction on purpose; the digest is still made from the real arguments.
- **More name/value spellings.** The rule links `name` or `key` with `value` only. It does not cover `val` or `data`, `{"id": "password", "value": ...}`, `{"field": ...}`, parallel arrays such as `{"names": [...], "values": [...]}`, or argument lists of three or more items.
- **More than two words after `Authorization:`.** `Authorization: Basic abc hunter2` hides `Basic abc` and leaves `hunter2`, with or without a marker (`Authorization: Basic [REDACTED] hunter2` gives `Authorization: [REDACTED] hunter2`). `Authorization: [REDACTED] hunter2` is left alone because it looks exactly like the output of a two-word value followed by a third word.
- **A value that swallows the next label.** A value runs up to white space or a delimiter and, for `Authorization:`, may take a second word. If the next label is stuck to it (`Authorization:[REDACTED]cookie ...`, `password=xcookie: ...`) the label is part of the value and is hidden with it; what follows it is then not read as the value of that label. This is the same with or without a marker.
- **Values ended by a delimiter.** `password=[REDACTED],hunter2` and `password=x,hunter2` both leave `,hunter2`: a value ends at white space, `,`, `;`, `&`, a quote or a closing bracket.
- **A URL user name that is a label word.** `http://secret:hunter2@host/x` becomes `http://secret:[REDACTED]`: the URL pattern leaves `[REDACTED]@host`, and the label `secret:` in front of it takes `@host` as part of its value. The password is hidden, the host is lost (over-redaction).
- **Very long texts.** A text over 200,000 characters inside `redact_args`, or over 1,000,000 inside `sanitize_error_text`, is replaced entirely. Nothing leaks, but the content is lost.

If a secret must never be stored, do not pass it to a tool as an argument. Use a reference (a name or an id) and look the secret up inside the tool.

## The keyed digest

An audit record needs a way to tell whether two calls had the same arguments, and an approval needs to be tied to exactly the arguments the person saw. A digest does that without storing the arguments.

A plain hash is not safe for this. If the arguments hold a weak secret (a short password, a four digit PIN), anyone who can read the stored hash can try every likely value, hash each one, and compare. Hashing does not stop that, because it needs no secret.

`ArgsDigester` uses HMAC-SHA256 with a random 32 byte key made when the object is created (`secrets.token_bytes(32)`). The key lives only in that object in memory. It is not printed by `repr`, the object cannot be pickled or copied, and each digester has its own key. Without the key, a guess cannot be checked against the digest, so the stored value gives an attacker nothing to test against.

```python
from aria.security.redact import default_digester

digester = default_digester()
digest = digester.digest({"path": "/tmp/x", "mode": "w"})        # 64 hex characters
digester.matches({"path": "/tmp/x", "mode": "w"}, digest)        # True
```

- **Canonical JSON.** The arguments are first written as canonical JSON: sorted keys, no spaces, ASCII only. Equal arguments give the same digest whatever the key order. A tuple and a list are the same in JSON, so they give the same digest.
- **Purpose.** `digest(obj, purpose="args")` puts the purpose in front of the data inside the HMAC input, with a 4 byte length so that no purpose and data can be mistaken for another. The same value digested as `"args"` and as `"result"` gives different digests. A purpose is 1 to 32 ASCII characters: a letter, then letters, digits, `_`, `.` or `-`. A bad purpose raises `ValueError` (`TypeError` if it is not text).
- **`matches(obj, digest, purpose="args")`** compares in constant time (`hmac.compare_digest`) and returns `False`, and never raises, for a malformed digest, a bad purpose, or arguments that `digest` would reject.
- **One digester, one meaning.** One digester gives the same digest for the same input every time. Two digesters (or two runs of the program) give different digests, so a digest is only meaningful inside the process that made it.
- **Only exact types.** `dict` (with `str` keys), `list`, `tuple`, `str`, `int`, `float`, `bool` and `None` are accepted, and only the exact types. A subclass of any of them (an `OrderedDict`, an `IntEnum`, a `str` subclass) is rejected with `TypeError`, because its own methods could make the data that is hashed differ from the data that was checked. No method of a user class is ever called, and the digest is made from a fresh copy built during the check.
- **Rejected with `TypeError`:** bytes, sets, arbitrary objects, subclasses, `nan`, `inf`, dict keys that are not text, any text or key with a lone surrogate (it would give the same JSON as the matching emoji), integers over 4,096 bits, and cycles.
- **Limits.** One digest looks at no more than 200,000 values and 16,000,000 characters (text and keys), counting every visit, so a value that is shared many times is counted many times and `x = [x, x]` repeated 25 times stops with a `TypeError` instead of taking seconds. Nesting is limited to 100 levels. The error message never contains a value. Ordinary tool arguments, including a file of a few megabytes, are far below these limits.
- Pass the original arguments, not the output of `redact_args`, because the redacted copy would give the same digest for different secrets.
- The digest hides the secret from someone who only has the stored digest. It does not help against someone who can read the memory of the running process, because the key is there too.
