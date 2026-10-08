# Spilling large output to disk

This page explains how ARIA handles agent and tool output that is too big to
show to a model in full. The code is in `aria/core/spill.py`, and it is used
by `LLMAgent` in `aria/agents/base.py`. Tracked in issue #40.

## Why this exists

Before, `LLMAgent` cut long text with plain slices: results passed between
nodes were cut to 300 characters, and tool results to 1000. The cut was silent.
The model saw a sentence that just stopped, with no sign that anything was
missing and no way to get the rest. A model could then give a confident answer
based on half of a document.

## What happens now

* Text that fits the limit is passed through exactly as before. Nothing is
  saved and nothing is added.
* Text that does not fit is saved in full to a private file. The model gets a
  preview instead: the start of the text, the end of the text, and a note in
  between that says how much was left out and how to read it.
* If the file cannot be saved (disk full, no permission, anything), the whole
  text stays in the message and a warning is recorded. ARIA never drops
  information to save space.

An example of what the model sees (shortened here with `...` for the page):

```
line 00000
line 00001
...
line 00060
[... 5600 characters omitted. This output was SHORTENED: showing the first 667 and last 333 of 6600 characters. The full text is saved under spill id 3f9c1d0a5b7e4c2f8a6d9e1b0c4a7f23. To read the omitted part, call read_spilled(locator="3f9c1d0a5b7e4c2f8a6d9e1b0c4a7f23", offset=667, length=2000); offset and length count characters. ...]
...
line 00598
line 00599
```

## How a model reads the rest

When something is spilled, the agent offers the model a tool called
`read_spilled(locator, offset=0, length=2000)`. The model passes the spill id
from the note. Each answer says which characters it covered and, when there is
more, the exact call to continue:

```
...text...
[characters 667 to 2667 of 6600; to continue call read_spilled(locator="3f9c...", offset=2667)]
```

Offsets and lengths count characters (Python text positions), not bytes, so
accents, Asian scripts and emoji are never split in the middle. One call
returns at most 20,000 characters; the answer says where it stopped.

`read_spilled` is added to an agent only after the first spill. Agents that
never meet large output look exactly as they did before. It is not added to the
agent's advertised tool list used for routing.

## Where the files go, and who can read them

Spilled text can contain private data (web pages, user files, tool output), so
the folder is locked down:

* The folder is created with mode `0700` (owner only), and an existing folder
  that is looser is tightened. A folder owned by someone else, or one that is a
  symbolic link, is not used.
* Each item is its own file with a random name and mode `0600` (owner only),
  created exclusively, so an existing file is never overwritten and a planted
  link is never followed.
* The model never sees a file path. It sees a random 32 character id. `read`
  accepts only ids that this store issued in this process, checks them against
  a strict pattern, and confirms the file really sits inside the folder, so
  `../x`, absolute paths, empty or huge ids and NUL characters are refused.
* On Windows the mode bits do not apply the same way; the folder is a normal
  per-user temp folder there.

Default location: the folder named by the `ARIA_SPILL_DIR` environment
variable, or else `aria-spill-<user id>` in the system temp folder. It is never
inside the repository. (`aria_spill/` is also in `.gitignore`, in case someone
points the setting at a folder in a checkout.) You can also give an agent its
own store: `LLMAgent(..., spill_store=SpillStore("/some/private/folder"))`.

## Retention

Spill files are temporary. By default they are kept for one day.
`store.cleanup()` removes files older than the retention time
(`cleanup(max_age_seconds=...)` overrides it) and `store.clear_all()` removes
all of them. The default store sweeps old files once when it is first created.
Only files this module could have created are ever touched. Reading an item
after it was removed returns a clear "not found, it may have expired" error.

## Health information

`store.describe()` returns the folder, file count, bytes on disk, how many items
were spilled and how many characters were left out of previews in this run, and
the warning counts. `store.warnings` lists the most recent warnings (up to 100),
for example a failed save.

## Limits

* The preview limits are `LLMAgent.DEPENDENCY_PREVIEW_CHARS` and
  `LLMAgent.TOOL_RESULT_PREVIEW_CHARS`, both 1000 characters. The note is added
  on top of that.
* This is not a security guard. Checking what a model may do with spilled
  output, or what a tool may return, belongs to the guarded pipeline (#42).
