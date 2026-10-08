# DeepSeek Harness study and intake plan for ARIA

Issue: #35 (spike). Status: findings and recommendation, waiting for maintainer decisions. No ARIA code was changed by this study.

Studied: https://github.com/deepseek-ai/deepseek-harness at commit `5badb15` (2026-10-03), tag `dsh-v0.2.1-alpha.1`, shallow clone.

## 1. The short answer

DeepSeek Harness, called `dsh`, is a large TypeScript agent platform. It is not a library that can be dropped into a Python framework. It has about 60 package families, a Web UI, an Electron desktop app, two SDKs and a plugin framework called Cordis underneath everything. The maintainers label it a developer preview, warn that compatibility will break, and say plainly that it has not had a security audit.

So the useful question is not "how do we copy dsh into ARIA". It is "which of its ideas make ARIA better without breaking ARIA's own rules". Our recommendation has three parts.

1. **Adopt ideas natively in Python.** Several dsh designs are excellent and small enough to rebuild in standard library Python in a few hundred lines each. These are the guarded tool pipeline, the append-only session log, spill to disk, compaction, skills, the sub-agent contract and the workflow limits.
2. **Treat embedding dsh as a later, optional step.** ARIA can talk to dsh as an outside coding agent through a small adapter, but only through ACP, the one protocol that keeps a human approval for every action. This must never become a required dependency.
3. **Do not port code.** The code is TypeScript, welded to Cordis, alpha quality and very large. Porting it would cost far more than rebuilding the few good ideas.

There is also a bonus. Reading how dsh contains file writes exposed a real security bug in ARIA's own permission engine, now tracked in issue #36. That fix should come before any new capability, because the new capabilities would all sit on top of the permission engine.

## 2. How this study was done, and how far to trust it

Four read-only agents each studied one group of packages and were told to label every claim as verified (they read the source file) or docs only. I then re-checked the claims the recommendation depends on by reading the source myself. Nothing from the dsh repository was run, built or installed, and its instruction files were treated as data.

**Verified by reading source** (by the agents, and re-checked by me where marked):
- Approval requests fail closed. With no answerer the outcome is `unavailable`, which means deny. (re-checked)
- The tool pipeline has pre-execute, guard, execute and post-execute stages, and only an `allowed-once` approval lets a call proceed. (re-checked)
- Spill to disk triggers on a configurable token size. (re-checked)
- The session-log upload plugin is enabled by default. (re-checked)
- Neither the Linux sandbox profile nor the Landlock launcher handles network access. I searched for network flags and found none. (re-checked)
- Hooks block only on exit code 2, so any other failure lets the action go ahead. (re-checked)
- Skills put only a name and a trimmed description in the prompt, and load the full text on demand. (re-checked, in the `tool-skill` package)
- The Python SDK launches `dsh` as a child process and speaks JSON-RPC over standard input and output. It has no channel for asking a human to approve an action.

**Taken from the agents without my own re-check:** package sizes and test counts, the exact shape of the compaction bracket, the sub-agent depth rules, the workflow limits, and the goal and schedule behaviour.

**Not looked at by anyone:** the UI packages, the desktop app source, the experimental inspector, voice and Claude-Code-mod packages, how MCP tool calls are approved, how the scheduler does time arithmetic, and the git history (the clone is shallow with one commit). Nothing was tested against a real model.

## 3. What dsh is, in plain words

dsh runs an AI agent that can read and write files, run commands, search, plan, split work into sub-agents and talk to the user through a browser or a desktop window. Everything is a plugin, including the model adapter, the tool list, the session log and the agent loop itself. Each plugin can be swapped from configuration.

The parts that matter most to ARIA:

| Area | What dsh does |
|---|---|
| Session log | An append-only list of facts (turn started, user message, tool call, tool result). The model's view of the conversation is computed from the log and never stored separately. |
| Tool pipeline | Every tool call is frozen, checked by policy, optionally sent to a human, checked by deny-only guards, run with a timeout, then post-processed. Anything unexpected becomes an error result, never a crash. |
| Safety | A kernel sandbox (bwrap, Landlock, Seatbelt) limits where files can be written. Approvals are one-shot. There are no remembered rules. |
| Context | Large tool output is saved to disk and replaced by a short preview. Long conversations are summarised inside a start and end bracket in the log. |
| Delegation | Sub-agents with depth limits, frozen permissions, structured output and "fork versus spawn". Workflows run many agents with hard limits. |
| Continuity | Skills, plans, todo lists, goals, background jobs and scheduled reminders. |
| Surfaces | Web UI, desktop app, JSON-RPC SDK, ACP server, Python and TypeScript SDKs. |

## 4. How it compares with ARIA

Where ARIA is stronger today:
- **Permissions.** ARIA has real grants with a scope, an expiry, a use limit and a hash-chained audit trail. dsh only has one-shot approvals and remembers nothing. (Correction added later: the audit trail is only an ordered log. See section 14.)
- **Self-knowledge.** ARIA scans its own code, patches itself with rollback, grades its own health and learns from outcomes. dsh has nothing comparable.
- **Weight.** ARIA needs no outside packages. dsh needs Node, pnpm, a plugin framework and dozens of dependencies.

Where dsh is stronger:
- **Execution discipline.** A strict pipeline for every tool call, with fail-closed approvals and results that are always recorded.
- **Context handling.** ARIA cuts text with fixed slices such as `[:300]` and `[:1000]`. dsh spills big output to disk, tells the model what was left out, and compacts history safely.
- **Recoverability.** dsh can repair a session that crashed halfway through a tool call. ARIA has no session log to repair.
- **Delegation and limits.** dsh caps how deep sub-agents can nest and how many agents a workflow can start in total. ARIA does cap how deeply a task is broken down (`max_decomposition_depth` in the decomposer), but I found no cap on the total number of agents or tasks a run can start, and no rule that narrows a child task's permissions below its parent's.
- **Process.** Snapshot tests of prompts, an install smoke test in a clean environment, upgrade guides, and a dry-run release.

## 5. Capability matrix

Fit is how well the idea matches ARIA's rules. Effort is a rough size for a Python version using only the standard library.

| # | Capability | ARIA today | Fit | Effort | Main risk | Recommendation |
|---|---|---|---|---|---|---|
| 1 | Guarded tool pipeline (freeze, policy, human ask, deny-only guards, timeout, post-process, always record) | Executor runs agents, permission check is separate | High | About 300 lines | Low | Adopt first |
| 2 | Append-only session event log with crash repair | None (audit trail and memory only) | High | 400 to 600 lines | Medium, it touches everything | Adopt, second |
| 3 | Spill large output to disk with an honest preview | Blind slicing | High | About 120 lines | Low | Adopt, quick win |
| 4 | Bracketed context compaction | None | High value, medium effort | About 400 lines, needs #2 | Medium | Adopt after #2 |
| 5 | Skills with progressive disclosure | Agents are Python classes | High | Small, needs a front matter reader | Medium, skill text is untrusted | Adopt |
| 6 | Sub-agent contract (fail loud on missing capability, depth cap, frozen non-widening permissions, fork versus spawn) | Decomposition depth is capped, no total agent cap, no narrowing of child permissions found | High for the contract | Medium | Medium | Adopt |
| 7 | Workflow limits and Ralph style handoff (concurrency cap, total cap, null on child failure, validated handoff) | Static task graph | Medium to high | Small to medium | Low if model-written scripts are excluded | Adopt the limits |
| 8 | Goal and todo loop with compare-and-set and round caps | Learning, no goal loop | Medium | Small | Medium, completion is self-reported | Adopt, add an independent check |
| 9 | Failure taxonomy and honest sandbox reporting | Free-form errors | High | Small | Low | Adopt |
| 10 | Background jobs with bounded output ring, and schedule with missed-run collapsing | None | Medium to high | Medium | Medium | Later |
| 11 | Path containment (separator aware, resolved before use) | Prefix bug, see #36 | High, urgent | Small | Low | Do now as a bug fix |
| 12 | Environment scrubbing for child processes | None | Medium to high | Tiny | Low | Adopt, use an allowlist |
| 13 | Retry with backoff and structured failure codes for LLM calls | Single call | Medium | About 150 lines | Low | Adopt |
| 14 | Process practices (prompt snapshots, install smoke test, upgrade guides, dry-run release, final gate) | Partly done (CI Gate) | High | Small each | Low | Adopt gradually |
| 15 | Embed dsh as an outside agent through ACP | None | Medium | Medium, a hand-written ACP client | Medium | Optional, later |
| 16 | Kernel sandbox launcher (bwrap or Landlock) | Approval gates only, approved code runs in process | High value | Medium | High, see section 7 | Separate decision |
| 17 | MCP client | None | Medium | Medium | High, third-party servers and prompt text | Later, behind permissions |
| 18 | LSP tools | None | Low | Medium | Low | Skip for now |

## 6. What not to take, and why

- **Automatic review that replaces the human.** The `auto-review` package lets another model approve tool calls. That breaks ARIA's rule that side effects need a human.
- **Plan mode as protection.** In dsh, plan mode is only a prompt section. It does not stop writes. If ARIA copies the idea, read-only must be enforced by permissions, not by asking nicely.
- **Hooks that fail open.** A hook that crashes lets the action continue. Security checks must fail closed.
- **Default telemetry and log upload.** The session-log upload plugin defaults to on and points at a DeepSeek collector. ARIA must never ship a default like that.
- **Plaintext credential file.** dsh keeps credentials in a YAML file with strict file permissions and no encryption. Keep only the idea of storing references, not values.
- **Name-pattern scrubbing of environment variables.** Matching `KEY|PASSWORD|SECRET|TOKEN` misses names like `DATABASE_URL`. Use an allowlist.
- **Model-written code execution.** The programmatic tool runtime lets a model write a program that calls tools. Even its authors say the Python version is not a security boundary.
- **The UI, desktop app, typed RPC generator and Cordis plumbing.** They solve problems ARIA does not have.

## 7. Security view

- dsh's own safety notice says it is not audited and must not be relied on as the only protection.
- Its Linux sandbox restricts file writes only. Reads, network and process visibility are open, so a confined process can still read secrets and send them out. Anything ARIA builds on this idea must say so honestly and should add network isolation.
- Stderr text is used to detect sandbox denials, and a confined child can forge that text. Their own postmortem admits it.
- Approvals in dsh are one-shot only and never remembered. That is safer in one way and noisier in another. ARIA's grants are richer, so ARIA should keep them and add dsh's paired "asked" and "decided" audit entries.
- Skill files and MCP server instructions become part of the prompt. Both must be treated as untrusted input.
- Shipping a native binary would add supply chain risk and break the "no mandatory dependencies" rule. A Python launcher that calls `bwrap` is the cheaper route.

## 8. Licence and legal

- dsh is MIT, the same as ARIA. Ideas are free to use. If code is ever copied, the dsh copyright and licence text must travel with it.
- Its dependency list is mostly MIT and Apache-2.0. A few are different: a LibreOffice kit under MPL-2.0 with source obligations, an LGPL lint tool used only while developing, and Anthropic's own Claude Code programs, which carry Anthropic's terms.
- dsh bundles those Claude Code programs and states that its project owner authorises that. That permission covers dsh's own product. ARIA cannot inherit it, so ARIA must not copy or redistribute those parts.
- This is a technical reading of the notices file, not legal advice.

## 9. Findings about ARIA itself

- **#36, permission scope matching, priority 1.** A grant for `/home/u/project` also allows `/home/u/project-secrets/key.pem`. Confirmed by running ARIA's own code. This should be fixed before building anything new on the permission engine.
- The patcher bugs already tracked under epic #25 matter more now. The sandbox ideas and the tool pipeline would rely on a patcher and permission engine that behave correctly.
- `CODE_EXECUTE` is gated twice, by a permission check and by a user approval (`aria/core/improvement.py` around line 528). After that, the approved code runs inside the ARIA process with `exec` and a trimmed list of built-in functions. Python's own guidance is that restricting built-ins is not a security boundary, so an approved capability has the same power as ARIA itself. The two gates are real protection, and the missing part is containment after approval. That is a gap to decide on, not a bug.

## 10. Proposed roadmap

Nothing below starts until the maintainer picks scope. Every item follows the SDLC document: an issue first, acceptance criteria, design, tests, independent review, a green CI Gate.

**Phase 0, safe foundations**
- Fix #36 (scope matching).
- Fix the highest risk patcher items from epic #25 (symlink bypass, line number bound, backup collisions).

**Phase 1, execution discipline** (epic: "Guarded tool pipeline and session log")
- Story: every tool call runs through a pipeline that denies by default, records every stage, and turns any failure into a recorded error result. Acceptance: an unavailable approval denies, a guard can only deny, a crash mid-call leaves a repairable record, and every stage appears in the audit log.
- Story: an append-only session log in SQLite or JSONL with crash repair. Acceptance: a restart after a killed run closes open turns and marks unmatched tool calls as unknown outcome.
- Story: spill large output to disk (mode 0600, private folder) with a preview that says what was omitted. Acceptance: no silent truncation anywhere in the agents.
- Story: environment scrub by allowlist for child processes.

**Phase 2, context and delegation**
- Story: bracketed compaction on top of the session log.
- Story: skills with a catalog in the prompt and the full text loaded on demand, treated as untrusted.
- Story: sub-agent contract with depth cap, frozen non-widening permissions, and a loud error when a requested feature is missing.
- Story: workflow limits (concurrency cap, total cap, null on child failure) in the executor.
- Story: retry with backoff and structured failure codes for LLM plugins.

**Phase 3, longer running work**
- Goal and todo loop with round caps and an independent completion check.
- Background jobs with a bounded output ring.
- Scheduled reminders with missed-run collapsing.

**Phase 4, optional and separate decisions**
- ACP adapter to drive dsh as an outside coding agent, with per-call human approval.
- A Python launcher for `bwrap` or Landlock, with network isolation and honest reporting of what is and is not enforced.
- MCP client behind ARIA permissions.

**Process work that can run alongside:** prompt snapshot tests, an install smoke test in a clean environment, upgrade guides per breaking change, a dry-run release workflow.

## 11. Different points of view

- **Maintainer.** Phases 0 and 1 give the biggest gain for the least risk and keep the ARIA identity intact. Phases 2 and 3 are where ARIA starts to feel like a full agent platform. Phase 4 is where scope and risk grow quickly.
- **Security.** Build the pipeline and fix the permission engine before adding any power. Do not add model-written code execution or LLM approval, even though dsh has both.
- **Engineering effort.** Phases 0 to 2 are roughly a few thousand lines of Python plus tests, which is about the size of ARIA itself. That is a real investment and should be sliced into small pull requests.
- **Stability.** dsh may change or break at any time. Copying ideas protects ARIA from that. Depending on it directly does not.
- **Contrarian view.** Another option is to do nothing now and watch dsh mature. That is safe but leaves the context handling gaps and the permission bug in place. We recommend at least Phase 0.

## 12. Decisions needed from the maintainer

1. Which phases to start with. Our recommendation is Phase 0 and Phase 1.
2. Whether ARIA should ever drive dsh as an outside agent (Phase 4), or stay fully self-contained.
3. Whether a kernel sandbox launcher is in scope. It is the only way to make `CODE_EXECUTE` safer, but it carries the highest risk.
4. Whether to allow one branch per issue for this larger programme, as the SDLC document expects, instead of one shared branch.
5. Whether this report should be broken into an epic and story issues now.

## 13. Source map

Paths are inside the dsh clone at `/home/user/deepseek-ai/deepseek-harness`.

- Architecture overview: `docs/architecture.md`, `docs/tool-execution-pipeline.md`, `docs/defensive-patterns.md`, `docs/postmortem/`
- Safety notice and licence list: `SAFETY.md`, `THIRD_PARTY_NOTICES.md`
- Tool pipeline: `packages/core/tools/src/index.ts`
- Session log: `packages/core/session/src/` (log, surface, repair, fork)
- Approval: `packages/interaction/user-approval/src/index.ts`
- Sandbox: `packages/sandbox/`, `native/system/packages/entry/src/main.c`
- Spill: `packages/spill/spill-policy/src/index.ts`, `packages/spill/spill-local/src/store.ts`
- Compaction: `packages/compaction/`
- Skills: `packages/skill/skill-filesystem/`, `packages/skill/tool-skill/`
- Sub-agents and workflow: `packages/subagent/`, `packages/workflow/`
- Hooks: `packages/hooks/hook-protocol/src/`
- Python SDK: `python/sdk/src/deepseek_harness/`
- ACP: `packages/acp/acp/src/index.ts`
- Process: `docs/testing.md`, `.github/workflows/`

## 14. Outcome and corrections (updated 2026-10-08)

The maintainer asked the coordinator to decide what to take. This is what happened after this report was written.

### What was decided
Take the ideas, not the code. Start with the safe foundations, then execution discipline, as recommended in section 10. Cordis was also studied directly (its core is small and MIT) and its event and scope ideas were rebuilt in plain Python. Everything is tracked under the epic #37.

### What is built and merged
| Item | Issue | Pull request |
|---|---|---|
| Permission scope matching fix (the bug found while reading dsh) | #36 | #47 |
| Kernel events (emit, bail, serial, parallel, waterfall) | #38 | #49 |
| LLM retry with backoff, failure codes and secret redaction | #41 | #51 |
| Spill large output to disk with an honest preview | #40 | #50 |
| Correct overstated permission and audit claims, pipeline design | #54, #42 | #55 |

In review when this was written: kernel scopes (#39, pull request #61).

### Planned next
The guarded execution pipeline (#42) is designed in `docs/design/0042-guarded-pipeline.md` and split into stories #56 to #60 and #62 to #65. Later stories: session log (#43), compaction (#44), skills (#45), run limits (#46). Follow-ups found on the way: #66 (patcher rollback unguarded), #67 (web agent asks for scope `*`), #68 (save the audit trail), #69 (decide how approved synthesized agents are contained).

### Not taken, on purpose
Embedding dsh as an outside agent (ACP), a kernel sandbox launcher, an MCP client, LSP tools, LLM based approval, fail-open hooks, default telemetry, and model-written code execution. The sandbox question continues as the spike #69.

### Corrections to this report
- Section 4 listed a "hash-chained audit trail" as a strength of ARIA. Checking the code later showed that the hash skips `scope`, `reason`, `agent_id` and `task_id`, is cut to 64 bits, is never verified, and the log is memory only. It is an ordered log, not tamper evidence (issue #54, fixed in the docs, hardening is pipeline story #56).
- Section 9 said the permission engine is the place to build on. It is, but it only answers questions. Agent tool calls never asked it, which is what #42 fixes.
- Section 9 said `CODE_EXECUTE` is gated twice and "the missing part is containment after approval". That is still true and is now the spike #69.
