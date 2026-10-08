# Design: guarded execution pipeline for tool calls (issue #42)

**Status:** proposed. Written by the architect agent from a read of the code, then checked by the coordinator. Nothing in this document is built yet. The stories that build it are sub-issues of #42.

**How to read this:** (U) marks a statement nobody verified by running code. The audit and enforcement facts in section 1 were re-checked against `origin/main` before this document was committed.

Basis: `origin/main` at 9f7023b, the spill branch `feat/40-spill-to-disk` at ce42cae, the DeepSeek Harness sources and the study in `docs/research/deepseek-harness-intake.md`. The architect read the code and ran nothing.

## 1. What the code does today

- `aria/agents/base.py` and `aria/core/executor.py` never call the permission engine. `LLMAgent._execute_tools` runs any callable: sync tools on the event loop, no timeout, results not in call order. engine.py:254 claims `check()` runs "before EVERY tool execution"; it does not.
- `AuditEntry.compute_hash` covers entry_id, timestamp, user_id, capability, action and prev_hash only. scope, reason, agent_id, task_id are not tamper-evident. There is no verify function and the audit list is memory only, though docs/architecture.md claims tamper evidence.
- `check()` spends a use on every allow; there is no non-consuming query.
- scope.py: a specific grant never covers a request for `*`. Tools must declare a scope. web_agent.py:454 and :480 request `*`, so a URL-scoped grant can never satisfy them.
- Spill (feat/40) binds locators to `agent_id` only, so users sharing an agent can read each other's spills.

### Gap table (side effects that skip the engine)

| Side effect | Where | Today |
|---|---|---|
| Any LLMAgent tool | `_execute_tools` | No check |
| `@aria_agent`, `BaseAgent.execute` bodies | agents | No gate; only WebAgent self-checks |
| `exec` of synthesized code | improvement.py:580 | Gated once at load, then runs in-process; trimmed builtins are no sandbox (U) |
| Patch write | patcher.py:275 | SELF_MODIFY, scope is vuln_id, not the path |
| `rollback`, `rollback_all` | patcher.py:327-362 | Overwrite source files; no check, no audit |
| Web fetch and search | web_agent.py | WEB_* with scope `*` only |
| LLM provider calls | llm_plugins.py | Task data leaves the machine; no NETWORK check |
| State files | many | No FILE_WRITE check; all share `memory_persist_path` (orchestrator.py:83-121), writers may clobber each other (U) |

No subprocess or os.system in aria/. Other `check`/`require` sites: orchestrator.py:200 (only if the caller passes required_capabilities), :380; improvement.py:409, :528; patcher.py:229.

## 2. Answers

### (a) Declaring capability; undeclared tools
New package `aria/tools/`: frozen `ToolSpec(name, needs: tuple[Need, ...], timeout_s=30.0, concurrency_safe=False, approval="auto"|"always", internal=False)` and `Need(capability, scope_arg: str | Callable[[Mapping], str] = "*")`. Declare with decorator `@tool(Capability.WEB_BROWSE, scope_arg="url")`, or `LLMAgent(tools=[...], tool_specs={name: spec})` for third-party callables; explicit wins. The pipeline copies specs into its own registry at registration, so later attribute edits or direct `_tools` writes change nothing. Risk comes from `RISK_LEVEL`.

Call sites: LLMAgent is constructed 0 times in aria/, examples/, tests/ (docstring only, base.py:26); `tools=` 0; `add_tool` 0; tests of `_execute_tools` 0. The 8 `@aria_agent` uses are agent level and bypass it.

Recommend: undeclared means DENY, with `ToolPipeline(undeclared="deny"|"allow_logged")`. `allow_logged` is OFF by default; it still freezes, times out and audits (`tool.undeclared_allowed`) and warns once per tool. Reason: the engine already promises enforcement, so this is a bug fix; in-repo migration cost is zero. Rejected: ON with a deprecation warning (hole open by default).

### (b) Hook points
- `_execute_tools(tool_calls)` keeps its name and return shape (easy rebase over #40) and calls `self._pipeline.run_batch(calls, current_run.get())`. The pipeline wraps everything from the model's tool_call dict to the string put in messages. `register_agent` attaches it beside `set_llm` (orchestrator.py:150). Tools but no pipeline: deny all.
- Identity must not travel in `context` (it is printed into the prompt). In `_run_node_once` the executor sets a `ContextVar` `current_run = ToolRunContext(user_id, genome_id, node_id, agent_id, deadline)` around `wait_for(agent.execute...)` and resets it in `finally`. The executor holds no pipeline. No run context: deny.

### (c) Human approval in a library
`ApprovalProvider` protocol: `async request(req) -> ApprovalOutcome`. `req` carries call_id, user, agent, tool, capability, scope, risk, redacted args preview and digest. Outcomes: `ALLOWED_ONCE`, `REJECTED`, `CANCELLED`, `UNAVAILABLE`. Only `is ALLOWED_ONCE` proceeds. Exception, timeout (120 s), garbage value or no provider mean `UNAVAILABLE`, which denies. Default: `NoApprovals`. Ship `CallbackApprovalProvider(async_fn)` and `QueueApprovalProvider` (a UI task reads `.requests` and calls `.resolve(id, outcome)`). Approval is one-shot, bound to the args digest. No "always allow"; durable permission is only an engine grant.

Pairing: approval is an extra "and", never a replacement. No grant means deny, as today. Approval is needed when the spec says `always`, a hook votes ask, or risk is high or critical with a standing grant. A grant with `max_uses=1` is already a per-call human act, which is what `confirm_self_modify` creates (scope=patch_id, 15 minutes, one use), so that flow works unchanged and is not asked twice. Use counts: a new non-consuming `engine.peek()` decides early; one consuming `check()` runs after the guards, right before execution, so a guard denial never burns a single-use grant. Rejected: approval minting a grant (the pipeline would create permissions).

### (d) Concurrency, timeout, cancellation
- Default exclusive. Overlap only when `concurrency_safe=True`; exclusive calls are barriers (as in dsh). Decision stages and approvals run one call at a time; only the run stage overlaps. Results return in call order. The pipeline mints its own call_id.
- Timeout = min(spec.timeout_s, run deadline left, 300). Python 3.10 lacks `asyncio.timeout`: use `wait_for`, catch both timeout classes. Async tools get cancel plus 2 s grace. Sync tools run in a worker thread (needed for a timeout to mean anything) and cannot be killed: outcome `timeout_unfinished`, "effects unknown".
- Always recorded: `tool.received` is written first and `finally` writes exactly one `tool.end` (ok, error, denied, timeout, cancelled), synchronously, so cancellation cannot interrupt it. `CancelledError` and other `BaseException`s are recorded and re-raised; siblings are cancelled and awaited. `Exception` becomes an error result. A begin with no end after a crash means "unknown outcome".

### (e) Events and deny-wins
One private `Events` per pipeline:
- `tool.policy`: `serial(view, vote)`; hooks add a denial or ask.
- `tool.guard`: `serial(view)`; return a deny reason or None.
- `tool.around`: `awaterfall`, inner = the run; metrics only, cannot alter args or decision.
- `tool.result`: `awaterfall`, inner = spill; extra transforms.
- `tool.audit`: `emit`; observers only.

Mechanism: `vote` is a `PolicyVote` with `deny(reason)` and `ask(reason)` and no allow method. `view` is a read-only frozen snapshot without the engine verdict. Hooks must return None. `serial` bails on the first non-None, so any return (True, "allow", a dict) stops the chain and becomes a denial "hook protocol violation". A raise, a hang (stage timeout) or a TypeError from touching the frozen view also denies. Only then does the pipeline compute the engine verdict and call a pure `fold(engine, vote)` with DENY over ASK over ALLOW; the consuming engine check after the guards must still allow. Waterfall is rejected for policy: a layer that skips `next` ends the chain and hides later denials. Durable audit is written directly, not via `tool.audit`, so removing a listener cannot remove a record. Limit: events.py is not a sandbox; this stops mistakes, not malicious in-process hooks.

### (f) Audit format
Add `call_id`, `stage`, `detail` (redacted JSON) and `args_digest` to `AuditEntry`. Actions: `tool.received`, `tool.policy`, `tool.approval.asked`, `tool.approval.decided`, `tool.guard`, `tool.commit`, `tool.executed`, `tool.post`, `tool.end`. A new public `engine.record_audit(...)` appends to the same chain as grants and checks. `compute_hash` must cover every field and use the full sha256 (now cut to 64 bits); add `verify_chain()`. Arguments: key-name redaction plus `sanitize_error_text` on string leaves (200 characters) and an HMAC digest with a per-process key, so weak secrets cannot be guessed from it. Results: length, digest and a locator hash only. Move `sanitize_error_text` to `aria/security/redact.py`, re-exported from retry.py. Redaction is best effort. The chain is unanchored: whoever can rewrite process memory can recompute it.

### (g) Spill, retry, read_spilled
- Post-process = `SpillStore.preview_or_spill(text, max_chars, owner)`, moved out of `_tool_text` after #40 merges. If post-processing fails, withhold the raw result and return an error.
- Retry: the LLM retry policy wraps only `llm.chat`. The pipeline never retries a tool; a model re-issue is a new call through every stage. Node retry (`_run_node_with_retry`) can repeat side effects on a backup agent; a later guard can deny the same (tool, digest) after an ok or unknown outcome.
- `read_spilled` declares no capability: `internal=True`, settable only via a module-private token (not enforceable in-process). It still gets freeze, timeout and audit; its authority is the owner-bound locator. The no-re-spill exemption is keyed on the spec, not the name. Rejected: a SPILL_READ capability (grants for data ARIA withheld itself) or an auto-grant (self-granting).

### (h) Tests
- Exhaustive truth table: engine {allow, deny, error} x hook {none, deny, ask, raise, hang, returns True, mutates view} x approval {4 outcomes, garbage, raises}. The body runs iff engine allows, no denial, and (no ask or ALLOWED_ONCE).
- Ordering: audit actions in exact stage order; no body before `tool.commit`; guard denial leaves a `max_uses=1` grant unspent.
- Freeze: mutate the source dict after approval; the tool still sees the approved digest.
- Crash: `password=` in error text redacted; non-str result; hang; blocking sync tool; SystemExit recorded and propagated; cancel during run and approval; audit write failure means no run. After each test: begins equal ends and `verify_chain()` passes.
- Executor: node timeout cancels a running tool and its end record exists; concurrent nodes keep separate `current_run`.

## 3. Pull requests (dependency order)

| # | PR | Acceptance |
|---|---|---|
| 1 | engine.py audit hardening: full hash, `verify_chain`, `record_audit`, new fields | Edited scope or reason detected; old audit test passes |
| 2 | `engine.peek()` returning `PermissionDecision` | Never changes use_count; matches `check` in a table test |
| 3 | `aria/security/redact.py`: move function, add `redact_args` | test_retry unchanged; nested and key redaction tested |
| 4 | `aria/tools/`: ToolSpec, `@tool`, registry, freeze, digest | Non-JSON, deep or huge args rejected; frozen copy immutable |
| 5 | Decision stages: Events, `PolicyVote`, `fold`, approval providers, guards | Truth-table and ordering tests; no execution yet |
| 6 | Run stage and `run_batch`: timeout, cancel, errors, concurrency, records | Crash and cancel tests; begins equal ends |
| 7 | Wire LLMAgent, executor ContextVar, orchestrator | Undeclared tool denied; declared and granted tool behaves as before |
| 8 | Spill stage and internal `read_spilled` (after #40) | No double spill; same-named user tool not exempt |
| 9 | `allow_logged` mode, docs/tools.md, demo, release note | Off by default; one warning per tool |
| 10 | Follow-up issues: patcher rollback gate, URL scope in web_agent, audit JSONL, synthesized-agent containment | Filed, outside #42 |

## 4. Risks
- PRs 1 and 2 touch security-sensitive engine code: security review needed.
- Sync tools in threads change thread affinity (U).
- Audit memory grows (about nine entries per call).
- Conflicts with #40 in `_execute_tools`: land it first.
- Approval fatigue; an undrained queue (times out to deny).
- Non-LLMAgent agents stay unguarded until the follow-ups.

## 5. Decisions taken

The maintainer asked the coordinator to decide. These are the decisions for version 1. Each can be changed by a comment on #42 before the story that depends on it starts.

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | Undeclared tools | **Deny.** An opt-in `allow_logged` compatibility mode exists, off by default, and it warns once per tool | The engine already promises enforcement, so this is a bug fix. No tool in this repository, its examples or its tests is affected, because `LLMAgent` is not constructed anywhere in them |
| 2 | When a human must approve | **High risk and above**, plus any tool whose spec says `approval="always"`, plus any policy hook that votes "ask" | Matches the risk levels the engine already defines |
| 3 | May an approval create a grant when none exists | **No** in version 1 | The pipeline must not create permissions. Only the user grants |
| 4 | Sync tools in worker threads | **Yes** | A timeout means nothing for a blocking function on the event loop. Documented limit: a thread cannot be killed, so a timed out sync tool is recorded as "outcome unknown" |
| 5 | LLM provider calls and internal state files | **Out of scope for version 1**, tracked as follow-up stories | They need their own capability model (what is the scope of "send task data to a provider"?) |
| 6 | Spill locator owner | Becomes `user_id:agent_id` in the spill wiring story | Two users sharing one agent id must not read each other's spills |
| 7 | Audit hash change | **Accepted.** Full SHA256 over every field, plus a verify function | The chain exists only in memory today, so nothing stored is invalidated |

## 6. Honest limits

- Events in `aria/kernel/events.py` give clean extension points. They are not a sandbox. The deny-wins design stops mistakes and careless hooks, not malicious code running inside the same process.
- Even with the pipeline, agents that are not `LLMAgent` and call functions directly stay unguarded until the follow-up stories land. The gap table in section 1 is the list.
- The audit chain stays unanchored: whoever can rewrite process memory can recompute it. Saving it to disk and anchoring it outside the process is a follow-up.
