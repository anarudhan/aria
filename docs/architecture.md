# ARIA Architecture Deep Dive

> For developers who want to understand how the internals work before building on top of ARIA.

---

## The execution pipeline

When you call `aria.run(task, user_id)`, this is the exact sequence of events:

```
1. PERMISSION CHECK
   Extract required_capabilities from the call.
   Check each against the PermissionEngine for this user_id.
   Fail immediately with PermissionDeniedError if any are missing.
   No LLM call is made. No work is done.

2. MEMORY RECALL
   Search the four-ring memory system for context relevant to this task.
   Flash ring: most recent exchanges (volatile, fast).
   Hot ring: frequently accessed memories.
   Warm ring: session-level context.
   Cold ring: long-term learned patterns.

3. LESSON RECALL (v1.1+)
   Query the SelfImprovementEngine for lessons from similar past tasks.
   Inject relevant lessons into the decomposition prompt.
   "Last time we tried this, the routing to agent X failed because..."

4. TASK DECOMPOSITION
   Call the LLM with the task + context + lessons.
   Produce a TaskGenome — a DAG of TaskNode objects.
   Each node has: description, required_capabilities, dependencies, quality_threshold.
   Simple tasks produce a single-node genome. Complex tasks produce a multi-layer DAG.

5. COALITION FORMATION
   For each node in the genome, find the best agent.
   Match required_capabilities against registered agents using TF-IDF fingerprinting.
   Blend semantic score (0.6 weight) with empirical routing weight (0.4 weight).
   Form a coalition: {node_id → agent_id}.

6. PARALLEL EXECUTION
   Execute the DAG respecting dependency order.
   Nodes in the same layer run in parallel (asyncio.gather).
   Each agent receives: node, context (with user_id), memory reference.
   Agent returns a string result. Node is marked COMPLETED or FAILED.

7. RETROSPECTIVE
   Score the overall execution quality.
   Update the Bayesian trust ledger for each agent that participated.
   Agents that produced high-quality results get higher trust scores.
   Trust scores influence future coalition formation.

8. LEARNING (v1.1+)
   Call SelfImprovementEngine.learn_from_execution().
   Analyze outcome, identify root cause if failed.
   Record a Lesson with adjustment for next time.
   Update routing weights for all (agent, task_category) pairs.
   Check if failure pattern has hit the gap threshold (3 failures).
   If so, identify a CapabilityGap and propose a solution.

9. MEMORY WRITE (if MEMORY_WRITE permission granted)
   Store the task, result, and key context in the appropriate memory ring.
   Future executions can recall this via step 2.
```

---

## The TaskGenome

The TaskGenome is how ARIA thinks about complex work. Instead of running one big prompt, ARIA decomposes the task into a DAG of smaller, more focused subtasks. Each subtask is assigned to the agent best suited for it.

```python
@dataclass
class TaskNode:
    node_id: str                    # "n1", "n2", etc.
    description: str                # What this specific subtask does
    required_capabilities: list[str] # What kind of agent should handle it
    dependencies: list[str]         # node_ids that must complete first
    quality_threshold: float        # Minimum acceptable quality (0-1)
    status: TaskStatus              # PENDING, RUNNING, COMPLETED, FAILED
    result: Optional[Any]           # The agent's output
    error: Optional[str]            # Error message if failed

@dataclass
class TaskGenome:
    original_task: str
    nodes: dict[str, TaskNode]
    execution_layers: list[list[str]]  # Nodes grouped by parallel execution layer
```

A three-layer genome looks like this:

```
Layer 1 (parallel):   [n1: Research topic A]  [n2: Research topic B]
Layer 2 (parallel):   [n3: Synthesize findings]  (depends on n1, n2)
Layer 3:              [n4: Format final output]   (depends on n3)
```

---

## The four-ring memory system

```
┌─────────────────────────────────────────────────────┐
│  FLASH RING (500 entries, volatile)                 │
│  Current session context. Cleared on restart.       │
│  Speed: instant lookup                              │
├─────────────────────────────────────────────────────┤
│  HOT RING (2,000 entries)                           │
│  Frequently accessed, recently used context.        │
│  Eviction: LRU when capacity is reached.            │
├─────────────────────────────────────────────────────┤
│  WARM RING (10,000 entries)                         │
│  Session-level learning and context.                │
│  Entries decay in relevance over time.              │
├─────────────────────────────────────────────────────┤
│  COLD RING (20,000 entries)                         │
│  Long-term learned patterns and preferences.        │
│  Lowest access frequency. Compressed storage.       │
└─────────────────────────────────────────────────────┘
```

Memory entries bubble up (cold → warm → hot → flash) as they are accessed more frequently. Entries that are never accessed sink toward cold storage and eventually expire.

---

## The trust ledger

Every agent has a trust record that gets updated after every task it participates in.

```python
@dataclass
class TrustRecord:
    agent_id: str
    executions: int         # Total tasks participated in
    successes: int          # Tasks completed above quality threshold
    failures: int           # Tasks failed or below threshold
    avg_quality: float      # Running average quality score
    trust_score: float      # Bayesian-updated trust, 0.0 to 1.0
    last_updated: float
```

The trust score influences coalition formation. An agent with trust_score=0.9 on research tasks beats one with trust_score=0.5 even if their semantic similarity scores are equal. Trust is per-agent globally — it is not per-(agent, task_category). The routing weight system (in improvement.py) provides the per-task-category empirical layer.

---

## The permission engine internals

```
PermissionEngine
├── _grants: dict[str, list[PermissionGrant]]  # user_id → active grants
│   Each grant has: grant_id, user_id, capability, scope,
│                   expires_at, max_uses, uses_so_far
│
├── _audit: list[AuditEvent]                    # Append-only, hash-chained
│   Each event has: timestamp, user_id, capability, scope,
│                   action (granted/denied/revoked), hash_chain
│
└── check(user_id, capability, scope) → bool
    Iterates active grants for user_id.
    Checks: capability matches AND scope matches (or grant scope is "*")
    Checks: not expired (expires_at > now())
    Checks: not exhausted (max_uses is None or uses < max_uses)
    If all pass: increment uses_so_far, log audit event, return True.
    Otherwise: log denied event, return False.

### How scopes are matched

A scope names the resource a grant applies to. The rules live in
`aria/permissions/scope.py`, and every failure to parse means "not covered".

| Scope kind | Example | Rule |
|---|---|---|
| Everything | `*` | A grant of `*` covers all. A specific grant never covers a request for `*` |
| File path | `/home/u/project`, `~/notes` | `~`, `..` and symlinks are resolved first. A grant covers the folder and what is inside it, matched on whole path parts. `/a/b` covers `/a/b/c` and never `/a/bc` |
| URL | `https://example.com/api` | Scheme, host and port must be equal. The path matches on whole segments after `..` and `%2e%2e` are removed. A request with embedded credentials is not covered. URLs that clients and servers read differently are refused: backslashes, spaces and control characters, `;` in the path, empty path segments (`//`), and the encoded forms `%2f`, `%5c`, `%25` and `%00` in the path |
| Anything else | `vuln-12`, `cap-3` | Exact match only |

How grants keep one meaning:

- A path grant is stored as an absolute real path when it is created, so `~` and symlinks are settled once. Relative path grants (`.`, `./src`, `../x`) are refused, because they would mean something different whenever the working folder changes. A saved relative grant from an older version is kept but covers nothing.
- At check time a stored path must still resolve to itself. If a granted folder is later replaced by a symlink, the grant stops matching instead of following the link.
- Saved grants obey the same rules as new ones. A saved path that is not in canonical form (for example one with a trailing slash, a `~` or a symlink in it), a saved URL with a query, a fragment or dot segments, and a saved relative path all cover nothing. ARIA prints a warning at load for each, and the grant has to be made again.
- A path is checked at one moment and used at a later one. Code that writes to a path after a check should use `resolve_path()` and keep that gap short.

Behaviour change from earlier versions: scopes used to match on a text prefix. Bare relative paths such as `src/` and host-only names such as `example.com/api` were covered by a grant for their start. They now match only when equal. Use an absolute path or a full URL for a grant that covers children.
```

Each audit event carries a hash that includes the previous event's hash, so the events form an ordered chain. Be clear about what that does and does not give you today:

- The hash covers `entry_id`, `timestamp`, `user_id`, `capability`, `action` and the previous hash. It does **not** cover `scope`, `reason`, `agent_id` or `task_id`, so changing those is not detectable.
- It is cut to 16 hex characters (64 bits) of SHA256.
- **Nothing verifies the chain yet.** There is no function that walks it and reports a break.
- The audit list lives in memory only. It is not saved with the grants, so it is gone after a restart.
- Anyone who can change process memory can recompute the whole chain.

So the audit trail is an ordered log, not tamper evidence. A full hash over every field, a verify function, and saving to disk are planned as part of the guarded pipeline work (issue #42).

### What is and is not enforced today

The permission engine answers questions and denies by default. It does not stop code that never asks.

| Place | Asks the permission engine? |
|---|---|
| Orchestrator, for `required_capabilities` passed by the caller | Yes |
| Orchestrator, memory writes | Yes (`MEMORY_WRITE`) |
| Web agent, search and browse | Yes (`WEB_SEARCH`, `WEB_BROWSE`, with scope `*` only) |
| Improvement engine, spawning and loading a synthesized agent | Yes (`AGENT_SPAWN`, `CODE_EXECUTE`) |
| Patcher, applying a patch | Yes (`SELF_MODIFY`, with the vulnerability id as scope) |
| `LLMAgent` tool calls | **No** |
| `@aria_agent` and `BaseAgent.execute` bodies, other than the web agent | **No** |
| Patcher `rollback` and `rollback_all` | **No** |
| LLM provider calls (task data leaves the machine) | **No** |
| State files (memory, trust, permissions) | **No** |
| Running a synthesized agent after it was loaded | **No**, and it runs inside the ARIA process |

The guarded execution pipeline (issue #42, design in `docs/design/0042-guarded-pipeline.md`) is the planned fix for tool calls. Until it lands, code you write that calls tools or touches files must ask the engine itself.

---

## The security scanner internals

Two-pass analysis:

**Pass 1: Regex patterns**

Scans each line of each .py file against a dictionary of regex patterns. Each pattern maps to a severity, category, description, suggestion, and patch_safe flag. Fast. Catches well-known vulnerability patterns.

**Pass 2: AST structural analysis**

Parses each file into an AST. Walks the tree looking for structural issues that regex cannot catch:
- `str.format_map()` or `str.format()` called with user-controlled input (injection risk)
- `assert` statements used for security checks (disabled with python -O)
- Functions that accept `*args` or `**kwargs` without validation

The two passes run independently and their findings are merged into a single ScanReport.

---

## The self-improvement learning cycle

```
After every task execution:

learn_from_execution(task, result, coalition, genome)
    │
    ├── Classify outcome: success / failure / poor_quality / timeout
    │
    ├── Analyze root cause:
    │     If LLM available: send failure details to LLM, get structured analysis
    │     Heuristic fallback: pattern match on error messages
    │     Root causes: wrong_agent, decomposition_error, capability_gap,
    │                  contract_violation, tool_error, ambiguous_task
    │
    ├── Record Lesson:
    │     task_fingerprint (MD5 of normalized task text)
    │     task_sample (first 100 chars)
    │     outcome, root_cause, agents_used
    │     adjustment (what to do differently)
    │     tags (email, research, writing, etc.)
    │
    ├── Update RoutingWeight for each (agent, task_category) pair:
    │     Incremental running average of quality and success_rate
    │     Once executions >= 5, weight is "reliable"
    │
    └── Track failure patterns:
          key = f"{task_category}_{root_cause}"
          _failure_patterns[key] += 1
          If count >= 3: handle_capability_gap()
                            Identify gap, propose new agent
                            If LLM available: get specific proposal
                            Log gap to _gaps dict
```

---

## Adding a new agent

The minimum viable agent:

```python
from aria.agents.base import aria_agent

@aria_agent(
    capabilities=["list", "of", "capability", "strings"],
    domains=["domain_name"],
)
async def my_agent(node, context, memory):
    """What this agent does — used for semantic matching."""
    task = node.description
    user_id = context.get("user_id")
    # do the work
    return "result as string"
```

Register it:

```python
aria.register_agents([my_agent])
```

That is it. ARIA handles routing, quality scoring, trust tracking, and learning from this agent's performance automatically.

---

## LLM retry policy

`AnthropicPlugin` and `OpenAIPlugin` send every call through `call_with_retry` in `aria/plugins/retry.py`. Temporary failures (rate limit, server error, timeout, dropped connection, empty answer) are retried with exponential backoff, a cap and jitter. A wrong key or a bad request is never retried. A retry-after value from the provider is respected up to the cap, and cancellation is never swallowed. When the retries run out the caller gets an `LLMError` that carries the failure code, the attempt count and the original error. Error text kept in the records is redacted for common secret shapes and cut to 500 characters, on a best-effort basis; the original exception stays on `LLMError.cause` unredacted, so log `str(err)` and `err.code`, not the traceback. Every attempt is recorded and counted, so `retry_stats()` and `describe()` on a plugin can feed the health report and no failure is hidden. The module uses only the standard library. Full details are in `docs/llm-retry.md`.

---

## Security model

**What ARIA cannot do by default:**
- Access any resource (files, network, email, calendar)
- Store anything in long-term memory
- Spawn or register new agents
- Modify its own source code

**What grants are required for each category:**
- Reading user files: `FILE_READ`
- Writing user files: `FILE_WRITE`
- Any network call: `NETWORK`
- Email inbox access: `EMAIL_READ`
- Email sending (1-hour expiry default): `EMAIL_SEND`
- Teams channel reading: `TEAMS_READ`
- Teams posting (1-hour expiry default): `TEAMS_POST`
- Running generated code: `CODE_EXECUTE`
- Storing in long-term memory: `MEMORY_WRITE`
- Security self-patching (30-min expiry): `SELF_MODIFY`
- Registering new agents: `AGENT_SPAWN`

**Protected files (cannot be auto-patched):**
- `permissions/engine.py`
- `security/scanner.py`
- `security/patcher.py`

These files are excluded from the self-patcher's auto-patch list. Any changes to them require manual human review and application. This prevents a compromised patch from disabling the permission system or falsifying scan results.

---

## Spill store: large output is kept, never silently cut

Long tool and agent output is saved in full by `SpillStore` (`aria/core/spill.py`). The model sees the head and tail, a marker that says how many characters were left out, and an opaque id. The `read_spilled` tool returns more on request. Files are private (folder 0700, files 0600, created exclusively), ids cannot be turned into paths, and if saving fails the full text stays inline with a warning. Old files are removed after a retention period. Spill files from an earlier run are not readable after a restart, because ids are only valid in the process that issued them. Where text must be shortened and cannot be spilled (tool descriptions, web search snippets), the cut is marked with the number of characters omitted. Pages read back with `read_spilled` are never shortened again, locators are bound to the agent that spilled the item, the number and total size of spill files are bounded (oldest evicted first), and shortened previews start with an untrusted-data notice. Details are in `docs/spill.md`.
