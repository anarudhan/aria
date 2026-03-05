"""
ARIA Self-Improvement Engine
==============================
This is the part that makes ARIA genuinely grow over time.

Not just security patching. Not just remembering facts.
Actual learning from experience — mistakes, successes, edge cases,
capability gaps, routing errors, and quality patterns.

Three layers of learning:

1. MISTAKE LEARNING
   Every task failure and every low-quality result is analyzed.
   ARIA identifies what went wrong — wrong agent routed, bad decomposition,
   missing capability, contract violation — and records a lesson.
   Next time a similar task arrives, the lesson shapes the approach.

2. CAPABILITY EXPANSION
   When ARIA repeatedly fails at a class of tasks because no agent
   can handle them, it identifies the capability gap and generates
   a new tool/agent specification. In safe mode it proposes this to the user.
   In autonomous mode (if granted) it synthesizes a new Python tool function
   and registers it as a new agent — from scratch — using its LLM.

3. ROUTING REFINEMENT
   After enough executions, ARIA builds a routing model — which agent
   actually performed well on which type of task. This supplements the
   semantic fingerprinting with empirical evidence. An agent that said
   it could do research but consistently underperforms on research tasks
   gets a lower routing priority for those tasks over time.

The lessons, capability gaps, and routing weights are stored in the
TemporalMemory system and optionally persisted to disk.

Nothing in this module modifies code without going through the patcher's
safety protocol (backup → apply → verify → rollback on failure).
User permission is still required for any code modification.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ─── Lesson ──────────────────────────────────────────────────────────────────

@dataclass
class Lesson:
    """
    A single lesson ARIA learned from one execution.

    Lessons are the atomic unit of experience. Each one captures:
      - What was tried (task_fingerprint — hashed pattern of the task)
      - What happened (outcome — success, failure, partial, poor quality)
      - Why it happened (root_cause — ARIA's analysis of what went wrong)
      - What to do differently (adjustment — concrete change for next time)
      - Whether the lesson has been applied (applied_count)
    """
    lesson_id:       str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    task_fingerprint: str = ""     # Semantic hash of the task type
    task_sample:     str = ""      # First 100 chars of the task (for display)
    outcome:         str = ""      # "success", "failure", "poor_quality", "timeout", "permission_denied"
    root_cause:      str = ""      # "wrong_agent", "decomposition_error", "capability_gap", "contract_violation", "tool_error"
    agents_used:     list[str] = field(default_factory=list)
    quality_score:   float = 0.0
    duration_s:      float = 0.0
    adjustment:      str = ""      # What to do differently next time
    confidence:      float = 0.5   # How confident ARIA is in this lesson
    learned_at:      float = field(default_factory=time.time)
    applied_count:   int = 0
    tags:            list[str] = field(default_factory=list)

    def to_prompt_fragment(self) -> str:
        """Formatted for injection into decomposer/supervisor prompts."""
        return (
            f"Past lesson [{self.outcome}]: {self.task_sample[:80]}\n"
            f"  Root cause: {self.root_cause}\n"
            f"  What to do differently: {self.adjustment}"
        )

    def matches(self, task: str, threshold: float = 0.3) -> bool:
        """Check if this lesson is relevant to a new task using simple overlap."""
        task_words = set(task.lower().split())
        sample_words = set(self.task_sample.lower().split())
        if not task_words or not sample_words:
            return False
        overlap = len(task_words & sample_words) / len(task_words | sample_words)
        return overlap >= threshold


# ─── Capability Gap ───────────────────────────────────────────────────────────

@dataclass
class CapabilityGap:
    """
    A class of tasks that ARIA repeatedly cannot handle well.

    When the same type of failure occurs 3+ times and no existing agent
    can fill the gap, ARIA identifies a CapabilityGap and either:
      - Proposes a new agent spec to the user (safe mode)
      - Synthesizes a new tool and registers it (autonomous mode, with permission)
    """
    gap_id:          str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    task_type:       str = ""      # Description of the task type ARIA can't handle
    failure_count:   int = 0
    example_tasks:   list[str] = field(default_factory=list)
    proposed_cap:    str = ""      # Capability name that would fill this gap
    proposed_desc:   str = ""      # Description for the new agent
    proposed_tools:  list[str] = field(default_factory=list)
    status:          str = "open"  # "open", "proposed", "synthesized", "resolved"
    identified_at:   float = field(default_factory=time.time)
    resolved_at:     Optional[float] = None

    @property
    def is_critical(self) -> bool:
        return self.failure_count >= 5


# ─── Routing Weight ───────────────────────────────────────────────────────────

@dataclass
class RoutingWeight:
    """
    Empirical routing performance for a specific (agent, task_category) pair.

    This goes beyond semantic fingerprinting — it tracks ACTUAL outcomes.
    An agent that claims to do 'data analysis' but consistently scores
    below 0.5 quality on data analysis tasks will get a lower empirical weight.
    The routing system blends semantic similarity (0.6 weight) with empirical
    performance (0.4 weight) once enough data is available.
    """
    agent_id:       str = ""
    task_category:  str = ""
    executions:     int = 0
    avg_quality:    float = 0.5
    avg_duration_s: float = 0.0
    success_rate:   float = 0.5
    last_updated:   float = field(default_factory=time.time)

    @property
    def empirical_score(self) -> float:
        """Combined empirical performance score, 0.0 to 1.0."""
        if self.executions < 3:
            return 0.5   # Not enough data yet — neutral
        return (self.avg_quality * 0.6) + (self.success_rate * 0.4)

    @property
    def is_reliable(self) -> bool:
        return self.executions >= 5


# ─── Synthesized Capability ───────────────────────────────────────────────────

@dataclass
class SynthesizedCapability:
    """
    A new tool or agent that ARIA generated to fill a capability gap.

    The generated code is stored here and can be loaded dynamically.
    It is always shown to the user before being executed.
    """
    cap_id:          str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    name:            str = ""
    description:     str = ""
    source_code:     str = ""
    generated_at:    float = field(default_factory=time.time)
    generated_for:   str = ""   # gap_id it resolves
    user_reviewed:   bool = False
    user_approved:   bool = False
    loaded:          bool = False
    test_result:     str = ""   # "passed", "failed", "not_tested"
    capabilities:    list[str] = field(default_factory=list)
    domains:         list[str] = field(default_factory=list)


# ─── Improvement Engine ────────────────────────────────────────────────────────

class SelfImprovementEngine:
    """
    ARIA's learning and growth system.

    This runs in the background of every task execution, continuously:
      - Analyzing outcomes for lessons
      - Tracking which agents work well for which task types
      - Identifying capability gaps from repeated failures
      - Optionally synthesizing new tools to fill those gaps

    The engine never modifies agent behavior directly. It accumulates
    knowledge that influences future routing, decomposition, and planning.
    It also generates new capability proposals for the user to review.

    Learning Cycle (runs after every task):
        execution_result → analyze_outcome() → record_lesson()
                        → update_routing_weights()
                        → check_capability_gaps()
                        → (if gap critical) → propose_new_capability()
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        llm_plugin: Optional[object] = None,
        autonomous_synthesis: bool = False,   # Must be explicitly enabled
    ):
        self._lessons: list[Lesson] = []
        self._gaps: dict[str, CapabilityGap] = {}
        self._routing_weights: dict[str, RoutingWeight] = {}
        self._synthesized: list[SynthesizedCapability] = []
        self._llm = llm_plugin
        self._autonomous = autonomous_synthesis
        self._persist_path = Path(persist_path) / "improvement_data.json" if persist_path else None

        # Pattern frequency tracker for gap detection
        self._failure_patterns: dict[str, int] = {}

        if self._persist_path and self._persist_path.exists():
            self._load()

        print(f"[Learning] 🎓 Self-Improvement Engine ready "
              f"(autonomous_synthesis={'ON' if autonomous_synthesis else 'OFF'})")

    # ── Main Learning Hook (called after every task) ───────────────────────────

    async def learn_from_execution(
        self,
        task:          str,
        result,        # TaskResult
        coalition:     dict[str, str],
        genome,        # TaskGenome
        llm_plugin:    Optional[object] = None,
    ) -> list[Lesson]:
        """
        The primary learning function. Called after every task finishes.

        Analyzes the execution result, generates lessons, updates routing
        weights, and checks for capability gaps. Returns new lessons learned.
        """
        llm = llm_plugin or self._llm
        lessons = []

        from .models import TaskStatus

        # 1. Determine outcome category
        if result.status == TaskStatus.COMPLETED and result.quality_score >= 0.7:
            outcome = "success"
        elif result.status == TaskStatus.COMPLETED and result.quality_score < 0.5:
            outcome = "poor_quality"
        elif result.status == TaskStatus.FAILED:
            outcome = "failure"
        elif result.status == TaskStatus.TIMEOUT:
            outcome = "timeout"
        else:
            outcome = "partial"

        # 2. Analyze root cause (LLM if available, heuristic otherwise)
        root_cause, adjustment = await self._analyze_root_cause(
            task, result, genome, coalition, outcome, llm
        )

        # 3. Create and store lesson
        agents_used = list(set(coalition.values()))
        lesson = Lesson(
            task_fingerprint = self._fingerprint(task),
            task_sample      = task[:100],
            outcome          = outcome,
            root_cause       = root_cause,
            agents_used      = agents_used,
            quality_score    = result.quality_score,
            duration_s       = result.total_seconds,
            adjustment       = adjustment,
            confidence       = 0.7 if llm else 0.4,
            tags             = self._extract_tags(task),
        )
        self._lessons.append(lesson)
        lessons.append(lesson)

        # 4. Update routing weights
        task_category = self._categorize_task(task)
        for agent_id in agents_used:
            self._update_routing_weight(
                agent_id      = agent_id,
                task_category = task_category,
                quality       = result.quality_score,
                duration_s    = result.total_seconds,
                success       = result.status == TaskStatus.COMPLETED,
            )

        # 5. Track failure patterns for gap detection
        if outcome in ("failure", "poor_quality"):
            pattern_key = f"{task_category}_{root_cause}"
            self._failure_patterns[pattern_key] = self._failure_patterns.get(pattern_key, 0) + 1

            if self._failure_patterns[pattern_key] >= 3:
                await self._handle_capability_gap(task, task_category, root_cause, outcome, llm)

        # 6. Log what was learned
        if outcome != "success":
            print(
                f"[Learning] 📖 Lesson recorded: {outcome} | "
                f"Cause: {root_cause} | "
                f"Adjustment: {adjustment[:60]}..."
            )

        self._save()
        return lessons

    # ── Lesson Retrieval for Future Tasks ─────────────────────────────────────

    def recall_lessons(self, task: str, limit: int = 5) -> list[Lesson]:
        """
        Find lessons relevant to an incoming task.
        Called by the orchestrator before decomposition — lessons shape planning.
        """
        relevant = [ls for ls in self._lessons if ls.matches(task)]
        # Sort: failures first (more important to avoid), then by recency
        relevant.sort(key=lambda ls: (ls.outcome != "failure", -ls.learned_at))
        return relevant[:limit]

    def format_lessons_for_prompt(self, task: str) -> str:
        """Return lessons as a formatted string for injection into LLM prompts."""
        lessons = self.recall_lessons(task)
        if not lessons:
            return ""
        lines = ["Past experience with similar tasks:"]
        for ls in lessons:
            icon = "❌" if ls.outcome == "failure" else "⚠️" if ls.outcome == "poor_quality" else "✅"
            lines.append(f"{icon} {ls.task_sample[:60]}: {ls.adjustment[:100]}")
        return "\n".join(lines)

    # ── Routing Intelligence ───────────────────────────────────────────────────

    def get_empirical_score(self, agent_id: str, task: str) -> float:
        """
        Get the empirical performance score for an agent on a task type.
        Returns 0.5 if not enough data yet (neutral — don't penalize new agents).
        """
        category = self._categorize_task(task)
        key = f"{agent_id}|{category}"
        weight = self._routing_weights.get(key)
        if not weight or not weight.is_reliable:
            return 0.5
        return weight.empirical_score

    def blend_routing_score(
        self,
        agent_id:       str,
        task:           str,
        semantic_score: float,
        semantic_weight: float = 0.6,
    ) -> float:
        """
        Blend semantic fingerprint similarity with empirical performance.
        Once an agent has 5+ executions on a task type, empirical data adds weight.
        Returns a blended score between 0.0 and 1.0.
        """
        empirical = self.get_empirical_score(agent_id, task)
        category  = self._categorize_task(task)
        key       = f"{agent_id}|{category}"
        weight    = self._routing_weights.get(key)

        if not weight or not weight.is_reliable:
            # Not enough empirical data — lean on semantics
            return semantic_score

        empirical_weight = 1.0 - semantic_weight
        return (semantic_score * semantic_weight) + (empirical * empirical_weight)

    # ── Capability Gap Handling ────────────────────────────────────────────────

    def get_capability_gaps(self) -> list[CapabilityGap]:
        return list(self._gaps.values())

    def get_open_gaps(self) -> list[CapabilityGap]:
        return [g for g in self._gaps.values() if g.status == "open"]

    def get_synthesized_capabilities(self, approved_only: bool = False) -> list[SynthesizedCapability]:
        if approved_only:
            return [s for s in self._synthesized if s.user_approved]
        return self._synthesized

    async def synthesize_capability(
        self,
        gap_id:   str,
        user_id:  str,
        permission_engine: object,
        llm_plugin: Optional[object] = None,
    ) -> Optional[SynthesizedCapability]:
        """
        Generate a new agent/tool to fill a capability gap.

        This uses the LLM to write a new Python function decorated with @aria_agent.
        The code is shown to the user for review before being loaded.
        It is never executed without explicit user approval (CODE_EXECUTE permission).
        """
        from ..permissions.engine import Capability as Cap
        try:
            permission_engine.require(user_id, Cap.AGENT_SPAWN, scope=gap_id)
        except Exception as e:
            print(f"[Learning] ❌ Cannot synthesize capability: {e}")
            return None

        # Search by gap_id since dict is keyed by pattern
        gap = next((g for g in self._gaps.values() if g.gap_id == gap_id), None)
        if not gap:
            print(f"[Learning] Gap {gap_id} not found")
            return None

        llm = llm_plugin or self._llm
        if not llm:
            print("[Learning] No LLM available for capability synthesis")
            return None

        print(f"\n[Learning] 🧬 Synthesizing new capability for gap: {gap.task_type}")

        prompt = textwrap.dedent(f"""
            You are generating a new Python agent tool for ARIA (an AI orchestration framework).

            CAPABILITY GAP TO FILL:
            Task type: {gap.task_type}
            Example failures: {chr(10).join(gap.example_tasks[:3])}
            Proposed capability: {gap.proposed_cap}
            Description: {gap.proposed_desc}

            REQUIREMENTS:
            1. Write a single Python async function decorated with @aria_agent
            2. Use only the Python standard library (no external dependencies)
            3. The function signature must be: async def agent_name(node, context, memory) -> str
            4. The function must return a string result
            5. Include proper error handling
            6. Add a descriptive docstring
            7. The @aria_agent decorator needs: capabilities=["..."], domains=["..."]

            ARIA AGENT TEMPLATE:
            ```python
            from ..agents.base import aria_agent

            @aria_agent(
                capabilities=["capability1", "capability2"],
                domains=["domain1"],
                max_load=3,
            )
            async def my_agent(node, context, memory):
                \"\"\"Clear description of what this agent does.\"\"\"
                task = node.description
                # implementation here
                return result
            ```

            Generate ONLY the Python code, no explanation, no markdown fences.
            The code must be syntactically valid Python 3.10+.
        """)

        try:
            generated = await llm.complete(prompt, max_tokens=800)
            # Strip markdown if LLM added it
            code = re.sub(r'```python\s*', '', generated)
            code = re.sub(r'```\s*', '', code).strip()

            # Syntax check before anything else
            ast.parse(code)

            synth = SynthesizedCapability(
                name          = gap.proposed_cap.replace(" ", "_").lower(),
                description   = gap.proposed_desc,
                source_code   = code,
                generated_for = gap_id,
                capabilities  = [gap.proposed_cap],
                domains       = [gap.task_type],
            )
            self._synthesized.append(synth)
            gap.status = "synthesized"

            print(f"\n[Learning] ✅ New capability synthesized: '{synth.name}'")
            print(f"[Learning] 📋 Code preview:\n{'─'*50}")
            print(code[:400] + ("..." if len(code) > 400 else ""))
            print("─" * 50)
            print(f"[Learning] ⚠️  Review and approve with: aria.approve_capability('{synth.cap_id}')")

            self._save()
            return synth

        except SyntaxError as e:
            print(f"[Learning] ❌ Generated code has syntax error: {e}")
            return None
        except Exception as e:
            print(f"[Learning] ❌ Synthesis failed: {e}")
            return None

    def approve_capability(self, cap_id: str, user_id: str) -> Optional[SynthesizedCapability]:
        """User explicitly approves a synthesized capability for loading."""
        cap = next((s for s in self._synthesized if s.cap_id == cap_id), None)
        if not cap:
            print(f"[Learning] Capability {cap_id} not found")
            return None
        cap.user_reviewed = True
        cap.user_approved = True
        print(f"[Learning] ✅ Capability '{cap.name}' approved by {user_id}")
        self._save()
        return cap

    async def load_approved_capability(
        self,
        cap_id:         str,
        orchestrator:   object,
        permission_engine: object,
        user_id:        str,
    ) -> bool:
        """
        Load an approved synthesized capability into the running ARIA instance.

        Executes the generated code in a restricted namespace to extract the
        agent object, then registers it with the orchestrator.
        """
        from ..permissions.engine import Capability as Cap
        try:
            permission_engine.require(user_id, Cap.CODE_EXECUTE, scope=cap_id)
        except Exception as e:
            print(f"[Learning] ❌ Cannot load capability: {e}")
            return False

        cap = next((s for s in self._synthesized if s.cap_id == cap_id), None)
        if not cap or not cap.user_approved:
            print(f"[Learning] Capability {cap_id} not found or not approved")
            return False

        print(f"[Learning] 🔌 Loading capability '{cap.name}'...")

        try:
            namespace = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "bool": bool,
                    "range": range,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "sorted": sorted,
                    "min": min,
                    "max": max,
                    "sum": sum,
                    "abs": abs,
                    "isinstance": isinstance,
                    "hasattr": hasattr,
                    "getattr": getattr,
                    "type": type,
                    "Exception": Exception,
                    "ValueError": ValueError,
                    "RuntimeError": RuntimeError,
                }
            }

            # Import aria_agent into namespace for the decorator to work
            from ..agents.base import aria_agent
            namespace["aria_agent"] = aria_agent

            # Strip import statements - aria_agent is already injected into namespace
            exec_code = '\n'.join(
                line for line in cap.source_code.splitlines()
                if not line.strip().startswith(('import ', 'from '))
            )
            exec(exec_code, namespace)

            # Find the decorated function in the namespace
            loaded_agent = None
            for name, obj in namespace.items():
                if hasattr(obj, '_is_aria_agent') and obj._is_aria_agent:
                    loaded_agent = obj
                    break

            if not loaded_agent:
                print("[Learning] ❌ No @aria_agent found in generated code")
                return False

            orchestrator.register_agent(loaded_agent)
            cap.loaded = True
            cap.test_result = "loaded"

            if gap_id := cap.generated_for:
                if gap_id in self._gaps:
                    self._gaps[gap_id].status = "resolved"
                    self._gaps[gap_id].resolved_at = time.time()

            print(f"[Learning] ✅ Capability '{cap.name}' loaded and registered!")
            self._save()
            return True

        except Exception as e:
            cap.test_result = f"failed: {e}"
            print(f"[Learning] ❌ Failed to load capability: {e}")
            return False

    # ── Improvement Summary ────────────────────────────────────────────────────

    def print_learning_report(self):
        """Print a human-readable learning summary."""
        print("\n" + "━" * 60)
        print("  ARIA Self-Improvement Report")
        print("━" * 60)

        print(f"\n  Lessons Learned: {len(self._lessons)}")
        outcomes = {}
        for ls in self._lessons:
            outcomes[ls.outcome] = outcomes.get(ls.outcome, 0) + 1
        for outcome, count in sorted(outcomes.items()):
            icon = {"success": "✅", "failure": "❌", "poor_quality": "⚠️",
                    "timeout": "⏱️", "partial": "◑"}.get(outcome, "?")
            print(f"    {icon} {outcome:<15} {count}")

        if self._lessons:
            recent = sorted(self._lessons, key=lambda ls: -ls.learned_at)[:3]
            print("\n  Most Recent Lessons:")
            for ls in recent:
                print(f"    • [{ls.outcome}] {ls.task_sample[:50]}")
                print(f"      → {ls.adjustment[:70]}")

        print(f"\n  Routing Weights: {len(self._routing_weights)} agent-category pairs tracked")
        if self._routing_weights:
            best = sorted(self._routing_weights.values(), key=lambda w: -w.empirical_score)
            print("  Top performers:")
            for w in best[:3]:
                if w.is_reliable:
                    print(f"    • {w.agent_id:<20} on {w.task_category:<15} "
                          f"score={w.empirical_score:.2f} n={w.executions}")

        gaps = self.get_open_gaps()
        print(f"\n  Capability Gaps: {len(gaps)} open")
        for g in gaps:
            critical = " 🔴 CRITICAL" if g.is_critical else ""
            print(f"    • {g.task_type}{critical} (failed {g.failure_count}x)")
            if g.proposed_cap:
                print(f"      Proposed solution: {g.proposed_cap}")

        synths = [s for s in self._synthesized if s.user_approved]
        print(f"\n  Synthesized Capabilities: {len(self._synthesized)} generated, {len(synths)} approved")

        print("━" * 60 + "\n")

    def get_summary(self) -> dict:
        return {
            "lessons_total":      len(self._lessons),
            "lessons_by_outcome": {
                k: sum(1 for ls in self._lessons if ls.outcome == k)
                for k in ["success", "failure", "poor_quality", "timeout", "partial"]
            },
            "open_gaps":          len(self.get_open_gaps()),
            "synthesized_caps":   len(self._synthesized),
            "approved_caps":      len([s for s in self._synthesized if s.user_approved]),
            "routing_pairs":      len(self._routing_weights),
        }

    # ── Internal Helpers ───────────────────────────────────────────────────────

    async def _analyze_root_cause(
        self,
        task: str,
        result,
        genome,
        coalition: dict,
        outcome: str,
        llm: Optional[object],
    ) -> tuple[str, str]:
        """
        Determine WHY a task failed or underperformed.

        Uses LLM analysis if available, otherwise falls back to
        heuristic pattern matching on the result object.
        """
        if llm and outcome in ("failure", "poor_quality"):
            failed_nodes = [
                f"• {n.description[:60]}: {n.error or 'no output'}"
                for n in genome.nodes.values()
                if n.error or not n.result
            ]
            errors = "\n".join(result.error_messages[:3])
            prompt = textwrap.dedent(f"""
                Analyze this AI task execution failure briefly.

                Task: {task[:200]}
                Outcome: {outcome}
                Quality score: {result.quality_score:.2f}
                Failed subtasks:
                {chr(10).join(failed_nodes) or 'None specifically failed'}
                Errors: {errors or 'None recorded'}
                Agents used: {list(set(coalition.values()))}

                Respond with exactly two lines:
                ROOT_CAUSE: one of [wrong_agent, decomposition_error, capability_gap, contract_violation, tool_error, data_quality, ambiguous_task]
                ADJUSTMENT: one sentence on what to do differently next time
            """)
            try:
                response = await llm.complete(prompt, max_tokens=100)
                lines = response.strip().splitlines()
                root_cause = "unknown"
                adjustment = "Review task requirements more carefully"
                for line in lines:
                    if line.startswith("ROOT_CAUSE:"):
                        root_cause = line.split(":", 1)[1].strip()
                    elif line.startswith("ADJUSTMENT:"):
                        adjustment = line.split(":", 1)[1].strip()
                return root_cause, adjustment
            except Exception:
                pass

        # Heuristic fallback
        if result.error_messages:
            err = " ".join(result.error_messages).lower()
            if "timeout" in err:
                return "contract_violation", "Break this task into smaller subtasks or increase time contract"
            if "permission" in err:
                return "permission_denied", "Request the necessary permissions before starting this task type"
            if "no agent" in err or "coalition" in err:
                return "capability_gap", "No registered agent can handle this task type — consider adding one"
            if "tool" in err or "api" in err:
                return "tool_error", "Check tool connectivity and API credentials before retrying"

        if result.quality_score < 0.4:
            return "decomposition_error", "Task likely needs clearer decomposition into simpler, more focused subtasks"

        return "unknown", "Review the task description for ambiguity and try again with more context"

    async def _handle_capability_gap(
        self,
        task:          str,
        task_category: str,
        root_cause:    str,
        outcome:       str,
        llm:           Optional[object],
    ):
        """Identify and record a capability gap from repeated failures."""
        gap_key = f"{task_category}_{root_cause}"

        if gap_key in self._gaps:
            self._gaps[gap_key].failure_count += 1
            if len(self._gaps[gap_key].example_tasks) < 5:
                self._gaps[gap_key].example_tasks.append(task[:100])
            print(f"[Learning] ⚠️  Capability gap growing: {gap_key} "
                  f"(failed {self._gaps[gap_key].failure_count}x)")
            return

        # New gap identified
        proposed_cap  = f"{task_category}_specialist"
        proposed_desc = f"Agent specialized in handling {task_category} tasks"

        if llm and root_cause == "capability_gap":
            try:
                prompt = (
                    f"For an AI assistant that keeps failing at this task type: '{task_category}'\n"
                    f"Example: {task[:100]}\n"
                    f"Suggest in ONE LINE: what capability name and description would fill this gap.\n"
                    f"Format: CAPABILITY_NAME | Description of what the agent should do"
                )
                suggestion = await llm.complete(prompt, max_tokens=60)
                parts = suggestion.strip().split("|", 1)
                if len(parts) == 2:
                    proposed_cap  = parts[0].strip().lower().replace(" ", "_")
                    proposed_desc = parts[1].strip()
            except Exception:
                pass

        gap = CapabilityGap(
            task_type      = task_category,
            failure_count  = self._failure_patterns.get(gap_key, 1),
            example_tasks  = [task[:100]],
            proposed_cap   = proposed_cap,
            proposed_desc  = proposed_desc,
        )
        self._gaps[gap_key] = gap

        print("\n[Learning] 🔴 New capability gap identified!")
        print(f"  Task type:   {task_category}")
        print(f"  Root cause:  {root_cause}")
        print(f"  Suggested:   {proposed_cap} — {proposed_desc}")
        print(f"  Gap ID:      {gap.gap_id}")
        print(f"  To synthesize a new agent: await aria.learning.synthesize_capability('{gap.gap_id}', user_id, aria.permissions)\n")

    def _update_routing_weight(
        self,
        agent_id:      str,
        task_category: str,
        quality:       float,
        duration_s:    float,
        success:       bool,
    ):
        key = f"{agent_id}|{task_category}"
        if key not in self._routing_weights:
            self._routing_weights[key] = RoutingWeight(
                agent_id      = agent_id,
                task_category = task_category,
            )
        w = self._routing_weights[key]
        n = w.executions
        w.avg_quality   = (w.avg_quality * n + quality)   / (n + 1)
        w.avg_duration_s = (w.avg_duration_s * n + duration_s) / (n + 1)
        w.success_rate  = (w.success_rate * n + (1.0 if success else 0.0)) / (n + 1)
        w.executions    += 1
        w.last_updated  = time.time()

    def _categorize_task(self, task: str) -> str:
        """Simple keyword-based task categorization."""
        task_lower = task.lower()
        if any(w in task_lower for w in ["email", "outlook", "inbox", "reply", "send"]):
            return "email"
        if any(w in task_lower for w in ["teams", "channel", "message", "chat", "slack"]):
            return "messaging"
        if any(w in task_lower for w in ["research", "search", "find", "look up", "what is"]):
            return "research"
        if any(w in task_lower for w in ["write", "draft", "document", "report", "summary"]):
            return "writing"
        if any(w in task_lower for w in ["code", "python", "function", "script", "debug"]):
            return "coding"
        if any(w in task_lower for w in ["calendar", "schedule", "meeting", "remind"]):
            return "scheduling"
        if any(w in task_lower for w in ["analyze", "data", "chart", "trend", "compare"]):
            return "analysis"
        return "general"

    def _fingerprint(self, task: str) -> str:
        """Create a stable semantic fingerprint for task similarity matching."""
        normalized = re.sub(r'\W+', ' ', task.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    def _extract_tags(self, task: str) -> list[str]:
        tags = [self._categorize_task(task)]
        words = task.lower().split()
        keywords = {"urgent", "important", "review", "quick", "detailed", "weekly", "daily"}
        tags += [w for w in words if w in keywords]
        return list(set(tags))

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self):
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "lessons": [
                    {
                        "lesson_id":  ls.lesson_id,
                        "task_sample": ls.task_sample,
                        "outcome":    ls.outcome,
                        "root_cause": ls.root_cause,
                        "agents_used": ls.agents_used,
                        "quality_score": ls.quality_score,
                        "adjustment": ls.adjustment,
                        "learned_at": ls.learned_at,
                        "tags":       ls.tags,
                    }
                    for ls in self._lessons[-1000:]  # Keep last 1000 lessons
                ],
                "routing_weights": {
                    k: {
                        "agent_id":       v.agent_id,
                        "task_category":  v.task_category,
                        "executions":     v.executions,
                        "avg_quality":    v.avg_quality,
                        "success_rate":   v.success_rate,
                        "avg_duration_s": v.avg_duration_s,
                    }
                    for k, v in self._routing_weights.items()
                },
                "gaps": {
                    k: {
                        "gap_id":       v.gap_id,
                        "task_type":    v.task_type,
                        "failure_count": v.failure_count,
                        "proposed_cap": v.proposed_cap,
                        "proposed_desc": v.proposed_desc,
                        "status":       v.status,
                    }
                    for k, v in self._gaps.items()
                },
            }
            self._persist_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[Learning] Could not save: {e}")

    def _load(self):
        try:
            data = json.loads(self._persist_path.read_text())
            for entry in data.get("lessons", []):
                lesson = Lesson(**{k: v for k, v in entry.items() if k in Lesson.__dataclass_fields__})
                self._lessons.append(lesson)
            for k, w in data.get("routing_weights", {}).items():
                self._routing_weights[k] = RoutingWeight(**w)
            for k, g in data.get("gaps", {}).items():
                self._gaps[k] = CapabilityGap(**{
                    kk: vv for kk, vv in g.items() if kk in CapabilityGap.__dataclass_fields__
                })
            print(f"[Learning] Loaded {len(self._lessons)} lessons, "
                  f"{len(self._routing_weights)} routing weights, "
                  f"{len(self._gaps)} gaps")
        except Exception as e:
            print(f"[Learning] Could not load saved data: {e}")
