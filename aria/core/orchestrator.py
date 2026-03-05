"""
ARIA Orchestrator — Version 1.2.0
------------------------------------
The top-level coordinator that brings all components together.

Pipeline:
  1. PERMISSION CHECK  — verify the user has granted access for what is being asked
  2. RECALL            — search memory for similar past tasks and context
  2b. PRISM PRE-REASON — honest cognitive assessment before any planning begins
  3. DECOMPOSE         — use LLM to break task into a TaskGenome DAG
  4. COALESCE          — form agent coalition via semantic fingerprint matching
  5. EXECUTE           — run the genome with contract enforcement and auto-retry
  5b. PRISM MONITOR    — watch for execution drift against predicted strategy
  6. REFLECT           — retrospective analysis for continuous improvement
  6b. PRISM REFLECT    — calibration update and self-knowledge recording
  7. REMEMBER          — store results and routing hints in memory
  8. HEALTH UPDATE     — report outcomes to self-consciousness for health tracking

New in v1.1.0:
  - PermissionEngine: every capability requires explicit user grant
  - SelfConsciousness: ARIA monitors its own health and security
  - SecurityScanner: ARIA reads its own code for vulnerabilities
  - SelfPatcher: applies security patches with user approval and rollback
  - Source integrity: detects unexpected external file changes

New in v1.2.0:
  - PRISM: Progressive Reasoning with Integrated Self-Monitoring
    ARIA now thinks honestly before acting, watches herself during execution,
    and reflects on calibration accuracy after every task. Over time she builds
    genuine self-knowledge about where she is strong and where she misjudges herself.
  - SQLite memory backend: 27x faster writes, full persistence across restarts
  - Claude-native web search: web agent uses Claude API search natively
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

from ..agents.base import BaseAgent
from ..permissions.engine import Capability, PermissionDeniedError, PermissionEngine
from .decomposer import TaskDecomposer
from .executor import ExecutionEngine
from .memory import TemporalMemory
from .models import ARIAConfig, ExecutionContract, MemoryRing, TaskResult, TaskStatus
from .prism import PRISMEngine
from .registry import CapabilityRegistry
from .self_aware.consciousness import SelfConsciousness
from .trust import TrustLedger


class ARIAOrchestrator:
    """
    The ARIA Orchestrator — run any task with any number of agents.

    Quick start:

        aria = ARIAOrchestrator(aria_root="/path/to/aria", llm_plugin=AnthropicPlugin(api_key="sk-..."))

        # Grant yourself permissions first (ARIA never assumes access)
        aria.permissions.grant("alice", Capability.MEMORY_WRITE)
        aria.permissions.grant("alice", Capability.WEB_SEARCH)

        # Register agents and run
        aria.register_agent(my_researcher)
        result = await aria.run("Research AI trends", user_id="alice")
        print(result.final_answer)
    """

    def __init__(
        self,
        config: Optional[ARIAConfig] = None,
        llm_plugin: Optional[object] = None,
        aria_root: Optional[str] = None,
    ):
        self.config   = config or ARIAConfig()
        self.registry = CapabilityRegistry()
        self.trust    = TrustLedger(
            persist_path = self.config.memory_persist_path,
            suspension_threshold = self.config.trust_suspension_threshold,
            min_executions_for_suspension = self.config.trust_min_executions_for_suspension,
        )
        self.memory = TemporalMemory(
            max_entries  = {
                MemoryRing.FLASH: self.config.flash_ring_max_entries,
                MemoryRing.HOT:   self.config.hot_ring_max_entries,
                MemoryRing.WARM:  self.config.warm_ring_max_entries,
                MemoryRing.COLD:  20000,
            },
            persist_path = self.config.memory_persist_path,
        )
        self.executor = ExecutionEngine(
            registry            = self.registry,
            trust_ledger        = self.trust,
            memory              = self.memory,
            enable_graph_print  = self.config.enable_execution_graph,
            max_parallel_agents = self.config.max_parallel_agents,
        )

        # NEW v1.1.0: Permissions and self-awareness
        self.permissions = PermissionEngine(
            persist_path = self.config.memory_persist_path
        )

        # Determine ARIA root for self-awareness
        _root = aria_root or str(Path(__file__).parent.parent)
        self.self_awareness = SelfConsciousness(aria_root=_root)

        # LLM plugin
        self._llm_plugin = llm_plugin
        self.decomposer = TaskDecomposer(config=self.config, llm_caller=llm_plugin)

        # PRISM — cognitive engine that runs before, during, and after every task
        self.prism = PRISMEngine(persist_path=self.config.memory_persist_path)

        # Initialize self-awareness baseline
        self.self_awareness.initialize(self)

        print("=" * 62)
        print("  ARIA — Autonomous Reasoning and Intelligence Architecture")
        print("  Version 1.2.0 | Zero-Trust | Self-Aware | PRISM-Guided")
        print("=" * 62)

    # ── Agent Registration ─────────────────────────────────────────────────────

    def register_agent(self, agent_or_func) -> str:
        """Register an agent. Accepts BaseAgent instances or @aria_agent-decorated functions."""
        if hasattr(agent_or_func, '_is_aria_agent') and agent_or_func._is_aria_agent:
            agent = agent_or_func._aria_agent
        elif isinstance(agent_or_func, BaseAgent):
            agent = agent_or_func
        else:
            raise TypeError(
                f"Cannot register {type(agent_or_func)}. "
                "Pass a BaseAgent instance or @aria_agent-decorated function."
            )

        if hasattr(agent, '_llm') and agent._llm is None and self._llm_plugin:
            agent.set_llm(self._llm_plugin)

        self.registry.register(agent.fingerprint, agent)
        return agent.agent_id

    def register_agents(self, agents: list) -> list[str]:
        return [self.register_agent(a) for a in agents]

    def set_llm(self, llm_plugin: object):
        self._llm_plugin = llm_plugin
        self.decomposer.set_llm(llm_plugin)
        for agent_id in [r["agent_id"] for r in self.registry.list_agents()]:
            agent = self.registry.get_agent(agent_id)
            if agent and hasattr(agent, 'set_llm'):
                agent.set_llm(llm_plugin)

    # ── Main Execution Pipeline ────────────────────────────────────────────────

    async def run(
        self,
        task: str,
        user_id: str = "default",
        contract: Optional[ExecutionContract] = None,
        context: Optional[dict] = None,
        skip_decomposition: bool = False,
        required_capabilities: Optional[list[Capability]] = None,
    ) -> TaskResult:
        """
        Run a task through the full ARIA pipeline.

        IMPORTANT: ARIA checks that the user has granted MEMORY_WRITE permission
        before storing any results. If a task requires specific capabilities
        (e.g., EMAIL_SEND), pass them in required_capabilities and ARIA will
        verify permission before even starting decomposition.

        Args:
            task:                   Natural language task description
            user_id:                Who is running this task (permission scoping)
            contract:               Resource constraints
            context:                Additional context for agents
            skip_decomposition:     Run as single-node (faster for simple tasks)
            required_capabilities:  Additional capabilities this task needs
        """
        if len(self.registry) == 0:
            raise RuntimeError("No agents registered. Call register_agent() first.")

        # ── Step 0: Permission check for task-level capabilities ──────────────
        if required_capabilities:
            for cap in required_capabilities:
                if not self.permissions.check(user_id, cap, agent_id="orchestrator"):
                    denied = PermissionDeniedError(user_id=user_id, capability=cap)
                    result = TaskResult(original_task=task, status=TaskStatus.FAILED)
                    result.error_messages.append(denied.to_user_message())
                    return result

        contract = contract or ExecutionContract(
            max_seconds  = self.config.coalition_timeout_seconds,
            max_cost_usd = 2.0,
        )

        print(f"\n[ARIA] 🚀 Task: {task[:80]}...")
        print(f"[ARIA]    User: {user_id}")
        pipeline_start = time.time()

        # ── Step 1: Recall ─────────────────────────────────────────────────────
        print("[ARIA] 📚 Searching memory...")
        past_memories = self.memory.recall(
            query   = task,
            rings   = [MemoryRing.FLASH, MemoryRing.HOT, MemoryRing.WARM],
            top_k   = 5,
            user_id = user_id,
        )
        memory_context = "\n".join(f"- {m.content[:200]}" for m in past_memories)
        if memory_context:
            print(f"[ARIA]   Found {len(past_memories)} relevant memories")

        # ── Step 1b: PRISM Pre-Reasoning ───────────────────────────────────────
        # Before planning anything, ARIA honestly assesses the task.
        # Four questions: comprehension, novelty, history, strategy.
        # The output shapes the decomposer prompt and the execution approach.
        past_lessons = None
        if hasattr(self, "learning"):
            past_lessons = list(getattr(self.learning, "_lessons", []))
        trust_summaries = self.trust.get_all_summaries()

        prism_state = self.prism.pre_reason(
            task            = task,
            past_memories   = past_memories,
            user_id         = user_id,
            past_lessons    = past_lessons,
            trust_summaries = trust_summaries,
        )

        # Inject PRISM context into memory context so the decomposer
        # LLM knows exactly what PRISM thinks before it plans the task
        prism_context = prism_state.to_context_string()
        if memory_context:
            memory_context = prism_context + "\n\n" + memory_context
        else:
            memory_context = prism_context

        # ── Step 2: Decompose ──────────────────────────────────────────────────
        if skip_decomposition:
            print("[ARIA] ⚡ Skip decomposition — single-node execution")
            from .decomposer import TaskDecomposer as TD
            from .models import TaskGenome
            genome = TD(self.config)._single_node_genome(task, TaskGenome(original_task=task), contract)
        else:
            complexity = self.decomposer.estimate_complexity(task)
            print(f"[ARIA] 🔬 Decomposing task (complexity={complexity:.2f})...")
            genome = await self.decomposer.decompose(
                task            = task,
                agent_summaries = self.registry.list_agents(),
                memory_context  = memory_context or None,
                contract        = contract,
            )
            print(f"[ARIA]   Genome: {len(genome.nodes)} subtasks in {len(genome.execution_order)} layers")

        # ── Step 3: Coalition Formation ────────────────────────────────────────
        print("[ARIA] 🤝 Forming coalition...")
        coalition = self.registry.find_coalition(
            task_nodes   = list(genome.nodes.values()),
            trust_ledger = self.trust,
        )
        unassigned = [nid for nid in genome.nodes if nid not in coalition]
        if unassigned:
            fallback = self.registry.find_best_agent(task)
            if fallback:
                for nid in unassigned:
                    coalition[nid] = fallback

        if not coalition:
            result = TaskResult(genome_id=genome.genome_id, original_task=task, status=TaskStatus.FAILED)
            result.error_messages.append("No agents available to form a coalition")
            return result

        agents_in_coalition = set(coalition.values())
        print(f"[ARIA]   Coalition: {', '.join(agents_in_coalition)}")

        # ── Step 4: Execute ────────────────────────────────────────────────────
        print("[ARIA] ⚙️  Executing...")
        result = await self.executor.run(genome=genome, coalition=coalition, contract=contract)

        # ── Step 4b: PRISM Monitoring Pass ────────────────────────────────────
        # Walk through every node that ran and tell PRISM what happened.
        # This is how PRISM detects drift between predicted strategy and reality.
        for node_id, node in genome.nodes.items():
            agent_id = coalition.get(node_id, "unknown")
            if node.status.value == "completed":
                node_quality = 0.75  # Reasonable default; replaced by quality_score below
                self.prism.on_node_complete(
                    state           = prism_state,
                    node_id         = node_id,
                    agent_id        = agent_id,
                    quality         = node_quality,
                    expected_quality = 0.6,
                )
            elif node.status.value in ("failed", "timeout", "contract_violated"):
                self.prism.on_node_fail(
                    state    = prism_state,
                    node_id  = node_id,
                    agent_id = agent_id,
                    error    = node.error or "",
                )

        # ── Step 5: Quality Assessment ─────────────────────────────────────────
        result.quality_score = self._assess_quality(task, result)

        # ── Step 5b: PRISM Reflection ──────────────────────────────────────────
        # Compare what PRISM predicted to what actually happened.
        # This is where ARIA updates her self-knowledge and calibration history.
        outcome = "success" if result.status == TaskStatus.COMPLETED else "failure"
        if result.status.value == "partial" or (
            result.status == TaskStatus.FAILED and result.node_results
        ):
            outcome = "partial"

        prism_state = self.prism.reflect(
            state   = prism_state,
            outcome = outcome,
            quality = result.quality_score,
        )

        # Attach PRISM state to result metadata for downstream inspection
        result.metadata = getattr(result, "metadata", {})
        result.metadata["prism"] = prism_state.to_dict()

        # ── Step 6: Retrospective ──────────────────────────────────────────────
        if self.config.enable_retrospective and result.status == TaskStatus.COMPLETED:
            result.retrospective = self._run_retrospective(genome, result, coalition)

        # ── Step 7: Store in Memory (only if user has MEMORY_WRITE permission) ─
        if self.permissions.check(user_id, Capability.MEMORY_WRITE, agent_id="orchestrator"):
            self.memory.store(
                content  = f"Task: {task} → Result: {(result.final_answer or '')[:300]}",
                ring     = MemoryRing.HOT,
                tags     = ["task", f"quality:{result.quality_score:.1f}"],
                metadata = {"genome_id": genome.genome_id, "agents": list(agents_in_coalition),
                            "quality": result.quality_score, "seconds": result.total_seconds},
                user_id  = user_id,
            )

        # ── Step 8: Health Update ──────────────────────────────────────────────
        self.self_awareness.record_task_outcome(
            success = result.status == TaskStatus.COMPLETED,
            seconds = result.total_seconds,
            quality = result.quality_score,
            error   = result.error_messages[0] if result.error_messages else None,
        )

        total_time = time.time() - pipeline_start
        print(f"[ARIA] ✅ Done in {total_time:.1f}s | "
              f"Quality: {result.quality_score:.2f} | "
              f"Cost: ${result.total_cost_usd:.4f}")
        print("-" * 62)
        return result

    async def run_batch(
        self,
        tasks: list[str],
        user_id: str = "default",
        contract: Optional[ExecutionContract] = None,
        max_parallel: int = 3,
    ) -> list[TaskResult]:
        semaphore = asyncio.Semaphore(max_parallel)
        async def run_one(task: str) -> TaskResult:
            async with semaphore:
                return await self.run(task, user_id=user_id, contract=contract)
        return await asyncio.gather(*[run_one(t) for t in tasks])

    # ── Self-Awareness Interface ───────────────────────────────────────────────

    async def security_scan(self, user_id: str = "system") -> dict:
        """
        Run a security scan of ARIA's own source code.
        Returns a dict summary of findings.
        """
        print("[ARIA] 🔍 Running security self-scan...")
        report = self.self_awareness.security_scan(force=True)
        if report is None:
            return {"error": "Scan failed"}
        report.print_report()
        return report.to_dict()

    def health_report(self) -> dict:
        """Return current health metrics."""
        health = self.self_awareness.health_check()
        return {
            "grade":              health.health_grade,
            "uptime_hours":       round(health.uptime_seconds / 3600, 2),
            "tasks_completed":    health.tasks_completed,
            "tasks_failed":       health.tasks_failed,
            "error_rate":         round(health.error_rate, 3),
            "avg_quality":        round(health.avg_quality_score, 2),
            "agents_registered":  health.agents_registered,
            "agents_suspended":   health.agents_suspended,
            "memory_flash":       f"{health.memory_flash_usage:.0%}",
            "memory_hot":         f"{health.memory_hot_usage:.0%}",
            "open_vulnerabilities": health.open_vulnerabilities,
            "patch_count":        health.patch_count,
            "is_healthy":         health.is_healthy,
        }

    def self_report(self):
        """Print a full self-assessment including health, security, and integrity."""
        self.self_awareness.print_self_report()

    def what_i_am(self) -> str:
        """ARIA's honest self-description."""
        return self.self_awareness.what_i_am()

    def integrity_check(self) -> list[dict]:
        """Check if any source files have changed unexpectedly."""
        violations = self.self_awareness.check_integrity()
        return [
            {
                "file":        Path(v.file_path).name,
                "changed":     v.changed,
                "change_type": v.change_type,
            }
            for v in violations
        ]

    # ── PRISM Interface ───────────────────────────────────────────────────────

    def prism_report(self) -> None:
        """
        Print PRISM's calibration report.

        Shows how well ARIA knows herself — where her confidence ratings are
        accurate and where she consistently over or underestimates her ability.
        More tasks = more accurate self-knowledge. Run this after a few dozen
        tasks to see ARIA's genuine self-model taking shape.
        """
        self.prism.print_calibration_report()

    def prism_stats(self) -> dict:
        """
        Return PRISM's calibration data as a dictionary.

        Useful for dashboards, logging, or programmatic inspection.
        """
        return self.prism.calibration_report()

    # ── Inspection and Management ──────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "agents":      self.registry.list_agents(),
            "memory":      self.memory.stats(),
            "trust":       self.trust.get_all_summaries(),
            "llm_ready":   self._llm_plugin is not None,
            "health":      self.health_report(),
        }

    def print_status(self):
        s = self.status()
        print("\n" + "=" * 62)
        print("  ARIA System Status")
        print("=" * 62)
        print(f"\n  Agents ({len(s['agents'])}):")
        for agent in s['agents']:
            trust_summary = self.trust.get_agent_summary(agent['agent_id'])
            trust_score = trust_summary.get('overall_trust', 0.6)
            suspended = " 🚫 SUSPENDED" if trust_summary.get('suspended') else ""
            print(f"    • {agent['agent_id']:<28} [{', '.join(agent['domains'])}] "
                  f"trust={trust_score:.2f} load={agent['load']}/{agent['max_load']}{suspended}")

        print("\n  Memory Rings:")
        for ring, stats in s['memory'].items():
            if not isinstance(stats, dict):
                continue
            bar_len = int(stats['count'] / max(stats['max'], 1) * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"    {ring:<8} [{bar}] {stats['count']:>5}/{stats['max']}")

        h = s['health']
        print(f"\n  Health: {h['grade']} | "
              f"Tasks: {h['tasks_completed']} done / {h['tasks_failed']} failed | "
              f"Quality: {h['avg_quality']}")

        print(f"\n  LLM Plugin: {'✅ Ready' if s['llm_ready'] else '❌ Not configured'}")
        print("=" * 62 + "\n")

    # ── Internal Helpers ───────────────────────────────────────────────────────

    def _assess_quality(self, task: str, result: TaskResult) -> float:
        score = 0.5
        if result.status == TaskStatus.COMPLETED:
            score += 0.2
        elif result.status == TaskStatus.FAILED:
            return 0.1
        answer_len = len(result.final_answer or "")
        if answer_len > 200:
            score += 0.15
        elif answer_len > 50:
            score += 0.05
        elif answer_len < 10:
            score -= 0.2
        if result.error_messages:
            score -= 0.1 * len(result.error_messages)
        return max(0.0, min(1.0, score))

    def _run_retrospective(self, genome, result: TaskResult, coalition: dict[str, str]) -> str:
        lines = ["Post-execution analysis:"]
        for node_id, agent_id in coalition.items():
            node = genome.nodes.get(node_id)
            if not node:
                continue
            if node.status == TaskStatus.COMPLETED:
                lines.append(f"  {agent_id} completed '{node.description[:40]}' ✅ ({node.duration_seconds:.1f}s)")
            elif node.status == TaskStatus.FAILED:
                lines.append(f"  {agent_id} failed on '{node.description[:40]}' ❌")
        lines.append(f"Overall quality: {result.quality_score:.2f}")
        return "\n".join(lines)


# ── Learning integration (appended) ─────────────────────────────────────────
# This adds the SelfImprovementEngine into the orchestrator so it is called
# after every task automatically and exposed as aria.learning

def _attach_learning(orchestrator_instance, llm_plugin=None, persist_path=None):
    """Attach the learning engine to an existing orchestrator instance."""
    from .improvement import SelfImprovementEngine
    orchestrator_instance.learning = SelfImprovementEngine(
        persist_path=persist_path,
        llm_plugin=llm_plugin,
    )

    # Patch the run() method to call learning after each task
    original_run = orchestrator_instance.run

    async def run_with_learning(task, user_id="default", contract=None,
                                context=None, skip_decomposition=False,
                                required_capabilities=None):
        result = await original_run(
            task, user_id=user_id, contract=contract, context=context,
            skip_decomposition=skip_decomposition,
            required_capabilities=required_capabilities,
        )
        # After task completes, learn from it
        if hasattr(orchestrator_instance, '_last_genome') and hasattr(orchestrator_instance, '_last_coalition'):
            await orchestrator_instance.learning.learn_from_execution(
                task=task, result=result,
                coalition=orchestrator_instance._last_coalition,
                genome=orchestrator_instance._last_genome,
                llm_plugin=llm_plugin,
            )
        return result

    orchestrator_instance.run = run_with_learning
    return orchestrator_instance
