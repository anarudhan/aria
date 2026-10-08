"""
ARIA Execution Engine
---------------------
Runs a TaskGenome by executing nodes in topological order,
parallelizing independent nodes automatically.

The executor is the runtime heart of ARIA. It:
  - Reads the TaskGenome's execution layers
  - Runs all nodes in each layer concurrently (asyncio.gather)
  - Enforces execution contracts (asyncio.wait_for with timeout)
  - Updates the trust ledger after each node
  - Stores results in memory
  - Draws a live execution graph in the terminal (optional)
  - Handles retries with exponential backoff
  - Routes contract violations to backup agents

The execution model is: layers are sequential, nodes within a layer are parallel.
Layer N starts only when ALL nodes in Layer N-1 have settled (completed or failed).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Optional

from .models import (
    AgentMessage,
    ExecutionContract,
    MemoryRing,
    TaskGenome,
    TaskNode,
    TaskResult,
    TaskStatus,
)

if TYPE_CHECKING:
    from .memory import TemporalMemory
    from .registry import CapabilityRegistry
    from .trust import TrustLedger


# ─── Execution Graph Printer ──────────────────────────────────────────────────

STATUS_ICONS = {
    TaskStatus.PENDING:            "⬜",
    TaskStatus.QUEUED:             "🔵",
    TaskStatus.RUNNING:            "🔄",
    TaskStatus.COMPLETED:          "✅",
    TaskStatus.FAILED:             "❌",
    TaskStatus.TIMEOUT:            "⏰",
    TaskStatus.CONTRACT_VIOLATED:  "🚫",
    TaskStatus.CANCELLED:          "⛔",
    TaskStatus.PAUSED:             "⏸️",
}


def print_execution_graph(genome: TaskGenome, coalition: dict[str, str]):
    """Print a live ASCII execution tree to the terminal."""
    print("\n" + "─" * 60)
    print(f"  ARIA Execution  |  Task: {genome.original_task[:40]}...")
    print("─" * 60)

    for layer_idx, layer in enumerate(genome.execution_order):
        print(f"\n  Layer {layer_idx + 1} {'(parallel)' if len(layer) > 1 else ''}")
        for node_id in layer:
            node = genome.nodes[node_id]
            icon  = STATUS_ICONS.get(node.status, "?")
            agent = coalition.get(node_id, "unassigned")
            desc  = node.description[:45]
            duration = f"{node.duration_seconds:.1f}s" if node.duration_seconds else ""
            print(f"    {icon}  [{node_id}] {desc:<46} → {agent:<20} {duration}")

    print("─" * 60 + "\n")


# ─── Execution Engine ─────────────────────────────────────────────────────────

class ExecutionEngine:
    """
    Runs TaskGenome instances, enforcing contracts and managing agent lifecycle.

    Usage:
        engine = ExecutionEngine(registry, trust_ledger, memory)
        result = await engine.run(genome, coalition, contract)
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        trust_ledger: TrustLedger,
        memory: TemporalMemory,
        enable_graph_print: bool = True,
        max_parallel_agents: int = 8,
    ):
        self._registry      = registry
        self._trust_ledger  = trust_ledger
        self._memory        = memory
        self._graph_print   = enable_graph_print
        self._max_parallel  = max_parallel_agents
        self._semaphore     = asyncio.Semaphore(max_parallel_agents)
        self._message_bus: dict[str, asyncio.Queue] = {}  # agent_id -> inbox
        # Set by the orchestrator before run(); copied into each run's context.
        self._initial_context: dict[str, Any] = {}

    # ── Main Entry Point ──────────────────────────────────────────────────────

    async def run(
        self,
        genome: TaskGenome,
        coalition: dict[str, str],    # node_id -> agent_id
        contract: ExecutionContract,
    ) -> TaskResult:
        """
        Execute a TaskGenome with the given coalition assignment.
        Returns a TaskResult with all node results and metadata.
        """
        result = TaskResult(
            genome_id     = genome.genome_id,
            original_task = genome.original_task,
            status        = TaskStatus.RUNNING,
        )
        start_time = time.time()

        # Register message queues for all coalition agents
        for agent_id in set(coalition.values()):
            self._message_bus[agent_id] = asyncio.Queue()

        # Context accumulates results from completed nodes
        # so downstream nodes can access upstream outputs.
        # Also picks up any initial context injected by the orchestrator
        # (e.g. llm_plugin so agents can make real LLM calls).
        completed_ids: set[str] = set()
        context: dict[str, Any] = dict(getattr(self, "_initial_context", {}))

        try:
            # Execute layers in order
            for layer_idx, layer in enumerate(genome.execution_order):
                if contract.is_expired:
                    print(f"[Executor] Contract expired before layer {layer_idx + 1}")
                    break

                if self._graph_print:
                    print_execution_graph(genome, coalition)

                # Filter layer to only nodes that still need to run
                pending_in_layer = [
                    nid for nid in layer
                    if genome.nodes[nid].status == TaskStatus.PENDING
                ]

                if not pending_in_layer:
                    continue

                # Mark all nodes in this layer as QUEUED
                for nid in pending_in_layer:
                    genome.nodes[nid].status = TaskStatus.QUEUED

                # Run all nodes in this layer concurrently
                layer_tasks = [
                    self._run_node_with_retry(
                        node     = genome.nodes[nid],
                        agent_id = coalition.get(nid),
                        context  = dict(context),  # Snapshot of current context
                        contract = contract,
                    )
                    for nid in pending_in_layer
                ]

                layer_results = await asyncio.gather(*layer_tasks, return_exceptions=True)

                # Process layer results
                for nid, layer_result in zip(pending_in_layer, layer_results):
                    node = genome.nodes[nid]
                    if isinstance(layer_result, Exception):
                        node.status = TaskStatus.FAILED
                        node.error  = str(layer_result)
                        result.error_messages.append(f"[{nid}] {str(layer_result)[:200]}")
                    else:
                        completed_ids.add(nid)
                        context[nid] = layer_result
                        result.node_results[nid] = layer_result

            # Final execution graph print
            if self._graph_print:
                print_execution_graph(genome, coalition)

        except asyncio.CancelledError:
            result.status = TaskStatus.CANCELLED
            return result
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error_messages.append(f"Executor error: {str(e)}")
            return result

        # Compute final result
        result.total_seconds  = time.time() - start_time
        result.total_tokens   = contract.tokens_consumed
        result.total_cost_usd = contract.cost_consumed
        result.agents_used    = list(set(coalition.values()))
        result.completed_at   = time.time()

        if genome.failed_nodes():
            result.status = TaskStatus.FAILED if not genome.successful_nodes() else TaskStatus.COMPLETED
        else:
            result.status = TaskStatus.COMPLETED

        # Synthesize final answer from last layer output.
        # If the last layer failed or timed out, fall back to any completed
        # node result we have — never throw away good work from earlier layers.
        last_layer = genome.execution_order[-1] if genome.execution_order else []
        last_results = [str(context[nid]) for nid in last_layer if nid in context]
        if not last_results:
            skip = {"llm_plugin"}
            all_results = [str(v) for k, v in context.items() if k not in skip and v]
            last_results = all_results
        result.final_answer = "\n\n".join(last_results) if last_results else "No results produced"

        # Store result in memory
        self._memory.store_result(
            genome_id = genome.genome_id,
            node_id   = "final",
            result    = result.final_answer,
            agent_id  = "orchestrator",
            ring      = MemoryRing.HOT,
        )

        return result

    # ── Node Execution ────────────────────────────────────────────────────────

    async def _run_node_with_retry(
        self,
        node: TaskNode,
        agent_id: Optional[str],
        context: dict[str, Any],
        contract: ExecutionContract,
    ) -> Any:
        """
        Execute a single node with retry logic and contract enforcement.

        Retry strategy: if the first attempt fails and retries are available,
        we try a different agent (not the same one that just failed).
        This is a key ARIA feature — automatic failover to backup agents.
        """
        if not agent_id:
            raise RuntimeError(f"No agent assigned to node {node.node_id}")

        tried_agents = set()
        current_agent_id = agent_id

        for attempt in range(contract.max_retries + 1):
            if node.contract.is_expired:
                raise TimeoutError(f"Node {node.node_id} contract expired")

            node.attempt_count += 1
            node.status        = TaskStatus.RUNNING
            node.started_at    = time.time()
            tried_agents.add(current_agent_id)

            try:
                result = await self._run_node_once(node, current_agent_id, context)
                # Success
                node.status       = TaskStatus.COMPLETED
                node.result       = result
                node.completed_at = time.time()

                self._trust_ledger.record_success(
                    agent_id         = current_agent_id,
                    task_description = node.description,
                    quality          = 0.8,
                    seconds          = node.duration_seconds or 0.0,
                )
                self._registry.mark_idle(current_agent_id)
                return result

            except asyncio.TimeoutError:
                node.status       = TaskStatus.TIMEOUT
                node.completed_at = time.time()
                self._trust_ledger.record_timeout(current_agent_id, node.description)
                self._registry.mark_idle(current_agent_id)
                print(f"[Executor] ⏰ Node {node.node_id} timed out on agent '{current_agent_id}'")

            except Exception as e:
                node.status       = TaskStatus.FAILED
                node.error        = str(e)
                node.completed_at = time.time()
                self._trust_ledger.record_failure(current_agent_id, node.description, 0.0)
                self._registry.mark_idle(current_agent_id)
                print(f"[Executor] ❌ Node {node.node_id} failed on agent '{current_agent_id}': {e}")

            # Find a backup agent for retry
            if attempt < contract.max_retries:
                backup = self._find_backup_agent(node, tried_agents)
                if backup:
                    current_agent_id = backup
                    node.status = TaskStatus.PENDING
                    print(f"[Executor] 🔄 Retrying node {node.node_id} with backup agent '{backup}'")
                else:
                    print(f"[Executor] No backup agent available for node {node.node_id}")
                    break

        raise RuntimeError(f"Node {node.node_id} failed after {node.attempt_count} attempts")

    async def _run_node_once(
        self,
        node: TaskNode,
        agent_id: str,
        context: dict[str, Any],
    ) -> Any:
        """Run a single execution attempt on the given agent with timeout."""
        agent = self._registry.get_agent(agent_id)
        if not agent:
            raise RuntimeError(f"Agent '{agent_id}' not found in registry")

        # Build a clean summary of upstream results so the agent can see
        # what previous layers actually produced. This is what allows Layer 3
        # to synthesise Layer 1+2, and Layer 4 to write from Layer 3's output.
        skip_keys = {"llm_plugin"}
        upstream = {
            k: v for k, v in context.items()
            if k not in skip_keys and isinstance(k, str) and k.startswith("t")
        }
        if upstream:
            parts = [
                f"[Result from subtask {k}]:\n{str(v)[:2000]}"
                for k, v in sorted(upstream.items())
            ]
            upstream_text = "\n\n".join(parts)
            context = dict(context)
            context["upstream_results"] = upstream_text

        self._registry.mark_busy(agent_id)
        timeout = node.contract.remaining_seconds

        async with self._semaphore:
            return await asyncio.wait_for(
                agent.execute(node, context, self._memory),
                timeout=timeout,
            )

    def _find_backup_agent(
        self,
        node: TaskNode,
        already_tried: set[str],
    ) -> Optional[str]:
        """Find an alternative agent for a failed node."""
        return self._registry.find_best_agent(
            task_description = node.description,
            required_tools   = node.required_tools,
            trust_ledger     = self._trust_ledger,
            exclude_agents   = list(already_tried),
        )

    # ── Coalition Message Bus ─────────────────────────────────────────────────

    async def _deliver_messages(self, genome_id: str, coalition: dict[str, str]):
        """
        Background task: deliver messages between coalition members.
        Runs for the duration of a genome execution.
        """
        all_agent_ids = set(coalition.values())
        while True:
            await asyncio.sleep(0.1)
            for agent_id in all_agent_ids:
                agent = self._registry.get_agent(agent_id)
                if not agent:
                    continue
                # Drain outbox and route to recipients
                while not agent._outbox.empty():
                    msg: AgentMessage = await agent._outbox.get()
                    if msg.genome_id != genome_id:
                        continue
                    if msg.recipient_id:
                        # Direct message
                        target_agent = self._registry.get_agent(msg.recipient_id)
                        if target_agent:
                            await target_agent._message_queue.put(msg)
                    else:
                        # Broadcast to all coalition members
                        for target_id in all_agent_ids:
                            if target_id != agent_id:
                                target = self._registry.get_agent(target_id)
                                if target:
                                    await target._message_queue.put(msg)
