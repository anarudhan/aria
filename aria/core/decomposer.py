"""
ARIA Task Decomposer
--------------------
Converts a plain English task description into a TaskGenome (DAG of subtasks).

This is the part of ARIA that makes it genuinely autonomous — you don't draw
a workflow. You describe what you want, and the decomposer figures out the
optimal execution plan.

The decomposer uses an LLM to:
  1. Assess the complexity and domain of the task
  2. Break it into atomic subtasks
  3. Identify dependencies between subtasks (what must finish before what)
  4. Estimate effort and set execution contracts per subtask
  5. Identify which capabilities and tools each subtask needs

The decomposer also checks the memory system for similar past tasks —
if we've solved something like this before, we can reuse the genome
structure and skip the LLM call entirely (or use it as a starting point).

Decomposition depth is bounded by ARIAConfig.max_decomposition_depth.
A depth of 1 means: break the task into independent parallel steps.
A depth of 2 means: each of those steps can itself be broken down.
Most tasks need depth 1 or 2. Very complex tasks (build a full app) may need 3.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from .models import ARIAConfig, ExecutionContract, TaskGenome, TaskNode

# ─── Decomposition Prompt Template ────────────────────────────────────────────

DECOMPOSE_PROMPT = """You are ARIA's Task Decomposer. Your job is to break a task into an optimal execution plan.

TASK: {task}

AVAILABLE AGENTS AND CAPABILITIES:
{agent_capabilities}

RELEVANT PAST CONTEXT:
{memory_context}

INSTRUCTIONS:
Analyze the task and produce a JSON execution plan. Each subtask must be:
- Atomic: one agent can complete it without needing another agent mid-execution
- Clear: unambiguous description of what to do and what output is expected
- Properly sequenced: dependencies must be correct (no circular deps)

Return ONLY valid JSON in this exact format:
{{
  "complexity_score": 0.0-1.0,
  "estimated_total_seconds": number,
  "subtasks": [
    {{
      "node_id": "t1",
      "description": "...",
      "expected_output": "...",
      "dependencies": [],
      "required_capabilities": ["research", "web_search"],
      "required_tools": ["tavily_search"],
      "estimated_seconds": number,
      "priority": "high|medium|low"
    }}
  ]
}}

RULES:
- Keep subtasks focused. Maximum {max_tasks} subtasks total.
- If the task is simple enough to do in one step, return just one subtask.
- Use node_ids like t1, t2, t3 etc.
- Dependencies reference node_ids that must complete first: "dependencies": ["t1", "t2"]
- Parallelizable tasks should have NO overlapping dependencies
- The last subtask should synthesize all results into a final answer if needed
"""


# ─── Decomposer ───────────────────────────────────────────────────────────────

class TaskDecomposer:
    """
    Uses an LLM to decompose a natural language task into a TaskGenome.

    The decomposer is the most LLM-intensive part of ARIA — it runs once
    per task (not once per subtask). After decomposition, the actual execution
    uses the genome structure to orchestrate agents efficiently.

    Memory integration: before calling the LLM, the decomposer searches
    HOT and WARM memory for similar past genomes and provides them as
    context. This improves decomposition quality and reduces hallucination
    of non-existent tools.
    """

    def __init__(
        self,
        config: ARIAConfig,
        llm_caller: Optional[object] = None,  # Will be set by orchestrator
    ):
        self.config = config
        self._llm_caller = llm_caller

    def set_llm(self, llm_caller: object):
        self._llm_caller = llm_caller

    async def decompose(
        self,
        task: str,
        agent_summaries: list[dict],
        memory_context: Optional[str] = None,
        contract: Optional[ExecutionContract] = None,
    ) -> TaskGenome:
        """
        Decompose a task into a TaskGenome.

        Args:
            task:             The plain English task description
            agent_summaries:  List of dicts describing available agents
            memory_context:   Relevant past context from memory recall
            contract:         The execution contract for the whole task

        Returns:
            A TaskGenome ready for the executor to run
        """
        genome = TaskGenome(original_task=task)

        if self._llm_caller is None:
            # Fallback: single-node genome when no LLM available for decomposition
            return self._single_node_genome(task, genome, contract)

        agent_text = self._format_agent_summaries(agent_summaries)
        memory_text = memory_context or "No relevant past context found."
        max_tasks = min(10, self.config.max_decomposition_depth * 4)

        prompt = DECOMPOSE_PROMPT.format(
            task               = task,
            agent_capabilities = agent_text,
            memory_context     = memory_text,
            max_tasks          = max_tasks,
        )

        try:
            response = await self._llm_caller.complete(
                prompt    = prompt,
                max_tokens = 2000,
                model      = self.config.decomposer_model,
            )
            genome = self._parse_response(response, task, genome, contract)
        except Exception as e:
            print(f"[Decomposer] LLM error: {e}. Falling back to single-node genome.")
            genome = self._single_node_genome(task, genome, contract)

        genome.decomposition_model = self.config.decomposer_model
        return genome

    def _parse_response(
        self,
        response: str,
        original_task: str,
        genome: TaskGenome,
        contract: Optional[ExecutionContract],
    ) -> TaskGenome:
        """Parse the LLM JSON response into a TaskGenome."""
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return self._single_node_genome(original_task, genome, contract)

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return self._single_node_genome(original_task, genome, contract)

        genome.complexity_score        = float(data.get("complexity_score", 0.5))
        genome.estimated_total_seconds = float(data.get("estimated_total_seconds", 60.0))

        subtasks = data.get("subtasks", [])
        if not subtasks:
            return self._single_node_genome(original_task, genome, contract)

        for s in subtasks:
            node_contract = contract.child_contract() if contract else ExecutionContract(
                max_seconds = float(s.get("estimated_seconds", 30.0)) * 2,
            )

            node = TaskNode(
                node_id               = s.get("node_id", f"t{len(genome.nodes)+1}"),
                description           = s.get("description", ""),
                expected_output       = s.get("expected_output", ""),
                dependencies          = s.get("dependencies", []),
                required_capabilities = s.get("required_capabilities", []),
                required_tools        = s.get("required_tools", []),
                contract              = node_contract,
            )
            genome.add_node(node)

        # Compute execution order (topological sort by layer)
        genome.execution_order = self._topological_layers(genome)
        return genome

    def _single_node_genome(
        self,
        task: str,
        genome: TaskGenome,
        contract: Optional[ExecutionContract],
    ) -> TaskGenome:
        """Fallback: one node that does everything."""
        node = TaskNode(
            node_id      = "t1",
            description  = task,
            expected_output = "Complete answer to the user's request",
            dependencies = [],
            contract     = contract or ExecutionContract(),
        )
        genome.add_node(node)
        genome.execution_order = [["t1"]]
        genome.complexity_score = 0.3
        return genome

    def _topological_layers(self, genome: TaskGenome) -> list[list[str]]:
        """
        Compute topological sort in layers.
        Each layer contains nodes that can run in parallel.
        Layer N+1 only starts when all nodes in layer N are done.

        Uses Kahn's algorithm for topological sorting.
        """
        in_degree: dict[str, int] = {nid: 0 for nid in genome.nodes}
        children: dict[str, list[str]] = {nid: [] for nid in genome.nodes}

        for node in genome.nodes.values():
            for dep in node.dependencies:
                if dep in in_degree:
                    in_degree[node.node_id] += 1
                    children[dep].append(node.node_id)

        layers = []
        ready = [nid for nid, deg in in_degree.items() if deg == 0]

        while ready:
            layers.append(list(ready))
            next_ready = []
            for nid in ready:
                for child in children[nid]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_ready.append(child)
            ready = next_ready

        return layers

    @staticmethod
    def _format_agent_summaries(summaries: list[dict]) -> str:
        if not summaries:
            return "No agents registered yet."
        lines = []
        for s in summaries:
            domains = ", ".join(s.get("domains", []))
            lines.append(f"  - {s['agent_id']} [{domains}] (tools: {s.get('status', 'idle')})")
        return "\n".join(lines)

    def estimate_complexity(self, task: str) -> float:
        """
        Quick heuristic complexity estimate without an LLM call.
        Used to decide whether to decompose at all.
        Score: 0.0 (trivial) to 1.0 (very complex)
        """
        score = 0.0
        tokens = task.lower().split()
        word_count = len(tokens)

        # Length heuristic
        if word_count > 50:
            score += 0.3
        elif word_count > 20:
            score += 0.15

        # Multi-step keywords
        multi_step_words = {"then", "after", "next", "also", "additionally",
                           "finally", "and", "summarize", "compare", "analyze"}
        if sum(1 for t in tokens if t in multi_step_words) >= 2:
            score += 0.2

        # Tool requirement keywords
        tool_words = {"search", "browse", "email", "send", "create", "write",
                     "generate", "download", "analyze", "compare", "research"}
        matched_tools = sum(1 for t in tokens if t in tool_words)
        score += min(0.3, matched_tools * 0.1)

        # Question complexity
        if "?" in task and any(w in tokens for w in ["why", "how", "what", "compare"]):
            score += 0.1

        return min(1.0, score)
