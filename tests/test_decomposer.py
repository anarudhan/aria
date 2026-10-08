"""
Unit tests for aria.core.decomposer.TaskDecomposer.

All tests use a scripted mock LLM plugin, so no network access or API keys
are needed.
"""
import json

import pytest

from aria.core.decomposer import TaskDecomposer
from aria.core.models import ARIAConfig, ExecutionContract, TaskGenome, TaskNode


class ScriptedLLM:
    """Mock LLM plugin: returns a canned response (or raises) and records calls."""

    def __init__(self, response: str = "", error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    async def complete(self, prompt: str, max_tokens: int = 2000, model=None) -> str:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "model": model})
        if self.error is not None:
            raise self.error
        return self.response


def plan(*subtasks: dict, complexity: float = 0.7, seconds: float = 90.0) -> str:
    return json.dumps({
        "complexity_score": complexity,
        "estimated_total_seconds": seconds,
        "subtasks": list(subtasks),
    })


def sub(node_id: str, deps=None, **extra) -> dict:
    data = {
        "node_id": node_id,
        "description": f"do {node_id}",
        "expected_output": f"output of {node_id}",
        "dependencies": deps or [],
        "required_capabilities": ["research"],
        "required_tools": ["search"],
        "estimated_seconds": 10,
    }
    data.update(extra)
    return data


AGENTS = [
    {"agent_id": "researcher", "domains": ["research", "web"], "status": "idle"},
    {"agent_id": "writer", "domains": ["writing"], "status": "idle"},
]


@pytest.fixture
def config() -> ARIAConfig:
    return ARIAConfig(decomposer_model="test-model")


# ─── Fallback: no LLM ─────────────────────────────────────────────────────────

class TestNoLLM:

    async def test_returns_single_node_genome(self, config):
        decomposer = TaskDecomposer(config)
        genome = await decomposer.decompose("Write a report", AGENTS)

        assert isinstance(genome, TaskGenome)
        assert genome.original_task == "Write a report"
        assert list(genome.nodes) == ["t1"]
        assert genome.nodes["t1"].description == "Write a report"
        assert genome.nodes["t1"].dependencies == []
        assert genome.execution_order == [["t1"]]
        assert genome.complexity_score == pytest.approx(0.3)

    async def test_given_contract_is_attached_to_the_node(self, config):
        contract = ExecutionContract(max_seconds=42)
        genome = await TaskDecomposer(config).decompose("task", [], contract=contract)
        assert genome.nodes["t1"].contract is contract

    async def test_default_contract_when_none_given(self, config):
        genome = await TaskDecomposer(config).decompose("task", [])
        assert genome.nodes["t1"].contract.max_seconds == ExecutionContract().max_seconds

    async def test_set_llm_enables_llm_decomposition(self, config):
        decomposer = TaskDecomposer(config)
        llm = ScriptedLLM(plan(sub("t1"), sub("t2", ["t1"])))
        decomposer.set_llm(llm)

        genome = await decomposer.decompose("task", AGENTS)

        assert len(llm.calls) == 1
        assert set(genome.nodes) == {"t1", "t2"}


# ─── LLM decomposition ────────────────────────────────────────────────────────

class TestLLMDecomposition:

    async def test_parses_nodes_and_metadata(self, config):
        llm = ScriptedLLM(plan(sub("t1"), sub("t2"), sub("t3", ["t1", "t2"]),
                               complexity=0.8, seconds=120))
        genome = await TaskDecomposer(config, llm).decompose("big task", AGENTS)

        assert set(genome.nodes) == {"t1", "t2", "t3"}
        assert genome.complexity_score == pytest.approx(0.8)
        assert genome.estimated_total_seconds == pytest.approx(120)
        assert genome.decomposition_model == "test-model"

        node = genome.nodes["t3"]
        assert isinstance(node, TaskNode)
        assert node.description == "do t3"
        assert node.expected_output == "output of t3"
        assert node.dependencies == ["t1", "t2"]
        assert node.required_capabilities == ["research"]
        assert node.required_tools == ["search"]

    async def test_execution_order_groups_parallel_nodes_into_layers(self, config):
        llm = ScriptedLLM(plan(sub("t1"), sub("t2"), sub("t3", ["t1", "t2"])))
        genome = await TaskDecomposer(config, llm).decompose("task", AGENTS)

        assert len(genome.execution_order) == 2
        assert sorted(genome.execution_order[0]) == ["t1", "t2"]
        assert genome.execution_order[1] == ["t3"]

    async def test_prompt_contains_task_agents_memory_and_limits(self, config):
        llm = ScriptedLLM(plan(sub("t1")))
        await TaskDecomposer(config, llm).decompose(
            "compare vendors", AGENTS, memory_context="we used vendor X last year",
        )

        call = llm.calls[0]
        assert "compare vendors" in call["prompt"]
        assert "researcher" in call["prompt"] and "writer" in call["prompt"]
        assert "we used vendor X last year" in call["prompt"]
        assert call["model"] == "test-model"
        assert call["max_tokens"] == 2000

    async def test_prompt_uses_placeholder_when_no_memory_context(self, config):
        llm = ScriptedLLM(plan(sub("t1")))
        await TaskDecomposer(config, llm).decompose("task", AGENTS)
        assert "No relevant past context found." in llm.calls[0]["prompt"]

    @pytest.mark.parametrize("depth,expected", [(1, 4), (2, 8), (3, 10), (4, 10)])
    async def test_max_subtasks_limit_scales_with_depth_and_is_capped(self, depth, expected):
        cfg = ARIAConfig(max_decomposition_depth=depth)
        llm = ScriptedLLM(plan(sub("t1")))
        await TaskDecomposer(cfg, llm).decompose("task", AGENTS)
        assert f"Maximum {expected} subtasks total" in llm.calls[0]["prompt"]

    async def test_json_inside_markdown_fence_is_extracted(self, config):
        fenced = "Here is the plan:\n```json\n" + plan(sub("t1"), sub("t2", ["t1"])) + "\n```\nDone."
        genome = await TaskDecomposer(config, ScriptedLLM(fenced)).decompose("task", AGENTS)
        assert set(genome.nodes) == {"t1", "t2"}
        assert genome.execution_order == [["t1"], ["t2"]]

    async def test_missing_optional_fields_get_defaults(self, config):
        llm = ScriptedLLM(json.dumps({"subtasks": [{"description": "just this"}]}))
        genome = await TaskDecomposer(config, llm).decompose("task", AGENTS)

        node = genome.nodes["t1"]
        assert node.description == "just this"
        assert node.dependencies == []
        assert node.required_tools == []
        assert genome.complexity_score == pytest.approx(0.5)
        assert genome.estimated_total_seconds == pytest.approx(60.0)

    async def test_node_contract_derived_from_estimate_without_parent_contract(self, config):
        llm = ScriptedLLM(plan(sub("t1", estimated_seconds=25)))
        genome = await TaskDecomposer(config, llm).decompose("task", AGENTS)
        assert genome.nodes["t1"].contract.max_seconds == pytest.approx(50.0)

    async def test_node_contracts_are_children_of_the_parent_contract(self, config):
        parent = ExecutionContract(max_seconds=100, max_cost_usd=1.0)
        llm = ScriptedLLM(plan(sub("t1"), sub("t2")))
        genome = await TaskDecomposer(config, llm).decompose("task", AGENTS, contract=parent)

        for node in genome.nodes.values():
            assert node.contract is not parent
            assert node.contract.max_seconds <= 100 * 0.8 + 0.01
            assert node.contract.max_cost_usd == pytest.approx(0.8)

    async def test_decomposition_model_recorded_even_on_fallback(self, config):
        llm = ScriptedLLM("not json at all")
        genome = await TaskDecomposer(config, llm).decompose("task", AGENTS)
        assert genome.decomposition_model == "test-model"


# ─── Failure paths ────────────────────────────────────────────────────────────

class TestFallbackOnBadLLMOutput:

    def _assert_single_node_fallback(self, genome: TaskGenome, task: str):
        assert list(genome.nodes) == ["t1"]
        assert genome.nodes["t1"].description == task
        assert genome.execution_order == [["t1"]]
        assert genome.complexity_score == pytest.approx(0.3)

    async def test_llm_exception_falls_back_to_single_node(self, config):
        llm = ScriptedLLM(error=RuntimeError("API down"))
        genome = await TaskDecomposer(config, llm).decompose("do it", AGENTS)
        self._assert_single_node_fallback(genome, "do it")

    async def test_timeout_from_llm_falls_back(self, config):
        llm = ScriptedLLM(error=TimeoutError("slow"))
        genome = await TaskDecomposer(config, llm).decompose("do it", AGENTS)
        self._assert_single_node_fallback(genome, "do it")

    @pytest.mark.parametrize("bad", ["", "no braces here", "{broken json", "{'single': 'quotes'}"])
    async def test_unparseable_response_falls_back(self, config, bad):
        genome = await TaskDecomposer(config, ScriptedLLM(bad)).decompose("do it", AGENTS)
        self._assert_single_node_fallback(genome, "do it")

    async def test_empty_subtask_list_falls_back(self, config):
        genome = await TaskDecomposer(config, ScriptedLLM(plan())).decompose("do it", AGENTS)
        self._assert_single_node_fallback(genome, "do it")

    async def test_missing_subtasks_key_falls_back(self, config):
        llm = ScriptedLLM(json.dumps({"complexity_score": 0.9}))
        genome = await TaskDecomposer(config, llm).decompose("do it", AGENTS)
        self._assert_single_node_fallback(genome, "do it")

    async def test_non_numeric_complexity_falls_back(self, config):
        llm = ScriptedLLM(json.dumps({"complexity_score": "high", "subtasks": [sub("t1")]}))
        genome = await TaskDecomposer(config, llm).decompose("do it", AGENTS)
        self._assert_single_node_fallback(genome, "do it")

    async def test_malformed_subtask_entry_falls_back(self, config):
        llm = ScriptedLLM(json.dumps({"subtasks": ["a string, not an object"]}))
        genome = await TaskDecomposer(config, llm).decompose("do it", AGENTS)
        self._assert_single_node_fallback(genome, "do it")


    @pytest.mark.xfail(
        strict=True,
        reason="bug: decomposer.py _parse_response adds nodes to the genome one by one, so "
               "when a later subtask is malformed the single-node fallback keeps the earlier "
               "nodes as orphans that are in genome.nodes but not in execution_order",
    )
    async def test_fallback_after_partial_parse_leaves_no_orphan_nodes(self, config):
        llm = ScriptedLLM(json.dumps({"subtasks": [sub("a"), "malformed"]}))
        genome = await TaskDecomposer(config, llm).decompose("do it", AGENTS)
        assert list(genome.nodes) == ["t1"]


# ─── Topological layering ─────────────────────────────────────────────────────

def genome_of(**deps: list[str]) -> TaskGenome:
    genome = TaskGenome(original_task="x")
    for node_id, dependencies in deps.items():
        genome.add_node(TaskNode(node_id=node_id, description=node_id, dependencies=dependencies))
    return genome


class TestTopologicalLayers:

    @pytest.fixture
    def decomposer(self, config):
        return TaskDecomposer(config)

    def test_independent_nodes_share_a_layer(self, decomposer):
        layers = decomposer._topological_layers(genome_of(a=[], b=[], c=[]))
        assert len(layers) == 1
        assert sorted(layers[0]) == ["a", "b", "c"]

    def test_linear_chain_has_one_node_per_layer(self, decomposer):
        layers = decomposer._topological_layers(genome_of(a=[], b=["a"], c=["b"]))
        assert layers == [["a"], ["b"], ["c"]]

    def test_diamond_shape(self, decomposer):
        layers = decomposer._topological_layers(
            genome_of(a=[], b=["a"], c=["a"], d=["b", "c"])
        )
        assert layers[0] == ["a"]
        assert sorted(layers[1]) == ["b", "c"]
        assert layers[2] == ["d"]

    def test_every_dependency_is_in_an_earlier_layer(self, decomposer):
        genome = genome_of(a=[], b=["a"], c=["a", "b"], d=["c"], e=[], f=["e", "d"])
        layers = decomposer._topological_layers(genome)
        position = {nid: i for i, layer in enumerate(layers) for nid in layer}
        assert set(position) == set(genome.nodes)
        for node in genome.nodes.values():
            for dep in node.dependencies:
                assert position[dep] < position[node.node_id]

    def test_unknown_dependency_is_ignored(self, decomposer):
        layers = decomposer._topological_layers(genome_of(a=["ghost"], b=["a"]))
        assert layers == [["a"], ["b"]]

    def test_empty_genome_has_no_layers(self, decomposer):
        assert decomposer._topological_layers(TaskGenome()) == []

    def test_cycle_terminates_and_cyclic_nodes_are_left_out(self, decomposer):
        # Characterisation of current behaviour: Kahn's algorithm never reaches
        # nodes that sit on a cycle, so they are silently absent from the plan.
        genome = genome_of(a=[], b=["c"], c=["b"], d=["a"])
        layers = decomposer._topological_layers(genome)
        flat = [nid for layer in layers for nid in layer]
        assert layers[0] == ["a"]
        assert "d" in flat
        assert "b" not in flat and "c" not in flat

    async def test_cyclic_plan_from_llm_still_returns_a_genome(self, config):
        llm = ScriptedLLM(plan(sub("t1", ["t2"]), sub("t2", ["t1"])))
        genome = await TaskDecomposer(config, llm).decompose("task", AGENTS)
        assert set(genome.nodes) == {"t1", "t2"}


# ─── Helpers ──────────────────────────────────────────────────────────────────

class TestFormatAgentSummaries:

    def test_empty_list(self):
        assert TaskDecomposer._format_agent_summaries([]) == "No agents registered yet."

    def test_lists_each_agent_with_domains(self):
        text = TaskDecomposer._format_agent_summaries(AGENTS)
        lines = text.splitlines()
        assert len(lines) == 2
        assert "researcher" in lines[0] and "research, web" in lines[0]
        assert "writer" in lines[1] and "writing" in lines[1]

    def test_missing_domains_and_status_are_tolerated(self):
        text = TaskDecomposer._format_agent_summaries([{"agent_id": "solo"}])
        assert "solo" in text


class TestEstimateComplexity:

    @pytest.fixture
    def decomposer(self, config):
        return TaskDecomposer(config)

    def test_trivial_task_scores_zero(self, decomposer):
        assert decomposer.estimate_complexity("hello world") == 0.0

    def test_empty_task_scores_zero(self, decomposer):
        assert decomposer.estimate_complexity("") == 0.0

    def test_multi_step_task_with_tool_words(self, decomposer):
        # multi-step (then/summarize/and/compare >= 2) = 0.2, tool words (search, compare) = 0.2
        score = decomposer.estimate_complexity("search then summarize and compare")
        assert score == pytest.approx(0.4)

    def test_tool_word_bonus_is_capped(self, decomposer):
        score = decomposer.estimate_complexity("search browse email send create write generate")
        assert score == pytest.approx(0.3)

    def test_long_task_adds_length_bonus(self, decomposer):
        medium = decomposer.estimate_complexity(" ".join(["word"] * 25))
        long = decomposer.estimate_complexity(" ".join(["word"] * 60))
        assert medium == pytest.approx(0.15)
        assert long == pytest.approx(0.3)

    def test_question_words_add_bonus(self, decomposer):
        assert decomposer.estimate_complexity("why is the sky blue?") == pytest.approx(0.1)
        assert decomposer.estimate_complexity("the sky is blue?") == 0.0

    def test_score_is_case_insensitive_and_bounded(self, decomposer):
        text = ("Search THEN Analyze AND Compare, additionally research and write and "
                "generate a report? how " + "filler " * 60)
        score = decomposer.estimate_complexity(text)
        assert 0.0 < score <= 1.0
        assert score > decomposer.estimate_complexity("say hi")

    def test_harder_task_scores_higher_than_easier_one(self, decomposer):
        easy = decomposer.estimate_complexity("what time is it")
        hard = decomposer.estimate_complexity(
            "research competitors then analyze pricing and finally write and send a report"
        )
        assert hard > easy
