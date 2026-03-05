"""
Tests for ARIA Orchestrator
"""
import pytest

from aria import ARIAConfig, ARIAOrchestrator, Capability, ExecutionContract, aria_agent
from aria.core.models import TaskStatus


class MockLLM:
    async def complete(self, prompt: str, **kwargs) -> str:
        if "decompose" in prompt.lower():
            return '{"nodes": [{"node_id": "n1", "description": "Execute the task", "required_capabilities": ["general"], "dependencies": []}]}'
        return "Task completed successfully."
    async def chat(self, messages, **kwargs) -> str:
        return await self.complete(messages[-1].get("content", "") if messages else "")


@aria_agent(capabilities=["general", "testing"], domains=["general"])
async def general_agent(node, context, memory):
    return f"Completed: {node.description[:50]}"


@aria_agent(capabilities=["research", "information retrieval"], domains=["research"])
async def research_agent(node, context, memory):
    return f"Research done: {node.description[:50]}"


@pytest.fixture
def aria():
    instance = ARIAOrchestrator(
        config=ARIAConfig(enable_retrospective=False),
        llm_plugin=MockLLM(),
    )
    instance.register_agents([general_agent, research_agent])
    instance.permissions.grant("test_user", Capability.MEMORY_WRITE, hours=1)
    return instance


@pytest.mark.asyncio
async def test_basic_task_completes(aria):
    result = await aria.run(
        task="Write a brief summary of today's agenda",
        user_id="test_user",
        contract=ExecutionContract(max_seconds=10),
    )
    assert result.status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_result_has_answer(aria):
    result = await aria.run(
        task="List three benefits of exercise",
        user_id="test_user",
        contract=ExecutionContract(max_seconds=10),
    )
    assert result.final_answer is not None
    assert len(result.final_answer) > 0


@pytest.mark.asyncio
async def test_quality_score_range(aria):
    result = await aria.run(
        task="Summarize the key points from a meeting",
        user_id="test_user",
        contract=ExecutionContract(max_seconds=10),
    )
    assert 0.0 <= result.quality_score <= 1.0


def test_agent_registration(aria):
    agents = aria.registry.list_agents()
    names = [a["agent_id"] for a in agents]
    assert any("general" in n for n in names)
    assert any("research" in n for n in names)


def test_permission_check(aria):
    aria.permissions.grant("new_user", Capability.FILE_READ, hours=1)
    assert aria.permissions.check("new_user", Capability.FILE_READ) is True
    assert aria.permissions.check("new_user", Capability.EMAIL_SEND) is False


def test_health_report_structure(aria):
    health = aria.health_report()
    assert "grade" in health
    assert "is_healthy" in health
    assert "tasks_completed" in health
    assert "error_rate" in health
    assert health["grade"] in ("A", "B", "C", "D", "F")


def test_what_i_am_returns_string(aria):
    description = aria.what_i_am()
    assert isinstance(description, str)
    assert len(description) > 50
    assert "ARIA" in description


def test_integrity_check_returns_list(aria):
    result = aria.integrity_check()
    assert isinstance(result, list)
