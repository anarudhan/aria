"""Regression tests for TaskResult.metadata (issue #20).

Before the fix, metadata was attached to the result only on the success path,
so results from early failures had no such attribute and readers got an
AttributeError.
"""
import pytest

from aria import ARIAConfig, ARIAOrchestrator, aria_agent
from aria.core.models import TaskResult, TaskStatus
from aria.permissions.engine import Capability


class MockLLM:
    async def complete(self, prompt: str, **kwargs) -> str:
        return "ok"

    async def chat(self, messages, **kwargs) -> str:
        return "ok"


@aria_agent(capabilities=["general"], domains=["general"])
async def general_agent(node, context, memory):
    return "done"


def test_new_result_has_empty_metadata():
    assert TaskResult().metadata == {}


def test_metadata_is_not_shared_between_results():
    first, second = TaskResult(), TaskResult()
    first.metadata["key"] = "value"
    assert second.metadata == {}


def test_metadata_does_not_affect_repr_or_equality():
    plain, annotated = TaskResult(), TaskResult()
    annotated.metadata["prism"] = {"confidence": 0.9}
    assert "prism" not in repr(annotated)
    assert plain.original_task == annotated.original_task
    assert plain.metadata != annotated.metadata
    assert plain.status == annotated.status


@pytest.mark.asyncio
async def test_permission_denied_result_still_has_metadata():
    aria = ARIAOrchestrator(
        config=ARIAConfig(enable_retrospective=False),
        llm_plugin=MockLLM(),
    )
    aria.register_agents([general_agent])

    result = await aria.run(
        task="Send an email to the team",
        user_id="nobody",
        required_capabilities=[Capability.EMAIL_SEND],
    )

    assert result.status == TaskStatus.FAILED
    assert result.error_messages
    assert result.metadata == {}
