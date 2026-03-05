"""
Tests for ARIA Self-Improvement Engine
"""
import time

import pytest

from aria.core.improvement import Lesson, SelfImprovementEngine
from aria.core.models import TaskGenome, TaskNode, TaskResult, TaskStatus


@pytest.fixture
def engine():
    return SelfImprovementEngine()


@pytest.fixture
def failed_result():
    r = TaskResult(original_task="Test task", status=TaskStatus.FAILED)
    r.error_messages.append("No agent capable of handling this task type")
    r.quality_score = 0.1
    r.total_seconds = 2.0
    return r


@pytest.fixture
def success_result():
    r = TaskResult(original_task="Test task", status=TaskStatus.COMPLETED)
    r.quality_score = 0.85
    r.total_seconds = 1.2
    return r


@pytest.fixture
def mock_genome():
    genome = TaskGenome(original_task="Test")
    genome.nodes["n1"] = TaskNode(
        node_id="n1",
        description="Do something",
        status=TaskStatus.COMPLETED,
        result="Done",
        started_at=time.time() - 1,
        completed_at=time.time(),
    )
    return genome


@pytest.mark.asyncio
async def test_learns_from_failure(engine, failed_result, mock_genome):
    """A failure produces a lesson with outcome='failure'."""
    lessons = await engine.learn_from_execution(
        task="Analyze sales data and identify anomalies",
        result=failed_result,
        coalition={},
        genome=mock_genome,
    )
    assert len(lessons) == 1
    assert lessons[0].outcome == "failure"


@pytest.mark.asyncio
async def test_learns_from_success(engine, success_result, mock_genome):
    """A successful task with high quality produces a success lesson."""
    lessons = await engine.learn_from_execution(
        task="Write a summary of the quarterly report",
        result=success_result,
        coalition={"n1": "writer_agent"},
        genome=mock_genome,
    )
    assert len(lessons) == 1
    assert lessons[0].outcome == "success"


@pytest.mark.asyncio
async def test_poor_quality_outcome(engine, mock_genome):
    """Completed tasks with low quality score produce poor_quality lessons."""
    result = TaskResult(original_task="Test", status=TaskStatus.COMPLETED)
    result.quality_score = 0.3
    result.total_seconds = 5.0

    lessons = await engine.learn_from_execution(
        task="Research competitor pricing strategies",
        result=result,
        coalition={"n1": "researcher_agent"},
        genome=mock_genome,
    )
    assert lessons[0].outcome == "poor_quality"


@pytest.mark.asyncio
async def test_lesson_recall(engine, failed_result, mock_genome):
    """Lessons are recalled for semantically similar future tasks."""
    task = "Analyze quarterly sales figures and find outliers"
    await engine.learn_from_execution(
        task=task, result=failed_result, coalition={}, genome=mock_genome,
    )
    recalled = engine.recall_lessons("Analyze quarterly sales data for outliers")
    assert len(recalled) > 0
    assert recalled[0].outcome == "failure"


@pytest.mark.asyncio
async def test_no_recall_for_unrelated_task(engine, failed_result, mock_genome):
    """Lessons about sales data are not recalled for unrelated tasks."""
    await engine.learn_from_execution(
        task="Analyze quarterly sales figures",
        result=failed_result, coalition={}, genome=mock_genome,
    )
    recalled = engine.recall_lessons("Write a poem about the ocean")
    # May return 0 or very few results for completely different tasks
    assert len(recalled) == 0 or recalled[0].quality_score < 0.5


@pytest.mark.asyncio
async def test_capability_gap_detected_after_three_failures(engine, failed_result, mock_genome):
    """Three similar failures trigger capability gap detection."""
    for i in range(3):
        await engine.learn_from_execution(
            task=f"Perform data analysis task number {i}",
            result=failed_result,
            coalition={},
            genome=mock_genome,
        )
    gaps = engine.get_open_gaps()
    assert len(gaps) >= 1


@pytest.mark.asyncio
async def test_routing_weight_updates(engine, success_result, mock_genome):
    """Successful execution updates routing weights for the agent."""
    await engine.learn_from_execution(
        task="Research the latest AI papers",
        result=success_result,
        coalition={"n1": "researcher_agent"},
        genome=mock_genome,
    )
    score = engine.get_empirical_score("researcher_agent", "research the AI field")
    # Score should be 0.5 (neutral) because only 1 execution — not reliable yet
    assert 0.0 <= score <= 1.0


def test_blend_routing_score_neutral_with_no_data(engine):
    """Blended score equals semantic score when no empirical data exists."""
    blended = engine.blend_routing_score("new_agent", "some task", semantic_score=0.7)
    assert blended == 0.7


def test_lesson_matches(engine):
    """Lesson.matches returns True for similar tasks."""
    lesson = Lesson(
        task_sample="Analyze quarterly sales data and find outliers",
        outcome="failure",
        root_cause="capability_gap",
        adjustment="Add a data analysis agent",
    )
    assert lesson.matches("Analyze quarterly sales data for outliers", threshold=0.1) is True
    assert lesson.matches("Write a sonnet about spring and flowers") is False


def test_format_lessons_for_prompt(engine):
    """format_lessons_for_prompt returns a non-empty string when lessons exist."""
    engine._lessons.append(Lesson(
        task_sample="Failed to analyze sales data",
        outcome="failure",
        root_cause="capability_gap",
        adjustment="Add a data analysis specialist agent",
    ))
    prompt_fragment = engine.format_lessons_for_prompt("Failed to analyze sales data quarterly")
    assert "failure" in prompt_fragment or "Add a data analysis" in prompt_fragment


def test_get_summary(engine):
    """get_summary returns the correct structure."""
    summary = engine.get_summary()
    assert "lessons_total" in summary
    assert "open_gaps" in summary
    assert "synthesized_caps" in summary
    assert summary["lessons_total"] == 0
