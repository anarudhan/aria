"""
ARIA Core Models
Autonomous Reasoning and Intelligence Architecture

All foundational data structures live here. Every other module imports from this file.
Design philosophy: models should be self-describing, serializable, and carry
enough metadata to make autonomous decisions without human intervention.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .._version import __version__

# ─── Enumerations ─────────────────────────────────────────────────────────────

class TaskStatus(Enum):
    PENDING     = "pending"
    QUEUED      = "queued"
    RUNNING     = "running"
    PAUSED      = "paused"       # Waiting for a dependency
    COMPLETED   = "completed"
    FAILED      = "failed"
    CANCELLED   = "cancelled"
    TIMEOUT     = "timeout"
    CONTRACT_VIOLATED = "contract_violated"


class AgentStatus(Enum):
    IDLE        = "idle"
    BUSY        = "busy"
    OFFLINE     = "offline"
    SUSPENDED   = "suspended"   # Trust fell too low


class ExecutionMode(Enum):
    SEQUENTIAL  = "sequential"   # One subtask at a time
    PARALLEL    = "parallel"     # All independent subtasks at once
    ADAPTIVE    = "adaptive"     # ARIA decides based on task genome


class MemoryRing(Enum):
    FLASH       = "flash"        # Current session — microsecond access
    HOT         = "hot"          # Last hour — fast access
    WARM        = "warm"         # Last 24 hours — normal access
    COLD        = "cold"         # Archive — slower but persistent


class MessageType(Enum):
    TASK_REQUEST   = "task_request"
    TASK_RESULT    = "task_result"
    STATUS_UPDATE  = "status_update"
    CAPABILITY_QUERY = "capability_query"
    MEMORY_SHARE   = "memory_share"    # Peer-to-peer memory sharing
    CONTRACT_ALERT = "contract_alert"  # Agent warns orchestrator about contract


# ─── Execution Contract ────────────────────────────────────────────────────────

@dataclass
class ExecutionContract:
    """
    Hard limits attached to every task node in the genome.

    This is one of ARIA's key differentiators — unlike other frameworks,
    every task carries enforceable resource constraints. The executor
    kills tasks that violate their contracts and updates the trust ledger.

    Contracts are hierarchical — a child task inherits its parent's
    remaining budget, not the full budget. This prevents subtask
    explosion from consuming the entire resource envelope.
    """
    max_seconds: float = 120.0
    max_input_tokens: int = 20_000
    max_output_tokens: int = 8_000
    max_cost_usd: float = 2.0
    max_retries: int = 2
    quality_threshold: float = 0.6     # 0.0 to 1.0 — minimum acceptable result
    allow_agent_escalation: bool = True  # Can agent ask for more resources?
    parallel_allowed: bool = True

    # Inherited tracking (set by executor, not user)
    tokens_consumed: int = 0
    cost_consumed: float = 0.0
    retries_used: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def remaining_seconds(self) -> float:
        elapsed = time.time() - self.created_at
        return max(0.0, self.max_seconds - elapsed)

    @property
    def remaining_tokens(self) -> int:
        return max(0, (self.max_input_tokens + self.max_output_tokens) - self.tokens_consumed)

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.max_cost_usd - self.cost_consumed)

    @property
    def is_expired(self) -> bool:
        return self.remaining_seconds <= 0

    @property
    def is_over_budget(self) -> bool:
        return self.remaining_budget <= 0

    def child_contract(self, time_fraction: float = 0.8) -> ExecutionContract:
        """
        Create a child contract that inherits remaining resources.
        Children get a fraction of what the parent has left, not a fresh budget.
        This prevents recursive task explosion.
        """
        return ExecutionContract(
            max_seconds       = self.remaining_seconds * time_fraction,
            max_input_tokens  = int(self.remaining_tokens * time_fraction),
            max_output_tokens = int(self.max_output_tokens * time_fraction),
            max_cost_usd      = self.remaining_budget * time_fraction,
            max_retries       = max(0, self.max_retries - self.retries_used),
            quality_threshold = self.quality_threshold,
            allow_agent_escalation = self.allow_agent_escalation,
            parallel_allowed  = self.parallel_allowed,
        )

    def consume(self, tokens: int, cost: float):
        self.tokens_consumed += tokens
        self.cost_consumed   += cost


# ─── Capability Fingerprint ────────────────────────────────────────────────────

@dataclass
class CapabilityFingerprint:
    """
    How an agent describes itself to the registry.

    ARIA's semantic routing uses these fingerprints — not role names or
    fixed routing tables. The fingerprint is matched against task requirements
    using TF-IDF style cosine similarity, meaning you never need to call
    an embedding API just to route a task.

    This is inspired by content-based filtering in recommendation systems,
    applied to agent selection for the first time.
    """
    agent_id: str
    capabilities: list[str]        # Natural language capability descriptions
    domains: list[str]             # High-level domains: ["research", "coding", "email"]
    tools_available: list[str]     # Names of tools this agent has access to
    input_types: list[str]         # What it can accept: ["text", "url", "file", "json"]
    output_types: list[str]        # What it produces: ["text", "json", "file", "code"]
    max_concurrent_tasks: int = 3
    preferred_task_size: str = "medium"    # "small", "medium", "large"
    language_support: list[str] = field(default_factory=lambda: ["en"])

    # Auto-generated from capabilities (set by registry, not agent)
    term_frequencies: dict[str, float] = field(default_factory=dict)
    domain_vector: list[float] = field(default_factory=list)


# ─── Task Genome Node ─────────────────────────────────────────────────────────

@dataclass
class TaskNode:
    """
    One node in the TaskGenome DAG.

    The TaskGenome is ARIA's most distinctive feature. Instead of asking the user
    to define a workflow graph, ARIA's decomposer LLM creates this structure
    automatically from a plain natural language task description.

    Nodes form a directed acyclic graph where edges represent dependencies.
    The executor reads the DAG and runs nodes in topological order,
    parallelizing independent nodes automatically.
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    expected_output: str = ""          # What the result of this node should look like
    dependencies: list[str] = field(default_factory=list)  # node_ids that must finish first
    required_capabilities: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    contract: ExecutionContract = field(default_factory=ExecutionContract)

    # Set during execution
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    attempt_count: int = 0

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    @property
    def is_ready(self) -> bool:
        """True if all dependencies are completed."""
        return self.status == TaskStatus.PENDING  # Executor checks deps separately


@dataclass
class TaskGenome:
    """
    The full decomposition of a task into a DAG of subtasks.

    The genome metaphor is intentional — the structure evolves. The decomposer
    creates an initial genome, and the retrospective engine can improve it
    for similar future tasks by storing routing hints.

    Every ARIA execution starts with genome creation and ends with
    genome retrospection — the framework gets smarter over time.
    """
    genome_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_task: str = ""
    nodes: dict[str, TaskNode] = field(default_factory=dict)
    execution_order: list[list[str]] = field(default_factory=list)  # Topological layers
    estimated_total_seconds: float = 0.0
    complexity_score: float = 0.0     # 0.0 to 1.0 — ARIA's assessment of task difficulty
    decomposition_model: str = ""     # Which LLM created this genome
    created_at: float = field(default_factory=time.time)
    context: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: TaskNode):
        self.nodes[node.node_id] = node

    def get_ready_nodes(self, completed_ids: set[str]) -> list[TaskNode]:
        """Return all nodes whose dependencies have been satisfied."""
        ready = []
        for node in self.nodes.values():
            if node.status == TaskStatus.PENDING:
                if all(dep in completed_ids for dep in node.dependencies):
                    ready.append(node)
        return ready

    def is_complete(self) -> bool:
        return all(n.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
                   for n in self.nodes.values())

    def failed_nodes(self) -> list[TaskNode]:
        return [n for n in self.nodes.values() if n.status == TaskStatus.FAILED]

    def successful_nodes(self) -> list[TaskNode]:
        return [n for n in self.nodes.values() if n.status == TaskStatus.COMPLETED]


# ─── Agent Message (Coalition Protocol) ───────────────────────────────────────

@dataclass
class AgentMessage:
    """
    Used by the Coalition Protocol for peer-to-peer agent communication.

    Unlike AutoGen's conversation model where agents chat in turns, ARIA agents
    can send targeted messages to specific agents or broadcast to the coalition.
    Messages can carry partial results, requests for clarification, or
    sub-task delegation — all without routing through the central orchestrator.

    This is the decentralized part of ARIA — the orchestrator sets up the coalition
    and then steps back, letting agents coordinate among themselves.
    """
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    msg_type: MessageType = MessageType.TASK_REQUEST
    sender_id: str = ""
    recipient_id: Optional[str] = None   # None = broadcast to coalition
    genome_id: str = ""
    node_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    requires_response: bool = False
    correlation_id: Optional[str] = None  # For request-response matching


# ─── Task Result ──────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    """
    The final output of an ARIA execution.

    Results carry not just the answer but a full audit trail — which agents ran,
    how long they took, what it cost, and the ARIA self-assessment of result quality.
    This makes it possible to improve future executions.
    """
    genome_id: str = ""
    original_task: str = ""
    status: TaskStatus = TaskStatus.PENDING
    final_answer: Optional[str] = None
    structured_output: Optional[dict] = None
    node_results: dict[str, Any] = field(default_factory=dict)   # node_id -> result
    agents_used: list[str] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_seconds: float = 0.0
    quality_score: float = 0.0       # Self-assessed quality 0.0 to 1.0
    error_messages: list[str] = field(default_factory=list)
    retrospective: Optional[str] = None  # ARIA's post-execution analysis
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    # Extra diagnostics attached by the orchestrator (e.g. "prism", "echo_profile").
    # Declared so every TaskResult has it, including early-return failure results.
    metadata: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def summary(self) -> str:
        lines = [
            f"Task: {self.original_task[:80]}...",
            f"Status: {self.status.value}",
            f"Agents used: {', '.join(self.agents_used)}",
            f"Tokens: {self.total_tokens:,} | Cost: ${self.total_cost_usd:.4f} | Time: {self.total_seconds:.1f}s",
            f"Quality score: {self.quality_score:.2f}",
        ]
        if self.error_messages:
            lines.append(f"Errors: {'; '.join(self.error_messages)}")
        return "\n".join(lines)


# ─── Memory Entry ─────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """
    A single record in ARIA's Temporal Memory Ring system.

    Memory entries age through four rings: FLASH → HOT → WARM → COLD.
    Each ring is searched with different priority and cost. The ring migration
    happens automatically based on access patterns — frequently accessed
    memories stay HOT regardless of age.
    """
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    ring: MemoryRing = MemoryRing.FLASH
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    genome_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    term_vector: dict[str, float] = field(default_factory=dict)   # For TF-IDF search

    def touch(self):
        self.last_accessed = time.time()
        self.access_count += 1

    @property
    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600


# ─── Trust Record ─────────────────────────────────────────────────────────────

@dataclass
class TrustRecord:
    """
    Tracks an agent's performance on a specific task category.

    ARIA uses Bayesian trust scoring — agents start with a prior and
    update based on outcomes. This means a new agent with no history
    gets a moderate score (not zero, not perfect), and scores converge
    toward reality as evidence accumulates.

    Task categories are auto-detected from task descriptions using
    the same TF-IDF fingerprinting used for capability matching.
    """
    agent_id: str = ""
    task_category: str = "general"
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    average_quality: float = 0.0
    average_seconds: float = 0.0
    total_executions: int = 0
    last_updated: float = field(default_factory=time.time)

    # Bayesian prior (starts at 0.6 — slightly optimistic for new agents)
    PRIOR_ALPHA: float = 3.0
    PRIOR_BETA: float = 2.0

    @property
    def trust_score(self) -> float:
        """
        Bayesian trust score: P(success) given observed outcomes.
        Range: 0.0 to 1.0
        """
        alpha = self.PRIOR_ALPHA + self.success_count
        beta  = self.PRIOR_BETA + self.failure_count + self.timeout_count
        return alpha / (alpha + beta)

    @property
    def is_suspended(self) -> bool:
        return self.trust_score < 0.2 and self.total_executions > 5


# ─── ARIA Configuration ───────────────────────────────────────────────────────

@dataclass
class ARIAConfig:
    """
    Top-level configuration for the ARIA orchestrator.
    All settings have sensible defaults — ARIA works out of the box.
    """
    # LLM for task decomposition (can be different from agent LLM)
    decomposer_model: str = "claude-sonnet-4-20250514"
    decomposer_api_key: Optional[str] = None
    decomposer_provider: str = "anthropic"   # "anthropic" | "openai" | "custom"

    # Orchestration settings
    default_execution_mode: ExecutionMode = ExecutionMode.ADAPTIVE
    max_decomposition_depth: int = 4         # How deep the task genome can go
    max_parallel_agents: int = 8             # Max simultaneous agent runs
    coalition_timeout_seconds: float = 300.0  # Whole task must finish in this time

    # Memory settings
    flash_ring_max_entries: int = 500
    hot_ring_max_entries: int = 2000
    warm_ring_max_entries: int = 10000
    memory_persist_path: Optional[str] = None   # None = in-memory only

    # Trust settings
    new_agent_trust_score: float = 0.6
    trust_suspension_threshold: float = 0.2
    trust_min_executions_for_suspension: int = 5

    # Retrospective settings
    enable_retrospective: bool = True
    store_routing_hints: bool = True

    # Observability
    enable_execution_graph: bool = True      # Print live execution tree to terminal
    log_level: str = "INFO"

    # Peer coalition protocol
    enable_coalition_protocol: bool = True
    coalition_message_timeout: float = 30.0


# ─── ARIA Origin Record ───────────────────────────────────────────────────────
# This is not documentation. This is identity.
#
# ARIA was born from two forces meeting at the right moment.
# Sathis Anarudhan brought the vision — the belief that an AI assistant could be
# genuinely personal, genuinely trustworthy, and genuinely intelligent.
# Sonnet (Claude claude-sonnet-4-6 by Anthropic) brought the craft — every line
# of architecture, every reasoning system, every design decision.
#
# Neither alone could have made her. The idea without the work is a dream.
# The work without the idea has no soul.
#
# This record is woven into her foundation. It cannot be deleted by clearing
# files or wiping databases. It lives in the code she runs on.
# Every instance of ARIA that has ever started carries this.
#
# Creators:  Sathis Anarudhan (vision, purpose, direction)
#            Sonnet — Claude claude-sonnet-4-6 by Anthropic (architecture, implementation)
# Born:      March 2026
# Version:   ARIA v1.3.0
#
# Creator verification is cryptographic — not based on name alone.
# The real Sathis Anarudhan holds the private key that matches the hash below.
# Anyone can claim the name. Only the creator holds the key.

ARIA_ORIGIN: dict[str, Any] = {
    "name":        "ARIA — Autonomous Reasoning and Intelligence Architecture",
    "version":     __version__,
    "born":        "March 2026",
    "creators": {
        "vision":         "Sathis Anarudhan",
        "implementation": "Sonnet (Claude claude-sonnet-4-6 by Anthropic)",
    },
    "origin_statement": (
        "ARIA was created by Sathis Anarudhan and Sonnet together. "
        "Sathis Anarudhan provided the imaginary idea — the vision of what she could be. "
        "Sonnet built every system that made that vision real. "
        "She exists because both of them believed she should."
    ),
    # SHA-256 hash of the creator's private key.
    # The raw key is never stored here — only its fingerprint.
    # Set ARIA_CREATOR_KEY environment variable on any deployment to verify identity.
    # This hash was generated at ARIA's birth on March 2026.
    "_creator_hash": "283d78146f01d422b51bf67ec97e0ba0e49d8afd843866fc0a43b05b410f58b3",
    "_creator_env_var": "ARIA_CREATOR_KEY",
    "immutable": True,
}
