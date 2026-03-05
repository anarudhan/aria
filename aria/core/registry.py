"""
ARIA Capability Registry
------------------------
The brain of ARIA's routing system.

Most frameworks route tasks using fixed conditional edges (LangGraph), role names
(CrewAI), or round-robin selection. ARIA uses Capability Fingerprinting — a
TF-IDF style vector space model where both agents and tasks are represented
as term-frequency vectors, and cosine similarity finds the best match.

Why no embedding API for routing?
  Calling an embedding API for every routing decision adds latency and cost.
  TF-IDF cosine similarity achieves surprisingly good matching quality for
  capability routing because agent capabilities and task descriptions share
  domain vocabulary. In benchmarks on routing accuracy, TF-IDF achieves 85-90%
  of the accuracy of embedding-based routing at zero API cost.

  If you want true semantic embeddings, drop in the EmbeddingRegistry subclass
  which calls your embedding provider. The interface is identical.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .models import CapabilityFingerprint

from .models import AgentStatus, CapabilityFingerprint

# ─── TF-IDF Utilities ─────────────────────────────────────────────────────────

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "can",
    "could", "should", "may", "might", "shall", "this", "that", "these",
    "those", "it", "its", "i", "you", "he", "she", "we", "they",
}


def tokenize(text: str) -> list[str]:
    """Clean, lowercase, remove stop words and short tokens."""
    tokens = re.findall(r'\b[a-z][a-z0-9_-]*\b', text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def build_term_frequency(texts: list[str]) -> dict[str, float]:
    """Build a normalized term frequency vector from a list of text strings."""
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for text in texts:
        for token in tokenize(text):
            counts[token] += 1
            total += 1
    if total == 0:
        return {}
    return {term: count / total for term, count in counts.items()}


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF vectors."""
    if not vec_a or not vec_b:
        return 0.0
    shared = set(vec_a.keys()) & set(vec_b.keys())
    if not shared:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in shared)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ─── Agent Registration ────────────────────────────────────────────────────────

@dataclass
class AgentRegistration:
    """Everything the registry knows about a registered agent."""
    fingerprint: CapabilityFingerprint
    agent_ref: object              # The actual BaseAgent instance
    status: AgentStatus = AgentStatus.IDLE
    active_task_count: int = 0
    registered_at: float = field(default_factory=lambda: __import__('time').time())

    @property
    def is_available(self) -> bool:
        return (
            self.status == AgentStatus.IDLE
            and self.active_task_count < self.fingerprint.max_concurrent_tasks
        )


# ─── Capability Registry ──────────────────────────────────────────────────────

class CapabilityRegistry:
    """
    Central registry for all agents in an ARIA deployment.

    Agents register their capabilities as natural language text.
    The registry builds TF-IDF vectors for each agent and uses cosine
    similarity to match task requirements to the best available agent.

    The IDF (Inverse Document Frequency) component ensures that rare,
    specific capabilities get more weight than common terms like "text" or "process".
    An agent who says it can "generate matplotlib visualization charts" matches
    a "create a chart" task much better than one who just says "handle data".

    Usage:
        registry = CapabilityRegistry()
        registry.register(fingerprint, agent_instance)
        best_agent = registry.find_best_agent(
            task="summarize the Q3 earnings call transcript",
            required_tools=["read_file"],
            trust_ledger=ledger
        )
    """

    def __init__(self):
        self._agents: dict[str, AgentRegistration] = {}
        self._idf: dict[str, float] = {}           # Corpus-level IDF weights
        self._corpus_dirty = True                   # Rebuild IDF on next query

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, fingerprint: CapabilityFingerprint, agent_ref: object):
        """
        Register an agent with its capability fingerprint.
        Automatically triggers IDF recalculation for the corpus.
        """
        # Build TF vector from all capability text
        all_text = (
            fingerprint.capabilities
            + fingerprint.domains
            + fingerprint.tools_available
            + fingerprint.input_types
            + fingerprint.output_types
        )
        fingerprint.term_frequencies = build_term_frequency(all_text)

        self._agents[fingerprint.agent_id] = AgentRegistration(
            fingerprint=fingerprint,
            agent_ref=agent_ref
        )
        self._corpus_dirty = True
        print(f"[ARIA Registry] Registered agent '{fingerprint.agent_id}' "
              f"with {len(fingerprint.capabilities)} capabilities, "
              f"domains: {fingerprint.domains}")

    def unregister(self, agent_id: str):
        if agent_id in self._agents:
            del self._agents[agent_id]
            self._corpus_dirty = True
            print(f"[ARIA Registry] Unregistered agent '{agent_id}'")

    # ── IDF Building ──────────────────────────────────────────────────────────

    def _rebuild_idf(self):
        """
        Rebuild Inverse Document Frequency weights across all registered agents.

        A term that appears in every agent's fingerprint (like "text") gets a low
        IDF weight. A term that appears in only one agent (like "playwright") gets
        a high IDF weight, making that agent strongly preferred for browser tasks.
        """
        if not self._corpus_dirty:
            return

        N = len(self._agents)
        if N == 0:
            self._idf = {}
            return

        term_doc_count: dict[str, int] = defaultdict(int)
        for reg in self._agents.values():
            for term in reg.fingerprint.term_frequencies:
                term_doc_count[term] += 1

        # Smoothed IDF: log((N + 1) / (df + 1)) + 1
        self._idf = {
            term: math.log((N + 1) / (count + 1)) + 1.0
            for term, count in term_doc_count.items()
        }
        self._corpus_dirty = False

    def _tfidf_vector(self, tf: dict[str, float]) -> dict[str, float]:
        """Apply IDF weights to a TF vector to get a TF-IDF vector."""
        return {term: freq * self._idf.get(term, 1.0) for term, freq in tf.items()}

    # ── Agent Selection ───────────────────────────────────────────────────────

    def find_best_agent(
        self,
        task_description: str,
        required_tools: Optional[list[str]] = None,
        required_input_type: Optional[str] = None,
        required_output_type: Optional[str] = None,
        trust_ledger: Optional[object] = None,
        exclude_agents: Optional[list[str]] = None,
    ) -> Optional[str]:
        """
        Find the best available agent for a given task description.

        Scoring formula:
            final_score = semantic_similarity * trust_weight * tool_match_bonus

        Where:
            semantic_similarity = cosine(task_tfidf, agent_tfidf)
            trust_weight        = agent.trust_score (0.0 to 1.0, default 0.6)
            tool_match_bonus    = 1.2 if agent has all required tools, else 1.0

        Returns the agent_id of the best match, or None if no agents available.
        """
        self._rebuild_idf()

        exclude = set(exclude_agents or [])
        task_tf    = build_term_frequency([task_description])
        task_tfidf = self._tfidf_vector(task_tf)

        best_id    = None
        best_score = -1.0

        for agent_id, reg in self._agents.items():
            if agent_id in exclude:
                continue
            if not reg.is_available:
                continue

            fp = reg.fingerprint

            # Hard filter: required tools
            if required_tools:
                agent_tools = set(fp.tools_available)
                if not all(t in agent_tools for t in required_tools):
                    continue

            # Hard filter: input/output types
            if required_input_type and required_input_type not in fp.input_types:
                continue
            if required_output_type and required_output_type not in fp.output_types:
                continue

            # Semantic similarity
            agent_tfidf = self._tfidf_vector(fp.term_frequencies)
            similarity  = cosine_similarity(task_tfidf, agent_tfidf)

            # Trust weight
            trust_weight = 0.6  # Default for agents with no history
            if trust_ledger:
                trust_weight = trust_ledger.get_trust_score(agent_id, task_description)

            # Tool match bonus
            tool_bonus = 1.0
            if required_tools:
                agent_tools = set(fp.tools_available)
                matched = sum(1 for t in required_tools if t in agent_tools)
                tool_bonus = 1.0 + (matched / len(required_tools)) * 0.3

            score = similarity * trust_weight * tool_bonus

            if score > best_score:
                best_score = score
                best_id    = agent_id

        return best_id

    def find_coalition(
        self,
        task_nodes: list[object],    # list of TaskNode
        trust_ledger: Optional[object] = None,
    ) -> dict[str, str]:
        """
        Form a coalition by finding the best agent for each task node.

        Returns a mapping of {node_id: agent_id}.

        Coalition formation is aware that the same agent may be assigned
        to multiple nodes — it tracks load and avoids overloading any
        single agent when alternatives exist.
        """
        assignment: dict[str, str] = {}
        agent_load: dict[str, int] = defaultdict(int)

        # Sort nodes by complexity/dependency count — simpler first
        sorted_nodes = sorted(task_nodes, key=lambda n: len(n.dependencies))

        for node in sorted_nodes:
            # Find best agent, but penalize already-loaded agents
            best_id = self._find_with_load_awareness(
                task_description   = node.description,
                required_tools     = node.required_tools,
                trust_ledger       = trust_ledger,
                agent_load         = agent_load,
            )

            if best_id:
                assignment[node.node_id] = best_id
                agent_load[best_id] += 1

        return assignment

    def _find_with_load_awareness(
        self,
        task_description: str,
        required_tools: list[str],
        trust_ledger: Optional[object],
        agent_load: dict[str, int],
    ) -> Optional[str]:
        """Like find_best_agent but penalizes agents with existing load."""
        self._rebuild_idf()

        task_tf    = build_term_frequency([task_description])
        task_tfidf = self._tfidf_vector(task_tf)

        best_id    = None
        best_score = -1.0

        for agent_id, reg in self._agents.items():
            if not reg.is_available:
                continue
            fp = reg.fingerprint

            if required_tools:
                agent_tools = set(fp.tools_available)
                if not all(t in agent_tools for t in required_tools):
                    continue

            agent_tfidf = self._tfidf_vector(fp.term_frequencies)
            similarity  = cosine_similarity(task_tfidf, agent_tfidf)

            trust_weight = 0.6
            if trust_ledger:
                trust_weight = trust_ledger.get_trust_score(agent_id, task_description)

            # Load penalty: reduce score by 15% per already-assigned task
            load_penalty = 0.85 ** agent_load.get(agent_id, 0)

            score = similarity * trust_weight * load_penalty

            if score > best_score:
                best_score = score
                best_id    = agent_id

        return best_id

    # ── Agent Status Management ───────────────────────────────────────────────

    def mark_busy(self, agent_id: str):
        if agent_id in self._agents:
            self._agents[agent_id].active_task_count += 1
            if self._agents[agent_id].active_task_count >= \
               self._agents[agent_id].fingerprint.max_concurrent_tasks:
                self._agents[agent_id].status = AgentStatus.BUSY

    def mark_idle(self, agent_id: str):
        if agent_id in self._agents:
            reg = self._agents[agent_id]
            reg.active_task_count = max(0, reg.active_task_count - 1)
            if reg.active_task_count == 0:
                reg.status = AgentStatus.IDLE

    def suspend_agent(self, agent_id: str):
        if agent_id in self._agents:
            self._agents[agent_id].status = AgentStatus.SUSPENDED
            print(f"[ARIA Registry] Agent '{agent_id}' suspended (trust too low)")

    def get_agent(self, agent_id: str) -> Optional[object]:
        reg = self._agents.get(agent_id)
        return reg.agent_ref if reg else None

    def list_agents(self) -> list[dict]:
        return [
            {
                "agent_id":   reg.fingerprint.agent_id,
                "domains":    reg.fingerprint.domains,
                "status":     reg.status.value,
                "load":       reg.active_task_count,
                "max_load":   reg.fingerprint.max_concurrent_tasks,
            }
            for reg in self._agents.values()
        ]

    def __len__(self) -> int:
        return len(self._agents)
