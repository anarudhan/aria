"""
ARIA Agent Base Classes
-----------------------
The building blocks for all agents in an ARIA system.

Two ways to create an ARIA agent:

1. Subclass BaseAgent and implement the execute() method:

       class MyAgent(BaseAgent):
           async def execute(self, node, context, memory):
               return await my_tool(node.description)

2. Use the @aria_agent decorator to wrap any Python function:

       @aria_agent(
           capabilities=["search the web", "find information"],
           domains=["research"],
           tools=["tavily_search"]
       )
       async def web_search_agent(node, context, memory):
           return await tavily.search(node.description)

3. Use LLMAgent with a list of callable tools:

       agent = LLMAgent(
           agent_id="researcher",
           capabilities=["research", "summarize articles"],
           tools=[search_tool, fetch_url_tool],
           llm=anthropic_llm,
       )

The @aria_agent decorator is the "Universal Adapter" — it wraps any existing
function, LangChain tool, or third-party agent into the ARIA framework
with full trust ledger integration, contract enforcement, and memory access.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional

from ..core.models import (
    AgentMessage,
    CapabilityFingerprint,
    MessageType,
    TaskNode,
)
from ..core.spill import (
    ESCAPE_SENTENCE,
    READ_TOOL_NAME,
    UNTRUSTED_NOTICE,
    SpillError,
    SpillStore,
    get_default_store,
    neutralize_markers,
)

if TYPE_CHECKING:
    from ..core.memory import TemporalMemory

_log = logging.getLogger("aria.agents")

# ─── Base Agent ───────────────────────────────────────────────────────────────

def _cut(text: str, limit: int) -> str:
    """Shorten ``text`` to ``limit`` characters and say so, never silently."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [{len(text) - limit} more characters not shown]"


class BaseAgent(ABC):
    """
    Abstract base class for all ARIA agents.

    Every agent must implement execute(). Everything else — contract enforcement,
    trust ledger updates, memory integration, coalition messaging — is handled
    by the framework. Your agent just focuses on doing its job.
    """

    def __init__(
        self,
        agent_id: str,
        capabilities: list[str],
        domains: list[str],
        tools_available: Optional[list[str]] = None,
        input_types: Optional[list[str]] = None,
        output_types: Optional[list[str]] = None,
        max_concurrent_tasks: int = 3,
        preferred_task_size: str = "medium",
    ):
        self.agent_id = agent_id
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._outbox: asyncio.Queue = asyncio.Queue()  # Messages TO other agents

        # Build fingerprint
        self.fingerprint = CapabilityFingerprint(
            agent_id             = agent_id,
            capabilities         = capabilities,
            domains              = domains,
            tools_available      = tools_available or [],
            input_types          = input_types or ["text"],
            output_types         = output_types or ["text"],
            max_concurrent_tasks = max_concurrent_tasks,
            preferred_task_size  = preferred_task_size,
        )

    @abstractmethod
    async def execute(
        self,
        node: TaskNode,
        context: dict[str, Any],   # Results from completed dependency nodes
        memory: Optional[TemporalMemory],  # ARIA's memory (may be None)
    ) -> Any:
        """
        Execute the task described in the node.

        Args:
            node:    The TaskNode to execute (contains description, contract, etc.)
            context: Accumulated results from dependency nodes
            memory:  ARIA's memory system for reading past context

        Returns:
            Any value — string, dict, list, etc. ARIA will serialize it.

        Raise an exception to mark the node as FAILED.
        """
        pass

    async def on_message(self, message: AgentMessage) -> Optional[Any]:
        """
        Handle a peer-to-peer message from another agent in the coalition.
        Override this to participate in the Coalition Protocol.
        Default: ignore all messages.
        """
        return None

    async def send_message(
        self,
        msg_type: MessageType,
        recipient_id: Optional[str],
        genome_id: str,
        node_id: str,
        payload: dict,
        requires_response: bool = False,
    ) -> AgentMessage:
        """
        Send a message to another agent or broadcast to the coalition.
        Messages are queued — the framework delivers them asynchronously.
        """
        msg = AgentMessage(
            msg_type    = msg_type,
            sender_id   = self.agent_id,
            recipient_id = recipient_id,
            genome_id   = genome_id,
            node_id     = node_id,
            payload     = payload,
            requires_response = requires_response,
        )
        await self._outbox.put(msg)
        return msg

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.agent_id} domains={self.fingerprint.domains}>"


# ─── LLM Agent ────────────────────────────────────────────────────────────────

class LLMAgent(BaseAgent):
    """
    An agent that uses an LLM with a set of tools to complete tasks.

    This is the most common agent type. It runs a ReAct-style loop:
    1. Show the LLM the task + available tools + context
    2. LLM decides which tool to call (or produces a final answer)
    3. Execute the tool call
    4. Feed the result back to the LLM
    5. Repeat until the LLM produces a final answer or the contract expires

    The LLM is called through the plugin system — you swap the LLM by
    changing the plugin, not by changing this class.
    """

    SYSTEM_PROMPT = """You are {agent_id}, a specialized AI agent in the ARIA multi-agent system.

Your capabilities: {capabilities}
Your domains: {domains}

You have access to the following tools:
{tool_descriptions}

PAST CONTEXT (from memory):
{memory_context}

DEPENDENCY RESULTS (from previous subtasks):
{dependency_results}

EXECUTION CONTRACT:
- Max time remaining: {max_seconds:.0f} seconds
- Max tokens remaining: {max_tokens:,}
- Quality threshold: {quality_threshold:.0%}

TASK: {task_description}
Expected output: {expected_output}

Think step by step. Use tools when needed. Be concise and focused.
When you have enough information, give a direct final answer.
Prefix your final answer with FINAL_ANSWER: on its own line."""

    # Text longer than these is never cut silently. It is saved to the spill
    # store and the model gets a head and tail preview that says how many
    # characters were left out and how to read them (see aria/core/spill.py).
    # Text that fits is passed through unchanged.
    DEPENDENCY_PREVIEW_CHARS = 1000
    TOOL_RESULT_PREVIEW_CHARS = 1000

    def __init__(
        self,
        agent_id: str,
        capabilities: list[str],
        domains: list[str],
        tools: Optional[list[Callable]] = None,
        llm: Optional[Any] = None,       # LLM plugin instance (duck-typed: needs chat())
        tools_available: Optional[list[str]] = None,
        input_types: Optional[list[str]] = None,
        output_types: Optional[list[str]] = None,
        max_concurrent_tasks: int = 3,
        preferred_task_size: str = "medium",
        max_react_iterations: int = 6,
        spill_store: Optional[SpillStore] = None,
    ):
        tool_names = tools_available or [getattr(t, '__name__', str(t)) for t in (tools or [])]

        super().__init__(
            agent_id             = agent_id,
            capabilities         = capabilities,
            domains              = domains,
            tools_available      = tool_names,
            input_types          = input_types,
            output_types         = output_types,
            max_concurrent_tasks = max_concurrent_tasks,
            preferred_task_size  = preferred_task_size,
        )

        self._tools: dict[str, Callable] = {}
        for tool in (tools or []):
            name = getattr(tool, '__name__', str(tool))
            self._tools[name] = tool

        self._llm = llm
        self._max_react_iterations = max_react_iterations
        self._spill_store = spill_store   # None: use the shared default on first need
        self._warned_read_tool = False

    @property
    def spill_store(self) -> SpillStore:
        """Where oversized output is saved (created on first use if not given)."""
        if self._spill_store is None:
            self._spill_store = get_default_store()
        return self._spill_store

    def read_spilled(self, locator: str, offset: int = 0, length: int = 2000) -> str:
        """Read more of an output that was shortened. Pass the spill id from the omission note; offset and length count characters."""
        try:
            page = self.spill_store.read_page(locator, offset, length, owner=self.agent_id)
        except SpillError as exc:
            return f"Error: {exc}"
        note = f"[characters {page.offset} to {page.end} of {page.total_chars}"
        if page.has_more:
            note += (
                f"; to continue call {READ_TOOL_NAME}(locator=\"{locator}\", "
                f"offset={page.end}, length={self.spill_store.suggested_read_chars})"
            )
        else:
            note += "; end of text"
        # Lines of the content that imitate our own notes get a "> " prefix.
        shown = neutralize_markers(page.text)
        if shown != page.text:
            note += f"; {ESCAPE_SENTENCE}"
        return f"[{UNTRUSTED_NOTICE}]\n{shown}\n{note}]"

    def _owns_read_tool(self) -> bool:
        """True when the registered ``read_spilled`` is this agent's own method."""
        tool = self._tools.get(READ_TOOL_NAME)
        # Identity, never ``==``: a user object could define __eq__ to say yes.
        return (
            getattr(tool, "__self__", None) is self
            and getattr(tool, "__func__", None) is type(self).read_spilled
        )

    def _show(self, text: str, max_chars: int) -> str:
        """Return text as-is if it fits, else an honest preview (never a silent cut).

        The first time something is spilled the ``read_spilled`` tool is made
        available to the model. It is not added to the advertised fingerprint,
        and agents that never meet large output are unchanged.
        """
        result = self.spill_store.preview_or_spill(text, max_chars, owner=self.agent_id)
        if result.spilled:
            if READ_TOOL_NAME not in self._tools:
                self._tools[READ_TOOL_NAME] = self.read_spilled
            elif not self._owns_read_tool() and not self._warned_read_tool:
                # A user tool already has this name; the note would point at it.
                self._warned_read_tool = True
                message = (
                    f"agent '{self.agent_id}' has its own tool named '{READ_TOOL_NAME}', so "
                    f"the spill reader was not registered; omission notes point at that tool"
                )
                _log.warning(message)
                self.spill_store.warn(message)
        return result.text

    def _tool_text(self, result: Any, tool_name: str = "") -> str:
        # Pages read back are already bounded by the store's read limit. Shortening
        # them again would hand the model a new preview instead of what it asked for.
        if tool_name == READ_TOOL_NAME and self._owns_read_tool():
            return str(result)
        return self._show(str(result), self.TOOL_RESULT_PREVIEW_CHARS)

    def set_llm(self, llm: Any):
        """Attach an LLM plugin after construction (common pattern)."""
        self._llm = llm

    def add_tool(self, tool: Callable):
        """Add a tool to this agent at runtime."""
        name = getattr(tool, '__name__', str(tool))
        self._tools[name] = tool
        if name not in self.fingerprint.tools_available:
            self.fingerprint.tools_available.append(name)

    async def execute(
        self,
        node: TaskNode,
        context: dict[str, Any],
        memory: Optional[TemporalMemory],
    ) -> Any:
        """
        Execute using a ReAct loop with tool use.
        Contract enforcement is handled by the executor — we just run.
        """
        if self._llm is None:
            raise RuntimeError(f"Agent '{self.agent_id}' has no LLM attached. Call set_llm() first.")

        # Recall relevant memories
        memories = memory.recall(node.description, top_k=5) if memory else []
        memory_text = "\n".join(f"- {m.content}" for m in memories) or "No relevant past context."

        # Format dependency results
        dep_text = "None" if not context else "\n".join(
            f"[{nid}]: {self._show(str(result), self.DEPENDENCY_PREVIEW_CHARS)}"
            for nid, result in context.items()
        )

        # Build tool descriptions
        tool_desc = self._format_tool_descriptions()

        system_prompt = self.SYSTEM_PROMPT.format(
            agent_id           = self.agent_id,
            capabilities       = ", ".join(self.fingerprint.capabilities),
            domains            = ", ".join(self.fingerprint.domains),
            tool_descriptions  = tool_desc,
            memory_context     = memory_text,
            dependency_results = dep_text,
            max_seconds        = node.contract.remaining_seconds,
            max_tokens         = node.contract.remaining_tokens,
            quality_threshold  = node.contract.quality_threshold,
            task_description   = node.description,
            expected_output    = node.expected_output or "A helpful, accurate answer",
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": f"Please complete this task: {node.description}"})

        # ReAct loop
        for iteration in range(self._max_react_iterations):
            if node.contract.is_expired:
                raise TimeoutError(f"Contract expired after {iteration} ReAct iterations")

            response = await self._llm.chat(
                messages   = messages,
                tools      = self._get_tool_schemas(),
                max_tokens = min(2000, node.contract.remaining_tokens // 2),
            )

            # Update token consumption
            tokens_used = response.get("tokens_used", 0)
            node.contract.consume(tokens=tokens_used, cost=tokens_used * 0.000003)

            content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])

            # Check for final answer
            if "FINAL_ANSWER:" in content:
                final = content.split("FINAL_ANSWER:")[-1].strip()
                return final

            # Execute tool calls
            if tool_calls:
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
                tool_results = await self._execute_tools(tool_calls)
                messages.append({"role": "tool", "content": str(tool_results)})
            else:
                # LLM finished without explicit final answer marker
                messages.append({"role": "assistant", "content": content})
                if iteration >= 1:  # Give it at least one round
                    return content

        # Reached max iterations — return what we have
        last_content = messages[-1].get("content", "Task completed but no explicit final answer produced.")
        return last_content

    async def _execute_tools(self, tool_calls: list[dict]) -> list[dict]:
        """Execute a list of tool calls in parallel."""
        results = []
        tasks = []

        for call in tool_calls:
            tool_name = call.get("name", "")
            tool_args = call.get("arguments", {})

            if tool_name in self._tools:
                tool_fn = self._tools[tool_name]
                if asyncio.iscoroutinefunction(tool_fn):
                    tasks.append((tool_name, asyncio.create_task(tool_fn(**tool_args))))
                else:
                    try:
                        result = tool_fn(**tool_args)
                        results.append({"tool": tool_name, "result": self._tool_text(result, tool_name)})
                    except Exception as e:
                        results.append({"tool": tool_name, "error": str(e)})
            else:
                results.append({"tool": tool_name, "error": f"Tool '{tool_name}' not available"})

        # Await async tool tasks
        for tool_name, task in tasks:
            try:
                result = await task
                results.append({"tool": tool_name, "result": self._tool_text(result, tool_name)})
            except Exception as e:
                results.append({"tool": tool_name, "error": str(e)})

        return results

    def _format_tool_descriptions(self) -> str:
        if not self._tools:
            return "No tools available — rely on your training knowledge."
        lines = []
        for name, fn in self._tools.items():
            doc = inspect.getdoc(fn) or "No description available."
            sig = inspect.signature(fn)
            lines.append(f"  {name}{sig}: {_cut(doc, 200)}")
        return "\n".join(lines)

    def _get_tool_schemas(self) -> list[dict]:
        """Build tool schemas for LLM function calling."""
        schemas = []
        for name, fn in self._tools.items():
            doc = inspect.getdoc(fn) or ""
            sig = inspect.signature(fn)
            params = {}
            required = []
            for pname, param in sig.parameters.items():
                if pname == "self":
                    continue
                ptype = "string"
                if param.annotation != inspect.Parameter.empty:
                    ann = str(param.annotation)
                    if "int" in ann:
                        ptype = "integer"
                    elif "float" in ann:
                        ptype = "number"
                    elif "bool" in ann:
                        ptype = "boolean"
                    elif "list" in ann:
                        ptype = "array"
                    elif "dict" in ann:
                        ptype = "object"
                params[pname] = {"type": ptype}
                if param.default == inspect.Parameter.empty:
                    required.append(pname)
            schemas.append({
                "name": name,
                "description": _cut(doc, 300),
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": required,
                }
            })
        return schemas


# ─── Functional Agent Decorator ───────────────────────────────────────────────

def aria_agent(
    capabilities: list[str],
    domains: list[str],
    tools: Optional[list[str]] = None,
    input_types: Optional[list[str]] = None,
    output_types: Optional[list[str]] = None,
    max_concurrent: int = 3,
    agent_id: Optional[str] = None,
):
    """
    The Universal Adapter decorator — wrap any async function as an ARIA agent.

    This is the simplest way to create an ARIA agent. The function receives
    the same arguments as BaseAgent.execute() and should return the result.

    Example:

        @aria_agent(
            capabilities=["search the web for information", "find current news"],
            domains=["research", "web_browsing"],
            tools=["tavily_search"],
        )
        async def web_researcher(node, context, memory):
            query = node.description
            results = await tavily_search(query)
            return summarize(results)

        # Register with ARIA
        orchestrator.register_agent(web_researcher)
    """
    def decorator(func: Callable) -> Callable:
        _agent_id = agent_id or func.__name__
        doc = inspect.getdoc(func) or ""
        all_capabilities = capabilities + ([_cut(doc, 100)] if doc else [])

        class FunctionalAgent(BaseAgent):
            def __init__(self):
                super().__init__(
                    agent_id             = _agent_id,
                    capabilities         = all_capabilities,
                    domains              = domains,
                    tools_available      = tools or [],
                    input_types          = input_types or ["text"],
                    output_types         = output_types or ["text"],
                    max_concurrent_tasks = max_concurrent,
                )

            async def execute(self, node, context, memory):
                if asyncio.iscoroutinefunction(func):
                    return await func(node, context, memory)
                else:
                    return func(node, context, memory)

        # Attach the agent instance to the function for easy access
        # setattr: arbitrary callables are not typed to carry these marker attributes.
        agent_instance = FunctionalAgent()
        setattr(func, "_aria_agent", agent_instance)
        setattr(func, "_is_aria_agent", True)
        setattr(func, "_agent_id", _agent_id)

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        setattr(wrapper, "_aria_agent", agent_instance)
        setattr(wrapper, "_is_aria_agent", True)
        setattr(wrapper, "_agent_id", _agent_id)
        return wrapper

    return decorator
