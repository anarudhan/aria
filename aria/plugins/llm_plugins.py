"""
ARIA Anthropic LLM Plugin
--------------------------
Connects ARIA to Anthropic's Claude API.

The plugin is the only place in ARIA where a specific LLM provider is mentioned.
Swap this plugin for the OpenAI plugin and ARIA switches to GPT-4o.
Write your own plugin to use Ollama, Mistral, or any other provider.

All plugins must implement:
    async complete(prompt, max_tokens, model) -> str
    async chat(messages, tools, max_tokens)   -> dict
"""

from __future__ import annotations

import json
import os
from typing import Any, Awaitable, Callable, Optional, TypeVar

from .retry import AttemptRecord, EmptyResponseError, RetryPolicy, call_with_retry

_T = TypeVar("_T")


def _anthropic_refused(response: Any) -> bool:
    """True when the model declined (stop_reason "refusal"). That answer is final: not retried."""
    return getattr(response, "stop_reason", None) == "refusal"


def _openai_declined(choice: Any) -> bool:
    """True when the model refused or the content filter blocked the answer. Not retried."""
    message = getattr(choice, "message", None)
    return bool(getattr(message, "refusal", None)) or (
        getattr(choice, "finish_reason", None) == "content_filter"
    )


class _RetryMixin:
    """
    Shared retry behaviour for LLM plugins.

    ``retry_policy`` holds the backoff settings, the bounded list of attempt records and
    the counters. Pass your own RetryPolicy (for example RetryPolicy(max_retries=0) to turn
    retries off) or leave it out for the defaults. ``on_attempt`` is called with every
    AttemptRecord so the audit trail can listen.
    """

    retry_policy: RetryPolicy
    on_attempt: Optional[Callable[[AttemptRecord], Any]]

    def _init_retry(
        self,
        retry_policy: Optional[RetryPolicy],
        on_attempt: Optional[Callable[[AttemptRecord], Any]],
    ) -> None:
        self.retry_policy = retry_policy if retry_policy is not None else RetryPolicy()
        self.on_attempt = on_attempt

    async def _with_retry(self, operation: str, fn: Callable[[], Awaitable[_T]]) -> _T:
        return await call_with_retry(
            fn, self.retry_policy, on_attempt=self.on_attempt, operation=operation,
        )

    def retry_stats(self) -> dict[str, Any]:
        """Counters: calls, attempts, retries, failures by error code, last error."""
        return self.retry_policy.stats()

    def retry_history(self) -> list[AttemptRecord]:
        """The most recent attempt records, oldest first (bounded)."""
        return list(self.retry_policy.history)

    def describe(self) -> dict[str, Any]:
        """Retry settings, counters and recent attempts, for the health report."""
        return {"plugin": type(self).__name__, "model": getattr(self, "model", None),
                "retry": self.retry_policy.describe()}


class AnthropicPlugin(_RetryMixin):
    """
    Anthropic Claude plugin for ARIA.

    Usage:
        plugin = AnthropicPlugin(api_key="sk-ant-...", model="claude-sonnet-4-20250514")
        aria = ARIAOrchestrator(config=ARIAConfig(), llm_plugin=plugin)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.3,
        retry_policy: Optional[RetryPolicy] = None,
        on_attempt: Optional[Callable[[AttemptRecord], Any]] = None,
    ):
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self._api_key    = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._init_retry(retry_policy, on_attempt)

        if not self._api_key:
            raise ValueError(
                "Anthropic API key not provided. "
                "Pass api_key= or set ANTHROPIC_API_KEY environment variable."
            )

        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required. "
                "Install it with: pip install anthropic"
            )

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 2000,
        model: Optional[str] = None,
    ) -> str:
        """
        Single-turn text completion. Used by the decomposer.

        Failed calls are retried (see aria/plugins/retry.py). When the provider keeps
        returning no content blocks, or the retries run out, LLMError is raised.
        """
        async def attempt() -> str:
            response = await self._client.messages.create(
                model      = model or self.model,
                max_tokens = max_tokens,
                messages   = [{"role": "user", "content": prompt}],
            )
            if not response.content:
                if _anthropic_refused(response):
                    return ""   # a refusal is final; retrying it only costs money
                raise EmptyResponseError("Anthropic returned a response with no content blocks")
            block = response.content[0]
            if block.type == "text":
                return block.text
            return ""

        return await self._with_retry("anthropic.complete", attempt)

    async def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        Multi-turn chat with optional tool use. Used by LLMAgent.

        Returns:
            {
                "content":     str,           # Text response
                "tool_calls":  list[dict],    # Tool calls (if any)
                "tokens_used": int,           # Total tokens consumed
            }
        """
        # Separate system message from conversation
        system_content = ""
        conv_messages  = []

        for msg in messages:
            if msg["role"] == "system":
                system_content += msg["content"] + "\n"
            else:
                conv_messages.append(msg)

        # Ensure we have at least one user message
        if not conv_messages:
            conv_messages = [{"role": "user", "content": "Please proceed."}]

        # Build API call kwargs
        kwargs: dict[str, Any] = {
            "model":      self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages":   conv_messages,
        }
        if system_content.strip():
            kwargs["system"] = system_content.strip()

        # Convert ARIA tool schema to Anthropic tool format
        if tools:
            kwargs["tools"] = [
                {
                    "name":        t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
                }
                for t in tools
            ]

        async def attempt() -> dict:
            response = await self._client.messages.create(**kwargs)

            # Parse response
            text_content = ""
            tool_calls   = []

            for block in response.content:
                if block.type == "text":
                    text_content += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "name":      block.name,
                        "arguments": block.input,
                        "id":        block.id,
                    })

            if not text_content and not tool_calls and not _anthropic_refused(response):
                raise EmptyResponseError("Anthropic returned no text and no tool calls")

            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            return {
                "content":    text_content,
                "tool_calls": tool_calls,
                "tokens_used": tokens_used,
            }

        return await self._with_retry("anthropic.chat", attempt)


class OpenAIPlugin(_RetryMixin):
    """
    OpenAI GPT plugin for ARIA.

    Usage:
        plugin = OpenAIPlugin(api_key="sk-...", model="gpt-4o")
        aria = ARIAOrchestrator(llm_plugin=plugin)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        temperature: float = 0.3,
        retry_policy: Optional[RetryPolicy] = None,
        on_attempt: Optional[Callable[[AttemptRecord], Any]] = None,
    ):
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self._api_key    = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._init_retry(retry_policy, on_attempt)

        if not self._api_key:
            raise ValueError("OpenAI API key not provided.")

        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key)
        except ImportError:
            raise ImportError("Install openai with: pip install openai")

    async def complete(self, prompt: str, max_tokens: int = 2000, model: Optional[str] = None) -> str:
        async def attempt() -> str:
            response = await self._client.chat.completions.create(
                model    = model or self.model,
                messages = [{"role": "user", "content": prompt}],
                max_tokens = max_tokens,
            )
            if not response.choices:
                raise EmptyResponseError("OpenAI returned a response with no choices")
            return response.choices[0].message.content or ""

        return await self._with_retry("openai.complete", attempt)

    async def chat(self, messages: list[dict], tools: Optional[list[dict]] = None, max_tokens: Optional[int] = None) -> dict:
        kwargs: dict[str, Any] = {
            "model":      self.model,
            "messages":   messages,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            kwargs["tool_choice"] = "auto"

        async def attempt() -> dict:
            response = await self._client.chat.completions.create(**kwargs)
            if not response.choices:
                raise EmptyResponseError("OpenAI returned a response with no choices")
            msg = response.choices[0].message

            tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append({"name": tc.function.name, "arguments": args, "id": tc.id})

            if not msg.content and not tool_calls and not _openai_declined(response.choices[0]):
                raise EmptyResponseError("OpenAI returned no text and no tool calls")

            tokens_used = response.usage.total_tokens if response.usage else 0
            return {
                "content":     msg.content or "",
                "tool_calls":  tool_calls,
                "tokens_used": tokens_used,
            }

        return await self._with_retry("openai.chat", attempt)
