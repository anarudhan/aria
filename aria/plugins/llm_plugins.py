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
from typing import Optional, Any


class AnthropicPlugin:
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
    ):
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self._api_key    = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

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
        """Single-turn text completion. Used by the decomposer."""
        response = await self._client.messages.create(
            model      = model or self.model,
            max_tokens = max_tokens,
            messages   = [{"role": "user", "content": prompt}],
        )
        return response.content[0].text

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

        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        return {
            "content":    text_content,
            "tool_calls": tool_calls,
            "tokens_used": tokens_used,
        }


class OpenAIPlugin:
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
    ):
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self._api_key    = api_key or os.environ.get("OPENAI_API_KEY", "")

        if not self._api_key:
            raise ValueError("OpenAI API key not provided.")

        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key)
        except ImportError:
            raise ImportError("Install openai with: pip install openai")

    async def complete(self, prompt: str, max_tokens: int = 2000, model: Optional[str] = None) -> str:
        response = await self._client.chat.completions.create(
            model    = model or self.model,
            messages = [{"role": "user", "content": prompt}],
            max_tokens = max_tokens,
        )
        return response.choices[0].message.content or ""

    async def chat(self, messages: list[dict], tools: Optional[list[dict]] = None, max_tokens: Optional[int] = None) -> dict:
        kwargs: dict[str, Any] = {
            "model":      self.model,
            "messages":   messages,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({"name": tc.function.name, "arguments": args, "id": tc.id})

        tokens_used = response.usage.total_tokens if response.usage else 0
        return {
            "content":     msg.content or "",
            "tool_calls":  tool_calls,
            "tokens_used": tokens_used,
        }
