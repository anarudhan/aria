"""
Unit tests for aria.plugins.llm_plugins (AnthropicPlugin and OpenAIPlugin).

The vendor SDKs are replaced by in-process fakes injected into sys.modules, so these
tests need neither the real packages, network access nor API keys.
"""
import inspect
import json
import sys
import types
from types import SimpleNamespace

import pytest

from aria.plugins import AnthropicPlugin, OpenAIPlugin
from aria.plugins.llm_plugins import AnthropicPlugin as DirectAnthropicPlugin

# ─── Fake SDKs ────────────────────────────────────────────────────────────────


class FakeAnthropicSDK:
    """Builds a fake `anthropic` module whose client records every request."""

    def __init__(self, response=None, error=None):
        self.requests = []
        self.clients = []
        self.response = response
        self.error = error
        sdk = self

        class _Messages:
            async def create(self, **kwargs):
                sdk.requests.append(kwargs)
                if sdk.error is not None:
                    raise sdk.error
                return sdk.response

        class AsyncAnthropic:
            def __init__(self, api_key=None):
                self.api_key = api_key
                self.messages = _Messages()
                sdk.clients.append(self)

        self.module = types.ModuleType("anthropic")
        self.module.AsyncAnthropic = AsyncAnthropic


class FakeOpenAISDK:
    def __init__(self, response=None, error=None):
        self.requests = []
        self.clients = []
        self.response = response
        self.error = error
        sdk = self

        class _Completions:
            async def create(self, **kwargs):
                sdk.requests.append(kwargs)
                if sdk.error is not None:
                    raise sdk.error
                return sdk.response

        class AsyncOpenAI:
            def __init__(self, api_key=None):
                self.api_key = api_key
                self.chat = SimpleNamespace(completions=_Completions())
                sdk.clients.append(self)

        self.module = types.ModuleType("openai")
        self.module.AsyncOpenAI = AsyncOpenAI


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(name, arguments, tool_id="toolu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=arguments, id=tool_id)


def anthropic_response(blocks, input_tokens=10, output_tokens=5):
    return SimpleNamespace(
        content=blocks,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def openai_message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def openai_tool_call(name, arguments, call_id="call_1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def openai_response(message, total_tokens=42):
    usage = None if total_tokens is None else SimpleNamespace(total_tokens=total_tokens)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def anthropic_sdk(monkeypatch):
    sdk = FakeAnthropicSDK(response=anthropic_response([text_block("hello")]))
    monkeypatch.setitem(sys.modules, "anthropic", sdk.module)
    return sdk


@pytest.fixture
def openai_sdk(monkeypatch):
    sdk = FakeOpenAISDK(response=openai_response(openai_message("hello")))
    monkeypatch.setitem(sys.modules, "openai", sdk.module)
    return sdk


# ─── Package surface ──────────────────────────────────────────────────────────

class TestPackageSurface:

    def test_plugins_are_exported_from_the_package(self):
        assert AnthropicPlugin is DirectAnthropicPlugin
        import aria.plugins
        assert set(aria.plugins.__all__) == {"AnthropicPlugin", "OpenAIPlugin"}

    @pytest.mark.parametrize("cls", [AnthropicPlugin, OpenAIPlugin])
    def test_plugins_implement_the_async_plugin_interface(self, cls):
        assert inspect.iscoroutinefunction(cls.complete)
        assert inspect.iscoroutinefunction(cls.chat)


# ─── AnthropicPlugin: construction ────────────────────────────────────────────

class TestAnthropicConstruction:

    def test_missing_key_raises_value_error(self, anthropic_sdk):
        with pytest.raises(ValueError, match="API key"):
            AnthropicPlugin()
        assert anthropic_sdk.clients == []        # no client was built

    def test_empty_string_key_counts_as_missing(self, anthropic_sdk):
        with pytest.raises(ValueError):
            AnthropicPlugin(api_key="")

    def test_explicit_key_is_passed_to_the_client(self, anthropic_sdk):
        plugin = AnthropicPlugin(api_key="sk-explicit")
        assert anthropic_sdk.clients[0].api_key == "sk-explicit"
        assert plugin.model == "claude-sonnet-4-20250514"
        assert plugin.max_tokens == 4096

    def test_key_is_read_from_environment(self, anthropic_sdk, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
        AnthropicPlugin()
        assert anthropic_sdk.clients[0].api_key == "sk-env"

    def test_explicit_key_wins_over_environment(self, anthropic_sdk, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
        AnthropicPlugin(api_key="sk-explicit")
        assert anthropic_sdk.clients[0].api_key == "sk-explicit"

    def test_missing_sdk_raises_helpful_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "anthropic", None)    # makes `import anthropic` fail
        with pytest.raises(ImportError, match="pip install anthropic"):
            AnthropicPlugin(api_key="sk-x")

    def test_missing_key_is_reported_before_missing_sdk(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "anthropic", None)
        with pytest.raises(ValueError):
            AnthropicPlugin()

    def test_custom_settings_are_stored(self, anthropic_sdk):
        plugin = AnthropicPlugin(api_key="k", model="m", max_tokens=99, temperature=0.9)
        assert (plugin.model, plugin.max_tokens, plugin.temperature) == ("m", 99, 0.9)


# ─── AnthropicPlugin: complete ────────────────────────────────────────────────

class TestAnthropicComplete:

    async def test_returns_text_and_sends_a_single_user_message(self, anthropic_sdk):
        plugin = AnthropicPlugin(api_key="k")

        text = await plugin.complete("say hello")

        assert text == "hello"
        request = anthropic_sdk.requests[0]
        assert request["messages"] == [{"role": "user", "content": "say hello"}]
        assert request["model"] == "claude-sonnet-4-20250514"
        assert request["max_tokens"] == 2000

    async def test_model_and_max_tokens_can_be_overridden_per_call(self, anthropic_sdk):
        plugin = AnthropicPlugin(api_key="k", model="default-model")
        await plugin.complete("p", max_tokens=123, model="other-model")
        request = anthropic_sdk.requests[0]
        assert request["model"] == "other-model" and request["max_tokens"] == 123

    async def test_instance_model_is_used_when_no_override(self, anthropic_sdk):
        await AnthropicPlugin(api_key="k", model="default-model").complete("p")
        assert anthropic_sdk.requests[0]["model"] == "default-model"

    async def test_non_text_first_block_returns_empty_string(self, anthropic_sdk):
        anthropic_sdk.response = anthropic_response([tool_block("search", {})])
        assert await AnthropicPlugin(api_key="k").complete("p") == ""

    async def test_api_errors_propagate(self, anthropic_sdk):
        anthropic_sdk.error = RuntimeError("rate limited")
        with pytest.raises(RuntimeError, match="rate limited"):
            await AnthropicPlugin(api_key="k").complete("p")

    @pytest.mark.xfail(
        strict=True,
        reason="bug: llm_plugins.py accepts a temperature argument and stores it, but never "
               "sends it to the Anthropic API, so the setting has no effect",
    )
    async def test_temperature_is_sent_to_the_api(self, anthropic_sdk):
        await AnthropicPlugin(api_key="k", temperature=0.7).complete("p")
        assert anthropic_sdk.requests[0].get("temperature") == 0.7


# ─── AnthropicPlugin: chat ────────────────────────────────────────────────────

class TestAnthropicChat:

    async def test_basic_chat_result_shape(self, anthropic_sdk):
        anthropic_sdk.response = anthropic_response([text_block("hi there")], 7, 3)
        plugin = AnthropicPlugin(api_key="k")

        result = await plugin.chat([{"role": "user", "content": "hello"}])

        assert result == {"content": "hi there", "tool_calls": [], "tokens_used": 10}

    async def test_system_messages_are_lifted_out_of_the_conversation(self, anthropic_sdk):
        plugin = AnthropicPlugin(api_key="k")
        await plugin.chat([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "question"},
            {"role": "system", "content": "be kind"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow up"},
        ])

        request = anthropic_sdk.requests[0]
        assert request["system"] == "be brief\nbe kind"
        assert [m["role"] for m in request["messages"]] == ["user", "assistant", "user"]
        assert [m["content"] for m in request["messages"]] == ["question", "answer", "follow up"]

    async def test_no_system_key_when_there_is_no_system_message(self, anthropic_sdk):
        await AnthropicPlugin(api_key="k").chat([{"role": "user", "content": "q"}])
        assert "system" not in anthropic_sdk.requests[0]

    async def test_blank_system_message_is_dropped(self, anthropic_sdk):
        await AnthropicPlugin(api_key="k").chat([
            {"role": "system", "content": "   "}, {"role": "user", "content": "q"},
        ])
        assert "system" not in anthropic_sdk.requests[0]

    async def test_only_system_messages_get_a_placeholder_user_turn(self, anthropic_sdk):
        await AnthropicPlugin(api_key="k").chat([{"role": "system", "content": "rules"}])
        request = anthropic_sdk.requests[0]
        assert request["messages"] == [{"role": "user", "content": "Please proceed."}]
        assert request["system"] == "rules"

    async def test_empty_message_list_gets_a_placeholder_user_turn(self, anthropic_sdk):
        await AnthropicPlugin(api_key="k").chat([])
        assert anthropic_sdk.requests[0]["messages"] == [
            {"role": "user", "content": "Please proceed."}
        ]

    async def test_max_tokens_default_and_override(self, anthropic_sdk):
        plugin = AnthropicPlugin(api_key="k", max_tokens=555)
        await plugin.chat([{"role": "user", "content": "q"}])
        await plugin.chat([{"role": "user", "content": "q"}], max_tokens=77)
        assert [r["max_tokens"] for r in anthropic_sdk.requests] == [555, 77]

    async def test_tools_are_converted_to_the_anthropic_schema(self, anthropic_sdk):
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        tools = [
            {"name": "search", "description": "Search the web", "parameters": schema},
            {"name": "ping"},
        ]
        await AnthropicPlugin(api_key="k").chat([{"role": "user", "content": "q"}], tools=tools)

        sent = anthropic_sdk.requests[0]["tools"]
        assert sent[0] == {"name": "search", "description": "Search the web",
                           "input_schema": schema}
        assert sent[1] == {"name": "ping", "description": "",
                           "input_schema": {"type": "object", "properties": {}}}

    async def test_no_tools_key_without_tools(self, anthropic_sdk):
        await AnthropicPlugin(api_key="k").chat([{"role": "user", "content": "q"}], tools=[])
        assert "tools" not in anthropic_sdk.requests[0]

    async def test_text_blocks_are_concatenated_and_tool_calls_extracted(self, anthropic_sdk):
        anthropic_sdk.response = anthropic_response([
            text_block("Let me look. "),
            tool_block("search", {"q": "aria"}, "toolu_a"),
            text_block("Done."),
            tool_block("fetch", {"url": "http://x"}, "toolu_b"),
        ], 100, 50)

        result = await AnthropicPlugin(api_key="k").chat([{"role": "user", "content": "go"}])

        assert result["content"] == "Let me look. Done."
        assert result["tool_calls"] == [
            {"name": "search", "arguments": {"q": "aria"}, "id": "toolu_a"},
            {"name": "fetch", "arguments": {"url": "http://x"}, "id": "toolu_b"},
        ]
        assert result["tokens_used"] == 150

    async def test_api_errors_propagate(self, anthropic_sdk):
        anthropic_sdk.error = ConnectionError("network down")
        with pytest.raises(ConnectionError):
            await AnthropicPlugin(api_key="k").chat([{"role": "user", "content": "q"}])

    async def test_message_without_role_is_rejected(self, anthropic_sdk):
        with pytest.raises(KeyError):
            await AnthropicPlugin(api_key="k").chat([{"content": "no role"}])
        assert anthropic_sdk.requests == []

    @pytest.mark.xfail(
        strict=True,
        reason="bug: llm_plugins.py stores temperature but never passes it in chat() requests",
    )
    async def test_temperature_is_sent_with_chat(self, anthropic_sdk):
        plugin = AnthropicPlugin(api_key="k", temperature=0.1)
        await plugin.chat([{"role": "user", "content": "q"}])
        assert anthropic_sdk.requests[0].get("temperature") == 0.1


# ─── OpenAIPlugin: construction ───────────────────────────────────────────────

class TestOpenAIConstruction:

    def test_missing_key_raises_value_error(self, openai_sdk):
        with pytest.raises(ValueError, match="API key"):
            OpenAIPlugin()
        assert openai_sdk.clients == []

    def test_explicit_key_and_defaults(self, openai_sdk):
        plugin = OpenAIPlugin(api_key="sk-explicit")
        assert openai_sdk.clients[0].api_key == "sk-explicit"
        assert plugin.model == "gpt-4o"
        assert plugin.max_tokens == 4096

    def test_key_from_environment_and_explicit_key_wins(self, openai_sdk, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        OpenAIPlugin()
        OpenAIPlugin(api_key="sk-explicit")
        assert [c.api_key for c in openai_sdk.clients] == ["sk-env", "sk-explicit"]

    def test_anthropic_key_is_not_used_for_openai(self, openai_sdk, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        with pytest.raises(ValueError):
            OpenAIPlugin()

    def test_missing_sdk_raises_helpful_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "openai", None)
        with pytest.raises(ImportError, match="pip install openai"):
            OpenAIPlugin(api_key="sk-x")


# ─── OpenAIPlugin: complete ───────────────────────────────────────────────────

class TestOpenAIComplete:

    async def test_returns_content_and_sends_a_single_user_message(self, openai_sdk):
        text = await OpenAIPlugin(api_key="k").complete("say hello")

        assert text == "hello"
        request = openai_sdk.requests[0]
        assert request["messages"] == [{"role": "user", "content": "say hello"}]
        assert request["model"] == "gpt-4o" and request["max_tokens"] == 2000

    async def test_overrides(self, openai_sdk):
        await OpenAIPlugin(api_key="k").complete("p", max_tokens=9, model="gpt-x")
        request = openai_sdk.requests[0]
        assert request["model"] == "gpt-x" and request["max_tokens"] == 9

    async def test_none_content_becomes_empty_string(self, openai_sdk):
        openai_sdk.response = openai_response(openai_message(None))
        assert await OpenAIPlugin(api_key="k").complete("p") == ""

    async def test_api_errors_propagate(self, openai_sdk):
        openai_sdk.error = TimeoutError("slow")
        with pytest.raises(TimeoutError):
            await OpenAIPlugin(api_key="k").complete("p")

    @pytest.mark.xfail(
        strict=True,
        reason="bug: llm_plugins.py stores temperature but never sends it to the OpenAI API",
    )
    async def test_temperature_is_sent_to_the_api(self, openai_sdk):
        await OpenAIPlugin(api_key="k", temperature=0.2).complete("p")
        assert openai_sdk.requests[0].get("temperature") == 0.2


# ─── OpenAIPlugin: chat ───────────────────────────────────────────────────────

class TestOpenAIChat:

    async def test_basic_result_shape_and_messages_passed_through(self, openai_sdk):
        openai_sdk.response = openai_response(openai_message("hi"), total_tokens=33)
        messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "q"}]

        result = await OpenAIPlugin(api_key="k").chat(messages)

        assert result == {"content": "hi", "tool_calls": [], "tokens_used": 33}
        request = openai_sdk.requests[0]
        assert request["messages"] == messages
        assert request["max_tokens"] == 4096
        assert "tools" not in request and "tool_choice" not in request

    async def test_max_tokens_override(self, openai_sdk):
        await OpenAIPlugin(api_key="k").chat([{"role": "user", "content": "q"}], max_tokens=64)
        assert openai_sdk.requests[0]["max_tokens"] == 64

    async def test_tools_are_wrapped_as_functions_with_auto_choice(self, openai_sdk):
        tool = {"name": "search", "description": "d", "parameters": {"type": "object"}}
        await OpenAIPlugin(api_key="k").chat([{"role": "user", "content": "q"}], tools=[tool])

        request = openai_sdk.requests[0]
        assert request["tools"] == [{"type": "function", "function": tool}]
        assert request["tool_choice"] == "auto"

    async def test_empty_tool_list_sends_no_tools(self, openai_sdk):
        await OpenAIPlugin(api_key="k").chat([{"role": "user", "content": "q"}], tools=[])
        assert "tools" not in openai_sdk.requests[0]

    async def test_tool_calls_are_parsed(self, openai_sdk):
        openai_sdk.response = openai_response(openai_message(
            None,
            [openai_tool_call("search", json.dumps({"q": "aria"}), "call_a"),
             openai_tool_call("fetch", json.dumps({"url": "http://x"}), "call_b")],
        ))

        result = await OpenAIPlugin(api_key="k").chat([{"role": "user", "content": "go"}])

        assert result["content"] == ""
        assert result["tool_calls"] == [
            {"name": "search", "arguments": {"q": "aria"}, "id": "call_a"},
            {"name": "fetch", "arguments": {"url": "http://x"}, "id": "call_b"},
        ]

    async def test_invalid_tool_arguments_become_empty_dict(self, openai_sdk):
        openai_sdk.response = openai_response(openai_message(
            "calling", [openai_tool_call("search", "{not valid json", "call_a")],
        ))

        result = await OpenAIPlugin(api_key="k").chat([{"role": "user", "content": "go"}])

        assert result["tool_calls"] == [{"name": "search", "arguments": {}, "id": "call_a"}]
        assert result["content"] == "calling"

    async def test_missing_usage_counts_zero_tokens(self, openai_sdk):
        openai_sdk.response = openai_response(openai_message("hi"), total_tokens=None)
        result = await OpenAIPlugin(api_key="k").chat([{"role": "user", "content": "q"}])
        assert result["tokens_used"] == 0

    async def test_api_errors_propagate(self, openai_sdk):
        openai_sdk.error = RuntimeError("quota exceeded")
        with pytest.raises(RuntimeError, match="quota"):
            await OpenAIPlugin(api_key="k").chat([{"role": "user", "content": "q"}])

    @pytest.mark.xfail(
        strict=True,
        reason="bug: llm_plugins.py stores temperature but never passes it in chat() requests",
    )
    async def test_temperature_is_sent_with_chat(self, openai_sdk):
        await OpenAIPlugin(api_key="k", temperature=0.4).chat([{"role": "user", "content": "q"}])
        assert openai_sdk.requests[0].get("temperature") == 0.4
