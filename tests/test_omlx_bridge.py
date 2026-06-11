from __future__ import annotations

import json

from mtplx.server.omlx_bridge import (
    ToolCallStreamFilter,
    extract_thinking,
    normalize_messages_for_template,
    parse_tool_calls,
)


def test_omlx_adapter_preserves_reasoning_tool_calls_and_tool_results():
    messages = [
        {"role": "system", "content": "You are OpenCode."},
        {"role": "developer", "content": "Keep changes narrow."},
        {"role": "user", "content": "status?"},
        {
            "role": "assistant",
            "content": "Let me inspect.",
            "reasoning_content": "Need to read files first.",
            "tool_calls": [
                {
                    "id": "call_read",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{\"path\":\"x\"}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_read", "content": "file text"},
    ]

    normalized = normalize_messages_for_template(messages)

    assert normalized[0]["role"] == "system"
    assert "You are OpenCode." in normalized[0]["content"]
    assert "Keep changes narrow." in normalized[0]["content"]
    assert normalized[2]["reasoning_content"] == "Need to read files first."
    assert normalized[2]["tool_calls"][0]["id"] == "call_read"
    assert normalized[3]["role"] == "tool"
    assert normalized[3]["tool_call_id"] == "call_read"


def test_omlx_thinking_unclosed_recovers_visible_content():
    reasoning, visible = extract_thinking("<think>Need to answer naturally.")

    assert reasoning == "Need to answer naturally."
    assert visible == "Need to answer naturally."


def test_omlx_tool_parser_tries_qwen_xml_and_normalizes_arguments():
    extraction = parse_tool_calls(
        "<tool_call>{\"name\":\"add\",\"arguments\":{\"a\":1,\"b\":2}}</tool_call>",
        tokenizer=None,
        tools=[{"type": "function", "function": {"name": "add"}}],
    )

    assert extraction.status == "parsed"
    assert extraction.parser_source == "qwen_xml"
    arguments = extraction.tool_calls[0]["function"]["arguments"]
    assert json.loads(arguments) == {"a": 1, "b": 2}


def test_omlx_tool_parser_accepts_opencode_drifted_json_shapes():
    tools = [{"type": "function", "function": {"name": "read"}}]
    samples = [
        '<tool_call>{"tool":"read","args":{"filePath":"src/game/Game.ts"}}</tool_call>',
        '<tool_call>{"function":{"name":"read","arguments":{"filePath":"src/game/Game.ts"}}}</tool_call>',
        '<tool_call>[{"name":"read","parameters":{"filePath":"src/game/Game.ts"}}]</tool_call>',
        '<tool_call>read({"filePath":"src/game/Game.ts"})</tool_call>',
    ]

    for sample in samples:
        extraction = parse_tool_calls(sample, tokenizer=None, tools=tools)
        assert extraction.status == "parsed"
        assert extraction.parser_source == "qwen_xml"
        assert extraction.tool_calls[0]["function"]["name"] == "read"
        arguments = extraction.tool_calls[0]["function"]["arguments"]
        assert json.loads(arguments) == {"filePath": "src/game/Game.ts"}


def test_omlx_tool_parser_accepts_name_attribute_function_shape():
    extraction = parse_tool_calls(
        '<tool_call><function name="read"><parameter name="filePath">src/game/Game.ts</parameter></function></tool_call>',
        tokenizer=None,
        tools=[{"type": "function", "function": {"name": "read"}}],
    )

    assert extraction.status == "parsed"
    assert extraction.tool_calls[0]["function"]["name"] == "read"
    assert json.loads(extraction.tool_calls[0]["function"]["arguments"]) == {
        "filePath": "src/game/Game.ts"
    }


def test_omlx_tool_parser_orders_opencode_read_target_before_limit():
    extraction = parse_tool_calls(
        "<tool_call>\n"
        "<function=read>\n"
        "<parameter=limit>\n100\n</parameter>\n"
        "<parameter=filePath>\nsrc/game/Game.ts\n</parameter>\n"
        "</function>\n"
        "</tool_call>",
        tokenizer=None,
        tools=[{"type": "function", "function": {"name": "read"}}],
    )

    assert extraction.status == "parsed"
    arguments = extraction.tool_calls[0]["function"]["arguments"]
    assert arguments.startswith('{"filePath":"src/game/Game.ts","limit":100')
    assert json.loads(arguments) == {"filePath": "src/game/Game.ts", "limit": 100}


def test_omlx_tool_parser_malformed_markup_remains_content():
    text = "<tool_call>not json</tool_call>"
    extraction = parse_tool_calls(
        text,
        tokenizer=None,
        tools=[{"type": "function", "function": {"name": "add"}}],
    )

    assert extraction.status == "malformed_as_content"
    assert extraction.tool_calls is None
    assert extraction.cleaned_text == text


def test_omlx_stream_filter_suppresses_tool_markup_without_cancelling():
    stream_filter = ToolCallStreamFilter()
    visible = []
    for char in "Before <tool_call>{\"name\":\"add\"}</tool_call> after":
        chunk = stream_filter.feed(char)
        if chunk:
            visible.append(chunk)
    tail = stream_filter.finish()
    if tail:
        visible.append(tail)

    assert "".join(visible) == "Before  after"
    assert stream_filter.suppressed_markup is True
