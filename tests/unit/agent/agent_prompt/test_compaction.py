import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from windows_use.agent.context.service import Context
from windows_use.messages import AIMessage, HumanMessage, ImageMessage, SystemMessage, ToolMessage
from windows_use.providers.events import LLMEvent, LLMEventType
from windows_use.providers.views import TokenUsage


@pytest.fixture
def context():
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    return Context(llm=llm)


def test_history_preserves_json_parameter_types_and_missing_content(context):
    params = {
        "text": "café",
        "count": 3,
        "ratio": 0.5,
        "enabled": True,
        "disabled": False,
        "optional": None,
        "coordinates": [10, 20],
        "nested": {"items": [None, True, {"label": "Save"}]},
        "thought": "Continue the task",
    }
    message = ToolMessage(id="call_1", name="example_tool", params=params, content=None)

    history = context._format_history_for_compaction([message])

    parameters = next(line for line in history.splitlines() if line.startswith("TOOL CALL"))
    serialized = parameters.removeprefix("TOOL CALL call_1: example_tool(").removesuffix(")")
    assert json.loads(serialized) == {
        key: value for key, value in params.items() if key != "thought"
    }
    assert "THOUGHT: Continue the task" in history
    assert "TOOL RESULT call_1: " in history
    assert message.params == params
    assert message.content is None


def test_history_handles_message_variants_without_serializing_images(context):
    image = object()
    history = context._format_history_for_compaction(
        [
            SystemMessage(content="private system instructions"),
            HumanMessage(content="Original task"),
            ImageMessage(content="Current screen", images=[image]),
            AIMessage(content=None),
            AIMessage(content={"remaining": ["Save", "Verify"]}),
        ]
    )

    assert "private system instructions" not in history
    assert "USER: Original task" in history
    assert "USER: Current screen" in history
    assert "ASSISTANT: " in history
    assert 'ASSISTANT: {"remaining": ["Save", "Verify"]}' in history
    assert repr(image) not in history


@pytest.mark.parametrize(
    "value",
    ["x" * 20_000, list(range(10_000)), {"nested": {"text": "x" * 20_000}}],
    ids=["long-string", "long-list", "nested-object"],
)
def test_history_bounds_tool_parameters_thought_and_result(context, value):
    message = ToolMessage(
        id="call_1",
        name="example_tool",
        params={"large": value, "thought": "t" * 20_000},
        content="r" * 20_000,
    )

    history = context._format_history_for_compaction([message])

    assert history.count("...[TRUNCATED]") == 3
    assert len(history) < 7300
    assert "TOOL CALL call_1: example_tool(" in history
    assert "TOOL RESULT call_1:" in history
    assert message.content == "r" * 20_000


def test_history_bounds_many_parameters_as_a_group(context):
    message = ToolMessage(
        id="call_1",
        name="example_tool",
        params={f"field_{index}": "x" * 100 for index in range(1000)},
        content="ok",
    )

    history = context._format_history_for_compaction([message])

    assert "...[TRUNCATED]" in history
    assert len(history) < 4300


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
async def test_compaction_uses_matching_provider_and_resets_usage(context, async_mode):
    event = LLMEvent(
        type=LLMEventType.TEXT,
        content="  Saved the draft. Next, verify its contents.  ",
        usage=TokenUsage(total_tokens=500),
    )
    context.llm.invoke.return_value = event
    context.llm.ainvoke.return_value = event
    context.token_usage = TokenUsage(total_tokens=90_000)
    messages = [HumanMessage(content="Create a draft")]

    result = await context.acompact(messages) if async_mode else context.compact(messages)

    assert "Saved the draft. Next, verify its contents." in result
    assert "Don't repeat already completed ACTIONS" in result
    assert context.token_usage.total_tokens == 0
    assert messages == [HumanMessage(content="Create a draft")]
    if async_mode:
        context.llm.ainvoke.assert_awaited_once()
        context.llm.invoke.assert_not_called()
        sent_messages = context.llm.ainvoke.call_args.args[0]
    else:
        context.llm.invoke.assert_called_once()
        context.llm.ainvoke.assert_not_called()
        sent_messages = context.llm.invoke.call_args.args[0]
    assert isinstance(sent_messages[0], SystemMessage)
    assert isinstance(sent_messages[1], HumanMessage)
    assert "USER: Create a draft" in sent_messages[1].content


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "response",
    [
        None,
        LLMEvent(type=LLMEventType.TEXT, content=None, usage=TokenUsage()),
        LLMEvent(type=LLMEventType.TEXT, content="", usage=TokenUsage()),
        LLMEvent(type=LLMEventType.TEXT, content=" \n\t", usage=TokenUsage()),
        LLMEvent(type=LLMEventType.ERROR, content="Rate limited", usage=TokenUsage()),
        LLMEvent(type=LLMEventType.TOOL_CALL, content="Unusable answer", usage=TokenUsage()),
    ],
    ids=["none", "no-content", "empty", "whitespace", "error", "tool-call"],
)
async def test_invalid_summary_preserves_history_and_usage(context, async_mode, response):
    context.llm.invoke.return_value = response
    context.llm.ainvoke.return_value = response
    usage = TokenUsage(total_tokens=90_000)
    context.token_usage = usage
    messages = [HumanMessage(content="Original task"), AIMessage(content="Progress")]
    original = [message.model_copy(deep=True) for message in messages]

    result = await context.acompact(messages) if async_mode else context.compact(messages)

    assert result is None
    assert messages == original
    assert context.token_usage == usage


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
async def test_provider_exception_preserves_history_and_usage(context, async_mode):
    context.llm.invoke.side_effect = RuntimeError("Provider unavailable")
    context.llm.ainvoke.side_effect = RuntimeError("Provider unavailable")
    context.token_usage = TokenUsage(total_tokens=90_000)
    messages = [HumanMessage(content="Original task")]

    with pytest.raises(RuntimeError, match="Provider unavailable"):
        if async_mode:
            await context.acompact(messages)
        else:
            context.compact(messages)

    assert messages == [HumanMessage(content="Original task")]
    assert context.token_usage.total_tokens == 90_000


async def test_async_compaction_propagates_cancellation(context):
    context.llm.ainvoke.side_effect = asyncio.CancelledError
    context.token_usage = TokenUsage(total_tokens=90_000)

    with pytest.raises(asyncio.CancelledError):
        await context.acompact([HumanMessage(content="Original task")])

    assert context.token_usage.total_tokens == 90_000
    context.llm.invoke.assert_not_called()
