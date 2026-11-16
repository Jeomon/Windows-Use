"""Exercise history compaction through both real agent loops without desktop actions."""

from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from windows_use.agent.registry.service import Registry
from windows_use.agent.service import Agent
from windows_use.agent.tools.service import done_tool
from windows_use.messages import HumanMessage, SystemMessage, ToolMessage
from windows_use.providers.events import LLMEvent, LLMEventType, ToolCall
from windows_use.providers.views import TokenUsage
from windows_use.tools import Tool


class RecordParams(BaseModel):
    flag: bool
    count: int
    coordinates: list[int]


@Tool("record_tool", model=RecordParams)
def record_tool(flag: bool, count: int, coordinates: list[int], **kwargs) -> str:
    return "recorded completed step"


def tool_event(name, params, tokens=0):
    return LLMEvent(
        type=LLMEventType.TOOL_CALL,
        tool_call=ToolCall(id=name, name=name, params=params),
        usage=TokenUsage(total_tokens=tokens),
    )


@pytest.fixture
def agent_factory():
    with (
        patch("windows_use.agent.service.Desktop"),
        patch("windows_use.agent.service.WatchDog"),
        patch("windows_use.agent.service.ProductTelemetry"),
    ):

        def make_agent(llm):
            agent = Agent(llm=llm, max_steps=3, log_to_console=False, disable_loop_detection=True)
            agent.context.system = MagicMock(return_value=SystemMessage(content="test system"))
            agent.context.state = MagicMock(return_value=HumanMessage(content="current UI"))
            agent.registry = Registry([record_tool, done_tool])
            return agent

        yield make_agent


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("summary_result", ["valid", "error", "empty", "exception"])
def test_compaction_preserves_progress_and_failed_history(
    agent_factory, async_mode, summary_result
):
    import asyncio

    task_events = deque(
        [
            tool_event("record_tool", {"flag": True, "count": 3, "coordinates": [10, 20]}),
            tool_event("missing_tool", {"attempt": 2}, tokens=90),
            tool_event("done_tool", {"answer": "finished", "thought": "verified"}),
        ]
    )
    task_inputs = []
    summary_inputs = []

    def respond(messages, tools=None):
        if tools is not None:
            task_inputs.append([message.model_copy(deep=True) for message in messages])
            return task_events.popleft()
        summary_inputs.append([message.model_copy(deep=True) for message in messages])
        if summary_result == "exception":
            raise RuntimeError("summary provider unavailable")
        return LLMEvent(
            type=LLMEventType.ERROR if summary_result == "error" else LLMEventType.TEXT,
            content={
                "valid": "Saved the completed step; missing tool failed",
                "error": "busy",
                "empty": " ",
            }[summary_result],
            usage=TokenUsage(),
        )

    llm = MagicMock()
    llm.model_name = "test-model"
    llm.provider = "test-provider"
    llm.get_metadata.return_value.context_window = 100
    llm.invoke.side_effect = respond
    llm.ainvoke = AsyncMock(side_effect=respond)
    agent = agent_factory(llm)

    result = (
        asyncio.run(agent.ainvoke("complete the task"))
        if async_mode
        else agent.invoke("complete the task")
    )

    assert result.is_done is True
    assert result.content == "finished"
    assert len(task_inputs) == 3
    assert len(summary_inputs) == 1
    formatted_history = summary_inputs[0][1].content
    assert "recorded completed step" in formatted_history
    assert "missing_tool" in formatted_history
    assert '"count": 3' in formatted_history
    assert '"flag": true' in formatted_history

    next_messages = task_inputs[2]
    if summary_result == "valid":
        assert any(
            "Saved the completed step" in (message.content or "") for message in next_messages
        )
        assert not any(isinstance(message, ToolMessage) for message in next_messages)
        assert agent.state.error_messages == []
    else:
        assert any(
            isinstance(message, ToolMessage) and message.name == "record_tool"
            for message in next_messages
        )
        assert any(
            isinstance(message, ToolMessage) and message.name == "missing_tool"
            for message in next_messages
        )
        assert agent.state.error_messages
    if async_mode:
        llm.invoke.assert_not_called()
        assert llm.ainvoke.await_count == 4
    else:
        llm.ainvoke.assert_not_called()
        assert llm.invoke.call_count == 4


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_new_task_does_not_compact_previous_tasks_token_usage(agent_factory, async_mode):
    import asyncio

    llm = MagicMock()
    llm.get_metadata.return_value.context_window = 100
    completion = tool_event("done_tool", {"answer": "finished", "thought": "verified"}, tokens=90)
    llm.invoke.return_value = completion
    llm.ainvoke = AsyncMock(return_value=completion)
    agent = agent_factory(llm)
    agent.context.compact = MagicMock(side_effect=AssertionError("stale token usage"))
    agent.context.acompact = AsyncMock(side_effect=AssertionError("stale token usage"))

    for task in ("first task", "second task"):
        result = asyncio.run(agent.ainvoke(task)) if async_mode else agent.invoke(task)
        assert result.is_done is True
    agent.context.compact.assert_not_called()
    agent.context.acompact.assert_not_called()
