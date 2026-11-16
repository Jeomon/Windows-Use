"""Task result contracts without desktop actions, network requests, or telemetry."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from windows_use.agent.events import EventType
from windows_use.agent.registry.views import ToolResult
from windows_use.agent.service import Agent
from windows_use.agent.views import AgentResult
from windows_use.messages import HumanMessage, SystemMessage
from windows_use.providers.events import LLMEvent, LLMEventType, ToolCall
from windows_use.providers.views import TokenUsage


def tool_call(name="done_tool", **params):
    return LLMEvent(
        type=LLMEventType.TOOL_CALL,
        tool_call=ToolCall(id="call_1", name=name, params={"thought": "Test", **params}),
        usage=TokenUsage(),
    )


def run_task(agent, asynchronous):
    if asynchronous:
        return asyncio.run(agent.ainvoke("A test task"))
    return agent.invoke("A test task")


@pytest.fixture
def agent_factory():
    with (
        patch("windows_use.agent.service.Desktop") as desktop_class,
        patch("windows_use.agent.service.WatchDog"),
        patch("windows_use.agent.service.ProductTelemetry"),
        patch("windows_use.agent.service.Context") as context_class,
        patch("windows_use.agent.service.time.sleep"),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        desktop_class.return_value.use_vision = False
        context = context_class.return_value
        context.need_compaction = False
        context.system.return_value = SystemMessage(content="Test system instructions")
        context.task.return_value = HumanMessage(content="A test task")
        context.state.side_effect = lambda **kwargs: HumanMessage(content="A mocked desktop")

        def make_agent(**kwargs):
            llm = MagicMock()
            llm.model_name = "test-model"
            llm.provider = "test-provider"
            llm.invoke.return_value = tool_call(answer="Finished")
            llm.ainvoke = AsyncMock(return_value=tool_call(answer="Finished"))
            events = []
            agent = Agent(
                llm=llm,
                log_to_console=False,
                disable_loop_detection=True,
                event_subscriber=events.append,
                **kwargs,
            )
            return agent, events

        yield make_agent


@pytest.mark.parametrize("asynchronous", [False, True])
def test_unexpected_execution_error_returns_failed_result(agent_factory, asynchronous):
    agent, events = agent_factory()
    agent.context.state.side_effect = RuntimeError("Desktop observation failed")

    result = run_task(agent, asynchronous)

    assert isinstance(result, AgentResult)
    assert result.is_done is False
    assert result.error == "Desktop observation failed"
    assert result.content is None
    assert events[-1].type == EventType.ERROR
    telemetry_event = agent.telemetry.capture.call_args.args[0]
    assert telemetry_event.is_success is False
    assert telemetry_event.error_type == "RuntimeError"
    agent.watchdog.__exit__.assert_called_once()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_watchdog_start_failure_returns_failed_result(agent_factory, asynchronous):
    agent, _ = agent_factory()
    agent.watchdog.__enter__.side_effect = RuntimeError("Could not start watchdog")

    result = run_task(agent, asynchronous)

    assert result.is_done is False
    assert result.error == "Could not start watchdog"
    agent.llm.invoke.assert_not_called()
    agent.llm.ainvoke.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_invalid_done_call_cannot_report_success(agent_factory, asynchronous):
    agent, events = agent_factory(max_steps=1, max_consecutive_failures=3)
    invalid_done = tool_call()  # Missing the required answer; real registry validation runs.
    agent.llm.invoke.return_value = invalid_done
    agent.llm.ainvoke.return_value = invalid_done

    result = run_task(agent, asynchronous)

    assert result.is_done is False
    assert result.error
    assert not any(event.type == EventType.DONE for event in events)
    failures = [event for event in events if event.type == EventType.TOOL_RESULT]
    assert len(failures) == 1
    assert failures[0].data["is_success"] is False
    assert "validation failed" in failures[0].data["content"]
    assert "answer" in agent.state.error_messages[0].content


@pytest.mark.parametrize("asynchronous", [False, True])
def test_invalid_done_can_be_corrected_on_next_step(agent_factory, asynchronous):
    agent, events = agent_factory(max_steps=2)
    responses = [tool_call(), tool_call(answer="Corrected completion")]
    agent.llm.invoke.side_effect = responses
    agent.llm.ainvoke.side_effect = responses

    result = run_task(agent, asynchronous)

    assert result == AgentResult(is_done=True, content="Corrected completion")
    assert len([event for event in events if event.type == EventType.DONE]) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_completion_uses_validated_tool_output(agent_factory, asynchronous):
    agent, events = agent_factory()
    executed = ToolResult(is_success=True, content="Validated completion")
    agent.registry.execute = MagicMock(return_value=executed)
    agent.registry.aexecute = AsyncMock(return_value=executed)

    result = run_task(agent, asynchronous)

    assert result.content == "Validated completion"
    assert events[-1].data["content"] == "Validated completion"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("detail", ["Provider rate limit reached", None])
def test_provider_error_events_produce_meaningful_failed_results(
    agent_factory, asynchronous, detail
):
    agent, events = agent_factory(max_consecutive_failures=2)
    response = LLMEvent(type=LLMEventType.ERROR, content=detail, usage=TokenUsage())
    agent.llm.invoke.return_value = response
    agent.llm.ainvoke.return_value = response

    result = run_task(agent, asynchronous)

    assert result.is_done is False
    expected_detail = detail or "Provider returned an error without details."
    assert expected_detail in result.error
    assert events[-1].data["error_source"] == "llm_exhausted_retries"
    failures = events[-1].data["llm_errors"]
    assert len(failures) == 2
    assert [failure["attempt"] for failure in failures] == [1, 2]
    assert all(failure["error_type"] == "provider_error" for failure in failures)
    assert all(failure["error"] == expected_detail for failure in failures)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_text_only_retries_explain_why_completion_failed(agent_factory, asynchronous):
    agent, _ = agent_factory(max_consecutive_failures=2)
    response = LLMEvent(type=LLMEventType.TEXT, content="I am done", usage=TokenUsage())
    agent.llm.invoke.return_value = response
    agent.llm.ainvoke.return_value = response

    result = run_task(agent, asynchronous)

    assert result.is_done is False
    assert "text instead of a tool call" in result.error
    assert "None" not in result.error


@pytest.mark.parametrize("asynchronous", [False, True])
def test_provider_exceptions_report_each_retry_and_final_failure(agent_factory, asynchronous):
    agent, events = agent_factory(max_consecutive_failures=2)
    agent.llm.invoke.side_effect = RuntimeError("Provider unavailable")
    agent.llm.ainvoke.side_effect = RuntimeError("Provider unavailable")

    result = run_task(agent, asynchronous)

    assert result.is_done is False
    assert "Provider unavailable" in result.error
    assert events[-1].data["error_source"] == "llm_exhausted_retries"
    assert len(events[-1].data["llm_errors"]) == 2
    assert all(error["error_type"] == "RuntimeError" for error in events[-1].data["llm_errors"])


@pytest.mark.parametrize("asynchronous", [False, True])
def test_tool_failure_limit_is_enforced(agent_factory, asynchronous):
    agent, _ = agent_factory(max_steps=5, max_consecutive_failures=2)
    agent.llm.invoke.return_value = tool_call("click_tool")
    agent.llm.ainvoke.return_value = tool_call("click_tool")
    failure = ToolResult(is_success=False, error="Element disappeared")
    agent.registry.execute = MagicMock(return_value=failure)
    agent.registry.aexecute = AsyncMock(return_value=failure)

    result = run_task(agent, asynchronous)

    assert result.is_done is False
    assert "2 consecutive tool failures" in result.error
    assert "Element disappeared" in result.error
    action = agent.registry.aexecute if asynchronous else agent.registry.execute
    assert action.call_count == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("operation", ["capture", "flush"])
@pytest.mark.parametrize("task_fails", [False, True])
def test_telemetry_failure_preserves_task_outcome(
    agent_factory, asynchronous, operation, task_fails
):
    agent, _ = agent_factory()
    getattr(agent.telemetry, operation).side_effect = RuntimeError("Telemetry unavailable")
    if task_fails:
        agent.context.state.side_effect = RuntimeError("Original task failure")

    result = run_task(agent, asynchronous)

    if task_fails:
        assert result == AgentResult(is_done=False, error="Original task failure")
    else:
        assert result == AgentResult(is_done=True, content="Finished")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("exception", [KeyboardInterrupt, asyncio.CancelledError])
def test_cancellation_propagates_and_cleans_up_watchdog(agent_factory, asynchronous, exception):
    agent, _ = agent_factory()
    agent.llm.invoke.side_effect = exception()
    agent.llm.ainvoke.side_effect = exception()

    with pytest.raises(exception):
        run_task(agent, asynchronous)

    agent.watchdog.__exit__.assert_called_once()
    agent.telemetry.capture.assert_not_called()


@pytest.mark.parametrize("name", ["max_steps", "max_consecutive_failures"])
@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_invalid_limits_fail_before_desktop_initialization(name, limit):
    with patch("windows_use.agent.service.Desktop") as desktop:
        with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
            Agent(**{name: limit})
        desktop.assert_not_called()
