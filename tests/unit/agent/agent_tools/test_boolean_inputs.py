"""Exercise schema-to-desktop boolean handling without sending native input."""

import asyncio
from unittest.mock import MagicMock, call

import pytest
from pydantic import ValidationError

import windows_use.agent.desktop.service as desktop_module
from windows_use.agent.desktop.service import Desktop
from windows_use.agent.registry.service import Registry
from windows_use.agent.tools.service import move_tool, multi_edit_tool, multi_select_tool, type_tool


@pytest.fixture
def desktop_and_uia(monkeypatch):
    uia = MagicMock()
    uia.Keys.VK_CONTROL = 0x11
    uia.GetCursorPos.return_value = (1, 2)
    monkeypatch.setattr(desktop_module, "uia", uia)
    monkeypatch.setattr(desktop_module, "sleep", MagicMock())
    # These input methods need no initialized tree, windows, or COM resources.
    return object.__new__(Desktop), uia


@pytest.fixture(params=["sync", "async"])
def dispatch(request):
    registry = Registry([type_tool, multi_select_tool, multi_edit_tool, move_tool])

    def run(tool_name, params, desktop):
        params = {"thought": "Exercise input behavior", **params}
        if request.param == "sync":
            return registry.execute(tool_name, params, desktop)
        return asyncio.run(registry.aexecute(tool_name, params, desktop))

    return run


@pytest.mark.parametrize(
    "clear,press_enter,expected_keys",
    [
        (True, True, ["{Ctrl}a", "{Back}", "hello", "{Enter}"]),
        (True, False, ["{Ctrl}a", "{Back}", "hello"]),
        (False, True, ["hello", "{Enter}"]),
        (False, False, ["hello"]),
        ("true", "true", ["{Ctrl}a", "{Back}", "hello", "{Enter}"]),
        ("true", "false", ["{Ctrl}a", "{Back}", "hello"]),
        ("false", "true", ["hello", "{Enter}"]),
        ("false", "false", ["hello"]),
    ],
)
def test_typing_flags_reach_native_input(
    dispatch, desktop_and_uia, clear, press_enter, expected_keys
):
    desktop, uia = desktop_and_uia
    result = dispatch(
        "type_tool",
        {"loc": [10, 20], "text": "hello", "clear": clear, "press_enter": press_enter},
        desktop,
    )

    assert result.is_success
    uia.Click.assert_called_once_with(10, 20)
    assert [entry.args[0] for entry in uia.SendKeys.call_args_list] == expected_keys


def test_omitted_typing_flags_do_not_clear_or_submit(dispatch, desktop_and_uia):
    desktop, uia = desktop_and_uia
    result = dispatch("type_tool", {"loc": [10, 20], "text": "hello"}, desktop)

    assert result.is_success
    uia.SendKeys.assert_called_once_with("hello", interval=0.01, waitTime=0.05)


@pytest.mark.parametrize(
    "press_ctrl,hold_ctrl", [(True, True), (False, False), ("true", True), ("false", False)]
)
def test_multiselect_holds_ctrl_only_when_requested(
    dispatch, desktop_and_uia, press_ctrl, hold_ctrl
):
    desktop, uia = desktop_and_uia
    result = dispatch(
        "multi_select_tool",
        {"elements": [[10, 20], [30, 40]], "press_ctrl": press_ctrl},
        desktop,
    )

    assert result.is_success
    expected = [call.Click(10, 20, waitTime=0.2), call.Click(30, 40, waitTime=0.2)]
    if hold_ctrl:
        expected = [
            call.PressKey(0x11, waitTime=0.05),
            *expected,
            call.ReleaseKey(0x11, waitTime=0.05),
        ]
    assert uia.method_calls == expected


def test_omitted_multiselect_flag_does_not_touch_ctrl(dispatch, desktop_and_uia):
    desktop, uia = desktop_and_uia
    result = dispatch("multi_select_tool", {"elements": [[10, 20]]}, desktop)

    assert result.is_success
    assert uia.method_calls == [call.Click(10, 20, waitTime=0.2)]


@pytest.mark.parametrize("press_ctrl", [True, False])
def test_multiselect_click_failure_releases_only_ctrl_it_pressed(
    dispatch, desktop_and_uia, press_ctrl
):
    desktop, uia = desktop_and_uia
    uia.Click.side_effect = RuntimeError("Input failed")
    result = dispatch(
        "multi_select_tool",
        {"elements": [[10, 20]], "press_ctrl": press_ctrl},
        desktop,
    )

    assert not result.is_success
    assert "Input failed" in result.error
    if press_ctrl:
        assert uia.method_calls == [
            call.PressKey(0x11, waitTime=0.05),
            call.Click(10, 20, waitTime=0.2),
            call.ReleaseKey(0x11, waitTime=0.05),
        ]
    else:
        assert uia.method_calls == [call.Click(10, 20, waitTime=0.2)]


def test_failed_ctrl_press_does_not_click_or_release_an_unheld_key(dispatch, desktop_and_uia):
    desktop, uia = desktop_and_uia
    uia.PressKey.side_effect = RuntimeError("Cannot press Ctrl")
    result = dispatch("multi_select_tool", {"elements": [[10, 20]], "press_ctrl": True}, desktop)

    assert not result.is_success
    assert "Cannot press Ctrl" in result.error
    uia.Click.assert_not_called()
    uia.ReleaseKey.assert_not_called()


@pytest.mark.parametrize(
    "drag,should_drag", [(True, True), (False, False), ("true", True), ("false", False)]
)
def test_move_dispatch_uses_normalized_booleans_and_coordinates(
    dispatch, desktop_and_uia, drag, should_drag
):
    desktop, uia = desktop_and_uia
    # Validation must dispatch its normalized values, not the original strings.
    result = dispatch("move_tool", {"loc": ["10", "20"], "drag": drag}, desktop)

    assert result.is_success
    if should_drag:
        uia.DragTo.assert_called_once_with(1, 2, 10, 20)
        uia.MoveTo.assert_not_called()
    else:
        uia.MoveTo.assert_called_once_with(10, 20, moveSpeed=10)
        uia.DragTo.assert_not_called()


@pytest.mark.parametrize(
    "tool_name,params",
    [
        ("type_tool", {"loc": [10, 20], "text": "hello", "clear": "invalid"}),
        ("type_tool", {"loc": [10, 20], "text": "hello", "press_enter": None}),
        ("multi_select_tool", {"elements": [[10, 20]], "press_ctrl": "invalid"}),
        ("move_tool", {"loc": [10, 20], "drag": "invalid"}),
    ],
)
def test_invalid_boolean_is_rejected_before_input(dispatch, desktop_and_uia, tool_name, params):
    desktop, uia = desktop_and_uia
    result = dispatch(tool_name, params, desktop)

    assert not result.is_success
    assert "validation failed" in result.error
    assert uia.method_calls == []


def test_multi_edit_still_clears_each_field(dispatch, desktop_and_uia):
    desktop, uia = desktop_and_uia
    result = dispatch(
        "multi_edit_tool", {"elements": [[10, 20, "first"], [30, 40, "second"]]}, desktop
    )

    assert result.is_success
    assert uia.Click.call_args_list == [call(10, 20), call(30, 40)]
    assert [entry.args[0] for entry in uia.SendKeys.call_args_list] == [
        "{Ctrl}a",
        "{Back}",
        "first",
        "{Ctrl}a",
        "{Back}",
        "second",
    ]


@pytest.mark.parametrize(
    "flag,enabled",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
        ("TRUE", True),
        ("FALSE", False),
    ],
)
def test_direct_desktop_flags_are_parsed_before_input(desktop_and_uia, flag, enabled):
    desktop, uia = desktop_and_uia
    desktop.type((10, 20), "hello", clear=flag, press_enter=flag)
    expected_keys = ["{Ctrl}a", "{Back}", "hello", "{Enter}"] if enabled else ["hello"]
    assert [entry.args[0] for entry in uia.SendKeys.call_args_list] == expected_keys

    uia.reset_mock()
    desktop.multi_select(press_ctrl=flag, elements=[(10, 20)])
    assert uia.PressKey.call_count == int(enabled)
    assert uia.ReleaseKey.call_count == int(enabled)


@pytest.mark.parametrize("flag", ["invalid", None, [], 2])
@pytest.mark.parametrize("parameter", ["clear", "press_enter", "press_ctrl"])
def test_direct_invalid_flags_are_rejected_before_input(desktop_and_uia, parameter, flag):
    desktop, uia = desktop_and_uia

    with pytest.raises(ValidationError):
        if parameter == "press_ctrl":
            desktop.multi_select(elements=[(10, 20)], press_ctrl=flag)
        else:
            desktop.type((10, 20), "hello", **{parameter: flag})

    assert uia.method_calls == []


@pytest.mark.parametrize("caret_position", ["none", "idle"])
def test_direct_typing_accepts_both_unchanged_caret_names(desktop_and_uia, caret_position):
    desktop, uia = desktop_and_uia
    desktop.type((10, 20), "hello", caret_position=caret_position)
    uia.SendKeys.assert_called_once_with("hello", interval=0.01, waitTime=0.05)


def test_direct_multiselect_rejects_invalid_coordinates_before_input(desktop_and_uia):
    desktop, uia = desktop_and_uia
    with pytest.raises(ValidationError):
        desktop.multi_select(press_ctrl=True, elements=[(10, 20), 30])
    assert uia.method_calls == []
