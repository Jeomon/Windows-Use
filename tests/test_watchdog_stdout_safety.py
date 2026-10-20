"""The UIA watchdog handlers must never write to stdout.

stdout is the transport for newline delimited JSON-RPC. Any MCP server (or
other stdout based protocol harness) embedding windows-use breaks as soon as a
focus/structure/property callback raises, because the stray ``print`` lands in
the middle of the protocol stream.

The handlers must report failures through the logger (stderr) instead.
"""

import logging
import weakref

from windows_use.agent.watchdog import event_handlers


class _Parent:
    """Minimal stand-in for the watchdog service."""

    def __init__(self):
        def _boom(*_args):
            raise RuntimeError("callback exploded")

        self._focus_callback = _boom
        self._structure_callback = _boom
        self._property_callback = _boom


class _Stub:
    """Stands in for the COM object; the handlers only use ``_parent``."""

    def __init__(self, parent):
        self._parent = weakref.ref(parent)


def test_focus_changed_handler_does_not_write_to_stdout(capsys):
    parent = _Parent()
    event_handlers.FocusChangedEventHandler.HandleFocusChangedEvent(_Stub(parent), None)
    assert capsys.readouterr().out == ""


def test_structure_changed_handler_does_not_write_to_stdout(capsys):
    parent = _Parent()
    event_handlers.StructureChangedEventHandler.HandleStructureChangedEvent(
        _Stub(parent), None, None, None
    )
    assert capsys.readouterr().out == ""


def test_property_changed_handler_does_not_write_to_stdout(capsys):
    parent = _Parent()
    event_handlers.PropertyChangedEventHandler.HandlePropertyChangedEvent(
        _Stub(parent), None, None, None
    )
    assert capsys.readouterr().out == ""


def test_handlers_report_failures_through_the_logger(caplog):
    parent = _Parent()
    with caplog.at_level(logging.WARNING, logger=event_handlers.__name__):
        event_handlers.FocusChangedEventHandler.HandleFocusChangedEvent(_Stub(parent), None)
    assert any("focus callback" in record.message for record in caplog.records)
