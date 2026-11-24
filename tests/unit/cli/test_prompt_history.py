from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from windows_use.cli import __main__ as cli_main


def test_create_prompt_session_uses_file_history(tmp_path, monkeypatch):
    config_dir = tmp_path / ".windows-use"
    history_file = config_dir / "history"
    monkeypatch.setattr(cli_main, "WINDOWS_USE_DIR", config_dir)
    monkeypatch.setattr(cli_main, "HISTORY_FILE", history_file)

    with create_pipe_input() as pipe_in:
        monkeypatch.setattr(
            cli_main,
            "PromptSession",
            lambda **kwargs: PromptSession(input=pipe_in, output=DummyOutput(), **kwargs),
        )
        session = cli_main._create_prompt_session()
        assert isinstance(session.history, FileHistory)
        assert session.history.filename == str(history_file)
        assert config_dir.exists()


def test_prompt_task_reuses_session_and_navigates_history(tmp_path):
    config_dir = tmp_path / ".windows-use"
    history_file = config_dir / "history"
    config_dir.mkdir(parents=True, exist_ok=True)
    history = FileHistory(str(history_file))

    with create_pipe_input() as pipe_in:
        session = PromptSession(
            completer=cli_main._CommandCompleter(),
            complete_while_typing=True,
            history=history,
            input=pipe_in,
            output=DummyOutput(),
        )

        pipe_in.send_text("open notepad\r")
        assert cli_main._prompt_task(session=session) == "open notepad"

        pipe_in.send_text("open calculator\r")
        assert cli_main._prompt_task(session=session) == "open calculator"

        # Verify history file recorded both commands in order
        assert list(history.load_history_strings()) == ["open calculator", "open notepad"]

        # Verify Up/Down buffer navigation inside an active event loop
        import asyncio

        async def _check_nav():
            buf = session.default_buffer
            buf.history_backward()
            assert buf.text == "open calculator"
            buf.history_backward()
            assert buf.text == "open notepad"
            buf.history_forward()
            assert buf.text == "open calculator"

        asyncio.run(_check_nav())
