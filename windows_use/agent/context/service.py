import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from getpass import getuser
from importlib.resources import files
from pathlib import Path
from typing import Literal

import windows_use.uia as uia
from windows_use.agent.desktop.service import Desktop
from windows_use.agent.desktop.views import Browser
from windows_use.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ImageMessage,
    SystemMessage,
    ToolMessage,
)
from windows_use.providers import BaseChatLLM, TokenUsage
from windows_use.providers.events import LLMEvent, LLMEventType

_template_cache: dict[str, str] = {}

_NON_TOOL_PARAMS = {"thought"}
_COMPACTION_CONTENT_LIMIT = 2000
_COMPACTION_PARAMS_LIMIT = 4000
_COMPACTION_THOUGHT_LIMIT = 1000
_TRUNCATED = "...[TRUNCATED]"


def _bounded_text(value: object, limit: int) -> str:
    """Render JSON values before truncating, preserving their types in history."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) > limit:
        return text[: limit - len(_TRUNCATED)] + _TRUNCATED
    return text


def _load_template(filename: str) -> str:
    """Load a prompt template from disk, caching after first read."""
    if filename not in _template_cache:
        _template_cache[filename] = Path(
            files("windows_use.agent.context").joinpath("prompt", filename)
        ).read_text(encoding="utf-8")
    return _template_cache[filename]


class Context:
    def __init__(self, llm: BaseChatLLM):
        self.token_usage: TokenUsage = TokenUsage()
        self.llm = llm

    def _build_system_prompt(
        self,
        mode: Literal["flash", "normal"],
        desktop: Desktop,
        browser: Browser,
        max_steps: int,
        instructions: list[str] = [],
    ) -> str:
        width, height = uia.GetScreenSize()
        match mode:
            case "flash":
                template = _load_template("system_flash.md")
                os_version = desktop.get_windows_version()
                return template.format(
                    **{
                        "max_steps": max_steps,
                        "datetime": datetime.now().strftime("%A, %B %d, %Y"),
                        "os": os_version,
                        "browser": browser.value,
                    }
                )
            case "normal":
                template = _load_template("system.md")
                with ThreadPoolExecutor(max_workers=3) as executor:
                    os_future = executor.submit(desktop.get_windows_version)
                    lang_future = executor.submit(desktop.get_default_language)
                    user_future = executor.submit(desktop.get_user_account_type)
                    os_version = os_future.result()
                    language = lang_future.result()
                    user_account_type = user_future.result()
                return template.format(
                    **{
                        "datetime": datetime.now().strftime("%A, %B %d, %Y"),
                        "instructions": "\n".join(instructions),
                        "download_directory": Path.home().joinpath("Downloads").as_posix(),
                        "os": os_version,
                        "language": language,
                        "browser": browser.value,
                        "home_dir": Path.home().as_posix(),
                        "user": f"{getuser()} ({user_account_type})",
                        "resolution": f"Primary Monitor ({width}x{height}) with DPI Scale: {desktop.get_dpi_scaling()}",
                        "max_steps": max_steps,
                    }
                )
            case _:
                raise ValueError(f"Invalid mode: {mode} (must be 'flash' or 'normal')")

    def _format_history_for_compaction(self, messages: list[BaseMessage]) -> str:
        lines = ["Following is the conversation that needs to be compacted:"]
        for message in messages:
            if isinstance(message, SystemMessage):
                continue
            content = _bounded_text(
                message.content if message.content is not None else "", _COMPACTION_CONTENT_LIMIT
            )
            if isinstance(message, HumanMessage | ImageMessage):
                lines.append(f"USER: {content}")
            elif isinstance(message, AIMessage):
                lines.append(f"ASSISTANT: {content}")
            elif isinstance(message, ToolMessage):
                id = message.id
                name = message.name
                params = message.params
                if "thought" in params:
                    thought = _bounded_text(params["thought"], _COMPACTION_THOUGHT_LIMIT)
                    lines.append(f"THOUGHT: {thought}")
                parts = _bounded_text(
                    {k: v for k, v in params.items() if k not in _NON_TOOL_PARAMS},
                    _COMPACTION_PARAMS_LIMIT,
                )
                lines.append(f"TOOL CALL {id}: {name}({parts})")
                lines.append(f"TOOL RESULT {id}: {content}")
            else:
                pass
        return "\n\n---\n\n".join(lines)

    @property
    def need_compaction(self) -> bool:
        metadata = self.llm.get_metadata()
        context_window = metadata.context_window
        total_tokens = self.token_usage.total_tokens
        return total_tokens > context_window * 0.8

    def compact(
        self,
        messages: list[BaseMessage],
    ) -> str | None:
        """Summarize history, returning None when the provider gives no usable text."""
        llm_response = self.llm.invoke(self._compaction_messages(messages))
        return self._format_compaction_response(llm_response)

    async def acompact(self, messages: list[BaseMessage]) -> str | None:
        """Summarize history without making a synchronous provider call."""
        llm_response = await self.llm.ainvoke(self._compaction_messages(messages))
        return self._format_compaction_response(llm_response)

    def _compaction_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        template = _load_template("compact.md")
        return [
            SystemMessage(content=template),
            HumanMessage(content=self._format_history_for_compaction(messages)),
        ]

    def _format_compaction_response(self, response: LLMEvent | None) -> str | None:
        if (
            response is None
            or response.type != LLMEventType.TEXT
            or not isinstance(response.content, str)
            or not response.content.strip()
        ):
            return None
        # The previous request's token count no longer describes the compacted history.
        self.token_usage = TokenUsage()
        summary = response.content.strip()
        return f"""
# Context Restoration (Previous Session Compacted)

The previous conversation was compacted due to context window limitations. Below is the detailed summary of work done so far.

**CRITICAL: Don't repeat already completed ACTIONS**

---

{summary}

---

Continue work from where the previous session left off. FOCUS only on the remaining tasks.
"""

    def system(
        self,
        mode: Literal["flash", "normal"],
        desktop: Desktop,
        browser: Browser,
        max_steps: int,
        instructions: list[str] = [],
    ) -> SystemMessage:
        content = self._build_system_prompt(
            mode=mode,
            desktop=desktop,
            browser=browser,
            max_steps=max_steps,
            instructions=instructions,
        )
        return SystemMessage(content=content)

    def state(
        self,
        query: str,
        step: int,
        max_steps: int,
        desktop: Desktop,
        nudge: str = "",
    ) -> HumanMessage | ImageMessage:
        desktop_state = desktop.get_state()  # Populates desktop.desktop_state
        content = self._build_state_prompt(
            query=query,
            step=step,
            max_steps=max_steps,
            desktop=desktop,
            nudge=nudge,
        )
        if desktop.use_vision and desktop_state.screenshot:
            return ImageMessage(images=[desktop_state.screenshot], content=content)
        return HumanMessage(content=content)

    def _build_state_prompt(
        self, query: str, step: int, max_steps: int, desktop: Desktop, nudge: str = ""
    ) -> str:
        desktop_state = desktop.desktop_state
        cursor_x, cursor_y = uia.GetCursorPos()
        template = _load_template("state.md")
        loop_warning = f"[Loop Warning]\n{nudge}\n[End of Loop Warning]\n" if nudge else ""
        return template.format(
            **{
                "steps": step,
                "max_steps": max_steps,
                "loop_warning": loop_warning,
                "active_window": desktop_state.active_window_to_string(),
                "windows": desktop_state.windows_to_string(),
                "cursor_location": f"({cursor_x},{cursor_y})",
                "interactive_elements": (
                    desktop_state.tree_state.interactive_elements_to_string()
                    if desktop.use_accessibility
                    else "No accessibility data is available"
                ),
                "scrollable_elements": (
                    desktop_state.tree_state.scrollable_elements_to_string()
                    if desktop.use_accessibility
                    else "No accessibility data is available"
                ),
                "active_desktop": desktop_state.active_desktop_to_string(),
                "desktops": desktop_state.desktops_to_string(),
                "query": query,
            }
        )

    def task(self, task: str) -> HumanMessage:
        return HumanMessage(content=f"TASK: {task}")

    def update_token_usage(self, token_usage: TokenUsage):
        self.token_usage = token_usage
