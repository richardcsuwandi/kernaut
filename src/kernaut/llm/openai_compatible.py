from __future__ import annotations

import importlib.util
import json
import sys
import time
from typing import Any, cast

import openai

from .base import AssistantReply, ConversationMessage, LanguageModel, RequestedTool


def _transient_error_types() -> tuple[type[BaseException], ...]:
    """Return SDK request errors and network errors that can occur during streaming.

    The OpenAI SDK does not wrap failures raised by the HTTP layer during streaming.
    """
    types: list[type[BaseException]] = [
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.APIStatusError,
        # _consume_stream combines fragments of the streamed `arguments` string.
        # A missing or corrupt fragment can produce invalid JSON without a network exception.
        # Treat invalid JSON like the other temporary stream failures listed above.
        # One corrupt fragment should not end the campaign.
        json.JSONDecodeError,
    ]
    for module_name in ("httpx", "httpx2"):
        if importlib.util.find_spec(module_name) is None:
            continue
        module = importlib.import_module(module_name)
        cls = getattr(module, "HTTPError", None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            types.append(cls)
    return tuple(types)


_TRANSIENT_ERRORS = _transient_error_types()


class OpenAICompatibleModel(LanguageModel):
    """Streaming adapter for OpenAI and compatible gateways such as ModelScope."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        base_url: str | None,
        timeout_seconds: float,
        max_retries: int,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        self.client = openai.OpenAI(
            api_key=api_key or "unused",
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )

    def complete(
        self,
        messages: list[ConversationMessage],
        tools: list[dict[str, Any]],
        system_prompt: str,
    ) -> AssistantReply:
        wire_messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for message in messages:
            entry: dict[str, Any] = {"role": message.role, "content": message.content or None}
            if message.role == "assistant" and message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ]
            if message.role == "tool":
                entry["tool_call_id"] = message.tool_call_id
            wire_messages.append(entry)

        wire_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]
        kwargs = {"extra_body": self.extra_body} if self.extra_body else {}
        for attempt in range(self.max_retries + 1):
            try:
                return self._consume_stream(wire_messages, wire_tools, kwargs)
            except _TRANSIENT_ERRORS as error:
                if isinstance(error, openai.APIStatusError) and (
                    error.status_code not in {408, 409, 429} and error.status_code < 500
                ):
                    raise
                if attempt == self.max_retries:
                    raise
                delay = self._backoff_delay(attempt, error)
                print(
                    f"[llm] transient {type(error).__name__}; retrying in {delay:.0f}s "
                    f"({attempt + 1}/{self.max_retries})",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable")

    @staticmethod
    def _backoff_delay(attempt: int, error: BaseException) -> float:
        headers = getattr(getattr(error, "response", None), "headers", None) or {}
        raw = headers.get("retry-after")
        if raw is not None:
            try:
                return min(max(float(raw), 1.0), 120.0)
            except ValueError:
                pass
        return min(2**attempt * 2, 32)

    def _consume_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> AssistantReply:
        create = cast(Any, self.client.chat.completions.create)
        started = time.monotonic()
        stream = create(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )
        content = ""
        pending: dict[int, dict[str, Any]] = {}
        stop_reason = ""
        usage: dict[str, int | None] | None = None
        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            if chunk_count == 1:
                print(
                    f"[llm] stream connected after {time.monotonic() - started:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
            elif chunk_count % 100 == 0:
                print(f"[llm] received {chunk_count} chunks...", file=sys.stderr, flush=True)
            if chunk.usage is not None:
                details = getattr(chunk.usage, "prompt_tokens_details", None)
                usage = {
                    "input_tokens": chunk.usage.prompt_tokens,
                    "output_tokens": chunk.usage.completion_tokens,
                    "cache_read_tokens": getattr(details, "cached_tokens", None),
                }
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            content += delta.content or ""
            for fragment in delta.tool_calls or []:
                item = pending.setdefault(fragment.index, {"id": "", "name": "", "arguments": ""})
                item["id"] = fragment.id or item["id"]
                if fragment.function:
                    item["name"] = fragment.function.name or item["name"]
                    item["arguments"] += fragment.function.arguments or ""
            stop_reason = choice.finish_reason or stop_reason
        calls = [
            RequestedTool(
                call_id=item["id"],
                name=item["name"],
                arguments=json.loads(item["arguments"] or "{}"),
            )
            for _, item in sorted(pending.items())
        ]
        print(
            f"[llm] stream complete | chunks={chunk_count} "
            f"elapsed={time.monotonic() - started:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        return AssistantReply(
            content=content, tool_calls=calls, stop_reason=stop_reason, usage=usage
        )
