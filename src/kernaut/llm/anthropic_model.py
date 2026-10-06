from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from .base import AssistantReply, ConversationMessage, LanguageModel, RequestedTool


class AnthropicModel(LanguageModel):
    def __init__(
        self, model: str, *, api_key: str | None, timeout_seconds: float, max_retries: int = 4
    ) -> None:
        try:
            anthropic = import_module("anthropic")
        except ImportError as error:
            raise RuntimeError("Install kernaut[anthropic] to use Anthropic") from error
        if not api_key:
            raise ValueError("Anthropic API key is not configured")
        self.model = model
        self.client = anthropic.Anthropic(
            api_key=api_key, timeout=timeout_seconds, max_retries=max_retries
        )

    def complete(
        self,
        messages: list[ConversationMessage],
        tools: list[dict[str, Any]],
        system_prompt: str,
    ) -> AssistantReply:
        wire: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                wire.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id,
                                "content": message.content,
                            }
                        ],
                    }
                )
            elif message.role == "assistant" and message.tool_calls:
                blocks: list[dict[str, Any]] = []
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
                blocks.extend(
                    {
                        "type": "tool_use",
                        "id": call.call_id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                    for call in message.tool_calls
                )
                wire.append({"role": "assistant", "content": blocks})
            else:
                wire.append({"role": message.role, "content": message.content})
        create = cast(Any, self.client.messages.create)
        response = create(
            model=self.model,
            max_tokens=8192,
            system=system_prompt,
            messages=wire,
            tools=tools,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        calls = [
            RequestedTool(call_id=block.id, name=block.name, arguments=block.input)
            for block in response.content
            if block.type == "tool_use"
        ]
        return AssistantReply(
            content=text,
            tool_calls=calls,
            stop_reason=response.stop_reason or "",
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )
