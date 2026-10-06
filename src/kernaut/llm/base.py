from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field


class RequestedTool(BaseModel):
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[RequestedTool] = Field(default_factory=list)
    tool_call_id: str | None = None


class AssistantReply(BaseModel):
    content: str = ""
    tool_calls: list[RequestedTool] = Field(default_factory=list)
    stop_reason: str = ""
    usage: dict[str, int | None] | None = None


class LanguageModel(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[ConversationMessage],
        tools: list[dict[str, Any]],
        system_prompt: str,
    ) -> AssistantReply:
        raise NotImplementedError
