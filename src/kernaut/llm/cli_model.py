"""Send model requests through a signed-in ``claude`` or ``codex`` CLI subscription.

Neither CLI exposes custom function calling. Instead, the prompt describes the
tools, and each reply must follow this JSON schema:
{content, tool_calls[{name, arguments_json}]}.
Each request is stateless, so the adapter sends the full transcript on every turn.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from .base import AssistantReply, ConversationMessage, LanguageModel, RequestedTool

REPLY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments_json": {"type": "string"},
                },
                "required": ["name", "arguments_json"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["content", "tool_calls"],
    "additionalProperties": False,
}

PROTOCOL = """\
You are driving a tool-using loop. You cannot run tools yourself. Instead, request them in
your reply: put each call in `tool_calls` with the tool `name` and its arguments encoded as a
JSON object string in `arguments_json`. Put any prose in `content`. Leave `tool_calls` empty
only when you are finished. Results come back as TOOL RESULT blocks on the next turn.

Available tools (JSON schema of each tool's arguments):
{tools}"""


def render_transcript(messages: list[ConversationMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        if message.role == "tool":
            parts.append(f"### TOOL RESULT ({message.tool_call_id})\n{message.content}")
            continue
        text = message.content
        for call in message.tool_calls:
            text += f"\n[requested {call.name}({json.dumps(call.arguments)}) id={call.call_id}]"
        parts.append(f"### {message.role.upper()}\n{text.strip()}")
    parts.append("### ASSISTANT (your turn, reply per the JSON schema)")
    return "\n\n".join(parts)


def parse_reply(raw: Any) -> AssistantReply:
    if isinstance(raw, str):
        # codex -o can repeat the final message. Read only the first JSON object.
        raw, _ = json.JSONDecoder().raw_decode(raw.strip())
    calls = [
        RequestedTool(
            call_id=f"call_{uuid.uuid4().hex[:12]}",
            name=call["name"],
            arguments=json.loads(call.get("arguments_json") or "{}"),
        )
        for call in raw.get("tool_calls", [])
    ]
    return AssistantReply(
        content=raw.get("content", ""),
        tool_calls=calls,
        stop_reason="tool_use" if calls else "end_turn",
    )


RETRY_BASE_SECONDS = 30.0


class SubscriptionCLIModel(LanguageModel):
    def __init__(
        self, provider: str, model: str, *, timeout_seconds: float, max_retries: int
    ) -> None:
        if provider not in {"claude-code", "codex"}:
            raise ValueError(f"Unsupported CLI provider: {provider}")
        self.provider = provider
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def complete(
        self,
        messages: list[ConversationMessage],
        tools: list[dict[str, Any]],
        system_prompt: str,
    ) -> AssistantReply:
        tool_specs = json.dumps(
            [
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "arguments": tool["input_schema"],
                }
                for tool in tools
            ],
            indent=1,
        )
        system = f"{system_prompt}\n\n{PROTOCOL.format(tools=tool_specs)}"
        prompt = render_transcript(messages)
        for attempt in range(self.max_retries + 1):
            try:
                if self.provider == "claude-code":
                    return self._claude(system, prompt)
                return self._codex(system, prompt)
            except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, TypeError) as error:
                if attempt == self.max_retries:
                    raise
                # Wait before retrying so a brief rate limit or network failure does not end the
                # campaign.
                delay = RETRY_BASE_SECONDS * 2**attempt
                print(
                    f"[llm] {self.provider} call failed ({error}); retrying in {delay:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def _run(self, command: list[str], stdin: str, cwd: str, drop_env: str) -> str:
        # Remove the API key so the CLI uses the signed-in subscription account.
        env = {key: value for key, value in os.environ.items() if key != drop_env}
        result = subprocess.run(
            command,
            input=stdin,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env,
            timeout=self.timeout_seconds,
        )
        if result.returncode != 0:
            # The CLI reports usage limits, authentication failures, and API errors on stdout or
            # stderr.
            detail = (result.stderr.strip() or result.stdout.strip())[-500:]
            raise subprocess.SubprocessError(
                f"{command[0]} exited with status {result.returncode}: {detail}"
            )
        return result.stdout

    def _claude(self, system: str, prompt: str) -> AssistantReply:
        command = [
            "claude", "-p",
            "--output-format", "json",
            "--json-schema", json.dumps(REPLY_SCHEMA),
            "--system-prompt", system,
            "--tools", "",
            "--strict-mcp-config",
            "--setting-sources", "",
            "--no-session-persistence",
        ]  # fmt: skip
        if self.model:
            command += ["--model", self.model]
        with tempfile.TemporaryDirectory() as workdir:
            payload = json.loads(self._run(command, prompt, workdir, "ANTHROPIC_API_KEY"))
        if payload.get("is_error"):
            raise subprocess.SubprocessError(payload.get("result", "claude returned an error"))
        reply = parse_reply(payload.get("structured_output") or payload["result"])
        usage = payload.get("usage") or {}
        reply.usage = {
            # The CLI caches the transcript sent on each turn. Most input tokens appear in cache
            # fields.
            "input_tokens": sum(
                usage.get(key) or 0
                for key in (
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            ),
            "output_tokens": usage.get("output_tokens"),
        }
        return reply

    def _codex(self, system: str, prompt: str) -> AssistantReply:
        with tempfile.TemporaryDirectory() as workdir:
            schema = Path(workdir, "schema.json")
            schema.write_text(json.dumps(REPLY_SCHEMA))
            output = Path(workdir, "reply.json")
            command = [
                "codex", "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox", "read-only",
                "--color", "never",
                "--output-schema", str(schema),
                "-o", str(output),
            ]  # fmt: skip
            if self.model:
                command += ["-m", self.model]
            command.append("-")
            self._run(command, f"{system}\n\n{prompt}", workdir, "OPENAI_API_KEY")
            return parse_reply(output.read_text())
