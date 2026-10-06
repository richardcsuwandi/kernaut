from kernaut.llm.base import ConversationMessage, RequestedTool
from kernaut.llm.cli_model import parse_reply, render_transcript


def test_parse_reply_decodes_arguments_and_repeated_codex_output():
    raw = '{"content":"","tool_calls":[{"name":"add","arguments_json":"{\\"a\\":1}"}]}' * 2
    reply = parse_reply(raw)
    assert reply.tool_calls[0].name == "add"
    assert reply.tool_calls[0].arguments == {"a": 1}
    assert reply.stop_reason == "tool_use"
    assert parse_reply({"content": "done", "tool_calls": []}).stop_reason == "end_turn"


def test_render_transcript_includes_calls_and_results():
    call = RequestedTool(call_id="c1", name="add", arguments={"a": 1})
    text = render_transcript(
        [
            ConversationMessage(role="user", content="go"),
            ConversationMessage(role="assistant", tool_calls=[call]),
            ConversationMessage(role="tool", tool_call_id="c1", content="2"),
        ]
    )
    assert "add" in text and "TOOL RESULT (c1)\n2" in text


def test_cli_failure_reports_reason_and_backs_off(monkeypatch):
    import subprocess

    import pytest

    from kernaut.llm import cli_model

    calls, sleeps = [], []

    def failing_run(*args, **kwargs):
        calls.append(1)
        out = '{"result":"usage limit reached"}'
        return subprocess.CompletedProcess(args, 1, stdout=out, stderr="")

    monkeypatch.setattr(cli_model.subprocess, "run", failing_run)
    monkeypatch.setattr(cli_model.time, "sleep", sleeps.append)
    model = cli_model.SubscriptionCLIModel("claude-code", "", timeout_seconds=5, max_retries=2)
    with pytest.raises(subprocess.SubprocessError, match="usage limit reached"):
        model.complete([ConversationMessage(role="user", content="hi")], [], "system")
    assert len(calls) == 3 and sleeps == [30.0, 60.0]
