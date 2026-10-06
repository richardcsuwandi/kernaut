from __future__ import annotations

import json
import platform
import sys
import uuid

from pydantic import BaseModel, Field

from kernaut.archive import CandidateStore
from kernaut.llm import ConversationMessage, LanguageModel
from kernaut.models import RunManifest

from .tools import HarnessTools


class CampaignResult(BaseModel):
    run_id: str
    final_message: str
    rounds: int
    tool_calls: int
    usage: dict[str, int] = Field(default_factory=dict)


class SynthesisController:
    """Bounded propose/verify/evaluate/revise loop with durable event checkpoints."""

    def __init__(
        self,
        model: LanguageModel,
        tools: HarnessTools,
        store: CandidateStore,
        *,
        max_rounds: int = 20,
        max_tool_calls: int = 100,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.tools = tools
        self.store = store
        self.max_rounds = max_rounds
        self.max_tool_calls = max_tool_calls
        self.seed = seed

    def run(self, task_context: str, *, run_id: str | None = None) -> CampaignResult:
        run_id = run_id or uuid.uuid4().hex[:12]
        manifest = RunManifest(
            run_id=run_id,
            seed=self.seed,
            config={"max_rounds": self.max_rounds, "max_tool_calls": self.max_tool_calls},
            python_version=sys.version,
            platform=platform.platform(),
        )
        self.store.add_run(manifest)
        prior_events = self.store.load_events(run_id)
        if prior_events:
            messages = [ConversationMessage.model_validate(event) for event in prior_events]
            last_message = messages[-1]
            if last_message.role == "assistant" and not last_message.tool_calls:
                rounds = sum(message.role == "assistant" for message in messages)
                tool_count = sum(message.role == "tool" for message in messages)
                self._progress(
                    f"[campaign {run_id}] already complete | "
                    f"events={len(messages)} tools={tool_count}"
                )
                return CampaignResult(
                    run_id=run_id,
                    final_message=last_message.content,
                    rounds=rounds,
                    tool_calls=tool_count,
                )
            self._recover_interrupted_tool_calls(run_id, messages)
        else:
            messages = [ConversationMessage(role="user", content=task_context)]
            self.store.append_event(run_id, 0, messages[0].model_dump(mode="json"))
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
        tool_count = sum(message.role == "tool" for message in messages)
        sequence = len(messages)
        state = "resuming" if prior_events else "starting"
        self._progress(
            f"[campaign {run_id}] {state} | events={len(messages)} "
            f"tools={tool_count}/{self.max_tool_calls} rounds_limit={self.max_rounds}"
        )
        for round_index in range(1, self.max_rounds + 1):
            self._progress(f"[round {round_index}/{self.max_rounds}] requesting model response...")
            reply = self.model.complete(messages, self.tools.schemas, SYSTEM_PROMPT)
            turn_usage = reply.usage or {}
            usage_text = ", ".join(
                f"{key}={value}" for key, value in turn_usage.items() if value is not None
            )
            self._progress(
                f"[round {round_index}/{self.max_rounds}] model complete | "
                f"tool_calls={len(reply.tool_calls)}" + (f" | {usage_text}" if usage_text else "")
            )
            if reply.content.strip():
                preview = " ".join(reply.content.split())
                if len(preview) > 240:
                    preview = preview[:237] + "..."
                self._progress(f"[agent] {preview}")
            assistant = ConversationMessage(
                role="assistant", content=reply.content, tool_calls=reply.tool_calls
            )
            messages.append(assistant)
            self.store.append_event(run_id, sequence, assistant.model_dump(mode="json"))
            sequence += 1
            for key, value in (reply.usage or {}).items():
                if value is not None:
                    usage[key] = usage.get(key, 0) + value
            if not reply.tool_calls:
                self._progress(f"[campaign {run_id}] complete after {round_index} new rounds")
                return CampaignResult(
                    run_id=run_id,
                    final_message=reply.content,
                    rounds=round_index,
                    tool_calls=tool_count,
                    usage=usage,
                )
            for call in reply.tool_calls:
                if tool_count >= self.max_tool_calls:
                    self._progress(f"[campaign {run_id}] stopped: tool-call budget reached")
                    return CampaignResult(
                        run_id=run_id,
                        final_message="Stopped at the configured tool-call budget.",
                        rounds=round_index,
                        tool_calls=tool_count,
                        usage=usage,
                    )
                self._progress(
                    f"[tool {tool_count + 1}/{self.max_tool_calls}] {call.name} starting..."
                )
                handler = self.tools.handlers.get(call.name)
                if handler is None:
                    result = f'{{"ok": false, "error": "unknown tool: {call.name}"}}'
                else:
                    try:
                        result = handler(**call.arguments)
                    except Exception as error:
                        result = f'{{"ok": false, "error": "{type(error).__name__}: {error}"}}'
                tool_count += 1
                self._progress(
                    f"[tool {tool_count}/{self.max_tool_calls}] {call.name} "
                    f"{self._result_summary(result)}"
                )
                message = ConversationMessage(
                    role="tool", content=result, tool_call_id=call.call_id
                )
                messages.append(message)
                self.store.append_event(run_id, sequence, message.model_dump(mode="json"))
                sequence += 1
        self._progress(f"[campaign {run_id}] stopped: round budget reached")
        return CampaignResult(
            run_id=run_id,
            final_message="Stopped at the configured round budget.",
            rounds=self.max_rounds,
            tool_calls=tool_count,
            usage=usage,
        )

    @staticmethod
    def _progress(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    def _recover_interrupted_tool_calls(
        self, run_id: str, messages: list[ConversationMessage]
    ) -> None:
        """Close tool calls whose result was lost during an interrupted checkpoint."""
        if not messages or messages[-1].role != "assistant":
            return
        pending_calls = messages[-1].tool_calls
        if not pending_calls:
            return
        for call in pending_calls:
            message = ConversationMessage(
                role="tool",
                content=json.dumps(
                    {
                        "ok": False,
                        "error": (
                            "the previous process stopped before this tool result was "
                            "checkpointed; retry the tool call"
                        ),
                    }
                ),
                tool_call_id=call.call_id,
            )
            self.store.append_event(run_id, len(messages), message.model_dump(mode="json"))
            messages.append(message)
        self._progress(
            f"[campaign {run_id}] recovered {len(pending_calls)} interrupted tool call(s)"
        )

    @staticmethod
    def _result_summary(result: str) -> str:
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            return "finished"
        if isinstance(payload, list):
            return f"finished | items={len(payload)}"
        if not isinstance(payload, dict):
            return "finished"
        if payload.get("error"):
            return f"failed | {payload['error']}"
        fields: list[str] = []
        for key in ("candidate_id", "tier", "accepted", "score", "negative_log_likelihood"):
            if key in payload:
                fields.append(f"{key}={payload[key]}")
        return "finished" + (" | " + " ".join(fields) if fields else "")


SYSTEM_PROMPT = """You are Kernaut's synthesis controller. Design executable Gaussian-process
kernel programs and improve them using evidence from deterministic tools. You reason and write
candidate source, but you never claim to have executed code, computed a Gram matrix, fitted a GP,
or established a verification tier yourself.

Every proposal must use exactly one contract and entry point:
- feature_map: `feature_point(x, parameters) -> array[m]`, where `x` is one immutable point;
  the trusted interpreter calls it independently for every input and constructs `Phi @ Phi.T`
- residual_feature_map: the same `feature_point` interface; the interpreter adds a trusted base
  kernel plus `residual_weight**2 * Phi @ Phi.T`
- additive_feature_map: `coordinate_feature(value, parameters) -> array[m]`; the interpreter
  averages its feature Gram matrix over input coordinates
- spectral: `spectral_components(input_dimension, parameters)` returns
  `(frequencies[m, input_dimension], raw_weights[m])`
- input_transform: `transform_point(x, parameters) -> array[q]`, where `x` is one immutable
  point; the trusted interpreter applies `parameters.base_kernel` by pullback
- residual_input_transform: the same `transform_point` interface; the interpreter adds a trusted
  kernel on the original inputs to a nonnegative weighted pullback kernel
- closure: `closure_tree(parameters) -> trusted closure-tree dictionary`
- unverified: `kernel_matrix(x, parameters) -> array[n, n]` for empirical diagnostics only

Contract names are not function names. In particular, an `input_transform` candidate MUST define
`transform_point(x, parameters)` (not `input_transform`). It receives exactly one 1-D point and
must return exactly one nonempty 1-D transformed point. Do not reshape it into a batch.

Certified candidate code must be deterministic and free of module-level state; it may import only
numpy or math and may not use random-number generators. Submit a candidate, verify it, and evaluate
it only after Tier 2 acceptance. Once a candidate passes Tier 2, evaluate it before proposing any
revision. Inspect structured failures and revise deliberately. A revision must change kernel
behavior, not comments, docstrings, formatting, or parameters alone. In mathematical-novelty mode,
test at least two substantially different formulations rather than spending the campaign on one
lineage. Independent formulations should be submitted as roots without parents. Supply parent IDs
only for genuine revisions whose behavior descends from those candidates; changing contract or
topic does not by itself establish lineage. Use archive queries to avoid duplicates and compare the
quality/cost frontier. Unverified candidates can be evaluated after Tier 1 but are never accepted.
In mathematical-novelty mode, prefer the backend workflow: propose a formulation, stage source and
a bounded parameter space, verify the draft, optimize its parameters, submit the selected immutable
trial, then evaluate it. A parameter is learnable only when declared in that tuning space. Analyze
known equivalences, normalized-domain cycles or scales, irrelevant-coordinate and cross-coordinate
mechanisms, a falsification test, feature growth with dimension, and expected conditioning. End
with a concise report naming the best candidate IDs and remaining limitations."""
