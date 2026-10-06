"""Small, installable examples of all three extension interfaces."""

from __future__ import annotations

import numpy as np

from kernaut.evaluation.gp import Dataset, GaussianProcessEvaluator
from kernaut.llm.base import AssistantReply, LanguageModel, RequestedTool
from kernaut.models import CandidateBundle
from kernaut.tasks import Task

SOURCE = "def feature_point(x, parameters): return [1.0] + list(x)"


def make_task(executor, options):
    points = int(options.get("points", 12))
    if points < 3:
        raise ValueError("points must be at least 3")
    x = np.linspace(-1.0, 1.0, points)
    return Task(
        dataset=Dataset(x=x[:, None].tolist(), y=np.sin(3.0 * x).tolist()),
        context="Find a kernel for a smooth one-dimensional response on [-1, 1].",
        evaluator=GaussianProcessEvaluator(executor),
    )


def make_baselines(task):
    # A fixed reference for the example, not a tuned paper baseline.
    return [CandidateBundle(name="linear_demo", contract="feature_map", source=SOURCE)]


class OfflineDemoModel(LanguageModel):
    """Submit, verify, and score one fixed candidate with no network access."""

    def complete(self, messages, tools, system_prompt):
        import json

        replies = [m for m in messages if m.role == "tool"]
        if not replies:
            return AssistantReply(
                tool_calls=[
                    RequestedTool(
                        call_id="submit-demo",
                        name="submit_candidate",
                        arguments={
                            "name": "demo_features",
                            "contract": "feature_map",
                            "source": SOURCE,
                        },
                    )
                ]
            )
        candidate_id = json.loads(replies[0].content)["candidate_id"]
        if len(replies) == 1:
            name = "verify_candidate"
        elif len(replies) == 2:
            name = "evaluate_candidate"
        else:
            return AssistantReply(content="Offline extension example complete.")
        return AssistantReply(
            tool_calls=[
                RequestedTool(
                    call_id=f"{name}-demo",
                    name=name,
                    arguments={"candidate_id": candidate_id},
                )
            ]
        )


def make_model(config):
    return OfflineDemoModel()
