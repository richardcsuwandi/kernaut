"""Public interfaces for task, context, evaluator, and baseline extensions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from kernaut.agent import HarnessTools, SynthesisController
from kernaut.agent.controller import CampaignResult
from kernaut.archive import CandidateStore
from kernaut.config import CampaignConfig
from kernaut.evaluation.gp import Dataset
from kernaut.evaluation.novelty import NoveltyPolicy
from kernaut.execution import KernelExecutor
from kernaut.extensions import load_extension
from kernaut.llm.base import LanguageModel
from kernaut.models import CandidateBundle, CandidateOrigin, EvaluationRecord
from kernaut.verification import VerificationPolicy, Verifier


class CandidateEvaluator(Protocol):
    """Score candidates on training data. Larger scores must mean better results."""

    def evaluate(self, candidate: CandidateBundle, dataset: Dataset) -> EvaluationRecord: ...


@dataclass(frozen=True)
class Task:
    """A task's training data, proposer context, and deterministic evaluator.

    Keep held-out test data outside this object. An episodic evaluator can keep
    training episodes internally and use dataset only as a representative input.
    """

    dataset: Dataset
    context: str
    evaluator: CandidateEvaluator
    verification_policy: VerificationPolicy | None = None
    novelty_policy: NoveltyPolicy | None = None
    parameter_evaluator: CandidateEvaluator | None = None


def load_task(name: str, executor: KernelExecutor, options: dict[str, Any]) -> Task:
    task = load_extension("kernaut.tasks", name)(executor, options)
    if not isinstance(task, Task):
        raise TypeError("Task factories must return a kernaut.tasks.Task")
    task.dataset.arrays()
    if not task.context.strip():
        raise ValueError("Task context must not be empty")
    return task


def load_baselines(name: str, task: Task) -> list[CandidateBundle]:
    candidates = list(load_extension("kernaut.baselines", name)(task))
    if not candidates or not all(isinstance(c, CandidateBundle) for c in candidates):
        raise TypeError("Baseline factories must return a nonempty iterable of CandidateBundle")
    return candidates


def run_task(
    task: Task,
    model: LanguageModel,
    store: CandidateStore,
    executor: KernelExecutor,
    *,
    campaign: CampaignConfig | None = None,
    baselines: Iterable[CandidateBundle] = (),
    run_id: str | None = None,
) -> CampaignResult:
    """Run a conversational search with the same verification gate as built-in tasks.

    Baseline candidates are verified before evaluation. This API does not retune
    supplied baseline parameters or reproduce the paper's evolutionary protocol.
    """
    task.dataset.arrays()
    if not task.context.strip():
        raise ValueError("Task context must not be empty")
    settings = campaign or CampaignConfig()
    verifier = Verifier(
        executor,
        task.verification_policy
        or VerificationPolicy(seed=settings.seed, input_dimension=len(task.dataset.x[0])),
    )
    for supplied in baselines:
        candidate = supplied.model_copy(update={"origin": CandidateOrigin.BASELINE})
        evidence = verifier.verify(candidate)
        if not evidence.accepted:
            raise ValueError(f"Baseline {candidate.name!r} failed verification")
        evaluation = task.evaluator.evaluate(candidate, task.dataset)
        if evaluation.candidate_id != candidate.candidate_id:
            raise ValueError("Evaluator returned a record for a different candidate")
        store.add_candidate(candidate)
        store.add_evidence(evidence)
        store.add_evaluation(evaluation)
    tools = HarnessTools(
        store,
        verifier,
        task.evaluator,
        task.dataset,
        novelty_policy=task.novelty_policy,
        parameter_evaluator=task.parameter_evaluator,
    )
    return SynthesisController(
        model,
        tools,
        store,
        max_rounds=settings.max_rounds,
        max_tool_calls=settings.max_tool_calls,
        seed=settings.seed,
    ).run(task.context, run_id=run_id)
