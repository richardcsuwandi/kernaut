from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

from pydantic import BaseModel, Field

from kernaut.archive import CandidateStore
from kernaut.llm import EnsembleModel, LanguageModel
from kernaut.models import CandidateBundle, utc_now

from .controller import CampaignResult, SynthesisController
from .tools import HarnessTools

DEFAULT_NICHES = (
    "input_geometry",
    "spectral_periodic",
    "feature_projection",
    "additive_interaction",
    "multiresolution_local",
    "residual_composite",
    "open_ended",
)


@dataclass(frozen=True)
class Elite:
    candidate: CandidateBundle
    selection_score: float
    cell: tuple[str, str, str]


class EvolutionResult(BaseModel):
    run_id: str
    iterations: int
    completed_iterations: int
    candidate_count_before: int
    candidate_count_after: int
    occupied_cells: int
    campaigns: list[CampaignResult] = Field(default_factory=list)


class QualityDiversityArchive:
    """Select the best archived candidates within groups of similar kernel behavior."""

    def __init__(self, store: CandidateStore) -> None:
        self.store = store

    def elites(self) -> list[Elite]:
        by_cell: dict[tuple[str, str, str], Elite] = {}
        for row in self.store.recent(limit=100_000):
            if row.get("origin") != "discovered" or not row.get("accepted"):
                continue
            candidate = self.store.get_candidate(str(row["candidate_id"]))
            evaluation = self.store.latest_evaluation(str(row["candidate_id"]))
            if candidate is None or evaluation is None:
                continue
            score = evaluation.metadata.get("selection_score", evaluation.score)
            if score is None or not math.isfinite(float(score)):
                continue
            descriptors = evaluation.metadata.get("quality_descriptors", {})
            cell = (
                str(descriptors.get("construction_niche", candidate.contract.value)),
                str(descriptors.get("feature_growth", candidate.feature_growth or "unknown")),
                str(descriptors.get("novelty_band", "unknown")),
            )
            elite = Elite(
                candidate=candidate,
                selection_score=float(score),
                cell=cell,
            )
            previous = by_cell.get(cell)
            if previous is None or elite.selection_score > previous.selection_score:
                by_cell[cell] = elite
        return sorted(by_cell.values(), key=lambda item: item.selection_score, reverse=True)

    def choose_context(
        self,
        rng: random.Random,
        *,
        root_probability: float,
        exploration_probability: float,
        target_niche: str,
    ) -> tuple[str, Elite | None, list[Elite]]:
        elites = self.elites()
        if not elites or rng.random() < root_probability:
            inspirations = rng.sample(elites, k=min(3, len(elites))) if elites else []
            return "root", None, inspirations

        if rng.random() < exploration_probability:
            parent = rng.choice(elites)
        else:
            parent = rng.choice(elites[: min(3, len(elites))])
        cross_niche = [item for item in elites if item.cell[0] != target_niche]
        pool = cross_niche or [item for item in elites if item != parent]
        inspirations = rng.sample(pool, k=min(3, len(pool))) if pool else []
        return "revision", parent, inspirations


class EvolutionarySynthesisController:
    """Run new searches within fixed budgets, using candidates from the archive.

    The archive retains candidates with high scores and different kernel behaviors.
    """

    def __init__(
        self,
        model: LanguageModel,
        tools: HarnessTools,
        store: CandidateStore,
        *,
        iterations: int = 12,
        rounds_per_iteration: int = 10,
        tool_calls_per_iteration: int = 30,
        root_probability: float = 0.35,
        exploration_probability: float = 0.35,
        seed: int = 0,
        niches: tuple[str, ...] = DEFAULT_NICHES,
    ) -> None:
        self.model = model
        self.tools = tools
        self.store = store
        self.iterations = iterations
        self.rounds_per_iteration = rounds_per_iteration
        self.tool_calls_per_iteration = tool_calls_per_iteration
        self.root_probability = root_probability
        self.exploration_probability = exploration_probability
        self.seed = seed
        self.niches = niches

    def run(self, task_context: str, *, run_id: str) -> EvolutionResult:
        rng = random.Random(self.seed)
        archive = QualityDiversityArchive(self.store)
        count_before = len(self.store.recent(limit=100_000))
        campaigns: list[CampaignResult] = []
        for iteration in range(self.iterations):
            target_niche = self._target_niche(archive, iteration)
            mode, parent, inspirations = archive.choose_context(
                rng,
                root_probability=self.root_probability,
                exploration_probability=self.exploration_probability,
                target_niche=target_niche,
            )
            context = self._iteration_context(
                task_context,
                iteration=iteration,
                mode=mode,
                target_niche=target_niche,
                parent=parent,
                inspirations=inspirations,
            )
            campaign_model = self.model
            ensemble_arm: int | None = None
            marker = utc_now().isoformat()
            if isinstance(self.model, EnsembleModel):
                ensemble_arm, campaign_model = self.model.begin_campaign()
            controller = SynthesisController(
                campaign_model,
                self.tools,
                self.store,
                max_rounds=self.rounds_per_iteration,
                max_tool_calls=self.tool_calls_per_iteration,
                seed=self.seed + iteration,
            )
            campaigns.append(controller.run(context, run_id=f"{run_id}-i{iteration:03d}"))
            if isinstance(self.model, EnsembleModel) and ensemble_arm is not None:
                self.model.end_campaign(ensemble_arm, self.store.best_selection_score_since(marker))
        count_after = len(self.store.recent(limit=100_000))
        return EvolutionResult(
            run_id=run_id,
            iterations=self.iterations,
            completed_iterations=len(campaigns),
            candidate_count_before=count_before,
            candidate_count_after=count_after,
            occupied_cells=len(archive.elites()),
            campaigns=campaigns,
        )

    def _target_niche(self, archive: QualityDiversityArchive, iteration: int) -> str:
        occupied = {elite.cell[0] for elite in archive.elites()}
        unoccupied = [niche for niche in self.niches if niche not in occupied]
        return unoccupied[0] if unoccupied else self.niches[iteration % len(self.niches)]

    @staticmethod
    def _iteration_context(
        task_context: str,
        *,
        iteration: int,
        mode: str,
        target_niche: str,
        parent: Elite | None,
        inspirations: list[Elite],
    ) -> str:
        parent_payload = None
        if parent is not None:
            parent_payload = {
                "candidate_id": parent.candidate.candidate_id,
                "name": parent.candidate.name,
                "contract": parent.candidate.contract.value,
                "score": parent.selection_score,
                "source": parent.candidate.source,
                "parameters": parent.candidate.parameters,
            }
        inspiration_payload = [
            {
                "candidate_id": item.candidate.candidate_id,
                "name": item.candidate.name,
                "niche": item.cell[0],
                "score": item.selection_score,
                "mathematical_form": item.candidate.mathematical_form,
            }
            for item in inspirations
        ]
        if mode == "root":
            lineage_instruction = "Create an independent root and leave parents empty."
        else:
            if parent is None:
                raise ValueError("revision mode requires a sampled parent")
            lineage_instruction = (
                f"Make a genuine behavioral revision of parent {parent.candidate.candidate_id} "
                "and cite exactly that parent unless you truly recombine another candidate."
            )
        return (
            f"{task_context}\n\n"
            "# Evolutionary search assignment\n\n"
            f"Iteration: {iteration}\nSearch mode: {mode}\nTarget niche: {target_niche}\n"
            f"{lineage_instruction}\n"
            "Inspirations are context only and are not parents unless their mechanism is actually "
            "incorporated. Follow the formulation -> stage -> verify -> optimize parameters -> "
            "submit staged candidate -> evaluate workflow. Complete one candidate, then stop.\n\n"
            f"Parent:\n{json.dumps(parent_payload, indent=2, default=str)}\n\n"
            f"Inspirations:\n{json.dumps(inspiration_payload, indent=2, default=str)}\n"
        )
