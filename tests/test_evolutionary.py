from __future__ import annotations

import random

from kernaut.agent import QualityDiversityArchive
from kernaut.archive import CandidateStore
from kernaut.models import CandidateBundle, EvaluationRecord, EvidenceRecord, EvidenceTier


def _archive_candidate(
    store: CandidateStore,
    *,
    name: str,
    niche: str,
    growth: str,
    novelty_band: str,
    score: float,
) -> CandidateBundle:
    candidate = CandidateBundle(
        name=name,
        contract="feature_map",
        source="def feature_point(x, parameters):\n    return x\n",
        construction_niche=niche,
        feature_growth=growth,
    )
    store.add_candidate(candidate)
    store.add_evidence(
        EvidenceRecord(
            candidate_id=candidate.candidate_id,
            tier=EvidenceTier.CONTRACT_CERTIFIED,
            accepted=True,
            checks=[],
            contract=candidate.contract,
        )
    )
    store.add_evaluation(
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            score=score,
            negative_log_likelihood=0.0,
            runtime_seconds=1.0,
            jitter=1e-4,
            condition_number=1.0,
            metadata={
                "selection_score": score,
                "quality_descriptors": {
                    "construction_niche": niche,
                    "feature_growth": growth,
                    "novelty_band": novelty_band,
                },
            },
        )
    )
    return candidate


def test_quality_diversity_archive_keeps_best_candidate_per_cell(tmp_path) -> None:
    store = CandidateStore(tmp_path / "archive.sqlite")
    _archive_candidate(
        store,
        name="weak_spectral",
        niche="spectral_periodic",
        growth="O(d)",
        novelty_band="distinct",
        score=0.1,
    )
    best = _archive_candidate(
        store,
        name="best_spectral",
        niche="spectral_periodic",
        growth="O(d)",
        novelty_band="distinct",
        score=0.3,
    )
    _archive_candidate(
        store,
        name="additive",
        niche="additive_interaction",
        growth="O(d)",
        novelty_band="strongly_distinct",
        score=0.2,
    )

    elites = QualityDiversityArchive(store).elites()
    assert len(elites) == 2
    assert elites[0].candidate.candidate_id == best.candidate_id


def test_evolutionary_sampling_distinguishes_roots_and_revisions(tmp_path) -> None:
    store = CandidateStore(tmp_path / "archive.sqlite")
    parent = _archive_candidate(
        store,
        name="parent",
        niche="input_geometry",
        growth="O(d)",
        novelty_band="distinct",
        score=0.2,
    )
    archive = QualityDiversityArchive(store)

    mode, selected, _ = archive.choose_context(
        random.Random(0),
        root_probability=1.0,
        exploration_probability=0.0,
        target_niche="spectral_periodic",
    )
    assert mode == "root"
    assert selected is None

    mode, selected, _ = archive.choose_context(
        random.Random(0),
        root_probability=0.0,
        exploration_probability=0.0,
        target_niche="input_geometry",
    )
    assert mode == "revision"
    assert selected is not None
    assert selected.candidate.candidate_id == parent.candidate_id


def test_failed_selection_score_does_not_break_evolution(tmp_path):
    store = CandidateStore(tmp_path / "archive.sqlite")
    candidate = _archive_candidate(
        store,
        name="failed",
        niche="input_geometry",
        growth="O(d)",
        novelty_band="unknown",
        score=-2.0,
    )
    failed = store.latest_evaluation(candidate.candidate_id).model_copy(
        update={"condition_number": float("inf"), "metadata": {"selection_score": float("-inf")}}
    )
    store.add_evaluation(failed)
    assert QualityDiversityArchive(store).elites() == []
