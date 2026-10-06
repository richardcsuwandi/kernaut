from kernaut.archive import CandidateStore
from kernaut.evaluation import Dataset, GaussianProcessEvaluator, StandardBaselineRunner
from kernaut.evaluation.baselines import STANDARD_BASELINES
from kernaut.execution import SubprocessExecutor
from kernaut.models import CandidateOrigin
from kernaut.verification import Verifier


def test_standard_baselines_are_verified_evaluated_and_archived(tmp_path) -> None:
    store = CandidateStore(tmp_path / "archive.sqlite")
    executor = SubprocessExecutor()
    runner = StandardBaselineRunner(
        store,
        Verifier(executor),
        GaussianProcessEvaluator(executor),
        lengthscales=(1.0,),
        variances=(1.0,),
    )
    dataset = Dataset(x=[[-1.0], [0.0], [1.0]], y=[1.0, 0.0, 1.0])

    winners = runner.run(dataset)

    assert len(winners) == len(StandardBaselineRunner.__mro__ and STANDARD_BASELINES)
    assert all(candidate.origin == CandidateOrigin.BASELINE for candidate, _ in winners)
    for candidate, evaluation in winners:
        assert store.get_candidate(candidate.candidate_id) == candidate
        assert store.latest_evidence(candidate.candidate_id).accepted
        assert store.latest_evaluation(candidate.candidate_id).score == evaluation.score
