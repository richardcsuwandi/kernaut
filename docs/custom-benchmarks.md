# Custom benchmarks and tasks

A benchmark defines task families, episode generation, fitting rules, and a scoring protocol.
A Kernaut `Task` exposes the training portion of that benchmark to the search agent.
Validation and test evaluators remain outside the search task.
This separation supports the [meta-evaluation protocol](meta-evaluation.md) without requiring changes to the kernel construction rules.

## Define the evaluation protocol

Specify the protocol before running discovery:

1. Define the unit of generalization, such as a function family, patient, material, site, or time interval.
2. Assign groups to meta-training, meta-validation, and meta-test partitions. Record identifiers and generation seeds in a versioned manifest.
3. Within each episode, separate fitting observations from query observations. Fit preprocessing and model parameters using only permitted fitting data.
4. Define the primary metric, failure penalty, aggregation weights, and evaluation budget. Use the same rules for candidates and baselines.
5. Fix the candidate-selection rule. Keep meta-validation and meta-test outcomes outside the model context and search evaluator.

For example, validation may use new episodes from training families, while testing uses different families.
For patient-level transfer, partition by patient rather than randomly dividing observations from the same patient.
For temporal transfer, specify the time boundary explicitly.

## Register a task using an existing meta-evaluator

The following example exposes the built-in predictive BBO evaluator as a task extension.
It illustrates how to configure meta-training episodes through `--options`.
It uses the existing BBO families, rather than defining new ones.

After [installing the example extension](extensions.md#example-extension), add this function to `examples/extension/src/kernaut_example/__init__.py`:

```python
from kernaut.evaluation.gp import Dataset
from kernaut.evaluation.meta_bo import MetaKernelEvaluator
from kernaut.tasks import Task
from kernaut.verification import VerificationPolicy


def make_meta_task(executor, options):
    evaluator = MetaKernelEvaluator(
        executor,
        split="train",
        episodes_per_family=int(options.get("episodes_per_family", 1)),
    )
    return Task(
        dataset=Dataset(x=[[0.0] * 8, [1.0] * 8], y=[0.0, 1.0]),
        context=(
            "Search for a kernel across generated training functions. "
            "Inputs have 2 to 8 coordinates in [0, 1]. "
            "The evaluator returns negative mean predictive CRPS."
        ),
        evaluator=evaluator,
        verification_policy=VerificationPolicy(
            input_dimension=8,
            input_dimensions=(2, 5, 6, 8),
        ),
    )
```

The two rows in `dataset` specify representative inputs for the framework.
`MetaKernelEvaluator` generates its own episode data and ignores those representative targets during scoring.
The task is explicitly restricted to `split="train"`.

Add an entry under the example package's existing task group:

```toml
[project.entry-points."kernaut.tasks"]
sine = "kernaut_example:make_task"
meta-regression = "kernaut_example:make_meta_task"
```

Create `meta-options.json` in the repository root:

```json
{"episodes_per_family": 1}
```

Reinstall the modified package and run an offline integration check:

```bash
pip install -e examples/extension
kernaut task-run --task meta-regression --options meta-options.json \
  --config examples/extension/offline.toml --baseline linear-demo \
  --archive runs/meta-extension/archive.sqlite
```

The offline model submits a fixed candidate. Replace its configuration with a configured provider for model-generated proposals.
`task-run` uses one conversation and evaluates baseline parameters as supplied.
The built-in `meta-run` additionally supports evolutionary search, baseline selection, and short optimization diagnostics.
These commands therefore do not define identical search protocols.

Because this example uses the built-in BBO families, evaluate its archived candidates with `meta-benchmark`:

```bash
kernaut meta-benchmark --archive runs/meta-extension/archive.sqlite \
  --split validation --episodes-per-family 1 > runs/meta-extension/validation.json
```

Select a fixed candidate and evaluate it on test families using the [selection procedure](meta-evaluation.md#search-select-and-evaluate).

## Implement a new domain evaluator

For a new domain, implement an object with the following interface:

```python
from kernaut.models import EvaluationRecord


def evaluate(self, candidate, dataset) -> EvaluationRecord: ...
```

The evaluator can manage multiple episodes internally, as the built-in domain evaluators do.
Its implementation should:

1. Run the candidate through the supplied `KernelExecutor` to obtain kernel matrices for each episode.
2. Fit the chosen kernel method using the episode's fitting observations, under a fixed parameter-fitting budget.
3. Predict the query observations and compute the declared metric.
4. Aggregate episode results with explicit weights. Apply the declared penalty to failed episodes rather than omitting them.
5. Return an `EvaluationRecord` with the same `candidate_id`, a higher-is-better `score`, runtime, and fitting diagnostics.

Store the split name, episode identifiers, raw metrics, failure counts, and protocol version in `EvaluationRecord.metadata`.
For a loss such as mean CRPS, use its negative as the score.
Keep the raw loss available for interpretation, especially if search adds other penalties.

The built-in evaluators provide implementations for [regression](https://github.com/richardcsuwandi/kernaut/blob/main/src/kernaut/evaluation/meta_bo.py),
[forecasting](https://github.com/richardcsuwandi/kernaut/blob/main/src/kernaut/evaluation/timeseries.py),
[enzyme kinetics](https://github.com/richardcsuwandi/kernaut/blob/main/src/kernaut/evaluation/chembench.py), and
[glucose prediction](https://github.com/richardcsuwandi/kernaut/blob/main/src/kernaut/evaluation/glucose.py).
Their private helper functions are implementation details, not stable extension interfaces.

Register a factory that returns your training `Task` through `kernaut.tasks`, as shown above.
Use `--options` for training configuration, such as a data path or episode budget.
Reject validation or test partitions in that search factory rather than forwarding an unrestricted split option.

## Evaluate fixed candidates outside search

The package does not provide a generic `benchmark --task` command for arbitrary extensions.
A new domain therefore needs a separate evaluation script or command in its extension package.
That command should load a frozen `CandidateBundle`, verify it, and apply the domain evaluator without calling a language model.

The following function illustrates that boundary for an evaluator with the interface above:

```python
from kernaut.verification import Verifier


def evaluate_fixed(candidate, evaluator, executor, verification_policy):
    evidence = Verifier(executor, verification_policy).verify(candidate)
    if not evidence.accepted:
        raise ValueError("The frozen candidate failed verification")
    return evaluator.evaluate(candidate, None)
```

This example assumes that the evaluator manages all episode data internally and accepts an unused second argument.
Evaluate the prespecified candidate set on validation episodes, select by the declared rule, and then evaluate the selected program on test episodes.
Write these reports separately from the training archive so that validation or test outcomes cannot enter a resumed search.

## Verify a benchmark contribution

Test episode reproducibility, group separation, score direction, candidate rejection, and the handling of failed episodes.
Check that changing query targets cannot affect preprocessing or parameter fitting.
Include one small offline end-to-end example and state which external data or software it requires.

Document the dataset license, source revision, partition manifest, selection metric, and reference budgets.
The [contribution guide](contributing.md) describes how to submit the benchmark.
