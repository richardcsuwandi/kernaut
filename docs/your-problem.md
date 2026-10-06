# Custom regression tasks

The default regression workflow accepts an input matrix and a target vector.
Add a task extension when you need a different data loader or evaluator.

## 1. Prepare training data

Create `my-data.json` with one row of inputs per observation and one corresponding target:

```json
{
  "x": [[0.0, 0.2], [0.3, 0.4], [0.7, 0.1], [1.0, 0.8]],
  "y": [0.1, 0.6, 0.4, 1.2]
}
```

Rows must have the same number of features. All values must be finite numbers.
Choose a sensible scale for each feature and document any transformations.
Keep validation and test observations separate from the training file.

## 2. Describe the problem

Create `my-task.md`. Explain what each input means and which patterns a useful kernel should capture.
For example:

```markdown
# Response across temperature and pressure

Each observation has two inputs, both scaled to [0, 1]:
- x[0]: temperature.
- x[1]: pressure.

The target is a measured response. Nearby settings should usually have similar responses.
The temperature effect may change with pressure. Explore smooth kernels that represent this interaction.
Use the supplied training observations to compare candidates.
```

Include input units, known periodicity, smoothness, interactions, or invariances when these matter.
An invariance is a transformation that should leave the response unchanged.
Do not include API keys or held-out answers in the context.

## 3. Run a search

After [installing Kernaut and configuring a model](getting-started.md), run from the repository root:

```bash
kernaut run --config configs/openai.toml --context my-task.md \
  --data my-data.json --archive runs/my-task/archive.sqlite
```

The default evaluator uses Gaussian process marginal likelihood on the training observations.
The score is log marginal likelihood, so higher scores are better.
A higher training score does not establish better held-out predictions.
Use a separate evaluation to select and assess the final kernel.
This command does not create a meta-level split. For evaluation across episodes or families, define a [custom benchmark](custom-benchmarks.md).

Keep one task and scoring protocol per archive so its comparisons remain meaningful.
Then [open the visualizer](visualize.md):

```bash
kernaut viz --archive runs/my-task/archive.sqlite --open
```

## 4. Change the evaluator when needed

A task combines training data, a problem description, and an evaluator.
You can supply these from Python:

```python
from kernaut.evaluation.gp import Dataset, GaussianProcessEvaluator
from kernaut.execution import SubprocessExecutor
from kernaut.tasks import Task

executor = SubprocessExecutor()
dataset = Dataset(x=[[0.0], [0.5], [1.0]], y=[0.0, 0.8, 0.2])
task = Task(
    dataset=dataset,
    context="Find a smooth kernel for a one-dimensional response.",
    evaluator=GaussianProcessEvaluator(executor),
)
```

Replace `evaluator` with your own object whose `evaluate(candidate, dataset)` method returns an `EvaluationRecord`.
Higher scores must mean better results.
For episode-based tasks, the evaluator can manage its training episodes internally.

To run your task from the command line and share it with others, follow the
[task extension guide](extensions.md#add-a-task-and-its-context).
The [built-in tasks](tasks.md) provide examples for optimization, forecasting, chemistry, and glucose data.
