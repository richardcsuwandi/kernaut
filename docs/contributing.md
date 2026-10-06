# Contribution guidelines

Contributions can extend the benchmark coverage, evaluation methods, model interfaces, or documentation.
Each contribution should include a reproducible example and a clear statement of its scope.

## Contribution categories

| Contribution | Required scope |
| --- | --- |
| New task | A data loader, a clear problem description, an evaluator, and a small offline example |
| New baseline | A reference kernel with stated parameters or a documented tuning procedure |
| Model adapter | An adapter that preserves tool calls, plus tests using recorded or synthetic responses |
| Tutorial | A runnable example that explains its inputs, outputs, and limitations |
| Bug report | The command, software versions, expected behavior, and a small reproduction |

Start with [an issue](https://github.com/richardcsuwandi/kernaut/issues/new/choose) to discuss a new task or baseline.
For a small fix or documentation improvement, a pull request is welcome directly.

## Add a task

Copy the [example extension](https://github.com/richardcsuwandi/kernaut/tree/main/examples/extension) into your own package.
Change its data loader, context, and evaluator, then register a task name.
The [extension guide](extensions.md) explains registration.
The [custom benchmark guide](custom-benchmarks.md) covers task partitions, evaluators, and fixed-candidate assessment.

Your example should state what the inputs mean, how the score is computed, and which data the search can access.
Keep held-out test answers outside the search context.
Include a small offline test so contributors can check the integration without API keys.

## Add a baseline

Follow the [baseline comparison guide](baselines.md).
State whether parameters are fixed or tuned, and use the same task and evaluation budget as the comparison method.
Contributions should make the comparison easy to reproduce.

## Develop locally

In a cloned repository with the Python environment active, install the development tools:

```bash
pip install -e ".[dev]"
pip install -e examples/extension
ruff check .
ruff format --check .
mypy src/kernaut
pytest
```

For documentation changes:

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Open `http://127.0.0.1:8000` to preview the site. Run `mkdocs build --strict` before submitting the change.
Keep generated archives, credentials, and local results out of the pull request.
Describe the change and the checks you ran.

The extension API is experimental and may change before version 1.0.
See the [repository contribution guidelines](https://github.com/richardcsuwandi/kernaut/blob/main/CONTRIBUTING.md)
for construction-contract requirements and review expectations.
