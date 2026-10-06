# Add tasks, model providers, and baselines

You can use Kernaut's built-in tasks and model providers without writing an extension.
An extension is a separate Python package that adds a task, model provider, or baseline to Kernaut.
For example, a task extension can load your dataset, describe the problem, and score proposed kernels.
An extension lets you add this behavior without editing Kernaut itself.

## Example extension

The repository includes a small extension in `examples/extension`. It provides a sine-wave task,
a fixed offline model, and a linear reference kernel. After installing Kernaut, run these commands
from the repository root:

```bash
pip install -e examples/extension
kernaut extensions
kernaut task-run --task sine --config examples/extension/offline.toml \
  --baseline linear-demo --archive runs/extension-demo/archive.sqlite
```

The first command installs the example in your Python environment. The second lists the names
that Kernaut can use. The third runs a search without an API key or network access.

## Register your extension

Each extension provides a function that creates a task, model adapter, or set of baseline kernels.
This function is called a **factory function**. An **entry point** maps a name, such as
`my-task`, to that function. Kernaut reads these mappings from installed packages.

To create your own extension:

1. Copy `examples/extension` into your repository.
2. Change the package name and Python module name.
3. Implement the functions for the components you want to add.
4. Register each function under the appropriate group in `pyproject.toml`:

```toml
[project.entry-points."kernaut.tasks"]
my-task = "my_extension:make_task"

[project.entry-points."kernaut.models"]
my-provider = "my_extension:make_model"

[project.entry-points."kernaut.baselines"]
my-baseline = "my_extension:make_baselines"
```

For example, `my_extension:make_task` identifies the `make_task` function in the `my_extension`
module. You only need the groups that your extension provides. Install the package in the same
Python environment as Kernaut. Reinstall it after changing the entry-point mappings.

Kernaut lists installed names without importing extension code. It imports the selected
extension when you use it. If multiple packages register the same selected name, Kernaut reports an error.
Built-in model provider names take precedence over extension names, so choose a different name.
Extensions run as trusted Python code. Install only packages that you trust.

## Add a task and its context

`make_task(executor: KernelExecutor, options: dict) -> Task` creates a `kernaut.tasks.Task`.
The task contains:

- `dataset`: training inputs and targets stored in a `Dataset`.
- `context`: the domain description sent to the model that proposes kernels.
- `evaluator`: an object whose `evaluate(candidate, dataset)` method returns an `EvaluationRecord`.
- `verification_policy`: optional input dimensions and verification settings.
- `novelty_policy`: optional settings for recording kernel ideas and checking novelty.
- `parameter_evaluator`: an optional separate evaluator for tuning parameters on training data.

The evaluator determines the kernel method and scoring rule. It can use kernel ridge regression, a support vector machine, or another method.
Use the supplied `executor` to run candidate kernels. Higher `EvaluationRecord.score` values must
mean better results. Each record must identify the candidate that it evaluates.
An evaluator can store multiple training episodes internally. In that case, `dataset` can provide
representative inputs that specify the input dimensions. Keep validation and held-out test
evaluation separate from the search task.

Use `--options options.json` to pass task settings as a JSON object. Use `--context task.md` to
replace the task's context. Describe input units, known structure, the objective, and transformations
that should leave predictions unchanged. Exclude credentials and held-out test answers.

In Python, you can create a `Task` directly and call `run_task(task, model, store, executor)`.
This function uses one model conversation for the search. The built-in domain commands also
support the paper's evolutionary search workflows and scoring rules.

## Add a model provider

`make_model(config: ModelConfig) -> LanguageModel` creates an adapter that connects Kernaut to a
model provider. Subclass `kernaut.llm.base.LanguageModel` and implement:

```python
def complete(self, messages, tools, system_prompt) -> AssistantReply: ...
```

Convert provider responses to `AssistantReply` and `RequestedTool` objects. Preserve tool call IDs
so Kernaut can match results to requests. Read credentials with `config.api_key()` after setting
`api_key_env`. Respect the configured timeout and retry limits. Declare optional dependencies in
your extension package. Model extensions also work in ensembles and built-in domain commands.

Select your provider in the model configuration:

```toml
[llm]
provider = "my-provider"
model = "your-model-id"
api_key_env = "MY_PROVIDER_API_KEY"
```

An OpenAI-compatible server usually needs only an existing provider configuration with the
server's `base_url`. It does not usually need a new adapter.

## Add baseline kernels

`make_baselines(task: Task) -> Iterable[CandidateBundle]` returns reference kernel programs with
specified parameters. Pass `--baseline NAME` to `task-run`. Repeat the option to use multiple
baseline extensions.

Kernaut marks the returned candidates as baselines and verifies them before evaluation.
It then evaluates each candidate with the task evaluator and stores its evidence and score.
If verification rejects a baseline, Kernaut stops the run before calling the LLM.
The runner uses the supplied parameters without tuning them.

These baselines are kernel programs. An external method, such as a neural training pipeline,
needs its own training and comparison protocol. Extensions do not change the trusted kernel
construction rules. Use a supported construction contract for each candidate.

## Test your extension

Test that task creation is deterministic, higher scores mean better results, and rejected
baselines cannot reach evaluation. For a model adapter, test the conversion of provider responses.
Run a complete offline search with a model that returns predefined responses.

Build a wheel, which is Python's installable package format, and test it in a separate environment.
Importing code from the source directory does not check whether the wheel includes the entry-point
mappings. The API is experimental in version 0.1. Pin compatible versions in packages that use it.
The [Python entry-point specification](https://packaging.python.org/en/latest/specifications/entry-points/)
defines the mapping format.
