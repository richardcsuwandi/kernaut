# Extending Kernaut

Kernaut discovers installed plugins through Python package entry points. A plugin is trusted
Python code installed in the same environment as Kernaut. Discovery lists metadata without
importing plugins. Only a selected factory is loaded. Duplicate names produce an error.
Built-in model provider names are reserved and take precedence over plugin names.

## Start from a working example

In the repository root, after installing Kernaut:

```bash
pip install -e examples/extension
kernaut extensions
kernaut task-run --task sine --config examples/extension/offline.toml \
  --baseline linear-demo --archive runs/plugin-demo/archive.sqlite
```

Copy `examples/extension` into your own repository, change the distribution name and module,
and register unique extension names in `pyproject.toml`:

```toml
[project.entry-points."kernaut.tasks"]
my-task = "my_extension:make_task"

[project.entry-points."kernaut.models"]
my-provider = "my_extension:make_model"

[project.entry-points."kernaut.baselines"]
my-baseline = "my_extension:make_baselines"
```

Reinstall your extension after changing entry-point metadata.

## Task and context interface

`make_task(executor: KernelExecutor, options: dict) -> Task` returns a `kernaut.tasks.Task`.
Its fields are:

- `dataset`: training inputs and targets as `Dataset`.
- `context`: the domain description sent to the proposer.
- `evaluator`: an object with `evaluate(candidate, dataset) -> EvaluationRecord`.
- `verification_policy`: optional input dimensions and verification settings.
- `novelty_policy`: optional formulation and novelty workflow settings.
- `parameter_evaluator`: optional separate training evaluator for parameter tuning.

Use the supplied executor to evaluate candidate kernels. A higher `EvaluationRecord.score`
must mean a better result, and the record must identify the evaluated candidate. Multi-episode
evaluators can hold training episodes internally and use a representative dataset for dimensions.
Keep validation and held-out test evaluation separate from the search task.

Pass task-specific settings as a JSON object with `--options options.json`. Override the
context with `--context task.md`. Describe units, useful invariances, known structure, and the
objective. Do not include credentials or held-out test answers in the context.

Python callers can construct a `Task` directly and call `run_task(task, model, store, executor)`.
The generic runner uses the existing conversational controller. Existing domain commands retain
the paper's evolutionary workflows and scoring rules.

## Model interface

`make_model(config: ModelConfig) -> LanguageModel` constructs your adapter.
Subclass `kernaut.llm.base.LanguageModel` and implement:

```python
def complete(self, messages, tools, system_prompt) -> AssistantReply: ...
```

Translate provider replies into `AssistantReply` and `RequestedTool`. Preserve tool call IDs so
results can be matched to requests. Use `config.api_key()` for an explicitly configured
`api_key_env`, and respect timeout and retry settings. Optional dependencies belong in your
extension package. Installed model factories also work in ensembles and existing domain commands.

```toml
[llm]
provider = "my-provider"
model = "your-model-id"
api_key_env = "MY_PROVIDER_API_KEY"
```

An OpenAI-compatible server generally needs only an existing provider configuration with its
`base_url`, without a new adapter.

## Baseline interface

`make_baselines(task: Task) -> Iterable[CandidateBundle]` supplies candidate programs with
explicit parameters. Pass one or more `--baseline NAME` options to `task-run`. The runner marks
these candidates as baselines, verifies them, evaluates them with the task evaluator, and stores
the evidence and scores. Rejected candidates stop the run before the LLM is called.

These are kernel-program baselines. A full external method such as a neural training pipeline
needs a separate training and comparison protocol. A plugin does not change
the trusted kernel construction rules. Use a supported contract for each candidate.

## Test and package an extension

Test deterministic task creation, score direction, the baseline verification gate, and model
response translation. Run an offline end-to-end campaign with a scripted model. Test your built
wheel in a separate environment because source imports alone cannot check entry-point packaging.

The API is experimental in version 0.1. Pin compatible versions in downstream packages.
Python packaging's [entry-point specification](https://packaging.python.org/en/latest/specifications/entry-points/)
describes the metadata format.
