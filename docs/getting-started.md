# Installation and examples

The offline example checks installation and the task interface. A configured model provider enables kernel search on a research task.

## Install

Use Python 3.11 or later on macOS or Linux. In a terminal, clone the repository and install Kernaut:

```bash
git clone https://github.com/richardcsuwandi/kernaut.git
cd kernaut
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the following commands from this directory with the environment active.
Windows has not been tested and lacks the worker's Unix resource limits.

## Run without an API key

Install the example package, which supplies a sine-wave task, a linear baseline, and predefined model replies:

```bash
pip install -e examples/extension
kernaut task-run --task sine --config examples/extension/offline.toml \
  --baseline linear-demo --archive runs/demo/archive.sqlite
```

The example submits, verifies, and evaluates one candidate.
When it finishes, the terminal prints `Offline extension example complete.`
This is a demonstration of the framework, not a new model-generated discovery.

Inspect the results and open the dashboard:

```bash
kernaut inspect --archive runs/demo/archive.sqlite
kernaut viz --archive runs/demo/archive.sqlite --open
```

You should see the `linear_demo` baseline and the `demo_features` candidate, each with a score.
Follow the [visualizer walkthrough](visualize.md) to read the evidence.

## Choose a model

Copy the environment template, then enter the key for your provider in `.env`:

```bash
cp .env.example .env
```

Choose a configuration from the [configs directory](https://github.com/richardcsuwandi/kernaut/tree/main/configs).
Check that its model name is available through your account.

| Provider | Configuration | Setup |
| --- | --- | --- |
| OpenAI | `configs/openai.toml` | Set `OPENAI_API_KEY` |
| Anthropic | `configs/anthropic.toml` | Install `.[anthropic]` and set `ANTHROPIC_API_KEY` |
| OpenRouter | `configs/openrouter.toml` | Set `OPENROUTER_API_KEY` |
| Ollama | `configs/local-ollama.toml` | Start your local server and load the configured model |
| Claude Code or Codex | `configs/claude-code.toml` or `configs/codex.toml` | Install and sign in to the corresponding CLI |
| Another OpenAI-compatible server | `configs/modelscope-qwen.toml` | Set the server URL, model name, and key variable |

API searches can incur provider charges.
The `[campaign]` settings limit model rounds and tool calls.
Use `configs/mixed-ensemble.toml` if you want to combine providers.

## Run model-guided discovery

With a provider configured, run the bundled regression example:

```bash
kernaut run --config configs/openai.toml --context examples/task.md \
  --data examples/data.json --archive runs/search/archive.sqlite
```

Kernaut first tunes eight reference kernels, then asks the model to propose candidates.
Open the archive with `kernaut viz` while the search runs, or after it finishes.
For another dataset, follow [custom regression tasks](your-problem.md).
For domain-level evaluation, consult the [benchmark catalogue](tasks.md) and [meta-evaluation protocol](meta-evaluation.md).

## Check a kernel directly

The repository includes the dual warp–fold (DWF) kernel, plus recurrent-feature and spectral examples.
Verify DWF without a model provider:

```bash
kernaut verify examples/candidates/dual_warp_fold.json
```

An accepted candidate has `"accepted": true` and `"tier": 2` in the output.
See [verification](verification.md) for what that result establishes.
