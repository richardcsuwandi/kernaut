# Built-in tasks

Run these examples from the repository root after installing Kernaut. Search commands require
a configured LLM. Benchmark commands do not call an LLM. Use training data for discovery,
validation data for selection, and held-out test data only for final evaluation.

## Your own dataset

Provide a JSON object with `x` as a matrix and `y` as a vector, plus a Markdown task description:

```bash
kernaut run --config configs/openai.toml --context examples/task.md \
  --data examples/data.json --archive runs/custom/archive.sqlite
```

`run` tunes eight standard kernel references before starting a conversational search.
For a custom evaluator or data loader, use the [task extension interface](extensions.md).

## Procedural black-box optimization

No external dataset is required. The adapter generates deterministic optimization tasks:

```bash
kernaut meta-run --config configs/openai.toml --context examples/meta_bo_task.md \
  --archive runs/bbo/archive.sqlite --strategy evolutionary
kernaut meta-benchmark --archive runs/bbo/archive.sqlite --split validation
```

## Greenhouse-gas forecasting

Monthly NOAA series are bundled with [source attribution](../src/kernaut/data/greenhouse/SOURCE.md):

```bash
kernaut ts-run --config configs/openai.toml --context examples/greenhouse_task.md \
  --archive runs/forecast/archive.sqlite --strategy evolutionary
kernaut ts-benchmark --archive runs/forecast/archive.sqlite --split validation
```

## Enzyme kinetics

The adapter requires the ChemBench oracle from LLM-AutoSciLab. Prepare it before running:

```bash
git clone https://github.com/scientific-discovery/LLM-AutoSciLab.git ../LLM-AutoSciLab
git -C ../LLM-AutoSciLab checkout 9a37255ff5432a7155b6cde8985e9654f2f8f272
kernaut chem-run --chembench-root ../LLM-AutoSciLab --config configs/openai.toml \
  --context examples/chembench_task.md --archive runs/chem/archive.sqlite --strategy evolutionary
kernaut chem-benchmark --chembench-root ../LLM-AutoSciLab \
  --archive runs/chem/archive.sqlite --split validation
```

## Glucose forecasting

Install the external `glucosebench` package in the same environment, then export its public
training payloads. The exporter does not include sealed test outcomes:

```bash
python examples/glucose/export_training.py --out runs/glucose/training.json
kernaut glucose-run --data runs/glucose/training.json --config configs/openai.toml \
  --context examples/glucosebench_task.md --archive runs/glucose/archive.sqlite
kernaut glucose-benchmark --data runs/glucose/training.json \
  --archive runs/glucose/archive.sqlite --split validation
```

This adapter uses training and validation episodes. Sealed adult testing requires the external
GlucoseBench evaluator. Run `kernaut COMMAND --help` for candidate selection and budget options.
These are usage examples, not a complete protocol for reproducing the paper's numerical results.
