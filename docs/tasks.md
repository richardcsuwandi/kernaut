# Benchmarks and evaluation tasks

Kernaut includes four domain benchmarks, a user-supplied regression workflow, and an offline integration example.
Run commands from the repository root after [installation](getting-started.md).
Search commands require a configured large language model. Benchmark commands evaluate fixed candidates without calling a language model.

The [meta-evaluation protocol](meta-evaluation.md) separates discovery, candidate selection, and final evaluation.
These examples describe the package interfaces. They are not a complete reproduction protocol for the paper's numerical results.

## Black-box optimization

The benchmark generates deterministic transformations of analytic functions.
Transformations include coordinate permutations, reflections, power warps, triangular couplings, and nuisance dimensions.
Predictive evaluation measures continuous ranked probability score (CRPS), negative log predictive density (NLPD), and root mean squared error (RMSE).
Bayesian optimization evaluation reports regret over a sequence of queries and final regret. Lower values are preferable for these metrics.

| Split | Function families and intrinsic dimensions |
| --- | --- |
| Meta-training | Branin (2), Ackley (2), Cosine (8), Hartmann (6), Levy (6) |
| Meta-validation | The same five families with separate episode seeds |
| Meta-test | Bukin (2), Drop-Wave (2), Griewank (5), Hölder Table (2), Rosenbrock (6), Rastrigin (6) |

Nuisance dimensions can increase the observed input dimension to at most eight.
The functions are generated locally and require no external dataset.

```bash
kernaut meta-run --config configs/openai.toml --context examples/meta_bo_task.md \
  --archive runs/bbo/archive.sqlite --strategy evolutionary --episodes-per-family 2
kernaut meta-benchmark --archive runs/bbo/archive.sqlite \
  --split validation --episodes-per-family 2 --bo-steps 8 --candidate-pool-size 128
```

The search score is negative mean predictive CRPS, with an optional novelty penalty.
Short optimization rollouts are recorded as additional search diagnostics.
They do not change the score returned by `MetaSearchEvaluator`.
See [Goldie et al. (2026)](references.md#discogen-and-meta-evaluation) for the related task-generation and meta-evaluation framework.

## Greenhouse-gas forecasting

This benchmark uses bundled monthly global mean records from the NOAA Global Monitoring Laboratory.
Meta-training and meta-validation use CO2, CH4, and N2O. Meta-test uses SF6.
Each episode fits a prefix and predicts a subsequent window in the episode's sequence.
The default forecast horizon is 48 months, with at least 216 training months.

Episode generation varies the prefix length, can reverse the record, and adds noise only to the training targets.
These are generated forecasting tasks, not exclusively forward calendar-time forecasts.
Training and validation episodes use different seeds but can share months from the underlying records.

```bash
kernaut ts-run --config configs/openai.toml --context examples/greenhouse_task.md \
  --archive runs/forecast/archive.sqlite --strategy evolutionary --episodes-per-family 3
kernaut ts-benchmark --archive runs/forecast/archive.sqlite \
  --split validation --episodes-per-family 3 --horizon-months 48
```

Reports include CRPS, NLPD, and RMSE. The search score is negative mean CRPS, with an optional novelty penalty.
See the [NOAA source and data attribution](references.md#noaa-greenhouse-gas-records).

## ChemBench enzyme kinetics

The adapter uses the ChemBench oracle from [LLM-AutoSciLab](references.md#chembench).
Each observation maps seven experimental inputs to a reaction rate.
The inputs encode concentrations, enzyme loading, temperature, and pH.
This adapter evaluates reaction-rate prediction. It does not evaluate symbolic recovery of a rate law.

| Split | Mechanism domains |
| --- | --- |
| Meta-training | `c0_michaelis_menten`, `c1_competitive_inhibition`, `c2_product_inhibition`, `c3_arrhenius_temperature`, `c4_ph_activity`, `c5_pingpong_bisubstrate`, `c6_uncompetitive_inhibition`, `c7_substrate_inhibition`, `c8_hill_cooperativity`, `c9_noncompetitive_inhibition` |
| Meta-validation | The same ten domains with separate input-sampling seeds |
| Meta-test | `c65_ordered_bi_bi`, `c66_reversible_mm`, `c67_allosteric_act`, `c69_fractal_kinetics`, `c73_metal_activation` |

Install the external source at the revision used by the adapter:

```bash
git clone https://github.com/scientific-discovery/LLM-AutoSciLab.git ../LLM-AutoSciLab
git -C ../LLM-AutoSciLab checkout 9a37255ff5432a7155b6cde8985e9654f2f8f272
kernaut chem-run --chembench-root ../LLM-AutoSciLab --config configs/openai.toml \
  --context examples/chembench_task.md --archive runs/chem/archive.sqlite \
  --strategy evolutionary --episodes-per-domain 1 --train-points 20 --test-points 20
kernaut chem-benchmark --chembench-root ../LLM-AutoSciLab \
  --archive runs/chem/archive.sqlite --split validation \
  --episodes-per-domain 1 --train-points 20 --test-points 20
```

Reports include CRPS, NLPD, RMSE, and failures by episode.
`--test-points` specifies held-out observations within each episode, including meta-training episodes.
It does not select the meta-test domains.
`--difficulty`, `--noise-level`, and `--worst-domain-weight` configure the oracle and score. See the [configuration table](meta-evaluation.md#configuration).

## GlucoseBench forecasting

[GlucoseBench](references.md#glucosebench) uses simulated patient profiles and meal and insulin interventions.
A candidate forecasts continuous glucose monitor readings after a 45-minute observed prefix.
Its inputs encode reading time, meal amount and start, and insulin bolus amount and start.

Meta-training uses `child#001` through `child#010`.
Meta-validation uses `adolescent#001` through `adolescent#010`.
The intended meta-test group is `adult#001` through `adult#010`.
The local adapter evaluates folds that withhold one public training episode at a time, using the remaining episodes and the query prefix.

Install the external `glucosebench` package in the same Python environment.
The exporter collects two passive episodes and four interventions for each patient, using a fixed acquisition policy:

```bash
python examples/glucose/export_training.py --out runs/glucose/training.json
kernaut glucose-run --data runs/glucose/training.json --config configs/openai.toml \
  --context examples/glucosebench_task.md --archive runs/glucose/archive.sqlite
kernaut glucose-benchmark --data runs/glucose/training.json \
  --archive runs/glucose/archive.sqlite --split validation
```

The local report includes CRPS, NLPD, RMSE, and normalized mean squared error (NMSE).
NMSE is averaged within each patient, then combined with a geometric mean across patients.
Search uses negative mean CRPS, with an optional novelty penalty.

The CLI supports only `train` and `validation` for this adapter.
Hidden adult test outcomes remain inside the external `glucosebench.Evaluator`.
The exported adult training episodes do not replace that hidden test evaluation.

## User-supplied regression

The default `run` workflow reads JSON arrays `x` and `y` and a Markdown task description:

```bash
kernaut run --config configs/openai.toml --context examples/task.md \
  --data examples/data.json --archive runs/custom/archive.sqlite
```

It tunes eight reference kernels and scores candidates by training log marginal likelihood.
It does not create meta-training, meta-validation, or meta-test splits automatically.
See [custom regression tasks](your-problem.md) and [custom benchmarks](custom-benchmarks.md) for separate evaluation protocols.

## Offline integration example

The sine-wave task in `examples/extension` verifies the extension interfaces without a model provider.
It includes a fixed linear baseline and a model that returns predefined tool requests.
It is an integration example, not a scientific benchmark of discovery performance.
Follow the [offline setup instructions](getting-started.md#run-without-an-api-key).
