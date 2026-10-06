# Meta-evaluation protocol

Meta-evaluation assesses whether a discovered kernel transfers beyond the tasks used during search.
Kernaut follows the separation between discovery tasks and evaluation tasks discussed by [Goldie et al. (2026)](references.md#discogen-and-meta-evaluation).
The task split and the observation split within a task serve different purposes.

## Two levels of separation

An **episode** is one small regression, forecasting, or optimization problem.
Within an episode, fitting observations determine the Gaussian process parameters.
Separate query observations measure predictive error.
For forecasting, those queries are the withheld suffix of the episode's sequence.

At the **meta level**, episodes or task families are assigned to three roles:

| Partition | Role | Permitted adaptation |
| --- | --- | --- |
| Meta-training | Generate and revise kernel programs, tune their parameters, and choose reference settings | Search can use evaluator feedback from these episodes |
| Meta-validation | Select among the frozen candidate programs using a prespecified metric | Select a candidate without revising its code or kernel parameters |
| Meta-test | Assess the selected program on families or patient groups withheld from search | Apply the same predefined fitting procedure without revising or reselecting the program |

A meta-training episode can contain query observations that are held out from its GP fit.
Their evaluation results still provide feedback to search, so these observations are not a final test set.
Conversely, a meta-test episode includes fitting observations that its GP may use under the fixed evaluation protocol.

In the domain evaluators, the kernel program and its parameters remain fixed during benchmark evaluation.
The GP fits covariance amplitude and observation noise from each episode's fitting observations using fixed grids.
Target standardization also uses those fitting observations.
For GlucoseBench, parameter fitting uses the other complete training episodes, and conditioning additionally uses the query prefix.

## Built-in partition definitions

| Benchmark | Meta-training | Meta-validation | Meta-test |
| --- | --- | --- | --- |
| Black-box optimization | Five function families | New seeded transformations of the same five families | Six different function families |
| Greenhouse gases | CO2, CH4, N2O | New seeded episodes from those records | SF6 |
| ChemBench | Ten canonical mechanism domains | New input samples in those domains | Five structurally distinct domains |
| GlucoseBench | Ten child profiles | Ten adolescent profiles | Ten adult profiles through the external hidden-test evaluator |

The [benchmark catalogue](tasks.md) lists every function, gas record, and mechanism domain.
The greenhouse split separates generated episodes and the held-out gas family. It does not make training and validation calendar months disjoint.
For custom temporal benchmarks, define disjoint time intervals if that is the intended generalization test.

Black-box optimization (BBO), greenhouse, and ChemBench episodes use the seed formula:

```text
seed = split_offset + 1000 * family_index + episode_index
split_offset = 0 (train), 10000 (validation), 20000 (test)
```

Keep fewer than 1,000 episodes per family or domain with this scheme to avoid overlapping seed ranges.
ChemBench uses this seed for input sampling. Oracle noise follows the external oracle's implementation.
The GlucoseBench exporter uses a fixed seed derived from each patient identifier.

## Configuration

Partition membership is defined in the domain adapters, rather than as a percentage split in TOML.
Command-line options control the evaluated partition and episode budgets.
`[campaign].seed` controls search and verification randomness. It does not change the built-in meta partitions or their episode seed offsets.

| Interface | Relevant options and defaults |
| --- | --- |
| `meta-run` | `--episodes-per-family 2`, always meta-training |
| `meta-benchmark` | `--split all`, `--episodes-per-family 1`, `--bo-steps 8`, `--candidate-pool-size 128` |
| `ts-run` | `--episodes-per-family 3`, always meta-training, 48-month full evaluation horizon |
| `ts-benchmark` | `--split all`, `--episodes-per-family 1`, `--horizon-months 48` |
| `chem-run` | Always meta-training. `--episodes-per-domain 1`, `--train-points 20`, `--test-points 20`, `--difficulty easy`, `--noise-level 0.01` |
| `chem-benchmark` | The same episode and oracle options, plus `--split all` |
| ChemBench scoring | `--worst-domain-weight 0.0` in both commands. Positive values add a worst-domain CRPS penalty |
| `glucose-run` | `--data` selects the exported observations. Search uses child profiles |
| `glucose-benchmark` | `--split all` means training and validation only. There is no hidden-test CLI option |

Use explicit `--split validation` during selection because the first three benchmark commands otherwise include test results.
Keep episode sizes, transformations, preprocessing, and oracle settings consistent between candidate and baseline evaluations.
Changing family membership, seed offsets, or the greenhouse training-window minimum requires a custom evaluator or an adapter change.
There is no general `--train-ratio`, `--validation-ratio`, or `--test-ratio` option.
Built-in parameter-tuning tools can use reduced meta-training episodes. Their scores are not validation results.

## Search, select, and evaluate

After configuring a model, run a search on BBO meta-training episodes:

```bash
kernaut meta-run --config configs/openai.toml --context examples/meta_bo_task.md \
  --archive runs/bbo/archive.sqlite --strategy evolutionary --episodes-per-family 2
```

Stop the search before selection. Preserve the resulting archive and evaluate its accepted candidates on meta-validation episodes:

```bash
kernaut meta-benchmark --archive runs/bbo/archive.sqlite \
  --split validation --episodes-per-family 2 --bo-steps 8 --candidate-pool-size 128 \
  > runs/bbo/validation.json
```

Choose the metric before inspecting results, such as `splits.validation.predictive.mean_crps` for prediction or `splits.validation.bo.regret_auc` for optimization.
Select the candidate with the lowest value under that rule and record its `candidate_id`.
The default candidate limit is 100. Increase `--limit` or repeat `--candidate-id` when your prespecified candidate set requires it.

Replace `SELECTED_ID` below with the chosen identifier before evaluating the held-out families:

```bash
kernaut meta-benchmark --archive runs/bbo/archive.sqlite \
  --split test --candidate-id SELECTED_ID --episodes-per-family 2 \
  --bo-steps 8 --candidate-pool-size 128 > runs/bbo/test.json
```

Benchmark commands include their reference baselines even when `--candidate-id` is supplied.
Reference settings are selected using meta-training evaluations.
The reported JSON is separate from the search archive's evaluation records.
Do not use the order of the test report to select another program.

The same select-then-test procedure applies to `ts-benchmark` and `chem-benchmark`.
For GlucoseBench, select on adolescent folds and use the external evaluator for hidden adult testing.
Repeated decisions based on validation results also adapt to that partition, so record the number of candidates and the selection rule.

## Metrics and reporting

CRPS evaluates a predictive distribution against an observed value. Lower CRPS is preferable.
See [Gneiting and Raftery (2007)](references.md#probabilistic-scoring) for proper scoring rules.
Kernaut negates loss-based scores so that larger `EvaluationRecord.score` values indicate better candidates.
Search can also include novelty or worst-domain penalties, so report the raw predictive metrics separately from search fitness.

Report the split definitions, episode counts, candidate set, selection rule, baseline budgets, failures, and software revisions with any scientific comparison.
Scores from different tasks or preprocessing protocols are not directly comparable merely because both use CRPS.
For a new dataset or domain, follow [custom benchmarks](custom-benchmarks.md).
