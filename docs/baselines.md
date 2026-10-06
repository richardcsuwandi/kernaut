# Baseline comparisons

Compare discovered kernels with reference kernels on the same task and evaluator.
Keep preprocessing, training data, and evaluation budgets consistent.

## Use the standard references

The `kernaut run` command tunes eight reference kernels before starting a search:
linear, RBF, Matérn-3/2, Matérn-5/2, periodic, rational quadratic, spectral mixture, and random Fourier features.
The default workflow uses the same Gaussian process evaluator for references and discoveries.

To evaluate only these references, without calling a language model:

```bash
kernaut baselines --data examples/data.json --archive runs/baselines/archive.sqlite
kernaut viz --archive runs/baselines/archive.sqlite --open
```

The dashboard labels reference candidates `BASE`.
A positive **Vs best baseline** value means that the selected candidate scored higher than the best stored baseline in that archive.
Read this value using the task's scoring rule. It is not a significance test.

## Comparisons in the paper

The paper also evaluates learned and domain-specific references:

| Benchmark | Additional comparisons |
| --- | --- |
| Black-box optimization | FSBO, fitted input warping, grid spectral mixtures, compositional kernel search (CKS), and CAKE |
| Forecasting | Fitted input warping, grid spectral mixtures, CKS, CAKE, and AutoGP |
| ChemBench | Optimized ARD, input warping, two deep-kernel architectures, and a domain-feature library |
| GlucoseBench | Relative-time ARD Matérn, a physiological library, and per-patient marginal-likelihood fits |

These comparisons have separate training and selection protocols in the paper.
The package commands do not reproduce all of these baselines. Use the extension interface below for additional kernel programs.

## Add your own reference kernel

A baseline extension is a small Python package that returns one or more `CandidateBundle` objects.
Use the [installable extension example](https://github.com/richardcsuwandi/kernaut/tree/main/examples/extension) as a starting point.
For example, replace its `make_baselines` function with:

```python
from kernaut.models import CandidateBundle


def make_baselines(task):
    return [
        CandidateBundle(
            name="my_rbf",
            contract="closure",
            source="def closure_tree(parameters): return parameters['tree']",
            parameters={
                "tree": {
                    "op": "base",
                    "kind": "rbf",
                    "lengthscale": 0.5,
                }
            },
            rationale="RBF reference with a fixed lengthscale of 0.5.",
        )
    ]
```

Register the function in the extension's `pyproject.toml`:

```toml
[project.entry-points."kernaut.baselines"]
my-rbf = "kernaut_example:make_baselines"
```

After reinstalling the modified example package, compare it with the offline candidate:

```bash
pip install -e examples/extension
kernaut task-run --task sine --config examples/extension/offline.toml \
  --baseline my-rbf --archive runs/comparison/archive.sqlite
kernaut viz --archive runs/comparison/archive.sqlite --open
```

For your own package, replace `kernaut_example` with its Python module name.
Repeat `--baseline NAME` to include several baseline extensions.
Use the same task with a real model configuration when you are ready to search.

## Comparison protocol

`task-run` verifies each supplied baseline and evaluates its given parameters.
It does not tune those parameters. The default `run` command does tune its built-in references.
State which tuning procedure and budget you use when reporting comparisons.

Use meta-training tasks for search, meta-validation tasks for selection, and meta-test tasks for final evaluation.
Each episode also separates fitting observations from query observations, as specified in the [meta-evaluation protocol](meta-evaluation.md).
The [built-in benchmark commands](tasks.md) evaluate archived candidates on separate splits.
Methods outside the kernel interfaces, such as a neural training pipeline, need their own training and comparison protocol.
