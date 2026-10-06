# Procedural meta-BO kernel discovery

Design a single dimension-agnostic Gaussian-process kernel that generalizes across a distribution
of normalized, low-to-moderate-dimensional Bayesian optimization (BO) tasks. Training episodes include
smooth anisotropic, periodic, multimodal, and rugged functions under hidden coordinate
permutations, reflections, monotone input warpings, invertible cross-coordinate couplings, and a
small number of irrelevant coordinates.

Each task standardizes its outputs and fits the same small grid of outer covariance amplitudes
and diagonal noise values.

Prefer Tier-2 candidates that:

- accept any input dimension rather than hard-coding a two-dimensional feature map.
- express useful inductive biases beyond renaming standard RBF or Matérn kernels.
- remain numerically stable for input dimensions 2, 5, 6, and 8.
- balance smooth global structure with periodic or localized variation.
- use parent lineage when making a genuine revision, and leave parents empty for an independent
  root formulation.

This is a mathematical-novelty campaign. Closure trees are disallowed for discovered candidates:
ordinary sums, products, and rescalings of RBF/Matérn kernels belong in the reference bank, not the
search space. Start each idea by calling `propose_formulation`. State the exact mathematical form,
why it is PSD, its closest known kernel and concrete difference, and the BO behavior it should
improve. Only then implement it and submit the returned formulation ID.

Prefer the `input_transform` contract when inventing geometry. Its required function is
`transform_point(x, parameters)`. The function receives one immutable 1-D point and must return
one nonempty 1-D transformed point. Do not define a function named `input_transform`, and do not reshape the
point into a batch. The trusted runtime stacks those vectors and evaluates
`parameters["base_kernel"]` on the transformed inputs. For example, the base specification is
`{"kind": "matern52", "lengthscale": 0.5}`. The transform must be dimension-agnostic and independent
of other observations. A spectral candidate defines
`spectral_components(input_dimension, parameters)` so its returned frequencies can always have
shape `[m, input_dimension]`. Pointwise, residual, additive, and genuinely new spectral
constructions are also permitted. Independent formulations are root candidates and must not cite unrelated discoveries as
parents. A genuine revision must cite its actual parent, and a parameter-only edit of a parent is
rejected.

Candidate-specific parameters are fixed unless they are declared in a staged candidate's tuning
space and optimized by the backend. Do not call fixed constants learnable. On normalized `[0, 1]`
inputs express frequencies in cycles, for example `sin(2*pi*f*x)`. Prefer the formulation, staging,
verification, parameter-optimization, immutable-submission, and full-evaluation tool sequence.

Explore at least two substantially different constructions. Inspect the archive before proposing a
revision, and do not treat a syntactic rewrite or parameter-only change as a novel kernel.
