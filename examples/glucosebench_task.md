# CGM forecasting kernel discovery

Design one certified Gaussian-process kernel that acts as a transferable prior for forecasting a
simulated type-1 diabetes patient's continuous glucose monitor (CGM) response to a meal and an
insulin bolus. Each patient runs six-hour episodes. In each episode the patient eats one meal and
may receive one insulin bolus, and CGM is read every five minutes. Every episode of a patient
starts from the same physiological state, so episodes differ only through their intervention and
sensor noise.

Every GP input is one CGM reading, a five-dimensional vector in `[0, 1]` with coordinates in
this order:

1. reading time, minutes divided by 360;
2. meal size, grams divided by 60;
3. meal start time, minutes divided by 360;
4. bolus size, insulin units divided by 1.5 (zero means no bolus); and
5. bolus start time, minutes divided by 360 (equal to the meal start when there is no bolus).

Coordinates 2-5 are constant within an episode, so the kernel decides both how readings
correlate over time and how whole trajectories share strength across interventions.

This is a mathematical-novelty campaign. Discovered closure trees are rejected because ordinary
sums and products of standard kernels belong in the baseline bank. Start every candidate with
`propose_formulation`, giving its exact mathematical form, PSD argument, closest known kernel,
behavioral distinction, and a falsification test. Then stage, verify, tune, and submit it through the
normal immutable-candidate workflow.

Prefer `input_transform`, `feature_map`, `residual_input_transform`, or genuinely new `spectral`
constructions. The trusted interpreter constructs the Gram matrix; candidate code should only
return the certified pointwise transform, features, or spectral components required by its
contract. The input dimension is fixed at five for this benchmark. Independent ideas have no
parents, while a real revision must cite the candidate it changes.

Develop at least two structurally different proposals before refining the strongest one. Do not
claim that a covariance is itself a recovered physiological model.
