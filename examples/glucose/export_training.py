"""Export public GlucoseBench training data using a fixed acquisition policy.

Kernaut does not select experiments. Each kernel receives the same training data:
two passive episodes and four distinct interventions, sampled with a fixed seed
for each patient. If the bolus is zero, different bolus delays describe the same
action. Remove those duplicates before sampling.

Export only public training data. Keep hidden test outcomes inside
glucosebench.Evaluator.

Usage: .venv/bin/python examples/glucose/export_training.py --out runs/glucosebench/training.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from glucosebench import Benchmark

PATIENTS = [f"{group}#{i:03d}" for group in ("child", "adolescent", "adult") for i in range(1, 11)]


DEMO_ACTIONS = [0, 12, 24, 36]  # the GlucoseBench README's fixed example policy


def acquisition(patient, menu, salt=""):
    """Select four distinct interventions using patient identity as the seed, never outcomes."""
    distinct = sorted(
        i for i, p in menu.items() if p["bolus_U"] > 0 or p["bolus_minute"] == p["meal_minute"]
    )
    seed = int(hashlib.sha256(f"kernaut-glucose-v1|{patient}{salt}".encode()).hexdigest()[:8], 16)
    return [int(i) for i in np.random.default_rng(seed).choice(distinct, size=4, replace=False)]


def benchmark_with_training(patient, acq="default"):
    """Create a benchmark and collect training episodes with the selected policy.

    Set ``acq`` to ``default`` for the search policy, ``seedK`` for another random
    draw, or ``demo`` for the fixed example policy.
    """
    run = Benchmark(patient, replicate=0)
    if acq == "default":
        actions = acquisition(patient, run.actions)
    elif acq == "demo":
        actions = list(DEMO_ACTIONS)
    else:
        actions = acquisition(patient, run.actions, salt=f"|{acq}")
    for index in actions:
        run.observe(index)
    return run, actions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for patient in PATIENTS:
        run, actions = benchmark_with_training(patient)
        data[patient] = dict(actions=actions, training=run.training)
        print(patient, actions, flush=True)
    args.out.write_text(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
