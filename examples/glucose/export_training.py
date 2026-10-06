"""Shared GlucoseBench harness side: acquisition policy and public training export.

Kernaut has no experiment design, so every kernel sees the same training episodes: the two
passive GlucoseBench episodes plus four distinct menu actions drawn with a fixed per-patient
seed. Zero-bolus actions with a nonzero delay duplicate their zero-delay twin, so they are
excluded before sampling. Only public training payloads are exported; sealed test outcomes
stay inside glucosebench.Evaluator and are not exported.

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
    """Four distinct interventions, seeded by patient identity only (never by outcomes)."""
    distinct = sorted(
        i for i, p in menu.items() if p["bolus_U"] > 0 or p["bolus_minute"] == p["meal_minute"]
    )
    seed = int(hashlib.sha256(f"kernaut-glucose-v1|{patient}{salt}".encode()).hexdigest()[:8], 16)
    return [int(i) for i in np.random.default_rng(seed).choice(distinct, size=4, replace=False)]


def benchmark_with_training(patient, acq="default"):
    """acq: 'default' (the searched protocol), 'seedK' (another random draw), or 'demo'."""
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
