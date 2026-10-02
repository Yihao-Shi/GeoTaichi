"""Plot GeoTaichi's IPC barrier and a penalty-law comparison."""

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.physics_model.contact_model.ipc.IPC import (
    ipc_barrier_distance_terms_py,
)


def barrier_profile(distances, active_distance, kappa):
    samples = [
        ipc_barrier_distance_terms_py(
            distance,
            active_distance,
            kappa=kappa,
        )
        for distance in distances
    ]
    energy, gradient, hessian = np.asarray(samples).T
    return energy, -gradient, hessian


def penalty_profile(overlap, kappa):
    energy = kappa / 2.5 * overlap**2.5
    force = kappa * overlap**1.5
    tangent = 1.5 * kappa * np.sqrt(overlap)
    return energy, force, tangent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kappa", type=float, default=1.0e4)
    parser.add_argument("--output")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    distances = np.linspace(1.0e-3, 0.12, 200)
    for active_distance in (0.04, 0.06, 0.12):
        _, force, _ = barrier_profile(distances, active_distance, args.kappa)
        plt.plot(
            distances,
            force,
            label=rf"IPC barrier, $\hat d={active_distance:g}$",
        )

    overlap = np.linspace(0.12, 0.0, 200)
    _, force, _ = penalty_profile(overlap, args.kappa)
    plt.plot(0.12 - overlap, force, "--", label="penalty")
    plt.xlabel("distance")
    plt.ylabel("contact force")
    plt.legend(frameon=False)
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
