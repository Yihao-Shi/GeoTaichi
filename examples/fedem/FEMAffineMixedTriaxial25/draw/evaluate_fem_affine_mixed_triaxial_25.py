"""Evaluation and postprocessing for fem_affine_mixed_triaxial_25."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv
import json
import math
import os
import numpy as np

from examples.fedem.FEMAffineMixedTriaxial25.fem_affine_mixed_triaxial_25_parameters import (
    CONTACT_DHAT,
    CONTACT_KAPPA,
    FEM_EE_STENCILS_PER_COMPONENT_PAIR,
    FEM_PT_STENCILS_PER_COMPONENT_PAIR,
    LOWER,
    PARTICLE_COUNT,
    PLATE_SPAN,
    SOFT_PERCENT,
    UPPER,
)


def plate_specs():
    inner = LOWER + 0.008
    return (
        ("left", (LOWER - 0.020, inner, inner), (0.020, PLATE_SPAN, PLATE_SPAN)),
        ("right", (UPPER, inner, inner), (0.020, PLATE_SPAN, PLATE_SPAN)),
        ("front", (inner, LOWER - 0.020, inner), (PLATE_SPAN, 0.020, PLATE_SPAN)),
        ("back", (inner, UPPER, inner), (PLATE_SPAN, 0.020, PLATE_SPAN)),
        ("bottom", (inner, inner, LOWER - 0.020), (PLATE_SPAN, PLATE_SPAN, 0.020)),
        ("top", (inner, inner, UPPER), (PLATE_SPAN, PLATE_SPAN, 0.020)),
    )


def write_results(output, parsed, soft_ids, rigid_ids, curve, result):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "stress_strain.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strain = np.asarray([row["axial_strain"] for row in curve])
    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.plot(strain, np.asarray([row["deviator_stress"] for row in curve]) / 1.0e3, label="q")
    axis.plot(strain, np.asarray([row["axial_stress"] for row in curve]) / 1.0e3, label="sigma_z")
    axis.plot(strain, np.asarray([row["mean_stress"] for row in curve]) / 1.0e3, label="p")
    axis.set(xlabel="Axial compressive strain", ylabel="Stress (kPa)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "stress_strain.png", dpi=180)
    plt.close(fig)

    final = curve[-1]
    summary = {
        "case": "confined_triaxial_compression_fem_abd",
        "soft_percent": SOFT_PERCENT,
        "soft_percent_realized": 100.0 * soft_ids.size / PARTICLE_COUNT,
        "soft_particle_count": int(soft_ids.size),
        "stiff_particle_count": int(rigid_ids.size),
        "particle_count": PARTICLE_COUNT,
        "tetrahedra_per_soft_particle": 20 * 4**parsed.subdivisions,
        "friction_coefficient": parsed.friction,
        "fem_mass_damping_per_second": parsed.fem_damping,
        "linear_solver_relative_tolerance": parsed.linear_solver_relative_tolerance,
        "contact_dhat": CONTACT_DHAT,
        "physical_contact_barrier_stiffness": CONTACT_KAPPA,
        "abd_contact_barrier_stiffness": parsed.abd_contact_kappa,
        "steps": int(result["step"]),
        "time": float(result["time"]),
        "converged": bool(result["converged"]),
        "final_axial_strain": final["axial_strain"],
        "final_deviator_stress": final["deviator_stress"],
        "peak_deviator_stress": max(row["deviator_stress"] for row in curve),
        "maximum_force_imbalance": max(row["force_imbalance"] for row in curve),
        "final_force_imbalance": final["force_imbalance"],
        "minimum_jacobian": min(row["minimum_jacobian"] for row in curve[1:]),
        "soft_particle_ids": soft_ids.tolist(),
        "stiff_particle_ids": rigid_ids.tolist(),
        "capacity_basis": {
            "fem_connected_component_count": int(soft_ids.size) + len(plate_specs()),
            "fem_unordered_component_pair_bound": math.comb(int(soft_ids.size) + len(plate_specs()), 2),
            "point_triangle_stencils_per_fem_component_pair": FEM_PT_STENCILS_PER_COMPONENT_PAIR,
            "edge_edge_stencils_per_fem_component_pair": FEM_EE_STENCILS_PER_COMPONENT_PAIR,
            "maximum_rigid_grain_coordination": max(int(rigid_ids.size) - 1, 0),
            "default_mixed_cross_body_pair_bound": 60,
            "point_triangle_stencils_per_mixed_body_pair": 64,
            "edge_edge_stencils_per_mixed_body_pair": 128,
            "fem_point_triangle_pairs": parsed.fem_pt_cap,
            "fem_edge_edge_pairs": parsed.fem_ee_cap,
            "abd_point_triangle_pairs": parsed.abd_pt_cap,
            "abd_edge_edge_pairs": parsed.abd_ee_cap,
            "mixed_point_triangle_pairs": parsed.mixed_pt_cap,
            "mixed_edge_edge_pairs": parsed.mixed_ee_cap,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + os.linesep, encoding="utf-8")
