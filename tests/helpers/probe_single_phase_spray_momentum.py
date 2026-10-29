"""Diagnostic only: audit pressure impulse of two isolated, underresolved liquid particles."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import taichi as ti
from geotaichi import MPM, init


@ti.kernel
def seed(particle: ti.template(), uniform: int, dimension: ti.template()):
    for i in range(2):
        fraction = 0.04 if i == 0 else 0.015
        position = ti.Vector.zero(float, dimension)
        position[0] = 0.075 + 0.01 * i
        for d in ti.static(range(1, dimension)):
            position[d] = 0.034
        particle[i].x = position
        particle[i].vol = fraction * 0.01**dimension
        particle[i].m = 1000.0 * particle[i].vol
        velocity = ti.Vector.zero(float, dimension)
        velocity[0] = 1.7 - (0.0 if uniform else 0.5) * i
        particle[i].v = velocity
        particle[i].velocity_gradient = ti.Matrix.zero(float, dimension, dimension)
        particle[i].pressure = 0.0


@ti.kernel
def measure_grid(node: ti.template(), result: ti.template(), dimension: ti.template()):
    result.fill(0.0)
    for d in ti.static(range(dimension)):
        for I in ti.grouped(node.m[d]):
            ti.atomic_add(result[d], node.m[d][I] * node.velocity[d][I])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=("MGPCG", "PCG"), default="MGPCG")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=2)
    args = parser.parse_args()
    dimension = args.dimension
    args.output.mkdir(parents=True, exist_ok=True)
    init(dim=dimension, arch="cpu", cpu_max_num_threads=2, default_fp="float64", offline_cache=True, log=False)
    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=[0.16] + [0.10] * (dimension - 1),
        gravity=[0.0] * dimension,
        background_damping=0.0,
        alphaPIC=1.0,
        mapping="USL",
        shape_function="QuadBSpline",
        velocity_projection="Affine",
        material_type="Fluid",
        solver_type="Implicit",
        discretization="FDM",
        fluid_level_set=True,
        particle_shifting=False,
        density_projection=False,
    )
    mpm.set_implicit_solver_parameters(
        linear_solver=args.solver, multilevel=2, max_iteration_number=200, residual_tolerance=1e-10
    )
    mpm.set_solver({"Timestep": 2e-5, "SimulationTime": 2e-5, "SaveInterval": 2e-5, "SavePath": str(args.output)})
    mpm.memory_allocate({"max_material_number": 1, "max_particle_number": 8, "dof_multiplier": 3})
    mpm.add_material(
        model="Newtonian",
        material={
            "MaterialID": 1,
            "Density": 1000.0,
            "Modulus": 2e6,
            "Viscosity": 0.0,
            "ElementLength": 0.01,
            "cL": 1.5,
            "cQ": 2.0,
            "atmospheric_pressure": 0.0,
        },
    )
    mpm.add_element({"ElementType": "Staggered", "ElementSize": [0.01] * dimension, "GhostCell": 1})
    mpm.add_region(
        {
            "Name": "pair",
            "Type": "Rectangle2D" if dimension == 2 else "Rectangle",
            "BoundingBoxPoint": [0.07] + [0.03] * (dimension - 1),
            "BoundingBoxSize": [0.02] + [0.01] * (dimension - 1),
        }
    )
    mpm.add_body(
        {
            "Template": [
                {
                    "RegionName": "pair",
                    "nParticlesPerCell": 1,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "InitialVelocity": [0.0] * dimension,
                    "FixVelocity": ["Free"] * dimension,
                }
            ]
        }
    )
    mpm.add_essentials()
    assert int(mpm.scene.particleNum[0]) == 2
    seed(mpm.scene.particle, 1, dimension)
    mpm.enginer.pre_calculation(mpm.sims, mpm.scene, mpm.neighbor)
    engine, scene = mpm.enginer, mpm.scene
    moment = ti.field(float, shape=dimension)

    def grid_momentum():
        measure_grid(scene.node, moment, dimension)
        return moment.to_numpy().copy()

    def particle_momentum():
        return (scene.particle.m.to_numpy()[:2, None] * scene.particle.v.to_numpy()[:2]).sum(axis=0)

    attribute = "apply_pressures_mgpcg_step" if args.solver == "MGPCG" else "apply_pressures_step"
    original = getattr(engine, attribute)
    current = {}

    def pressure(sims, scene):
        before = grid_momentum()
        current["grid_momentum_before_pressure"] = before.tolist()
        if not current["pressure_bypassed"]:
            original(sims, scene)
        after = grid_momentum()
        current["grid_momentum_after_pressure"] = after.tolist()
        current["pressure_impulse"] = (after - before).tolist()
        current["occupied_cells"] = []
        for x in (7, 8):
            cell = (x,) + (3,) * (dimension - 1)
            linear = sum(cell[d] * int(np.prod(list(scene.element.cnum)[:d])) for d in range(dimension))
            current["occupied_cells"].append(
                {
                    "cell": cell,
                    "type": int(scene.element.cell.type[cell]),
                    "fraction": float(scene.element.cell_volumefrac[linear]),
                    "sdf": float(scene.element.cell.fluid_sdf[cell]),
                    "pressure": float(scene.element.cell.pressure[cell]),
                }
            )

    setattr(engine, attribute, pressure)
    rows = []
    for name, uniform, bypass in [
        ("uniform_control", 1, False),
        ("unequal_velocity", 0, False),
        ("pressure_bypass_control", 0, True),
    ]:
        seed(scene.particle, uniform, dimension)
        scene.element.cell.pressure.fill(0.0)
        current = {"case": name, "pressure_bypassed": bypass}
        initial = particle_momentum()
        current["initial_particle_momentum"] = initial.tolist()
        mpm.solver.core(scene, mpm.neighbor)
        final = particle_momentum()
        current["final_particle_momentum"] = final.tolist()
        current["particle_impulse"] = (final - initial).tolist()
        current["grid_momentum_after_boundary"] = grid_momentum().tolist()
        current["finite"] = bool(np.isfinite(scene.particle.v.to_numpy()[:2]).all())
        current["active_particles"] = int(np.count_nonzero(scene.particle.active.to_numpy()[:2]))
        assert current["finite"]
        assert current["active_particles"] == 2
        np.testing.assert_allclose(current["grid_momentum_before_pressure"], initial, rtol=1e-7, atol=1e-12)
        if uniform or bypass:
            np.testing.assert_allclose(final, initial, rtol=1e-7, atol=1e-12)
        rows.append(current)
        print("SPRAY_MOMENTUM " + json.dumps(current), flush=True)
    result = {
        "diagnostic_only": True,
        "solver": args.solver,
        "dimension": dimension,
        "zero_gravity_viscosity_surface_tension": True,
        "particle_support_away_from_walls": True,
        "units": "2D unit-depth momentum" if dimension == 2 else "3D momentum",
        "rows": rows,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("src/mpm/engines/EngineKernel.py", "src/mpm/engines/IncompressibleEngine.py")
        },
    }
    (args.output / "momentum_audit.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
