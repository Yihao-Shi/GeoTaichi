import numpy as np
import pytest

pytestmark = [
    pytest.mark.verification,
    pytest.mark.mpm,
    pytest.mark.cpu,
    pytest.mark.slow,
    pytest.mark.serial,
]


def run_case(
    save_path,
    *,
    use_shifting,
    use_density_projection,
    density_projection_interior_only=True,
    shape_function="QuadBSpline",
    linear_solver="MGPCG",
    steps=4,
):
    from geotaichi import MPM, init

    save_path = str(save_path)

    domain = np.array([0.16, 0.10], dtype=np.float64)
    element_size = np.array([0.01, 0.01], dtype=np.float64)

    init(dim=2, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=domain.tolist(),
        background_damping=0.0,
        alphaPIC=0.5,
        mapping="USL",
        shape_function=shape_function,
        gravity=[0.0, -9.8],
        material_type="Fluid",
        solver_type="Implicit",
        discretization="FDM",
        velocity_projection="Affine",
        particle_shifting=use_shifting,
        density_projection=use_density_projection,
        density_projection_interior_only=density_projection_interior_only,
        log=False,
    )
    if linear_solver == "MGPCG":
        mpm.set_implicit_solver_parameters(
            linear_solver=linear_solver,
            multilevel=2,
            pre_and_post_smoothing=2,
            bottom_smoothing=8,
            max_iteration_number=80,
            residual_tolerance=1.0e-8,
        )
    else:
        mpm.set_implicit_solver_parameters(
            linear_solver=linear_solver,
            max_iteration_number=80,
            residual_tolerance=1.0e-8,
        )
    mpm.set_solver(
        {
            "Timestep": 5.0e-4,
            "SimulationTime": steps * 5.0e-4,
            "SaveInterval": steps * 5.0e-4,
            "SavePath": save_path,
        },
        log=False,
    )
    mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": 20000,
            "dof_multiplier": 3,
            "max_constraint_number": {},
        },
        log=False,
    )
    mpm.add_material(
        model="Newtonian",
        material={
            "MaterialID": 1,
            "Density": 1000.0,
            "Modulus": 2.0e6,
            "Viscosity": 1.0e-3,
            "ElementLength": 0.01,
            "cL": 1.5,
            "cQ": 2.0,
            "atmospheric_pressure": 0.0,
        },
    )
    mpm.add_element(
        element={
            "ElementType": "Staggered",
            "ElementSize": element_size.tolist(),
            "GhostCell": 1,
        }
    )
    mpm.add_region(
        region={
            "Name": "water",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": [0.0, 0.0],
            "BoundingBoxSize": [0.08, 0.05],
        }
    )
    mpm.add_body(
        body={
            "Template": [
                {
                    "RegionName": "water",
                    "nParticlesPerCell": 3,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "FixVelocity": ["Free", "Free", "Free"],
                }
            ]
        }
    )
    mpm.add_boundary_condition(
        boundary=[
            {"BoundaryType": "SolidCell", "Norm": [0.0, -1.0], "StartPoint": [0.0, 0.0], "EndPoint": [domain[0], 0.0], "CellThickness": 1},
            {"BoundaryType": "SolidCell", "Norm": [-1.0, 0.0], "StartPoint": [0.0, 0.0], "EndPoint": [0.0, domain[1]], "CellThickness": 1},
            {"BoundaryType": "SolidCell", "Norm": [1.0, 0.0], "StartPoint": [domain[0], 0.0], "EndPoint": [domain[0], domain[1]], "CellThickness": 1},
            {"BoundaryType": "SolidCell", "Norm": [0.0, 1.0], "StartPoint": [0.0, domain[1]], "EndPoint": [domain[0], domain[1]], "CellThickness": 1},
        ]
    )

    mpm.add_essentials(gravity_field=True)
    mpm.enginer.pre_calculation(mpm.sims, mpm.scene, mpm.neighbor)
    for _ in range(steps):
        mpm.solver.core(mpm.scene, mpm.neighbor)
        mpm.sims.current_time += mpm.sims.delta
        mpm.sims.current_step += 1

    particle_num = int(mpm.scene.particleNum[0])
    position = mpm.scene.particle.x.to_numpy()[:particle_num]
    pressure = mpm.scene.particle.pressure.to_numpy()[:particle_num]
    active = mpm.scene.particle.active.to_numpy()[:particle_num] == 1
    position = position[active]
    pressure = pressure[active]

    if position.size == 0:
        raise RuntimeError("All particles became inactive in the dam-break control test")
    if not np.isfinite(position).all() or not np.isfinite(pressure).all():
        raise RuntimeError("Non-finite particle state in the dam-break control test")

    lower_violation = np.min(position, axis=0)
    upper_violation = np.max(position, axis=0) - domain
    if np.any(lower_violation < -2.0e-3) or np.any(upper_violation > 2.0e-3):
        raise RuntimeError(f"Particle escaped domain: lower={lower_violation}, upper={upper_violation}")

    cell_id = np.floor(np.clip(position, 0.0, domain - 1.0e-12) / element_size).astype(np.int64)
    flat = cell_id[:, 0] + cell_id[:, 1] * int(round(domain[0] / element_size[0]))
    counts = np.bincount(flat)
    max_occupancy = int(counts.max()) if counts.size else 0
    near_right_wall = int(np.count_nonzero(position[:, 0] > domain[0] - 1.5 * element_size[0]))
    print(
        f"shape={shape_function} particle_shifting={use_shifting} "
        f"linear_solver={linear_solver} "
        f"density_projection={use_density_projection} "
        f"density_projection_interior_only={density_projection_interior_only} steps={steps} "
        f"active_particles={position.shape[0]} bbox_min={np.min(position, axis=0)} "
        f"bbox_max={np.max(position, axis=0)} max_cell_occupancy={max_occupancy} "
        f"near_right_wall={near_right_wall} pressure_min={np.nanmin(pressure):.6e} "
        f"pressure_max={np.nanmax(pressure):.6e}"
    )
    if max_occupancy > 64:
        raise RuntimeError(f"Particle clustering is too large: max cell occupancy={max_occupancy}")
    return {
        "active_particles": int(position.shape[0]),
        "max_cell_occupancy": max_occupancy,
        "near_right_wall": near_right_wall,
        "position": position,
        "pressure": pressure,
    }


@pytest.mark.parametrize(
    ("use_shifting", "use_density_projection"),
    [(True, False), (False, True)],
    ids=("particle-shifting", "density-projection"),
)
def test_incompressible_dam_break_controls(
    tmp_path, use_shifting, use_density_projection
):
    result = run_case(
        tmp_path
        / (
            f"dam_break_shift{int(use_shifting)}_"
            f"density_projection{int(use_density_projection)}"
        ),
        use_shifting=use_shifting,
        use_density_projection=use_density_projection,
    )

    assert result["active_particles"] > 0
    assert result["max_cell_occupancy"] <= 64
    assert np.isfinite(result["position"]).all()
    assert np.isfinite(result["pressure"]).all()
