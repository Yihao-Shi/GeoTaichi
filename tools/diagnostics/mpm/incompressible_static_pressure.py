import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


def run_case(save_path=None):
    """Run once in an isolated directory, or preserve output at ``save_path``."""

    if save_path is not None:
        explicit_path = os.fspath(save_path)
        os.makedirs(explicit_path, exist_ok=True)
        return _run_case(explicit_path)

    with tempfile.TemporaryDirectory(
        prefix="geotaichi-incompressible-static-pressure-"
    ) as temporary_directory:
        return _run_case(temporary_directory)


def _run_case(save_path):
    import numpy as np

    from geotaichi import MPM, init

    use_shifting = os.environ.get("GT_SHIFTING", "1") != "0"
    use_density_projection = os.environ.get("GT_DENSITY_PROJECTION", "1") != "0"
    steps = int(os.environ.get("GT_STEPS", "2"))

    init(dim=3, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)

    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=[0.3, 0.2, 0.3],
        dimension="3-Dimension",
        background_damping=0.0,
        alphaPIC=0.5,
        mapping="USL",
        shape_function="QuadBSpline",
        gravity=[0.0, 0.0, -9.8],
        material_type="Fluid",
        velocity_projection="Affine",
        solver_type="Implicit",
        discretization="FDM",
        fluid_level_set=False,
        particle_shifting=use_shifting,
        density_projection=use_density_projection,
        log=False,
    )
    mpm.set_implicit_solver_parameters(
        linear_solver="MGPCG",
        multilevel=2,
        pre_and_post_smoothing=2,
        bottom_smoothing=8,
        max_iteration_number=80,
        residual_tolerance=1.0e-8,
    )
    mpm.set_solver(
        solver={
            "Timestep": 1.0e-4,
            "SimulationTime": 1.0e-4,
            "SaveInterval": 1.0e-4,
            "SavePath": save_path,
        },
        log=False,
    )
    mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": 4096,
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
            "ElementLength": 0.025,
            "cL": 1.5,
            "cQ": 2.0,
            "atmospheric_pressure": 0.0,
        },
    )
    mpm.add_element(
        element={
            "ElementType": "Staggered",
            "ElementSize": [0.05, 0.05, 0.05],
            "GhostCell": 3,
        }
    )
    mpm.add_region(
        region=[
            {
                "Name": "fluid",
                "Type": "Rectangle",
                "BoundingBoxPoint": [0.05, 0.0, 0.0],
                "BoundingBoxSize": [0.1, 0.2, 0.15],
            }
        ]
    )
    mpm.add_body(
        body={
            "Template": [
                {
                    "RegionName": "fluid",
                    "nParticlesPerCell": 2,
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
            {
                "BoundaryType": "SolidCell",
                "Norm": [-1.0, 0.0, 0.0],
                "StartPoint": [0.0, 0.0, 0.0],
                "EndPoint": [0.0, 0.2, 0.3],
                "CellThickness": 3,
            },
            {
                "BoundaryType": "SolidCell",
                "Norm": [1.0, 0.0, 0.0],
                "StartPoint": [0.3, 0.0, 0.0],
                "EndPoint": [0.3, 0.2, 0.3],
                "CellThickness": 3,
            },
            {
                "BoundaryType": "SolidCell",
                "Norm": [0.0, -1.0, 0.0],
                "StartPoint": [0.0, 0.0, 0.0],
                "EndPoint": [0.3, 0.0, 0.3],
                "CellThickness": 3,
            },
            {
                "BoundaryType": "SolidCell",
                "Norm": [0.0, 1.0, 0.0],
                "StartPoint": [0.0, 0.2, 0.0],
                "EndPoint": [0.3, 0.2, 0.3],
                "CellThickness": 3,
            },
            {
                "BoundaryType": "SolidCell",
                "Norm": [0.0, 0.0, -1.0],
                "StartPoint": [0.0, 0.0, 0.0],
                "EndPoint": [0.3, 0.2, 0.0],
                "CellThickness": 3,
            },
        ]
    )

    mpm.add_essentials(gravity_field=True)
    mpm.enginer.pre_calculation(mpm.sims, mpm.scene, mpm.neighbor)
    for _ in range(steps):
        mpm.solver.core(mpm.scene, mpm.neighbor)

    pressure = mpm.scene.element.cell.pressure.to_numpy()
    cell_type = mpm.scene.element.cell.type.to_numpy()
    ghost = int(mpm.scene.element.ghost_cell)
    active = np.asarray(mpm.scene.element.cnum, dtype=np.int64) - 2 * ghost
    interior_pressure = pressure[ghost : ghost + active[0], ghost : ghost + active[1], ghost : ghost + active[2]]
    interior_type = cell_type[ghost : ghost + active[0], ghost : ghost + active[1], ghost : ghost + active[2]]

    max_span = 0.0
    for i in range(active[0]):
        for k in range(active[2]):
            column_mask = interior_type[i, :, k] == 1
            if np.count_nonzero(column_mask) > 1:
                values = interior_pressure[i, column_mask, k]
                max_span = max(max_span, float(np.max(values) - np.min(values)))

    pressure_scale = 1000.0 * 9.8 * 0.15
    tolerance = 2.0e-3 * pressure_scale
    print(
        f"particle_shifting={use_shifting} density_projection={use_density_projection} "
        f"steps={steps} max_y_pressure_span={max_span:.6e} tolerance={tolerance:.6e}"
    )
    if not np.isfinite(max_span) or max_span > tolerance:
        raise RuntimeError(f"Unexpected pressure variation along y: {max_span}")


if __name__ == "__main__":
    run_case(os.environ.get("GT_OUTPUT_PATH"))
