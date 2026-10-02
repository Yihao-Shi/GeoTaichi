import argparse
import os
import subprocess
import sys

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.serial]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def _test_arch():
    arch = os.environ.get("GEOTAICHI_TEST_ARCH", "cpu").lower()
    return "gpu" if arch == "cuda" else arch


def run_case(case_name, save_path):
    from geotaichi import MPM, init

    save_path = os.fspath(save_path)
    multi_material = case_name == "pcg_multi"
    affine_projection = case_name == "affine"
    loaded_pcg = case_name == "loaded_pcg"
    up_formulation = case_name.removesuffix("_gimp") in ("up", "hydrostatic_up")
    hydrostatic = case_name.startswith("hydrostatic_")
    hydrostatic_mgpcg = case_name == "hydrostatic_mgpcg"
    fic = "fic" in case_name
    if hydrostatic:
        domain = [0.1, 0.2]
        element_size = [0.01, 0.01]
        region_origin = [0.0, 0.0]
        region_size = [0.1, 0.1]
        semi_parameters = {
            "assemble_type": "MatrixFree",
            "linear_solver": "MGPCG" if hydrostatic_mgpcg else "PCG",
            "pressure_solver": "MGPCG" if hydrostatic_mgpcg else "PCG",
            "pressure_beta": 0.0 if hydrostatic_mgpcg else 1.0,
            "max_iteration_number": 500,
            "residual_tolerance": 1.0e-8,
        }
        if hydrostatic_mgpcg:
            semi_parameters.update(multilevel=2, pre_and_post_smoothing=2, bottom_smoothing=8)
    elif loaded_pcg:
        domain = [0.3, 1.2]
        element_size = [0.05, 0.05]
        region_origin = [0.05, 0.05]
        region_size = [0.2, 1.0]
        semi_parameters = {
            "assemble_type": "MatrixFree",
            "linear_solver": "PCG",
            "max_iteration_number": 500,
            "residual_tolerance": 1.0e-8,
        }
    elif case_name == "mgpcg":
        domain = [0.4, 0.4]
        element_size = [0.1, 0.1]
        region_origin = [0.1, 0.1]
        region_size = [0.2, 0.2]
        semi_parameters = {
            "assemble_type": "MatrixFreeMGPCG",
            "linear_solver": "MGPCG",
            "max_iteration_number": 100,
            "residual_tolerance": 1.0e-8,
            "multilevel": 2,
            "pre_and_post_smoothing": 2,
            "bottom_smoothing": 8,
        }
    else:
        domain = [0.1, 0.2] if up_formulation else [0.2, 0.2]
        element_size = [0.025, 0.025] if up_formulation else [0.1, 0.1]
        region_origin = [0.0, 0.0] if up_formulation else [0.05, 0.05]
        region_size = [0.1, 0.1]
        semi_parameters = {
            "assemble_type": "MatrixFree",
            "linear_solver": "PCG",
            "max_iteration_number": 200,
            "residual_tolerance": 1.0e-8,
        }

    if fic:
        semi_parameters["pressure_stabilize"] = "FIC"
        if "coo" in case_name:
            semi_parameters["assemble_type"] = "COO"

    init(dim=2, arch=_test_arch(), cpu_max_num_threads=2, offline_cache=False, log=False)

    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=domain,
        background_damping=0.0,
        gravity=[0.0, -9.8] if up_formulation or hydrostatic else [0.0, 0.0],
        alphaPIC=1.0 if affine_projection else 0.5,
        mapping="USF" if loaded_pcg or up_formulation else "USL",
        shape_function="GIMP" if case_name.endswith("_gimp") else "Linear",
        material_type="TwoPhaseSingleLayer",
        free_surface_detection=loaded_pcg or hydrostatic,
        solver_type="SemiImplicit_u_p" if up_formulation else "SemiImplicit",
        velocity_projection="Affine" if affine_projection else "PIC/FLIP",
    )
    mpm.set_solver(
        solver={
            "Timestep": 1.0e-5 if hydrostatic else 1.0e-4,
            "SimulationTime": 0.01 if hydrostatic else (2.0e-4 if loaded_pcg else 1.0e-4),
            "SaveInterval": 0.001 if hydrostatic else 1.0e-4,
            "SavePath": save_path,
        },
        log=False,
    )
    mpm.set_semi_implicit_solver_parameters(semi_parameters)
    mpm.memory_allocate(
        memory={
            "max_material_number": 2 if multi_material else 1,
            "max_particle_number": 512 if loaded_pcg or hydrostatic else (256 if multi_material else 128),
            "max_constraint_number": {
                "max_velocity_constraint": 256,
                "max_particle_traction_constraint": 512 if loaded_pcg else 0,
            },
        },
        log=False,
    )
    mpm.add_material(
        model="LinearElastic",
        material={
            "MaterialID": 1,
            "SolidDensity": 2650.0,
            "FluidDensity": 1000.0,
            "Porosity": 0.4,
            "FluidBulkModulus": 2.2e9 if loaded_pcg else 2.2e8,
            "CavitationPressure": 0.0,
            "Permeability": 1.0e-3 if loaded_pcg else 1.0e-4,
            "FluidViscosity": 1.0e-3,
            "GrainDiameter": 1.0e-2,
            "YoungModulus": 1.0e7 if loaded_pcg else 1.0e6,
            "PoissonRatio": 0.3,
        },
    )
    if multi_material:
        mpm.add_material(
            model="LinearElastic",
            material={
                "MaterialID": 2,
                "SolidDensity": 2200.0,
                "FluidDensity": 980.0,
                "Porosity": 0.5,
                "FluidBulkModulus": 2.2e8,
                "Permeability": 5.0e-5,
                "FluidViscosity": 2.0e-3,
                "GrainDiameter": 8.0e-3,
                "YoungModulus": 8.0e5,
                "PoissonRatio": 0.28,
            },
        )
    mpm.add_element(element={"ElementType": "Q4N2D", "ElementSize": element_size})
    regions = [
        {
            "Name": "sample",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": region_origin,
            "BoundingBoxSize": region_size,
        }
    ]
    if loaded_pcg:
        regions.append(
            {
                "Name": "loaded_surface",
                "Type": "Rectangle2D",
                "BoundingBoxPoint": [0.05, 1.025],
                "BoundingBoxSize": [0.2, 0.025],
            }
        )
    mpm.add_region(region=regions)
    templates = [
        {
            "RegionName": "sample",
            "nParticlesPerCell": 2,
            "BodyID": 0,
            "MaterialID": 1,
            "InitialVelocity": [0.0, 0.0],
            "FixVelocity": ["Free", "Free"],
        }
    ]
    if loaded_pcg:
        templates[0].update(
            {
                "ParticleStress": {
                    "GravityField": False,
                    "InternalStress": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "PorePressure": 0.0,
                },
                "Traction": [
                    {
                        "Pressure": [0.0, -1.0e4],
                        "FluidPressure": [0.0, 0.0],
                        "RegionName": "loaded_surface",
                    }
                ],
            }
        )
    if multi_material:
        templates.append(
            {
                "RegionName": "sample",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 2,
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Free", "Free"],
            }
        )
    mpm.add_body(body={"Template": templates})
    initial_pressure = None
    if up_formulation or hydrostatic:
        position = mpm.scene.particle.x.to_numpy()
        initial_pressure = mpm.scene.particle.pressure.to_numpy()
        stress = mpm.scene.particle.stress.to_numpy()
        depth = np.maximum(region_origin[1] + region_size[1] - position[: int(mpm.scene.particleNum[0]), 1], 0.0)
        initial_pressure[: len(depth)] = 1000.0 * 9.8 * depth
        stress[: len(depth), 1] = -(1.0 - 0.4) * (2650.0 - 1000.0) * 9.8 * depth
        mpm.scene.particle.pressure.from_numpy(initial_pressure)
        mpm.scene.particle.stress.from_numpy(stress)
    mpm.add_boundary_condition(
        boundary=[
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [None, 0.0],
                "StartPoint": [region_origin[0], region_origin[1]] if hydrostatic else [0.0, 0.0],
                "EndPoint": [region_origin[0] + region_size[0], region_origin[1]] if hydrostatic else [domain[0], 0.0],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None],
                "StartPoint": [region_origin[0], region_origin[1]] if hydrostatic else [0.0, 0.0],
                "EndPoint": [region_origin[0], region_origin[1] + region_size[1]] if hydrostatic else [0.0, domain[1]],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None],
                "StartPoint": (
                    [region_origin[0] + region_size[0], region_origin[1]] if hydrostatic else [domain[0], 0.0]
                ),
                "EndPoint": (
                    [region_origin[0] + region_size[0], region_origin[1] + region_size[1]]
                    if hydrostatic
                    else [domain[0], domain[1]]
                ),
            },
        ]
    )
    mpm.select_save_data(particle=True, grid=False, object=False)
    particle_num = int(mpm.scene.particleNum[0])
    initial_solid_mass = mpm.scene.particle.ms.to_numpy()[:particle_num].copy()
    mpm.run()
    if up_formulation:
        assert mpm.sims.dof_multiplier == 1
    np.testing.assert_allclose(
        mpm.scene.node.m.to_numpy(),
        mpm.scene.node.ms.to_numpy() + mpm.scene.node.mf.to_numpy(),
        rtol=1.0e-12,
    )

    position = mpm.scene.particle.x.to_numpy()[:particle_num]
    velocity = mpm.scene.particle.v.to_numpy()[:particle_num]
    solid_velocity = mpm.scene.particle.vs.to_numpy()[:particle_num]
    fluid_mass = mpm.scene.particle.mf.to_numpy()[:particle_num]
    pressure = mpm.scene.particle.pressure.to_numpy()[:particle_num]
    material_id = mpm.scene.particle.materialID.to_numpy()[:particle_num]
    porosity = mpm.scene.particle.porosity.to_numpy()[:particle_num]
    volume = mpm.scene.particle.vol.to_numpy()[:particle_num]
    solid_mass = mpm.scene.particle.ms.to_numpy()[:particle_num]
    mixture_mass = mpm.scene.particle.m.to_numpy()[:particle_num]
    assert particle_num > 0
    if multi_material:
        assert set(np.unique(material_id).tolist()) == {1, 2}
    assert np.isfinite(position).all()
    assert np.isfinite(velocity).all()
    assert np.isfinite(pressure).all()
    assert np.min(pressure) >= -1.0e-12
    np.testing.assert_array_equal(solid_mass, initial_solid_mass)
    # The material point follows the skeleton: fluid can enter/leave its pores.
    fluid_density = np.where(material_id == 2, 980.0, 1000.0)
    np.testing.assert_allclose(fluid_mass, porosity * volume * fluid_density, rtol=1.0e-12)
    np.testing.assert_allclose(mixture_mass, solid_mass + fluid_mass, rtol=1.0e-12)
    if hydrostatic:
        import glob
        import json

        records = []
        for filename in sorted(glob.glob(os.path.join(save_path, "particles", "MPMParticle*.npz"))):
            with np.load(filename) as frame:
                surface = region_origin[1] + region_size[1]
                expected = 1000.0 * 9.8 * np.maximum(surface - frame["position"][:, 1], 0.0)
                interior = frame["position"][:, 1] < surface - 0.02
                relative_error = np.linalg.norm((frame["pressure"] - expected)[interior]) / np.linalg.norm(
                    expected[interior]
                )
                records.append(
                    {
                        "time": float(frame["t_current"]),
                        "pressure_relative_l2": float(relative_error),
                        "pressure_min": float(frame["pressure"].min()),
                        "pressure_max": float(frame["pressure"].max()),
                        "solid_vmax": float(np.linalg.norm(frame["solid_velocity"], axis=1).max()),
                    }
                )
        with open(os.path.join(save_path, "hydrostatic_metrics.json"), "w") as output:
            json.dump(records, output, indent=2)
        print("Hydrostatic history:", records, flush=True)
        assert len(records) == 11
        assert max(row["pressure_relative_l2"] for row in records) < 0.05
        assert max(row["solid_vmax"] for row in records) < 0.01
    if not up_formulation:
        np.testing.assert_allclose(velocity, solid_velocity)
    elif not hydrostatic:
        assert np.max(np.abs(pressure - initial_pressure[:particle_num])) < 50.0
    if loaded_pcg:
        assert np.max(np.abs(velocity)) < 1.0
        assert np.max(np.abs(pressure)) < 1.0e4
        assert np.max(np.abs(pressure)) > 1.0e-3
        assert np.min(position) > 0.0
        assert np.max(position[:, 1]) < domain[1]
    particle_file = os.path.join(save_path, "particles", "MPMParticle000001.npz")
    assert os.path.exists(particle_file)
    if loaded_pcg:
        with np.load(particle_file) as output:
            assert {"pressure", "solid_velocity", "fluid_velocity", "porosity", "free_surface"} <= set(output.files)
        vtk_file = os.path.join(save_path, "vtks", "GraphicMPMParticle000001.vtu")
        with open(vtk_file, "rb") as vtk:
            vtk_data = vtk.read()
        for field in ("pressure", "porosity", "solid_velocity", "fluid_velocity", "free_surface"):
            assert f'Name="{field}"'.encode() in vtk_data


def _run_subprocess(case_name, output_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ROOT + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            __file__,
            "--case",
            case_name,
            "--output-root",
            os.fspath(output_path),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_single_layer_matrixfree_pcg(tmp_path):
    _run_subprocess("pcg", tmp_path / "pcg")


def test_single_layer_matrixfree_pcg_multimaterial(tmp_path):
    _run_subprocess("pcg_multi", tmp_path / "pcg-multi")


def test_single_layer_matrixfree_mgpcg(tmp_path):
    _run_subprocess("mgpcg", tmp_path / "mgpcg")


def test_single_layer_affine_projection(tmp_path):
    _run_subprocess("affine", tmp_path / "affine")


def test_single_layer_loaded_pcg_free_surface(tmp_path):
    _run_subprocess("loaded_pcg", tmp_path / "loaded-pcg")


def test_single_layer_up_pcg(tmp_path):
    _run_subprocess("up", tmp_path / "up")


@pytest.mark.parametrize("formulation", ("up", "uvp"))
@pytest.mark.parametrize("shape", ("", "_gimp"))
def test_single_layer_multistep_hydrostatic(formulation, shape, tmp_path):
    _run_subprocess("hydrostatic_" + formulation + shape, tmp_path / (formulation + shape))


def test_single_layer_mgpcg_multistep_hydrostatic(tmp_path):
    _run_subprocess("hydrostatic_mgpcg", tmp_path / "mgpcg")


def test_single_layer_fic_hydrostatic_assembly_equivalence(tmp_path):
    for assembly in ("fic", "fic_coo"):
        _run_subprocess("hydrostatic_" + assembly + "_gimp", tmp_path / assembly)
    # Both paths must retain the same hydrostatic field, not merely finish.
    for frame in range(11):
        name = f"particles/MPMParticle{frame:06d}.npz"
        with np.load(tmp_path / "fic" / name) as matrixfree, np.load(tmp_path / "fic_coo" / name) as coo:
            for field in ("position", "solid_velocity", "fluid_velocity", "pressure", "porosity"):
                np.testing.assert_allclose(coo[field], matrixfree[field], rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("solver_type", ("SemiImplicit_u_p", "Explicit"))
def test_single_layer_rejects_ignored_pressure_stabilization(solver_type):
    import taichi as ti

    from src.mpm.Simulation import Simulation

    ti.init(arch=ti.cpu, offline_cache=False, log_level=ti.ERROR)
    try:
        sims = Simulation()
        sims.dimension = 2
        sims.solver_type = solver_type
        sims.material_type = "TwoPhaseSingleLayer"
        sims.set_pressure_stabilize_technique("FIC")
        with pytest.raises(RuntimeError, match="FIC"):
            sims.validate_configuration(require_solver_parameters=False)
    finally:
        ti.reset()


@pytest.mark.parametrize(
    ("solver_type", "mapping"),
    (("SemiImplicit", "MUSL"), ("SemiImplicit_u_p", "USL")),
)
def test_single_layer_rejects_unimplemented_mappings(solver_type, mapping):
    import taichi as ti

    from src.mpm.Simulation import Simulation

    ti.init(arch=ti.cpu, offline_cache=False, log_level=ti.ERROR)
    try:
        sims = Simulation()
        sims.dimension = 2
        sims.discretization = "FEM"
        sims.material_type = "TwoPhaseSingleLayer"
        sims.solver_type = solver_type
        sims.mapping = mapping
        with pytest.raises(RuntimeError, match="mapping"):
            sims.validate_configuration(require_solver_parameters=False)
    finally:
        ti.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "pcg",
            "pcg_multi",
            "mgpcg",
            "affine",
            "loaded_pcg",
            "up",
            "hydrostatic_up",
            "hydrostatic_uvp",
            "hydrostatic_mgpcg",
            "hydrostatic_up_gimp",
            "hydrostatic_uvp_gimp",
            "hydrostatic_fic_gimp",
            "hydrostatic_fic_coo_gimp",
        ),
        default="pcg",
    )
    parser.add_argument("--output-root", required=True)
    arguments = parser.parse_args()
    run_case(arguments.case, arguments.output_root)
