"""Native affine transfer survives particle motion in UL, TL, and Newmark engines."""

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.serial]


@pytest.mark.isolated_dimension(2)
@pytest.mark.parametrize(
    "configuration,solver_type,mapping,mode,stabilize,affine_field",
    [
        ("ULMPM", "Explicit", "USF", "Normal", None, False),
        ("ULMPM", "Explicit", "G2P2G", "Normal", None, False),
        ("TLMPM", "Explicit", "MUSL", "Normal", None, False),
        ("ULMPM", "Implicit", "USL", "Normal", None, False),
        ("ULMPM", "Explicit", "USF", "Lightweight", None, False),
        ("ULMPM", "Explicit", "MUSL", "Lightweight", "B-Bar Method", False),
        ("ULMPM", "Explicit", "USL", "Lightweight", "F-Bar Method", False),
        ("ULMPM", "Explicit", "USF", "Normal", None, True),
        ("ULMPM", "Explicit", "USF", "Normal", "Displacement F-Bar Method", True),
        ("ULMPM", "Explicit", "USF", "Lightweight", None, True),
        ("ULMPM", "Explicit", "USL", "Lightweight", "F-Bar Method", True),
    ],
)
def test_native_affine_uniform_translation(
    tmp_path, taichi_runtime, configuration, solver_type, mapping, mode, stabilize, affine_field
):
    from src.mpm.mainMPM import MPM

    mpm = MPM(log=False)
    displacement_fbar = stabilize == "Displacement F-Bar Method"
    mpm.set_configuration(
        dimension=2,
        domain=[0.4, 0.4],
        gravity=[0, 0],
        configuration=configuration,
        solver_type=solver_type,
        velocity_projection="Affine",
        mapping=mapping,
        material_type="Fluid" if displacement_fbar else "Solid",
        mode=mode,
        stabilize=stabilize,
        shape_function="GIMP" if stabilize == "B-Bar Method" else "QuadBSpline",
        visualize=False,
        log=False,
    )
    if solver_type == "Implicit":
        mpm.set_implicit_solver_parameters(
            assemble_type="MatrixFree", max_iteration_number=8, displacement_tolerance=1e-5, residual_tolerance=1e-8
        )
    duration = 1e-4 if affine_field else 2e-4
    mpm.set_solver(
        solver={"Timestep": 1e-4, "SimulationTime": duration, "SaveInterval": duration, "SavePath": str(tmp_path)},
        log=False,
    )
    mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": 16,
            "max_constraint_number": {"max_velocity_constraint": 16, "max_displacement_constraint": 16},
        },
        log=False,
    )
    mpm.add_material(
        model="Newtonian" if displacement_fbar else "LinearElastic",
        material={
            "MaterialID": 1,
            "Density": 1800,
            "Modulus": 2e4,
            "Viscosity": 1e-3,
            "YoungModulus": 2e4,
            "PoissonRatio": 0.3,
        },
    )
    mpm.add_element(element={"ElementType": "Q4N2D", "ElementSize": [0.1, 0.1]})
    mpm.add_region(
        region={"Name": "block", "Type": "Rectangle2D", "BoundingBoxPoint": [0.15, 0.15], "BoundingBoxSize": [0.1, 0.1]}
    )
    mpm.add_body(
        body={
            "Template": {
                "RegionName": "block",
                "nParticlesPerCell": 1,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [0.2, 0.1],
                "FixVelocity": ["Free", "Free"],
            }
        }
    )
    mpm.add_boundary_condition()
    mpm.select_save_data(particle=True, grid=False, object=False)
    count = int(mpm.scene.particleNum[0])
    gradient = np.array([[0.7, 0.2], [-0.1, 0.3]]) if affine_field else np.zeros((2, 2))
    initial_volume = mpm.scene.particle.vol.to_numpy()[:count].copy()
    if affine_field:
        gradients = mpm.scene.particle.velocity_gradient.to_numpy()
        gradients[:count] = gradient
        mpm.scene.particle.velocity_gradient.from_numpy(gradients)
    mpm.run()
    assert count > 0
    assert mpm.enginer.compute == mpm.enginer.affine_updating
    np.testing.assert_allclose(mpm.scene.particle.v.to_numpy()[:count], [[0.2, 0.1]] * count, atol=2e-6)
    np.testing.assert_allclose(mpm.scene.particle.velocity_gradient.to_numpy()[:count], [gradient] * count, atol=2e-5)
    np.testing.assert_allclose(
        mpm.scene.particle.x.to_numpy()[:count], [[0.2 + duration * 0.2, 0.2 + duration * 0.1]] * count, atol=2e-6
    )
    if affine_field:
        np.testing.assert_allclose(
            mpm.scene.particle.vol.to_numpy()[:count],
            initial_volume * np.linalg.det(np.eye(2) + duration * gradient),
            rtol=2e-6,
        )
        assert np.isfinite(mpm.scene.particle.stress.to_numpy()[:count]).all()
