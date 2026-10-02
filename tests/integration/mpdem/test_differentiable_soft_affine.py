import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.mpdem,
    pytest.mark.ipc,
    pytest.mark.contact,
    pytest.mark.cpu,
    pytest.mark.slow,
    pytest.mark.serial,
]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def _build_case(tmp_path):
    from geotaichi import MPDEM, init, polyhedron

    init(
        arch=os.environ.get("GEOTAICHI_TEST_ARCH", "cpu"),
        cpu_max_num_threads=2,
        default_fp="float64",
        offline_cache=False,
        log=False,
    )
    coupling = MPDEM(log=False)
    coupling.set_configuration(
        domain=[1.0, 1.0, 1.0],
        coupling_scheme="MPDEM",
        particle_interaction=True,
        wall_interaction=False,
        gravity=[0.0, 0.0, 0.0],
        search="LinkedCell",
        visualize=False,
        log=False,
    )
    dem = coupling.dem
    dem.set_configuration(
        domain=[1.0, 1.0, 1.0],
        boundary=["Destroy", "Destroy", "Destroy"],
        gravity=[0.0, 0.0, 0.0],
        search="LinkedCell",
        scheme="LSMPM",
        soft_rigid_contact="IPC",
        shape_function="Linear",
        soft_mechanical_grid_spacing_ratio=0.3,
        visualize=False,
        log=False,
    )
    dem.set_affine_body_parameters(
        contact_model="BarrierIPC",
        assemble_type="HashTriplet",
        dhat=0.03,
        barrier_stiffness=2.0e4,
        friction_epsv=1.0e-3,
        friction_mode="lagged",
        friction_iterations=1,
        max_newton_iteration=40,
        newton_tolerance=1.0e-7,
        linear_tolerance=1.0e-10,
        linear_max_iteration=2000,
        line_search_max_iteration=16,
        max_step=0.02,
        ccd=True,
        ccd_type="ccd",
    )
    dem.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_rigid_body_number": 0,
            "max_soft_body_number": 1,
            "max_material_point_number": 4096,
            "max_rigid_template_number": 1,
            "levelset_grid_number": 4096,
            "soft_grid_number": 10000,
            "surface_node_number": 1024,
            "body_coordination_number": 8,
            "wall_coordination_number": 0,
            "verlet_distance_multiplier": [0.15, 0.15],
            "point_coordination_number": [16, 8],
            "compaction_ratio": [1.0, 1.0],
        },
        log=False,
    )
    dt = 5.0e-4
    coupling.set_solver(
        {
            "Timestep": dt,
            "SimulationTime": 2.0 * dt,
            "SaveInterval": 2.0 * dt,
            "SavePath": os.fspath(tmp_path),
            "enable_step_retry": False,
        },
        log=False,
    )
    dem.add_attribute(
        materialID=0,
        attribute={
            "Density": 1200.0,
            "ConstitutiveModel": "DruckerPrager",
            "YoungModulus": 5.0e3,
            "PoissonRatio": 0.3,
            "FrictionAngle": 20.0,
            "DilationAngle": 20.0,
            "Cohesion": 10.0,
            "dpType": "Circumscribed",
            "ForceLocalDamping": 0.0,
            "TorqueLocalDamping": 0.0,
        },
    )
    dem.add_template(
        {
            "Name": "soft_sphere",
            "Object": polyhedron(file=os.path.join(ROOT, "assets/mesh/LSDEM/sphere.stl")).grids(space=0.25, extent=1),
        }
    )
    dem.add_template(
        {
            "Name": "affine_octahedron",
            "TemplateType": "AffineBody",
            "Object": polyhedron(file=os.path.join(ROOT, "tests/data/affine_octahedron.obj")),
        }
    )
    dem.create_body(
        {
            "BodyType": "SoftBody",
            "Template": {
                "Name": "soft_sphere",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoint": [0.38, 0.5, 0.5],
                "BoundingRadius": 0.08,
                "MaterialPointsPerCell": 1,
                "InitialVelocity": [0.20, 0.12, 0.0],
                "InitialAngularVelocity": [0.0, 0.0, 0.0],
                "FixMotion": ["Free", "Free", "Free"],
            },
        }
    )
    dem.create_body(
        {
            "BodyType": "AffineBody",
            "Template": {
                "Name": "affine_octahedron",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoint": [0.54, 0.5, 0.5],
                "BoundingRadius": 0.08,
                "InitialVelocity": [-0.10, -0.05, 0.0],
                "InitialAngularVelocity": [0.0, 0.0, 0.0],
                "Friction": 0.35,
            },
        }
    )
    dem.add_property(
        0,
        0,
        {
            "Dhat": 0.03,
            "BarrierStiffness": 2.0e4,
            "ContactDampingStiffness": 0.0,
            "Friction": 0.35,
        },
        dType="all",
    )
    dem.add_essentials()
    dem.enginer.initialize(dem.sims, dem.scene)
    return dem


def test_soft_affine_rejects_plastic_material(tmp_path):
    with pytest.raises(ValueError, match="hyperelastic-only"):
        _build_case(tmp_path)
    return

    # Legacy plastic trajectory regression retained below as documentation of
    # the removed soft-particle path; plasticity now belongs to ordinary Direct
    # MPM only.
    dem = _build_case(tmp_path)
    engine = dem.enginer
    operator = engine.operator
    scene = dem.scene
    model = engine.soft_material.matProps.model
    soft_count = int(operator.soft_point_num)
    affine_count = int(operator.affine.control_num)

    deformation = scene.soft_point.F.to_numpy()
    deformation[:soft_count] = np.array([[1.08, 0.03, 0.0], [0.01, 0.94, 0.0], [0.0, 0.0, 1.0]])
    scene.soft_point.F.from_numpy(deformation)
    model.equivalent_plastic_strain.fill(0.02)

    trajectory = dem.differentiable_soft_affine(steps=2)
    initial = {
        "affine_position": operator.affine.y.to_numpy()[:affine_count].copy(),
        "affine_velocity": operator.affine.velocity_y.to_numpy()[:affine_count].copy(),
        "position": scene.soft_point.x.to_numpy()[:soft_count].copy(),
        "velocity": scene.soft_point.v.to_numpy()[:soft_count].copy(),
        "deformation_gradient": scene.soft_point.F.to_numpy()[:soft_count].copy(),
        "plastic_deformation_inverse": model.plastic_deformation_inverse.to_numpy()[:soft_count].copy(),
        "equivalent_plastic_strain": model.equivalent_plastic_strain.to_numpy()[:soft_count].copy(),
        "volumetric_plastic_strain": model.volumetric_plastic_strain.to_numpy()[:soft_count].copy(),
    }
    for _ in range(2):
        trajectory.step()

    rng = np.random.default_rng(27)
    seed = {
        "affine_position": 0.1 * rng.normal(size=(affine_count, 3)),
        "affine_velocity": 1.0e-3 * rng.normal(size=(affine_count, 3)),
        "position": 0.1 * rng.normal(size=(soft_count, 3)),
        "velocity": 1.0e-3 * rng.normal(size=(soft_count, 3)),
        "deformation_gradient": 0.03 * rng.normal(size=(soft_count, 3, 3)),
        "plastic_deformation_inverse": 0.03 * rng.normal(size=(soft_count, 3, 3)),
        "equivalent_plastic_strain": 0.05 * rng.normal(size=soft_count),
        "volumetric_plastic_strain": 0.05 * rng.normal(size=soft_count),
    }
    gradient = trajectory.backward(seed)
    assert int(operator.adjoint_mixed_friction_count[None]) > 0
    assert np.max(model.equivalent_plastic_strain.to_numpy()[:soft_count]) > 0.02000001

    translation = 0.1 * rng.normal(size=3)
    velocity_shift = 0.1 * rng.normal(size=3)
    direction = {
        "affine_position": np.tile(translation, (affine_count, 1)),
        "affine_velocity": np.tile(velocity_shift, (affine_count, 1)),
        "position": np.tile(translation, (soft_count, 1)),
        "velocity": np.tile(velocity_shift, (soft_count, 1)),
        # A common rigid shift preserves the frozen lagged contact cache.
        # Plastic/history directions are covered by the material-force and
        # standalone trajectory finite differences; perturbing them here
        # would differentiate cache refresh, which is intentionally stopped.
        "deformation_gradient": np.zeros_like(initial["deformation_gradient"]),
        "plastic_deformation_inverse": np.zeros_like(initial["plastic_deformation_inverse"]),
        "equivalent_plastic_strain": np.zeros_like(initial["equivalent_plastic_strain"]),
        "volumetric_plastic_strain": np.zeros_like(initial["volumetric_plastic_strain"]),
    }

    def reset(shift):
        operator.affine.y.from_numpy(
            np.ascontiguousarray(initial["affine_position"] + shift * direction["affine_position"])
        )
        operator.affine.velocity_y.from_numpy(
            np.ascontiguousarray(initial["affine_velocity"] + shift * direction["affine_velocity"])
        )
        for name, field in (
            ("position", scene.soft_point.x),
            ("velocity", scene.soft_point.v),
            ("deformation_gradient", scene.soft_point.F),
        ):
            values = field.to_numpy()
            values[:soft_count] = initial[name] + shift * direction[name]
            field.from_numpy(values)
        for name, field in (
            ("plastic_deformation_inverse", model.plastic_deformation_inverse),
            ("equivalent_plastic_strain", model.equivalent_plastic_strain),
            ("volumetric_plastic_strain", model.volumetric_plastic_strain),
        ):
            values = field.to_numpy()
            values[:soft_count] = initial[name] + shift * direction[name]
            field.from_numpy(values)
        dem.sims.current_time = trajectory.initial_time
        dem.sims.current_step = trajectory.initial_step

    def objective(shift):
        reset(shift)
        for _ in range(2):
            engine.step(dem.sims, scene)
            dem.sims.current_time += trajectory.dt
            dem.sims.current_step += 1
        return float(
            np.sum(seed["affine_position"] * operator.affine.y.to_numpy()[:affine_count])
            + np.sum(seed["affine_velocity"] * operator.affine.velocity_y.to_numpy()[:affine_count])
            + np.sum(seed["position"] * scene.soft_point.x.to_numpy()[:soft_count])
            + np.sum(seed["velocity"] * scene.soft_point.v.to_numpy()[:soft_count])
            + np.sum(seed["deformation_gradient"] * scene.soft_point.F.to_numpy()[:soft_count])
            + np.sum(seed["plastic_deformation_inverse"] * model.plastic_deformation_inverse.to_numpy()[:soft_count])
            + np.sum(seed["equivalent_plastic_strain"] * model.equivalent_plastic_strain.to_numpy()[:soft_count])
            + np.sum(seed["volumetric_plastic_strain"] * model.volumetric_plastic_strain.to_numpy()[:soft_count])
        )

    analytic = sum(np.sum(gradient[f"initial_{name}"] * direction[name]) for name in direction)
    step = 2.0e-6
    numerical = (objective(step) - objective(-step)) / (2.0 * step)
    assert analytic == pytest.approx(numerical, rel=5.0e-3, abs=2.0e-6)
    # Cache refresh is a stop-gradient operation in the lagged contract; the
    # fixed-cache scale derivative has its own finite-difference unit test.
    assert np.isfinite(gradient["friction_scale"])
    assert abs(gradient["friction_scale"]) > 1.0e-12
