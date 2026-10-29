"""Run through refinement and the following physical step with compact adaptive nodes."""

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.serial, pytest.mark.cpu]


@pytest.mark.parametrize(
    "dimension",
    [
        pytest.param(2, marks=pytest.mark.isolated_dimension(2)),
        pytest.param(3, marks=pytest.mark.isolated_dimension(3)),
    ],
)
@pytest.mark.parametrize("solver", ["Explicit", "Implicit"])
@pytest.mark.parametrize("sparse", [False, True])
def test_adaptive_refinement_and_following_step(dimension, solver, sparse, tmp_path, monkeypatch, options=None):
    from geotaichi import MPM, init

    options = dict(options or {})
    max_level = options.get("MaxLevel", 1)
    split = options.get("RefineParticles", True)
    assembly = options.get("assemble_type", "COO")
    monkeypatch.setenv("GEOTAICHI_FORCE_CPU", "1")
    init(dim=dimension, arch="cpu", cpu_max_num_threads=1, offline_cache=False, log=False)
    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=[0.4] * dimension,
        gravity=[0.0] * dimension,
        alphaPIC=0.0,
        mapping=options.get("mapping", "MUSL"),
        configuration=options.get("configuration", "ULMPM"),
        shape_function=options.get("shape", "Linear"),
        stabilize=options.get("stabilize"),
        is_2DAxisy=options.get("axisymmetric", False),
        particle_traction_method=options.get("traction_method", "Stable"),
        solver_type=solver,
        material_type="Solid",
        sparse_grid=sparse,
        pressure_smoothing=solver == "Explicit",
        visualize=False,
        log=False,
    )
    if solver == "Implicit":
        mpm.set_implicit_solver_parameters(
            assemble_type=assembly,
            max_iteration_number=20,
            displacement_tolerance=1e-6,
            residual_tolerance=1e-8,
        )
    mpm.set_solver(
        solver={
            "Timestep": 1e-5,
            "SimulationTime": 3e-5,
            "SaveInterval": 1.0,
            "SavePath": str(tmp_path),
        },
        log=False,
    )
    mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": 128,
            "max_constraint_number": {
                "max_displacement_constraint" if solver == "Implicit" else "max_velocity_constraint": 2048,
                "max_particle_traction_constraint": 128 if options.get("traction") else 0,
            },
        },
        log=False,
    )
    mpm.add_material(
        model="LinearElastic",
        material={
            "MaterialID": 1,
            "Density": 1800.0,
            "YoungModulus": 2e5,
            "PoissonRatio": 0.3,
        },
    )
    mpm.add_element(
        element={
            "ElementType": "Q4N2D" if dimension == 2 else "R8N3D",
            "ElementSize": [0.1] * dimension,
            "AdaptiveGrid": {
                "RefineInterval": 1,
                "RefineThreshold": 1.0,
                "RefineCriterion": "EquivalentStress",
                "RefineRatio": 1.0,
                "BufferCells": options.get("BufferCells", 0),
                "MaxLevel": max_level,
                "RefineParticles": split,
                "BridgingDomain": options.get("BridgingDomain", False),
                "HangingConstraintMode": options.get("constraint", "ShapeFunction"),
                "HangingShapeCapacityFactor": (
                    3 if dimension == 3 and options.get("shape") == "CubicBSpline" else 2**dimension
                ),
                "ParticleSplitBatch": options.get("ParticleSplitBatch", 8192),
            },
        }
    )
    mpm.add_region(
        region={
            "Name": "block",
            "Type": "Rectangle2D" if dimension == 2 else "Rectangle",
            # Keep all child positions inside the marked coarse cell through both split levels.
            "BoundingBoxPoint": [0.1] * dimension,
            "BoundingBoxSize": [0.1] * dimension,
        }
    )
    mpm.add_body(
        body={
            "Template": {
                "RegionName": "block",
                "nParticlesPerCell": 1,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [0.0] * dimension,
                "FixVelocity": ["Free"] * dimension,
                "Traction": [{"Pressure": [0.0, 100.0]}] if options.get("traction") else [],
            }
        }
    )
    count = int(mpm.scene.particleNum[0])
    stress = mpm.scene.particle.stress.to_numpy()
    stress[:count, 0] = 100.0
    mpm.scene.particle.stress.from_numpy(stress)
    if solver == "Implicit":
        mpm.scene.particle.stress0.from_numpy(stress)
    volume = mpm.scene.particle.vol.to_numpy()[:count].sum()
    mass = mpm.scene.particle.m.to_numpy()[:count].sum()
    mpm.add_boundary_condition(
        boundary=[
            {
                "BoundaryType": "DisplacementConstraint" if solver == "Implicit" else "VelocityConstraint",
                "Displacement" if solver == "Implicit" else "Velocity": [0.0] * dimension,
                "StartPoint": [0.0] * dimension,
                "EndPoint": [0.4] * (dimension - 1) + [0.0],
            }
        ]
    )
    if options.get("traction"):
        loads = mpm.scene.boundary.particle_traction
        initial_force = np.sum(2.0 * loads.traction.to_numpy() * loads.psize.to_numpy()[:, ::-1], axis=0)

    update_refinement = mpm.scene.element.update_refinement

    def device_refinement(sims, scene):
        def fail(*args, **kwargs):
            pytest.fail("Adaptive refinement attempted a bulk host/device transfer")

        with monkeypatch.context() as guard:
            for field_type in {type(scene.element.refined_cell), type(scene.element.calLength), type(scene.particle)}:
                guard.setattr(field_type, "to_numpy", fail)
                guard.setattr(field_type, "from_numpy", fail)
            update_refinement(sims, scene)

    monkeypatch.setattr(mpm.scene.element, "update_refinement", device_refinement)
    mpm.select_save_data(particle=True, grid=False, object=False)
    mpm.run()
    new_count = int(mpm.scene.particleNum[0])
    assert new_count == count * (2 ** (dimension * max_level) if split else 1)
    assert mpm.sims.current_step == 3
    assert mpm.scene.sparse_grid is None
    assert mpm.sims.sparse_grid == sparse
    np.testing.assert_allclose(mpm.scene.particle.m.to_numpy()[:new_count].sum(), mass, rtol=1e-6)
    np.testing.assert_allclose(mpm.scene.particle.vol.to_numpy()[:new_count].sum(), volume, rtol=1e-4)
    assert np.isfinite(mpm.scene.particle.v.to_numpy()[:new_count]).all()
    assert int(mpm.scene.element.refined_cell.to_numpy().max()) == max_level
    if options.get("traction"):
        loads = mpm.scene.boundary.particle_traction
        assert int(mpm.scene.boundary.ptraction_list[0]) == new_count
        tractions = loads.traction.to_numpy()[:new_count]
        if hasattr(loads, "td"):
            td = loads.td.to_numpy()[:new_count]
            tractions = (
                np.einsum("ni,nij->nj", tractions, np.linalg.inv(td).transpose(0, 2, 1)) * np.linalg.det(td)[:, None]
            )
        force = np.sum(2.0 * tractions * loads.psize.to_numpy()[:new_count, ::-1], axis=0)
        np.testing.assert_allclose(force, initial_force, rtol=1e-3, atol=1e-3)
    if solver == "Implicit":
        if assembly == "COO":
            assert mpm.enginer.iterator.coo_base_nnz > 0
        np.testing.assert_allclose(mpm.scene.particle.vol0.to_numpy()[:new_count].sum(), volume, rtol=1e-4)


MULTILEVEL_CASES = [
    ("ul_usl_linear", "Explicit", {"mapping": "USL"}),
    ("ul_usf_gimp_penalty", "Explicit", {"mapping": "USF", "shape": "GIMP", "constraint": "Penalty"}),
    ("ul_musl_quadratic", "Explicit", {"shape": "QuadBSpline"}),
    ("ul_musl_cubic", "Explicit", {"shape": "CubicBSpline"}),
    ("tl_usl_gimp", "Explicit", {"configuration": "TLMPM", "mapping": "USL", "shape": "GIMP"}),
    ("tl_usf_penalty", "Explicit", {"configuration": "TLMPM", "mapping": "USF", "constraint": "Penalty"}),
    ("tl_musl", "Explicit", {"configuration": "TLMPM"}),
    ("fbar", "Explicit", {"stabilize": "F-Bar Method"}),
    ("grid_only", "Explicit", {"RefineParticles": False, "shape": "GIMP"}),
    (
        "bridging_level_one",
        "Explicit",
        {"BridgingDomain": True, "MaxLevel": 1, "constraint": "Penalty", "shape": "GIMP"},
    ),
    ("implicit_coo", "Implicit", {"assemble_type": "COO"}),
    ("implicit_matrixfree", "Implicit", {"assemble_type": "MatrixFree"}),
    ("implicit_hashtriplet", "Implicit", {"assemble_type": "HashTriplet"}),
]


@pytest.mark.parametrize(
    "dimension",
    [
        pytest.param(2, marks=pytest.mark.isolated_dimension(2)),
        pytest.param(3, marks=pytest.mark.isolated_dimension(3)),
    ],
)
@pytest.mark.parametrize("name,solver,options", MULTILEVEL_CASES, ids=[case[0] for case in MULTILEVEL_CASES])
@pytest.mark.parametrize("sparse", [False, True])
def test_multilevel_solver_features(dimension, name, solver, options, sparse, tmp_path, monkeypatch):
    settings = {"MaxLevel": 2, **options}
    if dimension == 3:
        settings["ParticleSplitBatch"] = 2
    test_adaptive_refinement_and_following_step(dimension, solver, sparse, tmp_path, monkeypatch, settings)


@pytest.mark.isolated_dimension(2)
@pytest.mark.parametrize("options", [{"stabilize": "B-Bar Method"}, {"axisymmetric": True, "shape": "GIMP"}])
@pytest.mark.parametrize("sparse", [False, True])
def test_multilevel_2d_features(options, sparse, tmp_path, monkeypatch):
    test_adaptive_refinement_and_following_step(
        2, "Explicit", sparse, tmp_path, monkeypatch, {"MaxLevel": 2, **options}
    )


@pytest.mark.isolated_dimension(2)
@pytest.mark.parametrize("method", ["Stable", "Nanson", "Axisymmetric"])
@pytest.mark.parametrize("sparse", [False, True])
def test_multilevel_particle_traction_steps(method, sparse, tmp_path, monkeypatch):
    options = {"MaxLevel": 2, "traction": True}
    if method == "Axisymmetric":
        options.update(axisymmetric=True, shape="GIMP")
    else:
        options["traction_method"] = method
    test_adaptive_refinement_and_following_step(2, "Explicit", sparse, tmp_path, monkeypatch, options)
