"""Contact survives particle splitting and compact grid growth with either sparse flag."""

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.serial, pytest.mark.cpu]

CASES = [
    pytest.param(dim, contact, configuration, mapping, axisymmetric, marks=pytest.mark.isolated_dimension(dim))
    for dim, contact in [(2, "MPMContact"), (2, "GeoContact"), (2, "DEMContact"), (3, "MPMContact"), (3, "GeoContact")]
    for configuration, mapping, axisymmetric in (
        [("ULMPM", "USL", False), ("TLMPM", "USL", False)]
        + ([("ULMPM", "USF", False), ("ULMPM", "MUSL", False), ("ULMPM", "USL", True)] if dim == 2 else [])
    )
]


@pytest.mark.parametrize("dimension,contact,configuration,mapping,axisymmetric", CASES)
@pytest.mark.parametrize("sparse", [False, True])
def test_adaptive_contact_steps(
    dimension, contact, configuration, mapping, axisymmetric, sparse, tmp_path, monkeypatch, adaptive=True, max_level=1
):
    from geotaichi import MPM, init
    from src.mpm.engines.ULExplicitEngine import ULExplicitEngine

    monkeypatch.setenv("GEOTAICHI_FORCE_CPU", "1")
    init(dim=dimension, arch="cpu", cpu_max_num_threads=1, offline_cache=False, log=False)
    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=[0.4] * dimension,
        gravity=[0.0] * dimension,
        alphaPIC=0.0,
        mapping=mapping,
        configuration=configuration,
        is_2DAxisy=axisymmetric,
        shape_function="GIMP" if axisymmetric else "Linear",
        solver_type="Explicit",
        material_type="Solid",
        sparse_grid=sparse,
        visualize=False,
        log=False,
    )
    mpm.set_solver(
        solver={"Timestep": 1e-5, "SimulationTime": 3e-5, "SaveInterval": 1.0, "SavePath": str(tmp_path)}, log=False
    )
    mpm.memory_allocate(
        memory={
            "max_material_number": 2,
            "max_particle_number": 512,
            "max_constraint_number": {"max_velocity_constraint": 32},
        },
        log=False,
    )
    if contact == "DEMContact":
        mpm.add_contact(
            contact,
            first={"materialID": 1, "stiffness": [100.0, 50.0], "friction": 0.3},
            second={"materialID": 2, "stiffness": [100.0, 50.0], "friction": 0.3},
        )
    else:
        mpm.add_contact(contact, friction=0.3, penalty=[0.5, 2.0])
    for mat in (1, 2):
        mpm.add_material(
            model="LinearElastic",
            material={"MaterialID": mat, "Density": 1800.0, "YoungModulus": 2e5, "PoissonRatio": 0.3},
        )
    mpm.add_element(
        element={
            "ElementType": "Q4N2D" if dimension == 2 else "R8N3D",
            "ElementSize": [0.1] * dimension,
            "AdaptiveGrid": (
                {
                    "RefineInterval": 1,
                    "RefineThreshold": 1.0,
                    "RefineCriterion": "EquivalentStress",
                    "RefineRatio": 1.0,
                    "BufferCells": 0,
                    "MaxLevel": max_level,
                }
                if adaptive
                else None
            ),
        }
    )
    for body in (0, 1):
        origin = [0.1] * dimension
        origin[0] += body * 0.05
        mpm.add_region(
            region={
                "Name": f"body{body}",
                "Type": "Rectangle2D" if dimension == 2 else "Rectangle",
                "BoundingBoxPoint": origin,
                "BoundingBoxSize": [0.05] * dimension,
            }
        )
        template = {
            "RegionName": f"body{body}",
            "nParticlesPerCell": 2,
            "BodyID": 0 if contact == "DEMContact" else body,
            "MaterialID": body + 1,
            "InitialVelocity": [1.0 if body == 0 else 0.0] + [0.0] * (dimension - 1),
            "FixVelocity": ["Free"] * dimension,
        }
        if contact == "GeoContact" and body == 1:
            template["RigidBody"] = True
            template["FixVelocity"] = ["Fix"] * dimension
        mpm.add_body(body={"Template": template})
    polygon = np.array([[0.05, 0.05], [0.3, 0.05], [0.3, 0.1], [0.05, 0.1]])
    if contact == "DEMContact":
        mpm.add_polygons(body={"Vertices": polygon, "InitialVelocity": [0.0, 0.1]})
    count = int(mpm.scene.particleNum[0])
    stress = mpm.scene.particle.stress.to_numpy()
    stress[:count, 0] = 100.0
    mpm.scene.particle.stress.from_numpy(stress)
    mass = mpm.scene.particle.m.to_numpy()[:count].sum()
    material_ids = mpm.scene.particle.materialID.to_numpy()[:count]
    expected_count = np.count_nonzero(material_ids == 0) + np.count_nonzero(material_ids > 0) * (
        2 ** (dimension * max_level) if adaptive else 1
    )
    mpm.add_boundary_condition()
    mpm.select_save_data(particle=True, grid=False, object=False)
    forces = []
    method = {
        "MPMContact": "compute_contact_force",
        "GeoContact": "compute_geocontact_force",
        "DEMContact": "compute_demcontact_force_2D",
    }[contact]
    original = getattr(ULExplicitEngine, method)

    def record_contact(self, sims, scene):
        original(self, sims, scene)
        force = scene.node.contact_force.to_numpy()
        assert np.isfinite(force).all()
        if contact != "DEMContact":
            np.testing.assert_allclose(force.sum(axis=1), 0.0, atol=1e-8)
        forces.append(np.linalg.norm(force))

    monkeypatch.setattr(ULExplicitEngine, method, record_contact)
    mpm.run()
    new_count = int(mpm.scene.particleNum[0])
    assert new_count == expected_count
    assert mpm.sims.current_step == 3
    if adaptive:
        assert mpm.scene.element.strong_hanging_constraint
    assert mpm.scene.sparse_grid is None
    assert len(forces) == 3 and max(forces) > 0
    if adaptive:
        assert max(forces[1:]) > 0, "contact must still act after refinement"
    np.testing.assert_allclose(mpm.scene.particle.m.to_numpy()[:new_count].sum(), mass, rtol=1e-6)
    assert np.isfinite(mpm.scene.particle.v.to_numpy()[:new_count]).all()
    if contact == "DEMContact":
        np.testing.assert_allclose(mpm.scene.contact.polygon_vertices.to_numpy(), polygon + [0.0, 3e-6], atol=1e-10)


@pytest.mark.isolated_dimension(2)
@pytest.mark.parametrize("contact", ["MPMContact", "GeoContact", "DEMContact"])
def test_uniform_contact_regression(contact, tmp_path, monkeypatch):
    test_adaptive_contact_steps(2, contact, "ULMPM", "USL", False, False, tmp_path, monkeypatch, adaptive=False)


@pytest.mark.parametrize("dimension,contact,configuration,mapping,axisymmetric", CASES)
@pytest.mark.parametrize("sparse", [False, True])
def test_two_level_contact_refinement(
    dimension, contact, configuration, mapping, axisymmetric, sparse, tmp_path, monkeypatch
):
    test_adaptive_contact_steps(
        dimension, contact, configuration, mapping, axisymmetric, sparse, tmp_path, monkeypatch, max_level=2
    )
