"""BlockScan integration must match dense-grid contact, including remapping and boundaries."""

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.serial, pytest.mark.cpu]

CASES = [
    pytest.param(dim, contact, marks=pytest.mark.isolated_dimension(dim))
    for dim, contacts in [(2, ("MPMContact", "GeoContact", "DEMContact")), (3, ("MPMContact", "GeoContact"))]
    for contact in contacts
]


@pytest.mark.parametrize("dimension,contact", CASES)
@pytest.mark.parametrize("configuration", ["ULMPM", "TLMPM"])
@pytest.mark.parametrize("mapping", ["USL", "USF", "MUSL"])
def test_blockscan_contact_matches_dense(
    dimension, contact, configuration, mapping, tmp_path, monkeypatch, axisymmetric=False, stabilize=None
):
    from geotaichi import MPM, init
    from src.mpm.engines.Engine import Engine
    from src.mpm.engines.ULExplicitEngine import ULExplicitEngine

    monkeypatch.setenv("GEOTAICHI_FORCE_CPU", "1")
    method = {
        None: "compute_grid_kinematic",
        "MPMContact": "compute_contact_force",
        "GeoContact": "compute_geocontact_force",
        "DEMContact": "compute_demcontact_force_2D",
    }[contact]
    original_contact = getattr(ULExplicitEngine, method)
    original_interpolation = Engine.calculate_interpolations
    results = []
    for sparse in (False, True):
        init(dim=dimension, arch="cpu", cpu_max_num_threads=1, offline_cache=False, log=False)
        mpm = MPM(log=False)
        mpm.set_configuration(
            domain=[1.2] * dimension,
            gravity=([0.0, -9.8] if contact is None else [0.0] * dimension),
            alphaPIC=0.0,
            mapping=mapping,
            configuration=configuration,
            shape_function="GIMP" if axisymmetric or mapping == "USF" else "Linear",
            is_2DAxisy=axisymmetric,
            stabilize=stabilize,
            solver_type="Explicit",
            material_type="Solid",
            visualize=False,
            log=False,
            sparse_grid={"BlockSize": 2, "MaxActiveBlocks": 16} if sparse else False,
        )
        mpm.set_solver(
            solver={
                "Timestep": 1e-5,
                "SimulationTime": 3e-5,
                "SaveInterval": 1.0,
                "SavePath": str(tmp_path / str(sparse)),
            },
            log=False,
        )
        mpm.memory_allocate(
            memory={
                "max_material_number": 2,
                "max_particle_number": 64,
                "max_constraint_number": {"max_velocity_constraint": 1024, "max_reflection_constraint": 1024},
            },
            log=False,
        )
        if contact == "DEMContact":
            mpm.add_contact(
                contact,
                first={"materialID": 1, "stiffness": [100.0, 50.0], "friction": 0.3},
                second={"materialID": 2, "stiffness": [100.0, 50.0], "friction": 0.3},
            )
        elif contact is not None:
            mpm.add_contact(contact, friction=0.3, penalty=[0.5, 2.0])
        for material in (1, 2):
            mpm.add_material(
                model="LinearElastic",
                material={"MaterialID": material, "Density": 1800.0, "YoungModulus": 2e5, "PoissonRatio": 0.3},
            )
        mpm.add_element(
            element={"ElementType": "Q4N2D" if dimension == 2 else "R8N3D", "ElementSize": [0.1] * dimension}
        )
        for body in (0, 1):
            origin = [0.4] * dimension
            origin[0] = 0.35 + body * 0.05
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
                "BodyID": 0 if contact in (None, "DEMContact") else body,
                "MaterialID": body + 1,
                "InitialVelocity": [1.0 if body == 0 else 0.0] + [0.2] * (dimension - 1),
                "FixVelocity": ["Free"] * dimension,
            }
            if contact == "GeoContact" and body == 1:
                template.update(RigidBody=True, FixVelocity=["Fix"] * dimension)
            mpm.add_body(body={"Template": template})
        polygon = np.array([[0.3, 0.3], [0.9, 0.3], [0.9, 0.4], [0.3, 0.4]])
        if contact == "DEMContact":
            mpm.add_polygons(body={"Vertices": polygon, "InitialVelocity": [0.0, 0.1]})
        boundaries = []
        for kind, coordinate in (("VelocityConstraint", 0.4), ("ReflectionConstraint", 0.5)):
            start, end = [0.3] * dimension, [0.9] * dimension
            start[-1] = end[-1] = coordinate
            boundary = {"BoundaryType": kind, "StartPoint": start, "EndPoint": end}
            if kind == "VelocityConstraint":
                boundary["Velocity"] = [None] * (dimension - 1) + [0.0]
            else:
                boundary["Norm"] = [0.0] * (dimension - 1) + [1.0]
            boundaries.append(boundary)
        mpm.add_boundary_condition(boundary=boundaries)
        mpm.select_save_data(particle=True, grid=False, object=False)
        count = int(mpm.scene.particleNum[0])
        mass = mpm.scene.particle.m.to_numpy()[:count].sum()
        forces, blocks = [], []

        def record_contact(self, sims, scene):
            original_contact(self, sims, scene)
            force = (scene.node.force if contact is None else scene.node.contact_force).to_numpy()
            assert np.isfinite(force).all()
            forces.append(np.concatenate((force.sum(axis=0).ravel(), [np.linalg.norm(force)])))
            if scene.sparse_grid is not None:
                blocks.append(scene.sparse_grid.active_block_ids.to_numpy().copy())

        def translate_then_interpolate(self, sims, scene):
            # Force a block-map change independently of the stable physical timestep.
            if sims.current_step == 1:
                positions = scene.particle.x.to_numpy()
                positions[:count, 0] += 0.2
                scene.particle.x.from_numpy(positions)
                if contact == "DEMContact":
                    vertices = scene.contact.polygon_vertices.to_numpy()
                    vertices[:, 0] += 0.2
                    scene.contact.polygon_vertices.from_numpy(vertices)
            original_interpolation(self, sims, scene)

        with monkeypatch.context() as patch:
            patch.setattr(ULExplicitEngine, method, record_contact)
            patch.setattr(Engine, "calculate_interpolations", translate_then_interpolate)
            mpm.run()
        assert mpm.sims.current_step == 3
        assert len(forces) == 3 and forces[0][-1] > 0
        np.testing.assert_allclose(mpm.scene.particle.m.to_numpy()[:count].sum(), mass, rtol=1e-12)
        positions = mpm.scene.particle.x.to_numpy()[:count]
        velocities = mpm.scene.particle.v.to_numpy()[:count]
        assert np.isfinite(positions).all() and np.isfinite(velocities).all()
        if sparse:
            grid = mpm.scene.sparse_grid
            assert grid is not None and grid.get_active_blocks() > 0
            assert grid.node_capacity < mpm.scene.element.gridSum
            assert len(blocks) == 3
            if configuration == "ULMPM":
                assert not np.array_equal(blocks[0], blocks[1])
            else:
                np.testing.assert_array_equal(blocks[0], blocks[1])
        if contact == "DEMContact":
            shift = 0.2 if configuration == "ULMPM" else 0.0
            np.testing.assert_allclose(
                mpm.scene.contact.polygon_vertices.to_numpy(), polygon + [shift, 3e-6], atol=1e-10
            )
        results.append((positions, velocities, np.array(forces)))
    for dense, sparse in zip(*results):
        np.testing.assert_allclose(sparse, dense, rtol=1e-8, atol=1e-8)


@pytest.mark.isolated_dimension(2)
@pytest.mark.parametrize("contact", [None, "MPMContact", "GeoContact", "DEMContact"])
@pytest.mark.parametrize("mapping", ["USL", "USF", "MUSL"])
@pytest.mark.parametrize("stabilize", [None, "B-Bar Method"])
def test_axisymmetric_blockscan_matches_dense(contact, mapping, stabilize, tmp_path, monkeypatch):
    test_blockscan_contact_matches_dense(
        2, contact, "ULMPM", mapping, tmp_path, monkeypatch, axisymmetric=True, stabilize=stabilize
    )
