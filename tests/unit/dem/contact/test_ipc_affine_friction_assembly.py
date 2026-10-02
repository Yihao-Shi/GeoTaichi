import os
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

pytestmark = [
    pytest.mark.unit,
    pytest.mark.dem,
    pytest.mark.ipc,
    pytest.mark.contact,
    pytest.mark.assembly,
    pytest.mark.cpu,
    pytest.mark.serial,
]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))


def _build_affine_contact(save_path, with_wall=False, friction_mode="lagged"):
    from geotaichi import DEM, polyhedron

    dem = DEM(log=False)
    dem.set_configuration(
        domain=[4.0, 2.0, 2.0],
        scheme="AffineBody",
        search="LinkedCell",
        gravity=[0.0, 0.0, 0.0],
        visualize=False,
        log=False,
    )
    dem.set_affine_body_parameters(
        assemble_type="COO",
        young_modulus=2.0e4,
        friction_epsv=1.0e-3,
        local_damping=0.0,
        contact_damping_stiffness=0.0,
        hessian_shift=0.0,
        friction_mode=friction_mode,
    )
    memory = {
        "max_material_number": 1,
        "max_affine_body_number": 2,
        "surface_node_number": 16,
        "body_coordination_number": 8,
        "wall_coordination_number": 1,
        "compaction_ratio": [1.0, 1.0],
    }
    if with_wall:
        memory["max_plane_number"] = 1
    dem.memory_allocate(memory=memory, log=False)
    dem.set_solver(
        {
            "Timestep": 1.0e-3,
            "SimulationTime": 1.0e-3,
            "SaveInterval": 1.0e-3,
            "SavePath": os.fspath(save_path),
        },
        log=False,
    )
    dem.add_attribute(materialID=0, attribute={"Density": 1200.0})
    dem.add_template(
        template={
            "Name": "oct",
            "TemplateType": "AffineBody",
            "Object": polyhedron(file=os.path.join(ROOT, "tests/data/affine_octahedron.obj")),
        }
    )
    dem.create_body(
        body={
            "BodyType": "AffineBody",
            "Template": [
                {
                    "Name": "oct",
                    "GroupID": 0,
                    "MaterialID": 0,
                    "BodyPoint": [1.775, 1.0, 1.0],
                    "ScaleFactor": 0.2,
                    "InitialVelocity": [0.0, 0.0, 0.0],
                },
                {
                    "Name": "oct",
                    "GroupID": 0,
                    "MaterialID": 0,
                    "BodyPoint": [2.225, 1.0, 1.0],
                    "ScaleFactor": 0.2,
                    "InitialVelocity": [0.0, 0.0, 0.0],
                },
            ],
        }
    )
    if with_wall:
        dem.add_wall(
            {
                "WallType": "Plane",
                "MaterialID": 0,
                "WallCenter": np.array([0.0, 0.0, 0.75]),
                "OuterNormal": np.array([0.0, 0.0, 1.0]),
            }
        )
    dem.add_property(
        materialID1=0,
        materialID2=0,
        property={
            "Dhat": 0.25,
            "BarrierStiffness": 2.0e4,
            "ContactDampingStiffness": 0.0,
            "Friction": 0.4,
        },
        dType="particle-particle",
    )
    if with_wall:
        dem.add_property(
            materialID1=0,
            materialID2=0,
            property={
                "Dhat": 0.25,
                "BarrierStiffness": 2.0e4,
                "ContactDampingStiffness": 0.0,
                "Friction": 0.4,
            },
            dType="particle-wall",
        )
    dem.add_essentials()
    dem.enginer.initialize(dem.sims, dem.scene)
    state = dem.enginer.state
    state.begin_step(float(dem.sims.dt[None]))
    dem.enginer.operator.initialize_contact_damping(state.pack(), state.hat_y)
    if friction_mode == "lagged":
        assert int(dem.enginer.operator.friction_contact_count[0]) > 0
    else:
        # Fully implicit friction rebuilds current PT/EE geometry inside each
        # Newton residual/Jacobian assembly; no frozen lagged collisions exist.
        assert int(dem.enginer.operator.friction_contact_count[0]) == 0
    return dem, state.hat_y.reshape(-1).copy()


def _official_barrier_terms(distance, dhat, kappa):
    distance2 = distance * distance
    active_distance2 = dhat * dhat
    safe_distance2 = max(distance2, max(1.0e-30 * active_distance2, 1.0e-300))
    if safe_distance2 >= active_distance2:
        return 0.0, 0.0
    diff = safe_distance2 - active_distance2
    log_term = np.log(safe_distance2 / active_distance2)
    energy = -kappa * diff * diff * log_term
    gradient_distance2 = -kappa * (2.0 * diff * log_term + diff * diff / safe_distance2)
    normal_force = max(-2.0 * gradient_distance2 * distance, 0.0)
    return energy, normal_force


def _assert_official_collision_weights(dem, expected_kinds):
    from src.dem.engines.AffineBodyOperator import MATRIX_COO

    operator = dem.enginer.operator
    count = min(
        int(operator.friction_contact_count[0]),
        operator.friction_contact_capacity,
    )
    bodies = operator.friction_contact_bodies.to_numpy()[:count]
    vertices = operator.friction_contact_vertices.to_numpy()[:count]
    weights = operator.friction_contact_weights.to_numpy()[:count]
    coefficients = operator.friction_contact_coeff.to_numpy()[:count]
    x = operator.x.to_numpy()[: operator.vertex_num]
    node_area = operator.node_area.to_numpy()[: operator.vertex_num]
    edges = operator.edges.to_numpy()[: operator.edge_num]
    edge_area = operator.edge_area.to_numpy()[: operator.edge_num]
    edge_ids = {tuple(sorted((int(edge[0]), int(edge[1])))): edge_id for edge_id, edge in enumerate(edges)}
    body_material = operator.body_material.to_numpy()[: operator.body_num]
    body_mu = operator.body_mu.to_numpy()[: operator.body_num]
    pp_dhat = operator.pp_dhat.to_numpy()
    pp_kappa = operator.pp_kappa.to_numpy()
    pp_mu = operator.pp_mu.to_numpy()
    pw_dhat = operator.pw_dhat.to_numpy()
    pw_kappa = operator.pw_kappa.to_numpy()
    pw_mu = operator.pw_mu.to_numpy()
    wall_material = operator.wall_material.to_numpy()[: max(operator.wall_num, 1)]
    wall_mu = operator.wall_mu.to_numpy()[: max(operator.wall_num, 1)]
    wall_point = operator.wall_point.to_numpy()[: max(operator.wall_num, 1)]
    wall_normal = operator.wall_normal.to_numpy()[: max(operator.wall_num, 1)]

    energy_by_kind = {"vf": 0.0, "ee": 0.0, "wall": 0.0}
    seen_kinds = set()
    for contact_id in range(count):
        contact_bodies = bodies[contact_id]
        contact_vertices = vertices[contact_id]
        body_i = int(contact_bodies[0])
        material_i = int(body_material[body_i])

        if np.all(contact_bodies == contact_bodies[0]) and np.all(contact_vertices == contact_vertices[0]):
            kind = "wall"
            wall_id = 0
            material_j = int(wall_material[wall_id])
            dhat = float(pw_dhat[material_i, material_j])
            kappa = float(pw_kappa[material_i, material_j])
            mu = float(pw_mu[material_i, material_j])
            if mu < 0.0:
                mu = max(float(body_mu[body_i]), float(wall_mu[wall_id]))
            vertex_id = int(contact_vertices[0])
            distance = float(np.dot(x[vertex_id] - wall_point[wall_id], wall_normal[wall_id]))
            collision_weight = float(node_area[vertex_id])
        elif (
            contact_bodies[0] != contact_bodies[1]
            and contact_bodies[1] == contact_bodies[2]
            and contact_bodies[2] == contact_bodies[3]
        ):
            kind = "vf"
            body_j = int(contact_bodies[1])
            material_j = int(body_material[body_j])
            dhat = float(pp_dhat[material_i, material_j])
            kappa = float(pp_kappa[material_i, material_j])
            mu = float(pp_mu[material_i, material_j])
            if mu < 0.0:
                mu = max(float(body_mu[body_i]), float(body_mu[body_j]))
            distance = float(np.linalg.norm(np.sum(weights[contact_id, :, None] * x[contact_vertices], axis=0)))
            collision_weight = 0.25 * float(node_area[contact_vertices[0]])
        else:
            kind = "ee"
            assert contact_bodies[0] == contact_bodies[1]
            assert contact_bodies[2] == contact_bodies[3]
            body_j = int(contact_bodies[2])
            material_j = int(body_material[body_j])
            dhat = float(pp_dhat[material_i, material_j])
            kappa = float(pp_kappa[material_i, material_j])
            mu = float(pp_mu[material_i, material_j])
            if mu < 0.0:
                mu = max(float(body_mu[body_i]), float(body_mu[body_j]))
            distance = float(np.linalg.norm(np.sum(weights[contact_id, :, None] * x[contact_vertices], axis=0)))
            edge_i = edge_ids[tuple(sorted((int(contact_vertices[0]), int(contact_vertices[1]))))]
            edge_j = edge_ids[tuple(sorted((int(contact_vertices[2]), int(contact_vertices[3]))))]
            collision_weight = 0.25 * float(edge_area[edge_i] + edge_area[edge_j])

        barrier_energy, normal_force = _official_barrier_terms(distance, dhat, kappa)
        expected_coeff = mu * normal_force * collision_weight * operator.scale
        assert np.isclose(
            coefficients[contact_id], expected_coeff, rtol=2.0e-10, atol=1.0e-14
        ), f"{kind} collision weight must not contain an extra dhat"
        energy_by_kind[kind] += operator.scale * collision_weight * barrier_energy
        seen_kinds.add(kind)

    assert expected_kinds <= seen_kinds

    operator._clear_system(False, int(MATRIX_COO))
    operator._assemble_particle_contacts(
        False,
        operator.neighbor.candidate_count,
        operator.neighbor.candidate_vertex,
        operator.neighbor.candidate_face,
        int(MATRIX_COO),
    )
    assert np.isclose(
        float(operator.energy[None]), energy_by_kind["vf"], rtol=2.0e-10, atol=1.0e-14
    ), "VF barrier energy must use 0.25 * vertex_area without an extra dhat"

    operator._clear_system(False, int(MATRIX_COO))
    operator._assemble_edge_contacts(
        False,
        operator.neighbor.edge_candidate_count,
        operator.neighbor.candidate_edge0,
        operator.neighbor.candidate_edge1,
    )
    assert np.isclose(
        float(operator.energy[None]), energy_by_kind["ee"], rtol=2.0e-10, atol=1.0e-14
    ), "EE barrier energy must use 0.25 * summed edge_area without an extra dhat"

    if "wall" in expected_kinds:
        operator._clear_system(False, int(MATRIX_COO))
        operator._assemble_wall_contacts(False, int(MATRIX_COO))
        assert np.isclose(
            float(operator.energy[None]),
            energy_by_kind["wall"],
            rtol=2.0e-10,
            atol=1.0e-14,
        ), "wall barrier energy must use vertex_area without an extra dhat"


def _evaluate_friction(dem, hat_y, displacement):
    engine = dem.enginer
    operator = engine.operator
    y = hat_y + displacement

    operator.friction_scale[0] = 1.0
    energy_with, grad_with = engine._assemble_system(dem.sims, y, need_matrix=True)
    matrix_with = engine.coo_matrix._to_numpy().copy()

    operator.friction_scale[0] = 0.0
    energy_without, grad_without = engine._assemble_system(dem.sims, y, need_matrix=True)
    matrix_without = engine.coo_matrix._to_numpy().copy()
    operator.friction_scale[0] = 1.0

    friction_energy = energy_with - energy_without
    friction_force = -(grad_with - grad_without)
    friction_matrix = matrix_with - matrix_without
    return friction_energy, friction_force, friction_matrix


def _mixed_error(actual, expected):
    return np.linalg.norm(actual - expected, ord=np.inf) / max(
        np.linalg.norm(actual, ord=np.inf),
        np.linalg.norm(expected, ord=np.inf),
        1.0,
    )


def test_ipc_affine_affine_friction_production_assembly(tmp_path):
    os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")
    ti.init(arch=ti.cpu, default_fp=ti.f64, debug=False, offline_cache=False)
    dem, hat_y = _build_affine_contact(tmp_path / "body-contact")
    _assert_official_collision_weights(dem, {"vf", "ee"})
    tangent_y = np.zeros_like(hat_y).reshape((2, 4, 3))
    tangent_y[0, :, 1] = 0.5
    tangent_y[1, :, 1] = -0.5
    tangent_y = tangent_y.reshape(-1)
    tangent_z = np.zeros_like(hat_y).reshape((2, 4, 3))
    tangent_z[0, :, 2] = 0.5
    tangent_z[1, :, 2] = -0.5
    tangent_z = tangent_z.reshape(-1)
    normal_x = np.zeros_like(hat_y).reshape((2, 4, 3))
    normal_x[0, :, 0] = 0.5
    normal_x[1, :, 0] = -0.5
    normal_x = normal_x.reshape(-1)

    for label, amplitude, fd_step in (
        ("dynamic", 2.0e-4, 2.0e-7),
        ("smoothed", 2.0e-7, 2.0e-9),
    ):
        displacement = amplitude * tangent_y
        energy, force, matrix = _evaluate_friction(dem, hat_y, displacement)
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))

        assert energy > 0.0, label
        for tangent in (tangent_y, tangent_z, normal_x):
            energy_derivative = (
                _evaluate_friction(dem, hat_y, displacement + fd_step * tangent)[0]
                - _evaluate_friction(dem, hat_y, displacement - fd_step * tangent)[0]
            ) / (2.0 * fd_step)
            force_derivative = (
                _evaluate_friction(dem, hat_y, displacement + fd_step * tangent)[1]
                - _evaluate_friction(dem, hat_y, displacement - fd_step * tangent)[1]
            ) / (2.0 * fd_step)
            assert np.isclose(
                energy_derivative,
                -float(np.dot(force, tangent)),
                rtol=5.0e-5,
                atol=2.0e-8,
            ), label
            assert _mixed_error(force_derivative, -(matrix @ tangent)) < 2.0e-4, label
        assert np.allclose(matrix, matrix.T, rtol=1.0e-9, atol=1.0e-10), label
        assert eigenvalues.min() >= -1.0e-10 * max(eigenvalues.max(), 1.0), label
        assert np.allclose(
            force.reshape((-1, 3)).sum(axis=0),
            0.0,
            rtol=1.0e-8,
            atol=1.0e-8,
        ), label
        assert float(np.dot(force, displacement)) < 0.0, label


def test_ipc_affine_wall_collision_weight_has_no_extra_dhat(tmp_path):
    os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")
    ti.init(arch=ti.cpu, default_fp=ti.f64, debug=False, offline_cache=False)
    dem, unused_hat_y = _build_affine_contact(tmp_path / "wall-contact", with_wall=True)
    _assert_official_collision_weights(dem, {"vf", "ee", "wall"})

    # A deformable contact patch generally develops Cattaneo-style partial
    # slip while its normal and shear tractions settle.  Consequently,
    # ``mu > tan(theta)`` does not imply that every surface vertex of an
    # initially stress-free block has zero accumulated displacement.  Test the
    # actual static-friction branch in the setting for which the Coulomb
    # incline solution is exact: a uniform rigid translation of the frozen
    # wall-contact stencil.
    operator = dem.enginer.operator
    count = min(
        int(operator.friction_contact_count[0]),
        operator.friction_contact_capacity,
    )
    bodies = operator.friction_contact_bodies.to_numpy()[:count]
    vertices = operator.friction_contact_vertices.to_numpy()[:count]
    coefficients = operator.friction_contact_coeff.to_numpy()[:count]
    wall_contacts = np.logical_and(
        np.all(bodies == bodies[:, :1], axis=1),
        np.all(vertices == vertices[:, :1], axis=1),
    )
    coulomb_cap = float(np.sum(coefficients[wall_contacts]))
    assert coulomb_cap > 0.0

    translation = np.zeros_like(unused_hat_y).reshape((-1, 4, 3))
    translation[:, :, 0] = 1.0
    translation = translation.reshape(-1)
    # tan(theta)=0.2 and mu=0.4 give a strict sub-Coulomb load ratio 1/2.
    # The IPC C2 smoothing has
    #
    #   f1(v) = v (2 epsv - v) / epsv^2,  0 <= v < epsv,
    #
    # so balancing gravity against friction gives the closed-form sticking
    # speed below.  It is the regularized no-slip solution and tends to zero
    # linearly with epsv.
    load_ratio = 0.2 / 0.4
    expected_speed = operator.epsv * (1.0 - np.sqrt(1.0 - load_ratio))
    displacement = operator.dt * expected_speed * translation
    _, friction_force, friction_matrix = _evaluate_friction(dem, unused_hat_y, displacement)
    resistance = -float(np.dot(friction_force, translation))
    tangent_stiffness = float(translation @ friction_matrix @ translation)
    expected_stiffness = (
        coulomb_cap * 2.0 * (operator.epsv - expected_speed) / (operator.epsv * operator.epsv * operator.dt)
    )

    assert 0.0 < expected_speed < operator.epsv
    assert resistance == pytest.approx(load_ratio * coulomb_cap, rel=2.0e-10, abs=1.0e-13)
    assert tangent_stiffness == pytest.approx(expected_stiffness, rel=2.0e-9, abs=1.0e-10)

    dynamic_displacement = 2.0 * operator.dt * operator.epsv * translation
    _, dynamic_force, _ = _evaluate_friction(dem, unused_hat_y, dynamic_displacement)
    dynamic_resistance = -float(np.dot(dynamic_force, translation))
    assert dynamic_resistance == pytest.approx(coulomb_cap, rel=2.0e-10, abs=1.0e-13)


def test_soft_affine_step_initializes_friction_when_damping_is_disabled():
    os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")
    from src.mpdem.engines.SoftAffineIPCOperator import SoftAffineIPCOperator

    events = []
    state = SimpleNamespace(
        hat_y=np.zeros(3, dtype=np.float64),
        begin_step=lambda dt: events.append(("begin", dt)),
        pack=lambda: np.ones(3, dtype=np.float64),
    )
    affine = SimpleNamespace(
        contact_damping_stiffness=0.0,
        initialize_contact_damping=lambda y, hat_y: events.append(("initialize", y.copy(), hat_y.copy())),
    )
    operator = SimpleNamespace(
        dt=0.125,
        # ``begin_step`` is deliberately the NumPy-backed CPU oracle path.
        # Mirror the production guard dependency in this minimal unbound-method
        # test instead of bypassing the CUDA residency contract in production.
        _reject_cuda_host_vector_path=lambda operation: None,
        affine_state=state,
        affine=affine,
        _prepare_soft_grid=lambda: events.append(("grid",)),
        prefix_sum=SimpleNamespace(run=lambda field: events.append(("prefix", field))),
        soft_node2dof="node2dof",
        _fill_soft_dof=lambda: 7,
        affine_dof=12,
        _initialize_soft_step=lambda: events.append(("soft",)),
    )

    SoftAffineIPCOperator.begin_step(operator)

    assert events[1][0] == "initialize"
    assert operator.soft_active_nodes == 7
    assert operator.total_dof == 33


if __name__ == "__main__":
    test_ipc_affine_affine_friction_production_assembly()
    test_ipc_affine_wall_collision_weight_has_no_extra_dhat()
