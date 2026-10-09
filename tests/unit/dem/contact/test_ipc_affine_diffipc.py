import numpy as np
import pytest

from src.physics_model.contact_model.ipc.AffineDifferentiable import (
    AffineTriangleBody,
    AffineTriangleIPC,
    DiffIPCAffineEquilibrium,
    affine_edge_edge_gap,
    affine_point_triangle_gap,
)
from src.physics_model.contact_model.ipc.LevelSetAffine import affine_basis

pytestmark = [
    pytest.mark.unit,
    pytest.mark.dem,
    pytest.mark.ipc,
    pytest.mark.contact,
    pytest.mark.cpu,
]


def _identity_controls(center):
    center = np.asarray(center, dtype=np.float64)
    return np.array(
        [
            center,
            center + [1.0, 0.0, 0.0],
            center + [0.0, 1.0, 0.0],
            center + [0.0, 0.0, 1.0],
        ]
    )


def _triangle_basis():
    vertices = np.array(
        [
            [-0.20, -0.18, 0.0],
            [0.22, -0.17, 0.0],
            [0.01, 0.24, 0.0],
        ]
    )
    return np.asarray([affine_basis(vertex) for vertex in vertices])


@pytest.mark.parametrize("mode", ["ccd", "accd"])
def test_mesh_diffipc_swept_candidates_match_full_pair_ccd(taichi_runtime, tmp_path, mode):
    from tests.unit.dem.contact.test_ipc_affine_friction_assembly import _build_affine_contact
    from src.dem.engines.AffineDiffIPC import TaichiAffineMeshDiffIPCProjector

    dem, controls = _build_affine_contact(tmp_path)
    operator = dem.enginer.operator
    projector = TaichiAffineMeshDiffIPCProjector(operator, controls, np.tile([-2.0, 2.0], (2, 3, 1)))
    projector._ccd_mode = mode
    projector.direction.from_numpy(np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]]))
    neighbor = operator.neighbor
    nodes, faces = operator.node2body.to_numpy(), operator.face2body.to_numpy()
    edges = operator.edge2body.to_numpy()
    point_pairs = np.array(
        [(v, f) for v in range(operator.vertex_num) for f in range(operator.face_num) if nodes[v] != faces[f]],
        dtype=np.int32,
    )
    edge_pairs = np.array(
        [(i, j) for i in range(operator.edge_num) for j in range(i + 1, operator.edge_num) if edges[i] != edges[j]],
        dtype=np.int32,
    )
    neighbor._allocate_candidates(len(point_pairs))
    neighbor._allocate_edge_candidates(len(edge_pairs))
    actual = projector._maximum_feasible_step()
    compact_count = int(neighbor.candidate_count[None]) + int(neighbor.edge_candidate_count[None])
    assert compact_count < len(point_pairs) + len(edge_pairs)
    neighbor.candidate_vertex.from_numpy(point_pairs[:, 0].copy())
    neighbor.candidate_face.from_numpy(point_pairs[:, 1].copy())
    neighbor.candidate_edge0.from_numpy(edge_pairs[:, 0].copy())
    neighbor.candidate_edge1.from_numpy(edge_pairs[:, 1].copy())
    neighbor.candidate_count[None] = len(point_pairs)
    neighbor.edge_candidate_count[None] = len(edge_pairs)
    projector._compute_mesh_ccd_step(
        projector._ccd_eta,
        projector._ccd_thickness,
        projector._ccd_maximum_iterations,
        mode == "accd",
        neighbor.candidate_count,
        neighbor.candidate_vertex,
        neighbor.candidate_face,
        neighbor.edge_candidate_count,
        neighbor.candidate_edge0,
        neighbor.candidate_edge1,
    )
    expected = float(operator.ccd_alpha[None])
    assert 0 < actual < 1
    assert actual == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_affine_triangle_gap_full_24_dof_derivatives_match_finite_difference():
    source = _identity_controls([0.013, -0.007, 0.061])
    target = _identity_controls([0.0, 0.0, 0.0])
    source_basis = affine_basis([0.0, 0.0, 0.0])
    triangle_basis = _triangle_basis()
    packed = np.concatenate((source.reshape(-1), target.reshape(-1)))
    reference = affine_point_triangle_gap(source, target, source_basis, triangle_basis)
    assert reference.feature == "face"
    epsilon = 1.0e-6
    eye = np.eye(24)

    def evaluate(values):
        return affine_point_triangle_gap(
            values[:12].reshape(4, 3),
            values[12:].reshape(4, 3),
            source_basis,
            triangle_basis,
        )

    finite_gradient = np.array(
        [
            (evaluate(packed + epsilon * direction).distance - evaluate(packed - epsilon * direction).distance)
            / (2.0 * epsilon)
            for direction in eye
        ]
    )
    finite_hessian = np.column_stack(
        [
            (evaluate(packed + epsilon * direction).gradient - evaluate(packed - epsilon * direction).gradient)
            / (2.0 * epsilon)
            for direction in eye
        ]
    )
    np.testing.assert_allclose(reference.gradient, finite_gradient, rtol=2.0e-7, atol=2.0e-9)
    np.testing.assert_allclose(reference.hessian, finite_hessian, rtol=3.0e-6, atol=5.0e-8)


def test_affine_edge_edge_gap_full_24_dof_derivatives_match_finite_difference():
    controls0 = _identity_controls([0.0, 0.0, 0.0])
    controls1 = _identity_controls([0.0, 0.0, 0.0])
    edge0_basis = np.asarray([affine_basis([-0.2, 0.0, 0.0]), affine_basis([0.2, 0.0, 0.0])])
    edge1_basis = np.asarray([affine_basis([0.0, -0.2, 0.061]), affine_basis([0.0, 0.2, 0.061])])
    packed = np.concatenate((controls0.reshape(-1), controls1.reshape(-1)))
    reference = affine_edge_edge_gap(controls0, controls1, edge0_basis, edge1_basis)
    assert reference.feature == "edge_edge"
    epsilon = 1.0e-6
    eye = np.eye(24)

    def evaluate(values):
        return affine_edge_edge_gap(
            values[:12].reshape(4, 3),
            values[12:].reshape(4, 3),
            edge0_basis,
            edge1_basis,
        )

    finite_gradient = np.array(
        [
            (evaluate(packed + epsilon * direction).distance - evaluate(packed - epsilon * direction).distance)
            / (2.0 * epsilon)
            for direction in eye
        ]
    )
    finite_hessian = np.column_stack(
        [
            (evaluate(packed + epsilon * direction).gradient - evaluate(packed - epsilon * direction).gradient)
            / (2.0 * epsilon)
            for direction in eye
        ]
    )
    np.testing.assert_allclose(reference.gradient, finite_gradient, rtol=2.0e-7, atol=2.0e-9)
    np.testing.assert_allclose(reference.hessian, finite_hessian, rtol=4.0e-6, atol=7.0e-8)


def _triangle_contact(hessian_mode="exact"):
    basis = _triangle_basis()
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    weights = np.ones(3, dtype=np.float64) / 3.0
    bodies = [
        AffineTriangleBody(_identity_controls([0.0, 0.0, 0.0]), basis, weights, faces),
        AffineTriangleBody(_identity_controls([0.0, 0.0, 0.067]), basis, weights, faces),
    ]
    return AffineTriangleIPC(bodies, dhat=0.1, kappa=0.02, hessian_mode=hessian_mode)


def test_diffipc_affine_triangle_equilibrium_adjoint_matches_resolved_fd():
    contact = _triangle_contact(hessian_mode="projected")
    equilibrium = DiffIPCAffineEquilibrium(contact, anchor_stiffness=100.0)
    result = equilibrium.solve(maximum_iterations=300)
    assert result.success
    assert result.residual_norm < 2.0e-7
    assert contact.hessian_mode == "projected"

    rng = np.random.default_rng(1729)
    loss_gradient = rng.normal(size=contact.dof)
    loss_gradient /= np.linalg.norm(loss_gradient)
    adjoint = equilibrium.adjoint_anchor_gradient(loss_gradient)
    base_anchor = equilibrium.initial_controls.copy()
    epsilon = 2.0e-6
    for parameter in (2, 14):
        losses = []
        for sign in (-1.0, 1.0):
            perturbed_contact = _triangle_contact()
            perturbed = DiffIPCAffineEquilibrium(perturbed_contact, anchor_stiffness=100.0)
            anchor = base_anchor.copy()
            anchor[parameter] += sign * epsilon
            perturbed_result = perturbed.solve(anchor=anchor, maximum_iterations=300)
            assert perturbed_result.success
            losses.append(float(loss_gradient @ perturbed_result.controls.reshape(-1)))
        finite_difference = (losses[1] - losses[0]) / (2.0 * epsilon)
        assert adjoint[parameter] == pytest.approx(finite_difference, rel=8.0e-5, abs=2.0e-6)
