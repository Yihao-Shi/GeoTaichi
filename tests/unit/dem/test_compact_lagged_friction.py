import numpy as np
import pytest
import taichi as ti

from src.dem.engines.AffineBodyOperator import TaichiAffineBodyOperator, MATRIX_COO, MATRIX_HASH_TRIPLET
from src.linear_solver.BuildTriplet import BuildTriplet


@ti.data_oriented
class FrictionProbe(TaichiAffineBodyOperator):
    def __init__(self, speed):
        self.body_num, self.control_num = 2, 8
        self.fully_implicit, self.levelset_contact = False, False
        self.friction_contact_capacity = 96
        self.epsv = 0.01
        self.dt_device = ti.field(ti.f64, shape=())
        self.dt_device[None] = 0.001
        self.energy = ti.field(ti.f64, shape=())
        self.grad = ti.Vector.field(3, ti.f64, shape=8)
        self.hessian = ti.field(ti.f64, shape=(24, 24))
        self.x = ti.Vector.field(3, ti.f64, shape=4)
        self.basis = ti.field(ti.f64, shape=(4, 4))
        self.basis.from_numpy(
            np.array([[1.2, -0.4, 0.3, -0.1], [0.5, 0.1, 0.2, 0.2], [0.1, 0.3, 0.2, 0.4], [0.25, 0.25, 0.25, 0.25]])
        )
        self.friction_contact_count = ti.field(ti.i32, shape=1)
        self.friction_contact_count[0] = 96
        self.friction_contact_bodies = ti.Vector.field(4, ti.i32, shape=96)
        self.friction_contact_vertices = ti.Vector.field(4, ti.i32, shape=96)
        self.friction_contact_weights = ti.Vector.field(4, ti.f64, shape=96)
        self.friction_contact_normal = ti.Vector.field(3, ti.f64, shape=96)
        self.friction_contact_hat_rel = ti.Vector.field(3, ti.f64, shape=96)
        self.friction_contact_coeff = ti.field(ti.f64, shape=96)
        self.friction_scale = ti.field(ti.f64, shape=1)
        self.body_pair_segment_base = ti.field(ti.i32, shape=4)
        self.active_body_pair = ti.field(ti.i32, shape=4)
        self.active_body_pair_list = ti.field(ti.i32, shape=4)
        self.active_body_pair_count = ti.field(ti.i32, shape=())
        rest = np.array([[0.1, 0.2, 0.003], [0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.3, 0.0]])
        current = rest.copy()
        current[0] += speed * 0.001 * np.array([1.0, 0.25, 0.0])
        current[1:] -= speed * 0.0001 * np.array([1.0, 0.25, 0.0])
        self.x.from_numpy(current)
        stencils = [
            ([1, 0, 0, 0], [0, 1, 2, 3], [1.0, -0.2, -0.3, -0.5]),
            ([1, 0, 1, 0], [0, 1, 2, 3], [0.3, 0.7, -0.2, -0.8]),
            ([0, 0, 0, 0], [0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0]),
            ([1, 1, 1, 1], [1, 1, 1, 1], [1.0, 0.0, 0.0, 0.0]),
        ]
        for contact in range(96):
            bodies, vertices, weights = stencils[contact % len(stencils)]
            self.friction_contact_bodies[contact] = bodies
            self.friction_contact_vertices[contact] = vertices
            self.friction_contact_weights[contact] = weights
            self.friction_contact_normal[contact] = [0.2, -0.3, 1.0]
            self.friction_contact_hat_rel[contact] = np.asarray(weights) @ rest[vertices]
            self.friction_contact_coeff[contact] = 0.002 * (1 + 0.1 * (contact % 7))

    @ti.func
    def _add_matrix_block(self, row, column, block, target: ti.template()):
        for i, j in ti.static(ti.ndrange(3, 3)):
            ti.atomic_add(self.hessian[3 * row + i, 3 * column + j], block[i, j])

    @ti.kernel
    def seed_hash(self):
        for control in range(8):
            self.hash_triplet.add_block_entry(control, control, 0.5 * ti.Matrix.identity(ti.f64, 3))
        self.hash_triplet.add_block_entry(0, 5, 0.25 * ti.Matrix.identity(ti.f64, 3))
        if ti.static(self.hash_store_full_symmetric_input):
            self.hash_triplet.add_block_entry(5, 0, 0.25 * ti.Matrix.identity(ti.f64, 3))


@pytest.mark.parametrize("speed", [0.002, 0.02])
@pytest.mark.parametrize("majorize", [False, True])
@pytest.mark.parametrize("full_input", [False, True])
def test_many_friction_stencils_fit_pair_capacity_and_match_dense(taichi_runtime, speed, majorize, full_input):
    op = FrictionProbe(speed)
    matrix = BuildTriplet(
        dim=3,
        max_pairs_num=82,
        max_nonzeros=82,
        max_active_nodes=8,
        symmetric=False,
        matrix_symmetric=True,
        full_symmetric_input=full_input,
        device_reduction=True,
    )
    op.bind_hash_triplet(matrix)
    seed = 0.5 * np.eye(24)
    seed[:3, 15:18] = 0.25 * np.eye(3)
    seed[15:18, :3] = 0.25 * np.eye(3)
    for scale in (1.25, 0.0, 1.25):
        op.friction_scale[0] = scale
        op.grad.fill(0.0)
        op.energy[None] = 0.0
        op.hessian.from_numpy(seed)
        op._assemble_lagged_friction(True, MATRIX_COO, majorize)
        expected = op.hessian.to_numpy()
        gradient, energy = op.grad.to_numpy(), float(op.energy[None])
        matrix.reset_system()
        op.seed_hash()
        op.grad.fill(0.0)
        op.energy[None] = 0.0
        op._assemble_lagged_friction(True, MATRIX_HASH_TRIPLET, majorize)
        matrix.finalize_taichi_assembly()
        actual = matrix.to_scipy(8).toarray()
        # The budget covers f64 accumulation of 96 independent contact stencils.
        factor = 256 * 96 * np.finfo(np.float64).eps
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=factor * max(1.0, np.abs(expected).max()))
        np.testing.assert_allclose(
            op.grad.to_numpy(), gradient, rtol=0.0, atol=factor * max(1.0, np.abs(gradient).max())
        )
        assert float(op.energy[None]) == pytest.approx(energy, rel=factor, abs=factor)
        assert int(matrix.raw_non_diag_count[0]) <= 82


def test_pair_storage_overflow_remains_a_failure(taichi_runtime):
    op = FrictionProbe(0.02)
    matrix = BuildTriplet(
        dim=3,
        max_pairs_num=1,
        max_nonzeros=82,
        max_active_nodes=8,
        symmetric=False,
        matrix_symmetric=True,
        device_reduction=True,
    )
    op.bind_hash_triplet(matrix)
    matrix.reset_system()
    op.friction_scale[0] = 1.0
    op._assemble_lagged_friction(True, MATRIX_HASH_TRIPLET, False)
    with pytest.raises(RuntimeError, match="overflow"):
        matrix.finalize_taichi_assembly()
