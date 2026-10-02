"""Analytic oracles for the axisymmetric IGA element-local chain rule."""

from types import MethodType

import numpy as np
import pytest


pytestmark = [
    pytest.mark.unit,
    pytest.mark.cpu,
    pytest.mark.serial,
    pytest.mark.isolated_dimension(2),
]


def test_igampm_step_preparation_and_abort_share_one_transaction():
    from src.igampm.engines.ImplicitEngine import ImplicitEngineMixin

    class TransactionProbe:
        def __init__(self):
            self.implicit_step_in_progress = False
            self.events = []

        def snapshot(self):
            self.events.append("snapshot")

        def save_displacements(self):
            self.events.append("save_displacements")

        def prepare(self):
            self.events.append("prepare")
            raise RuntimeError("synthetic preparation failure")

        def restore(self):
            self.events.append("restore")

        def restore_displacements(self):
            self.events.append("restore_displacements")

    probe = TransactionProbe()
    probe._snapshot_implicit_physical_state_device = probe.snapshot
    probe._save_device_monolithic_entry_displacements = probe.save_displacements
    probe._prepare_implicit_ipc_step = probe.prepare
    probe._restore_implicit_physical_state_device = probe.restore
    probe._restore_device_monolithic_entry_displacements = (
        probe.restore_displacements
    )
    probe.begin_implicit_ipc_step = MethodType(
        ImplicitEngineMixin.begin_implicit_ipc_step, probe
    )
    probe.abort_implicit_ipc_step = MethodType(
        ImplicitEngineMixin.abort_implicit_ipc_step, probe
    )

    with pytest.raises(RuntimeError, match="synthetic preparation failure"):
        probe.begin_implicit_ipc_step()
    assert probe.events == [
        "snapshot",
        "save_displacements",
        "prepare",
        "restore",
        "restore_displacements",
    ]
    assert not probe.implicit_step_in_progress

    probe.events.clear()
    probe.implicit_step_in_progress = True
    probe.abort_implicit_ipc_step()
    assert probe.events == ["restore", "restore_displacements"]
    assert not probe.implicit_step_in_progress


def test_axisymmetric_local_pullbacks_match_no_swirl_oracle(
    taichi_runtime,
):
    import taichi as ti

    from src.iga.elements.Element import Element

    support_count = 4
    element = object.__new__(Element)
    element.total_knot_range = support_count

    shape_field = ti.Vector.field(support_count, ti.f64, shape=())
    shape_gradient_field = ti.Matrix.field(
        support_count, 2, ti.f64, shape=()
    )
    reference_field = ti.Matrix.field(
        support_count, 2, ti.f64, shape=()
    )
    current_field = ti.Matrix.field(support_count, 2, ti.f64, shape=())
    stress_field = ti.Vector.field(9, ti.f64, shape=())
    tangent_field = ti.Matrix.field(9, 9, ti.f64, shape=())
    deformation_field = ti.Matrix.field(3, 3, ti.f64, shape=())
    gradient_field = ti.field(ti.f64, shape=2 * support_count)
    hessian_field = ti.field(
        ti.f64, shape=(2 * support_count, 2 * support_count)
    )

    axis_offset = 0.25

    @ti.kernel
    def evaluate():
        shape = shape_field[None]
        shape_gradients = shape_gradient_field[None]
        reference = reference_field[None]
        current = current_field[None]
        reference_radius = (
            element.interpolate_component(shape, reference, 0)
            - ti.static(axis_offset)
        )
        deformation_field[None] = (
            element.compute_axisymmetric_deformation_gradient(
                shape,
                shape_gradients,
                reference,
                current,
                ti.static(axis_offset),
            )
        )
        for support_i in range(support_count):
            local_gradient = element.compute_axisymmetric_local_gradient(
                support_i,
                stress_field[None],
                shape,
                shape_gradients,
                reference_radius,
            )
            for component in ti.static(range(2)):
                gradient_field[2 * support_i + component] = local_gradient[
                    component
                ]
            for support_j in range(support_count):
                block = element.compute_axisymmetric_local_hessian(
                    support_i,
                    support_j,
                    tangent_field[None],
                    shape,
                    shape_gradients,
                    reference_radius,
                )
                for component_i, component_j in ti.static(ti.ndrange(2, 2)):
                    hessian_field[
                        2 * support_i + component_i,
                        2 * support_j + component_j,
                    ] = block[component_i, component_j]

    shape = np.array([0.12, 0.28, 0.24, 0.36], dtype=np.float64)
    shape_gradients = np.array(
        [
            [-0.45, -0.35],
            [0.50, -0.20],
            [-0.25, 0.40],
            [0.20, 0.15],
        ],
        dtype=np.float64,
    )
    reference = np.array(
        [[0.8, 0.0], [1.5, 0.1], [0.9, 0.9], [1.6, 1.1]],
        dtype=np.float64,
    )
    current = np.array(
        [[0.86, -0.03], [1.59, 0.16], [0.98, 0.96], [1.73, 1.20]],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260813)
    stress = rng.normal(size=9)
    tangent = rng.normal(size=(9, 9))

    shape_field[None] = shape
    shape_gradient_field[None] = shape_gradients
    reference_field[None] = reference
    current_field[None] = current
    stress_field[None] = stress
    tangent_field[None] = tangent
    evaluate()

    reference_radius = shape @ reference[:, 0] - axis_offset
    current_radius = shape @ current[:, 0] - axis_offset
    expected_deformation = np.zeros((3, 3), dtype=np.float64)
    expected_deformation[:2, :2] = current.T @ shape_gradients
    expected_deformation[2, 2] = current_radius / reference_radius

    derivative = np.zeros((2 * support_count, 9), dtype=np.float64)
    for support in range(support_count):
        for component in range(2):
            row = 2 * support + component
            for material_axis in range(2):
                derivative[
                    row, component + 3 * material_axis
                ] = shape_gradients[support, material_axis]
            if component == 0:
                derivative[row, 8] = shape[support] / reference_radius

    np.testing.assert_allclose(
        deformation_field.to_numpy(),
        expected_deformation,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        gradient_field.to_numpy(),
        derivative @ stress,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        hessian_field.to_numpy(),
        derivative @ tangent @ derivative.T,
        rtol=5.0e-14,
        atol=5.0e-14,
    )


def test_reference_mapping_rejects_nonpositive_jacobian(taichi_runtime):
    import taichi as ti

    from src.iga.engines.IGASolver import IGASolver

    solver = object.__new__(IGASolver)
    solver.reference_mapping_status = ti.field(ti.i32, shape=())

    @ti.kernel
    def invalid_reference_determinant() -> ti.f64:
        jacobian = ti.Matrix([[-1.0, 0.0], [0.0, 1.0]])
        return solver.require_positive_reference_jacobian(jacobian)

    assert invalid_reference_determinant() == pytest.approx(-1.0)
    assert int(solver.reference_mapping_status[None]) == 1


def test_precompute_reports_inverted_reference_patch(taichi_runtime):
    from src.iga import ImplicitIGA, Primitives, Rectangle

    rectangle = Rectangle()
    rectangle.set_parameters(start_point=[0.0, 0.0], size=[1.0, 0.5])
    rectangle.generate_knot_u(degree=2, num_ctrlpts=3)
    rectangle.generate_knot_v(degree=2, num_ctrlpts=3)
    rectangle.generate_ctrlpts()
    rectangle.generate_weights()
    primitives = Primitives()
    primitives.append(rectangle, "inverted-reference")
    primitives.finialize()
    engine = ImplicitIGA(
        primitives=primitives,
        degree=[2, 2],
        gravity=[0.0, 0.0],
        young_modulus=1.0e4,
        poisson_ratio=0.3,
        density=1.0,
    )
    inverted = engine.patch.rest_control_points.to_numpy()
    inverted[:, 0] *= -1.0
    engine.patch.rest_control_points.from_numpy(inverted)

    with pytest.raises(ValueError, match="finite positive Jacobians"):
        engine.precompute()


def test_precompute_reports_nonpositive_axisymmetric_radius(taichi_runtime):
    from src.iga import ImplicitIGA, Primitives, Rectangle

    rectangle = Rectangle()
    rectangle.set_parameters(start_point=[-1.0, 0.0], size=[0.2, 0.5])
    rectangle.generate_knot_u(degree=2, num_ctrlpts=3)
    rectangle.generate_knot_v(degree=2, num_ctrlpts=3)
    rectangle.generate_ctrlpts()
    rectangle.generate_weights()
    primitives = Primitives()
    primitives.append(rectangle, "negative-radius-reference")
    primitives.finialize()
    engine = ImplicitIGA(
        primitives=primitives,
        degree=[2, 2],
        gravity=[0.0, 0.0],
        young_modulus=1.0e4,
        poisson_ratio=0.3,
        density=1.0,
        axisymmetric=True,
        axis_offset=0.0,
    )

    with pytest.raises(ValueError, match="axisymmetric radii"):
        engine.precompute()


def test_igampm_reference_scan_checks_full_embedded_material_map(
    taichi_runtime,
):
    from types import SimpleNamespace

    import taichi as ti

    from src.igampm.engines.ImplicitEngine import ImplicitEngineMixin

    @ti.data_oriented
    class ReferenceProbe(ImplicitEngineMixin):
        def __init__(self):
            volume = ti.field(ti.f64, shape=1)
            particle_num = ti.field(ti.i32, shape=1)
            deformation = ti.Matrix.field(3, 3, ti.f64, shape=2)
            grid_type = ti.types.struct(m=ti.f64)
            grid = grid_type.field(shape=1)
            self.iga = SimpleNamespace(
                patch=SimpleNamespace(volume=volume)
            )
            self.mpm = SimpleNamespace(
                particleNum=particle_num,
                F0=deformation,
                material_dimension=3,
                grid=grid,
                val_lim=1.0e-12,
            )
            self.implicit_state_status = ti.field(ti.i32, shape=7)
            particle_num[0] = 2

    probe = ReferenceProbe()
    deformation = np.array(
        [np.eye(3), np.diag([1.0, 1.0, 0.0])], dtype=np.float64
    )
    probe.mpm.F0.from_numpy(deformation)
    probe._inspect_implicit_reference_state_device()
    status = probe.implicit_state_status.to_numpy()
    assert status[2] == 0
    assert status[3] == 1
    assert status[6] == 1

    deformation[1, 2, 2] = np.nan
    probe.mpm.F0.from_numpy(deformation)
    probe._inspect_implicit_reference_state_device()
    status = probe.implicit_state_status.to_numpy()
    assert status[2] == 1
    assert status[6] == 1
