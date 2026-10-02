from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def taichi_runtime():
    """Keep this deployment test independent of the repository conftest."""
    import taichi as ti

    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
        log_level=ti.ERROR,
    )
    try:
        yield ti
    finally:
        ti.reset()


def _reference_mass_fields(ti):
    vec3 = ti.types.vector(3, ti.f64)
    soft_type = ti.types.struct(
        startIndex=ti.i32,
        templatePointStart=ti.i32,
        mpmGridStart=ti.i32,
    )
    point_type = ti.types.struct(
        active=ti.i32,
        bodyID=ti.i32,
        m=ti.f64,
        v=vec3,
    )
    rigid_type = ti.types.struct(softID=ti.i32)
    grid_type = ti.types.struct(
        m=ti.f64,
        f=vec3,
        v=vec3,
        contact_force=vec3,
    )

    soft = soft_type.field(shape=1)
    point = point_type.field(shape=2)
    rigid = rigid_type.field(shape=1)
    grid = grid_type.field(shape=3)
    shape_node = ti.field(ti.i32, shape=(2, 2))
    shape = ti.field(ti.f64, shape=(2, 2))
    shape_count = ti.field(ti.i32, shape=2)

    soft.startIndex[0] = 0
    soft.templatePointStart[0] = 0
    soft.mpmGridStart[0] = 0
    rigid.softID[0] = 0
    for p, mass in enumerate((2.0, 3.0)):
        point.active[p] = 1
        point.bodyID[p] = 0
        point.m[p] = mass
        shape_count[p] = 2
    point.v[0] = [1.0, 0.0, 0.0]
    point.v[1] = [0.0, 2.0, 0.0]
    shape_node[0, 0], shape_node[0, 1] = 0, 1
    shape[0, 0], shape[0, 1] = 0.25, 0.75
    shape_node[1, 0], shape_node[1, 1] = 1, 2
    shape[1, 0], shape[1, 1] = 0.50, 0.50
    grid.m.fill(99.0)
    return soft, point, grid, shape_node, shape, shape_count, rigid


def test_reference_grid_mass_is_precomputed_and_idempotent(taichi_runtime):
    from src.mpm.soft_particle.SoftBodyKernel import (
        precompute_soft_grid_reference_mass_,
    )

    fields = _reference_mass_fields(taichi_runtime)
    soft, point, grid, shape_node, shape, shape_count, rigid = fields
    arguments = (
        1, 3, 2, soft, point, grid, shape_node, shape, shape_count, rigid
    )
    precompute_soft_grid_reference_mass_(*arguments)
    np.testing.assert_allclose(grid.m.to_numpy(), [0.5, 3.0, 1.5])
    np.testing.assert_allclose(
        grid.v.to_numpy(),
        [[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 2.0, 0.0]],
    )

    # A refresh must replace, rather than accumulate onto, cached mass.
    precompute_soft_grid_reference_mass_(*arguments)
    np.testing.assert_allclose(grid.m.to_numpy(), [0.5, 3.0, 1.5])
    np.testing.assert_allclose(
        grid.v.to_numpy(),
        [[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 2.0, 0.0]],
    )


def test_tlmpm_step_reset_preserves_cross_step_state(taichi_runtime):
    ti = taichi_runtime
    from src.mpm.soft_particle.Structs import DeformableGrid

    grid = DeformableGrid.field(shape=1)
    grid.m[0] = 4.25
    grid.v[0] = [1.0, 2.0, 3.0]
    grid.f[0] = [4.0, 5.0, 6.0]
    grid.contact_force[0] = [7.0, 8.0, 9.0]

    @ti.kernel
    def reset_dynamic_state():
        grid[0]._tlmpm_step_reset()

    reset_dynamic_state()
    assert float(grid.m[0]) == 4.25
    np.testing.assert_allclose(grid.v[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(grid.f[0], [7.0, 8.0, 9.0])
    np.testing.assert_allclose(grid.contact_force[0], [0.0, 0.0, 0.0])


def test_initial_constitutive_cache_handles_nonzero_prestrain(taichi_runtime):
    ti = taichi_runtime
    from src.mpm.MaterialManager import SoftParticleNeoHookeanMaterialTable
    from src.mpm.soft_particle.SoftBodyKernel import (
        initialize_soft_particle_constitutive_state_,
    )

    class Parameters:
        shear = 2.0
        lame_lambda = 3.0

    mat3 = ti.types.matrix(3, 3, ti.f64)
    point_type = ti.types.struct(
        active=ti.i32,
        materialID=ti.i32,
        F=mat3,
        stress=mat3,
        strain_energy=ti.f64,
        vol0=ti.f64,
    )
    state_type = ti.types.struct(estress=ti.f64)
    point = point_type.field(shape=1)
    state = state_type.field(shape=1)
    material = SoftParticleNeoHookeanMaterialTable(1, {0: Parameters()})

    deformation_gradient = np.diag([1.1, 0.9, 1.0])
    point.active[0] = 1
    point.materialID[0] = 0
    point.F[0] = deformation_gradient
    point.vol0[0] = 0.25
    initialize_soft_particle_constitutive_state_(1, point, material, state)

    inverse_transpose = np.linalg.inv(deformation_gradient).T
    expected = (
        2.0 * (deformation_gradient - inverse_transpose)
        + 3.0 * np.log(np.linalg.det(deformation_gradient))
        * inverse_transpose
    )
    np.testing.assert_allclose(point.stress.to_numpy()[0], expected)
    assert float(state.estress[0]) > 0.0


def test_soft_body_translation_constraint_uses_parallel_reduction(taichi_runtime):
    ti = taichi_runtime
    from src.mpm.soft_particle.SoftBodyKernel import (
        apply_soft_body_translation_constraint_,
        reduce_soft_body_kinematics_,
        reset_soft_body_kinematic_reduction_,
    )

    vec3 = ti.types.vector(3, ti.f64)
    vec3u8 = ti.types.vector(3, ti.u8)
    soft_type = ti.types.struct(
        m=ti.f64,
        mass_center0=vec3,
        previous_center=vec3,
        v=vec3,
    )
    point_type = ti.types.struct(
        active=ti.i32,
        bodyID=ti.i32,
        m=ti.f64,
        x=vec3,
        v=vec3,
    )
    rigid_type = ti.types.struct(softID=ti.i32, is_fix=vec3u8)
    soft = soft_type.field(shape=1)
    point = point_type.field(shape=2)
    rigid = rigid_type.field(shape=1)

    soft.m[0] = 4.0
    soft.mass_center0[0] = [0.0, 0.0, 5.0]
    rigid.softID[0] = 0
    rigid.is_fix[0] = [1, 1, 0]
    for p, mass in enumerate((1.0, 3.0)):
        point.active[p] = 1
        point.bodyID[p] = 0
        point.m[p] = mass
    point.x[0], point.x[1] = [0.0, 0.0, 2.0], [4.0, 0.0, 8.0]
    point.v[0], point.v[1] = [1.0, 2.0, 1.0], [3.0, 4.0, 5.0]

    reset_soft_body_kinematic_reduction_(1, soft)
    reduce_soft_body_kinematics_(2, soft, point, rigid)
    np.testing.assert_allclose(soft.previous_center.to_numpy()[0], [12.0, 0.0, 26.0])
    np.testing.assert_allclose(soft.v.to_numpy()[0], [10.0, 14.0, 16.0])

    apply_soft_body_translation_constraint_(2, soft, point, rigid)
    positions = point.x.to_numpy()
    velocities = point.v.to_numpy()
    np.testing.assert_allclose(np.average(positions, axis=0, weights=(1.0, 3.0)), [3.0, 0.0, 5.0])
    np.testing.assert_allclose(np.average(velocities, axis=0, weights=(1.0, 3.0)), [2.5, 3.5, 0.0])
    np.testing.assert_allclose(positions[1] - positions[0], [4.0, 0.0, 6.0])
    np.testing.assert_allclose(velocities[1] - velocities[0], [2.0, 2.0, 4.0])


def test_tlmpm_split_stages_reuse_cached_stress_and_update_it_once():
    source_path = (
        Path(__file__).parents[3]
        / "src/mpm/soft_particle/SoftBodyKernel.py"
    )
    source = source_path.read_text(encoding="utf-8")
    p2g_start = source.index("def soft_body_force_p2g_(")
    p2g_end = source.index("\n\n@ti.kernel", p2g_start)
    stress_start = source.index("def update_soft_particle_stress_(")
    stress_end = source.index("\n\n@ti.kernel", stress_start)

    assert "def soft_body_tlmpm_step_(" not in source
    assert "stress = material_point[p].stress" in source[p2g_start:p2g_end]
    assert source[stress_start:stress_end].count("soft_particle_pk1(") == 1


def test_tlmpm_scheduler_profiles_split_kernels_in_physical_order():
    source_path = (
        Path(__file__).parents[3]
        / "src/mpm/engines/SoftParticleEngine.py"
    )
    source = source_path.read_text(encoding="utf-8")
    start = source.index("    def euler_lsmpm_integration(")
    stage_names = [
        "reset_soft_grid_step_(",
        "soft_body_force_p2g_(",
        "soft_surface_force_p2g_(",
        "update_soft_grid_kinematic_(",
        "soft_body_g2p_(",
        "reset_soft_body_kinematic_reduction_(",
        "reduce_soft_body_kinematics_(",
        "apply_soft_body_translation_constraint_(",
        "remap_soft_grid_velocity_(",
        "update_soft_particle_stress_(",
        "prepare_soft_body_surface_frame_(",
        "reduce_soft_body_surface_frame_(",
        "finalize_soft_body_rotation_(",
        "track_soft_surface_points_(",
        "finalize_soft_body_bounds_(",
    ]
    stage_positions = [source.index(name, start) for name in stage_names]

    assert stage_positions == sorted(stage_positions)
    for label in (
        "LSMPM Grid reset",
        "LSMPM Force P2G",
        "LSMPM Surface force P2G",
        "LSMPM Grid kinematic",
        "LSMPM G2P",
        "LSMPM Body reduction",
        "LSMPM Constraint correction",
        "LSMPM Velocity remap P2G",
        "LSMPM Stress update",
        "LSMPM Surface frame reduction",
        "LSMPM Surface rotation",
        "LSMPM Surface point tracking",
        "LSMPM Bounding volume finalize",
        "LSMPM SDF transport",
    ):
        assert f'sims.timer.begin("{label}")' in source[start:]
        assert f'sims.timer.end("{label}")' in source[start:]
    assert "reduce_soft_material_bounds_(" not in source[start:]
