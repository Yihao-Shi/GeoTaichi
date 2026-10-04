"""CPT pressure stays attached to material points as grid support changes."""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu, pytest.mark.serial, pytest.mark.isolated_dimension(2)]


@pytest.mark.parametrize("transfer", ["ulmpm", "fempm", "igampm", "ipc_mpm"])
def test_cpt_particle_pressure_follows_material_points(taichi_runtime, tmp_path, monkeypatch, transfer):
    from examples.mpm.Contact.CPT2D.coupled_cpt import configure_direct_axisymmetric_mpm
    from examples.mpm.Contact.CPT2D.cpt_reference import SOIL_SIZE, SURFACE_PRESSURE
    from src.fempm.ImplicitEngine import FEMPMImplicitEngine
    from src.igampm.engines.ImplicitEngine import ImplicitEngineMixin
    from src.mpm import MPM
    from src.mpm import config
    from src.mpm.soft_particle.IPCULMPM import IPCULMPM
    from src.utils.linalg import no_operation

    monkeypatch.setattr(config, "DYNAMIC", False)
    mpm = MPM(log=False)
    dx, particle_count = configure_direct_axisymmetric_mpm(mpm, tmp_path, 1.0e-3, 1.0e-3, 1.0e-3, 50.0)
    engine = mpm.enginer
    engine.init_F0()
    assert particle_count == 40
    assert engine.neumann.num == 0
    assert engine.compute_traction
    count = int(engine.tractionNum[0])
    ids = engine.traction.particleID.to_numpy()[:count]
    pressure = engine.traction.pressure.to_numpy()[:count]
    areas = engine.traction.surface_area.to_numpy()[:count]
    np.testing.assert_array_equal(pressure, np.tile([0.0, -SURFACE_PRESSURE], (count, 1)))
    forces = pressure * areas[:, None]
    total_force = np.array([0.0, -SURFACE_PRESSURE * np.pi * SOIL_SIZE[0] ** 2])
    np.testing.assert_allclose(forces.sum(axis=0), total_force)
    initial_positions = engine.particle.x.to_numpy()
    assert np.all(initial_positions[ids, 1] == np.max(initial_positions[:, 1]))

    class Prepared(Exception):
        pass

    def stop_before_solve(_verbose):
        raise Prepared

    if transfer == "ulmpm":
        monkeypatch.setattr(engine, "_solve_lagged_material_inner", stop_before_solve)
    context = SimpleNamespace(
        mpm=engine,
        transfer_mpm_traction=engine.traction_p2g,
        update_mpm_mass_list=no_operation,
        compute_mpm_traction=engine.traction_p2g,
        compute_dynamic_mass_list=no_operation,
        _validate_active_mpm_dofs=no_operation,
    )
    # A safe but deliberately stale map exposes transfers that run before rebuilding it.
    engine.node2dof.fill(1)
    initial_loaded_nodes = None
    for moved in (False, True):
        if moved:
            positions = initial_positions.copy()
            positions[positions[:, 1] > SOIL_SIZE[1] - dx, 1] -= 2.25 * dx
            engine.particle.x.from_numpy(positions)
        if transfer == "ulmpm":
            with pytest.raises(Prepared):
                engine.substep(verbose=False)
        elif transfer == "fempm":
            FEMPMImplicitEngine._prepare_mass_mpm_transfer(context)
        elif transfer == "igampm":
            ImplicitEngineMixin._prepare_updated_lagrangian_mpm_step(context)
        else:
            IPCULMPM.prepare_step_device(context)

        mass = engine.grid.m.to_numpy()
        active = mass > engine.val_lim
        grid_force = np.zeros((mass.size, 2))
        nodes = engine.LnID.to_numpy()
        shapes = engine.shape.to_numpy()
        offsets = engine.offset.to_numpy()
        for particle_id, force in zip(ids, forces):
            support = slice(0, offsets[particle_id])
            np.add.at(grid_force, nodes[particle_id, support], shapes[particle_id, support, None] * force)
        expected = grid_force[active].ravel()
        np.testing.assert_allclose(engine.volume_force.to_numpy()[: engine.active_dof], expected, atol=1.0e-9)
        np.testing.assert_allclose(grid_force.sum(axis=0), total_force, atol=1.0e-9)
        if moved:
            assert np.any(mass[initial_loaded_nodes] == 0.0)
            np.testing.assert_array_equal(engine.traction.particleID.to_numpy()[:count], ids)
        else:
            initial_loaded_nodes = np.flatnonzero(np.any(grid_force != 0.0, axis=1))

        engine.rhs.fill(0.0)
        engine.assemble_inertia_force(engine.active_dof, 0.0, [0.0, 0.0], engine.integration, engine.grid_disp)
        np.testing.assert_allclose(engine.rhs.to_numpy()[: engine.active_dof], expected, atol=1.0e-9)
        engine.grid_disp.fill(0.01)
        engine.energy[None] = 0.0
        engine.get_inertia_energy(0.0, engine.integration, [0.0, 0.0], engine.grid_disp)
        assert engine.energy[None] == pytest.approx(-0.01 * total_force.sum())
