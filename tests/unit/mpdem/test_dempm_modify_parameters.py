from types import SimpleNamespace

import pytest

from src.mpdem.mainDEMPM import DEMPM


class _SimulationStub:
    def __init__(self):
        self.time = None
        self.gravity = None
        self.timestep = None
        self.dem_timestep = None

    def set_simulation_time(self, value):
        self.time = value

    def set_gravity(self, value):
        self.gravity = value

    def set_timestep(self, value):
        self.timestep = value

    def set_dem_timestep(self, value):
        self.dem_timestep = value


def test_modify_parameters_propagates_stage_time_and_gravity():
    coupling = _SimulationStub()
    mpm = _SimulationStub()
    dem = _SimulationStub()
    solver = DEMPM.__new__(DEMPM)
    solver.sims = coupling
    solver.mpm = SimpleNamespace(sims=mpm)
    solver.dem = SimpleNamespace(sims=dem)

    solver.modify_parameters(SimulationTime=1.25, gravity=[3.0, 0.0, -4.0])

    assert coupling.time == 1.25
    assert mpm.time == 1.25
    assert dem.time == 1.25
    assert mpm.gravity == [3.0, 0.0, -4.0]
    assert dem.gravity == [3.0, 0.0, -4.0]


@pytest.mark.parametrize("direct_affine", [False, True])
def test_modify_timestep_synchronizes_direct_affine_without_changing_other_dem_substeps(direct_affine):
    solver = DEMPM.__new__(DEMPM)
    solver.sims = _SimulationStub()
    solver.sims.dem_timestep = 0.0002
    solver.mpm = SimpleNamespace(sims=_SimulationStub())
    solver.dem = SimpleNamespace(sims=_SimulationStub())
    solver.is_direct_mpm_affine_ipc = lambda: direct_affine
    solver.modify_parameters(SimulationTime=1.25, Timestep=0.001)
    assert solver.sims.timestep == 0.001
    assert solver.sims.dem_timestep == (0.001 if direct_affine else 0.0002)
    assert solver.mpm.sims.timestep == solver.dem.sims.timestep == 0.001
