from types import SimpleNamespace

from src.mpdem.mainDEMPM import DEMPM


class _SimulationStub:
    def __init__(self):
        self.time = None
        self.gravity = None

    def set_simulation_time(self, value):
        self.time = value

    def set_gravity(self, value):
        self.gravity = value


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
