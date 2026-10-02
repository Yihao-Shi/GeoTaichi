import inspect

from src.dem.Simulation import Simulation


def test_contact_work_backend_contract_has_no_dead_size_dispatch():
    simulation = object.__new__(Simulation)
    simulation.max_particle_num = 1
    simulation.max_wall_num = 0
    simulation.define_work_load()

    assert simulation.particle_work == 2
    assert simulation.wall_work == 2
    source = inspect.getsource(Simulation.define_work_load)
    assert "max_particle_num <=" not in source
    assert "max_particle_num *" not in source
