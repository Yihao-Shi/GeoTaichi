"""Unit contracts for incompressible-cell pressure output sampling."""

import inspect

import numpy as np


def test_closed_domain_pressure_output_uses_zero_mean_gauge():
    from src.mpm.Recorder import _normalize_closed_domain_pressure

    pressure = np.array([100.0, 200.0, 300.0])
    cell_type = np.array([1, 2, 1])

    normalized = _normalize_closed_domain_pressure(pressure, cell_type)

    np.testing.assert_allclose(normalized, [-100.0, 200.0, 100.0])
    np.testing.assert_allclose(pressure, [100.0, 200.0, 300.0])


def test_free_surface_pressure_output_keeps_atmospheric_gauge():
    from src.mpm.Recorder import _normalize_closed_domain_pressure

    pressure = np.array([100.0, 0.0, 300.0])
    cell_type = np.array([1, 0, 1])

    np.testing.assert_array_equal(_normalize_closed_domain_pressure(pressure, cell_type), pressure)


def test_particle_boundary_is_reapplied_only_when_shifting_runs():
    from src.mpm.engines.IncompressibleEngine import IncompressibleEngine

    shifting = inspect.getsource(IncompressibleEngine.timed_particle_shifting)
    pcg_step = inspect.getsource(IncompressibleEngine.fdm_discretization_pcg)
    mgpcg_step = inspect.getsource(IncompressibleEngine.fdm_discretization_mgpcg)

    assert shifting.count("enforce_particle_boundary") == 1
    assert pcg_step.count("enforce_particle_boundary") == 1
    assert mgpcg_step.count("enforce_particle_boundary") == 1


class _ArrayField:
    def __init__(self, array):
        self.array = np.asarray(array)

    def to_numpy(self):
        return self.array.copy()


class _FakeCell:
    def __init__(self, pressure, cell_type, surface_tension):
        self.pressure = _ArrayField(pressure)
        self.type = _ArrayField(cell_type)
        self.surface_tension = _ArrayField(surface_tension)


class _FakeElement:
    def __init__(
        self,
        pressure,
        cell_type,
        surface_tension,
        ghost_cell,
        grid_size,
    ):
        self.cell = _FakeCell(pressure, cell_type, surface_tension)
        self.cnum = np.asarray(cell_type.shape, dtype=np.int64)
        self.ghost_cell = ghost_cell
        self.grid_size = np.asarray(grid_size, dtype=np.float64)


class _FakeScene:
    pass


class _FakeSims:
    dimension = 3


def test_incompressible_pressure_output_ignores_noninterface_air_corner():
    from src.mpm.Recorder import WriteFile

    pressure = np.zeros((4, 4, 4), dtype=np.float64)
    cell_type = np.full((4, 4, 4), 2, dtype=np.int32)
    surface_tension = np.zeros_like(pressure)
    cell_type[1, 1, 1] = 1
    pressure[1, 1, 1] = 1234.0
    cell_type[0, 0, 0] = 0

    scene = _FakeScene()
    scene.element = _FakeElement(
        pressure,
        cell_type,
        surface_tension,
        ghost_cell=1,
        grid_size=[0.05, 0.05, 0.05],
    )
    scene.material = object()

    writer = WriteFile.__new__(WriteFile)
    sampled = writer.sample_incompressible_cell_pressure(_FakeSims(), scene, np.array([[0.01, 0.01, 0.01]]))

    assert np.isclose(sampled[0], pressure[1, 1, 1], rtol=1.0e-12, atol=1.0e-12)


def test_incompressible_pressure_sampling_normalizes_closed_domain_gauge():
    from src.mpm.Recorder import WriteFile

    pressure = np.zeros((4, 4, 4), dtype=np.float64)
    cell_type = np.full((4, 4, 4), 2, dtype=np.int32)
    surface_tension = np.zeros_like(pressure)
    cell_type[1, 1, 1] = 1
    cell_type[2, 1, 1] = 1
    pressure[1, 1, 1] = 100.0
    pressure[2, 1, 1] = 300.0

    scene = _FakeScene()
    scene.element = _FakeElement(
        pressure,
        cell_type,
        surface_tension,
        ghost_cell=1,
        grid_size=[0.05, 0.05, 0.05],
    )
    scene.material = object()

    writer = WriteFile.__new__(WriteFile)
    sampled = writer.sample_incompressible_cell_pressure(
        _FakeSims(), scene, np.array([[0.025, 0.025, 0.025], [0.075, 0.025, 0.025]])
    )

    np.testing.assert_allclose(sampled, [-100.0, 100.0])
