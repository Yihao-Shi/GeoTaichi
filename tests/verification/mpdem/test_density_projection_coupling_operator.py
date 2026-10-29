"""Density correction must solve its own Laplacian, including inside CFDEM."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.linear_solver.MatrixFreePCG import MatrixFreePCG
from src.mpm.engines.IncompressibleEngine import IncompressibleEngine
from src.mpm.engines.Operator import PoissonEquationOperator
from tests.verification.mpdem.test_incompressible_coupling_kernels import FluidProperties

pytestmark = [pytest.mark.verification, pytest.mark.mpdem, pytest.mark.cpu, pytest.mark.isolated_dimension(3)]


@pytest.mark.parametrize("coupling_mode", [0, 1, 2])
@pytest.mark.parametrize("cut_cell", [False, True])
def test_density_projection_matches_laplacian_inside_coupled_solver(taichi_runtime, coupling_mode, cut_cell):
    shape, offset = (5, 5, 5), (-1, -1, -1)

    def scalar(dtype=float):
        return ti.field(dtype, shape=shape, offset=offset)

    cell_type = scalar(int)
    types = np.full(shape, 2, dtype=np.int32)
    types[1:4, 1:4, 1:4] = 1
    cell_type.from_numpy(types)
    sdf = scalar()
    sdf.fill(-1.0)
    solid_sdf = scalar()
    solid_sdf.fill(1.0)
    fraction = scalar()
    fraction.fill(0.25)
    solid_density = scalar()
    solid_density.fill(1140.0)
    face_fraction = [scalar() for _ in range(3)]
    for field in face_fraction:
        field.fill(0.5)
    flag = ti.field(int, shape=27)
    flag.fill(-1)
    flag[13] = 0
    spacing = np.array([0.125, 0.25, 0.5])
    element = SimpleNamespace(
        ghost_cell=1,
        cnum=ti.Vector(list(shape)),
        grid_size=ti.Vector(spacing),
        igrid_size=ti.Vector(1.0 / spacing),
        flag=flag,
        cell=SimpleNamespace(type=cell_type, fluid_sdf=sdf, solid_sdf=solid_sdf),
    )
    scene = SimpleNamespace(element=element)
    props = FluidProperties(density=1000.0)
    dt = ti.field(float, shape=())
    dt[None] = 1e-3
    sims = SimpleNamespace(
        dt=dt,
        density_projection=True,
        density_projection_interior_only=True,
        density_projection_tolerance=0.0,
        density_projection_error_clamp=0.1,
        iter_max=10,
        residual_tolerance=1e-8,
        linear_solver_relative_tolerance=1e-12,
    )
    engine = object.__new__(IncompressibleEngine)
    engine.dimension = 3
    engine.poisson_solver = MatrixFreePCG(1)
    engine.operator = PoissonEquationOperator(3)
    engine.operator.link_ptrs(
        scene, face_fraction if cut_cell else None, False, coupling_mode, fraction, fraction, solid_density, props
    )
    engine.unknow_vector = ti.field(float, shape=1)
    engine.right_hand_vector = ti.field(float, shape=1)
    engine.diag_A = ti.field(float, shape=1)
    engine.density_projection_cell_density = scalar()
    engine.density_projection_cell_density.fill(1000.0)
    engine.density_projection_cell_density[1, 1, 1] = 1010.0
    engine.density_projection_cell_fraction = scalar()
    engine.density_projection_cell_fraction.fill(1.0)
    engine.density_projection_pressure = scalar()
    engine.solid_face_open_fraction = face_fraction
    engine.solid_face_fraction2 = lambda: face_fraction[2]
    engine.get_single_fluid_mat_props = lambda _: props
    engine.prepare_density_projection_state = lambda *_: None
    engine.density_projection_is_required = lambda *_: True
    engine.find_density_projection_active_nodes = lambda *_: 1
    engine.apply_density_projection_position_correction = lambda *_: None
    engine.prepare_density_projection_equations_step = (
        engine.prepare_density_projection_equations_cut_cell
        if cut_cell
        else engine.prepare_density_projection_equations_regular
    )

    engine.density_projection_pcg(sims, scene)

    # One active interior cell with zero correction on all six neighboring cells.
    rhs = 1000.0 * 0.01 / dt[None] ** 2
    diagonal = 2.0 * np.sum(1.0 / spacing**2) * (0.5 if cut_cell else 1.0)
    assert engine.density_projection_pressure[1, 1, 1] == pytest.approx(rhs / diagonal, rel=1e-11)
    assert engine.operator.coupling_mode == coupling_mode, "The next physical pressure solve must retain its coupling"
