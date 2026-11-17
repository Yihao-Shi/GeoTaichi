"""Defect corrections share a fixed operator while recomputing the velocity RHS."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.linear_solver.MultiGridPCG import MGPCGPoissonSolver
from src.mpm.engines.ULSemiImplicitTwoPhaseDoubleLayerEngine import ULSemiImplicitTwoPhaseDoubleLayerEngine


@pytest.mark.parametrize("dimension", [2, 3])
def test_reused_pressure_operator_matches_fresh_solve(taichi_runtime, dimension):
    shape = (4,) * dimension
    engine = object.__new__(ULSemiImplicitTwoPhaseDoubleLayerEngine)
    engine.poisson_solver = MGPCGPoissonSolver(dimension, shape, n_mg_levels=1, bottom_smoothing=10)
    engine.poisson_iterations = 100
    for name, value in (
        ("cell_type", 1),
        ("cell_pressure", 0.0),
        ("cell_solid_density", 2500.0),
        ("cell_fluid_density", 1000.0),
        ("cell_fluid_sdf", -0.1),
        ("cell_pressure_mobility", 0.001),
    ):
        field = ti.field(ti.i32 if name == "cell_type" else ti.f64, shape=shape)
        field.fill(value)
        setattr(engine, name, field)
    kinds = np.ones(shape, dtype=np.int32)
    kinds[(0,) + (slice(None),) * (dimension - 1)] = 0
    engine.cell_type.from_numpy(kinds)
    for axis, label in enumerate("xyz"[:dimension]):
        face_shape = tuple(n + int(d == axis) for d, n in enumerate(shape))
        for prefix, value in (("face_porosity", 0.4), ("solid_velocity", 0.0), ("fluid_velocity", 0.0)):
            field = ti.field(ti.f64, shape=face_shape)
            field.fill(value)
            setattr(engine, f"{prefix}_{label}", field)
        velocity = getattr(engine, f"fluid_velocity_{label}")
        velocity.from_numpy(np.indices(face_shape, dtype=np.float64)[axis] * 0.01)
    setattr(engine, f"_build_fluid_sdf{dimension}d", lambda *_: None)
    setattr(engine, f"_assemble_pressure_mg_levels{dimension}d", lambda *_: None)
    timestep = ti.field(ti.f64, shape=())
    timestep[None] = 1.0e-4
    sims = SimpleNamespace(dt=timestep, residual_tolerance=1.0e-10)
    scene = SimpleNamespace(element=SimpleNamespace(grid_size=ti.Vector([0.1] * dimension)))
    solve = getattr(engine, f"_solve_pressure{dimension}d")
    solve(sims, scene)
    previous_rhs = engine.poisson_solver.b.to_numpy()
    diagonal = engine.poisson_solver.Adiag[0].to_numpy()
    for label in "xyz"[:dimension]:
        velocity = getattr(engine, f"fluid_velocity_{label}")
        velocity.from_numpy(velocity.to_numpy() * 1.7)
    solve(sims, scene, reuse_operator=True)
    reused = engine.cell_pressure.to_numpy()
    assert not np.array_equal(engine.poisson_solver.b.to_numpy(), previous_rhs)
    np.testing.assert_array_equal(engine.poisson_solver.Adiag[0].to_numpy(), diagonal)
    solve(sims, scene)
    np.testing.assert_allclose(engine.cell_pressure.to_numpy(), reused, rtol=1e-11, atol=1e-11)
