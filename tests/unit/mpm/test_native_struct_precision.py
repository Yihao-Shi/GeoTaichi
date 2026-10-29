"""Importing native MPM before ti.init must preserve configured storage precision."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("initialize_first", [False, True])
def test_particle_and_grid_scalar_precision_matches_vectors(precision, initialize_first):
    program = """
import os
import taichi as ti

def initialize():
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)

if os.environ['INITIALIZE_FIRST'] == '1':
    initialize()
from src.mpm.structs.Particle import ParticleCloudTwoPhase2D, ParticleCloudTwoPhase, ImplicitParticle2D
from src.mpm.structs.GridNode import Nodes2D, Nodes, ImplicitNodes2D
from src.mpm.soft_particle.Structs import DeformableGrid, SoftBody, SoftMaterialPoint, TemplateSoftNode
if os.environ['INITIALIZE_FIRST'] == '0':
    initialize()
expected = ti.f64 if os.environ['GEOTAICHI_REAL_DTYPE'] == 'float64' else ti.f32
for schema in (ParticleCloudTwoPhase2D, ParticleCloudTwoPhase):
    field = schema.field(shape=1)
    for name in ('m', 'ms', 'mf', 'vol', 'porosity', 'pressure', 'permeability'):
        assert getattr(field, name).dtype == expected, (schema, name)
    assert field.x.dtype == field.v.dtype == expected
for schema in (Nodes2D, Nodes, ImplicitNodes2D, ImplicitParticle2D):
    assert schema.field(shape=1).m.dtype == expected, schema
for schema, names in (
    (DeformableGrid, ('m',)),
    (SoftBody, ('scale', 'gridSpace', 'm', 'external_load_factor', 'damp_energy')),
    (SoftMaterialPoint, ('m', 'vol0', 'surface_weight', 'strain_energy')),
    (TemplateSoftNode, ('parameter',)),
):
    field = schema.field(shape=1)
    for name in names:
        assert getattr(field, name).dtype == expected, (schema, name)
ti.sync()
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "GEOTAICHI_REAL_DTYPE": precision, "INITIALIZE_FIRST": str(int(initialize_first))},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
