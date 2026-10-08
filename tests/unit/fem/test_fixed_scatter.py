"""Permanent FEM slots must represent the same operator as raw assembly."""

import numpy as np
import pytest

from src.fem.elements import create_element
from src.fem.generator import FEMGenerateManager
from src.fem.MaterialManager import FEMMaterialManager
from src.fem.engines.ClassicalAssembler import ClassicalAssembler
from src.linear_solver.BuildTriplet import BuildTriplet

pytestmark = [pytest.mark.unit, pytest.mark.fem, pytest.mark.assembly]


@pytest.mark.parametrize("element_type", ["TET4", "HEX8"])
def test_fixed_scatter_matches_raw_for_deformed_solids(taichi_runtime, element_type):
    import taichi as ti

    mesh = FEMGenerateManager().create_box(size=(1.0, 1.0, 1.0), divisions=(1, 1, 1), element_type=element_type)
    material = FEMMaterialManager().material_handle("StVK", young_modulus=1e4, poisson_ratio=0.3, density=1.0)
    assembler = ClassicalAssembler(mesh, create_element(mesh), material, project_pd=False)
    coordinates, slots = assembler.fixed_block_coordinates()
    slot_field = ti.field(ti.i32, shape=slots.shape)
    slot_field.from_numpy(slots)
    matrices = [
        BuildTriplet(
            dim=3,
            max_pairs_num=assembler.stiffness_block_pair_count,
            max_nonzeros=assembler.stiffness_unique_block_pair_count,
            max_active_nodes=mesh.number_of_nodes,
            symmetric=False,
            matrix_symmetric=True,
            full_symmetric_input=True,
        )
        for _ in range(2)
    ]
    raw, fixed = matrices
    fixed.install_fixed_pattern(coordinates)
    for stretch in (1.02, 0.98):
        positions = mesh.points @ np.array([[stretch, 0.01, 0.0], [0.0, 1.01, 0.02], [0.0, 0.0, 1.0]])
        assembler.positions.from_numpy(positions)
        assembler.assemble_device(positions=assembler.positions, need_stiffness=True)
        raw.reset_system()
        fixed.reset_system()
        assembler.scatter_stiffness_to_hash(raw)
        assembler.scatter_stiffness_to_fixed(fixed, slot_field, assembler.positions)
        raw.finalize_taichi_assembly()
        fixed.finalize_taichi_assembly()
        np.testing.assert_allclose(fixed.to_scipy().toarray(), raw.to_scipy().toarray(), rtol=2e-13, atol=1e-9)
        assert fixed.raw_non_diag_count[0] == 0
