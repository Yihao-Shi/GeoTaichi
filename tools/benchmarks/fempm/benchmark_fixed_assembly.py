"""Compare classical FEM raw reduction and permanent-slot device assembly.

Run from the repository root. Compilation is excluded; each sample synchronizes
Taichi. This measures material assembly, not a complete coupled timestep.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.fem.elements import create_element
from src.fem.generator import FEMGenerateManager
from src.fem.MaterialManager import FEMMaterialManager
from src.fem.engines.ClassicalAssembler import ClassicalAssembler
from src.linear_solver.BuildTriplet import BuildTriplet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--element", choices=("TET4", "HEX8"), default="HEX8")
    parser.add_argument("--divisions", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.divisions < 1 or args.repeats < 1:
        parser.error("divisions and repeats must be positive")
    ti.init(arch=getattr(ti, args.arch), default_fp=ti.f64, offline_cache=True)
    mesh = FEMGenerateManager().create_box(size=(1, 1, 1), divisions=(args.divisions,) * 3, element_type=args.element)
    material = FEMMaterialManager().material_handle("StVK", young_modulus=1e4, poisson_ratio=0.3, density=1.0)
    assembler = ClassicalAssembler(mesh, create_element(mesh), material, project_pd=True)
    coordinates, slots = assembler.fixed_block_coordinates()
    scatter = ti.field(ti.i32, shape=slots.shape)
    scatter.from_numpy(slots)
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
    matrices[1].install_fixed_pattern(coordinates)
    assembler.positions.from_numpy(mesh.points @ np.diag([1.01, 0.99, 1.0]))
    assembler.assemble_device(positions=assembler.positions, need_stiffness=True)

    def assemble(index):
        matrix = matrices[index]
        matrix.reset_system()
        if index:
            assembler.scatter_stiffness_to_fixed(matrix, scatter, assembler.positions)
        else:
            assembler.scatter_stiffness_to_hash(matrix)
        matrix.finalize_taichi_assembly()
        ti.sync()

    for index in (0, 1):
        assemble(index)
        assemble(index)
    reference, actual = (matrix.to_scipy().tocsr() for matrix in matrices)
    difference = actual - reference
    max_error = float(np.max(np.abs(difference.data), initial=0.0))
    if max_error > 1e-9 + 2e-12 * np.max(np.abs(reference.data)):
        raise AssertionError(f"fixed/raw matrix mismatch: {max_error}")
    samples = [[], []]
    for sample in range(args.repeats):
        for index in (0, 1) if sample % 2 == 0 else (1, 0):
            started = time.perf_counter()
            assemble(index)
            samples[index].append(time.perf_counter() - started)
    medians = [float(np.median(values)) for values in samples]
    report = dict(
        arch=args.arch,
        element=args.element,
        cells=mesh.number_of_cells,
        nodes=mesh.number_of_nodes,
        matrix_max_error=max_error,
        raw_seconds=medians[0],
        fixed_seconds=medians[1],
        speedup=medians[0] / medians[1],
        samples_seconds=samples,
        scope="material assembly only; excludes JIT",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
