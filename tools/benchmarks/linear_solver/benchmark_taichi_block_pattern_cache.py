"""CUDA microbenchmark for persistent block-pattern reduction.

Example::

    python tools/benchmarks/linear_solver/benchmark_taichi_block_pattern_cache.py \
        --pairs 1000000 --blocks 50000 --iterations 20

The benchmark changes block values on-device. Timings therefore measure the
Taichi reduction/pattern path rather than PCIe uploads from the test harness.
"""

import argparse
import os
import sys
import time

import numpy as np
import taichi as ti


sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
)

from src.linear_solver.HashReduction import HashReduction


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=250_000)
    parser.add_argument("--blocks", type=int, default=20_000)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Run the device kernels on CPU for debugging; results are not a GPU benchmark.",
    )
    return parser.parse_args()


def main():
    args = _arguments()
    cuda_available = ti._lib.core.with_cuda()
    if not cuda_available and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; pass --allow-cpu only for kernel debugging")
    arch = ti.cuda if cuda_available else ti.cpu
    ti.init(arch=arch, default_fp=ti.f64, offline_cache=False)

    rng = np.random.default_rng(101)
    block_i = rng.integers(0, args.blocks, size=args.pairs, dtype=np.int32)
    # A local bandwidth keeps the matrix FEM/MPM-like and creates substantial
    # duplicate block contributions across neighboring stencils.
    block_j = np.clip(
        block_i + rng.integers(-16, 17, size=args.pairs, dtype=np.int32),
        0,
        args.blocks - 1,
    ).astype(np.int32)
    block_h = rng.normal(size=(args.pairs, 9))
    unique_count = np.unique(
        (block_i.astype(np.int64) << 32) | block_j.astype(np.int64)
    ).size

    reduction = HashReduction(
        max_pairs_num=args.pairs,
        max_nnz=max(1, unique_count + unique_count // 50 + 8),
        hessian_size=9,
        pattern_cache=True,
        device_reduction=True,
    )
    reduction.set_triplets_from_numpy(block_i, block_j, block_h)

    @ti.kernel
    def update_values(scale: float):
        for pair in range(args.pairs):
            for component in ti.static(range(9)):
                reduction.blockH[pair][component] *= scale

    # Compile all kernels and establish one pattern.
    reduction.go(args.pairs)
    update_values(1.000001)
    ti.sync()

    reduction.configure_pattern_cache(enabled=False)
    start = time.perf_counter()
    for _ in range(args.iterations):
        update_values(1.000001)
        reduction.go(args.pairs)
    ti.sync()
    rebuild_seconds = time.perf_counter() - start

    reduction.configure_pattern_cache(enabled=True)
    reduction.go(args.pairs)
    ti.sync()
    start = time.perf_counter()
    for _ in range(args.iterations):
        update_values(0.999999)
        reduction.go(args.pairs)
    ti.sync()
    cached_seconds = time.perf_counter() - start

    print(
        {
            "arch": str(arch),
            "pairs": args.pairs,
            "unique_blocks": int(unique_count),
            "iterations": args.iterations,
            "rebuild_seconds": rebuild_seconds,
            "cached_seconds": cached_seconds,
            "reduction_speedup": rebuild_seconds / max(cached_seconds, 1.0e-30),
            "cache": reduction.pattern_cache_statistics(),
        }
    )


if __name__ == "__main__":
    main()
