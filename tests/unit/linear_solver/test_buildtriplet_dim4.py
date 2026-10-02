import numpy as np
import taichi as ti

from src.linear_solver.BuildTriplet import BuildTriplet


def test_buildtriplet_supports_dim4_and_scalar_paths():
    ti.init(arch=ti.cpu, default_fp=ti.f64, debug=False, offline_cache=False)
    matrix = BuildTriplet(dim=4, max_pairs_num=16, max_nonzeros=16, max_active_nodes=3, symmetric=False)

    @ti.kernel
    def fill():
        for d in ti.static(range(4)):
            matrix.diag[0][d * 4 + d] = 2.0
            matrix.diag[1][d * 4 + d] = 3.0

        idx = ti.atomic_add(matrix.raw_non_diag_count[0], 1)
        matrix.non_diag.blockI[idx] = 0
        matrix.non_diag.blockJ[idx] = 1
        k = 0
        while k < 16:
            matrix.non_diag.blockH[idx][k] = 0.0
            k += 1
        matrix.non_diag.blockH[idx][3] = 1.25

        idx = ti.atomic_add(matrix.raw_non_diag_count[0], 1)
        matrix.non_diag.blockI[idx] = 0
        matrix.non_diag.blockJ[idx] = 1
        k = 0
        while k < 16:
            matrix.non_diag.blockH[idx][k] = 0.0
            k += 1
        matrix.non_diag.blockH[idx][3] = 0.75

    matrix.reset_system()
    fill()
    matrix.finalize_taichi_assembly()
    assembled = matrix.to_scipy(2).toarray()
    print(
        f"BuildTriplet dim=4: shape={assembled.shape}, nnz={np.count_nonzero(assembled)}, merged={assembled[0, 7]:.3e}"
    )
    assert assembled.shape == (8, 8)
    assert np.isclose(assembled[0, 0], 2.0)
    assert np.isclose(assembled[4, 4], 3.0)
    assert np.isclose(assembled[0, 7], 2.0)

    scalar = BuildTriplet(dim=4, max_pairs_num=8, max_nonzeros=8, max_active_nodes=2, symmetric=False)
    scalar.reset_system()
    scalar.assemble_scalar_triplets(
        rows=np.asarray([0, 0, 4], dtype=np.int32),
        cols=np.asarray([7, 7, 4], dtype=np.int32),
        vals=np.asarray([1.25, 0.75, 3.0], dtype=np.float64),
    )
    scalar.finalize_taichi_assembly()
    assembled_scalar = scalar.to_scipy(2).toarray()
    print(f"BuildTriplet scalar dim=4: nnz={np.count_nonzero(assembled_scalar)}, merged={assembled_scalar[0, 7]:.3e}")
    assert np.isclose(assembled_scalar[0, 7], 2.0)
    assert np.isclose(assembled_scalar[4, 4], 3.0)

    scalar1 = BuildTriplet(dim=1, max_pairs_num=8, max_nonzeros=8, max_active_nodes=3, symmetric=False)
    scalar1.reset_system()
    scalar1.assemble_scalar_triplets(
        rows=np.asarray([0, 0, 1, 1], dtype=np.int32),
        cols=np.asarray([1, 1, 1, 2], dtype=np.int32),
        vals=np.asarray([1.0, 2.0, 4.0, 5.0], dtype=np.float64),
    )
    scalar1.finalize_taichi_assembly()
    assembled_dim1 = scalar1.to_scipy(3).toarray()
    print(f"BuildTriplet scalar dim=1: nnz={np.count_nonzero(assembled_dim1)}, merged={assembled_dim1[0, 1]:.3e}")
    assert np.isclose(assembled_dim1[0, 1], 3.0)
    assert np.isclose(assembled_dim1[1, 1], 4.0)
    assert np.isclose(assembled_dim1[1, 2], 5.0)

    direct = BuildTriplet(dim=3, max_pairs_num=8, max_nonzeros=8, max_active_nodes=2, symmetric=False)

    @ti.kernel
    def fill_direct():
        direct.add_scalar_entry(0, 0, 2.0, 0, 0)
        direct.add_scalar_entry(0, 1, 3.0, 0, 1)
        direct.add_scalar_entry(0, 3, 4.0, 0, 0)
        direct.add_scalar_entry(3, 0, 5.0, 0, 0)

    direct.reset_system()
    fill_direct()
    direct.finalize_taichi_assembly()
    assembled_direct = direct.to_scipy(2).toarray()
    print(
        f"BuildTriplet direct scalar entry: nnz={np.count_nonzero(assembled_direct)}, offdiag={assembled_direct[0, 3]:.3e}"
    )
    assert np.isclose(assembled_direct[0, 0], 2.0)
    assert np.isclose(assembled_direct[0, 1], 3.0)
    assert np.isclose(assembled_direct[0, 3], 4.0)
    assert np.isclose(assembled_direct[3, 0], 5.0)


def main():
    test_buildtriplet_supports_dim4_and_scalar_paths()


if __name__ == "__main__":
    main()
