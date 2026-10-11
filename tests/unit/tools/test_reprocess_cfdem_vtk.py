import meshio
import numpy as np
import pytest

from tools.postprocessing.mpdem.reprocess_cfdem_vtk import digest, reprocess_case


@pytest.mark.parametrize("has_air", [False, True])
def test_reprocess_validates_real_vtk_and_preserves_raw_data_and_old_vtks(tmp_path, has_air):
    folder = tmp_path / "result"
    (folder / "particles").mkdir(parents=True)
    (folder / "grids").mkdir()
    (folder / "vtks").mkdir()
    (folder / "vtks/previous.txt").write_text("old export")
    coords = np.stack(np.meshgrid(np.arange(3), np.arange(3), np.arange(3), indexing="ij"), axis=-1).reshape(-1, 3)
    pressure = np.zeros((2, 2, 2, 1))
    types = np.full_like(pressure, 2, dtype=int)
    pressure[0, 0, 0, 0], pressure[1, 1, 1, 0] = 100, 300
    types[0, 0, 0, 0] = types[1, 1, 1, 0] = 1
    if has_air:
        types[1, 1, 1, 0] = 0
    np.savez(folder / "grids/MPMGrid000000.npz", t_current=0, coords=coords, cell_type=types, cell_pressure=pressure)
    np.savez(
        folder / "particles/MPMParticle000000.npz",
        t_current=0,
        particleID=[0, 1],
        position=[[0, 0, 0], [1, 1, 1]],
        mass=[1, 1],
        bodyID=[0, 0],
        materialID=[1, 1],
        volume=[1, 1],
        velocity=np.zeros((2, 3)),
        pressure=[110.0, 310.0],
    )
    (folder / "SHA256SUMS").write_text(
        f'{digest(folder / "particles/MPMParticle000000.npz")}  ./particles/MPMParticle000000.npz\n'
        f'{digest(folder / "vtks/previous.txt")}  ./vtks/previous.txt\n'
    )
    report = reprocess_case(folder, tmp_path / "backup", 3, None)
    assert report["passed"] and report["vtk_files"] == 2
    assert report["frames"][0]["fluid_grid_pressure_mean_pa"] == (100 if has_air else 0)
    assert (tmp_path / "backup/result/vtks/previous.txt").read_text() == "old export"
    mesh = meshio.read(folder / "vtks/GraphicMPMParticle000000.vtu")
    np.testing.assert_array_equal(mesh.point_data["pressure"].reshape(-1), [110, 310] if has_air else [-90, 110])
    with np.load(folder / "particles/MPMParticle000000.npz") as original:
        np.testing.assert_array_equal(original["pressure"], [110, 310])
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert digest(folder / relative) == expected
