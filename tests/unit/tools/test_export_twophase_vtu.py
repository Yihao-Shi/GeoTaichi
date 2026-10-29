import meshio
import numpy as np
import pytest
from pathlib import Path
import re
import subprocess

from tools.postprocessing.mpdem.export_twophase_vtu import export


def snapshots(folder, grid_time=0, dimension=3):
    (folder / "particles").mkdir(parents=True)
    (folder / "grids").mkdir()
    np.savez(
        folder / "particles/MPMParticle000000.npz",
        t_current=0,
        particleID=[0, 1],
        position=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])[:, :dimension],
        mass=[1.0, 1.0],
        bodyID=[0, 0],
        materialID=[1, 1],
        volume=[1.0, 1.0],
        velocity=np.zeros((2, dimension)),
        solid_velocity=np.ones((2, dimension)),
        fluid_velocity=np.full((2, dimension), 2.0),
        pressure=[110.0, 310.0],
        phase=[1, 2],
    )
    coords = np.stack(np.meshgrid(*([np.arange(3)] * dimension), indexing="ij"), axis=-1).reshape(-1, dimension)
    np.savez(
        folder / "grids/MPMGrid000000.npz",
        t_current=grid_time,
        coords=coords,
        cell_type=np.ones((2,) * dimension + (1,), dtype=int),
        cell_pressure=np.full((2,) * dimension + (1,), 200.0),
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_export_preserves_two_phase_pressure_and_phase_velocity(tmp_path, dimension):
    snapshots(tmp_path, dimension=dimension)
    report = export(tmp_path)
    assert report["passed"] and report["frames"] == 1 and report["vtk_files"] == 3
    for name, pressure, velocity in (("Solid", 110.0, 1.0), ("Fluid", 310.0, 2.0)):
        mesh = meshio.read(tmp_path / f"vtks/GraphicMPM{name}Particle000000.vtu")
        np.testing.assert_array_equal(mesh.point_data["pressure"].reshape(-1), [pressure])
        np.testing.assert_array_equal(mesh.point_data["velocity"][:, :dimension], np.full((1, dimension), velocity))
        if dimension == 2:
            np.testing.assert_array_equal(mesh.points[:, 2], 0)
            np.testing.assert_array_equal(mesh.point_data["velocity"][:, 2], 0)
    with np.load(tmp_path / "particles/MPMParticle000000.npz") as original:
        np.testing.assert_array_equal(original["pressure"], [110.0, 310.0])
    with pytest.raises(FileExistsError):
        export(tmp_path)


def test_reject_unsynchronized_snapshot_before_export(tmp_path):
    snapshots(tmp_path, grid_time=0.1)
    with pytest.raises(ValueError, match="times differ"):
        export(tmp_path)
    assert not (tmp_path / "vtks").exists()


def test_reject_missing_frame_before_export(tmp_path):
    snapshots(tmp_path)
    (tmp_path / "particles/MPMParticle000000.npz").rename(tmp_path / "particles/MPMParticle000001.npz")
    with pytest.raises(ValueError, match="non-contiguous"):
        export(tmp_path)


def test_pull_snapshot_excludes_incomplete_save_header():
    script = Path(__file__).resolve().parents[3] / "research/remote_validation/pull_semi_implicit_outputs_20261013.sh"
    if not script.exists():
        pytest.skip("Local transfer script is not part of the tracked source tree")
    program = re.search(r"cutoff=\$\(awk '(.+)' \"\$3\"\)", script.read_text()).group(1)
    log = (
        "# MPM Save | Step = 110000 | Save Number = 22 | Simulation Time = 2.2\n"
        "# Time record accmulated(execute num): Output: 14.793(23)\n"
        "# MPM Save | Step = 115000 | Save Number = 23 | Simulation Time = 2.3\n"
    )
    result = subprocess.run(["awk", program], input=log, text=True, capture_output=True, check=True)
    assert result.stdout.strip() == "22"
