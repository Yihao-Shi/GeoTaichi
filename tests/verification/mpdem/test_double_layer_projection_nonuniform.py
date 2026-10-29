"""A nonuniform phase velocity exposes the pressure/correction operator mismatch."""

from pathlib import Path
import os
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def run_probe(output):
    import geotaichi as gt

    original_factory = gt.MPM

    def factory(*args, **kwargs):
        mpm = original_factory(*args, **kwargs)
        original_run = mpm.run

        def run(*args, **kwargs):
            particle = mpm.scene.particle
            position = particle.x.to_numpy()
            mask = particle.phase.to_numpy() == 2
            fluid = particle.vf.to_numpy()
            velocity = particle.v.to_numpy()
            fluid[mask, 0] += 0.15 * (position[mask, 0] - 0.15) / 0.1
            fluid[mask, 1] -= 0.07 * (position[mask, 1] - 0.15) / 0.1
            velocity[mask] = fluid[mask]
            particle.vf.from_numpy(fluid)
            particle.v.from_numpy(velocity)
            return original_run(*args, **kwargs)

        mpm.run = run
        return mpm

    gt.MPM = factory
    helper = runpy.run_path(str(Path(__file__).with_name("test_double_layer_projection_constraint.py")))
    helper["check_projection"](2, output)


def test_nonuniform_phase_velocity_projection(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "tests.verification.mpdem.test_double_layer_projection_nonuniform", str(tmp_path)],
        cwd=ROOT,
        env={**os.environ, "GEOTAICHI_REAL_DTYPE": "float64"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1800,
    )
    (tmp_path / "nonuniform_projection.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout[-6000:]


if __name__ == "__main__":
    run_probe(Path(sys.argv[1]))
