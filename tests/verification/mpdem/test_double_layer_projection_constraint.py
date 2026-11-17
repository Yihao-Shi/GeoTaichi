"""The actual solid-node/fluid-face correction must satisfy mixture continuity."""

from pathlib import Path
import json
import os
import runpy
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]


def check_projection(dimension, output):
    import geotaichi as gt

    original_init = gt.init
    rows, context = [], {}

    def initialize(*args, **kwargs):
        result = original_init(*args, **kwargs)
        import importlib
        import numpy as np
        import taichi as ti

        module = importlib.import_module("src.mpm.engines.ULSemiImplicitTwoPhaseDoubleLayerEngine")
        cls = module.ULSemiImplicitTwoPhaseDoubleLayerEngine
        update_name = f"_semi_implicit_update_{dimension}d"
        solve_name = f"_solve_pressure{dimension}d"
        g2p_name = f"kernel_g2p_double_layer{dimension}d"
        original_update, original_solve = getattr(cls, update_name), getattr(cls, solve_name)
        original_g2p = getattr(module, g2p_name)

        if hasattr(cls, "_project_coupled_pressure"):
            original_projection = cls._project_coupled_pressure

            def projection(self, sims, scene, dim):
                context["predictor_norm"] = self._actual_pressure_residual_squared(scene, dim) ** 0.5
                return original_projection(self, sims, scene, dim)

            cls._project_coupled_pressure = projection

        def update(self, sims, scene, *args, **kwargs):
            context.update(engine=self, sims=sims, scene=scene)
            return original_update(self, sims, scene, *args, **kwargs)

        def solve(self, sims, scene, **kwargs):
            result = original_solve(self, sims, scene, **kwargs)
            context.setdefault("predictor_norm", float(np.linalg.norm(self.poisson_solver.b.to_numpy())))
            return result

        def g2p(*args, **kwargs):
            e, sims, scene = (context[k] for k in ("engine", "sims", "scene"))
            if context.get("step") != sims.current_time:
                context["step"] = sims.current_time
                axes = "xyz"[:dimension]
                if "solid" not in context:
                    context["solid"] = tuple(
                        ti.field(ti.f64, shape=getattr(e, f"solid_velocity_{a}").shape) for a in axes
                    )
                    context["residual"] = ti.field(ti.f64, shape=e.cell_type.shape)
                solid, residual = context["solid"], context["residual"]
                for f, a in zip(solid, axes):
                    f.copy_from(getattr(e, f"solid_velocity_{a}"))
                project = getattr(module, f"kernel_project_solid_grid_velocity_to_mac{dimension}d")
                project(
                    scene.mass_cut_off,
                    scene.element.grid_size,
                    scene.element.gnum,
                    e.mac_shape_type,
                    e.mac_influenced_node,
                    scene.node,
                    scene.element.calLength,
                    e.cell_type,
                    *(getattr(e, f"solid_mass_{a}") for a in axes),
                    *solid,
                )
                rhs = getattr(module, "kernel_assemble_double_layer_pressure_rhs" + ("3d" if dimension == 3 else ""))
                rhs(
                    scene.element.grid_size,
                    e.cell_type,
                    *(getattr(e, f"face_porosity_{a}") for a in axes),
                    *(getattr(e, f"fluid_velocity_{a}") for a in axes),
                    *solid,
                    residual,
                )
                norm = float(np.linalg.norm(residual.to_numpy()))
                scale = context.pop("predictor_norm")
                rows.append(
                    {
                        "step_time": sims.current_time,
                        "predicted_norm": scale,
                        "actual_norm": norm,
                        "relative": norm / max(scale, 1e-30),
                    }
                )
            return original_g2p(*args, **kwargs)

        setattr(cls, update_name, update)
        setattr(cls, solve_name, solve)
        setattr(module, g2p_name, g2p)
        return result

    gt.init = initialize
    helper = runpy.run_path(str(Path(__file__).with_name("test_two_phase_checkpoint_continuation.py")))
    helper["run_stage"](dimension, output, 3, "")
    (output / "actual_projection_residual.json").write_text(json.dumps(rows, indent=2))
    assert len(rows) == 3
    assert all(r["actual_norm"] <= max(1e-10, 1e-5 * r["predicted_norm"]) for r in rows), rows


@pytest.mark.parametrize("dimension", [2, 3])
def test_actual_mixture_projection(tmp_path, dimension):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.verification.mpdem.test_double_layer_projection_constraint",
            str(dimension),
            str(tmp_path),
        ],
        cwd=ROOT,
        env={**os.environ, "GEOTAICHI_REAL_DTYPE": "float64"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1800,
    )
    (tmp_path / "projection.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout[-6000:]


if __name__ == "__main__":
    check_projection(int(sys.argv[1]), Path(sys.argv[2]))
