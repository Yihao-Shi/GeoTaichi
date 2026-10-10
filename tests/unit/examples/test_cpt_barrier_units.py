"""The IGA CPT barrier uses the same pressure units as FEM--MPM."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.unit, pytest.mark.cpu, pytest.mark.serial, pytest.mark.isolated_dimension(2)]


def test_quad_cpt_rejects_linear_checkpoint_before_runtime_initialization(tmp_path, monkeypatch):
    import geotaichi as gt
    from examples.igampm.cpt_dp import cpt_dp

    previous = tmp_path / "linear"
    previous.mkdir()
    (previous / "parameters.json").write_text(
        json.dumps(
            dict(
                contact="ipc",
                default_fp="float64",
                dilation_angle=0.0,
                resolution_scale=25.0,
                save_interval=0.001,
                shape_function="Linear",
            )
        )
    )
    arguments = SimpleNamespace(
        contact="ipc",
        arch="cpu",
        default_fp="float64",
        dt=0.001,
        time=0.003,
        save_interval=0.001,
        resolution_scale=25.0,
        dilation_angle=0.0,
        state_dependent=False,
        inexact_newton=True,
        resume=previous / "latest_state.npz",
        output_dir=str(tmp_path / "new_quad"),
    )
    monkeypatch.setattr(cpt_dp, "parse_arguments", lambda: arguments)
    monkeypatch.setattr(gt, "init", lambda **kwargs: pytest.fail("legacy interpolation must be rejected before init"))
    with pytest.raises(ValueError, match="shape_function; restart QuadBSpline from frame 0"):
        cpt_dp.main()
    assert not (tmp_path / "new_quad" / "parameters.json").exists()


@ti.kernel
def _terms(barrier: ti.template(), distance: ti.f64) -> ti.types.vector(3, ti.f64):
    energy, first, second = barrier._terms(distance)
    return ti.Vector([energy, first, second])


def test_iga_cpt_pressure_barrier_matches_fem_normalization(taichi_runtime, tmp_path, monkeypatch):
    import geotaichi as gt
    from examples.igampm.cpt_dp import cpt_dp
    from src.physics_model.contact_model.ipc.IPC import Barrier, ipc_barrier_distance_terms_py

    monkeypatch.setattr(gt, "init", lambda **kwargs: None)
    monkeypatch.setattr(
        cpt_dp,
        "parse_arguments",
        lambda: SimpleNamespace(
            contact="ipc",
            arch="cpu",
            default_fp="float64",
            dt=0.0005,
            time=0.0005,
            save_interval=0.0005,
            resolution_scale=25.0,
            dilation_angle=None,
            state_dependent=False,
            inexact_newton=False,
            resume=None,
            output_dir=str(tmp_path),
        ),
    )
    captured = {}

    class Configured(Exception):
        pass

    def capture_coupling(**kwargs):
        captured.update(kwargs)
        raise Configured

    monkeypatch.setattr(gt, "IGAMPM", capture_coupling)
    with pytest.raises(Configured):
        cpt_dp.main()

    barrier = Barrier(**captured)
    assert barrier.use_physical_barrier
    dhat, dmin, kappa = (float(captured[key]) for key in ("dhat", "dmin", "kappa"))
    gap2 = (2.0 * dmin + dhat) * dhat
    for distance in (dmin + 0.05 * dhat, dmin + 0.5 * dhat, dmin + dhat):
        # FEM's assembler explicitly uses kappa / gap2**2 and area * dhat.
        expected = (
            np.asarray(
                ipc_barrier_distance_terms_py(
                    distance,
                    dhat,
                    dmin,
                    kappa / gap2**2,
                )
            )
            * dhat
        )
        np.testing.assert_allclose(np.asarray(_terms(barrier, distance)), expected, rtol=1.0e-12, atol=1.0e-12)
