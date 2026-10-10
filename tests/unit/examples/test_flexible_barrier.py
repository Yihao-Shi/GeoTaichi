"""The barrier's 2D unit-depth section and 3D slab preserve declared soil mass."""

from types import SimpleNamespace
import json

import numpy as np
import pytest

from examples.igampm.flexible_barrier import flexible_barrier as case
from src.mpm.generator.Body import Body


@pytest.mark.parametrize("dimension", [2, 3])
def test_barrier_particle_geometry_and_quadratic_basis(dimension, tmp_path):
    p = case.parameters(
        SimpleNamespace(
            dimension=dimension,
            thickness=1.0,
            spacing=0.5,
            elastic_spacing=0.5,
            dt=0.001,
            time=0.002,
            save_interval=0.001,
            ppc=2,
            contact_capacity=256,
            dilation_angle=0.0,
            linear_solver="PCG",
        )
    )
    body = Body()
    configured = {}
    mpm = SimpleNamespace(
        create_body=lambda: body,
        set_configuration=lambda **values: configured.update(values),
        add_body=lambda value: None,
        add_material=lambda **values: None,
        add_element=lambda values: configured.update(element=values),
        add_boundary_condition=lambda **values: None,
        set_solver=lambda values: None,
    )
    case.configure_mpm(mpm, p, tmp_path)
    data = body.bodies["dp_soil"]
    points = data["points"]
    assert configured["dimension"] == dimension and points.shape == (p["particle_count"], dimension)
    assert configured["element"]["ShapeFunction"] == "QuadBSpline"
    np.testing.assert_allclose(p["density"] * data["volume"] * len(points), p["soil_mass"])
    assert p["soil_mass_unit"] == ("kg/m" if dimension == 2 else "kg")
    np.testing.assert_array_less(p["soil_origin"], points.min(axis=0))
    np.testing.assert_array_less(points.max(axis=0), np.array(p["soil_origin"]) + p["soil_size"])
    assert p["plane_strain"] == (dimension == 2)
    assert p["gravity"] == [0.0] * (dimension - 1) + [-9.8]


@pytest.mark.parametrize("invalid", [None, "deformation", "frames", "diagnostics"])
def test_barrier_resume_validates_state_and_preserves_frame_numbering(tmp_path, monkeypatch, invalid):
    class Field:
        def __init__(self, value):
            self.value = value.copy()

        def to_numpy(self):
            return self.value.copy()

        def from_numpy(self, value):
            self.value = value.copy()

    data = {
        "deformation_gradient": np.eye(3)[None],
        "plastic_inverse": np.eye(3)[None],
        "particle_mass": np.array([6.25]),
        "grid_mass": np.array([1.0, 0.0]),
    }
    state = {name: Field(np.zeros_like(value)) for name, value in data.items()}
    monkeypatch.setattr(case, "state_fields", lambda engine: state)
    if invalid == "deformation":
        data["deformation_gradient"][0, 0, 0] = -1.0
    np.savez(tmp_path / "latest_state.npz", metadata=np.array(json.dumps({"time": 0.138375, "step": 157})), **data)
    vtks = tmp_path / "vtks"
    vtks.mkdir()
    if invalid == "diagnostics":
        (tmp_path / "step_diagnostics.jsonl").write_text(json.dumps({"step": 158}) + "\n")
    for prefix in ("GraphicMPMParticle", "NurbsVolume"):
        for index in range(2):
            if invalid != "frames" or prefix != "NurbsVolume" or index != 1:
                (vtks / f"{prefix}{index:06d}.vtu").touch()
    engine = SimpleNamespace(iga=SimpleNamespace(), mpm=SimpleNamespace())
    model = SimpleNamespace(mpm=SimpleNamespace(sims=SimpleNamespace()))
    if invalid:
        with pytest.raises(ValueError):
            case.restore_checkpoint(model, engine, tmp_path / "latest_state.npz", 3.0)
        for field in state.values():
            assert np.count_nonzero(field.value) == 0
    else:
        case.restore_checkpoint(model, engine, tmp_path / "latest_state.npz", 3.0)
        assert engine.time == 0.138375 and engine.implicit_step_index == 157
        assert engine.iga.output_count == engine.mpm.output_count == model.mpm.sims.current_print == 2
        assert model._last_implicit_recorded_step == 157
        for name, field in state.items():
            np.testing.assert_array_equal(field.value, data[name])
