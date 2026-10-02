import json

import numpy as np
import pytest

from research.llm_assist.benchmarks import run_mixed_funnel as funnel


class _ArrayMember:
    def __init__(self, values):
        self.values = np.asarray(values).copy()

    def to_numpy(self):
        return self.values.copy()

    def from_numpy(self, values):
        self.values = np.asarray(values).copy()


class _StructField:
    def __init__(self, **members):
        self._members = {
            name: _ArrayMember(values) for name, values in members.items()
        }
        for name, member in self._members.items():
            setattr(self, name, member)

    def to_numpy(self):
        return {
            name: member.to_numpy()
            for name, member in self._members.items()
        }


def test_restart_grid_mapping_removes_accumulated_time_drift():
    grid_time, remapped_step = funnel._restart_grid_mapping(
        checkpoint_step=950_000,
        checkpoint_dt=1.0e-6,
        stored_time=0.9500000000066242,
        target_dt=5.0e-7,
    )

    assert grid_time == pytest.approx(0.95, abs=1.0e-15)
    assert remapped_step == 1_900_000


def test_restart_grid_mapping_rejects_an_unrepresentable_target_grid():
    with pytest.raises(ValueError, match="not representable"):
        funnel._restart_grid_mapping(
            checkpoint_step=1,
            checkpoint_dt=1.0e-6,
            stored_time=1.0e-6,
            target_dt=3.0e-7,
        )


def test_restart_parameter_snapshot_excludes_contact_work_ledgers():
    field = _StructField(
        kn=[100.0],
        mus=[0.3],
        elastic_energy=[4.0],
        friction_energy=[-2.0],
        damp_energy=[-1.0],
    )

    snapshot = funnel._struct_parameter_snapshot(field)

    assert set(snapshot) == {"kn", "mus"}


def test_restart_restores_dissipative_parameters_without_rewriting_energy():
    field = _StructField(
        kn=[100.0],
        mus=[0.3],
        ndratio=[0.2],
        friction_energy=[-2.0],
    )
    requested = {
        "kn": np.asarray([100.0]),
        "mus": np.asarray([0.0]),
        "ndratio": np.asarray([0.0]),
    }

    changed, energy_sensitive = funnel._restore_struct_parameters(
        field, requested
    )

    assert changed is True
    assert energy_sensitive is False
    np.testing.assert_allclose(field.mus.to_numpy(), [0.0])
    np.testing.assert_allclose(field.ndratio.to_numpy(), [0.0])
    np.testing.assert_allclose(field.friction_energy.to_numpy(), [-2.0])


def test_restart_reports_contact_stiffness_as_energy_sensitive():
    field = _StructField(kn=[100.0], ks=[50.0])

    changed, energy_sensitive = funnel._restore_struct_parameters(
        field,
        {"kn": np.asarray([200.0]), "ks": np.asarray([50.0])},
    )

    assert changed is True
    assert energy_sensitive is True


def test_restart_native_frame_audit_counts_only_files_in_new_output(tmp_path):
    checkpoint_root = tmp_path / "native" / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    for frame, step in ((19, 900_000), (20, 925_000)):
        metadata = {
            "scalar_state": {"coupling.current_step": step},
        }
        np.savez(
            checkpoint_root / f"FEDEMCheckpoint{frame:06d}.npz",
            metadata_json=json.dumps(metadata),
        )

    assert funnel._existing_native_checkpoint_steps(checkpoint_root) == [
        900_000,
        925_000,
    ]
    assert funnel._existing_native_checkpoint_steps(
        tmp_path / "separate_restart_output" / "native" / "checkpoints"
    ) == []


def test_funnel_has_four_slopes_chute_and_separate_closed_catcher():
    walls = funnel._facet_wall_specs()
    names = [name for name, _, _ in walls]

    assert len(walls) == 13
    assert names[:4] == [
        "left_hopper",
        "right_hopper",
        "front_hopper",
        "back_hopper",
    ]
    assert set(names[4:8]) == {
        "left_chute",
        "right_chute",
        "front_chute",
        "back_chute",
    }
    assert set(names[8:]) == {
        "catcher_floor",
        "catcher_left",
        "catcher_right",
        "catcher_front",
        "catcher_back",
    }

    interior = np.asarray([0.30, 0.20, 0.50])
    for name, vertices, normal in walls:
        vertices = np.asarray(vertices, dtype=np.float64)
        normal = np.asarray(normal, dtype=np.float64)
        assert vertices.shape == (4, 3)
        np.testing.assert_allclose(np.linalg.norm(normal), 1.0)
        np.testing.assert_allclose(
            (vertices - vertices[0]) @ normal,
            0.0,
            atol=1.0e-12,
        )
        if name.endswith("_hopper"):
            assert float((interior - vertices[0]) @ normal) > 0.0

    upper_vertices = np.vstack(
        [
            np.asarray(vertices)
            for name, vertices, _ in walls
            if name.endswith("_hopper") or name.endswith("_chute")
        ]
    )
    catcher_vertices = np.vstack(
        [
            np.asarray(vertices)
            for name, vertices, _ in walls
            if name.startswith("catcher_")
        ]
    )
    separation = np.linalg.norm(
        upper_vertices[:, None, :] - catcher_vertices[None, :, :], axis=2
    )
    assert float(np.min(separation)) > 0.0


def test_funnel_800_particle_lattice_clears_all_four_slopes():
    centers, radii = funnel._initial_packing(
        np.random.default_rng(funnel.SEED)
    )
    for name, vertices, normal in funnel._facet_wall_specs()[:4]:
        vertices = np.asarray(vertices, dtype=np.float64)
        normal = np.asarray(normal, dtype=np.float64)
        gap = (centers - vertices[0]) @ normal - radii
        assert float(np.min(gap)) > funnel.SOFT_CONTACT_THICKNESS, name


def test_sparse_validation_column_interleaves_both_phases(monkeypatch):
    monkeypatch.setattr(funnel, "PARTICLE_COUNT", 8)
    monkeypatch.setattr(funnel, "SOFT_COUNT", 4)
    monkeypatch.setattr(funnel, "RIGID_COUNT", 4)

    centers, radii = funnel._initial_packing(
        np.random.default_rng(funnel.SEED), packing="validation"
    )

    assert centers.shape == (8, 3)
    assert radii.shape == (8,)
    np.testing.assert_allclose(
        np.sort(centers[:4, 2]),
        [0.43, 0.52, 0.61, 0.70],
    )
    np.testing.assert_allclose(
        np.sort(centers[4:, 2]),
        [0.475, 0.565, 0.655, 0.745],
    )
    assert np.all(centers[:, 0] - radii > 0.23)
    assert np.all(centers[:, 0] + radii < 0.37)
    assert np.all(centers[:, 1] - radii > 0.12)
    assert np.all(centers[:, 1] + radii < 0.28)
    pair_distance = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=2
    )
    pair_clearance = pair_distance - radii[:, None] - radii[None, :]
    np.fill_diagonal(pair_clearance, np.inf)
    assert float(np.min(pair_clearance)) > funnel.SOFT_CONTACT_THICKNESS


def test_funnel_outlet_and_catcher_elevations():
    walls = {
        name: np.asarray(vertices, dtype=np.float64)
        for name, vertices, _ in funnel._facet_wall_specs()
    }
    for name in ("left_chute", "right_chute", "front_chute", "back_chute"):
        assert float(np.min(walls[name][:, 2])) == funnel.CHUTE_BOTTOM_Z
        assert float(np.max(walls[name][:, 2])) == funnel.OUTLET_Z
    for name in (
        "catcher_left",
        "catcher_right",
        "catcher_front",
        "catcher_back",
    ):
        assert float(np.max(walls[name][:, 2])) == funnel.CATCHER_TOP_Z
    assert funnel.CATCHER_TOP_Z > funnel.OUTLET_Z


def test_formal_catcher_is_centered_and_smaller_than_the_hopper():
    walls = {
        name: np.asarray(vertices, dtype=np.float64)
        for name, vertices, _ in funnel._facet_wall_specs(
            catcher_width_x=funnel.PRODUCTION_CATCHER_WIDTH_X,
            catcher_width_y=funnel.PRODUCTION_CATCHER_WIDTH_Y,
        )
    }
    floor = walls["catcher_floor"]
    width_x = float(np.ptp(floor[:, 0]))
    width_y = float(np.ptp(floor[:, 1]))

    assert width_x == pytest.approx(funnel.PRODUCTION_CATCHER_WIDTH_X)
    assert width_y == pytest.approx(funnel.PRODUCTION_CATCHER_WIDTH_Y)
    assert float(np.mean(floor[:, 0])) == pytest.approx(0.30)
    assert float(np.mean(floor[:, 1])) == pytest.approx(0.20)
    assert width_x * width_y < 0.60 * 0.40
    assert width_x > 0.14
    assert width_y > 0.16


def test_optional_outlet_gate_exactly_covers_chute_opening():
    default_walls = funnel._facet_wall_specs()
    gated_walls = funnel._facet_wall_specs(include_gate=True)
    assert len(default_walls) == 13
    assert len(gated_walls) == 14
    name, vertices, normal = gated_walls[-1]
    vertices = np.asarray(vertices, dtype=np.float64)
    assert name == "outlet_gate"
    assert np.allclose(vertices.min(axis=0), [0.23, 0.12, funnel.OUTLET_Z])
    assert np.allclose(vertices.max(axis=0), [0.37, 0.28, funnel.OUTLET_Z])
    assert np.allclose(normal, [0.0, 0.0, 1.0])
