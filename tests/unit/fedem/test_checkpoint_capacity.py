from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.fedem.Checkpoint import (
    _capture_object,
    _restore_array,
    _restore_object,
    _restore_scalar_state,
    _scalar_state,
)


class _ArrayField:
    def __init__(self, values):
        self.values = np.array(values, copy=True)

    def to_numpy(self):
        return np.array(self.values, copy=True)

    def from_numpy(self, values):
        self.values = np.array(values, copy=True)


class _ZeroInitializedCapacityField:
    def __init__(self, shape, components, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype
        numpy_dtype = np.float64 if dtype == ti.f64 else np.int32
        self.values = np.zeros(
            self.shape + tuple(components), dtype=numpy_dtype
        )

    def to_numpy(self):
        raise AssertionError("expanded contact restore must not stage GPU data")

    def from_numpy(self, values):
        self.values = np.array(values, copy=True)


class _ScalarField:
    def __init__(self, value=0.0):
        self.value = value

    def __getitem__(self, index):
        assert index is None
        return self.value

    def __setitem__(self, index, value):
        assert index is None
        self.value = value


def _coupling_scalar_fixture():
    fem_engine = SimpleNamespace(
        time=0.2,
        step_count=20,
        total_step=100,
        soft_particle_contact=None,
    )
    return SimpleNamespace(
        sims=SimpleNamespace(
            current_time=0.2,
            current_step=20,
            current_print=2,
            time=1.0,
            delta=0.01,
            dt=_ScalarField(0.01),
        ),
        fem=SimpleNamespace(engine=fem_engine),
        dem=SimpleNamespace(
            sims=SimpleNamespace(
                current_time=0.2,
                current_step=20,
                current_print=2,
                CurrentTime=_ScalarField(0.2),
            )
        ),
        contactor=SimpleNamespace(
            neighbor=SimpleNamespace(contact_count=3),
            initialized=True,
            wall_candidate_count=4,
        ),
        enginer=SimpleNamespace(minimum_jacobian=0.8),
        solver=None,
        removed_wall_contact_energy=1.25,
    )


def test_checkpoint_array_can_restore_into_larger_capacity_prefix():
    field = _ArrayField(np.full((5, 2), -7, dtype=np.int32))
    saved = np.arange(6, dtype=np.int32).reshape(3, 2)

    _restore_array(field, saved, "contact/candidates")

    np.testing.assert_array_equal(field.values[:3], saved)
    np.testing.assert_array_equal(field.values[3:], -7)


def test_expanded_soft_contact_restore_avoids_device_staging_copy():
    field = _ZeroInitializedCapacityField(
        shape=(5,), components=(2,), dtype=ti.i32
    )
    saved = np.arange(6, dtype=np.int32).reshape(3, 2)

    _restore_array(field, saved, "fem.soft_contact/ee_stencil")

    np.testing.assert_array_equal(field.values[:3], saved)
    np.testing.assert_array_equal(
        field.values[3:], np.zeros((2, 2), dtype=np.int32)
    )


def test_exact_shape_restore_avoids_device_staging_copy():
    field = _ZeroInitializedCapacityField(
        shape=(3,), components=(2,), dtype=ti.i32
    )
    saved = np.arange(6, dtype=np.int32).reshape(3, 2)

    _restore_array(field, saved, "fem.state/connectivity")

    np.testing.assert_array_equal(field.values, saved)


def test_checkpoint_array_rejects_capacity_shrink():
    field = _ArrayField(np.zeros((2, 2), dtype=np.float64))
    saved = np.zeros((3, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="checkpoint array"):
        _restore_array(field, saved, "contact/candidates")


def test_checkpoint_array_rejects_dtype_change_during_capacity_growth():
    field = _ArrayField(np.zeros(5, dtype=np.float64))
    saved = np.zeros(3, dtype=np.float32)

    with pytest.raises(ValueError, match="checkpoint array"):
        _restore_array(field, saved, "contact/history")


def test_checkpoint_does_not_capture_derived_element_hessian():
    arrays = {}
    manifest = {}
    assembler = SimpleNamespace(
        cell_hessian=_ArrayField(np.ones((8, 8, 3, 3))),
        cell_stress=_ArrayField(np.ones((8, 3, 3))),
    )

    _capture_object(arrays, manifest, "fem.assembler", assembler)

    assert arrays == {}
    assert manifest == {}


@pytest.mark.parametrize("allocate_hessian", [False, True])
def test_checkpoint_ignores_legacy_element_hessian(allocate_hessian):
    archive = {
        "state/fem.assembler/cell_hessian": np.ones(
            (8, 12, 12), dtype=np.float64
        )
    }
    manifest = {
        "fem.assembler/cell_hessian": {"kind": "field"}
    }
    assembler = SimpleNamespace(allocate_hessian=allocate_hessian)

    _restore_object(
        archive, manifest, "fem.assembler", assembler
    )


def test_checkpoint_round_trips_removed_wall_contact_energy():
    coupling = _coupling_scalar_fixture()

    state = _scalar_state(coupling)
    coupling.removed_wall_contact_energy = 0.0
    _restore_scalar_state(coupling, state)

    assert coupling.removed_wall_contact_energy == pytest.approx(1.25)


def test_legacy_checkpoint_defaults_removed_wall_contact_energy_to_zero():
    coupling = _coupling_scalar_fixture()
    state = _scalar_state(coupling)
    state.pop("coupling.removed_wall_contact_energy")

    _restore_scalar_state(coupling, state)

    assert coupling.removed_wall_contact_energy == pytest.approx(0.0)


def test_checkpoint_restores_all_fem_case_without_cross_neighbor():
    coupling = _coupling_scalar_fixture()
    coupling.contactor.neighbor = None
    state = _scalar_state(coupling)

    _restore_scalar_state(coupling, state)

    assert coupling.contactor.neighbor is None


def test_checkpoint_rejects_cross_contacts_without_cross_neighbor():
    coupling = _coupling_scalar_fixture()
    coupling.contactor.neighbor = None
    state = _scalar_state(coupling)
    state["cross.contact_count"] = 1

    with pytest.raises(ValueError, match="no FEM--LSDEM neighbor module"):
        _restore_scalar_state(coupling, state)
