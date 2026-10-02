"""Parity checks against the bundled frozen-friction fixture.

The 446 cube/cube cases are read directly from
``tests/data/ipc_toolkit`` without extracting files into a machine-local
checkout.
"""

import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import tarfile

import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.unit, pytest.mark.ipc, pytest.mark.contact]

from src.physics_model.contact_model.ipc.IPC import (
    ipc_friction_f0,
    ipc_friction_f1_over_speed,
    ipc_friction_hessian_term,
)

_CASE_COUNT = 446
_ARCHIVE_SHA256 = "1b3ddd33327957d759564a3aeb3e44ea80e3e4cead49e12dc534761e80570d38"
_CONTENTS_SHA256 = "72f21aeba625e7d2fb083d5edd726b20abfa25bd325479cbea3d0d1477e8c73b"
_BUNDLED_DATA_ROOT = Path(__file__).resolve().parents[4] / "data" / "ipc_toolkit"
_DATA_ROOT = Path(os.environ.get("IPC_TOOLKIT_TEST_DATA", _BUNDLED_DATA_ROOT)).expanduser()
_ARCHIVE_PATH = _DATA_ROOT / "friction" / "cube_cube.tar.gz"
_CASE_NAME = re.compile(r"friction_data_([0-9]+)\.json")


@pytest.fixture(scope="module", autouse=True)
def _initialize_taichi():
    ti.init(
        arch=ti.cpu,
        cpu_max_num_threads=2,
        offline_cache=False,
        default_fp=ti.f64,
        debug=False,
    )


@ti.kernel
def _friction_scalar_terms(speed: ti.f64, epsv: ti.f64) -> ti.types.vector(3, ti.f64):
    return ti.Vector(
        [
            ipc_friction_f0(speed, epsv, 1.0),
            ipc_friction_f1_over_speed(speed, epsv),
            ipc_friction_hessian_term(speed, epsv),
        ]
    )


def _reference_data_cases():
    if not _ARCHIVE_PATH.is_file():
        pytest.fail(f"bundled ipc-toolkit data is missing: {_ARCHIVE_PATH}")

    archive_bytes = _ARCHIVE_PATH.read_bytes()
    archive_digest = hashlib.sha256(archive_bytes).hexdigest()
    if archive_digest != _ARCHIVE_SHA256:
        pytest.fail("bundled ipc-toolkit archive checksum mismatch: " f"{archive_digest} != {_ARCHIVE_SHA256}")

    payloads = {}
    with tarfile.open(_ARCHIVE_PATH, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = PurePosixPath(member.name).name
            match = _CASE_NAME.fullmatch(name)
            if match is None:
                pytest.fail(f"unexpected ipc-toolkit archive member: {member.name}")
            case_id = int(match.group(1))
            if case_id in payloads:
                pytest.fail(f"duplicate ipc-toolkit friction case: {case_id}")
            stream = archive.extractfile(member)
            if stream is None:
                pytest.fail(f"cannot read ipc-toolkit archive member: {member.name}")
            payloads[case_id] = (name, stream.read())

    expected_ids = set(range(_CASE_COUNT))
    actual_ids = set(payloads)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        pytest.fail("ipc-toolkit friction case IDs are not continuous 0..445: " f"missing={missing}, extra={extra}")

    contents_digest = hashlib.sha256()
    cases = []
    for case_id in range(_CASE_COUNT):
        name, payload = payloads[case_id]
        contents_digest.update(name.encode("utf-8"))
        contents_digest.update(b"\0")
        contents_digest.update(payload)
        contents_digest.update(b"\0")
        cases.append((name, json.loads(payload)))
    if contents_digest.hexdigest() != _CONTENTS_SHA256:
        pytest.fail(
            "bundled ipc-toolkit payload checksum mismatch: " f"{contents_digest.hexdigest()} != {_CONTENTS_SHA256}"
        )
    return cases


def _collision_stencil(mmcvid, closest_point):
    """Return fixture vertex ordering and relative-velocity coefficients."""
    m0, m1, m2, m3 = (int(value) for value in mmcvid)
    beta0, beta1 = (float(value) for value in closest_point)
    if m0 >= 0:  # edge-edge
        return [m0, m1, m2, m3], [1.0 - beta0, beta0, beta1 - 1.0, -beta1]

    point = -m0 - 1
    if m2 < 0:  # vertex-vertex
        return [point, m1], [1.0, -1.0]
    if m3 < 0:  # edge-vertex, stored point first in the fixture
        return [point, m1, m2], [1.0, beta0 - 1.0, -beta0]
    # face-vertex, stored point first in the fixture
    return [point, m1, m2, m3], [1.0, beta0 + beta1 - 1.0, -beta0, -beta1]


def _assemble_frozen_friction(data):
    start = np.asarray(data["V_start"], dtype=np.float64)
    end = np.asarray(data["V_end"], dtype=np.float64)
    velocity = end - start
    dim = start.shape[1]
    ndof = start.size
    epsv = float(data["epsv_times_h_squared"]) ** 0.5
    mu = float(data["mu"])
    mmcvids = np.asarray(data["mmcvids"], dtype=np.int64)
    closest_points = np.asarray(data["closest_point_coordinates"], dtype=np.float64)
    tangent_bases = np.asarray(data["tangent_bases"], dtype=np.float64)
    normal_forces = np.asarray(data["normal_force_magnitudes"], dtype=np.float64)

    energy = 0.0
    gradient = np.zeros(ndof, dtype=np.float64)
    hessian = np.zeros((ndof, ndof), dtype=np.float64)
    identity = np.eye(dim, dtype=np.float64)

    for contact, mmcvid in enumerate(mmcvids):
        vertex_ids, coefficients = _collision_stencil(mmcvid, closest_points[contact])
        gamma = np.hstack([coefficient * identity for coefficient in coefficients])
        local_velocity = velocity[vertex_ids].reshape(-1)
        tangent_basis = tangent_bases[dim * contact : dim * (contact + 1)]
        tangent_map = gamma.T @ tangent_basis
        tangential_velocity = tangent_map.T @ local_velocity
        speed = float(np.linalg.norm(tangential_velocity))
        f0, f1_over_speed, radial_derivative = np.asarray(_friction_scalar_terms(speed, epsv))
        scale = mu * normal_forces[contact]

        energy += scale * f0
        local_gradient = scale * f1_over_speed * tangent_map @ tangential_velocity
        inner = f1_over_speed * np.eye(tangent_basis.shape[1])
        if speed > 0.0:
            inner += (radial_derivative / speed) * np.outer(tangential_velocity, tangential_velocity)
        local_hessian = scale * tangent_map @ inner @ tangent_map.T

        local_dofs = np.concatenate([np.arange(vertex * dim, (vertex + 1) * dim) for vertex in vertex_ids])
        np.add.at(gradient, local_dofs, local_gradient)
        hessian[np.ix_(local_dofs, local_dofs)] += local_hessian

    return energy, gradient, hessian


def _stored_hessian(data):
    ndof = len(data["gradient"])
    hessian = np.zeros((ndof, ndof), dtype=np.float64)
    for row, column, value in data["hessian_triplets"]:
        hessian[int(row), int(column)] += float(value)
    return hessian


def test_all_ipc_toolkit_cube_cube_friction_data():
    for name, data in _reference_data_cases():
        energy, gradient, hessian = _assemble_frozen_friction(data)
        expected_gradient = np.asarray(data["gradient"], dtype=np.float64)
        expected_hessian = _stored_hessian(data)

        np.testing.assert_allclose(
            energy,
            float(data["energy"]),
            rtol=2.0e-11,
            atol=2.0e-12,
            err_msg=name,
        )
        np.testing.assert_allclose(
            gradient,
            expected_gradient,
            rtol=2.0e-10,
            atol=2.0e-10,
            err_msg=name,
        )
        np.testing.assert_allclose(
            hessian,
            expected_hessian,
            rtol=5.0e-10,
            atol=5.0e-9,
            err_msg=name,
        )
