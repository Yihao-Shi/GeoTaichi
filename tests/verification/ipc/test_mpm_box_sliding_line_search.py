"""Deterministic box-sliding verification for IPC line search."""

import types

import pytest

import src.mpm.config as config

from src.mpm.generator.Body import Body
from src.mpm.generator.Ground import Ground
from src.mpm.soft_particle.IPCULMPM import IPCULMPM


pytestmark = [
    pytest.mark.verification,
    pytest.mark.ipc,
    pytest.mark.mpm,
    pytest.mark.cpu,
    pytest.mark.slow,
    pytest.mark.serial,
]


def install_line_search_probe(ipcmpm):
    records = []
    original_line_search = ipcmpm.ipc.line_search

    def probed_line_search(self, active_dof, verbose=False):
        success = original_line_search(active_dof, verbose)
        records.append(
            {
                "success": bool(success),
                "stop": bool(self.line_search_stop_iter),
                "g0": float(self.line_search_last_g0),
                "alpha0": float(self.line_search_last_alpha0),
                "alpha": float(self.line_search_last_alpha),
                "E0": float(self.line_search_last_previous_energy),
                "E_trial0": float(self.line_search_last_trial_energy),
                "E": float(self.line_search_last_accepted_energy),
                "backtracks": int(self.line_search_last_backtracks),
                "barrier_contacts": int(self.curr_barrier_contact_num),
                "friction_contacts": int(self.curr_friction_contact_num),
            }
        )
        return success

    ipcmpm.ipc.line_search = types.MethodType(probed_line_search, ipcmpm.ipc)
    return records


def build_case(output_path):
    dt = 1.0e-3
    gravity = [0.0, -6.929646456]
    domain = [5.0, 5.0]
    dx = 0.05
    fric = 1.0
    integration = [1.0, 0.5, 1.0]

    ground = Ground()
    ground.append([2.5, 0.1], [0.0, 1.0])

    body = Body()
    body.add_rectangle([1.0, 0.1 - 0.0115], [2.0, 0.6], dx, 2, init_v=[0.0, 0.0])

    return IPCULMPM(
        domain=domain,
        dx=dx,
        dt=dt,
        bodies=body,
        ground=ground,
        newmark=integration,
        young_modulus=1.0e12,
        poisson_ratio=0.2,
        density=1000.0,
        kappa=1.0e6,
        dhat=0.001,
        mu=0.0,
        epsv=0.001,
        residual=1.0e-6,
        friction_residual=1.0e-6,
        activate_friction=(fric != 0.0),
        gravity=gravity,
        step=1,
        interval=1,
        max_iters=8,
        line_search=True,
        shape_function="linear",
        visualize=False,
        path=str(output_path),
    )


def test_mpm_box_sliding_line_search_descends(taichi_runtime, tmp_path):
    config.set_dimension(2)
    ipcmpm = build_case(tmp_path / "box_sliding")
    records = install_line_search_probe(ipcmpm)
    ipcmpm.initial_simulation()

    for _ in range(3):
        ipcmpm.substep(verbose=False)

    ipcmpm.ipc.friction.friction = 1.0
    ipcmpm.mpm.damping = 0.0
    ipcmpm.mpm.gravity = [6.929646456, -6.929646456]

    for _ in range(3):
        ipcmpm.substep(verbose=False)

    tangential_force = ipcmpm.ipc.record_tangential_force()
    if abs(float(tangential_force[0][0])) <= 1.0e-8:
        raise AssertionError(f"recorded tangential contact force is zero: {tangential_force}")

    if not records:
        raise AssertionError("line search was not exercised")
    if any((not record["success"]) for record in records):
        raise AssertionError(f"line search failed: {records}")

    active_records = [record for record in records if not record["stop"]]
    if not active_records:
        raise AssertionError("line search did not accept any finite step")

    max_increase = max(record["E"] - record["E0"] for record in active_records)
    if max_increase > 1.0e-7:
        bad = max(active_records, key=lambda record: record["E"] - record["E0"])
        raise AssertionError(f"accepted line search energy increased: {bad}")
    print("box sliding line search records:")
    for idx, record in enumerate(records[:12]):
        print(
            f"{idx}: contacts=({record['barrier_contacts']},{record['friction_contacts']}), "
            f"success={record['success']}, stop={record['stop']}, "
            f"g0={record['g0']:.6e}, alpha0={record['alpha0']:.6e}, "
            f"alpha={record['alpha']:.6e}, backtracks={record['backtracks']}, "
            f"E0={record['E0']:.12e}, E_trial0={record['E_trial0']:.12e}, "
            f"E={record['E']:.12e}"
        )
    assert len(active_records) > 0
    assert max_increase <= 1.0e-7
