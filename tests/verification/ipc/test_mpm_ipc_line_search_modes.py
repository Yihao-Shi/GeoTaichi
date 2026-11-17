"""Verification of IPC line-search behavior for MPM contact modes."""

import types

import pytest

import src.mpm.config as config
from src.mpm.generator.Body import Body
from src.mpm.generator.Ground import Ground
from src.mpm.soft_particle.IPCULMPM import IPCULMPM
from src.mpm.soft_particle.IPCTLMPM import IPCTLMPM


pytestmark = [
    pytest.mark.verification,
    pytest.mark.ipc,
    pytest.mark.mpm,
    pytest.mark.cpu,
    pytest.mark.slow,
    pytest.mark.serial,
]


def _install_probe(solver):
    records = []
    original_line_search = solver.ipc.line_search

    def probed_line_search(self, active_dof, verbose=False):
        success = original_line_search(active_dof, verbose)
        records.append(
            {
                "success": bool(success),
                "stop": bool(self.line_search_stop_iter),
                "stop_reason": str(self.line_search_stop_reason),
                "failed": bool(self.line_search_failed),
                "fallback": bool(self.line_search_used_fallback),
                "g0": float(self.line_search_last_g0),
                "alpha0": float(self.line_search_last_alpha0),
                "alpha": float(self.line_search_last_alpha),
                "stagnation_alpha": float(self.line_search_stagnation_alpha),
                "backtracks": int(self.line_search_last_backtracks),
                "E0": float(self.line_search_last_previous_energy),
                "E": float(self.line_search_last_accepted_energy),
                "barrier_contacts": int(self.curr_barrier_contact_num),
            }
        )
        return success

    solver.ipc.line_search = types.MethodType(probed_line_search, solver.ipc)
    return records


def _base_kwargs(output_path, dhat, gravity):
    return dict(
        domain=[0.35, 0.30],
        dx=0.05,
        dt=1.0e-4,
        newmark=[1.0, 0.5, 1.0],
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
        kappa=1.0e4,
        dhat=dhat,
        mu=0.0,
        epsv=1.0e-3,
        residual=1.0e-14,
        friction_residual=1.0e-14,
        activate_friction=False,
        gravity=gravity,
        step=1,
        interval=1,
        max_iters=8,
        line_search=True,
        line_search_work_tol=1.0e-8,
        line_search_energy_rtol=1.0e-12,
        line_search_energy_atol=1.0e-14,
        line_search_max_backtracks=25,
        shape_function="linear",
        visualize=False,
        scale=1.0,
        path=str(output_path),
    )


def _build_wall_case(solver_cls, output_path):
    ground = Ground()
    ground.append([0.0, 0.1], [0.0, 1.0])

    body = Body()
    body.add_rectangle([0.10, 0.0755], [0.20, 0.1755], 0.05, ppc=1, init_v=[0.0, 0.0])
    return solver_cls(
        bodies=body,
        ground=ground,
        **_base_kwargs(output_path, dhat=2.0e-3, gravity=[0.0, -9.8]),
    )


def _build_particle_case(solver_cls, output_path):
    ground = Ground()
    ground.append([0.0, -1.0], [0.0, 1.0])

    body = Body()
    body.add_rectangle([0.100, 0.100], [0.151, 0.151], 0.05, ppc=1, init_v=[0.0, 0.0])
    body.add_rectangle([0.151, 0.100], [0.202, 0.151], 0.05, ppc=1, init_v=[0.0, 0.0])
    return solver_cls(
        bodies=body,
        ground=ground,
        **_base_kwargs(output_path, dhat=6.0e-2, gravity=[0.0, 0.0]),
    )


def _check_records(label, records):
    if not records:
        raise AssertionError(f"{label}: line search was not exercised")
    if not any(record["barrier_contacts"] > 0 for record in records):
        raise AssertionError(f"{label}: no IPC barrier contacts were found")
    if any(record["failed"] or not record["success"] for record in records):
        raise AssertionError(f"{label}: line search failed: {records}")

    active_records = [
        record for record in records
        if not record["stop"] or record["stop_reason"] == "stagnation"
    ]
    if not active_records:
        raise AssertionError(f"{label}: no finite line-search step was accepted")
    max_increase = max(record["E"] - record["E0"] for record in active_records)
    if max_increase > 1.0e-9:
        bad = max(active_records, key=lambda record: record["E"] - record["E0"])
        raise AssertionError(f"{label}: accepted energy increased: {bad}")
    for record in records:
        if record["stop"] and record["stop_reason"] == "tiny_work":
            scale = max(1.0, abs(record["E0"]))
            if abs(record["g0"]) > 1.1e-8 * scale:
                raise AssertionError(f"{label}: invalid tiny-work stop: {record}")
        if record["stop"] and record["stop_reason"] == "stagnation":
            if record["alpha"] > 1.1 * record["stagnation_alpha"]:
                raise AssertionError(f"{label}: invalid stagnation stop: {record}")
    print(
        f"{label}: records={len(records)}, contacts=max({max(r['barrier_contacts'] for r in records)}), "
        f"stops={sum(1 for r in records if r['stop'])}, "
        f"stop_reasons={[r['stop_reason'] for r in records if r['stop']]}, "
        f"fallbacks={sum(1 for r in records if r['fallback'])}, max_energy_increase={max_increase:.3e}"
    )


def _run_case(solver_cls, builder, label, output_path):
    solver = builder(solver_cls, output_path)
    records = _install_probe(solver)
    solver.initial_simulation()
    solver.substep(verbose=False)
    _check_records(label, records)


def test_mpm_ipc_line_search_accepts_wall_and_particle_contact_modes(
    taichi_runtime,
    tmp_path,
):
    config.set_dimension(2)
    _run_case(
        IPCULMPM,
        _build_wall_case,
        "ul_wall_ipc",
        tmp_path / "ul_wall",
    )
    _run_case(
        IPCTLMPM,
        _build_wall_case,
        "tl_wall_ipc",
        tmp_path / "tl_wall",
    )
    _run_case(
        IPCULMPM,
        _build_particle_case,
        "ul_particle_ipc",
        tmp_path / "ul_particle",
    )
    _run_case(
        IPCTLMPM,
        _build_particle_case,
        "tl_particle_ipc",
        tmp_path / "tl_particle",
    )
