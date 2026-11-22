"""Control targets must use the attempted endpoint, including timestep retries."""

from types import SimpleNamespace

import pytest

from src.mpdem.mainDEMPM import DEMPM
from src.mpm.engines.direct.MPMSolver import MPMSolver
from src.utils.StepRetry import StepRetryPolicy
from src.utils.errors import LineSearchError


@pytest.mark.parametrize("adjust_timestep", [False, True])
def test_preintegration_control_runs_before_every_attempt(adjust_timestep):
    coupling = object.__new__(DEMPM)
    mpm = object.__new__(MPMSolver)
    mpm.time, mpm.step_count, mpm.dt = 0.0, 0, 0.2
    mpm.history = []
    mpm.step_retry = StepRetryPolicy(enabled=True, maximum_retries=1)
    mpm.step_schedule = SimpleNamespace(history_due=lambda *_args, **_kwargs: True)
    mpm.record = lambda: None
    attempts, commands, observations = [], [], []
    prepared = []

    def solve(*_args, **_kwargs):
        attempts.append(mpm.time + mpm.dt)
        if len(attempts) == 1:
            raise LineSearchError("residual Armijo line search failed")
        return {}

    system = SimpleNamespace(
        mpm=mpm,
        solve_lagged_equilibrium_device=solve,
        commit_step_device=lambda: None,
        mixed=SimpleNamespace(diagnostics=lambda: {"active_contacts": 0, "candidate_contacts": 0}),
    )
    coupling.enginer = system
    coupling._add_direct_mpm_affine_ipc_essentials = lambda **_kwargs: None
    coupling._direct_affine_output_started = True
    coupling.sims = SimpleNamespace(
        current_time=0.0, current_step=0, CurrentTime={None: 0.0}, time=0.3, delta=0.2, save_interval=1.0
    )
    coupling.mpm = SimpleNamespace(
        sims=SimpleNamespace(),
        enginer=SimpleNamespace(mpm=mpm, prepare_step_device=lambda: prepared.append(len(commands))),
    )
    coupling.dem = SimpleNamespace(
        sims=SimpleNamespace(), enginer=SimpleNamespace(_save=lambda *_args: None), scene=None
    )

    def observe(active):
        observations.append(active.mpm.time)
        if adjust_timestep:
            coupling.sims.delta = 0.1

    result = coupling.run_direct_mpm_affine_ipc(
        verbose=False,
        preintegration_function=lambda active: commands.append((active.mpm.time, active.mpm.dt)),
        postprocessing=observe,
    )
    expected_commands = (
        [(0.0, 0.2), (0.0, 0.1), (0.1, 0.1), (0.2, 0.1)] if adjust_timestep else [(0.0, 0.2), (0.0, 0.1), (0.1, 0.2)]
    )
    assert len(commands) == len(expected_commands)
    for command, expected in zip(commands, expected_commands):
        assert command == pytest.approx(expected)
    assert attempts == pytest.approx([0.2, 0.1, 0.2, 0.3] if adjust_timestep else [0.2, 0.1, 0.3])
    assert prepared == list(range(1, len(expected_commands) + 1))
    assert observations == pytest.approx([0.1, 0.2, 0.3] if adjust_timestep else [0.1, 0.3])
    assert result["converged"] and result["step"] == (3 if adjust_timestep else 2)
    assert result["time"] == pytest.approx(0.3)
