"""CPT cutbacks survive output clipping and checkpoint restart."""

import json

import pytest

from examples.igampm.cpt_dp.cpt_dp import restore_cpt_timestep_control, update_cpt_timestep_control


def test_cutback_hold_growth_and_restart():
    maximum = 0.001
    control = restore_cpt_timestep_control(None, maximum)

    def accept(dt, iterations=8, retries=0):
        update_cpt_timestep_control(control, {"accepted_timestep": dt, "retry_count": retries}, iterations, maximum)

    accept(0.00025, retries=2)
    accept(0.00025)
    accept(0.00025)
    saved = json.loads(json.dumps(control))
    control = restore_cpt_timestep_control(saved, maximum)
    accept(0.00001)  # Clipped at an output time, not a solver cutback.
    assert control == {"cap": 0.00025, "easy_steps": 2}
    accept(0.00025, iterations=20)
    assert control["easy_steps"] == 0
    for _ in range(3):
        accept(0.00025)
        assert control["cap"] == 0.00025
    accept(0.00025)
    assert control == {"cap": 0.0005, "easy_steps": 0}
    for _ in range(8):
        accept(control["cap"])
    assert control["cap"] == maximum
    assert restore_cpt_timestep_control(saved, 0.0001) == {"cap": 0.0001, "easy_steps": 0}


@pytest.mark.parametrize("cap,easy", [(0, 0), (float("nan"), 0), (0.001, 4), (0.001, 1.5), (0.001, True)])
def test_invalid_checkpoint_control(cap, easy):
    with pytest.raises(ValueError, match="controller checkpoint"):
        restore_cpt_timestep_control({"cap": cap, "easy_steps": easy}, 0.001)
