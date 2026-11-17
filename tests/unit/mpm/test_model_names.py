"""Public contact model selection uses exact canonical names."""

import pytest

from src.mpm.mainMPM import MPM

pytestmark = [pytest.mark.unit, pytest.mark.mpm, pytest.mark.cpu]


def test_barrier_ipc_name_selects_direct_contact(taichi_runtime):
    model = MPM(log=False)
    model.add_contact("BarrierIPC")
    assert model.sims.ipc_contact


@pytest.mark.parametrize("name", ["IPC", "ipc", "IPCContact", "barrieripc", "Barrier-IPC", "typo"])
def test_aliases_and_misspellings_fail_before_changing_contact(taichi_runtime, name):
    model = MPM(log=False)
    with pytest.raises(ValueError, match="BarrierIPC"):
        model.add_contact(name)
    assert not model.sims.ipc_contact
