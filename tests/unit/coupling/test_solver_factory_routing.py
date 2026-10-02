import sys
import types

import src


class _Simulation:
    def __init__(self, direct):
        self.direct = bool(direct)
        self.coupling_calls = []

    def is_direct_backend(self):
        return self.direct

    def set_mpm_coupling(self, value):
        self.coupling_calls.append(value)


class _MPM:
    def __init__(self, direct):
        self.sims = _Simulation(direct)


def test_fempm_factory_binds_lagrangian_fields_only_for_native_mpm(monkeypatch):
    module = types.ModuleType("src.fempm.mainFEMPM")
    module.FEMPM = lambda fem, mpm, title, log: (fem, mpm, title, log)
    monkeypatch.setitem(sys.modules, "src.fempm.mainFEMPM", module)

    native = _MPM(direct=False)
    direct = _MPM(direct=True)
    fem = object()
    src.FEMPM(fem=fem, mpm=native, log=False)
    src.FEMPM(fem=fem, mpm=direct, log=False)

    assert native.sims.coupling_calls == ["Lagrangian"]
    assert direct.sims.coupling_calls == []
