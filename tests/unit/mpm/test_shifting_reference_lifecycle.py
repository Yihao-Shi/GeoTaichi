"""A geometric basis integral is built once, but refreshed for a new run."""

from types import SimpleNamespace
import importlib

import pytest


def test_reference_reused_until_precalculation_refresh(monkeypatch):
    module = importlib.import_module("src.mpm.engines.IncompressibleEngine")
    engine = object.__new__(module.IncompressibleEngine)
    engine.shifting_node_volume = engine.shifting_reference_volume = object()
    engine.shifting_reference_ready = False
    calls = []
    monkeypatch.setattr(module, "kernel_fdm_shifting_reference_volume", lambda *_: calls.append(1))
    monkeypatch.setattr(module, "kernel_volume_p2g_fdm_shifting_on_the_fly_2d", lambda *_: None)
    monkeypatch.setattr(module, "kernel_particle_shifting_delta_correction_fdm_on_the_fly_2d", lambda *_: None)
    element = SimpleNamespace(
        LnID=None,
        ghost_cell=1,
        gnum=None,
        grid_size=None,
        calLength=None,
        boundary_type=None,
        influenced_node=9,
        igrid_size=None,
        cnum=None,
        cell=SimpleNamespace(type=None),
    )
    scene = SimpleNamespace(element=element, particleNum=[0], particle=None)
    sims = SimpleNamespace(particle_shifting=True, dimension=2)
    engine.particle_shifting(sims, scene)
    engine.particle_shifting(sims, scene)
    assert calls == [1]

    # Stop at the existing first stage: invalidation must precede rebuilding
    # characteristic lengths, including when initialization subsequently fails.
    def stop_at_first_stage(*_):
        raise RuntimeError("initialization sentinel")

    engine.trace_precalculation_stage = stop_at_first_stage
    with pytest.raises(RuntimeError, match="initialization sentinel"):
        engine.pre_calculation(sims, scene, None)
    assert not engine.shifting_reference_ready
    engine.particle_shifting(sims, scene)
    assert calls == [1, 1]
