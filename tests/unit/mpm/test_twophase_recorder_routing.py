from types import SimpleNamespace

from src.mpm.Recorder import WriteFile
from src.utils.linalg import no_operation


def test_double_layer_keeps_two_phase_recorder_with_neighbor_detection():
    recorder = object.__new__(WriteFile)
    recorder.save_particle = no_operation
    recorder.save_grid = no_operation
    sims = SimpleNamespace(
        monitor_type=("particle",),
        visualize=False,
        dimension=2,
        material_type="TwoPhaseDoubleLayer",
        neighbor_detection=True,
        coupling=False,
        solver_type="SemiImplicit",
    )

    recorder.manage_function(sims)

    assert recorder.save_particle.__func__ is WriteFile.MonitorParticleTwoPhase
