from types import SimpleNamespace

import numpy as np
import pytest

from src.mpm.Recorder import WriteFile


@pytest.mark.parametrize("track_energy", (False, True))
def test_soft_point_recorder_preserves_checkpoint_fields_and_body_energy(tmp_path, track_energy):
    def field(values):
        array = np.asarray(values)
        return SimpleNamespace(to_numpy=lambda: array)

    points = SimpleNamespace(
        active=field([1, 1]),
        bodyID=field([7, 7]),
        materialID=field([2, 2]),
        groupID=field([3, 3]),
        x=field([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]),
        x0=field([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        v=field([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        contact_force=field(np.zeros((2, 3))),
        external_force=field(np.zeros((2, 3))),
        m=field([2.0, 3.0]),
        vol0=field([1.0, 1.0]),
        F=field(np.tile(np.eye(3), (2, 1, 1))),
        stress=field(np.zeros((2, 3, 3))),
        strain_energy=field([0.2, 0.3]),
    )
    scene = SimpleNamespace(
        softPointNum=[2],
        soft_point=points,
        softNum=[1],
        soft=SimpleNamespace(bodyID=field([7]), damp_energy=field([0.125])),
    )
    simulation = SimpleNamespace(
        current_time=0.25,
        current_print=0,
        visualize=False,
        energy_tracking=track_energy,
        gravity=np.asarray([0.0, -9.81, 0.0]),
    )
    recorder = object.__new__(WriteFile)
    recorder.particle_path = str(tmp_path)
    recorder.monitor_soft_material_point(simulation, scene)

    with np.load(tmp_path / "LSMPMSoftPoint000000.npz") as output:
        assert output["t_current"] == 0.25
        np.testing.assert_array_equal(output["bodyID"], [7, 7])
        np.testing.assert_array_equal(output["materialID"], [2, 2])
        np.testing.assert_array_equal(output["position"], points.x.to_numpy())
        np.testing.assert_array_equal(output["F"], points.F.to_numpy())
        if track_energy:
            np.testing.assert_allclose(output["soft_kinetic_energy"], [7.0])
            np.testing.assert_allclose(output["soft_strain_energy"], [0.5])
            np.testing.assert_allclose(output["soft_gravitational_potential_energy"], [78.48])
            np.testing.assert_allclose(output["soft_local_damping_energy"], [0.125])
        else:
            assert "soft_kinetic_energy" not in output
