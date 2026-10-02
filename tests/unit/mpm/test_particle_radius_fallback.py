from types import SimpleNamespace

import numpy as np

from src.mpm.SceneManager import myScene


def test_particle_radius_falls_back_to_circumscribed_psize():
    scene = myScene()
    scene.particle = SimpleNamespace()
    scene.particleNum[0] = 2
    scene.psize = np.asarray([[0.1, 0.2], [0.3, 0.4]])

    assert np.isclose(scene.find_particle_min_radius(), np.sqrt(0.05))
    assert np.isclose(scene.find_particle_max_radius(), 0.5)
