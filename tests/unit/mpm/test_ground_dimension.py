import pytest


@pytest.mark.parametrize("dimension", [2, 3])
def test_ground_default_velocity_uses_its_position_dimension(dimension, monkeypatch):
    import src.mpm.config as config
    from src.mpm.generator.Ground import Ground

    monkeypatch.setattr(config, "DIM", 5 - dimension)
    ground = Ground()
    ground.append([0.0] * dimension, [0.0] * (dimension - 1) + [1.0])
    assert ground.np_vel == [[0.0] * dimension]
