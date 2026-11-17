from src.mpm.Simulation import Simulation


def make_soft_options():
    sims = Simulation.__new__(Simulation)
    sims.initialize_soft_particle_options()
    return sims


def test_soft_levelset_transport_is_enabled_by_default():
    sims = make_soft_options()

    assert sims.soft_levelset_transport is True


def test_soft_levelset_transport_can_be_disabled_independently():
    sims = make_soft_options()

    sims.set_soft_levelset_reinitialization(
        transport_enabled=False,
        enabled=False,
        volume_correction=False,
        domain_check=False,
    )

    assert sims.soft_levelset_transport is False
    assert sims.soft_levelset_reinitialization is False
    assert sims.soft_levelset_volume_correction is False
    assert sims.soft_levelset_domain_check is False
