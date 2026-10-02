"""Regression coverage for import-time Taichi dimension isolation."""

import pytest


pytestmark = [pytest.mark.unit, pytest.mark.cpu, pytest.mark.serial]


@pytest.mark.isolated_dimension(3)
def test_isolated_child_sets_every_geotaichi_dimension_source_before_import():
    import src.iga.config as iga_config
    import src.igampm.config as igampm_config
    import src.mpm.config as mpm_config
    import src.utils.GlobalVariable as global_variable

    assert global_variable.DIMENSION == 3
    assert iga_config.get_dimension() == 3
    assert mpm_config.get_dimension() == 3
    assert igampm_config.get_dimension() == 3
