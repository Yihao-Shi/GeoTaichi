import pytest
from types import SimpleNamespace

from examples.cfdem.FullyResolved.IBMResolvedSphereSettling.sphere_settling import (
    case_parameters,
    grid_parameters,
)


def test_ten_cate_e3_uses_viscous_container_walls_and_experimental_peak():
    config = case_parameters()
    assert config["fluid_wall_no_slip"] is True
    assert config["reference_speed"] == pytest.approx(0.959 * 0.091)
    assert config["viscosity"] == pytest.approx(0.113)


def test_sphere_grid_study_preserves_physical_domain():
    config = case_parameters(cells_per_diameter=16)
    counts, spacing = grid_parameters(config)
    assert all(n * h == pytest.approx(length) for n, h, length in zip(counts, spacing, config["domain"]))
    assert config["diameter"] / spacing[0] > 15.9


def test_sphere_run_selects_compatible_viscous_wall_configuration(tmp_path, monkeypatch, taichi_runtime):
    import geotaichi
    from src.mpm.Simulation import Simulation
    from examples.cfdem.FullyResolved.IBMResolvedSphereSettling.sphere_settling import run

    class ConfigurationChecked(Exception):
        pass

    def check_configuration(**configuration):
        sims = Simulation()
        for name in (
            "dimension",
            "material_type",
            "solver_type",
            "discretization",
            "fluid_level_set",
            "density_projection",
            "solid_sdf_cut_cell",
            "fluid_wall_no_slip",
        ):
            setattr(sims, name, configuration[name])
        sims.validate_configuration(require_solver_parameters=False)
        assert sims.fluid_wall_no_slip
        raise ConfigurationChecked

    monkeypatch.setattr(geotaichi, "init", lambda **_: None)
    monkeypatch.setattr(
        geotaichi,
        "DEMPM",
        lambda: SimpleNamespace(
            set_configuration=lambda **_: None,
            mpm=SimpleNamespace(set_configuration=check_configuration),
        ),
    )
    args = SimpleNamespace(
        output_dir=tmp_path, dt=None, time=None, arch="cpu", device_memory=1, write_vtu=False, alpha_pic=0.01
    )
    with pytest.raises(ConfigurationChecked):
        run(case_parameters(), args)
