import pytest
import taichi as ti


@pytest.fixture(scope="module", autouse=True)
def taichi_runtime():
    ti.init(arch=ti.cpu, offline_cache=False, log_level=ti.ERROR)
    yield
    ti.reset()


def _mpm_sparse_sim(coupling):
    from src.mpm.Simulation import Simulation

    sims = Simulation()
    sims.sparse_grid = True
    sims.sparse_grid_backend = "BlockScan"
    sims.mapping = "USL"
    sims.coupling = coupling
    sims.velocity_projection_scheme = "PIC/FLIP"
    return sims


def test_sparse_grid_allows_lagrangian_coupling_configuration():
    sims = _mpm_sparse_sim("Lagrangian")

    sims.validate_configuration()


def test_sparse_grid_rejects_non_lagrangian_coupling_configuration():
    sims = _mpm_sparse_sim("Eulerian")

    with pytest.raises(RuntimeError, match="coupling='Lagrangian'"):
        sims.validate_configuration()


def test_cfdem_rejects_mpm_sparse_grid_configuration():
    from src.dem.Simulation import Simulation as DEMSimulation
    from src.mpdem.Simulation import Simulation as DEMPMSimulation

    msims = _mpm_sparse_sim("Lagrangian")
    msims.material_type = "Fluid"

    dsims = DEMSimulation()
    dsims.set_dem_scheme("DEM")

    csims = DEMPMSimulation()
    csims.set_coupling_scheme("CFDEM")

    with pytest.raises(RuntimeError, match="CFDEM does not support MPM sparse_grid"):
        csims.validate_coupling_configuration(msims, dsims)
