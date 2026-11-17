"""Regression probes for per-test configuration and module runtime cleanup."""

import os

import numpy as np
import taichi as ti


_PROBE_ENVIRONMENT = "GT_RUNTIME_ISOLATION_PROBE"
_ORIGINAL_PROBE_VALUE = os.environ.get(_PROBE_ENVIRONMENT)


def test_solver_setup_globals_can_be_changed_inside_one_test():
    import src.iga.config as iga_config
    import src.igampm.config as igampm_config
    import src.mpm.config as mpm_config
    import src.utils.GlobalVariable as global_variable

    global_variable.DIMENSION = 3
    global_variable.TRACKENERGY = True
    global_variable.RANDOMFIELD = True
    global_variable.MPMXPBC = True
    global_variable.MPMXSIZE = 7.5
    global_variable.ADAPTIVESTIFF = True
    iga_config.DIM = 3
    iga_config.DYNAMIC = False
    mpm_config.DIM = 3
    mpm_config.DYNAMIC = False
    igampm_config.DIM = 3
    igampm_config.DYNAMIC = False
    os.environ[_PROBE_ENVIRONMENT] = "dirty"

    assert global_variable.TRACKENERGY
    assert (iga_config.DIM, mpm_config.DIM, igampm_config.DIM) == (3, 3, 3)


def test_next_test_observes_the_canonical_configuration():
    import src.iga.config as iga_config
    import src.igampm.config as igampm_config
    import src.mpm.config as mpm_config
    import src.utils.GlobalVariable as global_variable

    assert global_variable.DIMENSION == 2
    assert global_variable.TRACKENERGY is False
    assert global_variable.RANDOMFIELD is False
    assert global_variable.MPMXPBC is False
    assert global_variable.MPMXSIZE == 0.0
    assert global_variable.ADAPTIVESTIFF is False
    assert iga_config.DYNAMIC is True
    assert mpm_config.DYNAMIC is True
    assert igampm_config.DYNAMIC is True
    assert (iga_config.DIM, mpm_config.DIM, igampm_config.DIM) == (2, 2, 2)
    if _ORIGINAL_PROBE_VALUE is None:
        assert _PROBE_ENVIRONMENT not in os.environ
    else:
        assert os.environ[_PROBE_ENVIRONMENT] == _ORIGINAL_PROBE_VALUE


def test_module_may_leave_an_initialized_taichi_runtime():
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
    )
    values = ti.field(ti.f64, shape=2)
    values.from_numpy(np.asarray([1.0, 2.0]))
    assert values.to_numpy().tolist() == [1.0, 2.0]
