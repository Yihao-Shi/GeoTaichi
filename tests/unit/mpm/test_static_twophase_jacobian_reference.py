"""Small deterministic oracles extracted from the legacy Jacobian scripts."""

import pytest

from tools.diagnostics.mpm.static_twophase import (
    verify_static_twophase_global_jacobian as global_2d,
)
from tools.diagnostics.mpm.static_twophase import (
    verify_static_twophase_global_jacobian_3d as global_3d,
)
from tools.diagnostics.mpm.static_twophase import (
    verify_static_twophase_local_jacobian as local_2d,
)
from tools.diagnostics.mpm.static_twophase import (
    verify_static_twophase_local_jacobian_3d as local_3d,
)


pytestmark = [
    pytest.mark.unit,
    pytest.mark.mpm,
    pytest.mark.materials,
    pytest.mark.assembly,
    pytest.mark.cpu,
]


@pytest.mark.parametrize("material", ["linearElastic", "neoHookean"])
@pytest.mark.parametrize("ppp", [False, True], ids=["ppp-off", "ppp-on"])
@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_local_two_phase_2d_jacobian_matches_finite_difference(
    material, ppp, mode
):
    _, relative_error = local_2d.run_case(
        material, ppp, mode, trials=2
    )
    assert relative_error < 1.0e-7


@pytest.mark.parametrize(
    "material", ["linearElastic", "neoHookean", "druckerPrager"]
)
@pytest.mark.parametrize("ppp", [False, True], ids=["ppp-off", "ppp-on"])
def test_local_two_phase_3d_jacobian_matches_finite_difference(
    material, ppp
):
    _, relative_error = local_3d.run_case(material, ppp, trials=2)
    assert relative_error < 1.0e-7


@pytest.mark.parametrize("material", ["linearElastic", "neoHookean"])
@pytest.mark.parametrize("ppp", [False, True], ids=["ppp-off", "ppp-on"])
@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_global_two_phase_2d_scatter_matches_finite_difference(
    material, ppp, mode
):
    _, relative_error = global_2d.run_case(
        material, ppp, mode, trials=1
    )
    assert relative_error < 1.0e-7


@pytest.mark.parametrize(
    "material", ["linearElastic", "neoHookean", "druckerPrager"]
)
@pytest.mark.parametrize("ppp", [False, True], ids=["ppp-off", "ppp-on"])
def test_global_two_phase_3d_scatter_matches_finite_difference(
    material, ppp
):
    _, relative_error = global_3d.run_case(material, ppp, trials=1)
    assert relative_error < 1.0e-7
