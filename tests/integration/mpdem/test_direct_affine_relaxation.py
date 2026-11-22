import numpy as np
import pytest

from tests.integration.mpdem.test_direct_affine_nonassociated import _case, runtime

pytestmark = [
    pytest.mark.integration,
    pytest.mark.mpdem,
    pytest.mark.ipc,
    pytest.mark.serial,
    pytest.mark.isolated_dimension(3),
]


def test_relaxed_plastic_contact_converges_and_rejects_unimplemented_trajectory_adjoint(tmp_path):
    coupling = _case(tmp_path, inexact=True, friction_relaxation=0.8)
    system = coupling.enginer
    assert coupling.run(verbose=False)["converged"]
    system.mpm.F0.from_numpy(np.array([np.diag([1.04, 0.96, 0.98])]))
    mass = system.mpm.particle.m.to_numpy().copy()
    for _ in range(2):
        result = coupling.run(verbose=False)
        assert result["converged"] and system.last_step_record["friction_converged"]
        assert system.last_linear_solve["equilibrated"]
        assert system.mixed.friction_count > 0 and system.mixed.diagnostics()["minimum_distance"] > 0
        assert np.isfinite(system.mpm.particle.x.to_numpy()).all()
        np.testing.assert_array_equal(system.mpm.particle.m.to_numpy(), mass)
    assert system.mpm.material.equivalent_plastic_strain[0] > 0
    with pytest.raises(ValueError, match="friction_relaxation"):
        coupling.differentiable(steps=1)
