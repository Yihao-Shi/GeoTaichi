from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = REPO_ROOT / "research" / "LSMPM" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from validation_common import contact_work_increment_audit  # noqa: E402


def test_contact_work_audit_separates_transfer_and_storage() -> None:
    step = np.arange(1, 5, dtype=np.int64)
    contact_power = np.asarray([0.0, 2.0, 2.0, 0.0])
    contact_work = np.asarray([0.0, 1.0, 2.0, 1.0])
    body_energy = 10.0 + np.cumsum(contact_work)
    contact_energy = -np.cumsum(contact_work)

    arrays, summary = contact_work_increment_audit(
        step=step,
        time_step=1.0,
        body_mechanical_energy=body_energy,
        contact_elastic_energy=contact_energy,
        contact_power=contact_power,
        initial_body_mechanical_energy=10.0,
        energy_reference=10.0,
        redistance_completed_steps=(3,),
    )

    np.testing.assert_allclose(arrays["contact_work_increment"], contact_work)
    np.testing.assert_allclose(arrays["mechanical_transfer_defect"], 0.0)
    np.testing.assert_allclose(arrays["contact_storage_defect"], 0.0)
    assert arrays["redistance_event"].tolist() == [False, False, True, False]
    assert summary["step_resolved"] is True
    assert summary["redistance_event_count"] == 1
    assert summary["cumulative_total_energy_increment"] == pytest.approx(0.0)


def test_contact_work_audit_rejects_nonmonotone_steps() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        contact_work_increment_audit(
            step=[1, 1],
            time_step=1.0,
            body_mechanical_energy=[1.0, 1.0],
            contact_elastic_energy=[0.0, 0.0],
            contact_power=[0.0, 0.0],
            initial_body_mechanical_energy=1.0,
            energy_reference=1.0,
        )
