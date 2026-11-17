"""FEM--MPM coupled contact and subsystem output."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class WriteFile:
    def __init__(self, simulation, fem, mpm, contactor):
        self.simulation = simulation
        self.fem = fem
        self.mpm = mpm
        self.contactor = contactor
        self.contact_path = None
        if simulation.path is not None:
            root = Path(simulation.path)
            root.mkdir(parents=True, exist_ok=True)
            self.contact_path = root / "FEMPMcontacts"
            self.contact_path.mkdir(parents=True, exist_ok=True)

    def output(self):
        if self.mpm.recorder is not None:
            self.mpm.recorder.output(self.mpm.sims, self.mpm.scene)
        elif self.mpm.enginer is not None:
            self.mpm.enginer.record(log=False)
        if self.simulation.path is not None:
            vtk_path = Path(self.simulation.path) / "vtks"
            vtk_path.mkdir(parents=True, exist_ok=True)
            self.fem.engine.record(
                vtk_path / f"FEM{self.simulation.current_print:06d}.vtu",
                log=False,
            )
        if not self.simulation.save_contact or self.contact_path is None:
            return
        if self.contactor.neighbor is None:
            return
        count = self.contactor.neighbor.contact_count
        contacts = self.contactor.neighbor.contacts
        np.savez(
            self.contact_path / f"FEMPMContact{self.simulation.current_print:06d}.npz",
            particle_id=np.ascontiguousarray(contacts.particle_id.to_numpy()[:count]),
            face_id=np.ascontiguousarray(contacts.face_id.to_numpy()[:count]),
            active=np.ascontiguousarray(contacts.active.to_numpy()[:count]),
            normal_force=np.ascontiguousarray(contacts.normal_force.to_numpy()[:count]),
            tangential_force=np.ascontiguousarray(contacts.tangential_force.to_numpy()[:count]),
            old_tangential_overlap=np.ascontiguousarray(contacts.old_tangential_overlap.to_numpy()[:count]),
        )


__all__ = ["WriteFile"]
