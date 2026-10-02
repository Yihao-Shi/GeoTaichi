"""Contact-model lifecycle for explicit and implicit FEM--MPM coupling."""

from __future__ import annotations

from src.fempm.contact import HertzMindlinModel, IPCModel, LinearModel
from src.fempm.contact.ContactKernel import (
    reset_contact_force,
    resolve_hertz_mindlin_contact,
    resolve_linear_contact,
)
from src.fempm.neighbor import FEMPMBVH, FEMPMLinkedCell
from src.utils.linalg import no_operation


class ContactManager:
    def __init__(self, simulation):
        self.simulation = simulation
        self.patch = None
        self.neighbor = None
        self.model = None
        self.initialized = False
        self.resolve_contact_step = no_operation

    def choose_contact_model(self, model, **kwargs):
        normalized = str(model).replace("_", "").replace("-", "").replace(" ", "").lower()
        if normalized in ("linear", "linearmodel", "linearspring"):
            self.model = LinearModel(self.simulation)
        elif normalized in ("hertz", "hertzmindlin", "hertzmindlinmodel"):
            self.model = HertzMindlinModel(self.simulation)
        elif normalized in ("ipc", "incrementalpotentialcontact", "barrier", "barrieripc", "semi", "semiipc"):
            kwargs.setdefault(
                "ipc_model",
                "SemiIPC" if normalized in ("semi", "semiipc") else "BarrierIPC",
            )
            self.model = IPCModel(self.simulation, **kwargs)
        else:
            raise ValueError("FEMPM contact model must be Linear, HertzMindlin, BarrierIPC, or SemiIPC")
        return self.model

    def add_property(self, mpm_material, fem_body, parameters):
        if self.model is None:
            raise RuntimeError("choose the FEMPM contact model before properties")
        self.model.add_property(mpm_material, fem_body, parameters)

    def initialize(self, patch, fem_engine, mpm_scene):
        if self.model is None:
            raise RuntimeError("FEMPM contact model has not been selected")
        if isinstance(self.model, IPCModel):
            raise RuntimeError("implicit FEMPM IPC is initialized by its monolithic engine")
        self.patch = patch
        self.patch.update(fem_engine.position_field)
        face_radius_min, face_radius_max = self.patch.bounding_radii()
        particle_radius_min = float(mpm_scene.find_particle_min_radius())
        particle_radius_max = float(mpm_scene.find_particle_max_radius())
        if particle_radius_min <= 0.0 or particle_radius_max <= 0.0:
            raise RuntimeError("FEMPM requires positive-radius MPM material points")
        if self.simulation.verlet_distance <= 0.0:
            self.simulation.verlet_distance = self.simulation.verlet_distance_multiplier * min(
                particle_radius_min, face_radius_min
            )
        if self.simulation.search == "LinkedCell":
            self.neighbor = FEMPMLinkedCell(self.simulation, self.patch, particle_radius_max, face_radius_max)
        else:
            self.neighbor = FEMPMBVH(self.simulation, self.patch)
        self.rebuild(mpm_scene)
        self.resolve_contact_step = (
            resolve_linear_contact if isinstance(self.model, LinearModel) else resolve_hertz_mindlin_contact
        )
        self.initialized = True

    def reset(self):
        if self.neighbor is not None:
            reset_contact_force(self.neighbor.contact_count, self.neighbor.contacts)

    def requires_rebuild(self, mpm_scene):
        if self.neighbor is None:
            return True
        surface_motion = float(self.patch.maximum_displacement[None]) > 0.5 * self.simulation.verlet_distance
        particle_motion = self.neighbor.particle_requires_rebuild(int(mpm_scene.couplingNum[0]), mpm_scene.particle)
        return surface_motion or particle_motion

    def rebuild(self, mpm_scene):
        result = self.neighbor.rebuild(int(mpm_scene.couplingNum[0]), mpm_scene.particle)
        mpm_scene.reset_verlet_disp()
        return result

    def resolve(self, mpm_scene, fem_engine):
        arguments = (
            self.neighbor.contact_count,
            self.simulation.delta,
            self.model.surface_properties,
            mpm_scene.particle,
            self.patch.nodes,
            self.patch.faces,
            self.patch.face_body,
            self.patch.normals,
            fem_engine.velocity_field,
            fem_engine.state.external_force,
            self.neighbor.contacts,
        )
        self.resolve_contact_step(*arguments)

    def critical_timestep(self, mpm_scene):
        return self.model.critical_timestep(
            mpm_scene.find_particle_min_mass(),
            mpm_scene.find_particle_max_radius(),
        )


__all__ = ["ContactManager"]
