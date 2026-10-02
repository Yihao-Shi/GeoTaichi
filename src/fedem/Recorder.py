"""FEM--DEM coupled contact and subsystem output."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.FieldIO import field_to_numpy_prefix


class WriteFile:
    def __init__(self, simulation, fem, dem, contactor, engine=None):
        self.simulation = simulation
        self.fem = fem
        self.dem = dem
        self.contactor = contactor
        self.engine = engine
        self.checkpoint_owner = None
        self.contact_path = None
        if simulation.path is not None:
            root = Path(simulation.path)
            root.mkdir(parents=True, exist_ok=True)
            self.contact_path = root / "FEDEMcontacts"
            self.contact_path.mkdir(parents=True, exist_ok=True)

    def output(self):
        if self.dem.recorder is not None:
            self.dem.recorder.output(self.dem.sims, self.dem.scene)
        if self.simulation.path is not None:
            vtk_path = Path(self.simulation.path) / "vtks"
            vtk_path.mkdir(parents=True, exist_ok=True)
            self.fem.engine.record(
                vtk_path / f"FEM{self.simulation.current_print:06d}.vtu",
                log=False,
            )
        if not self.simulation.save_contact or self.contact_path is None:
            self._output_checkpoint()
            return
        if self.contactor.neighbor is None:
            soft_contact = getattr(self.fem.engine, "soft_particle_contact", None)
            payload = {"contact_kind": np.asarray("FEM-FEM")}
            if soft_contact is not None:
                pt_count = int(soft_contact.pt_count)
                ee_count = int(soft_contact.ee_count)
                payload.update(
                    {
                        "point_triangle": np.ascontiguousarray(
                            field_to_numpy_prefix(soft_contact.culling.point_triangle, pt_count)
                        ),
                        "point_triangle_active": field_to_numpy_prefix(soft_contact.pt_active, pt_count),
                        "point_triangle_normal_force": field_to_numpy_prefix(soft_contact.pt_normal_force, pt_count),
                        "point_triangle_tangential_force": field_to_numpy_prefix(
                            soft_contact.pt_tangential_force, pt_count
                        ),
                        "edge_edge": field_to_numpy_prefix(soft_contact.culling.edge_edge, ee_count),
                        "edge_edge_active": field_to_numpy_prefix(soft_contact.ee_active, ee_count),
                        "edge_edge_normal_force": field_to_numpy_prefix(soft_contact.ee_normal_force, ee_count),
                        "edge_edge_tangential_force": field_to_numpy_prefix(soft_contact.ee_tangential_force, ee_count),
                    }
                )
            np.savez(
                self.contact_path / f"FEDEMContact{self.simulation.current_print:06d}.npz",
                **payload,
            )
            self._output_checkpoint()
            return
        count = self.contactor.neighbor.contact_count
        contacts = self.contactor.neighbor.contacts
        if self.contactor.level_set:
            payload = {
                "rigid_id": field_to_numpy_prefix(contacts.rigid_id, count),
                "node_id": field_to_numpy_prefix(contacts.node_id, count),
                "active": field_to_numpy_prefix(contacts.active, count),
                "normal_force": field_to_numpy_prefix(contacts.normal_force, count),
                "normal_gap": field_to_numpy_prefix(contacts.normal_gap, count),
                "tangential_force": field_to_numpy_prefix(contacts.tangential_force, count),
                "old_tangential_overlap": field_to_numpy_prefix(contacts.old_tangential_overlap, count),
            }
            wall_contacts = self.contactor.wall_contacts
            if wall_contacts is not None:
                wall_count = self.contactor.wall_count
                wall_pair_count = self.contactor.wall_candidate_count
                wall_active_all = field_to_numpy_prefix(wall_contacts.active, wall_pair_count)
                active_candidates = np.flatnonzero(wall_active_all)
                dense_pair_ids = field_to_numpy_prefix(self.contactor.wall_candidate_pairs, wall_pair_count)[
                    active_candidates
                ]
                facet_ids = dense_pair_ids % wall_count
                logical_wall_ids = field_to_numpy_prefix(self.dem.scene.wall.wallID, wall_count)
                payload.update(
                    {
                        "wall_node_id": np.ascontiguousarray(dense_pair_ids // wall_count),
                        "wall_facet_id": np.ascontiguousarray(facet_ids),
                        "wall_id": np.ascontiguousarray(logical_wall_ids[facet_ids]),
                        "wall_active": np.ascontiguousarray(wall_active_all[active_candidates]),
                        "wall_normal_gap": np.ascontiguousarray(
                            field_to_numpy_prefix(wall_contacts.normal_gap, wall_pair_count)[active_candidates]
                        ),
                        "wall_normal_force": np.ascontiguousarray(
                            field_to_numpy_prefix(wall_contacts.normal_force, wall_pair_count)[active_candidates]
                        ),
                        "wall_tangential_force": np.ascontiguousarray(
                            field_to_numpy_prefix(wall_contacts.tangential_force, wall_pair_count)[active_candidates]
                        ),
                        "wall_old_tangential_overlap": np.ascontiguousarray(
                            field_to_numpy_prefix(
                                wall_contacts.old_tangential_overlap,
                                wall_pair_count,
                            )[active_candidates]
                        ),
                    }
                )
            np.savez(
                self.contact_path / f"FEDEMContact{self.simulation.current_print:06d}.npz",
                **payload,
            )
            self._output_checkpoint()
            return
        np.savez(
            self.contact_path / f"FEDEMContact{self.simulation.current_print:06d}.npz",
            particle_id=field_to_numpy_prefix(contacts.particle_id, count),
            face_id=field_to_numpy_prefix(contacts.face_id, count),
            active=field_to_numpy_prefix(contacts.active, count),
            normal_force=field_to_numpy_prefix(contacts.normal_force, count),
            tangential_force=field_to_numpy_prefix(contacts.tangential_force, count),
            old_tangential_overlap=field_to_numpy_prefix(contacts.old_tangential_overlap, count),
        )
        self._output_checkpoint()

    def _output_checkpoint(self):
        if not self.simulation.save_checkpoint:
            return
        if self.checkpoint_owner is None or self.simulation.path is None:
            raise RuntimeError("FEDEM checkpoint output has no coupled owner/path")
        checkpoint_path = Path(self.simulation.path) / "checkpoints"
        self.checkpoint_owner.save_checkpoint(
            checkpoint_path / f"FEDEMCheckpoint{self.simulation.current_print:06d}.npz"
        )


__all__ = ["WriteFile"]
