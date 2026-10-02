import numpy as np
import taichi as ti

import src.igampm.config as config
from src.nurbs.core.NurbsGeometry import NurbsBasisFunction1d, NurbsBasisFunction2d


@ti.data_oriented
class CouplingContactSurface:
    """Boundary primitives used by the IGA--MPM contact layer.

    A patch in GeoTaichi currently owns independent degrees of freedom, so two
    geometrically coincident patch faces can mean either an internal
    multi-patch interface or two distinct contacting bodies.  The default is
    consequently conservative (keep both).  Applications that assemble a
    multi-patch solid can opt into ``canonical`` or ``remove_internal`` through
    ``contact_surface_duplicate_policy`` and can always select faces explicitly
    with ``contact_surface_include``/``contact_surface_exclude``.
    """

    def __init__(self, iga, **kwargs):
        self.iga = iga
        self.num_surfaces = 0
        self.num_knot_u = [0]
        self.num_knot_v = [0]
        self.num_ctrlpts = [0]
        self.num_ctrlpts_u = [0]
        self.num_ctrlpts_v = [0]
        self.prefix_num_knot_u = None
        self.prefix_num_knot_v = None
        self.prefix_num_ctrlpts = None
        self.prefix_num_ctrlpts_u = None
        self.prefix_num_ctrlpts_v = None
        self.basis = []
        self.max_support_size = 1
        self.surface_keys = []
        self.excluded_surface_keys = []
        self.duplicate_surface_groups = []

        include = self._surface_key_set(kwargs.get("contact_surface_include", kwargs.get("include_surfaces")))
        exclude = self._surface_key_set(kwargs.get("contact_surface_exclude", kwargs.get("exclude_surfaces"))) or set()
        duplicate_policy = str(kwargs.get("contact_surface_duplicate_policy", "keep")).strip().replace("-", "_").lower()
        duplicate_aliases = {
            "keep": "keep",
            "all": "keep",
            "canonical": "canonical",
            "first": "canonical",
            "remove_internal": "remove_internal",
            "remove_pairs": "remove_internal",
        }
        if duplicate_policy not in duplicate_aliases:
            raise ValueError("contact_surface_duplicate_policy must be 'keep', " "'canonical', or 'remove_internal'")
        duplicate_policy = duplicate_aliases[duplicate_policy]
        duplicate_tolerance = float(kwargs.get("contact_surface_duplicate_tolerance", 1.0e-10))
        if not np.isfinite(duplicate_tolerance) or duplicate_tolerance <= 0.0:
            raise ValueError("contact_surface_duplicate_tolerance must be finite and positive")

        all_control_points = []
        all_control_point_ids = []
        all_weights = []
        all_knot_u = []
        all_knot_v = []

        pending_surfaces = []
        gathered_surface_id = 0
        for patch_id, (_, meta) in enumerate(self.iga.patch.primitive.body.items()):
            primitive = meta["primitive"]
            patch_sizes, boundary_ctrlpts_id, knot_vector_list, degree_list = primitive.gather_boundary_ctrlpts()
            prefix_ctrlpt = self.iga.patch.prefix_total_num_ctrlpts[patch_id]

            local_begin = 0
            for face_id, face_size in enumerate(patch_sizes):
                local_face_ids = np.asarray(boundary_ctrlpts_id[local_begin : local_begin + face_size], dtype=np.int32)
                local_begin += face_size
                surface_key = (patch_id, face_id)
                selected = self._surface_selected(surface_key, gathered_surface_id, include, exclude)
                gathered_surface_id += 1
                if config.DIM == 3:
                    if face_id < 2:
                        num_ctrlpts_u = primitive.num_ctrlpts_u
                        num_ctrlpts_v = primitive.num_ctrlpts_v
                    elif face_id < 4:
                        num_ctrlpts_u = primitive.num_ctrlpts_u
                        num_ctrlpts_v = primitive.num_ctrlpts_w
                    else:
                        num_ctrlpts_u = primitive.num_ctrlpts_v
                        num_ctrlpts_v = primitive.num_ctrlpts_w

                    knot_u = np.asarray(knot_vector_list[face_id][0], dtype=np.float64)
                    knot_v = np.asarray(knot_vector_list[face_id][1], dtype=np.float64)
                    basis = NurbsBasisFunction2d(degree_list[face_id][0], degree_list[face_id][1], dimension=config.DIM)
                    support_size = (int(degree_list[face_id][0]) + 1) * (int(degree_list[face_id][1]) + 1)
                else:
                    num_ctrlpts_u = face_size
                    num_ctrlpts_v = 1
                    knot_u = np.asarray(knot_vector_list[face_id], dtype=np.float64)
                    knot_v = np.zeros(0, dtype=np.float64)
                    basis = NurbsBasisFunction1d(degree_list[face_id], dimension=config.DIM)
                    support_size = int(degree_list[face_id]) + 1

                if not selected:
                    self.excluded_surface_keys.append(surface_key)
                    continue
                control_points = np.asarray(primitive.control_points[local_face_ids], dtype=np.float64)
                face_weights = np.asarray(primitive.weights[local_face_ids], dtype=np.float64)
                pending_surfaces.append(
                    {
                        "key": surface_key,
                        "face_size": face_size,
                        "num_ctrlpts_u": num_ctrlpts_u,
                        "num_ctrlpts_v": num_ctrlpts_v,
                        "knot_u": knot_u,
                        "knot_v": knot_v,
                        "basis": basis,
                        "support_size": support_size,
                        "control_points": control_points,
                        "control_point_ids": local_face_ids + prefix_ctrlpt,
                        "weights": face_weights,
                        "signature": self._face_signature(control_points, face_weights, duplicate_tolerance),
                    }
                )

        signature_groups = {}
        for index, record in enumerate(pending_surfaces):
            signature_groups.setdefault(record["signature"], []).append(index)
        duplicate_groups = [indices for indices in signature_groups.values() if len(indices) > 1]
        self.duplicate_surface_groups = [
            tuple(pending_surfaces[index]["key"] for index in indices) for indices in duplicate_groups
        ]
        keep = np.ones(len(pending_surfaces), dtype=bool)
        if duplicate_policy == "canonical":
            for indices in duplicate_groups:
                keep[indices[1:]] = False
        elif duplicate_policy == "remove_internal":
            for indices in duplicate_groups:
                keep[indices] = False

        for index, record in enumerate(pending_surfaces):
            if not keep[index]:
                self.excluded_surface_keys.append(record["key"])
                continue
            self.surface_keys.append(record["key"])
            self.num_surfaces += 1
            self.num_knot_u.append(len(record["knot_u"]))
            self.num_knot_v.append(len(record["knot_v"]))
            self.num_ctrlpts.append(record["face_size"])
            self.num_ctrlpts_u.append(record["num_ctrlpts_u"])
            self.num_ctrlpts_v.append(record["num_ctrlpts_v"])
            self.basis.append(record["basis"])
            self.max_support_size = max(self.max_support_size, record["support_size"])
            all_control_points.append(record["control_points"])
            all_control_point_ids.append(record["control_point_ids"])
            all_weights.append(record["weights"])
            all_knot_u.append(record["knot_u"])
            if config.DIM == 3:
                all_knot_v.append(record["knot_v"])

        self.prefix_num_knot_u = np.cumsum(np.asarray(self.num_knot_u, dtype=np.int32))
        self.prefix_num_knot_v = np.cumsum(np.asarray(self.num_knot_v, dtype=np.int32))
        self.prefix_num_ctrlpts = np.cumsum(np.asarray(self.num_ctrlpts, dtype=np.int32))
        self.prefix_num_ctrlpts_u = np.cumsum(np.asarray(self.num_ctrlpts_u, dtype=np.int32))
        self.prefix_num_ctrlpts_v = np.cumsum(np.asarray(self.num_ctrlpts_v, dtype=np.int32))
        basis_groups = {}
        for surface_id, basis in enumerate(self.basis):
            if config.DIM == 2:
                signature = (int(basis.basis_u.degree),)
            else:
                signature = (
                    int(basis.basis_u.degree),
                    int(basis.basis_v.degree),
                )
            basis_groups.setdefault(signature, []).append(surface_id)
        self.accd_basis = []
        self.accd_basis_group_offsets = [0]
        grouped_surface_ids = []
        for surface_ids in basis_groups.values():
            self.accd_basis.append(self.basis[surface_ids[0]])
            grouped_surface_ids.extend(surface_ids)
            self.accd_basis_group_offsets.append(len(grouped_surface_ids))

        # Device mirrors let one ACCD kernel specialization process every
        # boundary with the same degree signature.
        self.prefix_num_knot_u_field = ti.field(ti.i32, shape=len(self.prefix_num_knot_u))
        self.prefix_num_knot_v_field = ti.field(ti.i32, shape=len(self.prefix_num_knot_v))
        self.prefix_num_ctrlpts_field = ti.field(ti.i32, shape=len(self.prefix_num_ctrlpts))
        self.accd_basis_group_surface_ids = ti.field(ti.i32, shape=max(1, self.num_surfaces))
        self.prefix_num_knot_u_field.from_numpy(np.asarray(self.prefix_num_knot_u, dtype=np.int32))
        self.prefix_num_knot_v_field.from_numpy(np.asarray(self.prefix_num_knot_v, dtype=np.int32))
        self.prefix_num_ctrlpts_field.from_numpy(np.asarray(self.prefix_num_ctrlpts, dtype=np.int32))
        if grouped_surface_ids:
            grouped_surface_ids_np = np.zeros(max(1, self.num_surfaces), dtype=np.int32)
            grouped_surface_ids_np[: self.num_surfaces] = np.asarray(grouped_surface_ids, dtype=np.int32)
            self.accd_basis_group_surface_ids.from_numpy(grouped_surface_ids_np)

        total_ctrlpts = int(self.prefix_num_ctrlpts[-1])
        total_knot_u = int(self.prefix_num_knot_u[-1])
        total_knot_v = int(self.prefix_num_knot_v[-1])
        self.total_ctrlpts = total_ctrlpts
        self.total_knot_u = total_knot_u
        self.total_knot_v = total_knot_v

        self.control_points_hat = ti.Vector.field(config.DIM, ti.f64, shape=max(1, total_ctrlpts))
        self.control_point_direction = ti.Vector.field(config.DIM, ti.f64, shape=max(1, total_ctrlpts))
        self.control_points_id = ti.field(ti.i32, shape=max(1, total_ctrlpts))
        self.weights = ti.field(ti.f64, shape=max(1, total_ctrlpts))
        self.knot_vector_u = ti.field(ti.f64, shape=max(1, total_knot_u))
        self.knot_vector_v = ti.field(ti.f64, shape=max(1, total_knot_v))

        ctrlpts_np = np.concatenate(all_control_points, axis=0) if all_control_points else np.zeros((0, config.DIM))
        ctrlpt_ids_np = (
            np.concatenate(all_control_point_ids, axis=0) if all_control_point_ids else np.zeros((0,), dtype=np.int32)
        )
        weights_np = np.concatenate(all_weights, axis=0) if all_weights else np.zeros((0,), dtype=np.float64)
        knot_u_np = np.concatenate(all_knot_u, axis=0) if all_knot_u else np.zeros((0,), dtype=np.float64)
        knot_v_np = np.concatenate(all_knot_v, axis=0) if all_knot_v else np.zeros((0,), dtype=np.float64)

        if total_ctrlpts > 0:
            self.control_points_hat.from_numpy(np.asarray(ctrlpts_np, dtype=np.float64))
            self.control_points_id.from_numpy(np.asarray(ctrlpt_ids_np, dtype=np.int32))
            self.weights.from_numpy(np.asarray(weights_np, dtype=np.float64))
        if total_knot_u > 0:
            self.knot_vector_u.from_numpy(np.asarray(knot_u_np, dtype=np.float64))
        if total_knot_v > 0:
            self.knot_vector_v.from_numpy(np.asarray(knot_v_np, dtype=np.float64))

    @staticmethod
    def _surface_key_set(value):
        if value is None:
            return None
        if isinstance(value, (str, int, np.integer)):
            value = [value]
        elif isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, (int, np.integer)) for v in value):
            value = [value]
        result = set()
        for item in value:
            if isinstance(item, (int, np.integer)):
                result.add(int(item))
            elif isinstance(item, str):
                parts = item.replace("/", ":").split(":")
                if len(parts) != 2:
                    raise ValueError("surface keys must be global indices or 'patch:face'")
                result.add((int(parts[0]), int(parts[1])))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                result.add((int(item[0]), int(item[1])))
            else:
                raise ValueError("surface keys must be global indices or (patch, face) pairs")
        return result

    @staticmethod
    def _surface_selected(key, global_index, include, exclude):
        if include is not None and key not in include and global_index not in include:
            return False
        return key not in exclude and global_index not in exclude

    @staticmethod
    def _face_signature(control_points, weights, tolerance):
        """Canonical signature for coincident equal-control-mesh boundaries."""
        points = np.asarray(control_points, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if not np.isfinite(points).all() or not np.isfinite(weights).all():
            raise ValueError("contact-surface control points and weights must be finite")
        scale = max(float(np.ptp(points, axis=0).max()), 1.0)
        quantized_points = np.rint(points / (tolerance * scale)).astype(np.int64)
        weight_scale = max(float(np.max(np.abs(weights))), 1.0e-30)
        quantized_weights = np.rint((weights / weight_scale) / tolerance).astype(np.int64)
        rows = [tuple(quantized_points[i].tolist()) + (int(quantized_weights[i]),) for i in range(points.shape[0])]
        return points.shape[0], tuple(sorted(rows))

    @ti.kernel
    def update_from_patch(self, control_points: ti.template()):
        for i in range(self.total_ctrlpts):
            self.control_points_hat[i] = control_points[self.control_points_id[i]]

    @ti.kernel
    def update_from_patch_displacement(self, control_points: ti.template(), grid_disp: ti.template()):
        """Evaluate the contact surface at the current IGA Newton iterate."""
        for i in range(self.total_ctrlpts):
            control_id = self.control_points_id[i]
            displacement = ti.Vector.zero(ti.f64, config.DIM)
            for d in ti.static(range(config.DIM)):
                displacement[d] = grid_disp[config.DIM * control_id + d]
            self.control_points_hat[i] = control_points[control_id] + displacement

    @ti.kernel
    def update_control_point_direction(self, grid_direction: ti.template()):
        """Gather an IGA search direction in contact-surface ordering."""
        for local_control_id in range(self.total_ctrlpts):
            global_control_id = self.control_points_id[local_control_id]
            direction = ti.Vector.zero(ti.f64, config.DIM)
            for component in ti.static(range(config.DIM)):
                direction[component] = grid_direction[config.DIM * global_control_id + component]
            self.control_point_direction[local_control_id] = direction
