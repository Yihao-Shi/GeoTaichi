import numpy as np
import taichi as ti
from src.physics_model.contact_model.ipc.ContactGeometry import aabb_overlap_with_clearance

import src.igampm.config as config
from src.nurbs.core.NurbsGeometry import NurbsBasisFunction1d, NurbsBasisFunction2d
from src.physics_model.contact_model.ipc.NurbsContact import squared_norm_nd


def _build_hull_tree(centers, groups):
    nodes, prefix = [], [0]

    def build(indices):
        node = len(nodes)
        nodes.append([-1, -1, -1, -1])
        if len(indices) == 1:
            nodes[node][3] = int(indices[0])
        else:
            axis = int(np.argmax(np.ptp(centers[indices], axis=0)))
            order = indices[np.argsort(centers[indices, axis], kind="stable")]
            middle = len(order) // 2
            nodes[node][0] = build(order[:middle])
            nodes[node][1] = build(order[middle:])
        nodes[node][2] = len(nodes)
        return node

    for indices in groups:
        if len(indices):
            build(indices)
        prefix.append(len(nodes))
    return nodes, prefix


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
            basis = self.basis[surface_ids[0]]
            # Basis objects contain only degree/dimension. Share their identity
            # so every contact kernel reuses its specialization across faces.
            for surface_id in surface_ids:
                self.basis[surface_id] = basis
            self.accd_basis.append(basis)
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
        self.direction_lower = ti.Vector.field(config.DIM, ti.f64, shape=max(1, self.num_surfaces))
        self.direction_upper = ti.Vector.field(config.DIM, ti.f64, shape=max(1, self.num_surfaces))
        self.surface_lower = ti.Vector.field(config.DIM, ti.f64, shape=max(1, self.num_surfaces))
        self.surface_upper = ti.Vector.field(config.DIM, ti.f64, shape=max(1, self.num_surfaces))
        self.surface_bounds_valid = ti.field(ti.i32, shape=max(1, self.num_surfaces))
        self.control_points_id = ti.field(ti.i32, shape=max(1, total_ctrlpts))
        self.weights = ti.field(ti.f64, shape=max(1, total_ctrlpts))
        self.knot_vector_u = ti.field(ti.f64, shape=max(1, total_knot_u))
        self.knot_vector_v = ti.field(ti.f64, shape=max(1, total_knot_v))

        ctrlpts_np = np.concatenate(all_control_points, axis=0) if all_control_points else np.zeros((0, config.DIM))
        ctrlpt_ids_np = (
            np.concatenate(all_control_point_ids, axis=0) if all_control_point_ids else np.zeros((0,), dtype=np.int32)
        )
        weights_np = np.concatenate(all_weights, axis=0) if all_weights else np.zeros((0,), dtype=np.float64)
        if not np.all(np.isfinite(ctrlpts_np)) or not np.all(np.isfinite(weights_np)) or np.any(weights_np <= 0):
            raise ValueError("NURBS contact requires finite control points and finite positive weights")
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
        centers = np.asarray(
            [
                0.5
                * (
                    ctrlpts_np[self.prefix_num_ctrlpts[sid] : self.prefix_num_ctrlpts[sid + 1]].min(axis=0)
                    + ctrlpts_np[self.prefix_num_ctrlpts[sid] : self.prefix_num_ctrlpts[sid + 1]].max(axis=0)
                )
                for sid in range(self.num_surfaces)
            ]
        )
        nodes, _ = _build_hull_tree(centers, [np.arange(self.num_surfaces)])
        self.surface_tree_count = len(nodes)
        self.surface_tree_nodes = ti.Vector.field(4, ti.i32, shape=max(1, len(nodes)))
        self.swept_surface_lower = ti.Vector.field(config.DIM, ti.f64, shape=max(1, self.num_surfaces))
        self.swept_surface_upper = ti.Vector.field(config.DIM, ti.f64, shape=max(1, self.num_surfaces))
        self.swept_tree_lower = ti.Vector.field(config.DIM, ti.f64, shape=max(1, len(nodes)))
        self.swept_tree_upper = ti.Vector.field(config.DIM, ti.f64, shape=max(1, len(nodes)))
        self.surface_group_order = ti.field(ti.i32, shape=max(1, self.num_surfaces))
        if nodes:
            self.surface_tree_nodes.from_numpy(np.asarray(nodes, dtype=np.int32))
            self.surface_group_order.from_numpy(np.argsort(grouped_surface_ids).astype(np.int32))
        if config.DIM == 3:
            self._initialize_projection_cache(knot_u_np, knot_v_np, ctrlpts_np)

    def _initialize_projection_cache(self, knot_u, knot_v, control_points):
        span_controls = []
        span_prefix = [0]
        greville = []
        for sid, basis in enumerate(self.basis):
            degree_u, degree_v = int(basis.basis_u.degree), int(basis.basis_v.degree)
            count_u, count_v = self.num_ctrlpts_u[sid + 1], self.num_ctrlpts_v[sid + 1]
            knots_u = knot_u[self.prefix_num_knot_u[sid] : self.prefix_num_knot_u[sid + 1]]
            knots_v = knot_v[self.prefix_num_knot_v[sid] : self.prefix_num_knot_v[sid + 1]]
            greville_u = [knots_u[i + 1 : i + degree_u + 1].sum() / degree_u for i in range(count_u)]
            greville_v = [knots_v[i + 1 : i + degree_v + 1].sum() / degree_v for i in range(count_v)]
            greville.extend((u, v) for v in greville_v for u in greville_u)
            # Include empty spans so device indexing matches the knot grid.
            for span_u in range(degree_u, count_u):
                for span_v in range(degree_v, count_v):
                    first_control = self.prefix_num_ctrlpts[sid] + (span_v - degree_v) * count_u + span_u - degree_u
                    span_controls.append((first_control, degree_u + 1, degree_v + 1, count_u))
            span_prefix.append(len(span_controls))
        self.prefix_num_spans_field = ti.field(ti.i32, shape=len(span_prefix))
        self.span_controls = ti.Vector.field(4, ti.i32, shape=max(1, len(span_controls)))
        self.span_lower = ti.Vector.field(3, ti.f64, shape=max(1, len(span_controls)))
        self.span_upper = ti.Vector.field(3, ti.f64, shape=max(1, len(span_controls)))
        self.control_greville = ti.Vector.field(2, ti.f64, shape=max(1, self.total_ctrlpts))
        self.prefix_num_spans_field.from_numpy(np.asarray(span_prefix, dtype=np.int32))
        if span_controls:
            self.span_controls.from_numpy(np.asarray(span_controls, dtype=np.int32))
        if greville:
            self.control_greville.from_numpy(np.asarray(greville, dtype=np.float64))

        centers = []
        for first, count_u, count_v, stride in span_controls:
            ids = first + np.arange(count_u)[None, :] + stride * np.arange(count_v)[:, None]
            points = control_points[ids.ravel()]
            centers.append(0.5 * (points.min(axis=0) + points.max(axis=0)))
        centers = np.asarray(centers)
        nodes, node_prefix = _build_hull_tree(
            centers, [np.arange(span_prefix[sid], span_prefix[sid + 1]) for sid in range(self.num_surfaces)]
        )
        self.swept_span_lower = ti.Vector.field(3, ti.f64, shape=max(1, len(span_controls)))
        self.swept_span_upper = ti.Vector.field(3, ti.f64, shape=max(1, len(span_controls)))
        self.swept_span_tree_lower = ti.Vector.field(3, ti.f64, shape=max(1, len(nodes)))
        self.swept_span_tree_upper = ti.Vector.field(3, ti.f64, shape=max(1, len(nodes)))
        self.span_tree_count = len(nodes)
        self.span_tree_prefix = ti.field(ti.i32, shape=len(node_prefix))
        self.span_tree_nodes = ti.Vector.field(4, ti.i32, shape=max(1, len(nodes)))
        self.span_tree_lower = ti.Vector.field(3, ti.f64, shape=max(1, len(nodes)))
        self.span_tree_upper = ti.Vector.field(3, ti.f64, shape=max(1, len(nodes)))
        self.span_tree_prefix.from_numpy(np.asarray(node_prefix, dtype=np.int32))
        if nodes:
            self.span_tree_nodes.from_numpy(np.asarray(nodes, dtype=np.int32))

    def update_span_bounds(self):
        """Refresh leaf hulls and refit the fixed-topology span hierarchy."""
        self._update_span_bounds()
        self._refit_span_tree()

    @ti.kernel
    def _update_span_bounds(self):
        """Refresh stationary span hulls once for all particle queries."""
        for span in range(self.prefix_num_spans_field[self.num_surfaces]):
            first, count_u, count_v, stride = self.span_controls[span]
            lower = ti.Vector([ti.math.inf, ti.math.inf, ti.math.inf], dt=ti.f64)
            upper = -lower
            for local_control in range(count_u * count_v):
                control_id = first + (local_control // count_u) * stride + local_control % count_u
                point = self.control_points_hat[control_id]
                lower = ti.min(lower, point)
                upper = ti.max(upper, point)
            self.span_lower[span] = lower
            self.span_upper[span] = upper

    @ti.kernel
    def _refit_span_tree(self):
        # ponytail: serial refit of a few thousand nodes; use per-level kernels
        # if refitting becomes significant relative to particle queries.
        ti.loop_config(serialize=True)
        for reverse_node in range(self.span_tree_prefix[self.num_surfaces]):
            node = self.span_tree_prefix[self.num_surfaces] - 1 - reverse_node
            left, right, _, span = self.span_tree_nodes[node]
            if span >= 0:
                self.span_tree_lower[node] = self.span_lower[span]
                self.span_tree_upper[node] = self.span_upper[span]
            else:
                self.span_tree_lower[node] = ti.min(self.span_tree_lower[left], self.span_tree_lower[right])
                self.span_tree_upper[node] = ti.max(self.span_tree_upper[left], self.span_tree_upper[right])

    def update_swept_bounds(self, max_step):
        self._update_swept_hulls(float(max_step))
        self._refit_swept_tree(
            self.surface_tree_count,
            self.surface_tree_nodes,
            self.swept_surface_lower,
            self.swept_surface_upper,
            self.swept_tree_lower,
            self.swept_tree_upper,
        )
        if config.DIM == 3:
            self._refit_swept_tree(
                self.span_tree_count,
                self.span_tree_nodes,
                self.swept_span_lower,
                self.swept_span_upper,
                self.swept_span_tree_lower,
                self.swept_span_tree_upper,
            )

    @ti.kernel
    def _update_swept_hulls(self, max_step: ti.f64):
        for sid in range(self.num_surfaces):
            lower = ti.Vector([ti.math.inf for _ in ti.static(range(config.DIM))])
            upper = -lower
            for control in range(self.prefix_num_ctrlpts_field[sid], self.prefix_num_ctrlpts_field[sid + 1]):
                begin = self.control_points_hat[control]
                end = begin + max_step * self.control_point_direction[control]
                lower, upper = ti.min(lower, begin, end), ti.max(upper, begin, end)
            self.swept_surface_lower[sid], self.swept_surface_upper[sid] = lower, upper
        if ti.static(config.DIM == 3):
            for span in range(self.prefix_num_spans_field[self.num_surfaces]):
                first, count_u, count_v, stride = self.span_controls[span]
                lower = ti.Vector([ti.math.inf, ti.math.inf, ti.math.inf], dt=ti.f64)
                upper = -lower
                for local in range(count_u * count_v):
                    control = first + (local // count_u) * stride + local % count_u
                    begin = self.control_points_hat[control]
                    end = begin + max_step * self.control_point_direction[control]
                    lower, upper = ti.min(lower, begin, end), ti.max(upper, begin, end)
                self.swept_span_lower[span], self.swept_span_upper[span] = lower, upper

    @ti.kernel
    def _refit_swept_tree(
        self,
        count: int,
        nodes: ti.template(),
        lower: ti.template(),
        upper: ti.template(),
        tree_lower: ti.template(),
        tree_upper: ti.template(),
    ):
        ti.loop_config(serialize=True)
        for reverse in range(count):
            node = count - 1 - reverse
            left, right, _, leaf = nodes[node]
            if leaf >= 0:
                tree_lower[node], tree_upper[node] = lower[leaf], upper[leaf]
            else:
                tree_lower[node] = ti.min(tree_lower[left], tree_lower[right])
                tree_upper[node] = ti.max(tree_upper[left], tree_upper[right])

    @ti.func
    def swept_box_overlap(self, point_lower, point_upper, lower, upper, clearance):
        return aabb_overlap_with_clearance(point_lower, point_upper, lower, upper, clearance)

    @ti.func
    def swept_span_overlap(self, sid, point_lower, point_upper, clearance):
        found = False
        node, end = self.span_tree_prefix[sid], self.span_tree_prefix[sid + 1]
        while node < end:
            left, _, escape, span = self.span_tree_nodes[node]
            if not self.swept_box_overlap(
                point_lower, point_upper, self.swept_span_tree_lower[node], self.swept_span_tree_upper[node], clearance
            ):
                node = escape
            elif span >= 0:
                found = True
                break
            else:
                node = left
        return found

    @ti.func
    def span_node_distance_squared(self, node, point):
        offset = ti.max(self.span_tree_lower[node] - point, point - self.span_tree_upper[node], 0.0)
        return squared_norm_nd(offset)

    @ti.func
    def span_distance_lower_bound(self, surface_id, point, threshold):
        """Certify distant pairs; return zero as soon as a span may be near."""
        scale = ti.max(1.0, point.norm(), self.surface_lower[surface_id].norm(), self.surface_upper[surface_id].norm())
        padding = 1.0e-12 * scale
        threshold2 = (threshold + padding) ** 2
        lower2 = ti.math.inf
        node = self.span_tree_prefix[surface_id]
        end = self.span_tree_prefix[surface_id + 1]
        while node < end:
            left, _, escape, span = self.span_tree_nodes[node]
            distance2 = self.span_node_distance_squared(node, point)
            if distance2 > threshold2:
                lower2 = ti.min(lower2, distance2)
                node = escape
            elif span >= 0:
                lower2 = 0.0
                break
            else:
                node = left
        return ti.max(0.0, ti.sqrt(lower2) - padding)

    @ti.func
    def _nearest_span_control(self, span, point, best_control, best_distance2):
        first, count_u, count_v, stride = self.span_controls[span]
        for local_control in range(count_u * count_v):
            control = first + (local_control // count_u) * stride + local_control % count_u
            distance2 = squared_norm_nd(self.control_points_hat[control] - point)
            if distance2 < ti.math.inf and (
                best_control < 0
                or distance2 < best_distance2
                or (distance2 == best_distance2 and control < best_control)
            ):
                best_control, best_distance2 = control, distance2
        return best_control, best_distance2

    @ti.func
    def nearest_control_point(self, surface_id, point):
        """Exact control-point seed search using conservative span hulls."""
        assert self.surface_bounds_valid[surface_id] != 0, "invalid NURBS control hull"
        root = self.span_tree_prefix[surface_id]
        end = self.span_tree_prefix[surface_id + 1]
        node = root
        # A greedy leaf provides an upper bound before stackless traversal.
        while self.span_tree_nodes[node][3] < 0:
            left, right = self.span_tree_nodes[node][0], self.span_tree_nodes[node][1]
            node = left
            if self.span_node_distance_squared(right, point) < self.span_node_distance_squared(left, point):
                node = right
        best_control, best_distance2 = self._nearest_span_control(self.span_tree_nodes[node][3], point, -1, 1.0e300)
        node = root
        while node < end:
            left, _, escape, span = self.span_tree_nodes[node]
            if best_control < 0 or self.span_node_distance_squared(node, point) <= best_distance2 + 1.0e-14 * (
                1.0 + best_distance2
            ):
                if span >= 0:
                    best_control, best_distance2 = self._nearest_span_control(span, point, best_control, best_distance2)
                    node = escape
                else:
                    node = left
            else:
                node = escape
        assert best_control >= 0, "NURBS surface controls have non-finite distances"
        return best_control

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
    def update_surface_bounds(self):
        """Bound positive-weight NURBS geometry by its current control hull."""
        for surface_id in range(self.num_surfaces):
            lower = ti.Vector.zero(ti.f64, config.DIM)
            upper = ti.Vector.zero(ti.f64, config.DIM)
            valid = 1
            for component in ti.static(range(config.DIM)):
                lower[component] = ti.math.inf
                upper[component] = -ti.math.inf
            for control_id in range(
                self.prefix_num_ctrlpts_field[surface_id], self.prefix_num_ctrlpts_field[surface_id + 1]
            ):
                if not (self.weights[control_id] > 0.0 and self.weights[control_id] < ti.math.inf):
                    valid = 0
                point = self.control_points_hat[control_id]
                for component in ti.static(range(config.DIM)):
                    if not (ti.abs(point[component]) < ti.math.inf):
                        valid = 0
                    lower[component] = ti.min(lower[component], point[component])
                    upper[component] = ti.max(upper[component], point[component])
            self.surface_lower[surface_id] = lower
            self.surface_upper[surface_id] = upper
            self.surface_bounds_valid[surface_id] = valid

    @ti.func
    def distance_lower_bound(self, surface_id, position):
        offset = ti.max(self.surface_lower[surface_id] - position, position - self.surface_upper[surface_id], 0.0)
        # Round down to keep pruning conservative on the activation boundary.
        scale = ti.max(
            1.0, position.norm(), self.surface_lower[surface_id].norm(), self.surface_upper[surface_id].norm()
        )
        distance = ti.max(0.0, offset.norm() - 1.0e-13 * scale)
        valid = self.surface_bounds_valid[surface_id] != 0
        for component in ti.static(range(config.DIM)):
            valid = valid and ti.abs(position[component]) < ti.math.inf
        if not valid:
            # The common minimum-distance reduction rejects this sentinel on
            # every backend, including CUDA with debug assertions disabled.
            distance = -1.0
        return distance

    @ti.kernel
    def update_control_point_direction(self, grid_direction: ti.template()):
        """Gather an IGA search direction in contact-surface ordering."""
        for local_control_id in range(self.total_ctrlpts):
            global_control_id = self.control_points_id[local_control_id]
            direction = ti.Vector.zero(ti.f64, config.DIM)
            for component in ti.static(range(config.DIM)):
                direction[component] = grid_direction[config.DIM * global_control_id + component]
            self.control_point_direction[local_control_id] = direction
        for surface_id in range(self.num_surfaces):
            lower = ti.Vector([ti.math.inf for _ in ti.static(range(config.DIM))])
            upper = -lower
            for control_id in range(
                self.prefix_num_ctrlpts_field[surface_id], self.prefix_num_ctrlpts_field[surface_id + 1]
            ):
                lower = ti.min(lower, self.control_point_direction[control_id])
                upper = ti.max(upper, self.control_point_direction[control_id])
            self.direction_lower[surface_id] = lower
            self.direction_upper[surface_id] = upper

    @ti.func
    def relative_motion_upper_bound(self, surface_id, point_direction):
        extent = ti.max(
            ti.abs(point_direction - self.direction_lower[surface_id]),
            ti.abs(point_direction - self.direction_upper[surface_id]),
        )
        return extent.norm() * (1.0 + 1.0e-12)
