"""GPU point--NURBS contact for explicit IGA--MPM coupling."""

import numpy as np
import taichi as ti

import src.igampm.config as config
import src.utils.GlobalVariable as GlobalVariable
from src.igampm.contact.ContactSurface import CouplingContactSurface
from src.utils.linalg import no_operation


vec2d = ti.types.vector(2, ti.f64)
vec3d = ti.types.vector(3, ti.f64)


@ti.func
def _closest_point_on_surface(
    start_knot_u,
    start_knot_v,
    start_ctrlpt,
    num_knot_u,
    num_knot_v,
    knot_vector_u,
    knot_vector_v,
    ctrlpts,
    weight,
    point,
    basis,
):
    """All-span projected Gauss--Newton closest point for explicit contact."""
    degree_u = ti.static(basis.basis_u.degree)
    degree_v = ti.static(basis.basis_v.degree)
    num_ctrlpts_u = num_knot_u - degree_u - 1
    num_ctrlpts_v = num_knot_v - degree_v - 1
    best_u = knot_vector_u[start_knot_u + degree_u]
    best_v = knot_vector_v[start_knot_v + degree_v]
    best_distance2 = ti.math.inf
    best_pointer = ti.Vector.zero(ti.f64, 3)
    for span_u in range(degree_u, num_ctrlpts_u):
        lower_u = knot_vector_u[start_knot_u + span_u]
        upper_u = knot_vector_u[start_knot_u + span_u + 1]
        if upper_u > lower_u:
            for span_v in range(degree_v, num_ctrlpts_v):
                lower_v = knot_vector_v[start_knot_v + span_v]
                upper_v = knot_vector_v[start_knot_v + span_v + 1]
                if upper_v > lower_v:
                    u = 0.5 * (lower_u + upper_u)
                    v = 0.5 * (lower_v + upper_v)
                    iteration = 0
                    converged = 0
                    while iteration < 25 and converged == 0:
                        position, tangent_u, tangent_v = basis.NurbsBasisInterpolationsDers2d(
                            start_knot_u,
                            start_knot_v,
                            start_ctrlpt,
                            num_knot_u,
                            num_knot_v,
                            u,
                            v,
                            knot_vector_u,
                            knot_vector_v,
                            ctrlpts,
                            weight,
                        )
                        residual = position - point
                        gradient_u = residual.dot(tangent_u)
                        gradient_v = residual.dot(tangent_v)
                        hessian_uu = tangent_u.dot(tangent_u)
                        hessian_uv = tangent_u.dot(tangent_v)
                        hessian_vv = tangent_v.dot(tangent_v)
                        determinant = hessian_uu * hessian_vv - hessian_uv * hessian_uv
                        du = 0.0
                        dv = 0.0
                        if ti.abs(determinant) > 1.0e-24:
                            du = (-hessian_vv * gradient_u + hessian_uv * gradient_v) / determinant
                            dv = (hessian_uv * gradient_u - hessian_uu * gradient_v) / determinant
                        else:
                            scale = ti.max(
                                tangent_u.norm_sqr() + tangent_v.norm_sqr(),
                                1.0e-24,
                            )
                            du = -gradient_u / scale
                            dv = -gradient_v / scale
                        next_u = ti.min(upper_u, ti.max(lower_u, u + du))
                        next_v = ti.min(upper_v, ti.max(lower_v, v + dv))
                        parameter_step2 = (next_u - u) * (next_u - u) + (next_v - v) * (next_v - v)
                        u = next_u
                        v = next_v
                        if parameter_step2 < 1.0e-24:
                            converged = 1
                        iteration += 1
                    position = basis.NurbsBasisInterpolations2d(
                        start_knot_u,
                        start_knot_v,
                        start_ctrlpt,
                        num_knot_u,
                        num_knot_v,
                        u,
                        v,
                        knot_vector_u,
                        knot_vector_v,
                        ctrlpts,
                        weight,
                    )
                    pointer = position - point
                    distance2 = pointer.norm_sqr()
                    if distance2 < best_distance2:
                        best_u = u
                        best_v = v
                        best_distance2 = distance2
                        best_pointer = pointer
    return best_u, best_v, ti.sqrt(ti.max(best_distance2, 0.0)), best_pointer


@ti.dataclass
class PointNurbsDEMContact:
    active: ti.i32
    particle_id: ti.i32
    surface_id: ti.i32
    gap: ti.f64
    knot_value: vec2d
    normal: vec3d
    tangential_overlap: vec3d
    normal_force: vec3d
    tangential_force: vec3d


@ti.data_oriented
class ExplicitNurbsContact:
    """Stable point--surface table with current-control-hull AABB culling."""

    def __init__(self, iga, mpm_scene, model, **kwargs):
        if config.DIM != 3:
            raise RuntimeError("explicit DEM-law IGA-MPM contact currently requires dimension=3")
        self.iga = iga
        self.mpm_scene = mpm_scene
        self.model = model
        self.contact_surface = CouplingContactSurface(iga, **kwargs)
        self.num_surfaces = int(self.contact_surface.num_surfaces)
        canonical_basis = {}
        self.surface_basis = []
        for basis in self.contact_surface.basis:
            signature = (int(basis.basis_u.degree), int(basis.basis_v.degree))
            canonical_basis.setdefault(signature, basis)
            self.surface_basis.append(canonical_basis[signature])
        self.particle_count = int(mpm_scene.couplingNum[0])
        if self.num_surfaces <= 0:
            raise RuntimeError("explicit IGA-MPM found no selected IGA boundaries")
        if self.particle_count <= 0:
            raise RuntimeError("explicit IGA-MPM requires coupled MPM points")
        weights = self.contact_surface.weights.to_numpy()[: self.contact_surface.total_ctrlpts]
        if np.any(weights <= 0.0):
            raise ValueError("explicit NURBS contact AABB culling requires positive weights")
        particle_radius = np.asarray(
            mpm_scene.particle.rad.to_numpy()[: self.particle_count],
            dtype=np.float64,
        )
        if not np.isfinite(particle_radius).all() or np.any(particle_radius <= 0.0):
            raise ValueError("explicit IGA-MPM requires finite positive MPM contact radii")

        body_ids = np.asarray(
            [surface_key[0] for surface_key in self.contact_surface.surface_keys],
            dtype=np.int32,
        )
        self.max_iga_body_num = max(
            int(iga.patch.primitive.num_primitives),
            int(body_ids.max()) + 1 if body_ids.size else 1,
        )
        particle_materials = np.asarray(
            mpm_scene.particle.materialID.to_numpy()[: self.particle_count],
            dtype=np.int32,
        )
        active_pairs = {
            (int(material), int(body)) for material in np.unique(particle_materials) for body in np.unique(body_ids)
        }
        registered_pairs = {(int(material), int(body)) for material, body, _ in model.pending_properties}
        missing_pairs = sorted(active_pairs - registered_pairs)
        if missing_pairs:
            rendered = ", ".join(f"({material}, {body})" for material, body in missing_pairs)
            raise RuntimeError("missing explicit IGA-MPM contact properties for " + rendered)
        max_mpm_material_num = int(
            getattr(getattr(mpm_scene, "material", None), "max_material_num", 0)
            or getattr(getattr(mpm_scene, "material", None), "material_num", 0)
            or getattr(getattr(mpm_scene, "material", None), "matProps", np.empty(1)).shape[0]
        )
        model.initialize(max_mpm_material_num, self.max_iga_body_num)
        self.reset_contact_energy = model.reset_instantaneous_energy if GlobalVariable.TRACKENERGY else no_operation
        self.read_contact_energy = (
            model.energy_diagnostics if GlobalVariable.TRACKENERGY else self._empty_energy_diagnostics
        )

        self.surface_body = ti.field(ti.i32, shape=self.num_surfaces)
        self.surface_body.from_numpy(body_ids)
        self.surface_lower = ti.Vector.field(3, ti.f64, shape=self.num_surfaces)
        self.surface_upper = ti.Vector.field(3, ti.f64, shape=self.num_surfaces)
        self.contacts = PointNurbsDEMContact.field(shape=self.particle_count * self.num_surfaces)
        self.active_contact_count = ti.field(ti.i32, shape=())
        self.update_geometry()
        self.reset_contact_state()

    @ti.kernel
    def reset_contact_state(self):
        self.active_contact_count[None] = 0
        for contact_id in self.contacts:
            self.contacts[contact_id].active = 0
            self.contacts[contact_id].particle_id = contact_id // ti.static(self.num_surfaces)
            self.contacts[contact_id].surface_id = contact_id % ti.static(self.num_surfaces)
            self.contacts[contact_id].gap = ti.math.inf
            self.contacts[contact_id].knot_value = ti.Vector.zero(ti.f64, 2)
            self.contacts[contact_id].normal = ti.Vector.zero(ti.f64, 3)
            self.contacts[contact_id].tangential_overlap = ti.Vector.zero(ti.f64, 3)
            self.contacts[contact_id].normal_force = ti.Vector.zero(ti.f64, 3)
            self.contacts[contact_id].tangential_force = ti.Vector.zero(ti.f64, 3)

    @ti.kernel
    def clear_step_state(self):
        self.active_contact_count[None] = 0
        for contact_id in self.contacts:
            self.contacts[contact_id].active = 0
            self.contacts[contact_id].gap = ti.math.inf
            self.contacts[contact_id].normal = ti.Vector.zero(ti.f64, 3)
            self.contacts[contact_id].normal_force = ti.Vector.zero(ti.f64, 3)
            self.contacts[contact_id].tangential_force = ti.Vector.zero(ti.f64, 3)

    @ti.kernel
    def reset_surface_bounds(self):
        for surface_id in range(self.num_surfaces):
            self.surface_lower[surface_id] = ti.Vector([ti.math.inf, ti.math.inf, ti.math.inf])
            self.surface_upper[surface_id] = ti.Vector([-ti.math.inf, -ti.math.inf, -ti.math.inf])

    @ti.kernel
    def accumulate_surface_bounds(self, surface_id: ti.i32, begin: ti.i32, end: ti.i32):
        for local_control_id in range(begin, end):
            point = self.contact_surface.control_points_hat[local_control_id]
            for direction in ti.static(range(3)):
                ti.atomic_min(self.surface_lower[surface_id][direction], point[direction])
                ti.atomic_max(self.surface_upper[surface_id][direction], point[direction])

    def update_geometry(self):
        self.contact_surface.update_from_patch(self.iga.patch.control_points)
        self.reset_surface_bounds()
        for surface_id in range(self.num_surfaces):
            self.accumulate_surface_bounds(
                surface_id,
                int(self.contact_surface.prefix_num_ctrlpts[surface_id]),
                int(self.contact_surface.prefix_num_ctrlpts[surface_id + 1]),
            )

    @ti.kernel
    def set_particle_contact_enabled(self, enabled: ti.i32):
        for particle_id in range(self.particle_count):
            self.mpm_scene.particle[particle_id].coupling = ti.cast(enabled, ti.u8)

    @ti.kernel
    def project_surface(
        self,
        surface_id: ti.i32,
        prefix_num_knot_u: ti.i32,
        prefix_num_knot_v: ti.i32,
        prefix_num_ctrlpts: ti.i32,
        num_knot_u: ti.i32,
        num_knot_v: ti.i32,
        num_ctrlpts_u: ti.i32,
        surface: ti.template(),
        basis: ti.template(),
        particles: ti.template(),
    ):
        for particle_id in range(self.particle_count):
            contact_id = particle_id * ti.static(self.num_surfaces) + surface_id
            particle = particles[particle_id]
            radius = ti.cast(particle.rad, ti.f64)
            position = ti.cast(particle.x, ti.f64)
            inside_aabb = particle.active != 0 and particle.coupling != 0
            for direction in ti.static(range(3)):
                inside_aabb = (
                    inside_aabb
                    and position[direction] >= self.surface_lower[surface_id][direction] - radius
                    and position[direction] <= self.surface_upper[surface_id][direction] + radius
                )
            if inside_aabb:
                uknot, vknot, distance, pointer = _closest_point_on_surface(
                    prefix_num_knot_u,
                    prefix_num_knot_v,
                    prefix_num_ctrlpts,
                    num_knot_u,
                    num_knot_v,
                    surface.knot_vector_u,
                    surface.knot_vector_v,
                    surface.control_points_hat,
                    surface.weights,
                    position,
                    basis,
                )
                gap = distance - radius
                if gap < 0.0:
                    normal = ti.Vector.zero(ti.f64, 3)
                    if distance > 1.0e-14:
                        normal = -pointer / distance
                    self.contacts[contact_id].active = 1
                    self.contacts[contact_id].gap = gap
                    self.contacts[contact_id].knot_value = ti.Vector([uknot, vknot])
                    self.contacts[contact_id].normal = normal
                    ti.atomic_add(self.active_contact_count[None], 1)
                else:
                    self.contacts[contact_id].tangential_overlap = ti.Vector.zero(ti.f64, 3)
            else:
                self.contacts[contact_id].tangential_overlap = ti.Vector.zero(ti.f64, 3)

    @ti.kernel
    def resolve_surface_force(
        self,
        surface_id: ti.i32,
        prefix_num_knot_u: ti.i32,
        prefix_num_knot_v: ti.i32,
        prefix_num_ctrlpts: ti.i32,
        num_knot_u: ti.i32,
        num_knot_v: ti.i32,
        num_ctrlpts_u: ti.i32,
        surface: ti.template(),
        basis: ti.template(),
        properties: ti.template(),
        active_properties: ti.template(),
        dt: ti.template(),
        particles: ti.template(),
        iga_velocity: ti.template(),
        iga_rhs: ti.template(),
    ):
        for particle_id in range(self.particle_count):
            contact_id = particle_id * ti.static(self.num_surfaces) + surface_id
            if self.contacts[contact_id].active != 0:
                particle = particles[particle_id]
                radius = particle.rad
                uknot = self.contacts[contact_id].knot_value[0]
                vknot = self.contacts[contact_id].knot_value[1]
                gap = self.contacts[contact_id].gap
                body_id = self.surface_body[surface_id]
                property_id = ti.cast(particle.materialID, ti.i32) * ti.static(self.max_iga_body_num) + body_id
                assert active_properties[property_id] != 0, "IGA-MPM DEM contact property is missing"
                num_ctrlpts_v = num_knot_v - basis.basis_v.degree - 1
                span_u = basis.basis_u.FindSpan(
                    prefix_num_knot_u,
                    num_ctrlpts_u,
                    uknot,
                    surface.knot_vector_u,
                )
                span_v = basis.basis_v.FindSpan(
                    prefix_num_knot_v,
                    num_ctrlpts_v,
                    vknot,
                    surface.knot_vector_v,
                )
                nshape = basis.NurbsBasis2d(
                    prefix_num_knot_u,
                    prefix_num_knot_v,
                    prefix_num_ctrlpts,
                    num_knot_u,
                    num_knot_v,
                    uknot,
                    vknot,
                    surface.knot_vector_u,
                    surface.knot_vector_v,
                    surface.weights,
                )
                surface_velocity = ti.Vector.zero(ti.f64, 3)
                for i in range(span_v - basis.basis_v.degree, span_v + 1):
                    for j in range(span_u - basis.basis_u.degree, span_u + 1):
                        local_offset = (j - span_u + basis.basis_u.degree) + (i - span_v + basis.basis_v.degree) * (
                            basis.basis_u.degree + 1
                        )
                        local_control_id = prefix_num_ctrlpts + j + i * num_ctrlpts_u
                        global_control_id = surface.control_points_id[local_control_id]
                        surface_velocity += nshape[local_offset] * iga_velocity[global_control_id]

                relative_velocity = ti.cast(particle.v, ti.f64) - surface_velocity
                normal = self.contacts[contact_id].normal
                if normal.norm_sqr() < 1.0e-28:
                    _, tangent_u, tangent_v = basis.NurbsBasisInterpolationsDers2d(
                        prefix_num_knot_u,
                        prefix_num_knot_v,
                        prefix_num_ctrlpts,
                        num_knot_u,
                        num_knot_v,
                        uknot,
                        vknot,
                        surface.knot_vector_u,
                        surface.knot_vector_v,
                        surface.control_points_hat,
                        surface.weights,
                    )
                    normal = tangent_u.cross(tangent_v).normalized()
                    if relative_velocity.dot(normal) > 0.0:
                        normal = -normal
                    self.contacts[contact_id].normal = normal

                normal_force, tangential_force, _, overlap = properties[property_id]._force_assemble(
                    particle.m,
                    radius,
                    gap,
                    1.0,
                    radius,
                    ti.cast(normal, float),
                    ti.cast(relative_velocity, float),
                    ti.Vector.zero(float, 3),
                    ti.cast(
                        self.contacts[contact_id].tangential_overlap,
                        float,
                    ),
                    dt,
                )
                total_force = ti.cast(normal_force + tangential_force, ti.f64)
                for direction in ti.static(range(3)):
                    ti.atomic_add(
                        particles[particle_id].external_force[direction],
                        total_force[direction],
                    )
                for i in range(span_v - basis.basis_v.degree, span_v + 1):
                    for j in range(span_u - basis.basis_u.degree, span_u + 1):
                        local_offset = (j - span_u + basis.basis_u.degree) + (i - span_v + basis.basis_v.degree) * (
                            basis.basis_u.degree + 1
                        )
                        local_control_id = prefix_num_ctrlpts + j + i * num_ctrlpts_u
                        global_control_id = surface.control_points_id[local_control_id]
                        for direction in ti.static(range(3)):
                            ti.atomic_add(
                                iga_rhs[3 * global_control_id + direction],
                                -nshape[local_offset] * total_force[direction],
                            )
                self.contacts[contact_id].tangential_overlap = ti.cast(overlap, ti.f64)
                self.contacts[contact_id].normal_force = ti.cast(normal_force, ti.f64)
                self.contacts[contact_id].tangential_force = ti.cast(tangential_force, ti.f64)

    def resolve(self, dt):
        self.reset_contact_energy()
        self.clear_step_state()
        self.update_geometry()
        for surface_id in range(self.num_surfaces):
            arguments = (
                surface_id,
                int(self.contact_surface.prefix_num_knot_u[surface_id]),
                int(self.contact_surface.prefix_num_knot_v[surface_id]),
                int(self.contact_surface.prefix_num_ctrlpts[surface_id]),
                int(self.contact_surface.num_knot_u[surface_id + 1]),
                int(self.contact_surface.num_knot_v[surface_id + 1]),
                int(self.contact_surface.num_ctrlpts_u[surface_id + 1]),
                self.contact_surface,
                self.surface_basis[surface_id],
            )
            self.project_surface(
                *arguments,
                self.mpm_scene.particle,
            )
            self.resolve_surface_force(
                *arguments,
                self.model.surface_properties,
                self.model.active_properties,
                dt,
                self.mpm_scene.particle,
                self.iga.patch.velocitys,
                self.iga.rhs,
            )
        return int(self.active_contact_count[None])

    @staticmethod
    def _empty_energy_diagnostics():
        return {}

    def compile_kernels(self, dt):
        """JIT contact specializations without changing physical state."""
        self.set_particle_contact_enabled(0)
        try:
            self.resolve(dt)
        finally:
            self.set_particle_contact_enabled(1)
            self.reset_contact_state()
            self.iga.rhs.fill(0.0)


__all__ = ["ExplicitNurbsContact", "PointNurbsDEMContact"]
