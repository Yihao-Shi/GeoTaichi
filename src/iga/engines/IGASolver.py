import taichi as ti
import numpy as np

import src.iga.config as config
from src.iga.generator.Primitives import Primitives
from src.iga.elements.ContactSurface import ContactSurface
from src.iga.elements.Patch import Patch
from src.iga.elements.Element import Element
from src.physics_model.consititutive_model.finite_strain.NeoHookean import NeoHookeanModel
from src.utils.SolverConsole import print_save_file_info
from src.utils.SolverRuntime import StepSchedule
from src.utils.TimeTicker import Timer
from src.iga.boundaries.BoundaryCondition import DirichletBoundary, NeumannBoundary
from src.iga.engines.EngineUtils import (
    jacobian2parent2parametric1d,
    jacobian2parent2parametric2d,
    linearize,
    vectorize_id,
)


@ti.data_oriented
class IGASolver:
    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = float(value)
        if hasattr(self, "TIdt"):
            self.TIdt[None] = self._dt

    def __init__(
        self,
        primitives: Primitives,
        dirichlet: DirichletBoundary = None,
        neumann: NeumannBoundary = None,
        name="case",
        **kwargs,
    ):
        self.TIdt = ti.field(ti.f64, shape=())
        self.dt = kwargs.get("dt", 1e-2)
        self.material = NeoHookeanModel().initialize_from_kwargs(**kwargs)
        self.is_axisymmetric = bool(
            kwargs.get(
                "axisymmetric",
                kwargs.get("is_axisymmetric", kwargs.get("is_2DAxisy", False)),
            )
        )
        self.axis_offset = float(kwargs.get("axis_offset", 0.0))
        if self.is_axisymmetric and config.DIM != 2:
            raise ValueError("axisymmetric IGA requires dimension=2")
        if not np.isfinite(self.axis_offset):
            raise ValueError("axis_offset must be finite")
        self.material_dimension = 3 if self.is_axisymmetric else config.DIM
        gravity = kwargs.get("gravity", [0.0, 0.0, -9.8])
        self.gravity = list(gravity[: config.DIM]) + [0.0] * max(0, config.DIM - len(gravity))
        self.output_interval = int(kwargs.get("interval", 1))
        self.total_step = int(kwargs.get("step", 100))
        if self.output_interval <= 0 or self.total_step < 0:
            raise ValueError("IGA interval must be positive and step cannot be negative")
        self.track_energy = bool(kwargs.get("track_energy", False))
        self.step_schedule = StepSchedule.from_options(kwargs, output_interval=self.output_interval)
        self.path = kwargs.get("path", "IGAData_" + name)
        self.output_count = 0
        self.time = 0.0
        self.step_count = 0
        self.history = []
        self.last_failure = None
        self.timer = Timer()
        self.compile_seconds = None

        self.patch = Patch(primitives)
        self.patch.add_patches(self.gravity, rest_shape=kwargs.get("rest_shape"))
        self.patch.finalize()
        self.element = Element(kwargs.get("degree", [2, 2, 2]))
        self.degree_of_freedom = int(config.DIM * self.patch.primitive.num_ctrlpts)
        total_elements = int(np.sum(self.patch.total_num_element[1:]))
        self.initial_deformation_gradients = ti.Matrix.field(
            self.material_dimension,
            self.material_dimension,
            ti.f64,
            shape=(total_elements, self.element.gauss_number),
        )
        self.initial_deformation_gradient = self.initial_deformation_gradients
        self.reference_mapping_status = ti.field(ti.i32, shape=())

        self.dirichlet = dirichlet
        self.neumann = neumann
        if self.dirichlet is not None:
            self.dirichlet.finalize(self.degree_of_freedom)
        else:
            self.dirichlet = DirichletBoundary()
        if self.neumann is not None:
            self.neumann.finalize()
        else:
            self.neumann = NeumannBoundary()

    def build_contact_surface(self):
        self.contact_surface = ContactSurface(self.patch)

    @ti.func
    def calculate_jacobian(self, dNdnat, rest_ctrl_coords):
        jacobian = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
        for j in range(self.element.total_knot_range):
            for a in ti.static(range(config.DIM)):
                for b in ti.static(range(config.DIM)):
                    jacobian[a, b] += rest_ctrl_coords[j, a] * dNdnat[j, b]
        return jacobian

    @ti.func
    def require_positive_reference_jacobian(self, jacobian):
        determinant = jacobian.determinant()
        if not (determinant > 0.0):
            ti.atomic_max(self.reference_mapping_status[None], 1)
        assert determinant > 0.0, "IGA reference mapping requires a finite positive Jacobian"
        return determinant

    @ti.func
    def require_positive_reference_radius(self, reference_radius):
        if not (reference_radius > 0.0):
            ti.atomic_max(self.reference_mapping_status[None], 2)
        assert reference_radius > 0.0, "axisymmetric IGA quadrature requires a finite positive radius"
        return reference_radius

    @ti.func
    def _deformation_gradient(
        self,
        N,
        dNdnat,
        dnatdX,
        rest_ctrl_coords,
        current_ctrl_coords,
    ):
        deformation_gradient = ti.Matrix.identity(ti.f64, self.material_dimension)
        if ti.static(self.is_axisymmetric):
            reference_radius = self.element.interpolate_component(N, rest_ctrl_coords, 0) - ti.static(self.axis_offset)
            self.require_positive_reference_radius(reference_radius)
            shape_gradients = self.element.compute_shape_gradients(dNdnat, dnatdX)
            deformation_gradient = self.element.compute_axisymmetric_deformation_gradient(
                N,
                shape_gradients,
                rest_ctrl_coords,
                current_ctrl_coords,
                ti.static(self.axis_offset),
            )
        else:
            deformation_gradient = self.element.compute_deformation_gradient(dNdnat, dnatdX, current_ctrl_coords)
        return deformation_gradient

    @ti.kernel
    def compute_deformation_gradient(
        self,
        total_num_element: ti.i32,
        prefix_num_knot: ti.types.vector(config.DIM, ti.i32),
        prefix_num_element: ti.types.vector(config.DIM, ti.i32),
        prefix_num_ctrlpts: int,
        num_knot: ti.types.vector(config.DIM, ti.i32),
        num_element: ti.types.vector(config.DIM, ti.i32),
        num_ctrlpts: ti.types.vector(config.DIM, ti.i32),
        deformation_gradient: ti.template(),
    ):
        for ele in range(total_num_element):
            eleid = vectorize_id(ele, num_element)
            elrange_u = self.patch.element_u[prefix_num_element[0] + eleid[0]]
            elrange_v = self.patch.element_v[prefix_num_element[1] + eleid[1]]
            elrange_w = ti.Vector.zero(ti.f64, 2)
            if ti.static(config.DIM == 3):
                elrange_w = self.patch.element_w[prefix_num_element[2] + eleid[2]]

            rest_ctrl_coords = ti.Matrix.zero(ti.f64, self.element.total_knot_range, config.DIM)
            current_ctrl_coords = ti.Matrix.zero(ti.f64, self.element.total_knot_range, config.DIM)
            for nodeID in ti.grouped(ti.ndrange(*self.element.knot_range)):
                global_nodeID = nodeID + eleid
                local_offset = linearize(nodeID, self.element.knot_range)
                global_offset = linearize(global_nodeID, num_ctrlpts)
                for j in ti.static(range(config.DIM)):
                    rest_ctrl_coords[local_offset, j] = self.patch.rest_control_points[
                        prefix_num_ctrlpts + global_offset
                    ][j]
                    current_ctrl_coords[local_offset, j] = self.patch.control_points[
                        prefix_num_ctrlpts + global_offset
                    ][j]

            for gauss_id in range(self.element.gauss_number):
                N, dNdnat = self.element.dshapefn(
                    gauss_id, elrange_u, elrange_v, elrange_w, prefix_num_knot, prefix_num_ctrlpts, num_knot, self.patch
                )
                jacobian = self.calculate_jacobian(dNdnat, rest_ctrl_coords)
                self.require_positive_reference_jacobian(jacobian)
                dnatdX = jacobian.inverse()
                deformation_gradient[ele, gauss_id] = self._deformation_gradient(
                    N,
                    dNdnat,
                    dnatdX,
                    rest_ctrl_coords,
                    current_ctrl_coords,
                )

    @ti.kernel
    def precompute_deformation_gradient(
        self,
        total_num_element: ti.i32,
        prefix_total_num_element: ti.i32,
        prefix_num_knot: ti.types.vector(config.DIM, ti.i32),
        prefix_num_element: ti.types.vector(config.DIM, ti.i32),
        prefix_num_ctrlpts: int,
        num_knot: ti.types.vector(config.DIM, ti.i32),
        num_element: ti.types.vector(config.DIM, ti.i32),
        num_ctrlpts: ti.types.vector(config.DIM, ti.i32),
    ):
        for ele in range(total_num_element):
            eleid = vectorize_id(ele, num_element)
            elrange_u = self.patch.element_u[prefix_num_element[0] + eleid[0]]
            elrange_v = self.patch.element_v[prefix_num_element[1] + eleid[1]]
            elrange_w = ti.Vector.zero(ti.f64, 2)
            if ti.static(config.DIM == 3):
                elrange_w = self.patch.element_w[prefix_num_element[2] + eleid[2]]

            rest_ctrl_coords = ti.Matrix.zero(ti.f64, self.element.total_knot_range, config.DIM)
            current_ctrl_coords = ti.Matrix.zero(ti.f64, self.element.total_knot_range, config.DIM)
            for node_id in ti.grouped(ti.ndrange(*self.element.knot_range)):
                global_node_id = node_id + eleid
                local_offset = linearize(node_id, self.element.knot_range)
                global_offset = linearize(global_node_id, num_ctrlpts)
                control_point_id = prefix_num_ctrlpts + global_offset
                for direction in ti.static(range(config.DIM)):
                    rest_ctrl_coords[local_offset, direction] = self.patch.rest_control_points[control_point_id][
                        direction
                    ]
                    current_ctrl_coords[local_offset, direction] = self.patch.control_points[control_point_id][
                        direction
                    ]

            for gauss_id in range(self.element.gauss_number):
                N, dNdnat = self.element.dshapefn(
                    gauss_id,
                    elrange_u,
                    elrange_v,
                    elrange_w,
                    prefix_num_knot,
                    prefix_num_ctrlpts,
                    num_knot,
                    self.patch,
                )
                jacobian = self.calculate_jacobian(dNdnat, rest_ctrl_coords)
                self.require_positive_reference_jacobian(jacobian)
                self.initial_deformation_gradients[prefix_total_num_element + ele, gauss_id] = (
                    self._deformation_gradient(
                        N,
                        dNdnat,
                        jacobian.inverse(),
                        rest_ctrl_coords,
                        current_ctrl_coords,
                    )
                )

    @ti.kernel
    def compute_nodal_volume(
        self,
        total_num_element: ti.i32,
        prefix_num_ctrlpts: ti.i32,
        prefix_num_knot: ti.types.vector(config.DIM, ti.i32),
        prefix_num_element: ti.types.vector(config.DIM, ti.i32),
        num_knot: ti.types.vector(config.DIM, ti.i32),
        num_element: ti.types.vector(config.DIM, ti.i32),
        num_ctrlpts: ti.types.vector(config.DIM, ti.i32),
    ):
        for ele in range(total_num_element):
            eleid = vectorize_id(ele, num_element)
            elrange_u = self.patch.element_u[prefix_num_element[0] + eleid[0]]
            elrange_v = self.patch.element_v[prefix_num_element[1] + eleid[1]]
            elrange_w = ti.Vector.zero(ti.f64, 2)
            j2 = jacobian2parent2parametric2d(elrange_u, elrange_v)
            if ti.static(config.DIM == 3):
                elrange_w = self.patch.element_w[prefix_num_element[2] + eleid[2]]
                j2 *= jacobian2parent2parametric1d(elrange_w)

            rest_ctrl_coords = ti.Matrix.zero(ti.f64, self.element.total_knot_range, config.DIM)
            for nodeID in ti.grouped(ti.ndrange(*self.element.knot_range)):
                global_nodeID = nodeID + eleid
                local_offset = linearize(nodeID, self.element.knot_range)
                global_offset = linearize(global_nodeID, num_ctrlpts)
                for j in ti.static(range(config.DIM)):
                    rest_ctrl_coords[local_offset, j] = self.patch.rest_control_points[
                        prefix_num_ctrlpts + global_offset
                    ][j]

            for gauss_id in range(self.element.gauss_number):
                N, dNdnat = self.element.dshapefn(
                    gauss_id, elrange_u, elrange_v, elrange_w, prefix_num_knot, prefix_num_ctrlpts, num_knot, self.patch
                )
                jacobian = self.calculate_jacobian(dNdnat, rest_ctrl_coords)
                j1 = self.require_positive_reference_jacobian(jacobian)
                volume = j1 * j2 * self.element.gauss_weights[gauss_id]
                if ti.static(self.is_axisymmetric):
                    reference_radius = self.element.interpolate_component(N, rest_ctrl_coords, 0) - ti.static(
                        self.axis_offset
                    )
                    reference_radius = self.require_positive_reference_radius(reference_radius)
                    volume *= 2.0 * ti.math.pi * reference_radius

                for nodeID in ti.grouped(ti.ndrange(*self.element.knot_range)):
                    global_nodeID = nodeID + eleid
                    local_offset = linearize(nodeID, self.element.knot_range)
                    global_offset = linearize(global_nodeID, num_ctrlpts)
                    self.patch.volume[prefix_num_ctrlpts + global_offset] += N[local_offset] * volume

    @ti.kernel
    def calculate_vonmises_stress(
        self,
        total_num_element: ti.i32,
        prefix_total_num_ctrlpts: ti.i32,
        prefix_num_knot: ti.types.vector(config.DIM, ti.i32),
        prefix_num_element: ti.types.vector(config.DIM, ti.i32),
        num_knot: ti.types.vector(config.DIM, ti.i32),
        num_element: ti.types.vector(config.DIM, ti.i32),
        num_ctrlpts: ti.types.vector(config.DIM, ti.i32),
    ):
        for ele in range(total_num_element):
            eleid = vectorize_id(ele, num_element)
            elrange_u = self.patch.element_u[prefix_num_element[0] + eleid[0]]
            elrange_v = self.patch.element_v[prefix_num_element[1] + eleid[1]]
            elrange_w = ti.Vector.zero(ti.f64, 2)
            j2 = jacobian2parent2parametric2d(elrange_u, elrange_v)
            if ti.static(config.DIM == 3):
                elrange_w = self.patch.element_w[prefix_num_element[2] + eleid[2]]
                j2 *= jacobian2parent2parametric1d(elrange_w)

            rest_ctrl_coords = ti.Matrix.zero(ti.f64, self.element.total_knot_range, config.DIM)
            current_ctrl_coords = ti.Matrix.zero(ti.f64, self.element.total_knot_range, config.DIM)
            for nodeID in ti.grouped(ti.ndrange(*self.element.knot_range)):
                global_nodeID = nodeID + eleid
                offset = linearize(nodeID, self.element.knot_range)
                global_offset = linearize(global_nodeID, num_ctrlpts)
                for j in ti.static(range(config.DIM)):
                    rest_ctrl_coords[offset, j] = self.patch.rest_control_points[
                        prefix_total_num_ctrlpts + global_offset
                    ][j]
                    current_ctrl_coords[offset, j] = self.patch.control_points[
                        prefix_total_num_ctrlpts + global_offset
                    ][j]

            for gauss_id in range(self.element.gauss_number):
                N, dNdnat = self.element.dshapefn(
                    gauss_id,
                    elrange_u,
                    elrange_v,
                    elrange_w,
                    prefix_num_knot,
                    prefix_total_num_ctrlpts,
                    num_knot,
                    self.patch,
                )
                jacobian = self.calculate_jacobian(dNdnat, rest_ctrl_coords)
                j1 = self.require_positive_reference_jacobian(jacobian)
                dnatdX = jacobian.inverse()
                deformation_gradient = self._deformation_gradient(
                    N,
                    dNdnat,
                    dnatdX,
                    rest_ctrl_coords,
                    current_ctrl_coords,
                )
                stress = self.material.VonMises(deformation_gradient)
                volume = j1 * j2 * self.element.gauss_weights[gauss_id]
                if ti.static(self.is_axisymmetric):
                    reference_radius = self.element.interpolate_component(N, rest_ctrl_coords, 0) - ti.static(
                        self.axis_offset
                    )
                    reference_radius = self.require_positive_reference_radius(reference_radius)
                    volume *= 2.0 * ti.math.pi * reference_radius

                for nodeID in ti.grouped(ti.ndrange(*self.element.knot_range)):
                    global_nodeID = nodeID + eleid
                    local_offset = linearize(nodeID, self.element.knot_range)
                    global_offset = linearize(global_nodeID, num_ctrlpts)
                    self.patch.stress[prefix_total_num_ctrlpts + global_offset] += N[local_offset] * stress * volume

    @ti.kernel
    def normalize_vonmises_stress(self, begin: ti.i32, count: ti.i32, cutoff: ti.f64):
        for i in range(count):
            node = begin + i
            volume = self.patch.volume[node]
            stress = self.patch.stress[node] / volume if volume > 0.0 else 0.0
            self.patch.stress[node] = stress if stress >= cutoff else 0.0

    def precompute(self):
        self.patch.volume.fill(0)
        self.reference_mapping_status[None] = 0
        for patch_id in range(self.patch.primitive.num_primitives):
            prefix_num_knot = self.patch.prefix_num_knot[patch_id]
            prefix_num_element = self.patch.prefix_num_element[patch_id]
            prefix_num_ctrlpts = self.patch.prefix_total_num_ctrlpts[patch_id]
            num_knot = self.patch.num_knot[patch_id + 1]
            num_element = self.patch.num_element[patch_id + 1]
            num_ctrlpts = self.patch.num_ctrlpts[patch_id + 1]
            total_num_element = self.patch.total_num_element[patch_id + 1]
            self.compute_nodal_volume(
                total_num_element,
                prefix_num_ctrlpts,
                prefix_num_knot,
                prefix_num_element,
                num_knot,
                num_element,
                num_ctrlpts,
            )
            self.precompute_deformation_gradient(
                total_num_element,
                self.patch.prefix_total_num_element[patch_id],
                prefix_num_knot,
                prefix_num_element,
                prefix_num_ctrlpts,
                num_knot,
                num_element,
                num_ctrlpts,
            )
        if int(self.reference_mapping_status[None]) != 0:
            raise ValueError(
                "IGA reference geometry requires finite positive Jacobians "
                "and axisymmetric radii at all quadrature points"
            )

    def visualize_stress(self):
        self.patch.stress.fill(0)
        for patch_id in range(self.patch.primitive.num_primitives):
            prefix_num_knot = self.patch.prefix_num_knot[patch_id]
            prefix_num_element = self.patch.prefix_num_element[patch_id]
            prefix_total_num_ctrlpts = self.patch.prefix_total_num_ctrlpts[patch_id]
            num_knot = self.patch.num_knot[patch_id + 1]
            num_element = self.patch.num_element[patch_id + 1]
            num_ctrlpts = self.patch.num_ctrlpts[patch_id + 1]
            total_num_element = self.patch.total_num_element[patch_id + 1]
            self.calculate_vonmises_stress(
                total_num_element,
                prefix_total_num_ctrlpts,
                prefix_num_knot,
                prefix_num_element,
                num_knot,
                num_element,
                num_ctrlpts,
            )
            self.normalize_vonmises_stress(
                prefix_total_num_ctrlpts,
                self.patch.total_num_ctrlpts[patch_id + 1],
                1.0e-12 * max(1.0, float(self.material.young)),
            )

    def visualize(self, log=True):
        import os

        vtk_path = os.path.join(self.path, "vtks")
        if not os.path.exists(vtk_path):
            os.makedirs(vtk_path)
        self.patch.current_print = self.output_count
        self.patch.visualize(vtk_path)
        if log:
            print_save_file_info(
                "IGA",
                self.step_count,
                self.output_count,
                self.time,
                self.path,
            )
        self.output_count += 1
