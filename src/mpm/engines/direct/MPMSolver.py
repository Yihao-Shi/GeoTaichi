import taichi as ti
import numpy as np
import math
import time

import src.mpm.config as config
from src.mpm.generator.Body import Body
from src.mpm.boundaries.BoundaryCondition import DirichletBoundary, NeumannBoundary
from src.utils.ShapeFunctions import (
    ShapeLinear,
    GShapeLinear,
    HShapeLinear,
    ShapeGIMP,
    GShapeGIMP,
    ShapeBsplineQ,
    GShapeBsplineQ,
    HShapeBsplineQ,
)
from third_party.pyevtk.hl import pointsToVTK

from src.physics_model.contact_model.ipc.ContactMeasure import reference_point_measure
from src.utils.RuntimeHook import runtime_checkpoint
from src.utils.SolverRuntime import StepSchedule, normalize_callbacks
from src.utils.SolverConsole import print_save_file_info
from src.utils.TimeTicker import Timer
from src.utils.linalg import no_operation
from src.utils.StepRetry import (
    StepRetryPolicy,
    is_recoverable_nonlinear_failure,
    nonlinear_failure_kind,
)


class MPMConvergenceError(RuntimeError):
    """A recoverable failure before a Direct MPM step is accepted."""


@ti.data_oriented
class _LinearShapeFunction:
    def __init__(self):
        self.offset = 0.0
        self.max_node_per_particle_one_axis = 2
        self.max_node_per_particle = self.max_node_per_particle_one_axis**config.DIM
        self.influenced_node = self.max_node_per_particle * config.DIM

    @ti.func
    def shapefn(self, xp, xg, idx, lp):
        return ShapeLinear(xp, xg, idx, lp)

    @ti.func
    def dshapefn(self, xp, xg, idx, lp):
        return GShapeLinear(xp, xg, idx, lp)

    @ti.func
    def hshapefn(self, xp, xg, idx, lp):
        return HShapeLinear(xp, xg, idx, lp)


@ti.data_oriented
class _GIMPShapeFunction:
    def __init__(self):
        self.offset = 0.0
        self.max_node_per_particle_one_axis = 3
        self.max_node_per_particle = self.max_node_per_particle_one_axis**config.DIM
        self.influenced_node = self.max_node_per_particle * config.DIM

    @ti.func
    def shapefn(self, xp, xg, idx, lp):
        return ShapeGIMP(xp, xg, idx, lp)

    @ti.func
    def dshapefn(self, xp, xg, idx, lp):
        return GShapeGIMP(xp, xg, idx, lp)


@ti.data_oriented
class _QuadBSplineShapeFunction:
    def __init__(self):
        self.offset = 0.5
        self.max_node_per_particle_one_axis = 3
        self.max_node_per_particle = self.max_node_per_particle_one_axis**config.DIM
        self.influenced_node = self.max_node_per_particle * config.DIM

    @ti.func
    def shapefn(self, xp, xg, idx, btype):
        return ShapeBsplineQ(xp, xg, idx, btype)

    @ti.func
    def dshapefn(self, xp, xg, idx, btype):
        return GShapeBsplineQ(xp, xg, idx, btype)

    @ti.func
    def hshapefn(self, xp, xg, idx, btype):
        return HShapeBsplineQ(xp, xg, idx, btype)


def _make_shape_function(shape_function):
    key = str(shape_function).replace("-", "").replace("_", "").replace(" ", "").lower()
    if key in ("linear", "smoothlinear"):
        return _LinearShapeFunction(), "linear"
    if key == "gimp":
        return _GIMPShapeFunction(), "gimp"
    if key in ("bspline", "quadbspline", "quadraticbspline"):
        return _QuadBSplineShapeFunction(), "bspline"
    raise ValueError("shape_function must be one of: linear, gimp, bspline/QuadBSpline")


@ti.data_oriented
class MPMSolver:
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
        bodies: Body,
        dirichlet: DirichletBoundary = None,
        neumann: NeumannBoundary = None,
        name="case",
        solver="Implicit",
        **kwargs,
    ):
        self.val_lim = 1e-12
        self.dimension = config.DIM
        self.n_particles = 0
        self.total_surface_num = 0
        self.compute_traction = False
        self.assemble_traction_step = no_operation
        self.solver = solver
        self.velocity_proj = kwargs.get("velocity_projection", False)
        self.is_axisymmetric = bool(
            kwargs.get(
                "axisymmetric",
                kwargs.get("is_axisymmetric", kwargs.get("is_2DAxisy", False)),
            )
        )
        self.axis_offset = float(kwargs.get("axis_offset", 0.0))
        if self.is_axisymmetric and config.DIM != 2:
            raise ValueError("axisymmetric Direct MPM requires dimension=2")
        if not np.isfinite(self.axis_offset):
            raise ValueError("axis_offset must be finite")

        self.coeffPIC = kwargs.get("alphaPIC", 0.0)
        self.damping = kwargs.get("damping", 0.0)
        self.TIdt = ti.field(ti.f64, shape=())
        self.dt = kwargs.get("dt", 1e-2)
        self.gravity = kwargs.get("gravity")
        self.shape_func, self.shape_function_name = _make_shape_function(kwargs.get("shape_function", "linear"))

        self.vis = kwargs.get("visualize", True)
        self.output_interval = int(kwargs.get("interval", 1))
        self.total_step = int(kwargs.get("step", 100))
        if self.output_interval <= 0 or self.total_step < 0:
            raise ValueError("Direct MPM interval must be positive and step cannot be negative")
        self.track_energy = bool(kwargs.get("track_energy", False))
        self.step_schedule = StepSchedule.from_options(kwargs, output_interval=self.output_interval)
        self.path = kwargs.get("path", "MPMData_" + name)
        self.output_count = 0
        self.step_retry = StepRetryPolicy(
            enabled=kwargs.get("enable_step_retry", False),
            maximum_retries=kwargs.get("step_retry_max_retries", 2),
            reduction=kwargs.get("step_retry_reduction", 0.5),
            minimum_timestep=kwargs.get("step_retry_minimum_timestep", 0.0),
        )
        if self.step_retry.enabled and self.solver != "Implicit":
            raise ValueError("Direct MPM step retry is available only for implicit solvers")
        self.time = 0.0
        self.step_count = 0
        self.history = []
        self.timer = Timer()
        self.compile_seconds = None
        self.last_failure = None

        self.body_dtype = ti.types.struct(
            goffset=ti.i32,
            grid_num=ti.types.vector(config.DIM, ti.i32),
            grid_size=ti.f64,
            xmin=ti.types.vector(config.DIM, ti.f64),
            xmax=ti.types.vector(config.DIM, ti.f64),
        )

        self.particle_dtype = ti.types.struct(
            x=ti.types.vector(config.DIM, ti.f64),  # position
            v=ti.types.vector(config.DIM, ti.f64),  # velocity
            a=ti.types.vector(config.DIM, ti.f64),  # acceleration
            vol0=ti.f64,  # initial volume
            m=ti.f64,  # mass
            bodyID=ti.i32,  # body ID
        )

        self.grid_dtype = ti.types.struct(
            v=ti.types.vector(config.DIM, ti.f64),  # velocity
            m=ti.f64,  # mass
            a=ti.types.vector(config.DIM, ti.f64),  # acceleration
        )

        self.traction_dtype = ti.types.struct(particleID=ti.i32, traction=ti.types.vector(config.DIM, ti.f64))

        self.domain = kwargs.get("domain")
        self.dx = kwargs.get("dx")
        self.inv_dx = 1.0 / self.dx
        self.grid_num = [int(self.domain[d] * self.inv_dx) + 1 for d in range(config.DIM)]
        self.total_grid_num = math.prod(self.grid_num)
        self.total_background_grid_num = 0

        self.bodies = bodies
        self.n_body = self.bodies.body_counter
        self.n_particles = self.bodies.particle_counter
        self.add_body_info()

        self.dirichlet = dirichlet
        self.neumann = neumann
        if self.dirichlet is not None:
            self.dirichlet.finalize(config.DIM * self.total_background_grid_num)
        else:
            self.dirichlet = DirichletBoundary()
        if self.neumann is not None:
            self.neumann.finalize()
        else:
            self.neumann = NeumannBoundary()

        # body data
        self.body = self.body_dtype.field(shape=self.n_body)

        # particle data
        self.particleNum = ti.field(int, shape=1)
        if self.solver == "Explicit":
            self.particle_dtype.members.update({"stress": ti.types.matrix(3, 3, ti.f64)})
        self.particle = self.particle_dtype.field(shape=self.n_particles)
        if self.velocity_proj:
            self.gradv = ti.Matrix.field(config.DIM, config.DIM, dtype=ti.f64, shape=self.n_particles)
        self.tractionNum = ti.field(int, shape=1)
        self.traction = self.traction_dtype.field(shape=kwargs.get("traction_number", self.n_particles))

        # grid node data
        self.grid = self.grid_dtype.field(shape=int(self.total_background_grid_num))

        # shape function data
        self.offset = ti.field(int, shape=self.n_particles)  # number of nodes for each particle
        self.LnID = ti.field(
            int, shape=(self.n_particles, self.shape_func.max_node_per_particle)
        )  # linear index of nodes for each particle
        self.shape = ti.field(
            ti.f64, shape=(self.n_particles, self.shape_func.max_node_per_particle)
        )  # shape function value for each particle-node pair
        self.dshape = ti.Vector.field(
            config.DIM, ti.f64, shape=(self.n_particles, self.shape_func.max_node_per_particle)
        )  # shape function gradient for each particle-node pair
        self.invalid_stencil_particle = ti.field(ti.i32, shape=())
        self.init()

    def build_surface_node(self, all_particles=False):
        all_surface_ids = []
        all_surface_measures = []
        for name, body in self.bodies.bodies.items():
            surface_id = self.bodies.get_surface_ids(name, all_particles=all_particles)
            body["surface_id"] = surface_id
            all_surface_ids.append(surface_id)
            local_surface_id = surface_id - int(body["poffset"])
            explicit_measure = body.get("surface_measure", body.get("boundary_measure", None))
            if explicit_measure is None:
                volume = np.asarray(body["volume"], dtype=np.float64)
                if volume.ndim == 0:
                    measures = np.full(
                        surface_id.shape[0],
                        reference_point_measure(float(volume), config.DIM),
                        dtype=np.float64,
                    )
                else:
                    volume = volume.reshape(-1)
                    if volume.size != body["points"].shape[0]:
                        raise ValueError(f"{name}: particle volume must be scalar or have one " "entry per particle")
                    measures = np.power(
                        volume[local_surface_id],
                        (config.DIM - 1.0) / config.DIM,
                    )
            else:
                explicit_measure = np.asarray(explicit_measure, dtype=np.float64)
                if explicit_measure.ndim == 0:
                    measures = np.full(surface_id.shape[0], float(explicit_measure), dtype=np.float64)
                else:
                    explicit_measure = explicit_measure.reshape(-1)
                    if explicit_measure.size == body["points"].shape[0]:
                        measures = explicit_measure[local_surface_id]
                    elif explicit_measure.size == surface_id.shape[0]:
                        measures = explicit_measure
                    else:
                        raise ValueError(
                            f"{name}: surface_measure must be scalar, per-particle, " "or per-boundary-particle"
                        )
            if self.is_axisymmetric:
                reference_radius = np.asarray(body["points"], dtype=np.float64)[local_surface_id, 0] - self.axis_offset
                if np.any(reference_radius <= 0.0):
                    raise ValueError(f"{name}: axisymmetric surface points must satisfy " "radius > axis_offset")
                # The configured measure is a meridional line weight. Its
                # revolution supplies the physical reference contact area.
                measures = 2.0 * np.pi * reference_radius * measures
            if np.any(~np.isfinite(measures)) or np.any(measures < 0.0):
                raise ValueError(f"{name}: surface measures must be finite and non-negative")
            all_surface_measures.append(np.asarray(measures, dtype=np.float64))
        all_surface_ids = np.concatenate(all_surface_ids, axis=0)
        all_surface_measures = np.concatenate(all_surface_measures, axis=0)
        self.total_surface_num = all_surface_ids.shape[0]
        self.surface_id = ti.field(ti.i32, shape=self.total_surface_num)
        self.surface_id.from_numpy(all_surface_ids)
        self.surface_measure = ti.field(ti.f64, shape=self.total_surface_num)
        self.surface_measure.from_numpy(all_surface_measures)

        self.p_temp = ti.Vector.field(config.DIM, ti.f64, shape=self.total_surface_num)
        self.pv_temp = ti.Vector.field(config.DIM, ti.f64, shape=self.total_surface_num)

    def add_body_info(self):
        goffset = 0
        for name, body in self.bodies.bodies.items():
            # grid_size
            if "grid_size" not in body or body["grid_size"] is None or body["grid_size"] <= 0:
                body["grid_size"] = self.dx
            # xmin
            if "xmin" not in body or body["xmin"] is None or np.any(~np.isfinite(body["xmin"])):
                if config.DIM == 3:
                    body["xmin"] = np.zeros(3)
                else:
                    body["xmin"] = np.zeros(2)
            # xmax
            if "xmax" not in body or body["xmax"] is None or np.any(~np.isfinite(body["xmax"])):
                body["xmax"] = np.array(self.domain, dtype=float)
            # grid_num
            body["grid_num"] = (
                np.ceil((np.array(body["xmax"]) - np.array(body["xmin"])) / body["grid_size"]).astype(int) + 1
            )
            # goffset
            body["goffset"] = goffset
            body["total_grid_num"] = np.prod(body["grid_num"])
            goffset += body["total_grid_num"]
        self.total_background_grid_num = int(goffset)

    def visualize(self, log=True):
        import os

        vtk_path = os.path.join(self.path, "vtks")
        if not os.path.exists(vtk_path):
            os.makedirs(vtk_path)
        particle_num = self.particleNum.to_numpy()[0]
        pos = np.ascontiguousarray(self.particle.x.to_numpy()[:particle_num])
        posx = np.ascontiguousarray(pos[:, 0])
        posy = np.ascontiguousarray(pos[:, 1])
        posz = np.zeros_like(posx)
        if config.DIM == 3:
            posz = np.ascontiguousarray(pos[:, 2])
        vel = np.ascontiguousarray(self.particle.v.to_numpy()[:particle_num])
        velx = np.ascontiguousarray(vel[:, 0])
        vely = np.ascontiguousarray(vel[:, 1])
        velz = np.zeros_like(vely)
        if config.DIM == 3:
            velz = np.ascontiguousarray(vel[:, 2])
        stress = self.calculate_von_mises()
        bodyID = np.ascontiguousarray(self.particle.bodyID.to_numpy()[:particle_num])
        point_data = {
            "velocity": (velx, vely, velz),
            "stress": stress,
            "bodyID": bodyID,
        }
        if hasattr(self, "surface_id"):
            contact_sample = np.zeros(particle_num, dtype=np.int32)
            surface_id = np.asarray(self.surface_id.to_numpy(), dtype=np.int64)
            surface_id = surface_id[(surface_id >= 0) & (surface_id < particle_num)]
            contact_sample[surface_id] = 1
            point_data["contact_sample"] = contact_sample
        if getattr(self, "is_finite_strain_plastic", False):
            point_data.update(
                equivalent_plastic_strain=np.ascontiguousarray(
                    self.material.equivalent_plastic_strain.to_numpy()[:particle_num]
                ),
                volumetric_plastic_strain=np.ascontiguousarray(
                    self.material.volumetric_plastic_strain.to_numpy()[:particle_num]
                ),
            )
        pointsToVTK(
            vtk_path + f"/GraphicMPMParticle{self.output_count:06d}",
            posx,
            posy,
            posz,
            data=point_data,
        )
        if log:
            print_save_file_info(
                "MPM",
                self.step_count,
                self.output_count,
                self.time,
                self.path,
            )
        self.output_count += 1

    def init(self):
        self.particleNum[0] = 0
        for idb, (name, b) in enumerate(self.bodies.bodies.items()):
            self.add_body(idb, b["goffset"], b["grid_num"], b["grid_size"], b["xmin"], b["xmax"])
        for idb, (name, b) in enumerate(self.bodies.bodies.items()):
            particle_count = int(b["points"].shape[0])
            volume = np.asarray(b["volume"], dtype=np.float64)
            if volume.ndim == 0:
                volume = np.full(particle_count, float(volume), dtype=np.float64)
            else:
                volume = np.ascontiguousarray(volume.reshape(-1), dtype=np.float64)
                if volume.size != particle_count:
                    raise ValueError(f"{name}: particle volume must be scalar or contain " "one value per particle")
            if np.any(~np.isfinite(volume)) or np.any(volume <= 0.0):
                raise ValueError(f"{name}: particle volumes must be finite and positive")
            self.add_particle(idb, volume, b["init_v"], self.gravity, b["points"])

    def init_traction(self, traction=None, region=None):
        if traction is None:
            traction = [0, 0, 0]
        self.compute_traction = True
        self.assemble_traction_step = self.traction_p2g
        self.add_traction(traction, ti.func(region))

    @ti.kernel
    def add_body(
        self,
        idb: ti.i32,
        goffset: ti.i32,
        grid_num: ti.types.vector(config.DIM, ti.i32),
        grid_size: ti.f64,
        xmin: ti.types.vector(config.DIM, ti.f64),
        xmax: ti.types.vector(config.DIM, ti.f64),
    ):
        self.body[idb].goffset = goffset
        self.body[idb].grid_num = grid_num
        self.body[idb].grid_size = grid_size
        self.body[idb].xmin = xmin
        self.body[idb].xmax = xmax

    @ti.kernel
    def add_particle(
        self,
        idb: ti.i32,
        volume: ti.types.ndarray(),
        init_v: ti.types.vector(config.DIM, ti.f64),
        init_a: ti.types.vector(config.DIM, ti.f64),
        body: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=True)
        for i in range(body.shape[0]):
            ind = ti.atomic_add(self.particleNum[0], 1)
            self.particle[ind].bodyID = idb
            self.particle[ind].x = ti.Vector([body[i, d] for d in ti.static(range(config.DIM))])
            physical_volume = volume[i]
            if ti.static(self.is_axisymmetric):
                radius = body[i, 0] - ti.static(self.axis_offset)
                assert radius > 0.0, "axisymmetric particles require radius > axis_offset"
                physical_volume = 2.0 * ti.math.pi * radius * volume[i]
            self.particle[ind].vol0 = physical_volume
            self.particle[ind].m = self.material.density * physical_volume
            self.particle[ind].v = init_v
            self.particle[ind].a = init_a

    @ti.kernel
    def add_traction(self, traction: ti.types.vector(config.DIM, ti.f64), region: ti.template()):
        ti.loop_config(serialize=True)
        for pid in range(self.particleNum[0]):
            if region(self.particle[pid].x):
                ind = ti.atomic_add(self.tractionNum[0], 1)
                self.traction[ind].particleID = pid
                self.traction[ind].traction = traction

    @ti.func
    def stencil_offset_from_flat(self, flat_index, width: ti.template()):
        offset = ti.Vector.zero(ti.i32, config.DIM)
        remainder = flat_index
        for d in ti.static(range(config.DIM)):
            axis = config.DIM - 1 - d
            offset[axis] = remainder % width
            remainder //= width
        return offset

    @ti.func
    def shape_hessian(self, particle_id, local_id):
        """Return d(grad N)/dx without storing an O(particle*stencil) tape."""
        body_id = self.particle[particle_id].bodyID
        node_id = self.LnID[particle_id, local_id] - self.body[body_id].goffset
        grid_num = self.body[body_id].grid_num
        grid_coord = ti.Vector.zero(ti.i32, config.DIM)
        remainder = node_id
        for axis in ti.static(range(config.DIM)):
            grid_coord[axis] = remainder % grid_num[axis]
            remainder //= grid_num[axis]
        dx = self.body[body_id].grid_size
        inv_dx = 1.0 / dx
        xmin = self.body[body_id].xmin
        position = self.particle[particle_id].x
        shape = ti.Vector.zero(ti.f64, config.DIM)
        gradient = ti.Vector.zero(ti.f64, config.DIM)
        curvature = ti.Vector.zero(ti.f64, config.DIM)
        for axis in ti.static(range(config.DIM)):
            grid_position = xmin[axis] + grid_coord[axis] * dx
            shape[axis] = self.shape_func.shapefn(position[axis], grid_position, inv_dx, 0.0)
            gradient[axis] = self.shape_func.dshapefn(position[axis], grid_position, inv_dx, 0.0)
            curvature[axis] = self.shape_func.hshapefn(position[axis], grid_position, inv_dx, 0.0)
        hessian = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
        if ti.static(config.DIM == 2):
            hessian[0, 0] = curvature[0] * shape[1]
            hessian[1, 1] = shape[0] * curvature[1]
            hessian[0, 1] = gradient[0] * gradient[1]
            hessian[1, 0] = hessian[0, 1]
        else:
            hessian[0, 0] = curvature[0] * shape[1] * shape[2]
            hessian[1, 1] = shape[0] * curvature[1] * shape[2]
            hessian[2, 2] = shape[0] * shape[1] * curvature[2]
            hessian[0, 1] = gradient[0] * gradient[1] * shape[2]
            hessian[1, 0] = hessian[0, 1]
            hessian[0, 2] = gradient[0] * shape[1] * gradient[2]
            hessian[2, 0] = hessian[0, 2]
            hessian[1, 2] = shape[0] * gradient[1] * gradient[2]
            hessian[2, 1] = hessian[1, 2]
        return hessian

    @ti.kernel
    def _compute_shapefn(self):
        self.offset.fill(0)
        self.invalid_stencil_particle[None] = 0
        for i in range(self.particleNum[0]):
            bid = self.particle[i].bodyID
            goffset = self.body[bid].goffset
            grid_num = self.body[bid].grid_num
            xmin = self.body[bid].xmin
            dx = self.body[bid].grid_size
            inv_dx = 1.0 / dx
            pos = self.particle[i].x
            base = ti.cast(
                ti.floor((pos - xmin) * inv_dx - self.shape_func.offset),
                ti.i32,
            )
            valid_stencil = 1

            for count in range(self.shape_func.max_node_per_particle):
                offset = self.stencil_offset_from_flat(count, self.shape_func.max_node_per_particle_one_axis)
                grid_id = base + offset
                for d in ti.static(range(config.DIM)):
                    if grid_id[d] < 0 or grid_id[d] >= grid_num[d]:
                        valid_stencil = 0

                if valid_stencil != 0:
                    shapefn = ti.Vector.zero(ti.f64, config.DIM)
                    shapefn_grad = ti.Vector.zero(ti.f64, config.DIM)
                    for d in ti.static(range(config.DIM)):
                        xg = xmin[d] + grid_id[d] * dx
                        shape_fn = self.shape_func.shapefn(pos[d], xg, inv_dx, 0.0)
                        shape_fn_grad = self.shape_func.dshapefn(pos[d], xg, inv_dx, 0.0)
                        shapefn[d] = shape_fn
                        shapefn_grad[d] = shape_fn_grad

                    if ti.static(config.DIM == 2):
                        N = shapefn[0] * shapefn[1]
                        dN = ti.Vector([shapefn_grad[0] * shapefn[1], shapefn_grad[1] * shapefn[0]])
                        linear_grid_id = int(grid_id[0] + grid_id[1] * grid_num[0])
                        self.LnID[i, count] = linear_grid_id + goffset
                        self.shape[i, count] = N
                        self.dshape[i, count] = dN
                    else:
                        N = shapefn[0] * shapefn[1] * shapefn[2]
                        dN = ti.Vector(
                            [
                                shapefn_grad[0] * shapefn[1] * shapefn[2],
                                shapefn_grad[1] * shapefn[0] * shapefn[2],
                                shapefn_grad[2] * shapefn[0] * shapefn[1],
                            ]
                        )
                        linear_grid_id = int(
                            grid_id[0] + grid_id[1] * grid_num[0] + grid_id[2] * grid_num[0] * grid_num[1]
                        )
                        self.LnID[i, count] = linear_grid_id + goffset
                        self.shape[i, count] = N
                        self.dshape[i, count] = dN
            if valid_stencil != 0:
                self.offset[i] = self.shape_func.max_node_per_particle
            else:
                ti.atomic_max(self.invalid_stencil_particle[None], i + 1)

    def compute_shapefn(self):
        self._compute_shapefn()
        invalid = int(self.invalid_stencil_particle[None]) - 1
        if invalid >= 0:
            position = self.particle.x.to_numpy()[invalid].tolist()
            raise RuntimeError(
                f"Direct MPM particle {invalid} at {position} has an interpolation stencil "
                "outside its background grid"
            )

    def calculate_von_mises(self):
        raise NotImplementedError

    @ti.kernel
    def calculate_mean_velocity(self) -> ti.types.vector(config.DIM, ti.f64):
        velocity = ti.Vector.zero(ti.f64, config.DIM)
        for i in range(self.particleNum[0]):
            velocity += self.particle[i].v
        return velocity / self.particleNum[0]

    @ti.kernel
    def calculate_mean_acceleration(self) -> ti.types.vector(config.DIM, ti.f64):
        acceleration = ti.Vector.zero(ti.f64, config.DIM)
        for i in range(self.particleNum[0]):
            acceleration += self.particle[i].a
        return acceleration / self.particleNum[0]

    def initial_simulation(self):
        self.record()

    def record(self, log=True):
        if self.vis:
            self.visualize(log=log)

    def substep(self, verbose=True):
        raise NotImplementedError

    def _failure_diagnostics(self, exception, attempt, timestep):
        return {
            "kind": nonlinear_failure_kind(exception),
            "exception": type(exception).__name__,
            "message": str(exception),
            "attempt": int(attempt),
            "timestep": float(timestep),
            "time": float(self.time),
            "step": int(self.step_count),
        }

    def diagnostics_snapshot(self):
        return {
            "schema_version": 1,
            "subsystem": "mpm_direct",
            "solver_type": self.solver,
            "time": float(self.time),
            "step": int(self.step_count),
            "timestep": float(self.dt),
            "last_failure": self.last_failure,
            "last_step": self.history[-1] if self.history else None,
        }

    def run_substep(self, substep, verbose=True, record_history=True):
        original_timestep = float(self.dt)
        attempt_timestep = original_timestep
        attempts = []
        for attempt in range(self.step_retry.maximum_retries + 1):
            self.dt = float(attempt_timestep)
            try:
                result = substep(verbose)
            except RuntimeError as exception:
                if not (isinstance(exception, MPMConvergenceError) or is_recoverable_nonlinear_failure(exception)):
                    raise
                failure = self._failure_diagnostics(exception, attempt, attempt_timestep)
                attempts.append(failure)
                next_timestep = self.step_retry.next_timestep(attempt_timestep, attempt)
                if next_timestep is None:
                    self.last_failure = {
                        **failure,
                        "original_timestep": original_timestep,
                        "attempts": attempts,
                    }
                    self.dt = original_timestep
                    raise
                attempt_timestep = next_timestep
                continue

            self.time += float(attempt_timestep)
            self.step_count += 1
            retry_record = {
                "enabled": bool(self.step_retry.enabled),
                "original_timestep": original_timestep,
                "accepted_timestep": float(attempt_timestep),
                "retry_count": int(attempt),
                "attempts": attempts,
            }
            step_record = {
                "step": int(self.step_count),
                "time": float(self.time),
                "step_retry": retry_record,
            }
            if isinstance(result, dict):
                step_record.update(result)
            if record_history:
                self.history.append(step_record)
            self.last_failure = None
            return result

        raise AssertionError("unreachable Direct MPM retry state")

    def run(self, verbose=True, postprocessing=()):
        postprocessing = normalize_callbacks(postprocessing)
        with self.timer.section("Output"):
            self.initial_simulation()
        self.timer.profile0()
        with self.timer.section("Postprocess"):
            for f in postprocessing:
                f()
        for output_index in range(self.total_step):
            for interval_index in range(self.output_interval):
                next_step = self.step_count + 1
                output_due = interval_index + 1 == self.output_interval
                final_step = output_index + 1 == self.total_step and output_due
                record_history = self.step_schedule.history_due(next_step, output=output_due, final=final_step)
                compiling = self.compile_seconds is None
                if compiling:
                    print("Compiling first ... ...")
                    compile_start = time.perf_counter()
                with self.timer.section("MPM direct step"):
                    self.run_substep(
                        self.substep,
                        verbose,
                        record_history=record_history,
                    )
                if compiling:
                    ti.sync()
                    self.compile_seconds = time.perf_counter() - compile_start
                    print(f"Compiling time = {self.compile_seconds} \n")
                    self.timer.profile1()
                runtime_checkpoint()
            with self.timer.section("Output"):
                self.record()
            self.timer.profile0()
            with self.timer.section("Postprocess"):
                for f in postprocessing:
                    f()


__all__ = ["MPMConvergenceError", "MPMSolver"]
