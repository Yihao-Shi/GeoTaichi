from pathlib import Path

import numpy as np
import taichi as ti

from src.utils.errors import ConfigurationError


@ti.kernel
def exchange_grid(
    node: ti.template(),
    values: ti.types.ndarray(ndim=2),
    dimension: ti.template(),
    force: ti.template(),
    pack: ti.template(),
):
    for n, b in node:
        row = n * node.shape[1] + b
        if ti.static(force):
            for d in ti.static(range(dimension)):
                if ti.static(pack):
                    values[row, d] = node[n, b].force[d]
                else:
                    node[n, b].force[d] = values[row, d]
        else:
            if ti.static(pack):
                values[row, 0] = node[n, b].m
            else:
                node[n, b].m = values[row, 0]
            for d in ti.static(range(dimension)):
                if ti.static(pack):
                    values[row, d + 1] = node[n, b].momentum[d]
                else:
                    node[n, b].momentum[d] = values[row, d + 1]


@ti.kernel
def exchange_field(field: ti.template(), values: ti.types.ndarray(ndim=2), width: ti.template(), pack: ti.template()):
    for index in ti.grouped(field):
        row = 0
        for axis in ti.static(range(len(field.shape))):
            row = row * field.shape[axis] + index[axis]
        if ti.static(width == 0):
            if ti.static(pack):
                values[row, 0] = field[index]
            else:
                field[index] = values[row, 0]
        else:
            for d in ti.static(range(width)):
                if ti.static(pack):
                    values[row, d] = field[index][d]
                else:
                    field[index][d] = values[row, d]


class DistributedMPM:
    def __init__(self, model, clock=None, coupled_settings=()):
        sims, scene = model.sims, model.scene
        if not (
            sims.mode in ("Normal", "Lightweight")
            and sims.solver_type == "Explicit"
            and sims.configuration in ("ULMPM", "TLMPM")
            and not sims.is_direct_backend()
            and sims.dimension in (2, 3)
            and (
                sims.material_type == "Solid"
                or (sims.material_type == "Fluid" and sims.mode == "Normal" and sims.configuration == "ULMPM")
            )
        ):
            raise ConfigurationError(
                "distributed MPM requires native 2D or 3D explicit MPM (solid, or Normal UL single-phase fluid)"
            )
        if sims.mode == "Lightweight" and sims.configuration != "ULMPM":
            raise ConfigurationError("distributed Lightweight MPM requires ULMPM")
        if sims.sparse_grid:
            raise ConfigurationError(
                "distributed MPM does not support BlockScan or other sparse grids. Set sparse_grid=False."
            )
        if sims.isTHB or getattr(scene.element, "adaptive", False):
            raise ConfigurationError("distributed MPM requires a regular dense grid without THB or adaptive refinement")
        if sims.free_surface_detection or sims.ptraction_method == "Virtual":
            raise ConfigurationError(
                "distributed MPM does not support free-surface detection or virtual particle traction"
            )
        if sims.shape_function not in ("Linear", "SmoothLinear", "GIMP", "QuadBSpline", "CubicBSpline"):
            raise ConfigurationError("distributed MPM requires Linear, SmoothLinear, GIMP, QuadBSpline or CubicBSpline")
        if sims.mode == "Lightweight" and (
            sims.gauss_number > 0
            or sims.mls
            or sims.contact_detection
            or sims.stabilize in ("F-Bar Method", "Displacement F-Bar Method")
        ):
            raise ConfigurationError(
                "distributed Lightweight MPM does not support MLS, quadrature, F-Bar or grid contact"
            )
        if sims.mls and (sims.shape_function not in ("QuadBSpline", "CubicBSpline") or sims.gauss_number):
            raise ConfigurationError("distributed MLS requires a quadratic or cubic B-spline without Gauss quadrature")
        if sims.stabilize == "Displacement F-Bar Method" and (
            sims.material_type != "Fluid" or sims.shape_function not in ("QuadBSpline", "CubicBSpline")
        ):
            raise ConfigurationError(
                "distributed Displacement F-Bar requires a fluid material and a quadratic or cubic B-spline"
            )
        if sims.mode == "Normal" and sims.TESTMODE:
            raise ConfigurationError("distributed Normal MPM does not support test mode")
        if sims.configuration == "TLMPM" and (sims.is_2DAxisy or sims.gauss_number or sims.mls or sims.stabilize):
            raise ConfigurationError(
                "distributed TLMPM supports Cartesian reference transfers without MLS, quadrature or stabilization"
            )
        if (
            sims.mode == "Normal"
            and sims.is_2DAxisy
            and (
                sims.stabilize == "Displacement F-Bar Method"
                or (sims.stabilize == "F-Bar Method" and sims.material_type == "Fluid")
                or sims.gauss_number
                or sims.velocity_projection_scheme == "Affine"
                or sims.mls
            )
        ):
            raise ConfigurationError(
                "distributed axisymmetric Normal MPM supports standard, B-Bar and solid F-Bar integration "
                "without Affine, MLS or Gauss quadrature"
            )
        if model.generator.myGenerator:
            raise ConfigurationError("distributed MPM requires pre-generated particles")
        node = scene.node
        if int(scene.particleNum[0]) <= 0 or node is None:
            raise ConfigurationError("each distributed MPM rank requires local particles and an MPM grid")

        try:
            import torch
            import torch.distributed as dist
        except ImportError as exc:
            raise ConfigurationError("distributed MPM requires PyTorch with CUDA and NCCL") from exc
        if not dist.is_initialized() or dist.get_backend() != "nccl" or dist.get_world_size() < 2:
            raise ConfigurationError("distributed MPM requires an NCCL group of at least two ranks")
        if ti.lang.impl.current_cfg().arch != ti.cuda or torch.cuda.device_count() != 1:
            raise ConfigurationError("distributed MPM requires Taichi CUDA and one visible GPU per rank")

        clock = sims if clock is None else clock
        self.restart_index = None
        if model._restart_checkpoint is not None:
            self.restart_index, checkpoint_path = model._restart_checkpoint
            if Path(checkpoint_path).resolve() == Path(clock.path).resolve():
                raise ConfigurationError("distributed restart requires a new SavePath on each rank")
            # shortcut: persist the additional solver state before enabling restart on these paths.
            if (
                clock.adaptive_timestep
                or sims.configuration != "ULMPM"
                or sims.gauss_number
                or sims.contact_detection == "DEMContact"
                or sims.material_type != "Solid"
            ):
                raise ConfigurationError(
                    "distributed restart requires fixed-step solid ULMPM without Gauss quadrature or DEMContact"
                )
            if not np.isclose(clock.current_time, clock.current_step * clock.delta, rtol=0, atol=1e-10):
                raise ConfigurationError(
                    "distributed restart requires the saved step count and the original fixed timestep"
                )
        contact = scene.contact
        contact_settings = ()
        if sims.contact_detection == "MPMContact":
            contact_settings = (contact.friction,)
        elif sims.contact_detection == "GeoContact":
            contact_settings = (contact.friction, contact.alpha, contact.beta, contact.cut_off)
        elif sims.contact_detection == "DEMContact":
            if contact.polygon_vertices is None:
                raise ConfigurationError("distributed DEMContact requires a polygon on every rank")
            contact_settings = (
                tuple(contact.velocity),
                contact.polygon_vertices.to_numpy().tobytes(),
                tuple(sorted(scene.contact.get_parameters(params).items()) for params in contact.contact_phys),
            )
        settings = (
            sims.mode,
            sims.configuration,
            sims.material_type,
            sims.mapping,
            sims.dimension,
            sims.is_2DAxisy,
            tuple(node.shape),
            str(node.m.dtype),
            str(node.force.dtype),
            tuple(scene.element.gnum),
            tuple(scene.element.grid_size),
            tuple(sims.domain),
            tuple(sims.boundary),
            sims.shape_function,
            sims.velocity_projection_scheme,
            sims.stabilize,
            sims.fbar_fraction,
            sims.gauss_number,
            sims.mls,
            sims.pressure_smoothing,
            sims.contact_detection,
            contact_settings,
            sims.shape_smooth,
            tuple(sims.gravity),
            sims.alphaPIC,
            sims.background_damping,
            clock.delta,
            clock.init_delta,
            clock.CFL,
            clock.time,
            clock.current_time,
            clock.current_step,
            clock.current_print,
            self.restart_index,
            clock.adaptive_timestep,
            coupled_settings,
        )
        ranks = [None] * dist.get_world_size()
        dist.all_gather_object(ranks, (settings, str(Path(clock.path).resolve()), scene.is_rigid.to_numpy()))
        if any(other_settings != settings for other_settings, _, _ in ranks):
            raise ConfigurationError("distributed MPM grid, solver settings, and replicated rigid state must match")
        if len({path for _, path, _ in ranks}) != len(ranks):
            raise ConfigurationError("distributed MPM requires a different SavePath for each rank")
        scene.is_rigid.from_numpy(np.maximum.reduce([flags for _, _, flags in ranks]))

        self.torch = torch
        self.dist = dist
        self.node = node
        self.dimension = sims.dimension
        self.buffers = {}
        self.scalar = self._buffer((1,), ti.f64)
        # shortcut: replicate the dense grid until full-grid traffic or memory limits the model.
        node_count = node.shape[0] * node.shape[1]
        self.mass_momentum = self._buffer((node_count, self.dimension + 1), node.m.dtype)
        self.force = self._buffer((node_count, self.dimension), node.force.dtype)

    def _buffer(self, shape, dtype):
        return self.torch.empty(
            shape, device="cuda", dtype={ti.f32: self.torch.float32, ti.f64: self.torch.float64}[dtype]
        )

    def _sum(self, values):
        ti.sync()
        self.dist.all_reduce(values)
        # Taichi and NCCL use different CUDA streams.
        self.torch.cuda.synchronize()

    def reduce_grid(self, force):
        values = self.force if force else self.mass_momentum
        exchange_grid(self.node, values, self.dimension, force, True)
        self._sum(values)
        exchange_grid(self.node, values, self.dimension, force, False)

    def sum_field(self, field):
        width = getattr(field, "n", 0)
        shape = (np.prod(field.shape).item(), max(1, width))
        key = (shape, field.dtype)
        if key not in self.buffers:
            self.buffers[key] = self._buffer(shape, field.dtype)
        values = self.buffers[key]
        exchange_field(field, values, width, True)
        self._sum(values)
        exchange_field(field, values, width, False)

    def minimum_timestep(self, value):
        return self._reduce_scalar(value, self.dist.ReduceOp.MIN)

    def any_invalid_radius(self, invalid):
        return bool(self._reduce_scalar(invalid, self.dist.ReduceOp.MAX))

    def _reduce_scalar(self, value, op):
        self.scalar.fill_(value)
        self.dist.all_reduce(self.scalar, op=op)
        return self.scalar.item()
