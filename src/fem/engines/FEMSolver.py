"""Shared configuration, preprocessing and output for FEM engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.fem.boundaries import (
    DeviceBoundaryData,
    DirichletBoundary,
    NeumannBoundary,
)
from src.fem.elements import create_element
from src.fem.engines.FEMState import FEMState
from src.mpm.SoftParticleOutput import von_mises_stress
from src.utils.SolverConsole import print_save_file_info
from src.utils.SolverRuntime import StepSchedule
from src.utils.TimeTicker import Timer
from src.utils.linalg import no_operation


def _volume_weighted_cell_to_node(
    cell_values,
    connectivity,
    reference_weights,
    node_count,
):
    """Project constant/averaged cell fields to nodes by reference measure."""

    values = np.asarray(cell_values, dtype=np.float64)
    connectivity = np.asarray(connectivity, dtype=np.int32)
    if values.shape[0] != connectivity.shape[0]:
        raise ValueError("one cell value is required per FEM element")
    cell_measure = np.sum(np.asarray(reference_weights, dtype=np.float64), axis=1)
    if np.any(cell_measure <= 0.0):
        raise ValueError("FEM cell-to-node projection requires positive measures")
    incidence_weight = cell_measure / connectivity.shape[1]
    nodal_weight = np.zeros(int(node_count), dtype=np.float64)
    nodal_values = np.zeros((int(node_count), *values.shape[1:]), dtype=np.float64)
    weighted_values = values * incidence_weight.reshape((-1, *([1] * (values.ndim - 1))))
    for local in range(connectivity.shape[1]):
        nodes = connectivity[:, local]
        np.add.at(nodal_weight, nodes, incidence_weight)
        np.add.at(nodal_values, nodes, weighted_values)
    if np.any(nodal_weight <= 0.0):
        raise ValueError("FEM mesh contains a node with no incident cell measure")
    nodal_values /= nodal_weight.reshape((-1, *([1] * (values.ndim - 1))))
    return np.ascontiguousarray(nodal_values)


@dataclass
class FEMResult:
    time: float
    positions: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    reaction: np.ndarray
    nodal_von_mises: np.ndarray
    strain_energy: float
    converged: bool = True
    history: list = field(default_factory=list)


class FEMSolver:
    """Common FEM state.

    Mesh integration and boundary topology are initialization work and may use
    NumPy.  Once constructed, nodal state and all equation operations live in
    :class:`FEMState` Taichi fields.  The array properties below are explicit
    user/output snapshots; production time integration never calls them.
    """

    def __init__(self, mesh, material, dirichlet=None, neumann=None, **kwargs):
        self.mesh = mesh.copy()
        self.material = material
        self.dimension = int(kwargs.get("dimension", 3))
        self.is_axisymmetric = bool(
            kwargs.get(
                "axisymmetric",
                kwargs.get("is_axisymmetric", kwargs.get("is_2DAxisy", False)),
            )
        )
        self.axis_offset = float(kwargs.get("axis_offset", 0.0))
        if self.is_axisymmetric and self.dimension != 2:
            raise ValueError("axisymmetric FEM requires dimension=2")
        self.dirichlet = dirichlet if dirichlet is not None else DirichletBoundary()
        self.neumann = neumann if neumann is not None else NeumannBoundary()
        formulation = "surface_cloth" if bool(getattr(material, "is_cloth", False)) else "classical"
        self.element = create_element(
            self.mesh,
            material.thickness,
            formulation=formulation,
            axisymmetric=self.is_axisymmetric,
            axis_offset=self.axis_offset,
        )
        self.backend = "taichi"

        # Rest coordinates define the material metric. Reference coordinates
        # define zero user displacement and may intentionally differ.
        self.rest_positions = self.mesh.rest_shape.copy()
        self.reference_positions = self.mesh.points.copy()
        self.initial_positions = self.reference_positions.copy()
        self.initial_deformation_gradients = self.element.deformation_gradients(self.reference_positions)

        initial_velocity = np.zeros_like(self.reference_positions)
        prescribed_initial_velocity = kwargs.get("initial_velocity")
        if prescribed_initial_velocity is not None:
            value = np.asarray(prescribed_initial_velocity, dtype=np.float64)
            initial_velocity[:] = value if value.shape == initial_velocity.shape else value.reshape(1, 3)

        self.gravity = kwargs.get("gravity", (0.0, 0.0, 0.0))
        self.damping = float(kwargs.get("damping", 0.0))
        if self.damping < 0.0:
            raise ValueError("FEM damping must be non-negative")

        self.path = kwargs.get("path", kwargs.get("save_path", None))
        self.output_interval = int(kwargs.get("interval", kwargs.get("output_interval", 1)))
        if self.output_interval <= 0:
            raise ValueError("FEM output interval must be positive")
        self.track_energy = bool(kwargs.get("track_energy", False))
        self.step_schedule = StepSchedule.from_options(kwargs, output_interval=self.output_interval)
        self.time = 0.0
        self.step_count = 0
        self.history = []
        self.timer = Timer()
        self.compile_seconds = None
        self._boundary_data_initialized = False
        self._boundary_dof_count = 0
        self._boundary_force_initialized = False
        self.boundary_data = None
        self.contact_assembler = None
        self._advance_constitutive_state = no_operation
        self._boundary_time_step = None
        self._adaptive_boundary_evaluation = False
        self._boundary_dofs = np.empty(0, dtype=np.int32)
        self._boundary_history_max_bytes = int(kwargs.get("boundary_history_max_bytes", 512 * 1024 * 1024))
        if self._boundary_history_max_bytes <= 0:
            raise ValueError("boundary_history_max_bytes must be positive")

        mass = self._assemble_lumped_mass()
        if np.any(mass <= 0.0):
            missing = np.flatnonzero(mass <= 0.0)
            raise ValueError(f"FEM mesh contains nodes with zero mass: {missing[:8].tolist()}")
        self.state = FEMState(self.reference_positions, initial_velocity, mass)
        if self.dimension == 2:
            # FEM state always stores three components so the same device
            # fields serve membranes, planar solids, and 3-D volumes.  The
            # virtual third component is not a physical 2-D unknown and must
            # be constrained even before public boundary preprocessing runs.
            constrained = np.zeros(self.degree_of_freedom, dtype=np.int32)
            constrained[2::3] = 1
            self.state.constrained.from_numpy(constrained)
        self.state.set_gravity(tuple(float(value) for value in self.gravity))
        self.set_boundary_data_step = self._boundary_data_not_initialized
        self.update_external_force_step = self._boundary_data_not_initialized
        self.prepare_explicit_step = self._boundary_data_not_initialized
        self.apply_boundary_step = no_operation

    @property
    def degree_of_freedom(self):
        return 3 * self.mesh.number_of_nodes

    @property
    def gravity(self):
        return self._gravity

    @gravity.setter
    def gravity(self, value):
        gravity = np.asarray(value, dtype=np.float64).reshape(-1)
        if gravity.size == 2:
            gravity = np.append(gravity, 0.0)
        if gravity.size != 3:
            raise ValueError("FEM gravity must contain two or three components")
        self._gravity = gravity
        state = getattr(self, "state", None)
        if state is not None:
            state.set_gravity(tuple(float(component) for component in gravity))

    @property
    def position_field(self):
        return self.state.position

    @property
    def velocity_field(self):
        return self.state.velocity

    @property
    def acceleration_field(self):
        return self.state.acceleration

    @property
    def mass_field(self):
        return self.state.mass

    @property
    def positions(self):
        return self.state.position.to_numpy()

    @positions.setter
    def positions(self, value):
        values = np.ascontiguousarray(value, dtype=self.state.numpy_type)
        if values.shape != (self.mesh.number_of_nodes, 3):
            raise ValueError("FEM positions have an invalid shape")
        self.state.position.from_numpy(values)

    @property
    def velocity(self):
        return self.state.velocity.to_numpy()

    @velocity.setter
    def velocity(self, value):
        self.state.velocity.from_numpy(np.ascontiguousarray(value, dtype=self.state.numpy_type))

    @property
    def acceleration(self):
        return self.state.acceleration.to_numpy()

    @acceleration.setter
    def acceleration(self, value):
        self.state.acceleration.from_numpy(np.ascontiguousarray(value, dtype=self.state.numpy_type))

    @property
    def reaction(self):
        return self.state.reaction.to_numpy()

    @property
    def mass(self):
        return self.state.mass.to_numpy()

    @property
    def displacement(self):
        # This property is an output snapshot, not a time-step backend path.
        return self.positions - self.reference_positions

    def _assemble_lumped_mass(self):
        """One-time reference-domain quadrature preprocessing."""
        mass = np.zeros(self.mesh.number_of_nodes, dtype=np.float64)
        for element_id, connectivity in enumerate(self.element.connectivity):
            for quadrature_id in range(self.element.quadrature_count):
                nodal_mass = (
                    self.material.density
                    * self.element.reference_weights[element_id, quadrature_id]
                    * self.element.shape_values[quadrature_id]
                )
                np.add.at(mass, connectivity, nodal_mass)
        return mass

    def _dirichlet_values(self, time):
        """Evaluate one host frame during boundary preprocessing."""
        dofs, displacement = self.dirichlet.values(self.mesh, time)
        if self.dimension == 2:
            dofs = np.asarray(dofs, dtype=np.int32)
            displacement = np.asarray(displacement, dtype=np.float64)
            virtual_dofs = 3 * np.arange(self.mesh.number_of_nodes, dtype=np.int32) + 2
            physical = dofs % 3 != 2
            dofs = np.concatenate((dofs[physical], virtual_dofs))
            displacement = np.concatenate(
                (
                    displacement[physical],
                    np.zeros(virtual_dofs.size, dtype=np.float64),
                )
            )
        return (
            np.ascontiguousarray(dofs, dtype=np.int32),
            np.ascontiguousarray(displacement, dtype=self.state.numpy_type),
        )

    def initialize_device_boundaries(
        self,
        *,
        timestep=None,
        total_step=None,
        force=False,
    ):
        """Sample user callables once and persist every runtime frame on device.

        Arbitrary Python callables cannot execute inside a Taichi kernel.  For a
        fixed time-step solve, sampling their nodal result before integration is
        the only general way to preserve the callable API without performing a
        host evaluation and array upload in every substep.
        """
        if self.boundary_data is not None and not force:
            return self.boundary_data
        timestep = float(self.dt if timestep is None else timestep)
        total_step = int(self.total_step if total_step is None else total_step)
        if not np.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("FEM boundary frame time step must be positive")
        if total_step < 0:
            raise ValueError("FEM boundary frame count cannot be negative")

        dirichlet_dynamic = bool(self.dirichlet.time_dependent)
        force_dynamic = bool(self.neumann.time_dependent)
        retry = getattr(self, "step_retry", None)
        self._adaptive_boundary_evaluation = bool(
            (dirichlet_dynamic or force_dynamic) and retry is not None and retry.enabled
        )

        dofs, initial_displacement = self._dirichlet_values(0.0)
        dirichlet_frames = total_step + 1 if dirichlet_dynamic and not self._adaptive_boundary_evaluation else 1
        force_frames = total_step + 1 if force_dynamic and not self._adaptive_boundary_evaluation else 1
        itemsize = np.dtype(self.state.numpy_type).itemsize
        required_bytes = (
            dofs.nbytes
            + dirichlet_frames * int(dofs.size) * itemsize
            + force_frames * self.mesh.number_of_nodes * 3 * itemsize
        )
        if required_bytes > self._boundary_history_max_bytes:
            raise MemoryError(
                "device-resident FEM boundary history requires "
                f"{required_bytes / (1024 ** 2):.1f} MiB, exceeding "
                f"boundary_history_max_bytes={self._boundary_history_max_bytes}; "
                "use static boundary values, shorten the configured solve, or "
                "raise the explicit memory limit"
            )

        displacement_history = np.empty((dirichlet_frames, dofs.size), dtype=self.state.numpy_type)
        displacement_history[0] = initial_displacement
        for frame in range(1, dirichlet_frames):
            frame_dofs, frame_values = self._dirichlet_values(frame * timestep)
            if not np.array_equal(frame_dofs, dofs):
                raise ValueError(
                    "FEM Dirichlet callable changed constraint topology; nodes "
                    "and components must remain fixed during a solve"
                )
            displacement_history[frame] = frame_values

        force_history = np.empty(
            (force_frames, self.mesh.number_of_nodes, 3),
            dtype=self.state.numpy_type,
        )
        for frame in range(force_frames):
            force_history[frame] = self.neumann.force(
                self.mesh,
                frame * timestep,
                axisymmetric=self.is_axisymmetric,
                axis_offset=self.axis_offset,
            )

        self.boundary_data = DeviceBoundaryData(
            dofs,
            displacement_history,
            force_history,
            real_type=self.state.real_type,
            numpy_type=self.state.numpy_type,
            dirichlet_dynamic=dirichlet_dynamic,
            force_dynamic=force_dynamic,
        )
        self.boundary_data.initialize_constraints(self.state.constrained, self.state.prescribed_displacement)
        self.boundary_data.set_force_frame(0, self.state.boundary_force)
        self._boundary_time_step = timestep
        self._boundary_dofs = dofs
        self._boundary_data_initialized = True
        self._boundary_dof_count = int(dofs.size)
        self._boundary_force_initialized = True
        self.state.apply_boundary(0.0, 0)
        self.bind_boundary_functions()
        return self.boundary_data

    def reconfigure_device_boundary_timeline(self, timestep, total_step):
        """Rebuild transient frames after an out-of-loop dt/time change."""
        timestep = float(timestep)
        total_step = int(total_step)
        self.total_step = total_step
        if self.boundary_data is None:
            return self.initialize_device_boundaries(timestep=timestep, total_step=total_step)
        if not (self.boundary_data.dirichlet_dynamic or self.boundary_data.force_dynamic):
            self._boundary_time_step = timestep
            return self.boundary_data
        if self._adaptive_boundary_evaluation:
            self._boundary_time_step = timestep
            return self.boundary_data
        if self.step_count != 0:
            raise RuntimeError(
                "cannot change dt/simulation time after advancing a FEM solve "
                "with preallocated Python-callable boundary frames"
            )
        return self.initialize_device_boundaries(timestep=timestep, total_step=total_step, force=True)

    def _boundary_frame(self, time, step, frame_count):
        if frame_count == 1:
            return 0
        if step is None:
            step = int(round(float(time) / self._boundary_time_step))
        step = int(step)
        if not 0 <= step < frame_count:
            raise IndexError(
                f"FEM boundary step {step} is outside the preallocated range "
                f"[0, {frame_count}); configure enough solver steps"
            )
        return step

    def _boundary_data_not_initialized(self, *_args, **_kwargs):
        raise RuntimeError("FEM device boundary data has not been initialized")

    def bind_boundary_functions(self):
        """Bind fixed, transient, and empty boundary paths once."""
        if self.boundary_data is None:
            raise RuntimeError("FEM device boundary data has not been initialized")
        self.set_boundary_data_step = (
            self._set_adaptive_boundary_data
            if self.boundary_data.dirichlet_dynamic and self._adaptive_boundary_evaluation
            else (
                self._set_dynamic_boundary_data
                if self.boundary_data.dirichlet_dynamic
                else self._keep_static_boundary_data
            )
        )
        self.update_external_force_step = (
            self._update_adaptive_external_force
            if self.boundary_data.force_dynamic and self._adaptive_boundary_evaluation
            else (
                self._update_dynamic_external_force
                if self.boundary_data.force_dynamic
                else self._update_static_external_force
            )
        )
        self.prepare_explicit_step = (
            self._prepare_dynamic_explicit_step
            if self.boundary_data.force_dynamic
            else self._prepare_static_explicit_step
        )
        self.apply_boundary_step = self.state.apply_boundary if self._boundary_dof_count > 0 else no_operation

    def _keep_static_boundary_data(self, _time, _step=None):
        return self._boundary_dof_count

    def _set_dynamic_boundary_data(self, time, step=None):
        frame = self._boundary_frame(time, step, self.boundary_data.dirichlet_frame_count)
        self.boundary_data.set_dirichlet_frame(frame, self.state.prescribed_displacement)
        return self._boundary_dof_count

    def _set_adaptive_boundary_data(self, time, _step=None):
        dofs, values = self._dirichlet_values(time)
        if not np.array_equal(dofs, self._boundary_dofs):
            raise ValueError(
                "FEM Dirichlet callable changed constraint topology; nodes "
                "and components must remain fixed during a solve"
            )
        self.state.set_boundary_data(dofs, values)
        return self._boundary_dof_count

    def _update_static_external_force(self, _time, _step=None):
        self.state.build_external_force()

    def _update_dynamic_external_force(self, time, step=None):
        frame = self._boundary_frame(time, step, self.boundary_data.force_frame_count)
        self.boundary_data.set_force_frame(frame, self.state.boundary_force)
        self.state.build_external_force()

    def _update_adaptive_external_force(self, time, _step=None):
        self.state.set_boundary_force(
            self.neumann.force(
                self.mesh,
                time,
                axisymmetric=self.is_axisymmetric,
                axis_offset=self.axis_offset,
            )
        )
        self.state.build_external_force()

    def _prepare_static_explicit_step(self, _time, _step=None):
        self.state.prepare_explicit_step(int(self._boundary_dof_count > 0))

    def _prepare_dynamic_explicit_step(self, time, step=None):
        frame = self._boundary_frame(time, step, self.boundary_data.force_frame_count)
        self.boundary_data.set_force_frame(frame, self.state.boundary_force)
        self.state.prepare_explicit_step(int(self._boundary_dof_count > 0))

    @property
    def is_fully_constrained_static(self):
        """True when every nodal DOF has a time-independent prescription."""
        return bool(
            self._boundary_data_initialized
            and not self.dirichlet.time_dependent
            and self._boundary_dof_count == self.degree_of_freedom
        )

    def assemble_internal(self, positions=None, need_stiffness=False, need_stress=False):
        raise NotImplementedError("FEM runtime assembly must be supplied by a Taichi element assembler")

    def stable_time_step(self, cfl=0.5):
        """Reference-mesh CFL estimate (initialization only)."""
        cfl = float(cfl)
        if cfl <= 0.0:
            raise ValueError("FEM CFL number must be positive")
        minimum_edge = np.inf
        for connectivity in self.mesh.cells:
            coordinates = self.rest_positions[connectivity]
            for first in range(len(connectivity)):
                for second in range(first + 1, len(connectivity)):
                    minimum_edge = min(
                        minimum_edge,
                        np.linalg.norm(coordinates[second] - coordinates[first]),
                    )
        lame_lambda, mu = (
            self.material.lame_parameters(self.element.constitutive_dimension)
            if callable(getattr(self.material, "lame_parameters", None))
            else self.material.lame_parameters
        )
        wave_modulus = lame_lambda + 2.0 * mu
        if wave_modulus <= 0.0:
            # Some pressure-dependent incremental models (for example MCC)
            # own only state-dependent Gauss-point moduli. They cannot supply
            # a single preprocessing wave bound, so an explicit user dt is
            # required and the coupled DEM/contact limits remain applicable.
            return np.inf
        wave_speed = np.sqrt(wave_modulus / self.material.density)
        return cfl * minimum_edge / wave_speed

    def _output_fields(self):
        """Download diagnostics only when output/result is explicitly requested."""
        energy, _, _, stress = self._assemble_output_diagnostics()
        nodal_stress = _volume_weighted_cell_to_node(
            stress,
            self.element.connectivity,
            self.element.reference_weights,
            self.mesh.number_of_nodes,
        )
        return float(energy), von_mises_stress(nodal_stress)

    def _assemble_output_diagnostics(self):
        return self.assemble_internal(self.positions, need_stress=True)

    def result(self, converged=True):
        energy, nodal_von_mises = self._output_fields()
        positions = self.positions
        return FEMResult(
            self.time,
            positions,
            positions - self.reference_positions,
            self.velocity,
            self.acceleration,
            self.reaction,
            nodal_von_mises,
            energy,
            bool(converged),
            list(self.history),
        )

    def record(self, filename=None, log=True):
        _, nodal_von_mises = self._output_fields()
        if filename is None:
            if self.path is None:
                return
            filename = Path(self.path) / "vtks" / f"FEM{self.step_count:06d}.vtu"
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        positions = self.positions
        output_mesh = self.mesh.copy()
        output_mesh.points[:] = positions
        output_mesh.write(
            filename,
            point_data={
                "displacement": positions - self.reference_positions,
                "velocity": self.velocity,
                "acceleration": self.acceleration,
                "reaction": self.reaction,
                "von_mises": nodal_von_mises,
            },
        )
        if log:
            print_save_file_info(
                "FEM",
                self.step_count,
                self.step_count // self.output_interval,
                self.time,
                self.path if self.path is not None else filename.parent,
            )


__all__ = ["FEMResult", "FEMSolver"]
