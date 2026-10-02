"""VTU snapshots for synchronized IGA--MPM gallery animations.

The low-level writers in this module operate only on host NumPy arrays.  The
``IGAMPMGalleryRecorder`` adapter is intentionally small: it snapshots the two
Taichi backends and delegates all mesh construction and file output to those
host writers.  This keeps the output contract independently testable without
running a simulation.
"""

from pathlib import Path

import numpy as np

from src.nurbs.NurbsBasis import (
    NurbsBasisInterpolations2d,
    NurbsBasisInterpolations3d,
)
from third_party.pyevtk.hl import pointsToVTK, unstructuredGridToVTK
from third_party.pyevtk.vtk import VtkHexahedron, VtkQuad


def _sampled_grid_connectivity(resolution):
    if len(resolution) == 2:
        nu, nv = resolution
        connectivity = []
        for j in range(nv - 1):
            for i in range(nu - 1):
                n0 = i + nu * j
                connectivity.append((n0, n0 + 1, n0 + nu + 1, n0 + nu))
        nodes_per_cell = 4
        cell_type = VtkQuad.tid
    elif len(resolution) == 3:
        nu, nv, nw = resolution
        connectivity = []
        layer = nu * nv
        for k in range(nw - 1):
            for j in range(nv - 1):
                for i in range(nu - 1):
                    n0 = i + nu * (j + nv * k)
                    connectivity.append(
                        (
                            n0,
                            n0 + 1,
                            n0 + nu + 1,
                            n0 + nu,
                            n0 + layer,
                            n0 + layer + 1,
                            n0 + layer + nu + 1,
                            n0 + layer + nu,
                        )
                    )
        nodes_per_cell = 8
        cell_type = VtkHexahedron.tid
    else:
        raise ValueError("IGA gallery output supports only 2-D and 3-D grids")

    connectivity = np.asarray(connectivity, dtype=np.int64).reshape(-1)
    cell_count = connectivity.size // nodes_per_cell
    offsets = np.arange(
        nodes_per_cell,
        (cell_count + 1) * nodes_per_cell,
        nodes_per_cell,
        dtype=np.int64,
    )
    cell_types = np.full(cell_count, cell_type, dtype=np.uint8)
    return connectivity, offsets, cell_types


def _sampled_grid_vtu_data(sampled_points, displacement, stress):
    resolution = sampled_points.shape[:-1]
    dimension = sampled_points.shape[-1]
    points = [np.ascontiguousarray(sampled_points[..., axis].reshape(-1, order="F")) for axis in range(dimension)]
    displacement_components = [
        np.ascontiguousarray(displacement[..., axis].reshape(-1, order="F")) for axis in range(dimension)
    ]
    if dimension == 2:
        points.append(np.zeros(int(np.prod(resolution)), dtype=np.float64))
        displacement_components.append(np.zeros(int(np.prod(resolution)), dtype=np.float64))
    connectivity, offsets, cell_types = _sampled_grid_connectivity(resolution)
    return (
        points,
        connectivity,
        offsets,
        cell_types,
        {
            "displacement": tuple(displacement_components),
            "stress": np.ascontiguousarray(stress.reshape(-1, order="F")),
        },
    )


def _resolution_tuple(resolution, dimension):
    if isinstance(resolution, (int, np.integer)):
        values = (int(resolution),) * dimension
    else:
        values = tuple(int(value) for value in resolution)
    if len(values) != dimension or any(value < 2 for value in values):
        raise ValueError(f"IGA resolution must contain {dimension} integers greater than one")
    return values


def _validated_frame_path(output_path, frame, stem):
    frame = int(frame)
    if frame < 0:
        raise ValueError("gallery frame must be non-negative")
    directory = Path(output_path) / "vtks"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{stem}{frame:06d}"


def _finite_array(values, name, ndim=None):
    values = np.asarray(values)
    if ndim is not None and values.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-D array")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


def sample_iga_primitive(
    primitive,
    control_points,
    initial_control_points,
    stress,
    resolution,
):
    """Sample one NURBS primitive and return pyevtk-ready grid arrays."""
    dimension = int(primitive.dimension)
    resolution = _resolution_tuple(resolution, dimension)
    control_points = _finite_array(control_points, "control_points", ndim=2)
    initial_control_points = _finite_array(initial_control_points, "initial_control_points", ndim=2)
    stress = _finite_array(stress, "stress").reshape(-1)
    expected = int(primitive.num_ctrlpts)
    if (
        control_points.shape != (expected, dimension)
        or initial_control_points.shape != (expected, dimension)
        or stress.shape != (expected,)
    ):
        raise ValueError("IGA control-point snapshot does not match the primitive")

    sampled_points = np.zeros((*resolution, dimension), dtype=np.float64)
    displacement = np.zeros_like(sampled_points)
    sampled_stress = np.zeros(resolution, dtype=np.float64)
    parameters = [np.linspace(0.0, 1.0, value) for value in resolution]
    control_displacement = control_points - initial_control_points

    for index in np.ndindex(*resolution):
        coordinate = [parameters[axis][index[axis]] for axis in range(dimension)]
        if dimension == 2:
            arguments = (
                *coordinate,
                *primitive.degree,
                primitive.knot_vector_u,
                primitive.knot_vector_v,
            )
            interpolate = NurbsBasisInterpolations2d
        elif dimension == 3:
            arguments = (
                *coordinate,
                *primitive.degree,
                primitive.knot_vector_u,
                primitive.knot_vector_v,
                primitive.knot_vector_w,
            )
            interpolate = NurbsBasisInterpolations3d
        else:
            raise ValueError("IGA gallery output supports only 2-D and 3-D patches")

        sampled_points[index] = interpolate(*arguments, control_points, primitive.weights)
        displacement[index] = interpolate(*arguments, control_displacement, primitive.weights)
        sampled_stress[index] = interpolate(*arguments, stress, primitive.weights)

    return _sampled_grid_vtu_data(sampled_points, displacement, sampled_stress)


def _merge_iga_grids(grids):
    if not grids:
        raise ValueError("at least one IGA primitive is required")

    point_components = [[], [], []]
    displacement_components = [[], [], []]
    connectivity = []
    offsets = []
    cell_types = []
    stresses = []
    patch_ids = []
    point_offset = 0
    connectivity_offset = 0

    for patch_id, grid in enumerate(grids):
        points, local_connectivity, local_offsets, local_types, point_data = grid
        point_count = len(points[0])
        for axis in range(3):
            point_components[axis].append(np.asarray(points[axis]))
            displacement_components[axis].append(np.asarray(point_data["displacement"][axis]))
        connectivity.append(np.asarray(local_connectivity) + point_offset)
        offsets.append(np.asarray(local_offsets) + connectivity_offset)
        cell_types.append(np.asarray(local_types))
        stresses.append(np.asarray(point_data["stress"]))
        patch_ids.append(np.full(point_count, patch_id, dtype=np.int32))
        point_offset += point_count
        connectivity_offset += len(local_connectivity)

    return (
        tuple(np.ascontiguousarray(np.concatenate(parts)) for parts in point_components),
        np.ascontiguousarray(np.concatenate(connectivity)),
        np.ascontiguousarray(np.concatenate(offsets)),
        np.ascontiguousarray(np.concatenate(cell_types)),
        {
            "displacement": tuple(np.ascontiguousarray(np.concatenate(parts)) for parts in displacement_components),
            "stress": np.ascontiguousarray(np.concatenate(stresses)),
            "patch_id": np.ascontiguousarray(np.concatenate(patch_ids)),
        },
    )


def write_iga_frame(output_path, frame, patch_snapshots, resolution):
    """Write ``GraphicIGA%06d.vtu`` from host-side patch snapshots.

    Each snapshot is ``(primitive, control_points, initial_points, stress)``.
    Multiple primitives are merged into one synchronized unstructured grid.
    """
    grids = [
        sample_iga_primitive(primitive, current, initial, stress, resolution)
        for primitive, current, initial, stress in patch_snapshots
    ]
    points, connectivity, offsets, cell_types, point_data = _merge_iga_grids(grids)
    filename = _validated_frame_path(output_path, frame, "GraphicIGA")
    unstructuredGridToVTK(
        str(filename),
        *points,
        connectivity=connectivity,
        offsets=offsets,
        cell_types=cell_types,
        pointData=point_data,
    )
    return filename.with_suffix(".vtu")


def write_mpm_particle_frame(
    output_path,
    frame,
    positions,
    velocities,
    body_ids,
    contact_samples=None,
    state_data=None,
):
    """Write ``GraphicMPMParticle%06d.vtu`` from host particle arrays."""
    positions = _finite_array(positions, "positions", ndim=2).astype(np.float64, copy=False)
    velocities = _finite_array(velocities, "velocities", ndim=2).astype(np.float64, copy=False)
    if positions.shape != velocities.shape or positions.shape[1] not in (2, 3):
        raise ValueError("positions and velocities must have matching 2-D/3-D shape")
    particle_count, dimension = positions.shape
    if particle_count == 0:
        raise ValueError("MPM gallery output requires at least one particle")
    body_ids = np.asarray(body_ids, dtype=np.int32).reshape(-1)
    if body_ids.shape != (particle_count,):
        raise ValueError("body_ids must contain one value per particle")
    if contact_samples is None:
        contact_samples = np.zeros(particle_count, dtype=np.int32)
    else:
        contact_samples = np.asarray(contact_samples, dtype=np.int32).reshape(-1)
        if contact_samples.shape != (particle_count,):
            raise ValueError("contact_samples must contain one value per particle")

    xyz = [np.ascontiguousarray(positions[:, axis]) for axis in range(dimension)]
    velocity = [np.ascontiguousarray(velocities[:, axis]) for axis in range(dimension)]
    if dimension == 2:
        xyz.append(np.zeros(particle_count, dtype=np.float64))
        velocity.append(np.zeros(particle_count, dtype=np.float64))

    data = {
        "velocity": tuple(velocity),
        "speed": np.ascontiguousarray(np.linalg.norm(velocities, axis=1)),
        "bodyID": np.ascontiguousarray(body_ids),
        "contact_sample": np.ascontiguousarray(contact_samples),
    }
    for name, values in (state_data or {}).items():
        values = _finite_array(values, name)
        if values.shape[0] != particle_count:
            raise ValueError(f"{name} must contain one value per particle")
        data[name] = np.ascontiguousarray(values)

    filename = _validated_frame_path(output_path, frame, "GraphicMPMParticle")
    pointsToVTK(
        str(filename),
        *xyz,
        data=data,
    )
    return filename.with_suffix(".vtu")


class IGAMPMGalleryRecorder:
    """Snapshot a built IGA--MPM coupling engine into aligned VTU frames."""

    def __init__(self, engine, output_path, iga_resolution=(18, 18, 3)):
        self.engine = engine
        self.output_path = Path(output_path)
        self.iga_resolution = tuple(int(value) for value in iga_resolution)

    def _iga_snapshots(self):
        patch = self.engine.iga.patch
        current = np.asarray(patch.control_points.to_numpy(), dtype=np.float64)
        initial = np.asarray(patch.initial_control_points.to_numpy(), dtype=np.float64)
        stress = np.asarray(patch.stress.to_numpy(), dtype=np.float64)
        snapshots = []
        begin = 0
        for meta in patch.primitive.body.values():
            primitive = meta["primitive"]
            end = begin + int(primitive.num_ctrlpts)
            snapshots.append((primitive, current[begin:end], initial[begin:end], stress[begin:end]))
            begin = end
        if begin != current.shape[0]:
            raise RuntimeError("IGA primitive offsets do not cover the control points")
        return snapshots

    def _mpm_snapshot(self):
        mpm = self.engine.mpm
        particle_count = int(np.asarray(mpm.particleNum.to_numpy()).reshape(-1)[0])
        positions = np.asarray(mpm.particle.x.to_numpy()[:particle_count])
        velocities = np.asarray(mpm.particle.v.to_numpy()[:particle_count])
        body_ids = np.asarray(mpm.particle.bodyID.to_numpy()[:particle_count])
        contact_samples = np.zeros(particle_count, dtype=np.int32)
        if hasattr(mpm, "surface_id"):
            surface_ids = np.asarray(mpm.surface_id.to_numpy(), dtype=np.int64)
            surface_ids = surface_ids[(surface_ids >= 0) & (surface_ids < particle_count)]
            contact_samples[surface_ids] = 1
        state_data = {
            "stress": np.asarray(mpm.calculate_von_mises(), dtype=np.float64),
        }
        if getattr(mpm, "is_finite_strain_plastic", False):
            state_data.update(
                equivalent_plastic_strain=np.asarray(
                    mpm.material.equivalent_plastic_strain.to_numpy()[:particle_count],
                    dtype=np.float64,
                ),
                volumetric_plastic_strain=np.asarray(
                    mpm.material.volumetric_plastic_strain.to_numpy()[:particle_count],
                    dtype=np.float64,
                ),
            )
        return positions, velocities, body_ids, contact_samples, state_data

    def record(self, frame, update_iga_stress=True):
        if update_iga_stress and hasattr(self.engine.iga, "visualize_stress"):
            self.engine.iga.visualize_stress()
        iga_path = write_iga_frame(
            self.output_path,
            frame,
            self._iga_snapshots(),
            self.iga_resolution,
        )
        mpm_path = write_mpm_particle_frame(self.output_path, frame, *self._mpm_snapshot())
        return iga_path, mpm_path


__all__ = [
    "IGAMPMGalleryRecorder",
    "sample_iga_primitive",
    "write_iga_frame",
    "write_mpm_particle_frame",
]
