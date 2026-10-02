import os
from itertools import product

import numpy as np

from src.mpm.elements.AdaptiveGridOutput import build_adaptive_leaf_mesh
from src.mpm.Simulation import Simulation
from src.mpm.SceneManager import myScene
from src.mpm.SoftParticleOutput import (
    pk1_to_cauchy,
    von_mises_stress,
    vtk_vector,
)
from src.utils.FieldIO import field_to_numpy_prefix
from src.utils.linalg import no_operation, get_dataclass_to_dict
from third_party.pyevtk.hl import pointsToVTK, gridToVTK, unstructuredGridToVTK
from third_party.pyevtk.vtk import VtkHexahedron, VtkQuad, VtkVertex


def _normalize_closed_domain_pressure(pressure, cell_type):
    """Use one zero-mean output gauge when no atmospheric cell fixes pressure."""
    normalized = np.array(pressure, copy=True)
    pressure_flat = normalized.reshape(-1)
    cell_type_flat = np.asarray(cell_type).reshape(-1)
    if pressure_flat.size != cell_type_flat.size:
        return normalized
    fluid = cell_type_flat == 1
    if np.any(fluid) and not np.any(cell_type_flat == 0):
        pressure_flat[fluid] -= np.mean(pressure_flat[fluid], dtype=np.float64)
    return normalized


def build_sparse_block_mesh(dimension, active_block_ids, block_count, block_size, grid_size, gnum):
    """Build one clipped VTK cell for every active BlockScan storage block."""
    dimension = int(dimension)
    if dimension not in (2, 3):
        raise ValueError(f"Sparse block output requires dimension 2 or 3, got {dimension}")

    active_block_ids = np.asarray(active_block_ids, dtype=np.int64).reshape(-1)
    block_count = np.asarray(block_count, dtype=np.int64).reshape(-1)[:dimension]
    grid_size = np.asarray(grid_size, dtype=np.float64).reshape(-1)[:dimension]
    gnum = np.asarray(gnum, dtype=np.int64).reshape(-1)[:dimension]
    block_size = int(block_size)
    if block_size <= 1:
        raise ValueError(f"Sparse block output requires block_size > 1, got {block_size}")
    if block_count.size != dimension or grid_size.size != dimension or gnum.size != dimension:
        raise ValueError("Sparse block output arrays must match the requested dimension")
    if np.any(block_count <= 0) or np.any(grid_size <= 0.0) or np.any(gnum <= 1):
        raise ValueError(
            "Sparse block output requires positive block counts/grid spacing and at least two nodes per axis"
        )

    total_blocks = int(np.prod(block_count))
    if np.any(active_block_ids < 0) or np.any(active_block_ids >= total_blocks):
        raise ValueError("Sparse block output received an active block id outside the physical block grid")

    vertices_per_cell = 4 if dimension == 2 else 8
    coords = np.empty((active_block_ids.size * vertices_per_cell, dimension), dtype=np.float64)
    connectivity = np.arange(coords.shape[0], dtype=np.int64).reshape(-1, vertices_per_cell)
    block_indices = np.empty((active_block_ids.size, dimension), dtype=np.int32)
    domain_upper = (gnum - 1) * grid_size

    for compact_id, physical_id in enumerate(active_block_ids):
        block_index = np.empty(dimension, dtype=np.int64)
        remaining = int(physical_id)
        for axis in range(dimension):
            block_index[axis] = remaining % int(block_count[axis])
            remaining //= int(block_count[axis])
        block_indices[compact_id, :] = block_index

        first_node = block_index * block_size
        lower = np.maximum(0.0, (first_node - 0.5) * grid_size)
        upper = np.minimum(domain_upper, (first_node + block_size - 0.5) * grid_size)
        offset = compact_id * vertices_per_cell
        if dimension == 2:
            x0, y0 = lower
            x1, y1 = upper
            coords[offset : offset + 4, :] = (
                (x0, y0),
                (x1, y0),
                (x1, y1),
                (x0, y1),
            )
        else:
            x0, y0, z0 = lower
            x1, y1, z1 = upper
            coords[offset : offset + 8, :] = (
                (x0, y0, z0),
                (x1, y0, z0),
                (x1, y1, z0),
                (x0, y1, z0),
                (x0, y0, z1),
                (x1, y0, z1),
                (x1, y1, z1),
                (x0, y1, z1),
            )

    return {
        "coords": np.ascontiguousarray(coords),
        "connectivity": np.ascontiguousarray(connectivity),
        "active_block_id": np.ascontiguousarray(active_block_ids),
        "compact_block_id": np.arange(active_block_ids.size, dtype=np.int64),
        "block_index": np.ascontiguousarray(block_indices),
    }


class WriteFile:
    def __init__(self, sims):
        self.vtk_path = None
        self.particle_path = None
        self.grid_path = None

        self.save_particle = no_operation
        self.save_grid = no_operation
        self.adaptive_grid_cache = None
        self.adaptive_grid_cache_signature = None

        self.mkdir(sims)
        self.manage_function(sims)

    def manage_function(self, sims: Simulation):
        self.visualizeParticle = no_operation
        if "particle" in sims.monitor_type:
            if sims.visualize:
                if sims.dimension == 2:
                    self.visualizeParticle = self.VisualizeParticle2D
                elif sims.dimension == 3:
                    self.visualizeParticle = self.VisualizeParticle
            if sims.material_type in ("TwoPhaseSingleLayer", "TwoPhaseDoubleLayer"):
                self.save_particle = self.MonitorParticleTwoPhase
            elif sims.neighbor_detection or sims.coupling:
                if "Implicit" in sims.solver_type:
                    if sims.material_type == "Fluid" or sims.material_type == "TwoPhaseDoubleLayer":
                        self.save_particle = self.MonitorIncompressibleParticleCoupling
                    elif sims.material_type == "Solid":
                        self.save_particle = self.MonitorImplicitParticleCoupling
                else:
                    self.save_particle = self.MonitorParticleCoupling
            else:
                if sims.material_type in ("TwoPhaseSingleLayer", "TwoPhaseDoubleLayer"):
                    self.save_particle = self.MonitorParticleTwoPhase
                else:
                    if "Implicit" in sims.solver_type:
                        if sims.material_type == "Fluid" or sims.material_type == "TwoPhaseDoubleLayer":
                            self.save_particle = self.MonitorIncompressibleParticle
                        elif sims.material_type == "Solid":
                            self.save_particle = self.MonitorImplicitParticle
                    else:
                        self.save_particle = self.MonitorParticle

        self.visualizeGrid = no_operation
        if "grid" in sims.monitor_type:
            if sims.visualize:
                if sims.dimension == 2:
                    self.visualizeGrid = self.VisualizeGrid2D
                elif sims.dimension == 3:
                    self.visualizeGrid = self.VisualizeGrid
            if sims.contact_detection:
                self.save_grid = self.MonitorContactGrid
            else:
                self.save_grid = self.MonitorGrid

        self.visualizedObject = no_operation
        if "object" in sims.monitor_type:
            self.visualizedObject = self.VisualizeObject2D

    def output(self, sims, scene):
        self.save_particle(sims, scene)
        self.save_grid(sims, scene)
        self.visualizedObject(sims, scene)

    def mkdir(self, sims: Simulation):
        if not os.path.exists(sims.path):
            os.makedirs(sims.path)

        self.vtk_path = None
        self.particle_path = None
        self.grid_path = None

        self.particle_path = sims.path + "/particles"
        self.vtk_path = sims.path + "/vtks"
        self.grid_path = sims.path + "/grids"
        if not os.path.exists(self.particle_path):
            os.makedirs(self.particle_path)
        if not os.path.exists(self.vtk_path):
            os.makedirs(self.vtk_path)
        if not os.path.exists(self.grid_path):
            os.makedirs(self.grid_path)

    def VisualizeObject2D(self, sims: Simulation, scene: myScene):
        polygon = scene.contact.polygon_vertices.to_numpy()
        points_flattened = polygon.flatten()
        posx = np.ascontiguousarray(polygon[:, 0])
        posy = np.ascontiguousarray(polygon[:, 1])
        posz = np.zeros(posx.shape[0])
        connectivity = np.arange(polygon.shape[0])
        offsets = np.array([polygon.shape[0]])
        cell_types = np.array([7])
        unstructuredGridToVTK(
            self.vtk_path + f"/GraphicObject{sims.current_print:06d}",
            posx,
            posy,
            posz,
            connectivity,
            offsets,
            cell_types,
        )

    def VisualizeParticle(self, sims: Simulation, position, velocity, volume, state_vars):
        posx = np.ascontiguousarray(position[:, 0], dtype=np.float64)
        posy = np.ascontiguousarray(position[:, 1], dtype=np.float64)
        posz = np.ascontiguousarray(position[:, 2], dtype=np.float64)
        velx = np.ascontiguousarray(velocity[:, 0], dtype=np.float64)
        vely = np.ascontiguousarray(velocity[:, 1], dtype=np.float64)
        velz = np.ascontiguousarray(velocity[:, 2], dtype=np.float64)
        data = {"velocity": (velx, vely, velz), "volume": volume}
        data.update(state_vars)
        pointsToVTK(self.vtk_path + f"/GraphicMPMParticle{sims.current_print:06d}", posx, posy, posz, data=data)

    def VisualizeParticle2D(self, sims: Simulation, position, velocity, volume, state_vars):
        posx = np.ascontiguousarray(position[:, 0], dtype=np.float64)
        posy = np.ascontiguousarray(position[:, 1], dtype=np.float64)
        posz = np.zeros(position.shape[0], dtype=np.float64)
        velx = np.ascontiguousarray(velocity[:, 0], dtype=np.float64)
        vely = np.ascontiguousarray(velocity[:, 1], dtype=np.float64)
        velz = np.zeros(velocity.shape[0], dtype=np.float64)
        data = {"velocity": (velx, vely, velz), "volume": volume}
        data.update(state_vars)
        pointsToVTK(self.vtk_path + f"/GraphicMPMParticle{sims.current_print:06d}", posx, posy, posz, data=data)

    def _visualize_particle_named(self, sims: Simulation, name, position, velocity, volume, state_vars):
        if position.shape[0] == 0:
            return
        if sims.dimension == 2:
            posx = np.ascontiguousarray(position[:, 0], dtype=np.float64)
            posy = np.ascontiguousarray(position[:, 1], dtype=np.float64)
            posz = np.zeros(position.shape[0], dtype=np.float64)
            velx = np.ascontiguousarray(velocity[:, 0], dtype=np.float64)
            vely = np.ascontiguousarray(velocity[:, 1], dtype=np.float64)
            velz = np.zeros(velocity.shape[0], dtype=np.float64)
        else:
            posx = np.ascontiguousarray(position[:, 0], dtype=np.float64)
            posy = np.ascontiguousarray(position[:, 1], dtype=np.float64)
            posz = np.ascontiguousarray(position[:, 2], dtype=np.float64)
            velx = np.ascontiguousarray(velocity[:, 0], dtype=np.float64)
            vely = np.ascontiguousarray(velocity[:, 1], dtype=np.float64)
            velz = np.ascontiguousarray(velocity[:, 2], dtype=np.float64)
        data = {"velocity": (velx, vely, velz), "volume": np.ascontiguousarray(volume)}
        data.update(state_vars)
        pointsToVTK(self.vtk_path + f"/{name}{sims.current_print:06d}", posx, posy, posz, data=data)

    def _slice_vtk_state_vars(self, state_vars, mask):
        sliced = {}
        expected_count = mask.shape[0]
        for key, value in state_vars.items():
            if isinstance(value, tuple):
                components = []
                valid = True
                for component in value:
                    array = np.asarray(component)
                    if array.ndim == 0 or array.shape[0] != expected_count:
                        valid = False
                        break
                    components.append(np.ascontiguousarray(array[mask]))
                if valid:
                    sliced[key] = tuple(components)
            else:
                array = np.asarray(value)
                if array.ndim > 0 and array.shape[0] == expected_count:
                    sliced[key] = np.ascontiguousarray(array[mask])
        return sliced

    def _flatten_vtk_data(self, data, expected_count):
        flattened = {}
        for key, value in data.items():
            if isinstance(value, tuple):
                components = tuple(np.ascontiguousarray(component).reshape(-1) for component in value)
                if all(component.size == expected_count for component in components):
                    flattened[key] = components
            else:
                array = np.ascontiguousarray(value).reshape(-1)
                if array.size == expected_count:
                    flattened[key] = array
        return flattened

    def _append_cell_field(self, cell_data, key, field, cell_shape, dtype=None):
        if field is None or not hasattr(field, "to_numpy"):
            return
        array = field.to_numpy()
        if dtype is not None:
            array = np.asarray(array, dtype=dtype)
        else:
            array = np.asarray(array)
        expected_size = int(np.prod(cell_shape))
        if array.size == expected_size:
            cell_data[key] = np.ascontiguousarray(array.reshape(*cell_shape, -1))

    def _regular_grid_axes(self, sims: Simulation, coords, element=None):
        if element is not None and hasattr(element, "cnum") and hasattr(element, "grid_size"):
            cnum = np.asarray(element.cnum, dtype=np.int64)[: sims.dimension]
            grid_size = np.asarray(element.grid_size, dtype=np.float64)[: sims.dimension]
            ghost_cell = int(getattr(element, "ghost_cell", 0))
            return tuple(
                np.linspace(
                    -ghost_cell * grid_size[d],
                    (cnum[d] - ghost_cell) * grid_size[d],
                    int(cnum[d] + 1),
                    dtype=np.float64,
                )
                for d in range(sims.dimension)
            )
        coordx = np.unique(np.ascontiguousarray(coords[:, 0], dtype=np.float64))
        coordy = np.unique(np.ascontiguousarray(coords[:, 1], dtype=np.float64))
        if sims.dimension == 2:
            return coordx, coordy
        coordz = np.unique(np.ascontiguousarray(coords[:, 2], dtype=np.float64))
        return coordx, coordy, coordz

    def _regular_grid_coords(self, sims: Simulation, coords, element=None):
        axes = self._regular_grid_axes(sims, coords, element)
        if sims.dimension == 2:
            xx, yy = np.meshgrid(*axes, indexing="ij")
            return np.ascontiguousarray(
                np.column_stack((xx.reshape(-1), yy.reshape(-1))),
                dtype=np.float64,
            )
        xx, yy, zz = np.meshgrid(*axes, indexing="ij")
        return np.ascontiguousarray(
            np.column_stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1))),
            dtype=np.float64,
        )

    def _write_regular_grid_as_unstructured_vtk(
        self, sims: Simulation, coords, point_data=None, cell_data=None, element=None
    ):
        point_data = {} if point_data is None else point_data
        cell_data = {} if cell_data is None else cell_data
        axes = self._regular_grid_axes(sims, coords, element)

        if sims.dimension == 2:
            coordx, coordy = axes
            xx, yy = np.meshgrid(coordx, coordy, indexing="ij")
            posx = np.ascontiguousarray(xx.reshape(-1), dtype=np.float64)
            posy = np.ascontiguousarray(yy.reshape(-1), dtype=np.float64)
            posz = np.zeros(posx.shape[0], dtype=np.float64)

            ny = coordy.shape[0]
            cell_count = max(coordx.shape[0] - 1, 0) * max(coordy.shape[0] - 1, 0)
            connectivity = np.empty((cell_count, 4), dtype=np.int64)
            cursor = 0
            for i in range(coordx.shape[0] - 1):
                for j in range(coordy.shape[0] - 1):
                    n0 = i * ny + j
                    connectivity[cursor, :] = (n0, n0 + ny, n0 + ny + 1, n0 + 1)
                    cursor += 1
            vtk_cell_type = VtkQuad.tid
        else:
            coordx, coordy, coordz = axes
            xx, yy, zz = np.meshgrid(coordx, coordy, coordz, indexing="ij")
            posx = np.ascontiguousarray(xx.reshape(-1), dtype=np.float64)
            posy = np.ascontiguousarray(yy.reshape(-1), dtype=np.float64)
            posz = np.ascontiguousarray(zz.reshape(-1), dtype=np.float64)

            ny = coordy.shape[0]
            nz = coordz.shape[0]
            cell_count = max(coordx.shape[0] - 1, 0) * max(coordy.shape[0] - 1, 0) * max(coordz.shape[0] - 1, 0)
            connectivity = np.empty((cell_count, 8), dtype=np.int64)
            cursor = 0
            for i in range(coordx.shape[0] - 1):
                for j in range(coordy.shape[0] - 1):
                    for k in range(coordz.shape[0] - 1):
                        n0 = (i * ny + j) * nz + k
                        n1 = ((i + 1) * ny + j) * nz + k
                        n2 = ((i + 1) * ny + j + 1) * nz + k
                        n3 = (i * ny + j + 1) * nz + k
                        connectivity[cursor, :] = (
                            n0,
                            n1,
                            n2,
                            n3,
                            n0 + 1,
                            n1 + 1,
                            n2 + 1,
                            n3 + 1,
                        )
                        cursor += 1
            vtk_cell_type = VtkHexahedron.tid

        nodes_per_cell = connectivity.shape[1]
        offsets = np.arange(
            nodes_per_cell,
            (cell_count + 1) * nodes_per_cell,
            nodes_per_cell,
            dtype=np.int64,
        )
        cell_types = np.full(cell_count, vtk_cell_type, dtype=np.uint8)
        unstructuredGridToVTK(
            self.vtk_path + f"/GraphicMPMGrid{sims.current_print:06d}",
            posx,
            posy,
            posz,
            np.ascontiguousarray(connectivity.reshape(-1), dtype=np.int64),
            offsets,
            cell_types,
            pointData=self._flatten_vtk_data(point_data, posx.shape[0]),
            cellData=self._flatten_vtk_data(cell_data, cell_count),
        )

    def VisualizeGrid(self, sims: Simulation, coords, point_data=None, cell_data=None, element=None):
        self._write_regular_grid_as_unstructured_vtk(sims, coords, point_data, cell_data, element)

    def VisualizeGrid2D(self, sims: Simulation, coords, point_data=None, cell_data=None, element=None):
        self._write_regular_grid_as_unstructured_vtk(sims, coords, point_data, cell_data, element)

    def VisualizeAdaptiveGrid(self, sims: Simulation, mesh):
        coords = mesh["coords"]
        posx = np.ascontiguousarray(coords[:, 0], dtype=np.float64)
        posy = np.ascontiguousarray(coords[:, 1], dtype=np.float64)
        if sims.dimension == 2:
            posz = np.zeros(coords.shape[0], dtype=np.float64)
            vtk_cell_type = VtkQuad.tid
        else:
            posz = np.ascontiguousarray(coords[:, 2], dtype=np.float64)
            vtk_cell_type = VtkHexahedron.tid

        connectivity = np.ascontiguousarray(mesh["connectivity"].reshape(-1), dtype=np.int64)
        nodes_per_cell = mesh["connectivity"].shape[1]
        cell_count = mesh["connectivity"].shape[0]
        offsets = np.arange(
            nodes_per_cell,
            (cell_count + 1) * nodes_per_cell,
            nodes_per_cell,
            dtype=np.int64,
        )
        cell_types = np.full(cell_count, vtk_cell_type, dtype=np.uint8)
        unstructuredGridToVTK(
            self.vtk_path + f"/GraphicMPMGrid{sims.current_print:06d}",
            posx,
            posy,
            posz,
            connectivity,
            offsets,
            cell_types,
            cellData={
                "grid_level": mesh["grid_level"],
                "coarse_cell_id": mesh["coarse_cell_id"],
            },
            pointData={"logical_node_id": mesh["logical_node_id"]},
        )

    def VisualizeTHBGrid(self, sims: Simulation, coords, point_data=None, cell_data=None):
        coords = np.ascontiguousarray(coords, dtype=np.float64)
        posx = np.ascontiguousarray(coords[:, 0], dtype=np.float64)
        posy = np.ascontiguousarray(coords[:, 1], dtype=np.float64)
        posz = np.zeros(coords.shape[0], dtype=np.float64)
        if sims.dimension == 3:
            posz = np.ascontiguousarray(coords[:, 2], dtype=np.float64)

        node_count = coords.shape[0]
        connectivity = np.arange(node_count, dtype=np.int64)
        offsets = np.arange(1, node_count + 1, dtype=np.int64)
        cell_types = np.full(node_count, VtkVertex.tid, dtype=np.uint8)
        unstructuredGridToVTK(
            self.vtk_path + f"/GraphicMPMGrid{sims.current_print:06d}",
            posx,
            posy,
            posz,
            connectivity,
            offsets,
            cell_types,
            pointData=self._flatten_vtk_data(point_data or {}, node_count),
            cellData=self._flatten_vtk_data(cell_data or {}, node_count),
        )

    def VisualizeSparseGrid(self, sims: Simulation, mesh):
        coords = mesh["coords"]
        posx = np.ascontiguousarray(coords[:, 0], dtype=np.float64)
        posy = np.ascontiguousarray(coords[:, 1], dtype=np.float64)
        if sims.dimension == 2:
            posz = np.zeros(coords.shape[0], dtype=np.float64)
            vtk_cell_type = VtkQuad.tid
            block_z = np.zeros(mesh["block_index"].shape[0], dtype=np.int32)
        else:
            posz = np.ascontiguousarray(coords[:, 2], dtype=np.float64)
            vtk_cell_type = VtkHexahedron.tid
            block_z = np.ascontiguousarray(mesh["block_index"][:, 2], dtype=np.int32)

        connectivity = np.ascontiguousarray(mesh["connectivity"].reshape(-1), dtype=np.int64)
        cell_count = mesh["connectivity"].shape[0]
        nodes_per_cell = mesh["connectivity"].shape[1]
        offsets = np.arange(nodes_per_cell, (cell_count + 1) * nodes_per_cell, nodes_per_cell, dtype=np.int64)
        cell_types = np.full(cell_count, vtk_cell_type, dtype=np.uint8)
        block_index = mesh["block_index"]
        unstructuredGridToVTK(
            self.vtk_path + f"/GraphicMPMGrid{sims.current_print:06d}",
            posx,
            posy,
            posz,
            connectivity,
            offsets,
            cell_types,
            cellData={
                "active_block_id": mesh["active_block_id"],
                "compact_block_id": mesh["compact_block_id"],
                "block_index": (
                    np.ascontiguousarray(block_index[:, 0], dtype=np.int32),
                    np.ascontiguousarray(block_index[:, 1], dtype=np.int32),
                    block_z,
                ),
            },
        )

    def MonitorParticleCoupling(self, sims: Simulation, scene: myScene):
        particle_num = scene.particleNum[0]
        output = self.MonitorParticleBase(sims, scene, particle_num)

        coupling = field_to_numpy_prefix(scene.particle.coupling, particle_num)
        radius = field_to_numpy_prefix(scene.particle.rad, particle_num)
        stress = field_to_numpy_prefix(scene.particle.stress, particle_num)
        external_force = field_to_numpy_prefix(scene.particle.external_force, particle_num)
        velocity_gradient = field_to_numpy_prefix(scene.particle.velocity_gradient, particle_num)
        state_vars: dict = scene.material.get_state_vars_dict(start_index=0, end_index=scene.particleNum[0])
        output.update(
            {
                "coupling": coupling,
                "stress": stress,
                "radius": radius,
                "external_force": external_force,
                "velocity_gradient": velocity_gradient,
                "state_vars": state_vars,
            }
        )

        if sims.free_surface_detection:
            free_surface = field_to_numpy_prefix(scene.particle.free_surface, particle_num)
            mass_density = field_to_numpy_prefix(scene.particle.mass_density, particle_num)
            state_vars.update({"free_surface": free_surface, "mass_density": mass_density})
        if sims.boundary_direction_detection:
            normal = field_to_numpy_prefix(scene.particle.normal, particle_num)
            normalx = np.ascontiguousarray(normal[:, 0])
            normaly = np.ascontiguousarray(normal[:, 1])
            normalz = np.ascontiguousarray(normal[:, 2])
            state_vars.update({"normal": (normalx, normaly, normalz)})
        self.visualizeParticle(sims, output["position"], output["velocity"], output["volume"], state_vars)
        np.savez(self.particle_path + f"/MPMParticle{sims.current_print:06d}", **output)

    def MonitorParticleTwoPhase(self, sims: Simulation, scene: myScene):
        particle_num = scene.particleNum[0]
        particleID = field_to_numpy_prefix(scene.particle.particleID, particle_num)
        position = field_to_numpy_prefix(scene.particle.x, particle_num)
        bodyID = field_to_numpy_prefix(scene.particle.bodyID, particle_num)
        materialID = field_to_numpy_prefix(scene.particle.materialID, particle_num)
        active = field_to_numpy_prefix(scene.particle.active, particle_num)
        phase = None
        if hasattr(scene.particle, "phase"):
            phase = field_to_numpy_prefix(scene.particle.phase, particle_num)
        coupling = None
        if hasattr(scene.particle, "coupling"):
            coupling = field_to_numpy_prefix(scene.particle.coupling, particle_num)
        velocity = field_to_numpy_prefix(scene.particle.v, particle_num)
        solid_velocity = field_to_numpy_prefix(scene.particle.vs, particle_num)
        fluid_velocity = field_to_numpy_prefix(scene.particle.vf, particle_num)
        mass = field_to_numpy_prefix(scene.particle.m, particle_num)
        solid_mass = field_to_numpy_prefix(scene.particle.ms, particle_num)
        fluid_mass = field_to_numpy_prefix(scene.particle.mf, particle_num)
        volume = field_to_numpy_prefix(scene.particle.vol, particle_num)
        stress = field_to_numpy_prefix(scene.particle.stress, particle_num)
        pressure = field_to_numpy_prefix(scene.particle.pressure, particle_num)
        permeability = field_to_numpy_prefix(scene.particle.permeability, particle_num)
        porosity = field_to_numpy_prefix(scene.particle.porosity, particle_num)
        solid_velocity_gradient = field_to_numpy_prefix(scene.particle.solid_velocity_gradient, particle_num)
        fluid_velocity_gradient = field_to_numpy_prefix(scene.particle.fluid_velocity_gradient, particle_num)
        fix_v = field_to_numpy_prefix(scene.particle.fix_v, particle_num)
        state_vars: dict = scene.material.get_state_vars_dict(start_index=0, end_index=scene.particleNum[0])
        state_vars.update({"pressure": pressure})
        if phase is not None:
            state_vars.update({"phase": phase, "bodyID": bodyID, "materialID": materialID})
        free_surface = None
        if sims.free_surface_detection:
            free_surface = field_to_numpy_prefix(scene.particle.free_surface, particle_num)
            state_vars.update({"free_surface": free_surface})
        vtk_state_vars = dict(state_vars)
        vtk_state_vars.update(
            {
                "porosity": porosity,
                "solid_velocity": vtk_vector(solid_velocity),
                "fluid_velocity": vtk_vector(fluid_velocity),
            }
        )
        if sims.visualize and sims.material_type == "TwoPhaseDoubleLayer" and phase is not None:
            solid_mask = phase == 1
            fluid_mask = phase == 2
            self._visualize_particle_named(
                sims,
                "GraphicMPMSolidParticle",
                position[solid_mask],
                solid_velocity[solid_mask],
                volume[solid_mask],
                self._slice_vtk_state_vars(vtk_state_vars, solid_mask),
            )
            self._visualize_particle_named(
                sims,
                "GraphicMPMFluidParticle",
                position[fluid_mask],
                fluid_velocity[fluid_mask],
                volume[fluid_mask],
                self._slice_vtk_state_vars(vtk_state_vars, fluid_mask),
            )
        else:
            self.visualizeParticle(sims, position, velocity, volume, vtk_state_vars)
        output = dict(
            t_current=sims.current_time,
            body_num=particle_num,
            particleID=particleID,
            bodyID=bodyID,
            materialID=materialID,
            active=active,
            mass=mass,
            volume=volume,
            position=position,
            velocity=velocity,
            stress=stress,
            solid_velocity_gradient=solid_velocity_gradient,
            fluid_velocity_gradient=fluid_velocity_gradient,
            fix_v=fix_v,
            state_vars=state_vars,
            solid_velocity=solid_velocity,
            fluid_velocity=fluid_velocity,
            solid_mass=solid_mass,
            fluid_mass=fluid_mass,
            pressure=pressure,
            permeability=permeability,
            porosity=porosity,
        )
        if phase is not None:
            output["phase"] = phase
        if coupling is not None:
            output["coupling"] = coupling
        if free_surface is not None:
            output["free_surface"] = free_surface
        np.savez(self.particle_path + f"/MPMParticle{sims.current_print:06d}", **output)

    def MonitorIncompressibleParticleCoupling(self, sims: Simulation, scene: myScene):
        particle_num = scene.particleNum[0]
        output = self.MonitorParticleBase(sims, scene, particle_num)

        pressure = self.sample_incompressible_cell_pressure(sims, scene, output["position"])
        if (
            sims.dimension == 2
            and hasattr(scene.particle, "xvelocity_gradient")
            and hasattr(scene.particle, "yvelocity_gradient")
        ):
            xvelocity_gradient = field_to_numpy_prefix(scene.particle.xvelocity_gradient, particle_num)
            yvelocity_gradient = field_to_numpy_prefix(scene.particle.yvelocity_gradient, particle_num)
            output.update({"xvelocity_gradient": xvelocity_gradient, "yvelocity_gradient": yvelocity_gradient})
        elif (
            sims.dimension == 3
            and hasattr(scene.particle, "xvelocity_gradient")
            and hasattr(scene.particle, "yvelocity_gradient")
            and hasattr(scene.particle, "zvelocity_gradient")
        ):
            xvelocity_gradient = field_to_numpy_prefix(scene.particle.xvelocity_gradient, particle_num)
            yvelocity_gradient = field_to_numpy_prefix(scene.particle.yvelocity_gradient, particle_num)
            zvelocity_gradient = field_to_numpy_prefix(scene.particle.zvelocity_gradient, particle_num)
            output.update(
                {
                    "xvelocity_gradient": xvelocity_gradient,
                    "yvelocity_gradient": yvelocity_gradient,
                    "zvelocity_gradient": zvelocity_gradient,
                }
            )
        elif hasattr(scene.particle, "velocity_gradient"):
            output["velocity_gradient"] = field_to_numpy_prefix(scene.particle.velocity_gradient, particle_num)
        output["coupling"] = field_to_numpy_prefix(scene.particle.coupling, particle_num)
        state_vars = scene.material.get_state_vars_dict(start_index=0, end_index=scene.particleNum[0])
        state_vars.update({"pressure": pressure})
        output.update({"pressure": pressure})

        self.visualizeParticle(sims, output["position"], output["velocity"], output["volume"], state_vars)
        np.savez(self.particle_path + f"/MPMParticle{sims.current_print:06d}", **output)

    def MonitorParticle(self, sims: Simulation, scene: myScene):
        particle_num = scene.particleNum[0]
        output = self.MonitorParticleBase(sims, scene, particle_num)

        stress = field_to_numpy_prefix(scene.particle.stress, particle_num)
        velocity_gradient = field_to_numpy_prefix(scene.particle.velocity_gradient, particle_num)
        state_vars = scene.material.get_state_vars_dict(start_index=0, end_index=scene.particleNum[0])
        if getattr(scene.element, "adaptive", False):
            grid_level = field_to_numpy_prefix(scene.element.particle_level, particle_num)
            particle_refined = field_to_numpy_prefix(scene.element.particle_refined, particle_num)
            state_vars.update({"grid_level": grid_level})
            output.update(
                {
                    "grid_level": grid_level,
                    "particle_refined": particle_refined,
                    "refined_cell": scene.element.refined_cell.to_numpy(),
                    "coarse_cnum": np.asarray(scene.element.coarse_cnum, dtype=np.int32),
                }
            )
            if scene.element.bridging_domain:
                output["bridge_alpha"] = field_to_numpy_prefix(scene.element.bridge_alpha, particle_num)
        output.update({"stress": stress, "velocity_gradient": velocity_gradient, "state_vars": state_vars})

        self.visualizeParticle(sims, output["position"], output["velocity"], output["volume"], state_vars)
        np.savez(self.particle_path + f"/MPMParticle{sims.current_print:06d}", **output)

    def MonitorImplicitParticle(self, sims: Simulation, scene: myScene):
        particle_num = scene.particleNum[0]
        output = self.MonitorParticleBase(sims, scene, particle_num)

        stress = field_to_numpy_prefix(scene.particle.stress, particle_num)
        acceleration = field_to_numpy_prefix(scene.particle.a, particle_num)
        state_vars = scene.material.get_state_vars_dict(start_index=0, end_index=scene.particleNum[0])
        output.update({"stress": stress, "acceleration": acceleration, "state_vars": state_vars})

        self.visualizeParticle(sims, output["position"], output["velocity"], output["volume"], state_vars)
        np.savez(self.particle_path + f"/MPMParticle{sims.current_print:06d}", **output)

    def MonitorIncompressibleParticle(self, sims: Simulation, scene: myScene):
        particle_num = scene.particleNum[0]
        output = self.MonitorParticleBase(sims, scene, particle_num)

        pressure = self.sample_incompressible_cell_pressure(sims, scene, output["position"])
        output.update(
            {
                "pressure": pressure,
                "coupling": field_to_numpy_prefix(scene.particle.coupling, particle_num),
                "velocity_gradient": field_to_numpy_prefix(scene.particle.velocity_gradient, particle_num),
            }
        )
        self.visualizeParticle(sims, output["position"], output["velocity"], output["volume"], {"pressure": pressure})
        np.savez(self.particle_path + f"/MPMParticle{sims.current_print:06d}", **output)

    def sample_incompressible_cell_pressure(self, sims: Simulation, scene: myScene, position):
        particle_num = position.shape[0]
        if particle_num == 0:
            return np.zeros(0, dtype=np.float64)
        if not hasattr(scene.element, "cell") or scene.element.cell is None or scene.element.cell.pressure is None:
            return np.zeros(particle_num, dtype=np.float64)

        dim = sims.dimension
        pressure_field = scene.element.cell.pressure.to_numpy()
        cell_type_field = scene.element.cell.type.to_numpy()
        pressure_field = _normalize_closed_domain_pressure(pressure_field, cell_type_field)
        surface_tension_field = scene.element.cell.surface_tension.to_numpy()
        cnum = np.asarray(scene.element.cnum, dtype=np.int64)[:dim]
        ghost_cell = int(scene.element.ghost_cell)
        active_cnum = cnum - 2 * ghost_cell
        grid_size = np.asarray(scene.element.grid_size, dtype=np.float64)[:dim]
        if np.any(active_cnum <= 0) or np.any(grid_size <= 0.0):
            return np.zeros(particle_num, dtype=np.float64)

        atmospheric_pressure = self.get_incompressible_atmospheric_pressure(scene)
        cell_coord = position[:, :dim] / grid_size - 0.5
        base = np.floor(cell_coord).astype(np.int64)
        fraction = cell_coord - base

        pressure = np.zeros(particle_num, dtype=np.float64)
        weight_sum = np.zeros(particle_num, dtype=np.float64)

        for offset in product((0, 1), repeat=dim):
            offset_array = np.asarray(offset, dtype=np.int64)
            logical_cell = base + offset_array
            storage_cell = logical_cell + ghost_cell
            valid = np.all((storage_cell >= 0) & (storage_cell < cnum), axis=1)
            if not np.any(valid):
                continue

            weight = np.ones(particle_num, dtype=np.float64)
            for d, o in enumerate(offset):
                if o == 0:
                    weight *= 1.0 - fraction[:, d]
                else:
                    weight *= fraction[:, d]

            valid_indices = np.flatnonzero(valid)
            storage_tuple = tuple(storage_cell[valid, d] for d in range(dim))
            cell_type = cell_type_field[storage_tuple]
            cell_pressure = pressure_field[storage_tuple].astype(np.float64)
            cell_surface_tension = surface_tension_field[storage_tuple].astype(np.float64)
            sample_pressure = np.full(cell_type.shape, np.nan, dtype=np.float64)
            fluid_sample = cell_type == 1
            sample_pressure[fluid_sample] = cell_pressure[fluid_sample]

            air_sample = cell_type == 0
            if np.any(air_sample):
                valid_storage_cell = storage_cell[valid]
                interface_air = np.zeros(cell_type.shape, dtype=bool)
                for d in range(dim):
                    for side in (-1, 1):
                        neighbor = valid_storage_cell.copy()
                        neighbor[:, d] += side
                        neighbor_inside = np.all((neighbor >= 0) & (neighbor < cnum), axis=1)
                        if not np.any(neighbor_inside):
                            continue
                        neighbor_type = np.full(cell_type.shape, -1, dtype=np.int32)
                        neighbor_tuple = tuple(neighbor[neighbor_inside, axis] for axis in range(dim))
                        neighbor_type[neighbor_inside] = cell_type_field[neighbor_tuple]
                        interface_air |= air_sample & (neighbor_type == 1)
                sample_pressure[interface_air] = atmospheric_pressure + cell_surface_tension[interface_air]
            sample_valid = np.isfinite(sample_pressure)
            if not np.any(sample_valid):
                continue

            particle_indices = valid_indices[sample_valid]
            pressure[particle_indices] += weight[particle_indices] * sample_pressure[sample_valid]
            weight_sum[particle_indices] += weight[particle_indices]

        sampled = weight_sum > 1e-12
        pressure[sampled] /= weight_sum[sampled]
        if np.any(~sampled):
            nearest_cell = np.floor(position[:, :dim] / grid_size).astype(np.int64)
            nearest_cell = np.minimum(np.maximum(nearest_cell, 0), active_cnum - 1)
            nearest_storage = nearest_cell + ghost_cell
            nearest_tuple = tuple(nearest_storage[~sampled, d] for d in range(dim))
            pressure[~sampled] = pressure_field[nearest_tuple].astype(np.float64)
        return pressure

    def get_incompressible_atmospheric_pressure(self, scene: myScene):
        mat_props = getattr(scene.material, "matProps", None)
        if mat_props is None:
            return 0.0
        try:
            if mat_props.size() > 1:
                return float(mat_props[1].atmospheric_pressure)
        except (AttributeError, IndexError, TypeError):
            return 0.0
        return 0.0

    def MonitorParticleBase(self, sims: Simulation, scene: myScene, particle_num):
        particleID = field_to_numpy_prefix(scene.particle.particleID, particle_num)
        position = field_to_numpy_prefix(scene.particle.x, particle_num)
        bodyID = field_to_numpy_prefix(scene.particle.bodyID, particle_num)
        materialID = field_to_numpy_prefix(scene.particle.materialID, particle_num)
        active = field_to_numpy_prefix(scene.particle.active, particle_num)
        velocity = field_to_numpy_prefix(scene.particle.v, particle_num)
        mass = field_to_numpy_prefix(scene.particle.m, particle_num)
        volume = field_to_numpy_prefix(scene.particle.vol, particle_num)
        fix_v = field_to_numpy_prefix(scene.particle.fix_v, particle_num)
        if getattr(scene.element, "adaptive", False):
            psize = field_to_numpy_prefix(scene.element.particle_size, particle_num)
        else:
            psize = scene.psize
        return {
            "t_current": sims.current_time,
            "body_num": particle_num,
            "active": active,
            "particleID": particleID,
            "bodyID": bodyID,
            "materialID": materialID,
            "mass": mass,
            "volume": volume,
            "position": position,
            "velocity": velocity,
            "fix_v": fix_v,
            "psize": psize,
        }

    def MonitorContactGrid(self, sims: Simulation, scene: myScene):
        coords = self._regular_grid_coords(sims, None, scene.element)
        output = {"t_current": sims.current_time, "dims": scene.element.gnum, "coords": coords}
        if hasattr(scene.node, "contact_force"):
            output["contact_force"] = scene.node.contact_force.to_numpy()
        if hasattr(scene.node, "contact_force_s"):
            output["contact_force_s"] = scene.node.contact_force_s.to_numpy()
        if hasattr(scene.node, "contact_force_f"):
            output["contact_force_f"] = scene.node.contact_force_f.to_numpy()
        if hasattr(scene.node, "grad_domain"):
            output["normal"] = scene.node.grad_domain.to_numpy()
        self.visualizeGrid(sims, coords, element=scene.element)
        np.savez(self.grid_path + f"/MPMGrid{sims.current_print:06d}", **output)

    def MonitorGrid(self, sims: Simulation, scene: myScene):
        if getattr(sims, "isTHB", False):
            return self.MonitorTHBGrid(sims, scene)
        if getattr(scene.element, "adaptive", False):
            return self.MonitorAdaptiveGrid(sims, scene)
        if scene.sparse_grid is not None and getattr(sims, "sparse_grid_visualize_active_blocks", False):
            return self.MonitorSparseGrid(sims, scene)

        coords = self._regular_grid_coords(sims, None, scene.element)

        point_data = {}
        if sims.shape_function == "QuadBspline" or sims.shape_function == "CubicBspline":
            boundary_type = np.ascontiguousarray(scene.element.boundary_type.to_numpy().reshape(-1, 3), dtype=np.int32)
            xboundary_type = np.ascontiguousarray(boundary_type[:, 0])
            yboundary_type = np.ascontiguousarray(boundary_type[:, 1])
            zboundary_type = np.ascontiguousarray(boundary_type[:, 2])
            point_data.update({"boundy_type": (xboundary_type, yboundary_type, zboundary_type)})

        cell_data = {}
        cell = getattr(scene.element, "cell", None)
        if cell is not None and hasattr(cell, "type") and getattr(cell, "type") is not None:
            cell_shape = tuple(np.asarray(scene.element.cnum, dtype=np.int64)[: sims.dimension])
            self._append_cell_field(cell_data, "cell_type", getattr(cell, "type", None), cell_shape, np.int32)
            self._append_cell_field(cell_data, "cell_pressure", getattr(cell, "pressure", None), cell_shape)
            self._append_cell_field(
                cell_data, "cell_surface_tension", getattr(cell, "surface_tension", None), cell_shape
            )
            self._append_cell_field(cell_data, "cell_fluid_sdf", getattr(cell, "fluid_sdf", None), cell_shape)
            self._append_cell_field(cell_data, "cell_solid_sdf", getattr(cell, "solid_sdf", None), cell_shape)
            if sims.material_type == "Fluid" and "Implicit" in sims.solver_type and "cell_pressure" in cell_data:
                cell_data["cell_pressure"] = _normalize_closed_domain_pressure(
                    cell_data["cell_pressure"], cell_data["cell_type"]
                )

        self.visualizeGrid(sims, coords, cell_data=cell_data, point_data=point_data, element=scene.element)
        output = {"t_current": sims.current_time, "dims": scene.element.gnum, "coords": coords}
        output.update(cell_data)
        np.savez(self.grid_path + f"/MPMGrid{sims.current_print:06d}", **output)

    def MonitorSparseGrid(self, sims: Simulation, scene: myScene):
        sparse_grid = scene.sparse_grid
        active_count = sparse_grid.get_active_blocks()
        active_block_ids = np.ascontiguousarray(
            sparse_grid.active_block_ids.to_numpy()[:active_count],
            dtype=np.int64,
        )
        mesh = build_sparse_block_mesh(
            sims.dimension,
            active_block_ids,
            sparse_grid.block_count_np,
            sparse_grid.block_size,
            scene.element.grid_size,
            sparse_grid.gnum_np,
        )
        if active_count > 0:
            self.VisualizeSparseGrid(sims, mesh)
        np.savez(
            self.grid_path + f"/MPMGrid{sims.current_print:06d}",
            t_current=sims.current_time,
            backend="BlockScan",
            block_size=sparse_grid.block_size,
            active_block_count=active_count,
            active_node_slots=sparse_grid.get_active_node_slots(),
            **mesh,
        )

    def MonitorTHBGrid(self, sims: Simulation, scene: myScene):
        coords = np.ascontiguousarray(scene.element.nodal_coords, dtype=np.float64)
        logical_node_id = np.arange(coords.shape[0], dtype=np.int64)
        node_level = np.asarray(scene.element.Nlevel, dtype=np.int32)
        node_type = np.asarray(scene.element.Ntype, dtype=np.int32)
        if node_level.ndim == 1:
            node_level = node_level.reshape(-1, 1)
        if node_type.ndim == 1:
            node_type = node_type.reshape(-1, 1)

        thb_level = np.max(node_level, axis=1).astype(np.int32, copy=False)
        point_data = {
            "logical_node_id": logical_node_id,
            "thb_type_x": np.ascontiguousarray(node_type[:, 0], dtype=np.int32),
            "thb_type_y": np.ascontiguousarray(node_type[:, 1], dtype=np.int32),
        }
        cell_data = {
            "grid_level": np.ascontiguousarray(thb_level, dtype=np.int32),
            "thb_level_x": np.ascontiguousarray(node_level[:, 0], dtype=np.int32),
            "thb_level_y": np.ascontiguousarray(node_level[:, 1], dtype=np.int32),
        }
        if sims.visualize:
            self.VisualizeTHBGrid(sims, coords, point_data=point_data, cell_data=cell_data)

        output = {
            "t_current": sims.current_time,
            "dims": scene.element.gnum,
            "coords": coords,
            "logical_node_id": logical_node_id,
            "grid_level": cell_data["grid_level"],
            "thb_level_x": cell_data["thb_level_x"],
            "thb_level_y": cell_data["thb_level_y"],
            "thb_type_x": point_data["thb_type_x"],
            "thb_type_y": point_data["thb_type_y"],
        }
        np.savez(self.grid_path + f"/MPMGrid{sims.current_print:06d}", **output)

    def MonitorAdaptiveGrid(self, sims: Simulation, scene: myScene):
        refined_cell = scene.element.refined_cell.to_numpy()
        coarse_cnum = np.asarray(scene.element.coarse_cnum, dtype=np.int32)
        signature = (
            sims.dimension,
            tuple(coarse_cnum),
            refined_cell.tobytes(),
        )
        if signature != self.adaptive_grid_cache_signature:
            self.adaptive_grid_cache = build_adaptive_leaf_mesh(
                refined_cell,
                coarse_cnum,
                scene.element.fine_grid_size,
                getattr(scene.element, "max_level", None),
            )
            self.adaptive_grid_cache_signature = signature

        mesh = self.adaptive_grid_cache
        if sims.visualize:
            self.VisualizeAdaptiveGrid(sims, mesh)
        nodes_per_cell = mesh["connectivity"].shape[1]
        cell_count = mesh["connectivity"].shape[0]
        offsets = np.arange(
            nodes_per_cell,
            (cell_count + 1) * nodes_per_cell,
            nodes_per_cell,
            dtype=np.int64,
        )
        vtk_cell_type = VtkQuad.tid if sims.dimension == 2 else VtkHexahedron.tid
        cell_types = np.full(cell_count, vtk_cell_type, dtype=np.uint8)
        output = {
            "t_current": sims.current_time,
            "dims": scene.element.gnum,
            "coarse_cnum": coarse_cnum,
            "refined_cell": refined_cell,
            "coords": mesh["coords"],
            "logical_node_id": mesh["logical_node_id"],
            "connectivity": mesh["connectivity"],
            "offsets": offsets,
            "cell_types": cell_types,
            "grid_level": mesh["grid_level"],
            "coarse_cell_id": mesh["coarse_cell_id"],
            "bridging_domain": np.uint8(scene.element.bridging_domain),
        }
        if scene.element.bridging_domain:
            output.update(
                {
                    "bridging_cells": scene.element.bridging_cells,
                    "bridge_coarse_weight": (scene.element.bridge_coarse_weight.to_numpy()),
                }
            )
        np.savez(
            self.grid_path + f"/MPMGrid{sims.current_print:06d}",
            **output,
        )


def monitor_soft_material_point(recorder, sims, scene):
    point_num = int(scene.softPointNum[0])
    active = np.ascontiguousarray(scene.soft_point.active.to_numpy()[0:point_num])
    bodyID = np.ascontiguousarray(scene.soft_point.bodyID.to_numpy()[0:point_num])
    materialID = np.ascontiguousarray(scene.soft_point.materialID.to_numpy()[0:point_num])
    groupID = np.ascontiguousarray(scene.soft_point.groupID.to_numpy()[0:point_num])
    position = np.ascontiguousarray(scene.soft_point.x.to_numpy()[0:point_num])
    reference = np.ascontiguousarray(scene.soft_point.x0.to_numpy()[0:point_num])
    velocity = np.ascontiguousarray(scene.soft_point.v.to_numpy()[0:point_num])
    contact_force = np.ascontiguousarray(scene.soft_point.contact_force.to_numpy()[0:point_num])
    external_force = np.ascontiguousarray(scene.soft_point.external_force.to_numpy()[0:point_num])
    mass = np.ascontiguousarray(scene.soft_point.m.to_numpy()[0:point_num])
    volume = np.ascontiguousarray(scene.soft_point.vol0.to_numpy()[0:point_num])
    deformation = np.ascontiguousarray(scene.soft_point.F.to_numpy()[0:point_num])
    stress = np.ascontiguousarray(scene.soft_point.stress.to_numpy()[0:point_num])
    displacement = np.ascontiguousarray(position - reference)
    cauchy_stress, _ = pk1_to_cauchy(stress, deformation)
    von_mises = von_mises_stress(cauchy_stress)
    energy_output = {}
    if sims.energy_tracking:
        point_strain_energy = np.ascontiguousarray(scene.soft_point.strain_energy.to_numpy()[0:point_num])
        soft_num = int(scene.softNum[0])
        soft_body_ids = np.ascontiguousarray(scene.soft.bodyID.to_numpy()[0:soft_num])
        soft_local_damping_energy = np.ascontiguousarray(scene.soft.damp_energy.to_numpy()[0:soft_num])
        soft_kinetic_energy = np.zeros(soft_num, dtype=np.float64)
        soft_strain_energy = np.zeros(soft_num, dtype=np.float64)
        soft_gravitational_potential_energy = np.zeros(soft_num, dtype=np.float64)
        gravity = np.asarray(
            [float(sims.gravity[index]) for index in range(position.shape[1])],
            dtype=np.float64,
        )
        for soft_id, body_id in enumerate(soft_body_ids):
            body_mask = bodyID == body_id
            soft_kinetic_energy[soft_id] = 0.5 * np.sum(
                mass[body_mask] * np.einsum("ij,ij->i", velocity[body_mask], velocity[body_mask])
            )
            soft_strain_energy[soft_id] = np.sum(point_strain_energy[body_mask])
            soft_gravitational_potential_energy[soft_id] = -np.sum(
                mass[body_mask] * np.einsum("ij,j->i", position[body_mask], gravity)
            )
        energy_output = {
            "point_strain_energy": point_strain_energy,
            "soft_body_id": soft_body_ids,
            "soft_kinetic_energy": soft_kinetic_energy,
            "soft_strain_energy": soft_strain_energy,
            "soft_gravitational_potential_energy": soft_gravitational_potential_energy,
            "soft_local_damping_energy": soft_local_damping_energy,
        }
    if sims.visualize and point_num > 0:
        pointsToVTK(
            recorder.vtk_path + f"/GraphicLSMPMPoint{sims.current_print:06d}",
            np.ascontiguousarray(position[:, 0]),
            np.ascontiguousarray(position[:, 1]),
            np.ascontiguousarray(position[:, 2]),
            data={
                "bodyID": bodyID,
                "group": groupID,
                "displacement": vtk_vector(displacement),
                "velocity": vtk_vector(velocity),
                "contact_force": vtk_vector(contact_force),
                "external_force": vtk_vector(external_force),
                "von_mises_stress": von_mises,
                "cauchy_stress_xx": np.ascontiguousarray(cauchy_stress[:, 0, 0]),
                "cauchy_stress_yy": np.ascontiguousarray(cauchy_stress[:, 1, 1]),
                "cauchy_stress_zz": np.ascontiguousarray(cauchy_stress[:, 2, 2]),
            },
        )
    np.savez(
        recorder.particle_path + f"/LSMPMSoftPoint{sims.current_print:06d}",
        t_current=sims.current_time,
        point_num=point_num,
        active=active,
        bodyID=bodyID,
        materialID=materialID,
        groupID=groupID,
        position=position,
        reference=reference,
        velocity=velocity,
        contact_force=contact_force,
        external_force=external_force,
        mass=mass,
        volume=volume,
        F=deformation,
        stress=stress,
        **energy_output,
    )
