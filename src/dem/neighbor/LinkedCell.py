import taichi as ti
import math

from src.dem.neighbor.neighbor_kernel import *
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.SceneManager import myScene
from src.dem.Simulation import Simulation
from src.utils.constants import Threshold
from src.utils.linalg import no_operation
from src.utils.PrefixSum import PrefixSumExecutor, BLOCK_SZ as PREFIX_BLOCK_SZ
from src.utils.TypeDefination import vec3i

MAX_TAICHI_FIELD_SHAPE = 2147483647


class LinkedCell(NeighborBase):
    def __init__(self, sims: Simulation, scene):
        super().__init__(sims, scene)
        self.cnum = vec3i([0, 0, 0])
        self.particle_count = None
        self.particle_current = None
        self.ParticleID = None
        self.wall_count = None
        self.WallID = None
        self.cell_shape_limited = False
        self.max_linked_cell_sum = MAX_TAICHI_FIELD_SHAPE
        self.minimum_grid_size = 0.0
        self.allocated_cell_sum = 0
        self.adaptive_regrid_count = 0
        self.adaptive_cell_size = (
            sims.scheme == "LSMPM"
            and getattr(sims, "max_soft_body_num", 0) > 0
        )

    @staticmethod
    def prefix_sum_storage_length(batch_length):
        if batch_length <= 2:
            return batch_length
        storage_length = 0
        ele_num = batch_length
        while ele_num > 1:
            ele_num = int((ele_num + PREFIX_BLOCK_SZ - 1) / PREFIX_BLOCK_SZ)
            storage_length += PREFIX_BLOCK_SZ * ele_num
        return storage_length

    @classmethod
    def max_cell_sum_for_field_shape(cls):
        lower, upper = 1, MAX_TAICHI_FIELD_SHAPE - 1
        while lower < upper:
            middle = (lower + upper + 1) // 2
            if cls.prefix_sum_storage_length(middle + 1) <= MAX_TAICHI_FIELD_SHAPE:
                lower = middle
            else:
                upper = middle - 1
        return lower

    def calculate_cell_grid(self):
        self.igrid_size = 1. / self.grid_size
        cnum = [max(int(domain * self.igrid_size), 1) for domain in self.sims.domain]
        cell_sum = int(cnum[0]) * int(cnum[1]) * int(cnum[2])
        return cnum, cell_sum
        
    def manage_function(self, scene):
        self.manage_function_base(scene)
        self.manage_particle_function(scene)
        self.manage_wall_function()

    def manage_particle_function(self, scene: myScene):
        self.place_particle_to_cells = no_operation
        self.particle_hash_tables = no_operation
        if self.sims.max_particle_num > 1 or (self.sims.coupling and scene.particleNum[0] > 0):
            self.place_particle_to_cells = self.place_particle_to_cell
            self.particle_hash_tables = self.particle_hash_table

    def manage_wall_function(self):
        self.wall_hash_tables = no_operation
        self.place_wall_to_cells = no_operation
        if not self.sims.wall_type is None:
            if self.sims.wall_type == 0:
                self.place_wall_to_cells = self.place_plane_to_cell
                self.wall_hash_tables = self.plane_hash_table
            elif self.sims.wall_type == 1:
                '''if self.sims.static_wall:
                    self.place_wall_to_cells = self.place_static_facet_to_cell
                    self.wall_hash_tables = self.static_facet_hash_table
                else:'''
                self.place_wall_to_cells = self.place_facet_to_cell
                self.wall_hash_tables = self.facet_hash_table
            elif self.sims.wall_type == 2:
                self.place_wall_to_cells = self.place_patch_to_cell
                self.wall_hash_tables = self.patch_hash_table
        self.check_memory()

    def resize_neighbor(self, scene):
        del self.cell_pse, self.igrid_size, self.cnum, self.cellSum
        del self.particle_current, self.particle_count
        self.particle_wall_delete_vars()
        self.neighbor_initialze(scene, max(self.sims.domain), 0.)

    def neighbor_initialze(self, scene: myScene, min_bounding_rad, max_bounding_rad):
        rad_min, rad_max = scene.find_bounding_sphere_radius(self.sims)
        self.sims.set_max_bounding_sphere_radius(rad_max)
        self.sims.set_min_bounding_sphere_radius(rad_min)
        rad_max = max(max_bounding_rad, rad_max) if abs(rad_max) > 1e-15 else max_bounding_rad
        rad_min = min(min_bounding_rad, rad_min) if abs(rad_min) > 1e-15 else min_bounding_rad
        self.sims.set_verlet_distance(
            self._verlet_reference_radius(scene, rad_min))
        self.sims.set_potential_list_size(rad_max)
        
        self.contact_grid_size = 2 * (rad_max + self.sims.verlet_distance)
        self.grid_size = self.contact_grid_size
        self.plane_insert_factor = 0.5 + rad_max / self.grid_size
        if self.grid_size < 1e-3 * Threshold:
            raise RuntimeError("Particle radius is equal to zero!")
        cnum, cell_sum = self.calculate_cell_grid()
        self.max_linked_cell_sum = self.max_cell_sum_for_field_shape()
        self.cell_shape_limited = cell_sum > self.max_linked_cell_sum
        if self.cell_shape_limited:
            domain_volume = max(float(self.sims.domain[0]) * float(self.sims.domain[1]) * float(self.sims.domain[2]), self.contact_grid_size ** 3)
            self.grid_size = max(self.contact_grid_size, (domain_volume / self.max_linked_cell_sum) ** (1. / 3.))
            self.plane_insert_factor = 0.5 + rad_max / self.grid_size
            cnum, cell_sum = self.calculate_cell_grid()
            while cell_sum > self.max_linked_cell_sum:
                self.grid_size *= 1.001
                self.plane_insert_factor = 0.5 + rad_max / self.grid_size
                cnum, cell_sum = self.calculate_cell_grid()
        if cell_sum > self.max_linked_cell_sum:
            raise RuntimeError(
                "LinkedCell grid is too large for DEM neighbor search: "
                f"grid_number={cnum}, cell_number={cell_sum}. "
                f"The Taichi field shape limit is {MAX_TAICHI_FIELD_SHAPE}."
            )
        self.cnum = vec3i(cnum)
        self.cellSum = cell_sum
        self.minimum_grid_size = self.grid_size
        self.allocated_cell_sum = self.cellSum
        self.wall_in_cell = self.sims.wall_per_cell * self.cellSum
        self.check_memory()
        
        if self.first_run:
            self.cell_pse = PrefixSumExecutor(self.cellSum + 1)
            self.particle_pse = PrefixSumExecutor(self.sims.max_particle_num + 1)
            if self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
                self.point_pse = PrefixSumExecutor(self.sims.max_ls_contact_node_num + 1)
            self.set_potential_contact_list(scene)
            self.set_hash_table()
        self.print_info()
        self.first_run = False

    def refresh_contact_grid(self, scene: myScene):
        if not self.adaptive_cell_size:
            return False
        rad_min, rad_max = scene.find_bounding_sphere_radius(self.sims)
        self.sims.set_max_bounding_sphere_radius(rad_max)
        self.sims.set_min_bounding_sphere_radius(rad_min)
        if rad_max <= Threshold:
            return False

        contact_grid_size = 2.0 * (rad_max + self.sims.verlet_distance)
        new_grid_size = max(self.minimum_grid_size, contact_grid_size)
        new_plane_insert_factor = 0.5 + rad_max / new_grid_size
        grid_changed = not math.isclose(
            new_grid_size, self.grid_size,
            rel_tol=1.0e-12, abs_tol=1.0e-15)
        plane_band_changed = (
            getattr(self.sims, "wall_type", None) == 0
            and not math.isclose(
                new_plane_insert_factor, self.plane_insert_factor,
                rel_tol=1.0e-12, abs_tol=1.0e-15)
        )
        if not grid_changed:
            self.contact_grid_size = contact_grid_size
            self.plane_insert_factor = new_plane_insert_factor
            return plane_band_changed

        old_grid_size = self.grid_size
        self.contact_grid_size = contact_grid_size
        self.grid_size = new_grid_size
        cnum, cell_sum = self.calculate_cell_grid()
        if cell_sum > self.allocated_cell_sum:
            self.grid_size = old_grid_size
            self.calculate_cell_grid()
            raise RuntimeError(
                "Adaptive LinkedCell grid exceeds its initialized capacity: "
                f"required={cell_sum}, allocated={self.allocated_cell_sum}."
            )
        self.cnum = vec3i(cnum)
        self.cellSum = cell_sum
        self.plane_insert_factor = new_plane_insert_factor
        self.adaptive_regrid_count += 1
        return True

    def print_info(self):
        print(" Neighbor Search Initialize ".center(71,"-"))
        print("Neighbor search method:  Linked-cell")
        print("Verlet distance: ", self.sims.verlet_distance)
        print("Grid size: ", self.grid_size)
        if self.cell_shape_limited:
            print("Contact grid size: ", self.contact_grid_size)
            print("Linked-cell shape cell limit: ", self.max_linked_cell_sum)
        print("Grid number: ", self.cnum)
        print("Potental contact number per particle: ", self.sims.potential_particle_num)
        print("Potental contact wall per particle: ", self.sims.wall_coordination_number, '\n')

    def set_hash_table(self):
        self.particle_hash_tables()
        self.wall_hash_tables()

    def particle_hash_table(self):
        self.particle_current = ti.field(int, shape=self.sims.max_particle_num)
        self.particle_count = ti.field(int, shape=self.cell_pse.get_length())
        self.ParticleID = ti.field(int, shape=self.sims.max_particle_num)

    def plane_hash_table(self):
        self.wall_count = ti.field(int, shape=self.cell_pse.get_length())
        self.WallID = ti.field(int, shape=self.sims.wall_per_cell * self.cellSum)
    
    def facet_hash_table(self):
        self.wall_count = ti.field(int, shape=self.cell_pse.get_length())
        if self.sims.wall_per_cell == "sparse_grid":
            self.wallID = ti.field(int)
            ti.root.pointer(ti.i, math.ceil(self.sims.wall_per_cell / 128)).pointer(ti.i, 128 // 4).dense(self.wallID)
        else:
            self.WallID = ti.field(int, shape=self.sims.wall_per_cell * self.cellSum)

    def static_facet_hash_table(self):
        self.static_facet_current = ti.field(int, shape=self.cellSum)
        self.wall_count = ti.field(int, shape=self.cell_pse.get_length())

    def patch_hash_table(self):
        self.patch_current = ti.field(int, shape=self.sims.max_wall_num)
        self.wall_count = ti.field(int, shape=self.cell_pse.get_length())
        self.WallID = ti.field(int, shape=self.sims.max_wall_num)

    def particle_plane_delete_vars(self):
        del self.potential_list_particle_wall, self.particle_wall, self.hist_particle_wall
        del self.wall_count, self.WallID

    def particle_facet_delete_vars(self):
        del self.potential_list_particle_wall, self.particle_wall, self.hist_particle_wall
        del self.wall_count, self.WallID

    def particle_patch_delete_vars(self):
        del self.potential_list_particle_wall, self.particle_wall, self.hist_particle_wall
        del self.patch_current, self.wall_count, self.WallID
    
    def pre_neighbor(self, scene: myScene):
        if self.sims.static_wall is True:
            self.manage_wall_function()
            # A facet wall may be declared static after the neighbor object
            # was constructed. In that case the first allocation pass used
            # the no-wall hooks; allocate the fixed wall hash explicitly
            # before its one-time build.
            if self.wall_count is None or self.WallID is None:
                self.wall_hash_tables()
            
        self.place_particle_to_cells(scene)
        self.update_verlet_tables_particle_particle(scene)
        self.place_wall_to_cells(scene)
        self.update_verlet_tables_particle_wall(scene)

        if self.sims.static_wall is True:
            self.place_wall_to_cells = no_operation

    def update_verlet_table(self, scene):
        search_parameters_changed = self.refresh_contact_grid(scene)
        if search_parameters_changed and self.sims.static_wall is True:
            self.manage_wall_function()
        self.place_particle_to_cells(scene)
        self.update_verlet_tables_particle_particle(scene)
        self.place_wall_to_cells(scene)
        self.update_verlet_tables_particle_wall(scene)
        if search_parameters_changed and self.sims.static_wall is True:
            self.place_wall_to_cells = no_operation
        self.reset_verletDisp()

    def place_particle_to_cell(self, scene: myScene):
        calculate_particles_position_(int(scene.particleNum[0]), self.igrid_size, scene.particle, self.particle_count, self.particle_current, self.cnum)
        self.cell_pse.run(self.particle_count)
        insert_particle_to_cell_(self.igrid_size, int(scene.particleNum[0]), scene.particle, self.particle_count, self.particle_current, self.ParticleID, self.cnum)

    def place_plane_to_cell(self, scene: myScene):
        insert_plane_to_cell_(self.plane_insert_factor, int(scene.wallNum[0]), self.grid_size, self.cellSum, self.sims.wall_per_cell, self.wall_count, self.WallID, scene.wall, self.cnum)

    def place_facet_to_cell(self, scene: myScene):
        insert_facet_to_cell_(int(scene.wallNum[0]), self.grid_size, self.igrid_size, self.sims.domain, self.sims.wall_per_cell, self.wall_count, self.WallID, scene.wall, self.cnum)
        maximum = int(self.wall_count.to_numpy().max())
        if maximum > self.sims.wall_per_cell:
            raise RuntimeError(
                "LinkedCell facet capacity is insufficient: "
                f"required wall_per_cell={maximum}, "
                f"configured={self.sims.wall_per_cell}."
            )

    def place_static_facet_to_cell(self, scene: myScene):
        calculate_static_facet_position_(int(scene.wallNum[0]), self.igrid_size, scene.wall, self.wall_count, self.static_facet_current, self.cnum)
        self.cell_pse.run(self.wall_count)
        total_length = get_total_static_wall_length(self.cellSum, self.wall_count)
        self.WallID = ti.field(int, shape=total_length)
        insert_static_facet_to_cell_(self.igrid_size, int(scene.wallNum[0]), scene.wall, self.wall_count, self.static_facet_current, self.WallID, self.cnum)

    def place_patch_to_cell(self, scene: myScene):
        calculate_patch_position_(int(scene.wallNum[0]), self.igrid_size, scene.wall, self.wall_count, self.patch_current, self.cnum)
        self.cell_pse.run(self.wall_count)
        insert_patch_to_cell_(self.igrid_size, int(scene.wallNum[0]), scene.wall, self.wall_count, self.patch_current, self.WallID, self.cnum)

    def update_verlet_table_particle_particle(self, scene: myScene):
        board_search_particle_particle_linked_cell_(int(scene.particleNum[0]), self.sims.potential_particle_num, self.sims.verlet_distance, self.igrid_size,
                                                    self.particle_count, self.ParticleID, scene.particle, self.potential_list_particle_particle, 
                                                    self.particle_particle, self.cnum)
        self.particle_pse.run(self.particle_particle)

    def update_verlet_table_particle_plane(self, scene: myScene):
        board_search_particle_plane_linked_cell_(int(scene.particleNum[0]), self.sims.wall_coordination_number, self.sims.wall_per_cell, self.sims.verlet_distance, self.igrid_size, self.wall_count,
                                                 self.WallID, scene.particle, scene.wall, self.potential_list_particle_wall, self.particle_wall, self.cnum)
        self.particle_pse.run(self.particle_wall)

    def update_verlet_table_particle_facet(self, scene: myScene):
        board_search_particle_facet_linked_cell_(int(scene.particleNum[0]), self.sims.wall_coordination_number, self.sims.wall_per_cell, self.sims.verlet_distance, self.igrid_size, self.wall_count,
                                                 self.WallID, scene.particle, scene.wall, self.potential_list_particle_wall, self.particle_wall, self.cnum)
        self.particle_pse.run(self.particle_wall)

    def update_verlet_table_particle_static_facet(self, scene: myScene):
        board_search_particle_static_facet_linked_cell_(int(scene.particleNum[0]), self.sims.wall_coordination_number, self.sims.verlet_distance, self.igrid_size, 
                                                        self.wall_count, self.WallID, scene.particle, scene.wall, self.potential_list_particle_wall, 
                                                        self.particle_wall, self.cnum)
        self.particle_pse.run(self.particle_wall)

    def update_verlet_table_particle_patch(self, scene: myScene):
        board_search_particle_patch_linked_cell_(int(scene.particleNum[0]), self.sims.wall_coordination_number, self.sims.verlet_distance, self.igrid_size, 
                                                 self.wall_count, self.WallID, scene.particle, scene.wall, self.potential_list_particle_wall, 
                                                 self.particle_wall, self.cnum)
        self.particle_pse.run(self.particle_wall)
