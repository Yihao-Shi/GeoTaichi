import taichi as ti
import numpy as np

from src.dem.structs.BaseStruct import HierarchicalCell, HierarchicalBody
from src.dem.neighbor.neighbor_kernel import *
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.SceneManager import myScene
from src.dem.Simulation import Simulation
from src.utils.constants import Threshold
from src.utils.PrefixSum import PrefixSumExecutor, serial
from src.utils.TypeDefination import vec3i
from src.utils.linalg import no_operation


class HierarchicalLinkedCell(NeighborBase):
    def __init__(self, sims: Simulation, scene):
        super().__init__(sims, scene)
        self.wall_in_cell = 0
        self.grid = None
        self.body = None
        self.particle_count = None
        self.particle_current = None
        self.ParticleID = None
        self.wall_count = None
        self.WallID = None 

    def manage_function(self, scene):
        self.manage_function_base(scene)
        self.manage_particle_function()
        self.manage_wall_function()

        if self.sims.max_particle_num > 1:
            if self.sims.search_direction == "Up":
                self.update_verlet_tables_particle_particle = self.update_verlet_table_particle_particle2
            elif self.sims.search_direction == "Down":
                self.update_verlet_tables_particle_particle = self.update_verlet_table_particle_particle

    def manage_particle_function(self):
        self.place_particle_to_cells = no_operation
        self.particle_hash_tables = no_operation
        if self.sims.max_particle_num > 1 or self.sims.coupling:
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
                self.place_wall_to_cells = self.place_facet_to_cell
                self.wall_hash_tables = self.facet_hash_table
            elif self.sims.wall_type == 2:
                self.place_wall_to_cells = self.place_patch_to_cell
                self.wall_hash_tables = self.patch_hash_table

    def resize_neighbor(self, scene):
        del self.cell_pse, self.cellSum
        del self.particle_current, self.particle_count
        self.particle_wall_delete_vars()
        self.neighbor_initialze(scene, max(self.sims.domain), 0.)

    def neighbor_initialze(self, scene: myScene, min_bounding_rad, max_bounding_rad):
        if self.first_run:
            if self.sims.scheme == "DEM":
                self.body = HierarchicalBody.field(shape=self.sims.max_particle_num + 1)
            elif self.sims.scheme == "LSDEM":
                self.body = HierarchicalBody.field(shape=self.sims.max_rigid_body_num + self.sims.max_soft_body_num + 1)
            self.grid = HierarchicalCell.field(shape=self.sims.hierarchical_level)
        self.grid.rad_min.fill(min_bounding_rad)

        particle_num_in_level = np.zeros(self.sims.hierarchical_level)
        initialize_radius_range(int(scene.particleNum[0]), self.sims.hierarchical_level, np.array(self.sims.hierarchical_size), particle_num_in_level, self.body, self.grid, scene.particle)
        rad_min, rad_max = self.grid.rad_min.to_numpy(), self.grid.rad_max.to_numpy()
        self.sims.set_max_bounding_sphere_radius(rad_max)
        self.sims.set_min_bounding_sphere_radius(rad_min)
        self.sims.set_verlet_distance(min(min_bounding_rad, rad_min[0]))
        self.sims.update_hierarchical_size(max(max_bounding_rad, rad_max[-1]))
        potential_particle_ratio = self.sims.compute_potential_ratios(rad_max)
        
        if self.sims.xpbc:
            pass
        if self.sims.ypbc:
            pass
        if self.sims.zpbc:
            pass

        initialize_body_information(int(scene.particleNum[0]), np.array(potential_particle_ratio), np.array(self.sims.body_coordination_number), np.array(self.sims.wall_coordination_number), self.body)
        serial(self.body.max_potential_particle_pairs)
        serial(self.body.max_potential_wall_pairs)

        gsize, cnum, csum, factor = self.calculate_total_cell_number()
        self.cellSum = sum(csum)
        self.wall_in_cell = initialize_grid_information(self.sims.hierarchical_level, np.array(gsize), np.array(cnum), np.array(csum), np.array(factor), np.array(self.sims.wall_per_cell), self.grid)
        if self.sims.wall_type == "Patch":
            initialize_wall_information(int(scene.wallNum[0]), self.sims.hierarchical_level, np.array(self.sims.hierarchical_size), scene.wallbody, scene.wall)
        
        if self.first_run:
            self.cell_pse = PrefixSumExecutor(self.cellSum + 1)
            self.particle_pse = PrefixSumExecutor(self.sims.max_particle_num + 1)
            if self.sims.scheme == "LSDEM":
                self.point_pse = PrefixSumExecutor(self.sims.max_surface_node_num * self.sims.max_particle_num + 1)
            pairs_num = get_potential_contact_pairs_num(int(scene.particleNum[0]), self.body)
            self.sims.set_hierarchical_list_size(pairs_num[0], pairs_num[1])
            self.set_potential_contact_list(scene)
            self.set_hash_table()
        self.print_info(potential_particle_ratio)
        self.first_run = False

    def print_info(self, potential_particle_ratio):
        print(" Neighbor Search Initialize ".center(71,"-"))
        print("Neighbor search method:  Hierarchical Linked-cell")
        print("Verlet distance: ", self.sims.verlet_distance)
        print("Grid size: ", self.sims.hierarchical_size)
        print("Potental contact number per particle: ", np.ceil(potential_particle_ratio * np.array(self.sims.body_coordination_number)))
        print("Potental contact wall per particle: ", self.sims.wall_coordination_number, '\n')

    def calculate_total_cell_number(self):
        gsize, cnum, csum, factor = [], [], [], []
        for i in range(self.sims.hierarchical_level):
            grid_size = 2 * (self.sims.hierarchical_size[i] + self.sims.verlet_distance)
            plane_insert_factor = 0.5 + self.sims.hierarchical_size[i] / grid_size
            if grid_size < self.sims.hierarchical_size[i] * Threshold:
                raise RuntimeError("Particle radius is equal to zero!")
            igrid_size = 1. / grid_size
            cell_num = vec3i([int(domain * igrid_size) + 1 for domain in self.sims.domain])
            cellSum = int(cell_num[0] * cell_num[1] * cell_num[2])
            gsize.append(grid_size)
            cnum.append(cell_num)
            csum.append(cellSum)
            factor.append(plane_insert_factor)
        return gsize, cnum, csum, factor

    def set_hash_table(self):
        self.particle_hash_tables()
        self.wall_hash_tables()

    def particle_hash_table(self):
        self.particle_current = ti.field(int, shape=self.sims.max_particle_num)
        self.particle_count = ti.field(int, shape=self.cell_pse.get_length())
        self.ParticleID = ti.field(int, shape=self.sims.max_particle_num)

    def plane_hash_table(self):
        self.wall_count = ti.field(int, shape=self.cell_pse.get_length())
        self.WallID = ti.field(int, shape=self.wall_in_cell)
    
    def facet_hash_table(self):
        self.wall_count = ti.field(int, shape=self.cell_pse.get_length())
        self.WallID = ti.field(int, shape=self.wall_in_cell)

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
            
        self.place_particle_to_cells(scene)
        self.update_verlet_tables_particle_particle(scene)
        self.place_wall_to_cells(scene)
        self.update_verlet_tables_particle_wall(scene)

        if self.sims.static_wall is True:
            self.place_wall_to_cells = no_operation

    def update_verlet_table(self, scene):
        self.place_particle_to_cells(scene)
        self.update_verlet_tables_particle_particle(scene)
        self.place_wall_to_cells(scene)
        self.update_verlet_tables_particle_wall(scene)
        self.reset_verletDisp()

    def place_particle_to_cell(self, scene: myScene):
        calculate_particles_position_hierarchical_(int(scene.particleNum[0]), scene.particle, self.particle_count, self.particle_current, self.body, self.grid)
        self.cell_pse.run(self.particle_count)
        insert_particle_to_cell_hierarchical_(int(scene.particleNum[0]), scene.particle, self.particle_count, self.particle_current, self.ParticleID, self.body, self.grid)

    def place_plane_to_cell(self, scene: myScene):
        insert_plane_to_cell_hierarchical_(int(scene.wallNum[0]), self.cellSum, self.sims.hierarchical_level, self.wall_count, self.WallID, scene.wall, self.grid)

    def place_facet_to_cell(self, scene: myScene):
        insert_facet_to_cell_hierarchical_(int(scene.wallNum[0]), self.sims.hierarchical_level, self.wall_count, self.WallID, scene.wall, self.grid)

    def place_patch_to_cell(self, scene: myScene):
        calculate_patch_position_hierarchical_(int(scene.wallNum[0]), scene.wall, self.wall_count, self.patch_current, self.grid, scene.wallbody)
        self.cell_pse.run(self.wall_count)
        insert_patch_to_cell_hierarchical_(int(scene.wallNum[0]), scene.wall, self.wall_count, self.patch_current, self.WallID, self.grid, scene.wallbody)

    def update_verlet_table_particle_particle(self, scene: myScene):
        board_search_particle_particle_linked_cell_hierarchical_(int(scene.particleNum[0]), self.sims.verlet_distance, self.particle_count, self.ParticleID, 
                                                                 scene.particle, self.potential_list_particle_particle, self.particle_particle, self.body, self.grid)
        self.particle_pse.run(self.particle_particle)

    def update_verlet_table_particle_particle2(self, scene: myScene):
        board_search_particle_particle_linked_cell_hierarchical2_(int(scene.particleNum[0]), self.sims.hierarchical_level, self.sims.verlet_distance, self.particle_count, self.ParticleID, 
                                                                  scene.particle, self.potential_list_particle_particle, self.particle_particle, self.body, self.grid)
        self.particle_pse.run(self.particle_particle)

    def update_verlet_table_particle_plane(self, scene: myScene):
        board_search_particle_plane_linked_cell_hierarchical_(int(scene.particleNum[0]), self.sims.verlet_distance, self.wall_count, self.WallID, scene.particle, 
                                                              scene.wall, self.potential_list_particle_wall, self.particle_wall, self.body, self.grid)
        self.particle_pse.run(self.particle_wall)

    def update_verlet_table_particle_facet(self, scene: myScene):
        board_search_particle_facet_linked_cell_hierarchical_(int(scene.particleNum[0]), self.sims.verlet_distance, self.wall_count, self.WallID, 
                                                              scene.particle, scene.wall, self.potential_list_particle_wall, self.particle_wall, self.body, self.grid)
        self.particle_pse.run(self.particle_wall)

    def update_verlet_table_particle_patch(self, scene: myScene):
        board_search_particle_patch_linked_cell_hierarchical_(int(scene.particleNum[0]), self.sims.verlet_distance, self.wall_count, self.WallID, 
                                                              scene.particle, scene.wall, self.potential_list_particle_wall, self.particle_wall, self.body, self.grid)
        self.particle_pse.run(self.particle_wall)

    def update_particle_verlet_table(self, scene: myScene):
        self.particle_particle.fill(0)
        self.particle_wall.fill(0)
        for nlevel in range(self.sims.hierarchical_level):
            calculate_particles_position_hierarchical_(int(scene.particleNum[0]), scene.particle, self.particle_count, self.body, self.grid)
            self.cell_pse.run(self.particle_count, nlevel)
            insert_particle_to_cell_hierarchical_(int(scene.particleNum[0]), scene.particle, self.particle_count, self.particle_current, self.ParticleID, self.body, self.grid)
            board_search_particle_particle_linked_cell_hierarchical_interlevel_(int(scene.particleNum[0]), self.sims.verlet_distance, self.particle_count, self.ParticleID, 
                                                                                scene.particle, self.potential_list_particle_particle, self.particle_particle, self.body, self.grid)
            board_search_particle_particle_linked_cell_hierarchical_crosslevel_(int(scene.particleNum[0]), self.sims.verlet_distance, self.particle_count, self.ParticleID, 
                                                                                scene.particle, self.potential_list_particle_particle, self.particle_particle, self.body, self.grid)
            self.particle_pse.run(self.particle_particle)

