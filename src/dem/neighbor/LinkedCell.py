import taichi as ti

from src.dem.neighbor.neighbor_kernel import *
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.SceneManager import myScene
from src.dem.Simulation import Simulation
from src.utils.constants import Threshold
from src.utils.PrefixSum import PrefixSumExecutor
from src.utils.TypeDefination import vec3i


class LinkedCell(NeighborBase):
    def __init__(self, sims: Simulation, scene):
        super().__init__(sims, scene)
        self.wall_in_cell = sims.wall_per_cell
        self.cnum = vec3i([0, 0, 0])
        self.particle_count = None
        self.particle_current = None
        self.ParticleID = None
        self.wall_count = None
        self.WallID = None
        
    def manage_function(self, scene: myScene):
        self.manage_function_base()
        
        self.place_particle_to_cells = self.no_operation
        self.particle_hash_tables = self.no_operation_without_paras
        if self.sims.max_particle_num > 1 or self.sims.coupling:
            self.place_particle_to_cells = self.place_particle_to_cell
            self.particle_hash_tables = self.particle_hash_table

        self.wall_hash_tables = self.no_operation_without_paras
        self.place_wall_to_cells = self.no_operation
        if self.sims.wall_type == 0:
            self.place_wall_to_cells = self.place_plane_to_cell
            self.wall_hash_tables = self.plane_hash_table
        elif self.sims.wall_type == 1:
            self.place_wall_to_cells = self.place_facet_to_cell
            self.wall_hash_tables = self.facet_hash_table
        elif self.sims.wall_type == 2:
            self.place_wall_to_cells = self.place_patch_to_cell
            self.wall_hash_tables = self.patch_hash_table

        self.reset_verletDisp = self.no_operation_without_paras
        if self.sims.max_particle_num > 0 and (self.sims.max_wall_num == 0 or self.sims.wall_type == 0):
            self.reset_verletDisp = scene.reset_particle_verlet_disp
        elif self.sims.max_particle_num == 0 and self.sims.max_wall_num > 0 and self.sims.wall_type != 0:
            self.reset_verletDisp = scene.reset_wall_verlet_disp
        elif self.sims.max_particle_num > 0 and self.sims.max_wall_num > 0 and self.sims.wall_type != 0:
            self.reset_verletDisp = scene.reset_verlet_disp

    def resize_neighbor(self, scene):
        del self.cell_pse, self.igrid_size, self.cnum, self.cellSum
        del self.particle_current, self.particle_count
        self.particle_wall_delete_vars()
        self.neighbor_initialze(scene)

    def neighbor_initialze(self, scene: myScene, max_bounding_rad=0.):
        rad_min, rad_max = scene.find_bounding_sphere_radius(self.sims)
        rad_max = max(max_bounding_rad, rad_max)
        self.sims.set_verlet_distance(rad_min)
        self.grid_size = 2 * (rad_max + self.sims.verlet_distance)
        self.plane_insert_factor = 0.5 + rad_max / self.grid_size
        if self.grid_size < 1e-3 * Threshold:
            raise RuntimeError("Particle radius is equal to zero!")
        self.sims.set_potential_list_size(rad_max)
        self.set_potential_contact_list()
        self.igrid_size = 1. / self.grid_size

        if self.sims.pbc:
            pass

        self.cnum = vec3i([int(domain * self.igrid_size) + 1 for domain in self.sims.domain])
        for d in ti.static(range(3)):
            if self.cnum[d] == 0:
                self.cnum[d] = int(1)
        
        self.cellSum = int(self.cnum[0] * self.cnum[1] * self.cnum[2])
        self.cell_pse = PrefixSumExecutor(self.cellSum)
        self.particle_pse = PrefixSumExecutor(self.sims.max_particle_num + 1)
        self.set_hash_table()

    def set_hash_table(self):
        self.particle_hash_tables()
        self.wall_hash_tables()

    def particle_hash_table(self):
        self.particle_current = ti.field(int)
        self.particle_count = ti.field(int)
        ti.root.dense(ti.i, self.cellSum).place(self.particle_current, self.particle_count)
        self.ParticleID = ti.field(int, shape=self.sims.max_particle_num)

    def plane_hash_table(self):
        self.wall_count = ti.field(int, shape=self.cellSum)
        self.WallID = ti.field(int, shape=self.wall_in_cell * self.cellSum)
    
    def facet_hash_table(self):
        self.wall_count = ti.field(int, shape=self.cellSum)
        self.WallID = ti.field(int, shape=self.wall_in_cell * self.cellSum)

    def patch_hash_table(self):
        self.patch_current = ti.field(int)
        self.wall_count = ti.field(int)
        ti.root.dense(ti.i, self.cellSum).place(self.patch_current, self.wall_count)
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
        if self.sims.wall_type == 0:
            self.place_wall_to_cells = self.place_plane_to_cell
            
        self.place_particle_to_cells(scene)
        self.update_verlet_tables_particle_particle(scene)
        self.place_wall_to_cells(scene)
        self.update_verlet_tables_particle_wall(scene)

        if self.sims.wall_type == 0:
            self.place_wall_to_cells = self.no_operation

    def update_verlet_table(self, scene: myScene):
        self.place_particle_to_cells(scene)
        self.update_verlet_tables_particle_particle(scene)
        self.place_wall_to_cells(scene)
        self.update_verlet_tables_particle_wall(scene)
        self.reset_verletDisp()

    def place_particle_to_cell(self, scene: myScene):
        calculate_particles_position_(int(scene.particleNum[0]), self.igrid_size, scene.particle, self.particle_count, self.cnum)
        self.cell_pse.run(self.particle_count, self.cellSum)
        insert_particle_to_cell_(self.igrid_size, int(scene.particleNum[0]), scene.particle, self.particle_count, self.particle_current, self.ParticleID, self.cnum)

    def place_plane_to_cell(self, scene: myScene):
        insert_plane_to_cell_(self.plane_insert_factor, int(scene.wallNum[0]), self.grid_size, self.cellSum, self.wall_in_cell, self.wall_count, self.WallID, scene.wall, self.cnum)

    def place_facet_to_cell(self, scene: myScene):
        insert_facet_to_cell_(int(scene.wallNum[0]), self.igrid_size, self.wall_in_cell, self.wall_count, self.WallID, scene.wall, self.cnum)

    def place_patch_to_cell(self, scene: myScene):
        calculate_patch_position_(int(scene.wallNum[0]), self.igrid_size, scene.wall, self.wall_count, self.cnum)
        self.cell_pse.run(self.wall_count, self.cellSum)
        insert_patch_to_cell_(self.igrid_size, int(scene.wallNum[0]), scene.wall, self.wall_count, self.patch_current, self.WallID, self.cnum)

    def update_verlet_table_particle_particle(self, scene: myScene):
        board_search_particle_particle_linked_cell_(int(scene.particleNum[0]), self.sims.potential_particle_num, self.sims.verlet_distance, self.igrid_size,
                                                    self.particle_count, self.particle_current, self.ParticleID, scene.particle, self.potential_list_particle_particle, 
                                                    self.particle_particle, self.cnum)
        self.particle_pse.run(self.particle_particle, int(scene.particleNum[0]) + 1)

    def update_verlet_table_particle_plane(self, scene: myScene):
        board_search_particle_plane_linked_cell_(int(scene.particleNum[0]), self.sims.wall_coordination_number, self.wall_in_cell, self.sims.verlet_distance, self.igrid_size, self.wall_count,
                                                 self.WallID, scene.particle, scene.wall, self.potential_list_particle_wall, self.particle_wall, self.cnum)
        self.particle_pse.run(self.particle_wall, int(scene.particleNum[0]) + 1)

    def update_verlet_table_particle_facet(self, scene: myScene):
        board_search_particle_facet_linked_cell_(int(scene.particleNum[0]), self.sims.wall_coordination_number, self.wall_in_cell, self.sims.verlet_distance, self.igrid_size, self.wall_count,
                                                 self.WallID, scene.particle, scene.wall, self.potential_list_particle_wall, self.particle_wall, self.cnum)
        self.particle_pse.run(self.particle_wall, int(scene.particleNum[0]) + 1)

    def update_verlet_table_particle_patch(self, scene: myScene):
        board_search_particle_patch_linked_cell_(int(scene.particleNum[0]), self.sims.wall_coordination_number, self.sims.verlet_distance, self.igrid_size, 
                                                 self.wall_count, self.patch_current, self.WallID, scene.particle, scene.wall, self.potential_list_particle_wall, 
                                                 self.particle_wall, self.cnum)
        self.particle_pse.run(self.particle_wall, int(scene.particleNum[0]) + 1)
