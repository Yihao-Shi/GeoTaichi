from src.dem.neighbor.neighbor_kernel import *
from src.contact_detection.bounding_volume_hierarchy.MultiPatchLBVH import LBVH
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.SceneManager import myScene
from src.utils.PrefixSum import PrefixSumExecutor
from src.utils.linalg import no_operation


class BoundingVolumeHierarchy(NeighborBase):
    def __init__(self, sims, scene) -> None:
        super().__init__(sims, scene)
        self.lbvh = LBVH(n_aabbs=[sims.max_particle_num, sims.max_wall_num])
        self.first_run = True

    def manage_function(self, scene):
        self.manage_function_base(scene)
        self.manage_wall_function()

    def manage_wall_function(self):
        self.place_wall_to_cells = no_operation
        if not self.sims.wall_type is None:
            if self.sims.wall_type == 1 or self.sims.wall_type == 2:
                self.place_wall_to_cells = self.place_wall_to_cell
            if self.sims.max_particle_num > 0:
                if self.sims.wall_type == 0:
                    self.update_verlet_tables_particle_wall = self.update_verlet_table_particle_plane
                elif self.sims.wall_type == 1 or self.sims.wall_type == 2:
                    self.update_verlet_tables_particle_wall = self.update_verlet_table_particle_triangle_wall

    def resize_neighbor(self, scene):
        self.particle_wall_delete_vars()
        self.neighbor_initialze(scene, max(self.sims.domain), 0.)

    def neighbor_initialze(self, scene: myScene, min_bounding_rad, max_bounding_rad):
        rad_min, rad_max = scene.find_bounding_sphere_radius(self.sims)
        self.sims.set_max_bounding_sphere_radius(rad_max)
        self.sims.set_min_bounding_sphere_radius(rad_min)
        rad_max = max(max_bounding_rad, rad_max) if abs(rad_max) > 1e-15 else max_bounding_rad
        rad_min = min(min_bounding_rad, rad_min) if abs(rad_min) > 1e-15 else min_bounding_rad
        self.sims.set_verlet_distance(rad_min)
        self.sims.set_potential_list_size(rad_max)
            
        if self.sims.xpbc:
            pass
        if self.sims.ypbc:
            pass
        if self.sims.zpbc:
            pass

        if self.first_run:
            self.particle_pse = PrefixSumExecutor(self.sims.max_particle_num + 1)
            self.set_potential_contact_list(scene)
        self.print_info()
        self.first_run = False

    def print_info(self):
        print(" Neighbor Search Initialize ".center(71,"-"))
        print("Neighbor search method:  Bounding Volume Hierarchy (LBVH)")
        print("Verlet distance: ", self.sims.verlet_distance)
        print("Potental contact number per particle: ", self.sims.potential_particle_num)
        print("Potental contact wall per particle: ", self.sims.wall_coordination_number, '\n')
    
    def pre_neighbor(self, scene: myScene):
        self.lbvh.initialize(active_aabbs=(scene.particleNum[0], scene.wallNum[0]))
        if self.sims.static_wall is True:
            self.manage_wall_function()
            
        self.place_particle_to_cell(scene)
        self.place_wall_to_cells(scene)
        self.lbvh.build()
        self.update_verlet_tables_particle_particle(scene)
        self.update_verlet_tables_particle_wall(scene)

        if self.sims.static_wall is True:
            self.lbvh.aabb.n_batches = 1
            self.place_wall_to_cells = no_operation

    def update_verlet_table(self, scene):
        self.place_particle_to_cell(scene)
        self.place_wall_to_cells(scene)
        if self.sims.current_step % 1000 == 0:
            self.lbvh.build()
        else:
            self.lbvh.refit()
        self.update_verlet_tables_particle_particle(scene)
        self.update_verlet_tables_particle_wall(scene)
        self.reset_verletDisp()

    def place_particle_to_cell(self, scene: myScene):
        self.lbvh.aabb.set_sphere_aabbs(int(scene.particleNum[0]), self.lbvh.aabb.prefix_batch_size[0], self.sims.verlet_distance, scene.particle)

    def place_wall_to_cell(self, scene: myScene):
        self.lbvh.aabb.set_triangle_aabbs(int(scene.wallNum[0]), self.lbvh.aabb.prefix_batch_size[1], self.sims.verlet_distance, scene.wall)

    def update_verlet_table_particle_particle(self, scene: myScene):
        board_search_particle_particle_bvh_(0, int(scene.particleNum[0]), self.sims.potential_particle_num, self.lbvh.prefix_batch_size, self.lbvh.nodes, self.lbvh.morton_codes, self.lbvh.aabb.aabbs, 
                                            scene.particle, self.potential_list_particle_particle, self.particle_particle)
        self.particle_pse.run(self.particle_particle)

    def update_verlet_table_particle_plane(self, scene: myScene):
        board_search_particle_wall_brust_(self.sims.wall_coordination_number, int(scene.particleNum[0]), int(scene.wallNum[0]), self.sims.verlet_distance, 
                                          scene.particle, scene.wall, self.potential_list_particle_wall, self.particle_wall)
        self.particle_pse.run(self.particle_wall)

    def update_verlet_table_particle_triangle_wall(self, scene: myScene):
        board_search_particle_wall_bvh_(0, 1, int(scene.particleNum[0]), self.sims.wall_coordination_number, self.lbvh.prefix_batch_size, self.lbvh.nodes, self.lbvh.morton_codes, self.lbvh.aabb.aabbs, 
                                                 scene.particle, self.potential_list_particle_wall, self.particle_wall)
        self.particle_pse.run(self.particle_wall)



