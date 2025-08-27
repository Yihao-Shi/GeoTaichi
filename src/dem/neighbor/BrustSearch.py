from src.dem.neighbor.neighbor_kernel import *
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.SceneManager import myScene
from src.utils.PrefixSum import PrefixSumExecutor


class BrustSearch(NeighborBase):
    def __init__(self, sims, scene) -> None:
        super().__init__(sims, scene)

    def manage_function(self, scene):
        self.manage_function_base(scene)

    def pre_calculation(self, scene):
        self.update_verlet_tables_particle_particle(scene)
        self.update_verlet_tables_particle_wall(scene)

    def update_verlet_table(self, scene):
        self.update_verlet_tables_particle_particle(scene)
        self.update_verlet_tables_particle_wall(scene)
        self.reset_verletDisp()

    def resize_neighbor(self, scene):
        self.neighbor_initialze(scene, max(self.sims.domain), 0.)

    def neighbor_initialze(self, scene: myScene, min_bounding_rad, max_bounding_rad):
        rad_min, rad_max = scene.find_bounding_sphere_radius(self.sims)
        self.sims.set_max_bounding_sphere_radius(rad_max)
        self.sims.set_min_bounding_sphere_radius(rad_min)
        rad_max = max(max_bounding_rad, rad_max)
        rad_min = min(min_bounding_rad, rad_min)
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
            if self.sims.scheme == "LSDEM":
                self.point_pse = PrefixSumExecutor(self.sims.max_surface_node_num * self.sims.max_particle_num + 1)
            
            self.set_potential_contact_list(scene)
        self.print_info()
        self.first_run = False

    def print_info(self):
        print(" Neighbor Search Initialize ".center(71,"-"))
        print("Neighbor search method:  Brust search")
        print("Verlet distance: ", self.sims.verlet_distance)
        print("Potental contact number per particle: ", self.sims.potential_particle_num)
        print("Potental contact wall per particle: ", self.sims.wall_coordination_number, '\n')

    def update_verlet_table_particle_particle(self, scene: myScene):
        board_search_particle_particle_brust_(self.sims.max_potential_particle_pairs, int(scene.particleNum[0]), self.sims.verlet_distance, scene.particle, self.potential_list_particle_particle, self.particle_particle)
        self.particle_pse.run(self.particle_particle)

    def update_verlet_table_particle_plane(self, scene: myScene):
        board_search_particle_wall_brust_(self.sims.max_potential_wall_pairs, int(scene.particleNum[0]), int(scene.wallNum[0]), self.sims.verlet_distance, scene.particle, scene.wall, self.potential_list_particle_wall, self.particle_wall)
        self.particle_pse.run(self.particle_wall)

    def update_verlet_table_particle_facet(self, scene: myScene):
        board_search_particle_wall_brust_(self.sims.max_potential_wall_pairs, int(scene.particleNum[0]), int(scene.wallNum[0]), self.sims.verlet_distance, scene.particle, scene.wall, self.potential_list_particle_wall, self.particle_wall)
        self.particle_pse.run(self.particle_wall)

    def update_verlet_table_particle_patch(self, scene: myScene):
        board_search_particle_wall_brust_(self.sims.max_potential_wall_pairs, int(scene.particleNum[0]), int(scene.wallNum[0]), self.sims.verlet_distance, scene.particle, scene.wall, self.potential_list_particle_wall, self.particle_wall)
        self.particle_pse.run(self.particle_wall)


