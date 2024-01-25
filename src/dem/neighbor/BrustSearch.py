from src.dem.neighbor.neighbor_kernel import *
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.SceneManager import myScene


class BrustSearch(NeighborBase):
    def __init__(self, sims, scene) -> None:
        super().__init__(sims, scene)

    def manage_function(self, scene):
        self.manage_function_base()

    def pre_calculation(self, scene):
        pass

    def resize_neighbor(self):
        pass

    def neighbor_initialze(self, scene: myScene):
        rad_min, rad_max = scene.find_bounding_sphere_radius()
        self.sims.set_verlet_distance(rad_min)
        self.sims.set_potential_list_size(rad_max)
        self.set_potential_contact_list()

    def update_verlet_table_particle_plane(self, scene: myScene):
        board_search_particle_wall_brust_(scene.planeNum, self.verlet_distance, scene.particle, scene.plane, self.potential_list_particle_wall, self.particle_wall)

    def update_verlet_table_particle_facet(self, scene: myScene):
        board_search_particle_wall_brust_(scene.facetNum, self.verlet_distance, scene.particle, scene.facet, self.potential_list_particle_wall, self.particle_wall)

    def update_verlet_table_particle_patch(self, scene: myScene):
        board_search_particle_wall_brust_(scene.patchNum, self.verlet_distance, scene.particle, scene.patch, self.potential_list_particle_wall, self.particle_wall)