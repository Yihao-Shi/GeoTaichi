import taichi as ti

from src.dem.neighbor.neighbor_kernel import *
from src.dem.neighbor.LinkedCell import LinkedCell
from src.dem.SceneManager import myScene as DEMScene
from src.dem.Simulation import Simulation as DEMSimulation
from src.mpdem.Simulation import Simulation as DEMPMSimulation
from src.mpdem.contact.ContactKernel import *
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.mpm.SceneManager import myScene as MPMScene
from src.mpm.Simulation import Simulation as MPMSimulation
from src.utils.PrefixSum import PrefixSumExecutor


class MultiLinkedCell(object):
    csims: DEMPMSimulation
    msims: MPMSimulation
    dsims: DEMSimulation
    mpm_spatial_grid: SpatialHashGrid
    dem_spatial_grid: LinkedCell

    def __init__(self, csims, msims, dsims, mpm_neighbor, dem_neighbor) -> None:
        super().__init__()
        self.csims = csims
        self.msims = msims
        self.dsims = dsims
        self.mpm_spatial_grid = mpm_neighbor
        self.dem_spatial_grid = dem_neighbor
        self.plane_in_cell = 4
        self.facet_in_cell = 8
        self.manage_function(msims, dsims)

        self.potential_list_particle_particle = None
        self.particle_particle = None
        self.hist_particle_particle = None
        self.potential_list_particle_wall = None
        self.particle_wall = None
        self.hist_particle_wall = None

        self.particle_pse = None

    def no_operation(self, mscene, dscene):
        pass

    def no_operation_without_paras(self):
        pass

    def manage_function(self, msims: MPMSimulation, dsims: DEMSimulation):
        self.particle_particle_delete_vars = self.no_operation
        self.update_verlet_tables_particle_particle = self.no_operation
        self.update_particle_particle_auxiliary_lists = self.no_operation_without_paras
        if self.csims.particle_interaction:
            self.particle_particle_delete_vars = self.particle_particle_delete_var
            self.update_verlet_tables_particle_particle = self.update_verlet_table_particle_particle
            self.update_particle_particle_auxiliary_lists = self. update_particle_particle_auxiliary_list

        self.particle_wall_delete_vars = self.no_operation
        self.update_verlet_table_particle_wall = self.no_operation
        self.update_particle_wall_auxiliary_lists = self. no_operation_without_paras
        if self.csims.wall_interaction:
            if not dsims.wall_type is None:
                self.update_particle_wall_auxiliary_lists = self.update_particle_wall_auxiliary_list

            if dsims.wall_type == 0:
                self.particle_wall_delete_vars = self.particle_plane_delete_var
                self.update_verlet_table_particle_wall = self.update_verlet_table_particle_plane
            elif dsims.wall_type == 1:
                self.particle_wall_delete_vars = self.particle_facet_delete_var
                self.update_verlet_table_particle_wall = self.update_verlet_table_particle_facet
            elif dsims.wall_type == 2:
                self.particle_wall_delete_vars = self.particle_patch_delete_var
                self.update_verlet_table_particle_wall = self.update_verlet_table_particle_patch

    def particle_particle_delete_var(self):
        del self.potential_list_particle_particle, self.particle_particle, self.hist_particle_particle
    
    def particle_plane_delete_var(self):
        del self.potential_list_particle_wall, self.particle_wall, self.hist_particle_wall

    def particle_facet_delete_var(self):
        del self.potential_list_particle_wall, self.particle_wall, self.hist_particle_wall

    def particle_patch_delete_var(self):
        del self.potential_list_particle_wall, self.particle_wall, self.hist_particle_wall

    def set_potential_contact_list(self, msims: MPMSimulation, dsims: DEMSimulation, mscene: MPMScene, dscene: DEMScene):
        msims.set_verlet_distance(mscene.find_particle_min_radius())
        dsims.set_verlet_distance(dscene.find_min_radius(dsims))
        self.csims.set_potential_list_size(msims, dsims, dscene.find_max_radius(dsims), mscene.find_particle_max_radius())
        if self.csims.particle_interaction:
            self.potential_list_particle_particle = ti.field(int, shape=self.csims.max_potential_particle_pairs)
            self.particle_particle = ti.field(int)
            self.hist_particle_particle = ti.field(int)
            ti.root.dense(ti.i, msims.max_particle_num + 1).place(self.particle_particle, self.hist_particle_particle)
        
        if self.csims.wall_interaction and not dsims.wall_type is None:
            self.potential_list_particle_wall = ti.field(int, shape=self.csims.max_potential_wall_pairs)
            self.particle_wall = ti.field(int)
            self.hist_particle_wall = ti.field(int)
            ti.root.dense(ti.i, msims.max_particle_num + 1).place(self.particle_wall, self.hist_particle_wall)

        self.particle_pse = PrefixSumExecutor(msims.max_particle_num + 1)

    def particle_plane_delete_vars(self):
        del self.potential_list_particle_wall, self.particle_wall

    def particle_facet_delete_vars(self):
        del self.potential_list_particle_wall, self.particle_wall

    def particle_patch_delete_vars(self):
        del self.potential_list_particle_wall, self.particle_wall

    def update_particle_particle_auxiliary_list(self):
        initial_object_object(self.particle_particle, self.hist_particle_particle)

    def update_particle_wall_auxiliary_list(self):
        initial_object_object(self.particle_wall, self.hist_particle_wall)

    def pre_neighbor(self, mscene: MPMScene, dscene: DEMScene):
        self.update_verlet_tables_particle_particle(mscene, dscene)
        self.update_verlet_table_particle_wall(mscene, dscene)

    def update_verlet_table(self, mscene: MPMScene, dscene: DEMScene):
        self.update_verlet_tables_particle_particle(mscene, dscene)
        self.update_verlet_table_particle_wall(mscene, dscene)
        mscene.reset_verlet_disp()

    def update_verlet_table_particle_particle(self, mscene: MPMScene, dscene: DEMScene):
        board_search_coupled_particle_linked_cell_(self.csims.potential_particle_num, self.msims.verlet_distance, self.dsims.verlet_distance, self.dsims.max_particle_radius, self.dem_spatial_grid.igrid_size, self.dem_spatial_grid.particle_count, 
                                                   self.dem_spatial_grid.particle_current, self.dem_spatial_grid.ParticleID, mscene.particle, dscene.particle, self.potential_list_particle_particle, 
                                                   self.particle_particle, self.dem_spatial_grid.cnum, int(mscene.particleNum[0]))
        self.particle_pse.run(self.particle_particle, int(mscene.particleNum[0]) + 1)

    def update_verlet_table_particle_plane(self, mscene: MPMScene, dscene: DEMScene):
        board_search_coupled_particle_plane_linked_cell_(self.csims.wall_coordination_number, self.plane_in_cell, self.msims.verlet_distance, self.dem_spatial_grid.igrid_size, self.dem_spatial_grid.wall_count,
                                                         self.dem_spatial_grid.WallID, mscene.particle, dscene.wall, self.potential_list_particle_wall, self.particle_wall, self.dem_spatial_grid.cnum, int(mscene.particleNum[0]))
        self.particle_pse.run(self.particle_wall, int(mscene.particleNum[0]) + 1)

    def update_verlet_table_particle_facet(self, mscene: MPMScene, dscene: DEMScene):
        board_search_coupled_particle_facet_linked_cell_(self.csims.wall_coordination_number, self.plane_in_cell, self.msims.verlet_distance, self.dem_spatial_grid.igrid_size, self.dem_spatial_grid.wall_count,
                                                         self.dem_spatial_grid.WallID, mscene.particle, dscene.wall, self.potential_list_particle_wall, self.particle_wall, self.dem_spatial_grid.cnum, int(mscene.particleNum[0]))
        self.particle_pse.run(self.particle_wall, int(mscene.particleNum[0]) + 1)

    def update_verlet_table_particle_patch(self, mscene: MPMScene, dscene: DEMScene):
        board_search_coupled_particle_patch_linked_cell_(self.csims.wall_coordination_number, self.msims.verlet_distance, self.dem_spatial_grid.igrid_size, 
                                                         self.dem_spatial_grid.wall_count, self.dem_spatial_grid.patch_current, self.dem_spatial_grid.WallID, mscene.particle, dscene.wall, 
                                                         self.potential_list_particle_wall, self.particle_wall, self.dem_spatial_grid.cnum, int(mscene.particleNum[0]))
        self.particle_pse.run(self.particle_wall, int(mscene.particleNum[0]) + 1)
    
