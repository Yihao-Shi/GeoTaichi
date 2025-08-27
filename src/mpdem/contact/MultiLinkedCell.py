import taichi as ti

from src.dem.neighbor.neighbor_kernel import *
from src.dem.neighbor.LinkedCell import LinkedCell
from src.dem.SceneManager import myScene as DEMScene
from src.dem.Simulation import Simulation as DEMSimulation
from src.mpdem.Simulation import Simulation as DEMPMSimulation
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

    def __init__(self, csims, msims, dsims: DEMSimulation, mpm_neighbor, dem_neighbor) -> None:
        super().__init__()
        self.csims = csims
        self.msims = msims
        self.dsims = dsims
        self.mpm_spatial_grid = mpm_neighbor
        self.dem_spatial_grid = dem_neighbor
        self.plane_in_cell = self.dsims.wall_per_cell
        self.facet_in_cell = self.dsims.wall_per_cell
        self.manage_function()

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

    def manage_function(self):
        self.particle_particle_delete_vars = self.no_operation
        self.update_verlet_tables_particle_particle = self.no_operation
        self.update_particle_particle_auxiliary_lists = self.no_operation_without_paras
        if self.csims.particle_interaction and self.dsims.max_particle_num > 0:
            self.particle_particle_delete_vars = self.particle_particle_delete_var
            self.update_particle_particle_auxiliary_lists = self. update_particle_particle_auxiliary_list
            if self.dsims.scheme == "DEM":
                if self.dsims.search == "HierarchicalLinkedCell":
                    self.update_verlet_tables_particle_particle = self.update_verlet_table_particle_particle_hierarchical
                elif self.dsims.search == "LinkedCell":
                    if self.csims.enhanced_coupling:
                        self.update_verlet_tables_particle_particle = self.update_verlet_table_enhanced_particle_particle
                    else:
                        self.update_verlet_tables_particle_particle = self.update_verlet_table_particle_particle
            elif self.dsims.scheme == "LSDEM":
                if self.dsims.search == "HierarchicalLinkedCell":
                    self.update_verlet_tables_particle_particle = self.update_verlet_table_particle_lsparticle_hierarchical
                elif self.dsims.search == "LinkedCell":
                    self.update_verlet_tables_particle_particle = self.update_verlet_table_particle_lsparticle

        self.particle_wall_delete_vars = self.no_operation
        self.update_verlet_table_particle_wall = self.no_operation
        self.update_particle_wall_auxiliary_lists = self. no_operation_without_paras
        if self.csims.wall_interaction:
            if not self.dsims.wall_type is None:
                self.update_particle_wall_auxiliary_lists = self.update_particle_wall_auxiliary_list

            if self.dsims.wall_type == 0:
                self.particle_wall_delete_vars = self.particle_plane_delete_var
                self.update_verlet_table_particle_wall = self.update_verlet_table_particle_plane
            elif self.dsims.wall_type == 1:
                self.particle_wall_delete_vars = self.particle_facet_delete_var
                self.update_verlet_table_particle_wall = self.update_verlet_table_particle_facet
            elif self.dsims.wall_type == 2:
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

    def set_potential_contact_list(self, mscene: MPMScene, dscene: DEMScene):
        mrad_min, mrad_max = mscene.find_particle_min_radius(), mscene.find_particle_max_radius()
        drad_min, drad_max = dscene.find_bounding_sphere_min_radius(self.dsims), dscene.find_bounding_sphere_max_radius(self.dsims)
        self.msims.set_verlet_distance(mrad_min)
        self.msims.set_max_radius(mrad_max)
        self.dsims.set_verlet_distance(drad_min)
        self.csims.set_bounding_sphere(min(mrad_min, drad_min), max(mrad_max, drad_max))
        self.csims.set_potential_list_size(self.msims, self.dsims, drad_max, mrad_max)
        if self.csims.enhanced_coupling:
            self.particle_pse = PrefixSumExecutor(self.dsims.max_particle_num + 1)
        else:
            self.particle_pse = PrefixSumExecutor(self.msims.max_coupling_particle_num + 1)

        if self.csims.particle_interaction:
            self.potential_list_particle_particle = ti.field(int, shape=self.csims.max_potential_particle_pairs)
            self.particle_particle = ti.field(int, shape=self.particle_pse.get_length())
            if self.csims.enhanced_coupling:
                self.hist_particle_particle = ti.field(int, shape=self.dsims.max_particle_num + 1)
            else:
                self.hist_particle_particle = ti.field(int, shape=self.msims.max_coupling_particle_num + 1)
        
        if self.csims.wall_interaction and not self.dsims.wall_type is None:
            self.potential_list_particle_wall = ti.field(int, shape=self.csims.max_potential_wall_pairs)
            self.particle_wall = ti.field(int, shape=self.particle_pse.get_length())
            self.hist_particle_wall = ti.field(int, shape=self.msims.max_coupling_particle_num + 1)

        if self.dsims.scheme == "LSDEM":
            self.dsims.check_grid_extent(*dscene.find_expect_extent(self.dsims, self.msims.verlet_distance + self.dsims.verlet_distance + mrad_max))

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
        board_search_coupled_particle_linked_cell_(self.csims.potential_particle_num, self.msims.verlet_distance, self.dsims.verlet_distance, self.dsims.max_bounding_sphere_radius, self.dem_spatial_grid.igrid_size, self.dem_spatial_grid.particle_count, 
                                                   self.dem_spatial_grid.ParticleID, mscene.particle, dscene.particle, self.potential_list_particle_particle, self.particle_particle, self.dem_spatial_grid.cnum, int(mscene.couplingNum[0]))
        self.particle_pse.run(self.particle_particle)

    def update_verlet_table_enhanced_particle_particle(self, mscene: MPMScene, dscene: DEMScene):
        board_search_enhanced_coupled_particle_linked_cell_(self.csims.potential_particle_num, self.msims.verlet_distance, self.dsims.verlet_distance, self.msims.max_radius, self.mpm_spatial_grid.igrid_size, self.mpm_spatial_grid.sorted.bin_count, 
                                                   self.mpm_spatial_grid.sorted.object_list, dscene.particle, mscene.particle, self.potential_list_particle_particle, self.particle_particle, self.mpm_spatial_grid.cnum, int(dscene.particleNum[0]))
        self.particle_pse.run(self.particle_particle)

    def update_verlet_table_particle_particle_hierarchical(self, mscene: MPMScene, dscene: DEMScene):
        board_search_coupled_particle_linked_cell_hierarchical_(self.csims.potential_particle_num, self.msims.verlet_distance, self.dsims.verlet_distance, self.dsims.max_bounding_sphere_radius, self.dem_spatial_grid.particle_count, 
                                                                self.dem_spatial_grid.ParticleID, mscene.particle, dscene.particle, self.potential_list_particle_particle, self.particle_particle, int(mscene.couplingNum[0]), self.dsims.hierarchical_level, self.dem_spatial_grid.grid)
        self.particle_pse.run(self.particle_particle)

    def update_verlet_table_particle_lsparticle(self, mscene: MPMScene, dscene: DEMScene):
        board_search_coupled_lsparticle_linked_cell_(self.csims.potential_particle_num, self.msims.verlet_distance, self.dsims.verlet_distance, self.dsims.max_bounding_sphere_radius, self.dem_spatial_grid.igrid_size, self.dem_spatial_grid.particle_count, 
                                                     self.dem_spatial_grid.ParticleID, mscene.particle, dscene.rigid, dscene.box, dscene.rigid_grid, self.potential_list_particle_particle, 
                                                     self.particle_particle, self.dem_spatial_grid.cnum, int(mscene.couplingNum[0]))
        self.particle_pse.run(self.particle_particle)

    def update_verlet_table_particle_lsparticle_hierarchical(self, mscene: MPMScene, dscene: DEMScene):
        board_search_coupled_lsparticle_linked_cell_hierarchical_(self.csims.potential_particle_num, self.msims.verlet_distance, self.dsims.verlet_distance, self.dsims.max_bounding_sphere_radius, self.dem_spatial_grid.particle_count, 
                                                                  self.dem_spatial_grid.ParticleID, mscene.particle, dscene.rigid, dscene.box, self.potential_list_particle_particle, 
                                                                  self.particle_particle, int(mscene.couplingNum[0]), self.dsims.hierarchical_level, self.dem_spatial_grid.grid)
        self.particle_pse.run(self.particle_particle)

    def update_verlet_table_particle_plane(self, mscene: MPMScene, dscene: DEMScene):
        board_search_particle_plane_linked_cell_(int(mscene.couplingNum[0]), self.csims.wall_coordination_number, self.plane_in_cell, self.msims.verlet_distance, self.dem_spatial_grid.igrid_size, self.dem_spatial_grid.wall_count,
                                                 self.dem_spatial_grid.WallID, mscene.particle, dscene.wall, self.potential_list_particle_wall, self.particle_wall, self.dem_spatial_grid.cnum)
        self.particle_pse.run(self.particle_wall)

    def update_verlet_table_particle_facet(self, mscene: MPMScene, dscene: DEMScene):
        board_search_particle_facet_linked_cell_(int(mscene.couplingNum[0]), self.csims.wall_coordination_number, self.plane_in_cell, self.msims.verlet_distance, self.dem_spatial_grid.igrid_size, self.dem_spatial_grid.wall_count,
                                                 self.dem_spatial_grid.WallID, mscene.particle, dscene.wall, self.potential_list_particle_wall, self.particle_wall, self.dem_spatial_grid.cnum)
        self.particle_pse.run(self.particle_wall)

    def update_verlet_table_particle_patch(self, mscene: MPMScene, dscene: DEMScene):
        board_search_particle_patch_linked_cell_(int(mscene.couplingNum[0]), self.csims.wall_coordination_number, self.msims.verlet_distance, self.dem_spatial_grid.igrid_size, 
                                                 self.dem_spatial_grid.wall_count, self.dem_spatial_grid.WallID, mscene.particle, dscene.wall, 
                                                 self.potential_list_particle_wall, self.particle_wall, self.dem_spatial_grid.cnum)
        self.particle_pse.run(self.particle_wall)

    def update_verlet_table_particle_digital_elevation(self, mscene: MPMScene, dscene: DEMScene):
        board_search_particle_digital_elevation_(int(mscene.couplingNum[0]), self.csims.wall_coordination_number, self.msims.verlet_distance, dscene.digital_elevation.idigital_size, dscene.digital_elevation.digital_dim,
                                                 mscene.particle, dscene.wall, self.dem_spatial_grid.digital_wall, self.potential_list_particle_wall, self.particle_wall)
        self.particle_pse.run(self.particle_wall)
    
