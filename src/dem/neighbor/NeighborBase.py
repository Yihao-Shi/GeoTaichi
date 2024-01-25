import taichi as ti

from src.dem.Simulation import Simulation
from src.dem.neighbor.neighbor_kernel import initial_object_object


class NeighborBase(object):
    sims: Simulation

    def __init__(self, sims, scene) -> None:
        self.sims = sims
        self.manage_function(scene)

        self.potential_list_particle_particle = None
        self.particle_particle = None
        self.hist_particle_particle = None
        self.potential_list_particle_wall = None
        self.particle_wall = None
        self.hist_particle_wall = None

    def neighbor_initialze(self):
        raise NotImplementedError

    def manage_function(self, scene):
        raise NotImplementedError
    
    def resize_neighbor(self):
        raise NotImplementedError
    
    def no_operation(self, scene):
        pass

    def no_operation_without_paras(self):
        pass

    def manage_function_base(self):
        self.particle_particle_delete_vars = self.no_operation
        self.update_verlet_tables_particle_particle = self.no_operation
        self.update_particle_particle_auxiliary_lists = self.no_operation_without_paras
        if self.sims.max_particle_num > 1:
            self.particle_particle_delete_vars = self.particle_particle_delete_var
            self.update_verlet_tables_particle_particle = self.update_verlet_table_particle_particle
            self.update_particle_particle_auxiliary_lists = self.update_particle_particle_auxiliary_list

        self.particle_wall_delete_vars = self.no_operation
        self.update_verlet_tables_particle_wall = self.no_operation
        self.update_particle_wall_auxiliary_lists = self.no_operation_without_paras
        if not self.sims.wall_type is None:
            self.update_particle_wall_auxiliary_lists = self.update_particle_wall_auxiliary_list
        if self.sims.wall_type == 0:
            self.particle_wall_delete_vars = self.particle_plane_delete_var
            self.update_verlet_tables_particle_wall = self.update_verlet_table_particle_plane
        elif self.sims.wall_type == 1:
            self.particle_wall_delete_vars = self.particle_facet_delete_var
            self.update_verlet_tables_particle_wall = self.update_verlet_table_particle_facet
        elif self.sims.wall_type == 2:
            self.particle_wall_delete_vars = self.particle_patch_delete_var
            self.update_verlet_tables_particle_wall = self.update_verlet_table_particle_patch

    def set_potential_contact_list(self):
        if self.sims.max_particle_num > 0:
            self.potential_list_particle_particle = ti.field(int, shape=self.sims.max_potential_particle_pairs)
            self.particle_particle = ti.field(int)
            self.hist_particle_particle = ti.field(int)
            ti.root.dense(ti.i, self.sims.max_particle_num + 1).place(self.particle_particle, self.hist_particle_particle)
        else:
            raise RuntimeError("KeyWord:: /max_particle_number/ must be larger than 0!")
        
        if not self.sims.wall_type is None:
            self.potential_list_particle_wall = ti.field(int, shape=self.sims.max_potential_wall_pairs)
            self.particle_wall = ti.field(int)
            self.hist_particle_wall = ti.field(int)
            ti.root.dense(ti.i, self.sims.max_particle_num + 1).place(self.particle_wall, self.hist_particle_wall)

    def particle_particle_delete_var(self):
        del self.potential_list_particle_particle, self.particle_particle, self.hist_particle_particle

    def particle_plane_delete_var(self):
        del self.potential_list_particle_wall, self.particle_wall, self.hist_particle_wall

    def particle_facet_delete_var(self):
        del self.potential_list_particle_wall, self.particle_wall, self.hist_particle_wall

    def particle_patch_delete_var(self):
        del self.potential_list_particle_wall, self.particle_wall, self.hist_particle_wall

    def update_particle_particle_auxiliary_list(self):
        initial_object_object(self.particle_particle, self.hist_particle_particle)

    def update_particle_wall_auxiliary_list(self):
        initial_object_object(self.particle_wall, self.hist_particle_wall)

    def pre_neighbor(self):
        raise NotImplementedError
    
    def update_verlet_table(self, scene):
        raise NotImplementedError
    
    def update_verlet_table_particle_particle(self, scene):
        raise NotImplementedError
    
    def update_verlet_table_particle_plane(self, scene):
        raise NotImplementedError

    def update_verlet_table_particle_facet(self, scene):
        raise NotImplementedError

    def update_verlet_table_particle_patch(self, scene):
        raise NotImplementedError
    
