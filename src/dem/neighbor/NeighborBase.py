import taichi as ti

from src.dem.structs.BaseStruct import VerletContactTable
from src.dem.Simulation import Simulation
from src.dem.SceneManager import myScene
from src.dem.neighbor.neighbor_kernel import *
from src.utils.constants import Threshold
from src.utils.linalg import no_operation


class NeighborBase(object):
    sims: Simulation

    def __init__(self, sims: Simulation, scene) -> None:
        self.sims = sims

        self.potential_list_particle_particle = None
        self.potential_list_point_particle = None
        self.particle_particle = None
        self.hist_particle_particle = None
        self.potential_list_particle_wall = None
        self.potential_list_point_wall = None
        self.particle_wall = None
        self.hist_particle_wall = None
        self.lsparticle_wall = None
        self.hist_lsparticle_wall = None
        self.digital_wall = None
        self.particle_pse = None
        self.point_pse = None
        self.point_rebuild_required = ti.field(dtype=ti.i32, shape=())

        self.update_verlet_tables_point_particle = None
        self.update_verlet_tables_point_wall = None
        self.reset_verletDisp = None
        self.reset_relative_displacements = None
        self.first_run = True
        self.wall_in_cell = 0
        self.brust_serach_facet = False

        self.manage_function(scene)

    def neighbor_initialze(self):
        raise NotImplementedError

    def _soft_particle_backend(self, scene):
        backend = getattr(scene, "soft_particle_backend", None)
        if backend is None:
            raise RuntimeError("LSMPM contact search is an MPM soft-particle mode. Use MPDEM/MPM soft_particle so the coupling layer installs the backend.")
        return backend

    def _verlet_reference_radius(self, scene, bounding_radius):
        if self.sims.scheme == "LSMPM":
            particle_radius = scene.find_particle_min_radius(self.sims)
            if particle_radius > Threshold:
                return particle_radius
        return bounding_radius

    def manage_function(self, scene):
        raise NotImplementedError
    
    def resize_neighbor(self):
        raise NotImplementedError

    def manage_function_base(self, scene: myScene):
        self.particle_particle_delete_vars = no_operation
        self.update_verlet_tables_particle_particle = no_operation
        self.update_particle_particle_auxiliary_lists = no_operation
        if self.sims.max_particle_num > 1:
            self.particle_particle_delete_vars = self.particle_particle_delete_var
            self.update_verlet_tables_particle_particle = self.update_verlet_table_particle_particle
            if self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
                self.update_particle_particle_auxiliary_lists = self.update_LSparticle_LSparticle_auxiliary_list
            else:
                self.update_particle_particle_auxiliary_lists = self.update_particle_particle_auxiliary_list

        self.particle_wall_delete_vars = no_operation
        self.update_verlet_tables_particle_wall = no_operation
        self.update_particle_wall_auxiliary_lists = no_operation
        if not self.sims.wall_type is None and self.sims.max_particle_num > 0 and not self.sims.use_digital_elevation_heightfield():
            if self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
                self.update_particle_wall_auxiliary_lists = self.update_LSparticle_wall_auxiliary_list
            else:
                self.update_particle_wall_auxiliary_lists = self.update_particle_wall_auxiliary_list

        
            self.particle_wall_delete_vars = self.particle_wall_delete_var
            if self.sims.max_particle_num > 0:
                if self.sims.wall_type == 0:
                    self.update_verlet_tables_particle_wall = self.update_verlet_table_particle_plane
                elif self.sims.wall_type == 1:
                    '''if self.sims.static_wall:
                        self.update_verlet_tables_particle_wall = self.update_verlet_table_particle_static_facet
                    else:'''
                    self.update_verlet_tables_particle_wall = self.update_verlet_table_particle_facet
                elif self.sims.wall_type == 2:
                    self.update_verlet_tables_particle_wall = self.update_verlet_table_particle_patch
                elif self.sims.wall_type == 3:
                    self.update_verlet_tables_particle_wall = self.update_verlet_table_particle_digital_elevation

        if self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
            self.update_verlet_tables_point_particle = no_operation
            self.update_verlet_tables_point_wall = no_operation
            if self.sims.max_particle_num > 1:
                self.update_verlet_tables_point_particle = self.update_verlet_table_lsparticle_lsparticle_scenod_layer
            if self.sims.max_wall_num > 0 and self.sims.max_particle_num > 0 and not self.sims.use_digital_elevation_heightfield():
                self.update_verlet_tables_point_wall = self.update_verlet_table_lsparticle_wall_scenod_layer

            self.reset_relative_displacements = no_operation
            if self.sims.max_particle_num > 1:
                if self.sims.max_wall_num == 0 or self.sims.wall_type == 3:
                    self.reset_relative_displacements = self.reset_particle_particle_relative_displacement
                elif self.sims.max_wall_num > 0:
                    self.reset_relative_displacements = self.reset_relative_displacement
            elif self.sims.max_particle_num == 1 and self.sims.max_wall_num > 0:
                self.reset_relative_displacements = self.reset_particle_wall_relative_displacement

        self.reset_verletDisp = no_operation
        if self.sims.max_particle_num > 0 and (self.sims.max_wall_num == 0 or self.sims.wall_type == 0 or self.sims.wall_type == 3):
            self.reset_verletDisp = scene.reset_particle_verlet_disp
        elif self.sims.max_particle_num == 0 and self.sims.max_wall_num > 0 and self.sims.wall_type != 0:
            self.reset_verletDisp = scene.reset_wall_verlet_disp
        elif self.sims.max_particle_num > 0 and self.sims.max_wall_num > 0 and self.sims.wall_type != 0:
            self.reset_verletDisp = scene.reset_verlet_disp

    def set_potential_contact_list(self, scene):
        if self.first_run:
            if self.sims.max_particle_num > 1:
                if self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
                    point_particle_capacity = self.sims.potential_contact_points_particle
                    if self.sims.scheme == "LSDEM":
                        point_particle_capacity *= self.sims.max_particle_num
                    self.potential_list_point_particle = ti.field(int, shape=max(point_particle_capacity, self.sims.max_potential_particle_pairs))
                    self.particle_particle = ti.field(int, shape=self.particle_pse.get_length())
                    self.pplist = VerletContactTable.field(shape=self.sims.particle_verlet_length)
                    self.lsparticle_lsparticle = ti.field(int, shape=self.point_pse.get_length())
                    self.hist_lsparticle_lsparticle = ti.field(int, shape=self.point_pse.get_length())
                    self.potential_list_particle_particle = self.potential_list_point_particle
                else:
                    self.potential_list_particle_particle = ti.field(int, shape=self.sims.max_potential_particle_pairs)
                    self.particle_particle = ti.field(int, shape=self.particle_pse.get_length())
                    self.hist_particle_particle = ti.field(int, shape=self.sims.max_particle_num + 1)
            elif self.sims.max_particle_num < 0:
                raise RuntimeError("KeyWord:: DEM scheme: /max_particle_number/ or LSDEM scheme: /max_rigid_body_number/ must be larger than 0!")
            
            if not self.sims.wall_type is None and not self.sims.use_digital_elevation_heightfield():
                if self.sims.max_particle_num > 0:
                    if self.sims.scheme == "LSDEM" or self.sims.scheme == "LSMPM":
                        point_wall_capacity = self.sims.potential_contact_points_wall
                        if self.sims.scheme == "LSDEM":
                            point_wall_capacity *= self.sims.max_particle_num
                        self.potential_list_point_wall = ti.field(int, shape=max(point_wall_capacity, self.sims.max_potential_wall_pairs))
                        self.particle_wall = ti.field(int, shape=self.particle_pse.get_length())
                        self.pwlist = VerletContactTable.field(shape=self.sims.wall_verlet_length)
                        self.lsparticle_wall = ti.field(int, shape=self.point_pse.get_length())
                        self.hist_lsparticle_wall = ti.field(int, shape=self.point_pse.get_length())
                        self.potential_list_particle_wall = self.potential_list_point_wall
                    else:
                        self.potential_list_particle_wall = ti.field(int, shape=self.sims.max_potential_wall_pairs)
                        self.particle_wall = ti.field(int, shape=self.particle_pse.get_length())
                        self.hist_particle_wall = ti.field(int, shape=self.sims.max_particle_num + 1)
                if self.sims.wall_type == 3 and not self.sims.use_digital_elevation_heightfield():
                    self.place_digital_elevation_facet(scene)
    
    def place_digital_elevation_facet(self, scene: myScene): # digital elevation model
        if self.first_run:
            digital_elevation_grid_number = self.sims.max_digital_elevation_grid_number
            self.digital_wall = ti.field(int, shape=int((digital_elevation_grid_number[0] - 1) * (digital_elevation_grid_number[1] - 1) + 1))
            insert_digital_elevation_facet_(scene.digital_elevation.idigital_size, int(scene.wallNum[0]), scene.digital_elevation.digital_dim, scene.wall, self.digital_wall)

    def check_memory(self):
        if (self.wall_in_cell > 2147483647 or self.sims.max_wall_num < 100) and (not self.sims.search == "HierarchicalLinkedCell"):
            if self.sims.wall_type == 0 or self.sims.wall_type == 1 or self.sims.wall_type == 2:
                #pass
                self.brust_serach_facet = True
                self.wall_hash_tables = no_operation
                self.place_wall_to_cells = no_operation
                if self.sims.max_particle_num > 0:
                    self.update_verlet_tables_particle_wall = self.update_verlet_table_particle_facet_brust
    
    def particle_particle_delete_var(self):
        del self.potential_list_particle_particle, self.particle_particle, self.hist_particle_particle

    def particle_wall_delete_var(self):
        del self.potential_list_particle_wall, self.particle_wall, self.hist_particle_wall

    def update_particle_particle_auxiliary_list(self):
        initial_object_object(self.particle_particle, self.hist_particle_particle)

    def update_particle_wall_auxiliary_list(self):
        initial_object_object(self.particle_wall, self.hist_particle_wall)

    def update_LSparticle_LSparticle_auxiliary_list(self):
        initial_object_object(self.lsparticle_lsparticle, self.hist_lsparticle_lsparticle)

    def update_LSparticle_wall_auxiliary_list(self):
        initial_object_object(self.lsparticle_wall, self.hist_lsparticle_wall)

    def is_particle_particle_point_need_update_verlet_table(self, limit, scene: myScene):
        return validate_pprelative_displacement_(limit, int(scene.particleNum[0]), self.sims.dt, scene.particle, scene.rigid, self.particle_particle, self.pplist)
    
    def is_particle_wall_point_need_update_verlet_table(self, limit, scene: myScene):
        return validate_pwrelative_displacement_(limit, int(scene.particleNum[0]), self.sims.dt, scene.particle, scene.wall, scene.rigid, self.particle_wall, self.pwlist)

    def accumulate_point_relative_displacement(self, scene: myScene):
        """Advance LSDEM second-level Verlet sweeps without synchronizing."""

        particle_num = int(scene.particleNum[0])
        if self.sims.max_particle_num > 1:
            accumulate_pprelative_displacement_(
                particle_num,
                self.sims.dt,
                scene.particle,
                scene.rigid,
                self.particle_particle,
                self.pplist,
            )
        if (
            self.sims.max_particle_num > 0
            and self.sims.max_wall_num > 0
            and self.sims.wall_type is not None
            and self.sims.wall_type != 3
            and not self.sims.use_digital_elevation_heightfield()
        ):
            accumulate_pwrelative_displacement_(
                particle_num,
                self.sims.dt,
                scene.particle,
                scene.wall,
                scene.rigid,
                self.particle_wall,
                self.pwlist,
            )

    def accumulated_point_displacement_requires_rebuild(
        self, limit, scene: myScene
    ):
        """Reduce all second-level Verlet sweeps with one host flag read."""

        particle_num = int(scene.particleNum[0])
        self.point_rebuild_required.fill(0)
        if self.sims.max_particle_num > 1:
            flag_relative_displacement_(
                limit,
                particle_num,
                self.particle_particle,
                self.pplist,
                self.point_rebuild_required,
            )
        if (
            self.sims.max_particle_num > 0
            and self.sims.max_wall_num > 0
            and self.sims.wall_type is not None
            and self.sims.wall_type != 3
            and not self.sims.use_digital_elevation_heightfield()
        ):
            flag_relative_displacement_(
                limit,
                particle_num,
                self.particle_wall,
                self.pwlist,
                self.point_rebuild_required,
            )
        return int(self.point_rebuild_required[None]) == 1

    def update_verlet_table_lsparticle_lsparticle_scenod_layer(self, scene: myScene):
        if self.sims.scheme == "LSMPM":
            self._soft_particle_backend(scene).update_verlet_table_lsparticle_lsparticle_scenod_layer(self, scene)
        else:
            board_search_lsparticle_lsparticle_linked_cell_(int(scene.particleNum[0]), self.sims.point_particle_coordination_number, self.sims.point_verlet_distance, self.pplist, 
                                                            self.potential_list_point_particle, self.particle_particle, self.lsparticle_lsparticle, scene.rigid, scene.box, scene.vertice, scene.rigid_grid)
            scene.lsContactNodeNum[0] = int(scene.surfaceNum[0])
        self.point_pse.run(self.lsparticle_lsparticle)

    def update_verlet_table_lsparticle_wall_scenod_layer(self, scene: myScene):
        if self.sims.scheme == "LSMPM":
            self._soft_particle_backend(scene).update_verlet_table_lsparticle_wall_scenod_layer(self, scene)
        else:
            board_search_lsparticle_wall_linked_cell_(int(scene.particleNum[0]), self.sims.point_wall_coordination_number, self.sims.point_verlet_distance, self.pwlist, 
                                                      self.potential_list_point_wall, self.particle_wall, self.lsparticle_wall, scene.wall, scene.rigid, scene.vertice, scene.box)
            scene.lsContactNodeNum[0] = int(scene.surfaceNum[0])
        self.point_pse.run(self.lsparticle_wall)

    def reset_particle_particle_relative_displacement(self, scene: myScene):
        reset_relative_displacement(int(scene.particleNum[0]), self.particle_particle, self.pplist)

    def reset_particle_wall_relative_displacement(self, scene: myScene):
        reset_relative_displacement(int(scene.particleNum[0]), self.particle_wall, self.pwlist)

    def reset_relative_displacement(self, scene: myScene):
        reset_relative_displacement(int(scene.particleNum[0]), self.particle_particle, self.pplist)
        reset_relative_displacement(int(scene.particleNum[0]), self.particle_wall, self.pwlist)
    
    def update_point_verlet_table(self, scene):
        self.reset_relative_displacements(scene)
        self.update_verlet_tables_point_particle(scene)
        self.update_verlet_tables_point_wall(scene)

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
    
    def update_verlet_table_lsparticle_lsparticle(self, scene):
        raise NotImplementedError

    def update_verlet_table_lsparticle_plane(self, scene):
        raise NotImplementedError

    def update_verlet_table_lsparticle_facet(self, scene):
        raise NotImplementedError
    
    def update_verlet_table_particle_static_facet(self, scene: myScene):
        raise NotImplementedError

    def update_verlet_table_lsparticle_patch(self, scene):
        raise NotImplementedError
    
    def update_verlet_table_particle_digital_elevation(self, scene: myScene):
        board_search_particle_digital_elevation_(int(scene.particleNum[0]), self.sims.wall_coordination_number, self.sims.verlet_distance, scene.digital_elevation.idigital_size, scene.digital_elevation.digital_dim,
                                                 scene.particle, scene.wall, self.digital_wall, self.potential_list_particle_wall, self.particle_wall)
        self.particle_pse.run(self.particle_wall)

    def update_verlet_table_particle_facet_brust(self, scene: myScene):
        board_search_particle_wall_brust_(self.sims.wall_coordination_number, int(scene.particleNum[0]), int(scene.wallNum[0]), self.sims.verlet_distance, scene.particle, scene.wall, self.potential_list_particle_wall, self.particle_wall)
        self.particle_pse.run(self.particle_wall)
    
    
