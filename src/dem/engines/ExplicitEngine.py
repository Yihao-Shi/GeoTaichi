from src.dem.contact.ContactModelBase import ContactModelBase
from src.dem.ContactManager import ContactManager
from src.dem.neighbor.NeighborBase import NeighborBase
from src.dem.engines.EngineKernel import *
from src.dem.SceneManager import myScene
from src.dem.Simulation import Simulation
from src.utils.SolverRuntime import normalize_callbacks
from src.utils.linalg import no_operation


class ExplicitEngine(object):
    scene: myScene
    neighbor: NeighborBase
    physpp: ContactModelBase
    physpw: ContactModelBase

    def __init__(self, scene, contactor: ContactManager):
        self.scene = scene
        self.neighbor = contactor.neighbor
        self.physpp = contactor.physpp
        self.physpw = contactor.physpw

        self.update_servo_wall = None
        self.callback = None
        self.calm = None
        self.get_servo_wall_contact_forces = None

        self.limit1 = 0.0
        self.limit2 = 0.0
        self.soft_particle_material = None
        self.soft_particle_backend = None
        self.verlet_enabled = False
        self.verlet_force_ready = False
        self.verlet_load_fields = []
        self.transform_translational_load = no_operation

    def _soft_particle_backend(self, scene):
        backend = getattr(scene, "soft_particle_backend", None)
        if backend is None:
            raise RuntimeError(
                "LSMPM integration is an MPM soft-particle mode. Use MPDEM/MPM soft_particle so the coupling layer installs the backend."
            )
        return backend

    def choose_engine(self, sims: Simulation, scene: myScene):
        self.verlet_enabled = sims.engine == "VelocityVerlet" and sims.scheme != "LSMPM"
        self.verlet_force_ready = False
        self.verlet_load_fields = []
        if self.verlet_enabled and sims.max_particle_num > 0:
            # One force and torque per existing capacity slot; no contact-pair
            # over-allocation. Reuse these buffers for DEM then external loads.
            body = scene.particle if sims.scheme == "DEM" else scene.rigid
            self.verlet_load_fields.append(
                (
                    scene.particleNum,
                    body,
                    ti.Vector.field(3, float, shape=sims.max_particle_num),
                    ti.Vector.field(3, float, shape=sims.max_particle_num),
                )
            )
            if sims.enable_shell and sims.max_wall_num > 0:
                self.verlet_load_fields.append(
                    (
                        scene.wallNum,
                        scene.wall,
                        ti.Vector.field(3, float, shape=sims.max_wall_num),
                        ti.Vector.field(3, float, shape=sims.max_wall_num),
                    )
                )
            self.verlet_probe_dt = ti.field(float, shape=())
            self.verlet_probe_dt[None] = 0.0
        if sims.scheme == "LSMPM":
            self.soft_particle_backend = self._soft_particle_backend(scene)
            self.soft_particle_backend.bind_explicit_engine(self, sims, scene)

        self.update_neighbor_lists = self.update_neighbor_list
        if (sims.scheme == "LSDEM" or sims.scheme == "LSMPM") and sims.max_particle_num > 0:
            self.update_neighbor_lists = self.update_LSneighbor_list

        self.reset_particle_message = no_operation
        if sims.max_particle_num > 0:
            if sims.scheme == "DEM":
                self.reset_particle_message = self.reset_particle
            else:
                self.reset_particle_message = self.reset_level_set_particle

        self.reset_wall_message = no_operation
        self.get_wall_contact_forces = no_operation
        if sims.max_wall_num > 0 and (sims.wall_type == 1 or sims.enable_shell):
            if sims.enable_shell or sims.servo_status == "On":
                self.reset_wall_message = self.reset_wall
            if "wall" in sims.monitor_type:
                self.reset_wall_message = self.reset_wall
                if sims.max_particle_num > 0:
                    if sims.scheme == "LSMPM":
                        self.get_wall_contact_forces = self.get_lsmpm_wall_contact_force
                    else:
                        self.get_wall_contact_forces = self.get_wall_contact_force

        self.calcu_sphere_position = no_operation
        self.calcu_clump_position = no_operation
        self.calcu_wall_position = no_operation
        if sims.engine == "SymplecticEuler":
            if sims.scheme == "DEM" and sims.max_sphere_num > 0:
                self.calcu_sphere_position = self.euler_sphere_integration
            elif (
                sims.scheme == "LSDEM" or sims.scheme == "PolySuperEllipsoid" or sims.scheme == "PolySuperQuadrics"
            ) and sims.max_rigid_body_num > 0:
                self.calcu_sphere_position = self.euler_level_set_integration
            elif sims.scheme == "LSMPM" and sims.max_particle_num > 0:
                self.calcu_sphere_position = self.euler_lsmpm_integration
            if sims.scheme == "DEM" and sims.max_clump_num > 0:
                self.calcu_clump_position = self.euler_clump_integration
        elif sims.engine == "VelocityVerlet":
            if sims.scheme == "DEM" and sims.max_sphere_num > 0:
                self.calcu_sphere_position = self.verlet_sphere_integration
            elif (
                sims.scheme == "LSDEM" or sims.scheme == "PolySuperEllipsoid" or sims.scheme == "PolySuperQuadrics"
            ) and sims.max_rigid_body_num > 0:
                self.calcu_sphere_position = self.verlet_level_set_integration
            elif sims.scheme == "LSMPM" and sims.max_particle_num > 0:
                self.calcu_sphere_position = self.euler_lsmpm_integration
            if sims.scheme == "DEM" and sims.max_clump_num > 0:
                self.calcu_clump_position = self.verlet_clump_integration
        elif sims.engine == "PredictCorrector":
            pass
        else:
            raise RuntimeError("Engine Type is error")

        if sims.max_wall_num > 0 and sims.static_wall is False:
            if sims.wall_type == 1:
                self.calcu_wall_position = self.wall_integration
            elif sims.wall_type == 2:
                self.calcu_wall_position = self.patch_integration

        self.calm = self.launch_calm1
        if sims.max_clump_num > 0:
            self.calm = self.launch_calm2
        if sims.scheme == "LSDEM" or sims.scheme == "LSMPM":
            self.calm = self.launch_calm3

        self.is_verlet_update = no_operation
        if sims.max_particle_num > 0 and (sims.max_wall_num == 0 or sims.wall_type == 0 or sims.wall_type == 3):
            self.is_verlet_update = scene.is_particle_need_update_verlet_table
        elif sims.max_particle_num == 0 and sims.max_wall_num > 0 and (sims.wall_type == 1 or sims.wall_type == 2):
            self.is_verlet_update = scene.is_wall_need_update_verlet_table
        elif sims.max_particle_num > 0 and sims.max_wall_num > 0 and (sims.wall_type == 1 or sims.wall_type == 2):
            self.is_verlet_update = scene.is_need_update_verlet_table
        if sims.scheme == "LSMPM" and sims.max_particle_num > 0 and sims.max_soft_body_num > 0:
            if sims.max_wall_num > 0 and (sims.wall_type == 1 or sims.wall_type == 2):
                self.is_verlet_update = scene.is_deformable_need_update_verlet_table
            else:
                self.is_verlet_update = scene.is_deformable_particle_need_update_verlet_table
        if sims.scheme == "LSDEM" or sims.scheme == "LSMPM":
            self.is_verlet_update_point = no_operation
            if sims.max_particle_num > 1:
                if sims.max_wall_num == 0 or sims.wall_type == 3:
                    self.is_verlet_update_point = self.is_point_particle_need_update_verlet_table
                elif sims.max_wall_num > 0:
                    self.is_verlet_update_point = self.is_point_need_update_verlet_table
            elif sims.max_particle_num == 1 and sims.max_wall_num > 0 and not sims.use_digital_elevation_heightfield():
                self.is_verlet_update_point = self.is_point_wall_need_update_verlet_table

    def set_servo_mechanism(self, sims: Simulation, callback=None):
        self.update_servo_wall = no_operation
        self.get_contact_stiffnesses = no_operation
        self.get_servo_wall_contact_forces = self.get_wall_contact_force
        if sims.scheme == "LSMPM":
            self.get_servo_wall_contact_forces = self.get_lsmpm_wall_contact_force
        if sims.max_servo_wall_num > 0 and sims.servo_status == "On":
            if sims.servo_type == "StiffnessControl":
                self.update_servo_wall = self.update_servo_stiffness_control
            elif sims.servo_type == "GainControl":
                self.update_servo_wall = self.update_servo_gain_control
            if callback is None:
                self.callback = no_operation
            else:
                self.callback = normalize_callbacks(callback, ti.kernel)[0]

            if sims.max_particle_num > 0:
                if sims.scheme == "DEM":
                    self.get_contact_stiffnesses = self.get_contact_stiffness
                elif sims.scheme == "LSDEM" or sims.scheme == "LSMPM":
                    self.get_contact_stiffnesses = self.get_ls_contact_stiffness
                else:
                    raise RuntimeError(f"Servo wall is not supported under this scheme {sims.scheme}")

    def launch_calm1(self, current_step, calm_interval, scene: myScene):
        if current_step % calm_interval == 0:
            scene.particle_calm()

    def launch_calm2(self, current_step, calm_interval, scene: myScene):
        if current_step % calm_interval == 0:
            scene.particle_calm()
            scene.clump_calm()

    def launch_calm3(self, current_step, calm_interval, scene: myScene):
        if current_step % calm_interval == 0:
            scene.rigid_calm()

    def reset_particle(self, scene: myScene):
        self.verlet_force_ready = False
        particle_force_reset_(int(scene.particleNum[0]), scene.particle)

    def reset_level_set_particle(self, scene: myScene):
        self.verlet_force_ready = False
        particle_force_reset_(int(scene.particleNum[0]), scene.rigid)
        if scene.soft_point is not None:
            # The reset precedes contact assembly.  Ensure newly inserted soft
            # bodies already expose their invariant TLMPM nodal mass here,
            # rather than waiting for the integration phase of the step.
            self.soft_particle_backend.ensure_soft_grid_reference_mass(self, scene)
            if scene.vertice is not None and scene.surface is not None:
                self.soft_particle_backend.soft_surface_force_reset(
                    int(scene.softNum[0]),
                    int(scene.surfaceNum[0]),
                    scene.soft,
                    scene.surface,
                    scene.rigid,
                    scene.vertice,
                )
            self.soft_particle_backend.soft_material_point_force_reset(int(scene.softPointNum[0]), scene.soft_point)

    def reset_contact_energy(self):
        self.physpp.reset()
        self.physpw.reset()

    def reset_wall(self, scene: myScene):
        wall_force_reset_(int(scene.wallNum[0]), scene.wall)

    def is_point_need_update_verlet_table(self, limit):
        return self.neighbor.is_particle_particle_point_need_update_verlet_table(
            limit, self.scene
        ) or self.neighbor.is_particle_wall_point_need_update_verlet_table(limit, self.scene)

    def is_point_particle_need_update_verlet_table(self, limit):
        return self.neighbor.is_particle_particle_point_need_update_verlet_table(limit, self.scene)

    def is_point_wall_need_update_verlet_table(self, limit):
        return self.neighbor.is_particle_wall_point_need_update_verlet_table(limit, self.scene)

    def pre_calculation(self, sims: Simulation, scene: myScene, neighbor: NeighborBase):
        scene.apply_boundary_conditions(sims)
        if sims.scheme == "LSDEM" or sims.scheme == "LSMPM":
            neighbor.pre_neighbor(scene)
            self.physpp.update_verlet_particle_particle_tables(sims, scene, neighbor)
            self.physpw.update_verlet_particle_wall_tables(sims, scene, neighbor)
            neighbor.update_point_verlet_table(scene)
            self.physpp.update_contact_table(sims, scene, neighbor)
            self.physpw.update_contact_table(sims, scene, neighbor)
            neighbor.update_particle_particle_auxiliary_lists()
            neighbor.update_particle_wall_auxiliary_lists()
            self.limit1 = sims.verlet_distance * sims.verlet_distance
            self.limit2 = sims.point_verlet_distance * sims.point_verlet_distance
        else:
            neighbor.pre_neighbor(scene)
            self.physpp.update_contact_table(sims, scene, neighbor)
            self.physpw.update_contact_table(sims, scene, neighbor)
            neighbor.update_particle_particle_auxiliary_lists()
            neighbor.update_particle_wall_auxiliary_lists()
            self.system_resolve(sims, scene, neighbor)
            self.limit1 = sims.verlet_distance * sims.verlet_distance

    def update_neighbor_list(self, sims: Simulation, scene: myScene, neighbor: NeighborBase, advance_history=False):
        sims.timer.begin("Others")
        need_verlet_update = self.is_verlet_update(self.limit1) == 1
        sims.timer.end("Others")
        if need_verlet_update:
            self.update_verlet_table(sims, scene, neighbor)
        self.system_resolve(sims, scene, neighbor, advance_history=advance_history)

    def update_verlet_table(self, sims: Simulation, scene: myScene, neighbor: NeighborBase):
        sims.timer.begin("Boundary condition")
        scene.apply_boundary_conditions(sims)
        sims.timer.end("Boundary condition")
        sims.timer.begin("Broad search")
        neighbor.update_verlet_table(scene)
        sims.timer.end("Broad search")
        sims.timer.begin("Narrow search")
        self.physpp.update_contact_table(sims, scene, neighbor)
        self.physpw.update_contact_table(sims, scene, neighbor)
        neighbor.update_particle_particle_auxiliary_lists()
        neighbor.update_particle_wall_auxiliary_lists()
        sims.timer.end("Narrow search")

    def update_LSneighbor_list(self, sims: Simulation, scene: myScene, neighbor: NeighborBase, advance_history=False):
        sims.timer.begin("Others 1")
        need_particle_particle_verlet_update = self.is_verlet_update(self.limit1) == 1
        sims.timer.end("Others 1")
        if need_particle_particle_verlet_update:
            self.update_LSDEM_verlet_table1(sims, scene, neighbor)
            self.update_LSDEM_verlet_table2(sims, scene, neighbor)
        else:
            sims.timer.begin("Others 2")
            need_point_particle_verlet_update = self.is_verlet_update_point(self.limit2) == 1
            sims.timer.end("Others 2")
            if need_point_particle_verlet_update:
                self.update_LSDEM_verlet_table2(sims, scene, neighbor)
        self.system_resolve(sims, scene, neighbor, advance_history=advance_history)

    def update_LSDEM_verlet_table1(self, sims: Simulation, scene: myScene, neighbor: NeighborBase):
        sims.timer.begin("Boundary condition")
        scene.apply_boundary_conditions(sims)
        sims.timer.end("Boundary condition")
        sims.timer.begin("Broad search 1")
        neighbor.update_verlet_table(scene)
        sims.timer.end("Broad search 1")
        sims.timer.begin("Narrow search 1")
        self.physpp.update_verlet_particle_particle_tables(sims, scene, neighbor)
        self.physpw.update_verlet_particle_wall_tables(sims, scene, neighbor)
        sims.timer.end("Narrow search 1")

    def update_LSDEM_verlet_table2(self, sims: Simulation, scene: myScene, neighbor: NeighborBase):
        sims.timer.begin("Broad search 2")
        neighbor.update_point_verlet_table(scene)
        sims.timer.end("Broad search 2")
        sims.timer.begin("Narrow search 2")
        self.physpp.update_contact_table(sims, scene, neighbor)
        self.physpw.update_contact_table(sims, scene, neighbor)
        neighbor.update_particle_particle_auxiliary_lists()
        neighbor.update_particle_wall_auxiliary_lists()
        sims.timer.end("Narrow search 2")

    def system_resolve(self, sims: Simulation, scene: myScene, neighbor: NeighborBase, advance_history=False):
        sims.timer.begin("Force calculate")
        probe = self.verlet_enabled and bool(self.verlet_load_fields) and not advance_history
        physical_dt = sims.dt
        if probe:
            self.verlet_force_ready = False
            for count, body, force, torque in self.verlet_load_fields:
                cache_dem_external_load(int(count[0]), body, force, torque)
            # Only DEM's elastic/history contact models are evaluated here.
            # Never use this zero history increment for a fluid/cross-contact
            # impulse law: those callers retain the physical timestep.
            sims.dt = self.verlet_probe_dt
        try:
            self.physpp.resolve(sims, scene, neighbor)
            self.physpw.resolve(sims, scene, neighbor)
        finally:
            sims.dt = physical_dt
            sims.timer.end("Force calculate")
        if probe:
            for count, body, force, torque in self.verlet_load_fields:
                subtract_dem_load_cache(int(count[0]), body, force, torque)
            self.verlet_force_ready = True

    def get_contact_stiffness(self, sims: Simulation, scene: myScene, neighbor: NeighborBase):
        get_contact_stiffness_(
            sims.max_material_num,
            int(scene.particleNum[0]),
            scene.particle,
            scene.wall,
            self.physpw.surfaceProps,
            self.physpw.cplist,
            neighbor.particle_wall,
        )

    def get_ls_contact_stiffness(self, sims: Simulation, scene: myScene, neighbor: NeighborBase):
        if sims.scheme == "LSMPM":
            get_LSMPM_contact_stiffness_(
                sims.max_material_num,
                int(scene.lsContactNodeNum[0]),
                scene.rigid,
                scene.vertice,
                scene.soft_point,
                scene.ls_contact_body,
                scene.ls_contact_kind,
                scene.ls_contact_ref,
                scene.wall,
                self.physpw.surfaceProps,
                self.physpw.cplist,
                neighbor.lsparticle_wall,
            )
        else:
            get_LScontact_stiffness_(
                sims.max_material_num,
                int(scene.surfaceNum[0]),
                scene.surface,
                scene.particle,
                scene.rigid,
                scene.vertice,
                scene.wall,
                self.physpw.surfaceProps,
                self.physpw.cplist,
                neighbor.lsparticle_wall,
            )

    def get_wall_contact_force(self, scene: myScene, neighbor: NeighborBase):
        get_wall_contact_force_(int(scene.particleNum[0]), scene.wall, self.physpw.cplist, neighbor.particle_wall)

    def get_lsmpm_wall_contact_force(self, scene: myScene, neighbor: NeighborBase):
        get_LSMPM_wall_contact_force_(
            int(scene.lsContactNodeNum[0]), scene.wall, self.physpw.cplist, neighbor.lsparticle_wall
        )

    def update_servo_stiffness_control(self, sims: Simulation, scene: myScene, neighbor: NeighborBase):
        sims.timer.begin("Servo wall")
        self.get_contact_stiffnesses(sims, scene, neighbor)
        self.callback()
        get_gain(sims.dt, int(scene.servoNum[0]), scene.servo, scene.wall)
        servo(int(scene.servoNum[0]), scene.wall, scene.servo)
        sims.timer.end("Servo wall")

    def update_servo_gain_control(self, sims: Simulation, scene: myScene, neighbor: NeighborBase):
        sims.timer.begin("Servo wall")
        self.get_servo_wall_contact_forces(scene, neighbor)
        self.callback()
        servo(int(scene.servoNum[0]), scene.wall, scene.servo)
        sims.timer.end("Servo wall")

    def integration(self, sims: Simulation, scene: myScene, neighbor: NeighborBase):
        self.update_servo_wall(sims, scene, neighbor)
        if sims.scheme == "LSMPM":
            self.calcu_sphere_position(sims, scene)
            sims.timer.begin("LSMPM Wall integration")
            self.calcu_wall_position(sims, scene)
            sims.timer.end("LSMPM Wall integration")
            sims.timer.begin("LSMPM Wall contact reduction")
            self.get_wall_contact_forces(scene, neighbor)
            sims.timer.end("LSMPM Wall contact reduction")
            return

        sims.timer.begin("Time integration")
        if self.verlet_enabled and self.verlet_load_fields:
            if not self.verlet_force_ready:
                raise RuntimeError("VelocityVerlet requires current DEM contact-force assembly before integration")
            # Preserve every non-DEM load added before OR after DEM assembly.
            # The coupled solver owns its sampling time; do not advance the
            # fluid/FEM/MPM a second time during the DEM correction.
            for count, body, force, torque in self.verlet_load_fields:
                subtract_dem_load_cache(int(count[0]), body, force, torque)
        self.transform_translational_load()
        self.calcu_sphere_position(sims, scene)
        self.calcu_clump_position(sims, scene)
        self.calcu_wall_position(sims, scene)
        if self.verlet_enabled and self.verlet_load_fields:
            for count, body, force, torque in self.verlet_load_fields:
                restore_dem_external_load(int(count[0]), body, force, torque)
            self.reset_contact_energy()
            self.update_neighbor_lists(sims, scene, neighbor, advance_history=True)
            self.transform_translational_load()
            if sims.scheme == "DEM":
                if sims.max_sphere_num > 0:
                    move_spheres_verlet_corrector_(
                        int(scene.sphereNum[0]), sims.dt, scene.sphere, scene.particle, scene.material, sims.gravity
                    )
                if sims.max_clump_num > 0:
                    move_clumps_verlet_corrector_(
                        int(scene.clumpNum[0]), sims.dt, scene.clump, scene.particle, scene.material, sims.gravity
                    )
            else:
                move_level_set_verlet_corrector_(
                    int(scene.particleNum[0]), sims.dt, scene.rigid, scene.material, sims.gravity
                )
            self.verlet_force_ready = False
        self.get_wall_contact_forces(scene, neighbor)
        sims.timer.end("Time integration")

    def euler_sphere_integration(self, sims: Simulation, scene: myScene):
        move_spheres_euler_(
            int(scene.sphereNum[0]), sims.dt, scene.sphere, scene.particle, scene.material, sims.gravity
        )

    def euler_clump_integration(self, sims: Simulation, scene: myScene):
        move_clumps_euler_(int(scene.clumpNum[0]), sims.dt, scene.clump, scene.particle, scene.material, sims.gravity)

    def euler_level_set_integration(self, sims: Simulation, scene: myScene):
        move_level_set_euler_(
            int(scene.particleNum[0]), sims.dt, scene.particle, scene.rigid, scene.material, sims.gravity
        )

    def euler_lsmpm_integration(self, sims: Simulation, scene: myScene):
        self.soft_particle_backend.euler_lsmpm_integration(self, sims, scene)

    def wall_integration(self, sims: Simulation, scene: myScene):
        move_walls_euler_(int(scene.wallNum[0]), sims.dt, scene.wall)

    def patch_integration(self, sims: Simulation, scene: myScene):
        scene.geometry.go(sims.dt, sims.delta, scene.wall)

    def verlet_sphere_integration(self, sims: Simulation, scene: myScene):
        move_spheres_verlet_predictor_(
            int(scene.sphereNum[0]), sims.dt, scene.sphere, scene.particle, scene.material, sims.gravity
        )

    def verlet_clump_integration(self, sims: Simulation, scene: myScene):
        move_clumps_verlet_predictor_(
            int(scene.clumpNum[0]), sims.dt, scene.clump, scene.particle, scene.material, sims.gravity
        )

    def verlet_level_set_integration(self, sims: Simulation, scene: myScene):
        move_level_set_verlet_predictor_(
            int(scene.particleNum[0]), sims.dt, scene.particle, scene.rigid, scene.material, sims.gravity
        )

    def compute_aratio(self, sims: Simulation, scene: myScene, neighbor: NeighborBase):
        currF = 0.0
        if int(scene.sphereNum[0]) > 0:
            currF += calculate_sphere_total_unbalance_force(
                sims.gravity, int(scene.sphereNum[0]), scene.sphere, scene.particle
            )
        if int(scene.clumpNum[0]) > 0:
            currF += calculate_clump_total_unbalance_force(
                sims.gravity, int(scene.clumpNum[0]), scene.clump, scene.particle
            )

        contactF, contactC = calculate_total_contact_force(
            int(scene.particleNum[0]), neighbor.particle_particle, self.physpp.cplist
        )
        if sims.max_wall_num > 0:
            contactFw, contactCw = calculate_total_contact_force(
                int(scene.particleNum[0]), neighbor.particle_wall, self.physpw.cplist
            )
            contactF += contactFw
            contactC += contactCw
        return currF * contactC / (contactF * (int(scene.sphereNum[0]) + int(scene.clumpNum[0])))

    def compute_mratio(self, sims: Simulation, scene: myScene, neighbor: NeighborBase):
        currF = 0.0
        if int(scene.sphereNum[0]) > 0:
            currF += calculate_sphere_maximum_unbalance_force(
                sims.gravity, int(scene.sphereNum[0]), scene.sphere, scene.particle
            )
        if int(scene.clumpNum[0]) > 0:
            currF += calculate_clump_maximum_unbalance_force(
                sims.gravity, int(scene.clumpNum[0]), scene.clump, scene.particle
            )

        contactF, contactC = calculate_total_contact_force(
            int(scene.particleNum[0]), neighbor.particle_particle, self.physpp.cplist
        )
        if sims.max_wall_num > 0:
            contactFw, contactCw = calculate_total_contact_force(
                int(scene.particleNum[0]), neighbor.particle_wall, self.physpw.cplist
            )
            contactF += contactFw
            contactC += contactCw
        return currF * contactC / contactF

    def adaptive_timestep(self, sims: Simulation, scene: myScene):
        if sims.adaptive_timestep > 0 and sims.current_step % sims.adaptive_timestep == 0:
            min_dt = scene.adaptive_timestep(sims)
            sims.set_timestep(min(sims.init_delta, sims.CFL * min_dt))
