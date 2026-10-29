import numpy as np
import os, types

from src.mpm.generator.Generator import Generator
from src.mpm.generator.InsertionKernel import *
from src.mpm.Contact import DEMContact
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from src.utils.RegionFunction import RegionFunction
from src.utils.TypeDefination import vec2u8, vec3f
from src.utils.FieldIO import field_from_numpy_prefix, validate_numpy_prefix
import trimesh as tm


class BodyReader(Generator):
    sims: Simulation

    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.file_type = None
        self.check_history = False

    def no_print(self):
        self.log = False

    def deactivate(self):
        self.active = False

    def set_region(self, region):
        self.myRegion = region

    def get_region_ptr(self, name):
        if not self.myRegion is None:
            return self.myRegion[name]
        else:
            raise RuntimeError("Region class should be activated first!")

    def set_system_strcuture(self, body_dict):
        self.file_type = DictIO.GetEssential(body_dict, "FileType")
        self.myTemplate = DictIO.GetEssential(body_dict, "Template")
        period = DictIO.GetAlternative(body_dict, "Period", [self.sims.current_time, self.sims.current_time, 1e6])
        self.check_history = DictIO.GetAlternative(body_dict, "CheckHistory", False)
        self.start_time = period[0]
        self.end_time = period[1]
        self.insert_interval = period[2]

    def finalize(self):
        del self.active  # , self.field_builder, self.snode_tree
        del self.start_time, self.end_time, self.insert_interval, self.myTemplate

    def begin(self, scene: myScene):
        if self.sims.current_time < self.next_generate_time:
            return 0
        if self.sims.current_time < self.start_time or self.sims.current_time > self.end_time:
            return 0

        start_particle = int(scene.particleNum[0])
        if self.file_type == "TXT":
            if type(self.myTemplate) is dict:
                self.add_txt_body(scene, self.myTemplate)
            elif type(self.myTemplate) is list:
                for template in self.myTemplate:
                    self.add_txt_body(scene, template)
        elif self.file_type == "NPZ":
            if type(self.myTemplate) is dict:
                self.add_npz_body(scene, self.myTemplate)
            elif type(self.myTemplate) is list:
                for template in self.myTemplate:
                    self.add_npz_body(scene, template)
        elif self.file_type == "OBJ":
            if type(self.myTemplate) is dict:
                self.add_obj_body(scene, self.myTemplate)
            elif type(self.myTemplate) is list:
                for template in self.myTemplate:
                    self.add_obj_body(scene, template)
        else:
            raise RuntimeError("Invalid file type. Only the following file types are aviliable: ['TXT', 'NPZ', 'OBJ']")

        if not scene.particle is None:
            scene.material.update_material_mapping(scene.particle, int(scene.particleNum[0]))

        if (
            self.sims.current_time + self.insert_interval > self.end_time
            or self.insert_interval > self.sims.time
            or self.end_time == 0
            or self.start_time > self.end_time
        ):
            self.deactivate()
        else:
            self.next_generate_time = self.sims.current_time + self.insert_interval
        return 1

    def check_bodyID(self, scene: myScene, bodyID):
        if bodyID > scene.grid_level - 1:
            raise RuntimeError(f"Keyword:: /bodyID/ must be smaller than {scene.grid_level - 1}")

    def rotate_body(self, orientation, coords, init_particle_num, particle_num):
        if self.sims.dimension == 3:
            bounding_center = kernel_calc_mass_of_center_(coords)
            kernel_position_rotate_(orientation, bounding_center, coords, init_particle_num, particle_num)
        elif self.sims.dimension == 2:
            bounding_center = kernel_calc_mass_of_center_2D(coords)
            kernel_position_rotate_2D(orientation, bounding_center, coords, init_particle_num, particle_num)

    def set_element_calLength(self, scene: myScene, bodyID, psize):
        if self.sims.shape_function == "Linear":
            scene.element.calLength[bodyID] = [0, 0, 0]
        elif self.sims.shape_function == "QuadBSpline":
            scene.element.calLength[bodyID] = 0.5 * scene.element.grid_size
        elif self.sims.shape_function == "CubicBSpline":
            scene.element.calLength[bodyID] = scene.element.grid_size
        elif self.sims.shape_function == "GIMP":
            scene.element.calLength[bodyID] = psize

    def particle_cloud_filter(self, particle_cloud, region: RegionFunction):
        if region.region_type == "Rectangle":
            start_points = region.local_start_point
            size_points = region.local_region_size
            bounding_box = [
                start_points[0],
                start_points[0] + size_points[0],
                start_points[1],
                start_points[1] + size_points[1],
            ]
            particle_cloud = particle_cloud[
                np.logical_and(particle_cloud[:, 0] >= bounding_box[0], particle_cloud[:, 0] <= bounding_box[1])
            ]
            particle_cloud = particle_cloud[
                np.logical_and(particle_cloud[:, 1] >= bounding_box[2], particle_cloud[:, 1] <= bounding_box[3])
            ]
            if self.sims.dimension == 3:
                bounding_box += [start_points[2], start_points[2] + size_points[2]]
                particle_cloud = particle_cloud[
                    np.logical_and(particle_cloud[:, 2] >= bounding_box[4], particle_cloud[:, 2] <= bounding_box[5])
                ]
        return particle_cloud

    def add_txt_body(self, scene: myScene, template):
        particle_file = DictIO.GetAlternative(template, "ParticleFile", "Particle.txt")
        print("#", f"Start adding material points from {particle_file}......")
        if not os.path.exists(particle_file):
            raise EOFError("Invaild path")
        particle_cloud = np.loadtxt(particle_file, unpack=True, comments="#").transpose()

        bodyID = DictIO.GetEssential(template, "BodyID")
        self.check_bodyID(scene, bodyID)

        particle_number = DictIO.GetAlternative(template, "ParticleNumber", particle_cloud.shape[0])
        particle_count = particle_number if particle_number != -1 else particle_cloud.shape[0]
        particle_stress = DictIO.GetAlternative(template, "ParticleStress", {"InternalStress": [0, 0, 0, 0, 0, 0]})
        orientation = DictIO.GetAlternative(
            template, "Orientation", vec3f([0, 0, 0]) if self.sims.dimension == 3 else 0.0
        )
        init_v = DictIO.GetAlternative(
            template, "InitialVelocity", [0.0, 0.0, 0.0] if self.sims.dimension == 3 else [0.0, 0.0]
        )
        fix_v_str = DictIO.GetAlternative(
            template, "FixVelocity", ["Free", "Free", "Free"] if self.sims.dimension == 3 else ["Free", "Free"]
        )
        rigid_body = DictIO.GetAlternative(template, "RigidBody", False)
        if rigid_body:
            materialID = 0
            density = DictIO.GetAlternative(template, "Density", 2650)
            scene.is_rigid[bodyID] = 1
            scene.material.matProps[materialID].density = density
        else:
            materialID = DictIO.GetEssential(template, "MaterialID")
            if self.sims.random_field:
                scene.material.matProps[materialID].read_random_field(
                    int(scene.particleNum[0]), int(scene.particleNum[0]) + particle_count, scene.material.stateVars
                )
                density = np.ascontiguousarray(scene.material.stateVars.density.to_numpy())
            else:
                density = scene.material.matProps[materialID].density

            if self.sims.material_type == "TwoPhaseSingleLayer":
                if not self.sims.random_field:
                    density = scene.material.matProps[materialID].solid_density
                densityf = scene.material.matProps[materialID].fluid_density
                porosity = scene.material.matProps[materialID].porosity
                permeability = scene.material.matProps[materialID].permeability
            if materialID <= 0:
                raise RuntimeError(f"Material ID {materialID} should be larger than 0")

        if self.sims.dimension == 3:
            fix_v = vec3u8([DictIO.GetEssential(self.FIX, is_fix) for is_fix in fix_v_str])
            if isinstance(init_v, (list, tuple)):
                init_v = vec3f(init_v)
        elif self.sims.dimension == 2:
            fix_v = vec2u8([DictIO.GetEssential(self.FIX, is_fix) for is_fix in fix_v_str])
            if isinstance(init_v, (list, tuple)):
                init_v = vec2f(init_v)

        if isinstance(density, (np.ndarray, list, tuple)):
            density = np.asarray(density)
        elif isinstance(density, (int, float)):
            density = np.repeat(density, particle_count)

        init_particle_num = int(scene.particleNum[0])
        coords, volume = None, None
        if self.sims.dimension == 3:
            coords = np.ascontiguousarray(particle_cloud[0:particle_number, [0, 1, 2]])
            volume = np.ascontiguousarray(particle_cloud[0:particle_number, 3])
            psize = np.ascontiguousarray(particle_cloud[0:particle_number, [4, 5, 6]])
        elif self.sims.dimension == 2:
            coords = np.ascontiguousarray(particle_cloud[0:particle_number, [0, 1]])
            volume = np.ascontiguousarray(particle_cloud[0:particle_number, 2])
            psize = np.ascontiguousarray(particle_cloud[0:particle_number, [3, 4]])
        particle_num = coords.shape[0]

        self.rotate_body(orientation, coords, init_particle_num, particle_num)
        scene.check_particle_num(self.sims, particle_number=particle_num)
        if self.check_history:
            scene.delete_overlap_particles(coords, volume)
        if self.sims.dimension == 3:
            self.set_element_calLength(scene, bodyID, vec3f(psize[0, 0], psize[0, 1], psize[0, 2]))
            kernel_read_particle_file_(
                scene.particle,
                int(scene.particleNum[0]),
                particle_num,
                coords,
                volume,
                bodyID,
                materialID,
                density,
                init_v,
                fix_v,
            )
        elif self.sims.dimension == 2:
            self.set_element_calLength(scene, bodyID, vec2f(psize[0, 0], psize[0, 1]))
            kernel_read_particle_file_2D(
                scene.particle,
                int(scene.particleNum[0]),
                particle_num,
                coords,
                volume,
                bodyID,
                materialID,
                density,
                init_v,
                fix_v,
            )

        self.set_particle_stress(scene, init_particle_num, particle_num, particle_stress)
        scene.push_psize(psize)
        traction = DictIO.GetAlternative(template, "Traction", {})
        self.set_traction(particle_num, traction, scene)
        scene.material.state_vars_initialize(
            materialID, int(scene.particleNum[0]), int(scene.particleNum[0]) + particle_num, scene.particle
        )
        scene.particleNum[0] += particle_num
        self.print_particle_info(bodyID, materialID, init_v, fix_v, particle_num)

    def _read_incompressible_restart_pressure(self, scene, particle_file, particle_info):
        grid_file = os.path.join(
            os.path.dirname(os.path.dirname(particle_file)),
            "grids",
            os.path.basename(particle_file).replace("MPMParticle", "MPMGrid", 1),
        )
        if "restart_cell_pressure" not in particle_info.files and not os.path.isfile(grid_file):
            return None
        shape = tuple(int(n) for n in scene.element.cnum)
        spacing = np.asarray(scene.element.grid_size, dtype=float)
        ghost_cell = int(scene.element.ghost_cell)
        pressure = None
        if "restart_cell_pressure" in particle_info.files:
            pressure = np.asarray(particle_info["restart_cell_pressure"], dtype=float)
            if (
                not np.allclose(particle_info["restart_grid_size"], spacing, rtol=1e-7, atol=1e-12)
                or int(particle_info["restart_ghost_cell"]) != ghost_cell
            ):
                raise ValueError("Incompressible restart grid geometry does not match the configured grid")
        else:
            # Older particle files can recover the exact field from their matching grid output.
            if os.path.isfile(grid_file):
                with np.load(grid_file, allow_pickle=False) as grid_info:
                    if "cell_pressure" in grid_info.files:
                        if not np.isclose(grid_info["t_current"], particle_info["t_current"], rtol=1e-10, atol=1e-12):
                            raise ValueError("Incompressible restart particle and grid times do not match")
                        pressure = np.asarray(grid_info["cell_pressure"], dtype=float)
                        if "coords" in grid_info.files:
                            coords = grid_info["coords"]
                            if not np.allclose(
                                coords.min(axis=0), -ghost_cell * spacing, rtol=1e-7, atol=1e-12
                            ) or not np.allclose(
                                coords.max(axis=0), (np.asarray(shape) - ghost_cell) * spacing, rtol=1e-7, atol=1e-12
                            ):
                                raise ValueError(
                                    "Incompressible restart grid geometry does not match the configured grid"
                                )
        if pressure is not None:
            if pressure.shape == shape + (1,):
                pressure = pressure[..., 0]
            if pressure.shape != shape or not np.isfinite(pressure).all():
                raise ValueError(
                    "Incompressible restart cell pressure must be finite and match the configured grid shape"
                )
            pressure = np.ascontiguousarray(pressure)
        return pressure

    def _restore_two_phase(self, scene, data, template):
        count, dim = int(data["body_num"]), self.sims.dimension
        if count < 0 or count > self.sims.max_particle_num:
            raise ValueError("Two-phase restart particle count exceeds the allocated capacity")
        if "dimension" in data and int(data["dimension"]) != dim:
            raise ValueError("Two-phase restart dimension does not match the configured model")
        if "material_type" in data and str(data["material_type"]) != self.sims.material_type:
            raise ValueError("Two-phase restart material type does not match the configured model")
        if float(data["body_num"]) != count:
            raise ValueError("Restart particle count must be an integer")
        if "restart_format_version" in data and int(data["restart_format_version"]) != 1:
            raise ValueError("Unsupported two-phase restart format version")
        phase = np.asarray(data.get("phase", np.zeros(count, dtype=np.uint8)))
        if self.sims.material_type == "TwoPhaseDoubleLayer" and np.any(~np.isin(phase, [1, 2])):
            raise ValueError("Double-layer restart requires valid saved phase labels")
        if self.sims.material_type == "TwoPhaseSingleLayer" and np.any(phase != 0):
            raise ValueError("A double-layer phase checkpoint cannot restart a single-layer model")
        sizes = data["psize"] if "psize" in data else template.get("ParticleSize")
        if sizes is None:
            raise ValueError("Legacy two-phase restart lacks psize; supply particle_size from the original generation")
        sizes = np.asarray(sizes, dtype=float)
        if sizes.shape == (dim,):
            sizes = np.broadcast_to(sizes, (count, dim)).copy()
        if sizes.shape != (count, dim) or not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError("Restart particle_size must be positive with shape (dimension,) or (particles, dimension)")
        saved_time = float(data["t_current"])
        if not np.isfinite(saved_time) or saved_time < 0:
            raise ValueError("Restart time must be finite and non-negative")
        step = data["current_step"] if "current_step" in data else template.get("CurrentStep")
        if self.sims.is_continue:
            if step is None:
                if saved_time == 0:
                    step = 0
                else:
                    raise ValueError(
                        "Legacy two-phase restart lacks current_step; supply the original saved step count"
                    )
            if not np.isfinite(step) or int(step) != step or int(step) < 0:
                raise ValueError("Restart current_step must be a non-negative integer")

        fields = {
            "particleID": "particleID",
            "bodyID": "bodyID",
            "materialID": "materialID",
            "active": "active",
            "m": "mass",
            "ms": "solid_mass",
            "mf": "fluid_mass",
            "vol": "volume",
            "x": "position",
            "v": "velocity",
            "vs": "solid_velocity",
            "vf": "fluid_velocity",
            "stress": "stress",
            "pressure": "pressure",
            "porosity": "porosity",
            "permeability": "permeability",
            "solid_velocity_gradient": "solid_velocity_gradient",
            "fluid_velocity_gradient": "fluid_velocity_gradient",
            "fix_v": "fix_v",
        }
        restores = []

        def prepare(field, value, name, entries=count):
            value = validate_numpy_prefix(field, value)
            if value.shape[0] != entries:
                raise ValueError(f"Restart field {name} has the wrong particle count")
            restores.append((field, value))

        for field, key in fields.items():
            if key not in data:
                raise ValueError(f"Two-phase restart lacks {key}")
            prepare(getattr(scene.particle, field), data[key], key)
        if self.sims.material_type == "TwoPhaseDoubleLayer" and "phase" not in data:
            raise ValueError("Double-layer restart requires each particle's phase")
        for name in (
            "phase",
            "coupling",
            "free_surface",
            "pressure_gradient",
            "normal",
            "rad",
            "mass_density",
            "verletDisp",
            "external_force",
        ):
            if hasattr(scene.particle, name):
                field = getattr(scene.particle, name)
                components = () if not hasattr(field, "n") else (field.n,)
                value = data[name] if name in data else np.zeros((count, *components))
                if name == "coupling" and name not in data and self.sims.coupling:
                    value = np.asarray(phase == 1, dtype=np.uint8)
                prepare(field, value, name)
        if np.any(data["bodyID"] >= scene.is_rigid.shape[0]) or np.any(data["bodyID"] < 0):
            raise ValueError("Restart body IDs exceed the allocated bodies")
        if scene.material.stateVars is not None:
            history_key = (
                "constitutive_state"
                if "constitutive_state" in data
                else ("restart_state_vars" if "restart_state_vars" in data else "state_vars")
            )
            history = data[history_key].item() if history_key in data else {}
            for name, field in scene.material.stateVars.field_dict.items():
                if name not in history:
                    raise ValueError(f"Two-phase restart lacks constitutive history {name}")
                value = np.asarray(history[name])
                # Older visualization snapshots expanded six-component symmetric tensors.
                if getattr(field, "n", None) == 6 and getattr(field, "m", None) == 1 and value.shape == (count, 3, 3):
                    value = value[:, [0, 1, 2, 0, 1, 0], [0, 1, 2, 1, 2, 2]]
                prepare(field, value, "state_vars." + name)

        for name in ("mass", "solid_mass", "fluid_mass"):
            if np.any(data[name] < 0):
                raise ValueError("Restart particle masses must be nonnegative")
        if np.any(data["volume"] <= 0) or np.any((data["porosity"] < 0) | (data["porosity"] > 1)):
            raise ValueError("Invalid restart particle volume or porosity")
        if hasattr(scene.material, "matProps") and np.any(data["materialID"] >= scene.material.matProps.size()):
            raise ValueError("Restart material IDs exceed the configured material table")
        initialized = data.get(
            "two_phase_fluid_mass_initialized",
            data.get("fluid_mass_initialized", template.get("FluidMassInitialized", saved_time > 0)),
        )
        if initialized not in (False, True):
            raise ValueError("Invalid fluid mass initialization flag")
        traction_count = None
        traction_key = "particle_traction_state" if "particle_traction_state" in data else "restart_particle_traction"
        if traction_key in data:
            if template.get("Traction"):
                raise ValueError("Saved particle tractions cannot be combined with new restart tractions")
            traction = data[traction_key].item()
            traction_count = len(traction["pid"]) if traction else 0
            if traction_count:
                target = scene.boundary.particle_traction
                if target is None or not hasattr(target, "field_dict"):
                    raise ValueError("Restart requires allocated particle traction constraints")
                if np.any(traction["pid"] < 0) or np.any(traction["pid"] >= count):
                    raise ValueError("Restart traction IDs are out of range")
                for name, field in target.field_dict.items():
                    if name not in traction:
                        raise ValueError(f"Restart lacks particle traction history {name}")
                    prepare(field, traction[name], "particle_traction." + name, traction_count)
        elif saved_time > 0 and scene.boundary.ptraction_list[0] > 0 and self.sims.particle_traction_update_area:
            raise ValueError("Legacy restart lacks evolving particle traction areas; use a new checkpoint")

        restart_pressure = self._read_incompressible_restart_pressure(scene, template["File"], data)
        if saved_time == 0.0 and (step is None or step == 0):
            # The initial output precedes projection; IBM must seed pressure
            # from the restored particles, as in an uninterrupted fresh run.
            restart_pressure = None

        # Validate every required field before changing the live particle state or clock.
        for field, value in restores:
            field_from_numpy_prefix(field, value)
        scene.is_rigid.fill(0)
        for body in np.unique(data["bodyID"][data["materialID"] == 0]):
            scene.is_rigid[int(body)] = 1
        scene.psize = sizes.copy()
        scene.particleNum[0] = count
        if traction_count is not None:
            scene.boundary.ptraction_list[0] = traction_count
        scene.two_phase_fluid_mass_initialized = bool(initialized)
        scene.restart_cell_pressure = restart_pressure
        if self.sims.is_continue:
            self.sims.current_time = saved_time
            self.sims.CurrentTime[None] = saved_time
            self.sims.current_step = int(step)
        self.set_traction(count, DictIO.GetAlternative(template, "Traction", {}), scene)
        print("Insert particle Number: ", count)

    def add_npz_body(self, scene: myScene, template):
        particle_file = DictIO.GetEssential(template, "File")
        if not os.path.exists(particle_file):
            raise EOFError("Invaild path")

        if DictIO.GetAlternative(template, "Restart", False):
            particle_info = np.load(particle_file, allow_pickle=True)
            if self.sims.material_type in ("TwoPhaseSingleLayer", "TwoPhaseDoubleLayer"):
                try:
                    self._restore_two_phase(scene, particle_info, template)
                finally:
                    particle_info.close()
                return
            is_incompressible = "pressure" in particle_info.files and "stress" not in particle_info.files
            restart_pressure = None
            if is_incompressible:
                restart_pressure = self._read_incompressible_restart_pressure(scene, particle_file, particle_info)
            if self.sims.is_continue:
                self.sims.current_time = DictIO.GetEssential(particle_info, "t_current")
                self.sims.CurrentTime[None] = DictIO.GetEssential(particle_info, "t_current")
            particle_number = int(DictIO.GetEssential(particle_info, "body_num"))

            if particle_number > self.sims.max_particle_num:
                raise RuntimeError("/max_particle_number/ should be enlarged")
            is_incompressible = "pressure" in particle_info.files and "stress" not in particle_info.files
            if is_incompressible:
                coupling = (
                    DictIO.GetEssential(particle_info, "coupling")
                    if "coupling" in particle_info.files
                    else np.ones(particle_number, dtype=np.uint8)
                )
                velocity_gradient = (
                    DictIO.GetEssential(particle_info, "velocity_gradient")
                    if "velocity_gradient" in particle_info.files
                    else np.zeros((particle_number, self.sims.dimension, self.sims.dimension), dtype=float)
                )
                if self.sims.dimension == 3:
                    kernel_rebulid_incompressible_particle(
                        particle_number,
                        scene.particle,
                        scene.is_rigid,
                        DictIO.GetEssential(particle_info, "particleID"),
                        DictIO.GetEssential(particle_info, "bodyID"),
                        DictIO.GetEssential(particle_info, "materialID"),
                        DictIO.GetEssential(particle_info, "active"),
                        coupling,
                        DictIO.GetEssential(particle_info, "mass"),
                        DictIO.GetEssential(particle_info, "position"),
                        DictIO.GetEssential(particle_info, "velocity"),
                        DictIO.GetEssential(particle_info, "volume"),
                        -DictIO.GetEssential(particle_info, "pressure"),
                        velocity_gradient,
                        DictIO.GetEssential(particle_info, "fix_v"),
                    )
                elif self.sims.dimension == 2:
                    kernel_rebulid_incompressible_particle_2D(
                        particle_number,
                        scene.particle,
                        scene.is_rigid,
                        DictIO.GetEssential(particle_info, "particleID"),
                        DictIO.GetEssential(particle_info, "bodyID"),
                        DictIO.GetEssential(particle_info, "materialID"),
                        DictIO.GetEssential(particle_info, "active"),
                        coupling,
                        DictIO.GetEssential(particle_info, "mass"),
                        DictIO.GetEssential(particle_info, "position"),
                        DictIO.GetEssential(particle_info, "velocity"),
                        DictIO.GetEssential(particle_info, "volume"),
                        -DictIO.GetEssential(particle_info, "pressure"),
                        velocity_gradient,
                        DictIO.GetEssential(particle_info, "fix_v"),
                    )
                scene.restart_cell_pressure = restart_pressure
            else:
                kernel_rebulid_particle(
                    particle_number,
                    scene.particle,
                    scene.is_rigid,
                    DictIO.GetEssential(particle_info, "particleID"),
                    DictIO.GetEssential(particle_info, "bodyID"),
                    DictIO.GetEssential(particle_info, "materialID"),
                    DictIO.GetEssential(particle_info, "active"),
                    DictIO.GetEssential(particle_info, "mass"),
                    DictIO.GetEssential(particle_info, "position"),
                    DictIO.GetEssential(particle_info, "velocity"),
                    DictIO.GetEssential(particle_info, "volume"),
                    DictIO.GetEssential(particle_info, "stress"),
                    DictIO.GetEssential(particle_info, "velocity_gradient"),
                    DictIO.GetEssential(particle_info, "fix_v"),
                )

            if self.sims.coupling and not is_incompressible:
                kernel_rebulid_particle_coupling(
                    particle_number,
                    scene.particle,
                    DictIO.GetEssential(particle_info, "coupling"),
                    DictIO.GetEssential(particle_info, "radius"),
                )
            if self.sims.neighbor_detection and not is_incompressible:
                kernel_rebulid_particle_neighbor_detection(
                    particle_number,
                    scene.particle,
                    DictIO.GetEssential(particle_info, "free_surface"),
                    DictIO.GetEssential(particle_info, "mass_density"),
                    DictIO.GetEssential(particle_info, "normal"),
                )

            psize = DictIO.GetEssential(particle_info, "psize")
            scene.push_psize(psize)
            traction = DictIO.GetAlternative(template, "Traction", {})
            self.set_traction(particle_number, traction, scene)
            if "state_vars" in particle_info.files:
                scene.material.reload_state_variables(DictIO.GetEssential(particle_info, "state_vars"))
            scene.particleNum[0] = particle_number
            print("Insert particle Number: ", particle_number)
        else:
            bodyID = DictIO.GetEssential(template, "BodyID")
            self.check_bodyID(scene, bodyID)
            rigid_body = DictIO.GetAlternative(template, "RigidBody", False)
            if rigid_body:
                materialID = 0
                density = DictIO.GetAlternative(template, "Density", 2650)
                scene.is_rigid[bodyID] = 1
            else:
                materialID = DictIO.GetEssential(template, "MaterialID")
                density = scene.material.matProps[materialID].density

                if materialID <= 0:
                    raise RuntimeError(f"Material ID {materialID} should be larger than 0")

            particle_stress = DictIO.GetAlternative(template, "ParticleStress", {"InternalStress": [0, 0, 0, 0, 0, 0]})
            orientation = DictIO.GetAlternative(
                template, "Orientation", vec3f([0, 0, 0]) if self.sims.dimension == 3 else 0.0
            )
            init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
            fix_v_str = DictIO.GetAlternative(template, "FixVelocity", ["Free", "Free", "Free"])
            fix_v = vec3u8([DictIO.GetEssential(self.FIX, is_fix) for is_fix in fix_v_str])
            init_particle_num = int(scene.particleNum[0])

            particle_cloud = np.load(particle_file, allow_pickle=True)
            coords = DictIO.GetEssential(particle_cloud, "position")
            psize = DictIO.GetEssential(particle_cloud, "psize")
            volume = DictIO.GetEssential(particle_cloud, "volume")
            particle_num = coords.shape[0]

            if isinstance(density, (np.ndarray, list, tuple)):
                density = np.asarray(density)
            elif isinstance(density, (int, float)):
                density = np.repeat(density, particle_num)

            self.rotate_body(orientation, coords, init_particle_num, particle_num)
            scene.check_particle_num(self.sims, particle_number=particle_num)
            kernel_read_particle_file_(
                scene.particle,
                int(scene.particleNum[0]),
                particle_num,
                coords,
                volume,
                bodyID,
                materialID,
                density,
                init_v,
                fix_v,
            )
            self.set_particle_stress(scene, init_particle_num, particle_num, particle_stress)
            scene.push_psize(psize)
            traction = DictIO.GetAlternative(template, "Traction", {})
            self.set_traction(particle_num, traction, scene)
            stateVars = DictIO.GetAlternative(particle_info, "state_vars", None)
            if not stateVars is None:
                scene.material.reload_state_variables(stateVars)
            else:
                scene.material.state_vars_initialize(
                    materialID, int(scene.particleNum[0]), int(scene.particleNum[0]) + particle_num, scene.particle
                )
            scene.particleNum[0] += particle_num
            self.print_particle_info(bodyID, materialID, init_v, fix_v, particle_num)

    def add_obj_body(self, scene: myScene, template):
        if self.sims.dimension == 2:
            raise RuntimeError("2D conditions do not support voxelization technique")
        particle_file = DictIO.GetAlternative(template, "ParticleFile", "Particle.obj")
        if not os.path.exists(particle_file):
            raise EOFError("Invaild path")

        bodyID = DictIO.GetEssential(template, "BodyID")
        self.check_bodyID(scene, bodyID)
        rigid_body = DictIO.GetAlternative(template, "RigidBody", False)
        if rigid_body:
            materialID = 0
            density = DictIO.GetAlternative(template, "Density", 2650)
            scene.is_rigid[bodyID] = 1
        else:
            materialID = DictIO.GetEssential(template, "MaterialID")
            density = scene.material.matProps[materialID].density

            if materialID <= 0:
                raise RuntimeError(f"Material ID {materialID} should be larger than 0")

        particle_stress = DictIO.GetAlternative(template, "ParticleStress", {"InternalStress": [0, 0, 0, 0, 0, 0]})
        traction = DictIO.GetAlternative(template, "Traction", {})
        init_v = DictIO.GetAlternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        fix_v_str = DictIO.GetAlternative(template, "FixVelocity", ["Free", "Free", "Free"])
        fix_v = vec3u8([DictIO.GetEssential(self.FIX, is_fix) for is_fix in fix_v_str])
        init_particle_num = int(scene.particleNum[0])

        scale_factor = DictIO.GetAlternative(template, "ScaleFactor", default=1.0)
        offset = np.array(DictIO.GetAlternative(template, "Offset", [0.0, 0.0, 0.0]))
        orientation = DictIO.GetAlternative(
            template, "Orientation", vec3f([0, 0, 0]) if self.sims.dimension == 3 else 0.0
        )
        nParticlesPerCell = DictIO.GetAlternative(template, "nParticlesPerCell", 2)

        mesh = tm.load(particle_file)
        com = mesh.vertices.mean(axis=0)
        mesh.apply_translation(offset - com)
        mesh.apply_scale(scale_factor)
        cell_size = scene.element.grid_size
        diameter = min(cell_size[0], cell_size[1], cell_size[2]) / nParticlesPerCell
        voxelized_mesh = mesh.voxelized(pitch=diameter).fill()
        voxelized_points_np = voxelized_mesh.points.copy()
        particle_num = voxelized_points_np.shape[0]
        particle_volume = 4.0 / 3.0 * np.pi * (0.5 * diameter) ** 3

        if isinstance(density, (np.ndarray, list, tuple)):
            density = np.asarray(density)
        elif isinstance(density, (int, float)):
            density = np.repeat(density, particle_num)

        psize = np.repeat(0.5 * diameter, particle_num * 3).reshape((particle_num, 3))
        volume = np.repeat(particle_volume, particle_num)
        self.rotate_body(orientation, voxelized_points_np, init_particle_num, particle_num)
        scene.check_particle_num(self.sims, particle_number=particle_num)
        kernel_read_particle_file_(
            scene.particle,
            int(scene.particleNum[0]),
            particle_num,
            voxelized_points_np,
            volume,
            bodyID,
            materialID,
            density,
            init_v,
            fix_v,
        )

        self.set_particle_stress(scene, init_particle_num, particle_num, particle_stress)
        scene.push_psize(psize)
        traction = DictIO.GetAlternative(template, "Traction", {})
        self.set_traction(particle_num, traction, scene)
        scene.material.state_vars_initialize(
            materialID, int(scene.particleNum[0]), int(scene.particleNum[0]) + particle_num, scene.particle
        )
        scene.particleNum[0] += particle_num
        self.print_particle_info(
            bodyID,
            materialID,
            init_v,
            fix_v,
            particle_num,
            particle_volume=particle_volume,
            nParticlesPerCell=nParticlesPerCell,
        )

    def set_polygons(self, contact: DEMContact, body_dict):
        vertices = DictIO.GetEssential(body_dict, "Vertices")
        if isinstance(vertices, str):
            print("#", f"Reading {vertices}......")
            if not os.path.exists(vertices):
                raise EOFError("Invaild path")
            vertice = np.loadtxt(vertices)[:, 0 : self.sims.dimension]
        elif isinstance(vertices, (list, tuple, np.ndarray)):
            vertice = np.array(vertices)

        velocity = DictIO.GetAlternative(body_dict, "InitialVelocity", [0.0, 0.0])
        contact.generate_vertice_field(self.sims.dimension, vertice.shape[0])
        contact.polygon_vertices.from_numpy(vertice)
        contact.velocity = ti.Vector(np.array(velocity))
