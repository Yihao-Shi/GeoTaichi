from src.mpm.Contact import DEMContact
from src.utils.errors import ConfigurationError, SimulationStateError
from contextlib import contextmanager
import numpy as np
import os, types

from src.mpm.generator.Generator import Generator
from src.mpm.generator.InsertionKernel import *
from src.mpm.SceneManager import SceneManager
from src.mpm.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from src.utils.RegionFunction import RegionFunction
from src.utils.TypeDefinition import vec2u8, vec3f
from src.utils.FieldIO import field_from_numpy_prefix, field_to_numpy_prefix, validate_numpy_prefix
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
            raise SimulationStateError("Region class should be activated first!")

    def set_system_strcuture(self, body_dict):
        self.file_type = DictIO.get_essential(body_dict, "FileType")
        self.myTemplate = DictIO.get_essential(body_dict, "Template")
        period = DictIO.get_alternative(body_dict, "Period", [self.sims.current_time, self.sims.current_time, 1e6])
        self.check_history = DictIO.get_alternative(body_dict, "CheckHistory", False)
        self.start_time = period[0]
        self.end_time = period[1]
        self.insert_interval = period[2]

    def finalize(self):
        del self.active  # , self.field_builder, self.snode_tree
        del self.start_time, self.end_time, self.insert_interval, self.myTemplate

    def begin(self, scene: SceneManager):
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
            raise SimulationStateError(
                "Invalid file type. Only the following file types are aviliable: ['TXT', 'NPZ', 'OBJ']"
            )

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

    def check_body_id(self, scene: SceneManager, bodyID):
        if bodyID > scene.grid_level - 1:
            raise ConfigurationError(f"Keyword:: /bodyID/ must be smaller than {scene.grid_level - 1}")

    def rotate_body(self, orientation, coords, init_particle_num, particle_num):
        if self.sims.dimension == 3:
            bounding_center = kernel_calc_mass_of_center(coords)
            kernel_position_rotate(orientation, bounding_center, coords, init_particle_num, particle_num)
        elif self.sims.dimension == 2:
            bounding_center = kernel_calc_mass_of_center_2d(coords)
            kernel_position_rotate_2d(orientation, bounding_center, coords, init_particle_num, particle_num)

    def set_element_cal_length(self, scene: SceneManager, bodyID, psize):
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

    def add_txt_body(self, scene: SceneManager, template):
        particle_file = DictIO.get_alternative(template, "ParticleFile", "Particle.txt")
        print("#", f"Start adding material points from {particle_file}......")
        if not os.path.exists(particle_file):
            raise EOFError("Invaild path")
        particle_cloud = np.loadtxt(particle_file, unpack=True, comments="#").transpose()

        body_id = DictIO.get_essential(template, "BodyID")
        self.check_body_id(scene, body_id)

        particle_number = DictIO.get_alternative(template, "ParticleNumber", particle_cloud.shape[0])
        particle_count = particle_number if particle_number != -1 else particle_cloud.shape[0]
        particle_stress = DictIO.get_alternative(template, "ParticleStress", {"InternalStress": [0, 0, 0, 0, 0, 0]})
        orientation = DictIO.get_alternative(
            template, "Orientation", vec3f([0, 0, 0]) if self.sims.dimension == 3 else 0.0
        )
        init_v = DictIO.get_alternative(
            template, "InitialVelocity", [0.0, 0.0, 0.0] if self.sims.dimension == 3 else [0.0, 0.0]
        )
        fix_v_str = DictIO.get_alternative(
            template, "FixVelocity", ["Free", "Free", "Free"] if self.sims.dimension == 3 else ["Free", "Free"]
        )
        rigid_body = DictIO.get_alternative(template, "RigidBody", False)
        if rigid_body:
            material_id = 0
            density = DictIO.get_alternative(template, "Density", 2650)
            scene.is_rigid[body_id] = 1
            scene.material.matProps[material_id].density = density
        else:
            material_id = DictIO.get_essential(template, "MaterialID")
            if self.sims.random_field:
                scene.material.matProps[material_id].read_random_field(
                    int(scene.particleNum[0]), int(scene.particleNum[0]) + particle_count, scene.material.stateVars
                )
                density = np.ascontiguousarray(scene.material.stateVars.density.to_numpy())
            else:
                density = scene.material.matProps[material_id].density

            if self.sims.material_type == "TwoPhaseSingleLayer":
                if not self.sims.random_field:
                    density = scene.material.matProps[material_id].solid_density
                densityf = scene.material.matProps[material_id].fluid_density
                porosity = scene.material.matProps[material_id].porosity
                permeability = scene.material.matProps[material_id].permeability
            if material_id <= 0:
                raise SimulationStateError(f"Material ID {material_id} should be larger than 0")

        if self.sims.dimension == 3:
            fix_v = vec3u8([DictIO.get_essential(self.FIX, is_fix) for is_fix in fix_v_str])
            if isinstance(init_v, (list, tuple)):
                init_v = vec3f(init_v)
        elif self.sims.dimension == 2:
            fix_v = vec2u8([DictIO.get_essential(self.FIX, is_fix) for is_fix in fix_v_str])
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
            self.set_element_cal_length(scene, body_id, vec3f(psize[0, 0], psize[0, 1], psize[0, 2]))
            kernel_read_particle_file(
                scene.particle,
                int(scene.particleNum[0]),
                particle_num,
                coords,
                volume,
                body_id,
                material_id,
                density,
                init_v,
                fix_v,
            )
        elif self.sims.dimension == 2:
            self.set_element_cal_length(scene, body_id, vec2f(psize[0, 0], psize[0, 1]))
            kernel_read_particle_file_2d(
                scene.particle,
                int(scene.particleNum[0]),
                particle_num,
                coords,
                volume,
                body_id,
                material_id,
                density,
                init_v,
                fix_v,
            )

        self.set_particle_stress(scene, init_particle_num, particle_num, particle_stress)
        scene.push_psize(psize)
        traction = DictIO.get_alternative(template, "Traction", {})
        self.set_traction(particle_num, traction, scene)
        scene.material.state_vars_initialize(
            material_id, int(scene.particleNum[0]), int(scene.particleNum[0]) + particle_num, scene.particle
        )
        scene.particleNum[0] += particle_num
        self.print_particle_info(body_id, material_id, init_v, fix_v, particle_num)

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

    @contextmanager
    def restart_transaction(self, scene, fields, tractions):
        """Restore the prior scene if committing a validated checkpoint fails."""
        backups = [(field, field_to_numpy_prefix(field, count)) for field, count in fields]
        backups.append((scene.is_rigid, scene.is_rigid.to_numpy()))
        boundary = scene.boundary
        if tractions and boundary.particle_traction is not None:
            backups += [(field, field.to_numpy()) for field in boundary.particle_traction.field_dict.values()]
        old_sizes, old_count = scene.psize, int(scene.particleNum[0])
        old_pressure = scene.restart_cell_pressure
        old_initialized = scene.two_phase_fluid_mass_initialized
        old_time, old_step, old_clock = self.sims.current_time, self.sims.current_step, self.sims.CurrentTime[None]
        old_essentials = (boundary.is_rigid, boundary.psize, boundary.myRegion)
        old_traction_count = boundary.ptraction_list.copy()
        try:
            yield
        except BaseException:
            for field, previous in backups:
                field_from_numpy_prefix(field, previous)
            scene.psize, scene.particleNum[0] = old_sizes, old_count
            scene.restart_cell_pressure = old_pressure
            scene.two_phase_fluid_mass_initialized = old_initialized
            boundary.is_rigid, boundary.psize, boundary.myRegion = old_essentials
            boundary.ptraction_list[:] = old_traction_count
            self.sims.current_time, self.sims.current_step = old_time, old_step
            self.sims.CurrentTime[None] = old_clock
            raise

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

        tractions = DictIO.get_alternative(template, "Traction", {})
        with self.restart_transaction(scene, [(field, len(value)) for field, value in restores], tractions):
            for field, value in restores:
                field_from_numpy_prefix(field, value)
            scene.is_rigid.fill(0)
            for body in np.unique(data["bodyID"][data["materialID"] == 0]):
                scene.is_rigid[int(body)] = 1
            scene.psize = sizes.copy()
            if traction_count is not None:
                scene.boundary.ptraction_list[0] = traction_count
            scene.two_phase_fluid_mass_initialized = bool(initialized)
            scene.restart_cell_pressure = restart_pressure
            scene.particleNum[0] = 0
            self.set_traction(count, tractions, scene)
            scene.particleNum[0] = count
            if self.sims.is_continue:
                self.sims.current_time = saved_time
                self.sims.CurrentTime[None] = saved_time
                self.sims.current_step = int(step)
        print("Insert particle Number: ", count)

    def _restore_single_phase(self, scene, data, template):
        count = int(data["body_num"])
        if float(data["body_num"]) != count or not 0 <= count <= self.sims.max_particle_num:
            raise ValueError("Restart particle count must be an integer within the allocated capacity")
        saved_time = float(data["t_current"])
        if not np.isfinite(saved_time) or saved_time < 0:
            raise ValueError("Restart time must be finite and non-negative")
        dim = self.sims.dimension
        sizes = np.asarray(data["psize"], dtype=float)
        if sizes.shape == (dim,):
            sizes = np.broadcast_to(sizes, (count, dim)).copy()
        if sizes.shape != (count, dim) or not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError("Restart psize must be positive with shape (particles, dimension)")
        incompressible = "pressure" in data and "stress" not in data
        fields = {
            "particleID": "particleID",
            "bodyID": "bodyID",
            "materialID": "materialID",
            "active": "active",
            "mass": "m",
            "position": "x",
            "velocity": "v",
            "volume": "vol",
            "fix_v": "fix_v",
            "velocity_gradient": "velocity_gradient",
        }
        if incompressible:
            fields.update(pressure="pressure", coupling="coupling")
        else:
            fields["stress"] = "stress"
            if self.sims.coupling:
                fields.update(coupling="coupling", radius="rad")
            if self.sims.neighbor_detection:
                fields.update(free_surface="free_surface", mass_density="mass_density", normal="normal")
        values = {}
        for key, name in fields.items():
            if incompressible and key == "coupling" and key not in data:
                value = np.ones(count, dtype=np.uint8)
            elif incompressible and key == "velocity_gradient" and key not in data:
                value = np.zeros((count, dim, dim))
            else:
                value = data[key]
            value = validate_numpy_prefix(getattr(scene.particle, name), value)
            if value.shape[0] != count:
                raise ValueError(f"Restart field {key} has the wrong particle count")
            values[key] = value
        if np.any(values["bodyID"] >= scene.is_rigid.shape[0]):
            raise ValueError("Restart body IDs exceed the allocated bodies")
        if np.any(values["mass"] < 0) or np.any(values["volume"] <= 0):
            raise ValueError("Restart requires nonnegative masses and positive volumes")
        restart_pressure = (
            self._read_incompressible_restart_pressure(scene, template["File"], data) if incompressible else None
        )
        history_key = "constitutive_state" if "constitutive_state" in data else "state_vars"
        history_fields = scene.material.stateVars.field_dict if scene.material.stateVars is not None else {}
        history_uploads = []
        if history_key in data:
            history = data[history_key].item()
            if not isinstance(history, dict):
                raise TypeError("Constitutive restart history must be a dictionary")
            if history_key == "state_vars":
                history = {name: history[name] for name in history_fields if name in history}
            if history_fields or history:
                history_uploads = scene.material.prepare_state_variables(history, particle_count=count)
        elif history_fields:
            raise ValueError("Restart lacks constitutive history")

        # Bulk restore modifies particle prefixes; history uploads also reset their tails.
        restore_fields = [(field, count) for field in scene.particle.field_dict.values()]
        restore_fields += [(field, len(value)) for field, value in history_uploads]
        tractions = DictIO.get_alternative(template, "Traction", {})
        with self.restart_transaction(scene, restore_fields, tractions):
            scene.is_rigid.fill(0)
            if incompressible:
                rebuild = (
                    kernel_rebulid_incompressible_particle if dim == 3 else kernel_rebulid_incompressible_particle_2d
                )
                rebuild(
                    count,
                    scene.particle,
                    scene.is_rigid,
                    values["particleID"],
                    values["bodyID"],
                    values["materialID"],
                    values["active"],
                    values["coupling"],
                    values["mass"],
                    values["position"],
                    values["velocity"],
                    values["volume"],
                    -values["pressure"],
                    values["velocity_gradient"],
                    values["fix_v"],
                )
            else:
                kernel_rebulid_particle(
                    count,
                    scene.particle,
                    scene.is_rigid,
                    values["particleID"],
                    values["bodyID"],
                    values["materialID"],
                    values["active"],
                    values["mass"],
                    values["position"],
                    values["velocity"],
                    values["volume"],
                    values["stress"],
                    values["velocity_gradient"],
                    values["fix_v"],
                )
                if self.sims.coupling:
                    kernel_rebulid_particle_coupling(count, scene.particle, values["coupling"], values["radius"])
                if self.sims.neighbor_detection:
                    kernel_rebulid_particle_neighbor_detection(
                        count, scene.particle, values["free_surface"], values["mass_density"], values["normal"]
                    )
            for field, value in history_uploads:
                field.from_numpy(value)
            scene.psize = sizes.copy()
            scene.particleNum[0] = 0
            self.set_traction(count, tractions, scene)
            scene.particleNum[0] = count
            if incompressible:
                scene.restart_cell_pressure = restart_pressure
            if self.sims.is_continue:
                self.sims.current_time = saved_time
                self.sims.CurrentTime[None] = saved_time
        print("Insert particle Number: ", count)

    def add_npz_body(self, scene: SceneManager, template):
        particle_file = DictIO.get_essential(template, "File")
        if not os.path.exists(particle_file):
            raise EOFError("Invaild path")

        if DictIO.get_alternative(template, "Restart", False):
            with np.load(particle_file, allow_pickle=True) as particle_info:
                if self.sims.material_type in ("TwoPhaseSingleLayer", "TwoPhaseDoubleLayer"):
                    self._restore_two_phase(scene, particle_info, template)
                else:
                    self._restore_single_phase(scene, particle_info, template)
        else:
            body_id = DictIO.get_essential(template, "BodyID")
            self.check_body_id(scene, body_id)
            rigid_body = DictIO.get_alternative(template, "RigidBody", False)
            if rigid_body:
                material_id = 0
                density = DictIO.get_alternative(template, "Density", 2650)
                scene.is_rigid[body_id] = 1
            else:
                material_id = DictIO.get_essential(template, "MaterialID")
                density = scene.material.matProps[material_id].density

                if material_id <= 0:
                    raise SimulationStateError(f"Material ID {material_id} should be larger than 0")

            particle_stress = DictIO.get_alternative(template, "ParticleStress", {"InternalStress": [0, 0, 0, 0, 0, 0]})
            orientation = DictIO.get_alternative(
                template, "Orientation", vec3f([0, 0, 0]) if self.sims.dimension == 3 else 0.0
            )
            init_v = DictIO.get_alternative(template, "InitialVelocity", vec3f([0, 0, 0]))
            fix_v_str = DictIO.get_alternative(template, "FixVelocity", ["Free", "Free", "Free"])
            fix_v = vec3u8([DictIO.get_essential(self.FIX, is_fix) for is_fix in fix_v_str])
            init_particle_num = int(scene.particleNum[0])

            particle_cloud = np.load(particle_file, allow_pickle=True)
            coords = DictIO.get_essential(particle_cloud, "position")
            psize = DictIO.get_essential(particle_cloud, "psize")
            volume = DictIO.get_essential(particle_cloud, "volume")
            particle_num = coords.shape[0]

            if isinstance(density, (np.ndarray, list, tuple)):
                density = np.asarray(density)
            elif isinstance(density, (int, float)):
                density = np.repeat(density, particle_num)

            self.rotate_body(orientation, coords, init_particle_num, particle_num)
            scene.check_particle_num(self.sims, particle_number=particle_num)
            kernel_read_particle_file(
                scene.particle,
                int(scene.particleNum[0]),
                particle_num,
                coords,
                volume,
                body_id,
                material_id,
                density,
                init_v,
                fix_v,
            )
            self.set_particle_stress(scene, init_particle_num, particle_num, particle_stress)
            scene.push_psize(psize)
            traction = DictIO.get_alternative(template, "Traction", {})
            self.set_traction(particle_num, traction, scene)
            state_vars = DictIO.get_alternative(
                particle_cloud, "constitutive_state", DictIO.get_alternative(particle_cloud, "state_vars", None)
            )
            if state_vars is not None:
                history = state_vars.item()
                fields = scene.material.stateVars.field_dict if scene.material.stateVars is not None else {}
                if "constitutive_state" not in particle_cloud.files:
                    history = {name: history[name] for name in fields if name in history}
                scene.material.reload_state_variables(history, particle_num, init_particle_num)
            else:
                scene.material.state_vars_initialize(
                    material_id, int(scene.particleNum[0]), int(scene.particleNum[0]) + particle_num, scene.particle
                )
            scene.particleNum[0] += particle_num
            self.print_particle_info(body_id, material_id, init_v, fix_v, particle_num)

    def add_obj_body(self, scene: SceneManager, template):
        if self.sims.dimension == 2:
            raise SimulationStateError("2D conditions do not support voxelization technique")
        particle_file = DictIO.get_alternative(template, "ParticleFile", "Particle.obj")
        if not os.path.exists(particle_file):
            raise EOFError("Invaild path")

        body_id = DictIO.get_essential(template, "BodyID")
        self.check_body_id(scene, body_id)
        rigid_body = DictIO.get_alternative(template, "RigidBody", False)
        if rigid_body:
            material_id = 0
            density = DictIO.get_alternative(template, "Density", 2650)
            scene.is_rigid[body_id] = 1
        else:
            material_id = DictIO.get_essential(template, "MaterialID")
            density = scene.material.matProps[material_id].density

            if material_id <= 0:
                raise SimulationStateError(f"Material ID {material_id} should be larger than 0")

        particle_stress = DictIO.get_alternative(template, "ParticleStress", {"InternalStress": [0, 0, 0, 0, 0, 0]})
        traction = DictIO.get_alternative(template, "Traction", {})
        init_v = DictIO.get_alternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        fix_v_str = DictIO.get_alternative(template, "FixVelocity", ["Free", "Free", "Free"])
        fix_v = vec3u8([DictIO.get_essential(self.FIX, is_fix) for is_fix in fix_v_str])
        init_particle_num = int(scene.particleNum[0])

        scale_factor = DictIO.get_alternative(template, "ScaleFactor", default=1.0)
        offset = np.array(DictIO.get_alternative(template, "Offset", [0.0, 0.0, 0.0]))
        orientation = DictIO.get_alternative(
            template, "Orientation", vec3f([0, 0, 0]) if self.sims.dimension == 3 else 0.0
        )
        n_particles_per_cell = DictIO.get_alternative(template, "nParticlesPerCell", 2)

        mesh = tm.load(particle_file)
        com = mesh.vertices.mean(axis=0)
        mesh.apply_translation(offset - com)
        mesh.apply_scale(scale_factor)
        cell_size = scene.element.grid_size
        diameter = min(cell_size[0], cell_size[1], cell_size[2]) / n_particles_per_cell
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
        kernel_read_particle_file(
            scene.particle,
            int(scene.particleNum[0]),
            particle_num,
            voxelized_points_np,
            volume,
            body_id,
            material_id,
            density,
            init_v,
            fix_v,
        )

        self.set_particle_stress(scene, init_particle_num, particle_num, particle_stress)
        scene.push_psize(psize)
        traction = DictIO.get_alternative(template, "Traction", {})
        self.set_traction(particle_num, traction, scene)
        scene.material.state_vars_initialize(
            material_id, int(scene.particleNum[0]), int(scene.particleNum[0]) + particle_num, scene.particle
        )
        scene.particleNum[0] += particle_num
        self.print_particle_info(
            body_id,
            material_id,
            init_v,
            fix_v,
            particle_num,
            particle_volume=particle_volume,
            nParticlesPerCell=n_particles_per_cell,
        )

    def set_polygons(self, contact: DEMContact, body_dict):
        vertices = DictIO.get_essential(body_dict, "Vertices")
        if isinstance(vertices, str):
            print("#", f"Reading {vertices}......")
            if not os.path.exists(vertices):
                raise EOFError("Invaild path")
            vertice = np.loadtxt(vertices)[:, 0 : self.sims.dimension]
        elif isinstance(vertices, (list, tuple, np.ndarray)):
            vertice = np.array(vertices)

        velocity = DictIO.get_alternative(body_dict, "InitialVelocity", [0.0, 0.0])
        contact.generate_vertice_field(self.sims.dimension, vertice.shape[0])
        contact.polygon_vertices.from_numpy(vertice)
        contact.velocity = ti.Vector(np.array(velocity))
