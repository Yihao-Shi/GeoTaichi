from src.dem.generator.AffineBodyTemplate import AffineBodyTemplate
from src.utils.errors import SimulationStateError

from src.dem.generator.InsertionKernel import *
from src.dem.generator.GeneralShapeTemplate import GeneralShapeTemplate
from src.dem.generator.ClumpTemplate import ClumpTemplate
from src.dem.SceneManager import SceneManager
from src.dem.Simulation import Simulation
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefinition import vec3f, vec3i
from src.utils.Orientation import Orientation


class ParticleCreator(object):
    def __init__(self) -> None:
        self.myTemplate = None
        self.FIX = {"Free": 0, "Fix": 1}

    def create(self, sims, scene, body_dict):
        btype = DictIO.get_essential(body_dict, "BodyType")
        template = DictIO.get_essential(body_dict, "Template")
        if btype == "Sphere":
            self.create_sphere(sims, scene, template)
        elif btype == "Clump":
            self.create_multisphere(sims, scene, template)
        elif btype == "RigidBody":
            self.create_rigid_body(sims, scene, template)
        elif btype == "AffineBody":
            self.create_affine_body(sims, scene, template)
        elif btype == "SoftBody":
            self.create_soft_body(sims, scene, template)
        else:
            lists = ["Sphere", "Clump", "RigidBody", "AffineBody", "SoftBody"]
            raise SimulationStateError(f"Invalid Keyword:: /BodyType/: {btype}. Only the following {lists} are valid")

    def create_sphere(self, sims, scene: SceneManager, template):
        if type(template) is dict:
            self.create_template_sphere(sims, scene, template)
        elif type(template) is list:
            for temp in template:
                self.create_template_sphere(sims, scene, temp)

    def create_template_sphere(self, sims, scene: SceneManager, template):
        particle = scene.get_particle_ptr()
        sphere = scene.get_sphere_ptr()
        material = scene.get_material_ptr()
        body_num = int(scene.sphereNum[0])
        particle_num = int(scene.particleNum[0])

        radius = DictIO.get_essential(template, "Radius")
        position = DictIO.get_essential(template, "BodyPoint")
        group_id = DictIO.get_essential(template, "GroupID")
        mat_id = DictIO.get_essential(template, "MaterialID")
        init_v = DictIO.get_alternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.get_alternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        fix_v_str = DictIO.get_alternative(template, "FixVelocity", ["Free", "Free", "Free"])
        fix_w_str = DictIO.get_alternative(template, "FixAngularVelocity", ["Free", "Free", "Free"])
        scene.check_particle_num(sims, particle_number=1)
        scene.check_sphere_number(sims, body_number=1)
        fix_v = vec3i([DictIO.get_essential(self.FIX, i) for i in fix_v_str])
        fix_w = vec3i([DictIO.get_essential(self.FIX, i) for i in fix_w_str])
        kernel_create_sphere(
            particle,
            sphere,
            material,
            body_num,
            particle_num,
            position,
            radius,
            group_id,
            mat_id,
            init_v,
            init_w,
            fix_v,
            fix_w,
        )
        print(" Sphere Information ".center(71, "-"))
        self.print_particle_info(group_id, mat_id, position, init_v, init_w, fix_v=fix_v_str, fix_w=fix_w_str)
        scene.sphereNum[0] += 1
        scene.particleNum[0] += 1

    def set_template(self, template_ptr):
        self.myTemplate = template_ptr

    def get_template_ptr_by_name(self, name):
        clump_ptr = DictIO.get_optional(self.myTemplate, name)
        if not clump_ptr:
            raise KeyError(f"Template name: {name} is not set before")
        return clump_ptr

    def _soft_particle_backend(self, scene):
        backend = getattr(scene, "soft_particle_backend", None)
        if backend is None:
            raise SimulationStateError(
                "SoftBody is an MPM soft-particle mode. Use MPDEM/MPM soft_particle instead of DEM generator."
            )
        return backend

    def create_soft_body(self, sims, scene, template):
        return self._soft_particle_backend(scene).create_soft_body(self, sims, scene, template)

    def create_multisphere(self, sims, scene: SceneManager, template):
        if type(template) is dict:
            self.create_template_multisphere(sims, scene, template)
        elif type(template) is list:
            for temp in template:
                self.create_template_multisphere(sims, scene, temp)

    def create_template_multisphere(self, sims, scene: SceneManager, template):
        particle = scene.get_particle_ptr()
        clump = scene.get_clump_ptr()
        material = scene.get_material_ptr()
        body_num = int(scene.clumpNum[0])
        particle_num = int(scene.particleNum[0])

        name = DictIO.get_essential(template, "Name")
        template_ptr: ClumpTemplate = self.get_template_ptr_by_name(name)
        com_pos = DictIO.get_essential(template, "BodyPoint")
        equiv_rad = DictIO.get_alternative(template, "Radius", None)
        scale_factor = DictIO.get_alternative(template, "ScaleFactor", None)
        orientation = DictIO.get_alternative(template, "BodyOrientation", None)
        set_orientations = Orientation(orientation)

        group_id = DictIO.get_essential(template, "GroupID")
        mat_id = DictIO.get_essential(template, "MaterialID")
        init_v = DictIO.get_alternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.get_alternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))

        if equiv_rad is None and isinstance(scale_factor, (float, int)):
            equiv_rad = float(scale_factor) * template_ptr.r_equiv
        elif scale_factor is None and isinstance(equiv_rad, (float, int)):
            scale_factor = float(equiv_rad) / template_ptr.r_equiv
        else:
            raise SimulationStateError(
                "Keyword conflict!, You should set either Keyword:: /Radius/ or Keyword:: /ScaleFactor/."
            )

        scene.check_particle_num(sims, particle_number=template_ptr.nspheres)
        scene.check_clump_number(sims, body_number=1)
        kernel_create_multisphere(
            particle,
            clump,
            material,
            body_num,
            particle_num,
            template_ptr.nspheres,
            scale_factor,
            template_ptr.inertia,
            template_ptr.x_pebble,
            template_ptr.rad_pebble,
            com_pos,
            equiv_rad,
            set_orientations.get_orientation,
            group_id,
            mat_id,
            init_v,
            init_w,
        )
        print(" Clump Information ".center(71, "-"))
        self.print_particle_info(group_id, mat_id, com_pos, init_v, init_w, name=name)
        scene.clumpNum[0] += 1
        scene.particleNum[0] += template_ptr.nspheres

    def create_rigid_body(self, sims, scene: SceneManager, template):
        if type(template) is dict:
            self.create_template_rigid_body(sims, scene, template)
        elif type(template) is list:
            for temp in template:
                self.create_template_rigid_body(sims, scene, temp)

    def create_affine_body(self, sims, scene: SceneManager, template):
        if type(template) is dict:
            self.create_template_affine_body(sims, scene, template)
        elif type(template) is list:
            for temp in template:
                self.create_template_affine_body(sims, scene, temp)

    def create_template_affine_body(self, sims: Simulation, scene: SceneManager, template):
        name = DictIO.get_essential(template, "Name")
        template_ptr = self.get_template_ptr_by_name(name)
        if not isinstance(template_ptr, AffineBodyTemplate):
            raise SimulationStateError("AffineBody create expects an AffineBody template.")

        com_pos = DictIO.get_essential(template, "BodyPoint")
        equiv_rad = DictIO.get_alternative(template, "Radius", None)
        bounding_rad = DictIO.get_alternative(template, "BoundingRadius", None)
        scale_factor = DictIO.get_alternative(template, "ScaleFactor", None)
        orientation = DictIO.get_alternative(template, "BodyOrientation", None)
        group_id = DictIO.get_essential(template, "GroupID")
        mat_id = DictIO.get_essential(template, "MaterialID")
        init_v = DictIO.get_alternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.get_alternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        fix_str = DictIO.get_alternative(template, "FixMotion", ["Free", "Free", "Free"])
        is_fix = vec3i([DictIO.get_essential(self.FIX, i) for i in fix_str])

        if isinstance(scale_factor, (float, int)):
            scale_factor = float(scale_factor)
        elif isinstance(equiv_rad, (float, int)):
            scale_factor = float(equiv_rad) / max(template_ptr.eqradius, 1.0e-30)
        elif isinstance(bounding_rad, (float, int)):
            scale_factor = float(bounding_rad) / max(template_ptr.bounding_radius, 1.0e-30)
        else:
            scale_factor = 1.0
        young = DictIO.get_alternative(template, "YoungModulus", None)
        friction = DictIO.get_alternative(template, "Friction", 0.0)
        local_damping = DictIO.get_alternative(template, "LocalDamping", None)
        fdamp = DictIO.get_alternative(
            template, "ForceLocalDamping", DictIO.get_alternative(template, "force_local_damping", local_damping)
        )
        tdamp = DictIO.get_alternative(
            template, "TorqueLocalDamping", DictIO.get_alternative(template, "torque_local_damping", local_damping)
        )
        scene.add_affine_body(
            sims,
            name,
            com_pos,
            scale_factor,
            group_id,
            mat_id,
            init_v,
            init_w,
            orientation,
            young_modulus=young,
            friction=friction,
            force_local_damping=fdamp,
            torque_local_damping=tdamp,
        )
        print(" Affine body Information ".center(71, "-"))
        self.print_particle_info(group_id, mat_id, com_pos, init_v, init_w, fix_v=is_fix, fix_w=is_fix, name=name)

    def create_template_rigid_body(self, sims: Simulation, scene: SceneManager, template):
        bounding_sphere = scene.get_bounding_sphere()
        bounding_box = scene.get_bounding_box()
        master = scene.get_surface()
        rigid_body = scene.get_rigid_ptr()
        material = scene.get_material_ptr()
        particle_num = int(scene.particleNum[0])
        surface_num = int(scene.surfaceNum[0])

        name = DictIO.get_essential(template, "Name")
        template_ptr: GeneralShapeTemplate = self.get_template_ptr_by_name(name)
        index = DictIO.get_essential(scene.prefixID, name)
        com_pos = DictIO.get_essential(template, "BodyPoint")
        equiv_rad = DictIO.get_alternative(template, "Radius", None)
        bounding_rad = DictIO.get_alternative(template, "BoundingRadius", None)
        scale_factor = DictIO.get_alternative(template, "ScaleFactor", None)
        orientation = DictIO.get_alternative(template, "BodyOrientation", None)
        check_overlap = DictIO.get_alternative(template, "CheckOverlap", True)
        set_orientations = Orientation(orientation)

        group_id = DictIO.get_essential(template, "GroupID")
        mat_id = DictIO.get_essential(template, "MaterialID")
        init_v = DictIO.get_alternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.get_alternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        fix_str = DictIO.get_alternative(template, "FixMotion", ["Free", "Free", "Free"])
        is_fix = vec3i([DictIO.get_essential(self.FIX, i) for i in fix_str])

        if sims.scheme == "AffineBody":
            if isinstance(scale_factor, (float, int)):
                scale_factor = float(scale_factor)
                equiv_rad = scale_factor * template_ptr.eqradius
            elif isinstance(equiv_rad, (float, int)):
                scale_factor = float(equiv_rad) / max(template_ptr.eqradius, 1.0e-30)
            elif isinstance(bounding_rad, (float, int)):
                scale_factor = float(bounding_rad) / max(template_ptr.bounding_radius, 1.0e-30)
                equiv_rad = scale_factor * template_ptr.eqradius
            else:
                scale_factor = 1.0
                equiv_rad = template_ptr.eqradius
            young = DictIO.get_alternative(template, "YoungModulus", None)
            friction = DictIO.get_alternative(template, "Friction", 0.0)
            local_damping = DictIO.get_alternative(template, "LocalDamping", None)
            fdamp = DictIO.get_alternative(
                template, "ForceLocalDamping", DictIO.get_alternative(template, "force_local_damping", local_damping)
            )
            tdamp = DictIO.get_alternative(
                template, "TorqueLocalDamping", DictIO.get_alternative(template, "torque_local_damping", local_damping)
            )
            scene.add_affine_body(
                sims,
                name,
                com_pos,
                scale_factor,
                group_id,
                mat_id,
                init_v,
                init_w,
                orientation,
                young_modulus=young,
                friction=friction,
                force_local_damping=fdamp,
                torque_local_damping=tdamp,
            )
            print(" Affine body Information ".center(71, "-"))
            self.print_particle_info(group_id, mat_id, com_pos, init_v, init_w, fix_v=is_fix, fix_w=is_fix, name=name)
            return

        if isinstance(scale_factor, (float, int)):
            equiv_rad = float(scale_factor) * template_ptr.objects.eqradius
        elif isinstance(equiv_rad, (float, int)):
            scale_factor = float(equiv_rad) / template_ptr.objects.eqradius
        elif isinstance(bounding_rad, (float, int)):
            equiv_rad = template_ptr.objects.eqradius / template_ptr.boundings.r_bound * bounding_rad
            scale_factor = float(equiv_rad) / template_ptr.objects.eqradius
        else:
            raise SimulationStateError(
                "Keyword conflict!, You should set either Keyword:: /Radius/ or Keyword:: /ScaleFactor/."
            )

        if sims.scheme == "PolySuperEllipsoid" or sims.scheme == "PolySuperQuadrics":
            if sims.iterative_model == "LagrangianMultiplier":
                if sims.scheme == "PolySuperEllipsoid":
                    assert (
                        0.25 <= template_ptr.objects.physical_parameter["epsilon_e"] <= 1
                    ), f"The optimal value range of parameter /epsilon_e/ is 0.25 to 1"
                    assert (
                        0.25 <= template_ptr.objects.physical_parameter["epsilon_n"] <= 1
                    ), f"The optimal value range of parameter /epsilon_n/ is 0.25 to 1"
                elif sims.scheme == "PolySuperQuadrics":
                    assert (
                        0.25 <= template_ptr.objects.physical_parameter["epsilon_x"] <= 1
                    ), f"The optimal value range of parameter /epsilon_x/ is 0.25 to 1"
                    assert (
                        0.25 <= template_ptr.objects.physical_parameter["epsilon_y"] <= 1
                    ), f"The optimal value range of parameter /epsilon_y/ is 0.25 to 1"
                    assert (
                        0.25 <= template_ptr.objects.physical_parameter["epsilon_z"] <= 1
                    ), f"The optimal value range of parameter /epsilon_z/ is 0.25 to 1"
            template_id = scene.prefixID[name]
            scene.check_rigid_body_number(sims, rigid_body_number=1)
            kernel_create_implicit_surface_rigid_body(
                rigid_body,
                bounding_sphere,
                material,
                particle_num,
                template_id,
                template_ptr.boundings.r_bound,
                vec3f(template_ptr.boundings.x_bound),
                scale_factor,
                vec3f(template_ptr.objects.inertia),
                com_pos,
                equiv_rad,
                set_orientations.get_orientation,
                group_id,
                mat_id,
                init_v,
                init_w,
                is_fix,
            )
            print(" Implicit surface body Information ".center(71, "-"))
        else:
            grid_num = scene.gridID[index]
            vertice_num = scene.verticeID[index]
            scene.check_rigid_body_number(sims, rigid_body_number=1)
            kernel_create_level_set_rigid_body(
                rigid_body,
                bounding_box,
                bounding_sphere,
                master,
                material,
                particle_num,
                grid_num,
                vertice_num,
                surface_num,
                vec3f(template_ptr.objects.grid.min_box()),
                vec3f(template_ptr.objects.grid.max_box()),
                template_ptr.boundings.r_bound,
                vec3f(template_ptr.boundings.x_bound),
                template_ptr.surface_node_number,
                template_ptr.surface_area,
                template_ptr.objects.grid.grid_space,
                vec3i(template_ptr.objects.grid.gnum),
                template_ptr.objects.grid.extent,
                scale_factor,
                vec3f(template_ptr.objects.inertia),
                com_pos,
                equiv_rad,
                set_orientations.get_orientation,
                group_id,
                mat_id,
                init_v,
                init_w,
                is_fix,
            )

            print(" Level set body Information ".center(71, "-"))
        self.print_particle_info(group_id, mat_id, com_pos, init_v, init_w, fix_v=is_fix, fix_w=is_fix, name=name)

        faces = scene.add_connectivity(1, template_ptr.surface_node_number, template_ptr.objects)
        scene.particleNum[0] += 1
        scene.rigidNum[0] += 1
        scene.surfaceNum[0] += template_ptr.surface_node_number

    def print_particle_info(self, groupID, matID, com_pos, init_v, init_w, fix_v=None, fix_w=None, name=None):
        if name:
            print("Template name: ", name)
        print("Group ID = ", groupID)
        print("Material ID = ", matID)
        print("Center of Mass", com_pos)
        print("Initial Velocity = ", init_v)
        print("Initial Angular Velocity = ", init_w)
        if fix_v:
            print("Fixed Velocity = ", fix_v)
        if fix_w:
            print("Fixed Angular Velocity = ", fix_w)
        print("\n")
