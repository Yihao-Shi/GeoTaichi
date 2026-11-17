from src.utils.errors import SimulationStateError

import numpy as np
from src.dem.GenerateManager import GenerateManager as DEMGenerator
from src.dem.SceneManager import SceneManager as DEMScene
from src.dem.Simulation import Simulation as DEMSimulation
from src.dem.generator.GeneralShapeTemplate import GeneralShapeTemplate
from src.mpm.SceneManager import SceneManager as MPMScene
from src.mpm.Simulation import Simulation as MPMSimulation
from src.utils.ObjectIO import DictIO
from src.utils.TypeDefinition import vec3f, vec3i
from src.mesh.GaussPointInTriangle import GaussPointInTriangle
from src.utils.Orientation import Orientation


class ParticleCreator(object):
    mscene: MPMScene
    msims: MPMSimulation
    dscene: DEMScene
    dsims: DEMSimulation
    dgenerator: DEMGenerator

    def __init__(self, mscene, msims, dscene, dsims, dgenerator) -> None:
        self.mscene = mscene
        self.msims = msims
        self.dscene = dscene
        self.dsims = dsims
        self.dgenerator = dgenerator
        self.myTemplate = None
        self.FIX = {"Free": 0, "Fix": 1}

    def create(self, body_dict):
        template = DictIO.get_alternative(body_dict, "Template", body_dict)
        self.create_soft_body(template)

    def create_soft_body(self, template):
        if type(template) is dict:
            self.create_template_soft_body(template)
        elif type(template) is list:
            for temp in template:
                self.create_template_soft_body(temp)

    def create_template_soft_body(self, template):
        bounding_sphere = self.dscene.get_bounding_sphere()
        bounding_box = self.dscene.get_bounding_box()
        master = self.dscene.get_surface()
        rigid_body = self.dscene.get_rigid_ptr()
        material = self.dscene.get_material_ptr()
        particle_num = int(self.dscene.particleNum[0])
        surface_num = int(self.dscene.surfaceNum[0])

        name = DictIO.get_essential(template, "Name")
        template_ptr: GeneralShapeTemplate = DictIO.get_essential(self.dgenerator.myTemplate, name)
        com_pos = DictIO.get_essential(template, "BodyPoint")
        equiv_rad = DictIO.get_alternative(template, "Radius", None)
        bounding_rad = DictIO.get_alternative(template, "BoundingRadius", None)
        scale_factor = DictIO.get_alternative(template, "ScaleFactor", None)
        orientation = DictIO.get_alternative(template, "BodyOrientation", None)
        npic = DictIO.get_alternative(template, "nParticlesPerCell", 1)
        set_orientations = Orientation(orientation)

        group_id = DictIO.get_essential(template, "GroupID")
        mat_id = DictIO.get_essential(template, "MaterialID")
        init_v = DictIO.get_alternative(template, "InitialVelocity", vec3f([0, 0, 0]))
        init_w = DictIO.get_alternative(template, "InitialAngularVelocity", vec3f([0, 0, 0]))
        fix_str = DictIO.get_alternative(template, "FixMotion", ["Free", "Free", "Free"])
        is_fix = vec3i([DictIO.get_essential(self.FIX, i) for i in fix_str])

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

        gauss = GaussPointInTriangle(npic)
        self.dscene.check_soft_body_number(self.dsims, body_number=1)
        self.mscene.check_particle_num(self.msims, particle_number=npic * template_ptr.objects.tetramesh.cell.shape[0])
        kernel_create_level_set_body_(
            rigid_body,
            bounding_box,
            bounding_sphere,
            master,
            material,
            particle_num,
            gridNum,
            verticeNum,
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
        faces = np.array(template_ptr.objects.mesh.faces, dtype=np.int32)
        self.dscene.connectivity = np.append(self.dscene.connectivity, faces + self.dscene.surfaceNum[0]).reshape(-1, 3)
        self.dscene.particleNum[0] += 1
        self.dscene.softNum[0] += 1
        self.dscene.surfaceNum[0] += template_ptr.surface_node_number

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
