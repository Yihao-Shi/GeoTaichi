import taichi as ti
from src.utils.constants import ZEROVEC3F
from src.utils.TypeDefinition import mat3x3, real, vec3f, vec3i


@ti.dataclass
class SoftBody:
    bodyID: int
    groupID: int
    materialID: int
    startIndex: int
    endIndex: int
    startNode: int
    endNode: int
    localNode: int
    surfacePointStart: int
    surfacePointEnd: int
    gridStart: int
    mpmGridStart: int
    gridNum: int
    mpmGridNum: int
    mpmGridOffset: vec3i
    mpmGridSize: vec3i
    templatePointStart: int
    templateSurfaceStart: int
    templateSdfStart: int
    gridType: ti.u8
    scale: real
    gridSpace: real
    referenceRotation: mat3x3
    m: real
    mass_center0: vec3f
    mass_center: vec3f
    previous_center: vec3f
    previous_rotation: mat3x3
    surface_inertia: mat3x3
    v: vec3f
    contact_force: vec3f
    external_load_factor: real
    damp_energy: real

    @ti.func
    def _restart(self, body_id, input_start_index, input_end_index, group_id, material_id):
        self.bodyID = int(body_id)
        self.startIndex = int(input_start_index)
        self.endIndex = int(input_end_index)
        self.groupID = int(group_id)
        self.materialID = int(material_id)

    @ti.func
    def _add_body_attribute(self, mass, mass_center):
        self.m = float(mass)
        self.mass_center0 = float(mass_center)
        self.mass_center = float(mass_center)
        self.previous_center = float(mass_center)
        self.v = ZEROVEC3F
        self.contact_force = ZEROVEC3F
        self.external_load_factor = 1.0
        self.damp_energy = 0.0

    @ti.func
    def _set_external_load_factor(self, factor):
        self.external_load_factor = float(factor)

    @ti.func
    def _add_body_properties(self, material_id, group_id):
        self.materialID = int(material_id)
        self.groupID = int(group_id)

    @ti.func
    def _add_material_point_index(self, start_index, end_index):
        self.startIndex = int(start_index)
        self.endIndex = int(end_index)

    @ti.func
    def _add_grid_index(self, grid_start, mpm_grid_start, grid_num, mpm_grid_num, mpm_grid_offset, mpm_grid_size):
        self.gridStart = int(grid_start)
        self.mpmGridStart = int(mpm_grid_start)
        self.gridNum = int(grid_num)
        self.mpmGridNum = int(mpm_grid_num)
        self.mpmGridOffset = int(mpm_grid_offset)
        self.mpmGridSize = int(mpm_grid_size)

    @ti.func
    def _add_surface_index(self, start_index, end_index, local_index):
        self.startNode = int(start_index)
        self.endNode = int(end_index)
        self.localNode = int(local_index)

    @ti.func
    def _add_surface_point_index(self, start_index, end_index):
        self.surfacePointStart = int(start_index)
        self.surfacePointEnd = int(end_index)

    @ti.func
    def _add_template_support(
        self,
        point_start,
        surface_start,
        sdf_start,
        grid_type,
        scale,
        grid_space,
        reference_rotation,
    ):
        self.templatePointStart = int(point_start)
        self.templateSurfaceStart = int(surface_start)
        self.templateSdfStart = int(sdf_start)
        self.gridType = ti.u8(grid_type)
        self.scale = float(scale)
        self.gridSpace = float(grid_space)
        self.referenceRotation = reference_rotation

    @ti.func
    def _get_vertice_number(self):
        return int(self.endNode - self.startNode)

    @ti.func
    def _start_node(self):
        return self.localNode

    @ti.func
    def _end_node(self):
        return int(self.localNode + self.endNode - self.startNode)

    @ti.func
    def local_node_to_global(self, node):
        return int(node - self.localNode + self.startNode)

    @ti.func
    def global_node_to_local(self, node):
        return int(node - self.startNode + self.localNode)

    @ti.func
    def _get_material(self):
        return self.materialID

    @ti.func
    def _get_group(self):
        return self.groupID
