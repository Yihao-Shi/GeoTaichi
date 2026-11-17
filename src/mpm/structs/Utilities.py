from src.utils.MatrixFunction import (
    _stress_increment_voigt,
    _stress_increment_mat3,
    _stress_increment_mat2,
    _stress_increment_pressure,
)
import taichi as ti
from src.utils.constants import ZEROVEC6F
from src.utils.TypeDefinition import vec6f, mat3x3
from src.utils.TypeDefinition import real, vec2f, vec2u8, vec6f, mat2x2
from src.utils.BitFunction import zero2_one_vector
import src.utils.GlobalVariable as GlobalVariable
from src.utils.constants import ZEROVEC3F
from src.utils.TypeDefinition import real, vec3f, vec3u8, vec6f, mat3x3


@ti.func
def project_coulomb_wall(velocity, acceleration, mu, normal_axis, outward_sign, dt):
    # Grid kinematics has already applied acceleration: project its trial velocity once.
    corrected = velocity
    normal_impulse = ti.max(velocity[normal_axis] * outward_sign, 0.0)
    if normal_impulse > 0.0:
        tangent = velocity
        tangent[normal_axis] = 0.0
        speed = tangent.norm()
        scale = 0.0
        if speed > 0.0:
            scale = ti.max(0.0, 1.0 - mu * normal_impulse / speed)
        corrected = scale * tangent
        acceleration += (corrected - velocity) / dt
    return corrected, acceleration
