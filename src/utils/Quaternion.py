import taichi as ti
import math

from src.utils.constants import ZEROVEC3F, DELTA, PI, ZEROVEC4F, THRESHOLD
from src.utils.TypeDefinition import real, vec3f, vec4f, mat2x2, mat3x3, mat4x4
from src.utils.ScalarFunction import clamp
from src.utils.VectorFunction import normalize

# https://api.flutter.dev/flutter/vector_math/Quaternion/setAxisAngle.html


# ================================= Operator ===================================== #
@ti.func
def add(q1, q2):
    return q1 + q2


@ti.func
def sub(q1, q2):
    return q1 - q2


@ti.func
def multiply(q1, q2):
    q1w, q1x, q1y, q1z = q1[3], q1[0], q1[1], q1[2]
    q2w, q2x, q2y, q2z = q2[3], q2[0], q2[1], q2[2]
    return vec4f(
        [
            q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y,
            q1w * q2y + q1y * q2w + q1z * q2x - q1x * q2z,
            q1w * q2z + q1z * q2w + q1x * q2y - q1y * q2x,
            q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z,
        ]
    )


@ti.func
def normalized(q):
    qr = normalize(q)
    if q[0] < -THRESHOLD:
        qr = -qr
    return qr


@ti.func
def average(q1, q2):
    aq = ZEROVEC4F
    q1w, q1x, q1y, q1z = q1[3], q1[0], q1[1], q1[2]
    q2w, q2x, q2y, q2z = q2[3], q2[0], q2[1], q2[2]
    q1Tq2 = q1w * q2w + q1x * q2x + q1y * q2y + q1z * q2z
    if q1Tq2 != 0.0:
        aq = q1 * ti.abs(q1Tq2) + q2 * q1Tq2
    else:
        w1 = 0.5
        w2 = 0.6
        z = ti.sqrt((w1 - w2) * (w1 - w2) + 4 * w1 * w2 * q1Tq2 * q1Tq2)
        aq = q1 * 2 * w1 * q1Tq2 + q2 * (w2 - w1 + z)
    aq = normalized(aq)
    return aq


@ti.func
def sacle(q, scale):
    return q * scale


@ti.func
def conjugate(q):
    return vec4f([-q[0], -q[1], -q[2], q[3]])


@ti.func
def inverse(q):
    return vec4f([-q[0], -q[1], -q[2], q[3]]).normalized()


@ti.func
def random_generator():
    # From http://planning.cs.uiuc.edu/node198.html.
    # See K. Shoemake - Uniform random rotations - In D. Kirk, editor, Graphics Gems III, pages 124-132. Academic, New York, 1992.
    u1, u2, u3 = ti.random(), ti.random(), ti.random()
    q = vec4f(
        [
            ti.sqrt(1 - u1) * ti.sin(2 * PI * u2),
            ti.sqrt(1 - u1) * ti.cos(2 * PI * u2),
            ti.sqrt(u1) * ti.sin(2 * PI * u3),
            ti.sqrt(u1) * ti.cos(2 * PI * u3),
        ]
    )
    return normalized(q)


# ============================================== Method ========================================== #
@ti.func
def set_to_rotate(q):
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    return mat3x3(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


@ti.func
def set_to_euler_xyz(q):
    euler = ZEROVEC3F
    q = normalized(q)
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    t2 = 2 * (qw * qy - qx * qz)
    t2 = clamp(-1.0, 1.0, t2)
    euler[1] = t2
    if ti.abs(t2) < 0.99999:
        euler[0] = ti.atan2(2 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz)
        euler[2] = ti.atan2(2 * (qx * qy + qw * qz), qw * qw + qx * qx - qy * qy - qz * qz)
    else:
        euler[0] = 0.0
        euler[2] = ti.atan2(2 * (qw * qz - qx * qy), qw * qw - qx * qx + qy * qy - qz * qz)
    return euler


@ti.func
def rotate(q, vec):
    q_inv = conjugate(q)
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    qwi, qxi, qyi, qzi = q_inv[3], q_inv[0], q_inv[1], q_inv[2]
    vx, vy, vz = vec[0], vec[1], vec[2]

    tx = qwi * vx + qyi * vz - qzi * vy
    ty = qwi * vy + qzi * vx - qxi * vz
    tz = qwi * vz + qxi * vy - qyi * vx
    tw = -qxi * vx - qyi * vy - qzi * vz

    return vec3f(
        [
            tw * qx + tx * qw + ty * qz - tz * qy,
            tw * qy + ty * qw + tz * qx - tx * qz,
            tw * qz + tz * qw + tx * qy - ty * qx,
        ]
    )


@ti.func
def get_axis(q):
    axis = ZEROVEC3F
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    den = 1.0 - qw * qw
    if den > 1e-8:
        scale = 1.0 / ti.sqrt(den)
        axis = vec3f([qx, qy, qz]) * scale
    return axis


@ti.func
def length(q):
    return q.norm()


@ti.func
def radians(q):
    return 2.0 * ti.acos(q[3])


# ======================================== Set up Quaternion ============================================ #
@ti.func
def set_dq(q, omega):
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    ox, oy, oz = omega[0], omega[1], omega[2]
    return 0.5 * vec4f(
        [
            ox * qw - oy * qz + oz * qy,
            oy * qw - oz * qx + ox * qz,
            oz * qw - ox * qy + oy * qx,
            -ox * qx - oy * qy - oz * qz,
        ]
    )


@ti.func
def update_q_accurate(q, omega, dt):
    beta = 0.25 * dt[None] * omega
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    bx, by, bz = beta[0], beta[1], beta[2]
    P = vec4f(
        qx + bz * qy - bx * qz - by * qw,
        -bz * qx + qy - by * qz + bx * qw,
        bx * qx + by * qy + qz + bz * qw,
        by * qx + bx * qy - bz * qz + qw,
    )
    R = mat4x4([1, -bz, bx, by], [bz, 1, by, -bx], [-bx, -by, 1, -bz], [-by, bx, bz, 1])
    kappa = (1 + bx * bx + by * by + bz * bz) / R.determinant()
    return kappa * R.transpose() @ P


@ti.func
def set_from_value(qx, qy, qz, qw):
    return vec4f([qx, qy, qz, qw])


@ti.func
def set_from_axis_angle(axis, radians):
    leng = length(axis)
    half_sin = ti.sin(0.5 * radians) / leng
    return vec4f([axis[0] * half_sin, axis[1] * half_sin, axis[2] * half_sin, ti.cos(0.5 * radians)])


@ti.func
def set_from_euler(roll, pitch, yaw):
    half_yaw = yaw * 0.5
    half_pitch = pitch * 0.5
    half_roll = roll * 0.5
    cos_yaw = ti.cos(half_yaw)
    sin_yaw = ti.sin(half_yaw)
    cos_pitch = ti.cos(half_pitch)
    sin_pitch = ti.sin(half_pitch)
    cos_roll = ti.cos(half_roll)
    sin_roll = ti.sin(half_roll)
    return vec4f(
        [
            sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw,
            cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw,
            cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw,
            cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw,
        ]
    )


@ti.func
def set_from_rotation(rotation_matrix):
    # see http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    q = ZEROVEC4F
    trace = rotation_matrix.trace()
    if trace > 0.0:
        s = 0.5 / ti.sqrt(trace + 1.0)
        q[3] = 0.25 / s
        q[0] = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
        q[1] = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
        q[2] = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
    else:
        if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            s = 1.0 / (2.0 * ti.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]))
            q[3] = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
            q[0] = 0.25 / s
            q[1] = (rotation_matrix[1, 0] + rotation_matrix[0, 1]) * s
            q[2] = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) * s
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            s = 1.0 / (2.0 * ti.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]))
            q[3] = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
            q[0] = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) * s
            q[1] = 0.25 / s
            q[2] = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) * s
        else:
            s = 1.0 / (2.0 * ti.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]))
            q[3] = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
            q[0] = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) * s
            q[1] = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) * s
            q[2] = 0.25 / s
    return normalized(q)


@ti.func
def set_from_two_vec(vec1, vec2):
    v1 = vec1.normalized()
    v2 = vec2.normalized()

    c = v1.dot(v2)
    angle = ti.acos(c)
    axis = v1.cross(v2)

    if ti.abs(1.0 + c) < 1e-8:
        angle = math.pi
        if v1[0] > v1[1] and v1[0] > v1[2]:
            axis = v1.cross(vec3f([0.0, 1.0, 0.0]))
        else:
            axis = v1.cross(vec3f([1.0, 0.0, 0.0]))
    elif ti.abs(1.0 - c) < 1e-8:
        angle = 0.0
        axis = vec3f([1.0, 0.0, 0.0])

    return set_from_axis_angle(axis.normalized(), angle)


@ti.pyfunc
def rodrigues_rotation_matrix(origin, target, dtype: ti.template() = real):
    cos_theta = origin.dot(target)
    norm_vec = origin.cross(target)
    norm_vec_invert = ti.Matrix(
        [[0.0, -norm_vec[2], norm_vec[1]], [norm_vec[2], 0.0, -norm_vec[0]], [-norm_vec[1], norm_vec[0], 0.0]],
        dt=dtype,
    )
    identity = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dt=dtype)
    rotation_martix = identity + norm_vec_invert + (norm_vec_invert @ norm_vec_invert) / (1 + cos_theta)
    return rotation_martix


@ti.pyfunc
def rotation_matrix2d(origin, target):
    cos_theta = origin.dot(target)
    sin_theta = ti.pow(1 - cos_theta * cos_theta, 0.5)
    rotation_martix = mat2x2([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    return rotation_martix


@ti.pyfunc
def theta_to_rotation_matrix(theta):
    content_in_rad = theta * PI / 180
    rotate_x = ti.Matrix(
        [
            [1.0, 0.0, 0.0],
            [0, ti.cos(content_in_rad[0]), -ti.sin(content_in_rad[0])],
            [0, ti.sin(content_in_rad[0]), ti.cos(content_in_rad[0])],
        ]
    )
    rotate_y = ti.Matrix(
        [
            [ti.cos(content_in_rad[1]), 0, ti.sin(content_in_rad[1])],
            [0.0, 1.0, 0.0],
            [-ti.sin(content_in_rad[1]), 0, ti.cos(content_in_rad[1])],
        ]
    )
    rotate_z = ti.Matrix(
        [
            [ti.cos(content_in_rad[2]), -ti.sin(content_in_rad[2]), 0],
            [ti.sin(content_in_rad[2]), ti.cos(content_in_rad[2]), 0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rotate_z @ rotate_y @ rotate_x


@ti.pyfunc
def theta_to_rotation_matrix2d(theta):
    angle_rad = theta * PI / 180
    R = ti.Matrix([[ti.cos(angle_rad), -ti.sin(angle_rad)], [ti.sin(angle_rad), ti.cos(angle_rad)]])
    return R
