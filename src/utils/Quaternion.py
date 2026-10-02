import taichi as ti
import math

from src.utils.constants import ZEROVEC3f, DELTA, PI, ZEROVEC4f, Threshold
from src.utils.TypeDefination import vec3f, vec4f, mat2x2, mat3x3, mat4x4
from src.utils.ScalarFunction import clamp
from src.utils.VectorFunction import Normalize
# https://api.flutter.dev/flutter/vector_math/Quaternion/setAxisAngle.html


# ================================= Operator ===================================== #
@ti.func
def Add(q1, q2): 
    return q1 + q2


@ti.func
def Sub(q1, q2):
    return q1 - q2


@ti.func
def Multiply(q1, q2):                        
    q1w, q1x, q1y, q1z = q1[3], q1[0], q1[1], q1[2]
    q2w, q2x, q2y, q2z = q2[3], q2[0], q2[1], q2[2]
    return vec4f([q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y,
                  q1w * q2y + q1y * q2w + q1z * q2x - q1x * q2z,
                  q1w * q2z + q1z * q2w + q1x * q2y - q1y * q2x,
                  q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z])


@ti.func
def Average(q1, q2):
    aq = ZEROVEC4f
    q1w, q1x, q1y, q1z = q1[3], q1[0], q1[1], q1[2]
    q2w, q2x, q2y, q2z = q2[3], q2[0], q2[1], q2[2]
    q1Tq2 = q1w * q2w + q1x * q2x + q1y * q2y + q1z * q2z
    if q1Tq2 != 0.:
        aq = q1 * ti.abs(q1Tq2) + q2 * q1Tq2
    else:
        w1 = 0.5
        w2 = 0.6
        z = ti.sqrt((w1 - w2) * (w1 - w2) + 4 * w1 * w2 * q1Tq2 * q1Tq2)
        aq = q1 * 2 * w1 * q1Tq2 + q2 * (w2 - w1 + z)
    aq = Normalized(aq)
    return aq


@ti.func
def Sacle(q, scale): 
    return q * scale


@ti.func
def Conjugate(q):  
    return vec4f([-q[0], -q[1], -q[2], q[3]])


@ti.func
def Inverse(q):
    return vec4f([-q[0], -q[1], -q[2], q[3]]).normalized()


@ti.func
def Normalized(q):
    qr = Normalize(q)
    if q[0] < -Threshold:
        qr = -qr
    return qr


@ti.func
def RandomGenerator():
    # From http://planning.cs.uiuc.edu/node198.html. 
    # See K. Shoemake - Uniform random rotations - In D. Kirk, editor, Graphics Gems III, pages 124-132. Academic, New York, 1992.
    u1, u2, u3 = ti.random(), ti.random(), ti.random()
    q = vec4f([ti.sqrt(1 - u1) * ti.sin(2 * PI * u2),
               ti.sqrt(1 - u1) * ti.cos(2 * PI * u2),
               ti.sqrt(u1) * ti.sin(2 * PI * u3),
               ti.sqrt(u1) * ti.cos(2 * PI * u3)
               ])
    return Normalized(q)


# ============================================== Method ========================================== #
@ti.func
def SetToRotate(q):
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    return mat3x3([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)], 
                   [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)], 
                   [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])

@ti.func
def SetToEulerXYZ(q):
    # see https://github.com/mrdoob/three.js/blob/dev/src/math/Euler.js (ZYX)
    euler = ZEROVEC3f
    q = Normalized(q)
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    t2 = 2 * (qw * qy - qx * qz)
    t2 = clamp(-1., 1., t2)
    euler[1] = t2
    if ti.abs(t2) < 0.99999:
        euler[0] = ti.atan2(2 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz)
        euler[2] = ti.atan2(2 * (qx * qy + qw * qz), qw * qw + qx * qx - qy * qy - qz * qz)
    else:
        euler[0] = 0.
        euler[2] = ti.atan2(2 * (qw * qz - qx * qy), qw * qw - qx * qx + qy * qy - qz * qz)
    return euler


@ti.func
def Rotate(q, vec):
    q_inv = Conjugate(q)
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    qwi, qxi, qyi, qzi = q_inv[3], q_inv[0], q_inv[1], q_inv[2]
    vx, vy, vz = vec[0], vec[1], vec[2]

    tx = qwi * vx + qyi * vz - qzi * vy
    ty = qwi * vy + qzi * vx - qxi * vz
    tz = qwi * vz + qxi * vy - qyi * vx
    tw = - qxi * vx - qyi * vy - qzi * vz

    return vec3f([tw * qx + tx * qw + ty * qz - tz * qy,
                  tw * qy + ty * qw + tz * qx - tx * qz,
                  tw * qz + tz * qw + tx * qy - ty * qx])

    
@ti.func
def GetAxis(q):
    axis = ZEROVEC3f
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    den = 1. - qw * qw
    if den > 1e-8:
        scale = 1. / ti.sqrt(den)
        axis = vec3f([qx, qy, qz]) * scale
    return axis


@ti.func
def Length(q):
    return q.norm()


@ti.func
def Radians(q):
    return 2. * ti.acos(q[3])


# ======================================== Set up Quaternion ============================================ #
@ti.func
def SetDQ(q, omega):
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    ox, oy, oz = omega[0], omega[1], omega[2]
    return 0.5 * vec4f([ox * qw - oy * qz + oz * qy,
                        oy * qw - oz * qx + ox * qz,
                        oz * qw - ox * qy + oy * qx,
                        -ox * qx - oy * qy - oz * qz])


@ti.func
def UpdateQAccurate(q, omega, dt):
    beta = 0.25 * dt[None] * omega
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    bx, by, bz = beta[0], beta[1], beta[2]
    P = vec4f(qx + bz * qy - bx * qz - by * qw,
              -bz * qx + qy - by * qz + bx * qw,
              bx * qx + by * qy + qz + bz * qw,
              by * qx + bx * qy - bz * qz + qw)
    R = mat4x4([1, -bz, bx, by], [bz, 1, by, -bx], [-bx, -by, 1, -bz], [-by, bx, bz, 1])
    kappa = (1 + bx * bx + by * by + bz * bz) / R.determinant()
    return kappa * R.transpose() @ P


@ti.func
def SetFromValue(qx, qy, qz, qw):
    return vec4f([qx, qy, qz, qw])


@ti.func
def SetFromAxisAngle(axis, radians):
    leng = Length(axis)
    halfSin = ti.sin(0.5 * radians) / leng
    return vec4f([axis[0] * halfSin,
                  axis[1] * halfSin,
                  axis[2] * halfSin,
                  ti.cos(0.5 * radians)])


@ti.func
def SetFromEuler(roll, pitch, yaw):
    halfYaw = yaw * 0.5
    halfPitch = pitch * 0.5
    halfRoll = roll * 0.5
    cosYaw = ti.cos(halfYaw)
    sinYaw = ti.sin(halfYaw)
    cosPitch = ti.cos(halfPitch)
    sinPitch = ti.sin(halfPitch)
    cosRoll = ti.cos(halfRoll)
    sinRoll = ti.sin(halfRoll)
    return vec4f([sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw,
                  cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw,
                  cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw,
                  cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw])


@ti.func
def SetFromRotation(rotationMatrix):
    # see http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    q = ZEROVEC4f
    trace = rotationMatrix.trace()
    if trace > 0.:
        s = 0.5 / ti.sqrt(trace + 1.0)
        q[3] = 0.25 / s
        q[0] = (rotationMatrix[2, 1] - rotationMatrix[1, 2]) * s
        q[1] = (rotationMatrix[0, 2] - rotationMatrix[2, 0]) * s
        q[2] = (rotationMatrix[1, 0] - rotationMatrix[0, 1]) * s
    else:
        if rotationMatrix[0, 0] > rotationMatrix[1, 1] and rotationMatrix[0, 0] > rotationMatrix[2, 2]:
            s = 1. / (2. * ti.sqrt(1. + rotationMatrix[0, 0] - rotationMatrix[1, 1] - rotationMatrix[2, 2]))
            q[3] = (rotationMatrix[2, 1] - rotationMatrix[1, 2]) * s
            q[0] = 0.25 / s
            q[1] = (rotationMatrix[1, 0] + rotationMatrix[0, 1]) * s
            q[2] = (rotationMatrix[0, 2] + rotationMatrix[2, 0]) * s
        elif rotationMatrix[1, 1] > rotationMatrix[2, 2]:
            s = 1. / (2. * ti.sqrt(1. + rotationMatrix[1, 1] - rotationMatrix[0, 0] - rotationMatrix[2, 2]))
            q[3] = (rotationMatrix[0, 2] - rotationMatrix[2, 0]) * s
            q[0] = (rotationMatrix[0, 1] + rotationMatrix[1, 0]) * s
            q[1] = 0.25 / s
            q[2] = (rotationMatrix[1, 2] + rotationMatrix[2, 1]) * s
        else:
            s = 1. / (2. * ti.sqrt(1. + rotationMatrix[2, 2] - rotationMatrix[0, 0] - rotationMatrix[1, 1]))
            q[3] = (rotationMatrix[1, 0] - rotationMatrix[0, 1]) * s
            q[0] = (rotationMatrix[0, 2] + rotationMatrix[2, 0]) * s
            q[1] = (rotationMatrix[1, 2] + rotationMatrix[2, 1]) * s
            q[2] = 0.25 / s
    return Normalized(q)
  

@ti.func
def SetFromTwoVec(vec1, vec2):
    v1 = vec1.normalized()
    v2 = vec2.normalized()

    c = v1.dot(v2)
    angle = ti.acos(c)
    axis = v1.cross(v2)

    if ti.abs(1.0 + c) < 1e-8:
        angle = math.pi
        if v1[0] > v1[1] and v1[0] > v1[2]:
            axis = v1.cross(vec3f([0., 1., 0.]))
        else:
          axis = v1.cross(vec3f([1., 0., 0.]))
    elif ti.abs(1.0 - c) < 1e-8:
        angle = 0.
        axis = vec3f([1., 0., 0.])

    return SetFromAxisAngle(axis.normalized(), angle)


@ti.pyfunc
def RodriguesRotationMatrix(origin, target):
    cos_theta = origin.dot(target)
    norm_vec = origin.cross(target)
    norm_vec_invert = mat3x3([[0., -norm_vec[2], norm_vec[1]], 
                             [norm_vec[2], 0., -norm_vec[0]],
                             [-norm_vec[1], norm_vec[0], 0.]])
    RotationMartix = DELTA + norm_vec_invert + (norm_vec_invert @ norm_vec_invert) / (1 + cos_theta)
    return RotationMartix


@ti.pyfunc
def RotationMatrix2D(origin, target):
    cos_theta = origin.dot(target)
    sin_theta = ti.pow(1 - cos_theta * cos_theta, 0.5)
    RotationMartix = mat2x2([[cos_theta, -sin_theta],
                             [sin_theta, cos_theta]])
    return RotationMartix


@ti.pyfunc
def ThetaToRotationMatrix(theta):
    contentInRad = theta * PI / 180
    rotateX = ti.Matrix([[1., 0., 0.],
                        [0, ti.cos(contentInRad[0]), -ti.sin(contentInRad[0])],
                        [0, ti.sin(contentInRad[0]), ti.cos(contentInRad[0])]])
    rotateY = ti.Matrix([[ti.cos(contentInRad[1]), 0, ti.sin(contentInRad[1])],
                        [0., 1., 0.],
                        [-ti.sin(contentInRad[1]), 0, ti.cos(contentInRad[1])]])
    rotateZ = ti.Matrix([[ti.cos(contentInRad[2]), -ti.sin(contentInRad[2]), 0],
                        [ti.sin(contentInRad[2]), ti.cos(contentInRad[2]), 0],
                        [0., 0., 1.]])
    return rotateZ @ rotateY @ rotateX

@ti.pyfunc
def ThetaToRotationMatrix2D(theta):
    angle_rad = theta * PI / 180
    R = ti.Matrix([[ti.cos(angle_rad), -ti.sin(angle_rad)],
                   [ti.sin(angle_rad),  ti.cos(angle_rad)]])
    return R
