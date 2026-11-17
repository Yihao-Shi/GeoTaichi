import taichi as ti
from src.utils.constants import THRESHOLD, DBL_EPSILON
from src.utils.ScalarFunction import sgn, sgn
from src.utils.TypeDefinition import vec3f, mat3x3


@ti.dataclass
class PolySuperEllipsoid:
    xrad1: float
    yrad1: float
    zrad1: float
    epsilon1: float
    epsilon2: float
    xrad2: float
    yrad2: float
    zrad2: float

    @ti.func
    def _add_template_parameter(self, xrad1, yrad1, zrad1, epsilon_e, epsilon_n, xrad2, yrad2, zrad2):
        self.xrad1 = xrad1
        self.yrad1 = yrad1
        self.zrad1 = zrad1
        self.epsilon1 = 2.0 / epsilon_n
        self.epsilon2 = 2.0 / epsilon_e
        self.xrad2 = xrad2
        self.yrad2 = yrad2
        self.zrad2 = zrad2

    @ti.func
    def physical_parameters(self, scale):
        return ti.Vector(
            [
                self.xrad1 * scale,
                self.yrad1 * scale,
                self.zrad1 * scale,
                self.xrad2 * scale,
                self.yrad2 * scale,
                self.zrad2 * scale,
                self.epsilon1,
                self.epsilon2,
            ]
        )

    @ti.func
    def evolving_physical_parameters(self, fraction, bounding_rad, scale):
        return ti.Vector(
            [
                bounding_rad + fraction * (scale * self.xrad1 - bounding_rad),
                bounding_rad + fraction * (scale * self.yrad1 - bounding_rad),
                bounding_rad + fraction * (scale * self.zrad1 - bounding_rad),
                bounding_rad + fraction * (scale * self.xrad2 - bounding_rad),
                bounding_rad + fraction * (scale * self.yrad2 - bounding_rad),
                bounding_rad + fraction * (scale * self.zrad2 - bounding_rad),
                2.0 + fraction * (self.epsilon1 - 2.0),
                2.0 + fraction * (self.epsilon2 - 2.0),
            ]
        )

    @ti.func
    def get_param(self, x, y, z, parameters):
        # ``*rad1`` is the extent on the negative side and ``*rad2`` is the
        # extent on the positive side (see ``SuperSurface``'s bounding box).
        # The old selection was reversed, which mirrored every asymmetric
        # implicit surface and gave the wall kernel the wrong support point.
        xrad = parameters[0] if x < 0.0 else parameters[3]
        yrad = parameters[1] if y < 0.0 else parameters[4]
        zrad = parameters[2] if z < 0.0 else parameters[5]
        return xrad, yrad, zrad

    @ti.func
    def fx(self, x, y, z, params):
        xrad, yrad, zrad = self.get_param(x, y, z, params)
        funcs = (
            ti.pow(ti.pow(ti.abs(x / xrad), params[7]) + ti.pow(ti.abs(y / yrad), params[7]), params[6] / params[7])
            + ti.pow(ti.abs(z / zrad), params[6])
            - 1.0
        )
        return funcs

    @ti.func
    def gradient(self, x, y, z, params):
        xrad, yrad, zrad = self.get_param(x, y, z, params)
        x0 = ti.abs(x / xrad)
        x1 = ti.abs(y / yrad)
        x2 = ti.abs(z / zrad)
        x3 = x0 ** (params[7])
        x4 = x1 ** (params[7])
        x5 = x0 ** (params[7] - 1.0)
        x6 = x1 ** (params[7] - 1.0)
        x7 = x2 ** (params[6] - 1.0)
        mu = (x3 + x4) ** (params[6] / params[7] - 1.0)
        return vec3f(
            params[6] / xrad * x5 * mu * sgn(x), params[6] / yrad * x6 * mu * sgn(y), params[6] / zrad * x7 * sgn(z)
        )

    @ti.func
    def hessian(self, x, y, z, params):
        xrad, yrad, zrad = self.get_param(x, y, z, params)
        inv_xrad2 = 1.0 / (xrad * xrad)
        inv_yrad2 = 1.0 / (yrad * yrad)
        inv_zrad2 = 1.0 / (zrad * zrad)
        inv_xyrad = 1.0 / (xrad * yrad)
        x0 = ti.abs(x / xrad)
        x1 = ti.abs(y / yrad)
        x2 = ti.abs(z / zrad)
        x3 = x0 ** (params[7] - 2.0)
        x4 = x1 ** (params[7] - 2.0)
        x5 = x2 ** (params[6] - 2.0)
        x6 = x0 ** (2.0 * params[7] - 2.0)
        x7 = x1 ** (2.0 * params[7] - 2.0)
        x8 = x0 ** (params[7])
        x9 = x1 ** (params[7])
        x10 = x0 ** (params[7] - 1.0)
        x11 = x1 ** (params[7] - 1.0)

        mu = x8 + x9
        mu0 = mu ** (params[6] / params[7] - 1.0)
        mu1 = mu ** (params[6] / params[7] - 2.0)

        n1n2 = params[6] * (params[7] - 1.0)
        n1n1 = params[6] * (params[6] - 1.0)
        n1n1n2 = params[6] * (params[6] - params[7])

        fxx = inv_xrad2 * (n1n2 * x3 * mu0 + n1n1n2 * x6 * mu1)
        fyy = inv_yrad2 * (n1n2 * x4 * mu0 + n1n1n2 * x7 * mu1)
        fxy = inv_xyrad * n1n1n2 * x10 * x11 * mu1 * sgn(x * y)
        fzz = inv_zrad2 * n1n1 * x5
        return mat3x3([fxx, fxy, 0.0], [fxy, fyy, 0.0], [0.0, 0.0, fzz])

    @ti.func
    def nearest_point(self, local_start_point, local_normal, params):
        t = 0.0
        x = local_start_point[0] + t * local_normal[0]
        y = local_start_point[1] + t * local_normal[1]
        z = local_start_point[2] + t * local_normal[2]

        iter = 0
        fval = self.fx(x, y, z, params)
        while ti.abs(fval) > 1e-12 and iter < 50:
            t -= fval / self.gradient(x, y, z, params).dot(local_normal)

            x = local_start_point[0] + t * local_normal[0]
            y = local_start_point[1] + t * local_normal[1]
            z = local_start_point[2] + t * local_normal[2]
            fval = self.fx(x, y, z, params)
            iter += 1
        return t * local_normal + local_start_point

    @ti.func
    def support(self, local_normal, params):
        xrad, yrad, zrad = self.get_param(*local_normal, params)
        eps1, eps2 = 2.0 / self.epsilon2, 2.0 / self.epsilon1

        cos_phi0, sin_phi0, cos_phi1, sin_phi1 = 0.0, 0.0, 0.0, 0.0
        if ti.abs(local_normal[0]) < DBL_EPSILON:
            cos_phi0 = 0.0
            sin_phi0 = 1.0
            if ti.abs(local_normal[1]) < DBL_EPSILON:
                cos_phi1 = 0.0
                sin_phi1 = 1.0
            else:
                phi1_cos2 = 1.0 / (
                    1.0
                    + pow(
                        ti.abs(zrad * local_normal[2] * pow(sin_phi0, 2.0 - eps1) / yrad / local_normal[1]),
                        2 / (2.0 - eps2),
                    )
                )
                cos_phi1 = pow(phi1_cos2, 0.5)
                sin_phi1 = pow(1.0 - phi1_cos2, 0.5)
        else:
            phi0_cos2 = 1.0 / (1.0 + pow((ti.abs(yrad * local_normal[1] / xrad / local_normal[0])), 2 / (2.0 - eps1)))
            cos_phi0 = pow(phi0_cos2, 0.5)
            sin_phi0 = pow(1.0 - phi0_cos2, 0.5)
            phi1_cos2 = 1.0 / (
                1.0
                + pow(
                    ti.abs(zrad * local_normal[2] * pow(cos_phi0, 2.0 - eps1) / xrad / local_normal[0]),
                    2 / (2.0 - eps2),
                )
            )
            cos_phi1 = pow(phi1_cos2, 0.5)
            sin_phi1 = pow(1.0 - phi1_cos2, 0.5)

        x = sgn(local_normal[0]) * xrad * ti.pow(cos_phi0, eps1) * ti.pow(cos_phi1, eps2)
        y = sgn(local_normal[1]) * yrad * ti.pow(sin_phi0, eps1) * ti.pow(cos_phi1, eps2)
        z = sgn(local_normal[2]) * zrad * ti.pow(sin_phi1, eps2)
        return vec3f([x, y, z])

    @ti.func
    def plane_minimum(self, local_normal, plane_offset, params):
        """Minimise the implicit potential on ``n.x = plane_offset``.

        A superellipsoid has homothetic level sets.  Consequently the
        constrained minimiser lies on the ray through the support point in
        the signed plane-normal direction, giving the lowest-potential
        particle/triangle feature construction.
        """
        point = vec3f(0.0, 0.0, 0.0)
        if ti.abs(plane_offset) > THRESHOLD:
            direction = sgn(plane_offset) * local_normal
            support_point = self.support(direction, params)
            denominator = local_normal.dot(support_point)
            if ti.abs(denominator) > THRESHOLD:
                point = (plane_offset / denominator) * support_point
        return point

    @ti.func
    def mean_curvature(self, p, params):
        grad = self.gradient(p[0], p[1], p[2], params)
        hess = self.hessian(p[0], p[1], p[2], params)
        grad_norm = grad.norm()
        return (
            0.5
            * (grad.T @ hess @ grad - grad_norm * grad_norm * (hess[0, 0] + hess[1, 1] + hess[2, 2]))
            / (grad_norm * grad_norm * grad_norm)
        )

    @ti.func
    def gauss_curvature(self, p, params):
        grad = self.gradient(p[0], p[1], p[2], params)
        hess = self.hessian(p[0], p[1], p[2], params)

        A = grad[2] * (hess[0, 0] * grad[2] - 2.0 * grad[0] * hess[0, 2]) + grad[0] * grad[0] * hess[2, 2]
        B = grad[2] * (hess[1, 1] * grad[2] - 2.0 * grad[1] * hess[1, 2]) + grad[1] * grad[1] * hess[2, 2]
        C = (
            grad[2] * (hess[0, 1] * grad[2] - grad[1] * hess[0, 2] - grad[0] * hess[1, 2])
            + grad[0] * grad[1] * hess[2, 2]
        )
        D = grad[2] * grad[2] * (grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2])
        return ti.sqrt(A * B - C * C) / D
