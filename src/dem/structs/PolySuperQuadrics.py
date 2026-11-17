import taichi as ti
from src.utils.constants import THRESHOLD
from src.utils.ScalarFunction import sgn, sgn
from src.utils.TypeDefinition import vec3f, mat3x3


@ti.dataclass
class PolySuperQuadrics:
    xrad1: float
    yrad1: float
    zrad1: float
    epsilon1: float
    epsilon2: float
    epsilon3: float
    xrad2: float
    yrad2: float
    zrad2: float

    @ti.func
    def _add_template_parameter(self, xrad1, yrad1, zrad1, epsilon_x, epsilon_y, epsilon_z, xrad2, yrad2, zrad2):
        self.xrad1 = xrad1
        self.yrad1 = yrad1
        self.zrad1 = zrad1
        self.epsilon1 = 2.0 / epsilon_x
        self.epsilon2 = 2.0 / epsilon_y
        self.epsilon3 = 2.0 / epsilon_z
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
                self.epsilon3,
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
                2.0 + fraction * (self.epsilon3 - 2.0),
            ]
        )

    @ti.func
    def get_param(self, x, y, z, parameters):
        xrad = parameters[0] if x < 0.0 else parameters[3]
        yrad = parameters[1] if y < 0.0 else parameters[4]
        zrad = parameters[2] if z < 0.0 else parameters[5]
        return xrad, yrad, zrad

    @ti.func
    def fx(self, x, y, z, params):
        xrad, yrad, zrad = self.get_param(x, y, z, params)
        funcs = (
            ti.pow(ti.abs(x / xrad), params[6])
            + ti.pow(ti.abs(y / yrad), params[7])
            + ti.pow(ti.abs(z / zrad), params[8])
            - 1.0
        )
        return funcs

    @ti.func
    def gradient(self, x, y, z, params):
        xrad, yrad, zrad = self.get_param(x, y, z, params)
        x0 = ti.abs(x / xrad)
        x1 = ti.abs(y / yrad)
        x2 = ti.abs(z / zrad)
        x5 = x0 ** (params[6] - 1.0)
        x6 = x1 ** (params[7] - 1.0)
        x7 = x2 ** (params[8] - 1.0)
        return vec3f(params[6] / xrad * x5 * sgn(x), params[7] / yrad * x6 * sgn(y), params[8] / zrad * x7 * sgn(z))

    @ti.func
    def hessian(self, x, y, z, params):
        xrad, yrad, zrad = self.get_param(x, y, z, params)
        inv_xrad2 = 1.0 / (xrad * xrad)
        inv_yrad2 = 1.0 / (yrad * yrad)
        inv_zrad2 = 1.0 / (zrad * zrad)
        x0 = ti.abs(x / xrad)
        x1 = ti.abs(y / yrad)
        x2 = ti.abs(z / zrad)
        x3 = x0 ** (params[6] - 2.0)
        x4 = x1 ** (params[7] - 2.0)
        x5 = x2 ** (params[8] - 2.0)
        n1n1 = params[6] * (params[6] - 1.0)
        n2n2 = params[7] * (params[7] - 1.0)
        n3n3 = params[8] * (params[8] - 1.0)
        fxx = inv_xrad2 * n1n1 * x3
        fyy = inv_yrad2 * n2n2 * x4
        fzz = inv_zrad2 * n3n3 * x5
        return mat3x3([fxx, 0.0, 0.0], [0.0, fyy, 0.0], [0.0, 0.0, fzz])

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
        # Maximise n.x subject to
        #   sum_i (|x_i| / a_i)^p_i = 1.
        # For p_i > 1, the KKT conditions give
        #   |x_i|/a_i = (|n_i| a_i / (lambda p_i))^(1/(p_i-1)).
        # Different axis exponents share the scalar multiplier ``lambda``;
        # solve its monotone equation with a small, deterministic bisection.
        radii = ti.Vector.zero(float, 3)
        radii[0], radii[1], radii[2] = self.get_param(*local_normal, params)

        support_point = vec3f(0.0, 0.0, 0.0)
        if local_normal.norm_sqr() > THRESHOLD * THRESHOLD:
            lambda_hi = THRESHOLD
            for d in ti.static(range(3)):
                exponent = params[6 + d]
                coefficient = ti.abs(local_normal[d]) * radii[d] / exponent
                lambda_hi = ti.max(lambda_hi, coefficient)

            # ``lambda_hi=max(coefficient)`` leaves every summand <= 1,
            # although their sum can still exceed one. Enlarge the bracket
            # until its upper end is outside the root.
            for _ in ti.static(range(8)):
                constraint = 0.0
                for d in ti.static(range(3)):
                    exponent = params[6 + d]
                    coefficient = ti.abs(local_normal[d]) * radii[d] / exponent
                    if coefficient > 0.0:
                        dual_exponent = exponent / (exponent - 1.0)
                        constraint += ti.pow(coefficient / lambda_hi, dual_exponent)
                if constraint > 1.0:
                    lambda_hi *= 2.0

            lambda_lo = 0.0
            for _ in range(40):
                lambda_mid = 0.5 * (lambda_lo + lambda_hi)
                constraint = 0.0
                for d in ti.static(range(3)):
                    exponent = params[6 + d]
                    coefficient = ti.abs(local_normal[d]) * radii[d] / exponent
                    if coefficient > 0.0:
                        dual_exponent = exponent / (exponent - 1.0)
                        constraint += ti.pow(coefficient / lambda_mid, dual_exponent)
                if constraint > 1.0:
                    lambda_lo = lambda_mid
                else:
                    lambda_hi = lambda_mid

            multiplier = lambda_hi
            for d in ti.static(range(3)):
                exponent = params[6 + d]
                coefficient = ti.abs(local_normal[d]) * radii[d] / exponent
                if coefficient > 0.0:
                    support_point[d] = (
                        sgn(local_normal[d]) * radii[d] * ti.pow(coefficient / multiplier, 1.0 / (exponent - 1.0))
                    )
        return support_point

    @ti.func
    def plane_minimum(self, local_normal, plane_offset, params):
        """Minimise this separable implicit potential on a plane.

        The three independent powers mean the level sets are not homothetic,
        so the superellipsoid ray construction is not valid here.  The KKT
        equations still share one positive multiplier, which is solved by a
        deterministic scalar bisection.
        """
        point = vec3f(0.0, 0.0, 0.0)
        target = ti.abs(plane_offset)
        if target > THRESHOLD:
            direction = sgn(plane_offset) * local_normal
            radii = ti.Vector.zero(float, 3)
            radii[0], radii[1], radii[2] = self.get_param(*direction, params)

            lambda_hi = 1.0
            for _ in ti.static(range(32)):
                constraint = 0.0
                for d in ti.static(range(3)):
                    exponent = params[6 + d]
                    coefficient = ti.abs(local_normal[d]) * radii[d] / exponent
                    if coefficient > 0.0:
                        constraint += (
                            ti.abs(local_normal[d])
                            * radii[d]
                            * ti.pow(
                                lambda_hi * coefficient,
                                1.0 / (exponent - 1.0),
                            )
                        )
                if constraint < target:
                    lambda_hi *= 2.0

            lambda_lo = 0.0
            for _ in range(48):
                multiplier = 0.5 * (lambda_lo + lambda_hi)
                constraint = 0.0
                for d in ti.static(range(3)):
                    exponent = params[6 + d]
                    coefficient = ti.abs(local_normal[d]) * radii[d] / exponent
                    if coefficient > 0.0:
                        constraint += (
                            ti.abs(local_normal[d])
                            * radii[d]
                            * ti.pow(
                                multiplier * coefficient,
                                1.0 / (exponent - 1.0),
                            )
                        )
                if constraint < target:
                    lambda_lo = multiplier
                else:
                    lambda_hi = multiplier

            multiplier = lambda_hi
            for d in ti.static(range(3)):
                exponent = params[6 + d]
                coefficient = ti.abs(local_normal[d]) * radii[d] / exponent
                if coefficient > 0.0:
                    point[d] = (
                        sgn(direction[d])
                        * radii[d]
                        * ti.pow(
                            multiplier * coefficient,
                            1.0 / (exponent - 1.0),
                        )
                    )
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
