"""Device scalar recurrences shared by scalar and block Krylov solvers.

Matvec stays a Python callback so existing matrix-free operators still work.
Only convergence/breakdown status crosses the host boundary inside iterations.
"""

from math import isfinite, sqrt
import os

import taichi as ti

from src.linear_solver.LinearOperator import LinearOperator


def device_krylov_enabled():
    return ti.lang.impl.current_cfg().arch == ti.cuda and os.environ.get("GEOTAICHI_KRYLOV_DEVICE", "1") != "0"


@ti.data_oriented
class DeviceKrylov:
    def __init__(self, owner, bicgstab=False, block=False):
        self.owner = owner
        self.block = block
        self.bicgstab = bicgstab
        self.p, self.r, self.Ax = owner.p, owner.r, owner.Ax
        self.z = getattr(owner, "z", self.r)
        self.Ap = owner.Ap if not bicgstab else (owner.v if block else getattr(owner, "Ay", None))
        if bicgstab:
            if self.Ap is None:
                self.Ap = owner.Ap
            self.As = owner.t if block else (owner.Az if hasattr(owner, "Az") else owner.As)
            self.shadow = owner.r_hat if block else owner.r_tld
            self.s = owner.s
            self.h = None if block else owner.h
            self.phat = owner.p_hat if block else getattr(owner, "y", self.p)
            self.shat = owner.s_hat if block else (owner.z if hasattr(owner, "y") else self.s)
        if block:
            from src.linear_solver.BuildTriplet import _dense_block_matvec, _dense_block_transpose_matvec

            self.block_apply = _dense_block_matvec
            self.block_apply_transpose = _dense_block_transpose_matvec
        # alpha, beta, omega, previous recurrence product, breakdown code.
        self.coefficients = ti.Vector.field(5, float, shape=())
        self.rr = ti.field(float, shape=())
        self.product = ti.field(float, shape=())
        self.denominator = ti.field(float, shape=())
        self.numerator = ti.field(float, shape=())
        self.status_reads = 0

    @ti.func
    def dot(self, a, b):
        value = 0.0
        if ti.static(self.block):
            value = a.dot(b)
        else:
            value = a * b
        return value

    @ti.func
    def precondition(self, i, value, M: ti.template(), transpose: ti.template()):
        result = value
        if ti.static(self.block):
            if ti.static(transpose):
                result = self.block_apply_transpose(self.owner.diag_inverse[i], value, ti.static(self.owner.dim))
            else:
                result = self.block_apply(self.owner.diag_inverse[i], value, ti.static(self.owner.dim))
        elif ti.static(M != None):
            result = value / M[i]
        return result

    @ti.func
    def invalid(self, value):
        return ti.math.isnan(value) or ti.math.isinf(value)

    @ti.kernel
    def initialize_cg(
        self, size: int, b: ti.template(), M: ti.template(), initial: ti.template(), work_items: ti.template()
    ):
        if ti.static(initial):
            self.coefficients[None] = ti.Vector.zero(float, 5)
        self.rr[None] = 0.0
        self.product[None] = 0.0
        for i in range(work_items):
            if i < size:
                residual = b[i] - self.Ax[i]
                z = self.precondition(i, residual, M, False)
                self.r[i], self.z[i], self.p[i] = residual, z, z
                self.rr[None] += self.dot(residual, residual)
                self.product[None] += self.dot(residual, z)
        self.coefficients[None][3] = self.product[None]
        self.coefficients[None][4] = 0.0

    @ti.kernel
    def cg_step(self, size: int, x: ti.template(), M: ti.template(), work_items: ti.template()):
        self.denominator[None] = 0.0
        for i in range(work_items):
            if i < size:
                self.denominator[None] += self.dot(self.p[i], self.Ap[i])
        if self.coefficients[None][4] == 0.0:
            denom = self.denominator[None]
            if self.invalid(denom) or denom <= 0.0:
                self.coefficients[None][4] = 1.0
            else:
                self.coefficients[None][0] = self.coefficients[None][3] / denom
                if self.invalid(self.coefficients[None][0]):
                    self.coefficients[None][4] = 2.0
        if self.coefficients[None][4] == 0.0:
            self.rr[None] = 0.0
            self.product[None] = 0.0
        for i in range(work_items):
            if i < size and self.coefficients[None][4] == 0.0:
                alpha = self.coefficients[None][0]
                x[i] += alpha * self.p[i]
                residual = self.r[i] - alpha * self.Ap[i]
                z = self.precondition(i, residual, M, False)
                self.r[i], self.z[i] = residual, z
                self.rr[None] += self.dot(residual, residual)
                self.product[None] += self.dot(residual, z)

    @ti.kernel
    def initialize_cg_external(self, size: int, b: ti.template(), initial: ti.template()):
        if ti.static(initial):
            self.coefficients[None] = ti.Vector.zero(float, 5)
        for i in range(size):
            self.r[i] = b[i] - self.Ax[i]

    @ti.kernel
    def reduce_cg_external(self, size: int, restart: ti.template(), work_items: ti.template()):
        self.rr[None] = 0.0
        self.product[None] = 0.0
        for i in range(work_items):
            if i < size:
                self.rr[None] += self.dot(self.r[i], self.r[i])
                self.product[None] += self.dot(self.r[i], self.z[i])
                if ti.static(restart):
                    self.p[i] = self.z[i]
        if ti.static(restart):
            self.coefficients[None][3] = self.product[None]
            self.coefficients[None][4] = 0.0

    @ti.kernel
    def cg_step_external(self, size: int, x: ti.template(), work_items: ti.template()):
        self.denominator[None] = 0.0
        for i in range(work_items):
            if i < size:
                self.denominator[None] += self.dot(self.p[i], self.Ap[i])
        if self.coefficients[None][4] == 0.0:
            denominator = self.denominator[None]
            if self.invalid(denominator) or denominator <= 0.0:
                self.coefficients[None][4] = 1.0
            else:
                self.coefficients[None][0] = self.coefficients[None][3] / denominator
                if self.invalid(self.coefficients[None][0]):
                    self.coefficients[None][4] = 2.0
        for i in range(size):
            if self.coefficients[None][4] == 0.0:
                alpha = self.coefficients[None][0]
                x[i] += alpha * self.p[i]
                self.r[i] -= alpha * self.Ap[i]

    @ti.kernel
    def cg_direction(self, size: int):
        product = self.product[None]
        old = self.coefficients[None][3]
        if self.invalid(product) or product <= 0.0:
            self.coefficients[None][4] = 3.0
        else:
            beta = product / old
            self.coefficients[None][1] = beta
            self.coefficients[None][3] = product
            if self.invalid(beta):
                self.coefficients[None][4] = 4.0
        for i in range(size):
            if self.coefficients[None][4] == 0.0:
                self.p[i] = self.z[i] + self.coefficients[None][1] * self.p[i]

    @ti.kernel
    def initialize_bicg(self, size: int, b: ti.template(), initial: ti.template(), work_items: ti.template()):
        if ti.static(initial):
            self.coefficients[None] = ti.Vector.zero(float, 5)
        self.rr[None] = 0.0
        self.product[None] = 0.0
        for i in range(work_items):
            if i < size:
                residual = b[i] - self.Ax[i]
                self.r[i], self.p[i] = residual, residual
                if ti.static(initial or self.block):
                    self.shadow[i] = residual
                self.Ap[i] = 0.0 * residual
                self.As[i] = 0.0 * residual
                self.rr[None] += self.dot(residual, residual)
                self.product[None] += self.dot(residual, self.shadow[i])
        self.coefficients[None][3] = self.product[None]
        self.coefficients[None][4] = 0.0

    @ti.kernel
    def prepare(
        self, size: int, source: ti.template(), dest: ti.template(), M: ti.template(), transpose: ti.template()
    ):
        for i in range(size):
            dest[i] = self.precondition(i, source[i], M, transpose)

    @ti.kernel
    def bicg_alpha_step(self, size: int, x: ti.template(), work_items: ti.template()):
        self.denominator[None] = 0.0
        for i in range(work_items):
            if i < size:
                self.denominator[None] += self.dot(self.shadow[i], self.Ap[i])
        if self.coefficients[None][4] == 0.0:
            rho, denom = self.coefficients[None][3], self.denominator[None]
            if self.invalid(rho) or rho == 0.0:
                self.coefficients[None][4] = 5.0
            elif self.invalid(denom) or denom == 0.0:
                self.coefficients[None][4] = 6.0
            else:
                self.coefficients[None][0] = rho / denom
                if self.invalid(self.coefficients[None][0]):
                    self.coefficients[None][4] = 2.0
        if self.coefficients[None][4] == 0.0:
            self.rr[None] = 0.0
        for i in range(work_items):
            if i < size and self.coefficients[None][4] == 0.0:
                alpha = self.coefficients[None][0]
                if ti.static(not self.block):
                    self.h[i] = x[i] + alpha * self.phat[i]
                value = self.r[i] - alpha * self.Ap[i]
                self.s[i] = value
                self.rr[None] += self.dot(value, value)

    @ti.kernel
    def accept_alpha(self, size: int, x: ti.template()):
        for i in range(size):
            if ti.static(self.block):
                x[i] += self.coefficients[None][0] * self.phat[i]
            else:
                x[i] = self.h[i]

    @ti.kernel
    def bicg_omega_step(self, size: int, x: ti.template(), work_items: ti.template()):
        self.numerator[None] = 0.0
        self.denominator[None] = 0.0
        for i in range(work_items):
            if i < size:
                self.numerator[None] += self.dot(self.As[i], self.s[i])
                self.denominator[None] += self.dot(self.As[i], self.As[i])
        denom = self.denominator[None]
        if self.invalid(denom) or denom <= 0.0:
            self.coefficients[None][4] = 7.0
        else:
            omega = self.numerator[None] / denom
            self.coefficients[None][2] = omega
            if self.invalid(omega) or (omega == 0.0 and ti.static(not self.block)):
                self.coefficients[None][4] = 8.0
        if self.coefficients[None][4] == 0.0:
            self.rr[None] = 0.0
            self.product[None] = 0.0
        for i in range(work_items):
            if i < size and self.coefficients[None][4] == 0.0:
                alpha, omega = self.coefficients[None][0], self.coefficients[None][2]
                if ti.static(self.block):
                    x[i] += alpha * self.phat[i] + omega * self.shat[i]
                else:
                    x[i] = self.h[i] + omega * self.shat[i]
                residual = self.s[i] - omega * self.As[i]
                self.r[i] = residual
                self.rr[None] += self.dot(residual, residual)
                self.product[None] += self.dot(residual, self.shadow[i])
        if ti.static(self.block):
            if self.coefficients[None][2] == 0.0 and self.coefficients[None][4] == 0.0:
                self.coefficients[None][4] = 8.0

    @ti.kernel
    def bicg_direction(self, size: int):
        rho, old = self.product[None], self.coefficients[None][3]
        if self.invalid(rho) or rho == 0.0:
            self.coefficients[None][4] = 5.0
        else:
            beta = (rho / old) * (self.coefficients[None][0] / self.coefficients[None][2])
            self.coefficients[None][1] = beta
            self.coefficients[None][3] = rho
            if self.invalid(beta):
                self.coefficients[None][4] = 4.0
        for i in range(size):
            if self.coefficients[None][4] == 0.0:
                self.p[i] = self.r[i] + self.coefficients[None][1] * (
                    self.p[i] - self.coefficients[None][2] * self.Ap[i]
                )

    @ti.kernel
    def _read_status(self) -> ti.types.vector(3, ti.f64):
        return ti.Vector([self.rr[None], self.product[None], self.coefficients[None][4]])

    @ti.kernel
    def _read_coefficients(self) -> ti.types.vector(5, ti.f64):
        return self.coefficients[None]

    def status(self):
        self.status_reads += 1
        rr, product, error = self._read_status()
        if error:
            self.owner.last_breakdown_reason = {
                1: "non_positive_curvature",
                2: "non_finite_alpha",
                3: "non_positive_preconditioned_residual",
                4: "non_finite_beta",
                5: "zero_shadow_residual_product",
                6: "zero_alpha_denominator",
                7: "zero_omega_denominator",
                8: "zero_or_non_finite_omega",
            }[int(error)]
        elif not isfinite(rr) or rr < 0.0:
            self.owner.last_residual = float("inf")
            self.owner.last_breakdown_reason = "non_finite_residual"
            error = -1
        else:
            self.owner.last_residual = sqrt(rr)
        return product, error

    def solve(self, A, b, x, M, size, tol, maxiter, rel_tol, transpose=False, reliable_interval=0):
        owner = self.owner
        owner.last_iterations = owner.last_residual_restarts = self.status_reads = 0
        owner.last_breakdown_reason = ""
        owner.last_converged = False
        A.matvec(x, self.Ax)
        initialize = self.initialize_bicg if self.bicgstab else self.initialize_cg
        external_preconditioner = isinstance(M, LinearOperator)
        # A bounded range avoids CUDA launching empty reduction blocks; bucket sizes limit specialization.
        work_items = 1 << max(0, (int(size) - 1).bit_length())

        def initialize_residual(initial):
            if external_preconditioner and not self.bicgstab:
                self.initialize_cg_external(size, b, initial)
                M.matvec(self.r, self.z)
                self.reduce_cg_external(size, True, work_items)
            elif self.bicgstab:
                initialize(size, b, initial, work_items)
            else:
                initialize(size, b, M, initial, work_items)
            return self.status()

        product, error = initialize_residual(True)
        owner.last_initial_residual = owner.last_residual
        threshold = max(tol, rel_tol * owner.last_initial_residual)
        if not error and owner.last_residual <= threshold:
            owner.last_converged = True
        elif not self.bicgstab and (not isfinite(product) or product <= 0.0):
            owner.last_breakdown_reason = "non_positive_preconditioned_residual"
            error = True

        for iteration in range(maxiter):
            if error or owner.last_converged:
                break
            if self.bicgstab:
                if external_preconditioner:
                    M.matvec(self.p, self.phat)
                elif M is not None or self.block:
                    self.prepare(size, self.p, self.phat, M, transpose)
                A.matvec(self.phat, self.Ap)
                self.bicg_alpha_step(size, x, work_items)
                _, error = self.status()
                if error:
                    break
                owner.last_iterations = iteration + 1
                if owner.last_residual <= threshold:
                    self.accept_alpha(size, x)
                    verify = True
                else:
                    if external_preconditioner:
                        M.matvec(self.s, self.shat)
                    elif M is not None or self.block:
                        self.prepare(size, self.s, self.shat, M, transpose)
                    A.matvec(self.shat, self.As)
                    self.bicg_omega_step(size, x, work_items)
                    _, error = self.status()
                    verify = False
            else:
                A.matvec(self.p, self.Ap)
                if external_preconditioner:
                    self.cg_step_external(size, x, work_items)
                    M.matvec(self.r, self.z)
                    self.reduce_cg_external(size, False, work_items)
                else:
                    self.cg_step(size, x, M, work_items)
                _, error = self.status()
                if error:
                    break
                owner.last_iterations = iteration + 1
                verify = False
            if error:
                break
            verify = (
                verify
                or owner.last_residual <= threshold
                or (reliable_interval and (iteration + 1) % reliable_interval == 0)
            )
            if verify:
                A.matvec(x, self.Ax)
                product, error = initialize_residual(False)
                if not error and owner.last_residual <= threshold:
                    owner.last_converged = True
                elif not error:
                    owner.last_residual_restarts += 1
                    invalid_product = not isfinite(product) or (product == 0.0 if self.bicgstab else product <= 0.0)
                    if invalid_product:
                        owner.last_breakdown_reason = (
                            "zero_shadow_residual_product" if self.bicgstab else "non_positive_preconditioned_residual"
                        )
                        error = True
            elif self.bicgstab:
                self.bicg_direction(size)
            else:
                self.cg_direction(size)

        self.recursive_residual = owner.last_residual
        if not owner.last_converged:
            # A pending direction breakdown is checked even at the iteration limit.
            self.status()
            A.matvec(x, self.Ax)
            _, error = initialize_residual(False)
            owner.last_converged = not error and owner.last_residual <= threshold
        if owner.last_converged:
            owner.last_breakdown_reason = ""
        elif not owner.last_breakdown_reason:
            owner.last_breakdown_reason = "maximum_iterations"
        coefficients = self._read_coefficients()
        for name, value in zip(("alpha", "beta", "omega", "rho_1"), coefficients[:4]):
            if hasattr(owner, name):
                setattr(owner, name, float(value))
        if hasattr(owner, "rho"):
            owner.rho = float(coefficients[3])
        return owner.last_converged
