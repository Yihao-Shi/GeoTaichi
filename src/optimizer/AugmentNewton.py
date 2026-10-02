import taichi as ti
import numpy as np


@ti.data_oriented
class AugmentedLagrangianNewton:
    """
    Augmented Lagrangian method with Newton's method for solving constrained optimization problems
    Solves: min f(x) subject to g(x) <= 0
    """

    def __init__(
        self,
        objective_func,
        objective_grad,
        objective_hess,
        constraints,
        constraints_grad,
        constraints_hess=None,
        n_vars=1,
        tolerance=1e-6,
        max_iterations=100,
        newton_max_iter=50,
        rho_update_factor=10.0,
    ):
        """
        Initialize the solver

        Parameters:
        objective_func: objective function f(x)
        objective_grad: gradient of objective function f'(x)
        objective_hess: Hessian of objective function f''(x)
        constraints: list of constraint functions g_i(x) <= 0
        constraints_grad: list of constraint gradients g_i'(x)
        x0: initial point
        """
        constraints = list(constraints)
        constraints_grad = list(constraints_grad)
        self.f = ti.func(objective_func)
        self.f_grad = ti.func(objective_grad)
        self.f_hess = ti.func(objective_hess)
        self.constraints = [ti.func(constraint) for constraint in constraints]
        self.constraints_grad = [ti.func(constraint_grad) for constraint_grad in constraints_grad]
        self.n_constraints = len(constraints)
        self.n_vars = n_vars

        if constraints_hess is None:
            self.constraints_hess = [self.constraint_hessian for _ in range(self.n_constraints)]
        else:
            constraints_hess = list(constraints_hess)
            self.constraints_hess = [ti.func(constraint_hess) for constraint_hess in constraints_hess]

        if self.n_vars > 20:
            raise RuntimeError("Only support variables less than 20.")

        self.rho = 10.0
        self.lambda_k = np.zeros(self.n_constraints)

        # Convergence parameters
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.newton_max_iter = newton_max_iter
        self.rho_update_factor = rho_update_factor

    @ti.func
    def constraint_hessian(self, x):
        return 0.0

    @ti.func
    def augmented_lagrangian(self, x, lambda_k, rho):
        """
        Augmented Lagrangian function
        L(x, lambda, rho) = f(x) + sum[lambda_i * max(0, g_i(x)) + rho/2 * max(0, g_i(x))^2]
        """
        L = self.f(x)

        for i in ti.static(range(self.n_constraints)):
            g_val = self.constraints[i](x)
            term = max(0, g_val + lambda_k[i] / rho)
            L += 0.5 * rho * term * term

        return L

    @ti.func
    def augmented_lagrangian_grad(self, x, lambda_k, rho):
        """
        Gradient of the augmented Lagrangian function
        """
        grad = self.f_grad(x)

        for i in ti.static(range(self.n_constraints)):
            g_val = self.constraints[i](x)
            term = g_val + lambda_k[i] / rho

            if term > 0:
                grad += rho * term * self.constraints_grad[i](x)

        return grad

    @ti.func
    def augmented_lagrangian_hess(self, x, lambda_k, rho):
        """
        Hessian of the augmented Lagrangian function
        """
        hess = self.f_hess(x)

        for i in ti.static(range(self.n_constraints)):
            g_val = self.constraints[i](x)
            term = g_val + lambda_k[i] / rho

            if term > 0:  # Only contribute when max(0, term) > 0
                g_grad_val = self.constraints_grad[i](x)
                g_hess_val = self.constraints_hess[i](x)
                if ti.static(self.n_vars == 1):
                    hess += rho * (g_grad_val * g_grad_val + term * g_hess_val)  # Outer product, but scalar here
                else:
                    hess += rho * (g_grad_val.outer_product(g_grad_val) + term * g_hess_val)

        return hess

    @ti.func
    def polar_decomposition(self, mat):
        L = ti.Matrix.zero(float, mat.n, mat.m)
        U = ti.Matrix.zero(float, mat.n, mat.m)
        for i in ti.static(range(mat.n)):
            L[i, i] = 1.0
        for j in ti.static(range(mat.n)):
            U[0, j] = mat[0, j]
        for i in ti.static(range(1, mat.n)):
            L[i, 0] = mat[i, 0] / U[0, 0]
        for i in ti.static(range(mat.n)):
            for j in ti.static(range(i, mat.n)):
                s = 0.0
                for k in ti.static(range(i)):
                    s += L[i, k] * U[k, j]
                U[i, j] = mat[i, j] - s
            for d in ti.static(range(i, mat.n)):
                s = 0.0
                for k in ti.static(range(i)):
                    s += L[d, k] * U[k, i]
                L[d, i] = (mat[d, i] - s) / U[i, i]
        return L, U

    @ti.func
    def forward_substitution(self, L, b):
        y = ti.Vector.zero(float, L.n)
        for i in ti.static(range(L.n)):
            sum = 0.0
            for j in ti.static(range(L.m)):
                sum += L[i, j] * y[j]
            y[i] = b[i] - sum
        return y

    @ti.func
    def backward_substitution(self, U, y):
        x = ti.Vector.zero(float, U.n)
        for i in ti.static(range(U.n)):
            sum = 0.0
            for j in ti.static(range(U.n - i, U.m)):
                sum += U[U.n - i - 1, j] * x[j]
            x[U.n - i - 1] = (y[U.n - i - 1] - sum) / U[U.n - i - 1, U.n - i - 1]
        return x

    @ti.func
    def vec_squ_norm(self, vec):
        sum = 0.0
        for i in ti.static(range(vec.n)):
            sum += vec[i] * vec[i]
        return sum

    @ti.func
    def mat_squ_norm(self, mat):
        sum = 0.0
        for i in ti.static(range(mat.n)):
            for j in ti.static(range(mat.m)):
                sum += mat[i, j] * mat[i, j]
        return sum

    @ti.func
    def newton_method(self, x0, lambda_k, rho):
        """
        Newton's method for solving the augmented Lagrangian subproblem
        with line search to prevent bounds violation
        """
        x = x0
        for _ in range(self.newton_max_iter):
            grad = self.augmented_lagrangian_grad(x, lambda_k, rho)
            hess = self.augmented_lagrangian_hess(x, lambda_k, rho)

            # Check gradient convergence
            if ti.static(self.n_vars == 1):
                if abs(grad) * abs(grad) < self.tolerance:
                    break
            else:
                if self.vec_squ_norm(grad) < self.tolerance:
                    break

            # Newton step: x_{k+1} = x_k - H^{-1} * grad
            # For 1D problem, Hessian is a scalar
            step = 0.0
            if ti.static(self.n_vars == 1):
                if abs(hess) * abs(hess) < 1e-12:  # Avoid division by zero
                    # If Hessian is near zero, use gradient descent
                    step = -grad * 0.1
                else:
                    step = -grad / hess
            elif ti.static(2 <= self.n_vars <= 4):
                if self.mat_squ_norm(hess) < 1e-12:  # Avoid division by zero
                    # If Hessian is near zero, use gradient descent
                    step = -grad * 0.1
                else:
                    step = -hess.inverse() @ grad
            else:
                if self.mat_squ_norm(hess) < 1e-12:  # Avoid division by zero
                    # If Hessian is near zero, use gradient descent
                    step = -grad * 0.1
                else:
                    L, U = self.polar_decomposition(hess)
                    y = self.forward_substitution(L, -grad)
                    step = self.backward_substitution(U, y)

            # Line search with bounds checking
            alpha = 1.0
            if ti.static(self.n_vars == 1):
                alpha = min(alpha, -(1.0 - x) / step if step < 0.0 else x / step)
            else:
                for tt in ti.static(range(x.n)):
                    alpha = min(alpha, -(1.0 - x[tt]) / step[tt] if step[tt] < 0.0 else x[tt] / step[tt])
            x_new = x + alpha * step

            # Then apply Armijo line search condition
            # f(x + alpha*p) <= f(x) + c1 * alpha * grad^T * p
            c1 = 1e-4  # Armijo parameter
            f_current = self.augmented_lagrangian(x, lambda_k, rho)
            armijo_condition = f_current
            if ti.static(self.n_vars == 1):
                armijo_condition += c1 * alpha * grad * step
            else:
                armijo_condition += c1 * alpha * grad.dot(step)

            iteration_count = 0
            while (
                (self.augmented_lagrangian(x_new, lambda_k, rho) > armijo_condition)
                and alpha > 1e-8
                and iteration_count < 100
            ):
                alpha *= 0.5
                x_new = x + alpha * step
                iteration_count += 1

            # If step size becomes too small, accept current step
            if alpha < 1e-8:
                x_new = x + 1e-8 * step
            x = x_new

        return x

    @ti.func
    def solve(self, x0):
        x_current = x0
        x_new = x0
        rho = self.rho
        lambda_k = ti.Vector(self.lambda_k)
        for _ in range(self.max_iterations):
            # Step 1: Minimize augmented Lagrangian using Newton's method
            x_new = self.newton_method(x_current, lambda_k, rho)

            # Step 2: Update Lagrange multipliers
            max_violation = 0.0
            prev_violation = 0.0

            for i in ti.static(range(self.n_constraints)):
                g_val = self.constraints[i](x_new)
                g_val0 = self.constraints[i](x_current)
                lambda_k[i] = max(0, lambda_k[i] + rho * g_val)
                max_violation = max(max_violation, abs(max(0, g_val)))
                prev_violation = max(prev_violation, abs(max(0, g_val0)))

            # Step 3: Check convergence
            if ti.static(self.n_vars == 1):
                if max_violation < self.tolerance and abs(x_new - x_current) * abs(x_new - x_current) < self.tolerance:
                    break
            else:
                if max_violation < self.tolerance and self.vec_squ_norm(x_new - x_current) < self.tolerance:
                    break

            # Step 4: Update parameters
            if max_violation > 0.25 * prev_violation:
                rho = ti.min(1e5, rho * self.rho_update_factor)

            x_current = x_new

        return x_new
