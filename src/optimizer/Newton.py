import taichi as ti


@ti.data_oriented
class Newton:
    """
    Augmented Lagrangian method with Newton's method for solving constrained optimization problems
    Solves: min f(x) subject to g(x) <= 0
    """
    def __init__(self, objective_func, objective_grad, objective_hess, n_vars=1, tolerance=1e-6, max_iterations=100, newton_max_iter=50, rho_update_factor=10.0):
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
        self.f = ti.func(objective_func)
        self.f_grad = ti.func(objective_grad)
        self.f_hess = ti.func(objective_hess)
        self.n_vars = n_vars

        if self.n_vars > 20:
            raise RuntimeError("Only support variables less than 20.")
        
        # Convergence parameters
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.newton_max_iter = newton_max_iter
        self.rho_update_factor = rho_update_factor
    
    @ti.func
    def polar_decomposition(self, mat):
        L = ti.Matrix.zero(float, mat.n, mat.m)
        U = ti.Matrix.zero(float, mat.n, mat.m)
        for i in ti.static(range(mat.n)):
            L[i, i] = 1.
        for j in ti.static(range(mat.n)):
            U[0, j] = mat[0, j]
        for i in ti.static(range(1, mat.n)):
            L[i, 0] = mat[i, 0] / U[0, 0]
        for i in ti.static(range(mat.n)):
            for j in ti.static(range(i, mat.n)):
                s = 0.
                for k in ti.static(range(i)):
                    s += L[i, k] * U[k, j]
                U[i, j] = mat[i, j] - s
            for d in ti.static(range(i, mat.n)):
                s = 0.
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
        sum = 0.
        for i in ti.static(range(vec.n)):
            sum += vec[i] * vec[i]
        return sum
    
    @ti.func
    def mat_squ_norm(self, mat):
        sum = 0.
        for i in ti.static(range(mat.n)):
            for j in ti.static(range(mat.m)):
                sum += mat[i, j] * mat[i, j]
        return sum
    
    @ti.func
    def solve(self, x0):
        """
        Newton's method for solving the augmented Lagrangian subproblem
        with line search to prevent bounds violation
        """
        x = x0
        for _ in range(self.newton_max_iter):
            grad = self.f_grad(x)
            hess = self.f_hess(x)
            
            # Check gradient convergence
            if ti.static(self.n_vars == 1):
                if abs(grad) * abs(grad) < self.tolerance:
                    break
            else:
                if self.vec_squ_norm(grad) < self.tolerance:
                    break
                
            # Newton step: x_{k+1} = x_k - H^{-1} * grad
            # For 1D problem, Hessian is a scalar
            step = 0.
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
            x_new = x + alpha * step
            
            # Then apply Armijo line search condition
            # f(x + alpha*p) <= f(x) + c1 * alpha * grad^T * p
            c1 = 1e-4  # Armijo parameter
            f_current = self.f(x)
            armijo_condition = f_current + c1 * alpha * grad * step
            
            iteration_count = 0
            while (self.f(x_new) > armijo_condition) and alpha > 1e-8 and iteration_count < 100:
                alpha *= 0.5
                x_new = x + alpha * step
                armijo_condition = f_current + c1 * alpha * grad * step
                iteration_count += 1
            
            # If step size becomes too small, accept current step
            if alpha < 1e-8:
                x_new = x + 1e-8 * step
            x = x_new
            
        return x