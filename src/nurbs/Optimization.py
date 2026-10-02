import numpy as np
from scipy.sparse.linalg import cg

from src.linear_solver.ScipyKrylov import checked_scipy_krylov
from src.nurbs.element.Element import Element


def surface_optimize_surface_energy(
    element: Element, fix_dofs=[], tol=1e-5, epsilon=1e-12, iteration=100, verbose=False
):
    DdeformationDx = element.compute_ddeformation_dx()
    deformation_gradient = element.compute_deformation_gradient(element.vertice)
    energy = element.surface_energy(deformation_gradient)

    iter = 0
    err = 1.0
    while iter < iteration and err > tol:
        constraint_grad = element.surface_energy_gradient(deformation_gradient, DdeformationDx)
        constraint_hess = element.surface_energy_hessian(deformation_gradient, DdeformationDx)
        constraint_hess[fix_dofs, :] = 0.0
        constraint_hess[:, fix_dofs] = 0.0
        constraint_hess[fix_dofs, fix_dofs] = 1.0
        constraint_grad[fix_dofs] = 0.0
        delta = checked_scipy_krylov(cg, constraint_hess, -constraint_grad, solver_name="NURBS optimization CG")
        delta = delta.reshape(-1, element.dimension)

        init_ctrlpts = element.vertice.copy()
        alpha = 1.0
        while alpha > tol:
            disp = alpha * delta
            element.vertice = init_ctrlpts + disp
            deformation_gradient = element.compute_deformation_gradient(element.vertice)
            energy_new = element.surface_energy(deformation_gradient)
            if verbose:
                print(f"New energy is {energy_new} and old energy is {energy}")
            if energy_new <= energy:
                break
            alpha *= 0.5
        err = np.sum(np.sqrt(alpha * delta * alpha * delta))
        energy, energy_new = energy_new, energy
        iter += 1

        if verbose:
            print(f"In iteration: {iter}")
            print(f"Step size: {alpha}")
            print(f"Current energy: {energy}")
            print(f"Residual error: {err}", "\n")

        if alpha <= tol or abs(energy) < epsilon:
            break

        if iter == iteration:
            raise RuntimeError(f"Convergence failed, the residual error is {err}")
    return element
