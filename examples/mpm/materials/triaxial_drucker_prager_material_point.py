import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from tools.diagnostics.mpm.static_twophase.verify_static_twophase_local_jacobian_3d import (
    dp_local_update,
    dp_material_constants,
    dp_yield_value,
    elastic_strain_from_stress,
)


def stress_invariants(stress):
    p_mean = np.trace(stress) / 3.0
    s = stress - p_mean * np.eye(3)
    q_dev = np.sqrt(max(1.5 * np.sum(s * s), 0.0))
    return p_mean, q_dev


def update_stress(total_strain_new, total_strain_old, elastic_strain_old, lam, mu, friction_angle, dilation_angle, cohesion, shape_factor):
    deps = total_strain_new - total_strain_old
    eps_trial = elastic_strain_old + deps
    stress_new, elastic_strain_new, *_ = dp_local_update(
        eps_trial,
        lam,
        mu,
        friction_angle,
        dilation_angle,
        cohesion,
        shape_factor,
    )
    return stress_new, elastic_strain_new


def radial_residual(radial_strain, total_strain_old, elastic_strain_old, axial_strain_new, sigma_conf, lam, mu, friction_angle, dilation_angle, cohesion, shape_factor):
    total_strain_new = np.diag([radial_strain, radial_strain, axial_strain_new])
    stress_new, elastic_strain_new = update_stress(
        total_strain_new,
        total_strain_old,
        elastic_strain_old,
        lam,
        mu,
        friction_angle,
        dilation_angle,
        cohesion,
        shape_factor,
    )
    return stress_new[0, 0] - sigma_conf, stress_new, elastic_strain_new


def solve_radial_strain(total_strain_old, elastic_strain_old, axial_strain_new, sigma_conf, lam, mu, friction_angle, dilation_angle, cohesion, shape_factor, tol=1.0e-10, max_iters=30):
    radial = total_strain_old[0, 0]
    stress = np.zeros((3, 3))
    elastic = elastic_strain_old.copy()
    for _ in range(max_iters):
        res, stress, elastic = radial_residual(
            radial,
            total_strain_old,
            elastic_strain_old,
            axial_strain_new,
            sigma_conf,
            lam,
            mu,
            friction_angle,
            dilation_angle,
            cohesion,
            shape_factor,
        )
        if abs(res) < tol:
            break
        h = 1.0e-8 * max(1.0, abs(radial))
        rp, _, _ = radial_residual(
            radial + h,
            total_strain_old,
            elastic_strain_old,
            axial_strain_new,
            sigma_conf,
            lam,
            mu,
            friction_angle,
            dilation_angle,
            cohesion,
            shape_factor,
        )
        rm, _, _ = radial_residual(
            radial - h,
            total_strain_old,
            elastic_strain_old,
            axial_strain_new,
            sigma_conf,
            lam,
            mu,
            friction_angle,
            dilation_angle,
            cohesion,
            shape_factor,
        )
        dres = (rp - rm) / (2.0 * h)
        if abs(dres) < 1.0e-20:
            raise RuntimeError("Radial Newton derivative is near zero.")
        radial -= res / dres
    else:
        raise RuntimeError("Radial Newton did not converge.")
    return radial, stress, elastic


def main():
    young_modulus = 25000.0
    poisson_ratio = 0.3
    friction_angle = 35.0
    dilation_angle = 5.0
    cohesion = 0.0
    shape_factor = 0.0

    sigma_conf = -150.0
    final_axial_strain = 0.1
    n_steps = 100
    axial_increment = -final_axial_strain / n_steps

    lam = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    mu = young_modulus / (2.0 * (1.0 + poisson_ratio))

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TriaxialDruckerPrager")
    os.makedirs(out_dir, exist_ok=True)

    stress0 = np.diag([sigma_conf, sigma_conf, sigma_conf])
    elastic_strain = elastic_strain_from_stress(stress0, lam, mu)
    total_strain = elastic_strain.copy()

    Af, Bf = dp_material_constants(friction_angle, cohesion)
    rows = []

    for step in range(n_steps + 1):
        stress = lam * np.trace(elastic_strain) * np.eye(3) + 2.0 * mu * elastic_strain
        p_mean, q_dev = stress_invariants(stress)
        t_dev = np.sqrt(max(np.sum((stress - p_mean * np.eye(3)) ** 2), 0.0))
        yield_value = dp_yield_value(p_mean, t_dev, Af, Bf, shape_factor)
        rows.append(
            [
                step,
                total_strain[0, 0],
                total_strain[2, 2],
                stress[0, 0],
                stress[1, 1],
                stress[2, 2],
                p_mean,
                q_dev,
                yield_value,
            ]
        )

        if step == n_steps:
            break

        axial_new = total_strain[2, 2] + axial_increment
        radial_new, stress_new, elastic_new = solve_radial_strain(
            total_strain,
            elastic_strain,
            axial_new,
            sigma_conf,
            lam,
            mu,
            friction_angle,
            dilation_angle,
            cohesion,
            shape_factor,
        )
        total_strain = np.diag([radial_new, radial_new, axial_new])
        elastic_strain = elastic_new

    data = np.array(rows, dtype=np.float64)
    csv_path = os.path.join(out_dir, "triaxial_history.csv")
    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header="step,eps_xx,eps_zz,sigma_xx,sigma_yy,sigma_zz,p_mean,q_dev,yield_value",
        comments="",
    )

    axial_strain_plot = -data[:, 2]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(axial_strain_plot, data[:, 3], label=r"$\sigma_{rr}$")
    axes[0].plot(axial_strain_plot, data[:, 5], label=r"$\sigma_{zz}$")
    axes[0].set_xlabel(r"$-\varepsilon_{zz}$")
    axes[0].set_ylabel("Stress")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(axial_strain_plot, data[:, 6], label="p")
    axes[1].plot(axial_strain_plot, data[:, 7], label="q")
    axes[1].set_xlabel(r"$-\varepsilon_{zz}$")
    axes[1].set_ylabel("Invariant")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(axial_strain_plot, data[:, 8], label="f(sigma)")
    axes[2].axhline(0.0, color="k", linewidth=1.0, linestyle="--")
    axes[2].set_xlabel(r"$-\varepsilon_{zz}$")
    axes[2].set_ylabel("Yield value")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(out_dir, "triaxial_curves.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    plastic_rows = data[np.where(data[:, 8] > -1.0e-8)]
    print(f"saved csv: {csv_path}")
    print(f"saved figure: {fig_path}")
    print(f"final sigma_rr={data[-1,3]:.6f}, final sigma_zz={data[-1,5]:.6f}, final q={data[-1,7]:.6f}")
    if plastic_rows.shape[0] > 0:
        first_yield_step = int(plastic_rows[0, 0])
        active_history = data[first_yield_step:, 8]
        print(f"first near-yield step = {first_yield_step}")
        print(f"max |yield value| after first yield = {np.max(np.abs(active_history)):.6e}")
    else:
        print("plastic loading was not activated in this run")


if __name__ == "__main__":
    main()
