import os

import matplotlib.pyplot as plt
import numpy as np


def undrained_theory(axial_strain, sigma_conf, young_modulus, poisson_ratio):
    shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio))
    radial_strain = -0.5 * axial_strain
    excess_pore_pressure = shear_modulus * axial_strain
    sigma_rr_eff = np.full_like(axial_strain, sigma_conf, dtype=np.float64) - excess_pore_pressure
    sigma_zz_eff = np.full_like(axial_strain, sigma_conf, dtype=np.float64) + 2.0 * shear_modulus * axial_strain
    sigma_rr_total = np.full_like(axial_strain, sigma_conf, dtype=np.float64)
    sigma_zz_total = sigma_zz_eff + excess_pore_pressure
    q_total = sigma_zz_total - sigma_rr_total
    q_eff = sigma_zz_eff - sigma_rr_eff
    volumetric_strain = axial_strain + 2.0 * radial_strain
    return {
        "radial_strain": radial_strain,
        "excess_pore_pressure": excess_pore_pressure,
        "sigma_rr_eff": sigma_rr_eff,
        "sigma_zz_eff": sigma_zz_eff,
        "sigma_rr_total": sigma_rr_total,
        "sigma_zz_total": sigma_zz_total,
        "q_total": q_total,
        "q_eff": q_eff,
        "volumetric_strain": volumetric_strain,
    }


def run_incremental_response(sigma_conf, young_modulus, poisson_ratio, final_axial_strain, n_steps):
    shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lame_lambda = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    axial_increment = final_axial_strain / n_steps

    strain = np.zeros((3, 3), dtype=np.float64)
    sigma_eff = np.diag([sigma_conf, sigma_conf, sigma_conf]).astype(np.float64)
    excess_pore_pressure = 0.0
    rows = []

    for step in range(n_steps + 1):
        axial_strain = strain[2, 2]
        radial_strain = strain[0, 0]
        sigma_total = sigma_eff + excess_pore_pressure * np.eye(3)
        q_total = sigma_total[2, 2] - sigma_total[0, 0]
        q_eff = sigma_eff[2, 2] - sigma_eff[0, 0]
        volumetric_strain = np.trace(strain)
        rows.append(
            [
                step,
                radial_strain,
                axial_strain,
                volumetric_strain,
                sigma_total[0, 0],
                sigma_total[2, 2],
                sigma_eff[0, 0],
                sigma_eff[2, 2],
                excess_pore_pressure,
                q_total,
                q_eff,
            ]
        )

        if step == n_steps:
            break

        deps = np.diag([-0.5 * axial_increment, -0.5 * axial_increment, axial_increment])
        dsigma_eff = lame_lambda * np.trace(deps) * np.eye(3) + 2.0 * shear_modulus * deps
        du = -dsigma_eff[0, 0]

        strain += deps
        sigma_eff += dsigma_eff
        excess_pore_pressure += du

    return np.asarray(rows, dtype=np.float64)


def main():
    young_modulus = 25000.0
    poisson_ratio = 0.3
    sigma_conf = -150.0
    final_axial_strain = 0.05
    n_steps = 100

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TriaxialUndrainedLinearElastic")
    os.makedirs(out_dir, exist_ok=True)

    data = run_incremental_response(
        sigma_conf=sigma_conf,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        final_axial_strain=final_axial_strain,
        n_steps=n_steps,
    )

    axial_strain = data[:, 2]
    theory = undrained_theory(
        axial_strain=axial_strain,
        sigma_conf=sigma_conf,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
    )

    csv_path = os.path.join(out_dir, "triaxial_undrained_history.csv")
    np.savetxt(
        csv_path,
        np.column_stack(
            [
                data,
                theory["excess_pore_pressure"],
                theory["sigma_rr_eff"],
                theory["sigma_zz_eff"],
                theory["sigma_rr_total"],
                theory["sigma_zz_total"],
                theory["q_total"],
                theory["q_eff"],
                theory["volumetric_strain"],
            ]
        ),
        delimiter=",",
        header=(
            "step,eps_rr,eps_zz,eps_v,sigma_rr_total,sigma_zz_total,sigma_rr_eff,sigma_zz_eff,"
            "excess_pore_pressure,q_total,q_eff,"
            "u_theory,sigma_rr_eff_theory,sigma_zz_eff_theory,sigma_rr_total_theory,"
            "sigma_zz_total_theory,q_total_theory,q_eff_theory,eps_v_theory"
        ),
        comments="",
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(axial_strain, data[:, 4], label=r"$\sigma_{rr}$ total")
    axes[0].plot(axial_strain, data[:, 5], label=r"$\sigma_{zz}$ total")
    axes[0].plot(axial_strain, data[:, 6], label=r"$\sigma_{rr}'$ effective")
    axes[0].plot(axial_strain, data[:, 7], label=r"$\sigma_{zz}'$ effective")
    axes[0].plot(axial_strain, theory["sigma_zz_eff"], "--", color="k", label=r"$\sigma_{zz}'$ theory")
    axes[0].set_xlabel(r"$\varepsilon_{zz}$")
    axes[0].set_ylabel("Stress / Pore pressure [kPa]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(axial_strain, data[:, 8], label="u numerical")
    axes[1].plot(axial_strain, theory["excess_pore_pressure"], "--", label="u theory")
    axes[1].plot(axial_strain, data[:, 9], label="q_total numerical")
    axes[1].plot(axial_strain, theory["q_total"], "--", label="q_total theory")
    axes[1].set_xlabel(r"$\varepsilon_{zz}$")
    axes[1].set_ylabel("Value [kPa]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(axial_strain, data[:, 3], label=r"$\varepsilon_v$ numerical")
    axes[2].plot(axial_strain, theory["volumetric_strain"], "--", label=r"$\varepsilon_v$ theory")
    axes[2].plot(axial_strain, data[:, 4] - sigma_conf, label=r"$\sigma_{rr} - \sigma_{conf}$")
    axes[2].axhline(0.0, color="k", linewidth=1.0, linestyle="--")
    axes[2].set_xlabel(r"$\varepsilon_{zz}$")
    axes[2].set_ylabel("Constraint check")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(out_dir, "triaxial_undrained_curves.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    max_u_err = float(np.max(np.abs(data[:, 8] - theory["excess_pore_pressure"])))
    max_q_err = float(np.max(np.abs(data[:, 9] - theory["q_total"])))
    max_ev = float(np.max(np.abs(data[:, 3])))
    max_conf_err = float(np.max(np.abs(data[:, 4] - sigma_conf)))

    print(f"saved csv: {csv_path}")
    print(f"saved figure: {fig_path}")
    print(f"final excess pore pressure = {data[-1, 8]:.6f} kPa")
    print(f"final q_total = {data[-1, 9]:.6f} kPa")
    print(f"max |u - theory| = {max_u_err:.6e}")
    print(f"max |q_total - theory| = {max_q_err:.6e}")
    print(f"max |eps_v| = {max_ev:.6e}")
    print(f"max |sigma_rr - sigma_conf| = {max_conf_err:.6e}")


if __name__ == "__main__":
    main()
