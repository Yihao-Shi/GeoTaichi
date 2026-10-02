import numpy as np

try:
    from .verify_static_twophase_local_jacobian import (
        constitutive_response,
        constitutive_tangent_apply,
        newmark_acceleration,
        newmark_pressure_rate,
        newmark_velocity,
    )
except ImportError:
    from verify_static_twophase_local_jacobian import (
        constitutive_response,
        constitutive_tangent_apply,
        newmark_acceleration,
        newmark_pressure_rate,
        newmark_velocity,
    )


def particle_residual_and_jacobian(material, u_nodes, p_nodes, p_old_nodes, data):
    N = data["N"]
    Navg = data["Navg"]
    Gref = data["Gref"]
    node_ids = data["node_ids"]
    F_old = data["F_old"]
    J_old = data["J_old"]
    stress_old = data["stress_old"]
    V0 = data["V0"]
    dt = data["dt"]
    mobility = data["mobility"]
    tau = data["tau"]
    phi_s0 = data["phi_s0"]
    rho_s = data["rho_s"]
    rho_f = data["rho_f"]
    gravity = data["gravity"]
    lam = data["lam"]
    mu = data["mu"]
    ppp = data["ppp"]
    mode = data["mode"]

    u_loc = u_nodes[node_ids]
    p_loc = p_nodes[node_ids]
    p_old_loc = p_old_nodes[node_ids]

    grad_u = np.zeros((2, 2))
    delta_u = np.zeros(2)
    p_new = 0.0
    p_old = 0.0
    p_avg = 0.0
    p_old_avg = 0.0
    pdot_old = 0.0
    pdot_old_avg = 0.0
    pddot_old = 0.0
    pddot_old_avg = 0.0
    for a in range(len(node_ids)):
        grad_u += np.outer(u_loc[a], Gref[a])
        delta_u += N[a] * u_loc[a]
        p_new += N[a] * p_loc[a]
        p_old += N[a] * p_old_loc[a]
        p_avg += Navg[a] * p_loc[a]
        p_old_avg += Navg[a] * p_old_loc[a]
        if mode == "dynamic":
            pdot_old += N[a] * data["p_rate_old_nodes"][node_ids[a]]
            pdot_old_avg += Navg[a] * data["p_rate_old_nodes"][node_ids[a]]
            pddot_old += N[a] * data["p_acc_old_nodes"][node_ids[a]]
            pddot_old_avg += Navg[a] * data["p_acc_old_nodes"][node_ids[a]]

    deltaF = np.eye(2) + grad_u
    A = np.linalg.inv(deltaF)
    F_new = deltaF @ F_old
    J_new = np.linalg.det(F_new)
    vol = V0 * J_new
    log_ratio = np.log(J_new) - np.log(J_old)
    porosity = 1.0 - phi_s0 / J_new
    rho_mix = porosity * rho_f + (1.0 - porosity) * rho_s
    coeff_rho = (rho_f - rho_s) * phi_s0 / J_new
    gcur = np.array([A @ Gref[a] for a in range(len(node_ids))])
    q = np.sum(p_loc[:, None] * gcur, axis=0)
    sigma_eff = constitutive_response(material, stress_old, F_old, F_new, J_new, grad_u, lam, mu)
    sigma_total = sigma_eff - p_new * np.eye(2)
    ppp_gap = (p_new - p_avg) - (p_old - p_old_avg)
    p_rate_gap = ppp_gap
    a_new = np.zeros(2)
    v_new = np.zeros(2)
    if mode == "dynamic":
        a_new = newmark_acceleration(delta_u, data["velocity_old"], data["acceleration_old"], data["beta"], dt)
        v_new = newmark_velocity(delta_u, data["velocity_old"], data["acceleration_old"], data["beta"], data["gamma"], dt)
        p_rate_new = newmark_pressure_rate(p_new, p_old, pdot_old, pddot_old, data["beta"], data["gamma"], dt)
        p_rate_new_avg = newmark_pressure_rate(p_avg, p_old_avg, pdot_old_avg, pddot_old_avg, data["beta"], data["gamma"], dt)
        p_rate_gap = (p_rate_new - p_rate_new_avg) - (pdot_old - pdot_old_avg)

    rloc = np.zeros(3 * len(node_ids))
    Kloc = np.zeros((3 * len(node_ids), 3 * len(node_ids)))
    for a in range(len(node_ids)):
        gi = gcur[a]
        base_force = (-sigma_total @ gi + N[a] * rho_mix * gravity) * vol
        if mode == "dynamic":
            base_force -= N[a] * data["mass"] * (a_new + data["damping"] * v_new)
        rloc[3 * a + 0] = base_force[0]
        rloc[3 * a + 1] = base_force[1]

        darcy = dt * mobility * gi.dot(q)
        mass = (N[a] * log_ratio + darcy) * vol
        if ppp:
            mass += tau * (N[a] - Navg[a]) * (p_rate_gap if mode == "dynamic" else ppp_gap) * vol
        rloc[3 * a + 2] = mass

        for b in range(len(node_ids)):
            Gj = Gref[b]
            gj = gcur[b]
            Kup = vol * N[b] * gi
            Kloc[3 * a + 0, 3 * b + 2] = Kup[0]
            Kloc[3 * a + 1, 3 * b + 2] = Kup[1]

            Kpp = dt * mobility * gi.dot(gj) * vol
            if ppp:
                if mode == "dynamic":
                    Kpp += tau * data["gamma"] / (data["beta"] * dt) * (N[a] - Navg[a]) * (N[b] - Navg[b]) * vol
                else:
                    Kpp += tau * (N[a] - Navg[a]) * (N[b] - Navg[b]) * vol
            Kloc[3 * a + 2, 3 * b + 2] = Kpp

            for comp in range(2):
                col = 3 * b + comp
                hcol = A[:, comp]
                eta = Gj.dot(hcol)
                dgi = -hcol * Gj.dot(gi)
                dv = vol * eta
                drho = coeff_rho * eta
                dsigma = constitutive_tangent_apply(material, F_old, F_new, J_new, Gj, comp, eta, lam, mu)

                mech = (-dsigma @ gi - sigma_total @ dgi + N[a] * drho * gravity) * vol
                mech += (-sigma_total @ gi + N[a] * rho_mix * gravity) * dv
                if mode == "dynamic":
                    mech[comp] -= data["mass"] * N[a] * N[b] * (
                        1.0 / (data["beta"] * dt * dt) + data["damping"] * data["gamma"] / (data["beta"] * dt)
                    )
                Kloc[3 * a + 0, col] = mech[0]
                Kloc[3 * a + 1, col] = mech[1]

                scalar_grad = (-Gj.dot(gi) * hcol.dot(q) - Gj.dot(q) * hcol.dot(gi) + gi.dot(q) * eta) * dt * mobility * vol
                kpu = N[a] * (1.0 + log_ratio) * eta * vol + scalar_grad
                if ppp:
                    kpu += tau * (N[a] - Navg[a]) * (p_rate_gap if mode == "dynamic" else ppp_gap) * dv
                Kloc[3 * a + 2, col] = kpu

    return rloc, Kloc


def pack_unknowns(u_nodes, p_nodes):
    n = len(p_nodes)
    x = np.zeros(3 * n)
    for i in range(n):
        x[3 * i:3 * i + 2] = u_nodes[i]
        x[3 * i + 2] = p_nodes[i]
    return x


def unpack_unknowns(x):
    n = len(x) // 3
    u = np.zeros((n, 2))
    p = np.zeros(n)
    for i in range(n):
        u[i] = x[3 * i:3 * i + 2]
        p[i] = x[3 * i + 2]
    return u, p


def global_residual(material, x, p_old_nodes, particles, n_nodes):
    u_nodes, p_nodes = unpack_unknowns(x)
    r = np.zeros(3 * n_nodes)
    for data in particles:
        rloc, _ = particle_residual_and_jacobian(material, u_nodes, p_nodes, p_old_nodes, data)
        for a, node in enumerate(data["node_ids"]):
            r[3 * node:3 * node + 3] += rloc[3 * a:3 * a + 3]
    return r


def global_jacobian_manual(material, x, p_old_nodes, particles, n_nodes):
    u_nodes, p_nodes = unpack_unknowns(x)
    K = np.zeros((3 * n_nodes, 3 * n_nodes))
    for data in particles:
        _, Kloc = particle_residual_and_jacobian(material, u_nodes, p_nodes, p_old_nodes, data)
        for a, row_node in enumerate(data["node_ids"]):
            for b, col_node in enumerate(data["node_ids"]):
                rs = slice(3 * row_node, 3 * row_node + 3)
                cs = slice(3 * col_node, 3 * col_node + 3)
                rsl = slice(3 * a, 3 * a + 3)
                csl = slice(3 * b, 3 * b + 3)
                K[rs, cs] += Kloc[rsl, csl]
    return K


def finite_difference_jacobian(material, x, p_old_nodes, particles, n_nodes, eps=1.0e-7):
    n = len(x)
    J = np.zeros((n, n))
    for j in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[j] += eps
        xm[j] -= eps
        rp = global_residual(material, xp, p_old_nodes, particles, n_nodes)
        rm = global_residual(material, xm, p_old_nodes, particles, n_nodes)
        J[:, j] = (rp - rm) / (2.0 * eps)
    return J


def random_particle(rng, n_nodes, ppp, mode):
    support = 4
    node_ids = np.sort(rng.choice(n_nodes, size=support, replace=False))
    N = rng.random(support)
    N /= N.sum()
    Navg = rng.random(support)
    Navg /= Navg.sum()
    Gref = rng.uniform(-1.0, 1.0, size=(support, 2))
    F_old = np.eye(2) + rng.uniform(-0.15, 0.15, size=(2, 2))
    if np.linalg.det(F_old) <= 0.25:
        F_old += 0.4 * np.eye(2)
    return {
        "node_ids": node_ids,
        "N": N,
        "Navg": Navg,
        "Gref": Gref,
        "F_old": F_old,
        "J_old": np.linalg.det(F_old),
        "stress_old": rng.uniform(-0.3, 0.3, size=(2, 2)),
        "V0": rng.uniform(8.0e-3, 2.5e-2),
        "dt": rng.uniform(0.05, 0.25),
        "mobility": rng.uniform(1.0e-5, 5.0e-4),
        "tau": 0.5 / rng.uniform(120.0, 900.0),
        "phi_s0": rng.uniform(0.2, 0.8),
        "rho_s": rng.uniform(0.9, 2.4),
        "rho_f": rng.uniform(0.8, 1.4),
        "gravity": np.array([0.0, -rng.uniform(0.0, 9.81)], dtype=np.float64),
        "lam": rng.uniform(150.0, 700.0),
        "mu": rng.uniform(100.0, 500.0),
        "ppp": ppp,
        "mode": mode,
        "mass": rng.uniform(0.02, 0.4),
        "damping": rng.uniform(0.0, 0.2),
        "beta": 0.3025,
        "gamma": 0.6,
        "velocity_old": rng.uniform(-0.3, 0.3, size=2),
        "acceleration_old": rng.uniform(-0.4, 0.4, size=2),
        "p_rate_old_nodes": rng.uniform(-0.3, 0.3, size=n_nodes),
        "p_acc_old_nodes": rng.uniform(-0.4, 0.4, size=n_nodes),
    }


def run_case(material, ppp, mode, trials=10):
    seed = 23 if (ppp and mode == "static") else 29 if mode == "static" else 31 if ppp else 41
    rng = np.random.default_rng(seed)
    worst_abs = 0.0
    worst_rel = 0.0
    worst_id = -1
    for trial in range(trials):
        n_nodes = 6
        particles = [random_particle(rng, n_nodes, ppp, mode) for _ in range(3)]
        u_nodes = rng.uniform(-4.0e-3, 4.0e-3, size=(n_nodes, 2))
        p_nodes = rng.uniform(-0.2, 0.2, size=n_nodes)
        p_old_nodes = rng.uniform(-0.2, 0.2, size=n_nodes)
        x = pack_unknowns(u_nodes, p_nodes)
        K_man = global_jacobian_manual(material, x, p_old_nodes, particles, n_nodes)
        K_fd = finite_difference_jacobian(material, x, p_old_nodes, particles, n_nodes)
        abs_err = np.max(np.abs(K_man - K_fd))
        rel_err = abs_err / max(np.max(np.abs(K_fd)), 1.0e-12)
        if rel_err > worst_rel:
            worst_rel = rel_err
            worst_abs = abs_err
            worst_id = trial
    tag = "PPP-on" if ppp else "PPP-off"
    print(f"{mode:7s} {material:13s} {tag:7s}: worst_abs={worst_abs:.3e}, worst_rel={worst_rel:.3e}, trial={worst_id}")
    return worst_abs, worst_rel


if __name__ == "__main__":
    for mode in ("static", "dynamic"):
        for material in ("linearElastic", "neoHookean"):
            for ppp in (False, True):
                run_case(material, ppp, mode, trials=12)
