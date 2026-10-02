import numpy as np

try:
    from .verify_static_twophase_local_jacobian_3d import (
        constitutive_response,
        constitutive_tangent_apply,
        dp_local_update,
    )
except ImportError:
    from verify_static_twophase_local_jacobian_3d import (
        constitutive_response,
        constitutive_tangent_apply,
        dp_local_update,
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

    u_loc = u_nodes[node_ids]
    p_loc = p_nodes[node_ids]
    p_old_loc = p_old_nodes[node_ids]

    grad_u = np.zeros((3, 3))
    p_new = 0.0
    p_old = 0.0
    p_avg = 0.0
    p_old_avg = 0.0
    for a in range(len(node_ids)):
        grad_u += np.outer(u_loc[a], Gref[a])
        p_new += N[a] * p_loc[a]
        p_old += N[a] * p_old_loc[a]
        p_avg += Navg[a] * p_loc[a]
        p_old_avg += Navg[a] * p_old_loc[a]

    deltaF = np.eye(3) + grad_u
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
    sigma_total = sigma_eff - p_new * np.eye(3)
    ppp_gap = (p_new - p_avg) - (p_old - p_old_avg)

    rloc = np.zeros(4 * len(node_ids))
    Kloc = np.zeros((4 * len(node_ids), 4 * len(node_ids)))
    for a in range(len(node_ids)):
        gi = gcur[a]
        base_force = (-sigma_total @ gi + N[a] * rho_mix * gravity) * vol
        rloc[4 * a:4 * a + 3] = base_force

        darcy = dt * mobility * gi.dot(q)
        mass = (N[a] * log_ratio + darcy) * vol
        if ppp:
            mass += tau * (N[a] - Navg[a]) * ppp_gap * vol
        rloc[4 * a + 3] = mass

        for b in range(len(node_ids)):
            Gj = Gref[b]
            gj = gcur[b]
            Kup = vol * N[b] * gi
            Kloc[4 * a:4 * a + 3, 4 * b + 3] = Kup

            Kpp = dt * mobility * gi.dot(gj) * vol
            if ppp:
                Kpp += tau * (N[a] - Navg[a]) * (N[b] - Navg[b]) * vol
            Kloc[4 * a + 3, 4 * b + 3] = Kpp

            for comp in range(3):
                col = 4 * b + comp
                hcol = A[:, comp]
                eta = Gj.dot(hcol)
                dgi = -hcol * Gj.dot(gi)
                dv = vol * eta
                drho = coeff_rho * eta
                dsigma = constitutive_tangent_apply(material, stress_old, F_old, F_new, J_new, grad_u, Gj, comp, eta, lam, mu)

                mech = (-dsigma @ gi - sigma_total @ dgi + N[a] * drho * gravity) * vol
                mech += (-sigma_total @ gi + N[a] * rho_mix * gravity) * dv
                Kloc[4 * a:4 * a + 3, col] = mech

                scalar_grad = (
                    -Gj.dot(gi) * hcol.dot(q)
                    - Gj.dot(q) * hcol.dot(gi)
                    + gi.dot(q) * eta
                ) * dt * mobility * vol
                kpu = N[a] * (1.0 + log_ratio) * eta * vol + scalar_grad
                if ppp:
                    kpu += tau * (N[a] - Navg[a]) * ppp_gap * dv
                Kloc[4 * a + 3, col] = kpu

    return rloc, Kloc


def pack_unknowns(u_nodes, p_nodes):
    n = len(p_nodes)
    x = np.zeros(4 * n)
    for i in range(n):
        x[4 * i:4 * i + 3] = u_nodes[i]
        x[4 * i + 3] = p_nodes[i]
    return x


def unpack_unknowns(x):
    n = len(x) // 4
    u = np.zeros((n, 3))
    p = np.zeros(n)
    for i in range(n):
        u[i] = x[4 * i:4 * i + 3]
        p[i] = x[4 * i + 3]
    return u, p


def global_residual(material, x, p_old_nodes, particles, n_nodes):
    u_nodes, p_nodes = unpack_unknowns(x)
    r = np.zeros(4 * n_nodes)
    for data in particles:
        rloc, _ = particle_residual_and_jacobian(material, u_nodes, p_nodes, p_old_nodes, data)
        for a, node in enumerate(data["node_ids"]):
            r[4 * node:4 * node + 4] += rloc[4 * a:4 * a + 4]
    return r


def global_jacobian_manual(material, x, p_old_nodes, particles, n_nodes):
    u_nodes, p_nodes = unpack_unknowns(x)
    K = np.zeros((4 * n_nodes, 4 * n_nodes))
    for data in particles:
        _, Kloc = particle_residual_and_jacobian(material, u_nodes, p_nodes, p_old_nodes, data)
        for a, row_node in enumerate(data["node_ids"]):
            for b, col_node in enumerate(data["node_ids"]):
                rs = slice(4 * row_node, 4 * row_node + 4)
                cs = slice(4 * col_node, 4 * col_node + 4)
                rsl = slice(4 * a, 4 * a + 4)
                csl = slice(4 * b, 4 * b + 4)
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


def random_particle(rng, n_nodes, ppp):
    support = 8
    node_ids = np.sort(rng.choice(n_nodes, size=support, replace=False))
    N = rng.random(support)
    N /= N.sum()
    Navg = rng.random(support)
    Navg /= Navg.sum()
    Gref = rng.uniform(-1.0, 1.0, size=(support, 3))
    F_old = np.eye(3) + rng.uniform(-0.1, 0.1, size=(3, 3))
    if np.linalg.det(F_old) <= 0.35:
        F_old += 0.35 * np.eye(3)
    stress_old = rng.uniform(-0.2, 0.2, size=(3, 3))
    stress_old = 0.5 * (stress_old + stress_old.T)
    return {
        "node_ids": node_ids,
        "N": N,
        "Navg": Navg,
        "Gref": Gref,
        "F_old": F_old,
        "J_old": np.linalg.det(F_old),
        "stress_old": stress_old,
        "V0": rng.uniform(8.0e-3, 2.5e-2),
        "dt": rng.uniform(0.05, 0.25),
        "mobility": rng.uniform(1.0e-5, 5.0e-4),
        "tau": 0.5 / rng.uniform(120.0, 900.0),
        "phi_s0": rng.uniform(0.2, 0.8),
        "rho_s": rng.uniform(0.9, 2.4),
        "rho_f": rng.uniform(0.8, 1.4),
        "gravity": np.array([0.0, 0.0, -rng.uniform(0.0, 9.81)], dtype=np.float64),
        "lam": rng.uniform(150.0, 700.0),
        "mu": rng.uniform(100.0, 500.0),
        "ppp": ppp,
    }


def make_admissible_old_stress(material, particle, rng):
    if material != "druckerPrager":
        return
    eps0 = rng.uniform(-2.0e-3, 2.0e-3, size=(3, 3))
    eps0 = 0.5 * (eps0 + eps0.T)
    sigma0, *_ = dp_local_update(
        eps0,
        particle["lam"],
        particle["mu"],
        20.0,
        10.0,
        1.0,
        0.0,
    )
    particle["stress_old"] = sigma0


def run_case(material, ppp, trials=10):
    rng = np.random.default_rng(31 if ppp else 37)
    worst_abs = 0.0
    worst_rel = 0.0
    worst_id = -1
    for trial in range(trials):
        n_nodes = 10
        particles = [random_particle(rng, n_nodes, ppp) for _ in range(3)]
        for particle in particles:
            make_admissible_old_stress(material, particle, rng)
        u_nodes = rng.uniform(-4.0e-3, 4.0e-3, size=(n_nodes, 3))
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
    print(f"{material:13s} {tag:7s}: worst_abs={worst_abs:.3e}, worst_rel={worst_rel:.3e}, trial={worst_id}")
    return worst_abs, worst_rel


if __name__ == "__main__":
    for material in ("linearElastic", "neoHookean", "druckerPrager"):
        for ppp in (False, True):
            run_case(material, ppp, trials=12)
