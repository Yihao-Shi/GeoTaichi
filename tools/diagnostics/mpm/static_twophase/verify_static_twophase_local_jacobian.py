"""Deterministic 2-D static two-phase local-Jacobian diagnostic."""

import numpy as np


def neo_hookean_stress(F, J, lam, mu):
    b = F @ F.T
    I = np.eye(2)
    return mu / J * (b - I) + lam * np.log(J) / J * I


def constitutive_response(material, stress_old, F_old, F_new, J_new, grad_u, lam, mu):
    if material == "linearElastic":
        eps = 0.5 * (grad_u + grad_u.T)
        return stress_old + lam * np.trace(eps) * np.eye(2) + 2.0 * mu * eps
    if material == "neoHookean":
        return neo_hookean_stress(F_new, J_new, lam, mu)
    raise ValueError(material)


def constitutive_tangent_apply(material, F_old, F_new, J_new, G, comp, eta, lam, mu):
    deps = np.zeros((2, 2))
    deps[comp, 0] += 0.5 * G[0]
    deps[0, comp] += 0.5 * G[0]
    deps[comp, 1] += 0.5 * G[1]
    deps[1, comp] += 0.5 * G[1]
    if material == "linearElastic":
        return lam * np.trace(deps) * np.eye(2) + 2.0 * mu * deps
    H = F_old.T @ G
    dF = np.zeros((2, 2))
    dF[comp, 0] = H[0]
    dF[comp, 1] = H[1]
    b = F_new @ F_new.T
    db = dF @ F_new.T + F_new @ dF.T
    dJ = J_new * eta
    return mu / J_new * db - mu / (J_new * J_new) * (b - np.eye(2)) * dJ + lam * (1.0 - np.log(J_new)) / (J_new * J_new) * dJ * np.eye(2)


def newmark_acceleration(delta_u, v_old, a_old, beta, dt):
    return delta_u / (beta * dt * dt) - v_old / (beta * dt) - (0.5 / beta - 1.0) * a_old


def newmark_velocity(delta_u, v_old, a_old, beta, gamma, dt):
    return gamma / (beta * dt) * delta_u - (gamma / beta - 1.0) * v_old - dt * (gamma / (2.0 * beta) - 1.0) * a_old


def newmark_pressure_rate(p_new, p_old, pdot_old, pddot_old, beta, gamma, dt):
    return gamma / (beta * dt) * (p_new - p_old) + (1.0 - gamma / beta) * pdot_old - dt * (gamma / (2.0 * beta) - 1.0) * pddot_old


def residual(material, u_new, p_new_nodes, p_old_nodes, data):
    N = data["N"]
    Navg = data["Navg"]
    Gref = data["Gref"]
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
    mode = data["mode"]

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
    for a in range(len(N)):
        grad_u += np.outer(u_new[a], Gref[a])
        delta_u += N[a] * u_new[a]
        p_new += N[a] * p_new_nodes[a]
        p_old += N[a] * p_old_nodes[a]
        p_avg += Navg[a] * p_new_nodes[a]
        p_old_avg += Navg[a] * p_old_nodes[a]
        if mode == "dynamic":
            pdot_old += N[a] * data["p_rate_old_nodes"][a]
            pdot_old_avg += Navg[a] * data["p_rate_old_nodes"][a]
            pddot_old += N[a] * data["p_acc_old_nodes"][a]
            pddot_old_avg += Navg[a] * data["p_acc_old_nodes"][a]

    deltaF = np.eye(2) + grad_u
    A = np.linalg.inv(deltaF)
    F_new = deltaF @ F_old
    J_new = np.linalg.det(F_new)
    vol = V0 * J_new
    log_ratio = np.log(J_new) - np.log(J_old)
    porosity = 1.0 - phi_s0 / J_new
    rho_mix = porosity * rho_f + (1.0 - porosity) * rho_s
    gcur = np.array([A @ Gref[a] for a in range(len(N))])
    q = np.sum(p_new_nodes[:, None] * gcur, axis=0)
    sigma_eff = constitutive_response(material, stress_old, F_old, F_new, J_new, grad_u, data["lam"], data["mu"])
    sigma_total = sigma_eff - p_new * np.eye(2)
    ppp_gap = (p_new - p_avg) - (p_old - p_old_avg)
    p_rate_gap = ppp_gap
    a_new = np.zeros(2)
    v_new = np.zeros(2)
    if mode == "dynamic":
        beta = data["beta"]
        gamma = data["gamma"]
        a_new = newmark_acceleration(delta_u, data["velocity_old"], data["acceleration_old"], beta, dt)
        v_new = newmark_velocity(delta_u, data["velocity_old"], data["acceleration_old"], beta, gamma, dt)
        p_rate_new = newmark_pressure_rate(p_new, p_old, pdot_old, pddot_old, beta, gamma, dt)
        p_rate_new_avg = newmark_pressure_rate(p_avg, p_old_avg, pdot_old_avg, pddot_old_avg, beta, gamma, dt)
        p_rate_gap = (p_rate_new - p_rate_new_avg) - (pdot_old - pdot_old_avg)

    r = np.zeros(3 * len(N))
    for a in range(len(N)):
        gi = gcur[a]
        base_force = (-sigma_total @ gi + N[a] * rho_mix * gravity) * vol
        if mode == "dynamic":
            base_force -= N[a] * data["mass"] * (a_new + data["damping"] * v_new)
        r[3 * a + 0] = base_force[0]
        r[3 * a + 1] = base_force[1]
        darcy = dt * mobility * gi.dot(q)
        mass = (N[a] * log_ratio + darcy) * vol
        if data["ppp"]:
            mass += tau * (N[a] - Navg[a]) * (p_rate_gap if mode == "dynamic" else ppp_gap) * vol
        r[3 * a + 2] = mass
    return r


def jacobian_manual(material, u_new, p_new_nodes, p_old_nodes, data):
    N = data["N"]
    Navg = data["Navg"]
    Gref = data["Gref"]
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
    mode = data["mode"]

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
    for a in range(len(N)):
        grad_u += np.outer(u_new[a], Gref[a])
        delta_u += N[a] * u_new[a]
        p_new += N[a] * p_new_nodes[a]
        p_old += N[a] * p_old_nodes[a]
        p_avg += Navg[a] * p_new_nodes[a]
        p_old_avg += Navg[a] * p_old_nodes[a]
        if mode == "dynamic":
            pdot_old += N[a] * data["p_rate_old_nodes"][a]
            pdot_old_avg += Navg[a] * data["p_rate_old_nodes"][a]
            pddot_old += N[a] * data["p_acc_old_nodes"][a]
            pddot_old_avg += Navg[a] * data["p_acc_old_nodes"][a]

    deltaF = np.eye(2) + grad_u
    A = np.linalg.inv(deltaF)
    F_new = deltaF @ F_old
    J_new = np.linalg.det(F_new)
    vol = V0 * J_new
    log_ratio = np.log(J_new) - np.log(J_old)
    porosity = 1.0 - phi_s0 / J_new
    rho_mix = porosity * rho_f + (1.0 - porosity) * rho_s
    coeff_rho = (rho_f - rho_s) * phi_s0 / J_new
    gcur = np.array([A @ Gref[a] for a in range(len(N))])
    q = np.sum(p_new_nodes[:, None] * gcur, axis=0)
    sigma_eff = constitutive_response(material, stress_old, F_old, F_new, J_new, grad_u, lam, mu)
    sigma_total = sigma_eff - p_new * np.eye(2)
    ppp_gap = (p_new - p_avg) - (p_old - p_old_avg)
    p_rate_gap = ppp_gap
    if mode == "dynamic":
        p_rate_new = newmark_pressure_rate(p_new, p_old, pdot_old, pddot_old, data["beta"], data["gamma"], dt)
        p_rate_new_avg = newmark_pressure_rate(p_avg, p_old_avg, pdot_old_avg, pddot_old_avg, data["beta"], data["gamma"], dt)
        p_rate_gap = (p_rate_new - p_rate_new_avg) - (pdot_old - pdot_old_avg)

    K = np.zeros((3 * len(N), 3 * len(N)))
    for a in range(len(N)):
        gi = gcur[a]
        for b in range(len(N)):
            Gj = Gref[b]
            gj = gcur[b]
            Kup = vol * N[b] * gi
            K[3 * a + 0, 3 * b + 2] = Kup[0]
            K[3 * a + 1, 3 * b + 2] = Kup[1]
            Kpp = dt * mobility * gi.dot(gj) * vol
            if data["ppp"]:
                if mode == "dynamic":
                    Kpp += tau * data["gamma"] / (data["beta"] * dt) * (N[a] - Navg[a]) * (N[b] - Navg[b]) * vol
                else:
                    Kpp += tau * (N[a] - Navg[a]) * (N[b] - Navg[b]) * vol
            K[3 * a + 2, 3 * b + 2] = Kpp

            for comp in range(2):
                row = 3 * b + comp
                hcol = A[:, comp]
                eta = Gj.dot(hcol)
                dgi = -hcol * (Gj.dot(gi))
                dv = vol * eta
                drho = coeff_rho * eta
                dsigma = constitutive_tangent_apply(material, F_old, F_new, J_new, Gj, comp, eta, lam, mu)
                mech = (-dsigma @ gi - sigma_total @ dgi + N[a] * drho * gravity) * vol + (-sigma_total @ gi + N[a] * rho_mix * gravity) * dv
                if mode == "dynamic":
                    mech[comp] -= data["mass"] * N[a] * N[b] * (
                        1.0 / (data["beta"] * dt * dt) + data["damping"] * data["gamma"] / (data["beta"] * dt)
                    )
                K[3 * a + 0, row] = mech[0]
                K[3 * a + 1, row] = mech[1]

                scalar_grad = (-Gj.dot(gi) * hcol.dot(q) - Gj.dot(q) * hcol.dot(gi) + gi.dot(q) * eta) * dt * mobility * vol
                kpu = N[a] * (1.0 + log_ratio) * eta * vol + scalar_grad
                if data["ppp"]:
                    kpu += tau * (N[a] - Navg[a]) * (p_rate_gap if mode == "dynamic" else ppp_gap) * dv
                K[3 * a + 2, row] = kpu
    return K


def pack_unknowns(u, p):
    x = np.zeros(3 * len(p))
    for a in range(len(p)):
        x[3 * a:3 * a + 2] = u[a]
        x[3 * a + 2] = p[a]
    return x


def unpack_unknowns(x):
    n = len(x) // 3
    u = np.zeros((n, 2))
    p = np.zeros(n)
    for a in range(n):
        u[a] = x[3 * a:3 * a + 2]
        p[a] = x[3 * a + 2]
    return u, p


def finite_difference_jac(material, x, p_old, data, eps=1.0e-7):
    n = len(x)
    J = np.zeros((n, n))
    for j in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[j] += eps
        xm[j] -= eps
        up, pp = unpack_unknowns(xp)
        um, pm = unpack_unknowns(xm)
        rp = residual(material, up, pp, p_old, data)
        rm = residual(material, um, pm, p_old, data)
        J[:, j] = (rp - rm) / (2.0 * eps)
    return J


def random_case(rng, ppp, mode):
    N = rng.random(4)
    N /= N.sum()
    Navg = rng.random(4)
    Navg /= Navg.sum()
    Gref = rng.uniform(-1.0, 1.0, size=(4, 2))
    F_old = np.eye(2) + rng.uniform(-0.12, 0.12, size=(2, 2))
    if np.linalg.det(F_old) <= 0.25:
        F_old += 0.35 * np.eye(2)
    return {
        "N": N,
        "Navg": Navg,
        "Gref": Gref,
        "F_old": F_old,
        "J_old": np.linalg.det(F_old),
        "stress_old": rng.uniform(-0.2, 0.2, size=(2, 2)),
        "V0": rng.uniform(1.0e-2, 3.0e-2),
        "dt": rng.uniform(0.05, 0.3),
        "mobility": rng.uniform(1.0e-5, 5.0e-4),
        "tau": 0.5 / rng.uniform(150.0, 800.0),
        "phi_s0": rng.uniform(0.25, 0.75),
        "rho_s": rng.uniform(0.8, 2.2),
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
        "p_rate_old_nodes": rng.uniform(-0.3, 0.3, size=4),
        "p_acc_old_nodes": rng.uniform(-0.4, 0.4, size=4),
    }


def run_case(material, ppp, mode, trials=20):
    worst_abs = 0.0
    worst_rel = 0.0
    worst_id = -1
    seed = 7 if (ppp and mode == "static") else 17 if mode == "static" else 27 if ppp else 37
    rng = np.random.default_rng(seed)
    for trial in range(trials):
        data = random_case(rng, ppp, mode)
        u = rng.uniform(-4.0e-3, 4.0e-3, size=(len(data["N"]), 2))
        p = rng.uniform(-0.2, 0.2, size=len(data["N"]))
        p_old = rng.uniform(-0.2, 0.2, size=len(data["N"]))
        x = pack_unknowns(u, p)
        K_man = jacobian_manual(material, u, p, p_old, data)
        K_fd = finite_difference_jac(material, x, p_old, data)
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
                run_case(material, ppp, mode, trials=25)
