"""Deterministic 3-D static two-phase local-Jacobian diagnostic."""

import numpy as np


def neo_hookean_stress(F, J, lam, mu):
    b = F @ F.T
    I = np.eye(3)
    return mu / J * (b - I) + lam * np.log(J) / J * I


def deviatoric(A):
    return A - np.trace(A) / 3.0 * np.eye(3)


def elastic_strain_from_stress(stress, lam, mu):
    return (stress - lam / (3.0 * lam + 2.0 * mu) * np.trace(stress) * np.eye(3)) / (2.0 * mu)


def dp_material_constants(angle_deg, cohesion):
    angle = np.deg2rad(angle_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    A = 2.0 * np.sqrt(6.0) * cohesion * c / (3.0 - s)
    B = 2.0 * np.sqrt(6.0) * s / (3.0 - s)
    return A, B


def dp_yield_value(p_mean, t_dev, Af, Bf, shape_factor):
    kf = shape_factor * Af
    return np.sqrt(t_dev * t_dev + kf * kf) - Af + Bf * p_mean


def dp_local_update(eps_trial, lam, mu, friction_angle, dilation_angle, cohesion, shape_factor, tol=1.0e-10, max_iters=25):
    I = np.eye(3)
    bulk = lam + 2.0 * mu / 3.0
    sigma_trial = lam * np.trace(eps_trial) * I + 2.0 * mu * eps_trial
    p_trial = np.trace(sigma_trial) / 3.0
    s_trial = deviatoric(sigma_trial)
    t_trial = np.sqrt(max(np.sum(s_trial * s_trial), 0.0))

    Af, Bf = dp_material_constants(friction_angle, cohesion)
    Ag, Bg = dp_material_constants(dilation_angle, cohesion)
    f_trial = dp_yield_value(p_trial, t_trial, Af, Bf, shape_factor)

    sigma = sigma_trial.copy()
    eps_elastic = eps_trial.copy()
    t_dev = t_trial
    delta_lambda = 0.0

    if f_trial > tol and p_trial < 0.0:
        for _ in range(max_iters):
            kf = shape_factor * Af
            kg = shape_factor * Ag
            rf = np.sqrt(max(t_dev * t_dev + kf * kf, 1.0e-30))
            rg = np.sqrt(max(t_dev * t_dev + kg * kg, 1.0e-30))
            R1 = t_dev - t_trial + 2.0 * mu * delta_lambda * t_dev / rg
            R2 = rf - Af + Bf * (p_trial - bulk * Bg * delta_lambda)
            if max(abs(R1), abs(R2)) < tol:
                break
            a11 = 1.0 + 2.0 * mu * delta_lambda * kg * kg / (rg ** 3)
            a12 = 2.0 * mu * t_dev / rg
            a21 = t_dev / rf
            a22 = -Bf * bulk * Bg
            det = a11 * a22 - a12 * a21
            if abs(det) < 1.0e-20:
                det = np.copysign(1.0e-20, det if det != 0.0 else 1.0)
            dt = (-a22 * R1 + a12 * R2) / det
            dl = (a21 * R1 - a11 * R2) / det
            t_dev = max(t_dev + dt, 0.0)
            delta_lambda += dl

        p_new = p_trial - bulk * Bg * delta_lambda
        alpha = t_dev / t_trial if t_trial > 1.0e-20 else 0.0
        s_new = alpha * s_trial
        sigma = p_new * I + s_new

        kg = shape_factor * Ag
        rg = np.sqrt(max(t_dev * t_dev + kg * kg, 1.0e-30))
        flow = Bg / 3.0 * I
        if rg > 1.0e-20:
            flow += s_new / rg
        eps_elastic = eps_trial - delta_lambda * flow

    return sigma, eps_elastic, s_trial, p_trial, t_trial, t_dev, delta_lambda


def dp_consistent_tangent_apply(deps, s_trial, p_trial, t_trial, t_dev, delta_lambda, lam, mu, friction_angle, dilation_angle, cohesion, shape_factor, tol=1.0e-10):
    I = np.eye(3)
    bulk = lam + 2.0 * mu / 3.0
    Af, Bf = dp_material_constants(friction_angle, cohesion)
    Ag, Bg = dp_material_constants(dilation_angle, cohesion)
    kf = shape_factor * Af
    kg = shape_factor * Ag

    dp_trial = bulk * np.trace(deps)
    ds_trial = 2.0 * mu * deviatoric(deps)
    sigma = lam * np.trace(deps) * I + 2.0 * mu * deps
    f_trial = dp_yield_value(p_trial, t_trial, Af, Bf, shape_factor)
    if f_trial <= tol or p_trial >= 0.0:
        return sigma

    n_trial = np.zeros((3, 3))
    dt_trial = 0.0
    alpha = 0.0
    if t_trial > 1.0e-20:
        n_trial = s_trial / t_trial
        dt_trial = np.sum(n_trial * ds_trial)
        alpha = t_dev / t_trial

    rf = np.sqrt(max(t_dev * t_dev + kf * kf, 1.0e-30))
    rg = np.sqrt(max(t_dev * t_dev + kg * kg, 1.0e-30))
    a11 = 1.0 + 2.0 * mu * delta_lambda * kg * kg / (rg ** 3)
    a12 = 2.0 * mu * t_dev / rg
    a21 = t_dev / rf
    a22 = -Bf * bulk * Bg
    det = a11 * a22 - a12 * a21
    if abs(det) < 1.0e-20:
        det = np.copysign(1.0e-20, det if det != 0.0 else 1.0)

    dt_dev_coeff = a22 / det
    dt_vol_coeff = a12 * Bf / det
    dl_dev_coeff = -a21 / det
    dl_vol_coeff = -a11 * Bf / det

    dt = dt_dev_coeff * dt_trial + dt_vol_coeff * dp_trial
    dlam = dl_dev_coeff * dt_trial + dl_vol_coeff * dp_trial
    dp_new = dp_trial - bulk * Bg * dlam
    ds_new = alpha * ds_trial
    if t_trial > 1.0e-20:
        ds_new += (dt - alpha * dt_trial) * n_trial
    return dp_new * I + ds_new


def constitutive_response(material, stress_old, F_old, F_new, J_new, grad_u, lam, mu):
    if material == "linearElastic":
        eps = 0.5 * (grad_u + grad_u.T)
        return stress_old + lam * np.trace(eps) * np.eye(3) + 2.0 * mu * eps
    if material == "neoHookean":
        return neo_hookean_stress(F_new, J_new, lam, mu)
    if material == "druckerPrager":
        eps_old = elastic_strain_from_stress(stress_old, lam, mu)
        sigma, *_ = dp_local_update(
            eps_old + 0.5 * (grad_u + grad_u.T),
            lam,
            mu,
            20.0,
            10.0,
            1.0,
            0.0,
        )
        return sigma
    raise ValueError(material)


def constitutive_tangent_apply(material, stress_old, F_old, F_new, J_new, grad_u, G, comp, eta, lam, mu):
    deps = np.zeros((3, 3))
    for d in range(3):
        deps[comp, d] += 0.5 * G[d]
        deps[d, comp] += 0.5 * G[d]
    if material == "linearElastic":
        return lam * np.trace(deps) * np.eye(3) + 2.0 * mu * deps
    if material == "neoHookean":
        H = F_old.T @ G
        dF = np.zeros((3, 3))
        for d in range(3):
            dF[comp, d] = H[d]
        b = F_new @ F_new.T
        db = dF @ F_new.T + F_new @ dF.T
        dJ = J_new * eta
        return (
            mu / J_new * db
            - mu / (J_new * J_new) * (b - np.eye(3)) * dJ
            + lam * (1.0 - np.log(J_new)) / (J_new * J_new) * dJ * np.eye(3)
        )
    if material == "druckerPrager":
        eps_old = elastic_strain_from_stress(stress_old, lam, mu)
        _, _, s_trial, p_trial, t_trial, t_dev, delta_lambda = dp_local_update(
            eps_old + 0.5 * (grad_u + grad_u.T),
            lam,
            mu,
            20.0,
            10.0,
            1.0,
            0.0,
        )
        return dp_consistent_tangent_apply(
            deps,
            s_trial,
            p_trial,
            t_trial,
            t_dev,
            delta_lambda,
            lam,
            mu,
            20.0,
            10.0,
            1.0,
            0.0,
        )
    raise ValueError(material)


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

    grad_u = np.zeros((3, 3))
    p_new = 0.0
    p_old = 0.0
    p_avg = 0.0
    p_old_avg = 0.0
    for a in range(len(N)):
        grad_u += np.outer(u_new[a], Gref[a])
        p_new += N[a] * p_new_nodes[a]
        p_old += N[a] * p_old_nodes[a]
        p_avg += Navg[a] * p_new_nodes[a]
        p_old_avg += Navg[a] * p_old_nodes[a]

    deltaF = np.eye(3) + grad_u
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
    sigma_total = sigma_eff - p_new * np.eye(3)
    ppp_gap = (p_new - p_avg) - (p_old - p_old_avg)

    r = np.zeros(4 * len(N))
    for a in range(len(N)):
        gi = gcur[a]
        base_force = (-sigma_total @ gi + N[a] * rho_mix * gravity) * vol
        r[4 * a:4 * a + 3] = base_force
        darcy = dt * mobility * gi.dot(q)
        mass = (N[a] * log_ratio + darcy) * vol
        if data["ppp"]:
            mass += tau * (N[a] - Navg[a]) * ppp_gap * vol
        r[4 * a + 3] = mass
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

    grad_u = np.zeros((3, 3))
    p_new = 0.0
    p_old = 0.0
    p_avg = 0.0
    p_old_avg = 0.0
    for a in range(len(N)):
        grad_u += np.outer(u_new[a], Gref[a])
        p_new += N[a] * p_new_nodes[a]
        p_old += N[a] * p_old_nodes[a]
        p_avg += Navg[a] * p_new_nodes[a]
        p_old_avg += Navg[a] * p_old_nodes[a]

    deltaF = np.eye(3) + grad_u
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
    sigma_total = sigma_eff - p_new * np.eye(3)
    ppp_gap = (p_new - p_avg) - (p_old - p_old_avg)

    K = np.zeros((4 * len(N), 4 * len(N)))
    for a in range(len(N)):
        gi = gcur[a]
        for b in range(len(N)):
            Gj = Gref[b]
            gj = gcur[b]
            Kup = vol * N[b] * gi
            K[4 * a + 0:4 * a + 3, 4 * b + 3] = Kup

            Kpp = dt * mobility * gi.dot(gj) * vol
            if data["ppp"]:
                Kpp += tau * (N[a] - Navg[a]) * (N[b] - Navg[b]) * vol
            K[4 * a + 3, 4 * b + 3] = Kpp

            for comp in range(3):
                col = 4 * b + comp
                hcol = A[:, comp]
                eta = Gj.dot(hcol)
                dgi = -hcol * (Gj.dot(gi))
                dv = vol * eta
                drho = coeff_rho * eta
                dsigma = constitutive_tangent_apply(material, stress_old, F_old, F_new, J_new, grad_u, Gj, comp, eta, lam, mu)
                mech = (-dsigma @ gi - sigma_total @ dgi + N[a] * drho * gravity) * vol
                mech += (-sigma_total @ gi + N[a] * rho_mix * gravity) * dv
                K[4 * a + 0:4 * a + 3, col] = mech

                scalar_grad = (
                    -Gj.dot(gi) * hcol.dot(q)
                    - Gj.dot(q) * hcol.dot(gi)
                    + gi.dot(q) * eta
                ) * dt * mobility * vol
                kpu = N[a] * (1.0 + log_ratio) * eta * vol + scalar_grad
                if data["ppp"]:
                    kpu += tau * (N[a] - Navg[a]) * ppp_gap * dv
                K[4 * a + 3, col] = kpu
    return K


def pack_unknowns(u, p):
    x = np.zeros(4 * len(p))
    for a in range(len(p)):
        x[4 * a:4 * a + 3] = u[a]
        x[4 * a + 3] = p[a]
    return x


def unpack_unknowns(x):
    n = len(x) // 4
    u = np.zeros((n, 3))
    p = np.zeros(n)
    for a in range(n):
        u[a] = x[4 * a:4 * a + 3]
        p[a] = x[4 * a + 3]
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


def random_case(rng, ppp):
    support = 8
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
        "N": N,
        "Navg": Navg,
        "Gref": Gref,
        "F_old": F_old,
        "J_old": np.linalg.det(F_old),
        "stress_old": stress_old,
        "V0": rng.uniform(8.0e-3, 2.5e-2),
        "dt": rng.uniform(0.05, 0.3),
        "mobility": rng.uniform(1.0e-5, 5.0e-4),
        "tau": 0.5 / rng.uniform(150.0, 800.0),
        "phi_s0": rng.uniform(0.25, 0.75),
        "rho_s": rng.uniform(0.8, 2.2),
        "rho_f": rng.uniform(0.8, 1.4),
        "gravity": np.array([0.0, 0.0, -rng.uniform(0.0, 9.81)], dtype=np.float64),
        "lam": rng.uniform(150.0, 700.0),
        "mu": rng.uniform(100.0, 500.0),
        "ppp": ppp,
    }


def make_admissible_old_stress(material, data, rng):
    if material != "druckerPrager":
        return
    eps0 = rng.uniform(-2.0e-3, 2.0e-3, size=(3, 3))
    eps0 = 0.5 * (eps0 + eps0.T)
    sigma0, *_ = dp_local_update(
        eps0,
        data["lam"],
        data["mu"],
        20.0,
        10.0,
        1.0,
        0.0,
    )
    data["stress_old"] = sigma0


def run_case(material, ppp, trials=16):
    worst_abs = 0.0
    worst_rel = 0.0
    worst_id = -1
    rng = np.random.default_rng(11 if ppp else 19)
    for trial in range(trials):
        data = random_case(rng, ppp)
        make_admissible_old_stress(material, data, rng)
        u = rng.uniform(-4.0e-3, 4.0e-3, size=(len(data["N"]), 3))
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
    print(f"{material:13s} {tag:7s}: worst_abs={worst_abs:.3e}, worst_rel={worst_rel:.3e}, trial={worst_id}")
    return worst_abs, worst_rel


if __name__ == "__main__":
    for material in ("linearElastic", "neoHookean", "druckerPrager"):
        for ppp in (False, True):
            run_case(material, ppp, trials=20)
