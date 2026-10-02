import numpy as np
import pytest
import taichi as ti

pytestmark = [
    pytest.mark.unit,
    pytest.mark.dem,
    pytest.mark.ipc,
    pytest.mark.contact,
    pytest.mark.cpu,
    pytest.mark.serial,
]

from src.physics_model.contact_model.ipc.ContactDistance import (
    edge_edge_distance_grad_hess_by_type,
    edge_edge_distance_type,
    point_triangle_distance_grad_hess_by_type,
    point_triangle_distance_type,
)


DHAT = 0.2
KAPPA = 3.0
COEFF = 1.7
DT = 1.0e-3
EPSV = 1.0e-4
FD_EPS = 1.0e-6

vf_x = None
ee_x = None
out_grad = None
out_hess = None
out_energy = None


@pytest.fixture(scope="module", autouse=True)
def _isolated_taichi_runtime():
    global vf_x, ee_x, out_grad, out_hess, out_energy

    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        debug=False,
        offline_cache=False,
        cpu_max_num_threads=1,
    )
    vf_x = ti.Vector.field(3, ti.f64, shape=4)
    ee_x = ti.Vector.field(3, ti.f64, shape=4)
    out_grad = ti.field(ti.f64, shape=12)
    out_hess = ti.field(ti.f64, shape=(12, 12))
    out_energy = ti.field(ti.f64, shape=())
    try:
        yield
    finally:
        ti.sync()
        ti.reset()


@ti.func
def barrier_distance2(dist2, active_gap2, kappa):
    s = ti.max(dist2, ti.max(1.0e-24 * active_gap2, 1.0e-30))
    energy = 0.0
    db = 0.0
    ddb = 0.0
    if s < active_gap2:
        diff = s - active_gap2
        ratio = diff / active_gap2
        log_term = ti.log(s / active_gap2)
        energy = -kappa * ratio * ratio * log_term
        db = kappa * (ratio * log_term * (-2.0 / active_gap2) - ratio * ratio / s)
        ddb = kappa * ((-2.0 * log_term - 4.0 * diff / s) / (active_gap2 * active_gap2) + ratio * ratio / (s * s))
    return energy, db, ddb


@ti.kernel
def eval_vf_barrier(contact_type: ti.template()):
    p = vf_x[0]
    a = vf_x[1]
    b = vf_x[2]
    c = vf_x[3]
    dist2, grad_d, hess_d = point_triangle_distance_grad_hess_by_type(p, a, b, c, contact_type)
    energy, db, ddb = barrier_distance2(dist2, DHAT * DHAT, KAPPA)
    out_energy[None] = COEFF * energy
    i = 0
    while i < 12:
        out_grad[i] = COEFF * db * grad_d[i]
        j = 0
        while j < 12:
            out_hess[i, j] = COEFF * (ddb * grad_d[i] * grad_d[j] + db * hess_d[i, j])
            j += 1
        i += 1


@ti.kernel
def eval_ee_barrier(contact_type: ti.template()):
    p0 = ee_x[0]
    p1 = ee_x[1]
    q0 = ee_x[2]
    q1 = ee_x[3]
    dist2, grad_d, hess_d = edge_edge_distance_grad_hess_by_type(p0, p1, q0, q1, contact_type)
    energy, db, ddb = barrier_distance2(dist2, DHAT * DHAT, KAPPA)
    out_energy[None] = COEFF * energy
    i = 0
    while i < 12:
        out_grad[i] = COEFF * db * grad_d[i]
        j = 0
        while j < 12:
            out_hess[i, j] = COEFF * (ddb * grad_d[i] * grad_d[j] + db * hess_d[i, j])
            j += 1
        i += 1


@ti.kernel
def classify_vf() -> ti.i32:
    return point_triangle_distance_type(vf_x[0], vf_x[1], vf_x[2], vf_x[3])


@ti.kernel
def classify_ee() -> ti.i32:
    return edge_edge_distance_type(ee_x[0], ee_x[1], ee_x[2], ee_x[3])


def barrier_distance2_np(dist2, active_gap2=DHAT * DHAT, kappa=KAPPA):
    s = max(float(dist2), max(1.0e-24 * active_gap2, 1.0e-30))
    if s >= active_gap2:
        return 0.0, 0.0, 0.0
    diff = s - active_gap2
    ratio = diff / active_gap2
    log_term = np.log(s / active_gap2)
    energy = -kappa * ratio * ratio * log_term
    db = kappa * (ratio * log_term * (-2.0 / active_gap2) - ratio * ratio / s)
    ddb = kappa * ((-2.0 * log_term - 4.0 * diff / s) / (active_gap2 * active_gap2) + ratio * ratio / (s * s))
    return energy, db, ddb


def point_triangle_energy_np(x):
    pts = x.reshape(4, 3)
    p, a, b, c = pts
    n = np.cross(b - a, c - a)
    n = n / np.linalg.norm(n)
    dist2 = np.dot(p - a, n) ** 2
    energy, _, _ = barrier_distance2_np(dist2)
    return COEFF * energy


def edge_edge_energy_np(x):
    pts = x.reshape(4, 3)
    p0, p1, q0, q1 = pts
    u = p1 - p0
    v = q1 - q0
    w0 = p0 - q0
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)
    denom = a * c - b * b
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    closest = p0 + s * u - (q0 + t * v)
    dist2 = np.dot(closest, closest)
    energy, _, _ = barrier_distance2_np(dist2)
    return COEFF * energy


def trilinear_sdf(point, values, h):
    x, y, z = point / h
    phi = 0.0
    grad_r = np.zeros(3)
    hess_r = np.zeros((3, 3))
    for i in range(2):
        wx, dwx = (1.0 - x, -1.0) if i == 0 else (x, 1.0)
        for j in range(2):
            wy, dwy = (1.0 - y, -1.0) if j == 0 else (y, 1.0)
            for k in range(2):
                wz, dwz = (1.0 - z, -1.0) if k == 0 else (z, 1.0)
                val = values[i, j, k]
                phi += val * wx * wy * wz
                grad_r += val * np.array([dwx * wy * wz, wx * dwy * wz, wx * wy * dwz])
                hess_r[0, 1] += val * dwx * dwy * wz
                hess_r[0, 2] += val * dwx * wy * dwz
                hess_r[1, 2] += val * wx * dwy * dwz
    hess_r[1, 0] = hess_r[0, 1]
    hess_r[2, 0] = hess_r[0, 2]
    hess_r[2, 1] = hess_r[1, 2]
    return phi, grad_r / h, hess_r / (h * h)


def barrier_gap_np(gap, active_gap=DHAT, kappa=KAPPA):
    s = max(float(gap), max(1.0e-12 * active_gap, 1.0e-30))
    if s >= active_gap:
        return 0.0, 0.0, 0.0
    diff = s - active_gap
    log_term = np.log(s / active_gap)
    energy = -kappa * diff * diff * log_term
    db = kappa * (diff * log_term * (-2.0) - diff * diff / s)
    ddb = kappa * (-2.0 * log_term - 4.0 * diff / s + diff * diff / (s * s))
    return energy, db, ddb


def soft_sdf_energy_grad_hess(point, values, h):
    phi, grad_phi, hess_phi = trilinear_sdf(point, values, h)
    energy, db, ddb = barrier_gap_np(phi)
    return COEFF * energy, COEFF * db * grad_phi, COEFF * (ddb * np.outer(grad_phi, grad_phi) + db * hess_phi)


def friction_f0(vbarnorm, epsv=EPSV, dt=DT):
    if vbarnorm < epsv:
        vh = vbarnorm * dt
        eh = epsv * dt
        return vh * vh * (-vh / 3.0 + eh) / (eh * eh) + eh / 3.0
    return vbarnorm * dt


def friction_f1_div_norm(vbarnorm, epsv=EPSV):
    if vbarnorm < epsv:
        return (-vbarnorm + 2.0 * epsv) / (epsv * epsv)
    return 1.0 / max(vbarnorm, 1.0e-30)


def friction_hess_term(vbarnorm, epsv=EPSV):
    if vbarnorm < epsv:
        return -1.0 / (epsv * epsv)
    return -1.0 / max(vbarnorm * vbarnorm, 1.0e-30)


def semi_implicit_friction(weights, normal, hat_rel, coeff, x):
    pts = x.reshape(len(weights), 3)
    rel = np.einsum("i,ij->j", weights, pts)
    n = normal / np.linalg.norm(normal)
    P = np.eye(3) - np.outer(n, n)
    vbar = P @ ((rel - hat_rel) / DT)
    vbarnorm = np.linalg.norm(vbar)
    energy = coeff * friction_f0(vbarnorm)
    f1 = friction_f1_div_norm(vbarnorm)
    grad_rel = coeff * f1 * (P @ vbar)
    f_hess = friction_hess_term(vbarnorm)
    inner = coeff * f1 * np.eye(3)
    if vbarnorm > 1.0e-30:
        inner += coeff * f_hess / vbarnorm * np.outer(vbar, vbar)
    hess_rel = P @ inner @ P.T / DT
    grad = np.concatenate([w * grad_rel for w in weights])
    hess = np.zeros((3 * len(weights), 3 * len(weights)))
    for i, wi in enumerate(weights):
        for j, wj in enumerate(weights):
            hess[3 * i:3 * i + 3, 3 * j:3 * j + 3] = wi * wj * hess_rel
    return energy, grad, hess


def fd_gradient(func, x, eps=FD_EPS):
    grad = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (func(xp) - func(xm)) / (2.0 * eps)
    return grad


def fd_hessian_from_grad(grad_func, x, eps=FD_EPS):
    hess = np.zeros((x.size, x.size))
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        hess[:, i] = (grad_func(xp) - grad_func(xm)) / (2.0 * eps)
    return 0.5 * (hess + hess.T)


def check_close(name, actual, expected, atol=2.0e-4, rtol=2.0e-4):
    if not np.all(np.isfinite(actual)):
        raise AssertionError(f"{name} produced a non-finite analytical value")
    if not np.all(np.isfinite(expected)):
        raise AssertionError(f"{name} produced a non-finite finite-difference reference")
    err = np.linalg.norm(actual - expected, ord=np.inf)
    ref = max(1.0, np.linalg.norm(expected, ord=np.inf))
    rel = err / ref
    print(f"{name}: abs={err:.3e}, rel={rel:.3e}")
    if err > atol and rel > rtol:
        raise AssertionError(f"{name} finite-difference mismatch: abs={err:.6e}, rel={rel:.6e}")


def eval_taichi_vf(x):
    vf_x.from_numpy(x.reshape(4, 3))
    dtype = int(classify_vf())
    eval_vf_barrier(dtype)
    return out_energy[None], out_grad.to_numpy(), out_hess.to_numpy()


def eval_taichi_ee(x):
    ee_x.from_numpy(x.reshape(4, 3))
    dtype = int(classify_ee())
    eval_ee_barrier(dtype)
    return out_energy[None], out_grad.to_numpy(), out_hess.to_numpy()


def run_barrier_checks():
    vf = np.array([
        [0.23, 0.31, 0.045],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64).reshape(-1)
    _, vf_grad, vf_hess = eval_taichi_vf(vf)
    check_close("affine-affine VF barrier grad", vf_grad, fd_gradient(point_triangle_energy_np, vf))
    check_close("affine-affine VF barrier hess", vf_hess, fd_hessian_from_grad(lambda x: eval_taichi_vf(x)[1], vf), atol=5.0e-3, rtol=5.0e-3)

    ee = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.35, -0.2, 0.055],
        [0.35, 0.8, 0.055],
    ], dtype=np.float64).reshape(-1)
    _, ee_grad, ee_hess = eval_taichi_ee(ee)
    check_close("affine-affine EE barrier grad", ee_grad, fd_gradient(edge_edge_energy_np, ee))
    check_close("affine-affine EE barrier hess", ee_hess, fd_hessian_from_grad(lambda x: eval_taichi_ee(x)[1], ee), atol=5.0e-3, rtol=5.0e-3)

    soft_weights = np.array([0.35, 0.65], dtype=np.float64)
    affine_basis = np.array([
        [0.4, 0.3, 0.2, 0.1],
        [0.1, 0.5, 0.25, 0.15],
        [0.2, 0.2, 0.45, 0.15],
    ], dtype=np.float64)
    ndof = 3 * (2 + 4)
    J = np.zeros((12, ndof), dtype=np.float64)
    for i, w in enumerate(soft_weights):
        J[0:3, 3 * i:3 * i + 3] = w * np.eye(3)
    for lv in range(3):
        for a in range(4):
            J[3 * (lv + 1):3 * (lv + 2), 3 * (2 + a):3 * (3 + a)] = affine_basis[lv, a] * np.eye(3)
    q = np.array([
        [0.20, 0.08, 0.04],
        [0.24615384615384617, 0.4338461538461539, 0.04769230769230769],
        [-0.1, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [0.0, 1.1, 0.0],
        [0.25, 0.2, 0.1],
    ], dtype=np.float64).reshape(-1)
    local = J @ q
    _, g_local, H_local = eval_taichi_vf(local)
    mapped_grad = J.T @ g_local
    mapped_hess = J.T @ H_local @ J

    def affine_soft_energy(qx):
        return eval_taichi_vf(J @ qx)[0]

    def affine_soft_grad(qx):
        _, gl, Hl = eval_taichi_vf(J @ qx)
        return J.T @ gl

    check_close("affine-soft mapped barrier grad", mapped_grad, fd_gradient(affine_soft_energy, q))
    check_close("affine-soft mapped barrier hess", mapped_hess, fd_hessian_from_grad(affine_soft_grad, q), atol=5.0e-3, rtol=5.0e-3)

    values = np.array([
        [[0.10, 0.13], [0.14, 0.18]],
        [[0.16, 0.21], [0.19, 0.27]],
    ], dtype=np.float64)
    h = 0.4
    p = np.array([0.13, 0.17, 0.19], dtype=np.float64)
    _, sdf_grad, sdf_hess = soft_sdf_energy_grad_hess(p, values, h)

    def sdf_energy(px):
        return soft_sdf_energy_grad_hess(px, values, h)[0]

    def sdf_grad_func(px):
        return soft_sdf_energy_grad_hess(px, values, h)[1]

    check_close("soft-soft SDF barrier grad", sdf_grad, fd_gradient(sdf_energy, p))
    check_close("soft-soft SDF barrier hess", sdf_hess, fd_hessian_from_grad(sdf_grad_func, p), atol=5.0e-3, rtol=5.0e-3)


def run_friction_checks():
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for label, rel_scale in (("dynamic", 3.0e-4), ("static-smoothed", 2.0e-8)):
        fd_eps = 1.0e-6 if label == "dynamic" else 1.0e-10
        weights = np.array([1.0, -0.45, -0.35, -0.20], dtype=np.float64)
        x = np.array([
            [0.25 + rel_scale, 0.30 - 0.5 * rel_scale, 0.04],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64).reshape(-1)
        hat_rel = np.array([0.25, 0.30, 0.04], dtype=np.float64) - (0.45 * x[3:6] + 0.35 * x[6:9] + 0.20 * x[9:12])
        coeff = 0.9
        _, grad, hess = semi_implicit_friction(weights, normal, hat_rel, coeff, x)
        energy = lambda xx: semi_implicit_friction(weights, normal, hat_rel, coeff, xx)[0]
        grad_func = lambda xx: semi_implicit_friction(weights, normal, hat_rel, coeff, xx)[1]
        check_close(f"affine-affine/affine-soft {label} friction grad", grad, fd_gradient(energy, x, fd_eps), atol=2.0e-5, rtol=2.0e-5)
        check_close(f"affine-affine/affine-soft {label} friction hess", hess, fd_hessian_from_grad(grad_func, x, fd_eps), atol=2.0e-3, rtol=2.0e-3)

        weights_soft = np.array([1.0], dtype=np.float64)
        xs = np.array([[0.25 + rel_scale, 0.30 - 0.5 * rel_scale, 0.04]], dtype=np.float64).reshape(-1)
        hat_soft = np.array([0.25, 0.30, 0.04], dtype=np.float64)
        _, grad_s, hess_s = semi_implicit_friction(weights_soft, normal, hat_soft, coeff, xs)
        energy_s = lambda xx: semi_implicit_friction(weights_soft, normal, hat_soft, coeff, xx)[0]
        grad_s_func = lambda xx: semi_implicit_friction(weights_soft, normal, hat_soft, coeff, xx)[1]
        check_close(f"soft-soft {label} friction grad", grad_s, fd_gradient(energy_s, xs, fd_eps), atol=2.0e-5, rtol=2.0e-5)
        check_close(f"soft-soft {label} friction hess", hess_s, fd_hessian_from_grad(grad_s_func, xs, fd_eps), atol=2.0e-3, rtol=2.0e-3)


def test_ipc_barrier_gradients_and_hessians_match_finite_difference():
    run_barrier_checks()


def test_ipc_lagged_friction_gradients_and_hessians_match_finite_difference():
    run_friction_checks()


def main():
    test_ipc_barrier_gradients_and_hessians_match_finite_difference()
    test_ipc_lagged_friction_gradients_and_hessians_match_finite_difference()
    print("IPC contact finite-difference checks passed")


if __name__ == "__main__":
    main()
