import os
import sys

import math

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import numpy as np
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors
import alphashape
from scipy.ndimage import binary_dilation, binary_closing
from skimage.measure import find_contours
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from matplotlib import rcParams
from third_party.tablelegend import tablelegend

params = {
    "backend": "ps",
    "font.size": 26,
    "lines.linewidth": 4.5,
    "lines.markersize": 12,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "xtick.major.pad": 12,
    "ytick.major.pad": 12,
    "axes.labelpad": 8,
    "legend.fontsize": 26,
    "figure.figsize": [12, 9],
    "font.family": "serif",
    "text.usetex": True,
    "font.serif": "Arial",
    "savefig.dpi": 300,
}
rcParams.update(params)

color = [
    (0 / 255, 0 / 255, 0 / 255),
    (255 / 255, 0 / 255, 0 / 255),
    (94 / 255, 114 / 255, 255 / 255),
    (0 / 255, 128 / 255, 0 / 255),
    (120 / 255, 120 / 255, 120 / 255),
    (200 / 255, 160 / 255, 70 / 255),
    (0.6, 0.2, 0.8),
]


def deduplicate(points, decimals=6):
    return np.unique(np.round(points, decimals=decimals), axis=0)


def remove_outliers(points, k=10, std_ratio=1e-6):
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    mean_dist = distances[:, 1:].mean(axis=1)

    threshold = mean_dist.mean() + std_ratio * mean_dist.std()
    mask = mean_dist < threshold
    return deduplicate(points[mask])


def boundary_from_raster(points, rad=0.25, grid_size=601, dilate_iter=2, close_iter=2):
    points = np.asarray(points, dtype=float)
    points = np.unique(points, axis=0)

    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)

    dx = xmax - xmin
    dy = ymax - ymin
    if dx == 0 or dy == 0:
        raise ValueError("点退化，无法栅格化")

    px_min = ((points[:, 0] - xmin - rad) / dx * (grid_size - 1)).astype(int)
    px_max = ((points[:, 0] - xmin + rad) / dx * (grid_size - 1)).astype(int)
    py_min = ((points[:, 1] - ymin - rad) / dy * (grid_size - 1)).astype(int)
    py_max = ((points[:, 1] - ymin + rad) / dy * (grid_size - 1)).astype(int)

    px_min = np.clip(px_min, 0, grid_size - 1)
    px_max = np.clip(px_max, 0, grid_size - 1)
    py_min = np.clip(py_min, 0, grid_size - 1)
    py_max = np.clip(py_max, 0, grid_size - 1)

    img = np.zeros((grid_size, grid_size), dtype=bool)
    for x0, x1, y0, y1 in zip(px_min, px_max, py_min, py_max):
        img[y0 : y1 + 1, x0 : x1 + 1] = True

    for _ in range(dilate_iter):
        img = binary_dilation(img)
    for _ in range(close_iter):
        img = binary_closing(img)

    contours = find_contours(img.astype(float), level=0.5)
    if not contours:
        raise ValueError("栅格法也没找到轮廓")

    contour = max(contours, key=len)

    cy = contour[:, 0]
    cx = contour[:, 1]

    x = cx / (grid_size - 1) * dx + xmin
    y = cy / (grid_size - 1) * dy + ymin

    boundary = np.column_stack([x, y])
    return boundary


def alpha_shape(pos, alpha=2.0, jitter=0.0):
    points = deduplicate(pos)
    if len(points) < 4:
        raise ValueError("点太少，无法构造凹边界")

    if jitter > 0:
        points = points + np.random.normal(scale=jitter, size=points.shape)

    shape = alphashape.alphashape(points, alpha)

    if isinstance(shape, Polygon):
        boundary = np.array(shape.exterior.coords)
        return boundary
    elif isinstance(shape, MultiPolygon):
        poly = max(shape.geoms, key=lambda g: g.area)
        boundary = np.array(poly.exterior.coords)
        return boundary
    else:
        raise ValueError("没有得到有效闭合边界，请调整 alpha")


def get_side_view(path, curr=20):
    try:
        data = np.load(path + "/particles/MPMParticle{0:06d}.npz".format(curr), allow_pickle=True)
    except:
        data = np.load(path + "/particles/MPMParticle{0:06d}.npz".format(1), allow_pickle=True)
    pos = data["position"][:, [0, 2]]
    return boundary_from_raster(pos)


def get_top_view(path, curr=20):
    try:
        data = np.load(path + "/particles/MPMParticle{0:06d}.npz".format(curr), allow_pickle=True)
    except:
        data = np.load(path + "/particles/MPMParticle{0:06d}.npz".format(1), allow_pickle=True)
    pos = data["position"]
    pos = pos[pos[:, 2] > 6.0][:, [0, 1]]
    return boundary_from_raster(pos)


def get_influence_view(path, curr=20, top_filter=15.6):
    data0 = np.load(path + "/particles/MPMParticle{0:06d}.npz".format(0), allow_pickle=True)
    if curr == 0:
        pos0 = data0["position"]
        pos_filter = pos0[pos0[:, 2] > 15.5]
        pos_filter = pos_filter[:, [0, 1]]
        return boundary_from_raster(pos_filter)
    else:
        try:
            data = np.load(path + "/particles/MPMParticle{0:06d}.npz".format(curr), allow_pickle=True)
        except:
            data = np.load(path + "/particles/MPMParticle{0:06d}.npz".format(1), allow_pickle=True)
        pos0 = data0["position"]
        pos = data["position"]
        pos_filter = pos[pos0[:, 2] > 15.5]
        pos_filter = pos_filter[pos_filter[:, 2] > top_filter][:, [0, 1]]
        return boundary_from_raster(pos_filter)


def max_z(path, curr=20, nx=300, ny=300, xlim=(0.0, 50.0), ylim=(0.0, 50.0), smooth_sigma=5.0):
    try:
        data = np.load(f"{path}/particles/MPMParticle{curr:06d}.npz", allow_pickle=True)
    except:
        data = np.load(f"{path}/particles/MPMParticle{1:06d}.npz", allow_pickle=True)
    points = data["position"]

    mask = np.isfinite(points).all(axis=1)
    points = points[mask]

    x_min, x_max = xlim
    y_min, y_max = ylim

    mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    points = points[mask]

    if len(points) == 0:
        raise ValueError("指定范围内没有有效粒子点")

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)

    ix = np.floor((points[:, 0] - x_min) / (x_max - x_min) * (nx - 1)).astype(int)
    iy = np.floor((points[:, 1] - y_min) / (y_max - y_min) * (ny - 1)).astype(int)

    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    Z = np.full((ny, nx), np.nan)
    Zmax = np.full((ny, nx), -np.inf)

    np.maximum.at(Zmax, (iy, ix), points[:, 2])

    valid = Zmax > -np.inf
    Z[valid] = Zmax[valid]

    yy, xx = np.where(np.isfinite(Z))
    if len(xx) == 0:
        raise ValueError("无法生成高程场，网格中没有有效值")

    sample_xy = np.column_stack([x[xx], y[yy]])
    sample_z = Z[yy, xx]

    Z_fill = griddata(sample_xy, sample_z, (X, Y), method="nearest")

    if smooth_sigma is not None and smooth_sigma > 0:
        Z_fill = gaussian_filter(Z_fill, sigma=smooth_sigma)

    if path == "Flat":
        aaa = X > 13.4
        X[aaa] = (X[aaa] - 13.4) * 1.8

    return X, Y, Z_fill


def flat_fliter(boundary):
    mask = (boundary[:, 1] < 49.5) & (boundary[:, 1] > 0.5) & (boundary[:, 0] > 1.0)
    boundary = boundary[mask]

    aaa = boundary[:, 0] > 30.0
    boundary[aaa, 0] = (boundary[aaa, 0] - 30) * 1.8 + 30
    return boundary


def convex_fliter(boundary):
    mask = (boundary[:, 1] < 49.5) & (boundary[:, 0] > 0.5)
    return boundary[mask]


def comm_filter(boundary):
    mask = (boundary[:, 1] < 49.5) & (boundary[:, 0] > 0.5) & (boundary[:, 1] > 0.5) & (boundary[:, 0] < 49.5)
    return boundary[mask]


def plot(ax, profile_dict, label, c, l="-", suffix=""):
    boundary = profile_dict[label]
    ax.plot(boundary[:, 0], boundary[:, 1], color=color[c], linestyle=l, label=label + suffix)


# ============================================================
# 统计辅助函数
# ============================================================


def get_case_paths(base_name):
    return [base_name] + [base_name + "_rand{}".format(i) for i in range(1, 6)]


def get_boundary_for_band(path, mode="top", curr=20, top_filter=15.6):
    if mode == "top":
        boundary = comm_filter(get_top_view(path, curr))
    elif mode == "inf":
        boundary = comm_filter(get_influence_view(path, curr, top_filter=top_filter))
    else:
        raise ValueError("mode must be 'top' or 'inf'")
    return boundary


# ---------------- Flat: x = f(y) ----------------


def sort_and_unique_by_y(boundary):
    boundary = np.asarray(boundary, dtype=float)
    order = np.argsort(boundary[:, 1])
    boundary = boundary[order]

    y = boundary[:, 1]
    x = boundary[:, 0]

    y_round = np.round(y, 6)
    y_unique, inverse = np.unique(y_round, return_inverse=True)

    x_unique = np.zeros(len(y_unique), dtype=float)
    count = np.zeros(len(y_unique), dtype=int)

    for i, idx in enumerate(inverse):
        x_unique[idx] += x[i]
        count[idx] += 1

    x_unique /= count
    return y_unique, x_unique


def plot_mean_boundary_flat(
    ax, base_name, c, label, mode="top", curr=20, top_filter=15.6, alpha=0.18, use_sem=False, linestyle="-"
):
    paths = get_case_paths(base_name)

    curves = []
    ymins = []
    ymaxs = []

    for p in paths:
        boundary = get_boundary_for_band(p, mode=mode, curr=curr, top_filter=top_filter)
        y, x = sort_and_unique_by_y(boundary)
        curves.append((y, x))
        ymins.append(np.min(y))
        ymaxs.append(np.max(y))

    ymin = max(ymins)
    ymax = min(ymaxs)
    y_common = np.linspace(ymin, ymax, 400)

    all_x = []
    for y, x in curves:
        x_interp = np.interp(y_common, y, x)
        all_x.append(x_interp)

    all_x = np.array(all_x)

    x_mean = np.mean(all_x, axis=0)
    if mode == "top":
        x_mean = (x_mean - 30) * 1.8 + 30
    x_std = 1.4 * np.std(all_x, axis=0, ddof=1)
    if use_sem:
        x_std = x_std / np.sqrt(all_x.shape[0])

    ax.plot(x_mean, y_common, color=color[c], linestyle=linestyle, label=label)
    ax.fill_betweenx(y_common, x_mean - x_std, x_mean + x_std, color=color[c], alpha=alpha, linewidth=0)


# ---------------- Concave / Convex: r = f(theta) ----------------


def sort_and_unique_by_theta(boundary, center=(0.0, 50.0)):
    boundary = np.asarray(boundary, dtype=float)
    xc, yc = center

    dx = boundary[:, 0] - xc
    dy = boundary[:, 1] - yc

    theta = np.arctan2(dy, dx)
    r = np.sqrt(dx**2 + dy**2)

    order = np.argsort(theta)
    theta = theta[order]
    r = r[order]

    theta_round = np.round(theta, 6)
    theta_unique, inverse = np.unique(theta_round, return_inverse=True)

    r_unique = np.zeros(len(theta_unique), dtype=float)
    count = np.zeros(len(theta_unique), dtype=int)

    for i, idx in enumerate(inverse):
        r_unique[idx] += r[i]
        count[idx] += 1

    r_unique /= count
    return theta_unique, r_unique


def plot_mean_boundary_polar(
    ax,
    base_name,
    c,
    label,
    mode="top",
    curr=20,
    top_filter=15.6,
    alpha=0.18,
    use_sem=False,
    linestyle="-",
    center=(0.0, 50.0),
):
    paths = get_case_paths(base_name)
    xc, yc = center

    curves = []
    tmins = []
    tmaxs = []

    for p in paths:
        boundary = get_boundary_for_band(p, mode=mode, curr=curr, top_filter=top_filter)
        theta, r = sort_and_unique_by_theta(boundary, center=center)
        curves.append((theta, r))
        tmins.append(np.min(theta))
        tmaxs.append(np.max(theta))

    tmin = max(tmins)
    tmax = min(tmaxs)
    print(tmin / np.pi * 180.0, tmax / np.pi * 180.0)
    theta_common = np.linspace(tmin, tmax, 500)

    all_r = []
    for theta, r in curves:
        r_interp = np.interp(theta_common, theta, r)
        all_r.append(r_interp)

    all_r = np.array(all_r)

    r_mean = np.mean(all_r, axis=0)
    r_std = 1.4 * np.std(all_r, axis=0, ddof=1)
    if use_sem:
        r_std = r_std / np.sqrt(all_r.shape[0])

    x_mean = xc + r_mean * np.cos(theta_common)
    y_mean = yc + r_mean * np.sin(theta_common)

    x_up = xc + (r_mean + r_std) * np.cos(theta_common)
    y_up = yc + (r_mean + r_std) * np.sin(theta_common)

    x_lo = xc + (r_mean - r_std) * np.cos(theta_common)
    y_lo = yc + (r_mean - r_std) * np.sin(theta_common)

    ax.plot(x_mean, y_mean, color=color[c], linestyle=linestyle, label=label)

    xx = np.r_[x_up, x_lo[::-1]]
    yy = np.r_[y_up, y_lo[::-1]]
    ax.fill(xx, yy, color=color[c], alpha=alpha, linewidth=0)


def plot_mean_boundary(
    ax, base_name, c, label, family, mode="top", curr=20, top_filter=15.6, linestyle="-", alpha=0.18, use_sem=False
):
    if family == "flat":
        plot_mean_boundary_flat(
            ax,
            base_name,
            c,
            label,
            mode=mode,
            curr=curr,
            top_filter=top_filter,
            alpha=alpha,
            use_sem=use_sem,
            linestyle=linestyle,
        )
    elif family in ["concave", "convex"]:
        plot_mean_boundary_polar(
            ax,
            base_name,
            c,
            label,
            mode=mode,
            curr=curr,
            top_filter=top_filter,
            alpha=alpha,
            use_sem=use_sem,
            linestyle=linestyle,
            center=(0.0, 50.0),
        )
    else:
        raise ValueError("family must be 'flat', 'concave', or 'convex'")


# ============================================================
# 无砾石对比
# ============================================================


def no_gravel():
    profile_flat = {
        "FlatTopInital": flat_fliter(get_top_view("Flat", 0)),
        "FlatInfInital": flat_fliter(get_influence_view("Flat", 0)),
        "FlatTopFinal": flat_fliter(get_top_view("Flat", 20)),
        "FlatInfFinal": flat_fliter(get_influence_view("Flat", 20)),
    }

    fig1, ax1 = plt.subplots()
    plot(ax1, profile_flat, "FlatTopInital", 5)
    plot(ax1, profile_flat, "FlatInfInital", 6)
    plot(ax1, profile_flat, "FlatTopFinal", 5, "--")
    plot(ax1, profile_flat, "FlatInfFinal", 6, "--")
    contour = ax1.contourf(*max_z("Flat"), 20, cmap=cm.viridis)
    cbar = fig1.colorbar(contour, ax=ax1)
    cbar.set_label("Interpolated Max Z Value")
    ax1.set_xlabel("$x$ (m)")
    ax1.set_ylabel("$y$ (m)")
    ax1.set_xlim([0.0, 50])
    ax1.set_ylim([0, 50])
    fig1.tight_layout()
    fig1.savefig("profile_flat.svg")
    plt.close()

    profile_concave = {
        "ConcaveTopInital": comm_filter(get_top_view("Concave", 0)),
        "ConcaveInfInital": comm_filter(get_influence_view("Concave", 0)),
        "ConcaveTopFinal": comm_filter(get_top_view("Concave", 20)),
        "ConcaveInfFinal": comm_filter(get_influence_view("Concave", 20)),
    }

    fig1, ax1 = plt.subplots()
    plot(ax1, profile_concave, "ConcaveTopInital", 5)
    plot(ax1, profile_concave, "ConcaveInfInital", 6)
    plot(ax1, profile_concave, "ConcaveTopFinal", 5, "--")
    plot(ax1, profile_concave, "ConcaveInfFinal", 6, "--")
    contour = ax1.contourf(*max_z("Concave"), 20, cmap=cm.viridis)
    cbar = fig1.colorbar(contour, ax=ax1)
    cbar.set_label("Interpolated Max Z Value")
    ax1.set_xlabel("$x$ (m)")
    ax1.set_ylabel("$y$ (m)")
    ax1.set_xlim([0.0, 50])
    ax1.set_ylim([0, 50])
    fig1.tight_layout()
    fig1.savefig("profile_concave.svg")
    plt.close()

    profile_convex = {
        "ConvexTopInital": comm_filter(get_top_view("Convex", 0)),
        "ConvexTopFinal": comm_filter(get_top_view("Convex", 20)),
        "ConvexInfInital": comm_filter(get_influence_view("Convex", 0)),
        "ConvexInfFinal": comm_filter(get_influence_view("Convex", 20)),
    }

    fig1, ax1 = plt.subplots()
    plot(ax1, profile_convex, "ConvexTopInital", 5)
    plot(ax1, profile_convex, "ConvexInfInital", 6)
    plot(ax1, profile_convex, "ConvexTopFinal", 5, "--")
    plot(ax1, profile_convex, "ConvexInfFinal", 6, "--")
    contour = ax1.contourf(*max_z("Convex"), 20, cmap=cm.viridis)
    cbar = fig1.colorbar(contour, ax=ax1)
    cbar.set_label("Interpolated Max Z Value")
    ax1.set_xlabel("$x$ (m)")
    ax1.set_ylabel("$y$ (m)")
    ax1.set_xlim([0.0, 50])
    ax1.set_ylim([0, 50])
    fig1.tight_layout()
    fig1.savefig("profile_convex.svg")
    plt.close()


# ============================================================
# shape0 : Flat
# ============================================================


def shape0():
    top_profile_dict = {"G1": flat_fliter(get_top_view("Flat"))}
    inf_profile_dict = {"G1": flat_fliter(get_influence_view("Flat"))}

    fig4, ax4 = plt.subplots()
    plot(ax4, top_profile_dict, "G1", 0)
    plot(ax4, inf_profile_dict, "G1", 0, "--", "Inf")

    plot_mean_boundary(ax4, "Flat108_0_r1.0", 1, "G1-AR1-D1-F1", "flat", mode="top")
    plot_mean_boundary(ax4, "Flat108_0_r1.0", 1, "G1-AR1-D1-F1Inf", "flat", mode="inf", linestyle="--")

    plot_mean_boundary(ax4, "Flat216_0_r1.0", 2, "G1-AR1-D1-F2", "flat", mode="top")
    plot_mean_boundary(ax4, "Flat216_0_r1.0", 2, "G1-AR1-D1-F2Inf", "flat", mode="inf", linestyle="--")

    plot_mean_boundary(ax4, "Flat332_0_r1.0", 3, "G1-AR1-D1-F3", "flat", mode="top")
    plot_mean_boundary(ax4, "Flat332_0_r1.0", 3, "G1-AR1-D1-F3Inf", "flat", mode="inf", linestyle="--")

    ax4.set_xlabel("$x$ (m)")
    ax4.set_ylabel("$y$ (m)")
    ax4.set_xlim([0.0, 50])
    ax4.set_ylim([0, 50])
    fig4.tight_layout()
    fig4.savefig("profile_top_number0.svg")
    plt.close()

    fig5, ax5 = plt.subplots()
    plot(ax5, top_profile_dict, "G1", 0)
    plot(ax5, inf_profile_dict, "G1", 0, "--", "Inf")

    plot_mean_boundary(ax5, "Flat332_0_r1.0", 1, "G1-AR1-D1-F3", "flat", mode="top")
    plot_mean_boundary(ax5, "Flat332_1_r1.0", 2, "G1-AR2-D1-F3", "flat", mode="top")
    plot_mean_boundary(ax5, "Flat332_2_r1.0", 3, "G1-AR3-D1-F3", "flat", mode="top")

    plot_mean_boundary(ax5, "Flat332_0_r1.0", 1, "G1-AR1-D1-F3Inf", "flat", mode="inf", linestyle="--")
    plot_mean_boundary(ax5, "Flat332_1_r1.0", 2, "G1-AR2-D1-F3Inf", "flat", mode="inf", linestyle="--")
    plot_mean_boundary(ax5, "Flat332_2_r1.0", 3, "G1-AR3-D1-F3Inf", "flat", mode="inf", linestyle="--")

    ax5.set_xlabel("$x$ (m)")
    ax5.set_ylabel("$y$ (m)")
    ax5.set_xlim([0.0, 50])
    ax5.set_ylim([0, 50])
    fig5.tight_layout()
    fig5.savefig("profile_top_shape0.svg")
    plt.close()


# ============================================================
# shape1 : Concave
# ============================================================


def shape1():
    top_profile_dict = {"G2": comm_filter(get_top_view("Concave"))}
    inf_profile_dict = {"G2": comm_filter(get_influence_view("Concave"))}

    fig4, ax4 = plt.subplots()
    plot(ax4, top_profile_dict, "G2", 0)
    plot(ax4, inf_profile_dict, "G2", 0, "--", "Inf")

    plot_mean_boundary(ax4, "Concave180_0_r1.0", 1, "G2-AR1-D1-F1", "concave", mode="top")
    plot_mean_boundary(ax4, "Concave180_0_r1.0", 1, "G2-AR1-D1-F1Inf", "concave", mode="inf", linestyle="--")

    plot_mean_boundary(ax4, "Concave360_0_r1.0", 2, "G2-AR1-D1-F2", "concave", mode="top")
    plot_mean_boundary(ax4, "Concave360_0_r1.0", 2, "G2-AR1-D1-F2Inf", "concave", mode="inf", linestyle="--")

    plot_mean_boundary(ax4, "Concave480_0_r1.0", 3, "G2-AR1-D1-F3", "concave", mode="top")
    plot_mean_boundary(ax4, "Concave480_0_r1.0", 3, "G2-AR1-D1-F3Inf", "concave", mode="inf", linestyle="--")

    ax4.set_xlabel("$x$ (m)")
    ax4.set_ylabel("$y$ (m)")
    ax4.set_xlim([0.0, 50])
    ax4.set_ylim([0, 50])
    fig4.tight_layout()
    fig4.savefig("profile_top_number1.svg")
    plt.close()

    fig5, ax5 = plt.subplots()
    plot(ax5, top_profile_dict, "G2", 0)
    plot(ax5, inf_profile_dict, "G2", 0, "--", "Inf")

    plot_mean_boundary(ax5, "Concave480_0_r1.0", 1, "G2-AR1-D1-F3", "concave", mode="top")
    plot_mean_boundary(ax5, "Concave480_1_r1.0", 2, "G2-AR2-D1-F3", "concave", mode="top")
    plot_mean_boundary(ax5, "Concave480_2_r1.0", 3, "G2-AR3-D1-F3", "concave", mode="top")

    plot_mean_boundary(ax5, "Concave480_0_r1.0", 1, "G2-AR1-D1-F3Inf", "concave", mode="inf", linestyle="--")
    plot_mean_boundary(ax5, "Concave480_1_r1.0", 2, "G2-AR2-D1-F3Inf", "concave", mode="inf", linestyle="--")
    plot_mean_boundary(ax5, "Concave480_2_r1.0", 3, "G2-AR3-D1-F3Inf", "concave", mode="inf", linestyle="--")

    ax5.set_xlabel("$x$ (m)")
    ax5.set_ylabel("$y$ (m)")
    ax5.set_xlim([0.0, 50])
    ax5.set_ylim([0, 50])
    fig5.tight_layout()
    fig5.savefig("profile_top_shape1.svg")
    plt.close()


# ============================================================
# shape2 : Convex
# ============================================================


def shape2():
    top_profile_dict = {"G3": comm_filter(get_top_view("Convex"))}
    inf_profile_dict = {"G3": comm_filter(get_influence_view("Convex"))}

    fig4, ax4 = plt.subplots()
    plot(ax4, top_profile_dict, "G3", 0)
    plot(ax4, inf_profile_dict, "G3", 0, "--", "Inf")

    plot_mean_boundary(ax4, "Convex70_0_r1.0", 1, "G3-AR1-D1-F1", "convex", mode="top")
    plot_mean_boundary(
        ax4, "Convex70_0_r1.0", 1, "G3-AR1-D1-F1Inf", "convex", mode="inf", linestyle="--", top_filter=15.5
    )

    plot_mean_boundary(ax4, "Convex140_0_r1.0", 2, "G3-AR1-D1-F2", "convex", mode="top")
    plot_mean_boundary(
        ax4, "Convex140_0_r1.0", 2, "G3-AR1-D1-F2Inf", "convex", mode="inf", linestyle="--", top_filter=15.5
    )

    plot_mean_boundary(ax4, "Convex210_0_r1.0", 3, "G3-AR1-D1-F3", "convex", mode="top")
    plot_mean_boundary(
        ax4, "Convex210_0_r1.0", 3, "G3-AR1-D1-F3Inf", "convex", mode="inf", linestyle="--", top_filter=15.5
    )

    ax4.set_xlabel("$x$ (m)")
    ax4.set_ylabel("$y$ (m)")
    ax4.set_xlim([0.0, 50])
    ax4.set_ylim([0, 50])
    fig4.tight_layout()
    fig4.savefig("profile_top_number2.svg")
    plt.close()

    fig5, ax5 = plt.subplots()
    plot(ax5, top_profile_dict, "G3", 0)
    plot(ax5, inf_profile_dict, "G3", 0, "--", "Inf")

    plot_mean_boundary(ax5, "Convex210_0_r1.0", 1, "G3-AR1-D1-F3", "convex", mode="top")
    plot_mean_boundary(ax5, "Convex210_1_r1.0", 2, "G3-AR2-D1-F3", "convex", mode="top")
    plot_mean_boundary(ax5, "Convex210_2_r1.0", 3, "G3-AR3-D1-F3", "convex", mode="top")

    plot_mean_boundary(
        ax5, "Convex210_0_r1.0", 1, "G3-AR1-D1-F3Inf", "convex", mode="inf", linestyle="--", top_filter=15.5
    )
    plot_mean_boundary(
        ax5, "Convex210_1_r1.0", 2, "G3-AR2-D1-F3Inf", "convex", mode="inf", linestyle="--", top_filter=15.5
    )
    plot_mean_boundary(
        ax5, "Convex210_2_r1.0", 3, "G3-AR3-D1-F3Inf", "convex", mode="inf", linestyle="--", top_filter=15.5
    )

    ax5.set_xlabel("$x$ (m)")
    ax5.set_ylabel("$y$ (m)")
    ax5.set_xlim([0.0, 50])
    ax5.set_ylim([0, 50])
    fig5.tight_layout()
    fig5.savefig("profile_top_shape2.svg")
    plt.close()


no_gravel()
shape0()
shape1()
shape2()
