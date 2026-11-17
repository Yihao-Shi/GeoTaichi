import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from geotaichi import *

# =========================
# Size Parameters
# =========================
CL = 12.7 * 1e-3
L = CL * 7
H = L * 5
Angle = 120
stl_path = os.path.join(ROOT, "assets", "mesh", "LSDEM", f"staple_a{Angle}.stl")
save_path = os.path.join("Projects", "Staple-likeEntangledParticles", "output", f"Ratio{H/L}", f"Angle{Angle}", "Gen")

BoxRegion_point = ti.Vector([CL * 1, CL * 1, CL * 1])
BoxRegion = ti.Vector([L, L, (H + CL * 5)])
GenerateRegion_point = ti.Vector([CL * 1, CL * 1, (H + CL * 5)])
GenerateRegion_point = ti.Vector([CL * 1, CL * 1, CL])
GenerateRegion_size = ti.Vector([L, L, CL * 5])
Domain = BoxRegion_point * 2 + BoxRegion + ti.Vector([0, 0, CL * 5])
timestep = 2e-6
interval = round(0.15 / timestep)

# ---- wall mesh density: target edge length for nearly-equilateral triangles ----
WALL_MESH_SIZE = CL  # ≈ 6.35 mm; smaller = denser; produces isosceles right triangles (45°-45°-90°)


def _vec_len(v):
    return np.sqrt(sum(x * x for x in v))


def _vec_sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def _vec_lerp(a, b, t):
    return [a[i] + (b[i] - a[i]) * t for i in range(3)]


def _triangle_normal(p0, p1, p2):
    normal = np.cross(np.asarray(p1) - np.asarray(p0), np.asarray(p2) - np.asarray(p0))
    norm = np.linalg.norm(normal)
    return normal / norm if norm > 0.0 else normal


def _write_ascii_stl(path, triangles, name="box_wall_patch"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"solid {name}\n")
        for tri in triangles:
            normal = _triangle_normal(tri[0], tri[1], tri[2])
            f.write(f"  facet normal {normal[0]:.16e} {normal[1]:.16e} {normal[2]:.16e}\n")
            f.write("    outer loop\n")
            for vertex in tri:
                f.write(f"      vertex {vertex[0]:.16e} {vertex[1]:.16e} {vertex[2]:.16e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {name}\n")


def _tri_min_angle(p0, p1, p2):
    """Return the minimum interior angle (degrees)."""
    a = _vec_len(_vec_sub(p1, p0))
    b = _vec_len(_vec_sub(p2, p1))
    c = _vec_len(_vec_sub(p0, p2))
    sides = sorted([a, b, c])
    if sides[2] < 1e-30:
        return 0.0
    cos_min = (sides[1] ** 2 + sides[2] ** 2 - sides[0] ** 2) / (2.0 * sides[1] * sides[2])
    cos_min = max(-1.0, min(1.0, cos_min))
    return float(np.degrees(np.arccos(cos_min)))


def _tri_quality(p0, p1, p2):
    """Regularity: 1.0 = equilateral, → 0 = degenerate.  2 * r_inscribed / r_circum."""
    a = _vec_len(_vec_sub(p1, p0))
    b = _vec_len(_vec_sub(p2, p1))
    c = _vec_len(_vec_sub(p0, p2))
    s = (a + b + c) / 2.0
    if s < 1e-30:
        return 0.0
    area2 = max(0.0, s * (s - a) * (s - b) * (s - c))
    area = np.sqrt(area2)
    r_in = area / s
    r_circ = (a * b * c) / (4.0 * area) if area > 1e-30 else 1e30
    return min(1.0, 2.0 * r_in / r_circ)


def _print_mesh_quality(triangles, label="mesh"):
    if not triangles:
        return
    q = [_tri_quality(*t) for t in triangles]
    a = [_tri_min_angle(*t) for t in triangles]
    print(
        f"  [{label}] {len(triangles)} triangles | "
        f"quality: min={min(q):.3f} mean={np.mean(q):.3f} | "
        f"min angle: {np.min(a):.1f}° mean={np.mean(a):.1f}°"
    )


def _subdivide_quad_regular(p0, p1, p2, p3, target_edge):
    """Subdivide a quadrilateral into nearly-equilateral triangles.

    Each grid cell is as square as possible and split into two isosceles right
    triangles (45°-45°-90°).  Diagonal direction alternates for isotropy.

    p0 -- p1  (u direction, length = w)
    |      |
    p3 -- p2  (v direction, length = h)
    """
    u_vec = _vec_sub(p1, p0)
    v_vec = _vec_sub(p3, p0)
    w = _vec_len(u_vec)
    h = _vec_len(v_vec)

    n_u = max(1, int(np.ceil(w / target_edge)))
    cell_w = w / n_u
    n_v = max(1, int(np.ceil(h / cell_w)))

    def _pt(u, v):
        return _vec_lerp(_vec_lerp(p0, p1, u), _vec_lerp(p3, p2, u), v)

    triangles = []
    for i in range(n_u):
        u0 = i / n_u
        u1 = (i + 1) / n_u
        for j in range(n_v):
            v0 = j / n_v
            v1 = (j + 1) / n_v
            a, b, c, d = _pt(u0, v0), _pt(u1, v0), _pt(u1, v1), _pt(u0, v1)
            if (i + j) % 2 == 0:
                triangles.append((a, b, d))
                triangles.append((b, c, d))
            else:
                triangles.append((a, b, c))
                triangles.append((a, c, d))

    return triangles


def create_box_wall_patch_stl(
    output_file, box_point=BoxRegion_point, box_size=BoxRegion, expand=1.05, target_edge=WALL_MESH_SIZE
):
    x0, y0, z0 = [float(v) for v in box_point]
    dx, dy, dz = [float(v) for v in box_size]
    ex = expand - 1.0

    quads = [
        # bottom, normal +z
        [
            [x0 - dx * ex, y0 - dy * ex, z0],
            [x0 + dx * (1 + ex), y0 - dy * ex, z0],
            [x0 + dx * (1 + ex), y0 + dy * (1 + ex), z0],
            [x0 - dx * ex, y0 + dy * (1 + ex), z0],
        ],
        # x-min wall, normal +x
        [
            [x0, y0 - dy * ex, z0 - dz * ex],
            [x0, y0 + dy * (1 + ex), z0 - dz * ex],
            [x0, y0 + dy * (1 + ex), z0 + dz * (1 + ex)],
            [x0, y0 - dy * ex, z0 + dz * (1 + ex)],
        ],
        # x-max wall, normal -x
        [
            [x0 + dx, y0 - dy * ex, z0 - dz * ex],
            [x0 + dx, y0 - dy * ex, z0 + dz * (1 + ex)],
            [x0 + dx, y0 + dy * (1 + ex), z0 + dz * (1 + ex)],
            [x0 + dx, y0 + dy * (1 + ex), z0 - dz * ex],
        ],
        # y-min wall, normal +y
        [
            [x0 - dx * ex, y0, z0 - dz * ex],
            [x0 - dx * ex, y0, z0 + dz * (1 + ex)],
            [x0 + dx * (1 + ex), y0, z0 + dz * (1 + ex)],
            [x0 + dx * (1 + ex), y0, z0 - dz * ex],
        ],
        # y-max wall, normal -y
        [
            [x0 - dx * ex, y0 + dy, z0 - dz * ex],
            [x0 + dx * (1 + ex), y0 + dy, z0 - dz * ex],
            [x0 + dx * (1 + ex), y0 + dy, z0 + dz * (1 + ex)],
            [x0 - dx * ex, y0 + dy, z0 + dz * (1 + ex)],
        ],
    ]

    print(f"Wall mesh: target edge = {target_edge * 1e3:.2f} mm")
    triangles = []
    for i, quad in enumerate(quads):
        face_tris = _subdivide_quad_regular(*quad, target_edge)
        _print_mesh_quality(face_tris, label=f"face {i}")
        triangles.extend(face_tris)

    print(f"  total: {len(triangles)} triangles")
    _write_ascii_stl(output_file, triangles)


def box_wall(
    box_point=BoxRegion_point,
    box_size=BoxRegion,
    expand=1.05,
):
    wall_file = os.path.join(save_path, "box_wall_patch.stl")
    create_box_wall_patch_stl(
        wall_file, box_point=box_point, box_size=box_size, expand=expand, target_edge=WALL_MESH_SIZE
    )
    lsdem.add_wall(
        body={
            "WallID": 0,
            "WallType": "Patch",
            "WallFile": wall_file,
            "MaterialID": 1,
            "InverseNorm": False,
        }
    )


init(arch="gpu", log=True, debug=True, device_memory_GB=26)

lsdem = DEM()

lsdem.set_configuration(
    domain=Domain,
    boundary=["Destroy", "Destroy", "Destroy"],
    gravity=ti.Vector([0.0, 0.0, -9.8]),
    engine="VelocityVerlet",
    search="BVH",
    # search="LinkedCell",
    scheme="LSDEM",
    visualize=True,
)

lsdem.set_solver({"Timestep": timestep, "CFL": 1.0, "SimulationTime": 60, "SaveInterval": 0.1, "SavePath": save_path})

lsdem.memory_allocate(
    memory={
        "max_material_number": 2,
        "max_rigid_body_number": 30000,
        "max_rigid_template_number": 1,
        "levelset_grid_number": 240465,
        "surface_node_number": 808,
        "max_patch_number": 3040,
        "body_coordination_number": 240,
        "wall_coordination_number": 8,
        "wall_per_cell": 32,
        "verlet_distance_multiplier": [0.10, 0.10],
        "point_coordination_number": [5, 4],
        "compaction_ratio": [0.50, 0.15, 0.08, 0.05],
    }
)

lsdem.add_attribute(materialID=0, attribute={"Density": 7850, "ForceLocalDamping": 0.5, "TorqueLocalDamping": 0.5})

lsdem.add_template(
    template={
        "Name": "Staple",
        "WriteFile": True,
        "Object": polyhedron(file=stl_path).grids(space=1e-1, extent=3),
        "WriteFile": True,
    }
)

lsdem.choose_contact_model(particle_particle_contact_model="Linear Model", particle_wall_contact_model="Linear Model")

lsdem.add_property(
    materialID1=0,
    materialID2=0,
    property={
        "NormalStiffness": 5e6,
        "TangentialStiffness": 5e6,
        "Friction": 0.4,
        "NormalViscousDamping": 0.0,
        "TangentialViscousDamping": 0.0,
    },
)

lsdem.add_property(
    materialID1=0,
    materialID2=1,
    property={
        "NormalStiffness": 1e7,
        "TangentialStiffness": 1e7,
        "Friction": 0.0,
        "NormalViscousDamping": 0.0,
        "TangentialViscousDamping": 0.0,
    },
    dType="particle-wall",
)

lsdem.select_save_data(
    particle=True, surface=True, bounding=True, wall=True, particle_particle_contact=True, particle_wall_contact=False
)

lsdem.add_region(
    region={
        "Name": "GenerateRegion",
        "Type": "Rectangle",
        "BoundingBoxPoint": GenerateRegion_point,
        "BoundingBoxSize": GenerateRegion_size,
    }
)
box_wall()
lsdem.add_body(
    body={
        "BodyType": "RigidBody",
        "GenerateType": "Generate",
        "RegionName": "GenerateRegion",
        "TryNumber": 10000,
        "Template": [
            {
                "Name": "Staple",
                # Source STL coordinates are in mm; scale to metres for LSDEM.
                "ScaleFactor": 1e-3,
                "BodyNumber": 100,
                "GroupID": 0,
                "MaterialID": 0,
                "InitialVelocity": ti.Vector([0.0, 0.0, -0.10]),
                "InitialAngularVelocity": ti.Vector([0.0, 0.0, 0.0]),
                "BodyOrientation": "uniform",
            }
        ],
    }
)


def add_particles():
    current_step = lsdem.sims.current_step
    if current_step > 1 and current_step % interval == 0:
        lsdem.add_body(
            body={
                "BodyType": "RigidBody",
                "GenerateType": "Generate",
                "RegionName": "GenerateRegion",
                "TryNumber": 20000,
                "Template": [
                    {
                        "Name": "Staple",
                        # Source STL coordinates are in mm; scale to metres for LSDEM.
                        "ScaleFactor": 1e-3,
                        "BodyNumber": 100,
                        "GroupID": current_step // interval,
                        "MaterialID": 0,
                        "InitialVelocity": ti.Vector([0.0, 0.0, -0.10]),
                        "InitialAngularVelocity": ti.Vector([0.0, 0.0, 0.0]),
                        "BodyOrientation": "uniform",
                    }
                ],
            }
        )


lsdem.run(callback_funcs=[add_particles], callback_is_kernel=[False])
