"""Render a saved TSC9d snapshot using the research stress-smoothing settings.

python tools/stone_column_render.py --output images/stone_column_9d.png
"""

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research/StoneColumn/simulation/AxialLoading/TSC9d/cohesion/StoneColumnLoadingCoh20kPa"


def render(source, output):
    os.environ["GEOTAICHI_FORCE_CPU"] = "1"
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    sys.path.insert(0, str(ROOT))
    import geotaichi as gt

    gt.init(dim=3, arch="cpu", default_fp="float64", log=False)
    from src.mpm.PostPlot import SmoothVar

    # Both inputs are the final saved state (frame 56); do not pair it with DEM frame 55.
    with np.load(source / "particles/MPMParticle000056.npz") as data:
        positions = data["position"]
        raw_stress = data["stress"][:, 2]
        mass = data["mass"]
        sizes = data["psize"]
        active = data["active"].astype(bool)
        time = float(data["t_current"])
        assert np.isfinite(positions).all() and np.isfinite(raw_stress).all()
        assert np.isfinite(mass).all() and (mass > 0).all()
        assert np.isfinite(sizes[active]).all() and (sizes[active] > 0).all()
        # Same mass-weighted quadratic P2G/G2P as AxialLoading/postprocessing.py.
        smoother = SmoothVar(
            {"smooth_rad": 0.016, "lower_bound": [0.0, 0.0, 0.0], "upper_bound": [0.8, 0.8, 0.55]}, len(positions)
        )
        stress = smoother.run(positions, mass, raw_stress) / 1000
        assert np.isfinite(stress).all()
        assert stress.min() >= raw_stress.min() / 1000 - 1e-8
        assert stress.max() <= raw_stress.max() / 1000 + 1e-8
        # Display the central soil window; retain native particle positions and sizes.
        visible = active & (np.linalg.norm(positions[:, :2] - 0.4, axis=1) <= 0.12)
        visible &= positions[:, 1] >= 0.4 - sizes[:, 1]
        assert visible.any()
        soil = pv.PolyData(positions[visible])
        soil["sigma_zz"] = stress[visible]
        half_size = np.median(sizes[visible], axis=0)
        assert np.allclose(sizes[visible], half_size)
        boxes = pv.Cube(x_length=2 * half_size[0], y_length=2 * half_size[1], z_length=2 * half_size[2])
        soil = soil.glyph(geom=boxes, scale=False, orient=False)
        soil = soil.clip(normal=(0, 1, 0), origin=(0.4, 0.4, 0), invert=False)
        limits = [float(stress[active].min()), float(stress[active].max())]

    grains = pv.read(source / "vtks/GraphicDEMParticle000056.vtu")
    assert grains.n_points > 0 and np.isfinite(grains.points).all()
    assert np.isfinite(grains["radius"]).all() and (grains["radius"] > 0).all()
    grains = grains.glyph(
        geom=pv.Sphere(radius=1, theta_resolution=16, phi_resolution=12), scale="radius", orient=False
    )

    plot = pv.Plotter(off_screen=True, window_size=(960, 1100))
    plot.set_background("white")
    plot.add_mesh(
        soil,
        scalars="sigma_zz",
        cmap="turbo_r",
        clim=limits,
        show_edges=False,
        ambient=0.35,
        diffuse=0.65,
        scalar_bar_args={
            "title": "Smoothed stress zz (kPa)",
            "vertical": True,
            "position_x": 0.82,
            "position_y": 0.22,
            "height": 0.52,
            "width": 0.07,
            "title_font_size": 20,
            "label_font_size": 18,
            "fmt": "%.0f",
            "color": "#263140",
        },
    )
    plot.add_mesh(grains, color="#8c9095", smooth_shading=True, ambient=0.25, diffuse=0.7, specular=0.25)
    plot.add_text("Stone column | H/D = 9", position=(40, 1040), font_size=22, color="#263140")
    plot.add_text("c = 20 kPa   E = 4 MPa", position=(40, 998), font_size=17, color="#566273")
    plot.add_text(f"t = {time:.1f} s | settlement = 50 mm", position=(40, 954), font_size=16, color="#566273")
    plot.add_text(
        "MPM: half cut, smoothing h = 0.016 m\nDEM: complete column, grey\nCompression is negative",
        position=(40, 35),
        font_size=15,
        color="#566273",
    )
    plot.camera_position = [(0.92, -1.4, 0.95), (0.4, 0.43, 0.27), (0, 0, 1)]
    plot.enable_parallel_projection()
    plot.camera.parallel_scale = 0.40
    plot.camera.SetWindowCenter(0.12, 0)
    plot.enable_anti_aliasing("ssaa")
    output.parent.mkdir(parents=True, exist_ok=True)
    plot.screenshot(output)
    plot.close()
    print(f"Rendered {output}: t={time:.6f} s, smoothed stress zz range={limits} kPa")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "images/stone_column_9d.png")
    args = parser.parse_args()
    render(args.source, args.output)
