# Developer tools

`fem_iga_gifs.py` renders the README's implicit elastic bar and horizontal 450° twisting
bar in ParaView with fixed, full-range von Mises stress (Pa), and twisting FEM
cloth in Blender with one material. Geometry keeps its native deformation.
The twisting bar is sampled more densely from its saved quadratic NURBS
control points and stress, then rotated 90° for display. Cloth triangles and
their connectivity are preserved; its saved validation summary must report
completion, finite positions, and active self-contact.

```bash
python examples/iga/explicit_cantilever_3d/explicit_cantilever_3d.py --solver implicit --arch cpu --dt 0.005 --steps 100 --output-interval 2 --output-dir /tmp/iga_implicit_elastic_bar
python tools/fem_iga_gifs.py iga_elastic_bar /tmp/iga_implicit_elastic_bar --work-dir /tmp/iga_implicit_elastic_bar_render
python tools/fem_iga_gifs.py iga_twisting examples/iga/twisting_column_3d/twisting_column_3d_implicit --work-dir /tmp/iga_twisting_render
python tools/fem_iga_gifs.py cloth_twisting examples/fem/cloth/implicit_cloth_twist/OutputData/cloth_twist_corrected_grid50_full_20260916 --work-dir /tmp/fem_cloth_twisting_render
```

The elastic bar's `--steps` counts output batches: 100 × 2 steps × 0.005 s
produces 101 frames over 1 s. The shared cantilever example retains explicit
as its default; pass `--solver implicit` for this gallery. It uses implicit
Newmark with HashTriplet/PCG and a correction-velocity tolerance of 1e-6 m/s.
The `--damping` option applies only to the explicit solver. Elastic-bar rendering
requires its completed implicit `summary.json`; the frame spacing comes from
that record. Twisting-bar and cloth times also come from saved run records.
Add `--preview` to render the final frame without replacing a GIF.
Outputs are `images/iga_elastic_bar.gif`, `images/iga_twisting.gif`, and
`images/fem_cloth_twisting.gif`.

Gallery GIFs use saved simulation frames. `blender_cfdem_gif.py`,
`incompressible_gifs.py`, `mmpm_gifs.py` and `two_phase_3d_gifs.py` render Blender
water surfaces with a single color; geometry comes from saved NPZs.
The shared water shader uses area/angle-weighted shading normals and softer
specular reflections without moving surface vertices. Resolved waves and
reconstruction ripples remain visible. IBM dam break, bunny impact and the
LSDEM wavemaker use 960 by 720 canvases with orthographic framing and uniform
scene scaling; their different physical aspect ratios are preserved.
The porous-column dam-break gallery uses the first 100 saved frames.
Granular-water leakage reconstructs water surfaces in the upper container,
outlet stream and receiver from saved fluid positions and bulk volumes.
Missing triangle/quad surface facets are repaired using existing vertices;
closed air pockets and surfaces touching along an edge are retained.
`semi_resolved_gifs.py --source RESULT_DIRECTORY` renders sphere impact into
water to `images/sphere_impact_water.gif`. It pairs the saved MPM/grid frames
with `DEMParticle` positions and radii and checks their timestamps.
For the LSDEM-coupled sphere, bunny and wavemaker cases, MPM soil is displayed
as a continuous surface. The blue wet-bulk envelope takes the maximum of
fluid support and soil support where saved fluid points lie within 1.5 native
grid cells; overlapping bulk regions are not summed. Water mass comes from
the saved fluid masses, rather than the displayed envelope volume. Exact wall
planes and domain caps close the displayed surfaces while retaining resolved
air channels. A 0.05-cell inward display offset separates soil and water faces
at walls; LSDEM bodies use their saved rigid meshes.
The ParaView cavity pressure plot in `incompressible_gifs.py` uses a fixed
[-0.05, 0.05] color range. Taylor–Green uses velocity magnitude with a color
range from zero to each frame's maximum speed; the physical speed values remain
visible on the changing color bar as the flow decays.
The Taylor–Green camera reads the physical domain from the saved `metrics.json`,
so historical one-vortex runs and full four-vortex runs are framed correctly.
`wavy_plate_gif.py` colors IGA von Mises stress (Pa) and MPM displacement from
frame zero (m), and also renders the 2D flexible barrier and axisymmetric CPT:
The CPT probe uses its saved displacement (m), since recovered frames lack stress.

```sh
pvpython --force-offscreen-rendering tools/wavy_plate_gif.py examples/igampm/flexible_barrier/OutputData/ipc_2d_quadbspline_20261011 --output images/flexible_barrier.gif
pvpython --force-offscreen-rendering tools/wavy_plate_gif.py examples/igampm/cpt_dp/OutputData/ipc_formal_20261006 --output images/cpt_dp.gif
```

`mpm_cloth_gif.py` renders saved BarrierIPC cloth–MPM contact with Blender,
using single-color cloth and MPM markers at their saved positions:

```sh
python tools/mpm_cloth_gif.py --work-dir /tmp/mpm_cloth_gallery
```

The default source is `fem_mpm_cloth_dp_spread_t060_gridpad_v2_20260917` and
the GIF is saved to `images/mpm_cloth_ipc.gif`. Marker size comes from initial
particle spacing; VTU timestamps are inferred from the uniform save sequence
and the completed duration in `validation_summary.json`.

This directory contains explicit developer workflows that are useful but are
not correctness tests and are never run during pytest collection.

- `diagnostics/`: targeted numerical probes and viewers. A diagnostic must
  expose a callable entry point, use deterministic inputs where possible, and
  perform work only under `if __name__ == "__main__"`.
- `derivations/`: symbolic/reference derivations used to check or generate
  formulas. Install the optional dependencies with
  `pip install -e '.[gen]'`.
- `benchmarks/`: standalone timing and memory studies. Correctness belongs in
  `tests/`; benchmark output belongs in a caller-selected directory outside
  the repository.
  `benchmarks/igampm/benchmark_wavy_contact.py` compares span candidate screening
  and CCD motion-bound screening on a saved wavy-plate checkpoint. It alternates
  warmed baseline/optimized kernels, checks active-distance and motion-screen
  equivalence, and reports kernel timings rather than whole-step speedups.
  `benchmarks/fempm/benchmark_fixed_assembly.py` compares raw reduction with
  permanent-slot FEM material assembly, including matrix equivalence, alternating
  warmed samples, and explicit CPU/CUDA selection.
  `benchmarks/igampm/benchmark_fixed_assembly.py` times cached fixed-slot IGA
  solid assembly and checks it against the raw matrix; run it in separate source
  snapshots for a kernel comparison.
  `benchmarks/linear_solver/benchmark_pcg_fusion.py` compares split and fused
  HashTriplet PCG launches on identical SPD block systems, with independent true
  residual checks, alternating samples, and CPU/CUDA/Metal selection.
- `postprocessing/`: parameterized command-line transforms and data summaries
  for existing simulation output; input and output paths must be explicit.

`postprocessing/mpdem/export_twophase_vtu.py RESULT_DIRECTORY` exports saved
2D/3D two-point MPM snapshots to `RESULT_DIRECTORY/vtks`: separate fluid/solid
particle VTUs, pressure-grid VTUs, and LSDEM surface VTUs when present.
It checks frame continuity and synchronized timestamps, preserves the saved
pressure/positions/phase velocities, and refuses to overwrite existing VTKs.
Its export report does not replace the simulation's physical acceptance metrics.

Do not add a script here merely to avoid writing an assertion. If a small,
deterministic oracle exists, put it under `tests/unit`; keep the tool only when
interactive output, symbolic code generation, or performance instrumentation
is itself useful.
