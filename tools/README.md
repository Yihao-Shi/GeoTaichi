# Developer tools

Gallery GIFs use saved simulation frames. `blender_cfdem_gif.py`,
`incompressible_gifs.py`, `mmpm_gifs.py` and `two_phase_3d_gifs.py` render Blender
water surfaces with a single color; geometry comes from saved NPZs.
The ParaView pressure plots in `incompressible_gifs.py` keep their pressure coloring.
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

Do not add a script here merely to avoid writing an assertion. If a small,
deterministic oracle exists, put it under `tests/unit`; keep the tool only when
interactive output, symbolic code generation, or performance instrumentation
is itself useful.
