# Stone column axial loading (H/D = 9)

Run both scripts from the repository root, using an environment with GeoTaichi's
dependencies. Each output directory must be empty; previous results are preserved.

```sh
python examples/mpdem/MultiSphere/StoneColumn/prepare.py
python examples/mpdem/MultiSphere/StoneColumn/stone_column.py
```

`prepare.py` generates MPM soil around a cylindrical hole first. It fixes the
MPM velocities and selects DEM-only integration, then repeatedly generates
1200 DEM spheres, lets them fall, trims excess grains and compresses each layer
with a force-controlled top wall. There is no temporary cylindrical DEM wall.
The wall targets 1 kPa over the column area, with a 0.01 m/s speed limit and
10 s compaction per layer. An underfilled column after six batches is an error.
The final snapshot and `preparation.json` provide the input for loading.

`stone_column.py` restores both solvers and their contact histories, frees the
MPM velocities, applies the soil constraints and 1 kPa surface surcharge,
equilibrates DEM for 0.5 s, then switches to MPDEM for 0.1 s with the initial
gravity stress field. The top wall then moves down at 0.001 m/s for 50 s
(50 mm settlement). MPM, DEM, walls and contact records are saved together.
The loading summary checks the imposed wall settlement and finite state.

| Model choice | Value / source |
| --- | --- |
| Units and frame | m, kg, s, Pa; global Cartesian, z upward |
| Geometry | 0.8 × 0.8 m soil bed, 0.54 m high; column D = 0.06 m, H/D = 9 |
| Soil | Mohr–Coulomb, ρ = 897 kg/m³, E = 4 MPa, ν = 0.4, c = 20 kPa, φ = ψ = 0° |
| Soil initial stress | −1 kPa isotropic; gravity stress added once at coupled equilibration |
| MPM | Explicit USF, quadratic B-spline, 0.016 m grid, 2 particles per cell per axis |
| DEM | ρ = 2580 kg/m³; uniform radii 0.002–0.005 m; linear rolling contact, E* = 10 MPa, μ = 0.81, μr = 0.5, μt = 0.3 |
| MPM–DEM | Linear contact, E* = 1.2 MPa; preparation μ = 0.3, kn/ks = 1.5; loading μ = 0.5, kn/ks = 1 |
| Boundary conditions | Fixed base; normal velocity fixed on the four soil sides; free top with surcharge |
| Time and precision | Requested Δt = 2e−5 s, reduced by the solver's critical-step check; float64 fields and arithmetic |
| Capacity | 764800 MPM points (historical specimen: 657096), 20000 spheres (at most 7200 generated), 4 facets, 1 servo |
| Contact allocation | DEM coordination 64/2, cross-contact coordination 32/0; overflow remains an error |

The source protocol is
[`GranularFallingInsideMaterialPoints.py`](../../../../research/StoneColumn/simulation/AxialLoading/TSC9d/GranularFallingInsideMaterialPoints.py),
with geometry and grading from `generate_material_points.py` and
`generate_discrete_particles.py` in the same research directory. The loading
protocol comes from its `stone_column.py`, with the cohesion-series value
c = 20 kPa. Generation now uses the public body generators directly instead of
machine-specific `Particle.txt` / `SpherePacking083.txt` inputs. The compaction
wall is slower than the historical script's 0.8 m/s limit, so the regenerated
packing is not identical to the archived specimen.

The public lifecycle is traced through `src/mpdem/mainDEMPM.py` and
`src/mpdem/Engine.py`: scheme `DEM` holds MPM fixed, and scheme `MPDEM` advances
both solvers. `src/mpm/mainMPM.py` and `src/dem/mainDEM.py` own generation and
restart. The preparation keeps the loading interface's coupling prefix, so
restored cross-contact indices remain consistent. DEM restart history paths
are cleared after the first equilibration run so later stages keep evolved
tangential history.

For CPU execution, pass `--arch cpu` to **both** scripts. This explicitly sets
`GEOTAICHI_FORCE_CPU=1`, including on Apple Silicon. GPU execution requires
a float64-capable Taichi backend; Metal does not support these float64 fields.
Custom locations use `--output-dir PATH` and loading's `--preparation PATH`.

```sh
python examples/mpdem/MultiSphere/StoneColumn/prepare.py --arch cpu --smoke --output-dir /private/tmp/stone_column_prep
python examples/mpdem/MultiSphere/StoneColumn/stone_column.py --arch cpu --smoke --preparation /private/tmp/stone_column_prep --output-dir /private/tmp/stone_column_load
```

`--smoke` uses a 0.04 m grid, one MPM point per cell per axis, 32 DEM spheres
and a few steps per stage. It checks stationary MPM during preparation,
restart, phase switching, finite state and prescribed wall displacement.
It does not produce a compacted specimen or establish bearing capacity.
Smoke preparations cannot be used for a production loading run.

## README rendering

![Half-clipped smoothed MPM stress zz and grey DEM grains](../../../../images/stone_column_9d.png)

The image uses the archived `TSC9d/cohesion/StoneColumnLoadingCoh20kPa`
state at t = 50.60002 s: `particles/MPMParticle000056.npz` and
`vtks/GraphicDEMParticle000056.vtu`. MPM `stress[:, 2]` is smoothed using
the original settings in
[`AxialLoading/postprocessing.py`](../../../../research/StoneColumn/simulation/AxialLoading/postprocessing.py):
`smooth_rad=0.016`, `lower_bound=[0, 0, 0]`, `upper_bound=[0.8, 0.8, 0.55]`.
The renderer calls the same `src/mpm/PostPlot.py::SmoothVar` mass-weighted
quadratic B-spline P2G/G2P operation on CPU/float64. Colors show the complete
smoothed stress zz range (−68.59 to 12.86 kPa), retaining the negative sign
for compression. This recomputes the historical smoothing method for this
snapshot; it does not modify archived data. Nominal particle boxes use saved
positions and sizes.
Only the soil is cut at y = 0.4 m; the central r ≤ 0.12 m soil window keeps the
column visible, and every DEM sphere retains its saved position and radius.
Only the MPM terminal frame remains in this archive, so the image is static.
It is a historical result, not a rerun of the newly packaged example.

```sh
python tools/stone_column_render.py --output images/stone_column_9d.png
```

Rendering needs GeoTaichi's dependencies, PyVista/VTK and a graphics context.
The smoothing run checks finite values and bounds against the raw stress
extrema. Production density,
quasi-static force balance, load–settlement agreement and CUDA execution
remain unverified for the packaged scripts.

Validation on 2026-10-29: both scripts passed static API inspection (the
stage-update warning is expected), and the smoke commands above passed on
Taichi 1.7.0, CPU `Arch.arm64`, with float64 particle fields and arithmetic.
The preparation retained all 5346 MPM positions exactly; 32 DEM spheres and
their saved histories were handed to loading. The final loading time was
0.00012 s, with finite state and 3.99999749e−8 m wall settlement
(expected 4e−8 m, relative tolerance 1e−5, absolute tolerance 1e−10 m).
`python tests/testing/run_partition.py pre-commit -q --taichi-arch cpu`
passed 107 tests (54 deselected); that small gate does not validate this
specimen's full bearing response.
