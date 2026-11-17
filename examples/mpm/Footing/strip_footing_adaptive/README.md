# Strip footing: uniform, adaptive L1 and adaptive L2

This is a 2D plane-strain section of
[`StripFootingMC.py`](../StripFootingMC.py). It preserves the
10 by 10 m soil and 1 by 2 m rigid half-footing. The uniform control uses
the normal saved result's four particles per grid direction. Coordinates are translated by -0.1 m to put the physical bottom and
symmetry boundary at zero; exterior ghost margins are omitted. The full
footing width is 2 m. Geometry, material, contact and loading belong entirely
to `strip_footing.py`; evaluation and orchestration are separate scripts.
Soil particles refine together with the background grid.

The soil uses Mohr-Coulomb with density 1000 kg/m³, E=100 kPa, nu=0.49,
cohesion 1 kPa and zero friction/dilation angles. Gravity and background
damping are zero. GIMP, B-Bar, USL and alphaPIC=0.005 match the reference.
All three runs use particle integration; Gaussian integration is disabled
as requested. GeoContact uses friction 0.1, penalty [0.78, 2.5] and
`cut_off=0.8`, the explicit reference value. The reference script previously
misspelled this key as `cutoff`; the constructor silently used 1.5 instead.
That typo is fixed, and GeoContact rejects it so contact cannot silently start
at a wider gap. The initial October 22 comparison used 1.5 and must not be
treated as a run of the corrected input. `run.json` records effective contact
parameters as well as the executed source.

| Run | Grid spacing (m) | Initial soil particles per cell direction | Initial soil count |
|---|---|---|---:|
| uniform | 0.1 | 4 | 160,000 |
| adaptive L1 | 0.2 → 0.1 | 4 | 40,000 |
| adaptive L2 | 0.4 → 0.2 → 0.1 | 4 | 10,000 |

Soil particle spacing starts at 0.025, 0.05 and 0.1 m, respectively. With
`RefineParticles=True`, each eligible soil parent splits into four children
per refinement level, preserving mass and material history. The native splitter
excludes rigid particles, so the footing retains the same 3,200 points at
0.025 m spacing in all runs. Its fixed indices remain valid for the prescribed
velocity callback because soil children are appended rather than inserted.
The device history also records the evolving total and soil particle counts.
The grid uses strong `ShapeFunction` hanging constraints without bridging.
Accumulated equivalent plastic deviatoric strain (`epdstrain`) of 0.01 triggers
refinement, checked every 100 steps; this threshold is dimensionless. The
physical buffer width is 0.4 m in both adaptive runs. The buffer expands
material-indicator seeds; following already split particles
preserves their cell resolution and 2:1 balance without creating another
0.4 m buffer. Earlier runs that buffered migration seeds can over-refine
soil below the plastic-strain threshold.
Splitting also waits until every child center lies inside cells at the child's
refinement level. This prevents particle creation across a coarse/fine boundary
from driving another wave of grid refinement. Deferred parents preserve their
state and can split at later checks; the interface can temporarily have coarser
particle sampling than the interior. Runs predating this check can still
over-refine even when migration seeds are not buffered.
The threshold is a starting validation choice, not a calibrated error estimate. The earlier 500 Pa
equivalent-stress trigger selected over 93% of the final soil volume, largely
eliminating the benefit of local refinement. Results using that trigger do not
represent the current plastic-strain criterion.
The same threshold applies to both levels: a cell that still qualifies at a
later check advances again. Levels are stored per original coarse cell,
balanced to 2:1, and never coarsened. L2 is not an independent error estimate.
`RefineRatio=1.0` reserves the complete finest-grid node capacity (12,221
nodes), including GIMP support halos and retained compact IDs. It does not
force refinement; the plastic-strain indicator still selects cells. The previous
0.7 allocation overflowed in both adaptive runs.

The original 5 s post-step velocity ramp and subsequent constant velocity
are retained, including the initial -0.02 m/s velocity. The default duration
is 17.5 s and dt=1e-5 s. The model rejects an unstable prescribed timestep.
Loading, contact-force reduction and refinement counts stay on device;
the force history is copied out only after the run. Particle snapshots are
ordinary output, saved every 1 s. Results remain under this example's own
`OutputData/` directory.

The retained completed L2 result is
`OutputData/split_boundary_fix_20261024/adaptive_l2/`: a CUDA/f64 run through
17.5 s with coupled grid and particle refinement and both migration-growth
fixes. Soil particles grow from 10,000 to 65,428; the footing retains 3,200
particles. All 19 particle snapshots, VTU outputs, force history, executed
model and validation records are retained without checkpoints. File checksums
and the frozen numerical source are saved alongside the data.
Old runs, diagnostic intermediates and remote-job records have been removed.
The parent directory contains `comparison.json` and
`contact_force_comparison.png`. At the historical saved times (approximately
0.25 s apart), L1 and L2 force RMS differences from the uniform control are
0.67% and 2.95%, respectively; both pass the numerical and force checks.
This is point sampling without averaging or smoothing and can miss brief
contact impulses. Shared GPU use prevents a controlled speedup claim.

Prepare display data and render independently of the simulation:

```bash
conda run -n geotaichi python examples/mpm/Footing/strip_footing_adaptive/draw/prepare_render.py \
  examples/mpm/Footing/strip_footing_adaptive/OutputData/split_boundary_fix_20261024/adaptive_l2
pvpython examples/mpm/Footing/strip_footing_adaptive/draw/render_paraview.py \
  examples/mpm/Footing/strip_footing_adaptive/OutputData/split_boundary_fix_20261024/adaptive_l2
```

ParaView renders `epdstrain` over the fixed range 0–1 (higher values saturate),
the rigid footing and the actual saved adaptive leaf mesh. Neighboring particle
centers form the displayed surface using Delaunay2D with alpha equal to 1.1
times each body's initial particle spacing: 0.11 m for soil and 0.0275 m for
the footing. This includes the initial coarse cloud before particle splitting.
The animation uses the 19 saved states without temporal interpolation, at
4 frames/s; `render/footing_l2.gif` supplies the root README's `images/footing.gif`.
Only the final GIF, final PNG and render metadata are retained. The commands
above regenerate the PVD/VTK display inputs, scene state and individual frames.

```bash
conda run -n geotaichi python examples/mpm/Footing/strip_footing_adaptive/draw/evaluate.py extract-reference \
  examples/mpm/Footing/FootingLargeStrain \
  examples/mpm/Footing/strip_footing_adaptive/OutputData/historical_normal.csv
conda run -n geotaichi python examples/mpm/Footing/strip_footing_adaptive/run_comparison.py \
  --reference examples/mpm/Footing/strip_footing_adaptive/OutputData/historical_normal.csv
```

`run_comparison.py` serializes the three jobs and records commands, PIDs,
exit status and a source hash in `OutputData/queue.json`. It then runs
`draw/evaluate.py compare`. Before production, it runs all three geometries
for 0.1 s under `preflight/` and evaluates them independently. With no gravity,
prestress, or appreciable penetration, this gate rejects particle speeds over
1 m/s or volume ratios outside [0.5, 2], in addition to the ordinary invariants.
These generous early-response bounds detect finite but explosive states;
they do not establish historical force agreement. A failed preflight stops
production. Finest-level activation remains a production acceptance check.
`--after-pid <simulation-pid> <postprocessing-pid> --wait-gpu` waits for both
existing processes and then for an idle GPU. Process creation times prevent
following a reused PID. Use a fresh `--output` under this example's directory
for each queue; failed earlier results are retained.
Use `--reuse-uniform <completed-uniform-directory>` to reuse a matching,
numerically accepted fixed-grid baseline and run only adaptive L1/L2, including
their preflights. The runner checks the recorded physical and numerical inputs
and records provenance in `uniform/reuse.json`. Reused wall time is retained
for reference and marked incomparable with new controlled timing measurements.
The simulator also runs independently with
`--level 0`, `--level 1` or `--level 2`. Existing run directories are never
overwritten. `--smoke --output <temporary-directory>` uses a smaller soil
block and must not be reported as a production comparison.

The evaluator compares the uniform baseline with the saved
normal `FootingLargeStrain` result, and each adaptive run with the
uniform baseline. Old 3D forces are divided by their measured 0.1 m thickness.
Settlement comes from rigid-particle positions, not assumed time times speed.
Historical comparisons use only the shared settlement interval. The normal
saved series contains 91 frames through approximately 22.5 s and 2 m of
penetration; the new runs last 17.5 s and approximately 1.5 m. The normal
series retains a 5 s ramp followed by constant velocity, as in this setup.
The neighboring historical log is not a reliable parameter record for these
snapshots: saved sliding-node tangential/normal force ratios saturate at 0.1,
matching this example. The exact historical source revision is unavailable.
The 3D thin-layer and 2D B-Bar operators differ, so historical agreement must
be measured rather than presumed.
Reassembling historical states at 1, 5 and 10 s gives forces within about
0.4% of the saved values with an unscaled normal impulse and cutoff 0.8.
This is consistent with the older friction-contact kernel. The exact original
source revision is unavailable; it does not establish that the historical
run used the current GeoContact relaxation [0.78, 2.5]. Keep that contact-law
control separate from changes to material parameters or numerical damping.

At the user's request, force comparisons now use the original historical
saved times (approximately 0.25 s apart), using linear interpolation in time
without averaging or smoothing. Targets remain at most 10% pointwise RMS,
sampled peak and tail-mean force errors. High-frequency peaks remain diagnostic
and no longer veto this comparison. The evaluator separately reports the median
peak-to-peak amplitude of complete 1 s cycles after the 5 s ramp; no independent
amplitude tolerance has been prescribed. These samples can miss short impulses.
The evaluator also checks finite states, positive
mass/volume, mass conservation, equal/opposite vertical contact force,
matching initial physical domains/mass/volume, four soil particles per initial
cell direction, identical rigid sampling and actual finest-level grid and
particle refinement. Volume ratios use mass and initial material density so
child volumes are compared with their own reference volumes. It writes
`comparison.json`, `comparison.md` and `contact_force_comparison.png`, and
returns a failing exit code if a required comparison fails. The loose response
bounds used by preflight also apply to the full run: volume ratios in [0.5, 2] and particle speeds at most
1 m/s. These benchmark bounds reject finite but collapsed/explosive states;
they are not general material admissibility limits. Missing historical
data cannot count as a pass. Passing this comparison does not establish mesh
convergence or controlled performance; remote GPU contention affects timing.
Pass `draw/evaluate.py compare ... --output <new-directory>` when reevaluating
older runs to preserve their original comparison artifacts.

Source trace: public lifecycle and callbacks in `src/mpm/mainMPM.py` and
`MPMBase.py`; effective contact parameters in `src/mpm/Contact.py`; adaptive
compatibility, markers and device topology in `AdaptiveQuadrilateralElement.py`,
`AdaptiveQuadrilateralKernel.py` and `AdaptiveTopologyKernel.py`. No solver,
constitutive model, coupling or MCP interface is changed by this example.
