# Five cubes on an incline

Five free affine cubes share a fixed 30° plane. Their friction coefficients,
from the front lane to the back lane, are 0.10, 0.30, 0.50, 0.70 and 0.90.
The first three slide; the last two remain macroscopically stationary because
their coefficients exceed `tan(30°) = 0.57735`. No body has prescribed motion.

```bash
python examples/dem/AffineBody/five_cubes_incline/five_cubes_incline.py \
  --arch cpu --output-dir FiveCubesIncline
python examples/dem/AffineBody/five_cubes_incline/render.py FiveCubesIncline
```

The default run uses CPU/f64, BackwardEuler, `dt=0.001 s`, duration `1.2 s`,
0.25 m cubes, density 1000 kg/m³ and a 2 TPa numerical orthogonality penalty.
Contact uses `Dhat=0.003 m`, `BarrierStiffness=8e8`, no damping and
`friction_epsv=1e-4 m/s`. Each cube has its own material pair with the plane.

Friction is solved to an outer fixed point: `friction_iterations=-1`,
`friction_max_iterations=50`, `friction_tolerance=1e-7 m/s`.
Newton uses `1e-8 m/s`, at most 1000 iterations; PCG uses an absolute residual
threshold of `1e-13`. The immutable cube mass matrix and surface basis give a
linear velocity-error bound of `2804.34 * residual` at the default timestep:
the chosen threshold bounds this error by `2.81e-10 m/s`, below 3% of the Newton
target. This is a fixed, case-specific accuracy budget, not an automatic
tolerance adjustment. An unconverged step fails rather than silently advancing.

Initial height and small orientation corrections balance both normal force
and friction-induced moment. A setup-only three-variable solve uses the
production IPC barrier function and surface quadrature; it does not advance
time. This avoids releasing an unbalanced contact pressure distribution.
Subsequent motion, contact and friction iterations use the ordinary device ABD
solver. The example's NumPy operations only prepare geometry and inspect output.

The script validates positive gaps, finite state, convergence at every step,
principal stretch deviation below `1e-4`, and the inclined-plane prediction
`a = g(sin(theta) - mu*cos(theta))`. Sliding displacement is checked against
the BackwardEuler discrete trajectory, with 1% relative or 20 µm absolute
tolerance. The two sticking cubes must move less than 0.2 mm. Regularized
static friction permits microscopic creep; it is not a kinematic lock.

The renderer reads the saved triangle VTUs without interpolating poses. The
finite slab is a display of the infinite contact plane; the whole trajectory
stays within it. Lane markings and colors are visual annotations. The GIF
plays at 0.4× physical speed, with the actual simulation time displayed.
Blender 5 (EEVEE) and FFmpeg are required; the renderer currently uses the macOS
Blender application path. Only triangle geometry is read for rendering; the
matching native recorder files supply the saved simulation timestamps.

API evidence follows `DEM.set_affine_body_parameters` into `Simulation` and
`AffineBodyEngine`, and `DEM.add_property` into the material-pair wall tables
in `AffineBodyOperator`. The static friction smoothing and wall quadrature are
covered by `tests/unit/dem/contact/test_ipc_affine_friction_assembly.py`.
The [model contract](model-contract.json) fixes the physical acceptance criteria;
each run writes `validation_summary.json`, `history.json` and
`validation-evidence.json` beside its native output.

The completed CPU/f64 run reached 1.2 s in 1200 steps. Final downslope
displacements were:

| Friction coefficient | Displacement | Regime |
| --- | --- | --- |
| 0.10 | 2.922342 m | Sliding |
| 0.30 | 1.697941 m | Sliding |
| 0.50 | 0.473539 m | Sliding |
| 0.70 | 0.06978 mm | Regularized sticking |
| 0.90 | 0.04816 mm | Regularized sticking |

The maximum sliding displacement error over all steps was 1.86 nm. Minimum
gap was 0.108 mm; maximum principal-stretch deviation was `7.09e-10`.
Every refreshed outer-friction check converged, with maximum residual
`2.22e-8 m/s`. The equilibrated, steady-contact case needed one outer solve per
step; it still refreshed the friction data and tested convergence, rather than
using a prescribed one-iteration limit. All five bodies' total energies
decreased at every saved 0.01 s frame, including IPC and orthogonality energy.
The local solve took about 13.3 minutes including first-use compilation.
