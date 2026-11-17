# Thousand cubes and a heavy sphere

The default is exactly **1000 cubes** (10×10×10), plus one impact sphere.
The sphere uses a watertight subdivision-4 icosphere: **2562 vertices and 5120
triangles**, replacing the old 12-vertex, 20-triangle contact surface. This is
actual collision geometry, also exported to VTU for rendering.

The old circumscribed radius is retained: 0.1997218684 m after scaling. Density
stays at 30000 kg/m³. Refining the inscribed surface increases its volume and
mass (about 999 kg, previously 606 kg); this follows the unchanged density and
outer radius. Cube geometry, densities, initial velocity, walls, friction,
damping and timestep are retained. Surface-node capacity
includes all 2562 sphere vertices. The previous restriction requiring more
than 1000 cubes has been removed.

```bash
python examples/dem/AffineBody/thousand_cubes_heavy_ball_impact/thousand_cubes_heavy_ball_impact.py --arch gpu --save-every 1
```

`--smoke --steps 2` checks construction and finite advancement without requiring
an impact yet. Full-size CUDA preflight precedes production; later capacity
overflow and convergence failure remain errors. The [model contract](model-contract.json)
and `tests/unit/dem/test_abd_retry_examples.py` cover counts and mesh geometry;
`src/dem/engines/AffineBodyOperator.py` owns the unchanged IPC solve.

Both large scenes start with a nominal **1e-3 s** step. A recoverable solve
failure rolls back the trial and retries at half the timestep, at most twice
per step and never below one quarter of the nominal timestep. Newton/PCG
thresholds stay fixed. Exhausted retries remain a failure. Summaries retain
the fixed-step diagnostic and verify the accepted time increments sum to the
requested duration. The user-requested cube target is **0.25 s**;
retrying can increase the number of accepted steps. A two-step gate proves
construction and capacity only, not the later impact or wall-release dynamics.

The Newton cap is 1000, with a fixed `0.05 m/s` velocity threshold and a
case-specific `1e-10` absolute PCG threshold. Every linear solve checks its true
residual. The full-size configuration uses at most 12012 PCG iterations,
8M raw triplets, 600k PT candidates and 1.5M EE candidates.

Both cubes and sphere use the numerical orthogonality penalty **E=2 TPa** in
`W=VE/8 ||F^T F-I||²`. This controls affine rigidity, rather than prescribing a
calibrated material modulus. The separate normal-contact coefficient is fixed
at **kappa=1e12** from initialization.

For the smallest cube, 1% principal compression costs about 9021.60 J at this E,
above the initial kinetic-plus-gravitational budget of 8166.99 J. This is a
conservative concentration-of-energy bound, not a necessary minimum modulus.
The completed CUDA/f64 run accepted all **250 steps at 1 ms without retries**;
maximum principal-stretch deviation was `3.8721e-5`, minimum volume ratio
`0.99996336`, and PT/EE and wall gaps stayed positive. All 1000 cubes moved
more than 0.01 m; the sphere advanced 0.66351 m. The per-step energy check
bounded kinetic + orthogonality + floor-relative gravity by its initial value;
it excludes nonnegative IPC barrier energy and is not a full conservation proof.

The fixed `newton_tolerance=0.05` m/s measures surface correction velocity.
At `dt=1e-3` it bounds the unapplied surface correction by 50 micrometres;
this is a stopping criterion, not a trajectory-error bound. The original
Newton/Hessian, friction regularization and PCG tests are used. There is no
semi-implicit early exit or automatic tolerance relaxation. The public
`set_affine_body_parameters(newton_tolerance=...)` keyword remains available.

The fixed PCG choice uses the [mass-norm error bound](../../../../src/linear_solver/README.md#matching-linear-and-newton-tolerances).
For the original geometry/masses and minimum `dt=2.5e-4 s`, the resulting
linear surface-velocity error bound is 1.2e-3 m/s. The Newton target is
0.05 m/s. These are explicit case settings; no tolerance changes after failure.
