# Bunny pile wall release

The default scene contains **1000** original watertight bunnies (2503 vertices,
5002 triangles each), arranged on a 10×10×10 grid. Grid pitch encloses every
rotated bunny's circumsphere with 12 mm between spheres. `container_bounds()`
defines the bottom and all four side walls; the top stays open. The renderer
uses the same bounds from `validation_summary.json`.

At or after 2 s, the right wall moves away once the maximum surface speed is
at most 0.15 m/s. The default simulation ends at 4 s. Passing requires wall
release, horizontal displacement above 0.05 m, finite vertices, and positive
retained-wall gaps above 1e-8 m. `--smoke --steps 2` skips the release/runout
acceptance, but retains finite-state, target-time and gap checks.

The fixed orthogonality penalty is **2 TPa** and contact barrier stiffness is
**8e12**. Density 1150 kg/m³, friction 0.45 and local damping 0.08 are retained.
The fixed Newton velocity tolerance is **1e-5 m/s**, with PCG absolute **1e-12**.
PT/EE candidate buffers reserve 16000/32000 slots per body (minimum 80000/180000).
These are provisional bounds: overflow remains an error. A full-size CUDA
capacity gate must pass before production; a short gate does not establish
capacity throughout settling. No reduced mesh or CPU solve is substituted.

The fresh 1000-body CUDA/f64 trajectory with these settings is running toward
4 s. Complete wall-release/runout acceptance and the full-run peak capacities
remain unverified. The running prefix used about 8953 MiB of device memory;
the 1 GB initial CUDA pool setting is not a total VRAM limit.

```bash
python examples/dem/AffineBody/bunny_pile_wall_release/bunny_pile_wall_release.py --arch gpu
```

The [model contract](model-contract.json) records units, acceptance and capacity
limits. Evidence owners: this example's public DEM configuration; the maintained
`tests/unit/dem/test_abd_retry_examples.py`; `src/dem/engines/AffineBodyOperator.py`
and `src/dem/neighbor/AffineBodyNeighbor.py` for device assembly and overflow.

Both large scenes start with a nominal **1e-3 s** step. A recoverable solve
failure rolls back the trial and retries at half the timestep, at most twice
per step and never below one quarter of the nominal timestep. Newton/PCG
thresholds stay fixed. Exhausted retries remain a failure. Summaries retain
the fixed-step diagnostic and verify the accepted time increments sum to the
requested duration. The cube target is 0.25 s and the bunny target is 4 s;
retrying can increase the number of accepted steps. A two-step gate proves
construction and capacity only, not the later impact or wall-release dynamics.

The Newton iteration cap is 1000 and PCG permits up to 12000 iterations.
In the remote sequence this case follows the cradle and cubes.

The CPU/CUDA operator now allocates friction-record payloads in device pages
for the active contact prefix, including its adjoint backup. The previous
dense reservation for 1000 bunnies consumed about 3.47 GiB for these two
buffers alone. Capacity/overflow checks and the mesh resolution are retained.

The fixed `newton_tolerance=1e-5` m/s limits the unapplied surface correction
to 10 nanometres at `dt=1e-3`. A strong orthogonality penalty can retain
substantial energy in a small affine stretch during rotation, so a correction
velocity threshold alone needs a case energy/shape check. A controlled free
translation/rotation test at 2 TPa produced artificial energy with 0.05 m/s;
one additional Newton update at 1e-5 m/s removed that increase while keeping
PCG fixed at 1e-12. The original Newton/Hessian and friction law are used.
The public `set_affine_body_parameters(newton_tolerance=...)` keyword remains
available; thresholds never relax after a failure.

The fixed PCG choice uses the [mass-norm error bound](../../../../src/linear_solver/README.md#matching-linear-and-newton-tolerances).
For the original geometry/masses and minimum `dt=2.5e-4 s`, the resulting
linear surface-velocity error bound is 9.9e-7 m/s. The Newton target is
1e-5 m/s. These are explicit case settings; no tolerance changes after failure.
