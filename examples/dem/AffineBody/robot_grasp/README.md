# Pure ABD grasp, lift and release

This is stage one of the mechanism workflow: pure ABD grasping first, then a
stationary excavator with MPM soil, wheeled mobility on soil, and humanoid
walking. Only the first stage is implemented here.

The model has one world-hinged arm, two opposing hinged fingers, a free rigid
box payload and a horizontal support plane. The three servos follow smooth
angle references. The payload has no joint, fixed motion or lifting force;
gravity and IPC contact determine its motion. Detailed electric-drive and
hydraulic models, inverse kinematics and a planning framework are not required
for this test.
Parameters are illustrative and are specified in [model-contract.json](model-contract.json).

The small `Dhat=0.003` m uses `BarrierStiffness=1e9`. The existing
squared-distance barrier gives a small-gap force approximately
`2 * area * kappa * Dhat**4 / gap`; stiffness must therefore be scaled with
geometry and activation distance. At `gap=1e-7` m and a 0.0025 m² weighted
contact patch, `kappa=1e6` supplies only about 4 N, below the pinch load.
The chosen stiffness keeps the required contact reaction away from the
existing ACCD clearance rather than reducing that clearance or loosening
convergence tests.

```sh
python examples/dem/AffineBody/robot_grasp/robot_grasp.py \
  --arch cpu --output-dir /private/tmp/robot_grasp
```

The default protocol lasts 1.6 s at a 0.005 s timestep:

| Time (s) | Action |
| --- | --- |
| 0--0.1 | Settle on the support plane |
| 0.1--0.3 | Close the fingers |
| 0.3--0.7 | Raise the arm |
| 0.7--1.0 | Hold |
| 1.0--1.2 | Open the fingers |
| 1.2--1.6 | Let the payload fall |

`--dt 0.01` runs the same motion at a coarser timestep; `--dt 0.0025`
refines it. `--stop-time 0.05` runs a short prefix for compilation/configuration
inspection. Prefix runs report `partially_validated` and do not claim that a
grasp succeeded. CPU uses float64 with `GEOTAICHI_FORCE_CPU=1`; GPU evidence
must be collected separately.

Outputs are native VTU surfaces, NPZ affine states, per-step `trajectory.csv`,
`summary.json`, and `validation-evidence.json`. The self-check requires the
payload to remain at least 0.12 m above its initial height during hold, drift
no more than 0.03 m in the arm frame during hold, and drop at least 0.08 m
after release. It also checks finite state, positive affine determinants,
positive support-plane gaps and an affine rigidity error below 1 percent.
A completed process without these task checks is not a successful grasp.

The example uses public `DEM` construction, joint targets and run callbacks.
Its observation callback reads `AffineBodyEngine.output_snapshot()` solely at
the diagnostic/output boundary; downloaded data never updates the solver.
`preintegration_function` sets the next endpoint's target before every solve,
including the first physical step used for JIT compilation. `python_callback`
marks this setup/control callback as Python rather than a Taichi kernel.

```sh
python agent/geotaichi-model-builder/scripts/score_physics.py \
  examples/dem/AffineBody/robot_grasp/model-contract.json \
  --evidence /private/tmp/robot_grasp/validation-evidence.json
```

Constraints and motors have finite compliance. This is a small contact-driven
task, not a calibrated industrial gripper or a validated MPM/vehicle/walking
model. Supported standalone ABD control-sequence differentiation is described
in the [DEM guide](../../../../src/dem/README.md#robot-control-and-differentiable-trajectories).

The release phase also exercises the original lagged solver near the Coulomb
sliding threshold. Newton uses its configured backtracking budget before the
quadratic friction upper-bound retry, with an earlier retry for a flat energy
reflection that would cycle across zero slip; replacing the direction after only two
rejected trials stalled this landing. This search repair changes no energy,
contact parameters, physical Hessian or convergence threshold.

## Forward validation

The full 1.6 s protocol passed on ARM64 CPU/float64 with all three timesteps:

| Timestep (s) | Steps | Minimum hold lift (m) | Maximum hold drift (mm) | Release fall (m) |
| --- | --- | --- | --- | --- |
| 0.01 | 160 | 0.221735 | 0.137659 | 0.222181 |
| 0.005 | 320 | 0.220904 | 0.163729 | 0.221870 |
| 0.0025 | 640 | 0.220494 | 0.192594 | 0.222166 |

The default-to-refined hold-height difference is about 0.41 mm. Every run
kept strictly positive support-plane clearances, positive affine determinants,
and a rigidity error below 0.0041. These checks cover this illustrative model;
GPU behavior and a full grasp/contact trajectory adjoint are not established
by the forward runs.

## Local control-gradient check

A separate two-step window starting from the saved `t=0.9` s holding state was
checked with `DEM.differentiable_affine(steps=2)`. The objective was the free
payload's center height. For this adjoint query, the existing public setting
`linear_relative_tolerance=1e-10` was selected explicitly; the absolute default
`1e-14` and all nonlinear/friction criteria were retained. A height-objective
seed has a different scale from the incremental forward force residual, so a
relative residual criterion is needed rather than assuming one absolute
threshold serves both. No implicit precision floor was added.

The adjoint uses the existing complete-body preconditioner and its pivoted-LU
BiCGSTAB fallback. With scalar control-point preconditioning, the original
strong-contact query stalled at a true residual around `1.6e-9`; complete-body
blocks reduced this to the floating-point scale while preserving the operator.
Central differences of the complete two-step forward solve at `1e-3` and
`5e-4` degrees gave directional derivatives `-8.43677494e-4` and
`-8.43677889e-4`, versus the adjoint `-8.43678672e-4`. Relative errors were
`1.40e-6` and `9.28e-7`. Backward restored the terminal accepted position.
This is evidence for the smooth holding window; the whole close/lift/release
trajectory across contact switches and GPU execution remain unverified.
