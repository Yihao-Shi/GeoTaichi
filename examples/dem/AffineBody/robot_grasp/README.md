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

## Complete trajectory check and optimization

```sh
python examples/dem/AffineBody/robot_grasp/robot_grasp.py \
  --arch cpu --gradient-check --output-dir /private/tmp/grasp_gradient
python examples/dem/AffineBody/robot_grasp/robot_grasp.py \
  --arch cpu --train-iterations 3 --output-dir /private/tmp/grasp_training
```

The gradient check records all 320 steps, reverses their actual angle commands,
and reruns the complete forward problem for positive/negative perturbations
of 0.01 and 0.005 degrees. Directions act on arm lifting, finger closing, and
finger opening. The terminal objective combines the free payload's center and
velocity; it retains sensitivity after release and landing. The declared
derivative error bound is `1e-6 + 0.01 * max(abs(AD), abs(FD))`.
`gradient-check.json` records every probe; a failed comparison raises an error.

Before reversing the friction refresh, arm-gradient errors were about 2.26%
and the closing-direction gradient had the wrong sign. `DifferentiableABD`
now reverses each executed frozen-friction equilibrium and its cache refresh,
including the fixed support plane. This changes the derivative chain rather
than the forward physics or convergence criteria.

The optimization minimizes half the squared error between the final payload
x-coordinate and 1.18 m. It adjusts the smooth arm-lift amplitude, retaining the
close/lift/hold/open protocol. Each update uses the adjoint derivative in a
bounded scalar Gauss-Newton step of at most two degrees. No finite-difference
gradient is used for training. `training.json` records actual forward losses;
acceptance requires strictly decreasing loss, at least 70% total loss reduction,
and the original lift/release height checks. This demonstrates optimization
near a successful grasp; it does not establish learning a grasp from arbitrary
controls. Add `--gradient-check` to the training command to run both in order.

Contact sequences can change during the recorded trajectory. The gradient is
still local to the executed activation/closest-feature branches and iteration
counts; exact nonsmooth switching points need one-sided checks or a separate
optimization treatment. GPU behavior remains unverified.

The complete check passed on ARM64 CPU/float64, `dt=0.005` s, with
`linear_relative_tolerance=1e-10` explicitly selected. The default absolute
linear tolerance `1e-14` and all forward nonlinear/friction criteria were
retained. The recorded friction-contact count ranged from zero to 12 and
changed 11 times during the checked trajectory.

| Direction | Relative error at 0.01 degrees | Relative error at 0.005 degrees |
| --- | --- | --- |
| Arm lifting | 2.38e-5 | 3.85e-5 |
| Finger closing | 6.00e-3 | 5.31e-4 |
| Finger opening | 5.72e-6 | 3.43e-6 |

Three optimization updates changed the arm amplitude from -20 to
-15.3995266 degrees. The released payload's final x-coordinate moved from
1.196434669 m to 1.179979891 m (target 1.18 m). The loss decreased from
1.35049e-4 to 2.02187e-10. The final protocol still lifted the free payload
by at least 0.168497 m during hold and let it fall 0.169144 m after release.
These are results for this one illustrative task and terminal objective.
