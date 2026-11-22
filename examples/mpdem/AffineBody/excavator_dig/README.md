# Stationary excavation with plastic MPM soil

This first excavation model fixes the base and drives two revolute joints: a
boom and a bucket. The bucket is a closed, consistently oriented solid with an
open front and top; its cavity is not filled with rigid material. All soil
points can contact the bucket, including points exposed as it cuts into the
bed. There is no attachment of soil to the bucket.
The finite soil domain has frictionless normal constraints on its four vertical
sides. Soil and the rigid mechanism both contact the same fixed bottom plane.

The existing continuum Direct ULMPM route provides nonassociated
Drucker-Prager soil, a fixed support plane and two-way lagged-friction mesh IPC
with ABD. Dimensions, material parameters, capacities and acceptance criteria
are in [model-contract.json](model-contract.json). The soil and mechanism are
illustrative and have not been calibrated.

```sh
python examples/mpdem/AffineBody/excavator_dig/excavator_dig.py \
  --arch cpu --output-dir /private/tmp/excavator
```

The 1.6 s protocol settles the bed, tilts the cutting edge downward, inserts
it, cuts horizontally, curls the bucket, then lifts and holds it. During the
cut, the two existing joint references keep the cutting tip 0.006 m above the
bottom plane. The insertion-to-cut transition starts with zero angular velocity,
including the inverse-trigonometric pitch correction at a vertical tip.
The boom lowers to 33 degrees, then returns to 30 degrees
while the bucket curls so its rear corner clears the bottom plane. The bucket
pitch is maintained during lifting through the second joint's reference.
A separate stick, undercarriage, cylinders,
hydraulics and drive system are unnecessary for this first task.

Both MPM and ABD explicitly use backward Euler `[1, 0.5, 1]` in this model.
Inner plastic Newton and outer friction convergence use the physical velocity
correction for each child. Inexact Newton adapts Krylov accuracy during early
iterations and requires strict terminal solves. The base linear absolute
tolerance is 1e-12 with an 8000-iteration budget; Newton/friction velocity
tolerances remain 1e-7 m/s. Ground-contact buffers use the library's point-count
defaults to cover every soil point.
The friction fixed-point limit is 200 outer solves: a recorded soil-contact
state required 122 to reach the unchanged velocity threshold.
The forward model also selects `friction_relaxation=0.8` to damp oscillating
friction iterates, with CCD limiting every backstep. This changes the iteration
path while retaining the same final force and velocity criteria. Plastic
linear systems use device diagonal scaling and are checked on the original matrix.

`--stop-time 0.005` checks construction and two coupled steps; it does not
establish that digging succeeded. Complete runs require at least 0.2 kg of soil
inside the lifted cavity above the original bed top plus 0.05 m, positive
mixed-contact clearance, relative mass error below 1e-10 and mixed reaction-force
imbalance below 1e-8 N. The final contact-force diagnostic uses the existing
mixed residual at its assembled geometry and does not alter accepted states.
Rigid surfaces must also remain strictly above the bottom plane.
Both affine determinants must stay positive and the maximum
`norm(A.T @ A - I)` must remain below 0.01. The rigidity penalty is 1e11 Pa;
the thinner boom needs this stiffness to keep the cutting edge close to its
designed position under load.
Joint-anchor errors must remain below 1 mm and angle-reference errors below
0.1 degree. The position/motor penalties represent ideal servos with finite
compliance; they do not model actuator power or torque limits.

Outputs include native MPM/ABD VTU and NPZ files, `trajectory.csv`,
`summary.json` and `validation-evidence.json`. CPU is forced explicitly and uses
float64. The 0.04 m grid/particle spacing is coarse; spatial and temporal
refinement, calibrated soil parameters and GPU execution remain separate
requirements for quantitative excavation predictions. Full-trajectory digging
gradients and excavation trajectory optimization are not validated by this
forward example. The trajectory-adjoint API requires `friction_relaxation=1.0`
and its documented fixed-step configuration.

The nominal timestep is 0.0025 s. A failed step may retry up to seven times,
halving the timestep to a minimum of 0.00001953125 s. The preintegration callback
sets joint references at `system.mpm.time + system.mpm.dt` before every attempt,
including retries. Accepted timesteps and retry counts are recorded in
`trajectory.csv`; their sum must equal the requested duration. Material/contact
parameters and convergence criteria remain fixed. This recovery mechanism does
not establish temporal convergence of an excavation result.
The accepted step is retained for the next attempt. Every eighth accepted step
can increase it by 20% if that step needed no retry, up to the nominal maximum.

## Six-second excavation-to-truck smoke run

`excavator_cycle.py` takes the authored `trajectory.csv` and three contact OBJ
files from `research/demo/excavator` (or `--scene-dir`). That asset directory
is local/ignored in Git; copy it to another machine before running there.
The earlier example reached the end of its motion without retaining soil in
the raised bucket. This run reports numerical completion separately from
successful transport.

```sh
python -u examples/mpdem/AffineBody/excavator_dig/excavator_cycle.py \
  --arch cpu --particles 180 --grid-size 0.5 --dt 0.01 \
  --output-dir /private/tmp/excavator-cycle-results
```

Use a new output directory. `--stop-time` can shorten a setup probe; only a
run reaching 6 s covers cut, curl, lift, swing, dump and retract. The pilot
samples equal-volume MPM points inside the original 13.57 m³ pile. The
original contact meshes (84 bucket and 92 bed triangles) are kept intact;
there are no ABD bodies for the tracks, links, cylinders or truck chassis.
The soil has an open footprint and a fixed ground plane at `z=0`.

The public revolute motor does not prescribe a six-DOF pose. The cycle uses
four finite-stiffness spherical anchors per body, updates the bucket's
world-side targets at the end of **every attempted** step, and holds the
truck targets constant. Targets use linear pin interpolation and quaternion
SLERP. Use the accepted ABD output, not the command motion, for replay.
The CPU pilot uses float64, the stationary example's illustrative soil and
contact laws, backward Euler, unchanged nonlinear/linear stopping criteria,
and rollback with halving retries. The larger model uses a 0.10 m Newton
correction cap with CCD and line search still enabled.

`run-config.json` records the selected inputs; `accepted-steps.csv` records
accepted time/step size, retry and iteration counts, mixed contact, gaps,
actual-vs-command tracking and cavity masses. `summary.json` records mass
conservation and the mass in the bucket and bed at the first accepted step
at or after each phase boundary. Its spill estimate counts the particles
carried above the initial pile at the end of lift that are subsequently
outside both cavities. `failure.json` is written for an aborted solve;
missing `summary.json` is not evidence of completion.

At 180 equal-volume points, one particle represents about 128 kg, so this is
not a quantitative truck-load prediction. For a remote pilot, change
`--particles` to 10,000 and then 20,000, refine `--grid-size` and `--dt`, and
size `--max-point-triangle-pairs` from contact/CCD high-water marks and device
memory. Run each at its intended backend; the local CPU pilot does not
validate CUDA, spatial/time convergence, or calibrated soil behavior.
