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
python examples/mpdem/AffineBody/ExcavatorDig/excavator_dig.py \
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

## Excavation-to-truck cycle

`excavator_cycle.py` uses the soil geometry, bucket and bed contact solids, and
1.75× slowed bucket trajectories from the [explicit cases](../../LevelSet/ExcavatorDig/README.md).
The bucket holds its initial pose for the corresponding explicit release time.
Four compliant spherical anchors per body drive the pose with continuous
clamped-cubic interpolation for the pin position, yaw and pitch. The truck
bed stays in place. Use `--scene-dir inputs` from this directory for the
legacy 6 s scene and its quaternion trajectory.

| `full` case | MPM points | Grid (m) | Bucket held (s) | End time (s) |
| --- | ---: | ---: | ---: | ---: |
| `cone` | 150,000 | 0.25 | 1.8 | 16.8 |
| `trench` | 149,454 | 0.3125 | 1.1 | 16.1 |

```sh
python -u examples/mpdem/AffineBody/ExcavatorDig/excavator_cycle.py \
  --case cone --resolution full --arch gpu --output-dir /tmp/abd-cone
python -u examples/mpdem/AffineBody/ExcavatorDig/excavator_cycle.py \
  --case trench --resolution full --arch gpu --output-dir /tmp/abd-trench
```

The cone has a 5.5 m radius and a 24° surface angle. The trench is
3.125 × 6.25 × 1.25 m with a regular 57 × 114 × 23 particle lattice.
Both use cohesionless, nonassociated Drucker–Prager. Zero cohesion places
the finite-strain yield apex at zero mean stress. The initial vertical
Kirchhoff stress is `-rho*g*depth`, with horizontal stress ratio `K0=0.5`.
The rough base fixes all displacement components at grid nodes at or below
`z=0`. The trench constrains the normal component along each wall, up to
1.5625 m on the 0.3125 m grid for the authored 1.5 m rim.

The full nominal step is 0.005 s. Backward Euler with Linear transfer and
mesh BarrierIPC govern the implicit solve. The paired hold time is fixed.
The explicit runs measured their release from soil velocities. Confirm the
ABD settling, material state, tracked pose and contact high-water marks at
the intended resolution. The default mixed-pair budget is derived from
a coordination number of `256 / 176` (about 256 pairs for 176 triangles).
The solver reports an error if this pilot budget overflows.

Use `--resolution low --stop-time 0.01 --arch cpu` to check initial setup.
`--checkpoint-interval` controls the atomic accepted-state `checkpoint.npz`
(default 0.05 s). Resume with the same case and discretization, a larger
absolute `--stop-time`, and a fresh output directory. `run-config.json` and
`accepted-steps.csv` record effective inputs, accepted steps, contact gaps,
tracking errors and cavity masses. `summary.json` records the mass balance
and whether the full motion completed. Use the mass inside the bed to
assess delivery.
