# CCD and ACCD Theory-to-Code Map

This note is the contract for the Taichi continuous-contact primitives. It
separates exact zero-thickness CCD from iterative additive CCD and maps each
mathematical object to the production function that implements it.

## Motion and return convention

Every primitive vertex follows a linear trial trajectory

```text
x(t) = x0 + t dx,    0 <= t <= 1.
```

Public functions return a safe normalized step in `[0, 1]`. `1` means no
impact was found on the trial interval. An initially infeasible stencil returns
`0`. `gap_fraction = eta` retains `eta` of the initial excess separation.

## Analytic zero-thickness CCD

| Primitive | Equation | Domain check | Code |
| --- | --- | --- | --- |
| point--point | `||r0 + t dr||^2 = 0` (quadratic, normally a double root) | root in `[0,1]` | `CCD.point_point_ccd` |
| point--triangle | `(p-a) dot ((b-a) cross (c-a)) = 0` (cubic) | closest point lies on the triangle at the root | `CCD.point_triangle_ccd` |
| edge--edge | `(b0-a0) dot ((a1-a0) cross (b1-b0)) = 0` (cubic) | closest points lie on both segments at the root | `CCD.edge_edge_ccd` |
| point--plane gap | `g0 + t dg = 0` (linear) | closing signed gap | `CCD.linear_gap_ccd` |

`PolynomialCCD.real_cubic_roots` returns all real roots because the smallest
algebraic coplanarity root need not satisfy the primitive domain check. If a PT
or EE coplanarity polynomial vanishes identically, there is no isolated cubic
event. The implementation uses the zero-clearance ACCD distance fallback for
that all-coplanar trajectory.

## Iterative additive CCD

Let `d(t)` be the exact primitive distance, `d_min >= 0` the requested
clearance, and `L` a bound on relative motion. The cancellation-safe excess
distance is

```text
e(t) = d(t) - d_min
     = (d(t)^2 - d_min^2) / (d(t) + d_min).
```

For the current iterate, ACCD advances by the certified lower bound

```text
delta_t = (1 - eta) e(t) / L.
```

It repeatedly updates all moving vertices, recomputes the exact distance, and
stops before the excess distance drops below `eta * e(0)`.

| Primitive | Relative-motion bound `L` | Code |
| --- | --- | --- |
| point--point | `||dp0-dp1||` | `AdditiveCCD.point_point_accd` |
| point--triangle | after common-motion removal, `||dp|| + max_i ||dt_i||` | `AdditiveCCD.point_triangle_accd` |
| edge--edge | after common-motion removal, `max_i ||da_i|| + max_j ||db_j||` | `AdditiveCCD.edge_edge_accd` |
| point--NURBS | `max_i ||dp-dP_i||` for fixed positive weights | `AdditiveCCD.point_nurbs_accd_increment` inside the per-pair IGAMPM Taichi loop |

The positive-weight NURBS bound follows from

```text
dp - sum_i R_i dP_i = sum_i R_i (dp - dP_i),
R_i >= 0, sum_i R_i = 1.
```

It accounts for simultaneous point and surface motion, is invariant under a
common translation, and remains valid when the closest NURBS parameter changes.
IGAMPM evaluates the point and every control point at an arbitrary trial
fraction without mutating shared geometry:

```text
p(alpha)   = p0   + alpha dp,
P_i(alpha) = P_i0 + alpha dP_i.
```

IGAMPM compacts swept-BVH candidates before ACCD. Each Taichi launch advances
one iteration for unfinished candidate pairs and updates their TOCs on the
device. Python dispatches degree groups and reads the unfinished count between
launches. A final device minimum reduces the current group's candidate TOCs.
This split avoids prohibitively expensive Taichi compilation of nested loops.

## Swept broad phase

For a proposed fraction `alpha_max`, a primitive's box encloses every endpoint
`x_i` and `x_i + alpha_max * dx_i`. A point uses its own two endpoints. Expand
overlap tests by the requested clearance and a rounding guard. Overlap only
creates candidates; the actual CCD/ACCD still determines the safe fraction.
Endpoint boxes cover linear trajectories even if both endpoints are separated
and a collision occurs between them. Fixed positive NURBS weights make the
swept control hull conservative for the moving rational curve/surface.

- FEM, FEM--MPM and FEDEM use their swept BVH/linked-cell primitive builders.
- DEM affine bodies and the mesh differentiable IPC projector use swept
  neighbor candidates, followed by point--triangle and edge--edge CCD/ACCD.
- IGA--MPM queries swept surface and (3D) knot-span BVHs per particle sample.
- Direct MPM and soft--soft MPDEM first screen swept body boxes, then screen
  each candidate point pair's swept boxes before point--point CCD/ACCD.
- MPDEM soft--affine mesh contact uses the existing mixed swept BVH. Level-set
  bodies retain their swept body boxes and level-set distance advancement.
- Infinite planes retain exact linear-gap checks; material determinant bounds
  remain separate from geometric collision detection.

A general point--NURBS first-impact equation is not treated as one fixed
low-degree polynomial: the surface is rational and the minimizing parameter
can switch during the trajectory. The iterative distance query is therefore
the production path, not an analytic-root fallback.

## Robustness boundaries

- Additive CCD uses floating-point distances and a deliberate safety gap. It
  is not rounding-error certified in the sense of Tight Inclusion CCD.
- "Analytic CCD" here means algebraic candidate generation followed by the
  primitive-domain test. It is not an interval-arithmetic certificate.
- Analytic CCD is zero-thickness. Finite `d_min` belongs to ACCD.
- Broad-phase swept AABBs, candidate culling, and the global minimum reduction
  are caller responsibilities.
- `deformation_gradient_ccd` is a material-inversion feasibility polynomial,
  not a geometric contact query.

## Verification properties

`tests/unit/contact_detection/test_continuous_contact_detection.py` checks all
real cubic roots, analytic TOIs, coplanar fallback, initial intersection,
finite clearance, common-translation invariance, and point--NURBS relative
motion. The shared IPC NURBS tests compare virtual moving curve/surface queries
against explicitly moved control meshes. Higher-level tests verify the device
reductions in FEM, affine bodies, soft particles, MPM, and IGAMPM.
