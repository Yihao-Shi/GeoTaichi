# Broad-Phase and Continuous Collision Detection for Computational Mechanics

`src/contact_detection` contains reusable broad-phase and continuous
contact-detection primitives. It is a low-level numerical package consumed by
DEM, FEM, MPM, affine-body, and coupling solvers; it is not a standalone
simulation facade.

## Package layout

| Path | Responsibility |
| --- | --- |
| `bounding_volume_hierarchy/` | AABB storage, Morton sorting, LBVH construction/refit, traversal helpers, and collision queries |
| `continuous_contact_detection/CCD.py` | Zero-thickness point/triangle, edge/edge, point/point, and linear-gap CCD primitives |
| `continuous_contact_detection/AdditiveCCD.py` | Finite-clearance point/triangle, edge/edge, point/point, linear-gap, and point/NURBS ACCD primitives |
| `continuous_contact_detection/PolynomialCCD.py` | Polynomial point/point and material-determinant feasibility queries |
| `continuous_contact_detection/THEORY.md` | Detailed CCD and ACCD mathematical derivation and robustness assumptions |

## Broad-phase geometry and linear BVH theory

For a primitive with vertices $`\boldsymbol{x}_a`$ and padding $`r\geq0`$, its
axis-aligned bounding box (AABB) is

```math
b_k^- = \min_a x_{a,k}-r,
\qquad
b_k^+ = \max_a x_{a,k}+r.
```

Two boxes $`A`$ and $`B`$, with optional expansions $`r_A`$ and $`r_B`$, overlap if
and only if their intervals overlap in every coordinate:

```math
b_{A,k}^- -r_A\leq b_{B,k}^+ +r_B,
\qquad
b_{A,k}^+ +r_A\geq b_{B,k}^- -r_B
\quad\text{for every }k.
```

For linear motion
$`\boldsymbol{x}_a(t)=\boldsymbol{x}_a^0+t\Delta\boldsymbol{x}_a`$ on
$`0\leq t\leq1`$, a conservative swept box takes componentwise extrema over
both endpoints:

```math
b_k^- =\min_a\min(x_{a,k}^0,x_{a,k}^0+\Delta x_{a,k})-r,
\qquad
b_k^+ =\max_a\max(x_{a,k}^0,x_{a,k}^0+\Delta x_{a,k})+r.
```

For an oriented box with local half extent $`\boldsymbol{e}`$, center
$`\boldsymbol{c}`$, and rotation $`\boldsymbol{R}`$, the exact world-space AABB
half extent is

```math
\boldsymbol{e}_{AABB}=|\boldsymbol{R}|\boldsymbol{e},
\qquad
\boldsymbol{b}^{\pm}=\boldsymbol{c}\pm\boldsymbol{e}_{AABB},
```

where the absolute value is componentwise. Linear BVH construction maps each
box center into normalized domain coordinates, quantizes those coordinates,
and interleaves their bits into a Morton key. Sorting the keys converts spatial
proximity into one-dimensional order. If $`z_i`$ and $`z_j`$ are two keys, their
radix-tree affinity is the longest common prefix

```math
\delta(i,j)=\mathrm{clz}(z_i\mathbin{\mathtt{xor}}z_j),
```

with an index tie-break for coincident keys. Internal nodes cover contiguous
key ranges; their boxes are componentwise unions of their descendants.

## Continuous collision-detection theory

All vertices follow the normalized linear trajectory

```math
\boldsymbol{x}_a(t)=\boldsymbol{x}_a^0+t\Delta\boldsymbol{x}_a,
\qquad 0\leq t\leq1.
```

For two points, with
$`\boldsymbol{r}_0=\boldsymbol{x}_0^0-\boldsymbol{x}_1^0`$ and
$`\Delta\boldsymbol{r}=\Delta\boldsymbol{x}_0-\Delta\boldsymbol{x}_1`$,
zero-thickness impact candidates are roots of

```math
\|\boldsymbol{r}_0+t\Delta\boldsymbol{r}\|^2
=a t^2+b t+c=0,
```

```math
a=\Delta\boldsymbol{r}\cdot\Delta\boldsymbol{r},
\qquad
b=2\boldsymbol{r}_0\cdot\Delta\boldsymbol{r},
\qquad
c=\boldsymbol{r}_0\cdot\boldsymbol{r}_0.
```

For point--triangle and edge--edge CCD, coplanarity is the scalar triple
product

```math
q(t)=\boldsymbol{o}(t)\cdot
\left[\boldsymbol{a}(t)\times\boldsymbol{b}(t)\right]=0.
```

Because each vector is affine in $`t`$, $`q`$ is cubic. Every real root in
$`[0,1]`$ must still satisfy the primitive-domain condition: the closest point
must lie in the triangle or both closest coordinates must lie on their line
segments. An identically zero coplanarity polynomial contains no isolated
impact time and is handled by distance-based conservative advancement.

For a linearly changing signed gap,

```math
g(t)=g_0+t\Delta g,
```

the first closing root is $`t_*=-g_0/\Delta g`$ when $`g_0>0`$ and
$`\Delta g<0`$.

### Additive CCD with finite clearance

Let $`d(t)`$ be exact primitive distance, $`d_{min}\geq0`$ the required
clearance, and $`L`$ a Lipschitz bound on relative motion. The cancellation-safe
excess distance is

```math
e(t)=d(t)-d_{min}
=\frac{d(t)^2-d_{min}^2}{d(t)+d_{min}}.
```

With retained-gap fraction $`0\leq\eta<1`$, conservative advancement uses

```math
\Delta t=\frac{(1-\eta)e(t)}{L}
```

and recomputes the exact closest distance after every increment. Useful
relative-motion bounds are

```math
L_{PP}=\|\Delta\boldsymbol{x}_0-\Delta\boldsymbol{x}_1\|,
```

```math
L_{PT}=\|\Delta\boldsymbol{p}\|
+\max_{a=0,1,2}\|\Delta\boldsymbol{t}_a\|,
```

```math
L_{EE}=\max_{a=0,1}\|\Delta\boldsymbol{a}_a\|
+\max_{b=0,1}\|\Delta\boldsymbol{b}_b\|,
```

after subtracting any common translational motion from all velocities in a
stencil.

For a positive-weight NURBS surface,
$`\boldsymbol{x}_s=\sum_iR_i\boldsymbol{P}_i`$ with $`R_i\geq0`$ and
$`\sum_iR_i=1`$, convexity gives

```math
\left\|\Delta\boldsymbol{p}
-\sum_iR_i\Delta\boldsymbol{P}_i\right\|
\leq
\max_i\|\Delta\boldsymbol{p}-\Delta\boldsymbol{P}_i\|.
```

This remains conservative when the closest surface parameter changes during
the trial motion.

## Continuous collision detection

The public CCD functions are exported from
`continuous_contact_detection.__init__`:

- `point_point_ccd`
- `point_triangle_ccd`
- `edge_edge_ccd`
- `linear_gap_ccd`
- `point_point_accd`
- `point_triangle_accd`
- `edge_edge_accd`
- `linear_gap_accd`
- `point_nurbs_accd_increment`
- `point_point_quadratic_ccd`
- `deformation_gradient_ccd`
- `ccd_mode_parameters`

Except for the host-side mode normalizer, these are `@ti.func` building
blocks and must be called from a Taichi kernel or another Taichi function.
They operate on one stencil and do not allocate global candidate lists or
perform reductions.

```python
import taichi as ti
from src.contact_detection.continuous_contact_detection import point_point_ccd

toi = ti.field(dtype=ti.f64, shape=())

@ti.kernel
def compute_toi():
    p0 = ti.Vector([0.0, 0.0, 0.0])
    p1 = ti.Vector([1.0, 0.0, 0.0])
    dp0 = ti.Vector([0.75, 0.0, 0.0])
    dp1 = ti.Vector([0.0, 0.0, 0.0])
    toi[None] = point_point_ccd(p0, p1, dp0, dp1, 0.1, 50)
```

Ordinary CCD uses zero clearance. Its analytic label means polynomial
candidate roots plus primitive-domain validation, not an interval-arithmetic
rounding certificate. ACCD takes an explicit positive additive clearance and
returns zero when the initial stencil is already infeasible.
Callers own broad-phase candidates, the global minimum-TOI reduction, and the
accepted-step policy.

The point--NURBS ACCD bound uses the convex-hull property of positive NURBS
weights: `max_i ||dp-dP_i||` bounds the simultaneous relative motion of the
point and surface. It is used iteratively by IGA--MPM before the contact-aware
line search. IGA and MPM material
``ccd()`` methods are a different feasibility guard on the deformation
gradient determinant; they are not contact ACCD.

## BVH infrastructure

The BVH package provides AABBs, Morton codes, linear BVH construction, refit,
and specialized collision traversal. Base classes such as `Bvh` and `LBvh`
are infrastructure classes: concrete solvers usually provide primitive AABB
access and query kernels rather than instantiating the base class directly.

Several solver modules also contain specialized dynamic BVHs optimized for
their contact stencil and capacity model. Reuse those adapters when possible
instead of duplicating traversal logic.

FEM wraps its selected dynamic linked-cell or refit-BVH backend in
`FEMCollisionCulling`. The backend returns conservative AABB overlaps; the
wrapper applies stitch exclusions, exact current-distance pruning or swept
conservative CCD, prefix-sum compaction, and minimum-step reduction. This
separation keeps `broad_phase` as a spatial-backend choice rather than using
the raw AABB list as the assembled contact set.

## Numerical conventions

- CCD returns a normalized step fraction in `[0, 1]`, or the documented
  no-impact sentinel for polynomial determinant queries.
- `ccd` uses zero collision thickness.
- `accd` uses the supplied additive thickness; `ccd_mode_parameters()` caps
  the public retained-gap fraction `eta` at 0.1.
- Point-triangle and edge-edge closest-point calculations reuse the shared IPC
  geometry functions in `src/physics_model/contact_model/ipc`.
- See `continuous_contact_detection/THEORY.md` for the analytic-polynomial and
  iterative-distance derivations and their robustness boundaries.

## Tests

CCD and LBVH tests are under `tests/unit/contact_detection/`. Higher-level
contact integration tests live with the FEM, DEM, MPM, and coupling modules.
