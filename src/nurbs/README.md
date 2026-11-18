# NURBS and B-Spline Geometry: Basis Evaluation, Refinement, and Fitting

`src/nurbs` provides the geometry, basis, refinement, fitting, projection, and
optimization tools used by GeoTaichi's IGA and IGA-MPM modules. It can also be
used independently for curve, surface, and volume processing.

## Capabilities

- B-spline and rational NURBS curves, surfaces, and volumes.
- Basis values, first derivatives, and second derivatives in one to three
  parametric dimensions.
- Point evaluation, projection, inversion, distance, tangents, normals, and
  curvature-related derivatives.
- Knot insertion, refinement, and primitive operations.
- Primitive rectangles, rings, cubes, tubes, and cylinders.
- Global interpolation and least-squares/Gauss-Newton fitting.
- Point-cloud parameterization, boundary extraction, PCA, and multi-surface
  fitting helpers.
- ASAP and MIPS surface energies and control-point/weight optimization.

## B-spline and NURBS basis theory

For degree $`p`$ and nondecreasing knot vector
$`\{u_0,\ldots,u_{n+p+1}\}`$, the degree-zero B-spline basis is

```math
N_{i,0}(u)=
\begin{cases}
1,&u_i\leq u<u_{i+1},\\[0pt]
0,&\text{otherwise},
\end{cases}
```

with the final knot assigned to the final active span. The Cox--de Boor
recursion is

```math
N_{i,p}(u)
=\frac{u-u_i}{u_{i+p}-u_i}N_{i,p-1}(u)
+\frac{u_{i+p+1}-u}{u_{i+p+1}-u_{i+1}}N_{i+1,p-1}(u),
```

where a fraction with zero denominator contributes zero. Its first derivative
is

```math
N'_{i,p}(u)
=\frac{p}{u_{i+p}-u_i}N_{i,p-1}(u)
-\frac{p}{u_{i+p+1}-u_{i+1}}N_{i+1,p-1}(u).
```

Tensor-product polynomial bases are products of the univariate functions. For
a surface,

```math
B_{ij}(u,v)=N_{i,p}(u)M_{j,q}(v),
```

and a volume adds $`L_{k,r}(w)`$. Given positive weights $`w_A`$, define

```math
W(\boldsymbol{\xi})=\sum_Aw_AB_A(\boldsymbol{\xi}),
\qquad
R_A(\boldsymbol{\xi})
=\frac{w_AB_A(\boldsymbol{\xi})}{W(\boldsymbol{\xi})}.
```

The rational functions form a nonnegative partition of unity:

```math
R_A\geq0,
\qquad
\sum_AR_A=1.
```

Their first parametric derivatives follow directly from the quotient rule,

```math
R_{A,\alpha}
=w_A\frac{B_{A,\alpha}W-B_AW_{,\alpha}}{W^2},
\qquad
W_{,\alpha}=\sum_Bw_BB_{B,\alpha}.
```

Second derivatives use the same quotient rule:

```math
R_{A,\alpha\beta}
=\frac{w_AB_{A,\alpha\beta}}{W}
-\frac{w_A(B_{A,\alpha}W_{,\beta}
+B_{A,\beta}W_{,\alpha}+B_AW_{,\alpha\beta})}{W^2}
+\frac{2w_AB_AW_{,\alpha}W_{,\beta}}{W^3}.
```

## Rational geometry and differential quantities

Moving contact surfaces accumulate the homogeneous numerator and denominator
and their derivatives directly over the active support. The quotient rule then
returns position, both tangents, and the three second derivatives. This avoids
support-sized rational derivative matrices inside the closest-point iteration,
reducing Taichi compilation work without changing the NURBS geometry.
`test_moving_surface_homogeneous_derivatives_match_rational_basis` compares all
six results against the independent rational basis evaluation, including
nonuniform weights, endpoints, and degree `(3, 1)` wavy surfaces.

For control points $`\boldsymbol{P}_A`$, a NURBS curve, surface, or volume is

```math
\boldsymbol{x}(\boldsymbol{\xi})
=\sum_AR_A(\boldsymbol{\xi})\boldsymbol{P}_A.
```

Therefore

```math
\boldsymbol{x}_{,\alpha}
=\sum_AR_{A,\alpha}\boldsymbol{P}_A,
\qquad
\boldsymbol{x}_{,\alpha\beta}
=\sum_AR_{A,\alpha\beta}\boldsymbol{P}_A.
```

For a surface, the unit normal and area measure are

```math
\boldsymbol{n}
=\frac{\boldsymbol{x}_{,u}\times\boldsymbol{x}_{,v}}
{\|\boldsymbol{x}_{,u}\times\boldsymbol{x}_{,v}\|},
\qquad
\mathrm dA
=\|\boldsymbol{x}_{,u}\times\boldsymbol{x}_{,v}\|\,\mathrm du\,\mathrm dv.
```

For a volume, with
$`\boldsymbol{J}_{\xi}=
[\boldsymbol{x}_{,u}\ \boldsymbol{x}_{,v}\ \boldsymbol{x}_{,w}]`$,

```math
\mathrm dV=|\det\boldsymbol{J}_{\xi}|\,\mathrm du\,\mathrm dv\,\mathrm dw.
```

These derivatives also define tangents, metric tensors, curvature inputs, and
the Jacobians used by isogeometric analysis.

## Point inversion and closest projection

For a query point $`\boldsymbol{q}`$, closest projection minimizes

```math
E(\boldsymbol{\xi})
=\frac12\|\boldsymbol{x}(\boldsymbol{\xi})-\boldsymbol{q}\|^2.
```

The stationarity equations and exact parameter-space Hessian are

```math
g_\alpha
=(\boldsymbol{x}-\boldsymbol{q})\cdot\boldsymbol{x}_{,\alpha}=0,
```

```math
H_{\alpha\beta}
=\boldsymbol{x}_{,\alpha}\cdot\boldsymbol{x}_{,\beta}
+(\boldsymbol{x}-\boldsymbol{q})\cdot
\boldsymbol{x}_{,\alpha\beta}.
```

A projected Newton step solves

```math
\boldsymbol{H}\Delta\boldsymbol{\xi}=-\boldsymbol{g}
```

and clamps or projects the trial coordinates to the admissible knot domain.
Multi-start projection is needed when a surface has several local closest
points.

## Knot insertion, continuity, and fitting

Inserting a knot changes the basis and control polygon but preserves the
geometry:

```math
\sum_iN_{i,p}(u)\boldsymbol{P}_i
=\sum_i\widetilde N_{i,p}(u)\widetilde{\boldsymbol{P}}_i.
```

At a knot of multiplicity $`m`$, a degree-$`p`$ polynomial B-spline is generally
$`C^{p-m}`$ continuous. Refinement increases resolution without changing the
represented shape, provided the corresponding control-point transformation is
applied.

For samples $`\boldsymbol{q}_s`$ at prescribed parameters
$`\boldsymbol{\xi}_s`$, weighted least-squares fitting minimizes

```math
E(\boldsymbol{P})
=\frac12\sum_sw_s
\left\|\sum_AR_A(\boldsymbol{\xi}_s)\boldsymbol{P}_A
-\boldsymbol{q}_s\right\|^2.
```

With design matrix $`\boldsymbol{N}_{sA}=R_A(\boldsymbol{\xi}_s)`$, the normal
equations are

```math
(\boldsymbol{N}^T\boldsymbol{W}\boldsymbol{N})\boldsymbol{P}
=\boldsymbol{N}^T\boldsymbol{W}\boldsymbol{Q}.
```

When the sample parameters are also unknown, projection and control-point
updates are alternated or combined in a Gauss--Newton iteration.

## Surface distortion energies

Let the singular values of a two-dimensional surface deformation gradient be
$`\sigma_1,\sigma_2>0`$. The as-similar-as-possible energy is

```math
\Psi_{ASAP}
=\sigma_1^2+\sigma_2^2-2(\sigma_1+\sigma_2)+2
=(\sigma_1-1)^2+(\sigma_2-1)^2.
```

The MIPS distortion energy is

```math
\Psi_{MIPS}
=\frac{\sigma_1^2+\sigma_2^2}{\sigma_1\sigma_2}.
```

For control variables $`\boldsymbol{z}`$ and deformation-gradient Jacobian
$`\boldsymbol{J}_F=\partial\mathrm{vec}\boldsymbol{F}/
\partial\boldsymbol{z}`$, the chain rule gives

```math
\nabla_{\boldsymbol{z}}\Psi
=\boldsymbol{J}_F^T\nabla_{\boldsymbol{F}}\Psi,
\qquad
\nabla_{\boldsymbol{z}}^2\Psi
=\boldsymbol{J}_F^T
\nabla_{\boldsymbol{F}}^2\Psi\,
\boldsymbol{J}_F
```

when $`\boldsymbol{F}`$ is affine in $`\boldsymbol{z}`$. Positive-semidefinite
spectral projection may replace the raw Hessian in a descent solve.

## Package layout

| Path | Responsibility |
| --- | --- |
| `Spline.py`, `SplineCurve.py`, `SplineSurface.py`, `SplineVolume.py` | Common curve/surface/volume state and validation |
| `BSplineCurve.py`, `BSplineSurface.py`, `BSplineVolume.py` | Non-rational B-spline implementations |
| `NurbsCurve.py`, `NurbsSurface.py`, `NurbsVolume.py` | Rational curve/surface/volume implementations |
| `Nurbs.py`, `NurbsBasis.py` | Span lookup and basis/derivative algorithms |
| `PrimitiveSurface.py`, `PrimitiveVolume.py`, `Rectangle.py`, `Ring.py`, `Cube.py`, `Tube.py`, `Cylinder.py` | Procedural NURBS primitives |
| `Operations.py` | Knot insertion, refinement, and geometry operations |
| `CurvePoints.py`, `SurfacePoints.py`, `Utilities.py` | Curve, surface, and point-cloud fitting workflows |
| `Energy.py`, `ASAP.py`, `MIPS.py`, `Optimization.py` | Geometry energies and optimization drivers |
| `element/` | IGA fitting quadrature, orientation constraints, and linear-element geometry helpers |
| `cnurbs/`, `core/` | Specialized and experimental geometry kernels |

## Primitive example

```python
import numpy as np
from src.nurbs.Cube import Cube

volume = Cube()
volume.set_parameters(start_point=[0.0, 0.0, 0.0], size=[2.0, 1.0, 0.5])
volume.generate_knot_u(degree=2, num_ctrlpts=7)
volume.generate_knot_v(degree=2, num_ctrlpts=5)
volume.generate_knot_w(degree=2, num_ctrlpts=3)
volume.generate_ctrlpts()
volume.generate_weights()

point = volume.single_point(0.5, 0.5, 0.5)
du, dv, dw = volume.derivative(0.5, 0.5, 0.5)
volume.refine_knot(density=[1, 1, 0])
```

For solver scenes, import the lazily exported primitives from `src.iga` and
append them to an IGA `Primitives` collection.

## Fitting

`Utilities.py` contains both direct interpolation and iterative approximation
tools. Choose the algorithm according to whether control-point count, knot
vectors, boundary points, and derivatives are prescribed. Fitting functions
use NumPy/SciPy host linear algebra because they are geometry preprocessing,
not a simulation stepping backend.

Point inversion and projection can be expensive on large point clouds. Cache
parametric coordinates when the geometry does not change.

### Four-corner planar fitting

`fit_surface_from_boundary` accepts one ordered, non-self-intersecting 2D
boundary loop. The last point may repeat the first. Supply four corner
coordinates with `corner_points`; each is snapped to the nearest boundary
sample. The first corner defines `(u, v) = (0, 0)`, and the remaining corners
are ordered counterclockwise along the loop. Clockwise input loops are
reoriented automatically. Four distinct corner samples are required.

```python
import numpy as np
from src.nurbs.Utilities import fit_surface_from_boundary, extrude_surface

boundary_points = np.array([
    [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 0.5],
    [2.0, 1.0], [1.0, 1.2], [0.0, 1.0], [0.0, 0.5],
])
corners = boundary_points[[0, 2, 4, 6]]
surface = fit_surface_from_boundary(
    boundary_points, corner_points=corners,
    degree=(2, 2), num_ctrlpts=(4, 4),
)
volume = extrude_surface(surface, vector=[0.0, 0.0, 0.2], degree=2)
```

Omitting `corner_points` opens a Matplotlib window for four corner picks.
Left click adds a point, right click undoes a pick, and middle click finishes.
Fewer than four picks cancel the operation. An interactive Matplotlib backend
is required; headless scripts should supply coordinates explicitly.
`get_boundary_categories` also accepts `corner_points`, while retaining its
existing `corner_index` argument.

Each edge is fitted by constrained least squares with fixed endpoints and
chord-length parameters. Opposite edges share degree and knots, and a Coons
blend fills the interior control net. The result is a `NurbsSurface` with
unit weights. This constructor rejects folded control cells or invalid
sampled spline Jacobians; it does not repair them or change the boundary
topology. Strongly concave domains may require another corner partition,
different resolution, or multiple patches.

### Orientation and iterative fitting

The three planar iterative fitters use `element/NurbsSurfaceElement.py` for
MIPS energy, gradients and Hessians over actual knot spans. Geometry
derivatives come from the spline's polynomial or rational basis, rather than
a bilinear approximation of its control points.

Along a control-point update, each sampled Jacobian is affine in the step
length, so its determinant is quadratic. The first positive zero limits the
step. This check includes both the real spline Jacobians and all four corner
orientations of every control quadrilateral. A step cannot pass through a
zero determinant even if its final determinant would be positive. Rejected
steps and projection failures restore the accepted control points and boundary
curves. The final spline is checked again before returning.

`validate_spline_jacobian(primitive)` returns the minimum sampled signed
Jacobian, raising `ValueError` for a nonpositive or nonfinite value. It supports
planar 2D patches and 3D volumes. Surface validation also requires an unfolded
control net. Each knot span is checked at Gauss points and a grid including
one-sided span endpoints. These finite samples are an IGA geometry check,
not a proof of positivity everywhere between samples.

### Extruded volumes

`extrude_surface(surface, vector, degree=1)` constructs the exact straight
sweep `V(u, v, w) = S(u, v) + w * vector`. A 2D source is embedded at `z=0`.
The source knots and rational weights are preserved, with `degree + 1`
control layers in the extrusion direction. The output is a `NurbsVolume`
that can be appended to a 3D IGA `Primitives` collection.

The vector must give positive signed volume Jacobians; in-plane, zero or
oppositely oriented extrusions raise `ValueError`. Extrusion validates the
volume before returning and leaves the source unchanged. Arbitrary fitting
of a solid from six boundary faces is not implemented by this workflow.

## Conventions and compatibility

- Set degree and knot vectors before assigning rational weights.
- Control-point ordering is part of the primitive contract and must match the
  corresponding `num_ctrlpts_u/v/w` values.
- Several historical public names, including `paramaterize`, retain their
  original spelling for compatibility.
- Exact circular primitives depend on rational weights; replacing all weights
  by one changes the geometry into a polynomial B-spline approximation.

## Tests

NURBS basis, refinement, geometry, and fitting tests are grouped under
`tests/unit/iga/geometry/` because IGA is the main production consumer.
