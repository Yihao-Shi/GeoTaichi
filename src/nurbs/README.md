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
| `SplinePrimitives.py` | Common curve/surface/volume state and validation |
| `BSplinePrimitives.py` | Non-rational B-spline implementations |
| `NurbsPrimitives.py` | Rational curve/surface/volume implementations |
| `Nurbs.py`, `NurbsBasis.py` | Span lookup and basis/derivative algorithms |
| `BasicSurface.py`, `BasicVolume.py` | Procedural NURBS primitives |
| `Operations.py` | Knot insertion, refinement, and geometry operations |
| `Fitting.py` | Curve, surface, and point-cloud fitting workflows |
| `Energy.py`, `Optimization.py` | Geometry energies and optimization drivers |
| `cnurbs/`, `core/`, `element/` | Specialized and experimental geometry kernels |

## Primitive example

```python
import numpy as np
from src.nurbs.BasicVolume import Cube

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

`Fitting.py` contains both direct interpolation and iterative approximation
tools. Choose the algorithm according to whether control-point count, knot
vectors, boundary points, and derivatives are prescribed. Fitting functions
use NumPy/SciPy host linear algebra because they are geometry preprocessing,
not a simulation stepping backend.

Point inversion and projection can be expensive on large point clouds. Cache
parametric coordinates when the geometry does not change.

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
