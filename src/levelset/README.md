# Level-Set Transport, Redistancing, and Interface Geometry

`src/levelset` implements structured-grid signed-distance and interface
advection tools. The package is shared by fluid MPM, level-set rigid bodies,
and deforming soft-particle workflows.

## Capabilities

- Two- and three-dimensional signed-distance fields on regular grids.
- Fast marching and fast sweeping redistancing.
- WENO/ENO spatial reconstruction and Hamilton-Jacobi fluxes.
- Semi-Lagrangian advection.
- High-order interpolation and finite-difference derivatives.
- Fluid free-surface sampling, normals, curvature, and smoothed Heaviside
  functions.
- Periodic, extrapolated, and constant boundary treatments.

## Level-set transport and interface theory

A level-set field $`\phi(\boldsymbol{x},t)`$ represents the interface and its
two phases by

```math
\Gamma(t)=\{\boldsymbol{x}:\phi(\boldsymbol{x},t)=0\},
\qquad
\phi<0\ \text{inside},
\qquad
\phi>0\ \text{outside}.
```

For a signed-distance field,

```math
|\nabla\phi|=1,
\qquad
\boldsymbol{n}=\frac{\nabla\phi}{|\nabla\phi|},
\qquad
\kappa=\nabla\cdot\boldsymbol{n}.
```

Advection by velocity $`\boldsymbol{u}`$ obeys the Hamilton--Jacobi equation

```math
\frac{\partial\phi}{\partial t}
+\boldsymbol{u}\cdot\nabla\phi=0.
```

In semi-Lagrangian form, the value at grid point $`\boldsymbol{x}_i`$ is sampled
at the departure point

```math
\phi_i^{n+1}=\mathcal I[\phi^n](\boldsymbol{x}_d),
\qquad
\boldsymbol{x}_d=\boldsymbol{x}_i
-\Delta t\,\boldsymbol{u}
\left(\boldsymbol{x}_i-\frac{\Delta t}{2}
\boldsymbol{u}(\boldsymbol{x}_i)\right)
```

for midpoint backtracing. Higher Runge--Kutta backtraces replace the departure
point while retaining the same interpolation statement. The finite-difference
path reconstructs one-sided derivatives $`D_k^-\phi`$ or $`D_k^+\phi`$ with
ENO/WENO and chooses the upwind side:

```math
(\boldsymbol{u}\cdot\nabla\phi)_i
\approx\sum_k u_{i,k}
\begin{cases}
D_k^-\phi_i,&u_{i,k}>0,\\[0pt]
D_k^+\phi_i,&u_{i,k}<0.
\end{cases}
```

## Redistancing and Eikonal update

Redistancing keeps the zero contour fixed while solving in pseudo-time
$`\tau`$

```math
\frac{\partial\phi}{\partial\tau}
=S_\varepsilon(\phi_0)(1-|\nabla\phi|),
\qquad
S_\varepsilon(\phi_0)
=\frac{\phi_0}{\sqrt{\phi_0^2+h^2|\nabla\phi_0|^2}}.
```

Fast marching and fast sweeping instead solve the static Eikonal problem

```math
|\nabla d|=1,
\qquad d|_\Gamma=0.
```

Let $`a_1\leq\cdots\leq a_m`$ be the smallest accepted neighbor distances in
the active coordinate directions on an isotropic grid of spacing $`h`$. The
monotone local update is the admissible root $`d\geq a_m`$ of

```math
\sum_{k=1}^{m}(d-a_k)^2=h^2.
```

For two contributing directions this gives

```math
d=\frac{a_1+a_2+\sqrt{2h^2-(a_2-a_1)^2}}{2},
```

when $`a_2-a_1<h`$; otherwise $`d=a_1+h`$. The three-dimensional update adds
neighbors in ascending order until the corresponding quadratic root satisfies
the causality condition.

When an edge joins samples $`\phi_i`$ and $`\phi_j`$ of opposite sign, linear
interface localization gives

```math
\theta=\frac{\phi_i}{\phi_i-\phi_j},
\qquad
d_i=\mathrm{sign}(\phi_i)\,\theta h.
```

## Free-surface sampling

Cell-centered fields use multilinear interpolation. With local coordinates
$`\boldsymbol{\xi}\in[0,1]^d`$,

```math
\phi(\boldsymbol{x})
=\sum_{\boldsymbol{a}\in\{0,1\}^d}
\left[\prod_{k=1}^{d}
\xi_k^{a_k}(1-\xi_k)^{1-a_k}\right]
\phi_{\boldsymbol{i}+\boldsymbol{a}}.
```

Across a fluid sample $`\phi_f<0`$ and air sample $`\phi_a>0`$, the interface
fraction is

```math
\theta_f=\frac{\phi_f}{\phi_f-\phi_a},
\qquad 0<\theta_f\leq1.
```

The compact negative-phase Heaviside approximation is

```math
H_\varepsilon(\phi)=
\begin{cases}
1,&\phi\leq-\varepsilon,\\[0pt]
\dfrac12-\dfrac34\dfrac{\phi}{\varepsilon}
+\dfrac14\left(\dfrac{\phi}{\varepsilon}\right)^3,
&|\phi|<\varepsilon,\\[0pt]
0,&\phi\geq\varepsilon.
\end{cases}
```

Central differences evaluate the geometric normal and curvature:

```math
(\nabla_h\phi)_{i,k}
=\frac{\phi_{i+\boldsymbol{e}_k}-
\phi_{i-\boldsymbol{e}_k}}{2h_k},
\qquad
\kappa_i\approx
\sum_k\frac{n_{i+\boldsymbol{e}_k,k}-
n_{i-\boldsymbol{e}_k,k}}{2h_k}.
```

## Main classes and files

| Path | Responsibility |
| --- | --- |
| `LevelSet.py` | Base field layout, interface initialization, Eikonal update, and smoothing |
| `FastMarchingLevelSet.py` | Priority-queue fast marching redistance |
| `FastSweepingLevelSet.py` | Directional fast sweeping redistance |
| `WENO.py`, `WENOFlux.py` | ENO/WENO interpolation and Hamilton-Jacobi operators |
| `SemiLagrangian.py` | Characteristic backtracing and advection |
| `FluidLevelSetKernel.py` | Device free-surface, normal, curvature, and sampling helpers |
| `BoundaryConditions.py` | Ghost-cell and stencil boundary policies |

## Redistancing example

```python
import numpy as np
import taichi as ti
from src.levelset.FastSweepingLevelSet import FastSweepingLevelSet

ti.init(arch=ti.cpu, default_fp=ti.f64)

resolution = (128, 128)
level_set = FastSweepingLevelSet(
    dimension=2,
    grid_size=1.0 / resolution[0],
    resolution=resolution,
    iteration_num=2,
    ghost_cell=3,
)

# Fill `level_set.distance_field` with an initial implicit field, then:
level_set.redistance()
phi = level_set.distance_field.to_numpy()
```

The exact field-upload method depends on whether the caller owns a dense
field, an MPM grid, or a level-set body. In production solvers the fields are
normally shared directly and are not downloaded between advection and
redistancing steps.

## Runtime conventions

The base grid layout, Eikonal propagation, WENO fluxes, normals, and curvature
are implemented as Taichi kernels/functions. NumPy routines in this package
serve coefficient construction, standalone interpolation utilities, and test
oracles. Callers must provide enough ghost cells for the selected high-order
stencil and must respect each function's sign convention.

## Tests

WENO accuracy and boundary behavior are tested under `tests/unit/levelset/`.
Level-set transport, redistancing, and volume safeguards are additionally
covered by DEM and MPM tests.
