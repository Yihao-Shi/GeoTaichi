# Isogeometric Analysis (IGA): NURBS Solids, Dynamics, and MPM Contact Coupling

`src/iga` implements explicit and implicit isogeometric analysis on NURBS
curves, surfaces, and volumes. The public `IGA` facade mirrors the FEM
workflow while retaining NURBS control points, knot vectors, weights, and
patch topology as the geometric representation.

[Theory log and derivations](#iga-theory-log) | [Examples](../../examples/)

## Example-backed capabilities

- NURBS-based isogeometric analysis (IGA): [2D elastic beam](../../examples/iga/elastic_beam2d/elastic_beam2d.py) and [3D explicit cantilever](../../examples/iga/explicit_cantilever_3d/explicit_cantilever_3d.py).
- Axisymmetric NURBS solids: [annulus](../../examples/iga/axisymmetric_annulus/axisymmetric_annulus.py).
- Independent IGA structures coupled to MPM continua: [implicit point–NURBS IPC](../../examples/igampm/iga_mpm_barrier_contact/iga_mpm_barrier_contact.py), [explicit DEM-law contact](../../examples/igampm/iga_mpm_explicit_dem_contact/iga_mpm_explicit_dem_contact.py), and [axisymmetric CPT](../../examples/igampm/cpt_dp/cpt_dp.py).

Implicit HashTriplet assembly caches reference basis values, gradients, physical
quadrature weights and radii until `precompute()` refreshes reference geometry.
Gauss contributions share one raw block per element/node pair. In IGA--MPM the
same blocks accumulate directly into permanent reduced slots; moving contact and
MPM mappings remain dynamic. Axisymmetric weights still include the full $`2\pi R`$.
Direct fixed-slot assembly skips absent off-diagonal slots before computing the
local Hessian block; diagonal blocks and every represented slot remain unchanged.

Multi-patch initialization uses separate element-span and knot-vector offsets;
each patch owns its reference-quadrature cache range.

## Package layout

| Path | Responsibility |
| --- | --- |
| `generator/` | NURBS primitives, patch collections, refinement, and preprocessing |
| `elements/` | Basis evaluation, quadrature, local operators, and interpolation |
| `engines/` | Explicit and implicit solvers, sparse assembly, state fields, and output |
| `boundaries/` | Dirichlet and Neumann control-point constraints |
| `mainIGA.py` | Public `IGA` facade |

The lower-level NURBS geometry implementation is shared with `src/nurbs`.
Patch geometry/state remains in `elements/Patch.py`; contact sampling and
surface ownership are isolated in `elements/ContactSurface.py`. This keeps
contact lifecycle out of the general NURBS patch container while preserving
the public element imports.

## IGA theory log

### 1. B-spline basis functions

Let

```math
\Xi=\{\xi_0,\xi_1,\ldots,\xi_{n+p+1}\},
\qquad
\xi_i\leq\xi_{i+1},
```

be a nondecreasing knot vector for degree $`p`$. The zeroth-degree B-spline is
$`N_{i,0}(\xi)=1`$ on $`[\xi_i,\xi_{i+1})`$ and zero elsewhere. Higher degrees
follow the Cox--de Boor recursion

```math
N_{i,p}(\xi)
=\frac{\xi-\xi_i}{\xi_{i+p}-\xi_i}N_{i,p-1}(\xi)
+\frac{\xi_{i+p+1}-\xi}{\xi_{i+p+1}-\xi_{i+1}}
N_{i+1,p-1}(\xi).
```

A fraction with a zero denominator contributes zero. The first derivative is

```math
\frac{\mathrm dN_{i,p}}{\mathrm d\xi}
=\frac{p}{\xi_{i+p}-\xi_i}N_{i,p-1}
-\frac{p}{\xi_{i+p+1}-\xi_{i+1}}N_{i+1,p-1}.
```

The basis has local support and forms a partition of unity:

```math
\mathrm{supp}N_{i,p}=[\xi_i,\xi_{i+p+1}),
\qquad
N_{i,p}\geq0,
\qquad
\sum_iN_{i,p}=1.
```

At an interior knot of multiplicity $`m`$, a degree-$`p`$ basis is
$`C^{p-m}`$ continuous. Thus knot multiplicity controls continuity without
changing the polynomial degree.

### 2. Tensor-product NURBS basis

For a three-dimensional parameter coordinate
$`\boldsymbol{\xi}=(\xi,\eta,\zeta)`$, first form the tensor-product B-spline

```math
B_A(\boldsymbol{\xi})
=N_{i,p}(\xi)M_{j,q}(\eta)L_{k,r}(\zeta),
\qquad A\leftrightarrow(i,j,k).
```

For positive control weight $`w_A`$, define the weight function and rational
basis

```math
W(\boldsymbol{\xi})=\sum_Bw_BB_B(\boldsymbol{\xi}),
\qquad
R_A(\boldsymbol{\xi})
=\frac{w_AB_A(\boldsymbol{\xi})}{W(\boldsymbol{\xi})}.
```

The two-dimensional surface basis follows by omitting the third factor. NURBS
basis functions retain non-negativity and partition of unity:

```math
R_A\geq0,
\qquad
\sum_AR_A=1,
\qquad
\sum_AR_{A,\alpha}=0.
```

Let $`Q_A=w_AB_A`$. Its first rational derivative is

```math
R_{A,\alpha}
=\frac{Q_{A,\alpha}-R_AW_{,\alpha}}{W},
\qquad
W_{,\alpha}=\sum_BQ_{B,\alpha}.
```

The second derivative can be evaluated without differentiating a quotient a
second time:

```math
R_{A,\alpha\beta}
=\frac{
Q_{A,\alpha\beta}
-R_{A,\alpha}W_{,\beta}
-R_{A,\beta}W_{,\alpha}
-R_AW_{,\alpha\beta}
}{W}.
```

Equal weights cancel from numerator and denominator, reducing NURBS to the
corresponding tensor-product B-spline basis.

### 3. Exact geometry and isogeometric trial space

Let $`\boldsymbol{P}_A^0`$ be reference control points and
$`\boldsymbol{p}_A(t)`$ current control points. Reference geometry, current
geometry, and displacement use the same rational basis:

```math
\boldsymbol{X}(\boldsymbol{\xi})
=\sum_AR_A(\boldsymbol{\xi})\boldsymbol{P}_A^0,
\qquad
\boldsymbol{x}(\boldsymbol{\xi},t)
=\sum_AR_A(\boldsymbol{\xi})\boldsymbol{p}_A(t).
```

```math
\boldsymbol{u}(\boldsymbol{\xi},t)
=\sum_AR_A(\boldsymbol{\xi})\boldsymbol{d}_A(t),
\qquad
\boldsymbol{d}_A=\boldsymbol{p}_A-\boldsymbol{P}_A^0.
```

The reference mapping Jacobian, material basis gradients, and deformation
gradient are

```math
\boldsymbol{J}_0
=\frac{\partial\boldsymbol{X}}{\partial\boldsymbol{\xi}}
=\sum_A\boldsymbol{P}_A^0\otimes\nabla_{\!\xi}R_A.
```

```math
\nabla_{\!X}R_A
=\boldsymbol{J}_0^{-T}\nabla_{\!\xi}R_A,
\qquad
\boldsymbol{F}
=\sum_A\boldsymbol{p}_A\otimes\nabla_{\!X}R_A
=\boldsymbol{1}
+\sum_A\boldsymbol{d}_A\otimes\nabla_{\!X}R_A.
```

Independent reference and initial current control points define the initial
map

```math
\boldsymbol{F}_0
=\sum_A\boldsymbol{p}_A(0)\otimes\nabla_{\!X}R_A,
\qquad
\boldsymbol{F}_0\neq\boldsymbol{1}.
```

This state is initially stressed wherever
$`\boldsymbol{P}(\boldsymbol{F}_0)\neq\boldsymbol{0}`$.

The reference geometry is admissible only where

```math
\det\boldsymbol{J}_0>0.
```

The essential isogeometric idea is that the basis representing the exact CAD
geometry is also the approximation space for the unknown field. Control
points are coefficients of that field; except at interpolatory locations,
they are not physical mesh nodes.

### 4. Knot-span elements and quadrature

An element is the Cartesian product of nonzero knot spans. In one parametric
direction, map a parent Gauss coordinate $`\hat\xi\in[-1,1]`$ to the span
$`[\xi_a,\xi_b]`$ by

```math
\xi(\hat\xi)
=\frac{1}{2}
\left[(\xi_b-\xi_a)\hat\xi+\xi_a+\xi_b\right],
\qquad
\frac{\mathrm d\xi}{\mathrm d\hat\xi}
=\frac{\xi_b-\xi_a}{2}.
```

For a $`d`$-dimensional tensor-product span, the parent-to-parametric Jacobian
factor is

```math
j_{\xi}=\prod_{\alpha=1}^{d}
\frac{\xi_b^{\alpha}-\xi_a^{\alpha}}{2}.
```

Therefore a reference-domain integral is approximated by

```math
\int_{\Omega_0^e}g(\boldsymbol{X})\,\mathrm dV
\approx
\sum_q
w_q\,j_{\xi}\,
\det\boldsymbol{J}_0(\boldsymbol{\xi}_q)\,
g(\boldsymbol{X}(\boldsymbol{\xi}_q)).
```

A degree vector $`(p_1,\ldots,p_d)`$ gives
$`\prod_{\alpha=1}^{d}(p_\alpha+1)`$ locally supported basis functions on an
interior span. A tensor Gauss rule with $`p_\alpha+1`$ points in direction
$`\alpha`$ exactly integrates one-dimensional polynomials through degree
$`2p_\alpha+1`$ before rational geometry and nonlinear material terms are
introduced.

### 5. Hyperelastic IGA equations

For a total-Lagrangian hyperelastic body,

```math
U(\boldsymbol{d})
=\int_{\Omega_0}\Psi(\boldsymbol{F})\,\mathrm dV,
\qquad
\boldsymbol{P}=\frac{\partial\Psi}{\partial\boldsymbol{F}},
\qquad
\mathbb{A}=\frac{\partial\boldsymbol{P}}
{\partial\boldsymbol{F}}.
```

The control-point internal force and tangent blocks are

```math
\boldsymbol{f}^{\mathrm{int}}_A
=\int_{\Omega_0}
\boldsymbol{P}\nabla_{\!X}R_A\,\mathrm dV.
```

```math
(\boldsymbol{K}_{AB})_{ik}
=\int_{\Omega_0}
\mathbb{A}_{iJkL}R_{A,J}R_{B,L}\,\mathrm dV.
```

The consistent and lumped masses are

```math
M_{AB}=\int_{\Omega_0}\rho_0R_AR_B\,\mathrm dV,
\qquad
m_A=\sum_BM_{AB}
=\int_{\Omega_0}\rho_0R_A\,\mathrm dV.
```

A boundary traction density produces the generalized force

```math
\boldsymbol{f}^{\Gamma}_A
=\int_{\Gamma_0^t}R_A\bar{\boldsymbol{t}}_0\,\mathrm dA.
```

The energy, stress, and tangent of each supported material are defined in the
[shared constitutive-model theory](../physics_model/consititutive_model/README.md#hyperelastic-solid-laws).
IGA contributes the rational basis gradients and quadrature measure; the
constitutive law is independent of the spline discretization.

### 6. Axisymmetric NURBS solids

For a no-swirl meridian patch, interpolate the reference and current radii
with the rational basis:

```math
R(\boldsymbol{\xi})
=\sum_AR_A(\boldsymbol{\xi})\widehat R_A,
\qquad
r(\boldsymbol{\xi})
=\sum_AR_A(\boldsymbol{\xi})\widehat r_A.
```

Here $`\widehat R_A`$ and $`\widehat r_A`$ are the reference and current radial
control coordinates. The reconstructed three-dimensional deformation
gradient is

```math
\boldsymbol{F}_{axi}
=\begin{bmatrix}
\partial r/\partial R & \partial r/\partial Z & 0 \\[0pt]
\partial z/\partial R & \partial z/\partial Z & 0 \\[0pt]
0 & 0 & r/R
\end{bmatrix}.
```

The hoop stretch and reference volume measure are

```math
\lambda_\theta=\frac{r}{R},
\qquad
\mathrm dV_0=2\pi R\,\mathrm dA_0.
```

Thus every energy, force, tangent, and mass quadrature weight is multiplied by
$`2\pi R`$. The direct meridian formulation requires positive reference radius
at every quadrature point.

### 7. Semi-discrete dynamics

The control-point equations are

```math
\boldsymbol{M}\boldsymbol{a}
+\boldsymbol{f}^{\mathrm{int}}(\boldsymbol{d})
=\boldsymbol{f}^{\mathrm{ext}}.
```

With lumped mass, a first-order velocity-decay explicit step can be written as

```math
\boldsymbol{a}_n
=\boldsymbol{M}_L^{-1}
\left(\boldsymbol{f}^{\mathrm{ext}}_n
-\boldsymbol{f}^{\mathrm{int}}_n\right).
```

```math
\boldsymbol{v}^{*}
=\boldsymbol{v}_n+\Delta t\boldsymbol{a}_n,
\qquad
\boldsymbol{v}_{n+1}
=\max(0,1-\zeta\Delta t)\boldsymbol{v}^{*},
\qquad
\boldsymbol{d}_{n+1}
=\boldsymbol{d}_n+\Delta t\boldsymbol{v}_{n+1}.
```

For an implicit displacement increment
$`\Delta\boldsymbol{d}=\boldsymbol{d}_{n+1}-\boldsymbol{d}_n`$, introduce
parameters $`(\alpha_N,\beta,\gamma)`$. The acceleration and velocity updates
are

```math
\boldsymbol{a}_{n+1}
=\frac{\Delta\boldsymbol{d}}
{2\alpha_N\beta\Delta t^2}
-\frac{\boldsymbol{v}_n}
{2\alpha_N\beta\Delta t}
-\left(\frac{1}{2\beta}-1\right)\boldsymbol{a}_n.
```

```math
\boldsymbol{v}_{n+1}
=\frac{\gamma}{2\alpha_N\beta\Delta t}
\Delta\boldsymbol{d}
+\left(1-\frac{\gamma}{2\alpha_N\beta}\right)
\boldsymbol{v}_n
+\Delta t\left(1-\frac{\gamma}{2\beta}\right)
\boldsymbol{a}_n.
```

The choice
$`(\alpha_N,\beta,\gamma)=(1/2,1/4,1/2)`$ recovers the average-acceleration
Newmark method. Define

```math
c_a=\frac{1}{2\alpha_N\beta\Delta t^2},
\qquad
c_v=\frac{1}{2\alpha_N\beta\Delta t},
\qquad
c_0=\frac{1}{2\beta}-1.
```

An incremental potential whose stationarity gives dynamic equilibrium is

```math
\Phi(\Delta\boldsymbol{d})
=U(\boldsymbol{d}_n+\Delta\boldsymbol{d})
-\boldsymbol{f}^{\mathrm{ext}}\cdot\Delta\boldsymbol{d}
+\frac{1}{2}\sum_A m_A\Delta\boldsymbol{d}_A\cdot
\left(
c_a\Delta\boldsymbol{d}_A
-2c_v\boldsymbol{v}_{A,n}
-2c_0\boldsymbol{a}_{A,n}
\right).
```

The Newton tangent therefore contains the positive inertial contribution

```math
\boldsymbol{K}_{eff}=\boldsymbol{K}+c_a\boldsymbol{M}.
```

Quasi-static analysis drops the inertial part and solves
$`\boldsymbol{f}^{\mathrm{int}}-\boldsymbol{f}^{\mathrm{ext}}=\boldsymbol{0}`$.

### 8. Refinement and continuity

Knot insertion is $`h`$-refinement: it increases the number of basis functions
without changing the represented geometry. Degree elevation is $`p`$-refinement:
it raises polynomial order while preserving the geometry. Performing degree
elevation together with continuity-preserving knot insertion gives
$`k`$-refinement, which can increase both order and inter-element continuity.
All geometry-preserving refinement is most naturally carried out in
homogeneous coordinates

```math
\widetilde{\boldsymbol{P}}_A
=\left[w_A\boldsymbol{P}_A,\;w_A\right],
```

followed by projection back to physical coordinates. This treats control
points and weights as one polynomial B-spline object.

## Minimal example

```python
import geotaichi as gt
from src.iga import Cube, DirichletBoundary, Primitives

gt.init(arch="gpu", default_fp="float64", log=False)

patch = Cube()
patch.set_parameters(start_point=[0, 0, 0], size=[2, 1, 0.2])
patch.generate_knot_u(degree=2, num_ctrlpts=9)
patch.generate_knot_v(degree=2, num_ctrlpts=5)
patch.generate_knot_w(degree=2, num_ctrlpts=3)
patch.generate_ctrlpts()
patch.generate_weights()

primitives = Primitives()
primitives.append(patch, "beam")
primitives.finialize()

fixed = [0, 1, 2]
boundary = DirichletBoundary()
boundary.append([fixed], [0.0] * len(fixed))

iga = gt.IGA(log=False)
iga.set_configuration(dimension=3, solver_type="Implicit")
iga.add_primitives(primitives)
iga.add_boundary_condition(dirichlet=boundary)
iga.add_element(degree=[2, 2, 2])
iga.add_material(
    young_modulus=1.0e6,
    poisson_ratio=0.3,
    density=1000.0,
    gravity=[0.0, 0.0, -9.81],
)
iga.set_solver(
    dt=1.0e-3,
    step=100,
    project_hessian_to_psd=True,
    assemble_type="Hash",
    linear_solver="PCG",
)
result = iga.run()
```

Implicit IGA accepts the bounded retry keys `enable_step_retry`,
`step_retry_max_retries`, `step_retry_reduction`, and
`step_retry_minimum_timestep`. A linear, Newton, or line-search failure occurs
before control-point state is advanced, so a reduced timestep can be retried
transactionally. Explicit IGA rejects an enabled policy. Both engines expose
accepted time/step history through the facade's `diagnostics_snapshot()`.

See `examples/iga/` for complete boundary selection and output setup.

For axisymmetric analysis, configure `dimension=2`, `axisymmetric=True`, and
one finite `axis_offset`.  The material map remains three-dimensional with
`F_theta_theta=r/R`, while inertia, internal energy, residual and Hessian use
the revolved `2*pi*R` quadrature measure.  All participating quadrature points
must have `R > 0`.  Both integrators are exercised by
`examples/iga/axisymmetric_annulus/axisymmetric_annulus.py --solver explicit|implicit`.

## Rest shape and initial pre-strain

The optional `rest_shape` contains material reference control points. When it
is omitted, each patch uses a copy of its current control points. When it is
provided, the initial current geometry is preserved while preprocessing uses
the rest shape to compute reference Jacobians, quadrature volumes, nodal
masses, and initial deformation gradients.

Rest control points may be supplied by primitive name:

```python
rest = patch.control_points.copy()
rest[:, 0] *= 0.5
iga.add_primitives(primitives, rest_shape={"beam": rest})

engine = iga.build()
engine.precompute()
F0 = engine.initial_deformation_gradients.to_numpy()
```

They may also be attached when a primitive is appended:

```python
primitives.append(patch, "beam", rest_shape=rest)
```

`patch.rest_control_points` stores the material reference shape, while
`patch.initial_control_points` stores the initial current shape used by
displacement boundary conditions. This separation prevents a zero
displacement condition from removing user-defined initial pre-strain.

All reference patch Jacobians must be finite and positive at quadrature
points. `precompute()` records this condition in a device status field and
raises `ValueError` after the kernel batch if a patch is folded, singular, or
orientation-reversed; invalid patches are never accepted as negative mass or
energy.

`NeumannBoundary.append(dof_id, dof_val)` stores already-integrated generalized
forces at control-point degrees of freedom. It does not integrate a traction
density over a NURBS boundary. In axisymmetric analysis the supplied value must
therefore already contain any required `2*pi*R` boundary measure.

Implicit `DirichletBoundary.append_velocity(dof_id, velocity)` prescribes a
constant velocity. The solver converts it to `velocity * current_dt` whenever
the timestep changes, including retries and shortened output steps. Ordinary
`append` retains its prescribed-increment semantics. Explicit IGA rejects
`append_velocity`.

## Sparse systems and runtime boundary

Implicit IGA supports coordinate and hash assembly. PCG requires
`project_hessian_to_psd=True`; use BiCGSTAB for an unprojected tangent. The
default numerical path keeps state, assembly, matrix-vector products,
preconditioning, and Krylov iterations in Taichi. Host arrays are used for
NURBS preprocessing, explicit output, and user-selected SciPy solves only.
Material CCD dispatches each patch with the element count owned by that patch;
the prefix table is not interpreted as a previous-patch count.

## Tests

IGA geometry, assembly, solver, contact-coupling, and output tests are grouped
under `tests/unit/iga/`. End-to-end examples are available in `examples/iga/`
and `examples/igampm/`.
