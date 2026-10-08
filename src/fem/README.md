# Finite Element Method (FEM): Elastic Solids, Cloth, and IPC Contact

`src/fem` is GeoTaichi's total-Lagrangian finite-element module. Its public
workflow follows the rest of the project: configure a solver, add a mesh and
material, define boundary conditions, select solver options, and call `run()`.

Repository-wide rules for
device-resident numerical backends are documented in
[DEVICE_BACKEND_AUDIT.md](../../agent/DEVICE_BACKEND_AUDIT.md).

[Theory log and derivations](#fem-theory-log) | [Examples](../../examples/)

## Example-backed capabilities

- Elastic volume FEM: [implicit cantilever](../../examples/fem/implicit_volume_cantilever/implicit_volume_cantilever.py) and [volume self-contact](../../examples/fem/implicit_volume_self_contact/implicit_volume_self_contact.py).
- Membrane and cloth mechanics: [explicit membrane](../../examples/fem/explicit_membrane/explicit_membrane.py), [cloth bending](../../examples/fem/cloth/newton_cloth_bending/newton_cloth_bending.py), and [cloth IPC contact](../../examples/fem/implicit_cloth_contact/implicit_cloth_contact.py).
- Axisymmetric FEM: [annulus](../../examples/fem/axisymmetric_annulus/axisymmetric_annulus.py).
- Prescribed rest geometry: [two-layer cloth and sphere](../../examples/fem/cloth/implicit_two_layer_cloth_sphere/implicit_two_layer_cloth_sphere.py).
- Frozen one-sided SDF supports: [cloth rollers](../../examples/fem/cloth/newton_cloth_rollers/newton_cloth_rollers.py).
- Coupled FEM contact: see [FEM–MPM](../fempm/README.md) and [FEM–DEM/ABD](../fedem/README.md) for concrete examples.

Classical total-Lagrangian assembly already caches reference gradients and
quadrature weights. FEM--MPM additionally builds a permanent element-pair scatter
map and accumulates material tangents directly into the coupled HashTriplet
matrix. Standalone FEM and non-classical contributions retain their existing
assembly paths. Quadrature and local-node pair loops stay compact instead of
unrolling a separate copy of each material tangent in the generated kernel.

## Solver, element, material, and contact compatibility

`HEX8` does **not** imply explicit integration, and `TET4` does **not** imply
implicit integration. The time integrator and the element are independent for
elastic volume FEM.

| FEM formulation | Time integration | Constitutive support | Plasticity | Contact and coupling limits |
| --- | --- | --- | --- | --- |
| Elastic `TET4` volume | Explicit or implicit | St. Venant--Kirchhoff or compressible Neo-Hookean | No; the incremental plastic assembler does not accept `TET4` | Standalone IPC/augmented-Lagrangian contact is implicit. Cross-solver contact is documented under [FEDEM](../fedem/README.md). |
| Elastic `HEX8` volume | Explicit or implicit | St. Venant--Kirchhoff or compressible Neo-Hookean | No on this elastic route | Same contact split as elastic `TET4`. |
| Embedded cloth surface | Explicit or implicit | Cloth ARAP or cloth Neo-Hookean, with optional bending/stitch/spring/SDF energies | No | Standalone IPC is implicit; explicit DEM contact and implicit AffineBody IPC are FEDEM routes. |

Constitutive equations remain centralized in the
[shared constitutive-model theory](../physics_model/consititutive_model/README.md).

## Package layout

| Path | Responsibility |
| --- | --- |
| `generator/` | Procedural meshes, import, normalization, and preprocessing |
| `elements/` | Classical FEM shape functions and reference operators |
| `cloth/` | Cloth reference operators and device energy/Hessian assembly |
| `engines/` | Explicit/implicit integrators, state fields, sparse matrices, and line search |
| `boundaries/` | Dirichlet and Neumann boundary descriptions and device application |
| `contact/` | Surface extraction, broad phase, Barrier IPC, CCD, and friction |
| `MaterialManager.py` | Material-name and parameter normalization |
| `mainFEM.py` | Public `FEM` facade |

## FEM theory log

### 1. Total-Lagrangian continuum mechanics

Let $`\Omega_0`$ be the reference body with material coordinate
$`\boldsymbol{X}`$. Its motion, displacement, deformation gradient, and local
volume ratio are

```math
\boldsymbol{x}=\boldsymbol{\varphi}(\boldsymbol{X},t),
\qquad
\boldsymbol{u}=\boldsymbol{x}-\boldsymbol{X},
\qquad
\boldsymbol{F}=\frac{\partial\boldsymbol{x}}{\partial\boldsymbol{X}}
=\boldsymbol{1}+\nabla_{\!X}\boldsymbol{u},
\qquad
J=\det\boldsymbol{F}>0.
```

For reference density $`\rho_0`$, body acceleration $`\boldsymbol{b}`$, and first
Piola--Kirchhoff stress $`\boldsymbol{P}`$, balance of linear momentum in the
reference configuration is

```math
\rho_0\ddot{\boldsymbol{u}}
=\mathrm{Div}_{X}\boldsymbol{P}+\rho_0\boldsymbol{b}
\quad\text{in }\Omega_0.
```

The essential and natural boundary conditions are

```math
\boldsymbol{u}=\bar{\boldsymbol{u}}
\quad\text{on }\Gamma_0^u,
\qquad
\boldsymbol{P}\boldsymbol{N}=\bar{\boldsymbol{t}}_0
\quad\text{on }\Gamma_0^t.
```

For every admissible virtual displacement $`\boldsymbol{w}`$ that vanishes on
$`\Gamma_0^u`$, the weak form is

```math
\int_{\Omega_0}\rho_0\boldsymbol{w}\cdot\ddot{\boldsymbol{u}}\,\mathrm dV
+\int_{\Omega_0}\nabla_{\!X}\boldsymbol{w}:\boldsymbol{P}\,\mathrm dV
=\int_{\Omega_0}\rho_0\boldsymbol{w}\cdot\boldsymbol{b}\,\mathrm dV
+\int_{\Gamma_0^t}\boldsymbol{w}\cdot\bar{\boldsymbol{t}}_0\,\mathrm dA.
```

For a hyperelastic solid with stored-energy density
$`\Psi(\boldsymbol{F})`$,

```math
\boldsymbol{P}=\frac{\partial\Psi}{\partial\boldsymbol{F}},
\qquad
\mathbb{A}=\frac{\partial\boldsymbol{P}}
{\partial\boldsymbol{F}}
=\frac{\partial^2\Psi}{\partial\boldsymbol{F}^2}.
```

In quasi-static analysis, equilibrium is equivalently a stationary point of
the total potential

```math
\Pi(\boldsymbol{u})
=\int_{\Omega_0}\Psi(\boldsymbol{F})\,\mathrm dV
-\int_{\Omega_0}\rho_0\boldsymbol{b}\cdot\boldsymbol{u}\,\mathrm dV
-\int_{\Gamma_0^t}\bar{\boldsymbol{t}}_0\cdot\boldsymbol{u}\,\mathrm dA.
```

### 2. Isoparametric finite-element discretization

On an element with nodes $`a=1,\ldots,n_e`$ and parent coordinate
$`\boldsymbol{\xi}`$, use the same shape functions for the reference and
current geometries:

```math
\boldsymbol{X}^h(\boldsymbol{\xi})
=\sum_{a=1}^{n_e}N_a(\boldsymbol{\xi})\boldsymbol{X}_a,
\qquad
\boldsymbol{x}^h(\boldsymbol{\xi})
=\sum_{a=1}^{n_e}N_a(\boldsymbol{\xi})\boldsymbol{x}_a.
```

The reference mapping Jacobian and material gradients are

```math
\boldsymbol{J}_0
=\frac{\partial\boldsymbol{X}}{\partial\boldsymbol{\xi}},
\qquad
\nabla_{\!X}N_a
=\boldsymbol{J}_0^{-T}\nabla_{\!\xi}N_a,
\qquad
\mathrm dV=\det(\boldsymbol{J}_0)\,\mathrm d\boldsymbol{\xi}.
```

Partition of unity gives the discrete deformation gradient in either of the
equivalent forms

```math
\boldsymbol{F}^h
=\sum_{a=1}^{n_e}\boldsymbol{x}_a\otimes\nabla_{\!X}N_a
=\boldsymbol{1}
+\sum_{a=1}^{n_e}\boldsymbol{u}_a\otimes\nabla_{\!X}N_a.
```

If the initial current nodes do not coincide with the reference nodes, the
body starts from the non-identity initial map

```math
\boldsymbol{F}_0
=\sum_{a=1}^{n_e}\boldsymbol{x}_a(0)\otimes\nabla_{\!X}N_a,
\qquad
\boldsymbol{F}_0\neq\boldsymbol{1}.
```

It is initially stressed wherever
$`\boldsymbol{P}(\boldsymbol{F}_0)\neq\boldsymbol{0}`$; a pure rigid rotation
is non-identity but remains stress-free for an objective elastic energy.

With quadrature points $`q`$, parent weights $`w_q`$, and
$`W_q=w_q\det\boldsymbol{J}_0(\boldsymbol{\xi}_q)`$, the element energy is

```math
U_e\approx\sum_q W_q\Psi(\boldsymbol{F}_q).
```

Its nodal internal force and consistent material tangent are

```math
\boldsymbol{f}^{\mathrm{int}}_a
=\frac{\partial U_e}{\partial\boldsymbol{x}_a}
=\sum_q W_q\boldsymbol{P}_q\nabla_{\!X}N_a.
```

```math
(\boldsymbol{K}_{ab})_{ik}
=\sum_q W_q\mathbb{A}_{iJkL}
N_{a,J}N_{b,L}.
```

The consistent and row-sum lumped mass matrices are

```math
M_{ab}=\int_{\Omega_0^e}\rho_0N_aN_b\,\mathrm dV,
\qquad
m_a=\sum_bM_{ab}
=\int_{\Omega_0^e}\rho_0N_a\,\mathrm dV.
```

### 3. Linear tetrahedron and hexahedron

For a TET4 element on
$`\xi\geq0`$, $`\eta\geq0`$, $`\zeta\geq0`$, and
$`\xi+\eta+\zeta\leq1`$, the shape functions are

```math
N_0=1-\xi-\eta-\zeta,
\qquad
N_1=\xi,
\qquad
N_2=\eta,
\qquad
N_3=\zeta.
```

Define the reference and current edge matrices

```math
\boldsymbol{D}_m
=\left[\boldsymbol{X}_1-\boldsymbol{X}_0\;\;
\boldsymbol{X}_2-\boldsymbol{X}_0\;\;
\boldsymbol{X}_3-\boldsymbol{X}_0\right],
```

```math
\boldsymbol{D}_s
=\left[\boldsymbol{x}_1-\boldsymbol{x}_0\;\;
\boldsymbol{x}_2-\boldsymbol{x}_0\;\;
\boldsymbol{x}_3-\boldsymbol{x}_0\right].
```

Because the gradient is constant inside a linear tetrahedron,

```math
\boldsymbol{F}=\boldsymbol{D}_s\boldsymbol{D}_m^{-1},
\qquad
V_0=\frac{1}{6}\left|\det\boldsymbol{D}_m\right|.
```

For a HEX8 element, associate each node with signs
$`(s_a,t_a,r_a)\in\{-1,1\}^3`$. On the parent cube
$`[-1,1]^3`$,

```math
N_a(\xi,\eta,\zeta)
=\frac{1}{8}(1+s_a\xi)(1+t_a\eta)(1+r_a\zeta).
```

The standard $`2\times2\times2`$ Gauss rule uses every sign combination of
$`1/\sqrt{3}`$, with unit one-dimensional weights. The reference Jacobian,
shape gradients, deformation gradient, and quadrature weight generally vary
over the element.

### 4. Constitutive material response

Hyperelastic energy, stress, and tangent equations are maintained in the
[shared constitutive-model theory](../physics_model/consititutive_model/README.md#hyperelastic-solid-laws).
FEM supplies the quadrature-point deformation gradient and assembles the returned
first Piola stress and algorithmic tangent through the element operators above.

### 5. Cloth membrane and bending theory

#### Shared membrane response

Surface kinematics and the tension--compression ARAP and plane-stress
Neo-Hookean membrane laws are defined in the
[shared cloth constitutive theory](../physics_model/consititutive_model/README.md#surface-cloth-constitutive-laws).
The FEM triangle multiplies the returned surface density by reference area and
thickness before assembling nodal forces and tangents.

#### Quadratic hinge bending

Consider two neighboring cloth patches sharing an edge. Let their reference
areas be $`A_0`$ and $`A_1`$, and let $`c_i`$ be the four cotangent hinge
coefficients, which satisfy $`\sum_i c_i=0`$. With bending stiffness $`k_b`$,
Poisson ratio $`\nu_b`$, and thickness $`h`$, define

```math
D_b=\frac{k_bh^3}{24(1-\nu_b^2)}.
```

The constant hinge matrix and quadratic bending energy are

```math
K_{ij}^{b}=\frac{3D_b}{2(A_0+A_1)}c_ic_j,
```

```math
U_b^{quad}
=\frac{1}{2}\sum_{i,j}K_{ij}^{b}\boldsymbol{x}_i\cdot\boldsymbol{x}_j
=\frac{3D_b}{4(A_0+A_1)}
\left\|\sum_i c_i\boldsymbol{x}_i\right\|^2.
```

#### Dihedral-angle bending

Let $`\ell`$ be the reference shared-edge length, $`\theta`$ the current signed
dihedral angle, and $`\theta_0`$ its rest value. With the dual height

```math
\bar h=\frac{A_0+A_1}{3\ell},
```

the nonlinear hinge energy is

```math
U_b^{dih}=D_b\frac{\ell}{\bar h}(\theta-\theta_0)^2.
```

Its gradient and exact Hessian are

```math
\nabla U_b^{dih}
=2D_b\frac{\ell}{\bar h}(\theta-\theta_0)\nabla\theta,
```

```math
\nabla^2U_b^{dih}
=2D_b\frac{\ell}{\bar h}
\left[
\nabla\theta\nabla\theta^T
+(\theta-\theta_0)\nabla^2\theta
\right].
```

A positive-semidefinite Gauss--Newton approximation retains only
$`2D_b(\ell/\bar h)\nabla\theta\nabla\theta^T`$.

#### Stitch, target, and one-sided support energies

For a stitch node $`\boldsymbol{x}`$ attached at interpolation coordinate $`r`$
on the segment $`(\boldsymbol{y}_0,\boldsymbol{y}_1)`$, define

```math
\boldsymbol{\delta}
=\boldsymbol{x}-(1-r)\boldsymbol{y}_0-r\boldsymbol{y}_1,
\qquad
U_{stitch}=\frac{kA_x}{2}\|\boldsymbol{\delta}\|^2.
```

A target spring has energy

```math
U_{target}=\frac{k}{2}\|\boldsymbol{x}-\boldsymbol{t}\|^2.
```

For a one-sided signed-distance support, let
$`d=(\boldsymbol{x}-\boldsymbol{t})\cdot\boldsymbol{n}`$ and
$`\eta=d/\hat d-1`$. Within the active range $`d\leq\hat d`$,

```math
U_{sdf}=-\frac{kA_x\hat d}{6}\eta^3,
\qquad
\nabla_{\!x}U_{sdf}=-\frac{kA_x}{2}\eta^2\boldsymbol{n}.
```

Outside that range, the one-sided support energy and force vanish.

### 7. Contact theory

#### Point--triangle and edge--edge contact kinematics

For a point--triangle pair, let $`\boldsymbol{\beta}`$ be the barycentric
coordinates of the closest point on the triangle. Its relative vector is

```math
\boldsymbol{r}
=\boldsymbol{x}_0-
\beta_0\boldsymbol{x}_1-
\beta_1\boldsymbol{x}_2-
\beta_2\boldsymbol{x}_3.
```

The corresponding stencil weights are

```math
\boldsymbol{w}^{PT}
=(1,-\beta_0,-\beta_1,-\beta_2).
```

For two closest edge points with coordinates $`s`$ and $`t`$,

```math
\boldsymbol{r}
=(1-s)\boldsymbol{x}_0+s\boldsymbol{x}_1
-(1-t)\boldsymbol{x}_2-t\boldsymbol{x}_3,
```

```math
\boldsymbol{w}^{EE}=(1-s,s,-(1-t),-t).
```

Both stencils can therefore use

```math
\boldsymbol{r}=\sum_iw_i\boldsymbol{x}_i,
\qquad
d=\|\boldsymbol{r}\|,
\qquad
\boldsymbol{n}=\frac{\boldsymbol{r}}{d},
```

```math
\boldsymbol{v}_{rel}=\sum_iw_i\boldsymbol{v}_i,
\qquad
v_n=\boldsymbol{v}_{rel}\cdot\boldsymbol{n},
\qquad
\boldsymbol{v}_t=(\boldsymbol{1}-\boldsymbol{n}\otimes\boldsymbol{n})
\boldsymbol{v}_{rel}.
```

#### Shared Barrier IPC law

The scalar barrier and regularized-friction laws are maintained in the
[shared contact-model theory](../physics_model/contact_model/README.md#incremental-potential-contact).
FEM supplies the PT or EE stencil geometry, contact measure, and closest-feature
derivatives; the signed stencil weights make the assembled internal residual
obey action--reaction.

#### Edge--edge mollification

Nearly parallel edges require a smooth transition that removes the singular
edge--edge parameterization. Let

```math
e=\|\boldsymbol{e}_a\times\boldsymbol{e}_b\|^2,
\qquad
\varepsilon_x
=10^{-3}\|\boldsymbol{e}_a^0\|^2\|\boldsymbol{e}_b^0\|^2.
```

For $`e<\varepsilon_x`$, define

```math
m_{EE}(e)=\frac{e}{\varepsilon_x}
\left(2-\frac{e}{\varepsilon_x}\right),
```

and use $`m_{EE}=1`$ otherwise. The mollified barrier is $`m_{EE}b`$. Its
derivatives follow the product rule,

```math
\nabla(m_{EE}b)=m_{EE}\nabla b+b\nabla m_{EE},
```

```math
\nabla^2(m_{EE}b)
=m_{EE}\nabla^2b+b\nabla^2m_{EE}
+\nabla m_{EE}\nabla b^T
+\nabla b\nabla m_{EE}^T.
```

#### Shared IPC friction

The lagged smooth Coulomb potential and its regularization are defined in the
[shared IPC friction theory](../physics_model/contact_model/README.md#regularized-ipc-friction).
FEM supplies the current PT or EE stencil weights, normals, and normal-force
magnitudes.

#### Continuous collision detection

For a search direction $`\boldsymbol{p}`$, line-search motion is

```math
\boldsymbol{x}(\alpha)
=\boldsymbol{x}_n+\alpha\boldsymbol{p},
\qquad
0\leq\alpha\leq1.
```

Continuous collision detection computes the earliest point--triangle or
edge--edge time of impact. The accepted step is bounded by a safety fraction
of that time so that $`d(\alpha)>d_{min}`$. The same line search may impose
$`\det\boldsymbol{F}(\alpha)>J_{min}`$, preventing both contact penetration and
element inversion. Spatial hashing or a bounding-volume hierarchy changes
only the candidate search, not the contact energy or constraints.

### 8. Semi-discrete dynamics and time integration

After spatial discretization, the unconstrained nodal equations have the
form

```math
\boldsymbol{M}\boldsymbol{a}
+\boldsymbol{C}\boldsymbol{v}
+\boldsymbol{f}^{\mathrm{int}}(\boldsymbol{x})
=\boldsymbol{f}^{\mathrm{ext}},
```

where mass-proportional damping is
$`\boldsymbol{C}=\zeta\boldsymbol{M}`$ when it is used.

With a lumped mass matrix, a symplectic Euler step is

```math
\boldsymbol{a}_n
=\boldsymbol{M}_L^{-1}
\left(
\boldsymbol{f}^{\mathrm{ext}}_n
-\boldsymbol{f}^{\mathrm{int}}_n
-\zeta\boldsymbol{M}_L\boldsymbol{v}_n
\right).
```

```math
\boldsymbol{v}_{n+1}
=\boldsymbol{v}_n+\Delta t\boldsymbol{a}_n,
\qquad
\boldsymbol{x}_{n+1}
=\boldsymbol{x}_n+\Delta t\boldsymbol{v}_{n+1}.
```

This conditionally stable update requires the time step to resolve the largest
discrete frequency; a common linear estimate is

```math
\Delta t\lesssim\frac{2}{\omega_{\max}}.
```

For implicit Newmark integration, define

```math
\boldsymbol{x}_{pred}
=\boldsymbol{x}_n+\Delta t\boldsymbol{v}_n
+\Delta t^2\left(\frac{1}{2}-\beta\right)\boldsymbol{a}_n.
```

For a trial $`\boldsymbol{x}_{n+1}`$,

```math
\boldsymbol{a}_{n+1}
=\frac{\boldsymbol{x}_{n+1}-\boldsymbol{x}_{pred}}
{\beta\Delta t^2}.
```

```math
\boldsymbol{v}_{n+1}
=\boldsymbol{v}_n+\Delta t
\left[(1-\gamma)\boldsymbol{a}_n
+\gamma\boldsymbol{a}_{n+1}\right].
```

The nonlinear residual and effective tangent are

```math
\boldsymbol{r}
=\boldsymbol{f}^{\mathrm{int}}(\boldsymbol{x}_{n+1})
-\boldsymbol{f}^{\mathrm{ext}}_{n+1}
+\boldsymbol{M}\boldsymbol{a}_{n+1}
+\zeta\boldsymbol{M}\boldsymbol{v}_{n+1}.
```

```math
\boldsymbol{K}_{eff}
=\boldsymbol{K}
+\left[
\frac{1}{\beta\Delta t^2}
+\frac{\zeta\gamma}{\beta\Delta t}
\right]\boldsymbol{M}.
```

Newton's method solves
$`\boldsymbol{K}_{eff}\Delta\boldsymbol{x}=-\boldsymbol{r}`$. For a descent
direction $`\boldsymbol{p}`$, an Armijo line search accepts a step
$`\alpha\in(0,1]`$ when

```math
\Pi(\boldsymbol{x}+\alpha\boldsymbol{p})
\leq
\Pi(\boldsymbol{x})
+c\alpha\nabla\Pi(\boldsymbol{x})\cdot\boldsymbol{p},
\qquad 0<c<1,
```

subject also to an admissibility condition such as
$`\det\boldsymbol{F}>J_{min}>0`$. In quasi-static analysis the inertia and
damping terms are omitted.

## Mesh construction and basic volume example

Procedural constructors are available for boxes, rectangles, circles, and
cylinders. A file can be passed directly instead:

```python
import geotaichi as gt
from src.fem import DirichletBoundary

gt.init(arch="gpu", default_fp="float64", log=False)

fem = gt.FEM(log=False)
fem.set_configuration(dimension=3, solver_type="Implicit")
mesh = fem.add_mesh({
    "Geometry": "Box",
    "Size": (2.0, 1.0, 0.5),
    "Divisions": (8, 4, 2),
    "ElementType": "HEX8",
})
fem.add_material(
    "NeoHookean",
    young_modulus=2.0e6,
    poisson_ratio=0.3,
    density=1000.0,
)
fem.add_boundary_condition(
    dirichlet=DirichletBoundary().add(mesh.node_sets["xmin"], "all", 0.0)
)
fem.set_solver(
    quasi_static=True,
    step=10,
    line_search=True,
    project_pd=True,
    assemble_type="HashTriplet",
    linear_solver="PCG",
)
result = fem.run()
```

## Cloth example and one-sided supports

```python
cloth = gt.FEM(log=False)
cloth.set_configuration(dimension=3, solver_type="Implicit")
mesh = cloth.add_mesh(
    geometry="rectangle", size=(1.0, 1.0), divisions=(32, 32)
)
cloth.add_material(
    "ClothARAP",
    stretch_stiffness=5.0e4,
    compression_stiffness=8.0e4,
    density=1000.0,
    thickness=1.0e-3,
    bending_stiffness=2.0e8,
    bending_poisson_ratio=0.3,
    bending_model="Dihedral",
)
cloth.add_stitch([[12, 20, 21]], stiffness=2.0e5, ratios=[0.35])
cloth.add_spring(
    nodes=[4, 5],
    targets=[[0.0, 1.0, 0.0], [0.1, 1.0, 0.0]],
    stiffness=1.0e4,
)
cloth.add_sdf(
    nodes=[30, 31],
    targets=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    normals=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    stiffness=5.0e4,
    dhat=2.0e-2,
)
cloth.set_solver(
    dt=2.0e-3,
    step=100,
    line_search=True,
    project_pd=True,
    assemble_type="COO",
    linear_solver="PCG",
)
result = cloth.run()
```

Implicit FEM accepts `enable_step_retry` (default `False`),
`step_retry_max_retries` (default `2`), `step_retry_reduction` (default
`0.5`), and `step_retry_minimum_timestep` (default `0`). A failed Newton,
linear solve, or line search restores nodal state before a bounded reduced-dt
retry. Explicit FEM rejects the option. Barrier IPC is transactional and
supported. `diagnostics_snapshot()` reports common progress
for both solver families and detailed nonlinear/contact state for implicit FEM.

The bending model may be `"Quadratic"`, `"Dihedral"`, or `"None"` and can
also be overridden with `set_bending_model()`. Dihedral angle gradients and
Hessians are analytic Taichi functions. `project_bending_pd=True` uses the
positive-semidefinite Gauss--Newton angle-gradient block directly, avoiding a
device eigendecomposition of the full `12 x 12` hinge Hessian; set it to
`False` for the exact tangent. Stitch, spring, and SDF energies also provide
analytic device Hessians.

`add_sdf(..., sdf=shape)` samples an existing GeoTaichi SDF during
preprocessing. The sampled targets and normals are frozen device fields. They
are not automatically resampled for an animated SDF.

## Rest shape

`rest_shape` is an optional per-node material reference configuration with
the same shape as `mesh.points`. When omitted, it is copied from the initial
current coordinates, giving an undeformed initial state. When supplied, it
is used to compute reference Jacobians, gradients, integration weights,
mass, bending data, and the initial deformation gradient while the current
coordinates remain unchanged.

```python
mesh = fem.create_mesh(
    "box",
    size=(1.0, 1.0, 1.0),
    divisions=(4, 4, 4),
    element_type="TET4",
)
rest = mesh.points.copy()
rest[:, 0] *= 0.8
fem.add_mesh(mesh, rest_shape=rest)
```

## Sparse assembly and linear solvers

- `assemble_type="COO"` stores scalar coordinate triplets.
- `assemble_type="HashTriplet"` assembles device-side block triplets through
  the shared hash sparse infrastructure.
- `linear_solver="PCG"` requires `project_pd=True`.
- `linear_solver="BiCGSTAB"` supports unprojected indefinite Newton tangents.
- `linear_solver="Scipy"` is the only supported host linear-solve boundary;
  it does not enable a NumPy FEM backend.

All forces, energies, residuals, time integration, element Hessians, contact
searches, and default Krylov solves remain in Taichi fields and kernels.
Python is used for configuration, topology preprocessing, scalar nonlinear
control flow, and explicit output adapters.

Implicit finalization reuses the internal-force field assembled for the last
accepted Newton state when recovering equilibrium and reactions; it does not
rebuild unchanged contact or element state solely for reporting.

## Contact

```python
fem.add_contact(
    "BarrierIPC",                 # strict barrier IPC (legacy name: "IPC")
    self_contact=True,
    broad_phase="BVH",          # or "LinkedCell"
    planes=[{"point": (0, 0, 0), "normal": (0, 0, 1)}],
    dhat=2.0e-2,
    dmin=2.0e-3,
    kappa=5.0e4,
    friction_coefficient=0.3,
    epsv=1.0e-3,
    friction_iterations=-1,       # iterate until the refreshed system converges
    friction_tolerance=1.0e-7,    # unapplied correction / dt
    friction_max_iterations=50,   # safety cap
    ccd_safety=0.9,
    project_pd=True,
    point_triangle_coordination_number=32,
    edge_edge_coordination_number=128,
)
```

Positive `friction_iterations` retain fixed-count lagging.  Use `-1` when the
step must be certified: after every complete Newton solve FEM refreshes the
lagged frames and normal forces, solves the updated linear system once without
applying that correction, and accepts only when its infinity norm divided by
`dt` is below `friction_tolerance`.

Contact candidates and their PT/EE Hessian blocks are allocated once when the
assembler is created. By default their capacities are the triangle count
multiplied by the two coordination numbers above. Use
`max_point_triangle_pairs` and `max_edge_edge_pairs` to override either
estimate explicitly. A capacity underestimate raises an error asking for a
larger value; contact assembly never resizes these fields at runtime.

Multiple disconnected FEM bodies may be stored in one compatible mesh block.
Pass one `cell_body_ids` value per cell, or let connected components define
body IDs, then assign independent IPC parameters to unordered body pairs:

```python
fem.add_contact("IPC", self_contact=False, broad_phase="LinkedCell")
fem.add_contact_property(0, 1, dhat=2.0e-2, kappa=5.0e4,
                         friction_coefficient=0.2)
fem.add_contact_property(1, 2, dhat=1.0e-2, kappa=2.0e5,
                         friction_coefficient=0.5)
```

When pair properties are present, only the listed pairs are active; global
plane parameters remain available. Pair-specific contact uses the same IPC
barrier law.

FEM contact uses collision culling rather than treating the spatial query as
the final contact set. First, the selected `broad_phase` backend produces
conservative PT/EE AABB overlaps. The common Taichi culling layer then removes
topological neighborhoods, evaluates exact PT/EE distances for the current
Newton configuration, and compacts active stencils with prefix sums. Shared
mesh topology is rejected by both spatial backends before this common layer.

For IPC line search, collision culling is rebuilt from the swept `x -> x+p`
AABBs on every CCD query. Conservative PT/EE CCD computes the admissible step
and compacts only stencils that can limit it. No Verlet multiplier is used.
The dynamic linked-cell backend performs count, prefix sum, and compact fill
without fixed per-cell primitive capacities; its EE boxes split the pairwise
search radius equally between the two edges. The BVH backend refits a reusable
topology. Thus `broad_phase="LinkedCell"` and `"BVH"` select only the spatial
backend; distance/CCD pruning and exclusions have identical semantics.

## Output and tests

Results can include displacement, velocity, acceleration, reactions, stress,
strain energy, and contact history. The main examples are under
`examples/fem/`. Unit and integration coverage is under `tests/unit/fem/` and
`tests/integration/fem/`.
