# Material Point Method (MPM): Solid Mechanics, Incompressible Flow, Two-Phase Coupling, and Contact

`src/mpm` provides GeoTaichi's material-point solvers and the public `MPM`
facade. It includes updated- and total-Lagrangian formulations, explicit and
implicit time integration, single- and two-phase materials, incompressible
flow, sparse/adaptive grids, and soft-particle
contact modes.

[Theory log and derivations](#mpm-theory-log) | [Examples](../../examples/)

## Example-backed capabilities

- Explicit solid MPM: [column collapse](../../examples/mpm/column_collapse/DPmaterial.py) and [total-Lagrangian bar](../../examples/mpm/ElasticBar/TLBar.py).
- Implicit solid MPM: [Column collapse](../../examples/mpm/column_collapse2D/DPmaterial2DImplicit.py), [ULMPM block](../../examples/mpm/direct/implicit_ulmpm_block2d.py), and [elastic bar](../../examples/mpm/ElasticBar/ImplicitBar.py).
- Semi-implicit incompressible flow: [lid-driven cavity](../../examples/mpm/IncompressibleFluid/lid_driven_cavity_2d/lid_driven_cavity_2d.py), [Taylor–Green vortex](../../examples/mpm/IncompressibleFluid/taylor_green_vortex_2d/taylor_green_vortex_2d.py), and [3D wavemaker](../../examples/mpm/IncompressibleFluid/wavemaker_tank_3d/wavemaker_tank_3d.py).
- Two-phase porous-media MPM: [porous-column dam break](../../examples/mmpm/DamBreakPorousElastic2D/double_point_dam_break_porous_elastic.py) and [U-tube flow](../../examples/mmpm/UTubeFlow2D/u_tube_flow_2d.py).
- Sparse and adaptive grids: [sparse flume](../../examples/mpm/FlumeTest/flume_test.py) and [adaptive slope](../../examples/mpm/random_field/flat_slope.py).
- Soft-particle MPM/LSDEM contact: [mixed rigid–soft drop](../../examples/mpdem/LevelSet/SoftRigid/rigid_soft_sphere_drop_box.py).

## Formulations and example selection

MPM is organized here by physical model, Lagrangian configuration, time integration, and contact formulation—not by internal software backend. The implementations share the same MPM balance laws and interpolation principles; their input APIs and supported options differ for engineering reasons. Use the configuration in the linked example rather than treating an implementation name as a different numerical method.

| Formulation | Concrete examples | Scope |
| --- | --- | --- |
| Explicit solid ULMPM/TLMPM | [Column collapse](../../examples/mpm/column_collapse/DPmaterial.py), [TL elastic bar](../../examples/mpm/ElasticBar/TLBar.py) | Solid dynamics and formulation-specific constitutive response |
| Implicit solid mechanics | [Column collapse](../../examples/mpm/column_collapse2D/DPmaterial2DImplicit.py), [UL block](../../examples/mpm/direct/implicit_ulmpm_block2d.py), [Elastic bar](../../examples/mpm/ElasticBar/ImplicitBar.py) | Newton equilibrium; the finite-strain plastic examples use ULMPM |
| Semi-implicit incompressible flow | [Cavity](../../examples/mpm/IncompressibleFluid/lid_driven_cavity_2d/lid_driven_cavity_2d.py), [Wavemaker](../../examples/mpm/IncompressibleFluid/wavemaker_tank_3d/wavemaker_tank_3d.py) | MAC pressure projection and particle transport |
| Two-phase porous media | [Porous-column dam break](../../examples/mmpm/DamBreakPorousElastic2D/double_point_dam_break_porous_elastic.py), [U-tube](../../examples/mmpm/UTubeFlow2D/u_tube_flow_2d.py) | Solid/fluid field exchange and semi-implicit pressure solve |
| Soft-particle and cross-solver contact | [Rigid–soft drop](../../examples/mpdem/LevelSet/SoftRigid/rigid_soft_sphere_drop_box.py), [MPM–ABD impact](../../examples/mpdem/AffineBody/ABDImpactDP/direct_mpm_abd_impact.py) | Body-attached soft-particle grids or continuum contact; explicit level-set forces and implicit IPC are distinct contact formulations |

Material equations remain in the [shared constitutive theory](../physics_model/consititutive_model/README.md). Fluid, continuum-solid, and soft-particle MPM–ABD configurations are compared in the [coupling route guide](../mpdem/README.md#mpm-affinebody-route-selection).

## Package layout

| Path | Responsibility |
| --- | --- |
| `generator/` | Regions, particle bodies, file input, and sampling |
| `elements/` | Background grids, shape functions, quadrature, and adaptivity |
| `structs/` | Taichi particle, grid, material, and boundary fields |
| `engines/` | Explicit, implicit, incompressible, and two-phase operators |
| `engines/SoftParticleEngine.py` | LSMPM/SDF-specific explicit stepping mixin |
| `engines/direct/` | Particle-coordinate input, MPM assembly, and nonlinear solvers |
| `sparse_grid/` | Block-sparse grid storage and compaction |
| `soft_particle/` | Level-set and IPC soft-particle mechanics |
| `boundaries/` | Device boundary constraints |
| `mainMPM.py` | Public facade and backend selection |

Constitutive models are selected from `src/physics_model`. Their elastic,
plastic, and fluid equations are collected in the
[shared constitutive-model theory](../physics_model/consititutive_model/README.md).
Scalar penalty, DEM, and IPC contact laws are collected in the
[shared contact-model theory](../physics_model/contact_model/README.md).

## MPM theory log

The equations below describe the numerical models independently of the data
layout. Subscripts $`p`$ and $`i`$ denote material points and grid nodes,
respectively; $`d`$ is the spatial dimension; $`N_{pi}=N_i(\boldsymbol{x}_p)`$;
and $`\boldsymbol{G}_{pi}=\nabla N_i(\boldsymbol{x}_p)`$. Quantities carrying a
subscript $`0`$ are measured in the reference configuration.

### 1. Material-point quadrature and balance laws

Material points carry the quadrature state

```math
\mathcal{S}_p
=\left\{
\boldsymbol{x}_p,\boldsymbol{v}_p,m_p,V_p,
\boldsymbol{\sigma}_p,\boldsymbol{F}_p,\boldsymbol{\alpha}_p
\right\},
\qquad
m_p=\rho_pV_p,
```

where $`\boldsymbol{\alpha}_p`$ denotes any constitutive history. A material
integral is approximated by

```math
\int_{\Omega_t}q(\boldsymbol{x})\,\mathrm{d}v
\approx
\sum_p V_pq_p.
```

The current-configuration mass and momentum balances are

```math
\dot{\rho}+\rho\nabla\cdot\boldsymbol{v}=0,
```

```math
\rho\dot{\boldsymbol{v}}
=\nabla\cdot\boldsymbol{\sigma}+\rho\boldsymbol{b}.
```

Their particle-quadrature weak form gives the lumped nodal quantities

```math
m_i=\sum_pN_{pi}m_p,
\qquad
\boldsymbol{f}_i^{ext}
=\sum_pN_{pi}m_p\boldsymbol{b}_p+\boldsymbol{t}_i,
```

```math
\boldsymbol{f}_i^{int}
=-\sum_pV_p\boldsymbol{\sigma}_p\boldsymbol{G}_{pi},
\qquad
m_i\boldsymbol{a}_i
=\boldsymbol{f}_i^{ext}+\boldsymbol{f}_i^{int}.
```

Partition of unity and linear completeness are the transfer invariants

```math
\sum_iN_{pi}=1,
\qquad
\sum_iN_{pi}\boldsymbol{x}_i=\boldsymbol{x}_p,
\qquad
\sum_i\boldsymbol{G}_{pi}=\boldsymbol{0}.
```

They are what make particle-grid transfer preserve constant fields and total
linear momentum.

### 2. Grid basis functions

For grid spacing $`h`$ and normalized distance
$`r=(x_p-x_i)/h`$, the one-dimensional linear basis is

```math
B_1(r)=1-|r| \quad \text{for } |r|<1,
```

and $`B_1(r)=0`$ outside its support. The multidimensional basis is a tensor
product,

```math
N_{pi}=\prod_{a=1}^{d}B(r_a),
```

```math
G_{pi,a}
=\frac{1}{h_a}B'(r_a)
\prod_{b\ne a}B(r_b).
```

GIMP averages the grid basis over the particle domain $`\Omega_p`$:

```math
N_{pi}^{GIMP}
=\frac{1}{V_p}\int_{\Omega_p}N_i(\boldsymbol{x})\,\mathrm{d}v,
```

```math
\boldsymbol{G}_{pi}^{GIMP}
=\frac{1}{V_p}\int_{\Omega_p}\nabla N_i(\boldsymbol{x})\,\mathrm{d}v.
```

This convolution enlarges the support by the particle half-length and makes
the transfer continuous while a particle crosses a cell boundary.

The smoothed-linear variant retains the linear weight at the particle center
but averages its gradient over the particle domain:

```math
N_{pi}^{SL}=N_i(\boldsymbol{x}_p),
\qquad
\boldsymbol{G}_{pi}^{SL}
=\frac{1}{V_p}\int_{\Omega_p}\nabla N_i(\boldsymbol{x})\,\mathrm{d}v.
```

The interior quadratic B-spline is

```math
B_2(r)=\frac{3}{4}-r^2
\quad \text{for } |r|<\frac12,
```

```math
B_2(r)=\frac12\left(\frac32-|r|\right)^2
\quad \text{for } \frac12\le |r|<\frac32,
```

and is zero otherwise. The interior cubic B-spline is

```math
B_3(r)=\frac23-r^2+\frac12|r|^3
\quad \text{for } |r|<1,
```

```math
B_3(r)=\frac16(2-|r|)^3
\quad \text{for } 1\le |r|<2,
```

and is zero otherwise. Boundary-modified B-splines retain partition of unity
and first-order consistency after their support is clipped by the domain.

Moving-least-squares transfer uses a polynomial vector
$`\boldsymbol{p}(\boldsymbol{x})`$, normally
$`\boldsymbol{p}=(1,x_1,\ldots,x_d)^T`$, and the moment matrix

```math
\boldsymbol{M}_p
=\sum_iw_{pi}\boldsymbol{p}(\boldsymbol{x}_i)
\boldsymbol{p}(\boldsymbol{x}_i)^T.
```

The corrected particle-to-node basis is

```math
\Phi_{pi}
=\boldsymbol{p}(\boldsymbol{x}_p)^T
\boldsymbol{M}_p^{-1}
w_{pi}\boldsymbol{p}(\boldsymbol{x}_i).
```

### 3. Particle-grid transfer

The standard particle-to-grid transfer is

```math
m_i=\sum_pN_{pi}m_p,
\qquad
\boldsymbol{p}_i=\sum_pN_{pi}m_p\boldsymbol{v}_p,
\qquad
\boldsymbol{v}_i=\frac{\boldsymbol{p}_i}{m_i}.
```

APIC augments each particle with an affine velocity tensor
$`\boldsymbol{C}_p`$:

```math
\boldsymbol{p}_i
=\sum_pN_{pi}m_p
\left[
\boldsymbol{v}_p
+\boldsymbol{C}_p(\boldsymbol{x}_i-\boldsymbol{x}_p)
\right].
```

Define

```math
\boldsymbol{D}_p
=\sum_iN_{pi}
(\boldsymbol{x}_i-\boldsymbol{x}_p)
(\boldsymbol{x}_i-\boldsymbol{x}_p)^T,
```

```math
\boldsymbol{B}_p
=\sum_iN_{pi}
(\boldsymbol{v}_i-\boldsymbol{v}_p^{PIC})
(\boldsymbol{x}_i-\boldsymbol{x}_p)^T.
```

Then the recovered affine field is

```math
\boldsymbol{C}_p=\boldsymbol{B}_p\boldsymbol{D}_p^{-1}.
```

The grid-to-particle velocity options are

```math
\boldsymbol{v}_p^{PIC}
=\sum_iN_{pi}\boldsymbol{v}_i^{n+1},
```

```math
\boldsymbol{v}_p^{FLIP}
=\boldsymbol{v}_p^n
+\Delta t\sum_iN_{pi}\boldsymbol{a}_i,
```

```math
\boldsymbol{v}_p^{n+1}
=\alpha_{PIC}\boldsymbol{v}_p^{PIC}
+(1-\alpha_{PIC})\boldsymbol{v}_p^{FLIP}.
```

Particle positions use the PIC transport field,

```math
\boldsymbol{x}_p^{n+1}
=\boldsymbol{x}_p^n+\Delta t\boldsymbol{v}_p^{PIC},
```

and the grid velocity gradient is

```math
\boldsymbol{L}_p
=\sum_i\boldsymbol{v}_i^{n+1}\boldsymbol{G}_{pi}^T.
```

The updated volume and deformation gradient are

```math
\boldsymbol{F}_p^{n+1}
=\left(\boldsymbol{I}+\Delta t\boldsymbol{L}_p\right)
\boldsymbol{F}_p^n,
```

```math
V_p^{n+1}
=V_p^n\det\left(\boldsymbol{I}+\Delta t\boldsymbol{L}_p\right).
```

The fused G2P2G path evaluates the same transfer identities without storing a
complete intermediate grid state.

### 4. Updated-Lagrangian explicit integration

With forces evaluated in the current configuration, the nodal update is

```math
\boldsymbol{a}_i^n
=\frac{\boldsymbol{f}_i^{ext,n}+\boldsymbol{f}_i^{int,n}}
{m_i},
```

```math
\boldsymbol{v}_i^{n+1}
=\boldsymbol{v}_i^n+\Delta t\boldsymbol{a}_i^n.
```

Component-wise background damping modifies an unbalanced force only when it
does positive work:

```math
f_{i,a}^{damp}
=-\zeta|f_{i,a}|\mathrm{sign}(v_{i,a})
\quad \text{if } v_{i,a}f_{i,a}>0.
```

USL evaluates stress after the particle update, USF evaluates stress before
the force assembly, and MUSL remaps the updated particle velocity to the grid
before evaluating $`\boldsymbol{L}_p`$. These orderings use the same balance and
transfer equations but sample the stress at different points in the time
step.

The constitutive closure is

```math
\boldsymbol{\sigma}_p^{n+1},\boldsymbol{\alpha}_p^{n+1}
=\mathcal{C}
\left(
\boldsymbol{\sigma}_p^n,
\boldsymbol{L}_p,
\boldsymbol{\alpha}_p^n,
\Delta t
\right).
```

The definitions of $`\mathcal{C}`$ are kept in the
[constitutive-model theory](../physics_model/consititutive_model/README.md).

### 5. Total-Lagrangian explicit integration

Total-Lagrangian MPM freezes basis gradients in the reference configuration.
The deformation map and first Piola stress are

```math
\boldsymbol{F}_p
=\boldsymbol{I}+\sum_i\boldsymbol{u}_i
\left(\nabla_0N_{pi}\right)^T,
```

```math
\boldsymbol{P}_p
=J_p\boldsymbol{\sigma}_p\boldsymbol{F}_p^{-T},
\qquad
J_p=\det\boldsymbol{F}_p.
```

The reference internal force is

```math
\boldsymbol{f}_{i,0}^{int}
=-\sum_pV_{p0}\boldsymbol{P}_p\nabla_0N_{pi}.
```

Mass is assembled once from the reference support,

```math
m_i=\sum_pN_{pi}^0m_p,
```

while momentum, force, and the explicit grid update retain the same form as
the updated-Lagrangian equations.

### 6. Volumetric-locking control and smoothing

For B-bar, split a velocity gradient into deviatoric and volumetric parts,

```math
\boldsymbol{L}_p
=\mathrm{dev}\boldsymbol{L}_p
+\frac{1}{d}\mathrm{tr}(\boldsymbol{L}_p)\boldsymbol{I}.
```

Replace its local volumetric rate by a cell-projected rate
$`\overline{\ell}_p`$:

```math
\overline{\boldsymbol{G}}_{pi}
=\frac{1}{V_c}\int_{\Omega_c}\nabla N_i\,\mathrm{d}v,
\qquad
\overline{\ell}_p
=\sum_i\boldsymbol{v}_i\cdot\overline{\boldsymbol{G}}_{pi},
```

```math
\boldsymbol{L}_p^{Bbar}
=\boldsymbol{L}_p
+\frac{1}{d}
\left(
\overline{\ell}_p-\mathrm{tr}\boldsymbol{L}_p
\right)\boldsymbol{I}.
```

The corresponding B-bar strain-displacement operator keeps the original
deviatoric rows and replaces only its volumetric row:

```math
\boldsymbol{B}_{pi}^{Bbar}
=\mathrm{dev}\boldsymbol{B}_{pi}
+\frac{1}{d}\boldsymbol{m}\,
\overline{\boldsymbol{G}}_{pi}^{T},
```

where $`\boldsymbol{m}`$ selects the normal-strain components and
$`\overline{\boldsymbol{G}}_{pi}`$ is the cell-projected basis gradient. The
same operator is used in both the strain update and internal virtual work.

F-bar first computes

```math
J_p^{\Delta}=\det(\boldsymbol{I}+\Delta t\boldsymbol{L}_p)
```

and a mass-weighted nodal projection

```math
\overline{J}_i
=\frac{\sum_pN_{pi}m_pJ_p^{\Delta}}
{\sum_pN_{pi}m_p},
\qquad
\overline{J}_p=\sum_iN_{pi}\overline{J}_i.
```

For blend fraction $`\eta_F`$,

```math
\widehat{J}_p
=\eta_F\overline{J}_p+(1-\eta_F)J_p^{\Delta},
\qquad
\lambda_p
=\left(\frac{\widehat{J}_p}{J_p^{\Delta}}\right)^{1/d}.
```

The corrected increment and equivalent velocity gradient are

```math
\Delta\boldsymbol{F}_p^{Fbar}
=\lambda_p
\left(\boldsymbol{I}+\Delta t\boldsymbol{L}_p\right),
```

```math
\boldsymbol{L}_p^{Fbar}
=\lambda_p\boldsymbol{L}_p
+\frac{\lambda_p-1}{\Delta t}\boldsymbol{I}.
```

Displacement F-bar applies the same volumetric replacement to
$`\boldsymbol{I}+\nabla\boldsymbol{u}`$ inside an implicit displacement solve.

Pressure smoothing projects the particle mean stress to nodes and back:

```math
p_p=\frac{1}{d}\mathrm{tr}\boldsymbol{\sigma}_p,
\qquad
p_i
=\frac{\sum_pN_{pi}m_pp_p}{\sum_pN_{pi}m_p},
\qquad
\overline{p}_p=\sum_iN_{pi}p_i.
```

Only the spherical stress changes,

```math
\boldsymbol{\sigma}_p^{sm}
=\mathrm{dev}\boldsymbol{\sigma}_p
+\overline{p}_p\boldsymbol{I}.
```

For cell Gauss smoothing, the volume-weighted stress at a quadrature point is

```math
\overline{\boldsymbol{\sigma}}_g
=\frac{\sum_pW_{gp}V_p\boldsymbol{\sigma}_p}
{\sum_pW_{gp}V_p},
```

and the cell-mean pressure replaces only the pressure of each Gauss stress.

### 7. Axisymmetric no-swirl MPM

In meridian coordinates $`(r,z)`$, the no-swirl velocity gradient is defined by

```math
L_{rr}=\frac{\partial v_r}{\partial r},
\qquad
L_{rz}=\frac{\partial v_r}{\partial z},
```

```math
L_{zr}=\frac{\partial v_z}{\partial r},
\qquad
L_{zz}=\frac{\partial v_z}{\partial z},
\qquad
L_{\theta\theta}=\frac{v_r}{r}.
```

The physical volume measure is

```math
\mathrm{d}V=2\pi r\,\mathrm{d}r\,\mathrm{d}z.
```

Consequently, the radial and axial internal forces are

```math
f_{i,r}^{int}
=-\int
\left(
\sigma_{rr}N_{i,r}
+\sigma_{rz}N_{i,z}
+\sigma_{\theta\theta}\frac{N_i}{r}
\right)\mathrm{d}V,
```

```math
f_{i,z}^{int}
=-\int
\left(
\sigma_{zr}N_{i,r}
+\sigma_{zz}N_{i,z}
\right)\mathrm{d}V.
```

For an implicit updated-Lagrangian displacement increment, the hoop stretch
is

```math
f_{\theta\theta}=1+\frac{u_r}{r_n},
```

and the three-dimensional no-swirl update is

```math
\boldsymbol{F}_{n+1}
=\boldsymbol{f}_{n+1}\boldsymbol{F}_n.
```

The axis must satisfy $`r>0`$; an axis offset simply replaces $`r`$ by the
distance from that axis.

### 8. Implicit solid equilibrium

The Newmark family relates a nodal displacement increment
$`\boldsymbol{u}_i`$ to acceleration and velocity through

```math
\boldsymbol{a}_i^{n+1}
=\frac{1}{\beta\Delta t^2}\boldsymbol{u}_i
-\frac{1}{\beta\Delta t}\boldsymbol{v}_i^n
-\left(\frac{1}{2\beta}-1\right)\boldsymbol{a}_i^n,
```

```math
\boldsymbol{v}_i^{n+1}
=\frac{\gamma}{\beta\Delta t}\boldsymbol{u}_i
-\left(\frac{\gamma}{\beta}-1\right)\boldsymbol{v}_i^n
-\frac{\Delta t}{2}
\left(\frac{\gamma}{\beta}-2\right)\boldsymbol{a}_i^n.
```

The dynamic residual is

```math
\boldsymbol{R}_i
=\boldsymbol{f}_i^{ext}
+\boldsymbol{f}_i^{int}
-m_i\boldsymbol{a}_i^{n+1},
```

while the quasi-static residual omits the inertia term. Newton iteration
solves

```math
\boldsymbol{K}^{(k)}\Delta\boldsymbol{u}^{(k)}
=\boldsymbol{R}^{(k)},
\qquad
\boldsymbol{u}^{(k+1)}
=\boldsymbol{u}^{(k)}+\Delta\boldsymbol{u}^{(k)}.
```

The consistent tangent has the material and Newmark contributions

```math
\boldsymbol{K}_{ij}
=\sum_pV_p
\boldsymbol{B}_{pi}^T
\mathbb{C}_p^{alg}
\boldsymbol{B}_{pj}
+\frac{m_i}{\beta\Delta t^2}
\delta_{ij}\boldsymbol{I}.
```

Here $`\mathbb{C}_p^{alg}`$ is the elastic or algorithmic constitutive tangent
defined in the shared constitutive document. Matrix-free, COO, and block
triplet assembly represent this same linearized operator.

### 9. Implicit updated- and total-Lagrangian mechanics

Implicit finite-strain mechanics solves for the active-grid displacement vector
$`\boldsymbol{u}`$. Its updated-Lagrangian particle map is

```math
\boldsymbol{F}_p(\boldsymbol{u})
=\left[
\boldsymbol{I}
+\sum_i\boldsymbol{u}_i\boldsymbol{G}_{pi}^T
\right]\boldsymbol{F}_{p,n},
```

whereas its total-Lagrangian map is

```math
\boldsymbol{F}_p(\boldsymbol{u})
=\boldsymbol{F}_{p,n}
+\sum_i\boldsymbol{u}_i
\left(\nabla_0N_{pi}\right)^T.
```

For a Newmark predictor

```math
\widetilde{\boldsymbol{u}}_i
=\Delta t\boldsymbol{v}_i^n
+\Delta t^2\left(\frac12-\beta\right)\boldsymbol{a}_i^n,
```

the incremental potential is

```math
\Pi(\boldsymbol{u})
=\sum_pV_{p0}\Psi_p\left(\boldsymbol{F}_p(\boldsymbol{u})\right)
+\frac12
\sum_i\frac{m_i}{\beta\Delta t^2}
\left\|\boldsymbol{u}_i-\widetilde{\boldsymbol{u}}_i\right\|^2
-\sum_i\boldsymbol{f}_i^{ext}\cdot\boldsymbol{u}_i.
```

The equilibrium equations and Newton matrix are

```math
\boldsymbol{g}(\boldsymbol{u})
=\nabla_{\boldsymbol{u}}\Pi=\boldsymbol{0},
\qquad
\boldsymbol{H}
=\nabla_{\boldsymbol{u}}^2\Pi.
```

Projected Newton replaces negative eigenvalues of a local symmetric Hessian
by nonnegative values,

```math
\boldsymbol{H}_p
=\boldsymbol{Q}_p
\mathrm{diag}(\lambda_{p,a})
\boldsymbol{Q}_p^T,
```

```math
\boldsymbol{H}_p^+
=\boldsymbol{Q}_p
\mathrm{diag}\left(\max(\lambda_{p,a},0)\right)
\boldsymbol{Q}_p^T.
```

Backtracking accepts a trial step
$`\boldsymbol{u}+\alpha\Delta\boldsymbol{u}`$ when

```math
\Pi(\boldsymbol{u}+\alpha\Delta\boldsymbol{u})
\le
\Pi(\boldsymbol{u})
+c\alpha\nabla\Pi(\boldsymbol{u})^T\Delta\boldsymbol{u}.
```

Material continuous collision detection also restricts $`\alpha`$ so that

```math
\det\boldsymbol{F}_p
\left(\boldsymbol{u}+\alpha\Delta\boldsymbol{u}\right)>0
```

for every particle. Plastic trial states, return maps, tangents, and accepted
history commits are defined in the
[finite-strain plasticity section](../physics_model/consititutive_model/README.md#finite-strain-multiplicative-plasticity).

### 10. Explicit single-layer solid-fluid mixture

A mixture material point carries porosity $`n_p`$, intrinsic densities
$`\rho_s,\rho_f`$, skeleton velocity $`\boldsymbol{v}_s`$, pore-fluid velocity
$`\boldsymbol{v}_f`$, effective skeleton stress $`\boldsymbol{\sigma}'`$, and pore
pressure $`p`$. Its phase masses are

```math
m_p^s=(1-n_p)\rho_sV_p,
\qquad
m_p^f=n_p\rho_fV_p,
\qquad
m_p=m_p^s+m_p^f.
```

The single layer transfers total, solid, and fluid momenta to one collocated
grid:

```math
m_i^a=\sum_pN_{pi}m_p^a,
\qquad
\boldsymbol{p}_i^a
=\sum_pN_{pi}m_p^a\boldsymbol{v}_p^a,
```

where $`a`$ is $`s`$, $`f`$, or the total mixture. With compression-positive pore
pressure, the total and fluid internal-force contributions are

```math
\boldsymbol{f}_{i}^{int}
=-\sum_pV_p
\left(\boldsymbol{\sigma}'_p-p_p\boldsymbol{I}\right)
\boldsymbol{G}_{pi},
```

```math
\boldsymbol{f}_{i}^{f,int}
=\sum_pV_pn_pp_p\boldsymbol{G}_{pi}.
```

Darcy drag on the fluid phase is

```math
\boldsymbol{f}_p^d
=-\frac{n_p^2\gamma_fV_p}{k_p}
\left(\boldsymbol{v}_p^f-\boldsymbol{v}_p^s\right).
```

The fluid acceleration is obtained from its phase balance, and the skeleton
acceleration follows from the total balance:

```math
\boldsymbol{a}_i^f
=\frac{\boldsymbol{f}_i^f}{m_i^f},
\qquad
\boldsymbol{a}_i^s
=\frac{\boldsymbol{f}_i^{tot}-m_i^f\boldsymbol{a}_i^f}{m_i^s}.
```

For incompressible grains, the porosity update is

```math
n_p^{n+1}
=1-\frac{1-n_p^n}{J_{s,p}^{\Delta}},
\qquad
J_{s,p}^{\Delta}
=\det\left(\boldsymbol{I}+\Delta t\boldsymbol{L}_{s,p}\right).
```

The pore pressure increment generated by finite fluid bulk modulus $`K_f`$ is

```math
p_p^{n+1}
=p_p^n
-\frac{K_f}{n_p}
\left[
(1-n_p)(J_{s,p}^{\Delta}-1)
+n_p(J_{f,p}^{\Delta}-1)
\right].
```

The skeleton constitutive law acts only on
$`\boldsymbol{L}_{s,p}`$ and is referenced from the shared constitutive theory.

### 11. Double-layer semi-implicit two-phase MPM

The double-layer formulation represents skeleton and water with separate
particle sets. The skeleton is transferred to a nodal MPM grid, while fluid
velocity components live on the faces of a staggered MAC grid. The mapped
solid fraction and porosity are

```math
\phi_s
=\frac{1}{V_c}
\sum_{p\in s}N_{cp}V_p(1-n_p),
\qquad
\phi_f=n=1-\phi_s.
```

Let the relative phase velocity be

```math
\boldsymbol{w}=\boldsymbol{v}_f-\boldsymbol{v}_s.
```

The drag force per mixture volume is

```math
\boldsymbol{d}=K_d\boldsymbol{w},
```

and gives equal-opposite phase accelerations

```math
\boldsymbol{a}_s^d
=\frac{K_d\boldsymbol{w}}{\rho_s(1-n)},
\qquad
\boldsymbol{a}_f^d
=-\frac{K_d\boldsymbol{w}}{\rho_fn}.
```

For Ergun drag,

```math
K_d
=\frac{150\mu_f(1-n)^2}{nd_g^2}
+\frac{1.75\rho_f(1-n)}{d_g}\|\boldsymbol{w}\|.
```

For permeability-form Darcy drag,

```math
K_d=\frac{n^2\gamma_f}{k}.
```

For Beetstra drag, define

```math
Re=\frac{\rho_fd_g\|\boldsymbol{w}\|}{\mu_f},
```

```math
f_0
=\frac{10(1-n)}{n^2}
+n^2\left(1+1.5\sqrt{1-n}\right),
```

```math
f_{Re}
=\frac{0.413Re}{24n^2}
\frac{n^{-1}+3(1-n)n+8.4Re^{-0.343}}
{1+10^{3(1-n)}Re^{-(5-4n)/2}},
```

```math
K_d=\frac{18\mu_fn(1-n)}{d_g^2}(f_0+f_{Re}).
```

Freezing $`K_d`$ over a time step and treating drag by backward Euler gives

```math
\boldsymbol{w}^{n+1}
=\frac{\boldsymbol{w}^{*}}
{1+\Delta tK_d
\left[
\rho_s^{-1}(1-n)^{-1}+\rho_f^{-1}n^{-1}
\right]}.
```

Before pressure projection, the fluid predictor also contains gravity and
viscous diffusion,

```math
\boldsymbol{v}_f^*
=\boldsymbol{v}_f^n
+\Delta t
\left(
\boldsymbol{g}+\boldsymbol{a}_f^d
+\nu_f\nabla^2\boldsymbol{v}_f^n
\right).
```

Mixture incompressibility requires

```math
\nabla\cdot
\left[
(1-n)\boldsymbol{v}_s+n\boldsymbol{v}_f
\right]=0.
```

Expanding the constraint produces the actual projection residual,

```math
(1-n)\nabla\cdot\boldsymbol{v}_s
+n\nabla\cdot\boldsymbol{v}_f
+\nabla n\cdot(\boldsymbol{v}_f-\boldsymbol{v}_s)=0.
```

The pressure mobility is

```math
M_p=\frac{1-n}{\rho_s}+\frac{n}{\rho_f}.
```

The pressure equation is therefore

```math
\Delta t\nabla\cdot(M_p\nabla p)
=\nabla\cdot
\left[
(1-n)\boldsymbol{v}_s^*
+n\boldsymbol{v}_f^*
\right].
```

After solving it, both phase velocities are corrected:

```math
\boldsymbol{v}_s^{n+1}
=\boldsymbol{v}_s^*-\frac{\Delta t}{\rho_s}\nabla p,
```

```math
\boldsymbol{v}_f^{n+1}
=\boldsymbol{v}_f^*-\frac{\Delta t}{\rho_f}\nabla p.
```

At a fluid-air face, let $`\varphi_F<0`$ and $`\varphi_A>0`$ be the signed-distance
values in the fluid and air cells. The Ghost-Fluid interface fraction is

```math
\theta
=\frac{\varphi_F}{\varphi_F-\varphi_A},
\qquad
0.01\le\theta\le1,
```

and the normal pressure gradient uses the shortened distance $`\theta h`$ with
$`p_A=0`$. Fluid-particle volume remains consistent with mapped porosity through

```math
V_p^f=\frac{m_p^f}{\rho_fn_p}.
```

### 12. Incompressible MAC projection

Incompressible flow satisfies

```math
\nabla\cdot\boldsymbol{u}=0,
```

```math
\frac{\partial\boldsymbol{u}}{\partial t}
+\boldsymbol{u}\cdot\nabla\boldsymbol{u}
=-\frac{1}{\rho}\nabla p
+\nu\nabla^2\boldsymbol{u}
+\boldsymbol{g}+\frac{\boldsymbol{f}}{\rho}.
```

The particle transfer supplies face-centered MAC velocities. A non-pressure
predictor is formed as

```math
\boldsymbol{u}^*
=\boldsymbol{u}^n
+\Delta t
\left(
\nu\nabla^2\boldsymbol{u}^n
+\boldsymbol{g}+\frac{\boldsymbol{f}}{\rho}
\right).
```

Pressure follows from

```math
\nabla\cdot\left(\frac{1}{\rho}\nabla p\right)
=\frac{1}{\Delta t}\nabla\cdot\boldsymbol{u}^*,
```

and the projected velocity is

```math
\boldsymbol{u}^{n+1}
=\boldsymbol{u}^*-\frac{\Delta t}{\rho}\nabla p.
```

The cell-centered level set $`\phi`$ gives

```math
\boldsymbol{n}
=\frac{\nabla\phi}{\|\nabla\phi\|},
\qquad
\kappa=\nabla\cdot\boldsymbol{n},
```

with the free-surface jump

```math
p_f-p_a=\sigma\kappa.
```

The same Ghost-Fluid fraction $`\theta`$ shortens the pressure stencil between a
fluid cell and an air cell.

For a cut cell, let $`A_f`$ be a face area, $`\alpha_f`$ its open fraction,
$`\boldsymbol{u}_f`$ the fluid face velocity, and $`\boldsymbol{u}_s`$ the solid
face velocity. The finite-volume constraint is

```math
\frac{1}{V_c}\sum_{f\in\partial c}A_f
\left[
\alpha_f\boldsymbol{u}_f
+(1-\alpha_f)\boldsymbol{u}_s
\right]\cdot\boldsymbol{n}_f=0.
```

Thus the pressure coefficient across that face is scaled by
$`\alpha_f/(\rho h_f^2)`$.

For volume-fraction immersed boundaries, let $`\phi_s`$ be solid volume
fraction. The mixture density and solid mass fraction are

```math
\rho_m=(1-\phi_s)\rho_f+\phi_s\rho_s,
```

```math
\varphi_s=\frac{\phi_s\rho_s}{\rho_m}.
```

The direct-forcing IBM source is

```math
\boldsymbol{f}_{IBM}
=\rho_m\varphi_s
\frac{\boldsymbol{u}_s-\boldsymbol{u}_f}{\Delta t}.
```

Density projection corrects accumulated quadrature error. With resolved fluid
fraction $`\chi_c`$, define

```math
e_c=\frac{\rho_c}{\rho_0\chi_c}-1.
```

After applying the configured dead band and clamp to obtain
$`\widetilde e_c`$, a scalar correction potential $`q`$ satisfies

```math
-\nabla^2q
=\frac{\rho_0\chi_c}{\Delta t^2}\widetilde e_c,
```

and its face displacement is

```math
\Delta\boldsymbol{x}_f
=-\frac{\Delta t^2}{\rho_0}\nabla q.
```

Particle advection uses a midpoint trajectory,

```math
\boldsymbol{x}_p^{n+1}
=\boldsymbol{x}_p^n
+\Delta t\boldsymbol{u}
\left(
\boldsymbol{x}_p^n
+\frac{\Delta t}{2}\boldsymbol{u}(\boldsymbol{x}_p^n)
\right),
```

which removes the first-order radius growth of a rigid rotation.

### 13. Particle shifting and free-surface geometry

Let the nodal particle volume and its reference capacity be

```math
V_i=\sum_pN_{pi}V_p,
\qquad
V_i^{ref}=\int_{\Omega}N_i(\boldsymbol{x})\,\mathrm{d}v.
```

The overcrowding energy is

```math
E_V
=\sum_i
\left[
\max(0,V_i-V_i^{ref})
\right]^2.
```

Its particle gradient is

```math
\nabla_{\boldsymbol{x}_p}E_V
=2V_p\sum_i
\max(0,V_i-V_i^{ref})\boldsymbol{G}_{pi}.
```

A normalized steepest-descent correction is

```math
\Delta\boldsymbol{x}_p^*
=-\frac{E_V}
{\sum_q\|\nabla_{\boldsymbol{x}_q}E_V\|^2}
\nabla_{\boldsymbol{x}_p}E_V,
```

with the continuous zero correction when the denominator vanishes. In
incompressible flow the accepted shift is capped by a fraction $`\eta_s`$ of
the smallest grid spacing:

```math
\Delta\boldsymbol{x}_p
=
\begin{cases}
\Delta\boldsymbol{x}_p^*,
&\|\Delta\boldsymbol{x}_p^*\|\leq\eta_s h_{\min},\\[0pt]
\eta_s h_{\min}
\dfrac{\Delta\boldsymbol{x}_p^*}
{\|\Delta\boldsymbol{x}_p^*\|},
&\|\Delta\boldsymbol{x}_p^*\|>\eta_s h_{\min}.
\end{cases}
```

Only interior fluid particles are shifted, so this quadrature correction does
not move a particle through a detected free surface. For double-layer
two-phase MPM the same energy acts only on fluid particles; a boundary node
with $`b_i`$ boundary-coordinate directions uses the control-volume capacity

```math
V_i^{ref}=\frac{V_c}{2^{b_i}}.
```

Because shifting is a quadrature correction rather than physical advection,
an APIC particle also receives

```math
\Delta\boldsymbol{v}_p
=\boldsymbol{C}_p\Delta\boldsymbol{x}_p.
```

A density indicator marks a particle as a free-surface candidate when

```math
\frac{\rho_p^{MPM}}{\rho_0}\le0.76.
```

For neighboring particles $`q`$, define

```math
\boldsymbol{M}_p
=\sum_qV_q\nabla W_{pq}
(\boldsymbol{x}_q-\boldsymbol{x}_p)^T,
\qquad
\boldsymbol{b}_p=\sum_qV_q\nabla W_{pq}.
```

The renormalized outward normal is

```math
\boldsymbol{n}_p
=-\frac{\boldsymbol{M}_p^{-1}\boldsymbol{b}_p}
{\|\boldsymbol{M}_p^{-1}\boldsymbol{b}_p\|}.
```

With a Gaussian kernel $`W_{pq}`$, a normalized particle curvature estimate and
surface-tension force density are

```math
\eta_p=\sum_qV_qW_{pq},
\qquad
\kappa_p
=-\frac{1}{\eta_p}
\sum_qV_q
(\boldsymbol{n}_q-\boldsymbol{n}_p)\cdot\nabla W_{pq},
```

```math
\boldsymbol{f}_{\gamma,p}
=\gamma\kappa_p\boldsymbol{n}_p\delta_{\Gamma,p}.
```

A linked-cell search with cell width $`h_s`$ assigns

```math
\boldsymbol{c}_p
=\left\lfloor
\frac{\boldsymbol{x}_p-\boldsymbol{x}_{min}}{h_s}
\right\rfloor,
```

and obtains candidates from the adjacent cell block
$`\|\boldsymbol{c}_q-\boldsymbol{c}_p\|_{\infty}\le1`$. Exact distance and
body-membership tests then remove false candidates.

### 14. Grid contact and boundary conditions

For two bodies sharing node $`i`$, the contact normal is reconstructed from
their mapped domain gradients:

```math
\boldsymbol{n}_i
=\frac{\boldsymbol{g}_i^1-\boldsymbol{g}_i^2}
{\|\boldsymbol{g}_i^1-\boldsymbol{g}_i^2\|},
\qquad
\boldsymbol{g}_i^a
=\sum_{p\in a}V_p\boldsymbol{G}_{pi}.
```

With reduced nodal mass

```math
m_i^*=\frac{m_i^1m_i^2}{m_i^1+m_i^2},
```

contact is active when

```math
(\boldsymbol{v}_i^1-\boldsymbol{v}_i^2)
\cdot\boldsymbol{n}_i>0.
```

The normal force that cancels the approaching normal velocity in one time
step is

```math
\boldsymbol{f}_{n,i}
=\frac{m_i^*}{\Delta t}
\left[
(\boldsymbol{v}_i^1-\boldsymbol{v}_i^2)
\cdot\boldsymbol{n}_i
\right]\boldsymbol{n}_i.
```

The trial tangential force is

```math
\boldsymbol{f}_{t,i}^{trial}
=\frac{m_i^*}{\Delta t}
(\boldsymbol{v}_i^1-\boldsymbol{v}_i^2)
-\boldsymbol{f}_{n,i},
```

and Coulomb projection gives

```math
\boldsymbol{f}_{t,i}
=\min
\left(
1,
\frac{\mu\|\boldsymbol{f}_{n,i}\|}
{\|\boldsymbol{f}_{t,i}^{trial}\|}
\right)
\boldsymbol{f}_{t,i}^{trial}.
```

The two bodies receive equal and opposite
$`\boldsymbol{f}_{n,i}+\boldsymbol{f}_{t,i}`$.

For improved soil-rigid contact, let $`d_e`$ be the rigid material-point offset
from its node and $`h`$ the mean grid spacing. Define

```math
d^*=\left|1-2\left(\frac{-d_e}{1.25h}\right)^{0.58}\right|
\quad \text{for } d_e\le0,
```

```math
d^*=\left|2\left(\frac{d_e}{1.25h}\right)^{0.58}-1\right|
\quad \text{for } d_e>0,
```

and scale the normal force by

```math
\chi_g
=\frac{1-\alpha_g(d^*)^{\beta_g}}
{1+\alpha_g(d^*)^{\beta_g}}.
```

DEM-style polygon and cross-solver contact use the shared scalar laws linked
above. Prescribed velocity imposes
$`\boldsymbol{v}_i\cdot\boldsymbol{e}_a=\bar v_a`$; reflection projects only an
outward velocity; and a friction boundary clips tangential impulse by the
same Coulomb cone. A traction boundary contributes

```math
\boldsymbol{f}_i^t
=\int_{\Gamma_t}N_i\overline{\boldsymbol{t}}\,\mathrm{d}a.
```

For a nonreflecting boundary, decompose velocity and displacement into normal
and tangential parts:

```math
\boldsymbol{v}_n=(\boldsymbol{v}\cdot\boldsymbol{n})\boldsymbol{n},
\qquad
\boldsymbol{v}_t=\boldsymbol{v}-\boldsymbol{v}_n,
```

```math
\boldsymbol{u}_n=(\boldsymbol{u}\cdot\boldsymbol{n})\boldsymbol{n},
\qquad
\boldsymbol{u}_t=\boldsymbol{u}-\boldsymbol{u}_n.
```

With compressional and shear wave speeds $`c_p,c_s`$, the viscous-spring
absorbing traction is

```math
\boldsymbol{t}_{abs}
=-\rho
\left(
a c_p\boldsymbol{v}_n+b c_s\boldsymbol{v}_t
+\frac{c_p^2}{\delta}\boldsymbol{u}_n
+\frac{c_s^2}{\delta}\boldsymbol{u}_t
\right),
```

where $`a`$ and $`b`$ scale normal and tangential absorption and $`\delta`$ is the
absorbing-layer length. Periodic boundaries identify opposite coordinates
and their grid states:

```math
\boldsymbol{x}^{+}
=\boldsymbol{x}^{-}+L_a\boldsymbol{e}_a,
\qquad
\boldsymbol{v}^{+}=\boldsymbol{v}^{-},
\qquad
\boldsymbol{f}^{per}
=\boldsymbol{f}^{+}+\boldsymbol{f}^{-}.
```

### 15. Soft-particle LSMPM and level-set transport

Soft-particle MPM uses a body-fixed reference grid. Its invariant lumped mass
and current nodal force are

```math
m_i=\sum_pN_{pi}m_p,
```

```math
\boldsymbol{f}_i
=\sum_p
\left[
N_{pi}m_p\boldsymbol{g}
-V_{p0}\boldsymbol{P}_p\nabla_0N_{pi}
+N_{pi}\boldsymbol{f}_p^{contact}
+N_{pi}\boldsymbol{f}_p^{ext}
\right].
```

The explicit grid and PIC/FLIP particle updates are

```math
\boldsymbol{v}_i^{n+1}
=\boldsymbol{v}_i^n+\Delta t\frac{\boldsymbol{f}_i}{m_i},
```

```math
\boldsymbol{v}_p^{n+1}
=\alpha_s\sum_iN_{pi}\boldsymbol{v}_i^{n+1}
+(1-\alpha_s)
\left(
\boldsymbol{v}_p^n
+\Delta t\sum_iN_{pi}\frac{\boldsymbol{f}_i}{m_i}
\right).
```

Because gradients are fixed in the template frame, the deformation-gradient
rate is additive:

```math
\dot{\boldsymbol{F}}_p
=\sum_i\boldsymbol{v}_i
\left(\nabla_0N_{pi}\right)^T,
\qquad
\boldsymbol{F}_p^{n+1}
=\boldsymbol{F}_p^n+\Delta t\dot{\boldsymbol{F}}_p.
```

The PK1 stress and strain energy are

```math
\boldsymbol{P}_p=\frac{\partial\Psi}{\partial\boldsymbol{F}_p},
\qquad
E_p=V_{p0}\Psi(\boldsymbol{F}_p).
```

The body-level diagnostic energies are

```math
T=\frac12\sum_pm_p\|\boldsymbol{v}_p\|^2,
\qquad
U=\sum_pV_{p0}\Psi(\boldsymbol{F}_p),
```

```math
\Pi_g=-\sum_pm_p\boldsymbol{g}\cdot\boldsymbol{x}_p,
\qquad
E_{mech}=T+U+\Pi_g+E_{damp}.
```

The body surface is the zero contour of a signed-distance field,

```math
\Gamma(t)=\left\{\boldsymbol{x}:\phi(\boldsymbol{x},t)=0\right\}.
```

The default explicit LSMPM update is `ReferenceMap`. Each body stores its
initial template field $`\phi_{init}`$ once at insertion. Reconstruction uses
the current material-point positions, initial positions, and total deformation
gradients; it does not resample the preceding SDF.

Let $`R_0`$ and $`R`$ be the initial and current body rotations, $`s`$ the constant
template scale, and $`c_0,c`$ the corresponding centers. In unscaled local
coordinates,

```math
X_p=R_0^T(x_{p0}-c_0)/s,\qquad
x_p^{loc}=R^T(x_p-c)/s,\qquad
B_p=R_0^TF_p^{-1}R.
```

At a current SDF node $`x_g`$, the existing linear or B-spline projection
weights $`\omega_{gp}=m_pN_{gp}`$ reconstruct an affine inverse-map patch:

```math
X_g=\frac{\sum_p\omega_{gp}[X_p+B_p(x_g-x_p^{loc})]}
{\sum_p\omega_{gp}},\qquad
B_g=\frac{\sum_p\omega_{gp}B_p}{\sum_p\omega_{gp}}.
```

The inverse displacement $`X_g-x_g`$ and its projected gradient estimate are
extended into uncovered neighboring cells by synchronous affine extrapolation.
Trilinear sampling then gives the raw pullback and the locally corrected field:

```math
\psi_g=\phi_{init}(X_g),\qquad
n_0=\frac{\nabla_X\phi_{init}}{\|\nabla_X\phi_{init}\|},\qquad
\phi_g=\frac{\psi_g}{\|B_g^Tn_0(X_g)\|}.
```

This is a reference-map form of semi-Lagrangian pullback, adapted to material
points by grid projection rather than the source paper's k-nearest-neighbor
interpolator. The metric factor is inverse normal stretch, not $`\det F`$.
It is exact for an affine deformation of a plane with exact reference data;
curved surfaces, changing closest points, and nonuniform motion introduce
approximation error. Projected $`B_g`$ is an estimate and need not equal the
spatial derivative of the interpolated $`X_g`$. Downstream normals are computed
from the reconstructed grid field. Unsupported exterior nodes receive a
positive cap; the full far-field field is not guaranteed to be a Euclidean SDF.
The implementation rejects nonfinite/nonpositive particle determinants,
nonpositive inverse Jacobians near the interface, and a support halo cutting
the material interior. Near-interface reference-domain departures use the
existing domain audit.

For explicit LSMPM configuration through the coupling-owned DEM facade:

```python
soft_levelset_reinitialization={
    "advection_scheme": "ReferenceMap",
    "enabled": False,
    "volume_correction": False,
}
```

`ReferenceMap` is the default; `InverseMapping` and `SemiLagrangian` are aliases
for it. `MacCormack` explicitly selects the previous RK2-backtraced, bounded
MacCormack update. `WENO5` selects WENO5--SSP-RK3 with default CFL `0.20`.
The generic `src/levelset/SemiLagrangian.py` class is a separate interface.
The update interval defaults to one mechanical step. Reference-map updates
reconstruct total motion at the scheduled step; their CFL option does not
subcycle the inverse map. Choosing a scheme without an explicit `enabled`
value disables redistance for `ReferenceMap` and enables it for `MacCormack`
or `WENO5`. Explicit maintenance settings take precedence.

The two incremental transport alternatives project material velocity and solve

```math
\frac{\partial\phi}{\partial t}
+\boldsymbol{u}\cdot\nabla\phi=0.
```

MacCormack or WENO advection supplies the transport discretization. Optional
pseudo-time reinitialization restores distance scaling when
$`\|\nabla\phi\|`$ drifts from unity:

```math
\frac{\partial\phi}{\partial\tau}
+S(\phi_0)(\|\nabla\phi\|-1)=0,
```

```math
S(\phi_0)
=\frac{\phi_0}
{\sqrt{\phi_0^2+h^2\|\nabla\phi_0\|^2}}.
```

Volume correction applies a uniform level-set shift $`c`$ such that

```math
V(c)=\int H_{\epsilon}[-(\phi+c)]\,\mathrm{d}v=V_{target}.
```

A safeguarded Newton step uses

```math
c_{k+1}
=c_k+\frac{V(c_k)-V_{target}}{A_{\Gamma}(c_k)},
```

where

```math
A_{\Gamma}(c)
=\int\delta_{\epsilon}(\phi+c)\,\mathrm{d}v.
```

Redistance and volume shifts modify the current field without overwriting
$`\phi_{init}`$. They can move the zero contour, so `ReferenceMap` defaults to
redistance disabled and volume correction remains opt-in. Geometry-only
checks of the production reconstruction on the four-corner Neo-Hookean
rectangle give final distance MAE $`0.00524h`$ and grid-normal angular-error
95th percentile $`1.73^\circ`$ at SDF spacing $`h=0.02`$; this is one benchmark,
not a general accuracy guarantee. Reproduction and scope are recorded in
`research/LSMPM/dai2026_reference_map_production/implementation_notes.txt`.

### 16. Sparse and adaptive grids

Sparse storage changes only the set of allocated nodes. The active set is the
union of particle supports,

```math
\mathcal{A}
=\bigcup_p
\left\{i:N_{pi}\ne0\right\},
```

so all transfer and balance equations above remain unchanged.

If $`\widehat{\boldsymbol{u}}`$ contains only independent degrees of freedom,
hanging-node constraints have the prolongation form

```math
\boldsymbol{u}=\boldsymbol{P}\widehat{\boldsymbol{u}}.
```

Residuals and tangents reduce consistently as

```math
\widehat{\boldsymbol{R}}=\boldsymbol{P}^T\boldsymbol{R},
\qquad
\widehat{\boldsymbol{K}}
=\boldsymbol{P}^T\boldsymbol{K}\boldsymbol{P}.
```

### 17. Ordinary Barrier IPC and friction

For an MPM surface sample $`s`$, freeze its transfer weights during a Newton
step and write

```math
\boldsymbol{x}_s(\boldsymbol{u})
=\boldsymbol{x}_{s,n}+\sum_iN_{si}\boldsymbol{u}_i.
```

The contact-augmented incremental potential is

```math
\Pi_{IPC}(\boldsymbol{u})
=\Pi_{MPM}(\boldsymbol{u})
+\sum_{c\in\mathcal{C}}A_cb(s_c)
+\Pi_f(\boldsymbol{u}).
```

Here

```math
s_c=d_c^2-d_{min}^2,
\qquad
\widehat{s}=(d_{min}+\widehat d)^2-d_{min}^2,
```

$`A_c`$ is the point or surface measure, $`b`$ is the finite-clearance IPC
barrier with activation value $`\widehat s`$, and $`\Pi_f`$ is regularized
Coulomb friction. Their scalar equations, gradients, and Hessians are in the
[IPC contact theory](../physics_model/contact_model/README.md#incremental-potential-contact).

Newton solves

```math
\left[
\boldsymbol{H}_{MPM}
+\boldsymbol{H}_{IPC}
+\boldsymbol{H}_{f}
\right]\Delta\boldsymbol{u}
=-
\left[
\boldsymbol{g}_{MPM}
+\boldsymbol{g}_{IPC}
+\boldsymbol{g}_{f}
\right].
```

Continuous collision detection bounds the line-search fraction by

```math
0<\alpha
\le\min(1,\alpha_{contact},\alpha_{material}),
```

so every accepted trial satisfies

```math
d_c(\alpha)>d_{min},
\qquad
\det\boldsymbol{F}_p(\alpha)>0.
```

Lagged friction freezes the active contact, tangent frame, and normal-force
magnitude inside one inner Newton solve, then refreshes them in an outer
fixed-point iteration.

### 18. Time-step limits

For explicit solid MPM, the acoustic CFL estimate is

```math
\Delta t_{CFL}
=C_{CFL}
\frac{h_{min}}
{\max_p\|\boldsymbol{v}_p\|+c_{max}},
```

where $`c_{max}`$ is the largest material wave speed. A purely kinematic
adaptive estimate uses

```math
\Delta t_{adv}
=C_{CFL}\frac{h_{min}}
{\max_p\|\boldsymbol{v}_p\|}.
```

For viscous incompressible flow, an explicit diffusion limit is

```math
\Delta t_{\nu}
\le\frac{h_{min}^2}{2d\nu_{max}}.
```

The operative explicit step is the minimum applicable acoustic, advective,
viscous, contact, and user limits. Implicit line searches may reduce a failed
trial step, but they do not remove material-inversion or contact-feasibility
constraints.

## Grid-based example

```python
import geotaichi as gt

gt.init(dim=2, arch="gpu", default_fp="float32", log=False)

mpm = gt.MPM(log=False)
mpm.set_configuration(
    dimension=2,
    domain=[1.0, 0.6],
    gravity=[0.0, -9.81],
    mapping="MUSL",
    shape_function="Linear",
    solver_type="Implicit",
    configuration="ULMPM",
    material_type="Solid",
    visualize=False,
    log=False,
)
mpm.set_implicit_solver_parameters(
    assemble_type="HashTriplet",
    quasi_static=False,
    max_iteration_number=20,
    residual_tolerance=1.0e-6,
)
mpm.set_solver(
    solver={
        "Timestep": 1.0e-4,
        "SimulationTime": 0.1,
        "SaveInterval": 1.0e-2,
        "SavePath": "OutputData/mpm_case",
    },
    log=False,
)
mpm.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": 100000,
        "max_constraint_number": {"max_displacement_constraint": 10000},
    },
    log=False,
)
mpm.add_material(
    model="DruckerPrager",
    material={
        "MaterialID": 1,
        "Density": 2500.0,
        "YoungModulus": 1.0e6,
        "PoissonRatio": 0.3,
        "Friction": 30.0,
        "Cohesion": 0.0,
        "Dilation": 0.0,
    },
)
mpm.add_element({"ElementType": "Q4N2D", "ElementSize": [0.02, 0.02]})
# Add a region, particle body, and boundary conditions before running.
mpm.run()
```

The dictionary input format is retained for compatibility with existing
GeoTaichi cases. The linked Python examples demonstrate the supported solver configurations; tests are separate verification and integration checks.

## Particle construction and implicit examples

The [implicit block example](../../examples/mpm/direct/implicit_ulmpm_block2d.py) constructs material points through `create_body()`. Its configuration includes `mpm_backend="Direct"` because that input API currently resides in a separate implementation; this is an implementation selector, not a separate MPM theory. Other examples use region/template input. Follow a complete example rather than mixing the two input lifecycles.

The example-backed implicit ULMPM materials include Neo-Hookean elasticity, finite-strain Drucker–Prager, and von Mises plasticity. The [implicit elastic bar](../../examples/mpm/ElasticBar/ImplicitBar.py) uses linear elasticity. Cross-solver contact is documented under [FEM–MPM](../fempm/README.md), [IGA–MPM](../igampm/README.md), and [MPM–ABD](../mpdem/README.md).

Direct nonassociated Drucker–Prager accepts independent `DilationAngle` and
keeps symmetric PCG solves by freezing its flow correction in each inner
potential. The outer material convergence test compares physical and inner
Kirchhoff stress, together with predicted plastic volume, rather than changes
of the correction alone.
Particle-local Aitken relaxation damps predictor updates; acceptance still
checks their unrelaxed physical mismatch. See the
[shared constitutive theory](../physics_model/consititutive_model/README.md#nonassociated-drucker--prager-and-symmetric-inner-solves).

The [axisymmetric annulus](../../examples/mpm/AxisyExample/axisymmetric_annulus.py) selects explicit or implicit time integration with `--solver explicit|implicit`. Its coordinates are `(r,z)` and require `radius > axis_offset`.

The linked implicit IPC examples describe accepted-state rollback and bounded timestep retry controls. These are solver-specific runtime options, not additional MPM formulations.

## Sparse and adaptive grids

Block-sparse storage is configured through `sparse_grid={...}`. Adaptive
elements rebuild topology when refinement changes, while particle marking,
counting, transfer, constitutive updates, residuals, and linear algebra remain
device operations. Sparse capacity and adaptive constraint options should be
chosen before the engine is built.

## Numerical backend

Particle-grid transfers, stress updates, force assembly, grid solves, state
updates, IPC contact, and default implicit Krylov iterations execute in
Taichi kernels. SciPy is available only when an API explicitly selects a host
linear solver or requests sparse output. Mesh/particle import, topology
preprocessing, recording, and diagnostics remain host-side boundaries.

Every public `MPM` facade exposes `diagnostics_snapshot()`. Implicit engines add
accepted-step/retry/contact details; native engines still report common time,
step, timestep, target, and terminal task diagnostics.

Soft-particle IPC retains the scalar energy of an accepted Armijo trial instead
of evaluating the same trial again for logging. Incompressible MPM reapplies
particle boundary projection after particle shifting only when shifting is
enabled; the disabled callback adds no extra grid/particle traversal.

### Incompressible immersed boundaries

- `ibm_field_function` supplies solid fraction, density, and velocity for a
  fixed SDF-derived immersed rigid body. Moving LSDEM bodies populate the same
  fields internally. Fictitious fluid remains inside these bodies. Legacy
  aliases are `ibm_source_function` and `ibm_function`.
- `cut_cell_function` updates the SDF or face velocity of an irregular wall
  and requires `solid_sdf_cut_cell=True`. Its legacy alias is
  `solid_sdf_function`.
- IBM and cut-cell may coexist for different geometry, but the same physical
  surface must not be represented by both.

## Tests

Unit coverage is under `tests/unit/mpm/`; integration and verification cases
are under `tests/integration/mpm/` and `tests/verification/mpm/`.
