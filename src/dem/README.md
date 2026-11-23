# Discrete Element Method (DEM), Level-Set DEM (LSDEM), and Affine Body Dynamics (ABD)

`src/dem` contains GeoTaichi's discrete-element solvers and the public `DEM`
facade. It supports spherical and clumped particles, level-set rigid bodies,
affine deformable bodies, walls, several neighbor-search backends, and a range
of contact laws.

[Theory log and derivations](#dem-theory-log) | [Examples](../../examples/)

## Example-backed capabilities

- Spherical and clumped DEM: [sphere packing](../../examples/dem/MultiSphere/RotatingDrum/sphere_packing.py) and [clump packing](../../examples/dem/MultiSphere/GranularPackings/clump_packing.py).
- Level-set discrete element method (LSDEM) for irregular particles: [polydisperse packing](../../examples/dem/LevelSet/GranularAssemble/polydisperse/packing_generate.py), [screw and nut](../../examples/dem/LevelSet/ParticleParticle/screw_and_nut.py), and [rotating drum](../../examples/dem/LevelSet/RotatingDrum/rotating_drum.py).
- Affine body dynamics (ABD) with incremental potential contact (IPC): [sphere–wall collision](../../examples/dem/AffineBody/sphere_wall_collision/sphere_wall_collision.py) and [inclined-plane sliding](../../examples/dem/AffineBody/cube_incline_sliding/cube_incline_sliding.py).
- ABD joints and motors: [multilink arm](../../examples/dem/AffineBody/robot_multilink_arm/robot_multilink_arm.py).
- Pure ABD grasp/lift/hold/release with a free payload: [minimal gripper](../../examples/dem/AffineBody/robot_grasp/README.md).
- Level-set MPM soft particles: [rigid–soft contact](../../examples/mpdem/LevelSet/SoftRigid/rigid_soft_sphere_drop_box.py).

## Package layout

| Path | Responsibility |
| --- | --- |
| `generator/` | Regions, clump/level-set/affine templates, bodies, walls, and restart input |
| `structs/` | Taichi particle, rigid-body, wall, contact, and level-set fields |
| `neighbor/` | Neighbor search and Verlet/contact candidate management |
| `contact/` | DEM contact models and force assembly |
| `engines/` | Time integration and affine nonlinear solvers |
| `mainDEM.py` | Public scene facade |

Shared constitutive and contact formulas live in `src/physics_model`, while
BVH and CCD primitives live in `src/contact_detection`.

`generator/AffineBodyTemplate.py` prepares a copied, centered surface mesh and
its mass properties without building an SDF. `GeneralShapeTemplate.py` prepares
level-set objects and their grids. Both use the shared host surface quadrature
in `src/utils/linalg.py`; their different generation and coordinate-transform
contracts remain separate.

Local contact finite-difference probes and NumPy surface-correction measures
live in `tests/helpers/affine_contact_probe.py`. Production stepping uses the
device measures and body mass blocks; dense reference mass matrices are built
only by the test helpers that consume them.

AffineBody time integration is selected with
`set_affine_body_parameters(newmark=[alpha, beta, gamma])`. Exactly three
finite positive values are required, using the IGA/Direct MPM convention:
`q_pred=q_n+h*v_n+alpha*(1-2*beta)*h²*a_n`,
`a_next=(q_next-q_pred)/(2*alpha*beta*h²)` and
`v_next=v_n+h*((1-gamma)*a_n+gamma*a_next)`.
The default `[1, 0.5, 1]` is backward Euler;
`[0.5, 0.25, 0.5]` is average acceleration.

Fully implicit friction uses force tolerances
`fully_implicit_force_atol=1e-9` N and `fully_implicit_force_rtol=1e-8`.
For the incremental residual, convergence requires
`max(abs(R)) <= 2*alpha*beta*h²*atol + rtol*max(abs(R_initial))`.
The absolute default accounts for f64 rounding when metre-scale absolute
coordinates are subtracted at sub-millisecond time steps. Other mass, length,
or time scales require case-specific tolerances. Explicit settings remain
authoritative; no hidden precision floor replaces them. Lagged friction keeps
its separate correction-velocity criterion and defaults.

The original ABD operator uses this integration for mesh IPC, local/contact/
joint damping, wall motion and FEM/Direct-MPM/soft-particle coupling. Lagged
friction keeps its existing displacement-increment potential; fully implicit
friction and damping use the endpoint velocity and its Newmark derivative.
Coupling scales the ABD residual by `2*alpha*beta*h²`. The existing restriction
to lagged friction in FEM/Direct-MPM coupling is unchanged. ABD trajectory
adjoints remain backward-Euler only.
Initial acceleration comes from the device mass matrix and initial physical
forces. Accepted acceleration and velocity stay on device and survive failed
trial steps; saved Newmark frames include `a_y` and `newmark`.
Output schedules include the first accepted step used for JIT warmup, also
when continuing a saved run.
The lagged COO, MatrixFree, and HashTriplet solves in `AffineBodyEngine` precondition each body's
complete 12-DOF principal block with device Cholesky, retaining coupling between the four
control points. This resolves stiff body modes that separate 3x3 control-point
blocks can converge on very slowly; the configured PCG and Newton tolerances
remain authoritative.
The exact ABD adjoint also retains those body blocks. If its unprojected
tangent has indefinite principal blocks, the existing BiCGSTAB fallback uses
pivoted LU on the same blocks; the adjoint does not silently downgrade to
separate control-point blocks. Linear residual tolerances remain explicit.
After the configured Newton backtracking search is exhausted or an accepted
energy trial stagnates despite a resolved descent prediction, lagged ABD
retries the search once with a quadratic upper bound of the frozen Coulomb
potential. Switching after only two rejected trials can stall near the sliding
threshold; ordinary Newton first gets its full search budget to reach the
regularized transition. Concavity of
`f0(sqrt(t))` supplies the additional radial search curvature in sliding
contacts. Energy and gradient are unchanged. The accepted iterate is assembled
with the original Hessian again, so Newton stopping and the physical adjoint
tangent retain their original definitions. This search safeguard changes no
friction parameters or convergence thresholds.
The [scalar near-threshold regression](../../tests/unit/dem/test_affine_solver_regressions.py)
and the [grasp/release example](../../examples/dem/AffineBody/robot_grasp/README.md)
exercise this search behavior.
The [Newton cradle example](../../examples/dem/AffineBody/newtons_cradle/README.md)
compares integration dissipation and checks time-step refinement before its
queued formal run.
Mesh IPC prepares a barrier-padded swept envelope for each CCD-clipped
line-search segment. Backtracking trials and the accepted Hessian reuse its
candidates, evaluating contact at the current trial position. A changed search
direction, rollback or standalone CCD query prepares its own candidates;
level-set contact keeps its existing body-pair refresh.

## DEM theory log

### 1. Rigid-particle balance laws

For a rigid particle $`i`$, let $`m_i`$, $`\boldsymbol{x}_i`$,
$`\boldsymbol{v}_i`$, and $`\boldsymbol{\omega}_i`$ denote its mass, center of
mass, translational velocity, and spatial angular velocity. Its Newton--Euler
equations are

```math
\dot{\boldsymbol{x}}_i=\boldsymbol{v}_i,
\qquad
m_i\dot{\boldsymbol{v}}_i=\boldsymbol{F}_i,
\qquad
\dot{\boldsymbol{L}}_i=\boldsymbol{M}_i.
```

With $`\mathcal C_i`$ the active contacts of particle $`i`$, the resultant
force and moment are

```math
\boldsymbol{F}_i
=m_i\boldsymbol{g}+\boldsymbol{F}_i^{\mathrm{ext}}
+\sum_{c\in\mathcal C_i}
\left(\boldsymbol{F}_{n,c}+\boldsymbol{F}_{t,c}\right).
```

```math
\boldsymbol{M}_i
=\boldsymbol{M}_i^{\mathrm{ext}}
+\sum_{c\in\mathcal C_i}
\left[
\left(\boldsymbol{x}_c-\boldsymbol{x}_i\right)
\times\left(\boldsymbol{F}_{n,c}+\boldsymbol{F}_{t,c}\right)
+\boldsymbol{M}_{r,c}+\boldsymbol{M}_{tw,c}
\right].
```

Here $`\boldsymbol{x}_c`$ is the contact point;
$`\boldsymbol{F}_{n,c}`$ and $`\boldsymbol{F}_{t,c}`$ are the normal and
tangential contact forces; and $`\boldsymbol{M}_{r,c}`$ and
$`\boldsymbol{M}_{tw,c}`$ are optional rolling and twisting moments. A simple
contact law sets the last two terms to zero.

Let $`\boldsymbol{R}_i`$ rotate body-frame vectors into the spatial frame and
define

```math
\boldsymbol{\omega}_i^b=\boldsymbol{R}_i^T\boldsymbol{\omega}_i,
\qquad
\boldsymbol{M}_i^b=\boldsymbol{R}_i^T\boldsymbol{M}_i.
```

Because the body-frame inertia $`\boldsymbol{I}_i^b`$ is constant, rotational
balance can be written as Euler's rigid-body equation

```math
\boldsymbol{I}_i^b\dot{\boldsymbol{\omega}}_i^b
+\boldsymbol{\omega}_i^b\times
 \left(\boldsymbol{I}_i^b\boldsymbol{\omega}_i^b\right)
=\boldsymbol{M}_i^b,
\qquad
\dot{\boldsymbol{R}}_i
=\boldsymbol{R}_i[\boldsymbol{\omega}_i^b]_{\times},
```

where $`[\boldsymbol{a}]_{\times}\boldsymbol{b}=\boldsymbol{a}\times\boldsymbol{b}`$.
Equivalently,

```math
\boldsymbol{L}_i
=\boldsymbol{R}_i\boldsymbol{I}_i^b\boldsymbol{\omega}_i^b.
```

For a body occupying $`\Omega_i`$ with density $`\rho(\boldsymbol{x})`$, its
mass properties are

```math
m_i=\int_{\Omega_i}\rho\,\mathrm dV.
```

```math
\boldsymbol{x}_i
=\frac{1}{m_i}\int_{\Omega_i}\rho\boldsymbol{x}\,\mathrm dV.
```

```math
\boldsymbol{I}_i
=\int_{\Omega_i}\rho
\left[
\|\boldsymbol{r}\|^2\boldsymbol{1}
-\boldsymbol{r}\otimes\boldsymbol{r}
\right]\mathrm dV,
\qquad
\boldsymbol{r}=\boldsymbol{x}-\boldsymbol{x}_i.
```

For a homogeneous sphere of radius $`R_i`$, these reduce to

```math
m_i=\frac{4}{3}\pi\rho_iR_i^3,
\qquad
\boldsymbol{I}_i^b=\frac{2}{5}m_iR_i^2\boldsymbol{1}.
```

For a clump represented by overlapping pebbles, $`\Omega_i`$ is the union of
the pebble volumes. The union must be integrated only once; summing the mass
and inertia of every full pebble would double-count overlap.

#### Local non-viscous damping

Local damping acts componentwise on the current resultant load. In the frame
where the balance equation is evaluated, define

```math
\mathcal D_{\alpha}(f_k,u_k)
=f_k\left[1-\alpha\,\mathrm{sgn}(f_ku_k)\right],
\qquad k\in\{1,2,3\},
```

with $`\mathrm{sgn}(0)=0`$. The damped force and moment are therefore

```math
F_{i,k}^{d}=\mathcal D_{\alpha_f}(F_{i,k},v_{i,k}),
\qquad
M_{i,k}^{d}=\mathcal D_{\alpha_t}(M_{i,k},\omega_{i,k}).
```

### 2. Contact geometry and force transfer

The shared [contact-model theory](../physics_model/contact_model/README.md)
contains sphere/wall kinematics, Linear, Hertz--Mindlin, energy-conserving
penalty, Coulomb, rolling, twisting, BarrierIPC, and regularized
friction formulas. DEM owns neighbor discovery and rigid-particle assembly,
not separate copies of those constitutive laws.

For an active pair, the contact model returns the force on particle $`i`$,

```math
\boldsymbol{F}_{ij}^c
=\boldsymbol{F}_n+\boldsymbol{F}_t,
```

and optional rolling/twisting resistance $`\boldsymbol{M}_{ij}^c`$. With
contact arms
$`\boldsymbol{r}_i=\boldsymbol{x}_c-\boldsymbol{x}_i`$ and
$`\boldsymbol{r}_j=\boldsymbol{x}_c-\boldsymbol{x}_j`$, pair contributions
to the rigid-particle balances are

```math
\boldsymbol{F}_i^c=\boldsymbol{F}_{ij}^c,
\qquad
\boldsymbol{F}_j^c=-\boldsymbol{F}_{ij}^c,
```

```math
\boldsymbol{M}_i^c
=\boldsymbol{r}_i\times\boldsymbol{F}_{ij}^c
+\boldsymbol{M}_{ij}^c,
```

```math
\boldsymbol{M}_j^c
=-\boldsymbol{r}_j\times\boldsymbol{F}_{ij}^c
-\boldsymbol{M}_{ij}^c.
```

Thus internal contact forces cancel exactly. The remaining angular-momentum
balance follows from using one common contact point and equal/opposite contact
moments.

### 3. Neighbor search and Verlet lists

#### Verlet skin

Let $`s>0`$ be the full pairwise Verlet skin. A sphere pair is stored as a
candidate when

```math
\|\boldsymbol{x}_i-\boldsymbol{x}_j\|
\leq R_i+R_j+s.
```

Suppose $`\Delta\boldsymbol{x}_i`$ is the displacement since the list was
built. No omitted pair can enter contact while

```math
\|\Delta\boldsymbol{x}_i\|
+\|\Delta\boldsymbol{x}_j\|<s
```

for every pair. A simple global sufficient condition is

```math
\max_i\|\Delta\boldsymbol{x}_i\|<\frac{s}{2}.
```

For a non-spherical rigid body with bounding radius $`r_i^b`$, translation
alone is insufficient. If its accumulated rotation angle is
$`\Delta\theta_i`$, a conservative surface-displacement bound is

```math
u_i
\leq
\|\Delta\boldsymbol{x}_i\|
+2r_i^b\sin\left(\frac{|\Delta\theta_i|}{2}\right).
```

The safe pair condition then becomes $`u_i+u_j<s`$. For deformable bodies, the
bound must additionally include outward growth of the bounding volume.

#### Uniform linked cells

For maximum particle radius $`R_{max}`$, choose the cubic cell width

```math
h\geq2R_{max}+s.
```

The integer cell coordinate of a particle is

```math
\boldsymbol{c}_i
=\left\lfloor
\frac{\boldsymbol{x}_i-\boldsymbol{x}_{min}}{h}
\right\rfloor.
```

All possible neighbors lie in the particle's own cell or one of the adjacent
cells

```math
\boldsymbol{c}_j
\in
\boldsymbol{c}_i+\{-1,0,1\}^d,
```

where $`d=2`$ or $`3`$. The exact Verlet-distance test is still required
after the cell lookup; the grid only removes impossible pairs.

For strongly polydisperse particles, one grid wastes work because $`h`$ is
set by the largest radius. A hierarchical grid assigns particle $`i`$ to the
smallest level $`\ell`$ satisfying

```math
R_i\leq R_\ell,
\qquad
h_\ell=2R_\ell+s.
```

Pairs are searched within a level and across compatible coarser levels, then
filtered by the exact distance inequality. This retains small cells for small
particles without missing large--small contacts.

#### Bounding-volume hierarchy

An alternative broad phase encloses each object in a skin-expanded axis-aligned
box

```math
\boldsymbol{B}_i
=
\left[
\boldsymbol{x}_i-\left(R_i+\frac{s}{2}\right)\boldsymbol{1},
\;
\boldsymbol{x}_i+\left(R_i+\frac{s}{2}\right)\boldsymbol{1}
\right].
```

Two boxes overlap when, in every coordinate $`k`$,

```math
\max(B_{i,k}^{min},B_{j,k}^{min})
\leq
\min(B_{i,k}^{max},B_{j,k}^{max}).
```

A linear BVH orders box centroids by a Morton key. If
$`\boldsymbol{u}_i\in[0,1)^3`$ is the normalized centroid and $`Q`$ is an
integer quantizer, then

```math
\boldsymbol{q}_i=\lfloor Q\boldsymbol{u}_i\rfloor,
\qquad
z_i=\mathrm{interleave}(q_{i,x},q_{i,y},q_{i,z}).
```

The binary tree is formed from common prefixes of the sorted $`z_i`$. Query
traversal rejects a complete subtree whenever its box does not overlap the
query box. The resulting candidate set must still pass the exact geometric
test.

### 4. Level-set DEM

#### Signed-distance representation

A level-set rigid body is described in its material frame by a signed-distance
field

```math
\phi^0(\boldsymbol{X})<0
\quad\text{inside the body},
```

```math
\phi^0(\boldsymbol{X})=0
\quad\text{on the surface},
```

```math
\phi^0(\boldsymbol{X})>0
\quad\text{outside the body}.
```

For an exact signed-distance field,

```math
\|\nabla_{\boldsymbol{X}}\phi^0\|=1
```

away from medial-axis singularities. Under translation
$`\boldsymbol{x}_i`$, rotation $`\boldsymbol{R}_i`$, and uniform scale $`s_i`$,

```math
\boldsymbol{X}
=\frac{1}{s_i}\boldsymbol{R}_i^T
\left(\boldsymbol{x}-\boldsymbol{x}_i\right),
```

```math
\phi_i(\boldsymbol{x})=s_i\phi_i^0(\boldsymbol{X}),
\qquad
\nabla_{\boldsymbol{x}}\phi_i
=\boldsymbol{R}_i\nabla_{\boldsymbol{X}}\phi_i^0.
```

#### Trilinear SDF interpolation

Inside one Cartesian grid cell, let
$`\boldsymbol{\xi}=(\xi,\eta,\zeta)\in[0,1]^3`$ be the reduced coordinate and
define

```math
N_0(u)=1-u,
\qquad
N_1(u)=u.
```

The trilinear signed distance is

```math
\phi_h(\boldsymbol{\xi})
=\sum_{a,b,c\in\{0,1\}}
\phi_{abc}N_a(\xi)N_b(\eta)N_c(\zeta).
```

For grid spacing $`h`$, its first gradient component is

```math
\frac{\partial\phi_h}{\partial x}
=\frac{1}{h}
\sum_{a,b,c\in\{0,1\}}
(2a-1)\phi_{abc}N_b(\eta)N_c(\zeta),
```

with cyclic expressions for the other two components. The outward normal is

```math
\boldsymbol{n}
=\frac{\nabla\phi_h}{\|\nabla\phi_h\|}.
```

Trilinear interpolation is continuous, but its gradient is generally
discontinuous across cell boundaries. Grid resolution therefore controls both
surface accuracy and contact-normal smoothness.

#### Surface-node contact

Let $`\boldsymbol{p}_a`$ be a quadrature point on body $`i`$, evaluated in the
signed-distance field of body $`j`$. Its signed gap is

```math
g_a=\phi_j(\boldsymbol{p}_a).
```

For $`g_a<0`$, the penetration, outward normal, closest-point approximation,
and symmetric contact point are

```math
\delta_a=-g_a,
\qquad
\boldsymbol{n}_a
=\frac{\nabla\phi_j(\boldsymbol{p}_a)}
       {\|\nabla\phi_j(\boldsymbol{p}_a)\|},
```

```math
\overline{\boldsymbol{p}}_a
=\boldsymbol{p}_a-g_a\boldsymbol{n}_a,
\qquad
\boldsymbol{x}_{c,a}
=\boldsymbol{p}_a-\frac{g_a}{2}\boldsymbol{n}_a.
```

If $`w_a`$ is the surface quadrature weight, the resultant contact force and
moment are

```math
\boldsymbol{F}_{ij}
=\sum_a w_a
\left(
f_{n,a}\boldsymbol{n}_a+\boldsymbol{F}_{t,a}
\right),
```

```math
\boldsymbol{M}_{ij}
=\sum_a w_a
\left(\boldsymbol{x}_{c,a}-\boldsymbol{x}_i\right)
\times
\left(
f_{n,a}\boldsymbol{n}_a+\boldsymbol{F}_{t,a}
\right).
```

For a penalty potential $`\Psi(g)`$, a compact conservative form is

```math
E_{ij}
=\sum_a w_a\Psi(g_a),
\qquad
\Psi(g)=\frac{1}{2}k_n\langle-g\rangle_+^2.
```

The nodal contact force follows from

```math
\boldsymbol{f}_a
=-\frac{\partial E_{ij}}{\partial\boldsymbol{p}_a}
=k_n\langle-g_a\rangle_+\boldsymbol{n}_a
```

when $`\phi`$ is an exact signed-distance field. Tangential, rolling, and
twisting laws use the same contact-frame kinematics as sphere contact.

#### Two-level LSDEM search

Level-set contact uses two conservative filters. First, body bounding volumes
must overlap after expansion by the body skin $`s_b`$. Second, a surface point
is retained only when

```math
\phi_j(\boldsymbol{p}_a)<s_p,
```

where $`s_p`$ is the point-level skin. The body list is rebuilt when its
bounding-volume displacement exhausts $`s_b`$; the point list is rebuilt when
the accumulated relative motion at a candidate point exhausts $`s_p`$. This
separates inexpensive rigid-body culling from the more expensive SDF queries.

### 5. Affine-body mechanics and IPC

#### Affine kinematics

Let a body use four vector controls
$`\boldsymbol{y}_0,\boldsymbol{y}_1,\boldsymbol{y}_2,\boldsymbol{y}_3`$.
For material coordinate
$`\boldsymbol{X}=(X_1,X_2,X_3)`$, define

```math
w_0=1-X_1-X_2-X_3,
\qquad
w_1=X_1,
\qquad
w_2=X_2,
\qquad
w_3=X_3.
```

The current position and velocity are

```math
\boldsymbol{x}(\boldsymbol{X})
=\sum_{a=0}^{3}w_a(\boldsymbol{X})\boldsymbol{y}_a,
\qquad
\boldsymbol{v}(\boldsymbol{X})
=\sum_{a=0}^{3}w_a(\boldsymbol{X})\dot{\boldsymbol{y}}_a.
```

Equivalently,

```math
\boldsymbol{x}(\boldsymbol{X})
=\boldsymbol{y}_0+\boldsymbol{F}\boldsymbol{X},
\qquad
\boldsymbol{F}
=
\left[
\boldsymbol{y}_1-\boldsymbol{y}_0,\;
\boldsymbol{y}_2-\boldsymbol{y}_0,\;
\boldsymbol{y}_3-\boldsymbol{y}_0
\right].
```

The map contains translation, rotation, stretch, and shear while retaining
only twelve scalar degrees of freedom per body.

The consistent $`4\times4`$ control mass matrix is

```math
M_{ab}
=\int_{\Omega_0}\rho_0w_aw_b\,\mathrm dV.
```

The kinetic energy is

```math
T
=\frac{1}{2}
\sum_{a=0}^{3}\sum_{b=0}^{3}
M_{ab}\,
\dot{\boldsymbol{y}}_a\cdot\dot{\boldsymbol{y}}_b.
```

For a body force $`\boldsymbol{b}`$, the generalized force is

```math
\boldsymbol{f}_a
=\int_{\Omega_0}\rho_0w_a\boldsymbol{b}\,\mathrm dV.
```

#### Affine rigidity

A rotation-invariant rigidity potential is

```math
E_r(\boldsymbol{F})
=\frac{VE}{8}
\left\|
\boldsymbol{F}^T\boldsymbol{F}-\boldsymbol{1}
\right\|_F^2,
```

where $`V`$ is the reference volume and $`E`$ controls affine rigidity. It
vanishes for every $`\boldsymbol{F}\in SO(3)`$, penalizes stretch and shear,
and does not penalize rigid rotation.

Let

```math
\boldsymbol{C}
=\boldsymbol{F}^T\boldsymbol{F}-\boldsymbol{1}.
```

The derivative with respect to the affine matrix is

```math
\frac{\partial E_r}{\partial\boldsymbol{F}}
=\frac{VE}{2}\boldsymbol{F}\boldsymbol{C}.
```

#### Incremental potential

With

```math
\widetilde{\boldsymbol{y}}
=\boldsymbol{y}^n+\Delta t\,\dot{\boldsymbol{y}}^n,
```

one implicit step minimizes

```math
\Pi(\boldsymbol{y})
=\frac{1}{2}
\|\boldsymbol{y}-\widetilde{\boldsymbol{y}}\|_{\boldsymbol{M}}^2
+\Delta t^2
\left[
E_r(\boldsymbol{y})
+E_c(\boldsymbol{y})
+E_j(\boldsymbol{y})
+U_{ext}(\boldsymbol{y})
\right]
+D(\boldsymbol{y};\boldsymbol{y}^n).
```

Here

```math
\|\boldsymbol{z}\|_{\boldsymbol{M}}^2
=\boldsymbol{z}^T\boldsymbol{M}\boldsymbol{z},
\qquad
U_{ext}
=-\sum_a\boldsymbol{f}_a\cdot\boldsymbol{y}_a,
```

and $`D`$ collects optional damping or friction potentials. The accepted
velocity is

```math
\dot{\boldsymbol{y}}^{n+1}
=\frac{\boldsymbol{y}^{n+1}-\boldsymbol{y}^n}{\Delta t}.
```

Newton's method solves

```math
\boldsymbol{H}\Delta\boldsymbol{y}
=-\nabla\Pi,
\qquad
\boldsymbol{H}=\nabla^2\Pi,
```

followed by a feasible, energy-decreasing line search.

#### Shared IPC contact law

The scalar BarrierIPC, regularized-friction, and CCD feasibility
equations are maintained in the
[shared contact-model theory](../physics_model/contact_model/README.md#incremental-potential-contact).
For affine bodies, each primitive position is first mapped from the four
control vectors by the affine weights; the resulting contact energy,
gradient, and Hessian then enter the incremental potential above. Lagged
friction freezes its contact frame in an outer iteration, while a fully
implicit law differentiates the current normal force and frame.

### 6. Explicit rigid-body integration and stress-controlled servo walls

For translational acceleration
$`\boldsymbol{a}_i^n=\boldsymbol{F}_i^n/m_i`$, symplectic Euler uses

```math
\boldsymbol{v}_i^{n+1}
=\boldsymbol{v}_i^n+\Delta t\boldsymbol{a}_i^n,
\qquad
\boldsymbol{x}_i^{n+1}
=\boldsymbol{x}_i^n+\Delta t\boldsymbol{v}_i^{n+1}.
```

Velocity Verlet instead uses

```math
\boldsymbol{x}_i^{n+1}
=\boldsymbol{x}_i^n
+\Delta t\boldsymbol{v}_i^n
+\frac{\Delta t^2}{2}\boldsymbol{a}_i^n,
```

```math
\boldsymbol{v}_i^{n+1}
=\boldsymbol{v}_i^n
+\frac{\Delta t}{2}
\left(\boldsymbol{a}_i^n+\boldsymbol{a}_i^{n+1}\right).
```

The rotational state obeys the same angular-momentum balance from Section 1;
for a unit quaternion $`\boldsymbol{q}`$ its kinematic equation is

```math
\dot{\boldsymbol{q}}
=\frac12\boldsymbol{q}\otimes(0,\boldsymbol{\omega}^b),
\qquad
\|\boldsymbol{q}\|=1.
```

For a servo wall of current area $`A`$, measured normal force $`F`$, target stress
$`\sigma_t`$, and aggregate normal contact stiffness $`K`$, define the force
error and adaptive gain

```math
e_F=\sigma_tA-F,
\qquad
G=\frac{\alpha}{\Delta t\,K},
```

where $`0<\alpha\leq1`$ is the relaxation factor. The bounded wall-normal
velocity is

```math
v_n
=\mathrm{clip}
\left(Ge_F,-v_{max},v_{max}\right).
```

Prescribed tangential wall velocity is retained independently. Recomputing
$`A`$, $`F`$, and $`K`$ closes the discrete stress-feedback loop as the specimen
deforms.

### 7. Affine-body revolute joints, motors, and angle limits

Let the two sides of a joint reconstruct their anchor and material direction
vectors from affine controls,

```math
\boldsymbol{a}_s=\sum_{c=0}^{3}w_{s,c}^{a}\boldsymbol{y}_{s,c},
\qquad
\boldsymbol{d}_s=\sum_{c=0}^{3}w_{s,c}^{d}\boldsymbol{y}_{s,c},
\qquad s\in\{A,B\}.
```

Coincident anchors and axes are imposed by quadratic energies

```math
E_{joint}
=\frac{k_a}{2}\|\boldsymbol{a}_A-\boldsymbol{a}_B\|^2
+\frac{k_d}{2}\|\boldsymbol{d}_A-\boldsymbol{d}_B\|^2.
```

Choose orthogonal transverse directions
$`\boldsymbol{u}_A,\boldsymbol{v}_A`$ on side $`A`$ and
$`\boldsymbol{u}_B`$ on side $`B`$. The signed revolute angle is

```math
\theta
=\mathrm{atan2}
\left(
\boldsymbol{u}_B\cdot\boldsymbol{v}_A,
\boldsymbol{u}_B\cdot\boldsymbol{u}_A
\right).
```

A motor target $`\theta_t`$ is represented by

```math
E_m
=\frac{k_m}{2}
\left\|
\boldsymbol{u}_B
-\cos\theta_t\boldsymbol{u}_A
-\sin\theta_t\boldsymbol{v}_A
\right\|^2.
```

For limits $`\theta_{min}\leq\theta\leq\theta_{max}`$, set

```math
\theta_c
=\min\left(\theta_{max},\max(\theta_{min},\theta)\right)
```

and activate the same quadratic direction penalty with target $`\theta_c`$
only when $`\theta\ne\theta_c`$. These joint energies enter the affine
incremental potential and therefore share its Newton solve and feasible line
search.

## Typical workflow

```python
import geotaichi as gt

gt.init(arch="gpu", default_fp="float32", log=False)

dem = gt.DEM(log=False)
dem.set_configuration(
    domain=[2.0, 1.0, 1.0],
    boundary=["Destroy", "Destroy", "Reflect"],
    gravity=[0.0, 0.0, -9.81],
    engine="SymplecticEuler",
    search="LinkedCell",
    scheme="DEM",
    visualize=False,
    log=False,
)
dem.memory_allocate(
    memory={
        "max_material_number": 2,
        "max_particle_number": 100000,
        "max_sphere_number": 100000,
        "body_coordination_number": 24,
        "wall_coordination_number": 8,
    },
    log=False,
)
dem.set_solver(
    {
        "Timestep": 1.0e-5,
        "SimulationTime": 0.5,
        "SaveInterval": 1.0e-2,
        "SavePath": "OutputData/dem_case",
    },
    log=False,
)
dem.add_attribute(
    materialID=0,
    attribute={"Density": 2500.0, "ForceLocalDamping": 0.05},
)
dem.choose_contact_model(
    particle_particle_contact_model="Hertz Mindlin Model",
    particle_wall_contact_model="Hertz Mindlin Model",
)
dem.add_property(
    materialID1=0,
    materialID2=0,
    property={
        "ShearModulus": 4.0e6,
        "Poisson": 0.25,
        "Friction": 0.4,
        "Restitution": 0.2,
    },
)
# Add regions, templates, particles, and walls before running.
dem.run()
```

The generator API is dictionary based because body definitions vary by DEM,
LSDEM, LSMPM, and affine-body scheme. Complete inputs are available under
`tests/integration/dem/` and the project gallery scripts.

For coupling-owned LSMPM soft bodies, the default SDF update is `ReferenceMap`:
current material points and their inverse deformation gradients reconstruct
coordinates in the immutable initial SDF, with a local distance-metric
correction. `SemiLagrangian` is an alias for this route; `MacCormack` selects
the former incremental transport, and `WENO5` remains available. Redistancing
defaults to disabled for `ReferenceMap`. Configuration, equations, and
accuracy limits are documented in the
[MPM soft-particle guide](../mpm/README.md#15-soft-particle-lsmpm-and-level-set-transport).

## Affine and IPC bodies

Set `scheme="AffineBody"` and configure nonlinear options with
`set_affine_body_parameters()`. Affine contact uses device-resident energy,
gradient, Hessian, CCD, line search, and Krylov paths. Assembly can be
`"MatrixFree"`, `"COO"`, or `"HashTriplet"` depending on the selected solver.

AffineBody mesh IPC sizes the global PT and EE buffers from the assembled
surface triangle count. Each capacity is `max(1, ceil(triangles * coordination))`.
Set `point_triangle_coordination_number` and `edge_edge_coordination_number`
(defaults 32 and 128), or use `contact_coordination_number` for both. The LSMPM
Soft–Affine mesh candidate search also uses the affine triangle count and PT
coordination number. These settings are separate from body-level coordination.

AffineBody matrix capacity is derived from the assembly stencils; remove the
old `affine_contact_block_capacity` allocation key. Lagged mesh friction in
HashTriplet groups contacts by body pair and reserves at most 56 non-diagonal
blocks per pair. Fully implicit friction writes at most 216 PT or 224 EE raw
non-diagonal blocks per contact. COO also stores the diagonal contributions.
The body-pair barrier reserves its Hessian and projection workspace separately.

IPC initial states must be strictly feasible: body-body and body-wall gaps
must be positive and inside the configured barrier activation distance only
when contact is intended.

The affine implementation follows the same state/operator/engine ownership as
the older DEM paths. `AffineBodyState.py` owns accepted and trial state,
`AffineBodyOperator.py` owns Taichi contact/linear operations,
`TaichiAffineDiffIPCProjector.py` and `TaichiAffineMeshDiffIPCProjector.py`
own affine-mesh nonpenetration projectors, and
`AffineBodyEngine.py` owns nonlinear orchestration and output. Configured
device/reference, matrix, and friction paths are selected during
initialization rather than rediscovered in every nonlinear iteration.

Revolute robot links use `add_joint(...)` before the first run. A joint keeps
its world anchor and axis coincident while allowing relative rotation about
that axis; optional `MotorStiffness`/`TargetAngle`, `AngleLimit`,
`LimitStiffness`, and `Damping` add actuation, stops, and angular damping.
Angles are degrees at the public API. `CollideConnected=False` is the default:
the connected pair is removed from body--body IPC, but each link still collides
with other links and walls. `set_joint_target_angle(...)` may update a motor
after initialization.

For time-dependent drives in `dem.run`, use
`preintegration_function=controller` and decorate the controller with
`geotaichi.python_callback` when it calls Python setup/control APIs such as
`set_joint_target_angles`. ABD executes it before each step, including the
initial physical/JIT step. The `function` callback runs after the accepted
solve; the solver increments its clock afterward, so endpoint commands and
observations use `current_time + delta`. Default callbacks remain Taichi
kernels. Neither callback changes the nonlinear solver's numerical method.

`add_joint` returns the zero-based joint ID. `JointType="Fixed"` constrains
the anchor, axis and one transverse director, locking relative rigid motion;
`JointType="Spherical"` constrains only the anchor. For these two types,
`WorldAxis` defaults to `[0, 0, 1]`; motors, angular limits and angular damping
are rejected. All three types use soft quadratic constraints, with errors
set by load and stiffness. They share the existing Taichi assembly and are
available in standalone ABD, Direct MPM--ABD, and FEM--ABD coupling.

`set_joint_target_angles(jointIDs, target_angles)` updates a subset of revolute
motors between steps. Both arguments are equal-length vectors; IDs must be
unique integers and angles finite degrees. Invalid batches leave every target
unchanged. `get_joint_states()` requires an initialized engine and returns
arrays in joint-ID order: `joint_type`, `angle` (degrees in `[-180, 180]`),
`angular_velocity` (degrees/s), `target_angle` (degrees), `anchor_error` (m),
and `axis_error` (dimensionless director difference). Body--body angles measure
side `BodyID2` relative to `BodyID1`; world joints measure `BodyID1` relative
to the world frame. These angles are relative to the initial joint frames.
The angular speed is the analytic derivative of that coordinate, including
affine deformation. Fixed/spherical joints report zero angle and speed;
`axis_error` is unconstrained for spherical joints. The scalar angle wraps at
180 degrees and does not count accumulated turns. Motor targets enter through
sine/cosine and are not restricted to that reporting interval. This does not
establish reliable continuous-wheel operation: repeated turns, phase tracking
and any angle-limit interaction still need a dedicated regression. There is
currently no public prescribed-angular-speed or joint-torque drive.

### Robot control and differentiable trajectories

The same joint model can drive an arm, an excavator linkage or a floating-base
leg mechanism. Use Direct MPM--ABD IPC for continuum soil interaction; see
[coupled trajectory adjoints](../mpdem/README.md#continuum-mpm--abd-trajectory-adjoints).
Keep a walking robot's base free and let contact carry support loads.

```python
trajectory = dem.differentiable_affine(steps=len(commands))
for target_angles in commands:  # degrees, one vector per fixed time step
    dem.set_joint_target_angles(motor_ids, target_angles)
    trajectory.step()
    feedback = dem.get_joint_states()
gradients = trajectory.backward(terminal_control_gradient)
du = gradients["joint_target_angle_degrees_per_step"]
```

Standalone ABD, Direct MPM--ABD and FEM--ABD tapes record the exact motor
targets for each accepted step. Their backward results include
`joint_target_angle_radians_per_step` and
`joint_target_angle_degrees_per_step`, shaped `(recorded_steps, joint_count)`.
The existing `joint_target_angle_radians`/`joint_target_angle_degrees` arrays
sum these rows: they differentiate a common offset added to all recorded
targets. Unactuated and non-revolute columns are zero. Backward replay restores
the caller's current targets even if they changed after the last forward step.
The derivatives treat commands as independent inputs; differentiation through
a feedback controller requires the controller's own chain rule. Stiffness,
damping, gravity and other model parameters must stay fixed during recording
and replay. Existing backward-Euler, contact-branch and friction restrictions
still apply. Legacy LSMPM Soft-Affine multi-step replay remains unavailable.

The [arm example](../../examples/dem/AffineBody/robot_multilink_arm/robot_multilink_arm.py)
accepts `--differentiable --motion-amplitude 5 --steps 20 --output-dir /tmp/arm`
to exercise changing targets and save their per-step gradients.
Focused evidence is in
[joint/control tests](../../tests/unit/dem/test_affine_robot_controls.py) and
[plastic coupling gradients](../../tests/integration/mpdem/test_direct_affine_trajectory_gradients.py).

Standalone mesh-ABD trajectories reverse every executed lagged-friction outer
solve, including the refresh of normal load, tangent projection and closest
weights. Fixed plane and triangular walls retain their wall identity in the
friction cache. Replay uses one step's outer-iterate workspace alongside the
position/velocity tape. The single-equilibrium `differentiate_affine_step`
interface retains its frozen-cache meaning; use `trajectory.backward` for the
derivative of a complete step with refreshed friction.

Contact activation, closest-feature choices and iteration counts are discrete.
The returned derivative is local to the executed branches. A trajectory may
create and lose contacts over time, but this does not supply a derivative of
every possible change in contact sequence. The
[complete grasp check and small trajectory optimization](../../examples/dem/AffineBody/robot_grasp/README.md)
compare against perturbed complete forward runs.

The minimum application model depends on the requested observable. Prescribed
joint motion can represent an ideal actuator without an electric motor,
hydraulic cylinder or powertrain model. Current angle motors are compliant
position servos; integrating a speed command into their target angle supplies
a position reference, not an exactly enforced angular-speed constraint.

| Application | Minimum model for the task | Detail to add only when required |
| --- | --- | --- |
| Arm grasping an object | Links with mass/inertia, joints and a supplied motion sequence; an actuated gripper; a dynamically free object; frictional gripper/object/environment contact | A prismatic joint only if the chosen gripper needs one; IK for Cartesian commands, URDF import, motor/electrical models and effort caps for hardware feasibility |
| Excavator digging soil | Bucket and driven boom/stick/bucket links; plastic MPM soil with appropriate parameters; two-way bucket--soil contact and friction; a fixed base for stationary digging, or a free supported base when machine motion matters | Explicit cylinders and hydraulics for pressure/flow/actuator studies; detailed bucket linkage, tracks or powertrain when those are the observable |
| Wheeled vehicle on soil | Free chassis and wheels with mass/inertia; wheel-axis joints; a rotation drive; plastic MPM soil and two-way wheel--soil contact/friction; steering only for turning | Suspension for ride/load-transfer studies; deformable tires when tire compliance matters; torque limits/powertrain for power-limited mobility |
| Robot walking on soil | Free base, leg/foot geometry and inertias, driven joints, a supplied gait that maintains support/balance, and two-way foot--soil frictional contact | Sensor emulation, adaptive balance control, MPC/RL, actuator detail and effort caps according to the task |

For vehicle mobility and walking, let chassis/base translation follow the
computed forces; prescribing it would remove the soil-dependent slip,
sinkage or loss of support being investigated. A speed-driven wheel can still
slip or spin while the vehicle barely moves. For grasping, let the payload
follow contact forces rather than attaching it to the hand when testing grip.
Rigid wheels and feet suffice for a first soil-interaction model. Detailed
hydraulics, deformable tires, IK, URDF and virtual sensors are not prerequisites
for the supplied-motion tasks above.

Current shared foundations cover joints, angle-reference sequences, contact
and the supported coupled state/control VJPs. The next application evidence
should be a grip/lift test, a driven bucket soil sweep, repeated wheel turns
with a free chassis, and a free-base gait on soil. The existing small derivative
tests do not establish those application outcomes. Add a torque input or
bounded servo when the question includes stalling, motor feasibility or
available traction at a given actuator limit. Prescribed-motion runs instead
answer what soil/contact response that imposed motion requires.

Angle motors are director springs, not ideal torque sources. Angle limits are
soft penalties. Contact switches and angle-wrap crossings are nonsmooth, so
trajectory derivatives are local to the recorded branches; these capabilities
do not establish successful excavation or stable walking.

Implicit AffineBody and LSMPM Soft-Affine IPC runs accept
`enable_step_retry`, `step_retry_max_retries`, `step_retry_reduction`, and
`step_retry_minimum_timestep` in `set_solver`. Retry is bounded, restores the
device/host step-start transaction, and catches only solve/line-search
nonconvergence. Standard DEM/LSDEM/LSMPM stepping is explicit and rejects an
enabled retry policy. `diagnostics_snapshot()` is available on every `DEM`
facade; the affine engines additionally report nonlinear, linear, CCD, and
friction state.

### Choosing affine IPC parameters

GeoTaichi's affine IPC geometry inputs are absolute values. Let `L` be a characteristic
contact length, usually a body or scene bounding-box diagonal:

| Input | Units | Practical starting point |
| --- | --- | --- |
| `Dhat` | m | About `1e-3 L`; enlarge only when the surface resolution or intended contact gap requires it. |
| `BarrierStiffness` | problem dependent | Tune with `Dhat` so the loaded equilibrium gap is positive, below `Dhat`, and comfortably above geometry/CCD round-off. |
| `friction_epsv` | m/s | About `1e-3 L/s`; reduce by decades only while measured slip changes materially. |
| `newton_tolerance` | m/s | Controls the unapplied surface correction; check strain and energy when penalty stiffness is high. |
| `friction_tolerance` | m/s | For verification, use `friction_iterations=-1` and tighten until force balance and slip stop changing. For production, one lagged solve is the reference default and 2--4 solves are usually enough. |

Inside the quadratic static-friction branch, the maximum tangential curvature
is proportional to `1 / (friction_epsv * dt)`. A smaller `friction_epsv`
therefore reduces regularized creep but makes the nonlinear system sharper;
it is not an unconditional accuracy knob.

The lagged defaults are `friction_tolerance=1e-7` m/s,
`newton_tolerance=1e-8` m/s, `linear_tolerance=1e-14`
(absolute assembled-system residual), and `linear_relative_tolerance=0`.
Set these through `dem.set_affine_body_parameters(...)`. The fixed linear target
is `max(linear_tolerance, linear_relative_tolerance * initial_residual)`;
setting only `linear_tolerance` selects absolute-only stopping, including when
replacing an earlier relative setting. Supply both keys to use both branches.
Either tolerance may be zero, but not both. Both HashTriplet and COO, supported
adjoints, and LSMPM soft--ABD consume these controls. Fully implicit force checks
can require a tighter inner solve; failure never increases a threshold.
Explicit values remain authoritative within that force safeguard.
For BackwardEuler, `newton_tolerance * dt` bounds the unapplied Newton
surface correction. It does not bound the accumulated trajectory error or
the remaining penalty energy. In particular, the orthogonality energy
`V * E / 8 * ||F^T F - I||^2` can be substantial even for a small residual
affine stretch at large `E`. Validate a fixed case tolerance through both
rotation and contact transients. Decreasing the incremental objective in
line search alone does not certify non-increasing physical energy.

Linear residual and correction velocity have different units: small masses,
small time steps, or ill-conditioned affine coordinates can require a tighter
linear solve. Validate case-specific settings by tightening the linear
tolerance and checking the physical correction, rather than increasing only
the Newton iteration cap. The [mass-norm bound](../linear_solver/README.md#matching-linear-and-newton-tolerances)
relates a true linear residual to the surface correction for projected ABD.
The [bunny](../../examples/dem/AffineBody/bunny_pile_wall_release/README.md)
and [cube](../../examples/dem/AffineBody/thousand_cubes_heavy_ball_impact/README.md)
examples document their explicit settings and validation scope. Short checkpoint
evidence does not establish full wall-release or impact acceptance.

Read convergence in layers. The linear solve must converge first; the lagged
Newton test is `c_v * ||delta_x_surface||_inf < newton_tolerance`, with
`c_v = 1 / dt` for BackwardEuler and `gamma / (2 * alpha * beta * dt)` for
the other integration triples. The outer friction test applies the same
correction-velocity measure after rebuilding
the frozen normal force and tangent basis. The line search starts from the CCD
feasible step and halves it until the frozen incremental potential is
non-increasing. ABD, soft-affine and Direct MPM--ABD also accept f64 reduction
roundoff only when both the predicted decrease and observed increase fit
`1e-12 * max(abs(E_old), abs(E_trial))`. There is no absolute energy floor;
Newton convergence remains mandatory. The outer friction threshold is mandatory
in convergence mode (`friction_iterations=-1`); a finite lagged count records
its residual and cap status even if the fixed point has not converged.
A healthy step has a negative finite slope, positive accepted
alpha, and modest backtracking. Repeated tiny alpha values indicate a scale,
time-step, stiffness, or linear-solve problem; increasing only the backtracking
cap does not fix it.

The affine diagnostics expose `linear_solver`, `line_search`, Newton, friction,
and CCD values. Static friction itself need not be small: validate it with
small tangential slip and the dynamic balance `external force - contact force -
momentum change / dt`, not with `Ft` approaching zero. Interpret the
parameter meanings and monotone line search with the convergence criteria
described above.

## Runtime and memory

`memory_allocate()` is part of the public lifecycle and must be called after
the scheme and search backend are selected. Capacity settings define Taichi
field sizes and cannot generally be enlarged without rebuilding the solver.
Time integration, contact detection, force assembly, and state updates are
performed in Taichi kernels. Python owns configuration, generator orchestration,
scalar nonlinear control flow, and output callbacks.

## Tests

DEM geometry, neighbor, contact, IPC, affine-body, and LSMPM tests are under
`tests/unit/dem/` and `tests/integration/dem/`.

## DEM facet-wall servos with IPC

For ABD--wall IPC, retain the ordinary DEM `Polygon` wall configuration:
allocate `max_facet_number` and `max_servo_wall_number`, set each wall's
`ControlType`, `TargetStress`, `LimitVelocity` and optional `Alpha`/`Gain`, then
call `dem.servo_switch("StiffnessControl")` or `"GainControl"`. No servo is added
to the ABD bodies.

`IPCWallBoundary` keeps the facet geometry and motion in f64 device fields.
After an accepted IPC step, the physical wall reactions and normal tangent
stiffness feed the existing `ServoWall` force error, gain and velocity limit.
The next wall motion is capped by point--triangle ACCD before the nonlinear
solve. Rejected steps restore wall geometry and leave servo feedback unchanged.
An externally supplied servo area is retained; otherwise the facet areas
initialize it. The controller writes DEM wall forces, stiffness, velocity and
servo pressure data on device, so ordinary wall/servo outputs remain usable.
Host access occurs during setup and explicit diagnostics/output.

These servo boundaries use Polygon/facet walls. Contact damping and trajectory
adjoints through the controller are not supported. The boundary update is
staggered between accepted IPC steps; it is not an additional Newton unknown.

## Explicit BVH updates

For DEM, LSDEM and LSMPM, set `bvh_rebuild_interval` in `memory_allocate`, for
example `{"bvh_rebuild_interval": 100}` alongside the other memory settings.
The value must be a positive integer; the default is `1000`, and `1` rebuilds
at the next search in each new solver step. Within an interval the search refits
current AABBs on device. It rebuilds when the elapsed step count reaches the
interval, even if no neighbor search ran at the exact interval boundary.
Repeated queries in one step refit. The initial search and changed active
particle/wall counts rebuild; static wall batches can otherwise remain cached.
Verlet displacement checks still decide when a new candidate list is needed.
Explicit DEM/LSDEM--MPM and LSMPM coupling reuse the DEM child's tree and interval.

## AffineBody BVH updates

The IPC BVH neighbor rebuilds its topology on device for every new current or
swept candidate query, including queries within a Newton iteration. Each build
recomputes Morton codes, sorts primitives and reduces node bounds. A prepared
swept candidate list can still be reused by line-search trials along the bounded
direction. Candidate completeness and overflow checks are unchanged.
Device-built subtree body labels skip entire branches belonging to the query
body, preserving the existing same-body exclusion before reaching individual
leaves. These labels and the stackless escape links are regenerated after each
topology rebuild; each requires one integer per BVH node.

## AffineBody friction storage

CPU and CUDA keep lagged-friction records and their adjoint backup in device
pages. Contact counts and page activation run on device before parallel writes;
this prevents cold-page activation from losing records. The adjoint backup and
restore also activate their destination pages before copying. The capacity bound
and overflow checks are unchanged; unused vertex–wall combinations do not reserve record
payloads. Other backends retain dense storage. Numerical assembly, frozen
friction history and adjoint restore use the same fields and active counts.

Triangle-mesh lagged friction condenses each contact's primitive weights onto
the affine control points and accumulates its HashTriplet Hessian in one
off-diagonal segment per body pair, including body--wall self-pairs. Repeated
contacts reuse those slots without changing the energy, gradient or Hessian.
COO and LevelSet assembly retain their stencil paths. Capacity overflow remains
an error; this does not limit the number of physical contacts.
