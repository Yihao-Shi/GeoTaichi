# Discrete Element Method (DEM), Level-Set DEM (LSDEM), and Affine Body Dynamics (ABD)

`src/dem` contains GeoTaichi's discrete-element solvers and the public `DEM`
facade. It supports spherical and clumped particles, level-set rigid bodies,
affine deformable bodies, walls, several neighbor-search backends, and a range
of contact laws.

[Theory log and derivations](#dem-theory-log) | [Examples](../../examples/)

## Example-backed capabilities

- Spherical and clumped DEM: [sphere packing](../../examples/dem/MultiSphere/RotatingDrum/sphere_packing.py) and [clump packing](../../examples/dem/MultiSphere/GranularPackings/clump_packing.py).
- Level-set discrete element method (LSDEM) for irregular particles: [polydisperse packing](../../examples/dem/LevelSet/GranularAssemble/polydisperse/packing_generate.py), [screw and nut](../../examples/dem/LevelSet/ParticleParticle/screw_and_nut.py), and [rotating drum](../../examples/dem/LevelSet/RotatingDrum/rotating_drum.py).
- Affine body dynamics (ABD) with incremental potential contact (IPC): [sphere–wall collision](../../examples/dem/AffineBody/sphere_wall_collision.py) and [inclined-plane sliding](../../examples/dem/AffineBody/cube_incline_sliding.py).
- ABD joints and motors: [multilink arm](../../examples/dem/AffineBody/robot_multilink_arm.py).
- Level-set MPM soft particles: [rigid–soft contact](../../examples/mpdem/LevelSet/SoftRigid/rigid_soft_sphere_drop_box.py).

## Package layout

| Path | Responsibility |
| --- | --- |
| `generator/` | Regions, templates, bodies, walls, and restart input |
| `structs/` | Taichi particle, rigid-body, wall, contact, and level-set fields |
| `neighbor/` | Neighbor search and Verlet/contact candidate management |
| `contact/` | DEM contact models and force assembly |
| `affine/` | Affine body templates and state |
| `engines/` | Time integration and affine nonlinear solvers |
| `mainDEM.py` | Public scene facade |

Shared constitutive and contact formulas live in `src/physics_model`, while
BVH and CCD primitives live in `src/contact_detection`.

## DEM theory log

### 1. Rigid-particle balance laws

For a rigid particle $i$, let $m_i$, $\boldsymbol{x}_i$,
$\boldsymbol{v}_i$, and $\boldsymbol{\omega}_i$ denote its mass, center of
mass, translational velocity, and spatial angular velocity. Its Newton--Euler
equations are

$$
\dot{\boldsymbol{x}}_i=\boldsymbol{v}_i,
\qquad
m_i\dot{\boldsymbol{v}}_i=\boldsymbol{F}_i,
\qquad
\dot{\boldsymbol{L}}_i=\boldsymbol{M}_i.
$$

With $\mathcal C_i$ the active contacts of particle $i$, the resultant
force and moment are

$$
\boldsymbol{F}_i
=m_i\boldsymbol{g}+\boldsymbol{F}_i^{\mathrm{ext}}
+\sum_{c\in\mathcal C_i}
\left(\boldsymbol{F}_{n,c}+\boldsymbol{F}_{t,c}\right).
$$

$$
\boldsymbol{M}_i
=\boldsymbol{M}_i^{\mathrm{ext}}
+\sum_{c\in\mathcal C_i}
\left[
\left(\boldsymbol{x}_c-\boldsymbol{x}_i\right)
\times\left(\boldsymbol{F}_{n,c}+\boldsymbol{F}_{t,c}\right)
+\boldsymbol{M}_{r,c}+\boldsymbol{M}_{tw,c}
\right].
$$

Here $\boldsymbol{x}_c$ is the contact point;
$\boldsymbol{F}_{n,c}$ and $\boldsymbol{F}_{t,c}$ are the normal and
tangential contact forces; and $\boldsymbol{M}_{r,c}$ and
$\boldsymbol{M}_{tw,c}$ are optional rolling and twisting moments. A simple
contact law sets the last two terms to zero.

Let $\boldsymbol{R}_i$ rotate body-frame vectors into the spatial frame and
define

$$
\boldsymbol{\omega}_i^b=\boldsymbol{R}_i^T\boldsymbol{\omega}_i,
\qquad
\boldsymbol{M}_i^b=\boldsymbol{R}_i^T\boldsymbol{M}_i.
$$

Because the body-frame inertia $\boldsymbol{I}_i^b$ is constant, rotational
balance can be written as Euler's rigid-body equation

$$
\boldsymbol{I}_i^b\dot{\boldsymbol{\omega}}_i^b
+\boldsymbol{\omega}_i^b\times
 \left(\boldsymbol{I}_i^b\boldsymbol{\omega}_i^b\right)
=\boldsymbol{M}_i^b,
\qquad
\dot{\boldsymbol{R}}_i
=\boldsymbol{R}_i[\boldsymbol{\omega}_i^b]_{\times},
$$

where $[\boldsymbol{a}]_{\times}\boldsymbol{b}=\boldsymbol{a}\times\boldsymbol{b}$.
Equivalently,

$$
\boldsymbol{L}_i
=\boldsymbol{R}_i\boldsymbol{I}_i^b\boldsymbol{\omega}_i^b.
$$

For a body occupying $\Omega_i$ with density $\rho(\boldsymbol{x})$, its
mass properties are

$$
m_i=\int_{\Omega_i}\rho\,\mathrm dV.
$$

$$
\boldsymbol{x}_i
=\frac{1}{m_i}\int_{\Omega_i}\rho\boldsymbol{x}\,\mathrm dV.
$$

$$
\boldsymbol{I}_i
=\int_{\Omega_i}\rho
\left[
\|\boldsymbol{r}\|^2\boldsymbol{1}
-\boldsymbol{r}\otimes\boldsymbol{r}
\right]\mathrm dV,
\qquad
\boldsymbol{r}=\boldsymbol{x}-\boldsymbol{x}_i.
$$

For a homogeneous sphere of radius $R_i$, these reduce to

$$
m_i=\frac{4}{3}\pi\rho_iR_i^3,
\qquad
\boldsymbol{I}_i^b=\frac{2}{5}m_iR_i^2\boldsymbol{1}.
$$

For a clump represented by overlapping pebbles, $\Omega_i$ is the union of
the pebble volumes. The union must be integrated only once; summing the mass
and inertia of every full pebble would double-count overlap.

#### Local non-viscous damping

Local damping acts componentwise on the current resultant load. In the frame
where the balance equation is evaluated, define

$$
\mathcal D_{\alpha}(f_k,u_k)
=f_k\left[1-\alpha\,\mathrm{sgn}(f_ku_k)\right],
\qquad k\in\{1,2,3\},
$$

with $\mathrm{sgn}(0)=0$. The damped force and moment are therefore

$$
F_{i,k}^{d}=\mathcal D_{\alpha_f}(F_{i,k},v_{i,k}),
\qquad
M_{i,k}^{d}=\mathcal D_{\alpha_t}(M_{i,k},\omega_{i,k}).
$$

### 2. Contact geometry and force transfer

The shared [contact-model theory](../physics_model/contact_model/README.md)
contains sphere/wall kinematics, Linear, Hertz--Mindlin, energy-conserving
penalty, Coulomb, rolling, twisting, BarrierIPC, and regularized
friction formulas. DEM owns neighbor discovery and rigid-particle assembly,
not separate copies of those constitutive laws.

For an active pair, the contact model returns the force on particle $i$,

$$
\boldsymbol{F}_{ij}^c
=\boldsymbol{F}_n+\boldsymbol{F}_t,
$$

and optional rolling/twisting resistance $\boldsymbol{M}_{ij}^c$. With
contact arms
$\boldsymbol{r}_i=\boldsymbol{x}_c-\boldsymbol{x}_i$ and
$\boldsymbol{r}_j=\boldsymbol{x}_c-\boldsymbol{x}_j$, pair contributions
to the rigid-particle balances are

$$
\boldsymbol{F}_i^c=\boldsymbol{F}_{ij}^c,
\qquad
\boldsymbol{F}_j^c=-\boldsymbol{F}_{ij}^c,
$$

$$
\boldsymbol{M}_i^c
=\boldsymbol{r}_i\times\boldsymbol{F}_{ij}^c
+\boldsymbol{M}_{ij}^c,
$$

$$
\boldsymbol{M}_j^c
=-\boldsymbol{r}_j\times\boldsymbol{F}_{ij}^c
-\boldsymbol{M}_{ij}^c.
$$

Thus internal contact forces cancel exactly. The remaining angular-momentum
balance follows from using one common contact point and equal/opposite contact
moments.

### 3. Neighbor search and Verlet lists

#### Verlet skin

Let $s>0$ be the full pairwise Verlet skin. A sphere pair is stored as a
candidate when

$$
\|\boldsymbol{x}_i-\boldsymbol{x}_j\|
\leq R_i+R_j+s.
$$

Suppose $\Delta\boldsymbol{x}_i$ is the displacement since the list was
built. No omitted pair can enter contact while

$$
\|\Delta\boldsymbol{x}_i\|
+\|\Delta\boldsymbol{x}_j\|<s
$$

for every pair. A simple global sufficient condition is

$$
\max_i\|\Delta\boldsymbol{x}_i\|<\frac{s}{2}.
$$

For a non-spherical rigid body with bounding radius $r_i^b$, translation
alone is insufficient. If its accumulated rotation angle is
$\Delta\theta_i$, a conservative surface-displacement bound is

$$
u_i
\leq
\|\Delta\boldsymbol{x}_i\|
+2r_i^b\sin\left(\frac{|\Delta\theta_i|}{2}\right).
$$

The safe pair condition then becomes $u_i+u_j<s$. For deformable bodies, the
bound must additionally include outward growth of the bounding volume.

#### Uniform linked cells

For maximum particle radius $R_{max}$, choose the cubic cell width

$$
h\geq2R_{max}+s.
$$

The integer cell coordinate of a particle is

$$
\boldsymbol{c}_i
=\left\lfloor
\frac{\boldsymbol{x}_i-\boldsymbol{x}_{min}}{h}
\right\rfloor.
$$

All possible neighbors lie in the particle's own cell or one of the adjacent
cells

$$
\boldsymbol{c}_j
\in
\boldsymbol{c}_i+\{-1,0,1\}^d,
$$

where $d=2$ or $3$. The exact Verlet-distance test is still required
after the cell lookup; the grid only removes impossible pairs.

For strongly polydisperse particles, one grid wastes work because $h$ is
set by the largest radius. A hierarchical grid assigns particle $i$ to the
smallest level $\ell$ satisfying

$$
R_i\leq R_\ell,
\qquad
h_\ell=2R_\ell+s.
$$

Pairs are searched within a level and across compatible coarser levels, then
filtered by the exact distance inequality. This retains small cells for small
particles without missing large--small contacts.

#### Bounding-volume hierarchy

An alternative broad phase encloses each object in a skin-expanded axis-aligned
box

$$
\boldsymbol{B}_i
=
\left[
\boldsymbol{x}_i-\left(R_i+\frac{s}{2}\right)\boldsymbol{1},
\;
\boldsymbol{x}_i+\left(R_i+\frac{s}{2}\right)\boldsymbol{1}
\right].
$$

Two boxes overlap when, in every coordinate $k$,

$$
\max(B_{i,k}^{min},B_{j,k}^{min})
\leq
\min(B_{i,k}^{max},B_{j,k}^{max}).
$$

A linear BVH orders box centroids by a Morton key. If
$\boldsymbol{u}_i\in[0,1)^3$ is the normalized centroid and $Q$ is an
integer quantizer, then

$$
\boldsymbol{q}_i=\lfloor Q\boldsymbol{u}_i\rfloor,
\qquad
z_i=\mathrm{interleave}(q_{i,x},q_{i,y},q_{i,z}).
$$

The binary tree is formed from common prefixes of the sorted $z_i$. Query
traversal rejects a complete subtree whenever its box does not overlap the
query box. The resulting candidate set must still pass the exact geometric
test.

### 4. Level-set DEM

#### Signed-distance representation

A level-set rigid body is described in its material frame by a signed-distance
field

$$
\phi^0(\boldsymbol{X})<0
\quad\text{inside the body},
$$

$$
\phi^0(\boldsymbol{X})=0
\quad\text{on the surface},
$$

$$
\phi^0(\boldsymbol{X})>0
\quad\text{outside the body}.
$$

For an exact signed-distance field,

$$
\|\nabla_{\boldsymbol{X}}\phi^0\|=1
$$

away from medial-axis singularities. Under translation
$\boldsymbol{x}_i$, rotation $\boldsymbol{R}_i$, and uniform scale $s_i$,

$$
\boldsymbol{X}
=\frac{1}{s_i}\boldsymbol{R}_i^T
\left(\boldsymbol{x}-\boldsymbol{x}_i\right),
$$

$$
\phi_i(\boldsymbol{x})=s_i\phi_i^0(\boldsymbol{X}),
\qquad
\nabla_{\boldsymbol{x}}\phi_i
=\boldsymbol{R}_i\nabla_{\boldsymbol{X}}\phi_i^0.
$$

#### Trilinear SDF interpolation

Inside one Cartesian grid cell, let
$\boldsymbol{\xi}=(\xi,\eta,\zeta)\in[0,1]^3$ be the reduced coordinate and
define

$$
N_0(u)=1-u,
\qquad
N_1(u)=u.
$$

The trilinear signed distance is

$$
\phi_h(\boldsymbol{\xi})
=\sum_{a,b,c\in\{0,1\}}
\phi_{abc}N_a(\xi)N_b(\eta)N_c(\zeta).
$$

For grid spacing $h$, its first gradient component is

$$
\frac{\partial\phi_h}{\partial x}
=\frac{1}{h}
\sum_{a,b,c\in\{0,1\}}
(2a-1)\phi_{abc}N_b(\eta)N_c(\zeta),
$$

with cyclic expressions for the other two components. The outward normal is

$$
\boldsymbol{n}
=\frac{\nabla\phi_h}{\|\nabla\phi_h\|}.
$$

Trilinear interpolation is continuous, but its gradient is generally
discontinuous across cell boundaries. Grid resolution therefore controls both
surface accuracy and contact-normal smoothness.

#### Surface-node contact

Let $\boldsymbol{p}_a$ be a quadrature point on body $i$, evaluated in the
signed-distance field of body $j$. Its signed gap is

$$
g_a=\phi_j(\boldsymbol{p}_a).
$$

For $g_a<0$, the penetration, outward normal, closest-point approximation,
and symmetric contact point are

$$
\delta_a=-g_a,
\qquad
\boldsymbol{n}_a
=\frac{\nabla\phi_j(\boldsymbol{p}_a)}
       {\|\nabla\phi_j(\boldsymbol{p}_a)\|},
$$

$$
\overline{\boldsymbol{p}}_a
=\boldsymbol{p}_a-g_a\boldsymbol{n}_a,
\qquad
\boldsymbol{x}_{c,a}
=\boldsymbol{p}_a-\frac{g_a}{2}\boldsymbol{n}_a.
$$

If $w_a$ is the surface quadrature weight, the resultant contact force and
moment are

$$
\boldsymbol{F}_{ij}
=\sum_a w_a
\left(
f_{n,a}\boldsymbol{n}_a+\boldsymbol{F}_{t,a}
\right),
$$

$$
\boldsymbol{M}_{ij}
=\sum_a w_a
\left(\boldsymbol{x}_{c,a}-\boldsymbol{x}_i\right)
\times
\left(
f_{n,a}\boldsymbol{n}_a+\boldsymbol{F}_{t,a}
\right).
$$

For a penalty potential $\Psi(g)$, a compact conservative form is

$$
E_{ij}
=\sum_a w_a\Psi(g_a),
\qquad
\Psi(g)=\frac{1}{2}k_n\langle-g\rangle_+^2.
$$

The nodal contact force follows from

$$
\boldsymbol{f}_a
=-\frac{\partial E_{ij}}{\partial\boldsymbol{p}_a}
=k_n\langle-g_a\rangle_+\boldsymbol{n}_a
$$

when $\phi$ is an exact signed-distance field. Tangential, rolling, and
twisting laws use the same contact-frame kinematics as sphere contact.

#### Two-level LSDEM search

Level-set contact uses two conservative filters. First, body bounding volumes
must overlap after expansion by the body skin $s_b$. Second, a surface point
is retained only when

$$
\phi_j(\boldsymbol{p}_a)<s_p,
$$

where $s_p$ is the point-level skin. The body list is rebuilt when its
bounding-volume displacement exhausts $s_b$; the point list is rebuilt when
the accumulated relative motion at a candidate point exhausts $s_p$. This
separates inexpensive rigid-body culling from the more expensive SDF queries.

### 5. Affine-body mechanics and IPC

#### Affine kinematics

Let a body use four vector controls
$\boldsymbol{y}_0,\boldsymbol{y}_1,\boldsymbol{y}_2,\boldsymbol{y}_3$.
For material coordinate
$\boldsymbol{X}=(X_1,X_2,X_3)$, define

$$
w_0=1-X_1-X_2-X_3,
\qquad
w_1=X_1,
\qquad
w_2=X_2,
\qquad
w_3=X_3.
$$

The current position and velocity are

$$
\boldsymbol{x}(\boldsymbol{X})
=\sum_{a=0}^{3}w_a(\boldsymbol{X})\boldsymbol{y}_a,
\qquad
\boldsymbol{v}(\boldsymbol{X})
=\sum_{a=0}^{3}w_a(\boldsymbol{X})\dot{\boldsymbol{y}}_a.
$$

Equivalently,

$$
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
$$

The map contains translation, rotation, stretch, and shear while retaining
only twelve scalar degrees of freedom per body.

The consistent $4\times4$ control mass matrix is

$$
M_{ab}
=\int_{\Omega_0}\rho_0w_aw_b\,\mathrm dV.
$$

The kinetic energy is

$$
T
=\frac{1}{2}
\sum_{a=0}^{3}\sum_{b=0}^{3}
M_{ab}\,
\dot{\boldsymbol{y}}_a\cdot\dot{\boldsymbol{y}}_b.
$$

For a body force $\boldsymbol{b}$, the generalized force is

$$
\boldsymbol{f}_a
=\int_{\Omega_0}\rho_0w_a\boldsymbol{b}\,\mathrm dV.
$$

#### Affine rigidity

A rotation-invariant rigidity potential is

$$
E_r(\boldsymbol{F})
=\frac{VE}{8}
\left\|
\boldsymbol{F}^T\boldsymbol{F}-\boldsymbol{1}
\right\|_F^2,
$$

where $V$ is the reference volume and $E$ controls affine rigidity. It
vanishes for every $\boldsymbol{F}\in SO(3)$, penalizes stretch and shear,
and does not penalize rigid rotation.

Let

$$
\boldsymbol{C}
=\boldsymbol{F}^T\boldsymbol{F}-\boldsymbol{1}.
$$

The derivative with respect to the affine matrix is

$$
\frac{\partial E_r}{\partial\boldsymbol{F}}
=\frac{VE}{2}\boldsymbol{F}\boldsymbol{C}.
$$

#### Incremental potential

With

$$
\widetilde{\boldsymbol{y}}
=\boldsymbol{y}^n+\Delta t\,\dot{\boldsymbol{y}}^n,
$$

one implicit step minimizes

$$
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
$$

Here

$$
\|\boldsymbol{z}\|_{\boldsymbol{M}}^2
=\boldsymbol{z}^T\boldsymbol{M}\boldsymbol{z},
\qquad
U_{ext}
=-\sum_a\boldsymbol{f}_a\cdot\boldsymbol{y}_a,
$$

and $D$ collects optional damping or friction potentials. The accepted
velocity is

$$
\dot{\boldsymbol{y}}^{n+1}
=\frac{\boldsymbol{y}^{n+1}-\boldsymbol{y}^n}{\Delta t}.
$$

Newton's method solves

$$
\boldsymbol{H}\Delta\boldsymbol{y}
=-\nabla\Pi,
\qquad
\boldsymbol{H}=\nabla^2\Pi,
$$

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
$\boldsymbol{a}_i^n=\boldsymbol{F}_i^n/m_i$, symplectic Euler uses

$$
\boldsymbol{v}_i^{n+1}
=\boldsymbol{v}_i^n+\Delta t\boldsymbol{a}_i^n,
\qquad
\boldsymbol{x}_i^{n+1}
=\boldsymbol{x}_i^n+\Delta t\boldsymbol{v}_i^{n+1}.
$$

Velocity Verlet instead uses

$$
\boldsymbol{x}_i^{n+1}
=\boldsymbol{x}_i^n
+\Delta t\boldsymbol{v}_i^n
+\frac{\Delta t^2}{2}\boldsymbol{a}_i^n,
$$

$$
\boldsymbol{v}_i^{n+1}
=\boldsymbol{v}_i^n
+\frac{\Delta t}{2}
\left(\boldsymbol{a}_i^n+\boldsymbol{a}_i^{n+1}\right).
$$

The rotational state obeys the same angular-momentum balance from Section 1;
for a unit quaternion $\boldsymbol{q}$ its kinematic equation is

$$
\dot{\boldsymbol{q}}
=\frac12\boldsymbol{q}\otimes(0,\boldsymbol{\omega}^b),
\qquad
\|\boldsymbol{q}\|=1.
$$

For a servo wall of current area $A$, measured normal force $F$, target stress
$\sigma_t$, and aggregate normal contact stiffness $K$, define the force
error and adaptive gain

$$
e_F=\sigma_tA-F,
\qquad
G=\frac{\alpha}{\Delta t\,K},
$$

where $0<\alpha\leq1$ is the relaxation factor. The bounded wall-normal
velocity is

$$
v_n
=\mathrm{clip}
\left(Ge_F,-v_{max},v_{max}\right).
$$

Prescribed tangential wall velocity is retained independently. Recomputing
$A$, $F$, and $K$ closes the discrete stress-feedback loop as the specimen
deforms.

### 7. Affine-body revolute joints, motors, and angle limits

Let the two sides of a joint reconstruct their anchor and material direction
vectors from affine controls,

$$
\boldsymbol{a}_s=\sum_{c=0}^{3}w_{s,c}^{a}\boldsymbol{y}_{s,c},
\qquad
\boldsymbol{d}_s=\sum_{c=0}^{3}w_{s,c}^{d}\boldsymbol{y}_{s,c},
\qquad s\in\{A,B\}.
$$

Coincident anchors and axes are imposed by quadratic energies

$$
E_{joint}
=\frac{k_a}{2}\|\boldsymbol{a}_A-\boldsymbol{a}_B\|^2
+\frac{k_d}{2}\|\boldsymbol{d}_A-\boldsymbol{d}_B\|^2.
$$

Choose orthogonal transverse directions
$\boldsymbol{u}_A,\boldsymbol{v}_A$ on side $A$ and
$\boldsymbol{u}_B$ on side $B$. The signed revolute angle is

$$
\theta
=\mathrm{atan2}
\left(
\boldsymbol{u}_B\cdot\boldsymbol{v}_A,
\boldsymbol{u}_B\cdot\boldsymbol{u}_A
\right).
$$

A motor target $\theta_t$ is represented by

$$
E_m
=\frac{k_m}{2}
\left\|
\boldsymbol{u}_B
-\cos\theta_t\boldsymbol{u}_A
-\sin\theta_t\boldsymbol{v}_A
\right\|^2.
$$

For limits $\theta_{min}\leq\theta\leq\theta_{max}$, set

$$
\theta_c
=\min\left(\theta_{max},\max(\theta_{min},\theta)\right)
$$

and activate the same quadratic direction penalty with target $\theta_c$
only when $\theta\ne\theta_c$. These joint energies enter the affine
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

## Affine and IPC bodies

Set `scheme="AffineBody"` and configure nonlinear options with
`set_affine_body_parameters()`. Affine contact uses device-resident energy,
gradient, Hessian, CCD, line search, and Krylov paths. Assembly can be
`"MatrixFree"`, `"COO"`, or `"HashTriplet"` depending on the selected solver.

IPC initial states must be strictly feasible: body-body and body-wall gaps
must be positive and inside the configured barrier activation distance only
when contact is intended.

The affine implementation follows the same state/operator/engine ownership as
the older DEM paths. `AffineBodyState.py` owns accepted and trial state,
`AffineBodyOperator.py` owns Taichi contact/linear operations,
`AffineDiffIPC.py` owns affine-mesh nonpenetration projectors, and
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

Implicit AffineBody and LSMPM Soft-Affine IPC runs accept
`enable_step_retry`, `step_retry_max_retries`, `step_retry_reduction`, and
`step_retry_minimum_timestep` in `set_solver`. Retry is bounded, restores the
device/host step-start transaction, and catches only solve/line-search
nonconvergence. Standard DEM/LSDEM/LSMPM stepping is explicit and rejects an
enabled retry policy. `diagnostics_snapshot()` is available on every `DEM`
facade; the affine engines additionally report nonlinear, linear, CCD, and
friction state.

### Choosing affine IPC parameters

GeoTaichi's affine IPC inputs are absolute values. Let `L` be a characteristic
contact length, usually a body or scene bounding-box diagonal:

| Input | Units | Practical starting point |
| --- | --- | --- |
| `Dhat` | m | About `1e-3 L`; enlarge only when the surface resolution or intended contact gap requires it. |
| `BarrierStiffness` | problem dependent | Tune with `Dhat` so the loaded equilibrium gap is positive, below `Dhat`, and comfortably above geometry/CCD round-off. |
| `friction_epsv` | m/s | About `1e-3 L/s`; reduce by decades only while measured slip changes materially. |
| `newton_tolerance` | m/s | Choose from the allowed per-step position error: `newton_tolerance * dt`. |
| `friction_tolerance` | m/s | For verification, use `friction_iterations=-1` and tighten until force balance and slip stop changing. For production, one lagged solve is the reference default and 2--4 solves are usually enough. |

Inside the quadratic static-friction branch, the maximum tangential curvature
is proportional to `1 / (friction_epsv * dt)`. A smaller `friction_epsv`
therefore reduces regularized creep but makes the nonlinear system sharper;
it is not an unconditional accuracy knob.

Read convergence in layers. The linear solve must converge first; the lagged
Newton test is `||delta_x_surface||_inf / dt < newton_tolerance`; the outer
friction test applies the same correction-velocity measure after rebuilding
the frozen normal force and tangent basis. The line search starts from the CCD
feasible step and halves it until the frozen incremental potential is
non-increasing. A healthy step has a negative finite slope, positive accepted
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
