# Monolithic Isogeometric–Material Point Coupling with IPC and Plasticity

`src/igampm` couples deformable NURBS IGA patches with MPM bodies through
either explicit DEM-law contact or one monolithic implicit IPC solve.

## Capabilities

- Two- and three-dimensional Cartesian IGA-MPM coupling plus no-swirl
  axisymmetric coupling in a two-dimensional `(r,z)` meridian.
- Three-dimensional explicit IGA--MPM contact with the shared DEM Linear or
  Hertz--Mindlin spring/dashpot/Coulomb laws.
- Finite-radius MPM point--NURBS-surface closest geometry, control-hull AABB
  culling, persistent tangential history, and equal/opposite rational-basis
  force transfer for the explicit branch.
- Implicit IGA plus implicit direct MPM in one IPC Newton system.
- Elastic IGA coupled to Direct implicit ULMPM with `NeoHookean`, associated
  finite-strain `DruckerPrager`, associated finite-strain `VonMises`, or
  associated finite-strain `ModifiedCamClay`.
- Point-curve and point-surface NURBS contact geometry.
- Device point--NURBS ACCD plus IGA/MPM material feasibility CCD.
- COO/Hash sparse assembly and device Krylov solves.
- Lagged and fully implicit friction Jacobian building blocks.
- Contact surface include/exclude filtering and gallery-aligned VTU output.

IPC requires both child solvers to be implicit. Linear/Hertz--Mindlin requires
both to be explicit, the native particle/grid MPM backend, and
`coupling="Lagrangian"`. The explicit branch is currently 3D Cartesian; use
IPC for 2D or axisymmetric IGA--MPM coupling.
Finite-strain plastic MPM additionally requires `configuration="ULMPM"`;
Direct TLMPM is rejected because it does not yet store an explicit plastic
gradient. In Cartesian 2D, the finite-strain plastic models use the classical
3D plane-strain embedding (`F_33=1` for the total incremental map and a 3-by-3
elastic state); intrinsic 2D plastic invariants are not used. The IGA child
remains elastic.

## Package layout

| Path | Responsibility |
| --- | --- |
| `geometry/` | NURBS closest-point geometry and surface topology |
| `contact/` | Barrier, friction, CCD, and coupled contact assembly |
| `contact/DEMContact.py` | Shared DEM property ownership for explicit contact |
| `contact/ExplicitContact.py` | Device point--NURBS culling, projection, history, and force transfer |
| `engines/CoupledEngine.py` | Composed public engine and field allocation |
| `engines/ExplicitEngine.py` | Explicit IGA/MPM/contact step orchestration |
| `engines/ContactEngine.py` | Point--NURBS queries, barrier assembly, and contact energy |
| `engines/ImplicitEngine.py` | Monolithic assembly, material/contact ACCD, Newton, and Armijo search |
| `engines/FrictionEngine.py` | Lagged and fully implicit friction orchestration and kernels |
| `engines/FullyImplicitFriction.py` | Exact Taichi friction residual/Jacobian blocks |
| `ContactManager.py` | Contact model and parameter validation |
| `GalleryRecorder.py` | Synchronized IGA and MPM VTU frames |
| `mainIGAMPM.py` | Public coupling facade |

## IGA--MPM coupling theory log

This section records the continuum and contact equations independently of any
particular data structure. Superscripts $I$ and $M$ denote IGA and MPM
quantities, respectively. Bold lower-case symbols are vectors, bold upper-case
symbols are second-order tensors, and repeated contact samples are summed.

### 1. Coupled unknowns and interpolation maps

The monolithic displacement unknown is

$$
\boldsymbol{q}
=\left(\boldsymbol{u}^{I},\boldsymbol{u}^{M}\right)^T.
$$

For a NURBS boundary, let $B_A(\boldsymbol{\xi})$ be the tensor-product
B-spline basis and $w_A>0$ its rational weight. The rational basis is

$$
R_A(\boldsymbol{\xi})
=\frac{w_AB_A(\boldsymbol{\xi})}
{\sum_Bw_BB_B(\boldsymbol{\xi})},
\qquad
\sum_AR_A=1.
$$

The current control points and boundary position are

$$
\boldsymbol{P}_A
=\boldsymbol{P}_{A,n}+\boldsymbol{u}_A^I,
\qquad
\boldsymbol{X}(\boldsymbol{\xi})
=\sum_AR_A(\boldsymbol{\xi})\boldsymbol{P}_A.
$$

An MPM boundary sample $s$ is carried by the background grid through

$$
\boldsymbol{x}_s
=\boldsymbol{x}_{s,n}+\sum_iS_{si}\boldsymbol{u}_i^M,
\qquad
\sum_iS_{si}=1.
$$

Here $S_{si}$ is the MPM interpolation weight frozen for the current updated-
Lagrangian step. In implicit IPC the sample is a point primitive and
$d_{min}$ is its complete finite clearance. In explicit DEM-law coupling the
sample is a finite-radius particle, so its radius enters the overlap directly.

For an axisymmetric meridian calculation the distance is still evaluated in
the $(r,z)$ plane, but a sample represents a ring. If $\bar w_s$ is its
meridional measure and $R_s$ its reference radius, the physical contact
measure is

$$
w_s=2\pi R_s\bar w_s.
$$

### 2. Point--NURBS closest-point geometry

Let the parameter dimension be $m=1$ for a curve and $m=2$ for a surface.
Define

$$
\boldsymbol{r}(\boldsymbol{\xi},\boldsymbol{q})
=\boldsymbol{X}(\boldsymbol{\xi})-\boldsymbol{x}_s,
\qquad
d=\|\boldsymbol{r}\|,
$$

and obtain the closest parameter from

$$
\boldsymbol{\xi}^*
=\underset{\boldsymbol{\xi}\in\Omega_\xi}{\arg\min}
\frac12\|\boldsymbol{r}(\boldsymbol{\xi},\boldsymbol{q})\|^2.
$$

For an interior closest point, introduce the tangents
$\boldsymbol{t}_a=\boldsymbol{X}_{,a}$ and the stationarity equations

$$
g_a=\boldsymbol{r}\cdot\boldsymbol{t}_a=0.
$$

The closest-parameter Hessian is

$$
H^\xi_{ab}
=\boldsymbol{t}_a\cdot\boldsymbol{t}_b
+\boldsymbol{r}\cdot\boldsymbol{X}_{,ab}.
$$

For a closest point on a parameter-domain edge, only the free parameter
coordinates are retained in $\boldsymbol{g}$ and $\boldsymbol{H}^\xi$. At a
fixed corner there is no closest-parameter derivative. This active-coordinate
interpretation makes all formulas below apply to interior, edge, and corner
features without inventing derivatives through a clamped parameter.

The differential formulas assume a unique closest point and a nonsingular
free-coordinate $\boldsymbol{H}^\xi$. At an equal-distance feature switch the
distance remains continuous, but its derivative is understood piecewise.

The unit vector from the MPM point to the NURBS boundary is

$$
\boldsymbol{n}=\frac{\boldsymbol{r}}{d}.
$$

### 3. Exact direct point--NURBS coupling derivatives

This subsection gives the complete derivative of the minimized distance,
including motion of the closest NURBS parameter. It is the central formula of
the implicit IGA--MPM coupling.

Collect the free tangents in a matrix $\boldsymbol{T}$ whose row $a$ is
$\boldsymbol{t}_a^T$. For any vector-valued generalized block
$\boldsymbol{y}_a$, define the fixed-parameter residual Jacobian and the
stationarity Jacobian by

$$
\boldsymbol{J}_a
=\frac{\partial\boldsymbol{r}}{\partial\boldsymbol{y}_a},
\qquad
\boldsymbol{M}_a
=\frac{\partial\boldsymbol{g}}{\partial\boldsymbol{y}_a}.
$$

For IGA control point $A$,

$$
\boldsymbol{J}_A=R_A\boldsymbol{I},
$$

$$
(\boldsymbol{M}_A)_{a,:}
=R_A\boldsymbol{t}_a^T
+R_{A,a}\boldsymbol{r}^T.
$$

For MPM grid node $i$,

$$
\boldsymbol{J}_i=-S_{si}\boldsymbol{I},
\qquad
\boldsymbol{M}_i=-S_{si}\boldsymbol{T}.
$$

Implicit differentiation of $\boldsymbol{g}=\boldsymbol{0}$ gives

$$
\frac{\partial\boldsymbol{\xi}^*}{\partial\boldsymbol{y}_a}
=-(\boldsymbol{H}^\xi)^{-1}\boldsymbol{M}_a.
$$

Let the minimized squared distance be

$$
z(\boldsymbol{q})
=\|\boldsymbol{r}(\boldsymbol{\xi}^*(\boldsymbol{q}),\boldsymbol{q})\|^2.
$$

The envelope theorem removes the closest-parameter derivative from the first
derivative:

$$
\frac{\partial z}{\partial\boldsymbol{P}_A}
=2R_A\boldsymbol{r},
\qquad
\frac{\partial z}{\partial\boldsymbol{u}_i^M}
=-2S_{si}\boldsymbol{r}.
$$

The exact second derivative is the Schur complement of the closest-parameter
problem:

$$
\boldsymbol{H}^{z}_{ab}
=2\left[
\boldsymbol{J}_a^T\boldsymbol{J}_b
-\boldsymbol{M}_a^T(\boldsymbol{H}^\xi)^{-1}\boldsymbol{M}_b
\right].
$$

Consequently, the three coupled block families are

$$
\boldsymbol{H}^{z}_{AB}
=2\left[
R_AR_B\boldsymbol{I}
-\boldsymbol{M}_A^T(\boldsymbol{H}^\xi)^{-1}\boldsymbol{M}_B
\right],
$$

$$
\boldsymbol{H}^{z}_{Ai}
=2\left[
-R_AS_{si}\boldsymbol{I}
+\boldsymbol{M}_A^T(\boldsymbol{H}^\xi)^{-1}S_{si}\boldsymbol{T}
\right],
$$

$$
\boldsymbol{H}^{z}_{ij}
=2S_{si}S_{sj}\left[
\boldsymbol{I}
-\boldsymbol{T}^T(\boldsymbol{H}^\xi)^{-1}\boldsymbol{T}
\right].
$$

For a curve, $\boldsymbol{H}^\xi$ is scalar. For a surface it is a $2$ by
$2$ matrix, reduced to a scalar when the closest point lies on a parameter
edge. The mixed identity

$$
\boldsymbol{H}^{z}_{iA}
=\left(\boldsymbol{H}^{z}_{Ai}\right)^T
$$

is the symmetry condition that a monolithic energy Hessian must satisfy.

### 4. Offset IPC barrier and monolithic potential

The scalar offset barrier $b(s)$, its first two derivatives, activation
distance, stiffness scaling, and admissible domain are defined in the
[shared contact-model theory](../physics_model/contact_model/README.md#incremental-potential-contact).
Here $s=d^2-d_{min}^2$ is used only to derive how that shared law pulls back
through the point--NURBS geometry.
For sample measure $w_s$, the contact energy is

$$
E_c=w_sb(s).
$$

The IGA and MPM contact residual blocks are therefore

$$
\boldsymbol{g}_A^c
=2w_sb'(s)R_A\boldsymbol{r},
\qquad
\boldsymbol{g}_i^c
=-2w_sb'(s)S_{si}\boldsymbol{r}.
$$

The exact contact tangent between any two generalized blocks is

$$
\boldsymbol{K}_{ab}^c
=w_s\left[
b''(s)
\frac{\partial s}{\partial\boldsymbol{y}_a}
\left(\frac{\partial s}{\partial\boldsymbol{y}_b}\right)^T
+b'(s)\boldsymbol{H}_{ab}^{z}
\right].
$$

Because both interpolation maps form a partition of unity,

$$
\sum_A\boldsymbol{g}_A^c
+\sum_i\boldsymbol{g}_i^c
=\boldsymbol{0}.
$$

Thus point--NURBS IPC satisfies action--reaction exactly, and common rigid
translation is a null mode of the exact contact Hessian. This conclusion does
not require matching IGA and MPM bases.

For conservative normal contact and lagged friction, the coupled incremental
potential is

$$
\Pi(\boldsymbol{q};\boldsymbol{h}_n)
=\Pi_I(\boldsymbol{u}^I)
+\Pi_M(\boldsymbol{u}^M;\boldsymbol{h}_n)
+\sum_cE_c(\boldsymbol{q})
+D_f(\boldsymbol{q};\widehat{\boldsymbol{q}}).
$$

Here $\boldsymbol{h}_n$ contains accepted MPM material history and the hat
denotes a lagged friction state. The Newton equations have the block form

$$
(\boldsymbol{K}_{II}+\boldsymbol{K}_{II}^c)\Delta\boldsymbol{u}^I
+\boldsymbol{K}_{IM}^c\Delta\boldsymbol{u}^M
=-(\boldsymbol{r}_I+\boldsymbol{g}_I^c),
$$

$$
\boldsymbol{K}_{MI}^c\Delta\boldsymbol{u}^I
+(\boldsymbol{K}_{MM}+\boldsymbol{K}_{MM}^c)\Delta\boldsymbol{u}^M
=-(\boldsymbol{r}_M+\boldsymbol{g}_M^c).
$$

The off-diagonal blocks $\boldsymbol{K}_{IM}^c$ and
$\boldsymbol{K}_{MI}^c$ are the direct IGA--MPM coupling. Omitting them turns
the problem into a staggered force exchange rather than a monolithic IPC
solve.

### 5. Low-rank exact spectral projection

The exact point--NURBS contact Hessian can be indefinite because closest-point
curvature appears in $(\boldsymbol{H}^\xi)^{-1}$. Its rank, however, is at
most the spatial dimension $d_x$ plus the number $m$ of free closest
parameters.

For each IGA control point and MPM node, define reduced Jacobians

$$
\boldsymbol{Q}_A
=\left(R_A\boldsymbol{I},\boldsymbol{M}_A^T\right)^T,
$$

$$
\boldsymbol{Q}_i
=\left(-S_{si}\boldsymbol{I},-S_{si}\boldsymbol{T}^T\right)^T,
$$

and concatenate them as $\boldsymbol{Q}$. Let
$\varphi(d)=b(d^2-d_{min}^2)$ and define

$$
\alpha=\frac{\varphi'(d)}{2d},
\qquad
\beta=\frac14\left[
\frac{\varphi''(d)}{d^2}
-\frac{\varphi'(d)}{d^3}
\right].
$$

The exact local Hessian factors as

$$
\boldsymbol{H}_c
=\boldsymbol{Q}^T\boldsymbol{K}_{red}\boldsymbol{Q},
$$

where the spatial block of $\boldsymbol{K}_{red}$ is

$$
w_s\left[
2\alpha\boldsymbol{I}
+4\beta\boldsymbol{r}\boldsymbol{r}^T
\right],
$$

and its closest-parameter block is

$$
-2w_s\alpha(\boldsymbol{H}^\xi)^{-1}.
$$

Set $\boldsymbol{G}=\boldsymbol{Q}\boldsymbol{Q}^T$. A Euclidean spectral
projection of the full stencil Hessian can be performed entirely in the small
reduced space:

$$
\boldsymbol{W}
=\boldsymbol{G}^{1/2}\boldsymbol{K}_{red}\boldsymbol{G}^{1/2},
$$

$$
\boldsymbol{H}_c^+
=\boldsymbol{Q}^T\boldsymbol{G}^{-1/2}
[\boldsymbol{W}]_+
\boldsymbol{G}^{-1/2}\boldsymbol{Q}.
$$

The inverse square root is a pseudoinverse on the numerical row space, and
$[\boldsymbol{W}]_+$ clamps negative eigenvalues to zero. Projecting this
single reduced operator preserves the IGA--MPM mixed blocks; projecting the
IGA and MPM diagonal blocks separately would not represent the same coupled
energy.

### 6. Friction in the coupled stencil

#### Lagged smooth Coulomb friction

At a lagged closest parameter $\widehat{\boldsymbol{\xi}}$, freeze the normal
$\widehat{\boldsymbol{n}}$, rational basis, MPM weights, and total normal
force $\lambda_n=-w_s\varphi'(d)$. Define the tangent projector

$$
\widehat{\boldsymbol{T}}_n
=\boldsymbol{I}
-\widehat{\boldsymbol{n}}\widehat{\boldsymbol{n}}^T.
$$

The relative tangential increment is

$$
\boldsymbol{z}_t
=\widehat{\boldsymbol{T}}_n\left[
\sum_iS_{si}\Delta\boldsymbol{u}_i^M
-\sum_AR_A(\widehat{\boldsymbol{\xi}})
\Delta\boldsymbol{u}_A^I
\right],
\qquad
v=\frac{\|\boldsymbol{z}_t\|}{\Delta t}.
$$

The lagged friction potential is

$$
D_f=\mu\lambda_nf_0(v).
$$

The scalar $C^1$ function $f_0$ is the
[shared regularized IPC friction potential](../physics_model/contact_model/README.md#regularized-ipc-friction).
Freezing the contact frame makes $D_f$ a symmetric coupled potential with
IGA--MPM mixed blocks.

#### Fully implicit friction

Fully implicit friction instead recomputes the closest parameter, normal,
normal force, and endpoint velocities from the current displacement. For
either child, a Newmark-type update makes the endpoint velocity affine in the
current displacement,

$$
\boldsymbol{v}^{X}
=c_q^X\boldsymbol{u}^{X}
+c_v^X\boldsymbol{v}_n^{X}
+c_a^X\boldsymbol{a}_n^{X},
\qquad
X\in\{I,M\}.
$$

The relative contact velocity is

$$
\boldsymbol{v}_{rel}
=\sum_iS_{si}\boldsymbol{v}_i^M
-\sum_AR_A(\boldsymbol{\xi}^*)\boldsymbol{v}_A^I,
$$

$$
\boldsymbol{T}_n=\boldsymbol{I}-\boldsymbol{n}\boldsymbol{n}^T,
\qquad
\boldsymbol{v}_t=\boldsymbol{T}_n\boldsymbol{v}_{rel},
\qquad
v=\|\boldsymbol{v}_t\|.
$$

The radial resistance $\boldsymbol{\eta}(\boldsymbol{v}_t,\lambda_n)$,
including its $C^1$ or stabilized speed profile, Stribeck interpolation,
viscous term, and exact constitutive differential, is defined in the
[shared fully implicit friction theory](../physics_model/contact_model/README.md#fully-implicit-stribeck-friction).
The IGA--MPM contribution that is not part of that scalar law is the
derivative of the moving point--NURBS frame.
The required geometric differentials include

$$
\mathrm d\boldsymbol{n}
=\frac{\boldsymbol{T}_n}{d}\mathrm d\boldsymbol{r},
\qquad
\mathrm d\lambda_n
=-w_s\varphi''(d)\,\mathrm d d,
$$

$$
\mathrm d\boldsymbol{v}_t
=\boldsymbol{T}_n\mathrm d\boldsymbol{v}_{rel}
-\left[
\mathrm d\boldsymbol{n}\,\boldsymbol{n}^T
+\boldsymbol{n}\,\mathrm d\boldsymbol{n}^T
\right]\boldsymbol{v}_{rel}.
$$

Together with $\mathrm d\boldsymbol{\xi}^*$ from Section 3, these terms
differentiate the moving contact frame and force magnitude. The resulting
Jacobian is generally nonsymmetric, so the nonlinear problem is naturally
viewed as a residual equation rather than minimization of one scalar friction
potential.

### 7. Additive continuous collision detection and material feasibility

For a Newton direction, additive continuous collision detection (ACCD) writes
the moving point and control points as

$$
\boldsymbol{x}_s(\alpha)
=\boldsymbol{x}_s^0+\alpha\Delta\boldsymbol{x}_s,
$$

$$
\boldsymbol{P}_A(\alpha)
=\boldsymbol{P}_A^0+\alpha\Delta\boldsymbol{P}_A,
\qquad
0\leq\alpha\leq1.
$$

Positive rational weights give the motion bound

$$
\left\|
\Delta\boldsymbol{x}_s
-\sum_AR_A(\boldsymbol{\xi})\Delta\boldsymbol{P}_A
\right\|
\leq
\max_A\|\Delta\boldsymbol{x}_s-\Delta\boldsymbol{P}_A\|
=L.
$$

At the current trial fraction, let $e=d-d_{min}$ and let
$0\leq\eta<1$ be the fraction of the current excess gap retained as a safety
margin. Conservative advancement uses

$$
\Delta\alpha
=(1-\eta)\frac{e}{L},
$$

and recomputes the global closest point after every increment. This keeps the
accepted path strictly outside $d=d_{min}$ even when the closest NURBS span
or parameter changes.

Contact feasibility alone is insufficient. For every IGA quadrature point
and MPM particle, material continuous collision detection also requires

$$
\det\boldsymbol{F}(\alpha)>J_{min}>0.
$$

The admissible line-search fraction is therefore

$$
\alpha_{max}
=\min(\alpha_{contact},\alpha_{IGA},\alpha_{MPM},1).
$$

### 8. Plastic constitutive models inside the IPC solve

The complete multiplicative Hencky, associated Drucker--Prager, von Mises,
and Modified Cam--Clay equations are maintained in the
[shared constitutive-model theory](../physics_model/consititutive_model/README.md#finite-strain-multiplicative-plasticity).
This section records only how that local material response enters the coupled
IGA--MPM solve.

The IGA body remains elastic. For a Direct ULMPM particle $p$, the current
total deformation gradient is

$$
\boldsymbol{F}_p(\boldsymbol{u}^M)
=\left[
\boldsymbol{I}
+\sum_i\boldsymbol{u}_i^M\otimes\nabla N_{pi}
\right]\boldsymbol{F}_{p,n}.
$$

Plane strain supplies the three-dimensional embedding with incremental
$F_{33}=1$. In an axisymmetric meridian, the additional hoop stretch is

$$
F_{\theta\theta}=1+\frac{u_r}{R}.
$$

For accepted material history $\boldsymbol{h}_{p,n}$, let the shared
constitutive model return an incremental density, first Piola stress, and
algorithmic tangent:

$$
W_p=W_p(\boldsymbol{F}_p;\boldsymbol{h}_{p,n}),
\qquad
\boldsymbol{P}_p=\frac{\partial W_p}{\partial\boldsymbol{F}_p},
\qquad
\mathbb{A}_p^{alg}
=\frac{\partial\boldsymbol{P}_p}{\partial\boldsymbol{F}_p}.
$$

Define

$$
\boldsymbol{B}_{pi}
=\frac{\partial\boldsymbol{F}_p}
{\partial\boldsymbol{u}_i^M}.
$$

The material residual and tangent entering the MPM diagonal block are

$$
\boldsymbol{r}_{i}^{mat}
=\sum_pV_p^0\boldsymbol{P}_p:\boldsymbol{B}_{pi},
$$

$$
\boldsymbol{K}_{ij}^{mat}
=\sum_pV_p^0
\boldsymbol{B}_{pi}:\mathbb{A}_p^{alg}:\boldsymbol{B}_{pj}.
$$

A projected Newton step may spectrally clamp the symmetric material tangent
to its positive-semidefinite part. This changes the search metric but not the
constitutive stress, local return, or accepted history.

The accepted history is frozen during each global Newton trial. At every new
$\boldsymbol{q}$, the material model recomputes its local return and tangent;
the same displacement simultaneously changes the MPM boundary sample,
closest NURBS parameter, active contact set, and all IPC mixed blocks.
Plasticity therefore enters IPC through the monolithic equilibrium path even
though the scalar barrier law is material independent.

Modified Cam--Clay adds one outer material fixed point: freeze the lagged
preconsolidation pressure, solve the complete coupled equilibrium, refresh the
hardening pressure using the shared constitutive law, and repeat until both
the material fixed point and global residual converge.

For plastic states, use the residual merit

$$
\mathcal{M}(\boldsymbol{q})
=\frac12\|\boldsymbol{R}_{free}(\boldsymbol{q})\|^2.
$$

For a search direction $\boldsymbol{p}$,

$$
\mathcal{M}'(0)
=\boldsymbol{R}^T\boldsymbol{K}\boldsymbol{p}.
$$

An exact Newton direction obeys
$\boldsymbol{K}\boldsymbol{p}=-\boldsymbol{R}$ and therefore
$\mathcal{M}'(0)=-\|\boldsymbol{R}\|^2$. Every accepted trial must also
satisfy the contact and determinant bounds of Section 7.

Plastic deformation, equivalent plastic strain, volumetric plastic strain,
and hardening variables are committed only after the complete IGA--MPM step
converges. A rejected trial or rejected time step leaves all accepted
constitutive history unchanged.

### 9. Explicit finite-radius IGA--MPM coupling

The explicit branch treats an MPM particle of radius $R_p$ against a NURBS
surface. With

$$
d=\min_{\boldsymbol{\xi}}
\|\boldsymbol{X}(\boldsymbol{\xi})-\boldsymbol{x}_p\|,
\qquad
\delta=R_p-d,
$$

contact is active for $\delta>0$. The outward normal acting on the particle is

$$
\boldsymbol{n}
=\frac{\boldsymbol{x}_p-\boldsymbol{X}(\boldsymbol{\xi}^*)}{d}.
$$

The surface and relative velocities are

$$
\boldsymbol{v}_s
=\sum_AR_A(\boldsymbol{\xi}^*)\boldsymbol{v}_A^I,
\qquad
\boldsymbol{v}_{rel}=\boldsymbol{v}_p-\boldsymbol{v}_s,
$$

$$
v_n=\boldsymbol{v}_{rel}\cdot\boldsymbol{n},
\qquad
\boldsymbol{v}_t
=\boldsymbol{v}_{rel}-v_n\boldsymbol{n}.
$$

The Linear and Hertz--Mindlin normal/tangential laws, damping, Coulomb return,
and stored contact energies are defined in the
[shared DEM contact theory](../physics_model/contact_model/README.md#discrete-contact-kinematics-and-dem-laws).
Their resultant force on the particle is

$$
\boldsymbol{F}_p
=f_n\boldsymbol{n}+\boldsymbol{F}_t.
$$

It is transferred to IGA control point $A$ as

$$
\boldsymbol{f}_A^I
=-R_A(\boldsymbol{\xi}^*)\boldsymbol{F}_p.
$$

Partition of unity gives exact linear-momentum balance,

$$
\boldsymbol{F}_p+\sum_A\boldsymbol{f}_A^I=\boldsymbol{0}.
$$

For the normal force, the moment also cancels exactly:

$$
\boldsymbol{x}_p\times(f_n\boldsymbol{n})
+\sum_A\boldsymbol{P}_A\times
[-R_Af_n\boldsymbol{n}]
=(\boldsymbol{x}_p-\boldsymbol{X}^*)
\times(f_n\boldsymbol{n})
=\boldsymbol{0}.
$$

The corresponding contact power is

$$
\mathcal{P}_c
=\boldsymbol{F}_p\cdot\boldsymbol{v}_p
+\sum_A\boldsymbol{f}_A^I\cdot\boldsymbol{v}_A^I
=\boldsymbol{F}_p\cdot
(\boldsymbol{v}_p-\boldsymbol{v}_s).
$$

The normal spring stores and returns energy, while dashpots and sliding
friction dissipate it. Tangential force at a finite lever arm produces the
couple $(\boldsymbol{x}_p-\boldsymbol{X}^*)\times\boldsymbol{F}_t$; exact
angular-momentum balance would additionally require particle spin or
rotational surface degrees of freedom.

The explicit contact scale suggests the stability estimate

$$
\Delta t_c
\sim\sqrt{\frac{m_{min}}{k_{max}}}.
$$

The synchronized explicit step must satisfy both this contact restriction and
the MPM material-wave CFL restriction.

## Minimal setup

```python
import geotaichi as gt

gt.init(dim=3, arch="gpu", default_fp="float64", log=False)

coupling = gt.IGAMPM(
    log=False,
    contact_model="IPC",
    kappa=4.0e5,
    dhat=0.05,
    dmin=0.005,
    contact_ccd_safety=0.9,
    assemble_type="HashTriplet",  # or "COO"
)
coupling.set_configuration(
    dimension=3,
    coupling_scheme="IGAMPM",
    contact_model="IPC",
    activate_friction=False,
)

# Configure `coupling.iga` as an implicit IGA model and `coupling.mpm` as an
# implicit Direct MPM model before building.
result = coupling.run(steps=20)
```

For plastic MPM, configure the Direct child with `configuration="ULMPM"` and
select a supported finite-strain model through its ordinary material API:

```python
coupling.mpm.add_material(
    model="DruckerPrager",
    young_modulus=2.0e5,
    poisson_ratio=0.3,
    density=1800.0,
    Cohesion=1200.0,
    FrictionAngle=30.0,
    DilationAngle=30.0,
    dpType="Circumscribed",
)
```

`VonMises` instead takes `YieldStress` and optional `HardeningModulus`.
`ModifiedCamClay` takes the critical-state ratio, compression and swelling
indices, reference void ratio, and preconsolidation pressure.
Assembly, point--NURBS ACCD, material feasibility CCD, PSD-projected lagged
Hessians, and Armijo search are unchanged. Step preparation, nonlinear solve,
and acceptance form one transaction. Plastic history is committed only after
the coupled step is accepted; a preparation, solve, or post-commit failure
restores IGA control-point state, Direct-MPM particle/grid state, total `F0`, plastic
history, and entry displacements. A valid preinitialized Direct-MPM `F0` is
preserved during `IGAMPM.build()`; only an all-zero field receives the default
identity state. A partially initialized field, a non-finite entry, or any
nonpositive determinant is rejected across the full material dimension
(including the third plane-strain/axisymmetric component).

The implicit IPC coupling accepts a `coupling={...}` dictionary in
`IGAMPM.set_solver()` with `enable_step_retry`,
`step_retry_max_retries`, `step_retry_reduction`, and
`step_retry_minimum_timestep`. Only recognized nonlinear/linear/line-search
nonconvergence is retried after the existing full coupled rollback; capacity
and configuration errors remain immediate failures. The accepted reduced
timestep is synchronized to both children. Explicit DEM-law contact rejects
an enabled policy. `diagnostics_snapshot()` is available for both branches.

Complete examples are available in `examples/igampm/`, including
`iga_mpm_explicit_dem_contact.py` for the explicit branch.

For explicit coupling, configure both children as `Explicit`, construct
`IGAMPM` before MPM particle allocation (so it selects Lagrangian coupling),
then add one law for every active MPM-material/IGA-patch pair:

```python
coupling = gt.IGAMPM(iga, mpm, contact_model="Linear", log=False)
coupling.set_configuration(dimension=3, contact_model="Linear")
coupling.add_property(
    MPMmaterial=1,
    IGAbody=0,
    property={
        "NormalStiffness": 1.0e5,
        "TangentialStiffness": 5.0e4,
        "StaticFriction": 0.4,
        "DynamicFriction": 0.3,
        "NormalViscousDamping": 0.1,
        "TangentialViscousDamping": 0.1,
    },
)
coupling.run(steps=100)
```

`HertzMindlin` instead takes `ShearModulus`, `Poisson`, `Restitution`, and
friction coefficients. Nonzero `RollingFriction` is rejected because IGA
control points have translational but no independent rotational DOFs.
Equal-and-opposite force transfer and the normal-force moment are exact. A
finite-radius tangential force can leave the same unresolved frictional couple
as the MPM--DEM point-force convention; this branch does not claim exact
angular-momentum conservation under friction.
The explicit runner applies the native MPM CFL factor to the minimum of the
MPM material-wave and contact-law timestep estimates and keeps both children
synchronized. IGA has no independent spectral stability estimator yet, so
users must still verify the chosen step against an IGA refinement study.

The explicit broad phase rebuilds one control-hull AABB per selected NURBS
boundary every step, followed by all-knot-span projected Gauss--Newton
closest-point search.
It deliberately has no Verlet multiplier: typical IGA solids expose only a
small number of whole NURBS boundaries, so a stable particle--surface table
plus AABB culling avoids neighbor-list rebuild/history remapping overhead.

Axisymmetric runs set `dimension=2`, `axisymmetric=True`, and one shared
`axis_offset` on the IGA child, Direct MPM child, and IGAMPM coupling. The
radial coordinate is component zero. Both material blocks use the full
three-dimensional no-swirl map, including `F_theta_theta=r/R`, while their
unknowns remain the two meridional displacement components. Reference volume
and MPM contact-ring weights include `2*pi*R`; point--curve NURBS distance,
ACCD, friction, and Armijo line search operate in the meridian plane. The axis
itself is excluded from material/contact quadrature, so all participating
reference points must satisfy `R > 0` relative to `axis_offset`.
IGA reference patch Jacobians must also be finite and positive at every
quadrature point. IGA Neumann values are resultant control-point loads; users
supplying an axisymmetric traction density must integrate its `2*pi*R` measure
before calling `NeumannBoundary.append`.

## Monolithic implicit solve

The implicit engine combines IGA elasticity, MPM mechanics, IPC barrier
terms, and optional friction in one sparse system. Candidate generation,
closest-point derivatives, contact Hessians, CCD, sparse scatter, and Krylov
iterations remain device resident. Newton and lagged-friction outer loops use
Python only for scalar control flow.

Contact and storage parameters are frozen after `build()`. Reconfiguring the
dimension, friction mode, or contact model after fields and kernels have been
created raises an error. Construct a new coupling object to change those
options.

The contact step is visible in `engines/ImplicitEngine.py`: every current
MPM-surface-sample/NURBS-surface pair receives the positive-weight bound
`max_i ||dp-dP_i||`. Thus both the point and all NURBS control points move, a
common translation cancels exactly, and a closest-parameter change remains
covered. The boundary sample is a point primitive and `dmin` is its finite
clearance; the earlier research prototype instead subtracted `particle.rad`.
One Taichi thread runs the complete bounded ACCD loop for each
sample--surface pair. It evaluates `p0 + alpha*dp` and
`P_i0 + alpha*dP_i` without changing the shared contact surface, reruns the
closest-point query after every increment, stores the pair TOC, and contributes
to a device atomic-min reduction. Boundaries with the same degree signature
share one kernel specialization; Python dispatches these basis groups but does
not iterate or synchronize each ACCD increment. The IGA and MPM `ccd()` calls
separately add deformation-gradient determinant bounds before Armijo search.
Each moving closest-point state evaluates the rational basis and its first and
second derivatives once, then interpolates both control-point positions and
directions in the same support loop. The production Armijo path also reuses the
base DCD set it has just prepared. The public standalone contact-step helper
keeps its conservative verification and restore pass by default.

The production fully implicit friction law has no NumPy implementation under
`src/igampm`.  Geometry derivatives, endpoint velocities, residual/Jacobian
blocks, sparse scatter, residual merit, line search, and BiCGSTAB remain in
Taichi.  The NumPy implementation is retained only as a test oracle under
`tests/helpers/igampm_fully_implicit_friction_reference.py`.

## Assembly-only use

Advanced callers may invoke `run(solve=False)` to initialize and return the
selected Taichi COO or HashTriplet system. This mode is intended for external nonlinear
drivers and does not emulate the standard lagged-friction outer iteration. It
does not construct a NumPy/SciPy assembly backend; an external SciPy solve must
be an explicitly selected linear-system boundary.
`assemble_monolithic_newton_system()` returns the device dictionary. A host
conversion is allowed only at an explicitly
selected external linear-solve/output boundary; COO uses `_to_scipy()`, while
HashTriplet uses `to_scipy(system["active_nodes"])`.
Legacy injection of preassembled NumPy/SciPy subsystem blocks is rejected.

## Tests

Unit coverage is grouped with the IGA tests under `tests/unit/iga/`. Coupled
integration tests are under `tests/integration/igampm/`, and complete scenes
are under `examples/igampm/`.
