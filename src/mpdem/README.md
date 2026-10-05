# MPM–DEM, MPM–LSDEM Immersed Boundary, and MPM–ABD IPC Coupling

`src/mpdem` couples GeoTaichi MPM particles or continua with DEM particles and
walls. The public facade is named `DEMPM` internally and is exposed as both
`geotaichi.DEMPM()` and `geotaichi.MPDEM()`.

[Theory log and derivations](#mpdem-theory-log) | [Examples](../../examples/)

## Example-backed capabilities

- Explicit MPM–DEM interaction: [sphere impact into a granular bed](../../examples/mpdem/MultiSphere/SphereImpactToGranularBed/plane_strain.py).
- MPM–LSDEM contact: [box water entry](../../examples/mpdem/LevelSet/WaterImpact/box.py) and [rigid–soft particle contact](../../examples/mpdem/LevelSet/SoftRigid/rigid_soft_sphere_drop_box.py).
- Semi-resolved incompressible sphere coupling: [sphere falling in oil](../../examples/cfdem/SemiResolved/SphereFallingOil/sphere.py).
- Fully resolved incompressible MPM–LSDEM immersed boundary method (IBM): [3D dam break](../../examples/cfdem/FullyResolved/IBMLevelSetDamBreak3D/dam_break_levelset_ibm_3d.py).
- Hybrid two-phase MPM–LSDEM coupling: [saturated-bed wavemaker](../../examples/mmpm/TwoPhaseLSDEMCoupling/wavemaker_lsdem_particles_3d.py) and [sphere impact](../../examples/mmpm/TwoPhaseLSDEMCoupling/sphere_impact_submerged_bed_3d.py).
- Fully coupled material point method–affine body dynamics (MPM–ABD) IPC: [Solid MPM impact](../../examples/mpdem/AffineBody/ABDImpactDP/direct_mpm_abd_impact.py) and [hyperelastic soft-MPM contact](../../examples/mpdem/AffineBody/AffineSoftSphereIPC/affine_soft_sphere_ipc.py).
- Incompressible fluid–ABD IBM: [moving affine-body example](../../examples/mpm/IncompressibleFluid/affine_body_coupling_3d.py).

## MPM AffineBody route selection

MPM–ABD configurations are distinguished by their physical model and contact operator, not by the internal MPM implementation name.

| Physical route | Coupling operator | Concrete example | Restrictions |
| --- | --- | --- | --- |
| Incompressible fluid–ABD | Volume-fraction IBM and pressure projection | [Moving affine body](../../examples/mpm/IncompressibleFluid/affine_body_coupling_3d.py) | 3D, fixed timestep, no DEM subcycling; not IPC |
| Hyperelastic soft-particle MPM–ABD | Soft-grid displacements and affine controls in one IPC Newton system | [Soft spheres](../../examples/mpdem/AffineBody/AffineSoftSphereIPC/affine_soft_sphere_ipc.py) | Hyperelastic only; follow the `scheme="LSMPM"` template lifecycle |
| Continuum solid MPM–ABD | Active MPM-grid displacements and affine controls in one IPC Newton system | [Solid-bed impact](../../examples/mpdem/AffineBody/ABDImpactDP/direct_mpm_abd_impact.py) | 3D ULMPM; Neo-Hookean, Drucker–Prager, or von Mises; lagged friction |

These examples contain the required API selectors and allocation order. They do not represent different MPM balance laws. Ordinary explicit MPM–DEM/LSDEM and soft-particle contact are documented in Sections 1–10; solid IPC coupling is described in Sections 11–13 and 15.

## Package layout

| Path | Responsibility |
| --- | --- |
| `mainDEMPM.py` | Public coupling facade |
| `Simulation.py` | Coupling configuration, capacities, and shared time state |
| `Engine.py`, `DEMPMBase.py` | Explicit exchange order, synchronized stepping, and DEM subcycling |
| `ContactManager.py`, `contact/` | Cross-system contact geometry, search, history, and scalar-law selection |
| `GenerateManager.py`, `generator/` | Mixed generation, overlap removal, and boundary-radius adaptation |
| `fluid_dynamics/` | Drag correlations, porosity coupling, and volume-fraction IBM |
| `engines/SoftAffineIPC*` | Hyperelastic soft-MPM/AffineBody fully coupled IPC |
| `engines/DirectAffineIPC*` | Continuum-solid MPM/AffineBody fully coupled IPC and plastic history |
| `Recorder.py` | Child output and persistent cross-contact records |

## MPDEM theory log

### 1. Explicit MPM--DEM contact kinematics

Let an MPM contact sample and a DEM sphere have centers
$\boldsymbol{x}_p,\boldsymbol{x}_d$, radii $R_p,R_d$, masses $m_p,m_d$,
velocities $\boldsymbol{v}_p,\boldsymbol{v}_d$, and angular velocities
$\boldsymbol{\omega}_p,\boldsymbol{\omega}_d$. Define

$$
\boldsymbol{n}
=\frac{\boldsymbol{x}_p-\boldsymbol{x}_d}
{\|\boldsymbol{x}_p-\boldsymbol{x}_d\|},
\qquad
g=\|\boldsymbol{x}_p-\boldsymbol{x}_d\|-R_p-R_d.
$$

The gap-shifted contact evaluation point is

$$
\boldsymbol{x}_c
=\boldsymbol{x}_d+\left(R_d+\frac{g}{2}\right)\boldsymbol{n},
$$

and the relative contact velocity is

$$
\boldsymbol{v}_{rel}
=\boldsymbol{v}_p
+\boldsymbol{\omega}_p\times(\boldsymbol{x}_c-\boldsymbol{x}_p)
-\boldsymbol{v}_d
-\boldsymbol{\omega}_d\times(\boldsymbol{x}_c-\boldsymbol{x}_d).
$$

The effective mass and radius are

$$
m^*=\left(\frac{1}{m_p}+\frac{1}{m_d}\right)^{-1},
\qquad
R^*=\left(\frac{1}{R_p+g/2}+\frac{1}{R_d+g/2}\right)^{-1}.
$$

If a shared contact law returns
$\boldsymbol{F}=\boldsymbol{F}_n+\boldsymbol{F}_t$ and a rolling/twisting
couple $\boldsymbol{M}_r$, the two bodies receive

$$
\boldsymbol{F}_p=\boldsymbol{F},
\qquad
\boldsymbol{F}_d=-\boldsymbol{F},
$$

$$
\boldsymbol{\tau}_p
=(\boldsymbol{x}_c-\boldsymbol{x}_p)\times\boldsymbol{F}
+\boldsymbol{M}_r,
$$

$$
\boldsymbol{\tau}_d
=(\boldsymbol{x}_c-\boldsymbol{x}_d)\times(-\boldsymbol{F})
-\boldsymbol{M}_r.
$$

The MPM particle resultant is transferred to its background nodes by

$$
\boldsymbol{f}_i^{M}=N_{pi}\boldsymbol{F}_p.
$$

Since $\sum_iN_{pi}=1$,

$$
\sum_i\boldsymbol{f}_i^{M}+\boldsymbol{F}_d=\boldsymbol{0}.
$$

Thus the cross-system stencil preserves linear action--reaction before the
child time integrators advance their states. The contact-point moments give
the corresponding angular-momentum balance.

For enhanced boundary coupling, a boundary MPM sample supplies an oriented
normal $\boldsymbol{n}_p$. The DEM sphere uses the local plane gap

$$
g=(\boldsymbol{x}_d-\boldsymbol{x}_p)\cdot\boldsymbol{n}_p-R_d-R_p.
$$

A conservative candidate test uses

$$
\|\boldsymbol{x}_d-\boldsymbol{x}_p\|
<R_d+\sqrt{2}R_p+s,
$$

where $s$ is the combined search skin. Samples with coincident normals are
collapsed to one local plane so a single physical boundary patch is not
counted repeatedly.

### 2. MPM point against an LSDEM signed-distance body

Let the LSDEM body have center $\boldsymbol{c}$, rotation
$\boldsymbol{R}$, and body-frame signed-distance interpolant
$\phi(\boldsymbol{X})$. For an MPM sample,

$$
\boldsymbol{X}
=\boldsymbol{R}^T(\boldsymbol{x}_p-\boldsymbol{c}),
\qquad
g=\phi(\boldsymbol{X})-R_p.
$$

The spatial raw gradient and unit normal are

$$
\boldsymbol{g}_{\phi}
=\boldsymbol{R}\nabla_X\phi(\boldsymbol{X}),
\qquad
\boldsymbol{n}
=\frac{\boldsymbol{g}_{\phi}}{\|\boldsymbol{g}_{\phi}\|}.
$$

The common contact point is

$$
\boldsymbol{x}_c
=\boldsymbol{x}_p
-\left(R_p+\frac{g}{2}\right)\boldsymbol{n}.
$$

Its relative velocity is

$$
\boldsymbol{v}_{rel}
=\boldsymbol{v}_p
+\boldsymbol{\omega}_p\times(\boldsymbol{x}_c-\boldsymbol{x}_p)
-\boldsymbol{v}_c
-\boldsymbol{\omega}_d\times(\boldsymbol{x}_c-\boldsymbol{c}).
$$

For any normal potential $U(g)$, the exact conservative force on the MPM
sample is

$$
\boldsymbol{F}_p^n
=-\frac{\partial U}{\partial g}\boldsymbol{g}_{\phi}.
$$

Using the raw interpolated gradient is required for an exact work identity;
normalizing it is only appropriate for direction-dependent damping and
friction. The rigid-body reaction and moment are

$$
\boldsymbol{F}_d=-\boldsymbol{F}_p,
\qquad
\boldsymbol{\tau}_d
=(\boldsymbol{x}_c-\boldsymbol{c})\times\boldsymbol{F}_d.
$$

### 3. Plane, finite-wall, and digital-elevation contact

For a wall with projection $\boldsymbol{x}_q$ and outward normal
$\boldsymbol{n}_w$, define

$$
d=(\boldsymbol{x}_p-\boldsymbol{x}_q)\cdot\boldsymbol{n}_w,
\qquad
g=d-R_p,
$$

$$
\boldsymbol{x}_c
=\boldsymbol{x}_q+\frac{g}{2}\boldsymbol{n}_w.
$$

For a finite wall, the sphere--plane intersection radius is

$$
r_c=\sqrt{R_p^2-d^2}.
$$

If $\mathcal W$ is the wall footprint in its plane, the finite-contact
multiplier is

$$
\chi
=\frac{
\mathrm{area}
\left[
\mathcal D(\boldsymbol{x}_q,r_c)\cap\mathcal W
\right]
}{\pi r_c^2}.
$$

The applied force and particle moment are

$$
\boldsymbol{F}_p
=\chi(\boldsymbol{F}_n+\boldsymbol{F}_t),
\qquad
\boldsymbol{\tau}_p
=(\boldsymbol{x}_c-\boldsymbol{x}_p)\times\boldsymbol{F}_p
+\chi\boldsymbol{M}_r.
$$

An active dynamic wall receives the equal and opposite wrench.

For a digital-elevation cell with spacing $h$ and corner heights
$z_{00},z_{10},z_{01},z_{11}$, write its local coordinates as

$$
\xi=\frac{x}{h}-\left\lfloor\frac{x}{h}\right\rfloor,
\qquad
\eta=\frac{y}{h}-\left\lfloor\frac{y}{h}\right\rfloor.
$$

The cell is split along the $00$--$11$ diagonal. When $\xi\geq\eta$, the
active triangle is $(00,10,11)$; otherwise it is $(11,01,00)$. Its normal is

$$
\boldsymbol{n}_h
=\frac{
(\boldsymbol{x}_1-\boldsymbol{x}_0)
\times
(\boldsymbol{x}_2-\boldsymbol{x}_0)
}{
\|(\boldsymbol{x}_1-\boldsymbol{x}_0)
\times
(\boldsymbol{x}_2-\boldsymbol{x}_0)\|
}.
$$

The terrain gap then uses the same plane formula,

$$
g=(\boldsymbol{x}_p-\boldsymbol{x}_0)\cdot\boldsymbol{n}_h-R_p.
$$

If exactly one elevation sample is missing, the remaining three corners form
the triangle; fewer than three valid samples make the cell inactive.

### 4. Scalar contact laws and stable explicit step

The Linear, Hertz--Mindlin, energy-conserving penalty, explicit logarithmic
Barrier, and rolling/twisting laws are defined in the
[shared contact-model theory](../physics_model/contact_model/README.md#discrete-contact-kinematics-and-dem-laws).
The fluid-particle no-slip penalty used by liquid MPM is defined in the
[same shared theory](../physics_model/contact_model/README.md#fluid-particle-no-slip-penalty-law).

For Linear, energy-conserving, and fluid-particle penalty contact, a coupling
stability estimate is

$$
\Delta t_c
=\sqrt{\frac{m_{min}}{k_{max}}}.
$$

When adaptive Linear stiffness is selected,

$$
k_n=\frac{\pi}{2}R_{max}E^*,
\qquad
k_t=\frac{k_n}{\gamma_k},
$$

where $\gamma_k$ is the prescribed normal-to-shear ratio. For Hertz--Mindlin
contact, the empirical estimate is

$$
\Delta t_c
=\frac{
\pi R_{min}\sqrt{\rho_{min}/E_{max}}
}{0.1631\nu_{max}+0.8766}.
$$

The explicit coupled step uses

$$
\Delta t
\leq C_{CFL}
\min\left(
\Delta t_M,
\Delta t_D,
\Delta t_c
\right).
$$

### 5. Broad phase, history, and compact contact storage

For sphere-like MPM and DEM samples, a conservative Verlet candidate obeys

$$
\|\boldsymbol{x}_p-\boldsymbol{x}_d\|
\leq R_p+R_d+s_M+s_D.
$$

LSDEM candidates first pass a bounding-volume test and then the narrow SDF
condition

$$
\phi(\boldsymbol{X})-R_p<s_M+s_D.
$$

The stored list remains valid while the accumulated relative motion stays
inside its skin budget,

$$
\Delta x_M^{max}+\Delta x_D^{max}<s_M+s_D.
$$

Linked cells, hierarchical linked cells, and BVH change only how this
conservative set is generated. If source object $i$ owns $c_i$ candidates,
the compact offsets satisfy

$$
P_0=0,
\qquad
P_{i+1}=P_i+c_i.
$$

The active contacts of object $i$ occupy $[P_i,P_{i+1})$. Tangential,
rolling, twisting, and energy-conserving normal histories are inherited by
the persistent endpoint pair; opening contacts are cleared rather than
carrying stale history into a later pair.

For maximum radii $R_M^{max},R_D^{max}$ and a base coordination estimate
$C_0$, the conservative particle-pair allocation factor is

$$
q_V
=\left(
\frac{R_M^{max}+R_D^{max}+s_M+s_D}
{R_M^{max}+R_D^{max}}
\right)^3,
$$

so a source capacity $N_M^{max}$ uses approximately

$$
N_{pair}^{pot}
=N_M^{max}\left\lfloor C_0q_V\right\rfloor.
$$

With compaction ratio $c_{pp}>0$, the persistent contact-list capacity is

$$
N_{pair}^{hist}
=\left\lceil c_{pp}N_{pair}^{pot}\right\rceil.
$$

For a digital-elevation grid with spacing $h$ and search radius $R_s$, a
safe triangle coordination bound is

$$
n_a=\left\lceil\frac{2R_s}{h}\right\rceil+2,
\qquad
C_h=2n_a^2.
$$

### 6. Mixed generation and overlap removal

When an MPM sample has no explicit contact radius, its conservative radius is
formed from its particle half-size vector $\boldsymbol{l}_p$:

$$
R_p=\|\boldsymbol{l}_p\|.
$$

An MPM sample is removed from a newly generated DEM sphere when

$$
\|\boldsymbol{x}_p-\boldsymbol{x}_d\|-R_p-R_d<0.
$$

For LSDEM, the bounding-sphere test is followed by

$$
\phi\left[
\boldsymbol{R}^T(\boldsymbol{x}_p-\boldsymbol{c})
\right]<R_p.
$$

Near a generated wall, let

$$
d_{min}=\min_w d_w(\boldsymbol{x}_p).
$$

The adaptive boundary rule is

$$
d_{min}<\frac{R_p}{2}
\quad\Longrightarrow\quad
\text{remove the sample},
$$

and, for the surviving transition layer,

$$
\frac{R_p}{2}\leq d_{min}<R_p
\quad\Longrightarrow\quad
R_p^{new}=d_{min}.
$$

For a level-set template scaled by $s$, its equivalent and bounding radii are

$$
R_{eq}=sR_{eq}^{0},
\qquad
R_b=sR_b^{0}.
$$

### 7. Explicit exchange order and DEM subcycling

After cross-contact assembly, each DEM body advances the rigid balance

$$
m_d\dot{\boldsymbol{v}}_d
=\boldsymbol{F}_{ext}+\boldsymbol{F}_{contact},
$$

$$
\boldsymbol{I}_d\dot{\boldsymbol{\omega}}_d
+\boldsymbol{\omega}_d\times
(\boldsymbol{I}_d\boldsymbol{\omega}_d)
=\boldsymbol{\tau}_{ext}+\boldsymbol{\tau}_{contact},
$$

while the MPM contact resultant enters the ordinary particle-to-grid momentum
balance. Both children advance over the same coupled interval $\Delta t$.

When the DEM stability limit is smaller, the number of rigid substeps is

$$
N_D
=\max\left(
1,
\left\lceil\frac{\Delta t}{\Delta t_D}\right\rceil
\right),
\qquad
\delta t_D=\frac{\Delta t}{N_D}.
$$

For incompressible fluid coupling, the hydrodynamic load evaluated over the
outer step is held during these rigid contact substeps. DEM contact and wall
forces are recomputed at every $\delta t_D$.

### 8. Explicit LSMPM soft-particle--level-set coupling

The explicit LSMPM route represents each deformable particle by material
points on a body-attached mechanical grid and by a level set used only for
contact geometry. It can coexist with rigid LSDEM bodies in the same contact
search. This is distinct from the ordinary MPM-continuum--LSDEM coupling in
Section 2 and from the fully coupled soft--AffineBody IPC route below.

For soft body $b$, let $p$ denote material points and $i$ its mechanical-grid
nodes. The reference lumped mass and current nodal force are

$$
m_i=\sum_{p\in b}N_{pi}m_p,
$$

$$
\boldsymbol{f}_i
=\sum_{p\in b}
\left[
N_{pi}m_p\boldsymbol{g}
-V_{p0}\boldsymbol{P}_p\nabla_0N_{pi}
\right]
+\boldsymbol{f}_i^c
+\boldsymbol{f}_i^{ext}.
$$

Here $\boldsymbol{P}_p=\partial\Psi/\partial\boldsymbol{F}_p$. The supported
finite-strain elastic energies are defined in the
[shared constitutive-model theory](../physics_model/consititutive_model/README.md);
the present section concerns their explicit coupling to level-set contact.

Let $a$ be a boundary quadrature sample of body $b$, with reference local
coordinate $\widehat{\boldsymbol{x}}_a$ and normalized area weight $\omega_a$.
For body center $\boldsymbol{c}_b$, rotation $\boldsymbol{R}_b$, and scale
$s_b$, its world position and reference surface measure are

$$
\boldsymbol{x}_a
=\boldsymbol{c}_b
+\boldsymbol{R}_b(s_b\widehat{\boldsymbol{x}}_a),
\qquad
A_a=s_b^2A_b^0\omega_a.
$$

When a template surface measure is unavailable, the spherical equivalent
$A_b^0=4\pi(R_b^{eq})^2$ supplies the reference area. A soft surface-node
contact mass is

$$
m_a=\frac{M_b}{N_{\Gamma,b}},
$$

where $M_b$ and $N_{\Gamma,b}$ are the body mass and number of boundary
samples. The usual two-body effective mass is

$$
m^*=\left(\frac{1}{m_a}+\frac{1}{m_s}\right)^{-1},
$$

with the same surface-node definition used for $m_s$ when the slave is also
soft.

For a slave level-set body $s$, transform the master sample into its local
frame and evaluate

$$
\boldsymbol{X}_a
=\boldsymbol{R}_s^T(\boldsymbol{x}_a-\boldsymbol{c}_s),
\qquad
g_a=\phi_s(\boldsymbol{X}_a),
$$

$$
\boldsymbol{g}_{\phi,a}
=\boldsymbol{R}_s\nabla_X\phi_s(\boldsymbol{X}_a),
\qquad
\boldsymbol{n}_a
=\frac{\boldsymbol{g}_{\phi,a}}
{\|\boldsymbol{g}_{\phi,a}\|}.
$$

The common evaluation point is

$$
\boldsymbol{x}_{c,a}
=\boldsymbol{x}_a-\frac{g_a}{2}\boldsymbol{n}_a.
$$

Soft--rigid contact is integrated only by querying the soft trace against the
rigid SDF. Soft--soft contact retains both directed queries, each with half
the surface measure:

$$
\widetilde A_a
=\begin{cases}
A_a, & \text{soft--rigid or soft--wall},\\
\frac12A_a, & \text{each directed soft--soft query}.
\end{cases}
$$

This removes duplicate soft--rigid work while keeping a symmetric
soft--soft surface quadrature.

The velocity of a soft master sample is its current surface-trace velocity
$\boldsymbol{v}_a$. A soft slave velocity is reconstructed from its moving
trace rather than from a rigid-body approximation. Let $W_{aJ}$ be the
trilinear weight from slave surface sample $a$ to slave SDF node $J$. Define

$$
q_J=\sum_r A_rW_{rJ},
\qquad
\overline{\boldsymbol{v}}_J
=\frac{\sum_rA_rW_{rJ}\boldsymbol{v}_r}{q_J}.
$$

At the projected slave point $\boldsymbol{X}_{c}$,

$$
D(\boldsymbol{X}_{c})
=\sum_JW_J(\boldsymbol{X}_{c})q_J,
$$

$$
\boldsymbol{v}_s(\boldsymbol{X}_{c})
=\frac{
\sum_JW_J(\boldsymbol{X}_{c})q_J
\overline{\boldsymbol{v}}_J
}{D(\boldsymbol{X}_{c})}.
$$

The contact relative velocity and gap rate are therefore

$$
\boldsymbol{v}_{rel,a}
=\boldsymbol{v}_a-\boldsymbol{v}_s(\boldsymbol{X}_{c}),
\qquad
\dot g_a
=\boldsymbol{v}_{rel,a}\cdot\boldsymbol{g}_{\phi,a}.
$$

For a scalar conservative normal potential $\widehat U(g)$, the
area-integrated contribution and master force are

$$
U_a=\widetilde A_a\widehat U(g_a),
$$

$$
\boldsymbol{F}_a^n
=-\widetilde A_a
\frac{\partial\widehat U}{\partial g}(g_a)
\boldsymbol{g}_{\phi,a}.
$$

Consequently,

$$
\boldsymbol{F}_a^n\cdot\boldsymbol{v}_{rel,a}
=-\frac{\mathrm dU_a}{\mathrm dt}.
$$

The raw SDF gradient, rather than only its normalized direction, is required
for this work identity. Normal damping, tangential friction, Hertz--Mindlin,
Linear, and the energy-conserving explicit laws use the scalar definitions in
the
[shared contact-model theory](../physics_model/contact_model/README.md#discrete-contact-kinematics-and-dem-laws).

For the work-conjugate soft--soft normal history, let $\delta_a^n\geq0$ be
the stored penetration. Its explicit update is

$$
\delta_a^{n+1}
=\max\left(
\delta_a^n-\Delta t\,\dot g_a,
0
\right).
$$

A newly activated contact starts with $\delta_a^n=0$; a negative transported
SDF value is allowed to detect closure but is not inserted as pre-existing
spring energy. An opening, newly inactive trace therefore cannot inject
normal contact energy.

For a rigid slave, the exact reaction is

$$
\boldsymbol{F}_s=-\boldsymbol{F}_a,
\qquad
\boldsymbol{\tau}_s
=(\boldsymbol{x}_{c,a}-\boldsymbol{c}_s)
\times(-\boldsymbol{F}_a).
$$

For a soft slave, distribute the reaction first to its SDF trace nodes,

$$
\boldsymbol{Q}_J
=-\frac{W_J(\boldsymbol{X}_{c})}
{D(\boldsymbol{X}_{c})}\boldsymbol{F}_a,
$$

then gather it to slave surface samples,

$$
\boldsymbol{f}_r^{reaction}
=A_r\sum_JW_{rJ}\boldsymbol{Q}_J.
$$

Because $q_J=\sum_rA_rW_{rJ}$,

$$
\sum_r\boldsymbol{f}_r^{reaction}
=\sum_Jq_J\boldsymbol{Q}_J
=-\boldsymbol{F}_a.
$$

Thus the soft-trace transfer preserves the complete reaction even though the
slave contact velocity was reconstructed through its SDF grid.

Each surface force is finally transferred to the mass-carrying mechanical
grid support. If

$$
S_a=\sum_{j:m_j>0}N_{aj},
$$

then

$$
\boldsymbol{f}_i^c
\mathrel{+}=\frac{N_{ai}}{S_a}\boldsymbol{f}_a,
\qquad m_i>0,
$$

and hence

$$
\sum_i\boldsymbol{f}_i^c=\boldsymbol{f}_a.
$$

After contact assembly, the explicit grid, particle, and deformation updates
are

$$
\boldsymbol{v}_i^{n+1}
=\boldsymbol{v}_i^n
+\Delta t\frac{\boldsymbol{f}_i^n}{m_i},
$$

$$
\boldsymbol{v}_p^{n+1}
=\alpha_s\sum_iN_{pi}\boldsymbol{v}_i^{n+1}
+(1-\alpha_s)
\left(
\boldsymbol{v}_p^n
+\Delta t\sum_iN_{pi}\frac{\boldsymbol{f}_i^n}{m_i}
\right),
$$

$$
\boldsymbol{x}_p^{n+1}
=\boldsymbol{x}_p^n
+\Delta t\sum_iN_{pi}\boldsymbol{v}_i^{n+1},
$$

$$
\boldsymbol{F}_p^{n+1}
=\boldsymbol{F}_p^n
+\Delta t\sum_i
\boldsymbol{v}_i^{n+1}
\left(\nabla_0N_{pi}\right)^T.
$$

The deformed boundary trace updates the contact surface, while the level set
may be transported by

$$
\frac{\partial\phi}{\partial t}
+\boldsymbol{u}\cdot\nabla\phi=0.
$$

MacCormack/WENO transport, signed-distance reinitialization, and conservative
volume correction are summarized in the
[MPM soft-particle theory](../mpm/README.md#15-soft-particle-lsmpm-and-level-set-transport).

The explicit step must satisfy both bulk and contact limits. With
$k_{max}^{\Gamma}$ the largest area-scaled contact tangent,

$$
\Delta t
\leq C_{CFL}
\min\left(
\Delta t_{soft},
\Delta t_{rigid},
\sqrt{\frac{m_{min}^{\Gamma}}{k_{max}^{\Gamma}}}
\right).
$$

The maintained soft--wall case is
`examples/mpdem/LevelSet/SoftRigid/soft_sphere_rolling.py`; the mixed
rigid--soft, soft--soft, and wall case is
`examples/mpdem/LevelSet/SoftRigid/rigid_soft_sphere_drop_box.py`.

### 9. Semi-resolved incompressible sphere coupling

Let a DEM sphere have volume

$$
V_p=\frac{4}{3}\pi R_p^3.
$$

The compact Gaussian used to distribute it to cell centers is

$$
W_h(\boldsymbol{r})
=\frac{1}{\pi^{3/2}h^3}
\exp\left(-\frac{\|\boldsymbol{r}\|^2}{h^2}\right)
$$

for $\|\boldsymbol{r}\|/h\leq c$, and zero outside the selected support.
For sphere $p$, the truncated-kernel normalization is

$$
Z_p=\sum_IW_{pI}\Delta V_I.
$$

The cell solid fraction and porosity are

$$
\phi_{s,I}
=\mathrm{clamp}
\left(
\sum_p\frac{V_pW_{pI}}{Z_p},
0,
1
\right),
\qquad
\epsilon_{f,I}=1-\phi_{s,I}.
$$

An expanded Gaussian support reconstructs the undisturbed fluid state at a
sphere:

$$
\overline{\boldsymbol{u}}_{f,p}
=\frac{\sum_I\widetilde W_{pI}\boldsymbol{u}_{f,I}}
{\sum_I\widetilde W_{pI}},
$$

$$
\overline\epsilon_{f,p}
=\mathrm{clamp}
\left(
\frac{\sum_I\widetilde W_{pI}\epsilon_{f,I}}
{\sum_I\widetilde W_{pI}},
0.05,
1
\right).
$$

With slip velocity

$$
\boldsymbol{u}_r
=\overline{\boldsymbol{u}}_{f,p}-\boldsymbol{v}_p,
$$

the particle Reynolds number is

$$
Re_p
=\frac{
2\overline\epsilon_{f,p}\rho_fR_p\|\boldsymbol{u}_r\|
}{\mu_f}.
$$

The Stokes branch uses

$$
C_D=\frac{24}{Re_p}
\qquad\text{for}\qquad Re_p\leq1,
$$

and the high-Reynolds branch uses

$$
C_D=0.44
\qquad\text{for}\qquad Re_p\geq1000.
$$

Between these limits, the selectable correlations are

$$
C_D^{SN}
=\frac{24}{Re_p}
\left(1+0.15Re_p^{0.687}\right),
$$

$$
C_D^{BL}
=\frac{24}{Re_p}
\left(1+0.15Re_p^{0.681}\right)
+\frac{0.407}{1+8710/Re_p},
$$

$$
C_D^{E}
=\left(0.63+\frac{4.8}{\sqrt{Re_p}}\right)^2,
$$

$$
C_D^{A}
=\frac{24}{9.06^2}
\left(1+\frac{9.06}{\sqrt{Re_p}}\right)^2.
$$

The dense Gidaspow coefficient for
$\overline\epsilon_{f,p}\leq0.8$ is

$$
\beta
=\frac{
150(1-\overline\epsilon_{f,p})^2\mu_f
}{
\overline\epsilon_{f,p}(2R_p)^2
}
+\frac{
1.75(1-\overline\epsilon_{f,p})\rho_f
}{2R_p}
\|\boldsymbol{u}_r\|.
$$

The corresponding linear-form drag is

$$
\boldsymbol{F}_d
=\frac{V_p\beta}{1-\overline\epsilon_{f,p}}
\boldsymbol{u}_r.
$$

For $\overline\epsilon_{f,p}>0.8$, its dilute continuation is

$$
\boldsymbol{F}_d
=V_p
\frac{0.75C_D^{SN}\rho_f\|\boldsymbol{u}_r\|}{2R_p}
\overline\epsilon_{f,p}^{-1.65}
\boldsymbol{u}_r.
$$

The alternative quadratic law defines

$$
\kappa
=3.7-0.65
\exp\left[
-\frac{1}{2}
\left(1.5-\log_{10}Re_p\right)^2
\right]
$$

and

$$
\boldsymbol{F}_d
=\frac{1}{2}\pi C_D\rho_fR_p^2
\overline\epsilon_{f,p}^{2-\kappa}
\|\boldsymbol{u}_r\|\boldsymbol{u}_r.
$$

The equal and opposite fluid reaction is normalized over its influence
support:

$$
w_{pI}
=\frac{W_{pI}}{\sum_JW_{pJ}},
\qquad
\boldsymbol{f}_{I}^{drag}
=-\sum_pw_{pI}\boldsymbol{F}_{d,p}.
$$

Therefore

$$
\sum_I\boldsymbol{f}_{I}^{drag}
=-\sum_p\boldsymbol{F}_{d,p}.
$$

The pressure projection enforces the porosity continuity equation

$$
\frac{\partial\epsilon_f}{\partial t}
+\nabla\cdot(\epsilon_f\boldsymbol{u}_f)=0.
$$

The pressure force returned to a sphere is

$$
\boldsymbol{F}_{p}^{pressure}
=-V_p\overline{\nabla p}_p.
$$

An optional added mass

$$
m_a=C_A\rho_fV_p
$$

changes the translational acceleration to

$$
\boldsymbol{a}_p
=\frac{\boldsymbol{F}_h+m_p\boldsymbol{g}}{m_p+m_a}.
$$

Without changing the stored true particle mass, the equivalent non-gravity
load is

$$
\boldsymbol{F}_h^{eff}
=\frac{m_p}{m_p+m_a}
\left(\boldsymbol{F}_h+m_p\boldsymbol{g}\right)
-m_p\boldsymbol{g}.
$$

For a sphere approaching a plane wall, the unresolved lubrication correction
is

$$
\boldsymbol{F}_{lub}
=-6\pi\mu_fR_p^2
\left(
\frac{1}{h_{eff}}-\frac{1}{h_a}
\right)
(\boldsymbol{v}_p\cdot\boldsymbol{n})\boldsymbol{n},
$$

where

$$
h_{eff}=\max(h,h_{min})
$$

and the correction is active only for $h<h_a$.

### 10. Fully resolved LSDEM and AffineBody volume-fraction IBM

For a fluid cell with eight corner SDF samples $\phi_a$, the solid fraction
estimate is

$$
\phi_s
=\mathrm{clamp}
\left(
\frac{\sum_{a=1}^{8}\langle-\phi_a\rangle_+}
{\sum_{a=1}^{8}|\phi_a|},
0,
1
\right).
$$

If several bodies overlap the same cell, let $\phi_s^b$ be the unclamped
contribution of body $b$. The stored mixture fields are

$$
\phi_s=\min\left(1,\sum_b\phi_s^b\right),
$$

$$
\rho_s
=\frac{\sum_b\phi_s^b\rho_s^b}{\sum_b\phi_s^b},
\qquad
\boldsymbol{u}_s
=\frac{\sum_b\phi_s^b\boldsymbol{u}_s^b}{\sum_b\phi_s^b}.
$$

For LSDEM rigid body $b$,

$$
\boldsymbol{u}_s^b(\boldsymbol{x})
=\boldsymbol{v}_b
+\boldsymbol{\omega}_b\times(\boldsymbol{x}-\boldsymbol{c}_b).
$$

The mixture density and solid mass fraction are

$$
\rho_m
=(1-\phi_s)\rho_f+\phi_s\rho_s,
$$

$$
\varphi_s
=\mathrm{clamp}
\left(
\frac{\phi_s\rho_s}{\rho_m},
0,
1
\right).
$$

The direct-forcing immersed-boundary source is

$$
\boldsymbol{f}_{IBM}
=\rho_m\varphi_s
\frac{\boldsymbol{u}_s-\boldsymbol{u}_f^*}{\Delta t}.
$$

The pressure projection uses $\rho_m$ in its variable-density face
coefficient. With

$$
\boldsymbol{b}_{\sigma}
=-\nabla p+\mu_f\nabla^2\boldsymbol{u}_f,
$$

the force density returned to the solid phase is

$$
\boldsymbol{f}_s
=\varphi_s\boldsymbol{b}_{\sigma}
-(1-\varphi_s)\boldsymbol{f}_{IBM}.
$$

The share belonging to body $b$ is

$$
\alpha_b
=\frac{\phi_s^b}{\sum_k\phi_s^k}.
$$

The rigid resultant and torque are

$$
\boldsymbol{F}_b
=\sum_I\alpha_{bI}\boldsymbol{f}_{s,I}\Delta V_I,
$$

$$
\boldsymbol{\tau}_b
=\sum_I
(\boldsymbol{x}_I-\boldsymbol{c}_b)
\times
\left(
\alpha_{bI}\boldsymbol{f}_{s,I}\Delta V_I
\right).
$$

For an AffineBody, define its current frame

$$
\boldsymbol{A}
=\left[
\boldsymbol{y}_1-\boldsymbol{y}_0,
\boldsymbol{y}_2-\boldsymbol{y}_0,
\boldsymbol{y}_3-\boldsymbol{y}_0
\right]
$$

and material coordinate

$$
\boldsymbol{\xi}
=\boldsymbol{A}^{-1}(\boldsymbol{x}-\boldsymbol{y}_0).
$$

Its affine weights and solid velocity are

$$
w_0=1-\xi_1-\xi_2-\xi_3,
\qquad
w_a=\xi_a,
$$

$$
\boldsymbol{u}_s(\boldsymbol{x})
=\sum_{a=0}^{3}w_a\dot{\boldsymbol{y}}_a.
$$

The cell force is pulled back to the four controls by virtual work:

$$
\boldsymbol{Q}_a
=\sum_Iw_a(\boldsymbol{x}_I)
\alpha_{bI}\boldsymbol{f}_{s,I}\Delta V_I.
$$

Since $\sum_aw_a=1$, the generalized forces recover the physical resultant,

$$
\sum_{a=0}^{3}\boldsymbol{Q}_a=\boldsymbol{F}_b.
$$

#### 10.4 Two-phase two-point MPM--LSDEM hybrid

For 3D `TwoPhaseDoubleLayer` with `solver_type="SemiImplicit"`, LSDEM acts on
the two point sets through different operators.  A solid material point uses
the ordinary point--level-set gap from Section 2,

$$
g_p=\phi_b(\boldsymbol{x}_p)-r_p,
$$

and its contact resultant is transferred to the solid MPM grid.  A fluid
point never enters that contact list.  Instead, the LSDEM signed distance is
sampled at the eight cell vertices to obtain a cell solid fraction
$\alpha_c$ and rigid velocity $\boldsymbol{u}_{b,c}$.  At a MAC face,

$$
\alpha_f=\mathrm{clamp}\!\left(
\frac{\alpha_L+\alpha_R}{2},0,1\right),
\qquad
u_f^{IBM}=u_f^*+\alpha_f(u_{b,f}-u_f^*).
$$

The direct-forcing reaction stored on the adjacent cells is

$$
R_f=-\frac{m_f(u_f^{IBM}-u_f^*)}{\Delta t}.
$$

The constraint is applied once to the predicted velocity before the pressure
solve and again after pressure correction.  Thus the projection cannot leave
a residual velocity through the immersed body.  The load returned to rigid
body $b$ is partitioned by its local fraction $\alpha_{b,c}$:

$$
\boldsymbol{F}_b=
\sum_c\frac{\alpha_{b,c}}{\sum_k\alpha_{k,c}}
\left[
\omega_c(-\nabla p+\mu\nabla^2\boldsymbol{u})V_c
+(1-\omega_c)\boldsymbol{R}_c
\right],
$$

$$
\omega_c=
\frac{\alpha_c\rho_s}
{(1-\alpha_c)\rho_f+\alpha_c\rho_s},
\qquad
\boldsymbol{\tau}_b=
\sum_c(\boldsymbol{x}_c-\boldsymbol{x}_b)\times\boldsymbol{F}_{b,c}.
$$

Because the cross-contact neighbor list stores a leading MPM prefix, solid
body templates must be inserted before fluid body templates.  Configuration
validation rejects a mixed prefix rather than silently applying contact to
fluid points.  This route is 3D, uses `scheme="LSDEM"`, and currently requires
`DEMTimestep == Timestep`.

### 11. Hyperelastic soft-MPM incremental mechanics

Soft particles use active background-grid displacements
$\boldsymbol{u}_i$. Their current positions are

$$
\boldsymbol{x}_p
=\boldsymbol{x}_{p,n}
+\sum_iN_{pi}\boldsymbol{u}_i,
$$

and their trial deformation gradients are

$$
\boldsymbol{F}_p
=\boldsymbol{F}_{p,n}
+\sum_i\boldsymbol{u}_i\otimes\nabla N_{pi}.
$$

For background damping coefficient $c_b$, the soft incremental potential is

$$
\Pi_S
=\Pi_{pred}
+\Delta t^2\sum_pV_{p,0}\Psi(\boldsymbol{F}_p),
$$

$$
\Pi_{pred}
=\sum_i
\left(
\frac{1}{2}m_i(1+c_b\Delta t)\|\boldsymbol{u}_i\|^2
-m_i
\left[
\boldsymbol{v}_{i,n}\Delta t
+\boldsymbol{g}\Delta t^2
\right]\cdot\boldsymbol{u}_i
\right).
$$

Its grid residual is

$$
\boldsymbol{r}_i^S
=m_i(1+c_b\Delta t)\boldsymbol{u}_i
-m_i\boldsymbol{v}_{i,n}\Delta t
-m_i\boldsymbol{g}\Delta t^2
+\Delta t^2\sum_pV_{p,0}
\boldsymbol{P}_p\nabla N_{pi},
$$

where

$$
\boldsymbol{P}_p
=\frac{\partial\Psi}{\partial\boldsymbol{F}_p}.
$$

With material tangent

$$
\mathbb{C}_p
=\frac{\partial^2\Psi}{\partial\boldsymbol{F}_p^2},
$$

the grid tangent has the form

$$
\boldsymbol{K}_{ij}^S
=m_i(1+c_b\Delta t)\delta_{ij}\boldsymbol{I}
+\Delta t^2\sum_pV_{p,0}
\boldsymbol{B}_{pi}^T\mathbb{C}_p\boldsymbol{B}_{pj}.
$$

The supported Neo-Hookean, Hencky, Gent, and Hydrogel energies are maintained
in the
[shared constitutive-model theory](../physics_model/consititutive_model/README.md).
This soft route is hyperelastic; finite-strain plasticity belongs to the
continuum-solid MPM route below.

### 12. Soft--soft Barrier IPC and exact grid pullback

Let $A_p$ be the lumped surface measure of soft point $p$. If no explicit
surface quadrature is available, use

$$
A_p=V_{p,0}^{2/3}.
$$

For two surface points,

$$
\boldsymbol{r}=\boldsymbol{x}_p-\boldsymbol{x}_q,
\qquad
s=\boldsymbol{r}\cdot\boldsymbol{r},
$$

$$
A_{pq}=\frac{A_p+A_q}{2}.
$$

The ordinary IPC contribution is

$$
E_{pq}^{c}
=\Delta t^2A_{pq}b(s).
$$

Its relative gradient and Hessian are

$$
\boldsymbol{g}_r
=2\Delta t^2A_{pq}b'(s)\boldsymbol{r},
$$

$$
\boldsymbol{H}_r
=\Delta t^2A_{pq}
\left[
4b''(s)\boldsymbol{r}\boldsymbol{r}^T
+2b'(s)\boldsymbol{I}
\right].
$$

The point gradients are

$$
\boldsymbol{g}_p=\boldsymbol{g}_r,
\qquad
\boldsymbol{g}_q=-\boldsymbol{g}_r.
$$

Pulling them through both MPM supports gives

$$
\boldsymbol{r}_i^c
=N_{pi}\boldsymbol{g}_p+N_{qi}\boldsymbol{g}_q,
$$

$$
\boldsymbol{K}_{ij}^c
=\sum_{\alpha\in\{p,q\}}
\sum_{\beta\in\{p,q\}}
N_{\alpha i}
\boldsymbol{H}_{\alpha\beta}
N_{\beta j}.
$$

Partition of unity preserves the common-translation null mode. The scalar
barrier and regularized friction potential are defined in the
[shared IPC theory](../physics_model/contact_model/README.md#incremental-potential-contact).

### 13. Soft-MPM--AffineBody mesh IPC

For soft point $p$ and AffineBody surface vertex $v$,

$$
\boldsymbol{x}_p
=\boldsymbol{x}_{p,n}
+\sum_iN_{pi}\boldsymbol{u}_i,
$$

$$
\boldsymbol{x}_v^A
=\sum_{a=0}^{3}W_{va}\boldsymbol{y}_a,
\qquad
\sum_{a=0}^{3}W_{va}=1.
$$

For a point--triangle stencil with local sites
$\boldsymbol{z}_s$, write every site as

$$
\boldsymbol{z}_s
=\sum_AB_{sA}\boldsymbol{q}_A,
\qquad
\boldsymbol{q}=(\boldsymbol{y},\boldsymbol{u}).
$$

The soft site uses supports $N_{pi}\boldsymbol{I}$ and each affine vertex
uses supports $W_{va}\boldsymbol{I}$. If the local contact energy has blocks

$$
\boldsymbol{g}_s
=\frac{\partial E_c}{\partial\boldsymbol{z}_s},
\qquad
\boldsymbol{H}_{st}
=\frac{\partial^2E_c}
{\partial\boldsymbol{z}_s\partial\boldsymbol{z}_t},
$$

the exact fully coupled pullback is

$$
\boldsymbol{r}_A^c
=\sum_sB_{sA}^T\boldsymbol{g}_s,
$$

$$
\boldsymbol{K}_{AB}^c
=\sum_s\sum_t
B_{sA}^T\boldsymbol{H}_{st}B_{tB}.
$$

The mixed face--vertex measure is

$$
A_{FV}=\frac{A_p}{4},
$$

so its barrier energy is

$$
E_{FV}^{c}
=\Delta t^2A_{FV}b(d_{PT}^2).
$$

With AffineBody self energy $\Pi_A$, the coupled potential is

$$
\Pi_{SA}
=\Pi_A(\boldsymbol{y})
+\Pi_S(\boldsymbol{u})
+\sum E_{SS}^{c}
+\sum E_{SA}^{c}
+D_f.
$$

The Newton system is

$$
\left(
\boldsymbol{K}_{AA}+\boldsymbol{K}_{AA}^{c}
\right)\Delta\boldsymbol{y}
+\boldsymbol{K}_{AS}^{c}\Delta\boldsymbol{u}
=-\left(\boldsymbol{r}_A+\boldsymbol{r}_A^c\right),
$$

$$
\boldsymbol{K}_{SA}^{c}\Delta\boldsymbol{y}
+\left(
\boldsymbol{K}_{SS}+\boldsymbol{K}_{SS}^{c}
\right)\Delta\boldsymbol{u}
=-\left(\boldsymbol{r}_S+\boldsymbol{r}_S^c\right).
$$

The nonzero off-diagonal blocks are the direct soft--affine coupling. Lagged
friction freezes the closest coordinates, normal, and normal-force magnitude
during each Newton solve. The accepted step satisfies

$$
\alpha_{max}
=\min\left(
1,
\alpha_A,
\alpha_S,
\alpha_{SS},
\alpha_{SA}
\right),
$$

where the bounds respectively protect AffineBody deformation/self-contact,
soft deformation, soft--soft contact, and mixed point--triangle contact.

### 15. Solid MPM--AffineBody IPC and plasticity

The continuum-solid MPM formulation uses active grid displacements and AffineBody controls
in one generalized vector,

$$
\boldsymbol{q}
=\left(
\boldsymbol{y},
\boldsymbol{u}^{M}
\right).
$$

Its coupled incremental potential for elastic or incremental-potential
materials is

$$
\Pi_{DA}
=\Pi_A(\boldsymbol{y})
+\Pi_M(\boldsymbol{u}^{M};\boldsymbol{h}_n)
+E_{AA}^{c}
+E_{MM}^{c}
+E_{AM}^{c}
+D_f.
$$

The mixed point--triangle sites use the two linear maps

$$
\boldsymbol{x}_p
=\boldsymbol{x}_{p,n}
+\sum_iN_{pi}\boldsymbol{u}_i^M,
$$

$$
\boldsymbol{x}_v^A
=\sum_{a=0}^{3}W_{va}\boldsymbol{y}_a.
$$

Therefore the mixed residual and Hessian use the exact pullback from Section
12. In block form,

$$
\boldsymbol{K}_{AA}^{tot}\Delta\boldsymbol{y}
+\boldsymbol{K}_{AM}^{c}\Delta\boldsymbol{u}^{M}
=-\boldsymbol{r}_A^{tot},
$$

$$
\boldsymbol{K}_{MA}^{c}\Delta\boldsymbol{y}
+\boldsymbol{K}_{MM}^{tot}\Delta\boldsymbol{u}^{M}
=-\boldsymbol{r}_M^{tot}.
$$

For finite-strain plasticity,

$$
\boldsymbol{F}_{p}^{tr}
=\left[
\boldsymbol{I}
+\sum_i\boldsymbol{u}_i^M\otimes\nabla N_{pi}
\right]\boldsymbol{F}_{p,n}.
$$

The accepted history $\boldsymbol{h}_n$ is frozen throughout Newton and line
search trials. The return mapping supplies

$$
\boldsymbol{P}_p
=\boldsymbol{P}
(\boldsymbol{F}_{p}^{tr},\boldsymbol{h}_n),
\qquad
\mathbb{C}_p^{alg}
=\frac{\partial\boldsymbol{P}_p}
{\partial\boldsymbol{F}_{p}^{tr}}.
$$

History is committed only after the entire MPM--AffineBody equilibrium and
friction fixed point converge. The Drucker--Prager and von Mises updates are defined in the
[shared constitutive theory](../physics_model/consititutive_model/README.md#finite-strain-multiplicative-plasticity).

The common feasible line-search limit is

$$
\alpha_{max}
=\min\left(
1,
\alpha_A,
\alpha_M,
\alpha_{AM}
\right).
$$

The state is committed transactionally: AffineBody controls, MPM grid
displacements, particles, deformation gradients, and plastic history either
all advance or all return to the beginning of the step.

### 16. Synchronized state and recorded balances

All explicit child clocks satisfy

$$
t_M^n=t_D^n=t_C^n,
\qquad
t^{n+1}=t^n+\Delta t.
$$

For a compact contact list, the recorded number of contacts is

$$
N_c=P_{N_{source}},
$$

and each stored endpoint pair carries its inherited tangential history. When
energy tracking is active, the reported coupling totals are

$$
E_{elastic}^{C}=\sum_cE_{elastic,c},
$$

$$
E_{friction}^{C}=\sum_cE_{friction,c},
\qquad
E_{damping}^{C}=\sum_cE_{damping,c}.
$$

These are coupling contributions only; complete-system balance additionally
includes MPM strain/kinetic energy, DEM kinetic energy, gravity, prescribed
loads, and child contact energies.

## Lifecycle

The coupling object owns existing DEM and MPM facades. Configure and allocate
both child solvers before allocating coupling contact storage:

```python
import geotaichi as gt

gt.init(arch="gpu", log=False)

dem = gt.DEM(log=False)
mpm = gt.MPM(log=False)
dem.sims.set_dem_coupling(True)
mpm.sims.set_mpm_coupling("Lagrangian")

# Configure, allocate, and populate `dem` and `mpm` first.
coupling = gt.MPDEM(dem=dem, mpm=mpm, coupling="Lagrangian", log=False)
coupling.set_configuration(
    domain=[2.0, 1.0, 1.0],
    coupling_scheme="MPDEM",
    particle_interaction=True,
    wall_interaction=True,
    log=False,
)
coupling.set_solver(
    {
        "Timestep": 1.0e-5,
        "SimulationTime": 0.2,
        "SaveInterval": 1.0e-3,
        "SavePath": "OutputData/mpdem_case",
    },
    log=False,
)
coupling.memory_allocate(
    memory={
        "body_coordination_number": 64,
        "wall_coordination_number": 8,
        "compaction_ratio": [0.4, 0.3],
    }
)
coupling.choose_contact_model(
    particle_particle_contact_model="Hertz Mindlin Model",
    particle_wall_contact_model="Hertz Mindlin Model",
)
coupling.add_property(
    DEMmaterial=0,
    MPMmaterial=1,
    property={
        "ShearModulus": 4.0e6,
        "Poisson": 0.25,
        "Friction": 0.4,
        "Restitution": 0.1,
    },
)
coupling.run()
```

Exact child-solver memory dictionaries and body definitions depend on the
chosen DEM/MPM schemes. Reuse the corresponding child-module README and the
integration tests rather than duplicating those definitions here.

## Runtime and ownership

DEM and MPM keep their own state fields. `src/mpdem` owns coupling candidate
lists, contact histories, and coupling forces. Search, contact assembly, and
state exchange run in Taichi kernels. Python coordinates the two solvers,
capacity validation, output, and optional external CFD callbacks.

The incompressible sphere path solves
`d(epsilon_f)/dt + div(epsilon_f u_f) = 0`; drag reaction is distributed to
available MAC faces with porosity-weighted mass. The 3D LSDEM path retains
fictitious fluid inside moving particles, uses mixed density in the pressure
projection, and integrates pressure, viscous stress, and IBM source through
the volume-fraction force formula. `solid_sdf_cut_cell` remains available for
fixed walls but is not generated from moving LSDEM bodies.

The time steps stay synchronized by default. In incompressible CFDEM--LSDEM,
`DEMTimestep` may be set below `Timestep`; the fluid load is then held over the
coupled step and the existing DEM contact/search kernels are subcycled only
while a Verlet candidate exists. Storage-defining options and maximum contact
capacities should be fixed before the coupled engine is built.

The standard MPDEM/DEMPM and CFDEM loops are explicit. The four bounded retry
keys (`enable_step_retry`, `step_retry_max_retries`,
`step_retry_reduction`, and `step_retry_minimum_timestep`) are accepted only
for the fully coupled LSMPM Soft-Affine and solid MPM--AffineBody IPC routes,
whose coupled device states are transactional. All public coupling facades
expose `diagnostics_snapshot()`; explicit routes report common progress and
the fully coupled routes add nonlinear/contact failure details.

The LSMPM Soft-Affine IPC route is split by responsibility:
`SoftAffineIPCBase.py` owns shared state and transfer helpers,
`SoftAffineIPCOperator.py` owns device contact assembly,
and `SoftAffineIPCEngine.py` owns the selected advance/retry lifecycle. Optional cross-contact
and friction modes are bound when the engine is initialized; physical
nonlinear convergence and retry acceptance remain runtime decisions.

Soft-particle/LSMPM paths are hyperelastic-only: finite-strain
Drucker--Prager and von Mises are rejected at material setup and remain
available only through continuum-solid MPM.

## Tests

Coupling lifecycle and soft-affine IPC tests are under `tests/unit/mpdem/`.
Verification cases are under `tests/verification/mpdem/`.
`test_incompressible_coupling_kernels.py` exercises the production Taichi
kernels for porosity continuity, conservative drag transfer, mixed-density
PCG/MGPCG coefficients and viscosity, light-particle IBM weighting, and the
Eq. (28) force. It also checks volume preservation for a dense eight-sphere
Gaussian map and convergence of the SDF sphere volume/pressure resultant.
`test_incompressible_lsdem_end_to_end.py` runs the public LSDEM example with
PCG and MGPCG on the same grid, verifies the actual LSDEM solid-fraction
volume and final pressure residual, and compares their saved state. These
tests also guard the staggered-grid reset that prevents inactive face values
from leaking into a later active step.
