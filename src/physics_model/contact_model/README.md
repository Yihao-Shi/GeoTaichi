# Contact Mechanics for DEM, Barrier IPC, Friction, and Rolling Resistance

`src/physics_model/contact_model` owns scalar contact potentials, force laws,
friction regularizations, and history evolution shared by GeoTaichi solvers.
Solver READMEs retain search, geometric discretization, assembly, and time
integration, then link here instead of copying a constitutive law.

## Discrete contact kinematics and DEM laws

### Contact geometry

For two spheres, let the unit normal point from particle $j$ to particle
$i$:

$$
\boldsymbol{n}
=\frac{\boldsymbol{x}_i-\boldsymbol{x}_j}
       {\|\boldsymbol{x}_i-\boldsymbol{x}_j\|}.
$$

The signed normal gap and penetration are

$$
g_n=\|\boldsymbol{x}_i-\boldsymbol{x}_j\|-(R_i+R_j),
\qquad
\delta_n=\langle-g_n\rangle_+,
\qquad
\langle z\rangle_+=\max(z,0).
$$

Contact exists when $g_n<0$. If
$\boldsymbol{p}_i=\boldsymbol{x}_i-R_i\boldsymbol{n}$ and
$\boldsymbol{p}_j=\boldsymbol{x}_j+R_j\boldsymbol{n}$ are the two undeformed
surface points, a symmetric contact point is

$$
\boldsymbol{x}_c=\frac{\boldsymbol{p}_i+\boldsymbol{p}_j}{2}.
$$

Define the contact arms
$\boldsymbol{r}_i=\boldsymbol{x}_c-\boldsymbol{x}_i$ and
$\boldsymbol{r}_j=\boldsymbol{x}_c-\boldsymbol{x}_j$. The relative contact
velocity of $i$ with respect to $j$ is

$$
\boldsymbol{v}_{rel}
=\left(\boldsymbol{v}_i+\boldsymbol{\omega}_i\times\boldsymbol{r}_i\right)
-\left(\boldsymbol{v}_j+\boldsymbol{\omega}_j\times\boldsymbol{r}_j\right).
$$

Its normal and tangential parts are

$$
v_n=\boldsymbol{v}_{rel}\cdot\boldsymbol{n},
\qquad
\boldsymbol{v}_t
=\boldsymbol{P}_t\boldsymbol{v}_{rel},
\qquad
\boldsymbol{P}_t
=\boldsymbol{1}-\boldsymbol{n}\otimes\boldsymbol{n}.
$$

With this convention, $v_n<0$ means that the bodies are approaching. The
effective mass and radius are

$$
m^*=\frac{m_im_j}{m_i+m_j},
\qquad
R^*=\frac{R_iR_j}{R_i+R_j}.
$$

For contact with an immovable wall, take $m_j\rightarrow\infty$ and
$R_j\rightarrow\infty$, giving $m^*=m_i$ and $R^*=R_i$.

### Linear spring--dashpot--Coulomb law

The unilateral normal force magnitude is

$$
f_n
=\max\left(0,\;k_n\delta_n-c_nv_n\right),
\qquad
\boldsymbol{F}_n=f_n\boldsymbol{n}.
$$

The normal elastic energy and viscous power are

$$
U_n=\frac{1}{2}k_n\delta_n^2,
\qquad
\mathcal P_n^{vis}=-c_nv_n^2\leq0.
$$

A convenient damping parametrization is

$$
c_n=2\zeta_n\sqrt{m^*k_n},
\qquad
c_t=2\zeta_t\sqrt{m^*k_t}.
$$

For a linear underdamped normal collision, the damping ratio corresponding to
a coefficient of restitution $e$ is

$$
\zeta_n
=\frac{-\ln e}{\sqrt{\pi^2+(\ln e)^2}}.
$$

Tangential contact requires a history variable
$\boldsymbol{\xi}_t$ in the current contact plane. When the normal changes,
the old history is first rotated into the new plane:

$$
\boldsymbol{\xi}_t^{tr}
=\boldsymbol{R}_c\boldsymbol{\xi}_t^n
+\Delta t\,\boldsymbol{v}_t,
\qquad
\boldsymbol{\xi}_t^{tr}\cdot\boldsymbol{n}=0.
$$

Here $\boldsymbol{R}_c$ is the smallest rotation that maps the old contact
normal to the new one. The elastic--viscous trial force is

$$
\boldsymbol{F}_t^{tr}
=-k_t\boldsymbol{\xi}_t^{tr}-c_t\boldsymbol{v}_t.
$$

The static/dynamic Coulomb return is

$$
\boldsymbol{F}_t=\boldsymbol{F}_t^{tr}
\quad\text{if}\quad
\|\boldsymbol{F}_t^{tr}\|\leq\mu_s f_n,
$$

and

$$
\boldsymbol{F}_t
=\mu_d f_n
\frac{\boldsymbol{F}_t^{tr}}{\|\boldsymbol{F}_t^{tr}\|}
\quad\text{otherwise}.
$$

After sliding, the elastic history is returned to the yield surface:

$$
\boldsymbol{\xi}_t^{n+1}
=-\frac{\boldsymbol{F}_t}{k_t}.
$$

### Hertz--Mindlin law

For isotropic materials, define the effective elastic moduli by

$$
\frac{1}{E^*}
=\frac{1-\nu_i^2}{E_i}
+\frac{1-\nu_j^2}{E_j},
$$

$$
\frac{1}{G^*}
=\frac{2-\nu_i}{G_i}
+\frac{2-\nu_j}{G_j},
\qquad
G_i=\frac{E_i}{2(1+\nu_i)}.
$$

The Hertz contact radius and incremental normal and tangential stiffnesses are

$$
a=\sqrt{R^*\delta_n},
\qquad
k_n=2E^*a,
\qquad
k_t=8G^*a.
$$

The elastic normal force and energy are

$$
f_n^{el}
=\frac{4}{3}E^*\sqrt{R^*}\,\delta_n^{3/2}
=\frac{2}{3}k_n\delta_n,
$$

$$
U_n
=\frac{8}{15}E^*\sqrt{R^*}\,\delta_n^{5/2}.
$$

Using

$$
\beta=\frac{-\ln e}{\sqrt{\pi^2+(\ln e)^2}},
$$

the damped Hertz normal force is

$$
f_n
=\max\left(
0,\;
\frac{2}{3}k_n\delta_n
-1.8257\,\beta\sqrt{m^*k_n}\,v_n
\right).
$$

The Mindlin tangential trial force is

$$
\boldsymbol{F}_t^{tr}
=-k_t\boldsymbol{\xi}_t^{tr}
-1.8257\,\beta\sqrt{m^*k_t}\,\boldsymbol{v}_t,
$$

followed by the same Coulomb return used by the linear law. Because $a$,
$k_n$, and $k_t$ depend on $\delta_n$, the Hertz--Mindlin contact becomes
progressively stiffer as compression increases.

### Explicit energy-conserving penalty law

Let $g$ be the signed gap, with $g<0$ in penetration, and define

$$
\delta=\langle-g\rangle_+.
$$

A power-law normal potential is

$$
U_n(\delta)
=\frac{k_n}{\theta}\delta^\theta,
\qquad
\theta\geq2.
$$

Its conservative contact force, written as the scalar force conjugate to the
gap, is

$$
Q_n
=\frac{\partial U_n}{\partial\delta}
=k_n\delta^{\theta-1}.
$$

The tangent stiffness is

$$
k_{tan}
=\frac{\partial Q_n}{\partial\delta}
=k_n(\theta-1)\delta^{\theta-2}.
$$

The restriction $\theta\geq2$ keeps this tangent finite when contact is first
activated. The choice $\theta=2$ gives the linear penalty law, while
$\theta=5/2$ gives the same penetration exponent as Hertz contact.

For a level-set gap $g(\boldsymbol{x})$, the force vector is

$$
\boldsymbol{F}_n
=Q_n\nabla g.
$$

The gap rate and contact power are

$$
\dot g
=\boldsymbol{v}_{rel}\cdot\nabla g,
\qquad
\mathcal P_n
=\boldsymbol{F}_n\cdot\boldsymbol{v}_{rel}
=Q_n\dot g.
$$

Since $\dot\delta=-\dot g$ during active contact,

$$
\frac{\mathrm dU_n}{\mathrm dt}
=-Q_n\dot g,
$$

and therefore

$$
\mathcal P_n+\frac{\mathrm dU_n}{\mathrm dt}=0.
$$

This work identity remains valid when $\|\nabla g\|\neq1$ because the same
gradient appears in both the force and the gap rate.

For an explicit finite step, let $\delta^n$ and $\delta^{n+1}$ be the stored
penetrations and define

$$
\Delta g
=g^{n+1}-g^n
=-\left(\delta^{n+1}-\delta^n\right).
$$

When $\delta^{n+1}\neq\delta^n$, use the discrete-gradient force

$$
\overline Q_n
=\frac{
U_n(\delta^{n+1})-U_n(\delta^n)
}{
\delta^{n+1}-\delta^n
}.
$$

For the power-law potential, this becomes

$$
\overline Q_n
=\frac{k_n}{\theta}
\frac{
(\delta^{n+1})^\theta-(\delta^n)^\theta
}{
\delta^{n+1}-\delta^n
}.
$$

In the zero-increment limit,

$$
\overline Q_n
=k_n(\delta^n)^{\theta-1}.
$$

The discrete contact work then satisfies, to arithmetic roundoff,

$$
\overline Q_n\Delta g
+U_n(\delta^{n+1})-U_n(\delta^n)
=0.
$$

For $\theta=2$, the discrete force has the particularly simple form

$$
\overline Q_n
=\frac{k_n}{2}
\left(\delta^{n+1}+\delta^n\right).
$$

The penetration history must be advanced with the same relative velocity used
to compute contact work:

$$
\delta^{n+1}
=\left\langle
\delta^n-\dot g\,\Delta t
\right\rangle_+.
$$

When a contact is newly activated, its stored penetration starts from zero;
an opening inactive pair must not inherit a stale negative geometric gap.

Normal viscous damping may be added through

$$
Q_n^{vis}
=-2\zeta_n\sqrt{m^*k_n}\,\dot g,
$$

with the unilateral total force

$$
Q_n^{tot}
=\max\left(0,\overline Q_n+Q_n^{vis}\right).
$$

Before unilateral clipping, its power is

$$
\mathcal P_n^{vis}
=Q_n^{vis}\dot g
=-2\zeta_n\sqrt{m^*k_n}\,\dot g^2
\leq0.
$$

Tangential elasticity uses

$$
U_t
=\frac{1}{2}k_t\|\boldsymbol{\xi}_t\|^2
$$

with the Coulomb return described above. For one contact step, the energy
statement is

$$
W_c+\Delta U_n+\Delta U_t
=W_{vis}+W_{slip}
\leq0,
$$

where equality holds when viscous damping and sliding are absent. The
energy-conserving name refers to this contact work--potential identity; the
total energy accuracy of a complete simulation still depends on the chosen
time integrator and external work discretization.

### Fluid-particle no-slip penalty law

For a liquid MPM sample in contact with a solid body, decompose the relative
velocity as

$$
v_n=\boldsymbol{v}_{rel}\cdot\boldsymbol{n},
\qquad
\boldsymbol{v}_t
=\left(
\boldsymbol{I}-\boldsymbol{n}\boldsymbol{n}^T
\right)\boldsymbol{v}_{rel}.
$$

For active penetration $g<0$, the normal force is

$$
\boldsymbol{F}_n
=\left[
-k_ng
-2\zeta_n\sqrt{m^*k_n}\,v_n
\right]\boldsymbol{n}.
$$

The prescribed no-slip fraction $0\leq\alpha_s\leq1$ removes a fraction of
the tangential relative momentum in one time step:

$$
\boldsymbol{F}_t
=-\frac{m^*\alpha_s}{\Delta t}\boldsymbol{v}_t.
$$

Thus $\alpha_s=0$ is tangential slip and $\alpha_s=1$ is the full one-step
no-slip impulse. This model has no tangential spring history or Coulomb
return; its tangential branch is an impulse regularized by the time step.

### Explicit finite-clearance logarithmic Barrier law

Let $g=-\delta$ be clearance relative to the contact thickness, $c>0$ the
admissible clearance shift, and

$$
\eta=g+c,
\qquad
D=2c.
$$

For $0<\eta<D$, define the normal contact potential

$$
U_b(\eta)
=-A_c\kappa(\eta-D)^2
\ln\left(\frac{\eta}{D}\right).
$$

It vanishes for $\eta\geq D$ and diverges as $\eta\rightarrow0^+$. The
repulsive force magnitude conjugate to penetration is

$$
f_n^{el}
=A_c\kappa(\eta-D)
\left[
2\ln\left(\frac{\eta}{D}\right)
-\frac{D}{\eta}+1
\right].
$$

Hence the undamped normal law is conservative:

$$
\mathcal P_n+\frac{\mathrm dU_b}{\mathrm dt}=0.
$$

For a displacement-regularized tangential law, let $r_f>0$ be the tangential
stiffness ratio and define

$$
\xi_c=\frac{\mu f_n}{r_f\kappa A_c},
\qquad
s=\|\boldsymbol{\xi}_t\|.
$$

The smooth Coulomb factor is

$$
q_f(s)=\frac{s(2\xi_c-s)}{\xi_c^2}
\quad\text{for}\quad 0\leq s<\xi_c,
$$

and $q_f=1$ for $s\geq\xi_c$. The tangential force is

$$
\boldsymbol{F}_t
=-\mu f_nq_f(s)
\frac{\boldsymbol{\xi}_t}{s},
$$

with its continuous zero limit at $s=0$. Normal damping and sliding friction
dissipate energy; the logarithmic normal term itself stores and returns work.

### Rolling and twisting resistance

Split the relative angular velocity into twisting and rolling parts:

$$
\boldsymbol{\omega}_{rel}
=\boldsymbol{\omega}_i-\boldsymbol{\omega}_j,
$$

$$
\boldsymbol{\omega}_{tw}
=\left(\boldsymbol{\omega}_{rel}\cdot\boldsymbol{n}\right)\boldsymbol{n},
\qquad
\boldsymbol{\omega}_r
=\boldsymbol{P}_t\boldsymbol{\omega}_{rel}.
$$

History-dependent rolling and twisting moments may be written as

$$
\boldsymbol{M}_r^{tr}
=-k_r\boldsymbol{\theta}_r-c_r\boldsymbol{\omega}_r,
\qquad
\boldsymbol{M}_{tw}^{tr}
=-k_{tw}\boldsymbol{\theta}_{tw}-c_{tw}\boldsymbol{\omega}_{tw}.
$$

Their yield limits are

$$
\|\boldsymbol{M}_r\|\leq\mu_rR^*f_n,
\qquad
\|\boldsymbol{M}_{tw}\|\leq\mu_{tw}R^*f_n.
$$

A history-free constant-torque rolling law is the limiting case

$$
\boldsymbol{M}_r
=-\mu_rR^*f_n
\frac{\boldsymbol{\omega}_r}{\|\boldsymbol{\omega}_r\|}
$$

whenever $\|\boldsymbol{\omega}_r\|>0$.

## Incremental Potential Contact

For a point--triangle, edge--edge, or point--wall stencil, let $d>0$ be its
Euclidean separation, $s=d^2$, and
$\widehat{s}=\widehat{d}^{\,2}$. The clamped logarithmic IPC barrier is

$$
b(s)=0
\quad\text{for}\quad
s\geq\widehat{s},
$$

$$
b(s)
=-\kappa(s-\widehat{s})^2
\ln\left(\frac{s}{\widehat{s}}\right)
\quad\text{for}\quad
0<s<\widehat{s},
$$

and $b(s)=+\infty$ for $s\leq0$. In the active region, define
$\Delta=s-\widehat{s}$. Its derivatives are

$$
b'(s)
=-\kappa
\left[
2\Delta\ln\left(\frac{s}{\widehat{s}}\right)
+\frac{\Delta^2}{s}
\right],
$$

$$
b''(s)
=-\kappa
\left[
2\ln\left(\frac{s}{\widehat{s}}\right)
+\frac{4\Delta}{s}
-\frac{\Delta^2}{s^2}
\right].
$$

The total barrier energy is a weighted sum

$$
E_c
=\sum_{c\in\mathcal A}A_c\,b(d_c^2),
$$

where $A_c$ is a contact measure and $\mathcal A$ contains stencils with
$d_c<\widehat d$. Its gradient and Hessian obey

$$
\nabla b
=b'(s)\nabla s,
$$

$$
\nabla^2 b
=b''(s)\nabla s\nabla s^T
+b'(s)\nabla^2s.
$$

Continuous collision detection limits the line-search step
$\alpha\in(0,1]$ so that

$$
d_c\left(
\boldsymbol{y}+\alpha\Delta\boldsymbol{y}
\right)>0
$$

for every candidate stencil. The logarithmic singularity and CCD together
keep accepted states strictly non-intersecting.

### Oriented signed-gap form

Plane and level-set contact supplies an oriented gap $g$ rather than an
unsigned closest-point distance. Define the composed barrier

$$
\bar b(g)=b(g^2),
\qquad g>0,
$$

with $\bar b(g)=+\infty$ for $g\leq0$. Its scalar derivatives are

$$
\bar b'(g)=2g\,b'(g^2),
$$

$$
\bar b''(g)=2b'(g^2)+4g^2b''(g^2).
$$

For generalized coordinates $\boldsymbol{q}$ and an oriented gap
$g(\boldsymbol{q})$, the exact derivatives of
$E_c=A_c\bar b(g)$ are

$$
\nabla E_c
=A_c\bar b'(g)\nabla g,
$$

$$
\nabla^2E_c
=A_c\left[
\bar b''(g)\nabla g\nabla g^T
+\bar b'(g)\nabla^2g
\right].
$$

This form preserves the sign needed to reject penetration while remaining
the same ordinary clamped-log IPC barrier under the change of variable
$s=g^2$.

### Regularized IPC friction

Let $\lambda_n\geq0$ be the contact normal force magnitude and

$$
\boldsymbol{v}_t
=\left(\boldsymbol{1}
-\boldsymbol{n}\otimes\boldsymbol{n}\right)
\boldsymbol{v}_{rel},
\qquad
v=\|\boldsymbol{v}_t\|.
$$

The $C^1$ transition from static to sliding friction is

$$
f_1(v)
=\frac{v(2\varepsilon_v-v)}{\varepsilon_v^2}
\quad\text{for}\quad
0\leq v<\varepsilon_v,
$$

$$
f_1(v)=1
\quad\text{for}\quad
v\geq\varepsilon_v.
$$

The regularized Coulomb force is

$$
\boldsymbol{F}_t
=-\mu\lambda_n f_1(v)
\frac{\boldsymbol{v}_t}{v}
$$

for $v>0$, with the continuous zero limit at $v=0$. A corresponding
dissipation potential, up to an irrelevant additive constant, is

$$
\psi_{\varepsilon}(v)
=\Delta t
\left(
\frac{v^2}{\varepsilon_v}
-\frac{v^3}{3\varepsilon_v^2}
\right)
\quad\text{for}\quad
0\leq v<\varepsilon_v,
$$

$$
\psi_{\varepsilon}(v)
=\Delta t
\left(
v-\frac{\varepsilon_v}{3}
\right)
\quad\text{for}\quad
v\geq\varepsilon_v.
$$

Thus

$$
E_f=\sum_c\mu_c\lambda_{n,c}\psi_{\varepsilon}(v_c).
$$

In a lagged solve, $\lambda_n$, $\boldsymbol{n}$, and the tangent basis are
frozen while minimizing the current incremental potential, then refreshed in
an outer iteration. A fully implicit formulation differentiates these contact
quantities together with the friction law.

### Fully implicit Stribeck friction

For tangential velocity $\boldsymbol{v}_t$ and speed
$v=\|\boldsymbol{v}_t\|$, the compact $C^1$ radial profile is

$$
p(v)=\frac{2-v/\varepsilon_v}{\varepsilon_v}
$$

when $v<\varepsilon_v$, and $p(v)=1/v$ otherwise. A stabilized alternative
is $p(v)=1/(v+0.1\varepsilon_v)$.

With Stribeck speed $v_s$ and $x=v/v_s$, define

$$
\chi(v)=(2x+1)(x-1)^2
$$

for $0\leq v\leq v_s$, with $\chi=0$ above $v_s$. The effective friction
coefficient and radial resistance are

$$
\mu_{eff}(v)
=\mu_d+(\mu_s-\mu_d)\chi(v),
$$

$$
a(v,\lambda_n)
=\lambda_n\mu_{eff}(v)p(v)+\mu_v,
\qquad
\boldsymbol{\eta}=a\boldsymbol{v}_t.
$$

Their exact constitutive differential is

$$
\mathrm d\boldsymbol{\eta}
=\left[
a\boldsymbol{I}
+\frac{a_v}{v}\boldsymbol{v}_t\boldsymbol{v}_t^T
\right]\mathrm d\boldsymbol{v}_t
+\mu_{eff}p\boldsymbol{v}_t\,\mathrm d\lambda_n,
$$

$$
a_v
=\lambda_n\left[
(\mu_s-\mu_d)\chi'(v)p(v)
+\mu_{eff}(v)p'(v)
\right].
$$

At $v=0$, the continuous radial limit replaces the written $a_v/v$ quotient.
Geometry-specific consumers additionally differentiate their normal, tangent
projector, closest coordinates, and endpoint interpolation.

### Finite-clearance offset barrier

For a prescribed minimum separation $d_{min}\geq0$, replace the squared
distance and activation value by

$$
s=d^2-d_{min}^2,
\qquad
\widehat{s}=(d_{min}+\widehat d)^2-d_{min}^2.
$$

The same clamped-log barrier and its derivatives then act on this shifted
$s$. A physical-unit normalization may multiply the barrier by
$\widehat d/\widehat{s}^2$; this changes units and stiffness scale but not
the chain rule used by any contact stencil.

## Solver references

- [DEM contact geometry and rigid-particle balance](../../dem/README.md#2-contact-geometry-and-force-transfer)
- [FEM contact discretization](../../fem/README.md#contact)
- [FEDEM soft-particle coupling](../../fedem/README.md#soft-particle-contact)
- [FEM--MPM contact pullback](../../fempm/README.md#3-exact-mpm-grid-pullback)
- [IGA--MPM point--NURBS coupling](../../igampm/README.md#3-exact-direct-point--nurbs-coupling-derivatives)
- [MPM--DEM geometry and coupled contact](../../mpdem/README.md#1-explicit-mpm--dem-contact-kinematics)
