# Constitutive Theory for Hyperelasticity, Cloth, Plasticity, and Granular Materials

`src/physics_model/consititutive_model` owns constitutive equations shared by
GeoTaichi solvers. The directory name is intentionally kept with its historical
spelling for import compatibility. Solver READMEs document discretization and
time integration, then link here instead of duplicating material laws.

## Model families

- Finite-strain hyperelasticity: St. Venant--Kirchhoff, Neo-Hookean, Hencky,
  Mooney--Rivlin, Gent, hydrogel, and surface-cloth responses.
- Multiplicative finite-strain plasticity: associated or nonassociated Drucker--Prager,
  von Mises with optional isotropic hardening.
- Infinitesimal and rate-form solids: linear elasticity, J2, Mohr--Coulomb,
  Drucker--Prager, Modified Cam--Clay, NorSand, SANISAND, granular and
  softening models.
- Rate-dependent fluids: Newtonian and Bingham responses.

## Hyperelastic solid laws

For Young's modulus $`E`$ and Poisson ratio $`\nu`$, the three-dimensional Lame
parameters are

```math
\mu=\frac{E}{2(1+\nu)},
\qquad
\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)}.
```

### St. Venant--Kirchhoff elasticity

Let $`\boldsymbol{C}=\boldsymbol{F}^T\boldsymbol{F}`$ and define the
Green--Lagrange strain

```math
\boldsymbol{E}_{GL}=\frac{1}{2}(\boldsymbol{C}-\boldsymbol{1}).
```

The energy density, second Piola stress, and first Piola stress are

```math
\Psi_{SVK}
=\frac{\lambda}{2}\mathrm{tr}(\boldsymbol{E}_{GL})^2
+\mu\boldsymbol{E}_{GL}:\boldsymbol{E}_{GL}.
```

```math
\boldsymbol{S}
=\lambda\mathrm{tr}(\boldsymbol{E}_{GL})\boldsymbol{1}
+2\mu\boldsymbol{E}_{GL},
\qquad
\boldsymbol{P}=\boldsymbol{F}\boldsymbol{S}.
```

Its material tangent is

```math
\mathbb{A}_{iJkL}
=\delta_{ik}S_{LJ}
+\lambda F_{iJ}F_{kL}
+\mu F_{iL}F_{kJ}
+\mu\delta_{JL}(\boldsymbol{F}\boldsymbol{F}^T)_{ik}.
```

### Compressible Neo-Hookean elasticity

For $`J=\det\boldsymbol{F}>0`$ and spatial dimension $`d`$,

```math
\Psi_{NH}
=\frac{\mu}{2}
\left(\boldsymbol{F}:\boldsymbol{F}-d\right)
-\mu\ln J
+\frac{\lambda}{2}(\ln J)^2.
```

The first Piola stress and elasticity tensor are

```math
\boldsymbol{P}
=\mu\left(\boldsymbol{F}-\boldsymbol{F}^{-T}\right)
+\lambda\ln J\,\boldsymbol{F}^{-T}.
```

```math
\mathbb{A}_{iJkL}
=\mu\delta_{ik}\delta_{JL}
+(\mu-\lambda\ln J)F^{-T}_{iL}F^{-T}_{kJ}
+\lambda F^{-T}_{iJ}F^{-T}_{kL}.
```

The associated Cauchy stress is

```math
\boldsymbol{\sigma}
=\frac{1}{J}\boldsymbol{P}\boldsymbol{F}^T
=\frac{\mu}{J}
\left(\boldsymbol{F}\boldsymbol{F}^T-\boldsymbol{1}\right)
+\frac{\lambda\ln J}{J}\boldsymbol{1}.
```


## Surface-cloth constitutive laws

### Surface kinematics and membrane energy

For a flat reference cloth patch, let
$`\boldsymbol{D}_m\in\mathbb{R}^{2\times2}`$ contain two independent reference
edges and let $`\boldsymbol{D}_s\in\mathbb{R}^{3\times2}`$ contain their current
counterparts. The surface deformation gradient, metric, principal stretches,
and area ratio are

```math
\boldsymbol{F}_s=\boldsymbol{D}_s\boldsymbol{D}_m^{-1},
\qquad
\boldsymbol{C}_s=\boldsymbol{F}_s^T\boldsymbol{F}_s,
```

```math
\sigma_a=\sqrt{\lambda_a(\boldsymbol{C}_s)},
\qquad
J_s=\sqrt{\det\boldsymbol{C}_s}=\sigma_1\sigma_2>0.
```

If $`A_0`$ and $`h`$ are the reference area and thickness, respectively, a
constant-strain patch stores

```math
U_m=A_0h\,\Psi_s(\boldsymbol{F}_s).
```

### Tension--compression asymmetric ARAP cloth

Let $`k_s`$ and $`k_c`$ be the tensile and compressive stiffnesses. For each
principal stretch, take $`k_a=k_c`$ when $`\sigma_a\leq1`$ and $`k_a=k_s`$ when
$`\sigma_a>1`$. The surface energy is

```math
\Psi_{ARAP}=\sum_{a=1}^{2}k_a(\sigma_a-1)^2.
```

If $`\boldsymbol{C}_s=\boldsymbol{Q}\mathrm{diag}(\sigma_1^2,\sigma_2^2)\boldsymbol{Q}^T`$,
its first Piola surface stress is

```math
\boldsymbol{P}_s
=\boldsymbol{F}_s\boldsymbol{Q}
\mathrm{diag}\!\left(
\frac{2k_1(\sigma_1-1)}{\sigma_1},
\frac{2k_2(\sigma_2-1)}{\sigma_2}
\right)\boldsymbol{Q}^T.
```

Thus the two in-plane principal modes may resist compression and extension
with different moduli without introducing an out-of-plane constitutive
stress.

### Plane-stress Neo-Hookean cloth

The surface Lame parameters are

```math
\mu_s=\frac{E}{2(1+\nu)},
\qquad
\lambda_s=\frac{E\nu}{1-\nu^2}.
```

The compressible surface energy and first Piola stress are

```math
\Psi_{NH}^{s}
=\frac{\mu_s}{2}
\left(\mathrm{tr}\boldsymbol{C}_s-2-2\ln J_s\right)
+\frac{\lambda_s}{2}(\ln J_s)^2,
```

```math
\boldsymbol{P}_s
=\mu_s\boldsymbol{F}_s
+(-\mu_s+\lambda_s\ln J_s)
\boldsymbol{F}_s\boldsymbol{C}_s^{-1}.
```


## Finite-strain multiplicative plasticity

The constitutive update receives a three-dimensional total deformation
gradient. Plane-strain and axisymmetric solvers are responsible for supplying
their three-dimensional embedding before evaluating the material law.

Use the multiplicative decomposition

```math
\boldsymbol{F}=\boldsymbol{F}^e\boldsymbol{F}^{pl},
\qquad
\boldsymbol{F}_{tr}^e
=\boldsymbol{F}(\boldsymbol{F}^{pl}_n)^{-1}.
```

With the singular-value decomposition

```math
\boldsymbol{F}_{tr}^e
=\boldsymbol{U}\,\mathrm{diag}(\sigma_1,\sigma_2,\sigma_3)
\boldsymbol{V}^T,
```

the trial principal Hencky strains are

```math
\epsilon_a^{tr}=\ln\sigma_a,
\qquad
\theta^{tr}=\sum_a\epsilon_a^{tr},
\qquad
\boldsymbol{e}^{tr}
=\boldsymbol{\epsilon}^{tr}-\frac{\theta^{tr}}{3}\boldsymbol{1},
\qquad
\rho^{tr}=\|\boldsymbol{e}^{tr}\|.
```

For the quadratic Hencky elastic response,

```math
\psi_e
=\mu\|\boldsymbol{e}^e\|^2
+\frac{K}{2}(\theta^e)^2,
```

```math
\boldsymbol{\tau}
=2\mu\boldsymbol{e}^e
+K\theta^e\boldsymbol{1}.
```

### Associated Drucker--Prager and von Mises return

A common principal-space yield function is

```math
f
=\|\mathrm{dev}\boldsymbol{\tau}\|
+\alpha\,\mathrm{tr}\boldsymbol{\tau}
-\tau_y(\bar\epsilon^{pl})\leq0.
```

When the trial state is outside the cone and $`\rho^{tr}>0`$, define

```math
\boldsymbol{m}
=\frac{\boldsymbol{e}^{tr}}{\rho^{tr}}.
```

The associated return multiplier is

```math
\Delta\gamma
=\frac{
2\mu\rho^{tr}+3\alpha K\theta^{tr}-\tau_y
}{
2\mu+9\alpha^2K+H_p
},
```

with the numerator clipped at zero. The returned logarithmic strain is

```math
\boldsymbol{\epsilon}^e
=\boldsymbol{\epsilon}^{tr}
-\Delta\gamma\boldsymbol{m}
-\alpha\Delta\gamma\boldsymbol{1}.
```

For von Mises plasticity,

```math
\alpha=0,
\qquad
\tau_y
=\sqrt{\frac23}
(\sigma_y+H\bar\epsilon^{pl}),
\qquad
H_p=\frac23H,
```

```math
\Delta\bar\epsilon^{pl}
=\sqrt{\frac23}\Delta\gamma.
```

For Drucker--Prager written as

```math
\sqrt{J_2}+q_\phi\frac{I_1}{3}-k_\phi\leq0,
```

the equivalent coefficients are

```math
\alpha=\frac{\sqrt2}{3}q_\phi,
\qquad
\tau_y=\sqrt2k_\phi,
\qquad
\Delta\epsilon_v^{pl}=3\alpha\Delta\gamma.
```

For the circumscribed cone,

```math
q_\phi
=\frac{6\sin\phi}{\sqrt3(3-\sin\phi)},
\qquad
k_\phi
=\frac{6c\cos\phi}{\sqrt3(3-\sin\phi)}.
```

Associated flow requires the dilation angle to equal the friction angle. A
cone-apex return replaces the deviatoric formula when the radial projection
would cross the hydrostatic apex.

The local incremental density for the smooth cone branch is

```math
\psi_{inc}
=\mu\|\boldsymbol{e}^e\|^2
+\frac{K}{2}(\theta^e)^2
+\tau_y\Delta\gamma
+\frac12H_p(\Delta\gamma)^2.
```

### Nonassociated Drucker--Prager and symmetric inner solves

`DilationAngle` may differ from `FrictionAngle`; when omitted it defaults to
the friction angle. Let $`\beta=\sqrt{2}q_\psi/3`$, using the same `dpType`
mapping as $`\alpha`$. The perfect-plastic smooth-cone return uses

```math
\Delta\gamma=\frac{2\mu\rho^{tr}+3\alpha K\theta^{tr}-\tau_y}
{2\mu+9\alpha\beta K},\qquad
\boldsymbol{\epsilon}^e=\boldsymbol{\epsilon}^{tr}
-\Delta\gamma\boldsymbol{m}-\beta\Delta\gamma\boldsymbol{1}.
```

Its volumetric plastic increment is $`3\beta\Delta\gamma`$. At the tensile
apex the returned trace is $`\theta_a=\tau_y/(3\alpha K)`$. For zero dilation,
the apex acts as a tensile cap: $`\Delta\gamma=\rho^{tr}`$ and
$`\Delta\epsilon_v^{pl}=\theta^{tr}-\theta_a`$. No dilation evolution is used.

The physical tangent generally is nonsymmetric. Standalone lagged MPM--IPC,
FEM--MPM, IGA--MPM, and Direct MPM--ABD use it
directly with BiCGSTAB, residual Armijo, and contact/material CCD. There is no
material-consistency outer loop on these coupled nonassociated routes.
The physical trial volume and total stress are

```math
J_p(\boldsymbol F)=\frac{\det\boldsymbol F}{\det\boldsymbol F_e^{returned}},
\qquad
\boldsymbol P=J_p(\boldsymbol F)\boldsymbol P_e(\boldsymbol F_e^{tr})
(\boldsymbol F_{p,n})^{-T}.
```

The complete total Jacobian includes
$`\boldsymbol P\otimes\partial_{\boldsymbol F}\log J_p`$ in addition to the
transformed local return tangent. This keeps trial stress equal to stress
evaluated after committing the same deformation. Associated DP retains its
PCG/frozen-volume solve and existing volume-consistency iteration.

Other Direct-MPM consumers retain their existing frozen-potential contract.
For those consumers, PCG and energy Armijo instead
solve a symmetric associated inner potential with the isotropic strain shift
$`\delta=\alpha\Delta\gamma-\Delta\epsilon_v^{pl}/3`$ frozen. For zero-dilation
apex states use $`\delta=\alpha\rho^{tr}`$ to stay inside the flat apex branch.
An outer material iteration refreshes this shift and the predicted plastic
volume without committing history. Convergence checks the maximum of

```math
\left|\log\frac{J_p^{candidate}}{\widehat J_p}\right|,
\qquad \frac{\|\boldsymbol{\tau}_{physical}
-\boldsymbol{\tau}_{inner}\|}{2\mu},
```

against the existing dimensionless material tolerance. Checking stress directly
avoids rejecting equivalent shifts inside the capped branch. The inner Hessian
remains symmetric and may be PSD projected; this does not make the physical
nonassociated tangent symmetric.

Predictor updates use particle-local Aitken relaxation. With
$`\boldsymbol{r}_k=(\log(J_p^{candidate}/\widehat J_{p,k}),
\delta^{candidate}-\delta_k)^T`$ and
$`\Delta\boldsymbol{r}_k=\boldsymbol{r}_k-\boldsymbol{r}_{k-1}`$,
the secant update is

```math
\omega_k=\operatorname{clip}_{[10^{-3},1]}
\left(-\omega_{k-1}
\frac{\boldsymbol{r}_{k-1}^T\Delta\boldsymbol{r}_k}
{\|\Delta\boldsymbol{r}_k\|^2}\right),
\qquad
\delta_{k+1}=\delta_k+\omega_k r_{k,2},
\qquad
\widehat J_{p,k+1}=\widehat J_{p,k}\exp(\omega_k r_{k,1}).
```

Each step or retry starts with $`\omega=1`$ and zero previous residual.
Degenerate secant differences retain the current relaxation. Convergence uses
the unrelaxed stress and volume errors above, so damping cannot conceal a
physical mismatch. This numerical iteration leaves the DP flow law and its
constant dilation angle unchanged.

### State-dependent finite-strain Drucker--Prager

`finite_strain/StateDependentDruckerPrager.py` defines the independent
`StateDependentDruckerPragerModel`. Selecting `StateDependentDruckerPrager`
reuses the finite-strain DP return while replacing its constant friction and
dilation with the law in `infinitesimal_strain/StateDependentMohrCoulomb.py`.
Ordinary `DruckerPrager` retains its defaults and constant angles.

With compression-positive **previous Cauchy pressure**, total volume ratio
$`J=\det F`$, and accepted void ratio $`e_n`$, the step uses

```math
p_n=\max\left(-K\log J_{e,n}/J_{e,n},1000\,\mathrm{Pa}\right),\qquad
e_c=e_{\mathrm{Tao}}-\lambda_c(p_n/101000\,\mathrm{Pa})^\xi,
```

```math
e_{tr}=\operatorname{clip}_{[0.1,1.5]}\left((1+e_n)J_{tr}/J_n-1\right),\qquad
S=e_{tr}-e_c,
```

```math
\phi=\max\left(\phi_c,\arctan(\tan\phi_c\exp(-n_f S))\right),\qquad
\psi=\max\left(0,\arctan(-n_d S)\right).
```

The volume formula is the finite-strain counterpart of SDMC's
$`\Delta e=(1+e)\Delta\epsilon_v`$. Axisymmetric volume includes hoop stretch;
Cartesian 2D remains plane strain. Pressure is frozen throughout Newton and
lagged friction; trial void ratio and angles change with the trial deformation.
Elastic moduli and cohesion remain constant. `dpType` defaults to
`MiddleCircumscribed` (the $`3+\sin\phi`$ mapping used by SDMC); all three DP
cone mappings remain available. This transfers SDMC's state law, retaining the
finite-strain DP tensile apex rather than SDMC's independent tensile cutoff.

On the smooth cone, let $`t=\operatorname{tr}\epsilon^{tr}`$,
$`m=\operatorname{dev}\epsilon^{tr}/\rho`$, and
$`D=2\mu+9K\alpha\beta`$. Including the evolving coefficients gives

```math
\partial_j\gamma=
\frac{2\mu m_j+3K(\alpha+t\alpha')-\tau_y'-\gamma D'}{D},\qquad
D'=9K(\alpha'\beta+\alpha\beta'),
```

```math
\partial_j\epsilon_i^e=\delta_{ij}-(m_i+\beta)\partial_j\gamma
-\gamma\beta'-\frac{\gamma}{\rho}
(\delta_{ij}-1/3-m_i m_j).
```

Primes denote derivatives with respect to trial trace. At the apex,
$`\partial_j\epsilon_i^e=\frac13\partial_t[\tau_y/(3K\alpha)]`$;
its hydrostatic tangent therefore need not vanish. These derivatives and the
plastic-volume derivative enter the analytic total PK1 Jacobian. The coupled
IPC solver uses residual Newton/BiCGSTAB without symmetrization or an energy
potential. Lagged friction retains its own outer iteration.

The 14-entry history adds `void_ratio`, `committed_jacobian`, and
`state_pressure` to the original 11 DP entries. Trials do not commit it;
accepted steps commit void ratio and total volume, and failed steps restore
all entries. Supported owners are Direct implicit ULMPM within FEM--MPM,
IGA--MPM, and 3D MPM--ABD IPC. Standalone Direct stepping, Direct TLMPM,
Native MPM, LSMPM soft materials, and parameter/history adjoints are not
implemented for this new model. VTU output includes `void_ratio`.

### Stress, tangent, and state commit

Let $`\widehat J_p`$ be the frozen predictor of the accepted plastic volume,
refreshed by the outer material iteration. A particle or quadrature-point
incremental density expressed through the total deformation gradient is

```math
W(\boldsymbol{F};\boldsymbol{h}_n)
=\widehat J_p\psi_{inc}\left(
\boldsymbol{F}(\boldsymbol{F}^{pl}_n)^{-1};
\boldsymbol{h}_n
\right).
```

The following derivatives define the frozen inner first Piola stress and
tangent. The physical nonassociated tangent is evaluated separately.

```math
\boldsymbol{P}
=\frac{\partial W}{\partial\boldsymbol{F}},
\qquad
\mathbb{A}^{alg}
=\frac{\partial\boldsymbol{P}}{\partial\boldsymbol{F}}.
```

The returned elastic state is committed as

```math
\boldsymbol{F}_{n+1}^{e}
=\boldsymbol{U}\,\mathrm{diag}
(\exp\epsilon_1^e,\exp\epsilon_2^e,\exp\epsilon_3^e)
\boldsymbol{V}^T,
```

```math
(\boldsymbol{F}_{n+1}^{pl})^{-1}
=\boldsymbol{F}_{n+1}^{-1}\boldsymbol{F}_{n+1}^{e}.
```

Equivalent plastic strain, volumetric plastic strain, and any hardening
variables such as $`p_c`$ are advanced with the plastic deformation gradient.
A nonlinear solver evaluates trial returns with accepted history frozen and
commits these quantities only after the global step is accepted.

## Incremental infinitesimal elastoplasticity

With tension-positive Cauchy stress, define compression-positive mean stress
$`p`$, deviatoric stress, and equivalent shear stress by

```math
p=-\frac{1}{3}\mathrm{tr}\boldsymbol{\sigma},
\qquad
\boldsymbol{s}=\boldsymbol{\sigma}+p\boldsymbol{1},
\qquad
J_2=\frac{1}{2}\boldsymbol{s}:\boldsymbol{s},
\qquad
q=\sqrt{3J_2}.
```

For yield function $`f(\boldsymbol{\sigma},\boldsymbol{\alpha})`$ and plastic
potential $`g`$, rate-independent plasticity obeys

```math
\Delta\boldsymbol{\varepsilon}^{p}
=\Delta\lambda\frac{\partial g}{\partial\boldsymbol{\sigma}},
\qquad
f\leq0,
\qquad
\Delta\lambda\geq0,
\qquad
\Delta\lambda f=0.
```

Writing $`\boldsymbol{n}=\partial f/\partial\boldsymbol{\sigma}`$ and
$`\boldsymbol{m}=\partial g/\partial\boldsymbol{\sigma}`$, a local consistency
linearization has the generic plastic-multiplier denominator

```math
\boldsymbol{n}:\mathbb{C}:\boldsymbol{m}-H,
```

where $`H=(\partial f/\partial\boldsymbol{\alpha})\cdot(\mathrm d\boldsymbol{\alpha}/\mathrm d\lambda)`$
collects the evolution of all hardening variables. With this sign convention,
ordinary hardening gives $`H<0`$.

An elastic-predictor/plastic-corrector step first forms

```math
\boldsymbol{\sigma}_{tr}
=\boldsymbol{\sigma}_{n}^{rot}
+\mathbb{C}:\Delta\boldsymbol{\varepsilon}.
```

If $`f(\boldsymbol{\sigma}_{tr},\boldsymbol{\alpha}_n)\leq0`$, the trial state
is accepted. Otherwise the corrected state satisfies

```math
\boldsymbol{\sigma}_{n+1}
=\boldsymbol{\sigma}_{tr}
-\Delta\lambda\mathbb{C}:\boldsymbol{m},
```

```math
\boldsymbol{\alpha}_{n+1}
=\boldsymbol{\alpha}_n
+\Delta\lambda\boldsymbol{h},
\qquad
f(\boldsymbol{\sigma}_{n+1},\boldsymbol{\alpha}_{n+1})=0.
```

Here $`\boldsymbol{\sigma}_{n}^{rot}`$ is the objective rotation of the old
stress. Nonlinear surfaces solve the final consistency equation iteratively;
substepping limits constitutive error when a full strain increment is too
large for one local solve.

### J2 elastic-perfectly-plastic model

The von Mises surface and associated flow rule are

```math
f=q-\sigma_y(\bar\varepsilon_d^p),
\qquad
g=f,
\qquad
\Delta\boldsymbol{\varepsilon}^{p}
=\Delta\lambda\frac{3\boldsymbol{s}}{2q}.
```

Perfect plasticity uses constant $`\sigma_y`$. Peak-to-residual softening makes
$`\sigma_y`$ a decreasing function of accumulated equivalent plastic
deviatoric strain $`\bar\varepsilon_d^p`$.

### Mohr--Coulomb model

Let $`p_1\geq p_2\geq p_3`$ be compression-positive principal stresses. The
shear surface and tensile cutoff are

```math
f_s=(p_1-p_3)-(p_1+p_3)\sin\phi-2c\cos\phi,
```

```math
f_t=-p_3-\sigma_t.
```

The admissible domain satisfies $`f_s\leq0`$ and $`f_t\leq0`$. Non-associated
flow replaces the friction angle $`\phi`$ in the pressure-dependent part of
the plastic potential by the dilation angle $`\psi`$; $`\psi=\phi`$ recovers
associated flow. Cohesion, friction, dilation, and tensile strength may
evolve from peak to residual values with accumulated plastic shear strain.

### Drucker--Prager model

The conical shear surface, tensile cap, and non-associated potential can be
written

```math
f_s=\sqrt{J_2}-\alpha_\phi p-k_\phi,
\qquad
f_t=-p-\sigma_t,
```

```math
g_s=\sqrt{J_2}-\alpha_\psi p.
```

For the circumscribed match to Mohr--Coulomb,

```math
\alpha_\phi=\frac{6\sin\phi}{\sqrt{3}(3-\sin\phi)},
\qquad
k_\phi=\frac{6c\cos\phi}{\sqrt{3}(3-\sin\phi)}.
```

For the inscribed match,

```math
\alpha_\phi=\frac{3\tan\phi}{\sqrt{9+12\tan^2\phi}},
\qquad
k_\phi=\frac{3c}{\sqrt{9+12\tan^2\phi}}.
```

The middle match replaces $`3-\sin\phi`$ in the circumscribed coefficients by
$`3+\sin\phi`$. The dilation coefficient $`\alpha_\psi`$ follows the same
chosen matching rule with $`\phi`$ replaced by $`\psi`$.

### State-dependent Mohr--Coulomb model

For void ratio $`e`$, reference pressure $`p_{ref}`$, and critical-state line

```math
e_c=e_{\tau}-\lambda_c
\left(\frac{p}{p_{ref}}\right)^{\xi},
\qquad
\Psi=e-e_c,
```

the state-dependent friction and dilation laws are

```math
\tan\phi=\tan\phi_c\exp(-n_f\Psi),
\qquad
\phi\geq\phi_c,
```

```math
\tan\psi=\max(0,-n_d\Psi).
```

The current $`\phi`$ and $`\psi`$ enter a pressure-sensitive shear surface and
non-associated flow potential. With tensile-positive volumetric strain, the
void ratio evolves as

```math
e_{n+1}=e_n+(1+e_n)\Delta\varepsilon_v.
```

### Modified Cam--Clay model

For critical-state slope $`M_\theta`$, preconsolidation pressure $`p_c`$, bonding
pressures $`p_{cd}`$ and $`p_{cc}`$, and subloading ratio $`R\in(0,1]`$, the yield
surface is

```math
f=\frac{q^2}{M_\theta^2}
+(p+p_{cc})\left[p-R(p_c+p_{cd}+p_{cc})\right].
```

The unbonded normal surface follows by setting $`p_{cd}=p_{cc}=0`$ and $`R=1`$:

```math
f=\frac{q^2}{M^2}+p(p-p_c).
```

Isotropic hardening follows

```math
\mathrm dp_c
=\frac{1+e}{\lambda-\kappa}(p_c+p_{cd})
\,\mathrm d\varepsilon_v^p.
```

A bonding variable $`\chi`$ may add

```math
p_{cd}=a(\chi s_h)^b,
\qquad
p_{cc}=c_b(\chi s_h)^d,
\qquad
\mathrm d\chi=-m_{deg}\chi\,\mathrm d\varepsilon_d^p.
```

The subloading surface contracts the normal surface through $`R`$ and evolves
toward $`R=1`$ during plastic loading.

### Regularized granular $`\mu(I)`$ rheology

For plastic shear rate $`\dot\gamma_p`$, grain diameter $`d_g`$, grain density
$`\rho_s`$, and compression $`p>0`$, the inertial number is

```math
I=\dot\gamma_p d_g\sqrt{\frac{\rho_s}{p}}.
```

A pressure- and rate-regularized friction coefficient is

```math
\mu
=\mu_s+
\frac{\dot\gamma_p(\mu_2-\mu_s)}
{I_0\sqrt{p/(\rho_s d_g^2)}
+\sqrt{\dot\gamma_p^2+\varepsilon^2}}.
```

The shear strength is $`\tau=\mu p`$. Given trial shear $`\tau_{tr}`$ and shear
modulus $`G`$, the plastic rate satisfies the scalar consistency equation

```math
\tau_{tr}-G\Delta t\,\dot\gamma_p
-p\,\mu(\dot\gamma_p,p)=0.
```

### SANISAND-MS model

SANISAND uses the compression-positive stress ratio tensor
$`\boldsymbol{r}=\boldsymbol{s}/p`$ and a backstress ratio
$`\boldsymbol{\alpha}`$. Its yield surface is

```math
f=\|\boldsymbol{s}-p\boldsymbol{\alpha}\|
-\sqrt{\frac{2}{3}}\,m p.
```

Pressure-dependent elasticity is described by

```math
G=G_0p_{atm}\frac{(2.97-e)^2}{1+e}
\sqrt{\frac{p}{p_{atm}}},
\qquad
K=\frac{2(1+\nu)}{3(1-2\nu)}G.
```

The state parameter and Lode-angle function are

```math
\Psi=e-\left[e_0-\lambda_c
\left(\frac{p}{p_{atm}}\right)^{\xi}\right],
```

```math
g(\theta,c)=
\frac{2c}{(1+c)-(1-c)\cos(3\theta)}.
```

Bounding and dilatancy stress ratios take the forms

```math
M_b=g(\theta,c)M_c\exp(-n_b\Psi)-m,
\qquad
M_d=g(\theta,c)M_c\exp(n_d\Psi)-m.
```

Backstress and memory-surface evolution may be written generically as

```math
\mathrm d\boldsymbol{\alpha}
=\frac{2}{3}h\boldsymbol{b}\,\mathrm d\gamma^p,
\qquad
\mathrm d\boldsymbol{\alpha}_M
=\frac{2}{3}h_M\boldsymbol{b}_M\,\mathrm d\gamma^p.
```

The memory surface changes the bounding distance, dilatancy, and hardening
upon cyclic loading and fabric reversal.

### NorSand model

NorSand uses image pressure $`p_i`$ and stress ratio $`\eta=q/p`$. For shape
parameter $`N\neq0`$, the image stress ratio is

```math
\eta_i
=\frac{M}{N}
\left[
1-(1-N)\left(\frac{p}{p_i}\right)^{N/(1-N)}
\right].
```

Its $`N\rightarrow0`$ limit is

```math
\eta_i=M\left[1+\ln\left(\frac{p_i}{p}\right)\right].
```

The yield function and state parameter are

```math
f=q-\eta_i p,
\qquad
\Psi=v-v_{c0}+\lambda\ln p,
\qquad
v=1+e.
```

Here $`p`$ is expressed in the same calibrated stress unit used to define
$`v_{c0}`$.

The target image pressure and hardening rule are

```math
p_i^*=p\exp\left(-\frac{3.5\Psi}{\beta_{dil}M}\right),
\qquad
\mathrm dp_i=h(p_i^*-p_i)\,\mathrm d\lambda.
```

A Rowe-type non-associated potential has stress gradient

```math
\frac{\partial g}{\partial\boldsymbol{\sigma}}
=\frac{\partial q}{\partial\boldsymbol{\sigma}}
+\left(M-\frac{q}{p}\right)
\frac{\partial p}{\partial\boldsymbol{\sigma}}.
```

The pressure-dependent elastic moduli may be expressed as

```math
K=\frac{vp}{\kappa},
\qquad
G=G_0\sqrt{\frac{p}{p_{ref}}}.
```


## Solver references

- [MPM material selection](../../mpm/README.md)
- [FEM material integration](../../fem/README.md)
- [IGA hyperelastic assembly](../../iga/README.md)
- [FEM--MPM plasticity inside IPC](../../fempm/README.md#4-fully-coupled-equilibrium-and-plasticity)
- [IGA--MPM plasticity inside IPC](../../igampm/README.md#8-plastic-constitutive-models-inside-the-ipc-solve)
- [Direct MPM--AffineBody plasticity inside IPC](../../mpdem/README.md#15-solid-mpm--affinebody-ipc-and-plasticity)
