# Finite Element Method–Material Point Method (FEM–MPM) Coupling with IPC

`src/fempm` provides two-way contact between MPM material points and deforming
FEM boundaries. It has an explicit DEM-law path and a fully coupled
implicit IPC path. The child solvers retain their constitutive state, while
FEMPM owns surface topology, broad-phase storage, contact assembly,
synchronized time, and coupled output.

[Theory log and derivations](#fem--mpm-coupling-theory-log) | [Examples](../../examples/)

## Example-backed capabilities

- Explicit finite element method–material point method (FEM–MPM) contact: [point–membrane example](../../examples/fempm/explicit_point_membrane/explicit_point_membrane.py).
- Fully coupled implicit FEM–MPM incremental potential contact (IPC): [elastic/Drucker–Prager contact](../../examples/fempm/implicit_ipc_elastic_contact/implicit_ipc_elastic_contact.py) and [von Mises contact](../../examples/fempm/implicit_ipc_von_mises_contact/implicit_ipc_von_mises_contact.py).
- Axisymmetric FEM–MPM soil–structure interaction: [Drucker–Prager CPT](../../examples/fempm/cpt_dp/cpt_dp.py).
- Three-dimensional solid structures with DP soil: [flexible barrier](../../examples/fempm/flexible_barrier/flexible_barrier.py) and [upper-clamped wavy plate](../../examples/fempm/wavy_plate_collapse/wavy_plate_collapse.py).
- The linked implicit examples use elastic FEM and ULMPM, with finite-radius barrier contact, coupled assembly, and lagged friction where enabled.

## Package layout

Direct-MPM children selecting `ShapeFunction="QuadBSpline"` use the shared
finite-grid boundary polynomials described in [MPM](../mpm/README.md).
Values, gradients and IPC position Hessians use the same basis; each grid
axis requires at least four nodes. This applies to Cartesian and axisymmetric
FEM–MPM IPC without changing the material or contact measure.

| Path | Responsibility |
| --- | --- |
| mainFEMPM.py | Public FEMPM facade and coupled lifecycle |
| Engine.py | Explicit FEM--MPM stepping and force exchange |
| ImplicitEngine.py | Fully coupled Barrier IPC Newton solve |
| ContactManager.py | Contact properties, capacities, and model selection |
| Patch.py | Deforming FEM contact surface |
| contact/ | Explicit laws and implicit IPC pullback |
| neighbor/ | Dynamic linked-cell and BVH broad phases |
| Recorder.py | Coupled FEM, MPM, and contact output |

## FEM--MPM coupling theory log

### 1. Explicit particle--triangle contact

An explicit MPM contact sample is a material point with center
$`\boldsymbol{x}_p`$, velocity $`\boldsymbol{v}_p`$, mass $`m_p`$, and contact
radius $`R_p`$. For an oriented FEM triangle,

```math
\boldsymbol{x}_f
=\frac{\boldsymbol{x}_0+\boldsymbol{x}_1+\boldsymbol{x}_2}{3},
\qquad
\boldsymbol{n}_f
=\frac{
(\boldsymbol{x}_1-\boldsymbol{x}_0)
\times
(\boldsymbol{x}_2-\boldsymbol{x}_0)
}{
\left\|
(\boldsymbol{x}_1-\boldsymbol{x}_0)
\times
(\boldsymbol{x}_2-\boldsymbol{x}_0)
\right\|
}.
```

The oriented distance, normal gap, and projection are

```math
d=(\boldsymbol{x}_p-\boldsymbol{x}_f)\cdot\boldsymbol{n}_f,
\qquad
g=d-R_p,
```

```math
\boldsymbol{x}_q
=\boldsymbol{x}_p-d\boldsymbol{n}_f.
```

For $`0<d<R_p`$, let

```math
r_c=\sqrt{R_p^2-d^2}
```

and define the finite-face overlap fraction

```math
\chi
=\frac{
\mathrm{area}
\left[
\mathcal{D}(\boldsymbol{x}_q,r_c)
\cap
\triangle(\boldsymbol{x}_0,\boldsymbol{x}_1,\boldsymbol{x}_2)
\right]
}{
\pi r_c^2
}.
```

MPM points have no rotational contact degree of freedom. The relative
velocity used by the explicit law is therefore

```math
\boldsymbol{v}_{rel}
=\boldsymbol{v}_p
-\frac{\boldsymbol{v}_0+\boldsymbol{v}_1+\boldsymbol{v}_2}{3}.
```

If the shared Linear or Hertz--Mindlin law returns
$`\boldsymbol{F}_n+\boldsymbol{F}_t`$, the particle resultant is

```math
\boldsymbol{F}_p
=\chi(\boldsymbol{F}_n+\boldsymbol{F}_t).
```

For an interior projection, let $`\lambda_a`$ be its triangle area
coordinates. The FEM reactions are

```math
\boldsymbol{f}_a^F
=-\lambda_a\boldsymbol{F}_p,
\qquad
\sum_{a=0}^{2}\lambda_a=1.
```

The particle force enters the MPM background grid through its current shape
functions:

```math
\boldsymbol{f}_i^M
=N_{pi}\boldsymbol{F}_p.
```

Partition of unity gives

```math
\sum_i\boldsymbol{f}_i^M
+\sum_{a=0}^{2}\boldsymbol{f}_a^F
=\boldsymbol{0}.
```

Thus the explicit interior stencil preserves linear action--reaction across
the MPM grid and FEM surface. Damping and sliding friction dissipate energy;
the normal elastic contribution follows the shared contact potential.

### 2. Implicit contact geometry

During one implicit step, the current position of MPM surface sample $`p`$ is

```math
\boldsymbol{x}_p(\boldsymbol{u}^M)
=\boldsymbol{x}_{p,n}
+\sum_iN_{pi}\boldsymbol{u}_i^M.
```

The interpolation weights are frozen over the Newton solve, while the active
MPM grid displacement is unknown. In three dimensions, for closest triangle
coordinates $`\beta_a`$,

```math
\boldsymbol{r}
=\boldsymbol{x}_p
-\sum_{a=0}^{2}\beta_a\boldsymbol{x}_a^F,
\qquad
d^2=\boldsymbol{r}\cdot\boldsymbol{r},
```

```math
\beta_0+\beta_1+\beta_2=1.
```

In two dimensions, for closest edge coordinate $`\xi`$,

```math
\boldsymbol{r}
=\boldsymbol{x}_p
-(1-\xi)\boldsymbol{x}_0^F
-\xi\boldsymbol{x}_1^F.
```

With prescribed minimum distance $`d_{min}`$, define

```math
s=d^2-d_{min}^2,
\qquad
\widehat{s}
=(d_{min}+\widehat d)^2-d_{min}^2.
```

For sample measure $`\omega_p`$, the active contact energy is

```math
E_p^c
=\omega_p\widehat d\,b(s),
```

where the scalar finite-clearance barrier $`b`$ and its derivatives are defined
in the
[shared IPC theory](../physics_model/contact_model/README.md#incremental-potential-contact).

### 3. Exact MPM-grid pullback

Let the local contact sites be the MPM point followed by the two or three FEM
primitive nodes. Write the local gradient and Hessian blocks as

```math
\boldsymbol{g}_p
=\frac{\partial E_p^c}{\partial\boldsymbol{x}_p},
\qquad
\boldsymbol{g}_a^F
=\frac{\partial E_p^c}{\partial\boldsymbol{x}_a^F},
```

```math
\boldsymbol{H}_{pp}
=\frac{\partial^2E_p^c}
{\partial\boldsymbol{x}_p\partial\boldsymbol{x}_p},
\qquad
\boldsymbol{H}_{pa}
=\frac{\partial^2E_p^c}
{\partial\boldsymbol{x}_p\partial\boldsymbol{x}_a^F},
```

```math
\boldsymbol{H}_{ab}
=\frac{\partial^2E_p^c}
{\partial\boldsymbol{x}_a^F\partial\boldsymbol{x}_b^F}.
```

The exact residual pullback is

```math
\boldsymbol{r}_i^{M,c}
=N_{pi}\boldsymbol{g}_p,
\qquad
\boldsymbol{r}_a^{F,c}
=\boldsymbol{g}_a^F.
```

Because the MPM interpolation is linear in the grid displacement, the
contact tangent blocks are

```math
\boldsymbol{K}_{ij}^{MM,c}
=N_{pi}N_{pj}\boldsymbol{H}_{pp},
```

```math
\boldsymbol{K}_{ia}^{MF,c}
=N_{pi}\boldsymbol{H}_{pa},
\qquad
\boldsymbol{K}_{ai}^{FM,c}
=N_{pi}\boldsymbol{H}_{ap},
```

```math
\boldsymbol{K}_{ab}^{FF,c}
=\boldsymbol{H}_{ab}.
```

No derivative of $`N_{pi}`$ appears inside this Newton slice. The local IPC
Hessian may be projected to its positive-semidefinite part before this
pullback; the same projected block then generates all four coupled tangent
families.

The local distance energy is translation invariant, so

```math
\boldsymbol{g}_p+\sum_a\boldsymbol{g}_a^F=\boldsymbol{0}.
```

Together with $`\sum_iN_{pi}=1`$, this gives

```math
\sum_i\boldsymbol{r}_i^{M,c}
+\sum_a\boldsymbol{r}_a^{F,c}
=\boldsymbol{0}.
```

Therefore the exact pullback preserves the contact action--reaction null mode
even though FEM and MPM use unrelated discretizations.

### 4. Fully coupled equilibrium and plasticity

Collect FEM nodal positions and active MPM grid displacements in

```math
\boldsymbol{q}
=
\left(
\boldsymbol{x}^F,
\boldsymbol{u}^M
\right).
```

For elastic or incremental-potential materials, the coupled incremental
potential is

```math
\Pi(\boldsymbol{q};\boldsymbol{h}_n)
=\Pi_F(\boldsymbol{x}^F)
+\Pi_M(\boldsymbol{u}^M;\boldsymbol{h}_n)
+\sum_pE_p^c(\boldsymbol{q})
+D_f(\boldsymbol{q};\widehat{\boldsymbol{q}}).
```

Here $`\boldsymbol{h}_n`$ is accepted MPM material history and the hat denotes
the lagged friction frame. Potential materials supply their first variation;
non-potential plastic updates supply the physical force residual directly.
Both are consistently linearized. The Newton equations are

```math
\left(
\boldsymbol{K}_{FF}+\boldsymbol{K}_{FF}^c
\right)\Delta\boldsymbol{x}^F
+\boldsymbol{K}_{FM}^c\Delta\boldsymbol{u}^M
=-\left(\boldsymbol{r}_F+\boldsymbol{r}_F^c\right),
```

```math
\boldsymbol{K}_{MF}^c\Delta\boldsymbol{x}^F
+\left(
\boldsymbol{K}_{MM}+\boldsymbol{K}_{MM}^c
\right)\Delta\boldsymbol{u}^M
=-\left(\boldsymbol{r}_M+\boldsymbol{r}_M^c\right).
```

The FEM body uses total-Lagrangian quadrature. The MPM body uses the
updated total deformation gradient

```math
\boldsymbol{F}_p(\boldsymbol{u}^M)
=\left[
\boldsymbol{I}
+\sum_i
\boldsymbol{u}_i^M\otimes\nabla N_{pi}
\right]\boldsymbol{F}_{p,n}.
```

Finite-strain Drucker--Prager and von Mises return
mappings are defined in the
[shared constitutive theory](../physics_model/consititutive_model/README.md#finite-strain-multiplicative-plasticity).
Accepted plastic history is frozen while evaluating every Newton trial and is
committed only after the complete FEM--MPM step converges.

Nonassociated DP recomputes its physical return and plastic volume at every
Newton trial. The complete material Jacobian includes the plastic-volume
derivative and is neither symmetrized nor PSD projected. FEM--MPM automatically
switches the device solver from PCG to BiCGSTAB with full nonsymmetric storage
and node-block Jacobi preconditioning (scalar Jacobi for COO). This route has
no material-consistency outer loop. Associated DP retains PCG and its existing
frozen-plastic-volume consistency iteration. The lagged-friction outer loop
is independent of this choice.

The Direct ULMPM child also accepts `StateDependentDruckerPrager`, using
SDMC's state-dependent friction/dilation with the finite-strain DP return.
Its consistent nonsymmetric Jacobian follows the same residual/BiCGSTAB
route, including for cloth contact. Lagged friction remains independent;
all 14 history entries, including void ratio and previous-pressure cache,
participate in accepted-step rollback. See
[parameters](../mpm/README.md#state-dependent-dp-in-coupled-implicit-ulmpm) and
[theory](../physics_model/consititutive_model/README.md#state-dependent-finite-strain-drucker--prager).
Parameter/history adjoints for this law are unsupported.

For nonassociated DP, `set_solver({"inexact_newton": True, ...})` optionally
adapts the Taichi Krylov relative tolerance from 0.01 toward a floor of
`max(linear_solver_relative_tolerance, 1e-7)` as Newton's free force residual
decreases. The forcing history restarts at each friction outer iteration.
Force balance, prescribed displacement, and physical correction velocity must
converge together. The first force reference persists across the attempted step's
friction refreshes and resets on retry. Terminal inexact corrections are re-solved
at the configured Krylov accuracy. Fixed-node reaction forces are
excluded from the free force norm. The default is `False`: small systems can
spend more time on extra Newton assemblies than they save in linear solves.

When the MPM material has no global incremental potential, line search uses
the residual merit

```math
\mathcal{M}(\boldsymbol{q})
=\frac{1}{2}
\|\boldsymbol{R}_{free}(\boldsymbol{q})\|^2.
```

For an exact Newton direction,

```math
\mathcal{M}'(0)
=-\|\boldsymbol{R}_{free}\|^2.
```

This keeps plastic return mapping inside the same fully coupled contact
equilibrium without pretending that a non-potential material is hyperelastic.

### 5. Friction and feasible line search

Lagged IPC friction freezes closest-feature weights, contact normals, and
normal-force magnitudes during one Newton solve. After refreshing that frame,
the fixed-point residual is the unapplied correction velocity

```math
\varepsilon_f
=\frac{
\|\Delta\boldsymbol{q}_{unapplied}\|_{phys}
}{
\Delta t
}.
```

The physical norm combines FEM nodal displacement, interpolated MPM particle
displacement, and `h` times the maximum displacement-gradient component. The
axisymmetric MPM contribution also includes `h * abs(delta_u_r) / r`. Inner
Newton and outer probes use this same norm; nonassociated DP also checks the
refreshed free force and prescribed-motion bounds. The updated probe is unapplied
and unclamped.

The accepted line-search step is bounded by

```math
\alpha_{max}
=\min\left(
1,
\alpha_M,
\alpha_F,
\alpha_c
\right).
```

Here $`\alpha_M`$ preserves admissible MPM deformation, $`\alpha_F`$ prevents FEM
element inversion, and $`\alpha_c`$ is point--edge or point--triangle
continuous collision detection. The swept MPM sample follows

```math
\boldsymbol{x}_p(\alpha)
=\boldsymbol{x}_{p,n}
+\sum_iN_{pi}
\left(
\boldsymbol{u}_i^M
+\alpha\Delta\boldsymbol{u}_i^M
\right),
```

while each FEM node follows

```math
\boldsymbol{x}_a^F(\alpha)
=\boldsymbol{x}_a^F
+\alpha\Delta\boldsymbol{x}_a^F.
```

A failed Newton solve, CCD-limited line search, friction fixed point, or
constitutive update restores FEM state, MPM particles and grid state,
deformation gradients, and plastic history to the beginning of the step.

### 6. Axisymmetric contact measure

In a no-swirl meridian solve, the geometric closest-point problem remains
two-dimensional in $`(r,z)`$, but one MPM surface sample represents a full
reference ring. If $`\omega_p^{mer}`$ is its meridional measure at reference
radius $`R_p`$, the contact weight is

```math
\omega_p^{axi}
=2\pi R_p\omega_p^{mer}.
```

Thus

```math
E_p^{c,axi}
=2\pi R_p\omega_p^{mer}
\widehat d\,b(s).
```

The ring factor enters the residual and every coupled Hessian block, while
contact search and CCD remain point--edge operations in the meridian plane.
The material kinematics remain three-dimensional through the hoop stretch

```math
F_{\theta\theta}=\frac{r}{R}.
```

### 7. Search and capacity invariants

Linked cells and BVH generate only conservative candidate sets. Exact
point--edge or point--triangle distance culling defines the active IPC set.
For explicit coupling, a Verlet list is rebuilt when the sum of MPM-point and
FEM-surface motion reaches half the stored skin. For implicit coupling,
current candidates are rebuilt for each changed Newton/trial state; the residual
and tangent at the same state share them. CCD rebuilds swept candidates, so no
Verlet skin participates in feasibility.

Classical FEM material blocks use permanent reduced HashTriplet slots built from
mesh connectivity, including axisymmetric and 3D solid elements. Existing FEM
reference gradients and quadrature weights are reused. Other device contributions
retain their raw assembly path. Newton checks the residual before requesting the
Hessian; the ensuing matrix assembly reuses that trial's contact candidates and
MPM material response. A changed trial or material outer iteration recomputes them.

## Lifecycle

```python
import geotaichi as gt

gt.init(dim=3, arch="gpu", default_fp="float64")
fem = gt.FEM(log=False)
mpm = gt.MPM(log=False)
coupling = gt.FEMPM(fem=fem, mpm=mpm, log=False)

# Configure, allocate, and populate coupling.mpm and coupling.fem first.
coupling.set_configuration(
    domain=[2.0, 2.0, 2.0],
    gravity=[0.0, 0.0, -9.81],
    search="BVH",  # or "LinkedCell"
)
coupling.set_solver(
    {
        "Timestep": 1.0e-5,
        "SimulationTime": 0.1,
        "SaveInterval": 1.0e-3,
        "SavePath": "OutputData/fempm",
    }
)
coupling.add_surface(
    body_ids=[0],
    modifier={"Orientation": "Parallel", "Direction": [0, 0, 1]},
)
coupling.memory_allocate(
    {
        "contact_coordination_number": 32,
        "max_contact_pairs": 32000,
        # LinkedCell only; BVH does not allocate triangle/cell memberships.
        "max_facet_cell_pairs": 200000,
        "verlet_distance_multiplier": 0.1,
    }
)
coupling.choose_contact_model("Linear")
coupling.add_property(
    MPMmaterial=1,
    FEMbody=0,
    property={
        "NormalStiffness": 1.0e6,
        "TangentialStiffness": 5.0e5,
        "Friction": 0.3,
        "NormalViscousDamping": 0.1,
        "TangentialViscousDamping": 0.1,
    },
)
coupling.run()
```

Use `add_surface`, `choose_contact_model`, and `add_property`.

## Implicit IPC lifecycle

The implicit branch uses `mpm_backend="Direct"`, implicit FEM/MPM solvers,
elastic FEM, and either elastic or supported associated finite-strain MPM
plasticity. `MPMmaterial` in the historical `add_property` signature denotes
the MPM body id on this branch; the clearer
`add_ipc_property(MPMbody, FEMbody, ...)` alias is preferred.

For a planar solve, configure both children with `dimension=2`. For an
axisymmetric meridian solve, additionally configure both children and the
coupling with `axisymmetric=True` and the same finite `axis_offset`. Coordinate
component zero is the radius and component one is the axial coordinate. TRI3
FEM geometry must therefore lie in the stored `(r,z)` plane. The axisymmetric
constitutive update is no-swirl three-dimensional kinematics with
`F_theta_theta=r/R`; FEM/MPM volume and contact measures include `2*pi*R`.
For Cartesian 2D DP or von Mises MPM, the material update instead uses the
classical 3D plane-strain embedding: the incremental total map has `F_33=1`,
the elastic state and tangent are 3-by-3, and contact/grid unknowns remain
two-dimensional. Passing `plane_strain=False` with these plastic laws is
rejected; no intrinsic-2D DP/J2 approximation is used.
The explicitly stored third FEM component is automatically constrained, and
the fully coupled coupling embeds each two-component MPM block into its physical
top-left block. Explicit DEM-law FEMPM remains three-dimensional.
MPM and FEM nodal Neumann data are resultant forces. FEM
`add_traction` performs the axisymmetric boundary integration, whereas a raw
resultant supplied at a degree of freedom must already include the desired
revolved measure.

```python
# ULMPM examples use NeoHookean, DruckerPrager, and VonMises.
# For associated Drucker--Prager, set DilationAngle to
# equal FrictionAngle. VonMises optionally accepts HardeningModulus.
mpm.add_material(
    model="VonMises",
    density=7800.0,
    young_modulus=2.0e5,
    poisson_ratio=0.3,
    YieldStress=1.2e3,
    HardeningModulus=5.0e3,
)
model.set_solver(
    {
        "Timestep": 1.0e-3,
        "SimulationTime": 0.1,
        "assemble_type": "HashTriplet",  # or "COO"
        "linear_solver": "PCG",          # or explicit "Scipy"
        "project_pd": True,
        # Optional bounded retry for recoverable Newton/linear/line-search failures.
        "enable_step_retry": True,
        "step_retry_max_retries": 2,
        "step_retry_reduction": 0.5,
        "step_retry_minimum_timestep": 1.0e-5,
    }
)
model.add_surface(body_ids=[0])
model.memory_allocate({})
model.choose_contact_model(
    "IPC",
    dhat=2.0e-2,
    dmin=0.0,
    kappa=1.0e5,
    friction_coefficient=0.3,
    epsv=1.0e-3,
    friction_mode="lagged",
    friction_iterations=-1,
    friction_tolerance=1.0e-7,
    friction_max_iterations=50,
)
model.run()
```

With `friction_iterations=-1`, each complete frozen-friction Newton solve is
followed by a cache refresh and an unapplied updated-system correction probe.
The step is accepted only when `||delta x||_inf / dt <= friction_tolerance`;
the maximum count is a safety cap, not a convergence substitute.

FEM nodes precede compact active MPM grid nodes in the fully coupled unknown.
The FEM internal energy is Total Lagrangian: its rest-shape gradients and
reference quadrature weights remain fixed and `F = dx/dX`. The MPM
side is Updated Lagrangian and advances total `F_n`; its shared plastic
material state owns `F_p,n^{-1}` and forms `F_e,tr = F_tr F_p,n^{-1}`. IPC distance, friction, and
CCD are evaluated in the current spatial configuration. The coupled method is
therefore TL FEM + UL MPM + spatial IPC, not a single shared kinematic update.
Each coupled step is transactional: a failed nonlinear solve or post-solve
device update restores FEM nodal state, MPM particles, grid velocity and
acceleration, `F0`, and plastic history before the exception is propagated.
Bounded step retry is disabled by default. When enabled, only
`NewtonConvergenceError` failures are retried with a geometrically smaller
timestep; capacity, configuration, and other runtime errors are never retried.
An accepted reduced timestep becomes the synchronized child/coupling timestep.
If all attempts fail, the configured entry timestep is restored and structured
failure data remains available through `diagnostics_snapshot()`.
An all-zero MPM `F0` field is initialized to identity; otherwise every
particle must provide a finite positive-determinant map in the full material
dimension.
The Cartesian/axisymmetric 2D branch pulls back the analytic point--edge
6-by-6 Hessian, while the 3D branch pulls back the point--triangle analytic
12-by-12 Hessian, to
the MPM grid with the current particle shape weights. Both body tangents and
the IPC Hessian are projected for PCG with associated materials.
Nonassociated DP retains the exact unprojected Jacobian for BiCGSTAB.
Candidate lists are rebuilt on every
Newton/trial/CCD query without a Verlet multiplier.

## Broad phase and capacities

`LinkedCell` counts every cell overlapped by each expanded, deforming edge or triangle,
runs a prefix sum, and fills compact cell memberships at every rebuild. It
therefore does not assume a fixed number of triangles per cell.

`BVH` rebuilds a Morton-code linear BVH over current primitive AABBs and queries
it directly with each expanded material-point sphere. Both searches perform
the same exact point–edge (2D) or point–triangle (3D) distance cull and feed
the same compact contact list and narrow phase. A rebuild occurs when either the material-point motion
or FEM surface motion exceeds half the coupling skin. Capacity overflow raises
an error instead of dropping contacts.

## Contact and compatibility

The contact equations and circular sphere-section/triangle area fraction match
the DEM-style FEDEM laws. MPM points have no rotational degree of freedom, so
their contact-point velocity is translational; otherwise the normal, damping,
tangential-history, and Coulomb branches are identical. FEM reactions retain
the FEDEM subtriangle-area distribution.

- The explicit branch requires 3D explicit MPM with the particle/grid backend,
  and it must be constructed
  with Lagrangian coupling before its particle field is allocated. The public
  `FEMPM` factory configures this automatically.
- The explicit branch requires explicit FEM. Standalone FEM Barrier IPC is not
  combined with this penalty-contact path.
- Elastic volume/TRI3 and cloth TRI3 FEM may
  supply the coupled triangular boundary. Cloth TRI3 uses its device
  membrane/bending assembler in the fully coupled line search.
- Search, distance culling, contact response, and force exchange run in Taichi.
  Python is restricted to setup, validation, time-loop orchestration, and I/O.
- Implicit IPC requires elastic implicit FEM and implicit ULMPM.
  Supported MPM laws are Neo-Hookean elasticity, perfect Drucker--Prager,
  and associated von Mises with optional linear isotropic
  hardening, with coupled history updates. The
  plastic laws use their shared finite-strain updates and consistent tangents.
  FEM elastoplasticity, TLMPM plasticity, and standalone child FEM/MPM
  IPC are rejected on this coupled path.
- Continuum-solid MPM may couple to cloth FEM; soft-particle MPM--cloth IPC is
  intentionally unsupported and raises explicitly.
- Finite-strain Drucker--Prager supports independent `DilationAngle`
  (defaulting to `FrictionAngle`), uses no dilation evolution or hardening,
  and interprets `Cohesion` as a stress. Zero-dilation hydrostatic tension
  uses the cone apex as a tensile cap, without a separate cutoff parameter.
  `dpType` selects
  `Circumscribed`, `MiddleCircumscribed`, or `Inscribed` Mohr--Coulomb cone
  matching. Associated plastic MPM defaults to `project_pd=True`; the coupled
  nonassociated DP route overrides material projection and selects BiCGSTAB.
  This route also applies to continuum-solid MPM coupled to cloth FEM. It
  removes the material consistency outer loop while retaining the lagged
  friction fixed point and contact/deformation CCD.
- Implicit friction is currently the symmetric lagged IPC potential. Fully
  implicit nonsymmetric friction remains available in IGAMPM but is not yet
  implemented for FEM triangle-to-MPM-grid pullback.
- Implicit `save_data()` records the synchronized FEM and MPM states and the
  run history reports contact diagnostics. The explicit persistent-contact
  NPZ schema has not yet been extended to IPC candidates and frozen friction
  frames.
- Implicit LinkedCell/BVH rebuild current or swept candidates every query and
  do not use `verlet_distance` or `verlet_distance_multiplier`.
- In axisymmetric IPC, a meridional MPM contact sample represents its full
  reference ring. Its fixed IPC weight is the configured meridional measure
  times `2*pi*R`; the contact search and CCD remain two-dimensional in `(r,z)`.

Focused coverage is under `tests/integration/fempm/`.
