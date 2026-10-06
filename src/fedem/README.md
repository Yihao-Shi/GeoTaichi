# FEM–DEM, FEM–Level-Set DEM, and FEM–Affine Body Dynamics (ABD) Contact Coupling

`src/fedem` is the documentation home for soft-particle contact and provides
two-way contact between deforming FEM boundary triangles and GeoTaichi DEM
bodies. Its ownership and lifecycle follow `src/mpdem`:
existing child facades retain their state, while the coupling module owns
surface topology, candidate lists, contact history or IPC stencils, exchange
forces, synchronized stepping, and coupled output.

The FEM body is a volume or membrane mesh and never needs its own SDF. A
classical volume soft particle uses TET4/HEX8 elements internally and exposes
its TRI3 boundary for contact. Only the rigid LSDEM side owns an SDF. The
FEM--LSDEM branch samples that existing rigid SDF at current FEM boundary
nodes; the FEM--AffineBody IPC branch uses direct point--triangle and
edge--edge distances and therefore uses no SDF on either side.

[Theory log and derivations](#fedem-coupling-theory-log) | [Examples](../../examples/)

## Example-backed capabilities

- DEM sphere–FEM membrane contact: [sphere–membrane example](../../examples/fedem/ExplicitSphereMembrane/explicit_sphere_membrane.py).
- Explicit deformable-particle contact: [Hertz contact](../../examples/fedem/HertzContact/hertz_contact.py), [mixed funnel](../../examples/fedem/MixedFunnel/mixed_funnel.py), and [isotropic compaction](../../examples/fedem/IsotropicCompaction/isotropic_compaction.py).
- FEM–LSDEM signed-distance contact: [rigid level-set/soft-particle example](../../examples/fedem/ExplicitLevelSetSoftParticle/explicit_levelset_soft_particle.py).
- Fully coupled FEM–affine body dynamics (FEM–ABD) IPC: [volume soft particle](../../examples/fedem/ImplicitAffineIPCSoftParticle/implicit_affine_ipc_soft_particle.py), [membrane](../../examples/fedem/ImplicitAffineIPCMembrane/implicit_affine_ipc_membrane.py), and [cloth/grain drop](../../examples/fedem/ClothAffineIrregularDrop/cloth_affine_irregular_drop.py).
- Pressure-controlled ABD platens: [FEM–ABD triaxial compression](../../examples/fedem/FEMAffinePressureTriaxial/fem_affine_pressure_triaxial.py).

## Solver, material, and soft-particle compatibility

| FEDEM route | Required child solvers | Deformable discretization | Plasticity | Principal limitation |
| --- | --- | --- | --- | --- |
| Deformable--deformable soft particles | Explicit FEM; contact is owned by the FEM child | `TET4` or `HEX8` volume bodies with extracted boundary triangles | Elastic `TET4/HEX8` | This is an explicit contact route; it is not the implicit AffineBody IPC system. |
| DEM sphere--FEM surface | Explicit FEM + `scheme="DEM"` | Volume, membrane, or cloth FEM surface | Elastic examples only | Uses the selected Linear/Hertz--Mindlin law; standalone FEM IPC/AL cannot be combined. |
| LSDEM rigid SDF--FEM surface | Explicit FEM + `scheme="LSDEM"` | Current FEM boundary nodes against the rigid body's existing SDF | Elastic examples only | Only the rigid LSDEM body owns an SDF; the FEM body does not. |
| AffineBody--FEM mesh IPC | Implicit FEM + `scheme="AffineBody"` | Elastic `TET4`, elastic `HEX8`, or cloth | No; FEM elastoplasticity is rejected | Fully coupled ordinary Barrier IPC with lagged friction. Existing standalone FEM contact, if present, must also be IPC rather than augmented Lagrangian. |

Thus “soft particle” describes a contact role, not a new bulk element. Its
bulk response is still determined by the FEM row selected above.

## Package layout

| Path | Responsibility |
| --- | --- |
| `mainFEDEM.py` | Public `FEDEM` facade and lifecycle validation |
| `Simulation.py` | Shared time controls and contact capacity model |
| `ContactManager.py` | Contact-law, property, search, and history lifecycle |
| `Engine.py` | Ordered DEM/contact/FEM explicit step |
| `AffineIPCEngine.py` | Fully coupled AffineBody-control/FEM-node Newton solve |
| `FEDEMBase.py` | Coupled time loop and callbacks |
| `Recorder.py` | Child and coupled-contact output |
| `Patch.py` | Device-resident deforming triangular surface |
| [`../fem/soft_particle/`](../fem/soft_particle/) | Deformable--deformable soft-particle contact owned by the FEM child |
| `contact/` | Linear and Hertz–Mindlin properties and force kernels |
| `neighbor/` | Dynamic linked-cell/BVH sphere and rigid-SDF search |
| `structs/` | Active and history contact records |

## FEDEM coupling theory log

### 1. Deforming FEM interface

For an oriented FEM boundary triangle with current vertices
$`\boldsymbol{x}_0,\boldsymbol{x}_1,\boldsymbol{x}_2`$, define

```math
\boldsymbol{a}
=
(\boldsymbol{x}_1-\boldsymbol{x}_0)
\times
(\boldsymbol{x}_2-\boldsymbol{x}_0),
```

```math
A_f=\frac{1}{2}\|\boldsymbol{a}\|,
\qquad
\boldsymbol{n}_f=\frac{\boldsymbol{a}}{\|\boldsymbol{a}\|}.
```

The current lumped surface measure of FEM node $`i`$ is

```math
A_i=\frac{1}{3}\sum_{f\ni i}A_f.
```

This measure converts a contact traction scale into a nodal force for
node--wall and node--level-set coupling. Point--triangle and edge--edge IPC
instead use the primitive measures associated with their closest-feature
stencils. The scalar force laws are maintained in the
[shared contact-model theory](../physics_model/contact_model/README.md).

### 2. Explicit sphere--triangle coupling

Let a DEM sphere have center $`\boldsymbol{x}_p`$, radius $`R_p`$, translational
velocity $`\boldsymbol{v}_p`$, and angular velocity
$`\boldsymbol{\omega}_p`$. For a FEM face with centroid

```math
\boldsymbol{x}_f
=\frac{\boldsymbol{x}_0+\boldsymbol{x}_1+\boldsymbol{x}_2}{3},
```

the oriented plane distance, gap, and projected sphere center are

```math
d=(\boldsymbol{x}_p-\boldsymbol{x}_f)\cdot\boldsymbol{n}_f,
\qquad
g=d-R_p,
```

```math
\boldsymbol{x}_q
=\boldsymbol{x}_p-d\boldsymbol{n}_f.
```

When $`0<d<R_p`$, the sphere--plane intersection circle has radius

```math
r_c=\sqrt{R_p^2-d^2}.
```

Let $`\mathcal{D}(\boldsymbol{x}_q,r_c)`$ be its disk in the face plane. The
finite-face contact fraction is

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

The symmetric contact point and relative contact velocity are

```math
\boldsymbol{x}_c
=\boldsymbol{x}_q+\frac{g}{2}\boldsymbol{n}_f,
```

```math
\boldsymbol{v}_{rel}
=
\boldsymbol{v}_p
+\boldsymbol{\omega}_p
\times(\boldsymbol{x}_c-\boldsymbol{x}_p)
-\frac{\boldsymbol{v}_0+\boldsymbol{v}_1+\boldsymbol{v}_2}{3}.
```

Let the shared Linear or Hertz--Mindlin law return
$`\boldsymbol{F}_n+\boldsymbol{F}_t`$. The force applied to the sphere is

```math
\boldsymbol{F}_p
=\chi(\boldsymbol{F}_n+\boldsymbol{F}_t).
```

For an interior projection, define the area coordinates

```math
\lambda_0
=\frac{
\mathrm{area}
\triangle(\boldsymbol{x}_q,\boldsymbol{x}_1,\boldsymbol{x}_2)
}{A_f},
```

```math
\lambda_1
=\frac{
\mathrm{area}
\triangle(\boldsymbol{x}_q,\boldsymbol{x}_0,\boldsymbol{x}_2)
}{A_f},
\qquad
\lambda_2
=\frac{
\mathrm{area}
\triangle(\boldsymbol{x}_q,\boldsymbol{x}_0,\boldsymbol{x}_1)
}{A_f}.
```

Then $`\lambda_0+\lambda_1+\lambda_2=1`$ and the FEM reactions are

```math
\boldsymbol{f}_a^F=-\lambda_a\boldsymbol{F}_p,
\qquad
\sum_{a=0}^{2}\boldsymbol{f}_a^F=-\boldsymbol{F}_p.
```

The rigid-particle torque is

```math
\boldsymbol{\tau}_p
=(\boldsymbol{x}_c-\boldsymbol{x}_p)
\times\boldsymbol{F}_p.
```

Thus an interior contact transfers equal and opposite linear momentum and the
finite contact arm supplies the particle moment. If the projected center lies
outside the triangle while the intersection disk still overlaps it, absolute
subtriangle areas extend these weights without renormalization; this is a
finite-face edge convention rather than barycentric interpolation.

### 3. FEM--LSDEM signed-distance coupling

Let an LSDEM rigid body have center $`\boldsymbol{c}`$, rotation
$`\boldsymbol{R}`$, and body-frame signed-distance interpolant
$`\phi(\boldsymbol{X})`$. For FEM node $`\boldsymbol{x}_i`$,

```math
\boldsymbol{X}_i
=\boldsymbol{R}^T(\boldsymbol{x}_i-\boldsymbol{c}),
\qquad
g_i=\phi(\boldsymbol{X}_i).
```

The spatial gradient and outward unit normal are

```math
\boldsymbol{g}_i
=\boldsymbol{R}\nabla_{\!X}\phi(\boldsymbol{X}_i),
\qquad
\boldsymbol{n}_i
=\frac{\boldsymbol{g}_i}{\|\boldsymbol{g}_i\|}.
```

The rigid velocity is evaluated at the same FEM node:

```math
\boldsymbol{v}_r(\boldsymbol{x}_i)
=\boldsymbol{v}_c
+\boldsymbol{\omega}
\times(\boldsymbol{x}_i-\boldsymbol{c}),
```

```math
\boldsymbol{v}_{rel}
=\boldsymbol{v}_i-\boldsymbol{v}_r(\boldsymbol{x}_i).
```

With FEM nodal mass $`m_i`$ and rigid mass $`m_r`$, the pair effective mass is

```math
m^*
=\left(\frac{1}{m_i}+\frac{1}{m_r}\right)^{-1}.
```

For the conservative linear normal energy

```math
U_i
=\frac{1}{2}A_i k_n
\langle-g_i\rangle_+^2,
```

the exact spatial force contains the raw SDF gradient:

```math
\boldsymbol{F}_i^n
=-\frac{\partial U_i}{\partial\boldsymbol{x}_i}
=-A_i k_ng_i\boldsymbol{g}_i
\quad\text{for}\quad g_i<0.
```

Keeping $`\|\boldsymbol{g}_i\|`$ is essential when trilinear interpolation is
not an exact signed-distance field. Damping, Hertz--Mindlin contact, the
energy-conserving Barrier law, and Coulomb return use the same gap, normal,
effective mass, and nodal measure.

The rigid body receives

```math
\boldsymbol{F}_r=-\boldsymbol{F}_i,
\qquad
\boldsymbol{\tau}_r
=(\boldsymbol{x}_i-\boldsymbol{c})
\times\boldsymbol{F}_r.
```

Because both forces act at $`\boldsymbol{x}_i`$, this exchange preserves
discrete action--reaction, contact power, and angular momentum before damping
and frictional dissipation are added.

### 4. Fully coupled AffineBody--FEM Barrier IPC

The AffineBody child uses four vector controls per body. Its kinematics and
incremental potential are defined in the
[shared affine-body theory](../dem/README.md#5-affine-body-mechanics-and-ipc).
For affine surface vertex $`v`$,

```math
\boldsymbol{x}_v^A
=\sum_{a=0}^{3}w_{va}\boldsymbol{y}_a,
\qquad
\sum_{a=0}^{3}w_{va}=1.
```

FEM surface nodes remain direct unknowns,

```math
\boldsymbol{x}_i^F=\boldsymbol{q}_i^F.
```

Collect the generalized unknowns as

```math
\boldsymbol{q}
=
\left(
\boldsymbol{y},
\boldsymbol{x}^F
\right).
```

For one point--triangle or edge--edge stencil, let
$`\boldsymbol{z}_s`$ be its four geometric sites. Each site is a linear map of
the global unknowns,

```math
\boldsymbol{z}_s
=\sum_A B_{sA}\boldsymbol{q}_A.
```

For a FEM node, the only nonzero support is $`B_{sA}=\boldsymbol{I}`$. For an
AffineBody vertex, its four supports are
$`B_{sA}=w_{va}\boldsymbol{I}`$. Let the local contact energy have site
gradient and Hessian blocks

```math
\boldsymbol{g}_s
=\frac{\partial E_c}{\partial\boldsymbol{z}_s},
\qquad
\boldsymbol{H}_{st}
=\frac{\partial^2E_c}
{\partial\boldsymbol{z}_s\partial\boldsymbol{z}_t}.
```

Because every support map is linear, the exact global pullback is

```math
\boldsymbol{r}_A^c
=\sum_sB_{sA}^T\boldsymbol{g}_s,
```

```math
\boldsymbol{K}_{AB}^c
=\sum_s\sum_t
B_{sA}^T\boldsymbol{H}_{st}B_{tB}.
```

No second derivative of the affine support map appears. The scalar
finite-clearance Barrier IPC, edge--edge mollifier, and regularized friction
are defined in the
[shared IPC theory](../physics_model/contact_model/README.md#incremental-potential-contact).

With a frozen friction frame $`\widehat{\boldsymbol{q}}`$, the fully coupled
incremental potential is

```math
\Pi_{AF}(\boldsymbol{y},\boldsymbol{x}^F)
=\Pi_A(\boldsymbol{y})
+\Pi_F(\boldsymbol{x}^F)
+\sum_c E_c(\boldsymbol{y},\boldsymbol{x}^F)
+D_f(\boldsymbol{y},\boldsymbol{x}^F;\widehat{\boldsymbol{q}}).
```

After placing AffineBody controls before FEM nodal blocks, the coupled Newton
system is

```math
\left(
\boldsymbol{K}_{AA}+\boldsymbol{K}_{AA}^c
\right)\Delta\boldsymbol{y}
+\boldsymbol{K}_{AF}^c\Delta\boldsymbol{x}^F
=-\left(\boldsymbol{r}_A+\boldsymbol{r}_A^c\right),
```

```math
\boldsymbol{K}_{FA}^c\Delta\boldsymbol{y}
+\left(
\boldsymbol{K}_{FF}+\boldsymbol{K}_{FF}^c
\right)\Delta\boldsymbol{x}^F
=-\left(\boldsymbol{r}_F+\boldsymbol{r}_F^c\right).
```

The off-diagonal blocks are the direct AffineBody--FEM coupling; dropping
them would produce a staggered force exchange rather than one IPC solve.

Partition of unity gives the virtual-work identity

```math
\sum_a
\boldsymbol{r}_a^c\cdot\delta\boldsymbol{y}_a
+\sum_i
\boldsymbol{r}_i^c\cdot\delta\boldsymbol{x}_i^F
=\sum_s
\boldsymbol{g}_s\cdot\delta\boldsymbol{z}_s.
```

It also preserves common translation as a null mode and transfers the local
contact action--reaction exactly to AffineBody controls and FEM nodes.

For a Newton direction, three feasibility limits are combined:

```math
\alpha_{max}
=\min\left(
1,
\alpha_A,
\alpha_F,
\alpha_c
\right),
```

where $`\alpha_A`$ limits AffineBody self-contact and deformation,
$`\alpha_F`$ prevents FEM element inversion, and $`\alpha_c`$ is mixed PT/EE
continuous collision detection. Swept AffineBody vertices obey

```math
\boldsymbol{x}_v^A(\alpha)
=\sum_a w_{va}
\left(
\boldsymbol{y}_a+\alpha\Delta\boldsymbol{y}_a
\right),
```

while FEM nodes follow

```math
\boldsymbol{x}_i^F(\alpha)
=\boldsymbol{x}_i^F+\alpha\Delta\boldsymbol{x}_i^F.
```

### 5. Friction, convergence, and accepted state

Lagged Coulomb friction freezes closest-feature weights, normals, and normal
forces during one Newton solve. After the solve, the contact frame is
refreshed and the unapplied coupled correction is measured in velocity units:

```math
\varepsilon_f
=\frac{\|\Delta\boldsymbol{q}_{unapplied}\|_{\infty}}{\Delta t}.
```

The outer fixed point is accepted only when
$`\varepsilon_f\leq\varepsilon_{tol}`$. A failed Newton solve, line search,
friction fixed point, or time step restores both children to their accepted
state. Affine controls, FEM positions, velocities, accelerations, and contact
history are committed together.

## Soft-particle contact

A soft particle is an ordinary TET4 or HEX8 deformable FEM body whose contact
surface is its extracted triangular boundary. Its bulk constitutive response,
mass, and deformation gradient remain classical FEM; FEDEM owns the contact
interpretation and coupled workflows documented here.

### Explicit deformable--deformable stencil

For a point--triangle or edge--edge stencil with signed interpolation weights
$`w_i`$, define

```math
\boldsymbol{r}=\sum_iw_i\boldsymbol{x}_i,
\qquad
\boldsymbol{n}=\frac{\boldsymbol{r}}{\|\boldsymbol{r}\|},
\qquad
\boldsymbol{v}_{rel}=\sum_iw_i\boldsymbol{v}_i.
```

With lumped nodal masses, its effective mass is

```math
m^*=\left(\sum_i\frac{w_i^2}{m_i}\right)^{-1}.
```

For contact thickness $`h`$, geometric gap $`g`$, and contact measure $`A_c`$,

```math
\delta=h-g,
\qquad
K_n=A_ck_n,
\qquad
K_t=A_ck_t.
```

The [Linear and Hertz--Mindlin laws](../physics_model/contact_model/README.md#discrete-contact-kinematics-and-dem-laws)
and the [energy-conserving Barrier law](../physics_model/contact_model/README.md#explicit-finite-clearance-logarithmic-barrier-law)
are defined in the shared contact-model theory.
If their resultant is $`\boldsymbol{F}=f_n\boldsymbol{n}+\boldsymbol{F}_t`$,
nodal transfer is

```math
\boldsymbol{f}_i=w_i\boldsymbol{F},
\qquad
\sum_i\boldsymbol{f}_i=\boldsymbol{0}.
```

Thus the contact stencil obeys action--reaction independently of the bulk FEM
material. Sliding and damping dissipate energy; the conservative normal law
stores and returns its contact potential.

### Deformable--deformable setup

The deformable bodies are configured on the FEM child before any optional
FEDEM construction. Multiple compatible meshes can be appended with
`add_soft_particle`:

```python
fem = gt.FEM(log=False)
fem.set_configuration(dimension=3, solver_type="Explicit")
fem.add_soft_particle(first_tet_mesh)
fem.add_soft_particle(second_tet_mesh)
fem.add_material(
    "NeoHookean", density=1000.0,
    young_modulus=2.0e5, poisson_ratio=0.3,
)
fem.add_soft_particle_contact(
    "Linear",
    search="BVH",                 # or "LinkedCell"
    ContactThickness=1.0e-2,
    NormalStiffness=1.0e7,
    TangentialStiffness=5.0e6,
    Friction=0.3,
    NormalViscousDamping=0.1,
    TangentialViscousDamping=0.1,
    verlet_distance_multiplier=0.1,
)
fem.run()
```

Default law parameters apply to every distinct body pair. Use
`add_soft_particle_property(body1, body2, ...)` to override one pair. The
`HertzMindlin` law accepts `ShearModulus`, `Poisson`, `Restitution`,
`Friction`, and `ContactThickness`.

The broad phase rejects same-body topology and produces both directed
point--triangle and unordered edge--edge candidates. The PT law uses the
oriented signed gap and current lumped nodal surface area; the two directed PT
passes each receive half that area. EE uses unsigned segment distance and
therefore requires a positive `ContactThickness` to act before edge crossing.
Tangential history survives Verlet rebuilds through a device hash keyed by
contact kind and stencil. All narrow-phase, force, friction, area-update, and
history operations are Taichi kernels/functions.

For a rigid LSDEM target, use `FEDEM` with `scheme="LSDEM"`: FEM boundary
nodes query the rigid body's existing SDF. For an implicit elastic volume soft
particle against an AffineBody, use `FEDEM` with `scheme="AffineBody"` and the
`IPC` model; that fully coupled route rebuilds PT/EE collision candidates during
Newton/CCD and does not use a Verlet multiplier.

## Construction order

```text
configure and allocate DEM/LSDEM/AffineBody; add its material and bodies
-> configure FEM; add mesh, material, and boundaries
-> construct FEDEM(dem, fem)
-> set coupling configuration and synchronized solver
-> select the FEM boundary surface
-> allocate coupling capacities
-> choose Linear/HertzMindlin or IPC and add every active pair property
-> select output and run
```

The public constructor is `geotaichi.FEDEM()`.

## Maintained examples

- `examples/fedem/ExplicitLevelSetSoftParticle/explicit_levelset_soft_particle.py` launches a free TET4
  soft particle at a fixed LSDEM sphere. Only the rigid sphere owns an SDF;
  current FEM boundary nodes query `gapn`, and the linear DEM law transfers
  equal-and-opposite force/torque with friction.
- `examples/fedem/ImplicitAffineIPCSoftParticle/implicit_affine_ipc_soft_particle.py` uses the same sphere
  asset and an elastic TET4 soft particle with a constrained top face. It runs
  pair-local frictional IPC with mixed PT/EE culling, CCD, Armijo line search,
  PSD projection, HashTriplet assembly, and device PCG.

Both scripts accept `GT_ARCH`, `GT_FEDEM_SEARCH`, `GT_FEDEM_DT`,
`GT_FEDEM_TIME`, `GT_FEDEM_SAVE_INTERVAL`, and `GT_FEDEM_SAVE_PATH`.
The implicit script additionally accepts `GT_FEDEM_ASSEMBLE_TYPE` and
`GT_FEDEM_LINEAR_SOLVER`; the impact speed is controlled by
`GT_FEDEM_IMPACT_SPEED`. The first implicit step includes Taichi JIT time for
the analytic volume/contact Hessian and is therefore much slower than later
steps.

## Minimal usage

```python
import geotaichi as gt

gt.init(arch="gpu", default_fp="float64", log=False)

dem = gt.DEM(log=False)
fem = gt.FEM(log=False)

# Configure, allocate, and populate both child models first.
coupling = gt.FEDEM(dem=dem, fem=fem, log=False)
coupling.set_configuration(
    domain=[2.0, 2.0, 2.0],
    gravity=[0.0, 0.0, -9.81],
    search="LinkedCell",  # or "BVH"
    log=False,
)
coupling.set_solver(
    {
        "Timestep": 1.0e-5,
        "SimulationTime": 0.1,
        "SaveInterval": 1.0e-3,
        "SavePath": "OutputData/fedem",
    },
    log=False,
)
coupling.add_surface(
    body_ids=[0],
    modifier={"Orientation": "Parallel", "Direction": [0, 0, 1]},
)
coupling.memory_allocate(
    {
        "contact_coordination_number": 32,
        "max_contact_pairs": 32000,
        "max_facet_cell_pairs": 200000,  # linked-cell explicit branch
        "max_levelset_cell_pairs": 64000,  # linked-cell LSDEM branch
        "verlet_distance_multiplier": 0.1,
    }
)
coupling.choose_contact_model("Linear")
coupling.add_property(
    DEMmaterial=0,
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

Use `choose_contact_model`, `add_property`, and `add_surface`, matching
`src/mpdem`.
The legacy `run(callback=...)` spelling is forwarded to the DEM engine's servo
callback. Prefix other child setup callbacks with `dem_`; `function` is the
coupled post-step Taichi callback.

For explicit LSDEM, keep the same public lifecycle, configure the DEM child
with `scheme="LSDEM"`, and select `Linear` or `HertzMindlin`. Each current FEM
boundary node queries `dem.scene.box.distance` and `calculate_gradient` for
the rigid target. Its nodal force is weighted by one third of the current area
of every incident boundary triangle; the rigid receives the exact opposite
force and moment. The coupling never builds or advects an SDF for the FEM
body.

For an elastic implicit FEM soft particle and `scheme="AffineBody"`, select
IPC instead. TET4/HEX8 volumes expose their triangulated boundary; TRI3
membranes use their mesh directly:

```python
coupling.choose_contact_model(
    "IPC", dhat=2.0e-2, dmin=0.0, kappa=5.0e4,
    friction_coefficient=0.3, epsv=1.0e-3,
    friction_mode="lagged", friction_iterations=-1,
    friction_tolerance=1.0e-7, friction_max_iterations=50,
)
coupling.add_ipc_property(
    AffineBody=0, FEMbody=0,
    property={"dhat": 1.0e-2, "kappa": 1.0e5,
              "friction_coefficient": 0.4},
)
```

The `-1` mode strictly certifies the mixed lagged-friction fixed point.  It
refreshes the coupled normal force and tangent frame after a complete Newton
solve, measures the unapplied physical surface/FEM correction in velocity
units, and rolls the whole step back if the safety cap is reached first.

The mixed IPC Newton system places AffineBody control blocks before FEM nodal
blocks. `assemble_type` may be `"COO"` or `"HashTriplet"` and
`linear_solver` may be device `"PCG"` or the explicitly selected host
`"Scipy"` solve. Device PCG accepts an absolute
`linear_solver_tolerance` and a scale-independent
`linear_solver_relative_tolerance`; convergence uses the larger of the two
thresholds. Current and swept contact candidates are rebuilt for every
Newton/CCD query, so this implicit route does not use a Verlet multiplier.

The implicit branch also accepts `enable_step_retry` (default `False`),
`step_retry_max_retries` (default `2`), `step_retry_reduction` (default `0.5`),
and `step_retry_minimum_timestep` (default `0`, meaning no lower bound).
Retries are transactional and apply only to `NewtonConvergenceError`; capacity,
configuration, and unrelated runtime errors propagate immediately. A reduced
timestep is retained after success. Exhaustion restores the entry timestep and
exposes the bounded attempt/contact/linear-solver record through
`diagnostics_snapshot()`.

### Pressure-controlled AffineBody platens

An AffineBody wall can be made kinematic inside the fully coupled IPC solve with
`prescribe_affine_body_velocity(body_id, velocity)`. All four affine controls
receive the same velocity and their end-of-step positions are eliminated from
the Newton correction, so the wall translates without rotating or deforming.

`add_affine_body_pressure_servo` adds normal stress control. The mixed IPC
gradient is reduced to the wall resultant

```math
\mathbf F_b=\sum_{a=0}^{3}\mathbf f_{b,a}^{\mathrm{IPC}},\qquad
p_b=\frac{\max(0,-\mathbf F_b\cdot\mathbf n_b)}{A_b},
```

where $`\mathbf n_b`$ points from the wall into the specimen. After an accepted
fully implicit step, the next prescribed normal velocity is

```math
\mathbf v_b=\mathbf n_b\,\mathrm{clip}\!\left[
g\frac{p^\star-p_b}{p^\star},-v_{\max},v_{\max}\right].
```

Thus contact and material response remain in the converged IPC system; the
controller changes only the next wall boundary value. The accepted pressure
and velocity history is available as `engine.affine_pressure_history`.

## Contact and search invariants

For the ordinary sphere branch, the sphere sees the mean velocity of the three
FEM face nodes, including its own rotational contact-point velocity. The normal law is one-sided with
respect to the oriented surface. Contact force is multiplied by the exact
sphere-section/triangle intersection fraction. The particle force is
distributed with opposite sign to the three FEM nodes using the source's
absolute subtriangle weights. Those weights sum to one for a projection inside
the face; they are intentionally not renormalized when the projected center is
outside the face but its circular section still intersects it. The migrated
source torque convention is retained exactly.

The deforming face size is not assumed constant. Every linked-cell
coupling-list rebuild
counts all linked cells overlapped by each face AABB expanded by particle
radius and skin, performs a prefix sum, then fills a compact membership list.
The sphere list is rebuilt when either the DEM Verlet criterion or FEM surface
motion exceeds half the skin. A rigid LSDEM SDF can rotate without translating
its center, so its broad phase is rebuilt every explicit step and does not
reuse a Verlet list. That search inserts every current rotated rigid level-set
AABB into linked cells or an LBVH, queries only FEM boundary nodes, and uses
the rigid SDF as the narrow phase. Linked-cell rigid memberships use a
count--prefix-sum--fill pass bounded by `max_levelset_cell_pairs`; BVH leaves
use each rigid body's own AABB rather than a global maximum-radius surrogate.
Capacity overflow raises an error instead of dropping contacts.

For IPC, collision culling applies topology and cross-system exclusions before
exact distance evaluation. The analytic Hessian path classifies the closest
feature first and dispatches PP, PE, PT, or EE device kernels separately. This
preserves the exact analytic formulas while avoiding one oversized Taichi AST
that statically expands every feature Hessian.

## Compatibility

- `scheme="DEM"` and `scheme="LSDEM"` use explicit FEM plus the selected
  Linear/Hertz--Mindlin DEM law. Internal FEM Barrier IPC is not combined with
  this penalty-contact path.
- `scheme="AffineBody"` uses implicit elastic FEM and coupled IPC. Standalone
  FEM contact and FEM elastoplasticity are rejected in this route.
- Mixed FEM--AffineBody IPC supports pair-local lagged Coulomb friction with
  `friction_coefficient`, `epsv`, and one or more coupled
  `friction_iterations`. The frozen PT/EE closest-feature frame and barrier
  normal force are refreshed between fixed-point iterations while the
  beginning-of-step positions remain fixed. AffineBody self-contact must still
  use lagged friction with its own `friction_iterations=1`; fully implicit
  mixed or affine self-friction is not accepted by this fully coupled engine.
- Classical TET4/HEX8, classical TRI3, cloth TRI3, and explicit HEX8
  elastic FEM supplies the explicit deforming surface. The implicit
  soft-particle--AffineBody IPC route is covered with a TET4 volume regression
  and remains elastic FEM only.
- All hot-path search, contact, force transfer, and time integration remains
  in Taichi kernels. Python owns setup, capacity validation, time-loop control,
  and output only.

Focused coverage is under `tests/integration/fedem/`.
