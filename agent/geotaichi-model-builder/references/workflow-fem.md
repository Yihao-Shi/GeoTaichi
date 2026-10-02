# FEM Workflow Profile

## Contents

1. Ownership and lifecycle
2. Mesh, rest shape, and material
3. Assembly and solvers
4. Cloth energies
5. Explicit FEM soft particles
6. Contact and collision culling
7. Validation

## 1. Ownership and lifecycle

Use `geotaichi.FEM`. Initialize Taichi first, then configure, add one mesh and
material, add boundaries/contact or cloth energies, set the solver, and call
`build()` or `run()`. Start from `assets/fem_model_template.py`.

## 2. Mesh, rest shape, and material

`create_mesh`/`add_mesh` accept generated box, rectangle, circle, and cylinder
geometry or OBJ/Gmsh/meshio input. An explicit `rest_shape` is optional; when
omitted, preprocessing copies current mesh coordinates. Volume models use the
standard element mapping. A cloth constitutive model on TRI3 automatically
selects the cloth `Ds @ inv(Dm)` membrane map; TRI3 alone does not.

HEX8 additionally accepts the shared incremental solid models: von Mises
elastic-perfectly-plastic, Mohr--Coulomb, Drucker--Prager, state-dependent
Mohr--Coulomb, Modified Cam-Clay, GranularMaterial, SANISAND-MS, and NorSand.
This route is updated-Lagrangian, explicit-only, uses eight Gauss-point stress
and state histories, and rejects FEM IPC/AL. An optional six-component
`initial_stress` initializes those histories. Models without one fixed
preprocessing modulus, currently Modified Cam-Clay, require a user-specified
finite `dt` instead of `dt="auto"`.

## 3. Assembly and solvers

The runtime backend is Taichi. Choose `assemble_type="COO"` or
`"HashTriplet"`. Choose device `PCG`/`BiCGSTAB`, or explicitly use Scipy only
for the linear solve. PCG requires `project_pd=True`. Implicit FEM uses
Newmark/Newton plus Armijo line search; explicit FEM uses lumped mass.

## 4. Cloth energies

Cloth supports quadratic, dihedral, or disabled bending, plus garment stitch,
target spring, and frozen-frame SDF energies. These use analytic device
gradients/Hessians. Stitch point--edge neighborhoods are excluded from
self-contact candidates.

## 5. Explicit FEM soft particles

For multiple explicit volume soft particles, append compatible TET4/HEX8
meshes with `add_soft_particle`, then call
`add_soft_particle_contact("Linear"|"HertzMindlin", ...)`. The body keeps
ordinary FEM constitutive integration; only its extracted TRI3 boundary is
used for contact, so no FEM SDF is allocated. `search` accepts `LinkedCell` or
`BVH`. `ContactThickness` must be positive for the unsigned EE response.
Default law parameters apply to every distinct body pair;
`add_soft_particle_property(body_id1, body_id2, ...)` overrides one pair.
The explicit neighbor list has a Verlet skin and device-resident tangential
history.

For rigid LSDEM SDF contact or implicit elastic AffineBody IPC, use `FEDEM`
instead of standalone FEM. LSDEM is queried at current FEM boundary nodes;
AffineBody uses direct PT/EE IPC with CCD, pair-local lagged friction, and no
Verlet multiplier.

## 6. Contact and collision culling

IPC and augmented-Lagrangian contact require implicit FEM. `broad_phase`
selects LinkedCell or BVH only for conservative AABB overlaps. Common Taichi
collision culling applies topology/stitch exclusions, exact current-distance
or swept conservative-CCD pruning, prefix-sum compaction, and minimum-step
reduction. IPC rebuilds every CCD query without a Verlet multiplier. Do not
describe raw broad-phase candidates as the active contact set.

For multiple disconnected bodies, `FEMMesh.cell_body_ids` owns one body ID
per cell; omission infers connected components. After `add_contact("IPC")`,
`add_contact_property(body_id1, body_id2, ...)` creates independent unordered
pair settings. Once pair settings exist, only listed pairs and any global
planes are active. Pair-specific settings are IPC-only.

## 7. Validation

Check rest/reference geometry, element Jacobians, energy-force-tangent finite
differences, constrained reactions, explicit stability or implicit residual,
and sparse-backend agreement. For explicit soft particles, compare
LinkedCell/BVH PT and EE candidates, total action--reaction, history across a
Verlet rebuild, and timestep sensitivity. For implicit contact, verify stitch
exclusions, barrier feasibility, CCD-limited steps, friction direction, and AL
multiplier/violation updates.
