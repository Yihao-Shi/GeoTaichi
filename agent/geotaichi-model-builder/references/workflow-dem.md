# DEM and LSMPM Workflow Profile

## Contents

1. Ownership and ordinary DEM
2. Contact completeness
3. Level-set and AffineBody branches
4. LSMPM soft bodies
5. Capacity and timestep
6. Validation

## 1. Ownership and ordinary DEM

Use `geotaichi.DEM`. Typical dependency order:

```text
environment -> import/init -> DEM -> set_configuration -> set_solver
-> memory_allocate -> add_attribute -> add_template/add_region
-> create_body/add_body/add_wall -> choose_neighbor
-> choose_contact_model -> add_property for every active pair
-> select_save_data -> run
```

Register prerequisites before consumers and allocate before inserting runtime
bodies. Use the exact scheme, search, wall, body, and template vocabulary from
the current facade/generators.

## 2. Contact completeness

Contact configuration has three separate parts:

1. geometry/search and list capacity;
2. selected particle-particle and particle-wall models;
3. material-pair properties for every pair that can become active.

Check stiffness or modulus interpretation, friction aliases, damping,
rolling/twisting terms, and model-specific parameters. Missing pair properties
are a model error, not a tuning opportunity.

## 3. Level-set and AffineBody branches

For LSDEM, size rigid bodies, templates, SDF grid nodes, surface nodes, and
point coordination. For AffineBody, use its dedicated configuration, solver,
contact, neighbor, output, and backend restrictions. Do not mix ordinary DEM
contact assumptions into the IPC affine branch.

## 4. LSMPM soft bodies

Use `DEM.set_configuration(scheme="LSMPM")` for the current runnable route.
Do not emit `MPM(mode="SoftParticle").run()`; direct MPM scene ownership is
currently rejected.

For exact soft-grid sizing:

1. configure LSMPM;
2. register the level-set template;
3. call `preprocess_soft_grid_template`;
4. feed returned material-point/grid/support counts into `memory_allocate`;
5. add soft material attributes and insert bodies.

Select level-set transport with
`soft_levelset_advection_scheme="SemiLagrangian"` or `"WENO5"`. WENO5 uses
its own positive CFL and may subcycle. Redistance, advection interval, volume
correction, and compact-domain checks are separate settings. Compact storage is
precomputed contiguous support storage; legacy sparse names normalize to it
and do not create pointer SNodes.

LSMPM contact uses reference-area surface quadrature and a conservative
surface-node effective mass in the critical timestep. For the
energy-conserving law, `FreeParameter` must be at least two.

## 5. Capacity and timestep

Justify:

- material/body/sphere/clump/template capacities;
- level-set, surface-node, soft material-point, and soft-grid capacities;
- body/wall/point coordination and Verlet padding;
- wall geometry and digital-elevation capacities;
- soft velocity constraints and cached template support.

Call `check_critical_timestep()` after material/contact setup. Treat a timestep
reduction as evidence that the requested step was unsafe; do not conceal it in
a production handoff. Use strict timestep routing when the coupled workflow
supports it.

## 6. Validation

Check body and contact counts, overlap/gap, momentum and energy behavior,
wall/servo activation, force symmetry, stable timestep, SDF gradient/domain
quality for LSMPM, and a benchmark observable such as restitution, angle of
repose, runout, or deformation.
