# Prove a numerical change in GeoTaichi

Use this guide to choose an oracle for a material, contact, assembly, or
differentiation change. The [test guide](../tests/README.md) owns fixtures,
partitions, and commit selection. This page explains what the numerical check
must establish; it does not add tests to a commit gate.

## Identify the quantity being checked

Before writing an assertion, record the consuming solver branch, configuration,
dimension, stress/strain measure, units, sign/frame convention, DOF ordering,
and accepted history. State which variables are held fixed during differentiation.
For a material, use the [coverage matrix](../tests/unit/physics_model/materials/README.md)
to locate the maintained law test and the solver-entry test.

Keep these different quantities explicit:

| Quantity | Suitable check |
| --- | --- |
| Energy gradient or physical consistent tangent | Derivative of the same energy/stress/residual, with the same weights and state convention. |
| PSD-projected or lagged Newton matrix | Compare against the declared projection/frozen formulation separately; it need not equal the physical derivative. |
| Fully implicit friction Jacobian | Differentiate the complete nonconservative residual, including the changing normal/friction terms required by that formulation. |
| Eliminated linear system | Check constrained corrections and free equations after applying the same prescribed-value convention. |
| Parameter/history VJP | Compare a seeded derivative while preserving the intended accepted-history and replay contract. |

In particular, the material tangent can be nonsymmetric for a nonassociated
law. Do not enforce symmetry, PSD, or a potential identity merely because the
lagged solve uses a projected matrix. Read the owning
[finite-strain formulation](FINITE_STRAIN_ASSOCIATED_PLASTICITY.md) for its scope.

## Compare derivatives without contaminating history

For a state vector `x`, direction `p`, and residual `R`, compare

```text
J(x) p  with  [R(x + h p) - R(x - h p)] / (2 h).
```

For a conservative energy, also compare `gradient(E) dot p` with the central
energy difference. Check the residual sign in the consuming equation: some
contact tests compare the assembled matrix to minus the force derivative.
Include quadrature, volume/surface measure, timestep scaling, and coordinate
transforms exactly once.

1. Restore the same accepted state before the baseline and each perturbation.
   Reset trial history, contact caches, and scratch accumulators consumed by
   evaluation; never advance accepted history between the plus and minus probes.
2. Choose a direction and perturbation scale appropriate to the variable's
   units. For mixed displacement/pressure/rotation variables, scale components
   deliberately. Evaluate neighboring step sizes and look for a stable error
   range before setting a tolerance; making `h` arbitrarily small loses accuracy.
3. Keep both probes inside the same smooth branch when testing its derivative.
   Check yield/apex, activation, stick/slip, and closest-feature transitions
   separately with the formulation's limiting or one-sided contract. Preserve
   positive determinants and IPC feasibility during probes.
4. Compare physical and projected tangents separately. A PSD projection can
   alter negative curvature even when the physical derivative is correct.
5. Assert finite values and report error with its scale and branch. Use absolute
   and relative tolerances that reflect precision, conditioning, and the finite-
   difference error; reuse neighboring tolerances only when those conditions match.

[Finite-strain energy tests](../tests/unit/physics_model/materials/test_finite_strain_energy.py)
show energy/PK1/tangent comparisons, including column-major flattening of `F`.
[Shared IPC law tests](../tests/unit/physics_model/contact/ipc/test_ipc_shared_scalar_laws.py)
cover regularization branches and full friction derivatives. Their numerical
oracles should remain independent of the production derivative being checked.

## Separate local laws from global assembly

A correct local block can still be scattered to the wrong global row. Check
the local law first, then a small production assembly with an independently
constructed dense operator or matrix-vector product.

- Include duplicate coordinates, off-diagonal blocks, and a nonzero prescribed
  displacement. Check local/global/compact indices and coupled-system offsets.
- For symmetric triangle storage, verify the transpose contribution is present
  exactly once. For a full nonsymmetric Jacobian, retain both distinct blocks.
- Compare `A p` across supported representations before comparing solutions.
  Equivalent operators can use different preconditioners and iteration counts.
- Exercise topology/count changes on a reused workspace, including an empty
  next call, so stale sparse mappings and entries cannot pass a one-shot test.
- Keep the physical tangent product separate from the constraint-eliminated
  matrix when evaluating a merit derivative.

Use [assembly-backend tests](../tests/unit/linear_solver/test_assembly_backends.py),
[physical constrained products](../tests/unit/iga/solver/test_physical_tangent_product.py),
and [fixed-pattern assembly](../tests/unit/iga/assembly/test_fixed_reference_assembly.py)
as concrete starting points. A standalone local Hessian check does not establish
correct coupling assembly.

Check a solved system with a fresh true residual `b - A x`, its finite norm,
and the actual absolute/relative threshold. A small update or preconditioned
recurrence scalar alone is insufficient. Use the
[linear-solver contract](../src/linear_solver/README.md) for stopping rules and
the [implicit-solve workflow](WORKFLOWS.md#diagnose-an-implicit-solve) for failures.

## Validate differentiation at the promised level

For a VJP `J^T v`, compare `p dot (J^T v)` with a directional finite difference
of the seeded output `v dot output`. This exposes transpose and flattening
errors without building a large dense Jacobian. Use deterministic local seeds.

Distinguish a partial derivative with frozen lagged data from the derivative of
a converged solve or a complete trajectory. For the latter, rerun the forward
solve from the same initial state for each perturbation; include the promised
state/history/replay dependence. Resolve nonlinear and linear errors below the
finite-difference signal. Record contact-feature or constitutive-branch changes
that make a central smooth derivative comparison inapplicable.

Existing [material parameter VJPs](../tests/unit/physics_model/test_finite_strain_parameter_vjp.py)
include a frozen-inner nonassociated check. The
[direct-MPM integration tests](../tests/integration/mpm/test_ipc_mpm_fully_implicit_friction.py)
separate local Jacobians, supported lagged adjoints, and trajectory checks.
Their coverage does not imply differentiation support for every friction mode
or every facade.

## Treat rejected trials and restart as numerical contracts

Inventory the state the affected step can mutate: particle/grid/body state,
material history, friction history, accepted geometry, and time/step counters.
Derived candidates and sparse caches must be restored or invalidated/rebuilt
before reuse, according to their owner. Check an accepted trial and a forced
failure; verify that a rejected step leaves the accepted state intact.

Existing [MPM nonconvergence checks](../tests/integration/mpm/test_ipc_mpm_fully_implicit_friction.py)
verify displacement rollback, while
[lagged-friction unit checks](../tests/unit/mpm/test_ipc_mpm_lagged_friction_fixed_point.py)
cover frozen-step data and failed adjoint restoration. Add assertions for newly
affected state rather than assuming these tests cover every history field.

For persistence changes, first check names, shapes, dtypes, active prefixes,
topology, and clock data. Then compare the next accepted step of a restarted
run against an uninterrupted run. Use the
[two-phase restart tests](../tests/unit/mpm/test_two_phase_restart.py) and
[FEM-DEM checkpoint tests](../tests/unit/fedem/test_checkpoint_capacity.py) for
existing ownership/compatibility examples. A VTU visualization round trip
proves only its recorded fields, not simulation continuation.

## Run and report only the relevant evidence

From the repository root, these commands select existing focused checks:

```bash
GEOTAICHI_REAL_DTYPE=float64 python -m pytest -q \
  tests/unit/physics_model/materials/test_finite_strain_energy.py::test_pk1_and_tangent_match_central_finite_differences
GEOTAICHI_REAL_DTYPE=float64 python -m pytest -q --taichi-arch cpu \
  tests/unit/linear_solver/test_assembly_backends.py::test_matrix_free_coo_and_hash_triplet_match_dense_oracle
```

Choose the corresponding test for the law/path being changed. Use isolated
dimension fixtures and verify the effective backend; CPU/f64 evidence does not
establish GPU parallel correctness. For a backend failure, follow
[Taichi debugging](TAICHI_DEBUGGING.md).

Report the oracle, branch/state assumptions, exact command, effective backend
and precision, observed error, and remaining coverage. Broader local law groups
follow the [commit policy](../tests/README.md#local-commit-checks); this guide
does not require a full-directory test run or a long simulation for every edit.
