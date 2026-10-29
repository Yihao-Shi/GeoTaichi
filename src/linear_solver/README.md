# Sparse Block Assembly and Krylov–Multigrid Solvers for Nonlinear Mechanics

`src/linear_solver` provides the sparse-matrix and Krylov infrastructure
shared by FEM, IGA, MPM, DEM affine bodies, and coupled IPC systems. The
default production paths operate on Taichi fields across CPU, Metal, and CUDA
backends. `device_reduction=False` is rejected on every architecture; NumPy/SciPy
reduction references and validation-matrix generators live in `tests/helpers/`.


`BuildTriplet.install_fixed_pattern()` reserves permanent reduced off-diagonal
slots for immutable topology. Device scatter adds directly to those slots;
dynamic raw sources may overlap them. Reset clears their values, while pattern
rebuilds preserve their coordinates and defer stale-entry compaction until the
next assembly. Symmetric matrices store the strict upper block triangle, with
mirrored matvec and constraint elimination. Finalization is idempotent; reset is
required before appending another assembly. This path requires dense blocks and
device hash reduction; dynamic contact can still use bucket reduction upstream.

## Available representations

| Component | Purpose |
| --- | --- |
| `CoordinateSparseMatrix` | Scalar COO triplets with device matvec and optional conversion |
| `CompressedSparseRow` | CSR storage and row-oriented operators |
| `BuildTriplet` | Device block-triplet assembly, hash reduction, block Jacobi, and Krylov solve |
| `HashReduction` | Duplicate block reduction and reusable sparsity patterns |
| `MatrixFreeCG`, `MatrixFreePCG` | Symmetric matrix-free Krylov methods |
| `MatrixFreeBICG`, `MatrixFreeBICGSTAB`, `MatrixFreePBICGSTAB` | Nonsymmetric matrix-free methods |
| `MultiGridPCG*` | Geometric multigrid pressure/Poisson solvers |

`MatrixFreeKrylovKernel.py` shares reductions and vector kernels across CG, PCG,
BiCGSTAB, and preconditioned BiCGSTAB. Algorithm-specific initialization and the
fused PCG update remain separate within that module. CSR, COO, and legacy
row-major CG kernels stay with their respective matrix representations.

`BuildTriplet` keeps its reduced block-triplet values in place. Its Taichi
matvec uses a cached row adjacency containing only signed indices into those
blocks; pattern version, active prefix, and transpose mode invalidate the
cache. CUDA assigns a warp to each block row, while CPU/Metal use one thread.
The adjacency is allocated on the first matvec, so assembly-only matrices do
not reserve it. This is independent of SciPy and does not change assembly.
BiCGSTAB fuses direction/preconditioning and vector/dot-product updates.
Its reference path returns four scalar reductions per ordinary iteration. It
preserves the shadow residual and recurrence until prospective convergence, then checks
`b - A*x`. A failed check restarts with the true residual and `p = r`;
termination also reports the true residual. Frequent periodic restarts can
destabilize the recurrence of strongly scaled nonsymmetric IPC systems.

Lagged forward solves in `AffineBodyEngine` retain all 12 degrees of freedom
of each body's four control points in its principal block: HashTriplet uses
`preconditioner_group_size=4`, and COO (including the legacy MatrixFree selection)
uses the same device `BlockJacobi` factorization. FEM--ABD (HashTriplet and COO), Direct MPM--ABD and soft-MPM--ABD
use the same device `BlockJacobi`: the affine-control prefix has complete
12-DOF body blocks and the remaining grid nodes have 3-DOF blocks.
`preconditioner_group_nodes` marks that prefix in block rows; omitting it
groups the entire active matrix. Blocks are extracted after all contact terms,
constraint elimination and sparse reduction, so mixed-contact stiffness is
included while the complete coupled matrix remains the Krylov operator.
Other matrices retain their default single-block preconditioner.

Reordered raw contact stencils probe the existing device hash table in parallel
before inserting new coordinates. `last_mapping_misses` counts raw entries
forwarded to the serial insertion pass. Stable raw slots retain their direct
mapping; eviction, duplicate reduction and overflow checks keep the same rules.

PCG uses scaled Cholesky with positive finite pivot checks. BiCGSTAB uses
scaled LU with partial row pivoting, including the transposed preconditioner
for nonsymmetric adjoints. `DeviceKrylov` applies these factors through the
existing `LinearOperator` interface; scalar COO PCG/BiCGSTAB also accept that
interface on device and reference dispatch paths. Invalid/singular factors
are failures. An exact adjoint explicitly requesting PCG-to-BiCGSTAB fallback
also takes that fallback if its principal blocks are not positive definite;
LU still uses the unchanged exact tangent. The requested true-residual test
is preserved. Standalone exact ABD adjoints retain their previous
single-block preconditioner and BiCGSTAB fallback.

CUDA CG/PCG and (preconditioned) BiCGSTAB use `DeviceKrylov`: inner products,
alpha/beta/omega and breakdown guards stay on device. Ordinary CG/PCG iterations
read one small status vector; BiCGSTAB reads two to retain its early alpha-step
convergence check. True-residual checks and final diagnostics add boundary reads.
Status vectors explicitly use FP64. CPU/Metal keep the reference paths;
`GEOTAICHI_KRYLOV_DEVICE=0` selects those paths on CUDA for comparison.
Reduction loops use a power-of-two bound above the active prefix, with inactive
entries excluded. This limits empty CUDA blocks without specializing a kernel
for every active-node count; recurrence equations and true-residual tests stay
the same.
Both mixture MGPCG variants reuse the graph dispatch while retaining their
own stencils, signed pressure operators and breakdown thresholds.

## Matching linear and Newton tolerances

A fixed absolute residual tolerance has units and must match the nonlinear
motion criterion. For BackwardEuler projected ABD with assembled matrix
`A = M + K`, positive-definite inertial mass `M` and `K >= 0`, the remaining
linear error satisfies

```text
max_i |b_i A^-1 r| / dt
    <= sqrt(max_i(b_i M^-1 b_i^T) / lambda_min(M)) * ||r||_2 / dt.
```

Here `b_i` maps affine controls to one surface coordinate and `r = rhs - A*x`
is the true, unpreconditioned residual. The immutable mass and basis, together
with the minimum allowed timestep, give a case-specific fixed linear tolerance
below a chosen fraction of the Newton velocity tolerance. This does not adapt
or relax a tolerance during a solve. The cube example explicitly uses Newton
`0.05 m/s` with PCG absolute `1e-10`, giving a minimum-step linear velocity
bound of `1.2e-3 m/s`. The bunny example uses Newton `1e-5 m/s` with PCG
absolute `1e-12`, giving a bound of `9.9e-7 m/s`. Its strong orthogonality
penalty requires tighter rotation corrections; the complete 4 s CUDA trajectory
is still under validation in the owning example.
Static systems and nonsymmetric or unprojected tangents need a different error
bound; this argument is not a universal absolute tolerance for every solver.
Other time integrators use their actual `dv/dq` factor in place of `1/dt`.

## Sparse block assembly

Let $`\mathcal I_e(a)`$ map local degree of freedom $`a`$ of element or contact
stencil $`e`$ to a global block row. Assembly is the additive scatter

```math
\boldsymbol{A}_{IJ}
=\sum_e\sum_{a,b}
[\mathcal I_e(a)=I]
[\mathcal I_e(b)=J]\,
\boldsymbol{A}_{ab}^{e},
```

```math
\boldsymbol{b}_{I}
=\sum_e\sum_a
[\mathcal I_e(a)=I],\boldsymbol{b}_a^e.
```

Raw triplets with the same coordinate are therefore reduced by

```math
(I,J,\boldsymbol{A}_{IJ}^{(1)}),\ldots,
(I,J,\boldsymbol{A}_{IJ}^{(m)})
\longmapsto
\left(I,J,\sum_{q=1}^{m}\boldsymbol{A}_{IJ}^{(q)}\right).
```

For block vector $`\boldsymbol{x}`$, matrix--vector multiplication is

```math
(\boldsymbol{A}\boldsymbol{x})_I
=\sum_J\boldsymbol{A}_{IJ}\boldsymbol{x}_J.
```

If only one triangle of a symmetric matrix is stored, every off-diagonal
block contributes both
$`\boldsymbol{A}_{IJ}\boldsymbol{x}_J`$ and
$`\boldsymbol{A}_{IJ}^{T}\boldsymbol{x}_I`$. A block-Jacobi preconditioner
uses

```math
\boldsymbol{M}^{-1}
=\mathrm{blockdiag}(\boldsymbol{A}_{00}^{-1},
\boldsymbol{A}_{11}^{-1},\ldots).
```

## Preconditioned conjugate gradients

For a symmetric positive-definite system
$`\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b}`$, start with

```math
\boldsymbol{r}_0=\boldsymbol{b}-\boldsymbol{A}\boldsymbol{x}_0,
\qquad
\boldsymbol{z}_0=\boldsymbol{M}^{-1}\boldsymbol{r}_0,
\qquad
\boldsymbol{p}_0=\boldsymbol{z}_0.
```

One PCG iteration is

```math
\alpha_k
=\frac{\boldsymbol{r}_k^T\boldsymbol{z}_k}
{\boldsymbol{p}_k^T\boldsymbol{A}\boldsymbol{p}_k},
\qquad
\boldsymbol{x}_{k+1}=\boldsymbol{x}_k+\alpha_k\boldsymbol{p}_k,
```

```math
\boldsymbol{r}_{k+1}
=\boldsymbol{r}_k-\alpha_k\boldsymbol{A}\boldsymbol{p}_k,
\qquad
\boldsymbol{z}_{k+1}=\boldsymbol{M}^{-1}\boldsymbol{r}_{k+1},
```

```math
\beta_k
=\frac{\boldsymbol{r}_{k+1}^T\boldsymbol{z}_{k+1}}
{\boldsymbol{r}_k^T\boldsymbol{z}_k},
\qquad
\boldsymbol{p}_{k+1}=\boldsymbol{z}_{k+1}+\beta_k\boldsymbol{p}_k.
```

Convergence is decided from the true Euclidean residual, not the
preconditioned recurrence scalar:

```math
\|\boldsymbol{b}-\boldsymbol{A}\boldsymbol{x}_k\|_2
\leq
\max\left(\varepsilon_{abs},
\varepsilon_{rel}\|\boldsymbol{r}_0\|_2\right).
```

Periodic residual replacement recomputes
$`\boldsymbol{r}=\boldsymbol{b}-\boldsymbol{A}\boldsymbol{x}`$ to prevent
finite-precision drift in stiff systems.

### Choosing fixed tolerances

An absolute residual threshold has the units and scale of the assembled right-hand
side. Multiplying both $`A`$ and $`b`$ by a positive constant leaves the solution
unchanged but scales that residual. The relative term above preserves this
uniform-scaling invariance when it controls the threshold. Both tolerance
parameters are fixed before the solve; iteration exhaustion does not relax them.
The public controls reach both the forward and supported adjoint device solves:

| Solver entry | Absolute residual keyword | Relative residual keyword |
| --- | --- | --- |
| ABD / LSMPM soft--ABD | `linear_tolerance` | `linear_relative_tolerance` |
| FEM, IGA, Direct MPM, FEM--ABD, FEM--MPM | `linear_solver_tolerance` | `linear_solver_relative_tolerance` |
| Direct IPC-MPM / Direct MPM--ABD | MPM child's `linear_solver_tolerance` | MPM child's `linear_solver_relative_tolerance` |
| Native MPM `set_implicit_solver_parameters` | `residual_tolerance` | `linear_solver_relative_tolerance` |
| Monolithic IGA--MPM | `monolithic_linear_solver_tolerance` | `monolithic_linear_solver_relative_tolerance` |

Zero relative tolerance selects absolute-only stopping; zero absolute tolerance
selects relative-only stopping when the relative value is positive. Both must be
finite and non-negative. Newly exposed relative controls default to zero pending
representative physical calibration; existing FEM/coupling defaults are retained.
Monolithic IGA--MPM still requires a positive absolute tolerance because some
of its nonlinear safeguards request an absolute-only inner solve.
Fully implicit friction can tighten both branches to its existing force-residual
bound, so a loose linear setting cannot bypass that nonlinear safeguard. This
does not change the stored user parameters. Pressure solvers retain their own
pressure-equation controls; nonlinear force, velocity, divergence and friction
tests are separate from this linear criterion. Direct SciPy factorization has no
Krylov stopping tolerance.

An explicitly supplied absolute linear threshold selects absolute-only stopping
unless the same configuration call also supplies a relative threshold. This
applies to the public entries above, including repeated setter calls: setting
only an absolute value replaces an earlier relative setting with zero. Supplying
both values explicitly selects the combined `max(atol, rtol * initial_residual)`
criterion. Omitted absolute values use the path's existing defaults/settings.
Native MPM's `residual_tolerance` is a linear threshold; the same keyword on
Direct MPM controls the outer nonlinear iteration and does not reset its linear
relative setting.

For a previously validated example, explicitly retain the threshold(s) used by
that run, along with its outer Newton/friction tolerances, iteration limits,
time step and integration parameters. A formerly absolute-only example needs
only its original absolute keyword, not an extra relative-zero keyword. Use the
run's effective configuration, not today's defaults, as the source. This
preserves the requested stopping policy; it does not guarantee bitwise
reproduction across solver or backend changes. A run that failed at its former
threshold is not a validated reference for that threshold.

A criterion $`|r_k^T M^{-1}r_k| \leq \eta |r_0^T M^{-1}r_0|`$ measures a squared
preconditioned-norm ratio. For example, $`\eta=10^{-3}`$ corresponds to a norm
ratio of about 0.0316 for an SPD preconditioner. This differs from GeoTaichi's
Euclidean relative residual criterion, so these numerical thresholds cannot
be substituted directly.

Neither relative residual nor a floating-point backward-error bound alone
certifies the physical correction: $`e=A^{-1}r`$ also depends on conditioning.
Choose an inner tolerance using representative systems and check the resulting
surface/node velocity error against the outer Newton target; retain independent
friction convergence checks where applicable. A tolerance supported by one
captured contact matrix is case-specific evidence, not a general default for
ABD, FEM, and their coupling. A stalled solve remains a failure; do not increase
its threshold until it passes.

## BiCGSTAB for nonsymmetric systems

Choose a fixed shadow residual $`\widehat{\boldsymbol{r}}=\boldsymbol{r}_0`$.
With $`\rho_k=\widehat{\boldsymbol{r}}^T\boldsymbol{r}_{k-1}`$, the stabilized
recurrence is

```math
\beta_k
=\frac{\rho_k}{\rho_{k-1}}
\frac{\alpha_{k-1}}{\omega_{k-1}},
\qquad
\boldsymbol{p}_k
=\boldsymbol{r}_{k-1}
+\beta_k(\boldsymbol{p}_{k-1}-\omega_{k-1}\boldsymbol{v}_{k-1}),
```

```math
\boldsymbol{v}_k=\boldsymbol{A}\boldsymbol{p}_k,
\qquad
\alpha_k=\frac{\rho_k}
{\widehat{\boldsymbol{r}}^T\boldsymbol{v}_k},
\qquad
\boldsymbol{s}_k=\boldsymbol{r}_{k-1}-\alpha_k\boldsymbol{v}_k,
```

```math
\boldsymbol{t}_k=\boldsymbol{A}\boldsymbol{s}_k,
\qquad
\omega_k=\frac{\boldsymbol{t}_k^T\boldsymbol{s}_k}
{\boldsymbol{t}_k^T\boldsymbol{t}_k},
```

```math
\boldsymbol{x}_k
=\boldsymbol{x}_{k-1}+\alpha_k\boldsymbol{p}_k
+\omega_k\boldsymbol{s}_k,
\qquad
\boldsymbol{r}_k=\boldsymbol{s}_k-\omega_k\boldsymbol{t}_k.
```

Zero denominators or non-finite recurrence coefficients are algebraic
breakdowns and must not be reported as convergence.

## Geometric multigrid preconditioning

On level $`\ell`$, the residual equation is

```math
\boldsymbol{A}_{\ell}\boldsymbol{e}_{\ell}
=\boldsymbol{r}_{\ell},
\qquad
\boldsymbol{r}_{\ell}
=\boldsymbol{b}_{\ell}-\boldsymbol{A}_{\ell}\boldsymbol{x}_{\ell}.
```

A V-cycle applies pre-smoothing, restricts the defect, solves or repeatedly
smooths on the coarsest grid, prolongates the correction, and post-smooths:

```math
\boldsymbol{r}_{\ell+1}
=\boldsymbol{R}_{\ell}\boldsymbol{r}_{\ell},
\qquad
\boldsymbol{e}_{\ell}
\leftarrow\boldsymbol{e}_{\ell}
+\boldsymbol{P}_{\ell}\boldsymbol{e}_{\ell+1}.
```

The resulting approximate inverse acts as the PCG preconditioner. Red--black
Gauss--Seidel and damped Jacobi use, respectively, color-separated relaxation
and

```math
\boldsymbol{x}^{(m+1)}
=\boldsymbol{x}^{(m)}
+\omega\boldsymbol{D}^{-1}
(\boldsymbol{b}-\boldsymbol{A}\boldsymbol{x}^{(m)}),
\qquad \omega=\frac23.
```

On CUDA, `MGPCGPoissonSolver` batches its V-cycle and CG updates with Taichi
graphs. CG coefficients remain on the device; each iteration returns one
residual/breakdown record. Convergence still requires a fresh `b - A*x`.
Set `GEOTAICHI_MGPCG_GRAPH=0` to compare with the Python dispatch path.
`GEOTAICHI_SYNC_TIMERS=0` also removes GPU fences used only for stage timing;
stage timings then measure submission time rather than GPU execution time.

## HashTriplet example

```python
import numpy as np
import taichi as ti
from src.linear_solver.BuildTriplet import BuildTriplet

ti.init(arch=ti.cpu, default_fp=ti.f64)

matrix = BuildTriplet(
    dim=2,
    max_pairs_num=16,
    max_nonzeros=16,
    max_active_nodes=2,
    symmetric=False,
    matrix_symmetric=False,
    solver="PCG",
    device_reduction=True,
)
matrix.reset_system()
matrix.assemble_scalar_triplets(
    np.array([0, 1, 2, 3], dtype=np.int32),
    np.array([0, 1, 2, 3], dtype=np.int32),
    np.array([4.0, 3.0, 2.0, 5.0]),
)
matrix.finalize_taichi_assembly()
result = matrix.solve(
    rhs=np.array([[1.0, 0.0], [0.0, 1.0]]),
    active_nodes=2,
    tol=1.0e-12,
    rel_tol=1.0e-8,
    maxiter=100,
)
solution = result["x"]
```

Solvers normally scatter directly from element/contact kernels rather than
passing NumPy triplets. `append_raw_from()` combines compatible device block
systems without a host round trip and is used by fully coupled coupling.
`tol` is the absolute residual floor and `rel_tol` scales with the initial
true residual; Krylov convergence uses
`max(tol, rel_tol * initial_residual)`. The returned dictionary also contains
`converged`, `iterations`, `residual`, `initial_residual`, and
`convergence_tolerance`; `x` is included when `return_solution=True`.

HashTriplet PCG fuses the solution/residual update, block-Jacobi application,
and the two reductions $`r^Tz`$ and $`r^Tr`$ into one kernel with one scalar-pair
return. Initialization and true-residual restarts use the same fused reduction
layout. Each workspace is overwritten before use, avoiding a full reset of
unused BiCGSTAB vectors. Prospective convergence still requires a fresh $`b-Ax`$.

## Symmetry and solver selection

- PCG requires a symmetric positive-definite or projected positive-semidefinite
  operator with sufficient constraints to remove null modes.
- BiCGSTAB supports nonsymmetric and indefinite systems.
- `solve(..., transpose=True)` and `solve_flat_system(..., transpose=True)`
  apply the transposed block operator and transposed block-Jacobi
  preconditioner without materializing a host matrix.
- `matrix_symmetric=True` stores one off-diagonal block triangle and applies
  the transpose contribution during matvec.
- `full_symmetric_input=True` accepts both raw triangles and canonicalizes
  them before symmetric elimination/reduction.
- Block sizes `1`, `2`, `3`, and dense nonsymmetric `4` are supported by
  `BuildTriplet`; compact symmetric block storage is limited to dimensions 2
  and 3.

## Host boundaries

`to_scipy()`, `_to_scipy()`, and NumPy solution adapters are explicit output
or host-solver boundaries. They are not used by the default device Krylov
path. Pattern caching and device reduction are enabled in production
assemblers to avoid rebuilding or downloading sparse systems each iteration.

## Tests

Backend equivalence, duplicate reduction, block Jacobi, dimensions 1-4,
matrix-free contracts, CSR, and multigrid tests are under
`tests/unit/linear_solver/`.
