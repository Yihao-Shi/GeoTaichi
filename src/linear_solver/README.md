# Sparse Block Assembly and Krylov–Multigrid Solvers for Nonlinear Mechanics

`src/linear_solver` provides the sparse-matrix and Krylov infrastructure
shared by FEM, IGA, MPM, DEM affine bodies, and coupled IPC systems. The
default production paths operate on Taichi fields across CPU, Metal, and CUDA
backends.

## Available representations

| Component | Purpose |
| --- | --- |
| `CoordinateSparseMatrix` | Scalar COO triplets with device matvec and optional conversion |
| `CompressedSparseRow` | CSR storage and row-oriented operators |
| `BuildTriplet` | Device block-triplet assembly, hash reduction, block Jacobi, and Krylov solve |
| `HashReduction` | Duplicate block reduction and reusable sparsity patterns |
| `BlockSparsityPatternCache` | Persistent block topology across nonlinear iterations |
| `MatrixFreeCG`, `MatrixFreePCG` | Symmetric matrix-free Krylov methods |
| `MatrixFreeBICG`, `MatrixFreeBICGSTAB`, `MatrixFreePBICGSTAB` | Nonsymmetric matrix-free methods |
| `MultiGridPCG*` | Geometric multigrid pressure/Poisson solvers |

## Sparse block assembly

Let $\mathcal I_e(a)$ map local degree of freedom $a$ of element or contact
stencil $e$ to a global block row. Assembly is the additive scatter

$$
\boldsymbol{A}_{IJ}
=\sum_e\sum_{a,b}
[\mathcal I_e(a)=I]
[\mathcal I_e(b)=J]\,
\boldsymbol{A}_{ab}^{e},
$$

$$
\boldsymbol{b}_{I}
=\sum_e\sum_a
[\mathcal I_e(a)=I],\boldsymbol{b}_a^e.
$$

Raw triplets with the same coordinate are therefore reduced by

$$
(I,J,\boldsymbol{A}_{IJ}^{(1)}),\ldots,
(I,J,\boldsymbol{A}_{IJ}^{(m)})
\longmapsto
\left(I,J,\sum_{q=1}^{m}\boldsymbol{A}_{IJ}^{(q)}\right).
$$

For block vector $\boldsymbol{x}$, matrix--vector multiplication is

$$
(\boldsymbol{A}\boldsymbol{x})_I
=\sum_J\boldsymbol{A}_{IJ}\boldsymbol{x}_J.
$$

If only one triangle of a symmetric matrix is stored, every off-diagonal
block contributes both
$\boldsymbol{A}_{IJ}\boldsymbol{x}_J$ and
$\boldsymbol{A}_{IJ}^{T}\boldsymbol{x}_I$. A block-Jacobi preconditioner
uses

$$
\boldsymbol{M}^{-1}
=\operatorname{blockdiag}(\boldsymbol{A}_{00}^{-1},
\boldsymbol{A}_{11}^{-1},\ldots).
$$

## Preconditioned conjugate gradients

For a symmetric positive-definite system
$\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b}$, start with

$$
\boldsymbol{r}_0=\boldsymbol{b}-\boldsymbol{A}\boldsymbol{x}_0,
\qquad
\boldsymbol{z}_0=\boldsymbol{M}^{-1}\boldsymbol{r}_0,
\qquad
\boldsymbol{p}_0=\boldsymbol{z}_0.
$$

One PCG iteration is

$$
\alpha_k
=\frac{\boldsymbol{r}_k^T\boldsymbol{z}_k}
{\boldsymbol{p}_k^T\boldsymbol{A}\boldsymbol{p}_k},
\qquad
\boldsymbol{x}_{k+1}=\boldsymbol{x}_k+\alpha_k\boldsymbol{p}_k,
$$

$$
\boldsymbol{r}_{k+1}
=\boldsymbol{r}_k-\alpha_k\boldsymbol{A}\boldsymbol{p}_k,
\qquad
\boldsymbol{z}_{k+1}=\boldsymbol{M}^{-1}\boldsymbol{r}_{k+1},
$$

$$
\beta_k
=\frac{\boldsymbol{r}_{k+1}^T\boldsymbol{z}_{k+1}}
{\boldsymbol{r}_k^T\boldsymbol{z}_k},
\qquad
\boldsymbol{p}_{k+1}=\boldsymbol{z}_{k+1}+\beta_k\boldsymbol{p}_k.
$$

Convergence is decided from the true Euclidean residual, not the
preconditioned recurrence scalar:

$$
\|\boldsymbol{b}-\boldsymbol{A}\boldsymbol{x}_k\|_2
\leq
\max\left(\varepsilon_{abs},
\varepsilon_{rel}\|\boldsymbol{r}_0\|_2\right).
$$

Periodic residual replacement recomputes
$\boldsymbol{r}=\boldsymbol{b}-\boldsymbol{A}\boldsymbol{x}$ to prevent
finite-precision drift in stiff systems.

## BiCGSTAB for nonsymmetric systems

Choose a fixed shadow residual $\widehat{\boldsymbol{r}}=\boldsymbol{r}_0$.
With $\rho_k=\widehat{\boldsymbol{r}}^T\boldsymbol{r}_{k-1}$, the stabilized
recurrence is

$$
\beta_k
=\frac{\rho_k}{\rho_{k-1}}
\frac{\alpha_{k-1}}{\omega_{k-1}},
\qquad
\boldsymbol{p}_k
=\boldsymbol{r}_{k-1}
+\beta_k(\boldsymbol{p}_{k-1}-\omega_{k-1}\boldsymbol{v}_{k-1}),
$$

$$
\boldsymbol{v}_k=\boldsymbol{A}\boldsymbol{p}_k,
\qquad
\alpha_k=\frac{\rho_k}
{\widehat{\boldsymbol{r}}^T\boldsymbol{v}_k},
\qquad
\boldsymbol{s}_k=\boldsymbol{r}_{k-1}-\alpha_k\boldsymbol{v}_k,
$$

$$
\boldsymbol{t}_k=\boldsymbol{A}\boldsymbol{s}_k,
\qquad
\omega_k=\frac{\boldsymbol{t}_k^T\boldsymbol{s}_k}
{\boldsymbol{t}_k^T\boldsymbol{t}_k},
$$

$$
\boldsymbol{x}_k
=\boldsymbol{x}_{k-1}+\alpha_k\boldsymbol{p}_k
+\omega_k\boldsymbol{s}_k,
\qquad
\boldsymbol{r}_k=\boldsymbol{s}_k-\omega_k\boldsymbol{t}_k.
$$

Zero denominators or non-finite recurrence coefficients are algebraic
breakdowns and must not be reported as convergence.

## Geometric multigrid preconditioning

On level $\ell$, the residual equation is

$$
\boldsymbol{A}_{\ell}\boldsymbol{e}_{\ell}
=\boldsymbol{r}_{\ell},
\qquad
\boldsymbol{r}_{\ell}
=\boldsymbol{b}_{\ell}-\boldsymbol{A}_{\ell}\boldsymbol{x}_{\ell}.
$$

A V-cycle applies pre-smoothing, restricts the defect, solves or repeatedly
smooths on the coarsest grid, prolongates the correction, and post-smooths:

$$
\boldsymbol{r}_{\ell+1}
=\boldsymbol{R}_{\ell}\boldsymbol{r}_{\ell},
\qquad
\boldsymbol{e}_{\ell}
\leftarrow\boldsymbol{e}_{\ell}
+\boldsymbol{P}_{\ell}\boldsymbol{e}_{\ell+1}.
$$

The resulting approximate inverse acts as the PCG preconditioner. Red--black
Gauss--Seidel and damped Jacobi use, respectively, color-separated relaxation
and

$$
\boldsymbol{x}^{(m+1)}
=\boldsymbol{x}^{(m)}
+\omega\boldsymbol{D}^{-1}
(\boldsymbol{b}-\boldsymbol{A}\boldsymbol{x}^{(m)}),
\qquad \omega=\frac23.
$$

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
