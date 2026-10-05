# Newton and Augmented-Lagrangian Optimization for Local Nonlinear Problems

`src/optimizer` contains small Taichi-native Newton and augmented-Lagrangian
optimizers used as local numerical building blocks. These classes are distinct
from the global Newton, line-search, and contact solvers implemented inside
FEM, IGA, MPM, and coupled engines.

## Components

- `Newton`: unconstrained Newton iterations with a gradient-descent fallback
  and Armijo-style backtracking.
- `AugmentedLagrangianNewton`: inequality-constrained optimization for
  constraints written as `g_i(x) <= 0`, with projected augmented terms and
  optional analytic constraint Hessians.

Both classes are `@ti.data_oriented`, and their solve routines are `@ti.func`
methods intended to be called from a Taichi kernel. Objective, gradient,
Hessian, and constraint callbacks must therefore be Taichi-compatible
functions.

## Damped Newton method

For an unconstrained objective $f:\mathbb{R}^n\rightarrow\mathbb{R}$,
Newton's direction solves

$$
\boldsymbol{H}_k\boldsymbol{p}_k=-\boldsymbol{g}_k,
\qquad
\boldsymbol{g}_k=\nabla f(\boldsymbol{x}_k),
\qquad
\boldsymbol{H}_k=\nabla^2f(\boldsymbol{x}_k).
$$

If the Hessian is numerically unusable, the local fallback direction is a
scaled negative gradient. Backtracking starts from $\alpha=1$ and accepts the
Armijo condition

$$
f(\boldsymbol{x}_k+\alpha\boldsymbol{p}_k)
\leq
f(\boldsymbol{x}_k)
+c_1\alpha\boldsymbol{g}_k^T\boldsymbol{p}_k,
\qquad 0<c_1<1,
$$

halving $\alpha$ until the inequality holds or the minimum step is reached.
The local stationarity test is

$$
\|\nabla f(\boldsymbol{x}_k)\|_2^2<\varepsilon.
$$

## Inequality-constrained augmented Lagrangian

Consider

$$
\min_{\boldsymbol{x}} f(\boldsymbol{x})
\quad\text{subject to}\quad
g_i(\boldsymbol{x})\leq0,
\qquad i=1,\ldots,m.
$$

With multipliers $\lambda_i\geq0$, penalty $\rho>0$, and
$\langle z\rangle_+=\max(z,0)$, the Powell--Hestenes--Rockafellar form used
for the primal subproblem is, up to the constant
$-\sum_i\lambda_i^2/(2\rho)$,

$$
\mathcal L_\rho(\boldsymbol{x},\boldsymbol{\lambda})
=f(\boldsymbol{x})
+\frac{\rho}{2}\sum_{i=1}^{m}
\left\langle
g_i(\boldsymbol{x})+\frac{\lambda_i}{\rho}
\right\rangle_+^2.
$$

Define

$$
q_i=g_i(\boldsymbol{x})+\frac{\lambda_i}{\rho},
\qquad
\mathcal A=\{i:q_i>0\}.
$$

Away from the active-set switching surface, its gradient and Hessian are

$$
\nabla\mathcal L_\rho
=\nabla f
+\rho\sum_{i\in\mathcal A}q_i\nabla g_i,
$$

$$
\nabla^2\mathcal L_\rho
=\nabla^2f
+\rho\sum_{i\in\mathcal A}
\left(
\nabla g_i\nabla g_i^T+q_i\nabla^2g_i
\right).
$$

The inner solve applies damped Newton to $\mathcal L_\rho$. The outer update
is the projected multiplier step

$$
\lambda_i^{k+1}
=\max\left(0,\lambda_i^k+\rho_k g_i(\boldsymbol{x}_{k+1})\right).
$$

If the maximum positive constraint violation is not reduced sufficiently, the
penalty is increased:

$$
\rho_{k+1}
=\min(\rho_{max},\gamma_\rho\rho_k),
\qquad \gamma_\rho>1.
$$

The local stopping test combines feasibility and iterate change,

$$
\max_i\langle g_i(\boldsymbol{x}_{k+1})\rangle_+<\varepsilon,
\qquad
\|\boldsymbol{x}_{k+1}-\boldsymbol{x}_k\|_2^2<\varepsilon.
$$

## Usage pattern

```python
import taichi as ti
from src.optimizer.Newton import Newton

def energy(x):
    return 0.5 * (x - 2.0) * (x - 2.0)

def gradient(x):
    return x - 2.0

def hessian(x):
    return 1.0

solver = Newton(energy, gradient, hessian, n_vars=1)
result = ti.field(dtype=ti.f64, shape=128)

@ti.kernel
def solve_local_problems():
    for i in result:
        result[i] = solver.solve(0.0)

solve_local_problems()
```

Initialize Taichi before constructing fields or compiling a calling kernel.
`Newton` wraps the three plain Python callbacks with `ti.func`, so do not
decorate the callbacks separately. The solver is a local operation: call it
inside the kernel's outer particle, node, element, or constraint loop. This
also keeps the Newton loop containing `break` nested after `ti.func` is
inlined, as required by Taichi.

## Scope and limitations

- The variable count is limited to at most 20.
- Small dense systems are solved locally; these classes do not assemble or
  solve global sparse matrices.
- Do not call `solve()` as the only operation in a kernel with no enclosing
  loop. After `ti.func` inlining, its Newton loop would become the kernel's
  outermost loop, where Taichi does not allow `break`.
- Callers are responsible for scaling, feasible initial states, and checking
  returned solutions.
- The augmented-Lagrangian implementation stores host-side multiplier state;
  do not treat it as a replacement for the persistent device contact state in
  `src/fem/contact` or the fully coupled IPC solvers.

There is currently no dedicated optimizer test directory. Production uses of
these utilities should be covered by the owning module's tests.
