# Hot-Path Reuse Audit

Audit date: 2026-08-12.

## Scope and method

This audit covers code added or substantially changed after 2026-06-17 in
AffineBody, FEM, soft-particle MPM, IGA, IGAMPM, u-p MPM, incompressible MPM,
and double-layer MPM. Call graphs were followed through complete step, Newton,
line-search, CCD, friction, assembly, and update paths. The review treats an
extra device traversal or candidate rebuild as real overhead even when no
NumPy transfer is involved.

The governing rule is narrower than “reuse whenever possible”: reuse a shared
implementation only when its assumptions and lifecycle match the caller. When
reuse introduces preparation/restore passes, repeated geometry queries,
unneeded Hessian assembly, or redundant field copies, share the smallest
mathematical operation and keep a specialized orchestration path.

## Changes made

| Area | Avoided work | Result |
| --- | --- | --- |
| FEM implicit finalization | Reassembled elements and rebuilt contact after the converged Newton state | Reuse the internal-force field from the last accepted Newton assembly; reaction/equilibrium recovery remains on device. |
| FEM augmented Lagrangian | Re-ran violation kernels immediately after contact preparation had already reduced the same state | Read the prepared device reduction for convergence. |
| Soft-particle IPC Armijo | Recomputed the accepted trial energy only for logging | Retain the accepted Armijo scalar; compute once only when line search is disabled. |
| IGAMPM Armijo/ACCD | Rebuilt the base DCD set in the public ACCD wrapper and optionally verified/restored a trial that Armijo immediately rebuilt | Keep the public standalone verified path, but use a private prepared-contact path from Armijo with verification disabled. |
| Moving point--NURBS closest point | Evaluated rational basis values and first/second derivatives once for control points and again for control-point directions | Interpolate position and direction derivatives together from one rational-basis evaluation. |
| Incompressible MPM | Re-applied particle boundary projection after a disabled shifting callback | The post-shift projection is owned by the enabled shifting path only. |
| IGA material CCD | Read a patch prefix as its element count, skipping the first patch | Use each patch's owned end-prefix count consistently. |

## Reuse retained deliberately

| Area | Decision |
| --- | --- |
| AffineBody accepted line-search assembly | Trial probes omit the Hessian, while the accepted state must build the tangent. The final assembly is required; separating every force/Hessian kernel would enlarge and complicate this change. |
| AffineBody and soft-particle lagged friction | The post-update solve is the fixed-point convergence criterion, not duplicate logging work. |
| IGA full native HashTriplet destination | IGAMPM consumes both row orientations for the exact nonsymmetric tangent product; changing to upper-only storage would alter the operator. |
| IGA patch dispatch | Patch metadata and basis signatures differ. The small host dispatch avoids a mirrored dynamic metadata table and does not transfer field data. |
| u-p and single-layer material dispatch | Constitutive objects own specialized Taichi templates/state. Dynamic generic dispatch would add branches and obscure material ownership. |
| Double-layer 2D/3D and material kernels | Grid storage, MAC layout, and G2P operations differ enough that local specialization is the simpler and cheaper boundary. |
| FEM host adapters | NumPy inspection, output, mesh preprocessing, and verification helpers remain outside the production solver hot path. |

## Regression gates

- Moving NURBS queries are compared against explicitly displaced curve and
  surface geometry.
- IGAMPM tests require Armijo to reuse prepared contacts while the public ACCD
  entry point retains standalone verification.
- FEM tests guard against post-Newton reassembly and repeated AL reduction.
- Soft-particle tests count energy evaluations at an accepted Armijo trial.
- Incompressible MPM tests keep the post-shift boundary projection conditional.
- IGA tests traverse the exact element count of every patch during material CCD.

Future performance changes should add a numerical equivalence test and, where
the cost is not structurally obvious, a benchmark under `tools/benchmarks/` or
the repository's opt-in benchmark test layer.
