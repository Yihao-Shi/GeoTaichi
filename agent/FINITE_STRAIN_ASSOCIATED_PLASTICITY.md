# Direct finite-strain plasticity

This note records the implemented model and every deliberate difference from
the references used while adding Direct implicit ULMPM Drucker--Prager and von
Mises plasticity.

## Implemented contract

- Kinematics: multiplicative split `F = F_e F_p`; Direct ULMPM stores the
  accepted total map and the material owns `F_p^{-1}`. The trial is
  `F_e,tr = (I + grad(u)) F_n F_p,n^{-1}`.
- Dimension reduction: the constitutive models always consume a 3-by-3 map.
  Cartesian 2D uses classical plane-strain embedding with incremental
  `F_33=1`; axisymmetric 2D uses the no-swirl map with
  `F_theta_theta=r/R`. Intrinsic 2D DP/J2 invariants are rejected.
- Elastic strain: principal Hencky strain from the SVD of `F_e,tr`.
- Drucker--Prager: physical cohesion in stress units, classical
  Mohr--Coulomb-matched cone, independent constant `DilationAngle` (default
  `FrictionAngle`), perfect plasticity, and hydrostatic cone-apex return.
  Zero dilation uses a tensile apex cap; no dilation evolution is implemented.
- Von Mises: standard `sqrt(3 J2) = YieldStress` convention with optional
  linear isotropic `HardeningModulus`.
- Class structure: Drucker--Prager and von Mises are parallel subclasses of
  the internal `HenckyAssociatedPlasticityModel`; neither yield model inherits
  the other's parameters or public semantics.
- Derivatives: analytic PK1 stress and spectral tangent. The physical
  nonassociated tangent is generally nonsymmetric. A frozen flow shift and
  predicted plastic-volume weight give a symmetric inner potential, PCG,
  PSD projection, and energy Armijo. The material outer loop checks physical
  versus inner Kirchhoff stress and plastic-volume error; a different shift
  giving the same capped stress does not require another global solve.
  Particle-local Aitken relaxation damps the two predictors with a factor in
  `[1e-3, 1]`, reset for each step/retry. Acceptance checks the unrelaxed
  physical mismatch; this does not evolve dilation or the yield surface.
  Production material derivatives do not use finite differences. Plastic
  Direct MPM defaults to PSD projection of the inner Hessian.
- State: accepted total deformation gradient, plastic inverse, equivalent
  plastic strain, and volumetric plastic strain are updated in Taichi kernels.
- Coupled solve: FEM remains elastic. FEM--MPM IPC retains current/swept
  collision culling, point--edge (2D) or point--triangle (3D) CCD/ACCD,
  deformation-gradient CCD, lagged IPC friction, COO/HashTriplet assembly,
  and Armijo line search.
  Coupled preparation, solve, and acceptance form one transaction. Failures
  restore FEM/IGA, MPM particle/grid, displacement, elastic-gradient, and
  plastic-history state on the device.

## Reference and deviation ledger

| Reference | Reused principle | Deliberate difference in GeoTaichi |
| --- | --- | --- |
| Drucker and Prager (1952) | Smooth pressure-dependent cone | GeoTaichi exposes the three established Mohr--Coulomb matching choices through `dpType`; the implementation uses a tension-positive invariant convention documented in the helper theory manual. |
| Li and Feng (2014), *Algorithmic tangent modulus at finite strains based on multiplicative decomposition* | Multiplicative finite-strain elastoplastic kinematics, logarithmic strain, consistent tangent requirement | GeoTaichi uses a closed principal-space return and analytic physical tangent. Symmetric inner solves for nonassociated DP use a frozen flow correction, rather than the reference's general closest-point algorithm. |
| Li, Li, and Jiang (2022), *Energetically Consistent Inelasticity for Optimization Time Integration* and Supplemental Document | MPM force-equivalent elastic trial map and incremental-potential formulation | GeoTaichi uses physical cohesion, classical MC cone matching, independent constant dilation, a tensile apex, and a closed local return. Its material outer loop refreshes the flow correction and plastic-volume predictor; it does not evolve dilation or the yield surface. |

## Current restrictions

- Nonassociated Drucker--Prager requires convergence of the material outer
  loop. A failed loop rejects the step and invokes configured bounded retry.
- Drucker--Prager hardening/softening and a separate tensile cutoff are not
  implemented. The cone apex is the only tensile terminal branch.
- Direct TLMPM finite-strain plasticity is rejected because a fixed reference
  needs explicit `F_p` or evolving rest-basis storage. Direct ULMPM and the
  FEM--MPM and IGA--MPM IPC paths are supported.
- FEM--MPM IPC supports plastic MPM but still requires elastic FEM.
- IGA--MPM IPC supports plastic Direct ULMPM but still requires elastic IGA.
  It retains point--NURBS ACCD, material CCD, Armijo search, lagged PSD
  projection, and user-selected COO/HashTriplet assembly. Plastic history and
  `F0` are committed and rolled back as one device transaction.
- The finite-strain Direct models are separate from the existing Native MPM
  small-strain `DruckerPrager` and `ElasticPerfectlyPlastic` models.
- Plane-stress finite-strain DP/von Mises is not implemented. Cartesian 2D is
  plane strain only; `plane_strain=False` is rejected for these laws.

## Verification

The focused tests check classical cone parameters, independent dilation,
yield-surface return, standard von-Mises scaling, hardening-state commit,
analytic tangent agreement with numerical differentiation, Hessian symmetry,
one-step FEM--MPM IPC runs, and IGA--MPM assembly/ACCD/transaction tests with
contact, CCD, PSD assembly, Armijo, and both sparse formats. The tensile-apex
regression checks that physically identical capped stresses pass the material
criterion while a genuinely different inner stress fails.
The constrained-load regression also checks convergence to an analytical
nonassociated equilibrium when the unrelaxed material fixed point diverges,
despite positive physical and inner tangents.

Primary references:

- D. C. Drucker and W. Prager, “Soil mechanics and plastic analysis or limit
  design,” *Quarterly of Applied Mathematics*, 10(2), 1952.
- C.-J. Li and J.-L. Feng, “Algorithmic tangent modulus at finite strains
  based on multiplicative decomposition,” *Applied Mathematics and Mechanics*,
  35(3), 2014. DOI: 10.1007/s10483-014-1795-6.
- X. Li, Y. Li, and C. Jiang, “Energetically Consistent Inelasticity for
  Optimization Time Integration,” *ACM Transactions on Graphics*, 41(4),
  2022. DOI: 10.1145/3528223.3530072.
