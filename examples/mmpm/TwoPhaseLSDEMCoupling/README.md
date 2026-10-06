# Two-Phase Material Point Method–Level-Set DEM (MPM–LSDEM) Immersed Boundary Coupling

These 3D examples couple a saturated MPM bed, free-surface fluid, and rigid LSDEM grains. Fluid points use volume-fraction immersed boundary forcing (IBM); solid points use level-set contact. See the [MPM–LSDEM IBM theory log](../../../src/mpdem/README.md#10-fully-resolved-lsdem-and-affinebody-volume-fraction-ibm) for geometry, load exchange, and the hybrid two-phase formulation.

- `wavemaker_lsdem_particles_3d.py`: piston wavemaker, saturated trapezoidal
  MPM bed, and three fully resolved LSDEM spheres.
- `sphere_impact_submerged_bed_3d.py`: Section 5.2 of
  `1-s2.0-S0045782526006742-main.pdf`; 0.10 m saturated bed, 0.05 m overlying
  water, and a 0.025 m Teflon sphere.  The 0.20 m circular vessel is represented
  by 32 tangent-plane wall segments, and the air drop is replaced by its
  equivalent water-entry velocity.

Both examples insert the solid MPM template first because only the leading
solid-point prefix enters ordinary LSDEM contact.  Fluid points use IBM.

```bash
python examples/mmpm/TwoPhaseLSDEMCoupling/wavemaker_lsdem_particles_3d/wavemaker_lsdem_particles_3d.py --strict
python examples/mmpm/TwoPhaseLSDEMCoupling/sphere_impact_submerged_bed_3d/sphere_impact_submerged_bed_3d.py --drop-height 1.0 --strict
```
