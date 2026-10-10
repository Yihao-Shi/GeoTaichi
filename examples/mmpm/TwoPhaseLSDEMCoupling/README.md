# Two-Phase Material Point Method–Level-Set DEM (MPM–LSDEM) Immersed Boundary Coupling

These 3D examples couple a saturated MPM bed, free-surface fluid, and rigid LSDEM grains. Fluid points use volume-fraction immersed boundary forcing (IBM); solid points use level-set contact. See the [MPM–LSDEM IBM theory log](../../../src/mpdem/README.md#10-fully-resolved-lsdem-and-affinebody-volume-fraction-ibm) for geometry, load exchange, and the hybrid two-phase formulation.

- `wavemaker_lsdem_particles_3d/wavemaker_lsdem_particles_3d.py`: piston wavemaker,
  saturated trapezoidal MPM bed, an irregular trunk embedded in the bed and
  four irregular LSDEM grains on top.
- `sphere_impact_submerged_bed_3d.py`: Section 5.2 of
  `1-s2.0-S0045782526006742-main.pdf`; 0.10 m saturated bed, 0.05 m overlying
  water, and a 0.025 m Teflon sphere.  The 0.20 m circular vessel is represented
  by 32 tangent-plane wall segments, and the air drop is replaced by its
  equivalent water-entry velocity.
- The same impact script accepts `--impactor bunny`: a closed Stanford bunny
  replaces the sphere, preserving its equivalent diameter, volume, Teflon
  density and entry speed. This is a Section 5.2-derived geometry experiment,
  not a reproduction of the paper's spherical impactor. The bunny is upright,
  free to translate and rotate, and starts with its lowest surface at the water.
  Its fluid coverage and penetration checks use the actual moving LSDEM mesh.

`assets/mesh/LSDEM/bunny_impact_watertight.ply` closes the holes in the existing
`assets/bunny_sparse.obj`: Open3D `fill_holes(hole_size=1.0)` adds 34 cap faces,
keeping all 2,503 original vertices and their outline (5,002 faces in total).
The closed-mesh volume differs by less than 1%; simulation scaling uses that
closed volume to recover the original sphere's exact volume. The source mesh
and the default spherical example are unchanged.

Both examples insert the solid MPM template first because only the leading
solid-point prefix enters ordinary LSDEM contact.  Fluid points use IBM.

```bash
python examples/mmpm/TwoPhaseLSDEMCoupling/wavemaker_lsdem_particles_3d/wavemaker_lsdem_particles_3d.py --strict
python examples/mmpm/TwoPhaseLSDEMCoupling/sphere_impact_submerged_bed_3d/sphere_impact_submerged_bed_3d.py --drop-height 1.0 --strict
python examples/mmpm/TwoPhaseLSDEMCoupling/sphere_impact_submerged_bed_3d/sphere_impact_submerged_bed_3d.py --impactor bunny --ppc 2 --strict
```
