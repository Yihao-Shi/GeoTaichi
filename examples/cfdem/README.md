# Adopted CFDEM results

Each example's `OutputData` retains only the adopted result listed below.
`particles`, `grids`, `vtks`, and contact/wall data inside that result are parts
of one simulation, not duplicate simulations.

| Example | Adopted result, relative to this directory | Acceptance evidence |
| --- | --- | --- |
| Drafting–kissing–tumbling | [`FullyResolved/IBMDraftingKissingTumbling/OutputData/ibm_dkt_strict_final_d8p35`](FullyResolved/IBMDraftingKissingTumbling/OutputData/ibm_dkt_strict_final_d8p35/) | `metrics.strict.json`: passed |
| Fully resolved sphere settling in oil | [`FullyResolved/IBMResolvedSphereSettling/OutputData/sphere_settling_strict_d10_t0p8_fixed`](FullyResolved/IBMResolvedSphereSettling/OutputData/sphere_settling_strict_d10_t0p8_fixed/) | `metrics.json`: passed, complete 0.8 s run |
| IBM level-set dam break | [`FullyResolved/IBMLevelSetDamBreak3D/OutputData/dam3d_highres_sdf_vf/output`](FullyResolved/IBMLevelSetDamBreak3D/OutputData/dam3d_highres_sdf_vf/output/) | `metrics.json` and `validation_connectivity_metrics.json`: passed, complete 1.2 s run; enclosed air explicitly allowed |
| Fixed-cylinder flow | [`FullyResolved/IBMFixedCylinder2D/OutputData/ibm2d_karman_postfix_20260907/model`](FullyResolved/IBMFixedCylinder2D/OutputData/ibm2d_karman_postfix_20260907/model/) | `metrics.json`: passed; current cylinder gallery source |
| Semi-resolved sphere falling in oil | [`SemiResolved/SphereFallingOil/OutputData/E4_CA2_wall_lubrication_formal_20260910`](SemiResolved/SphereFallingOil/OutputData/E4_CA2_wall_lubrication_formal_20260910/) | `acceptance_pre_wall_20261003.json`: accepted on the pre-wall interval; historical full-window `metrics.json` is unchanged |

## Pressure VTU postprocessing

The retained raw NPZs are historical simulation data. In the sealed DKT and
oil tanks, their initial hydrostatic pressure used a top-zero reference while
the later solver pressure used an approximately zero-mean reference. The
re-exported VTUs subtract the fluid-cell mean from both the particle and grid
pressure, using the same offset, when the saved grid has no resolved air reference.
An isolated one-cell void surrounded by fluid/walls does not switch this gauge. This changes neither pressure
gradients nor velocity, geometry, force histories, or acceptance records.
Negative values in these sealed-tank VTUs are relative pressure, not negative
absolute pressure.

The dam break and fixed-cylinder output contain air-reference cells: their
pressure remains in its atmospheric/outlet gauge. Do not subtract their mean
to flatten legitimate startup or free-surface pressure changes.

The DKT data have one saved air cell at each of `t=0.48 s` and `t=0.51 s`,
near the wall (`[50, 1, 82]` and `[50, 15, 63]` respectively). These two frames
now use the same zero-mean output gauge as the neighboring frames. Their original
fluid-pressure means were approximately −27.94 and −73.52 Pa; the refreshed grid
VTUs have means within `2e-14 Pa` of zero. Raw NPZs and kinematic acceptance are
unchanged. The source classifier now handles isolated wall voids, but a historical
pressure-gradient error cannot be repaired by subtracting an output offset.

Each retained result has a `vtk_postprocess_pressure_report_20261011.json`
with frame times, pressure offsets, output counts and checksums for selected
unchanged raw snapshots, trajectories and acceptance files. Exports include
`GraphicMPMParticle*.vtu`, `GraphicMPMGrid*.vtu`, and the appropriate DEM/LSDEM
geometry. Initial, next and final exported VTUs are read back to verify pressure
and particle kinematics against the source NPZs.

Re-export without running the solver:

```sh
python tools/postprocessing/mpdem/reprocess_cfdem_vtk.py --backup-root /path/to/a/new/backup
```

Old VTUs and five superseded result directories were moved to the recoverable
cleanup directory recorded in
[`research/remote_validation/cfdem_output_cleanup_20261011.md`](../../research/remote_validation/cfdem_output_cleanup_20261011.md).
The raw simulation NPZs and existing GIFs are not rewritten by this operation.
