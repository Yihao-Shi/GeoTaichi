# Adopted CFDEM results

The adopted results for production reruns and gallery rendering are listed below.
`particles`, `grids`, `vtks`, and contact/wall data inside that result are parts
of one simulation, not duplicate simulations.

| Example | Adopted result, relative to this directory | Acceptance evidence |
| --- | --- | --- |
| Drafting–kissing–tumbling | [`FullyResolved/IBMDraftingKissingTumbling/OutputData/remote_20261019/matching_and_sphere_queue_20261019__ibmdraftingkissingtumbling`](FullyResolved/IBMDraftingKissingTumbling/OutputData/remote_20261019/matching_and_sphere_queue_20261019__ibmdraftingkissingtumbling/) | `metrics.json`: passed; 72 × 72 × 288 cells, Taylor projection, alphaPIC=0, CUDA/f64, complete 0.6 s run |
| Fully resolved sphere settling in oil | [`FullyResolved/IBMResolvedSphereSettling/OutputData/remote_20261019/sphere_grid16_cpu_retry_20261019__resolved_sphere_grid16_cpu`](FullyResolved/IBMResolvedSphereSettling/OutputData/remote_20261019/sphere_grid16_cpu_retry_20261019__resolved_sphere_grid16_cpu/) | `metrics.json`: passed; 108 × 108 × 172 cells, alphaPIC=0.01, CPU/f64, complete 0.8 s run |
| IBM level-set dam break | [`FullyResolved/IBMLevelSetDamBreak3D/OutputData/dam3d_highres_sdf_vf/output`](FullyResolved/IBMLevelSetDamBreak3D/OutputData/dam3d_highres_sdf_vf/output/) | `metrics.json` and `validation_connectivity_metrics.json`: passed, complete 1.2 s run; enclosed air explicitly allowed |
| Fixed-cylinder flow | [`FullyResolved/IBMFixedCylinder2D/OutputData/ibm2d_karman_postfix_20260907/model`](FullyResolved/IBMFixedCylinder2D/OutputData/ibm2d_karman_postfix_20260907/model/) | Historical gallery source; passed the older acceptance rules, not the current stricter rules |
| Semi-resolved sphere falling in oil | [`SemiResolved/SphereFallingOil/OutputData/E4_CA2_wall_lubrication_formal_20260910`](SemiResolved/SphereFallingOil/OutputData/E4_CA2_wall_lubrication_formal_20260910/) | `acceptance_pre_wall_20261003.json`: accepted on the pre-wall interval; historical full-window `metrics.json` is unchanged |

IBM production runs reproduce each adopted result's `configuration.json`,
`metrics.json`, and `rerun_execution.json`, including the backend, time step,
PIC coefficient, projection scheme, geometry, and physical acceptance gates.
The adopted DKT run explicitly requests 12 cells per diameter (effective 12.024),
alphaPIC=0, dt=5e-5 s, and DEM dt=1e-7 s. The adopted sphere run explicitly
requests 16 cells per diameter (effective 16.2), alphaPIC=0.01, and dt=2.5e-4 s.
The example defaults match these adopted run settings; the sphere defaults to CPU.
Production queues still record the parameters explicitly. The adopted sphere evidence is CPU/f64;
the finer grid has not established equivalent GPU memory use or GPU acceptance.

The fixed cylinder uses volume-fraction IBM, retaining fictitious fluid inside
the body. Its geometry callback does not write the cut-cell SDF; that SDF belongs
to the channel walls. Representing the cylinder by both methods removes interior
particles, leaves air cells where IBM needs fluid, and suppresses its forcing.
Negative analytic cylinder distances therefore identify expected fictitious
fluid; `particles_inside_sdf` still checks penetration of the channel walls.
The no-slip and maximum-speed limits remain unchanged. The historical September
gallery result used looser gates and is not acceptance evidence for a repaired run.

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
