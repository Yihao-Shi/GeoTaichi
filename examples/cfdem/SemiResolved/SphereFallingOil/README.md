# Oil sphere: semi-resolved incompressible MPM–DEM

## Latest validation status (2026-09-10): solver behaviour repaired; strict workbook comparison not passed

The retained formal run is
[`E4_CA2_wall_lubrication_formal_20260910`](OutputData/E4_CA2_wall_lubrication_formal_20260910/).
It contains the complete 1.25 s result: 51 frames and 153 VTUs, the raw NPZ
frames and trajectory, and the CSV/JSON/PNG comparison with every workbook row.
The five superseded diagnostic outputs were removed.

The DEM force is now evaluated at both Velocity-Verlet force stages of every
DEM substep. This removes the former 1.665 m/s first-contact rebound. The
formal result approaches the bottom smoothly, with only 0.041 micrometres of
penalty-contact overlap and no meaningful rebound. Initial acceleration differs
by 0.40%, peak speed by 2.89%, and the velocity-history RMSE through 1.0 s is
2.39% of the experimental peak.

The unscaled full-window RMSE remains 40.02%, so `metrics.json` deliberately
keeps `passed: false`. This is not an early-flow mismatch: integrating the
workbook velocity from release gives about 124.2 mm of travel, while the
published initial sphere-bottom clearance is 120 mm, and its final duplicate
samples still report downward velocities. Consequently no trajectory can both
obey that geometry and match every late workbook velocity. The example does
not increase the release height, shift time, or tune material properties to
hide this inconsistency.

## Setup and reference

`sphere.py` uses ten Cate et al. (2002), case **E4**: sphere diameter
15 mm, sphere density 1120 kg/m³, oil density 960 kg/m³, dynamic viscosity
0.058 Pa·s, and a 100×100×160 mm tank. The release center is
(50, 50, **127.5**) mm because the published 120 mm dimension is the
 sphere-bottom clearance, also shown in [Liu et al. figure 9](https://arxiv.org/html/2506.09517v1/Geo_sedimentation.png).

```sh
python examples/cfdem/SemiResolved/SphereFallingOil/sphere.py --arch gpu --time 1.25 --save-interval 0.025 --write-vtu
python examples/cfdem/SemiResolved/SphereFallingOil/draw.py
```

The fluid uses semi-implicit incompressible FDM pressure projection,
Schiller–Naumann drag, pressure-gradient coupling, a CA=2 integrated-inertia
closure, and one-grid-cell bottom-wall lubrication. At the default three cells
per diameter, the mesh has 20×20×32 cells and eight material points per cell.

The repaired setup also represents all six tank faces as DEM contact planes;
fluid `SolidCell` boundaries alone never constrain the sphere. The existing
elastic linear contact uses k=2e6 N/m, zero dry friction/damping, and DEM
substeps of at most 1e-5 s. The dry impact energy estimate
delta=v*sqrt(m/k) at v=0.128 m/s is about 4 micrometres (<0.03% of D),
while the coupled fluid timestep remains 2.5e-4 s. This is a contact
nonpenetration approximation, **not** a substitute for hydrodynamic lubrication
or permission to accept a geometrically inconsistent late-time reference.

The supplied workbook is named **`experiment.xls`**, not `experiments.xls`. Its data are vertical velocity in **m/s**, not displacement. `draw.py` compares every experimental row against the unshifted, unscaled computed trajectory, preserving the duplicate final time. It writes a comparison plot, CSV and quantitative errors in the selected output folder. New E4 `metrics.json` requires the **entire experimental time window** and velocity-history RMSE below 10% of the measured peak, in addition to the earlier speed/clearance checks. This is an explicitly chosen engineering validation tolerance, not experimental uncertainty. Older datasets' `passed` fields checked only peak speed, quasi-steady initial acceleration and clearance; those fields must not be read as full transient validation.

Reference: [ten Cate et al., Physics of Fluids 14, 4012–4025 (2002)](https://doi.org/10.1063/1.1512918). The E4 experimental peak ratio is 0.955 times the unbounded reference speed 0.128 m/s. The paper explicitly discusses unresolved near-wall lubrication; agreement cannot be assumed from a peak-speed match.

## Other publications reproducing this oil-sphere benchmark

- [Song and Park (2020), JMSE 8(12), 983](https://doi.org/10.3390/jmse8120983),
  section 3.4: unresolved CFD–DEM reproduces the ten Cate oil cases, including
  E4. Its inertia coefficient CA=2 combines added-mass and Basset-history
  effects approximately; it is not the isolated sphere's classical added-mass
  coefficient 0.5. The authors explicitly omit bottom lubrication.
- [Liu, Jing, Fu and Shi (2025 preprint)](https://arxiv.org/html/2506.09517v1#S5.SS1),
  section 5.1: semi-resolved CFD–DEM tests exactly the E4 material parameters
  and tank geometry, comparing kernel-based and point-cloud mappings. It also
  uses CA=2. Figure 10 captures acceleration and the nearly steady speed but
  shows an abrupt numerical stop, not the experiment's smooth bottom approach.
  A journal record is now verified: [JCP 565 (2026), 115227](https://doi.org/10.1016/j.jcp.2026.115227).
  Publisher-deposited Crossref metadata was created on 24 July 2026 and assigns
  the article to the November 2026 issue. The detailed settings cited here
  were checked in the public 2025 preprint, not the inaccessible final full text.
- [Wang, Teng and Liu (2019), JCP 384, 151–169](https://doi.org/10.1016/j.jcp.2019.01.017):
  the kernel-based semi-resolved method used as a baseline by Liu et al.
  Only its abstract/metadata were verified directly here; detailed E4 settings
  should not be attributed to it without checking its full text.

These references establish that semi-resolved modelling has been used for this
benchmark. They do not justify fitting CA, viscosity, density, or a time shift
to the supplied workbook, nor do they validate a drag-only near-wall transient.
