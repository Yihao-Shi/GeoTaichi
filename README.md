# GeoTaichi

![Github License](https://img.shields.io/github/license/Yihao-Shi/GeoTaichi)          ![Github stars](https://img.shields.io/github/stars/Yihao-Shi/GeoTaichi)          ![Github forks](https://img.shields.io/github/forks/Yihao-Shi/GeoTaichi)         [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) 

[**Quick start**](#quick-start) | [**Capabilities**](#core-capabilities) | [**Coupling formulations**](#featured-coupling-formulations) | [**Examples**](#examples) | [**Documentation**](#documentation) | [**Citation**](#citation) | [**Contact**](#contact-us)

## Brief description

GeoTaichi is an open-source, [Taichi](https://github.com/taichi-dev/taichi)-powered simulation framework for granular materials, deformable solids, incompressible fluids, and coupled soil–water–structure interaction. It combines the material point method (MPM), discrete element method (DEM), level-set DEM (LSDEM), affine body dynamics (ABD), finite element method (FEM), and isogeometric analysis (IGA) with explicit, semi-implicit, and implicit solver routes.

Developed by [Multiscale Geomechanics Lab](https://person.zju.edu.cn/en/nguo), Zhejiang University.

<p align="center">
    <img src="images/GeoTaichi.png" width="90%" height="90%" />
</p>


## Overview

GeoTaichi targets multiscale and multiphysics geomechanics: large-deformation and elastoplastic solids, arbitrarily shaped grains, free-surface flow, saturated porous media, and contact with deformable or nearly rigid structures. Solver modules can be used independently or combined through explicit contact exchange, immersed boundary methods (IBM), or fully coupled incremental potential contact (IPC).

GeoTaichi is an actively developed research project released under GPL-3.0. Taichi provides parallel CPU and GPU execution on supported Windows, Linux, and macOS configurations. Backend, precision, dimensionality, and material support depend on the selected solver; the module READMEs document these restrictions.

## Examples

Have an example to share? Submit a [PR](https://github.com/Yihao-Shi/GeoTaichi/pulls)!

### [Material point method (MPM)](src/mpm/README.md#mpm-theory-log)
#### Explicit MPM

| [Column collapse](examples/mpm/column_collapse/DPmaterial.py) | [Dam break](examples/mpm/column_collapse/NewtonianFluid.py) | [Adaptive MPM strip footing](examples/mpm/Footing/strip_footing_adaptive/README.md) | [Progressive failure process of sensitive clay](examples/mpm/sensitive_clay/sensitive_clay.py)|
| --- | --- | --- | --- |
| <img src="images/soil.gif" alt="Column collapse" width="200" height="125" /> | <img src="images/newtonian.gif" alt="Dam break" width="200" height="132" /> | <img src="images/footing.gif" alt="Adaptive MPM strip footing, L2 grid and particle refinement" width="125" height="140" /> | <img src="images/clay.gif" alt="Clay" width="200" height="75" /> |

#### Semi-implicit Incompressible MPM

| [Flow around a cylinder](examples/cfdem/FullyResolved/IBMFixedCylinder2D/cylinder_flow_ibm_2d.py) | [Lid-driven cavity](examples/mpm/IncompressibleFluid/lid_driven_cavity_2d/lid_driven_cavity_2d.py) | [Taylor–Green vortex](examples/mpm/IncompressibleFluid/taylor_green_vortex_2d/taylor_green_vortex_2d.py) |
| --- | --- | --- |
| <img src="images/cylinder_vorticity.gif" alt="Flow around a cylinder vorticity" width="182" height="85" /> | <img src="images/lid_driven_cavity_n64.gif" alt="Lid-driven cavity pressure" width="187" height="140" /> | <img src="images/taylor_green_vortex.gif" alt="Taylor–Green vortex speed" width="186" height="140" /> |

| [Dam break around a five-point star](examples/mpm/IncompressibleFluid/dam_break_visible_sdf_2d/dam_break_visible_sdf_2d.py) | [3D large-tank dam break](examples/mpm/IncompressibleFluid/large_tank_incompressible_3d/large_tank_incompressible_3d.py) | [3D piston wavemaker](examples/mpm/IncompressibleFluid/wavemaker_tank_3d/wavemaker_tank_3d.py) |
| --- | --- | --- |
| <img src="images/dam_break_star_obstacle_2d.gif" alt="Dam break around a five-point star" width="187" height="140" /> | <img src="images/large_tank_3d.gif" alt="3D large-tank dam break" width="186" height="140" /> | <img src="images/wavemaker_3d.gif" alt="3D piston wavemaker" width="186" height="140" /> |

#### Two-phase Semi-implicit MPM

| [Dam break through a porous column](examples/mmpm/DamBreakPorousElastic2D/double_point_dam_break_porous_elastic.py) | [Submarine granular landslide](examples/mmpm/SubmarineLandslide/submarine_landslide_2d/submarine_landslide_2d.py) | [U-tube flow through a porous bed](examples/mmpm/UTubeFlow2D/u_tube_flow_2d.py) | [Granular-water leakage](examples/mmpm/GranularWaterLeakage3D/granular_water_leakage_3d.py) |
| --- | --- | --- | --- |
| <img src="images/dam_break_porous_elastic_2d.gif" alt="Dam break through a porous column" width="200" height="133" /> | <img src="images/submarine_landslide.gif" alt="Submarine granular landslide" width="200" height="133" /> | <img src="images/u_tube_flow_2d.gif" alt="U-tube flow through a porous bed" width="153" height="140" /> | <img src="images/granular_water_leakage_3d.gif" alt="Granular-water leakage" width="187" height="140" /> |

### [Discrete element method (DEM)](src/dem/README.md#dem-theory-log)
#### Explicit Soft Constraint

| [Granular packing](examples/dem/LevelSet/GranularAssemble/polydisperse/packing_generate.py) | [Screw and nut](examples/dem/LevelSet/ParticleParticle/screw_and_nut.py) | [Debris flow](examples/dem/LevelSet/DebrisFlow/) |
| --- | --- | --- |
| <img src="images/lsdem.gif" alt="Granular packing" width="200" height="116" /> | <img src="images/screw_nut.gif" alt="Screw and nut" width="200" height="99" /> | <img src="images/debris_flow.gif" alt="Debris Flow" width="200" height="119" /> |

| [Rotating drum](examples/dem/LevelSet/RotatingDrum/rotating_drum.py) | [Triaxial shear test](examples/dem/LevelSet/TriaxialShear/) |
| --- | --- |
| <img src="images/drums.gif" alt="Rotating drum" width="200" height="103" /> | <img src="images/force_chain.gif" alt="Triaxial shear test" width="200" height="125" /> |

#### Implicit Affine Body Dynamics (ABD)

| [Cubes on an incline](examples/dem/AffineBody/five_cubes_incline/five_cubes_incline.py) | [Cube impact](examples/dem/AffineBody/thousand_cubes_heavy_ball_impact/thousand_cubes_heavy_ball_impact.py) | [Newton's cradle](examples/dem/AffineBody/newtons_cradle/newtons_cradle.py) |
| --- | --- | --- |
| <img src="images/abd_incline.gif" alt="Five cubes on an incline: sliding and static friction" width="187" height="140" />| <img src="images/abd_cubes.gif" alt="Cube impact" width="191" height="140" />| <img src="images/abd_cradle.gif" alt="Newton&#x27;s cradle: first 5 impacts" width="191" height="140" />|

### [FEM](src/fem/README.md#fem-theory-log) / [IGA](src/iga/README.md#iga-theory-log)

| [IGA elastic bar](examples/iga/explicit_cantilever_3d/explicit_cantilever_3d.py) | [IGA twisting](examples/iga/twisting_column_3d/twisting_column_3d.py) | [Cloth twisting with self-contact](examples/fem/cloth/implicit_cloth_twist/implicit_cloth_twist.py) |
| --- | --- | --- |
| <img src="images/iga_elastic_bar.gif" alt="Implicit IGA elastic bar: von Mises stress" width="200" height="133" /> | <img src="images/iga_twisting.gif" alt="IGA twisting bar: von Mises stress" width="200" height="133" /> | <img src="images/fem_cloth_twisting.gif" alt="FEM cloth twisting with self-contact" width="200" height="133" /> |

### [FEM-MPM](src/fempm/README.md#fem--mpm-coupling-theory-log) / [IGA-MPM](src/igampm/README.md#iga--mpm-coupling-theory-log)

#### Implicit Incremental Potential Contact

| Flexible barrier ([FEM–MPM](examples/fempm/flexible_barrier/flexible_barrier.py) / [IGA–MPM](examples/igampm/flexible_barrier/flexible_barrier.py)) | Wavy plate collapse ([FEM–MPM](examples/fempm/wavy_plate_collapse/wavy_plate_collapse.py) / [IGA–MPM](examples/igampm/wavy_plate_collapse/wavy_plate_collapse.py)) | Axisymmetric CPT ([FEM–MPM](examples/fempm/cpt_dp/cpt_dp.py) / [IGA–MPM](examples/igampm/cpt_dp/cpt_dp.py)) |
| --- | --- | --- |
| <img src="images/flexible_barrier.gif" alt="IGA–MPM flexible barrier" width="200" height="134" /> | <img src="images/wavy_plate_collapse.gif" alt="IGA–MPM wavy plate collapse" width="168" height="140" /> | <img src="images/cpt_dp.gif" alt="IGA–MPM axisymmetric CPT" width="93" height="140" /> |

### [FEM-DEM](src/fedem/README.md#fedem-coupling-theory-log)
#### Explicit Soft Constraint

| [Hertz contact](examples/fedem/HertzContact/hertz_contact.py) | [Mixed funnel](examples/fedem/MixedFunnel/mixed_funnel.py) | [Isotropic compaction](examples/fedem/IsotropicCompaction/isotropic_compaction.py) |
| --- | --- | --- |
| <img src="images/hertz_contact.png" alt="Hertz contact" width="200" height="116" /> | <img src="images/mixed_funnel.gif" alt="Mixed funnel" width="153" height="140" /> | <img src="images/isotropic_compaction_100.gif" alt="Isotropic compaction" width="187" height="140" /> |

### [MPM-DEM](src/mpdem/README.md#mpdem-theory-log)
#### Explicit MPM-DEM

| [A sphere impacting granular bed](examples/mpdem/MultiSphere/SphereImpactToGranularBed/plane_strain.py) | [Granular column impacting cubic particles](examples/mpdem/MultiSphere/GranularImpact/granular_impact.py) | [Box sinking into water](examples/mpdem/LevelSet/WaterImpact/box.py) | [Stone column axial loading](examples/mpdem/MultiSphere/StoneColumn/stone_column.py) |
| --- | --- | --- | --- |
| <img src="images/mpdem1.gif" alt="A sphere impacting granular bed" width="200" height="132" /> | <img src="images/mpdem2.gif" alt="Granular column impacting cubic particles" width="200" height="132" /> | <img src="images/box_sinking.gif" alt="Box sinking into water" width="200" height="137" /> | <img src="images/stone_column_9d.png" alt="9D stone column: half-clipped MPM soil colored by smoothed stress zz, complete grey DEM column; c=20 kPa, E=4 MPa" width="200" height="229" /> |

#### Semi-implicit MPM with Immersed Boundary Method

| [Drafting, kissing and tumbling](examples/cfdem/FullyResolved/IBMDraftingKissingTumbling/drafting_kissing_tumbling.py) | [Bunny impacting a saturated granular bed](examples/mmpm/TwoPhaseLSDEMCoupling/sphere_impact_submerged_bed_3d/sphere_impact_submerged_bed_3d.py) | [IBM dam break](examples/cfdem/FullyResolved/IBMLevelSetDamBreak3D/dam_break_levelset_ibm_3d.py) | [Wavemaker with LSDEM grains and a branched trunk](examples/mmpm/TwoPhaseLSDEMCoupling/wavemaker_lsdem_particles_3d/wavemaker_lsdem_particles_3d.py) |
| --- | --- | --- | --- |
| <img src="images/dkt.gif" alt="Drafting, kissing and tumbling" width="93" height="140" /> | <img src="images/bunny_impact_saturated_soil.gif" alt="Bunny impacting a saturated granular bed" width="187" height="140" /> | <img src="images/ibm_break.gif" alt="IBM dam break" width="186" height="140" /> | <img src="images/two_phase_lsdem_wavemaker.gif" alt="Two-phase wavemaker with LSDEM grains and a branched trunk" width="186" height="140" /> |

#### Implicit Incremental Potential Contact

Implemented routes include [Solid MPM–ABD impact](examples/mpdem/AffineBody/ABDImpactDP/direct_mpm_abd_impact.py). 

## Core capabilities

- [MPM](src/mpm/README.md#mpm-theory-log): explicit and implicit solid mechanics with sparse grid and adaptive refinement, semi-implicit incompressible flow, and two-phase porous-media formulations.
- [DEM, LSDEM, and ABD](src/dem/README.md#dem-theory-log): spheres and clumps, signed-distance representations of irregular rigid particles, granular contact, and implicit affine-body mechanics with IPC.
- [FEM](src/fem/README.md#fem-theory-log) and [IGA](src/iga/README.md#iga-theory-log): volume, membrane, and cloth FEM; NURBS-based IGA; explicit and implicit mechanics with formulation-specific material and contact support.
- [MPM–DEM](src/mpdem/README.md#mpdem-theory-log): explicit continuum–grain contact and semi-resolved incompressible fluid–sphere coupling.
- [MPM–LSDEM](src/mpdem/README.md#mpdem-theory-log): explicit level-set contact, fully resolved incompressible IBM, and hybrid two-phase solid–fluid–grain coupling.
- [MPM–ABD](src/mpdem/README.md#mpm-affinebody-route-selection): solid and hyperelastic soft-particle IPC.
- [FEM–DEM](src/fedem/README.md#fedem-coupling-theory-log): explicit grain–deformable-surface contact. 
- [FEM–LSDEM](src/fedem/README.md#3-fem--lsdem-signed-distance-coupling): deforming FEM boundaries interact with rigid-particle signed-distance geometry.
- [FEM–ABD](src/fedem/README.md#4-fully-coupled-affinebody--fem-barrier-ipc): fully coupled IPC for volume, membrane, and cloth structures.
- [FEM–MPM](src/fempm/README.md#fem--mpm-coupling-theory-log): explicit contact exchange and fully coupled implicit IPC with elastic or plastic MPM.
- [IGA–MPM](src/igampm/README.md#iga--mpm-coupling-theory-log): independent NURBS structures and MPM continua coupled through explicit contact or implicit point–NURBS IPC.
- [Verification and tests](tests/README.md) distinguish unit/integration checks from numerical comparisons against analytical or published references. Each solver README links its example-backed features to the relevant theory log.

## Novel coupling formulations

GeoTaichi introduces three new coupling formulations: MPM–ABD IPC, incompressible MPM–DEM coupling (fully resolved LSDEM volume-fraction IBM and semi-resolved sphere coupling), and IGA–MPM point–NURBS contact coupling. These contributions concern the coupling operators and formulations built on established MPM, ABD, DEM, LSDEM, IGA, IBM, and IPC methods. Their derivations, implementation details, and concrete examples are documented in the linked theory logs below.

| Formulation | Method and scope | Theory |
| --- | --- | --- |
| Incompressible material point method–DEM coupling | Fully resolved 3D MPM–LSDEM uses rigid-body SDFs to estimate cell solid fractions and apply volume-fraction IBM forcing. Semi-resolved 3D MPM–sphere DEM uses normalized Gaussian volume mapping, porosity-aware pressure projection, and drag/pressure-force exchange. | [LSDEM volume-fraction IBM](src/mpdem/README.md#10-fully-resolved-lsdem-and-affinebody-volume-fraction-ibm); [Semi-resolved sphere coupling](src/mpdem/README.md#9-semi-resolved-incompressible-sphere-coupling) |
| Isogeometric analysis–material point method coupling (IGA–MPM) | Independent deformable NURBS IGA structures and MPM continua interact through point–NURBS contact. The linked 3D examples demonstrate explicit DEM-law exchange and fully coupled implicit IPC with elastic or plastic MPM blocks. | [IGA–MPM theory log](src/igampm/README.md#iga--mpm-coupling-theory-log) |
| Material point method–affine body dynamics coupling (MPM–ABD) | Fully coupled solid MPM–ABD IPC couples active MPM-grid displacements and affine-body controls, including supported finite-strain plasticity. This route is currently 3D with lagged friction. Hyperelastic soft-MPM–ABD IPC and incompressible fluid–ABD IBM are separate routes. | [MPM–ABD derivation](src/mpdem/README.md#15-solid-mpm--affinebody-ipc-and-plasticity) |

## Quick start

### Install from source

GeoTaichi is distributed through this source repository. There is no supported PyPI installation workflow. Use Python 3.10 or newer; the dependency declarations are maintained in [pyproject.toml](pyproject.toml).

Clone the repository and create a Python environment:

```bash
git clone https://github.com/Yihao-Shi/GeoTaichi.git
cd GeoTaichi
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the local checkout, not a GeoTaichi package from PyPI:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

The editable install makes `geotaichi` importable without manually setting `PYTHONPATH`. Some geometry dependencies may need platform-specific system libraries. Select a backend supported by your hardware and solver; not all implicit GPU routes support the same precision or devices.

### Run an example

Run commands from the repository root. For a small, CPU-selectable coupling example:

```bash
python examples/mpdem/AffineBody/ABDImpactDP/direct_mpm_abd_impact.py --arch cpu --steps 10
```

For the explicit column-collapse example:

```bash
python examples/mpm/column_collapse/DPmaterial.py
```

The latter uses the default GPU initialization. If using CPU, change its initialization to `init(arch="cpu", default_fp="float64")`; `arch` must be passed by name, not as `init('cpu')`. Check each example's configuration, input assets, and output directory before running a full simulation.

### Visualization

Simulation outputs can be inspected in ParaView by opening a `.vtu` or `.vts` time series, selecting **Apply**, and choosing the representation and field to display. [Developer tools](tools/README.md) describe the particle-field and Blender rendering workflows used for the gallery. Rendered surfaces are visualization reconstructions, not a replacement for the underlying simulation data.

## Documentation

Documentation is maintained alongside the source. Module READMEs describe supported formulations, theory logs, configuration, workflows, and limitations:

- Core solvers: [MPM](src/mpm/README.md#mpm-theory-log), [DEM](src/dem/README.md#dem-theory-log), [FEM](src/fem/README.md#fem-theory-log), and [IGA](src/iga/README.md#iga-theory-log).
- Coupled solvers: [MPM–DEM](src/mpdem/README.md#mpdem-theory-log), [FEM–DEM](src/fedem/README.md#fedem-coupling-theory-log), [FEM–MPM](src/fempm/README.md#fem--mpm-coupling-theory-log), and [IGA–MPM](src/igampm/README.md#iga--mpm-coupling-theory-log).
- Shared models and numerics: [Constitutive models](src/physics_model/consititutive_model/README.md), [Contact models](src/physics_model/contact_model/README.md), [Contact detection](src/contact_detection/README.md), and [Linear solvers](src/linear_solver/README.md).
- Usage and development: [Python examples](examples/), [Solver integration principles](agent/SOLVER_INTEGRATION_PRINCIPLES.md), [Blender add-on](blender/README.md), [Developer tools](tools/README.md), and [Tests and verification](tests/README.md).

Start with the README for your solver and a matching Python example. Additional derivations and technical notes are available in [docs](docs/helper/geotaichi_user_theory_manual.pdf).

## License
This project is licensed under the GNU General Public License v3 - see the [LICENSE](https://www.gnu.org/licenses/) for details.

## Citation
If GeoTaichi supports your research, please consider starring the repository and citing the relevant publications.

The publications below describe the original framework and GPU level-set DEM. For newer coupling formulations, consult the linked theory logs, examples, and source revision; these papers should not be read as publications of every feature currently in the repository.

If you publish work that makes use of GeoTaichi, please cite the relevant references:
```latex
@article{shi2024geotaichi,
  title={GeoTaichi: A Taichi-powered high-performance numerical simulator for multiscale geophysical problems},
  author={Shi, YH and Guo, N and Yang, ZX},
  journal={Computer Physics Communications},
  volume={301},
  pages={109219},
  year={2024},
  publisher={Elsevier}
}
@article{shi2025gpu,
  title={GPU-accelerated level-set DEM for arbitrarily shaped particles with broad size distributions},
  author={Shi, YH and Guo, N and Yang, ZX},
  journal={Powder Technology},
  pages={121293},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgements
We thank all amazing contributors for their great work and open source spirit. We welcome all kinds of contributions to file an issue at [GitHub Issues](https://github.com/Yihao-Shi/GeoTaichi/issues).

### Contributors
<a href="https://github.com/Yihao-Shi/GeoTaichi/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Yihao-Shi/GeoTaichi" />
</a>

### Contact us
- If you spot any issue or need any help, please mail directly to <a href = "mailto:shiyh@zju.edu.cn">shiyh@zju.edu.cn</a>.

## Release Notes
V0.5.0 (Oct 2, 2026)

- Please click [here](https://github.com/Yihao-Shi/GeoTaichi/releases/tag/GeoTaichi-v0.5) for more details

V0.4.0 (Aug 27, 2025)

- Please click [here](https://github.com/Yihao-Shi/GeoTaichi/releases/tag/GeoTaichi-v0.4) for more details

V0.3.0 (December 12, 2024)

- Please click [here](https://github.com/Yihao-Shi/GeoTaichi/releases/tag/GeoTaichi-v0.3) for more details

V0.2.2 (July 22, 2024)

- Fix computing the intersection area between circles and triangles
- Add "Destory" and "Reflect" boundaries in DEM modules, see the [v0.2.2 release](https://github.com/Yihao-Shi/GeoTaichi/releases/tag/GeoTaichi-v0.2.2)

V0.2 (July 1, 2024)

- Fix some bugs in DEM and MPM modules, see [details](https://github.com/Yihao-Shi/GeoTaichi/releases/tag/GeoTaichi-v0.2)
- Add some advanced constitutive model

V0.1 (January 21, 2024)

- First release GeoTaichi
