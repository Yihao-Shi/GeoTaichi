# GeoTaichi

![Github License](https://img.shields.io/github/license/Yihao-Shi/GeoTaichi)          ![Github stars](https://img.shields.io/github/stars/Yihao-Shi/GeoTaichi)          ![Github forks](https://img.shields.io/github/forks/Yihao-Shi/GeoTaichi)         [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) 

[**Quick start**](#quick-start) | [**Examples**](#examples) | [**Paper**](https://www.researchgate.net/publication/380048019_GeoTaichi_A_Taichi-powered_high-performance_numerical_simulator_for_multiscale_geophysical_problems) | [**Citation**](#citation) | [**Contact**](#acknowledgements)

## Brief description

A [Taichi](https://github.com/taichi-dev/taichi)-based numerical package for high-performance simulations of multiscale and multiphysics geophysical problems. 
Developed by [Multiscale Geomechanics Lab](https://person.zju.edu.cn/en/nguo), Zhejiang University.

<p align="center">
    <img src="https://github.com/Yihao-Shi/GeoTaichi/blob/main/images/GeoTaichi.png" width="90%" height="90%" />
</p>


## Overview

GeoTaichi is a collection of several numerical tools, currently including __Discrete Element Method (DEM)__, __Material Point Method (MPM)__, __Material Point-Discrete element method (MPDEM)__, and __Finite Element Method (FEM)__, that cover the analysis of the __Soil-Gravel-Structure-Interaction__ in geotechnical engineering. The main components of GeoTaichi is illustrated as follows:
<p align="center">
    <img src="https://github.com/Yihao-Shi/GeoTaichi/blob/main/images/main_component.png" width="50%" height="50%" />
</p>

GeoTaichi is a research project that is currently __under development__. Our vision is to share with the geotechnical community a free, open-source (under the GPL-3.0 License) software that facilitates the relevant computational research. In the Taichi ecosystem, we hope to emphasize the potential of Taichi for scientific computing. Furthermore, GeoTaichi is high parallelized, multi-platform (supporting for Windows, Linux and Macs) and multi-architecture (supporting for both CPU and GPU).

## Examples

Have a cool example? Submit a [PR](https://github.com/Yihao-Shi/GeoTaichi/pulls)!

### Material point method (MPM)
| [Column collapse](example/mpm/ColumnCollapse/DPmaterial.py) | [Dam break](example/mpm/ColumnCollapse/NewtonianFluid.py) | [Strip footing](example/mpm/Footing/StripFootingTresca.py) | [Progressive failure process of sensitive clay](example/mpm/ColumnCollapse/SoftDP.py) |
| --- | --- | --- | --- |
| ![Column collapse](images/soil.gif) | ![Dam break](images/newtonian.gif) | ![Strip footing](images/footing.gif) | ![Clay](images/clay.gif) |

### Discrete element method (DEM)
| [Granular packing](example/dem/GranularPackings/polyLevelSet/packing_generate.py) | [Screw and nut](example/dem/ParticleSliding/screw_and_nut.py) | [Debris Flow](example/dem/DebrisFlow) | 
| --- | --- | --- | 
| ![Granular packing](images/lsdem.gif) | ![Screw and nut](images/screw_nut.gif) | ![Debris Flow](images/debris_flow.gif) | 

|[Rotating drum](example/dem/RotatingDrums) | [Triaxial shear test](example/dem/TriaxialTest) |
| --- | --- | 
| ![Rotating drum](images/drums.gif) | ![Triaxial shear test](images/force_chain.gif) |

### Coupled material point-discrete element method (MPDEM)
| [A sphere impacting granular bed](example/dempm/SphereImpact/plane_strain.py) | [Granular column impacting cubic particles](example/dempm/GranularImpact/granular_impact.py) | [Box sinking into water](example/dempm/BoxSinking/box.py) |
| --- | --- | --- |
| ![A sphere impacting granular bed](images/mpdem1.gif) | ![Granular column impacting cubic particles](images/mpdem2.gif) | ![Box sinking into water](images/box_sinking.gif) |

## Quick start
### Installation
#### Install from source code (recommand)
##### Ubuntu
1. Change the current working directory to the desired location and download the GeoTaichi code:
```
cd /path/to/desired/location/
git clone https://github.com/Yihao-Shi/GeoTaichi
cd GeoTaichi
```
2. Install essential dependencies
```
# Install python and pip
sudo apt-get install python3.8
sudo apt-get install python3-pip

# Install python packages (recommand to add package version)
bash requirements.sh
```
3. Install CUDA, detailed information can be referred to [official installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
4. Set up environment variables
```
sudo gedit ~/.bashrc
$ export PYTHONPATH="$PYTHONPATH:/path/to/desired/location/GeoTaichi"
source ~/.bashrc
```
##### Windows
1. Install Anaconda 
2. start Anaconda Prompt 
3. Navigate to a folder where geotaichi_env.yml is located. 
4. clone geotaichi as:
```
git clone https://github.com/Yihao-Shi/GeoTaichi 
```
5. run command:
```
conda env create -f geotaichi_env.yml 
```
6. run command:
```
conda activate geotaichi 
```
7. correct the environment (the last part should be modified to the path of geotaichi):
```
conda env config vars set PYTHONPATH=%PYTHONPATH%;.\path\to\GeoTaichi 
```
8. run command:
```
conda activate geotaichi 
```
9. run a benchmark (column collapse):
```
python DPmaterial 
```
Remark: line 3 of the examples should be modified based on the availability of the GPU. If CPU is available, the following should be used;
```
init('cpu')
```
#### Install from pip (easy)
```
pip install geotaichi
```

### Working with vtu files

To visualize the VTS files produced by some of the scripts, it is recommended to use [ParaView](http://www.paraview.org/). To visualize the output in ParaView, use the following
procedure:
1. Open the .vts or .vtu file in ParaView
2. Click on the "Apply" button on the left side of the screen
3. Make sure under "Representation" that "Surface" or "Surface with Edges" is selected
4. Under "Coloring" select variables and the approriate measure (i.e. "Magnitude", X-direction displacement, etc.)

### Document

Currently, only the tutorial of DEM in Chinese version is available in [doc](https://github.com/Yihao-Shi/GeoTaichi/blob/main/docs/GeoTaichi_tutorial_DEM_Chinese_version.pdf). 
Users can set up simulations by specifying numerical parameters and configuring the desired simulation settings in a Python script. More detailed about Python scripts can be found in the [example floder](https://github.com/Yihao-Shi/GeoTaichi/tree/main/example).

## Features
### Discrete Element Method 
Discrete element method is a powerful tool to simulate the movement of granular materials through a series of calculations that trace individual particles constituting the granular material.
  - Sphere, multisphere particles and level-set DEM
  - Unified approach for creating level-set functions for irregularly shaped particle
  - Generating particle packings by specifying initial void ratio or particle number in a box/cylinder/sphere/triangular prism
  - Three neighbor search algorithms, brust search/linked-cell/multilevel linked-cell
  - Two velocity updating schemes, symlectic Euler/velocity Verlet
  - Four contact models, including linear elastic, hertz-mindlin, linear rolling and energy conserving model
  - Supporting plane (infinite plane)/facet (servo wall)/triangle patch (suitable for complex boundary condition)
  - Supporting [periodic boundary](example/dem/PeriodicBoundary) for sphere particles

### Material Point Method 
The material point method (MPM) is a numerical technique used to simulate the behavior of solids, liquids, gases, and any other continuum material. Unlike other mesh-based methods like the finite element method, MPM does not encounter the drawbacks of mesh-based methods (high deformation tangling, advection errors etc.) which makes it a promising and powerful tool in computational mechanics. 
  - Nine Constitutive Models, including linear elastic/neo-hookean/Von-Mises/isotropic hardening plastic/(state-dependent) Mohr-Coulomb/Drucker-Prager/(cohesive) modified cam-clay/Newtonian fluid/Bingham fluid
  - Two improved velocity projection techniques, including TPIC/APIC/MLS
  - Three stress update schemes, including USF/USL/MUSL
  - Three stabilization techniques, including mix integration/B-bar method/F-bar method
  - Two smoothing mehod, including strain/pressure smoothing
  - Supporting Dirichlet (Fix/Reflect/Friction)/Neumann boundary conditions
  - Supporting total/updating Lagrangian explicit MPM 
  - Free surface detection
  - Supporting input [external CAD files](example/mpm/ExternalOBJ)

### MPDEM coupling
  - Two contact models, including linear elastic, hertz-mindlin, Energy conserving model (Barrier functions)
  - Support DEM-MPM-Mesh contact, feasible simulating complex boundary conditions 
  - Multilevel neighbor search
  - Two way or one way coupling

### Postprocessing
  - Restart from a specific time step
  - A simple GUI powered by [Taichi](https://github.com/taichi-dev/taichi)
  - VTU([Paraview](http://www.paraview.org/)) and NPZ(binary files) files are generated in the process of simualtion
  - Supporting force chain visualization

## Under development
  - Developing a well-structured IGA modules
  
## License
This project is licensed under the GNU General Public License v3 - see the [LICENSE](https://www.gnu.org/licenses/) for details.

## Citation
Please kindly star :star: this project if it helps you. We take great efforts to develope and maintain it :grin::grin:.

If you publish work that makes use of GeoTaichi, we would appreciate if you would cite the following reference:
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
- If you spot any issues or need help, please find our contact information on the lab website.

## Release Notes
V0.4.0 (Aug 27, 2025)

- Please click [here](https://github.com/Yihao-Shi/GeoTaichi/releases/tag/GeoTaichi-v0.4) for more details

V0.3.0 (December 12, 2024)

- Please click [here](https://github.com/Yihao-Shi/GeoTaichi/releases/tag/GeoTaichi-v0.3) for more details

V0.2.2 (July 22, 2024)

- Fix computing the intersection area between circles and triangles
- Add "Destory" and "Reflect" boundaries in DEM modules, see [examples](https://github.com/Yihao-Shi/GeoTaichi/blob/main/example/dem/SimpleChute/simple_chute.py)

V0.2 (July 1, 2024)

- Fix some bugs in DEM and MPM modules, see [details](https://github.com/Yihao-Shi/GeoTaichi/releases/tag/GeoTaichi-v0.2)
- Add some advanced constitutive model

V0.1 (January 21, 2024)

- First release GeoTaichi
