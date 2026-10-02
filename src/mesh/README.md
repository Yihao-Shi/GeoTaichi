# Structured Mesh Topology and Numerical Quadrature for Computational Mechanics

`src/mesh` contains structured mesh generators, connectivity utilities, and
Gaussian quadrature tables shared by MPM, FEM, level-set, and postprocessing
code. It is a lower-level utility package; solver-specific mesh normalization
remains in each solver module.

## Capabilities

- Structured quadrilateral and hexahedral meshes.
- Simplex quadrature tables and an experimental tetrahedral-mesh wrapper.
- Rectangle and box mesh generation.
- Optional Gmsh-based box and cylinder generation.
- Cell-to-node connectivity, face centers, face-to-cell maps, and
  boundary/internal face tags.
- Triangle, tetrahedron, rectangle, and tensor-product Gaussian quadrature.

## Main classes

| Class/file | Purpose |
| --- | --- |
| `QuadrilateralMesh` | Structured 2D cells and face topology |
| `HexahedronMesh` | Structured 3D cells and face topology |
| `TetraMesh` | Experimental tetrahedral meshing wrapper |
| `GaussPointInTriangle` | Triangle and tetrahedron quadrature rules |
| `GaussPointInRectangle` | Tensor-product Gauss-Legendre rules |
| `GenerateMesh.py` | Procedural and Gmsh-backed mesh generation |

## Structured-grid geometry and topology

For cell counts $n_k$, spacings $h_k$, and $g$ ghost layers, structured
nodes lie on the Cartesian lattice

$$
x_{\boldsymbol{i},k}=(i_k-g)h_k,
\qquad
0\leq i_k\leq n_k.
$$

Each interior face is incident to two cells and each domain-boundary face to
one. For a face with vertices $\boldsymbol{x}_{a}$, its geometric center is

$$
\boldsymbol{x}_f=\frac{1}{n_f}\sum_{a=1}^{n_f}\boldsymbol{x}_{a}.
$$

On a tensor-product parent cell $[-1,1]^d$, associate each corner $a$ with
signs $s_{a,k}\in\{-1,1\}$. The multilinear corner basis is

$$
N_a(\boldsymbol{\xi})
=2^{-d}\prod_{k=1}^{d}(1+s_{a,k}\xi_k),
\qquad
\sum_aN_a=1.
$$

The physical map and its Jacobian are

$$
\boldsymbol{x}(\boldsymbol{\xi})
=\sum_aN_a(\boldsymbol{\xi})\boldsymbol{x}_a,
\qquad
\boldsymbol{J}
=\frac{\partial\boldsymbol{x}}{\partial\boldsymbol{\xi}}
=\sum_a\boldsymbol{x}_a\otimes\nabla_{\xi}N_a.
$$

These identities define the geometry behind the quadrilateral and
hexahedral connectivity; solver packages attach their own field variables and
shape-gradient conventions.

## Gaussian quadrature

The one-dimensional Gauss--Legendre rule with nodes $\xi_q$ and weights
$w_q$ approximates

$$
\int_{-1}^{1}f(\xi)\,\mathrm d\xi
\approx\sum_{q=1}^{n}w_qf(\xi_q).
$$

Its tensor-product extension is

$$
\int_{[-1,1]^d}f(\boldsymbol{\xi})\,\mathrm d\boldsymbol{\xi}
\approx
\sum_{q_1=1}^{n_1}\cdots\sum_{q_d=1}^{n_d}
\left(\prod_{k=1}^{d}w_{q_k}\right)
f(\xi_{q_1},\ldots,\xi_{q_d}).
$$

After an isoparametric map, physical integration includes the Jacobian:

$$
\int_{\Omega_e}f(\boldsymbol{x})\,\mathrm d\boldsymbol{x}
\approx
\sum_q w_q f(\boldsymbol{x}(\boldsymbol{\xi}_q))
\left|\det\boldsymbol{J}(\boldsymbol{\xi}_q)\right|.
$$

For triangle and tetrahedron rules, the stored simplex weights are normalized
to sum to one. If $|S_e|$ is the physical area or volume,

$$
\int_{S_e}f(\boldsymbol{x})\,\mathrm d\boldsymbol{x}
\approx
|S_e|\sum_qw_qf(\boldsymbol{x}(\boldsymbol{\lambda}_q)),
\qquad
\sum_qw_q=1,
$$

where $\boldsymbol{\lambda}_q$ are barycentric coordinates, with the final
coordinate supplied by
$\lambda_0=1-\sum_{k=1}^{d}\lambda_k$.

## Examples

```python
from src.mesh.QuadMesh import QuadrilateralMesh
from src.mesh.GaussPoint import GaussPointInRectangle

mesh = QuadrilateralMesh(
    nx=20,
    ny=10,
    dx=0.05,
    dy=0.05,
    ghost_cell=1,
)
print(mesh.node_connectivity.shape)
print(mesh.face_map["x"]["tags"])

quadrature = GaussPointInRectangle(gauss_point=(2, 2), dimemsion=2)
quadrature.create_gauss_point()
```

The parameter name `dimemsion` is retained in the quadrature API for backward
compatibility.

## Solver-specific meshes

- General FEM mesh import and procedural geometry are implemented in
  `src/fem/generator`.
- MPM background elements and adaptive grids are implemented in
  `src/mpm/elements`.
- NURBS patch topology is implemented in `src/nurbs` and `src/iga/generator`.

Use those higher-level packages when creating a solver scene; use `src/mesh`
when working directly with structured connectivity or integration rules.

## Tests

Quadrature tests are under `tests/unit/mesh/`. Structured mesh behavior is
also exercised by MPM and level-set tests.
