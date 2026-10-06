# Signed-Distance Geometry for Implicit Surfaces, Boolean Modeling, and Contact

`src/sdf` provides analytic and sampled signed-distance geometry used by DEM
templates, level-set bodies, mesh generation, contact, and cloth SDF energies.
Many primitives are re-exported directly from `geotaichi`.

## Capabilities

- Analytic 2D primitives such as circles, rectangles, lines, polygons, and
  text/image-derived shapes.
- Analytic 3D primitives such as spheres, boxes, cylinders, cones, tori,
  capsules, polyhedra, and superquadrics.
- Boolean union, intersection, difference, negation, blend, shell, dilation,
  and erosion.
- Translation, scale, rotation, repetition, twist, bend, and transition
  transforms.
- 2D-to-3D extrusion and revolution.
- Mesh/polyhedron-backed distance queries and sampled level-set grids.
- Surface-node generation and fast-marching support.

## Signed-distance definition and differential geometry

For a closed set $`\Omega`$ with boundary $`\partial\Omega`$, the signed distance
used by closed primitives is

```math
d(\boldsymbol{x})=
\begin{cases}
-\mathrm{dist}(\boldsymbol{x},\partial\Omega),
&\boldsymbol{x}\in\Omega,\\[0pt]
\mathrm{dist}(\boldsymbol{x},\partial\Omega),
&\boldsymbol{x}\notin\Omega.
\end{cases}
```

Where the closest point is unique, an exact signed-distance field satisfies

```math
\|\nabla d\|=1,
\qquad
\boldsymbol{n}=\nabla d,
\qquad
\kappa=\nabla\cdot\boldsymbol{n}.
```

For a general implicit field whose gradient is not unit length, use
$`\boldsymbol{n}=\nabla d/\|\nabla d\|`$. The closest-point approximation is

```math
\boldsymbol{x}_{\Gamma}
\approx\boldsymbol{x}-
\frac{d(\boldsymbol{x})}{\|\nabla d(\boldsymbol{x})\|^2}
\nabla d(\boldsymbol{x}).
```

Representative exact primitives include the sphere

```math
d_{sphere}(\boldsymbol{x})
=\|\boldsymbol{x}-\boldsymbol{c}\|-r,
```

the axis-aligned box with center $`\boldsymbol{c}`$ and half size
$`\boldsymbol{h}`$,

```math
\boldsymbol{q}=|\boldsymbol{x}-\boldsymbol{c}|-\boldsymbol{h},
```

```math
d_{box}(\boldsymbol{x})
=\|\max(\boldsymbol{q},\boldsymbol{0})\|
+\min\left(\max_k q_k,0\right),
```

and the capsule around segment $`[\boldsymbol{a},\boldsymbol{b}]`$,

```math
t=\mathrm{clamp}
\left(
\frac{(\boldsymbol{x}-\boldsymbol{a})\cdot
(\boldsymbol{b}-\boldsymbol{a})}
{\|\boldsymbol{b}-\boldsymbol{a}\|^2},0,1
\right),
```

```math
d_{capsule}(\boldsymbol{x})
=\|\boldsymbol{x}-\boldsymbol{a}
-t(\boldsymbol{b}-\boldsymbol{a})\|-r.
```

## Boolean composition and offsets

For compatible negative-inside fields $`d_A`$ and $`d_B`$, hard constructive
solid geometry uses

```math
d_{A\cup B}=\min(d_A,d_B),
\qquad
d_{A\cap B}=\max(d_A,d_B),
```

```math
d_{A\setminus B}=\max(d_A,-d_B),
\qquad
d_{\neg A}=-d_A.
```

Dilation, erosion, and a centered shell of thickness $`t`$ are

```math
d_{dilate}=d-r,
\qquad
d_{erode}=d+r,
\qquad
d_{shell}=|d|-\frac{t}{2}.
```

For smoothing radius $`k>0`$, the polynomial smooth union is

```math
h=\mathrm{clamp}
\left(\frac12+\frac{d_B-d_A}{2k},0,1\right),
```

```math
d_{smooth\ union}
=h d_A+(1-h)d_B-kh(1-h).
```

Smooth intersection and difference use the corresponding sign changes in the
hard max operations. Hard min/max composition preserves the intended zero set
and sign, but at medial axes and after smooth blending the result is generally
an implicit distance-like field rather than an exact Eikonal solution.

## Coordinate transforms and dimensional constructions

Translation and rotation use inverse coordinate maps:

```math
d_{translated}(\boldsymbol{x})=d(\boldsymbol{x}-\boldsymbol{t}),
\qquad
d_{rotated}(\boldsymbol{x})=d(\boldsymbol{R}^T\boldsymbol{x}).
```

Uniform positive scale $`s`$ preserves distance under

```math
d_s(\boldsymbol{x})=s\,d(\boldsymbol{x}/s).
```

For anisotropic scale
$`\boldsymbol{s}=(s_1,\ldots,s_d)`$, the package uses the conservative field

```math
d_{\boldsymbol{s}}(\boldsymbol{x})
=\min_k(s_k)\,
d(\boldsymbol{x}\oslash\boldsymbol{s}),
```

which preserves sign but is not the exact Euclidean distance except when all
scales are equal. Twists, bends, repetitions, and transitions likewise act by
evaluating the source field at an inverse-warped point; a non-isometric warp
does not in general preserve $`\|\nabla d\|=1`$.

Extruding a 2D field $`d_2(x,y)`$ through height $`h`$ defines

```math
\boldsymbol{w}
=\left(d_2(x,y),\ |z|-\frac{h}{2}\right),
```

```math
d_{ext}
=\min(\max(w_1,w_2),0)
+\|\max(\boldsymbol{w},\boldsymbol{0})\|.
```

Revolution about the $`z`$ axis evaluates the profile at

```math
d_{rev}(x,y,z)
=d_2\left(\sqrt{x^2+y^2}-r_0,z\right).
```

## Package layout

| Path | Responsibility |
| --- | --- |
| `BasicShape.py` | Base shape state, mesh-backed geometry, normals, mass properties, and bounds |
| `SDFs.py` | `SDF2D`/`SDF3D` boolean and transform APIs |
| `SDFs2D.py`, `SDFs3D.py` | Analytic primitives |
| `MultiSDF.py` | Boolean and smooth composition functions |
| `utils2D.py`, `utils3D.py`, `utils23D.py` | Geometric transforms |
| `LevelSetGrid.py` | Sampled distance-grid representation |
| `BuildSurfaceNode.py` | Surface sampling for level-set bodies |
| `mesh.py`, `FastMarchingMethod.py` | Mesh extraction and distance propagation |

## Example

```python
import numpy as np
from geotaichi import box, capped_cylinder, sphere

shape = sphere(radius=0.6, center=(0.0, 0.0, 0.0))
shape = shape | box(size=(0.8, 0.8, 0.8))
shape = shape - capped_cylinder(
    point1=(0.0, 0.0, -1.0), point2=(0.0, 0.0, 1.0), radius=0.15
)
shape = shape.translate((1.0, 0.0, 0.0)).rotate(
    np.pi / 6.0, vector=(0.0, 1.0, 0.0)
)

points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
distance = shape(points)
normal = shape._normal(points)
```

Boolean operators create composed SDF objects; they do not modify the input
objects in place.

## Evaluation and preprocessing

Analytic SDF evaluation is vectorized NumPy geometry and is primarily used
during scene/template preprocessing. Solver modules transfer sampled geometry
or compact primitive parameters into Taichi fields before stepping. Do not
call Python SDF evaluation inside a device time-step loop.

Normals are evaluated through the existing `_normal()` interface. Although
the leading underscore is historical, it is currently the shared interface
used by geometry preprocessing and cloth SDF sampling.

## Sign and bounds

Most closed primitives follow the usual convention of negative distance
inside and positive distance outside. Half-space primitives define their sign
from the supplied normal. Boolean composition assumes compatible sign
conventions. Mesh-backed shapes should be watertight and consistently oriented
when inside/outside classification or mass properties are required.

SDF sampling resolution, smoothing, and bounding boxes directly affect
level-set contact quality and memory use.

## Tests

SDF behavior is exercised by DEM geometry, level-set, MPM soft-particle, FEM
cloth, and coupling tests rather than a single standalone test directory.
