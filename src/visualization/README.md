# Real-Time Scientific Visualization for Particle, Grid, and Surface Mechanics

`src/visualization` provides an interactive viewer for small GeoTaichi
simulations. It uses the vendored PyRender package together with Pyglet and
PyOpenGL and does not create a Taichi GGUI window.

## Capabilities

- Particle rendering with scalar or per-particle radii.
- Dynamic triangle meshes and wireframe surfaces.
- Adapters for DEM particles, DEM triangle walls, LSDEM surfaces, MPM
  particles, and IGA patches.
- Pause/resume, single-step, configurable steps per frame, and reset.
- Orbit/trackball camera controls, domain bounds, lights, and diagnostic text.
- Solver adapters invoked through existing `run(visualize=True)` paths.

## Camera and instance-transform model

Visualization owns no governing mechanical equation; it samples already
computed solver state. If $n$ simulation steps of size $\Delta t$ have been
accepted, the displayed physical time is

$$
t_n=n\Delta t.
$$

For camera eye $\boldsymbol{e}$, target $\boldsymbol{c}$, and up hint
$\boldsymbol{u}_0$, define the orthonormal frame

$$
\boldsymbol{f}
=\frac{\boldsymbol{c}-\boldsymbol{e}}
{\|\boldsymbol{c}-\boldsymbol{e}\|},
\qquad
\boldsymbol{r}
=\frac{\boldsymbol{f}\times\boldsymbol{u}_0}
{\|\boldsymbol{f}\times\boldsymbol{u}_0\|},
\qquad
\boldsymbol{u}=\boldsymbol{r}\times\boldsymbol{f}.
$$

The camera-to-world pose is

$$
\boldsymbol{T}_{cw}
=\begin{bmatrix}
\boldsymbol{r}&\boldsymbol{u}&-\boldsymbol{f}&\boldsymbol{e}\\
0&0&0&1
\end{bmatrix}.
$$

With vertical field of view $\theta$, aspect ratio $a$, and camera-space point
$(x_c,y_c,z_c)$ in front of the OpenGL camera ($z_c<0$), perspective division
gives

$$
x_{ndc}=\frac{x_c}{-z_c\,a\tan(\theta/2)},
\qquad
y_{ndc}=\frac{y_c}{-z_c\tan(\theta/2)}.
$$

A particle at $\boldsymbol{x}_p$ with physical visualization radius $r_p$ and
global scale $s$ is rendered by instancing a unit sphere with

$$
\boldsymbol{T}_p
=\begin{bmatrix}
sr_p\boldsymbol{I}_{3\times3}&\boldsymbol{x}_p\\
\boldsymbol{0}^T&1
\end{bmatrix}.
$$

These transforms affect only rendering; they never modify the mechanical
positions, radii, or time integration.

## Package layout

| Path | Responsibility |
| --- | --- |
| `sources.py` | Host-side particle and triangle render-source descriptions |
| `adapters.py` | DEM, MPM, level-set, and IGA source adapters |
| `realtime.py` | Scene model, playback controller, options, and viewer |
| `solver_adapters.py` | High-level solver-to-viewer lifecycle adapters |

## Callback example

```python
import numpy as np
from src.visualization import (
    CallbackSceneModel,
    ParticleRenderSource,
    RealtimeViewer,
    RealtimeViewerOptions,
)

positions = np.array([[0.2, 0.3, 0.4], [0.6, 0.7, 0.4]])

source = ParticleRenderSource(
    positions=lambda: positions,
    radii=0.05,
    color=(0.22, 0.58, 0.95),
)
model = CallbackSceneModel(
    name="particles",
    render_sources=(source,),
    time_step=1.0e-3,
    step_callback=lambda: None,
    domain_min=(0.0, 0.0, 0.0),
    domain_max=(1.0, 1.0, 1.0),
)
viewer = RealtimeViewer(
    model,
    RealtimeViewerOptions(start_paused=True, steps_per_frame=4),
)
viewer.run()
```

For a configured solver, prefer its facade integration:

```python
dem.run(visualize=True)
mpm.run(visualize=True)
iga.run(visualize=True, visualize_resolution=16)
```

## Runtime boundary

Render sources create host snapshots only when a frame is synchronized. They
read existing Taichi fields through `to_numpy()` but never allocate mechanical
state or feed values back into the numerical backend. Rendering therefore
remains an explicit output boundary.

Large scenes may be expensive because particle and dynamic mesh snapshots are
copied to the host every rendered frame. Use file-based VTU output for large
production runs and reserve the real-time viewer for setup checks and small
interactive cases.

## Dependencies and tests

The viewer depends on the vendored `third_party/pyrender` assets, Pyglet, and
PyOpenGL. Headless systems need an appropriate OpenGL context or should use
file output instead. Tests are under `tests/unit/visualization/` and
`tests/integration/visualization/`.
