# GeoTaichi Blender add-on

This directory is a thin Blender client for the canonical GeoTaichi
`SceneManifest` and `SolverJob` workflow. Numerical kernels do not run inside
Blender. The add-on calls the same `geotaichi-job` JSON CLI used by MCP, which
launches each model in an isolated Python worker.

## Workflow

1. Select the MPM, DEM, MPDEM, CFDEM, FEM, FEDEM, FEMPM, IGA, or IGAMPM
   solver family, its concrete solver mode, and the spatial dimension; then
   assign GeoTaichi roles to scene objects.
2. Select a GeoTaichi model entry script and optional model contract. Standard
   mode passes staged paths as `--contract`, `--scene-manifest`, and
   `--output-dir`.
3. Export `scene-manifest.json` and `solver-job.json`.
4. Validate the job.
5. Review the job and enable **Allow Trusted Script Execution**.
6. Submit, refresh/cancel the persistent task, and fetch terminal output into
   the Blender cache.
7. Open **VTU Results**, select the task output directory, and import its
   numbered VTU sequences. Scrubbing or playing Blender's timeline then loads
   the corresponding result frame automatically.

The add-on is deliberately local-only. SSH and Docker adapters can be added
behind the same job boundary without changing Blender scene data or the MCP
lifecycle.

Blocking CLI and filesystem operations run through one background worker at a
time. Blender data is read and updated only on Blender's main thread, and task
status is polled automatically until it reaches a terminal state.

The family/mode selector is descriptive metadata for MCP/model-builder routing;
it does not run or configure Taichi inside Blender.  `blender/defaults.py`
contains only add-on presentation and routing defaults. Numerical defaults
remain in the selected model script and solver, so Blender never becomes a
second owner of physical parameters. All families share the same versioned
scene/job export and immutable artifact cache.

FEM, IGA, and MPM accept both **Explicit** and **Implicit** with the
**Axisymmetric** spatial dimension.  The selected entry script remains
responsible for setting the solver's `axisymmetric=True` and `axis_offset`;
the Blender value is routing metadata rather than a hidden numerical default.

**Standard Model Arguments** is enabled by default. The generated SolverJob
uses `{model_contract}`, `{scene_manifest}`, and `{output_directory}`
placeholders; the task manager resolves them after immutable staging. Use
**Additional Arguments** for model-specific overrides such as
`--steps 20 --dt 1e-4`. Standard flags cannot be repeated there. Disable
standard arguments only for a legacy entry script that does not parse these
named options.

The bundled contract-driven templates require `--contract`, accept the
optional scene manifest, and use `--output-dir` only when the contract has no
explicit output path. Physical/material/contact configuration remains in the
model contract rather than being flattened into command-line arguments.

## VTU result animation

The importer is in **3D View > Sidebar (`N`) > GeoTaichi > VTU Results**:

1. Set **Result Directory** to a `vtks` directory or to a parent output
   directory. If it is empty, the most recently fetched cache/output directory
   is used. **Search Subdirectories** is enabled by default.
2. Leave **Prefix Filter** empty to import all numbered sequences. Enter a
   prefix such as `GraphicMPMParticle`, `GraphicLSMPMPoint`,
   `GraphicAffineBody`, or `FEM` to import only that result object.
3. Choose the Blender **Start** frame and **Step** between VTU files. **Point
   Radius** controls point-only MPM/DEM display; it is a Blender visualization
   value, not a solver parameter.
4. Select **Import VTU Sequences**. The add-on creates objects in the
   `GeoTaichi Results` collection, sets the timeline range, and binds the
   ordered VTU files to frame changes. **Reload Current Frame** rereads files
   that were replaced after import.

The reader uses Blender's bundled Python only; VTK, meshio, and NumPy are not
required inside Blender. It accepts the raw-appended binary written by pyevtk,
meshio's inline base64/zlib binary, and ASCII VTU. Vertex cells become a
renderable Geometry Nodes point cloud, surface cells become Blender faces, and
the exterior faces of tetrahedron, hexahedron, wedge, and pyramid volume cells
are extracted automatically. The `.blend` file stores the sequence path and
frame mapping, so the binding survives reopening while the source VTU files
remain at that path.

## Runtime boundary

Set **Python** to an interpreter where GeoTaichi is installed, or to a command
such as `python3` available on Blender's `PATH`. Set **Repository** when using a
checkout. The add-on invokes:

```text
python -m geotaichi_mcp.job_cli ...
```

The chosen model script is trusted arbitrary Python. Submission is disabled
until the explicit confirmation checkbox is selected. `GEOTAICHI_MCP_TASK_ID`,
the task directory, diagnostics path, and staged SolverJob path remain internal
process context; standard model input paths use arguments rather than
environment variables.

## Installation

For a legacy Blender add-on install, pass Blender's user add-on directory
explicitly:

```bash
python blender/install_blender_addon.py \
  --addon-path "/path/to/Blender/scripts/addons"
```

`make_release_blender_addon.py` creates a zip. The directory also contains
`blender_manifest.toml` for Blender extension builds.
