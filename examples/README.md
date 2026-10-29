# Example layout

Each independent case has a directory under its solver or example family:

```text
examples/<solver>/<family>/<case>/
    <case>.py
    draw/
        evaluate_<case>.py
        draw.py
```

The main script sets up and runs the simulation. Keep evaluation, error
comparisons, result summaries and plotting in `draw/`. The main script can
call these functions after recording results, so its existing postprocessing
options still work. Multi-stage cases can retain their preparation and
simulation scripts together.

Keep each case's model configuration and functions in its own directory;
do not import them from another example. Case-independent fluid diagnostics
and wave references live in `tools/postprocessing/mpm/fluid_metrics.py`.

When simulation and evaluation need the same fixed geometry or material
values, use the case's `<case>_parameters.py`. Pass realized CLI values to
postprocessing when those values change during configuration. Offline
evaluation must not import a script that starts a simulation.

Run examples from the repository root. A case that writes `OutputData/` must
resolve it beneath that case's own directory; solver-family directories must
not be shared output roots. An explicit `--output-dir` or output environment
variable can override the default. Historical plots that use
working-directory-relative inputs still run from the same data directory as
before.

For example, the cavity simulation and its independent comparison are:

```sh
python examples/mpm/IncompressibleFluid/lid_driven_cavity_2d/lid_driven_cavity_2d.py
python examples/mpm/IncompressibleFluid/lid_driven_cavity_2d/draw/compare_lid_driven_cavity_2d.py \
    examples/mpm/IncompressibleFluid/lid_driven_cavity_2d/OutputData/lid_driven_cavity_2d
```
