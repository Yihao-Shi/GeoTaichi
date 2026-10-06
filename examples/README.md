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

When simulation and evaluation need the same fixed geometry or material
values, use the case's `<case>_parameters.py`. Pass realized CLI values to
postprocessing when those values change during configuration. Offline
evaluation must not import a script that starts a simulation.

Run examples from the repository root. FEMPM and IGAMPM cases write to
`OutputData/` inside their own case directories by default; `--output-dir`
can override this location. Other examples preserve their existing output
locations. Historical plots that use working-directory-relative inputs
still run from the same data directory as before.

For example, the cavity simulation and its independent comparison are:

```sh
python examples/mpm/IncompressibleFluid/lid_driven_cavity_2d/lid_driven_cavity_2d.py
python examples/mpm/IncompressibleFluid/lid_driven_cavity_2d/draw/compare_lid_driven_cavity_2d.py \
    examples/mpm/IncompressibleFluid/OutputData/lid_driven_cavity_2d
```
