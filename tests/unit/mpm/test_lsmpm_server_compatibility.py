from types import SimpleNamespace


class _Field:
    def __init__(self, dtype):
        self.dtype = dtype


def _nested_scene(paths, dtype="f64"):
    scene = SimpleNamespace()
    for path in paths:
        owner = scene
        names = path.split(".")
        for name in names[:-1]:
            child = getattr(owner, name, None)
            if child is None:
                child = SimpleNamespace()
                setattr(owner, name, child)
            owner = child
        setattr(owner, names[-1], _Field(dtype))
    return scene


def test_precision_contract_matches_shared_surface_trace():
    from research.LSMPM.scripts.validation_common import (
        FORMAL_LSMPM_AUXILIARY_FIELDS,
        FORMAL_LSMPM_STATE_DTYPES,
        LSMPM_PRECISION_SCHEMA,
        lsmpm_precision_is_formal,
        lsmpm_precision_signature,
    )

    paths = tuple(FORMAL_LSMPM_STATE_DTYPES) + FORMAL_LSMPM_AUXILIARY_FIELDS
    scene = _nested_scene(paths)
    signature = lsmpm_precision_signature(scene, "float64")

    assert "surface_dshape" not in FORMAL_LSMPM_AUXILIARY_FIELDS
    assert signature["schema"] == LSMPM_PRECISION_SCHEMA
    assert signature["mode"] == "float64_state_float64_auxiliary"
    assert signature["mechanical_grid"]["nodal_state_bytes_per_slot"] == 80
    assert lsmpm_precision_is_formal(signature)


def test_hemisphere_sampler_import_does_not_load_optional_alphashape():
    from src.mpm.generator.Body import sample_solid_hemisphere_halton

    points = sample_solid_hemisphere_halton(
        [0.0, 0.0, 0.0],
        1.0,
        32,
        seed=7,
    )

    assert points.shape == (32, 3)
