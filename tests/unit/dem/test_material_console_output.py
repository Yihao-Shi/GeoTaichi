from pathlib import Path

import pytest

from src.dem.structs.Material import Material


@pytest.mark.parametrize(
    ("model_id", "name"),
    (
        (0, "Neo-Hookean"),
        (1, "Hencky Elastic"),
        (2, "Mooney-Rivlin"),
        (3, "Gent"),
        (4, "Hydrogel"),
        (5, "Drucker-Prager"),
        (-1, "Unknown (-1)"),
    ),
)
def test_soft_constitutive_model_ids_have_readable_names(model_id, name, capsys):
    material = Material(constitutive_model=model_id)
    material.print_info(7)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"Constitutive model:  {name}"
    assert lines[1] == "Material ID:  7"


def test_material_output_starts_with_model_name_then_material_id():
    source_path = Path("src/dem/structs/Material.py")
    source = source_path.read_text(encoding="utf-8")
    print_info = source.split("def print_info", 1)[1].split("\n\n", 1)[0]

    assert print_info.index('print("Constitutive model: ", model_name)') < (
        print_info.index('print("Material ID: ", matID)')
    )
    assert "Soft finite-strain model ID" not in print_info


def test_constitutive_sources_do_not_use_numeric_model_id_labels():
    source_root = Path("src/physics_model/constitutive_model")
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in source_root.rglob("*.py") if not path.name.startswith("._")
    )

    assert "Model ID" not in source
    assert "Constitutive model =" not in source


def test_affine_material_console_uses_effective_model_and_parameters():
    source = Path("src/dem/SceneManager.py").read_text(encoding="utf-8")

    assert "Constitutive model: Affine rigidity (Green-strain penalty)" in source
    assert "sims.affine_force_local_damping" in source
    assert "sims.affine_torque_local_damping" in source
    assert "sims.affine_young_modulus" in source
