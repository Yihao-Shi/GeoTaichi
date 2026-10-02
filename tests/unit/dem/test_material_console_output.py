from pathlib import Path

from src.dem.structs.BaseStruct import soft_constitutive_model_name


def test_soft_constitutive_model_ids_have_readable_names():
    assert soft_constitutive_model_name(0) == "Neo-Hookean"
    assert soft_constitutive_model_name(1) == "Hencky Elastic"
    assert soft_constitutive_model_name(2) == "Mooney-Rivlin"
    assert soft_constitutive_model_name(3) == "Gent"
    assert soft_constitutive_model_name(4) == "Hydrogel"
    assert soft_constitutive_model_name(5) == "Drucker-Prager"


def test_material_output_starts_with_model_name_then_material_id():
    source_path = Path("src/dem/structs/BaseStruct.py")
    source = source_path.read_text(encoding="utf-8")
    print_info = source.split("def print_info", 1)[1].split("\n\n", 1)[0]

    assert print_info.index('print("Constitutive model: ", model_name)') < (
        print_info.index('print("Material ID: ", matID)')
    )
    assert "Soft finite-strain model ID" not in print_info


def test_constitutive_sources_do_not_use_numeric_model_id_labels():
    source_root = Path("src/physics_model/consititutive_model")
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
