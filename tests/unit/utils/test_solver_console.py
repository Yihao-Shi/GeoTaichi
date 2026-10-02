import importlib
from types import SimpleNamespace

import pytest

from src.utils.SolverConsole import (
    constitutive_model_name,
    print_material_info,
    print_save_file_info,
    print_simulation_start,
    print_solver_section,
)


PUBLIC_SOLVER_FACADES = (
    ("src.mpm.mainMPM", "MPM"),
    ("src.dem.mainDEM", "DEM"),
    ("src.mpdem.mainDEMPM", "DEMPM"),
    ("src.fem.mainFEM", "FEM"),
    ("src.iga.mainIGA", "IGA"),
    ("src.fedem.mainFEDEM", "FEDEM"),
    ("src.fempm.mainFEMPM", "FEMPM"),
    ("src.igampm.mainIGAMPM", "IGAMPM"),
)


def test_solver_console_prints_sections_start_and_save_path(capsys):
    print_solver_section(
        "Example",
        "Basic Configuration",
        [("Dimension", 3), ("Backend", None)],
    )
    print_simulation_start("Example")
    print_save_file_info("Example", 12, 3, 0.25, "OutputData/part.vtu")

    output = capsys.readouterr().out
    assert "Example Basic Configuration" in output
    assert "Dimension: 3" in output
    assert "Backend: Not configured" in output
    assert "Start Example Simulation" in output
    assert "Step = 12" in output
    assert "Save Number = 3" in output
    assert "Simulation Time = 0.25" in output
    assert output.splitlines()[-1] == ("# Example Save | Step = 12 | Save Number = 3 | " "Simulation Time = 0.25")
    assert "Save Path =" not in output


def test_material_information_starts_with_readable_model_then_id(capsys):
    print_material_info("NeoHookeanModel", 3, [("Density", 1000)], solver_name="FEM")

    lines = capsys.readouterr().out.splitlines()
    assert "FEM Constitutive Model Information" in lines[0]
    assert lines[1] == "Constitutive model: Neo-Hookean"
    assert lines[2] == "Material ID: 3"
    assert lines[3] == "Density: 1000"
    assert constitutive_model_name("StVK") == "St. Venant-Kirchhoff"
    assert constitutive_model_name("ClothARAP") == "Cloth ARAP"


def test_iga_material_information_uses_model_before_id(capsys):
    from src.iga.mainIGA import IGA

    solver = IGA(log=False)
    solver.add_material(
        density=1200,
        young_modulus=2.5e5,
        poisson_ratio=0.35,
    )

    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == "Constitutive model: Neo-Hookean"
    assert lines[2] == "Material ID: 0"


def test_fem_material_information_uses_model_before_id(capsys):
    from src.fem.mainFEM import FEM

    solver = FEM(log=False)
    solver.add_material(
        "NeoHookean",
        density=1100.0,
        young_modulus=1.0e4,
        poisson_ratio=0.3,
    )

    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == "Constitutive model: Neo-Hookean"
    assert lines[2].split() == ["Material", "ID:", "0"]


@pytest.mark.parametrize(("module_name", "class_name"), PUBLIC_SOLVER_FACADES)
def test_public_solver_facades_expose_information_sections(module_name, class_name):
    solver_class = getattr(importlib.import_module(module_name), class_name)

    assert callable(getattr(solver_class, "print_basic_simulation_info"))
    assert callable(getattr(solver_class, "print_solver_info"))
    assert callable(getattr(solver_class, "print_memory_info"))
    assert callable(getattr(solver_class, "print_neighbor_search_info"))


def test_all_public_solver_information_sections_read_default_state(capsys):
    import taichi as ti

    from src.dem.mainDEM import DEM
    from src.fedem.mainFEDEM import FEDEM
    from src.fem.mainFEM import FEM
    from src.fempm.mainFEMPM import FEMPM
    from src.iga.mainIGA import IGA
    from src.igampm.mainIGAMPM import IGAMPM
    from src.mpdem.mainDEMPM import DEMPM
    from src.mpm.mainMPM import MPM

    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64)

    mpm = MPM(log=False)
    dem = DEM(log=False)
    fem = FEM(log=False)
    iga = IGA(log=False)
    solvers = (
        mpm,
        dem,
        DEMPM(dem, mpm, log=False),
        fem,
        iga,
        FEDEM(dem, fem, log=False),
        FEMPM(fem, mpm, log=False),
        IGAMPM(iga, mpm, log=False),
    )

    for solver in solvers:
        solver.print_basic_simulation_info()
        solver.print_solver_info()
        solver.print_memory_info()
        solver.print_neighbor_search_info()

    output = capsys.readouterr().out
    assert output.count("Basic Configuration") == len(solvers)
    assert output.count("Solver Information") == len(solvers)
    assert output.count("Memory Information") == len(solvers)
    # FEM/IGA and default grid-based MPM do not print unused search sections.
    assert output.count("Neighbor Search Information") == 5
    assert output.count("Save Path:") == len(solvers)


def test_all_primary_vtk_writers_use_the_vtks_directory():
    from pathlib import Path

    sources = {
        "FEM": Path("src/fem/engines/FEMSolver.py"),
        "FEDEM": Path("src/fedem/Recorder.py"),
        "FEMPM": Path("src/fempm/Recorder.py"),
        "IGA": Path("src/iga/engines/IGASolver.py"),
        "Direct MPM": Path("src/mpm/engines/direct/MPMSolver.py"),
        "Direct two-phase MPM": Path("src/mpm/engines/direct/StaticTwoPhaseULMPM.py"),
        "AffineBody": Path("src/dem/engines/AffineBodyEngine.py"),
    }

    for name, path in sources.items():
        source = path.read_text(encoding="utf-8")
        assert '"vtks"' in source, f"{name} does not use Save Path/vtks"
    assert '"fems"' not in Path("src/fedem/Recorder.py").read_text(encoding="utf-8")
    assert '"fems"' not in Path("src/fempm/Recorder.py").read_text(encoding="utf-8")


def test_unused_neighbor_search_sections_and_fields_are_omitted(capsys):
    from src.fem.mainFEM import FEM
    from src.mpm.mainMPM import MPM

    fem = FEM(log=False)
    fem.print_neighbor_search_info()
    assert capsys.readouterr().out == ""

    fem.scene.contact = SimpleNamespace(broad_phase="BVH")
    fem.print_neighbor_search_info()
    output = capsys.readouterr().out
    assert "FEM Contact Broad Phase: BVH" in output
    assert "Soft-particle" not in output
    assert "Not used" not in output

    mpm = MPM(log=False)
    mpm.sims.set_ipc_contact(True, {})
    mpm.print_neighbor_search_info()
    output = capsys.readouterr().out
    assert "Search Usage: IPC candidate search" in output
    assert "IPC Contact: True" in output
    assert "Verlet Distance" not in output
    assert "Not used" not in output


def test_every_primary_solver_branch_reports_mpm_style_compile_timing():
    from pathlib import Path

    sources = (
        Path("src/dem/DEMBase.py"),
        Path("src/mpm/MPMBase.py"),
        Path("src/mpm/engines/direct/MPMSolver.py"),
        Path("src/mpdem/DEMPMBase.py"),
        Path("src/mpdem/engines/SoftAffineIPCBase.py"),
        Path("src/fem/engines/ExplicitFEM.py"),
        Path("src/fem/engines/ImplicitFEM.py"),
        Path("src/iga/engines/ExplicitIGA.py"),
        Path("src/iga/engines/ImplicitIGA.py"),
        Path("src/fedem/FEDEMBase.py"),
        Path("src/fedem/AffineIPCEngine.py"),
        Path("src/fempm/FEMPMBase.py"),
        Path("src/fempm/ImplicitEngine.py"),
        Path("src/igampm/engines/ExplicitEngine.py"),
        Path("src/igampm/engines/ImplicitEngine.py"),
    )

    for path in sources:
        source = path.read_text(encoding="utf-8")
        assert "Compiling first ... ..." in source, path
        assert "Compiling time =" in source, path
        assert ".profile1()" in source, path


def test_solver_principles_document_the_cross_solver_contract():
    from pathlib import Path

    document = Path("docs/solver_integration_principles.md").read_text(encoding="utf-8")

    for solver_name in (
        "DEM",
        "MPM",
        "FEM",
        "IGA",
        "MPDEM",
        "FEDEM",
        "FEMPM",
        "IGAMPM",
    ):
        assert solver_name in document
    assert "argparse" in document
    assert "Neighbor Search" in document or "neighbor-search" in document
    assert "frame zero" in document
    assert "exactly one top-level save event" in document
