from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_every_supported_step_loop_has_a_cooperative_runtime_checkpoint():
    expected_calls = {
        "src/dem/DEMBase.py": 2,
        "src/dem/engines/AffineBodyEngine.py": 1,
        "src/mpm/MPMBase.py": 1,
        "src/mpm/engines/direct/MPMSolver.py": 1,
        "src/mpm/soft_particle/IPCULMPM.py": 1,
        "src/mpm/soft_particle/IPCTLMPM.py": 1,
        "src/mpdem/DEMPMBase.py": 1,
        "src/mpdem/engines/SoftAffineIPCBase.py": 1,
        "src/iga/engines/ExplicitIGA.py": 1,
        "src/iga/engines/ImplicitIGA.py": 1,
        "src/igampm/engines/ImplicitEngine.py": 1,
        "src/fem/engines/ExplicitFEM.py": 1,
        "src/fem/engines/ImplicitFEM.py": 1,
        "src/fedem/FEDEMBase.py": 1,
        # These implicit solvers execute the first JIT/compile step outside
        # the regular loop, so both accepted-step branches need a safe point.
        "src/fempm/ImplicitEngine.py": 2,
        "src/fedem/AffineIPCEngine.py": 2,
        "src/visualization/realtime.py": 1,
    }

    for relative_path, expected in expected_calls.items():
        content = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
        assert "from src.utils.RuntimeHook import runtime_checkpoint" in content
        assert content.count("runtime_checkpoint()") == expected
