import ast
from pathlib import Path

SOURCE = Path(__file__).parents[3] / "src/mpdem/engines/SoftAffineIPCOperator.py"


def test_mesh_mixed_hot_paths_use_stable_bvh_candidates():
    tree = ast.parse(SOURCE.read_text())
    operator = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SoftAffineIPCOperator"
    )
    methods = {
        node.name: ast.unparse(node)
        for node in operator.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    hot_paths = (
        "_differentiate_mixed_barrier_input_position",
        "_count_mixed_lagged_friction_contacts",
        "_write_mixed_lagged_friction_contacts",
        "_assemble_mixed_fully_implicit_friction",
        "_count_mixed_contact_types",
        "_assemble_mixed_contact_type",
        "_compute_mixed_ccd_alpha",
    )
    for name in hot_paths:
        assert "mixed_bvh.point_triangle" in methods[name]
        assert "affine.face_num" not in methods[name]

    assert "np.unique" not in methods["__init__"]
    assert "if ti.static(self.affine.levelset_contact)" in methods["_build_mixed_pairs"]
    assert "mixed_bvh.rebuild" in methods["_update_mixed_mesh_candidates"]
    assert "_initialize_mixed_lagged_friction_atomic" not in methods
    for name in ("refresh_lagged_friction_device", "assemble_device"):
        assert "_update_mixed_mesh_candidates(False)" in methods[name]
    assert "_update_mixed_mesh_candidates(True)" in methods["_mixed_ccd_step_size_device"]
