"""The next test module must not inherit the preceding Taichi program."""

import taichi as ti


def test_taichi_runtime_is_reset_at_test_module_boundary():
    assert ti.lang.impl.get_runtime().prog is None

