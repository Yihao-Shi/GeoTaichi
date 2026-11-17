"""Native material slots must remain valid Taichi template arguments."""

from types import SimpleNamespace

import taichi as ti

from src.utils.ListWrapper import ListWrapper


def test_material_slots_support_taichi_template_specialization(taichi_runtime):
    materials = ListWrapper()
    materials += [SimpleNamespace(density=1.0), SimpleNamespace(density=2.0)]
    materials[1] = SimpleNamespace(density=2.5)

    @ti.kernel
    def total_density(slots: ti.template()) -> ti.f64:
        total = 0.0
        for index in ti.static(range(slots.shape[0])):
            total += slots[index].density
        return total

    assert total_density(materials) == 3.5
