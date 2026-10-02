"""Shared Taichi kernels used by multiple utility modules."""

import taichi as ti


@ti.kernel
def serial(input: ti.template()):
    n = input.shape[0]
    ti.loop_config(serialize=True)
    for i in range(1, n):
        input[i] = input[i] + input[i - 1]


@ti.kernel
def serial_range(input: ti.template(), start: int, end: int):
    ti.loop_config(serialize=True)
    for i in range(start + 1, end):
        input[i] = input[i] + input[i - 1]
