"""Taichi kernels for FEM sparse matrix assembly."""

import taichi as ti


@ti.kernel
def _copy_coo_triplets(
    offset: ti.i32,
    rows: ti.types.ndarray(dtype=ti.i32, ndim=1),
    columns: ti.types.ndarray(dtype=ti.i32, ndim=1),
    values: ti.types.ndarray(dtype=ti.f64, ndim=1),
    target_rows: ti.template(),
    target_columns: ti.template(),
    target_values: ti.template(),
):
    for index in range(rows.shape[0]):
        target_rows[offset + index] = rows[index]
        target_columns[offset + index] = columns[index]
        target_values[offset + index] = values[index]


@ti.kernel
def _zero_constrained_coo_entries(
    count: ti.i32,
    constrained: ti.template(),
    rows: ti.template(),
    columns: ti.template(),
    values: ti.template(),
):
    for index in range(count):
        if constrained[rows[index]] != 0 or constrained[columns[index]] != 0:
            values[index] = 0.0


@ti.kernel
def _build_coo_diagonal(
    count: ti.i32,
    rows: ti.template(),
    columns: ti.template(),
    values: ti.template(),
    diagonal: ti.template(),
):
    for dof in range(diagonal.shape[0]):
        diagonal[dof] = 0.0
    for index in range(count):
        if rows[index] == columns[index]:
            ti.atomic_add(diagonal[rows[index]], values[index])
    for dof in range(diagonal.shape[0]):
        if ti.abs(diagonal[dof]) < 1.0e-14:
            diagonal[dof] = 1.0


@ti.kernel
def _apply_hash_constraints(
    matrix: ti.template(),
    constrained: ti.template(),
):
    for block in range(matrix.max_active_nodes):
        for row, column in ti.static(ti.ndrange(3, 3)):
            row_dof = 3 * block + row
            column_dof = 3 * block + column
            index = row * 3 + column
            if constrained[row_dof] != 0 or constrained[column_dof] != 0:
                matrix.diag[block][index] = 0.0
        for component in ti.static(range(3)):
            dof = 3 * block + component
            if constrained[dof] != 0:
                matrix.diag[block][component * 3 + component] = 1.0

    for entry in range(matrix.raw_non_diag_count[0]):
        if entry < matrix.non_diag.blockI.shape[0]:
            block_i = matrix.non_diag.blockI[entry]
            block_j = matrix.non_diag.blockJ[entry]
            if block_i >= 0 and block_j >= 0:
                block = matrix.non_diag.blockH[entry]
                for row, column in ti.static(ti.ndrange(3, 3)):
                    if constrained[3 * block_i + row] != 0 or constrained[3 * block_j + column] != 0:
                        block[row * 3 + column] = 0.0
                matrix.non_diag.blockH[entry] = block


@ti.kernel
def _load_scalar_field(values: ti.types.ndarray(dtype=ti.f64, ndim=1), field: ti.template()):
    for index in range(values.shape[0]):
        field[index] = values[index]


@ti.kernel
def _write_coo_diagonal_tail(
    offset: ti.i32,
    diagonal_shift: ti.f64,
    constrained: ti.template(),
    additional_diagonal: ti.template(),
    rows: ti.template(),
    columns: ti.template(),
    values: ti.template(),
):
    dof_count = constrained.shape[0]
    for dof in range(dof_count):
        rows[offset + dof] = dof
        columns[offset + dof] = dof
        values[offset + dof] = (additional_diagonal[dof] + diagonal_shift) * ti.cast(constrained[dof] == 0, ti.f64)
        rows[offset + dof_count + dof] = dof
        columns[offset + dof_count + dof] = dof
        values[offset + dof_count + dof] = ti.cast(constrained[dof] != 0, ti.f64)


@ti.kernel
def _add_hash_diagonal(
    matrix: ti.template(),
    diagonal_shift: ti.f64,
    constrained: ti.template(),
    additional_diagonal: ti.template(),
):
    for node, component in ti.ndrange(matrix.max_active_nodes, 3):
        dof = 3 * node + component
        if constrained[dof] == 0:
            ti.atomic_add(
                matrix.diag[node][component * 3 + component],
                additional_diagonal[dof] + diagonal_shift,
            )


@ti.kernel
def _pack_vector_rhs(vector: ti.template(), constrained: ti.template(), flat: ti.template()):
    for node, component in ti.ndrange(vector.shape[0], 3):
        dof = 3 * node + component
        if constrained[dof] == 0:
            flat[dof] = -vector[node][component]
        else:
            flat[dof] = 0.0


@ti.kernel
def _unpack_vector_solution(flat: ti.template(), constrained: ti.template(), vector: ti.template()):
    for node, component in ti.ndrange(vector.shape[0], 3):
        dof = 3 * node + component
        if constrained[dof] == 0:
            vector[node][component] = flat[dof]
        else:
            vector[node][component] = 0.0


@ti.kernel
def _set_mass_diagonal(mass: ti.template(), factor: ti.f64, diagonal: ti.template()):
    for node, component in ti.ndrange(mass.shape[0], 3):
        diagonal[3 * node + component] = factor * mass[node]
