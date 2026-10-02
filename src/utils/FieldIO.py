import os

import numpy as np
import taichi as ti


_TAICHI_TO_NUMPY_DTYPE = {
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "u8": np.uint8,
    "u16": np.uint16,
    "u32": np.uint32,
    "u64": np.uint64,
}

_MAX_COPY_ENTRIES_PER_KERNEL = max(
    1,
    int(os.environ.get("GEOTAICHI_FIELD_COPY_ENTRIES", "32768")),
)


def runtime_float_numpy_dtype():
    """Return the NumPy dtype matching Taichi's active default float."""
    return _TAICHI_TO_NUMPY_DTYPE.get(
        str(ti.lang.impl.current_cfg().default_fp), np.float64
    )


@ti.kernel
def _copy_scalar_field_slice(src: ti.template(), start: int, dst: ti.types.ndarray()):
    for i in range(dst.shape[0]):
        dst[i] = src[start + i]


@ti.kernel
def _copy_vector_field_slice(src: ti.template(), start: int, dst: ti.types.ndarray(), n: ti.template()):
    for i in range(dst.shape[0]):
        for j in ti.static(range(n)):
            dst[i, j] = src[start + i][j]


@ti.kernel
def _copy_matrix_field_slice(src: ti.template(), start: int, dst: ti.types.ndarray(), n: ti.template(), m: ti.template()):
    for i in range(dst.shape[0]):
        for j, k in ti.static(ti.ndrange(n, m)):
            dst[i, j, k] = src[start + i][j, k]


@ti.kernel
def _copy_matrix_component_field_slice(src: ti.template(), start: int, dst: ti.types.ndarray(), j: ti.template(), k: ti.template()):
    for i in range(dst.shape[0]):
        dst[i] = src[start + i][j, k]


def field_to_numpy_slice(field, start=0, end=-1):
    """Copy a one-dimensional Taichi field slice without materializing the full field."""
    if field is None:
        return None
    if len(field.shape) != 1:
        return np.ascontiguousarray(field.to_numpy()[start:end])

    start = int(start)
    field_length = int(field.shape[0])
    if end == -1:
        end = field_length
    end = int(end)
    if start < 0 or end < start or end > field_length:
        raise ValueError(f"Invalid Taichi field slice [{start}:{end}] for length {field_length}")
    if start == 0 and end == field_length:
        return np.ascontiguousarray(field.to_numpy())

    count = end - start
    dtype = _TAICHI_TO_NUMPY_DTYPE.get(str(getattr(field, "dtype", "")), np.float64)
    n = getattr(field, "n", None)
    m = getattr(field, "m", None)
    component_count = 1
    if n is not None:
        component_count = int(n)
        if m != 1:
            component_count *= int(m)
    chunk_size = max(1, _MAX_COPY_ENTRIES_PER_KERNEL // component_count)

    if count == 0:
        if n is None:
            return np.empty((0,), dtype=dtype)
        if m == 1:
            return np.empty((0, int(n)), dtype=dtype)
        return np.empty((0, int(n), int(m)), dtype=dtype)

    if n is None:
        out = np.empty((count,), dtype=dtype)
        for offset in range(0, count, chunk_size):
            chunk_count = min(chunk_size, count - offset)
            chunk = np.empty((chunk_count,), dtype=dtype)
            _copy_scalar_field_slice(field, start + offset, chunk)
            out[offset:offset + chunk_count] = chunk
    elif m == 1:
        out = np.empty((count, int(n)), dtype=dtype)
        for offset in range(0, count, chunk_size):
            chunk_count = min(chunk_size, count - offset)
            chunk = np.empty((chunk_count, int(n)), dtype=dtype)
            _copy_vector_field_slice(field, start + offset, chunk, int(n))
            out[offset:offset + chunk_count] = chunk
    else:
        out = np.empty((count, int(n), int(m)), dtype=dtype)
        for j in range(int(n)):
            for k in range(int(m)):
                for offset in range(0, count, chunk_size):
                    chunk_count = min(chunk_size, count - offset)
                    chunk = np.empty((chunk_count,), dtype=dtype)
                    _copy_matrix_component_field_slice(field, start + offset, chunk, j, k)
                    out[offset:offset + chunk_count, j, k] = chunk
    return np.ascontiguousarray(out)


def field_to_numpy_prefix(field, count):
    return field_to_numpy_slice(field, 0, int(count))
